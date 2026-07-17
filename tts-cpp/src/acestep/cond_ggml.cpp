#include "cond_ggml.h"

#include "qwen3_block.h"  // shared Qwen3 loaders + builders + DitGGUF IO

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ACE-Step condition encoder. Port from acestep.cpp/src/cond-enc.h.
// Lyric encoder (8L) + timbre encoder (4L) + text projector, packed into the
// DiT cross-attention states. Bidirectional Qwen3 with a sliding-window mask on
// even layers (|i-j| <= 128). Weights come from the DiT GGUF (encoder.* prefix).

namespace tts_cpp::acestep {

static constexpr int COND_H  = 2048;  // encoder hidden size
static constexpr int COND_W  = 128;   // bidirectional sliding window

static Qwen3Config lyric_config() {
    Qwen3Config c;
    c.hidden_size = 2048; c.intermediate_size = 6144; c.n_heads = 16; c.n_kv_heads = 8;
    c.head_dim = 128; c.n_layers = 8; c.rope_theta = 1000000.0f; c.rms_norm_eps = 1e-6f; c.is_causal = false;
    return c;
}
static Qwen3Config timbre_config() {
    Qwen3Config c = lyric_config();
    c.n_layers = 4;
    return c;
}

struct CondModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    Qwen3Config lyric_cfg;
    Qwen3Config timbre_cfg;

    std::vector<Qwen3Layer> lyric_layers;
    ggml_tensor *           lyric_embed_w = nullptr;  // [1024, 2048]
    ggml_tensor *           lyric_embed_b = nullptr;  // [2048] F32
    ggml_tensor *           lyric_norm    = nullptr;  // [2048] F32

    std::vector<Qwen3Layer> timbre_layers;
    ggml_tensor *           timbre_embed_w = nullptr;  // [64, 2048]
    ggml_tensor *           timbre_embed_b = nullptr;  // [2048] F32
    ggml_tensor *           timbre_norm    = nullptr;  // [2048] F32
    ggml_tensor *           timbre_cls     = nullptr;  // [2048,1,1] F32 or null
    bool                    use_timbre_cls = false;

    ggml_tensor * text_proj_w = nullptr;  // [1024, 2048]

    std::vector<float> null_emb;  // [2048] dequantized
};

static void dequant_to_f32(const DitGGUF & g, const std::string & name, std::vector<float> & out) {
    ggml_tensor * mt = dit_gmeta(g, name);
    const void *  s  = dit_gdata(g, name);
    if (!mt || !s) { out.clear(); return; }
    const size_t n = ggml_nelements(mt);
    out.resize(n);
    if (mt->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), s, n * sizeof(float));
    } else if (mt->type == GGML_TYPE_F16) {
        const ggml_fp16_t * p = (const ggml_fp16_t *) s;
        for (size_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(p[i]);
    } else if (mt->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) s;
        for (size_t i = 0; i < n; i++) out[i] = q3_bf16_to_f32(p[i]);
    } else {
        out.clear();
    }
}

CondModel * cond_model_load(const std::string & path, ggml_backend_t backend, bool verbose) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) {
        fprintf(stderr, "[acestep-cond] failed to parse %s\n", path.c_str());
        return nullptr;
    }

    CondModel * m   = new CondModel();
    m->backend      = backend;
    m->lyric_cfg    = lyric_config();
    m->timbre_cfg   = timbre_config();
    m->use_timbre_cls = dit_gguf_has(g, "encoder.timbre_encoder.special_token");

    const size_t n_tensors = (size_t) (m->lyric_cfg.n_layers + m->timbre_cfg.n_layers) * 11 + 16;
    ggml_init_params ip{ ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc=*/true };
    m->weight_ctx = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    // lyric encoder
    m->lyric_embed_w = q3_create_like(ctx, g, "encoder.lyric_encoder.embed_tokens.weight");
    m->lyric_embed_b = q3_create_f32_like(ctx, g, "encoder.lyric_encoder.embed_tokens.bias");
    m->lyric_norm    = q3_create_f32_like(ctx, g, "encoder.lyric_encoder.norm.weight");
    m->lyric_layers.resize(m->lyric_cfg.n_layers);
    for (int i = 0; i < m->lyric_cfg.n_layers; i++) {
        q3_create_layer(ctx, g, "encoder.lyric_encoder.layers." + std::to_string(i), m->lyric_layers[i]);
    }

    // timbre encoder
    m->timbre_embed_w = q3_create_like(ctx, g, "encoder.timbre_encoder.embed_tokens.weight");
    m->timbre_embed_b = q3_create_f32_like(ctx, g, "encoder.timbre_encoder.embed_tokens.bias");
    m->timbre_norm    = q3_create_f32_like(ctx, g, "encoder.timbre_encoder.norm.weight");
    m->timbre_layers.resize(m->timbre_cfg.n_layers);
    for (int i = 0; i < m->timbre_cfg.n_layers; i++) {
        q3_create_layer(ctx, g, "encoder.timbre_encoder.layers." + std::to_string(i), m->timbre_layers[i]);
    }
    if (m->use_timbre_cls) {
        m->timbre_cls = q3_create_f32_like(ctx, g, "encoder.timbre_encoder.special_token");
    }

    // text projector
    m->text_proj_w = q3_create_like(ctx, g, "encoder.text_projector.weight");

    m->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-cond] failed to allocate weight buffer\n");
        ggml_free(ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    // upload
    q3_load_raw(m->lyric_embed_w, g, "encoder.lyric_encoder.embed_tokens.weight");
    q3_load_f32(m->lyric_embed_b, g, "encoder.lyric_encoder.embed_tokens.bias");
    q3_load_f32(m->lyric_norm, g, "encoder.lyric_encoder.norm.weight");
    for (int i = 0; i < m->lyric_cfg.n_layers; i++) {
        q3_load_layer(g, "encoder.lyric_encoder.layers." + std::to_string(i), m->lyric_layers[i]);
    }
    q3_load_raw(m->timbre_embed_w, g, "encoder.timbre_encoder.embed_tokens.weight");
    q3_load_f32(m->timbre_embed_b, g, "encoder.timbre_encoder.embed_tokens.bias");
    q3_load_f32(m->timbre_norm, g, "encoder.timbre_encoder.norm.weight");
    for (int i = 0; i < m->timbre_cfg.n_layers; i++) {
        q3_load_layer(g, "encoder.timbre_encoder.layers." + std::to_string(i), m->timbre_layers[i]);
    }
    if (m->use_timbre_cls) q3_load_f32(m->timbre_cls, g, "encoder.timbre_encoder.special_token");
    q3_load_raw(m->text_proj_w, g, "encoder.text_projector.weight");

    dequant_to_f32(g, "null_condition_emb", m->null_emb);

    if (verbose) {
        fprintf(stderr, "[acestep-cond] loaded %s: %.1f MB, lyric(%dL) timbre(%dL%s) text_proj null_emb(%zu)\n",
                path.c_str(), cond_model_weight_bytes(m) / 1048576.0, m->lyric_cfg.n_layers, m->timbre_cfg.n_layers,
                m->use_timbre_cls ? ",CLS" : "", m->null_emb.size());
    }

    dit_gguf_close(g);
    return m;
}

void cond_model_free(CondModel * m) {
    if (!m) return;
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    delete m;
}

size_t cond_model_weight_bytes(const CondModel * m) {
    return m->weight_buf ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
}

const std::vector<float> & cond_model_null_emb(const CondModel * m) { return m->null_emb; }

// Bidirectional sliding-window mask [S, S] F16: 0 if |i-j| <= W else -inf.
static void fill_slide_mask(std::vector<uint16_t> & md, int S, int W) {
    md.resize((size_t) S * S);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int d = i - j; if (d < 0) d = -d;
            md[(size_t) i * S + j] = ggml_fp32_to_fp16(d <= W ? 0.0f : -INFINITY);
        }
}

bool cond_model_forward(CondModel * m, const float * text_hidden, int S_text, const float * lyric_embed, int S_lyric,
                        const float * timbre_feats, int S_ref, std::vector<float> & enc_hidden, int * out_enc_S) {
    const int  H          = COND_H;
    const bool has_timbre = (timbre_feats != nullptr && S_ref > 0);
    const int  S_timbre   = has_timbre ? S_ref + (m->use_timbre_cls ? 1 : 0) : 0;

    const size_t     nodes = 8192;
    ggml_init_params gp{ ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);

    // --- lyric path ---
    ggml_tensor * lyric_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S_lyric);
    ggml_set_input(lyric_pos);
    ggml_tensor * lyric_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, S_lyric);
    ggml_set_input(lyric_in);
    ggml_tensor * lyric_slide = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S_lyric, S_lyric);
    ggml_set_input(lyric_slide);

    ggml_tensor * lyric_h = q3_linear_bias(ctx, m->lyric_embed_w, m->lyric_embed_b, lyric_in);
    for (int i = 0; i < m->lyric_cfg.n_layers; i++) {
        ggml_tensor * mask = (i % 2 == 0) ? lyric_slide : nullptr;
        lyric_h = q3_build_layer(ctx, m->lyric_cfg, &m->lyric_layers[i], lyric_h, lyric_pos, mask, S_lyric);
    }
    lyric_h = q3_rms_norm_w(ctx, lyric_h, m->lyric_norm, m->lyric_cfg.rms_norm_eps);
    ggml_set_output(lyric_h);

    // --- text projection ---
    ggml_tensor * text_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, S_text);
    ggml_set_input(text_in);
    ggml_tensor * text_proj = q3_linear(ctx, m->text_proj_w, text_in);
    ggml_set_output(text_proj);

    // --- timbre path (optional) ---
    ggml_tensor * timbre_out   = nullptr;
    ggml_tensor * timbre_pos   = nullptr;
    ggml_tensor * timbre_in    = nullptr;
    ggml_tensor * timbre_slide = nullptr;
    if (has_timbre) {
        timbre_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S_timbre);
        ggml_set_input(timbre_pos);
        timbre_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, S_ref);
        ggml_set_input(timbre_in);
        timbre_slide = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S_timbre, S_timbre);
        ggml_set_input(timbre_slide);

        ggml_tensor * timbre_h = q3_linear_bias(ctx, m->timbre_embed_w, m->timbre_embed_b, timbre_in);
        if (m->use_timbre_cls) {
            ggml_tensor * cls = ggml_reshape_2d(ctx, m->timbre_cls, H, 1);
            timbre_h          = ggml_concat(ctx, cls, timbre_h, 1);
        }
        for (int i = 0; i < m->timbre_cfg.n_layers; i++) {
            ggml_tensor * mask = (i % 2 == 0) ? timbre_slide : nullptr;
            timbre_h = q3_build_layer(ctx, m->timbre_cfg, &m->timbre_layers[i], timbre_h, timbre_pos, mask, S_timbre);
        }
        timbre_h   = q3_rms_norm_w(ctx, timbre_h, m->timbre_norm, m->timbre_cfg.rms_norm_eps);
        timbre_out = ggml_cont(ctx, ggml_view_2d(ctx, timbre_h, H, 1, timbre_h->nb[1], 0));  // frame[0] / CLS
        ggml_set_output(timbre_out);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(gf, lyric_h);
    ggml_build_forward_expand(gf, text_proj);
    if (timbre_out) ggml_build_forward_expand(gf, timbre_out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-cond] forward alloc failed\n");
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    // inputs
    ggml_backend_tensor_set(lyric_in, lyric_embed, 0, (size_t) 1024 * S_lyric * sizeof(float));
    ggml_backend_tensor_set(text_in, text_hidden, 0, (size_t) 1024 * S_text * sizeof(float));
    {
        std::vector<int32_t> pos(S_lyric);
        for (int i = 0; i < S_lyric; i++) pos[i] = i;
        ggml_backend_tensor_set(lyric_pos, pos.data(), 0, (size_t) S_lyric * sizeof(int32_t));
        std::vector<uint16_t> md; fill_slide_mask(md, S_lyric, COND_W);
        ggml_backend_tensor_set(lyric_slide, md.data(), 0, md.size() * sizeof(uint16_t));
    }
    if (has_timbre) {
        ggml_backend_tensor_set(timbre_in, timbre_feats, 0, (size_t) 64 * S_ref * sizeof(float));
        std::vector<int32_t> pos(S_timbre);
        for (int i = 0; i < S_timbre; i++) pos[i] = i;
        ggml_backend_tensor_set(timbre_pos, pos.data(), 0, (size_t) S_timbre * sizeof(int32_t));
        std::vector<uint16_t> md; fill_slide_mask(md, S_timbre, COND_W);
        ggml_backend_tensor_set(timbre_slide, md.data(), 0, md.size() * sizeof(uint16_t));
    }

    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    // pack: [lyric | timbre[0:1] | text_proj]
    const int S_timbre_out = has_timbre ? 1 : 0;
    const int S_total      = S_lyric + S_timbre_out + S_text;
    enc_hidden.resize((size_t) H * S_total);
    *out_enc_S = S_total;

    size_t off = 0;
    ggml_backend_tensor_get(lyric_h, enc_hidden.data() + off * H, 0, (size_t) H * S_lyric * sizeof(float));
    off += S_lyric;
    if (timbre_out) {
        ggml_backend_tensor_get(timbre_out, enc_hidden.data() + off * H, 0, (size_t) H * sizeof(float));
        off += 1;
    }
    ggml_backend_tensor_get(text_proj, enc_hidden.data() + off * H, 0, (size_t) H * S_text * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return true;
}

} // namespace tts_cpp::acestep
