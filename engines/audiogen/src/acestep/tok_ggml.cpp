#include "tok_ggml.h"

#include "qwen3_block.h"  // shared Qwen3 loaders + builders + DitGGUF IO

#include "ggml.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ACE-Step FSQ tokenizer. Port from acestep.cpp/src/fsq-tok.h.

namespace tts_cpp::acestep {

static constexpr int TOK_H       = 2048;
static constexpr int TOK_IN_CH   = 64;
static constexpr int FSQ_NDIMS   = 6;
static constexpr int POOL        = 5;         // 5 latent frames -> 1 code
static constexpr int GROUP_S     = POOL + 1;  // CLS + 5 patches
static const int     FSQ_LEVELS[FSQ_NDIMS] = { 8, 8, 8, 5, 5, 5 };

// FSQ encode: matches vector_quantize_pytorch FSQ.symmetry_preserving_bound()
// + codes_to_indices(): code_d = floor((L-1) * (tanh(x_d) + 1) / 2 + 0.5).
int fsq_encode_index(const float * raw_vals) {
    int index  = 0;
    int stride = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        const int   L    = FSQ_LEVELS[d];
        const float t    = tanhf(raw_vals[d]);
        int         code = (int) floorf((float) (L - 1) * (t + 1.0f) / 2.0f + 0.5f);
        if (code < 0) code = 0;
        if (code >= L) code = L - 1;
        index += code * stride;
        stride *= L;
    }
    return index;
}

static Qwen3Config tok_config() {
    Qwen3Config c;
    c.hidden_size       = 2048;
    c.intermediate_size = 6144;
    c.n_heads           = 16;
    c.n_kv_heads        = 8;
    c.head_dim          = 128;
    c.n_layers          = 2;
    c.rope_theta        = 1000000.0f;
    c.rms_norm_eps      = 1e-6f;
    c.is_causal         = false;
    return c;
}

struct TokModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    Qwen3Config             cfg;
    std::vector<Qwen3Layer> layers;

    ggml_tensor * proj_w      = nullptr;  // [2048, 64]
    ggml_tensor * proj_b      = nullptr;  // [2048] F32
    ggml_tensor * embed_w     = nullptr;  // [2048, 2048]
    ggml_tensor * embed_b     = nullptr;  // [2048] F32
    ggml_tensor * special_tok = nullptr;  // [2048] (CLS token)
    ggml_tensor * norm        = nullptr;  // [2048] F32
    ggml_tensor * fsq_in_w    = nullptr;  // [6, 2048]
    ggml_tensor * fsq_in_b    = nullptr;  // [6] F32

    // First POOL silence frames, CPU-side, for padding the trailing group.
    std::vector<float> silence_pad;

    // CPU map-in-place: verbatim weights backed by `gguf`'s mmap via `map_buf`.
    DitGGUF               gguf;
    ggml_backend_buffer_t map_buf      = nullptr;
    bool                  mapped       = false;
    size_t                mapped_bytes = 0;
};

static bool tok_load_silence_pad(TokModel * m, const DitGGUF & g) {
    const ggml_tensor * meta = dit_gmeta(g, "silence_latent");
    const void *        data = dit_gdata(g, "silence_latent");
    if (!meta || !data || meta->type != GGML_TYPE_F32 ||
        ggml_nelements(meta) < (int64_t) POOL * TOK_IN_CH) {
        fprintf(stderr, "[acestep-tok] silence_latent missing or too short\n");
        return false;
    }
    m->silence_pad.resize((size_t) POOL * TOK_IN_CH);
    memcpy(m->silence_pad.data(), data, m->silence_pad.size() * sizeof(float));
    return true;
}

TokModel * tok_model_load(const std::string & path, ggml_backend_t backend, bool verbose) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) {
        fprintf(stderr, "[acestep-tok] failed to parse %s\n", path.c_str());
        return nullptr;
    }
    dit_gguf_require_hexagon_weights(g, path, backend, "acestep-tok");

    TokModel * m = new TokModel();
    m->backend   = backend;
    m->cfg       = tok_config();
    m->layers.resize(m->cfg.n_layers);

    if (!tok_load_silence_pad(m, g)) {
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    const bool            mapped  = dit_gguf_backend_maps_weights(backend);
    ggml_backend_buffer_t map_buf = mapped ? dit_gguf_cpu_map_buffer(g) : nullptr;

    const size_t n_tensors = (size_t) m->cfg.n_layers * 11 + 12;
    ggml_init_params ip{ ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc=*/true };
    m->weight_ctx      = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    // The auxiliary tensors (biases, CLS token, FSQ projector) are tiny;
    // materialise them as F32 like the detokenizer does so the graph never
    // re-dequantises them per group.
    m->proj_w      = q3_create_like(ctx, g, "tokenizer.audio_acoustic_proj.weight", map_buf);
    m->proj_b      = q3_create_f32_like(ctx, g, "tokenizer.audio_acoustic_proj.bias");
    m->embed_w     = q3_create_like(ctx, g, "tokenizer.attention_pooler.embed_tokens.weight", map_buf);
    m->embed_b     = q3_create_f32_like(ctx, g, "tokenizer.attention_pooler.embed_tokens.bias");
    m->special_tok = q3_create_f32_like(ctx, g, "tokenizer.attention_pooler.special_token");
    m->norm        = q3_create_f32_like(ctx, g, "tokenizer.attention_pooler.norm.weight");
    m->fsq_in_w    = q3_create_f32_like(ctx, g, "tokenizer.quantizer.project_in.weight");
    m->fsq_in_b    = q3_create_f32_like(ctx, g, "tokenizer.quantizer.project_in.bias");
    for (int i = 0; i < m->cfg.n_layers; i++) {
        q3_create_layer(ctx, g, "tokenizer.attention_pooler.layers." + std::to_string(i), m->layers[i], map_buf);
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, dit_gguf_weight_buffer_type(backend));
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-tok] failed to allocate weight buffer\n");
        if (map_buf) ggml_backend_buffer_free(map_buf);
        ggml_free(ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    q3_load_raw(m->proj_w, g, "tokenizer.audio_acoustic_proj.weight");
    q3_load_f32(m->proj_b, g, "tokenizer.audio_acoustic_proj.bias");
    q3_load_raw(m->embed_w, g, "tokenizer.attention_pooler.embed_tokens.weight");
    q3_load_f32(m->embed_b, g, "tokenizer.attention_pooler.embed_tokens.bias");
    q3_load_f32(m->special_tok, g, "tokenizer.attention_pooler.special_token");
    q3_load_f32(m->norm, g, "tokenizer.attention_pooler.norm.weight");
    q3_load_f32(m->fsq_in_w, g, "tokenizer.quantizer.project_in.weight");
    q3_load_f32(m->fsq_in_b, g, "tokenizer.quantizer.project_in.bias");
    for (int i = 0; i < m->cfg.n_layers; i++) {
        q3_load_layer(g, "tokenizer.attention_pooler.layers." + std::to_string(i), m->layers[i]);
    }
    m->mapped_bytes = mapped ? dit_gguf_mapped_bytes(ctx, g) : 0;

    if (verbose) {
        fprintf(stderr, "[acestep-tok] loaded %s: %.1f MB, proj(64->2048) + %dL encoder(S=6) + FSQ(2048->6)\n",
                path.c_str(), tok_model_weight_bytes(m) / 1048576.0, m->cfg.n_layers);
    }

    if (mapped) {
        m->mapped  = true;
        m->map_buf = map_buf;
        m->gguf    = g;  // keep the mmap alive; mapped weights point into it
    } else {
        dit_gguf_close(g);
    }
    return m;
}

void tok_model_free(TokModel * m) {
    if (!m) return;
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    if (m->map_buf) ggml_backend_buffer_free(m->map_buf);
    if (m->mapped) dit_gguf_close(m->gguf);
    delete m;
}

size_t tok_model_weight_bytes(const TokModel * m) {
    if (!m) return 0;
    const size_t alloc = m->weight_buf ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
    return alloc + m->mapped_bytes;
}

static void tok_pad_with_silence(const TokModel * m, const float * latents, int T_25Hz, int T_padded,
                                 std::vector<float> & padded) {
    padded.assign((size_t) T_padded * TOK_IN_CH, 0.0f);
    memcpy(padded.data(), latents, (size_t) T_25Hz * TOK_IN_CH * sizeof(float));
    const int pad = T_padded - T_25Hz;
    if (pad > 0) {
        memcpy(padded.data() + (size_t) T_25Hz * TOK_IN_CH, m->silence_pad.data(),
               (size_t) pad * TOK_IN_CH * sizeof(float));
    }
}

int tok_model_encode(TokModel * m, const float * latents, int T_25Hz, std::vector<int> & codes_out) {
    codes_out.clear();
    if (!m || !latents || T_25Hz <= 0) return -1;

    const int H        = TOK_H;
    const int T_padded = ((T_25Hz + POOL - 1) / POOL) * POOL;
    const int T_5Hz    = T_padded / POOL;

    std::vector<float> padded;
    tok_pad_with_silence(m, latents, T_25Hz, T_padded, padded);

    // Build the per-group graph once (S = 6 fixed), reuse across groups.
    const size_t     nodes = 4096;
    ggml_init_params gp{ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);

    ggml_tensor * tok_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, TOK_IN_CH, POOL);
    ggml_set_input(tok_in);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GROUP_S);
    ggml_set_input(positions);

    // audio_acoustic_proj + embed_tokens: [64,5] -> [2048,5]
    ggml_tensor * projected = q3_linear_bias(ctx, m->proj_w, m->proj_b, tok_in);
    ggml_tensor * embedded  = q3_linear_bias(ctx, m->embed_w, m->embed_b, projected);

    // Prepend the CLS token: [2048,1] ++ [2048,5] -> [2048,6]
    ggml_tensor * cls    = ggml_reshape_2d(ctx, q3_as_f32(ctx, m->special_tok), H, 1);
    ggml_tensor * hidden = ggml_concat(ctx, cls, embedded, 1);

    for (int i = 0; i < m->cfg.n_layers; i++) {
        hidden = q3_build_layer(ctx, m->cfg, &m->layers[i], hidden, positions, nullptr, GROUP_S);
    }
    hidden = q3_rms_norm_w(ctx, hidden, m->norm, m->cfg.rms_norm_eps);

    // CLS column -> project_in: [2048,1] -> [6,1]
    ggml_tensor * cls_out  = ggml_view_2d(ctx, hidden, H, 1, hidden->nb[1], 0);
    ggml_tensor * fsq_vals = q3_linear_bias(ctx, m->fsq_in_w, m->fsq_in_b, ggml_cont(ctx, cls_out));
    ggml_set_output(fsq_vals);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(gf, fsq_vals);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-tok] forward alloc failed\n");
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return -1;
    }

    int32_t pos_data[GROUP_S] = { 0, 1, 2, 3, 4, 5 };

    codes_out.resize(T_5Hz);
    float fsq_buf[FSQ_NDIMS];
    for (int g = 0; g < T_5Hz; g++) {
        // Positions can share buffers with intermediates; re-set every step.
        ggml_backend_tensor_set(positions, pos_data, 0, GROUP_S * sizeof(int32_t));
        ggml_backend_tensor_set(tok_in, padded.data() + (size_t) g * POOL * TOK_IN_CH, 0,
                                (size_t) POOL * TOK_IN_CH * sizeof(float));
        if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[acestep-tok] compute failed at group %d\n", g);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            codes_out.clear();
            return -1;
        }
        ggml_backend_tensor_get(fsq_vals, fsq_buf, 0, FSQ_NDIMS * sizeof(float));
        codes_out[g] = fsq_encode_index(fsq_buf);
    }

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return T_5Hz;
}

} // namespace tts_cpp::acestep
