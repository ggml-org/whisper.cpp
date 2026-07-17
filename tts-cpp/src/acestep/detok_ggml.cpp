#include "detok_ggml.h"

#include "qwen3_block.h"  // shared Qwen3 loaders + builders + DitGGUF IO

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ACE-Step FSQ detokenizer. Port from acestep.cpp/src/fsq-detok.h.

namespace tts_cpp::acestep {

static constexpr int DETOK_H     = 2048;
static constexpr int FSQ_NDIMS   = 6;
static constexpr int POOL        = 5;  // each 5Hz token -> 5 frames @ 25Hz
static const int     FSQ_LEVELS[FSQ_NDIMS] = { 8, 8, 8, 5, 5, 5 };

// FSQ decode: integer index -> 6 normalized floats in [-1, 1].
static void fsq_decode_index(int index, float * out) {
    int stride = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        int   L         = FSQ_LEVELS[d];
        int   level_idx = (index / stride) % L;
        float half_L    = (float) (L - 1) / 2.0f;
        out[d]          = (float) level_idx / half_L - 1.0f;
        stride *= L;
    }
}

static Qwen3Config detok_config() {
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

struct DetokModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    Qwen3Config             cfg;
    std::vector<Qwen3Layer> layers;

    ggml_tensor * fsq_proj_w  = nullptr;  // [2048, 6]
    ggml_tensor * fsq_proj_b  = nullptr;  // [2048] F32
    ggml_tensor * embed_w     = nullptr;  // [2048, 2048]
    ggml_tensor * embed_b     = nullptr;  // [2048] F32
    ggml_tensor * special_tok = nullptr;  // [2048, 5]
    ggml_tensor * norm        = nullptr;  // [2048] F32
    ggml_tensor * proj_out_w  = nullptr;  // [64, 2048]
    ggml_tensor * proj_out_b  = nullptr;  // [64] F32
};

DetokModel * detok_model_load(const std::string & path, ggml_backend_t backend, bool verbose) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) {
        fprintf(stderr, "[acestep-detok] failed to parse %s\n", path.c_str());
        return nullptr;
    }

    DetokModel * m = new DetokModel();
    m->backend     = backend;
    m->cfg         = detok_config();
    m->layers.resize(m->cfg.n_layers);

    const size_t n_tensors = (size_t) m->cfg.n_layers * 11 + 12;
    ggml_init_params ip{ ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc=*/true };
    m->weight_ctx      = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    m->fsq_proj_w  = q3_create_like(ctx, g, "tokenizer.quantizer.project_out.weight");
    m->fsq_proj_b  = q3_create_f32_like(ctx, g, "tokenizer.quantizer.project_out.bias");
    m->embed_w     = q3_create_like(ctx, g, "detokenizer.embed_tokens.weight");
    m->embed_b     = q3_create_f32_like(ctx, g, "detokenizer.embed_tokens.bias");
    // special_tokens is stored in the model's native type (e.g. BF16/Q8_0);
    // keep it raw and cast to F32 in the graph (q3_as_f32) like upstream.
    m->special_tok = q3_create_like(ctx, g, "detokenizer.special_tokens");
    m->norm        = q3_create_f32_like(ctx, g, "detokenizer.norm.weight");
    m->proj_out_w  = q3_create_like(ctx, g, "detokenizer.proj_out.weight");
    m->proj_out_b  = q3_create_f32_like(ctx, g, "detokenizer.proj_out.bias");
    for (int i = 0; i < m->cfg.n_layers; i++) {
        q3_create_layer(ctx, g, "detokenizer.layers." + std::to_string(i), m->layers[i]);
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-detok] failed to allocate weight buffer\n");
        ggml_free(ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    q3_load_raw(m->fsq_proj_w, g, "tokenizer.quantizer.project_out.weight");
    q3_load_f32(m->fsq_proj_b, g, "tokenizer.quantizer.project_out.bias");
    q3_load_raw(m->embed_w, g, "detokenizer.embed_tokens.weight");
    q3_load_f32(m->embed_b, g, "detokenizer.embed_tokens.bias");
    q3_load_raw(m->special_tok, g, "detokenizer.special_tokens");
    q3_load_f32(m->norm, g, "detokenizer.norm.weight");
    q3_load_raw(m->proj_out_w, g, "detokenizer.proj_out.weight");
    q3_load_f32(m->proj_out_b, g, "detokenizer.proj_out.bias");
    for (int i = 0; i < m->cfg.n_layers; i++) {
        q3_load_layer(g, "detokenizer.layers." + std::to_string(i), m->layers[i]);
    }

    if (verbose) {
        fprintf(stderr, "[acestep-detok] loaded %s: %.1f MB, FSQ(6->2048) + %dL encoder(S=5, 2048->64)\n",
                path.c_str(), detok_model_weight_bytes(m) / 1048576.0, m->cfg.n_layers);
    }

    dit_gguf_close(g);
    return m;
}

void detok_model_free(DetokModel * m) {
    if (!m) return;
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    delete m;
}

size_t detok_model_weight_bytes(const DetokModel * m) {
    return m->weight_buf ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
}

int detok_model_decode(DetokModel * m, const int * codes, int T_5Hz, float * context_out) {
    if (T_5Hz <= 0) return 0;
    const int H      = DETOK_H;
    const int P      = POOL;
    const int T_25Hz = T_5Hz * P;

    // FSQ decode all indices on CPU -> [6, T_5Hz] (per-token 6 floats).
    std::vector<float> fsq_decoded((size_t) T_5Hz * FSQ_NDIMS);
    for (int g = 0; g < T_5Hz; g++) {
        fsq_decode_index(codes[g], fsq_decoded.data() + (size_t) g * FSQ_NDIMS);
    }

    // Build the per-token graph once (S = 5 fixed), reuse across tokens.
    const size_t     nodes = 4096;
    ggml_init_params gp{ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);

    ggml_tensor * fsq_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FSQ_NDIMS);
    ggml_set_input(fsq_in);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    ggml_set_input(positions);

    // project_out: [6] -> [2048]; embed_tokens: [2048] -> [2048]
    ggml_tensor * quantized = q3_linear_bias(ctx, m->fsq_proj_w, m->fsq_proj_b, fsq_in);
    ggml_tensor * embedded  = q3_linear_bias(ctx, m->embed_w, m->embed_b, ggml_reshape_2d(ctx, quantized, H, 1));

    // broadcast [2048,1] -> [2048,5], add special_tokens [2048,5]
    ggml_tensor * special_2d  = ggml_reshape_2d(ctx, m->special_tok, H, P);
    ggml_tensor * special_f32 = q3_as_f32(ctx, special_2d);
    ggml_tensor * hidden      = ggml_add(ctx, ggml_repeat(ctx, embedded, special_f32), special_f32);

    for (int i = 0; i < m->cfg.n_layers; i++) {
        hidden = q3_build_layer(ctx, m->cfg, &m->layers[i], hidden, positions, nullptr, P);
    }
    hidden = q3_rms_norm_w(ctx, hidden, m->norm, m->cfg.rms_norm_eps);

    // proj_out: [2048,5] -> [64,5]
    ggml_tensor * output = q3_linear_bias(ctx, m->proj_out_w, m->proj_out_b, hidden);
    ggml_set_output(output);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(gf, output);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-detok] forward alloc failed\n");
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return -1;
    }

    int32_t pos_data[POOL] = { 0, 1, 2, 3, 4 };

    for (int g = 0; g < T_5Hz; g++) {
        // Positions can share buffers with intermediates; re-set every step.
        ggml_backend_tensor_set(positions, pos_data, 0, P * sizeof(int32_t));
        ggml_backend_tensor_set(fsq_in, fsq_decoded.data() + (size_t) g * FSQ_NDIMS, 0, FSQ_NDIMS * sizeof(float));
        if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[acestep-detok] compute failed at token %d\n", g);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return -1;
        }
        // output [64,5] -> context_out frames [g*5 .. g*5+4], each 64 channels.
        ggml_backend_tensor_get(output, context_out + (size_t) g * P * 64, 0, (size_t) P * 64 * sizeof(float));
    }

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return T_25Hz;
}

} // namespace tts_cpp::acestep
