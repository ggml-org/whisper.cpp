#include "textenc_ggml.h"

#include "qwen3_block.h"  // shared Qwen3 loaders + graph builders (uses DitGGUF IO)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ACE-Step text encoder (Qwen3-Embedding). Port from acestep.cpp/src/qwen3-enc.h.
// Thin wrapper over the shared Qwen3 block: vocab lookup -> N causal layers ->
// final RMSNorm. Correctness-first (F32 attention, separate Q/K/V, NEOX RoPE).

namespace tts_cpp::acestep {

struct TextEncModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    TextEncConfig cfg;
    Qwen3Config   q3;  // builder view of cfg

    ggml_tensor *           embed_tokens = nullptr;  // [H, V]
    ggml_tensor *           final_norm   = nullptr;  // [H] F32
    std::vector<Qwen3Layer> layers;
};

static Qwen3Config to_q3(const TextEncConfig & c) {
    Qwen3Config q;
    q.hidden_size       = c.hidden_size;
    q.intermediate_size = c.intermediate_size;
    q.n_heads           = c.n_heads;
    q.n_kv_heads        = c.n_kv_heads;
    q.head_dim          = c.head_dim;
    q.n_layers          = c.n_layers;
    q.rope_theta        = c.rope_theta;
    q.rms_norm_eps      = c.rms_norm_eps;
    q.is_causal         = c.is_causal;
    return q;
}

TextEncModel * textenc_model_load(const std::string & path, ggml_backend_t backend, bool verbose) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) {
        fprintf(stderr, "[acestep-txt] failed to parse %s\n", path.c_str());
        return nullptr;
    }

    TextEncModel * m = new TextEncModel();
    m->backend       = backend;
    m->q3            = to_q3(m->cfg);
    const Qwen3Config & c = m->q3;

    const size_t     n_tensors = (size_t) 2 + (size_t) c.n_layers * 11 + 8;
    ggml_init_params ip{ ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc=*/true };
    m->weight_ctx = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    m->embed_tokens = q3_create_like(ctx, g, "embed_tokens.weight");
    m->final_norm   = q3_create_f32_like(ctx, g, "norm.weight");
    m->layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        q3_create_layer(ctx, g, "layers." + std::to_string(i), m->layers[i]);
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-txt] failed to allocate weight buffer\n");
        ggml_free(ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    q3_load_raw(m->embed_tokens, g, "embed_tokens.weight");
    q3_load_f32(m->final_norm, g, "norm.weight");
    for (int i = 0; i < c.n_layers; i++) {
        q3_load_layer(g, "layers." + std::to_string(i), m->layers[i]);
    }

    if (verbose) {
        fprintf(stderr, "[acestep-txt] loaded %s: %.1f MB, %d layers H=%d Nh=%d/%d D=%d\n", path.c_str(),
                textenc_model_weight_bytes(m) / 1048576.0, c.n_layers, c.hidden_size, c.n_heads, c.n_kv_heads,
                c.head_dim);
    }

    dit_gguf_close(g);
    return m;
}

void textenc_model_free(TextEncModel * m) {
    if (!m) return;
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    delete m;
}

const TextEncConfig & textenc_model_config(const TextEncModel * m) { return m->cfg; }

size_t textenc_model_weight_bytes(const TextEncModel * m) {
    return m->weight_buf ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
}

bool textenc_model_forward(TextEncModel * m, const int32_t * token_ids, int S, std::vector<float> & hidden_out) {
    const Qwen3Config & c = m->q3;
    const int           H = c.hidden_size;

    const size_t     nodes = 4096;
    ggml_init_params gp{ ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);

    ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(t_ids);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(positions);
    ggml_tensor * mask = nullptr;
    if (c.is_causal) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, S);
        ggml_set_input(mask);
    }

    ggml_tensor * hidden = ggml_get_rows(ctx, m->embed_tokens, t_ids);  // [H, S]
    for (int i = 0; i < c.n_layers; i++) {
        hidden = q3_build_layer(ctx, c, &m->layers[i], hidden, positions, mask, S);
    }
    ggml_tensor * out = q3_rms_norm_w(ctx, hidden, m->final_norm, c.rms_norm_eps);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-txt] forward alloc failed (S=%d)\n", S);
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(t_ids, token_ids, 0, (size_t) S * sizeof(int32_t));
    std::vector<int32_t> pos(S);
    for (int i = 0; i < S; i++) pos[i] = i;
    ggml_backend_tensor_set(positions, pos.data(), 0, (size_t) S * sizeof(int32_t));
    if (mask) {
        std::vector<uint16_t> md((size_t) S * S);
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
                md[(size_t) i * S + j] = ggml_fp32_to_fp16((j <= i) ? 0.0f : -INFINITY);
        ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(uint16_t));
    }

    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    hidden_out.resize((size_t) H * S);
    ggml_backend_tensor_get(out, hidden_out.data(), 0, (size_t) H * S * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return true;
}

bool textenc_model_embed_lookup(TextEncModel * m, const int32_t * token_ids, int S, std::vector<float> & embed_out) {
    const int H = m->q3.hidden_size;

    const size_t     nodes = 64;
    ggml_init_params gp{ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);

    ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(t_ids);
    ggml_tensor * out = ggml_get_rows(ctx, m->embed_tokens, t_ids);  // [H, S]
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-txt] embed lookup alloc failed (S=%d)\n", S);
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(t_ids, token_ids, 0, (size_t) S * sizeof(int32_t));
    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    embed_out.resize((size_t) H * S);
    ggml_backend_tensor_get(out, embed_out.data(), 0, (size_t) H * S * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return true;
}

} // namespace tts_cpp::acestep
