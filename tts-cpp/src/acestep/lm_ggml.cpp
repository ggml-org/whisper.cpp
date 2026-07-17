#include "lm_ggml.h"

#include "qwen3_block.h"  // shared Qwen3 loaders + builders + DitGGUF IO

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ACE-Step LM core. Port from acestep.cpp/src/qwen3-lm.h, simplified for CPU:
// single KV set, per-call graph build, F32 soft_max attention, tied LM head.
// The KV cache lives in a persistent f16 buffer written via set_rows.

namespace tts_cpp::acestep {

struct LMModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    LMConfig    cfg;
    Qwen3Config q3;

    ggml_tensor *           embed_tokens = nullptr;  // [H, V] (also tied lm_head)
    ggml_tensor *           final_norm   = nullptr;  // [H] F32
    std::vector<Qwen3Layer> layers;

    // KV cache: n_sets independent caches, each per-layer [D, max_seq, Nkv] f16.
    // Indexed kv_k[set * n_layers + layer]. Set 0 is the default; CFG uses set 1
    // for the unconditional stream.
    ggml_context *             kv_ctx  = nullptr;
    ggml_backend_buffer_t      kv_buf  = nullptr;
    int                        n_sets  = 1;
    std::vector<ggml_tensor *> kv_k;
    std::vector<ggml_tensor *> kv_v;
    std::vector<int>           kv_pos;  // per set
};

static Qwen3Config to_q3(const LMConfig & c) {
    Qwen3Config q;
    q.hidden_size  = c.hidden_size;
    q.n_heads      = c.n_heads;
    q.n_kv_heads   = c.n_kv_heads;
    q.head_dim     = c.head_dim;
    q.n_layers     = c.n_layers;
    q.rope_theta   = c.rope_theta;
    q.rms_norm_eps = c.rms_norm_eps;
    q.is_causal    = true;
    return q;
}

LMModel * lm_model_load(const std::string & path, ggml_backend_t backend, int max_seq_len, bool verbose,
                        int n_kv_sets) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) {
        fprintf(stderr, "[acestep-lm] failed to parse %s\n", path.c_str());
        return nullptr;
    }

    LMModel * m = new LMModel();
    m->backend  = backend;
    m->n_sets   = n_kv_sets > 0 ? n_kv_sets : 1;

    // Derive config from tensor shapes.
    ggml_tensor * embed = dit_gmeta(g, "model.embed_tokens.weight");
    if (!embed) {
        fprintf(stderr, "[acestep-lm] missing model.embed_tokens.weight\n");
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }
    LMConfig & c   = m->cfg;
    c.hidden_size  = (int) embed->ne[0];
    c.vocab_size   = (int) embed->ne[1];
    c.head_dim     = 128;  // Qwen3 fixed
    // count layers
    int L = 0;
    while (dit_gguf_has(g, "model.layers." + std::to_string(L) + ".input_layernorm.weight")) L++;
    c.n_layers = L;
    // head counts from projection shapes (ne1 = heads*head_dim)
    ggml_tensor * qw = dit_gmeta(g, "model.layers.0.self_attn.q_proj.weight");
    ggml_tensor * kw = dit_gmeta(g, "model.layers.0.self_attn.k_proj.weight");
    c.n_heads    = (qw ? (int) qw->ne[1] : 2048) / c.head_dim;
    c.n_kv_heads = (kw ? (int) kw->ne[1] : 1024) / c.head_dim;
    c.rope_theta   = 1000000.0f;
    c.rms_norm_eps = 1e-6f;
    c.max_seq_len  = max_seq_len > 0 ? max_seq_len : 4096;
    m->q3 = to_q3(c);

    if (c.n_layers <= 0 || c.hidden_size <= 0 || c.vocab_size <= 0 || c.n_heads <= 0 || c.n_kv_heads <= 0) {
        fprintf(stderr, "[acestep-lm] bad derived config (L=%d H=%d V=%d Nh=%d Nkv=%d)\n", c.n_layers, c.hidden_size,
                c.vocab_size, c.n_heads, c.n_kv_heads);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    // Allocate + load weights.
    const size_t n_tensors = (size_t) 2 + (size_t) c.n_layers * 11 + 8;
    ggml_init_params ip{ ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc=*/true };
    m->weight_ctx = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    m->embed_tokens = q3_create_like(ctx, g, "model.embed_tokens.weight");
    m->final_norm   = q3_create_f32_like(ctx, g, "model.norm.weight");
    m->layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        q3_create_layer(ctx, g, "model.layers." + std::to_string(i), m->layers[i]);
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-lm] failed to allocate weight buffer\n");
        ggml_free(ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    q3_load_raw(m->embed_tokens, g, "model.embed_tokens.weight");
    q3_load_f32(m->final_norm, g, "model.norm.weight");
    for (int i = 0; i < c.n_layers; i++) {
        q3_load_layer(g, "model.layers." + std::to_string(i), m->layers[i]);
    }

    // KV cache: n_sets * n_layers tensors.
    const int D = c.head_dim, Nkv = c.n_kv_heads, S = c.max_seq_len, Lc = c.n_layers, NS = m->n_sets;
    ggml_init_params kp{ ggml_tensor_overhead() * (size_t) (NS * Lc * 2 + 4), nullptr, /*no_alloc=*/true };
    m->kv_ctx = ggml_init(kp);
    m->kv_k.resize((size_t) NS * Lc);
    m->kv_v.resize((size_t) NS * Lc);
    for (int s = 0; s < NS; s++) {
        for (int l = 0; l < Lc; l++) {
            int idx      = s * Lc + l;
            m->kv_k[idx] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv);
            m->kv_v[idx] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv);
            ggml_set_name(m->kv_k[idx], ("kv_k_" + std::to_string(s) + "_" + std::to_string(l)).c_str());
            ggml_set_name(m->kv_v[idx], ("kv_v_" + std::to_string(s) + "_" + std::to_string(l)).c_str());
        }
    }
    m->kv_buf = ggml_backend_alloc_ctx_tensors(m->kv_ctx, backend);
    if (!m->kv_buf) {
        fprintf(stderr, "[acestep-lm] failed to allocate KV cache\n");
        ggml_free(ctx);
        ggml_free(m->kv_ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }
    ggml_backend_buffer_clear(m->kv_buf, 0);
    m->kv_pos.assign(NS, 0);

    if (verbose) {
        const size_t kv_bytes = (size_t) NS * Lc * 2 * D * S * Nkv * sizeof(uint16_t);
        fprintf(stderr,
                "[acestep-lm] loaded %s: %.1f MB weights, %d layers H=%d V=%d Nh=%d/%d D=%d, KV %.1f MB (%d sets)\n",
                path.c_str(), lm_model_weight_bytes(m) / 1048576.0, c.n_layers, c.hidden_size, c.vocab_size, c.n_heads,
                c.n_kv_heads, c.head_dim, kv_bytes / 1048576.0, NS);
    }

    dit_gguf_close(g);
    return m;
}

void lm_model_free(LMModel * m) {
    if (!m) return;
    if (m->kv_buf) ggml_backend_buffer_free(m->kv_buf);
    if (m->kv_ctx) ggml_free(m->kv_ctx);
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    delete m;
}

const LMConfig & lm_model_config(const LMModel * m) { return m->cfg; }
size_t           lm_model_weight_bytes(const LMModel * m) {
    return m->weight_buf ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
}
int  lm_num_kv_sets(const LMModel * m) { return m->n_sets; }
void lm_reset(LMModel * m, int set) { if (set >= 0 && set < m->n_sets) m->kv_pos[set] = 0; }
int  lm_kv_pos(const LMModel * m, int set) { return (set >= 0 && set < m->n_sets) ? m->kv_pos[set] : 0; }

// KV-cache self-attention for one layer. x [H, S]. Writes S fresh rows at
// [kv_pos..kv_pos+S) then reads the [0, n_kv_pad) window. F32 attention.
static ggml_tensor * lm_attn(ggml_context * ctx, ggml_cgraph * gf, const Qwen3Config & c, Qwen3Layer * ly,
                             ggml_tensor * x, ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * kv_rows,
                             ggml_tensor * cache_k, ggml_tensor * cache_v, int n_kv_pad, int S) {
    const int D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;

    ggml_tensor * q = q3_linear(ctx, ly->q_proj, x);
    ggml_tensor * k = q3_linear(ctx, ly->k_proj, x);
    ggml_tensor * v = q3_linear(ctx, ly->v_proj, x);

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), q3_as_f32(ctx, ly->q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), q3_as_f32(ctx, ly->k_norm));

    q = ggml_rope_ext(ctx, q, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D, S, Nh]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D, S, Nkv]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [D, S, Nkv]

    // Write K,V into the persistent f16 cache at kv_rows (broadcast over Nkv).
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, k, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, v, kv_rows));

    // Read padded window [0, n_kv_pad).
    ggml_tensor * k_full = ggml_view_3d(ctx, cache_k, D, n_kv_pad, Nkv, cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * v_full = ggml_view_3d(ctx, cache_v, D, n_kv_pad, Nkv, cache_v->nb[1], cache_v->nb[2], 0);

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn  = q3_attn_f32(ctx, q, k_full, v_full, mask, scale);  // [D, Nh, S]
    attn                = ggml_reshape_2d(ctx, attn, Nh * D, S);
    return q3_linear(ctx, ly->o_proj, attn);
}

bool lm_model_forward(LMModel * m, const int32_t * token_ids, int n_tokens, std::vector<float> & logits_out, int set) {
    const Qwen3Config & c   = m->q3;
    const LMConfig &    lc  = m->cfg;
    const int           H   = c.hidden_size;
    const int           S   = n_tokens;
    if (set < 0 || set >= m->n_sets) {
        fprintf(stderr, "[acestep-lm] invalid kv set %d (n_sets=%d)\n", set, m->n_sets);
        return false;
    }
    const int           kv0 = m->kv_pos[set];
    const int           kv_len = kv0 + S;

    if (kv_len > lc.max_seq_len) {
        fprintf(stderr, "[acestep-lm] kv_len %d > max_seq %d\n", kv_len, lc.max_seq_len);
        return false;
    }
    int n_kv_pad = (int) GGML_PAD(kv_len, 256);
    if (n_kv_pad > lc.max_seq_len) n_kv_pad = lc.max_seq_len;

    const size_t nodes = 16384;
    ggml_init_params gp{ ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);

    ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(t_ids);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(positions);
    ggml_tensor * kv_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, S);
    ggml_set_input(kv_rows);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, S);
    ggml_set_input(mask);

    ggml_cgraph * gf     = ggml_new_graph_custom(ctx, nodes, false);
    ggml_tensor * hidden = ggml_get_rows(ctx, m->embed_tokens, t_ids);  // [H, S]
    for (int l = 0; l < c.n_layers; l++) {
        Qwen3Layer *  ly   = &m->layers[l];
        ggml_tensor * norm = q3_rms_norm_w(ctx, hidden, ly->input_norm, c.rms_norm_eps);
        int           idx  = set * c.n_layers + l;
        ggml_tensor * attn = lm_attn(ctx, gf, c, ly, norm, positions, mask, kv_rows, m->kv_k[idx], m->kv_v[idx],
                                     n_kv_pad, S);
        hidden             = ggml_add(ctx, hidden, attn);
        norm               = q3_rms_norm_w(ctx, hidden, ly->post_norm, c.rms_norm_eps);
        ggml_tensor * mlp  = q3_build_mlp(ctx, ly, norm);
        hidden             = ggml_add(ctx, hidden, mlp);
    }
    hidden = q3_rms_norm_w(ctx, hidden, m->final_norm, c.rms_norm_eps);
    if (S > 1) {
        hidden = ggml_view_1d(ctx, hidden, H, (int64_t) (S - 1) * H * sizeof(float));  // last token
    }
    ggml_tensor * lgt = ggml_mul_mat(ctx, m->embed_tokens, hidden);  // [V, 1]
    ggml_set_output(lgt);
    ggml_build_forward_expand(gf, lgt);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-lm] forward alloc failed (n=%d)\n", S);
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(t_ids, token_ids, 0, (size_t) S * sizeof(int32_t));
    std::vector<int32_t> pos(S);
    std::vector<int64_t> rows(S);
    for (int i = 0; i < S; i++) { pos[i] = kv0 + i; rows[i] = (int64_t) (kv0 + i); }
    ggml_backend_tensor_set(positions, pos.data(), 0, (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(kv_rows, rows.data(), 0, (size_t) S * sizeof(int64_t));

    std::vector<uint16_t> md((size_t) n_kv_pad * S);
    for (int i = 0; i < S; i++) {
        int qpos = kv0 + i;
        for (int j = 0; j < n_kv_pad; j++)
            md[(size_t) i * n_kv_pad + j] = ggml_fp32_to_fp16((j <= qpos) ? 0.0f : -INFINITY);
    }
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(uint16_t));

    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    logits_out.resize((size_t) lc.vocab_size);
    ggml_backend_tensor_get(lgt, logits_out.data(), 0, (size_t) lc.vocab_size * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    m->kv_pos[set] += S;
    return true;
}

} // namespace tts_cpp::acestep
