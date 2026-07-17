#pragma once

// Shared Qwen3 transformer block (QVAC-21921). The ACE-Step text encoder
// (Qwen3-Embedding), lyric encoder and timbre encoder are all the same Qwen3
// backbone: RMSNorm -> GQA self-attn (per-head QK-norm + NEOX RoPE) -> O proj,
// then RMSNorm -> SwiGLU MLP. Causality/windowing is expressed purely through
// the attention mask passed per layer (nullptr = full/bidirectional).
//
// Weights are loaded through the generic DitGGUF IO (mmap + metadata). Q/K/V and
// gate/up are kept separate (no fusion). CPU target: F32 soft_max attention.
//
// Layout: hidden [H, S] == ggml ne[0]=H, ne[1]=S.

#include "dit_gguf.h"  // DitGGUF, dit_gmeta, dit_gdata

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct Qwen3Config {
    int   hidden_size       = 0;
    int   intermediate_size = 0;
    int   n_heads           = 0;
    int   n_kv_heads        = 0;
    int   head_dim          = 0;
    int   n_layers          = 0;
    float rope_theta        = 1000000.0f;
    float rms_norm_eps      = 1e-6f;
    bool  is_causal         = false;
};

struct Qwen3Layer {
    ggml_tensor * input_norm = nullptr;  // [H] F32
    ggml_tensor * post_norm  = nullptr;  // [H] F32
    ggml_tensor * q_proj     = nullptr;  // [H, Nh*D]
    ggml_tensor * k_proj     = nullptr;  // [H, Nkv*D]
    ggml_tensor * v_proj     = nullptr;  // [H, Nkv*D]
    ggml_tensor * o_proj     = nullptr;  // [Nh*D, H]
    ggml_tensor * q_norm     = nullptr;  // [D] F32
    ggml_tensor * k_norm     = nullptr;  // [D] F32
    ggml_tensor * gate_proj  = nullptr;  // [H, FFN]
    ggml_tensor * up_proj    = nullptr;  // [H, FFN]
    ggml_tensor * down_proj  = nullptr;  // [FFN, H]
};

// ------------------------------------------------------------------ loaders
static inline float q3_bf16_to_f32(uint16_t v) {
    ggml_bf16_t b;
    b.bits = v;
    return ggml_bf16_to_fp32(b);
}

static inline ggml_tensor * q3_create_like(ggml_context * ctx, const DitGGUF & g, const std::string & name) {
    ggml_tensor * mt = dit_gmeta(g, name);
    if (!mt) {
        fprintf(stderr, "[qwen3] missing tensor: %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(ctx, mt->type, ggml_n_dims(mt), mt->ne);
    ggml_set_name(t, name.c_str());
    return t;
}

static inline ggml_tensor * q3_create_f32_like(ggml_context * ctx, const DitGGUF & g, const std::string & name) {
    ggml_tensor * mt = dit_gmeta(g, name);
    if (!mt) {
        fprintf(stderr, "[qwen3] missing tensor: %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(mt), mt->ne);
    ggml_set_name(t, name.c_str());
    return t;
}

static inline void q3_load_raw(ggml_tensor * dst, const DitGGUF & g, const std::string & name) {
    if (!dst) return;
    const void *  src = dit_gdata(g, name);
    ggml_tensor * mt  = dit_gmeta(g, name);
    if (!src || !mt) {
        fprintf(stderr, "[qwen3] cannot load %s\n", name.c_str());
        return;
    }
    ggml_backend_tensor_set(dst, src, 0, ggml_nbytes(mt));
}

static inline void q3_load_f32(ggml_tensor * dst, const DitGGUF & g, const std::string & name) {
    if (!dst) return;
    ggml_tensor * mt = dit_gmeta(g, name);
    const void *  s  = dit_gdata(g, name);
    if (!mt || !s) {
        fprintf(stderr, "[qwen3] cannot load %s\n", name.c_str());
        return;
    }
    const size_t       n = ggml_nelements(mt);
    std::vector<float> w(n);
    if (mt->type == GGML_TYPE_F32) {
        std::memcpy(w.data(), s, n * sizeof(float));
    } else if (mt->type == GGML_TYPE_F16) {
        const ggml_fp16_t * p = (const ggml_fp16_t *) s;
        for (size_t i = 0; i < n; i++) w[i] = ggml_fp16_to_fp32(p[i]);
    } else if (mt->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) s;
        for (size_t i = 0; i < n; i++) w[i] = q3_bf16_to_f32(p[i]);
    } else {
        fprintf(stderr, "[qwen3] load_f32: unsupported type for %s\n", name.c_str());
        return;
    }
    ggml_backend_tensor_set(dst, w.data(), 0, n * sizeof(float));
}

// Create the 11 per-layer weight tensors under `prefix` (e.g. "layers.0").
static inline void q3_create_layer(ggml_context * ctx, const DitGGUF & g, const std::string & prefix,
                                   Qwen3Layer & ly) {
    ly.input_norm = q3_create_f32_like(ctx, g, prefix + ".input_layernorm.weight");
    ly.post_norm  = q3_create_f32_like(ctx, g, prefix + ".post_attention_layernorm.weight");
    ly.q_proj     = q3_create_like(ctx, g, prefix + ".self_attn.q_proj.weight");
    ly.k_proj     = q3_create_like(ctx, g, prefix + ".self_attn.k_proj.weight");
    ly.v_proj     = q3_create_like(ctx, g, prefix + ".self_attn.v_proj.weight");
    ly.o_proj     = q3_create_like(ctx, g, prefix + ".self_attn.o_proj.weight");
    ly.q_norm     = q3_create_f32_like(ctx, g, prefix + ".self_attn.q_norm.weight");
    ly.k_norm     = q3_create_f32_like(ctx, g, prefix + ".self_attn.k_norm.weight");
    ly.gate_proj  = q3_create_like(ctx, g, prefix + ".mlp.gate_proj.weight");
    ly.up_proj    = q3_create_like(ctx, g, prefix + ".mlp.up_proj.weight");
    ly.down_proj  = q3_create_like(ctx, g, prefix + ".mlp.down_proj.weight");
}

static inline void q3_load_layer(const DitGGUF & g, const std::string & prefix, Qwen3Layer & ly) {
    q3_load_f32(ly.input_norm, g, prefix + ".input_layernorm.weight");
    q3_load_f32(ly.post_norm, g, prefix + ".post_attention_layernorm.weight");
    q3_load_raw(ly.q_proj, g, prefix + ".self_attn.q_proj.weight");
    q3_load_raw(ly.k_proj, g, prefix + ".self_attn.k_proj.weight");
    q3_load_raw(ly.v_proj, g, prefix + ".self_attn.v_proj.weight");
    q3_load_raw(ly.o_proj, g, prefix + ".self_attn.o_proj.weight");
    q3_load_f32(ly.q_norm, g, prefix + ".self_attn.q_norm.weight");
    q3_load_f32(ly.k_norm, g, prefix + ".self_attn.k_norm.weight");
    q3_load_raw(ly.gate_proj, g, prefix + ".mlp.gate_proj.weight");
    q3_load_raw(ly.up_proj, g, prefix + ".mlp.up_proj.weight");
    q3_load_raw(ly.down_proj, g, prefix + ".mlp.down_proj.weight");
}

// ------------------------------------------------------------------ graph ops
static inline ggml_tensor * q3_as_f32(ggml_context * ctx, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

static inline ggml_tensor * q3_rms_norm_w(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), q3_as_f32(ctx, w));
}

static inline ggml_tensor * q3_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

static inline ggml_tensor * q3_linear_bias(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return ggml_add(ctx, ggml_mul_mat(ctx, w, x), q3_as_f32(ctx, b));
}

// F32 attention. q[D,S,Nh], k[D,S,Nkv], v[D,S,Nkv] -> [D, Nh, S].
static inline ggml_tensor * q3_attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                        ggml_tensor * mask, float scale) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static inline ggml_tensor * q3_build_self_attn(ggml_context * ctx, const Qwen3Config & c, Qwen3Layer * ly,
                                               ggml_tensor * x, ggml_tensor * positions, ggml_tensor * mask, int S) {
    const int D   = c.head_dim;
    const int Nh  = c.n_heads;
    const int Nkv = c.n_kv_heads;

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

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn  = q3_attn_f32(ctx, q, k, v, mask, scale);
    attn                = ggml_reshape_2d(ctx, attn, Nh * D, S);
    return q3_linear(ctx, ly->o_proj, attn);
}

static inline ggml_tensor * q3_build_mlp(ggml_context * ctx, Qwen3Layer * ly, ggml_tensor * x) {
    ggml_tensor * gate = q3_linear(ctx, ly->gate_proj, x);
    ggml_tensor * up   = q3_linear(ctx, ly->up_proj, x);
    ggml_tensor * ff   = ggml_swiglu_split(ctx, gate, up);
    return q3_linear(ctx, ly->down_proj, ff);
}

// One layer: hidden [H, S] -> [H, S]. mask nullptr = full attention.
static inline ggml_tensor * q3_build_layer(ggml_context * ctx, const Qwen3Config & c, Qwen3Layer * ly,
                                           ggml_tensor * hidden, ggml_tensor * positions, ggml_tensor * mask, int S) {
    ggml_tensor * norm = q3_rms_norm_w(ctx, hidden, ly->input_norm, c.rms_norm_eps);
    ggml_tensor * attn = q3_build_self_attn(ctx, c, ly, norm, positions, mask, S);
    hidden             = ggml_add(ctx, hidden, attn);

    norm              = q3_rms_norm_w(ctx, hidden, ly->post_norm, c.rms_norm_eps);
    ggml_tensor * mlp = q3_build_mlp(ctx, ly, norm);
    hidden            = ggml_add(ctx, hidden, mlp);
    return hidden;
}

} // namespace tts_cpp::acestep
