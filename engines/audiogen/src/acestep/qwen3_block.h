#pragma once

// Shared Qwen3 transformer block. The ACE-Step text encoder
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
    // Matmul accumulation/arithmetic precision, applied to every matmul the block
    // builds: the four projections, the MLP, and both attention matmuls.
    //
    // GGML_PREC_F32 is required for models whose activations leave fp16 range, otherwise
    // fp16-based GPU matmul paths saturate them to inf at 65504. The ACE-Step LM reaches
    // ~1.9e6 and produces repetitive, robotic audio without it; the cond encoder reaches
    // ~4.2e5 at one hidden channel and, unfixed, returns rows of zeros carrying a single
    // NaN -- enough to silence a whole song through the DiT cross-attention.
    // The text encoder and detokenizer stay at GGML_PREC_DEFAULT. That is an untested
    // assumption about their activation ranges, not a measurement: check before trusting it.
    ggml_prec prec = GGML_PREC_DEFAULT;
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

// map_buf non-null (CPU backend) maps the verbatim weight in-place onto the GGUF
// mmap; its q3_load_raw then becomes a no-op. Null on the GPU path (allocate +
// upload as before). See dit_gguf_cpu_map_buffer.
static inline ggml_tensor * q3_create_like(ggml_context * ctx, const DitGGUF & g, const std::string & name,
                                           ggml_backend_buffer_t map_buf = nullptr) {
    ggml_tensor * mt = dit_gmeta(g, name);
    if (!mt) {
        fprintf(stderr, "[qwen3] missing tensor: %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(ctx, mt->type, ggml_n_dims(mt), mt->ne);
    ggml_set_name(t, name.c_str());
    if (map_buf) dit_gguf_map_tensor(t, g, name, map_buf);
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

// A tensor already backed by g's mmap (create_like mapped it) needs no copy;
// only allocated tensors do. Derived per-tensor via dit_gguf_is_mapped so there
// is no separate `mapped` flag to drift out of sync and memcpy into a PROT_READ
// page (SIGSEGV) or leave an allocated tensor unuploaded (silent garbage).
static inline void q3_load_raw(ggml_tensor * dst, const DitGGUF & g, const std::string & name) {
    if (!dst || dit_gguf_is_mapped(dst, g)) return;
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
    } else if (ggml_is_quantized(mt->type)) {
        // Same dequantisation ggml_cast would do, hoisted to load time.
        const ggml_type_traits * tt = ggml_get_type_traits(mt->type);
        if (!tt || !tt->to_float) {
            fprintf(stderr, "[qwen3] load_f32: no dequantiser for %s\n", name.c_str());
            return;
        }
        tt->to_float(s, w.data(), (int64_t) n);
    } else {
        fprintf(stderr, "[qwen3] load_f32: unsupported type for %s\n", name.c_str());
        return;
    }
    ggml_backend_tensor_set(dst, w.data(), 0, n * sizeof(float));
}

// Create the 11 per-layer weight tensors under `prefix` (e.g. "layers.0").
// map_buf (CPU) maps the 7 verbatim proj weights in-place; the 4 F32 norms are
// always allocated + converted.
static inline void q3_create_layer(ggml_context * ctx, const DitGGUF & g, const std::string & prefix,
                                   Qwen3Layer & ly, ggml_backend_buffer_t map_buf = nullptr) {
    ly.input_norm = q3_create_f32_like(ctx, g, prefix + ".input_layernorm.weight");
    ly.post_norm  = q3_create_f32_like(ctx, g, prefix + ".post_attention_layernorm.weight");
    ly.q_proj     = q3_create_like(ctx, g, prefix + ".self_attn.q_proj.weight", map_buf);
    ly.k_proj     = q3_create_like(ctx, g, prefix + ".self_attn.k_proj.weight", map_buf);
    ly.v_proj     = q3_create_like(ctx, g, prefix + ".self_attn.v_proj.weight", map_buf);
    ly.o_proj     = q3_create_like(ctx, g, prefix + ".self_attn.o_proj.weight", map_buf);
    ly.q_norm     = q3_create_f32_like(ctx, g, prefix + ".self_attn.q_norm.weight");
    ly.k_norm     = q3_create_f32_like(ctx, g, prefix + ".self_attn.k_norm.weight");
    ly.gate_proj  = q3_create_like(ctx, g, prefix + ".mlp.gate_proj.weight", map_buf);
    ly.up_proj    = q3_create_like(ctx, g, prefix + ".mlp.up_proj.weight", map_buf);
    ly.down_proj  = q3_create_like(ctx, g, prefix + ".mlp.down_proj.weight", map_buf);
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

static inline ggml_tensor * q3_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x,
                                      ggml_prec prec = GGML_PREC_DEFAULT) {
    ggml_tensor * t = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(t, prec);
    return t;
}

static inline ggml_tensor * q3_linear_bias(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
                                           ggml_prec prec = GGML_PREC_DEFAULT) {
    return ggml_add(ctx, q3_linear(ctx, w, x, prec), q3_as_f32(ctx, b));
}

// Attention. q[D,S,Nh], k[D,S,Nkv], v[D,S,Nkv] -> [D, Nh, S].
//
// Both matmuls run at `prec`, which the caller supplies; softmax exponentiates whatever
// error the scores carry, so this is not a knob to set casually. It is nonetheless a
// caller decision rather than a fixed F32: see Qwen3Config::prec for who asks for what.
static inline ggml_tensor * q3_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                    ggml_tensor * mask, float scale, ggml_prec prec = GGML_PREC_DEFAULT) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(scores, prec);
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    ggml_mul_mat_set_prec(out, prec);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static inline ggml_tensor * q3_build_self_attn(ggml_context * ctx, const Qwen3Config & c, Qwen3Layer * ly,
                                               ggml_tensor * x, ggml_tensor * positions, ggml_tensor * mask, int S) {
    const int D   = c.head_dim;
    const int Nh  = c.n_heads;
    const int Nkv = c.n_kv_heads;

    ggml_tensor * q = q3_linear(ctx, ly->q_proj, x, c.prec);
    ggml_tensor * k = q3_linear(ctx, ly->k_proj, x, c.prec);
    ggml_tensor * v = q3_linear(ctx, ly->v_proj, x, c.prec);

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
    ggml_tensor * attn  = q3_attn(ctx, q, k, v, mask, scale, c.prec);
    attn                = ggml_reshape_2d(ctx, attn, (int64_t) Nh * D, S);
    return q3_linear(ctx, ly->o_proj, attn, c.prec);
}

static inline ggml_tensor * q3_build_mlp(ggml_context * ctx, Qwen3Layer * ly, ggml_tensor * x,
                                         ggml_prec prec = GGML_PREC_DEFAULT) {
    ggml_tensor * gate = q3_linear(ctx, ly->gate_proj, x, prec);
    ggml_tensor * up   = q3_linear(ctx, ly->up_proj, x, prec);
    ggml_tensor * ff   = ggml_swiglu_split(ctx, gate, up);
    return q3_linear(ctx, ly->down_proj, ff, prec);
}

// One layer: hidden [H, S] -> [H, S]. mask nullptr = full attention.
static inline ggml_tensor * q3_build_layer(ggml_context * ctx, const Qwen3Config & c, Qwen3Layer * ly,
                                           ggml_tensor * hidden, ggml_tensor * positions, ggml_tensor * mask, int S) {
    ggml_tensor * norm = q3_rms_norm_w(ctx, hidden, ly->input_norm, c.rms_norm_eps);
    ggml_tensor * attn = q3_build_self_attn(ctx, c, ly, norm, positions, mask, S);
    hidden             = ggml_add(ctx, hidden, attn);

    norm              = q3_rms_norm_w(ctx, hidden, ly->post_norm, c.rms_norm_eps);
    ggml_tensor * mlp = q3_build_mlp(ctx, ly, norm, c.prec);
    hidden            = ggml_add(ctx, hidden, mlp);
    return hidden;
}

} // namespace tts_cpp::acestep
