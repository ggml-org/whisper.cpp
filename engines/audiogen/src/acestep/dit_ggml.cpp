#include "dit_ggml.h"

#include "audio_edit.h"
#include "dit_gguf.h"
#include "fit_measure.h"
#include "philox.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// ACE-Step DiT ggml engine. First port from acestep.cpp/src/dit-graph.h.
//
// Correctness-first: Q/K/V and gate/up are loaded as separate tensors (no
// fusion yet) and attention runs the F32 soft_max path (CPU target). proj_in /
// proj_out convs are pre-permuted to 2D F32 at load time so the graph is pure
// mul_mat. The sliding-window / cross-attn masks are built by the caller
// (sampler) and passed in via DitForwardInputs, matching the header contract.
//
// Wired into the engine (engine.cpp) and parity-checked against acestep.cpp
// --dump tensors on a fixed seed: feeding upstream noise/context/enc_hidden
// reproduces the reference DiT latent (corr ~0.999).

namespace tts_cpp::acestep {

// ------------------------------------------------------------------ structs
struct DitTemb {
    ggml_tensor * linear_1_w = nullptr;
    ggml_tensor * linear_1_b = nullptr;
    ggml_tensor * linear_2_w = nullptr;
    ggml_tensor * linear_2_b = nullptr;
    ggml_tensor * time_proj_w = nullptr;
    ggml_tensor * time_proj_b = nullptr;
};

struct DitLayer {
    ggml_tensor * self_attn_norm = nullptr;
    ggml_tensor * sa_q_proj = nullptr;
    ggml_tensor * sa_k_proj = nullptr;
    ggml_tensor * sa_v_proj = nullptr;
    ggml_tensor * sa_q_norm = nullptr;
    ggml_tensor * sa_k_norm = nullptr;
    ggml_tensor * sa_o_proj = nullptr;

    ggml_tensor * cross_attn_norm = nullptr;
    ggml_tensor * ca_q_proj = nullptr;
    ggml_tensor * ca_k_proj = nullptr;
    ggml_tensor * ca_v_proj = nullptr;
    ggml_tensor * ca_q_norm = nullptr;
    ggml_tensor * ca_k_norm = nullptr;
    ggml_tensor * ca_o_proj = nullptr;

    ggml_tensor * mlp_norm = nullptr;
    ggml_tensor * gate_proj = nullptr;
    ggml_tensor * up_proj = nullptr;
    ggml_tensor * down_proj = nullptr;

    ggml_tensor * scale_shift_table = nullptr;  // [H, 6]
    int layer_type = 0;                         // 0 = sliding window, 1 = full
};

// Reused forward graph: the sampler calls dit_model_forward once per Euler step
// with identical shapes, so the built graph and its allocation are kept.
struct DitGraphCache {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor *  input = nullptr, *enc_hidden = nullptr, *t_val = nullptr,
                *  tr_val = nullptr, *positions = nullptr, *sa_mask = nullptr,
                *  ca_mask = nullptr, *output = nullptr;
    // key
    int  T = -1, N = -1, enc_S = -1, H_enc = -1;
    bool has_sa = false, has_ca = false;

    void release() {
        if (ga)  ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
        *this = DitGraphCache{};
    }
};

struct DitModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    // CPU map-in-place: when `mapped`, the bulk (quantised) weights are backed
    // directly by `gguf`'s mmap via `map_buf` instead of `weight_buf`, so they
    // cost no dirty RAM. `gguf` (and its mmap) must outlive the model, so it is
    // kept open here and closed in dit_model_free. Unused on the GPU path.
    DitGGUF               gguf;
    ggml_backend_buffer_t map_buf = nullptr;
    bool                  mapped  = false;
    size_t                mapped_bytes = 0;  // sum of mmapped weight nbytes

    bool measuring = false;  // metadata-only load: weights sized, never read

    DitConfig cfg;

    DitTemb time_embed;
    DitTemb time_embed_r;

    ggml_tensor * proj_in_w = nullptr;   // [in_ch*P, H] F32
    ggml_tensor * proj_in_b = nullptr;   // [H] F32
    ggml_tensor * cond_emb_w = nullptr;  // [H_enc, H]
    ggml_tensor * cond_emb_b = nullptr;  // [H] F32

    std::vector<DitLayer> layers;

    ggml_tensor * norm_out = nullptr;         // [H] F32
    ggml_tensor * out_scale_shift = nullptr;  // [H, 2] F32
    ggml_tensor * proj_out_w = nullptr;       // [H, out_ch*P] F32
    ggml_tensor * proj_out_b = nullptr;       // [out_ch] F32

    ggml_tensor * scalar_one = nullptr;  // [1] = 1.0f

    bool use_flash_attn = false;

    DitGraphCache graph_cache;
};

// ------------------------------------------------------------------ loaders
static float bf16_to_f32(uint16_t v) {
    ggml_bf16_t b;
    b.bits = v;
    return ggml_bf16_to_fp32(b);
}

// Create a weight tensor mirroring the GGUF meta tensor's type + shape. When
// `map_buf` is non-null (CPU backend), the tensor is also mapped in-place onto
// the GGUF mmap so no backend buffer is allocated for it and no copy is needed
// (its matching load_raw becomes a no-op). map_buf is null on the GPU path, where
// weights are allocated + uploaded as before.
static ggml_tensor * create_like(ggml_context * ctx, const DitGGUF & g, const std::string & name,
                                 ggml_backend_buffer_t map_buf = nullptr) {
    ggml_tensor * mt = dit_gmeta(g, name);
    if (!mt) {
        fprintf(stderr, "[acestep-dit] missing tensor: %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(ctx, mt->type, ggml_n_dims(mt), mt->ne);
    ggml_set_name(t, name.c_str());
    if (map_buf) dit_gguf_map_tensor(t, g, name, map_buf);
    return t;
}

// Create an F32 tensor with the same shape as the GGUF meta tensor.
static ggml_tensor * create_f32_like(ggml_context * ctx, const DitGGUF & g, const std::string & name) {
    ggml_tensor * mt = dit_gmeta(g, name);
    if (!mt) {
        fprintf(stderr, "[acestep-dit] missing tensor: %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(mt), mt->ne);
    ggml_set_name(t, name.c_str());
    return t;
}

// Upload raw bytes verbatim (dst type == GGUF type). Used for mul_mat operands
// (kept in their native quant/precision type). A tensor already backed by the
// mmap (create_like mapped it) needs no copy — derived per-tensor via
// dit_gguf_is_mapped so no caller flag can drift and memcpy into a PROT_READ page.
static void load_raw(ggml_tensor * dst, const DitGGUF & g, const std::string & name) {
    if (!dst || dit_gguf_is_mapped(dst, g)) return;
    const void * src = dit_gdata(g, name);
    ggml_tensor * mt = dit_gmeta(g, name);
    if (!src || !mt) {
        fprintf(stderr, "[acestep-dit] cannot load %s\n", name.c_str());
        return;
    }
    ggml_backend_tensor_set(dst, src, 0, ggml_nbytes(mt));
}

// Dequantise a bf16/f16/f32 source into F32 and upload. (norms, biases, tables)
static void load_f32(ggml_tensor * dst, const DitGGUF & g, const std::string & name) {
    if (!dst) return;
    ggml_tensor * mt = dit_gmeta(g, name);
    const void *  s  = dit_gdata(g, name);
    if (!mt || !s) {
        fprintf(stderr, "[acestep-dit] cannot load %s\n", name.c_str());
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
        for (size_t i = 0; i < n; i++) w[i] = bf16_to_f32(p[i]);
    } else {
        fprintf(stderr, "[acestep-dit] load_f32: unsupported type for %s\n", name.c_str());
        return;
    }
    ggml_backend_tensor_set(dst, w.data(), 0, n * sizeof(float));
}

// proj_in conv weight GGUF [P, in_ch, H] -> pre-permuted 2D [in_ch*P, H] F32
// (data[h*in_ch*P + p*in_ch + ic] = src(h*P*in_ch + ic*P + p)).
static void load_proj_in(ggml_tensor * dst, const DitGGUF & g, const std::string & name, int H, int in_ch, int P) {
    if (!dst) return;
    ggml_tensor * mt = dit_gmeta(g, name);
    const void *  s  = dit_gdata(g, name);
    if (!mt || !s) return;
    std::vector<float> data((size_t) in_ch * P * H);
    auto cvt = [&](auto rd) {
        for (int h = 0; h < H; h++)
            for (int ic = 0; ic < in_ch; ic++)
                for (int p = 0; p < P; p++)
                    data[(size_t) h * in_ch * P + (size_t) p * in_ch + ic] = rd(h * P * in_ch + ic * P + p);
    };
    if (mt->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) s;
        cvt([&](int i) { return bf16_to_f32(p[i]); });
    } else if (mt->type == GGML_TYPE_F16) {
        const ggml_fp16_t * p = (const ggml_fp16_t *) s;
        cvt([&](int i) { return ggml_fp16_to_fp32(p[i]); });
    } else if (mt->type == GGML_TYPE_F32) {
        const float * p = (const float *) s;
        cvt([&](int i) { return p[i]; });
    } else {
        fprintf(stderr, "[acestep-dit] proj_in unsupported type for %s\n", name.c_str());
        return;
    }
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(float));
}

// proj_out conv weight GGUF [P, out_ch, H] -> pre-permuted+transposed 2D
// [H, out_ch*P] F32 (data[(p*out_ch+oc)*H + h] = src(h*P*out_ch + oc*P + p)).
static void load_proj_out(ggml_tensor * dst, const DitGGUF & g, const std::string & name, int H, int out_ch, int P) {
    if (!dst) return;
    ggml_tensor * mt = dit_gmeta(g, name);
    const void *  s  = dit_gdata(g, name);
    if (!mt || !s) return;
    std::vector<float> data((size_t) out_ch * P * H);
    auto cvt = [&](auto rd) {
        for (int h = 0; h < H; h++)
            for (int oc = 0; oc < out_ch; oc++)
                for (int p = 0; p < P; p++)
                    data[(size_t) (p * out_ch + oc) * H + h] = rd(h * P * out_ch + oc * P + p);
    };
    if (mt->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) s;
        cvt([&](int i) { return bf16_to_f32(p[i]); });
    } else if (mt->type == GGML_TYPE_F16) {
        const ggml_fp16_t * p = (const ggml_fp16_t *) s;
        cvt([&](int i) { return ggml_fp16_to_fp32(p[i]); });
    } else if (mt->type == GGML_TYPE_F32) {
        const float * p = (const float *) s;
        cvt([&](int i) { return p[i]; });
    } else {
        fprintf(stderr, "[acestep-dit] proj_out unsupported type for %s\n", name.c_str());
        return;
    }
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(float));
}

// pre-permute conv weight shape helpers create the destination shape.
static ggml_tensor * create_2d_f32(ggml_context * ctx, int64_t ne0, int64_t ne1, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    return t;
}

static void temb_create(DitTemb & w, ggml_context * ctx, const DitGGUF & g, const std::string & pfx,
                        ggml_backend_buffer_t map_buf = nullptr) {
    w.linear_1_w  = create_like(ctx, g, pfx + ".linear_1.weight", map_buf);
    w.linear_1_b  = create_f32_like(ctx, g, pfx + ".linear_1.bias");
    w.linear_2_w  = create_like(ctx, g, pfx + ".linear_2.weight", map_buf);
    w.linear_2_b  = create_f32_like(ctx, g, pfx + ".linear_2.bias");
    w.time_proj_w = create_like(ctx, g, pfx + ".time_proj.weight", map_buf);
    w.time_proj_b = create_f32_like(ctx, g, pfx + ".time_proj.bias");
}

static void temb_load(DitTemb & w, const DitGGUF & g, const std::string & pfx) {
    load_raw(w.linear_1_w, g, pfx + ".linear_1.weight");
    load_f32(w.linear_1_b, g, pfx + ".linear_1.bias");
    load_raw(w.linear_2_w, g, pfx + ".linear_2.weight");
    load_f32(w.linear_2_b, g, pfx + ".linear_2.bias");
    load_raw(w.time_proj_w, g, pfx + ".time_proj.weight");
    load_f32(w.time_proj_b, g, pfx + ".time_proj.bias");
}

// ------------------------------------------------------------------ graph ops
static ggml_tensor * as_f32(ggml_context * ctx, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

static ggml_tensor * rms_norm_w(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), as_f32(ctx, w));
}

static ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

static ggml_tensor * linear_b(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return ggml_add(ctx, ggml_mul_mat(ctx, w, x), as_f32(ctx, b));
}

// out = norm * (1 + scale) + shift  (scale/shift broadcast [H] over [H,S,N])
static ggml_tensor * adaln(ggml_context * ctx, ggml_tensor * norm, ggml_tensor * scale, ggml_tensor * shift,
                           ggml_tensor * one) {
    ggml_tensor * one_plus = ggml_add(ctx, scale, one);
    return ggml_add(ctx, ggml_mul(ctx, norm, one_plus), shift);
}

static ggml_tensor * gated_add(ggml_context * ctx, ggml_tensor * res, ggml_tensor * x, ggml_tensor * gate) {
    return ggml_add(ctx, res, ggml_mul(ctx, x, gate));
}

// F32 attention: Q[D,S,Nh,N], K[D,Skv,Nkv,N], V[D,Skv,Nkv,N] -> [D,Nh,S,N]
static ggml_tensor * attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                              ggml_tensor * mask, float scale) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static bool backend_supports_flash_attention(ggml_backend_t backend,
                                             const DitConfig &config) {
  if (!backend || config.head_dim <= 0 || config.n_heads <= 0 ||
      config.n_kv_heads <= 0)
    return false;
  const ggml_backend_dev_t device = ggml_backend_get_device(backend);
  if (!device)
    return false;
  const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
  if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
      type != GGML_BACKEND_DEVICE_TYPE_IGPU)
    return false;

  constexpr int sequence = 16;
  constexpr int batch = 2;
  ggml_init_params init{ggml_tensor_overhead() * 8, nullptr, true};
  ggml_context *context = ggml_init(init);
  if (!context)
    return false;
  ggml_tensor *query = ggml_new_tensor_4d(
      context, GGML_TYPE_F32, config.head_dim, sequence, config.n_heads, batch);
  ggml_tensor *key = ggml_new_tensor_4d(context, GGML_TYPE_F16, config.head_dim,
                                        sequence, config.n_kv_heads, batch);
  ggml_tensor *value =
      ggml_new_tensor_4d(context, GGML_TYPE_F16, config.head_dim, sequence,
                         config.n_kv_heads, batch);
  ggml_tensor *mask =
      ggml_new_tensor_4d(context, GGML_TYPE_F16, sequence, sequence, 1, batch);
  ggml_tensor *operation =
      ggml_flash_attn_ext(context, query, key, value, mask,
                          1.0f / sqrtf((float)config.head_dim), 0.0f, 0.0f);
  if (!operation) {
    ggml_free(context);
    return false;
  }
  ggml_flash_attn_ext_set_prec(operation, GGML_PREC_F32);
  const bool supported = ggml_backend_supports_op(backend, operation);
  ggml_free(context);
  return supported;
}

static constexpr float DIT_FLASH_ATTENTION_MAX_BIAS = 0.0f;
static constexpr float DIT_FLASH_ATTENTION_LOGIT_SOFTCAP = 0.0f;

static ggml_tensor * flash_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                     ggml_tensor * v, ggml_tensor * mask, float scale) {
    if (k->type == GGML_TYPE_F32) k = ggml_cast(ctx, k, GGML_TYPE_F16);
    if (v->type == GGML_TYPE_F32) v = ggml_cast(ctx, v, GGML_TYPE_F16);
    ggml_tensor * attention =
        ggml_flash_attn_ext(ctx, q, k, v, mask, scale,
                            DIT_FLASH_ATTENTION_MAX_BIAS, DIT_FLASH_ATTENTION_LOGIT_SOFTCAP);
    ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
    return attention;
}

static ggml_tensor * select_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                      ggml_tensor * v, ggml_tensor * mask, float scale,
                                      bool use_flash_attention) {
    if (use_flash_attention) return flash_attention(ctx, q, k, v, mask, scale);
    return attn_f32(ctx, q, k, v, mask, scale);
}

static ggml_tensor * build_temb(ggml_context * ctx, DitTemb * w, ggml_tensor * t_scalar, ggml_tensor ** out_tproj) {
    ggml_tensor * t_scaled   = ggml_scale(ctx, t_scalar, 1000.0f);
    ggml_tensor * sinusoidal = ggml_timestep_embedding(ctx, t_scaled, 256, 10000);
    ggml_tensor * h          = linear_b(ctx, w->linear_1_w, w->linear_1_b, sinusoidal);
    h                        = ggml_silu(ctx, h);
    ggml_tensor * temb       = linear_b(ctx, w->linear_2_w, w->linear_2_b, h);
    ggml_tensor * h2         = ggml_silu(ctx, temb);
    *out_tproj               = linear_b(ctx, w->time_proj_w, w->time_proj_b, h2);
    return temb;
}

static ggml_tensor * build_self_attn(ggml_context * ctx, DitModel * m, DitLayer * ly, ggml_tensor * norm_sa,
                                     ggml_tensor * positions, ggml_tensor * mask, int S, int N) {
    const DitConfig & c   = m->cfg;
    int               D   = c.head_dim;
    int               Nh  = c.n_heads;
    int               Nkv = c.n_kv_heads;

    ggml_tensor * q = linear(ctx, ly->sa_q_proj, norm_sa);
    ggml_tensor * k = linear(ctx, ly->sa_k_proj, norm_sa);
    ggml_tensor * v = linear(ctx, ly->sa_v_proj, norm_sa);

    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);
    v = ggml_reshape_4d(ctx, v, D, Nkv, S, N);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), as_f32(ctx, ly->sa_q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), as_f32(ctx, ly->sa_k_norm));

    q = ggml_reshape_3d(ctx, q, D, Nh, (int64_t) S * N);
    k = ggml_reshape_3d(ctx, k, D, Nkv, (int64_t) S * N);
    q = ggml_rope_ext(ctx, q, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    const float scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn = select_attention(ctx, q, k, v, mask, scale, m->use_flash_attn);
    attn                = ggml_reshape_3d(ctx, attn, (int64_t) Nh * D, S, N);
    return linear(ctx, ly->sa_o_proj, attn);
}

struct DitAttentionCaptureConfig {
    const DitAttentionHead * entries = nullptr;
    int                      count   = 0;
};

static bool capture_has_layer(const DitAttentionCaptureConfig * capture, int layer) {
    if (!capture) return false;
    for (int i = 0; i < capture->count; i++) {
        if (capture->entries[i].layer == layer) return true;
    }
    return false;
}

static bool capture_has_head(const DitAttentionCaptureConfig * capture, int layer, int head) {
    if (!capture) return false;
    for (int i = 0; i < capture->count; i++) {
        if (capture->entries[i].layer == layer && capture->entries[i].head == head) return true;
    }
    return false;
}

static int capture_last_layer(const DitAttentionCaptureConfig * capture) {
    int last = -1;
    for (int i = 0; i < capture->count; i++) {
        last = std::max(last, capture->entries[i].layer);
    }
    return last;
}

static void capture_name(char * buffer, size_t size, int layer, int head) {
    snprintf(buffer, size, "cross_attn_l%d_h%d", layer, head);
}

// Explicit-softmax cross-attention for captured layers: the per-head
// probabilities [enc_S, S] are exported as named contiguous graph outputs and
// the attention output is recomposed from them, mirroring the flash-free path.
static ggml_tensor * build_captured_cross_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                               ggml_tensor * v, ggml_tensor * mask, float scale,
                                               const DitAttentionCaptureConfig * capture,
                                               int layer_idx, int enc_S, int S, int N, int Nh) {
    ggml_tensor * probabilities = ggml_mul_mat(ctx, k, q);
    probabilities               = ggml_soft_max_ext(ctx, probabilities, mask, scale, 0.0f);
    for (int head = 0; head < Nh; head++) {
        if (!capture_has_head(capture, layer_idx, head)) continue;
        ggml_tensor * selected = ggml_view_3d(ctx, probabilities, enc_S, S, N, probabilities->nb[1],
                                              probabilities->nb[3], (size_t) head * probabilities->nb[2]);
        selected               = ggml_cont(ctx, selected);
        char name[64];
        capture_name(name, sizeof(name), layer_idx, head);
        ggml_set_name(selected, name);
        ggml_set_output(selected);
    }
    ggml_tensor * vt  = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out = ggml_mul_mat(ctx, vt, probabilities);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static ggml_tensor * build_cross_attn(ggml_context * ctx, DitModel * m, DitLayer * ly, ggml_tensor * norm_ca,
                                      ggml_tensor * enc, ggml_tensor * mask, int S, int enc_S, int N,
                                      const DitAttentionCaptureConfig * capture = nullptr,
                                      int layer_idx = -1) {
    const DitConfig & c   = m->cfg;
    int               D   = c.head_dim;
    int               Nh  = c.n_heads;
    int               Nkv = c.n_kv_heads;

    ggml_tensor * q = linear(ctx, ly->ca_q_proj, norm_ca);
    ggml_tensor * k = linear(ctx, ly->ca_k_proj, enc);
    ggml_tensor * v = linear(ctx, ly->ca_v_proj, enc);

    // QK norms run before the permute (self-attn's order): the norm reduces over
    // D per head either way, but the pre-permute layout is contiguous - the
    // strided form measured 3x slower on Vulkan at long sequence lengths.
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), as_f32(ctx, ly->ca_q_norm));
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_reshape_4d(ctx, k, D, Nkv, enc_S, N);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), as_f32(ctx, ly->ca_k_norm));
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_reshape_4d(ctx, v, D, Nkv, enc_S, N);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn  = capture_has_layer(capture, layer_idx)
                              ? build_captured_cross_attn(ctx, q, k, v, mask, scale, capture,
                                                          layer_idx, enc_S, S, N, Nh)
                              : select_attention(ctx, q, k, v, mask, scale, m->use_flash_attn);
    attn                = ggml_reshape_3d(ctx, attn, (int64_t) Nh * D, S, N);
    return linear(ctx, ly->ca_o_proj, attn);
}

static ggml_tensor * build_mlp(ggml_context * ctx, DitLayer * ly, ggml_tensor * norm_ffn) {
    ggml_tensor * gate = linear(ctx, ly->gate_proj, norm_ffn);
    ggml_tensor * up   = linear(ctx, ly->up_proj, norm_ffn);
    ggml_tensor * ff   = ggml_swiglu_split(ctx, gate, up);
    return linear(ctx, ly->down_proj, ff);
}

static ggml_tensor * build_layer(ggml_context * ctx, DitModel * m, int idx, ggml_tensor * hidden, ggml_tensor * tproj,
                                 ggml_tensor * enc, ggml_tensor * positions, ggml_tensor * sa_mask,
                                 ggml_tensor * ca_mask, int S, int enc_S, int N,
                                 const DitAttentionCaptureConfig * capture = nullptr) {
    const DitConfig & c  = m->cfg;
    DitLayer *        ly = &m->layers[idx];
    int               H  = c.hidden_size;

    ggml_tensor * ss = as_f32(ctx, ly->scale_shift_table);
    ss               = ggml_reshape_1d(ctx, ss, 6 * H);
    ggml_tensor * ad = ggml_add(ctx, ss, tproj);
    size_t        Hb = (size_t) H * sizeof(float);

    ggml_tensor * shift_sa  = ggml_view_1d(ctx, ad, H, 0 * Hb);
    ggml_tensor * scale_sa  = ggml_view_1d(ctx, ad, H, 1 * Hb);
    ggml_tensor * gate_sa   = ggml_view_1d(ctx, ad, H, 2 * Hb);
    ggml_tensor * shift_ffn = ggml_view_1d(ctx, ad, H, 3 * Hb);
    ggml_tensor * scale_ffn = ggml_view_1d(ctx, ad, H, 4 * Hb);
    ggml_tensor * gate_ffn  = ggml_view_1d(ctx, ad, H, 5 * Hb);

    ggml_tensor * res     = hidden;
    ggml_tensor * norm_sa = rms_norm_w(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    norm_sa               = adaln(ctx, norm_sa, scale_sa, shift_sa, m->scalar_one);
    ggml_tensor * sa_out  = build_self_attn(ctx, m, ly, norm_sa, positions, sa_mask, S, N);
    hidden                = gated_add(ctx, res, sa_out, gate_sa);

    if (enc) {
        ggml_tensor * norm_ca = rms_norm_w(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
        ggml_tensor * ca_out  = build_cross_attn(ctx, m, ly, norm_ca, enc, ca_mask, S, enc_S, N,
                                                 capture, idx);
        hidden                = ggml_add(ctx, hidden, ca_out);
    }

    res                    = hidden;
    ggml_tensor * norm_ffn = rms_norm_w(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    norm_ffn               = adaln(ctx, norm_ffn, scale_ffn, shift_ffn, m->scalar_one);
    ggml_tensor * ffn_out  = build_mlp(ctx, ly, norm_ffn);
    hidden                 = gated_add(ctx, res, ffn_out, gate_ffn);
    return hidden;
}

// ------------------------------------------------------------------ public
// Shared body of dit_model_load and dit_model_load_metadata_only. When
// `measure` is non-null the load is metadata-only: the weight allocation is
// sized into `measure` instead of performed and no tensor data is read.
static DitModel * dit_model_load_impl(const std::string & path, ggml_backend_t backend, bool verbose,
                                      AcestepStageMeasure * measure) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) return nullptr;
    dit_gguf_require_hexagon_weights(g, path, backend, "acestep-dit");

    DitModel * m = new DitModel();
    m->backend = backend;
    if (!dit_gguf_read_config(g, m->cfg)) {
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }
    m->use_flash_attn = backend_supports_flash_attention(backend, m->cfg);
    const DitConfig & c = m->cfg;
    const int         H = c.hidden_size;

    // CPU backend: map the bulk (quantised) weights straight off the GGUF mmap
    // instead of copying them into a backend buffer, so they stay clean/evictable
    // (see dit_gguf_cpu_map_buffer). `map_buf` is passed to create_like for the
    // verbatim weights and left null for the F32-converted / permuted tensors,
    // which are still allocated + uploaded. On a GPU backend map_buf stays null
    // and behaviour is unchanged (weights uploaded to device memory).
    const bool            mapped  = dit_gguf_backend_maps_weights(backend);
    ggml_backend_buffer_t map_buf = mapped ? dit_gguf_cpu_map_buffer(g) : nullptr;

    // enough overhead for all descriptors
    ggml_init_params ip{ ggml_tensor_overhead() * (size_t) (64 + 40 * c.n_layers), nullptr, /*no_alloc=*/true };
    m->weight_ctx = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    temb_create(m->time_embed, ctx, g, "decoder.time_embed", map_buf);
    temb_create(m->time_embed_r, ctx, g, "decoder.time_embed_r", map_buf);

    m->proj_in_w = create_2d_f32(ctx, (int64_t) c.in_channels * c.patch_size, H, "decoder.proj_in.1.weight");
    m->proj_in_b = create_f32_like(ctx, g, "decoder.proj_in.1.bias");
    m->cond_emb_w = create_like(ctx, g, "decoder.condition_embedder.weight", map_buf);
    m->cond_emb_b = create_f32_like(ctx, g, "decoder.condition_embedder.bias");
    m->cfg.enc_hidden_size = (int) m->cond_emb_w->ne[0];  // [H_enc, H] -> H_enc

    m->layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        DitLayer &  ly = m->layers[i];
        std::string p  = "decoder.layers." + std::to_string(i);
        ly.self_attn_norm = create_f32_like(ctx, g, p + ".self_attn_norm.weight");
        ly.sa_q_proj      = create_like(ctx, g, p + ".self_attn.q_proj.weight", map_buf);
        ly.sa_k_proj      = create_like(ctx, g, p + ".self_attn.k_proj.weight", map_buf);
        ly.sa_v_proj      = create_like(ctx, g, p + ".self_attn.v_proj.weight", map_buf);
        ly.sa_q_norm      = create_f32_like(ctx, g, p + ".self_attn.q_norm.weight");
        ly.sa_k_norm      = create_f32_like(ctx, g, p + ".self_attn.k_norm.weight");
        ly.sa_o_proj      = create_like(ctx, g, p + ".self_attn.o_proj.weight", map_buf);
        ly.cross_attn_norm = create_f32_like(ctx, g, p + ".cross_attn_norm.weight");
        ly.ca_q_proj      = create_like(ctx, g, p + ".cross_attn.q_proj.weight", map_buf);
        ly.ca_k_proj      = create_like(ctx, g, p + ".cross_attn.k_proj.weight", map_buf);
        ly.ca_v_proj      = create_like(ctx, g, p + ".cross_attn.v_proj.weight", map_buf);
        ly.ca_q_norm      = create_f32_like(ctx, g, p + ".cross_attn.q_norm.weight");
        ly.ca_k_norm      = create_f32_like(ctx, g, p + ".cross_attn.k_norm.weight");
        ly.ca_o_proj      = create_like(ctx, g, p + ".cross_attn.o_proj.weight", map_buf);
        ly.mlp_norm       = create_f32_like(ctx, g, p + ".mlp_norm.weight");
        ly.gate_proj      = create_like(ctx, g, p + ".mlp.gate_proj.weight", map_buf);
        ly.up_proj        = create_like(ctx, g, p + ".mlp.up_proj.weight", map_buf);
        ly.down_proj      = create_like(ctx, g, p + ".mlp.down_proj.weight", map_buf);
        ly.scale_shift_table = create_f32_like(ctx, g, p + ".scale_shift_table");
        ly.layer_type     = (i % 2 == 0) ? 0 : 1;
    }

    m->norm_out        = create_f32_like(ctx, g, "decoder.norm_out.weight");
    m->out_scale_shift = create_f32_like(ctx, g, "decoder.scale_shift_table");
    m->proj_out_w      = create_2d_f32(ctx, H, (int64_t) c.out_channels * c.patch_size, "decoder.proj_out.1.weight");
    m->proj_out_b      = create_f32_like(ctx, g, "decoder.proj_out.1.bias");
    m->scalar_one      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(m->scalar_one, "scalar_one");

    if (measure) {
        // Size-only twin of the allocation below: sizes only the tensors still
        // lacking data, exactly like ggml_backend_alloc_ctx_tensors skips the
        // mapped ones. All uploads (including scalar_one) are skipped.
        measure->weights_alloc_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            ctx, dit_gguf_weight_buffer_type(backend));
        m->measuring    = true;
        m->mapped_bytes = mapped ? dit_gguf_mapped_bytes(ctx, g) : 0;
        measure->weights_mapped_bytes = m->mapped_bytes;
        // Mark still-unallocated weights externally-allocated so graph sizing
        // excludes them (see textenc_model_load_impl). No data is read after.
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (!t->data && !t->view_src) {
                t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
        }
        if (mapped) {
            m->mapped  = true;
            m->map_buf = map_buf;
            m->gguf    = g;
        } else {
            dit_gguf_close(g);
        }
        return m;
    }

    // Allocates only the tensors still lacking data (the F32-converted / permuted
    // ones); every mapped weight has ->data set already and is skipped.
    m->weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, dit_gguf_weight_buffer_type(backend));
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-dit] failed to allocate weight buffer\n");
        if (map_buf) ggml_backend_buffer_free(map_buf);
        ggml_free(ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    // upload — load_raw is a per-tensor no-op for mapped weights (already backed
    // by the mmap); the F32-converted / permuted tensors are always uploaded.
    temb_load(m->time_embed, g, "decoder.time_embed");
    temb_load(m->time_embed_r, g, "decoder.time_embed_r");
    load_proj_in(m->proj_in_w, g, "decoder.proj_in.1.weight", H, c.in_channels, c.patch_size);
    load_f32(m->proj_in_b, g, "decoder.proj_in.1.bias");
    load_raw(m->cond_emb_w, g, "decoder.condition_embedder.weight");
    load_f32(m->cond_emb_b, g, "decoder.condition_embedder.bias");

    for (int i = 0; i < c.n_layers; i++) {
        DitLayer &  ly = m->layers[i];
        std::string p  = "decoder.layers." + std::to_string(i);
        load_f32(ly.self_attn_norm, g, p + ".self_attn_norm.weight");
        load_raw(ly.sa_q_proj, g, p + ".self_attn.q_proj.weight");
        load_raw(ly.sa_k_proj, g, p + ".self_attn.k_proj.weight");
        load_raw(ly.sa_v_proj, g, p + ".self_attn.v_proj.weight");
        load_f32(ly.sa_q_norm, g, p + ".self_attn.q_norm.weight");
        load_f32(ly.sa_k_norm, g, p + ".self_attn.k_norm.weight");
        load_raw(ly.sa_o_proj, g, p + ".self_attn.o_proj.weight");
        load_f32(ly.cross_attn_norm, g, p + ".cross_attn_norm.weight");
        load_raw(ly.ca_q_proj, g, p + ".cross_attn.q_proj.weight");
        load_raw(ly.ca_k_proj, g, p + ".cross_attn.k_proj.weight");
        load_raw(ly.ca_v_proj, g, p + ".cross_attn.v_proj.weight");
        load_f32(ly.ca_q_norm, g, p + ".cross_attn.q_norm.weight");
        load_f32(ly.ca_k_norm, g, p + ".cross_attn.k_norm.weight");
        load_raw(ly.ca_o_proj, g, p + ".cross_attn.o_proj.weight");
        load_f32(ly.mlp_norm, g, p + ".mlp_norm.weight");
        load_raw(ly.gate_proj, g, p + ".mlp.gate_proj.weight");
        load_raw(ly.up_proj, g, p + ".mlp.up_proj.weight");
        load_raw(ly.down_proj, g, p + ".mlp.down_proj.weight");
        load_f32(ly.scale_shift_table, g, p + ".scale_shift_table");
    }

    load_f32(m->norm_out, g, "decoder.norm_out.weight");
    load_f32(m->out_scale_shift, g, "decoder.scale_shift_table");
    load_proj_out(m->proj_out_w, g, "decoder.proj_out.1.weight", H, c.out_channels, c.patch_size);
    load_f32(m->proj_out_b, g, "decoder.proj_out.1.bias");
    const float one = 1.0f;
    ggml_backend_tensor_set(m->scalar_one, &one, 0, sizeof(float));

    // Exact mmapped weight footprint (sum of mapped tensor nbytes — excludes the
    // allocated F32/permuted tensors and GGUF metadata, so it does not double-count).
    m->mapped_bytes = mapped ? dit_gguf_mapped_bytes(ctx, g) : 0;

    if (mapped) {
        // Keep the mmap (and its backing buffer) alive for the model's lifetime;
        // the mapped weights point into it. Freed in dit_model_free.
        m->mapped  = true;
        m->map_buf = map_buf;
        m->gguf    = g;
    } else {
        dit_gguf_close(g);
    }

    if (verbose) {
        const float owned = (float) ggml_backend_buffer_get_size(m->weight_buf) / (1024 * 1024);
        const float mapsz = (float) m->mapped_bytes / (1024 * 1024);
        fprintf(stderr, "[acestep-dit] loaded %s: %.1f MB alloc + %.1f MB mmapped, %d layers H=%d Nh=%d/%d D=%d\n",
                path.c_str(), owned, mapsz, c.n_layers, H, c.n_heads, c.n_kv_heads, c.head_dim);
    }
    return m;
}

DitModel * dit_model_load(const std::string & path, ggml_backend_t backend, bool verbose) {
    return dit_model_load_impl(path, backend, verbose, /*measure=*/nullptr);
}

DitModel * dit_model_load_metadata_only(const std::string & path, ggml_backend_t backend,
                                        bool verbose, AcestepStageMeasure & measure) {
    measure = AcestepStageMeasure{};
    return dit_model_load_impl(path, backend, verbose, &measure);
}

size_t dit_model_compute_buffer_bytes(const DitModel * m) {
    return (m && m->graph_cache.ga) ? ggml_gallocr_get_buffer_size(m->graph_cache.ga, 0) : 0;
}

void dit_model_free(DitModel * m) {
    if (!m) return;
    m->graph_cache.release();
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    // Order: drop the CPU map buffer, then munmap/close the GGUF it wrapped.
    if (m->map_buf) ggml_backend_buffer_free(m->map_buf);
    if (m->mapped) dit_gguf_close(m->gguf);
    delete m;
}

const DitConfig & dit_model_config(const DitModel * m) { return m->cfg; }

size_t dit_model_weight_bytes(const DitModel * m) {
    if (!m) return 0;
    const size_t alloc = m->weight_buf ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
    return alloc + m->mapped_bytes;  // allocated (F32/permuted) + mmapped weights
}

bool dit_model_forward(DitModel * m, const DitForwardInputs & in, std::vector<float> & velocity_out,
                       size_t * measure_compute) {
    const DitConfig & c = m->cfg;
    const int         P = c.patch_size;
    const int         H = c.hidden_size;
    const int         T = in.T;
    const int         N = in.N;
    const int         S = T / P;
    const int         enc_S = in.enc_S;

    if (T % P != 0) {
        fprintf(stderr, "[acestep-dit] T (%d) must be a multiple of patch_size (%d)\n", T, P);
        return false;
    }

    const bool has_sa = in.sa_mask_sw != nullptr;
    const bool has_ca = in.ca_mask != nullptr;

    DitGraphCache & gc        = m->graph_cache;
    // Measure mode always rebuilds locally and never touches the cache, so a
    // projection on a live model cannot evict (or upload into) its real graph.
    const bool      cache_hit = !measure_compute &&
                                gc.ctx != nullptr && gc.T == T && gc.N == N && gc.enc_S == enc_S &&
                                gc.H_enc == in.H_enc && gc.has_sa == has_sa && gc.has_ca == has_ca;

    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor *input = nullptr, *enc_hidden = nullptr, *t_val = nullptr, *tr_val = nullptr,
                *positions = nullptr, *sa_mask = nullptr, *ca_mask = nullptr, *output = nullptr;

    if (cache_hit) {
        ctx    = gc.ctx;
        gf     = gc.gf;
        ga     = gc.ga;
        input  = gc.input;  enc_hidden = gc.enc_hidden;
        t_val  = gc.t_val;  tr_val = gc.tr_val; positions = gc.positions;
        sa_mask = gc.sa_mask; ca_mask = gc.ca_mask; output = gc.output;
    } else {
        if (!measure_compute) gc.release();

        const size_t nodes = (size_t) 8192;
        ggml_init_params gp{ ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
        ctx = ggml_init(gp);

        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.in_channels, T, N);
        ggml_set_input(input);
        // Step-invariant inputs are also flagged as outputs: gallocr never hands
        // an output's slot to the intermediate pool, so their bytes survive
        // replays and are uploaded only when constants_dirty.
        enc_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in.H_enc, enc_S, N);
        ggml_set_input(enc_hidden);
        ggml_set_output(enc_hidden);
        t_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_input(t_val);
        tr_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_input(tr_val);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) S * N);
        ggml_set_input(positions);
        ggml_set_output(positions);

        if (has_sa) {
            sa_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, N);
            ggml_set_input(sa_mask);
            ggml_set_output(sa_mask);
        }
        if (has_ca) {
            ca_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, enc_S, S, 1, N);
            ggml_set_input(ca_mask);
            ggml_set_output(ca_mask);
        }

        // timestep embeddings
        ggml_tensor * tproj_t;
        ggml_tensor * temb_t = build_temb(ctx, &m->time_embed, t_val, &tproj_t);
        ggml_tensor * tproj_r;
        ggml_tensor * t_diff = ggml_sub(ctx, t_val, tr_val);
        ggml_tensor * temb_r = build_temb(ctx, &m->time_embed_r, t_diff, &tproj_r);
        ggml_tensor * temb   = ggml_add(ctx, temb_t, temb_r);
        ggml_tensor * tproj  = ggml_add(ctx, tproj_t, tproj_r);

        // proj_in (patchify) + condition embedder
        ggml_tensor * patched = ggml_reshape_3d(ctx, input, (int64_t) c.in_channels * P, S, N);
        ggml_tensor * hidden  = linear_b(ctx, m->proj_in_w, m->proj_in_b, patched);
        ggml_tensor * enc     = linear_b(ctx, m->cond_emb_w, m->cond_emb_b, enc_hidden);

        for (int i = 0; i < c.n_layers; i++) {
            ggml_tensor * sm = (m->layers[i].layer_type == 0) ? sa_mask : nullptr;
            hidden = build_layer(ctx, m, i, hidden, tproj, enc, positions, sm, ca_mask, S, enc_S, N);
        }

        // output AdaLN + proj_out
        ggml_tensor * oss = ggml_reshape_1d(ctx, as_f32(ctx, m->out_scale_shift), 2 * H);
        size_t        Hb  = (size_t) H * sizeof(float);
        ggml_tensor * out_shift = ggml_add(ctx, ggml_view_1d(ctx, oss, H, 0), temb);
        ggml_tensor * out_scale = ggml_add(ctx, ggml_view_1d(ctx, oss, H, Hb), temb);
        ggml_tensor * norm_out  = rms_norm_w(ctx, hidden, m->norm_out, c.rms_norm_eps);
        norm_out                = adaln(ctx, norm_out, out_scale, out_shift, m->scalar_one);
        output                  = linear_b(ctx, m->proj_out_w, m->proj_out_b, norm_out);
        output                  = ggml_reshape_3d(ctx, output, c.out_channels, T, N);
        ggml_set_output(output);

        gf = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(gf, output);

        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
        if (!ga) {
            fprintf(stderr, "[acestep-dit] forward gallocr init failed (T=%d N=%d)\n", T, N);
            ggml_free(ctx);
            return false;
        }
        if (measure_compute) {
            // Size-only twin of the persistent graph-cache allocation below.
            *measure_compute = 0;
            ggml_gallocr_reserve_n_size(ga, gf, nullptr, nullptr, measure_compute);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return true;
        }
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            fprintf(stderr, "[acestep-dit] forward alloc failed (T=%d N=%d)\n", T, N);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return false;
        }

        gc.ctx   = ctx;
        gc.gf    = gf;
        gc.ga    = ga;
        gc.input = input;  gc.enc_hidden = enc_hidden;
        gc.t_val = t_val;  gc.tr_val = tr_val; gc.positions = positions;
        gc.sa_mask = sa_mask; gc.ca_mask = ca_mask; gc.output = output;
        gc.T = T; gc.N = N; gc.enc_S = enc_S; gc.H_enc = in.H_enc;
        gc.has_sa = has_sa; gc.has_ca = has_ca;
    }

    ggml_backend_tensor_set(input, in.input_latents, 0, (size_t) c.in_channels * T * N * sizeof(float));
    ggml_backend_tensor_set(t_val, &in.t, 0, sizeof(float));
    ggml_backend_tensor_set(tr_val, &in.t_r, 0, sizeof(float));
    if (!cache_hit || in.cond_dirty) {
        ggml_backend_tensor_set(enc_hidden, in.enc_hidden, 0, (size_t) in.H_enc * enc_S * N * sizeof(float));
        if (ca_mask) ggml_backend_tensor_set(ca_mask, in.ca_mask, 0, (size_t) enc_S * S * N * sizeof(uint16_t));
    }
    if (!cache_hit || in.constants_dirty) {
        std::vector<int32_t> pos((size_t) S * N);
        for (int n = 0; n < N; n++)
            for (int s = 0; s < S; s++) pos[(size_t) n * S + s] = s;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        if (sa_mask) ggml_backend_tensor_set(sa_mask, in.sa_mask_sw, 0, (size_t) S * S * N * sizeof(uint16_t));
    }

    int rc = ggml_backend_graph_compute(m->backend, gf);
    if (rc != GGML_STATUS_SUCCESS) {
        gc.release();
        return false;
    }

    velocity_out.resize((size_t) c.out_channels * T * N);
    ggml_backend_tensor_get(output, velocity_out.data(), 0, ggml_nbytes(output));
    return true;
}

static bool probe_inputs_valid(const DitConfig & c, const DitAttentionProbeInputs & in,
                               const std::vector<DitAttentionHead> & heads) {
    if (!in.context || !in.latent || !in.enc_hidden) return false;
    if (in.T <= 0 || in.T % c.patch_size != 0 || in.num_steps <= 0) return false;
    if (in.enc_S <= 0 || in.H_enc != c.enc_hidden_size) return false;
    if (in.real_enc_S < 0 || in.real_enc_S > in.enc_S) return false;
    if (heads.empty()) return false;
    for (size_t i = 0; i < heads.size(); i++) {
        if (heads[i].layer < 0 || heads[i].layer >= c.n_layers) return false;
        if (heads[i].head < 0 || heads[i].head >= c.n_heads) return false;
        for (size_t j = 0; j < i; j++) {
            if (heads[i].layer == heads[j].layer && heads[i].head == heads[j].head) return false;
        }
    }
    return true;
}

static float round_to_bf16(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

// Probe latent: context channels stay verbatim; the trailing latent channels
// carry x_t = t*noise + (1-t)*x0 rounded through bf16 (reference parity).
static std::vector<float> build_probe_input(const DitAttentionProbeInputs & in, const DitConfig & c,
                                            float timestep) {
    const int          ctx_ch = c.in_channels - c.out_channels;
    std::vector<float> noise((size_t) in.T * c.out_channels);
    philox_randn(in.seed, noise.data(), (int) noise.size(), true);
    std::vector<float> input((size_t) in.T * c.in_channels);
    for (int frame = 0; frame < in.T; frame++) {
        float * destination = input.data() + (size_t) frame * c.in_channels;
        memcpy(destination, in.context + (size_t) frame * ctx_ch, (size_t) ctx_ch * sizeof(float));
        for (int channel = 0; channel < c.out_channels; channel++) {
            const size_t index = (size_t) frame * c.out_channels + channel;
            const float  xt    = timestep * noise[index] + (1.0f - timestep) * in.latent[index];
            destination[ctx_ch + channel] = round_to_bf16(xt);
        }
    }
    return input;
}

static std::vector<uint16_t> build_probe_self_mask(const DitConfig & c, int S) {
    std::vector<uint16_t> mask((size_t) S * S);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < S; ki++) {
            const int  dist   = qi > ki ? qi - ki : ki - qi;
            const bool in_win = c.sliding_window <= 0 || S <= c.sliding_window || dist <= c.sliding_window;
            mask[(size_t) qi * S + ki] = ggml_fp32_to_fp16(in_win ? 0.0f : -INFINITY);
        }
    }
    return mask;
}

static std::vector<uint16_t> build_probe_cross_mask(int enc_S, int S, int real_enc_S) {
    std::vector<uint16_t> mask((size_t) enc_S * S);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < enc_S; ki++) {
            mask[(size_t) qi * enc_S + ki] = ggml_fp32_to_fp16(ki < real_enc_S ? 0.0f : -INFINITY);
        }
    }
    return mask;
}

static bool read_captured_heads(ggml_cgraph * gf, const std::vector<DitAttentionHead> & heads,
                                int enc_S, int S, std::vector<std::vector<float>> & captured_out) {
    for (const DitAttentionHead & head : heads) {
        char name[64];
        capture_name(name, sizeof(name), head.layer, head.head);
        ggml_tensor * tensor = ggml_graph_get_tensor(gf, name);
        if (!tensor || tensor->ne[0] != enc_S || tensor->ne[1] != S) {
            fprintf(stderr, "[acestep-dit] probe: missing captured tensor %s\n", name);
            return false;
        }
        std::vector<float> values((size_t) enc_S * S);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        captured_out.push_back(std::move(values));
    }
    return true;
}

bool dit_probe_cross_attention(DitModel *                            m,
                               const DitAttentionProbeInputs &       in,
                               const std::vector<DitAttentionHead> & heads,
                               std::vector<std::vector<float>> &     captured_out) {
    captured_out.clear();
    const DitConfig & c = m->cfg;
    if (!probe_inputs_valid(c, in, heads)) {
        fprintf(stderr, "[acestep-dit] probe: inputs do not match the DiT configuration\n");
        return false;
    }

    const int   S        = in.T / c.patch_size;
    const int   enc_S    = in.enc_S;
    const float timestep = 1.0f / (float) in.num_steps;

    const size_t     nodes = (size_t) 8192;
    ggml_init_params gp{ ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);
    if (!ctx) return false;

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.in_channels, in.T, 1);
    ggml_set_input(input);
    ggml_tensor * enc_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in.H_enc, enc_S, 1);
    ggml_set_input(enc_hidden);
    ggml_tensor * t_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(t_val);
    ggml_tensor * tr_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(tr_val);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(positions);
    ggml_tensor * sa_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, 1);
    ggml_set_input(sa_mask);
    ggml_tensor * ca_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, enc_S, S, 1, 1);
    ggml_set_input(ca_mask);

    ggml_tensor * tproj_t;
    ggml_tensor * temb_t = build_temb(ctx, &m->time_embed, t_val, &tproj_t);
    ggml_tensor * tproj_r;
    ggml_tensor * t_diff = ggml_sub(ctx, t_val, tr_val);
    build_temb(ctx, &m->time_embed_r, t_diff, &tproj_r);
    (void) temb_t;
    ggml_tensor * tproj = ggml_add(ctx, tproj_t, tproj_r);

    ggml_tensor * patched = ggml_reshape_3d(ctx, input, (int64_t) c.in_channels * c.patch_size, S, 1);
    ggml_tensor * hidden  = linear_b(ctx, m->proj_in_w, m->proj_in_b, patched);
    ggml_tensor * enc     = linear_b(ctx, m->cond_emb_w, m->cond_emb_b, enc_hidden);

    const DitAttentionCaptureConfig capture{ heads.data(), (int) heads.size() };
    const int                       last_layer = capture_last_layer(&capture);
    for (int i = 0; i <= last_layer; i++) {
        ggml_tensor * sm = (m->layers[i].layer_type == 0) ? sa_mask : nullptr;
        hidden = build_layer(ctx, m, i, hidden, tproj, enc, positions, sm, ca_mask, S, enc_S, 1, &capture);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, false);
    for (const DitAttentionHead & head : heads) {
        char name[64];
        capture_name(name, sizeof(name), head.layer, head.head);
        ggml_tensor * tensor = ggml_get_tensor(ctx, name);
        if (!tensor) {
            fprintf(stderr, "[acestep-dit] probe: capture %s was not built\n", name);
            ggml_free(ctx);
            return false;
        }
        ggml_build_forward_expand(gf, tensor);
    }

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-dit] probe: graph alloc failed (T=%d enc_S=%d)\n", in.T, enc_S);
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float>    probe_input = build_probe_input(in, c, timestep);
    const std::vector<uint16_t> self_mask   = build_probe_self_mask(c, S);
    const std::vector<uint16_t> cross_mask  = build_probe_cross_mask(enc_S, S, in.real_enc_S);
    std::vector<int32_t>        pos(S);
    for (int i = 0; i < S; i++) pos[(size_t) i] = i;

    ggml_backend_tensor_set(input, probe_input.data(), 0, probe_input.size() * sizeof(float));
    ggml_backend_tensor_set(enc_hidden, in.enc_hidden, 0, (size_t) in.H_enc * enc_S * sizeof(float));
    ggml_backend_tensor_set(t_val, &timestep, 0, sizeof(float));
    ggml_backend_tensor_set(tr_val, &timestep, 0, sizeof(float));
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(sa_mask, self_mask.data(), 0, self_mask.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(ca_mask, cross_mask.data(), 0, cross_mask.size() * sizeof(uint16_t));

    const int  rc = ggml_backend_graph_compute(m->backend, gf);
    bool       ok = rc == GGML_STATUS_SUCCESS;
    if (!ok) {
        fprintf(stderr, "[acestep-dit] probe: compute failed (%d)\n", rc);
    } else {
        ok = read_captured_heads(gf, heads, enc_S, S, captured_out);
        if (!ok) captured_out.clear();
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

void dit_build_schedule(float shift, int num_steps, std::vector<float> & schedule_out) {
    schedule_out.resize(num_steps);
    for (int i = 0; i < num_steps; i++) {
        float t          = 1.0f - (float) i / (float) num_steps;
        schedule_out[i]  = shift * t / (1.0f + (shift - 1.0f) * t);
    }
}

void dit_apply_haar_dcw(std::vector<float> &       x_next,
                        const std::vector<float> & denoised,
                        int                        T,
                        int                        C,
                        int                        N,
                        float                      low_scale,
                        float                      high_scale) {
    const size_t expected = (size_t) T * C * N;
    if (T <= 0 || C <= 0 || N <= 0 || x_next.size() != expected || denoised.size() != expected) return;
    if (low_scale == 0.0f && high_scale == 0.0f) return;

    // Single-level orthonormal Haar DWT/IDWT along time. For an odd T, the
    // missing right sample is zero, matching pytorch_wavelets mode="zero".
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    for (int b = 0; b < N; ++b) {
        const size_t batch = (size_t) b * T * C;
        for (int t = 0; t < T; t += 2) {
            const bool   has_odd = t + 1 < T;
            const size_t even    = batch + (size_t) t * C;
            const size_t odd     = even + C;
            for (int c = 0; c < C; ++c) {
                const float xe = x_next[even + c];
                const float xo = has_odd ? x_next[odd + c] : 0.0f;
                const float ye = denoised[even + c];
                const float yo = has_odd ? denoised[odd + c] : 0.0f;

                float x_low  = (xe + xo) * inv_sqrt2;
                float x_high = (xe - xo) * inv_sqrt2;
                const float y_low  = (ye + yo) * inv_sqrt2;
                const float y_high = (ye - yo) * inv_sqrt2;

                x_low += low_scale * (x_low - y_low);
                x_high += high_scale * (x_high - y_high);
                x_next[even + c] = (x_low + x_high) * inv_sqrt2;
                if (has_odd) x_next[odd + c] = (x_low - x_high) * inv_sqrt2;
            }
        }
    }
}

static constexpr float DIT_FINAL_TIMESTEP = 0.0f;

static float next_schedule_timestep(const float * schedule, int step, int num_steps) {
    return step == num_steps - 1 ? DIT_FINAL_TIMESTEP : schedule[step + 1];
}

static void inject_repaint_source(const DitSampleParams & params, int step, size_t latent_count,
                                  int channels, std::vector<float> & current) {
    const int injection_cutoff =
        audio_edit_round_ties_to_even(params.repaint_injection_ratio * params.num_steps);
    if (!params.repaint_mask || !params.clean_source_latents || step >= injection_cutoff) return;
    const float timestep = next_schedule_timestep(params.schedule, step, params.num_steps);
    GGML_ASSERT(latent_count == (size_t) params.T * channels * params.N);
    repaint_inject_source(
        current, params.clean_source_latents, params.noise, params.repaint_mask,
        (size_t) params.T, timestep, channels);
}

static void preserve_repaint_latent(const DitSampleParams & params, size_t latent_count,
                                    int channels, std::vector<float> & current) {
    if (!params.repaint_mask || !params.clean_source_latents || !params.repaint_preserve_latent) return;
    GGML_ASSERT(latent_count == (size_t) params.T * channels * params.N);
    repaint_blend_latent(
        current, params.clean_source_latents, params.repaint_mask,
        (size_t) params.T, params.repaint_crossfade_frames, channels);
}

static constexpr double DIT_APG_MOMENTUM       = -0.75;
static constexpr double DIT_APG_NORM_THRESHOLD = 2.5;

static void apg_accumulate_momentum(std::vector<double> & running, std::vector<double> & diff) {
    for (size_t i = 0; i < diff.size(); i++) {
        running[i] = diff[i] + DIT_APG_MOMENTUM * running[i];
        diff[i]    = running[i];
    }
}

static double apg_channel_norm(const double * values, int T, int Oc, int channel) {
    double sum = 0.0;
    for (int t = 0; t < T; t++) {
        const double v = values[(size_t) t * Oc + channel];
        sum += v * v;
    }
    return std::sqrt(sum);
}

static void apg_scale_channel(double * values, int T, int Oc, int channel, double scale) {
    for (int t = 0; t < T; t++) {
        values[(size_t) t * Oc + channel] *= scale;
    }
}

static void apg_clip_channel_norms(double * diff, int T, int Oc) {
    for (int c = 0; c < Oc; c++) {
        const double norm = apg_channel_norm(diff, T, Oc, c);
        if (norm > DIT_APG_NORM_THRESHOLD) {
            apg_scale_channel(diff, T, Oc, c, DIT_APG_NORM_THRESHOLD / norm);
        }
    }
}

static void fill_ca_mask_rows(std::vector<uint16_t> & ca_mask, const int * real_enc_S,
                              int enc_S, int S, int N) {
    for (int b = 0; b < N; b++) {
        int re = real_enc_S ? real_enc_S[b] : enc_S;
        for (int qi = 0; qi < S; qi++) {
            for (int ki = 0; ki < enc_S; ki++) {
                float v = (ki < re) ? 0.0f : -INFINITY;
                ca_mask[(size_t) b * enc_S * S + (size_t) qi * enc_S + ki] = ggml_fp32_to_fp16(v);
            }
        }
    }
}

static double apg_channel_norm_f32(const float * values, int T, int Oc, int channel) {
    double sum = 0.0;
    for (int t = 0; t < T; t++) {
        const double v = values[(size_t) t * Oc + channel];
        sum += v * v;
    }
    return std::sqrt(sum);
}

static void apg_remove_parallel_component(double * diff, const float * cond, int T, int Oc, int channel) {
    const double norm = apg_channel_norm_f32(cond, T, Oc, channel);
    if (norm <= 0.0) return;
    const double inv_norm = 1.0 / norm;
    double dot = 0.0;
    for (int t = 0; t < T; t++) {
        const size_t idx = (size_t) t * Oc + channel;
        dot += diff[idx] * (double) cond[idx] * inv_norm;
    }
    for (int t = 0; t < T; t++) {
        const size_t idx = (size_t) t * Oc + channel;
        diff[idx] -= dot * (double) cond[idx] * inv_norm;
    }
}

static void apg_project_orthogonal(double * diff, const float * cond, int T, int Oc) {
    for (int c = 0; c < Oc; c++) {
        apg_remove_parallel_component(diff, cond, T, Oc, c);
    }
}

static std::vector<double> apg_velocity_difference(const std::vector<float> & velocity,
                                                   const std::vector<float> & velocity_uncond) {
    std::vector<double> diff(velocity.size());
    for (size_t i = 0; i < diff.size(); i++) {
        diff[i] = (double) velocity[i] - (double) velocity_uncond[i];
    }
    return diff;
}

static void apg_shape_batch_updates(std::vector<double> & diff, const std::vector<float> & velocity,
                                    int T, int Oc, int N) {
    const size_t n_per = (size_t) T * Oc;
    for (int b = 0; b < N; b++) {
        apg_clip_channel_norms(diff.data() + (size_t) b * n_per, T, Oc);
        apg_project_orthogonal(diff.data() + (size_t) b * n_per, velocity.data() + (size_t) b * n_per, T, Oc);
    }
}

static void apg_apply_guided_update(std::vector<float> & velocity, const std::vector<double> & diff,
                                    float guidance_scale) {
    const double weight = (double) guidance_scale - 1.0;
    for (size_t i = 0; i < diff.size(); i++) {
        velocity[i] = (float) ((double) velocity[i] + weight * diff[i]);
    }
}

void dit_apg_guide(std::vector<float> &       velocity,
                   const std::vector<float> & velocity_uncond,
                   std::vector<double> &      momentum,
                   float                      guidance_scale,
                   int                        T,
                   int                        Oc,
                   int                        N) {
    std::vector<double> diff = apg_velocity_difference(velocity, velocity_uncond);
    apg_accumulate_momentum(momentum, diff);
    apg_shape_batch_updates(diff, velocity, T, Oc, N);
    apg_apply_guided_update(velocity, diff, guidance_scale);
}

static std::vector<float> make_null_enc_hidden(const float * null_emb, int H_enc, int enc_S, int N) {
    std::vector<float> hidden((size_t) H_enc * enc_S * N);
    for (int b = 0; b < N; b++) {
        for (int s = 0; s < enc_S; s++) {
            memcpy(&hidden[((size_t) b * enc_S + s) * H_enc], null_emb, (size_t) H_enc * sizeof(float));
        }
    }
    return hidden;
}

static std::vector<uint16_t> make_visible_ca_mask(int enc_S, int S, int N) {
    return std::vector<uint16_t>((size_t) enc_S * S * N, ggml_fp32_to_fp16(0.0f));
}

static void splice_context_channels(std::vector<float> & input_buf, const float * context,
                                    int T, int N, int in_ch, int ctx_ch) {
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            memcpy(&input_buf[(size_t) b * T * in_ch + (size_t) t * in_ch],
                   &context[(size_t) b * T * ctx_ch + (size_t) t * ctx_ch],
                   (size_t) ctx_ch * sizeof(float));
        }
    }
}

bool dit_sample(DitModel * m, const DitSampleParams & p, std::vector<float> & latent_out) {
    const DitConfig & c      = m->cfg;
    const int         Oc     = c.out_channels;                 // 64 (noisy latent channels)
    const int         ctx_ch = c.in_channels - Oc;             // 128 (conditioning channels)
    const int         in_ch  = c.in_channels;                  // 192
    const int         T      = p.T;
    const int         N      = p.N;
    const int         S      = T / c.patch_size;
    const int         enc_S  = p.enc_S;
    const int         win    = c.sliding_window;
    const size_t      n_per  = (size_t) T * Oc;                 // elements per sample

    if (T % c.patch_size != 0) {
        fprintf(stderr, "[acestep-dit] sample: T (%d) not a multiple of patch_size (%d)\n", T, c.patch_size);
        return false;
    }

    // Self-attention sliding-window mask [S, S, 1, N] (F16). Bidirectional window
    // |qi - ki| <= win; layer_type==1 layers ignore it (full attention).
    std::vector<uint16_t> sa_mask((size_t) S * S * N);
    for (int b = 0; b < N; b++) {
        for (int qi = 0; qi < S; qi++) {
            for (int ki = 0; ki < S; ki++) {
                int  dist   = qi > ki ? qi - ki : ki - qi;
                bool in_win = (win <= 0) || (S <= win) || (dist <= win);
                sa_mask[(size_t) b * S * S + (size_t) qi * S + ki] =
                    ggml_fp32_to_fp16(in_win ? 0.0f : -INFINITY);
            }
        }
    }

    // Cross-attention padding mask [enc_S, S, 1, N] (F16): block encoder positions
    // beyond the real (unpadded) length; value depends only on ki.
    std::vector<uint16_t> ca_mask((size_t) enc_S * S * N);
    fill_ca_mask_rows(ca_mask, p.real_enc_S, enc_S, S, N);

    // x_t (current noisy latent) starts at the supplied noise.
    std::vector<float> xt(p.noise, p.noise + (size_t) n_per * N);

    // Per-step DiT input [in_ch, T, N]: context channels are constant between
    // conditioning switches, the last Oc channels carry x_t and are refreshed
    // each step.
    std::vector<float> input_buf((size_t) in_ch * T * N);
    splice_context_channels(input_buf, p.context_latents, T, N, in_ch, ctx_ch);

    std::vector<float> vt;
    std::vector<float> xt_before;
    std::vector<float> denoised;
    if (p.dcw_enabled) {
        xt_before.resize(n_per * N);
        denoised.resize(n_per * N);
    }

    const bool use_cfg = p.guidance_scale > 1.0f && p.null_cond_emb != nullptr && p.H_enc > 0;
    std::vector<float>    vt_uncond;
    std::vector<float>    null_enc_hidden;
    std::vector<uint16_t> null_ca_mask;
    std::vector<double>   apg_momentum;
    if (use_cfg) {
        null_enc_hidden = make_null_enc_hidden(p.null_cond_emb, p.H_enc, enc_S, N);
        null_ca_mask    = make_visible_ca_mask(enc_S, S, N);
        apg_momentum.assign(n_per * N, 0.0);
    }

    const float * enc_hidden_active = p.enc_hidden;
    bool          cover_switched    = false;
    for (int step = 0; step < p.num_steps; step++) {
        if (p.on_step && !p.on_step(step, p.num_steps)) return false;
        const float t_curr          = p.schedule[step];
        bool        constants_dirty = (step == 0);

        if (p.cover_switch_step >= 0 && step >= p.cover_switch_step && !cover_switched &&
            p.context_switch != nullptr && p.enc_hidden_switch != nullptr) {
            cover_switched  = true;
            constants_dirty = true;
            splice_context_channels(input_buf, p.context_switch, T, N, in_ch, ctx_ch);
            enc_hidden_active = p.enc_hidden_switch;
            fill_ca_mask_rows(ca_mask, p.real_enc_S_switch, enc_S, S, N);
        }

        // splice x_t into the trailing Oc channels of the DiT input
        for (int b = 0; b < N; b++) {
            for (int t = 0; t < T; t++) {
                memcpy(&input_buf[(size_t) b * T * in_ch + (size_t) t * in_ch + ctx_ch],
                       &xt[(size_t) b * n_per + (size_t) t * Oc],
                       (size_t) Oc * sizeof(float));
            }
        }

        DitForwardInputs fin;
        fin.input_latents = input_buf.data();
        fin.T             = T;
        fin.N             = N;
        fin.enc_hidden    = enc_hidden_active;
        fin.enc_S         = enc_S;
        fin.H_enc         = p.H_enc;
        fin.t             = t_curr;
        // t_r == t (t_diff == 0, so time_embed_r sees 0) for both turbo and
        // base/sft: the reference passes timestep_r = timestep unconditionally.
        fin.t_r             = t_curr;
        fin.sa_mask_sw      = sa_mask.data();
        fin.ca_mask         = ca_mask.data();
        // CFG alternates cond/uncond enc_hidden and ca_mask on the shared graph
        // cache, so their uploads can never be skipped; positions and sa_mask
        // stay valid between steps and across the cond/uncond pair.
        fin.cond_dirty      = constants_dirty || use_cfg;
        fin.constants_dirty = constants_dirty;

        if (!dit_model_forward(m, fin, vt)) {
            fprintf(stderr, "[acestep-dit] sample: forward failed at step %d\n", step);
            return false;
        }

        if (use_cfg) {
            DitForwardInputs fin_uncond = fin;
            fin_uncond.enc_hidden       = null_enc_hidden.data();
            fin_uncond.ca_mask          = null_ca_mask.data();
            fin_uncond.constants_dirty  = false;
            if (!dit_model_forward(m, fin_uncond, vt_uncond)) {
                fprintf(stderr, "[acestep-dit] sample: uncond forward failed at step %d\n", step);
                return false;
            }
            dit_apg_guide(vt, vt_uncond, apg_momentum, p.guidance_scale, T, Oc, N);
        }

        // Euler ODE step. Final step integrates all the way to x0 (t_next = 0).
        const float t_next = (step == p.num_steps - 1) ? 0.0f : p.schedule[step + 1];
        const float dt     = t_curr - t_next;
        if (p.dcw_enabled) xt_before = xt;
        for (size_t i = 0; i < (size_t) n_per * N; i++) {
            xt[i] -= vt[i] * dt;
        }

        // Official ACE-Step DCW "double" mode. Reconstruct the model's clean
        // estimate from the pre-step latent, then push x_next's low/high Haar
        // bands away from it with complementary timestep schedules.
        if (p.dcw_enabled) {
            for (size_t i = 0; i < (size_t) n_per * N; ++i) {
                denoised[i] = xt_before[i] - vt[i] * t_curr;
            }
            dit_apply_haar_dcw(xt,
                               denoised,
                               T,
                               Oc,
                               N,
                               t_curr * p.dcw_scaler,
                               (1.0f - t_curr) * p.dcw_high_scaler);
        }

        inject_repaint_source(p, step, n_per * N, Oc, xt);
    }

    preserve_repaint_latent(p, n_per * N, Oc, xt);

    latent_out.swap(xt);
    return true;
}

static constexpr int FLOW_EDIT_BATCH_SIZE = 1;
static constexpr float ATTENTION_VISIBLE_VALUE = 0.0f;
static constexpr bool FLOW_EDIT_ROUND_NOISE_TO_BF16 = true;
static constexpr char FLOW_EDIT_PAIRED_FORWARD_ERROR[] =
    "[acestep-dit] flow-edit paired forward failed at step %d\n";
static constexpr char FLOW_EDIT_TARGET_FORWARD_ERROR[] =
    "[acestep-dit] flow-edit target forward failed at step %d\n";

struct FlowEditState {
    DitModel * model;
    const DitFlowEditParams & params;
    size_t count;
    std::vector<float> source;
    std::vector<float> edit;
    std::vector<float> target_running;
    std::vector<float> noise;
    std::vector<float> noisy_source;
    std::vector<float> noisy_target;
    std::vector<float> source_velocity;
    std::vector<float> target_velocity;
    std::vector<float> velocity_delta_sum;
    int min_step;
    int max_step;
    int64_t rng_subsequence = 0;

    FlowEditState(DitModel * model_value, const DitFlowEditParams & params_value)
        : model(model_value),
          params(params_value),
          count((size_t) params.T * model->cfg.out_channels),
          source(params.source_latents, params.source_latents + count),
          edit(source),
          noise(count),
          velocity_delta_sum(count),
          min_step((int) (params.num_steps * params.n_min)),
          max_step((int) (params.num_steps * params.n_max)) {}
};

static void fill_flow_edit_input(std::vector<float> & input, const std::vector<float> & latent,
                                 const float * context, int frames, int input_channels,
                                 int context_channels, int latent_channels) {
    for (int frame = 0; frame < frames; ++frame) {
        float * destination = input.data() + (size_t) frame * input_channels;
        memcpy(destination, context + (size_t) frame * context_channels,
               (size_t) context_channels * sizeof(float));
        memcpy(destination + context_channels, latent.data() + (size_t) frame * latent_channels,
               (size_t) latent_channels * sizeof(float));
    }
}

static void fill_flow_self_attention_mask(std::vector<uint16_t> & mask, int sequence,
                                          int sliding_window) {
    for (int query = 0; query < sequence; ++query) {
        for (int key = 0; key < sequence; ++key) {
            const int distance = query > key ? query - key : key - query;
            const bool visible = sliding_window <= 0 || sequence <= sliding_window ||
                                 distance <= sliding_window;
            mask[(size_t) query * sequence + key] =
                ggml_fp32_to_fp16(visible ? ATTENTION_VISIBLE_VALUE : -INFINITY);
        }
    }
}

static void fill_flow_cross_attention_mask(std::vector<uint16_t> & mask, int sequence,
                                           int encoded_sequence, int real_encoded_sequence) {
    for (int query = 0; query < sequence; ++query) {
        for (int key = 0; key < encoded_sequence; ++key) {
            mask[(size_t) query * encoded_sequence + key] =
                ggml_fp32_to_fp16(key < real_encoded_sequence ? ATTENTION_VISIBLE_VALUE : -INFINITY);
        }
    }
}

static DitForwardInputs make_flow_forward_inputs(const std::vector<float> & input,
                                                 const DitFlowEditCondition & condition,
                                                 const std::vector<uint16_t> & self_mask,
                                                 const std::vector<uint16_t> & cross_mask,
                                                 int frames, float timestep) {
    DitForwardInputs forward;
    forward.input_latents = input.data();
    forward.T = frames;
    forward.N = FLOW_EDIT_BATCH_SIZE;
    forward.enc_hidden = condition.enc_hidden;
    forward.enc_S = condition.enc_S;
    forward.H_enc = condition.H_enc;
    forward.t = timestep;
    forward.t_r = timestep;
    forward.sa_mask_sw = self_mask.data();
    forward.ca_mask = cross_mask.data();
    return forward;
}

static bool flow_edit_condition_is_valid(const DitFlowEditCondition & condition) {
    return condition.context_latents && condition.enc_hidden &&
           condition.enc_S > 0 && condition.H_enc > 0;
}

static bool flow_edit_forward(DitModel * model, const std::vector<float> & latent,
                              const DitFlowEditCondition & condition, int frames,
                              float timestep, std::vector<float> & velocity) {
    const DitConfig & config = model->cfg;
    const int latent_channels = config.out_channels;
    if (!flow_edit_condition_is_valid(condition) ||
        latent.size() != (size_t) frames * latent_channels) {
        return false;
    }
    const int context_channels = config.in_channels - latent_channels;
    const int sequence = frames / config.patch_size;
    std::vector<float> input((size_t) frames * config.in_channels);
    fill_flow_edit_input(input, latent, condition.context_latents, frames,
                         config.in_channels, context_channels, latent_channels);
    std::vector<uint16_t> self_mask((size_t) sequence * sequence);
    fill_flow_self_attention_mask(self_mask, sequence, config.sliding_window);
    const int real_encoded_sequence = condition.real_enc_S > 0
                                          ? std::min(condition.real_enc_S, condition.enc_S)
                                          : condition.enc_S;
    std::vector<uint16_t> cross_mask((size_t) condition.enc_S * sequence);
    fill_flow_cross_attention_mask(cross_mask, sequence, condition.enc_S, real_encoded_sequence);
    const DitForwardInputs forward =
        make_flow_forward_inputs(input, condition, self_mask, cross_mask, frames, timestep);
    return dit_model_forward(model, forward, velocity);
}

static void add_average_velocity_delta(std::vector<float> & delta,
                                       const std::vector<float> & target_velocity,
                                       const std::vector<float> & source_velocity, int averages) {
    for (size_t index = 0; index < delta.size(); ++index) {
        delta[index] += (target_velocity[index] - source_velocity[index]) / averages;
    }
}

static void integrate_flow_edit_delta(std::vector<float> & edit,
                                      const std::vector<float> & velocity_delta, float dt) {
    for (size_t index = 0; index < edit.size(); ++index) {
        edit[index] += dt * velocity_delta[index];
    }
}

static void integrate_flow_target(std::vector<float> & target,
                                  const std::vector<float> & velocity, float dt) {
    for (size_t index = 0; index < target.size(); ++index) {
        target[index] += dt * velocity[index];
    }
}

static void draw_flow_edit_noise(FlowEditState & state) {
  const int samples = (int)state.count;
  philox_randn_from((int64_t)state.params.seed, state.rng_subsequence,
                    state.noise.data(), samples, FLOW_EDIT_ROUND_NOISE_TO_BF16);
  state.rng_subsequence += samples;
}

static bool run_flow_edit_forward_pair(FlowEditState & state, float timestep) {
    if (flow_edit_forward(state.model, state.noisy_source, state.params.source,
                          state.params.T, timestep, state.source_velocity) &&
        flow_edit_forward(state.model, state.noisy_target, state.params.target,
                          state.params.T, timestep, state.target_velocity)) {
        return true;
    }
    return false;
}

static bool accumulate_flow_edit_averages(FlowEditState & state, int step, float timestep) {
    for (int average = 0; average < state.params.n_avg; ++average) {
        draw_flow_edit_noise(state);
        flow_edit_make_source(state.source, state.noise, timestep, state.noisy_source);
        flow_edit_make_target(state.edit, state.noisy_source, state.source, state.noisy_target);
        if (!run_flow_edit_forward_pair(state, timestep)) {
            fprintf(stderr, FLOW_EDIT_PAIRED_FORWARD_ERROR, step);
            return false;
        }
        add_average_velocity_delta(state.velocity_delta_sum, state.target_velocity,
                                   state.source_velocity, state.params.n_avg);
    }
    return true;
}

static bool run_flow_edit_delta_step(FlowEditState & state, int step, float timestep, float dt) {
    std::fill(state.velocity_delta_sum.begin(), state.velocity_delta_sum.end(),
              DIT_FLOW_EDIT_MIN_RATIO);
    if (!accumulate_flow_edit_averages(state, step, timestep)) return false;
    integrate_flow_edit_delta(state.edit, state.velocity_delta_sum, dt);
    return true;
}

static void initialize_flow_target(FlowEditState & state, float timestep) {
    draw_flow_edit_noise(state);
    flow_edit_make_source(state.source, state.noise, timestep, state.noisy_source);
    flow_edit_make_target(state.edit, state.noisy_source, state.source, state.target_running);
}

static bool run_flow_edit_target_step(FlowEditState & state, int step, float timestep, float dt) {
    if (state.target_running.empty()) initialize_flow_target(state, timestep);
    if (!flow_edit_forward(state.model, state.target_running, state.params.target,
                           state.params.T, timestep, state.target_velocity)) {
        fprintf(stderr, FLOW_EDIT_TARGET_FORWARD_ERROR, step);
        return false;
    }
    integrate_flow_target(state.target_running, state.target_velocity, dt);
    return true;
}

static bool run_flow_edit_step(FlowEditState & state, int step) {
    if (state.params.on_step && !state.params.on_step(step, state.params.num_steps)) return false;
    if (step < state.min_step) return true;
    const float timestep = state.params.schedule[step];
    const float next_timestep =
        next_schedule_timestep(state.params.schedule, step, state.params.num_steps);
    const float dt = next_timestep - timestep;
    if (step < state.max_step) return run_flow_edit_delta_step(state, step, timestep, dt);
    return run_flow_edit_target_step(state, step, timestep, dt);
}

static bool run_flow_edit_steps(FlowEditState & state) {
    for (int step = 0; step < state.params.num_steps; ++step) {
        if (!run_flow_edit_step(state, step)) return false;
    }
    return true;
}

static bool flow_edit_params_are_valid(DitModel * model, const DitFlowEditParams & params) {
    return model && params.source_latents && params.schedule && params.T > 0 &&
           params.num_steps > 0 && params.n_avg >= DIT_FLOW_EDIT_DEFAULT_AVERAGES &&
           params.n_min >= DIT_FLOW_EDIT_MIN_RATIO && params.n_min <= params.n_max &&
           params.n_max <= DIT_FLOW_EDIT_MAX_RATIO &&
           params.T % model->cfg.patch_size == 0;
}

bool dit_flow_edit(DitModel * model, const DitFlowEditParams & params,
                   std::vector<float> & latent_out) {
    if (!flow_edit_params_are_valid(model, params)) return false;
    FlowEditState state(model, params);
    if (!run_flow_edit_steps(state)) return false;
    if (params.on_step && !params.on_step(params.num_steps, params.num_steps)) return false;
    latent_out = state.target_running.empty() ? std::move(state.edit) : std::move(state.target_running);
    return true;
}

} // namespace tts_cpp::acestep
