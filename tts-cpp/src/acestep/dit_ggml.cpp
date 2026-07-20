#include "dit_ggml.h"

#include "dit_gguf.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

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

struct DitModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

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

    bool use_flash_attn = false;  // CPU target: F32 soft_max path
};

// ------------------------------------------------------------------ loaders
static float bf16_to_f32(uint16_t v) {
    ggml_bf16_t b;
    b.bits = v;
    return ggml_bf16_to_fp32(b);
}

// Create a weight tensor mirroring the GGUF meta tensor's type + shape.
static ggml_tensor * create_like(ggml_context * ctx, const DitGGUF & g, const std::string & name) {
    ggml_tensor * mt = dit_gmeta(g, name);
    if (!mt) {
        fprintf(stderr, "[acestep-dit] missing tensor: %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(ctx, mt->type, ggml_n_dims(mt), mt->ne);
    ggml_set_name(t, name.c_str());
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
// (kept in their native quant/precision type).
static void load_raw(ggml_tensor * dst, const DitGGUF & g, const std::string & name) {
    if (!dst) return;
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
                    data[(size_t) h * in_ch * P + p * in_ch + ic] = rd(h * P * in_ch + ic * P + p);
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

static void temb_create(DitTemb & w, ggml_context * ctx, const DitGGUF & g, const std::string & pfx) {
    w.linear_1_w  = create_like(ctx, g, pfx + ".linear_1.weight");
    w.linear_1_b  = create_f32_like(ctx, g, pfx + ".linear_1.bias");
    w.linear_2_w  = create_like(ctx, g, pfx + ".linear_2.weight");
    w.linear_2_b  = create_f32_like(ctx, g, pfx + ".linear_2.bias");
    w.time_proj_w = create_like(ctx, g, pfx + ".time_proj.weight");
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

    q = ggml_reshape_3d(ctx, q, D, Nh, S * N);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S * N);
    q = ggml_rope_ext(ctx, q, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    float         scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn  = attn_f32(ctx, q, k, v, mask, scale);
    attn                = ggml_reshape_3d(ctx, attn, Nh * D, S, N);
    return linear(ctx, ly->sa_o_proj, attn);
}

static ggml_tensor * build_cross_attn(ggml_context * ctx, DitModel * m, DitLayer * ly, ggml_tensor * norm_ca,
                                      ggml_tensor * enc, ggml_tensor * mask, int S, int enc_S, int N) {
    const DitConfig & c   = m->cfg;
    int               D   = c.head_dim;
    int               Nh  = c.n_heads;
    int               Nkv = c.n_kv_heads;

    ggml_tensor * q = linear(ctx, ly->ca_q_proj, norm_ca);
    ggml_tensor * k = linear(ctx, ly->ca_k_proj, enc);
    ggml_tensor * v = linear(ctx, ly->ca_v_proj, enc);

    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_reshape_4d(ctx, k, D, Nkv, enc_S, N);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_reshape_4d(ctx, v, D, Nkv, enc_S, N);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), as_f32(ctx, ly->ca_q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), as_f32(ctx, ly->ca_k_norm));

    float         scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn  = attn_f32(ctx, q, k, v, mask, scale);
    attn                = ggml_reshape_3d(ctx, attn, Nh * D, S, N);
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
                                 ggml_tensor * ca_mask, int S, int enc_S, int N) {
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
        ggml_tensor * ca_out  = build_cross_attn(ctx, m, ly, norm_ca, enc, ca_mask, S, enc_S, N);
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
DitModel * dit_model_load(const std::string & path, ggml_backend_t backend, bool verbose) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) return nullptr;

    DitModel * m = new DitModel();
    m->backend   = backend;
    if (!dit_gguf_read_config(g, m->cfg)) {
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }
    const DitConfig & c = m->cfg;
    const int         H = c.hidden_size;

    // enough overhead for all descriptors
    ggml_init_params ip{ ggml_tensor_overhead() * (size_t) (64 + 40 * c.n_layers), nullptr, /*no_alloc=*/true };
    m->weight_ctx = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    temb_create(m->time_embed, ctx, g, "decoder.time_embed");
    temb_create(m->time_embed_r, ctx, g, "decoder.time_embed_r");

    m->proj_in_w = create_2d_f32(ctx, (int64_t) c.in_channels * c.patch_size, H, "decoder.proj_in.1.weight");
    m->proj_in_b = create_f32_like(ctx, g, "decoder.proj_in.1.bias");
    m->cond_emb_w = create_like(ctx, g, "decoder.condition_embedder.weight");
    m->cond_emb_b = create_f32_like(ctx, g, "decoder.condition_embedder.bias");
    m->cfg.enc_hidden_size = (int) m->cond_emb_w->ne[0];  // [H_enc, H] -> H_enc

    m->layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        DitLayer &  ly = m->layers[i];
        std::string p  = "decoder.layers." + std::to_string(i);
        ly.self_attn_norm = create_f32_like(ctx, g, p + ".self_attn_norm.weight");
        ly.sa_q_proj      = create_like(ctx, g, p + ".self_attn.q_proj.weight");
        ly.sa_k_proj      = create_like(ctx, g, p + ".self_attn.k_proj.weight");
        ly.sa_v_proj      = create_like(ctx, g, p + ".self_attn.v_proj.weight");
        ly.sa_q_norm      = create_f32_like(ctx, g, p + ".self_attn.q_norm.weight");
        ly.sa_k_norm      = create_f32_like(ctx, g, p + ".self_attn.k_norm.weight");
        ly.sa_o_proj      = create_like(ctx, g, p + ".self_attn.o_proj.weight");
        ly.cross_attn_norm = create_f32_like(ctx, g, p + ".cross_attn_norm.weight");
        ly.ca_q_proj      = create_like(ctx, g, p + ".cross_attn.q_proj.weight");
        ly.ca_k_proj      = create_like(ctx, g, p + ".cross_attn.k_proj.weight");
        ly.ca_v_proj      = create_like(ctx, g, p + ".cross_attn.v_proj.weight");
        ly.ca_q_norm      = create_f32_like(ctx, g, p + ".cross_attn.q_norm.weight");
        ly.ca_k_norm      = create_f32_like(ctx, g, p + ".cross_attn.k_norm.weight");
        ly.ca_o_proj      = create_like(ctx, g, p + ".cross_attn.o_proj.weight");
        ly.mlp_norm       = create_f32_like(ctx, g, p + ".mlp_norm.weight");
        ly.gate_proj      = create_like(ctx, g, p + ".mlp.gate_proj.weight");
        ly.up_proj        = create_like(ctx, g, p + ".mlp.up_proj.weight");
        ly.down_proj      = create_like(ctx, g, p + ".mlp.down_proj.weight");
        ly.scale_shift_table = create_f32_like(ctx, g, p + ".scale_shift_table");
        ly.layer_type     = (i % 2 == 0) ? 0 : 1;
    }

    m->norm_out        = create_f32_like(ctx, g, "decoder.norm_out.weight");
    m->out_scale_shift = create_f32_like(ctx, g, "decoder.scale_shift_table");
    m->proj_out_w      = create_2d_f32(ctx, H, (int64_t) c.out_channels * c.patch_size, "decoder.proj_out.1.weight");
    m->proj_out_b      = create_f32_like(ctx, g, "decoder.proj_out.1.bias");
    m->scalar_one      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(m->scalar_one, "scalar_one");

    m->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-dit] failed to allocate weight buffer\n");
        ggml_free(ctx);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }

    // upload
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

    dit_gguf_close(g);

    if (verbose) {
        fprintf(stderr, "[acestep-dit] loaded %s: %.1f MB, %d layers H=%d Nh=%d/%d D=%d\n", path.c_str(),
                (float) ggml_backend_buffer_get_size(m->weight_buf) / (1024 * 1024), c.n_layers, H, c.n_heads,
                c.n_kv_heads, c.head_dim);
    }
    return m;
}

void dit_model_free(DitModel * m) {
    if (!m) return;
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    delete m;
}

const DitConfig & dit_model_config(const DitModel * m) { return m->cfg; }

size_t dit_model_weight_bytes(const DitModel * m) {
    return (m && m->weight_buf) ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
}

bool dit_model_forward(DitModel * m, const DitForwardInputs & in, std::vector<float> & velocity_out) {
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

    const size_t nodes = (size_t) 8192;
    ggml_init_params gp{ ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context *   ctx = ggml_init(gp);

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.in_channels, T, N);
    ggml_set_input(input);
    ggml_tensor * enc_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in.H_enc, enc_S, N);
    ggml_set_input(enc_hidden);
    ggml_tensor * t_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(t_val);
    ggml_tensor * tr_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(tr_val);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) S * N);
    ggml_set_input(positions);

    ggml_tensor * sa_mask = nullptr;
    if (in.sa_mask_sw) {
        sa_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, N);
        ggml_set_input(sa_mask);
    }
    ggml_tensor * ca_mask = nullptr;
    if (in.ca_mask) {
        ca_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, enc_S, S, 1, N);
        ggml_set_input(ca_mask);
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
    ggml_tensor * patched = ggml_reshape_3d(ctx, input, c.in_channels * P, S, N);
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
    ggml_tensor * output    = linear_b(ctx, m->proj_out_w, m->proj_out_b, norm_out);
    output                  = ggml_reshape_3d(ctx, output, c.out_channels, T, N);
    ggml_set_output(output);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(gf, output);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-dit] forward alloc failed (T=%d N=%d)\n", T, N);
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(input, in.input_latents, 0, (size_t) c.in_channels * T * N * sizeof(float));
    ggml_backend_tensor_set(enc_hidden, in.enc_hidden, 0, (size_t) in.H_enc * enc_S * N * sizeof(float));
    ggml_backend_tensor_set(t_val, &in.t, 0, sizeof(float));
    ggml_backend_tensor_set(tr_val, &in.t_r, 0, sizeof(float));
    std::vector<int32_t> pos((size_t) S * N);
    for (int n = 0; n < N; n++)
        for (int s = 0; s < S; s++) pos[(size_t) n * S + s] = s;
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (sa_mask) ggml_backend_tensor_set(sa_mask, in.sa_mask_sw, 0, (size_t) S * S * N * sizeof(uint16_t));
    if (ca_mask) ggml_backend_tensor_set(ca_mask, in.ca_mask, 0, (size_t) enc_S * S * N * sizeof(uint16_t));

    int rc = ggml_backend_graph_compute(m->backend, gf);
    if (rc != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }

    velocity_out.resize((size_t) c.out_channels * T * N);
    ggml_backend_tensor_get(output, velocity_out.data(), 0, ggml_nbytes(output));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return true;
}

void dit_build_schedule(float shift, int num_steps, std::vector<float> & schedule_out) {
    schedule_out.resize(num_steps);
    for (int i = 0; i < num_steps; i++) {
        float t          = 1.0f - (float) i / (float) num_steps;
        schedule_out[i]  = shift * t / (1.0f + (shift - 1.0f) * t);
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
    for (int b = 0; b < N; b++) {
        int re = p.real_enc_S ? p.real_enc_S[b] : enc_S;
        for (int qi = 0; qi < S; qi++) {
            for (int ki = 0; ki < enc_S; ki++) {
                float v = (ki < re) ? 0.0f : -INFINITY;
                ca_mask[(size_t) b * enc_S * S + (size_t) qi * enc_S + ki] = ggml_fp32_to_fp16(v);
            }
        }
    }

    // x_t (current noisy latent) starts at the supplied noise.
    std::vector<float> xt(p.noise, p.noise + (size_t) n_per * N);

    // Per-step DiT input [in_ch, T, N]: context channels are constant, the last
    // Oc channels carry x_t and are refreshed each step.
    std::vector<float> input_buf((size_t) in_ch * T * N);
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            memcpy(&input_buf[(size_t) b * T * in_ch + (size_t) t * in_ch],
                   &p.context_latents[(size_t) b * T * ctx_ch + (size_t) t * ctx_ch],
                   (size_t) ctx_ch * sizeof(float));
        }
    }

    std::vector<float> vt;
    for (int step = 0; step < p.num_steps; step++) {
        const float t_curr = p.schedule[step];

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
        fin.enc_hidden    = p.enc_hidden;
        fin.enc_S         = enc_S;
        fin.H_enc         = p.H_enc;
        fin.t             = t_curr;
        // t_r == t (t_diff == 0, so time_embed_r sees 0). Holds for turbo
        // text2music, which is also why the sampler runs a single conditional
        // pass (N == 1, no CFG). base/sft (50-step, CFG) parity is not yet
        // verified against the reference and would need t_r / uncond wiring.
        fin.t_r           = t_curr;
        fin.sa_mask_sw    = sa_mask.data();
        fin.ca_mask       = ca_mask.data();

        if (!dit_model_forward(m, fin, vt)) {
            fprintf(stderr, "[acestep-dit] sample: forward failed at step %d\n", step);
            return false;
        }

        // Euler ODE step. Final step integrates all the way to x0 (t_next = 0).
        const float t_next = (step == p.num_steps - 1) ? 0.0f : p.schedule[step + 1];
        const float dt     = t_curr - t_next;
        for (size_t i = 0; i < (size_t) n_per * N; i++) {
            xt[i] -= vt[i] * dt;
        }
    }

    latent_out.swap(xt);
    return true;
}

} // namespace tts_cpp::acestep
