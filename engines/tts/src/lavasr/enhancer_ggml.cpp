#include "enhancer_ggml.h"

#include "enhancer.h"      // enhance_with(), scalar enhance()
#include "enhancer_core.h" // EnhancerWeights, enhancer_spec_forward()
#include "../backend_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

// =============================================================================
// Persistent GGML state
// =============================================================================

namespace {

struct BlockW {
    ggml_tensor * dw_w   = nullptr; // depthwise conv   [K, 1, C]
    ggml_tensor * dwt_w  = nullptr; // dw kernel C-fastest [C, 1, K] (CWHN path)
    ggml_tensor * dw_b   = nullptr; //                  [C]
    ggml_tensor * norm_g = nullptr; // layer-norm gamma [C]
    ggml_tensor * norm_b = nullptr; //            beta  [C]
    ggml_tensor * pw1_w  = nullptr; // pointwise 1x1     [C, F]
    ggml_tensor * pw1_b  = nullptr; //                  [F]
    ggml_tensor * pw2_w  = nullptr; // pointwise 1x1     [F, C]
    ggml_tensor * pw2_b  = nullptr; //                  [C]
    ggml_tensor * gamma  = nullptr; // per-channel scale [C]
};

} // namespace

struct EnhancerGgml {
    ggml_backend_t        backend = nullptr; // borrowed
    ggml_context *        wctx    = nullptr; // weight tensor metadata
    ggml_backend_buffer_t wbuf    = nullptr; // weight data on the backend
    ggml_gallocr_t        allocr  = nullptr; // reusable compute allocator

    // Native CONV_2D_DW (Vulkan) vs im2col+matmul fallback (Metal/CPU): the k=7
    // per-channel matvecs are GPU-pathological, but Metal has no CONV_2D_DW.
    bool use_dw_direct = false;

    // Geometry snapshot.
    int   C = 0, F = 0, n_mels = 0, K = 0, n_blocks = 0, spec_bins = 0;
    float clip_max = 1000.0f;
    float ln_eps   = 1e-6f;

    // Weight handles.
    ggml_tensor *       embed_w = nullptr;   // [K, n_mels, C]
    ggml_tensor *       embed_b = nullptr;   // [1, C]
    ggml_tensor *       norm_g  = nullptr;   // [C]
    ggml_tensor *       norm_b  = nullptr;   // [C]
    std::vector<BlockW> blocks;
    ggml_tensor *       final_norm_g = nullptr; // [C]
    ggml_tensor *       final_norm_b = nullptr; // [C]
    ggml_tensor *       spec_w = nullptr;    // [1, C, 2*spec_bins]
    ggml_tensor *       spec_b = nullptr;    // [1, 2*spec_bins]
};

namespace {

// -----------------------------------------------------------------------------
// Graph-building primitives (F32, "same" symmetric zero-padding).
//
// These mirror the campplus / supertonic_vocoder idioms but use symmetric
// im2col padding (p0 = pad) rather than causal replicate padding, matching the
// enhancer's PyTorch Conv1d(padding=K/2) semantics.  All intermediates are F32
// (im2col dst_type F32) so parity with the scalar core holds — the stock
// ggml_conv_1d helpers use an F16 im2col that is too coarse for the 3e-3
// tolerance the enhancer is validated against.
// -----------------------------------------------------------------------------

// fp32-precise matmul: parity with the scalar core requires fp32 arithmetic
// even on GPUs whose default F32 matmul multiplies in fp16 (e.g. Vulkan).
static ggml_tensor * mul_mat_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * r = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(r, GGML_PREC_F32);
    return r;
}

// Regular / pointwise Conv1d.  kernel [K, IC, OC], input [T, IC, 1] -> [T, OC, 1].
// bias (optional) [1, OC] is broadcast over the time axis.
ggml_tensor * conv1d_same(ggml_context * ctx, ggml_tensor * kernel,
                          ggml_tensor * input, ggml_tensor * bias, int pad) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input,
                                       /*s0=*/1, /*s1=*/0, /*p0=*/pad, /*p1=*/0,
                                       /*d0=*/1, /*d1=*/0, /*is_2D=*/false,
                                       GGML_TYPE_F32);
    ggml_tensor * r = mul_mat_f32(
        ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    r = ggml_reshape_3d(ctx, r, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
    if (bias) {
        r = ggml_add(ctx, r, bias);
    }
    return r;
}

// Depthwise Conv1d (groups == C).  kernel [K, 1, C], input [T, C, 1] -> [T, C, 1].
// bias (optional) [1, C] broadcast over time.  Mirrors ggml_conv_1d_dw but with
// an F32 im2col and symmetric zero padding.  With dw_direct the whole conv is
// one native CONV_2D_DW op (1D as W-axis 2D; same taps, same accumulation order).
ggml_tensor * depthwise_same(ggml_context * ctx, ggml_tensor * kernel,
                             ggml_tensor * input, ggml_tensor * bias, int pad,
                             bool dw_direct) {
    ggml_tensor * y;
    if (dw_direct) {
        ggml_tensor * k4 = ggml_reshape_4d(ctx, kernel, kernel->ne[0], 1, 1,
                                           kernel->ne[2]);              // [K,1,1,C]
        ggml_tensor * x4 = ggml_reshape_4d(ctx, input, input->ne[0], 1,
                                           input->ne[1], input->ne[2]); // [T,1,C,1]
        y = ggml_conv_2d_dw_direct(ctx, k4, x4, 1, 1, pad, 0, 1, 1);    // [T,1,C,1]
    } else {
        ggml_tensor * new_b = ggml_reshape_4d(ctx, input, input->ne[0], 1,
                                              input->ne[1], input->ne[2]); // [T,1,C,1]
        ggml_tensor * im2col = ggml_im2col(ctx, kernel, new_b,
                                           /*s0=*/1, /*s1=*/0, /*p0=*/pad, /*p1=*/0,
                                           /*d0=*/1, /*d1=*/0, /*is_2D=*/false,
                                           GGML_TYPE_F32); // [K, T, C, 1]
        y = mul_mat_f32(ctx, im2col, kernel);              // [T, 1, C, 1]
    }
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);    // [T, C, 1]
    if (bias) {
        y = ggml_add(ctx, y, bias);
    }
    return y;
}

// Depthwise Conv1d, channel-major [C,T,1].  cwhn: native CONV_2D_DW on a
// channels-fastest [T,1,C,1] view; else time-major wrapped in two transposes.
ggml_tensor * depthwise_cm(ggml_context * ctx, const BlockW & b, ggml_tensor * x,
                           int pad, bool cwhn, bool dw_direct) {
    ggml_tensor * y;
    if (cwhn) {
        ggml_tensor * k4 = ggml_permute(ctx, b.dwt_w, 3, 2, 0, 1);           // [K,1,1,C]
        ggml_tensor * x4 = ggml_permute(ctx, x, 2, 0, 1, 3);                 // [T,1,C,1]
        y = ggml_conv_2d_dw_direct(ctx, k4, x4, 1, 1, pad, 0, 1, 1);         // [T,1,C,1]
        y = ggml_permute(ctx, y, 1, 2, 0, 3);                                // [C,T,1]
    } else {
        ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [T,C,1]
        y = depthwise_same(ctx, b.dw_w, xt, nullptr, pad, dw_direct);
        y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));                // [C,T,1]
    }
    return ggml_add(ctx, y, b.dw_b);
}

// LayerNorm over the channel dim on channel-major x [C, T, 1]; gamma/beta [C]
// broadcast per frame — ggml_norm reduces over ne0 = C directly, no permutes.
ggml_tensor * layernorm_cm(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * gamma, ggml_tensor * beta, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, gamma);
    return ggml_add(ctx, y, beta);
}

ggml_tensor * new_1d(ggml_context * ctx, int64_t ne0, const char * name) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    return t;
}
ggml_tensor * new_2d(ggml_context * ctx, int64_t ne0, int64_t ne1, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    return t;
}
ggml_tensor * new_3d(ggml_context * ctx, int64_t ne0, int64_t ne1, int64_t ne2,
                     const char * name) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(t, name);
    return t;
}

// Upload a named host weight into a declared tensor, checking the element count.
bool upload_weight(ggml_tensor * t, const EnhancerWeights & w,
                   const std::string & name) {
    if (!w.has(name)) {
        std::fprintf(stderr, "lavasr enhancer ggml: missing weight '%s'\n", name.c_str());
        return false;
    }
    const std::vector<float> & src = w.get(name).data;
    if (static_cast<int64_t>(src.size()) != ggml_nelements(t)) {
        std::fprintf(stderr,
                     "lavasr enhancer ggml: '%s' element count mismatch (have %zu, want %lld)\n",
                     name.c_str(), src.size(),
                     static_cast<long long>(ggml_nelements(t)));
        return false;
    }
    ggml_backend_tensor_set(t, src.data(), 0, src.size() * sizeof(float));
    return true;
}

// Upload a depthwise kernel transposed to C-fastest memory (t is [C, 1, K]) for
// the CWHN direct conv; host data is [C][K] (PyTorch [C,1,K] flattened).
bool upload_weight_dw_cm(ggml_tensor * t, const EnhancerWeights & w,
                         const std::string & name) {
    if (!w.has(name)) {
        std::fprintf(stderr, "lavasr enhancer ggml: missing weight '%s'\n", name.c_str());
        return false;
    }
    const std::vector<float> & src = w.get(name).data;
    const int64_t C = t->ne[0], K = t->ne[2];
    if (static_cast<int64_t>(src.size()) != C * K) {
        std::fprintf(stderr,
                     "lavasr enhancer ggml: '%s' element count mismatch (have %zu, want %lld)\n",
                     name.c_str(), src.size(), static_cast<long long>(C * K));
        return false;
    }
    std::vector<float> tr(src.size());
    for (int64_t c = 0; c < C; c++) {
        for (int64_t k = 0; k < K; k++) {
            tr[static_cast<size_t>(k * C + c)] = src[static_cast<size_t>(c * K + k)];
        }
    }
    ggml_backend_tensor_set(t, tr.data(), 0, tr.size() * sizeof(float));
    return true;
}

} // namespace

// =============================================================================
// Lifecycle
// =============================================================================

void enhancer_ggml_free(EnhancerGgml * g) {
    if (!g) {
        return;
    }
    if (g->allocr) {
        ggml_gallocr_free(g->allocr);
    }
    if (g->wbuf) {
        ggml_backend_buffer_free(g->wbuf);
    }
    if (g->wctx) {
        ggml_free(g->wctx);
    }
    delete g;
}

EnhancerGgml * enhancer_ggml_create(const EnhancerWeights & w, ggml_backend_t backend) {
    if (!backend) {
        return nullptr;
    }

    EnhancerGgml * g = new EnhancerGgml();
    g->backend       = backend;
    g->use_dw_direct = ::tts_cpp::detail::backend_is_vulkan(backend);
    g->C         = w.dim;
    g->F         = w.ffn_dim;
    g->n_mels    = w.n_mels;
    g->K         = w.kernel;
    g->n_blocks  = w.n_blocks;
    g->spec_bins = w.spec_bins;
    g->clip_max  = w.clip_max;
    g->ln_eps    = w.ln_eps;

    const int C = g->C, F = g->F, M = g->n_mels, K = g->K, B = g->spec_bins;

    // Metadata context sized for every weight tensor (10 per block + head/tail).
    const int n_tensors = 2 /*embed*/ + 2 /*norm*/ + 10 * g->n_blocks +
                          2 /*final_norm*/ + 2 /*spec_head*/ + 8 /*slack*/;
    ggml_init_params ip = {
        /*.mem_size   =*/ static_cast<size_t>(n_tensors) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    g->wctx = ggml_init(ip);
    if (!g->wctx) {
        std::fprintf(stderr, "lavasr enhancer ggml: weight ctx init failed\n");
        enhancer_ggml_free(g);
        return nullptr;
    }

    // Declare weight tensors in the ggml conv layout (see the header note on the
    // PyTorch<->ggml flat-memory equivalence: [out,in,K] <-> [K,in,out]).
    g->embed_w = new_3d(g->wctx, K, M, C, "embed.w");
    g->embed_b = new_2d(g->wctx, 1, C, "embed.b");
    g->norm_g  = new_1d(g->wctx, C, "norm.g");
    g->norm_b  = new_1d(g->wctx, C, "norm.b");

    g->blocks.resize(g->n_blocks);
    for (int i = 0; i < g->n_blocks; i++) {
        const std::string p = "blk" + std::to_string(i) + ".";
        BlockW & b = g->blocks[static_cast<size_t>(i)];
        b.dw_w   = new_3d(g->wctx, K, 1, C, (p + "dw.w").c_str());
        // CWHN transpose is consumed only on the Vulkan dw-direct path; skip elsewhere.
        if (g->use_dw_direct) b.dwt_w = new_3d(g->wctx, C, 1, K, (p + "dw.wt").c_str());
        b.dw_b   = new_1d(g->wctx, C, (p + "dw.b").c_str());
        b.norm_g = new_1d(g->wctx, C, (p + "norm.g").c_str());
        b.norm_b = new_1d(g->wctx, C, (p + "norm.b").c_str());
        b.pw1_w  = new_2d(g->wctx, C, F, (p + "pw1.w").c_str());
        b.pw1_b  = new_1d(g->wctx, F, (p + "pw1.b").c_str());
        b.pw2_w  = new_2d(g->wctx, F, C, (p + "pw2.w").c_str());
        b.pw2_b  = new_1d(g->wctx, C, (p + "pw2.b").c_str());
        b.gamma  = new_1d(g->wctx, C, (p + "gamma").c_str());
    }

    g->final_norm_g = new_1d(g->wctx, C, "final_norm.g");
    g->final_norm_b = new_1d(g->wctx, C, "final_norm.b");
    g->spec_w       = new_3d(g->wctx, 1, C, 2 * B, "spec.w");
    g->spec_b       = new_2d(g->wctx, 1, 2 * B, "spec.b");

    // Allocate + upload.
    g->wbuf = ggml_backend_alloc_ctx_tensors(g->wctx, backend);
    if (!g->wbuf) {
        std::fprintf(stderr, "lavasr enhancer ggml: weight buffer alloc failed\n");
        enhancer_ggml_free(g);
        return nullptr;
    }

    bool ok = true;
    ok = ok && upload_weight(g->embed_w, w, "enhancer.embed.weight");
    ok = ok && upload_weight(g->embed_b, w, "enhancer.embed.bias");
    ok = ok && upload_weight(g->norm_g, w, "enhancer.norm.weight");
    ok = ok && upload_weight(g->norm_b, w, "enhancer.norm.bias");
    for (int i = 0; ok && i < g->n_blocks; i++) {
        const std::string p = "enhancer.block." + std::to_string(i) + ".";
        const BlockW &    b = g->blocks[static_cast<size_t>(i)];
        ok = ok && upload_weight(b.dw_w, w, p + "dwconv.weight");
        if (g->use_dw_direct) ok = ok && upload_weight_dw_cm(b.dwt_w, w, p + "dwconv.weight");
        ok = ok && upload_weight(b.dw_b, w, p + "dwconv.bias");
        ok = ok && upload_weight(b.norm_g, w, p + "norm.weight");
        ok = ok && upload_weight(b.norm_b, w, p + "norm.bias");
        ok = ok && upload_weight(b.pw1_w, w, p + "pwconv1.weight");
        ok = ok && upload_weight(b.pw1_b, w, p + "pwconv1.bias");
        ok = ok && upload_weight(b.pw2_w, w, p + "pwconv2.weight");
        ok = ok && upload_weight(b.pw2_b, w, p + "pwconv2.bias");
        ok = ok && upload_weight(b.gamma, w, p + "gamma");
    }
    ok = ok && upload_weight(g->final_norm_g, w, "enhancer.final_norm.weight");
    ok = ok && upload_weight(g->final_norm_b, w, "enhancer.final_norm.bias");
    ok = ok && upload_weight(g->spec_w, w, "spec_head.out.weight");
    ok = ok && upload_weight(g->spec_b, w, "spec_head.out.bias");
    if (!ok) {
        enhancer_ggml_free(g);
        return nullptr;
    }

    g->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!g->allocr) {
        std::fprintf(stderr, "lavasr enhancer ggml: gallocr init failed\n");
        enhancer_ggml_free(g);
        return nullptr;
    }
    return g;
}

// =============================================================================
// Forward
// =============================================================================

bool enhancer_ggml_spec_forward(EnhancerGgml * g,
                                const std::vector<float> & mel, int T,
                                std::vector<float> & real, std::vector<float> & imag) {
    if (!g || T <= 0) {
        return false;
    }
    const int M = g->n_mels, K = g->K, B = g->spec_bins;
    if (static_cast<int64_t>(mel.size()) != static_cast<int64_t>(M) * T) {
        std::fprintf(stderr, "lavasr enhancer ggml: mel size %zu != %d*%d\n",
                     mel.size(), M, T);
        return false;
    }
    const int pad = K / 2;

    // Build the compute graph in a scratch context (no_alloc: tensor metadata
    // only; data lives in the gallocr-managed buffer after alloc_graph).
    const int    MAX_NODES = 2048;
    const size_t buf_size  = ggml_tensor_overhead() * MAX_NODES +
                            ggml_graph_overhead_custom(MAX_NODES, false);
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params     p   = {buf_size, buf.data(), /*no_alloc=*/true};
    ggml_context *       ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

    // Input: mel [T, n_mels, 1] (holds mel[c*T+t] = ne0=T, ne1=C).
    ggml_tensor * mel_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, M, 1);
    ggml_set_name(mel_in, "mel");
    ggml_set_input(mel_in);

    // embed Conv1d(n_mels -> C, k) "same" (time-major, one im2col — runs once),
    // then one transpose into channel-major [C,T,1] for the whole backbone.
    ggml_tensor * x = conv1d_same(ctx, g->embed_w, mel_in, g->embed_b, pad); // [T,C,1]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));                    // [C,T,1]
    x = layernorm_cm(ctx, x, g->norm_g, g->norm_b, g->ln_eps);

    // CWHN routing needs nb[1] > nb[0] on the input view, i.e. T > 1
    // (ggml_is_contiguous_channels); degenerate T falls back to time-major.
    const bool dw_cwhn = g->use_dw_direct && T > 1;

    // ConvNeXt blocks, all channel-major: the pointwise convs are bare GEMMs
    // and LayerNorm reduces over ne0 = C directly — no per-layer transposes.
    for (int i = 0; i < g->n_blocks; i++) {
        const BlockW & b   = g->blocks[static_cast<size_t>(i)];
        ggml_tensor *  res = x;
        ggml_tensor *  y   = depthwise_cm(ctx, b, x, pad, dw_cwhn,
                                          g->use_dw_direct);              // [C,T,1]
        y                  = layernorm_cm(ctx, y, b.norm_g, b.norm_b, g->ln_eps);
        y                  = ggml_add(ctx, mul_mat_f32(ctx, b.pw1_w, y),
                                      b.pw1_b);                           // [F,T,1]
        y                  = ggml_gelu_erf(ctx, y);
        y                  = ggml_add(ctx, mul_mat_f32(ctx, b.pw2_w, y),
                                      b.pw2_b);                           // [C,T,1]
        y                  = ggml_mul(ctx, y, b.gamma);                   // per-channel scale
        x                  = ggml_add(ctx, res, y);
    }

    x = layernorm_cm(ctx, x, g->final_norm_g, g->final_norm_b, g->ln_eps);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // back to [T,C,1] for the head

    // Spec head: Linear(C -> 2*spec_bins), split log-mag / phase, then
    //   mag = clip(exp(log-mag), max=clip_max); real = mag*cos(phase);
    //   imag = mag*sin(phase).
    ggml_tensor * lin = conv1d_same(ctx, g->spec_w, x, g->spec_b, 0); // [T, 2B, 1]
    lin               = ggml_reshape_2d(ctx, lin, T, 2 * B);         // [T, 2B]

    ggml_tensor * logmag =
        ggml_cont(ctx, ggml_view_2d(ctx, lin, T, B, lin->nb[1], 0));
    ggml_tensor * phase = ggml_cont(
        ctx, ggml_view_2d(ctx, lin, T, B, lin->nb[1], static_cast<size_t>(B) * lin->nb[1]));

    ggml_tensor * mag = ggml_exp(ctx, logmag);
    mag               = ggml_clamp(ctx, mag, 0.0f, g->clip_max);
    ggml_tensor * re  = ggml_mul(ctx, mag, ggml_cos(ctx, phase)); // [T, B]
    ggml_tensor * im  = ggml_mul(ctx, mag, ggml_sin(ctx, phase)); // [T, B]
    ggml_set_name(re, "real");
    ggml_set_output(re);
    ggml_set_name(im, "imag");
    ggml_set_output(im);
    ggml_build_forward_expand(gf, re);
    ggml_build_forward_expand(gf, im);

    // Allocate + run.
    if (!ggml_gallocr_reserve(g->allocr, gf)) {
        std::fprintf(stderr, "lavasr enhancer ggml: gallocr reserve failed\n");
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(g->allocr, gf)) {
        std::fprintf(stderr, "lavasr enhancer ggml: gallocr alloc failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(mel_in, mel.data(), 0,
                            static_cast<size_t>(M) * T * sizeof(float));

    if (ggml_backend_graph_compute(g->backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "lavasr enhancer ggml: graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read back real/imag.  ne = [T, B] gives x[f*T+t] linearised, matching the
    // scalar core's [spec_bins * T] channel-major output.
    real.assign(static_cast<size_t>(B) * T, 0.0f);
    imag.assign(static_cast<size_t>(B) * T, 0.0f);
    ggml_backend_tensor_get(re, real.data(), 0, real.size() * sizeof(float));
    ggml_backend_tensor_get(im, imag.data(), 0, imag.size() * sizeof(float));

    ggml_free(ctx);
    return true;
}

// =============================================================================
// Pipeline entry point (GPU neural core + shared scalar DSP)
// =============================================================================

std::vector<float> enhance(const EnhancerWeights & w, EnhancerGgml * gpu,
                           const std::vector<float> & pcm_in, int sr_in) {
    if (!gpu) {
        return enhance(w, pcm_in, sr_in);
    }
    return enhance_with(
        w, pcm_in, sr_in,
        [gpu, &w](const std::vector<float> & mel, int T, std::vector<float> & real,
                  std::vector<float> & imag) {
            if (!enhancer_ggml_spec_forward(gpu, mel, T, real, imag)) {
                // Robustness: a runtime backend failure falls back to the
                // scalar core so the pipeline still produces valid audio.
                enhancer_spec_forward(w, mel, T, real, imag);
            }
        });
}

} // namespace tts_cpp::lavasr
