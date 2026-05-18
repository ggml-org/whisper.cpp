#include "campplus.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declaration for the ggml-backed implementation lives in
// campplus_forward.inc (included at the bottom of this file).
static bool campplus_embed_ggml(const std::vector<float> & fbank_t_by_c, int T,
                                const campplus_weights & w, ggml_backend_t backend,
                                std::vector<float> & out);

// =============================================================================
// GGUF loader helpers
// =============================================================================

static bool copy_f32(ggml_context * ctx, const char * name,
                     std::vector<float> & out)
{
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) { fprintf(stderr, "campplus_load: missing tensor %s\n", name); return false; }
    out.resize(ggml_nelements(t));
    std::memcpy(out.data(), ggml_get_data(t), ggml_nbytes(t));
    return true;
}

static bool load_bn(ggml_context * ctx, const std::string & base, campplus_bn & bn)
{
    if (!copy_f32(ctx, (base + "/s").c_str(), bn.scale)) return false;
    if (!copy_f32(ctx, (base + "/b").c_str(), bn.shift)) return false;
    return true;
}

static bool load_conv1d(ggml_context * ctx,
                        const std::string & w_name,
                        const std::string & b_name_or_empty,
                        int k, int C_in, int C_out,
                        int stride, int pad, int dilation,
                        campplus_conv & conv)
{
    if (!copy_f32(ctx, w_name.c_str(), conv.w)) return false;
    if ((int64_t)conv.w.size() != (int64_t)k * C_in * C_out) {
        fprintf(stderr, "campplus_load: %s size mismatch (have %zu, want %d)\n",
                w_name.c_str(), conv.w.size(), k * C_in * C_out);
        return false;
    }
    if (!b_name_or_empty.empty()) {
        if (!copy_f32(ctx, b_name_or_empty.c_str(), conv.b)) return false;
    }
    conv.C_out = C_out;
    conv.C_in  = C_in;
    conv.k     = k;
    conv.stride_w = stride;
    conv.pad_w    = pad;
    conv.dilation_w = dilation;
    conv.is_2d = false;
    return true;
}

static bool load_conv2d(ggml_context * ctx,
                        const std::string & w_name,
                        int kH, int kW, int C_in, int C_out,
                        int sH, int sW, int pH, int pW,
                        campplus_conv & conv)
{
    if (!copy_f32(ctx, w_name.c_str(), conv.w)) return false;
    if ((int64_t)conv.w.size() != (int64_t)kH * kW * C_in * C_out) {
        fprintf(stderr, "campplus_load: %s size mismatch\n", w_name.c_str());
        return false;
    }
    conv.C_out = C_out;
    conv.C_in  = C_in;
    conv.kH = kH; conv.kW = kW;
    conv.stride_h = sH; conv.stride_w = sW;
    conv.pad_h = pH; conv.pad_w = pW;
    conv.dilation_h = 1; conv.dilation_w = 1;
    conv.is_2d = true;
    return true;
}

static bool load_res_block(ggml_context * ctx,
                           const std::string & base,
                           bool has_shortcut,
                           int in_planes, int planes, int stride,
                           campplus_res_block & blk)
{
    if (!load_conv2d(ctx, base + "/conv1/weight", 3, 3, in_planes, planes, stride, 1, 1, 1, blk.conv1)) return false;
    if (!load_bn    (ctx, base + "/bn1", blk.bn1)) return false;
    if (!load_conv2d(ctx, base + "/conv2/weight", 3, 3, planes,    planes, 1,      1, 1, 1, blk.conv2)) return false;
    if (!load_bn    (ctx, base + "/bn2", blk.bn2)) return false;
    if (has_shortcut) {
        if (!load_conv2d(ctx, base + "/shortcut/0/weight", 1, 1, in_planes, planes, stride, 1, 0, 0, blk.shortcut_conv)) return false;
        if (!load_bn    (ctx, base + "/shortcut/1", blk.shortcut_bn)) return false;
    }
    blk.stride_h = stride;
    return true;
}

static bool load_cam_block(ggml_context * ctx,
                           const std::string & base,
                           int num_layers, int kernel_size, int dilation,
                           int init_C_in, int growth_rate, int bn_channels,
                           campplus_cam_block & blk)
{
    blk.num_layers  = num_layers;
    blk.kernel_size = kernel_size;
    blk.dilation    = dilation;
    blk.layers.clear();
    blk.layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const int C_in = init_C_in + i * growth_rate;
        const std::string p = base + "/tdnnd" + std::to_string(i + 1);
        auto & L = blk.layers[i];
        if (!load_bn(ctx, p + "/nonlinear1/batchnorm", L.bn1)) return false;
        if (!load_conv1d(ctx, p + "/linear1/weight", "",
                         /*k=*/1, /*C_in=*/C_in, /*C_out=*/bn_channels,
                         1, 0, 1, L.linear1)) return false;
        if (!load_bn(ctx, p + "/nonlinear2/batchnorm", L.bn2)) return false;
        const int pad = (kernel_size - 1) / 2 * dilation;
        if (!load_conv1d(ctx, p + "/cam_layer/linear_local/weight", "",
                         kernel_size, bn_channels, growth_rate,
                         1, pad, dilation, L.cam_linear_local)) return false;
        if (!load_conv1d(ctx, p + "/cam_layer/linear1/weight",
                         p + "/cam_layer/linear1/bias",
                         1, bn_channels, bn_channels / 2, 1, 0, 1, L.cam_linear1)) return false;
        if (!load_conv1d(ctx, p + "/cam_layer/linear2/weight",
                         p + "/cam_layer/linear2/bias",
                         1, bn_channels / 2, growth_rate, 1, 0, 1, L.cam_linear2)) return false;
    }
    return true;
}

bool campplus_load(const std::string & path, campplus_weights & w)
{
    ggml_context * tmp = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { fprintf(stderr, "campplus_load: cannot open %s\n", path.c_str()); return false; }
    if (gguf_find_key(g, "campplus.embedding_size") < 0) {
        gguf_free(g); if (tmp) ggml_free(tmp); return false;
    }
    auto u32 = [&](const char * k, uint32_t fb) {
        int64_t id = gguf_find_key(g, k);
        return id < 0 ? fb : gguf_get_val_u32(g, id);
    };

    w.feat_dim       = (int)u32("campplus.feat_dim",       80);
    w.embedding_size = (int)u32("campplus.embedding_size", 192);
    w.seg_pool_len   = (int)u32("campplus.seg_pool_len",   100);
    w.sample_rate    = (int)u32("campplus.sample_rate",    16000);
    const int init_channels = (int)u32("campplus.init_channels", 128);
    const int growth_rate   = (int)u32("campplus.growth_rate",   32);
    const int bn_size       = (int)u32("campplus.bn_size",       4);
    const int bn_channels   = bn_size * growth_rate;
    const int b1_layers     = (int)u32("campplus.block1_layers", 12);
    const int b2_layers     = (int)u32("campplus.block2_layers", 24);
    const int b3_layers     = (int)u32("campplus.block3_layers", 16);
    const int b1_dil        = (int)u32("campplus.block1_dilation", 1);
    const int b2_dil        = (int)u32("campplus.block2_dilation", 2);
    const int b3_dil        = (int)u32("campplus.block3_dilation", 2);
    const int k_size        = (int)u32("campplus.kernel_size",   3);

    // FCM head.
    bool ok = true;
    ok &= load_conv2d(tmp, "campplus/head/conv1/weight", 3, 3, 1, 32, 1, 1, 1, 1, w.head.conv1);
    ok &= load_bn    (tmp, "campplus/head/bn1", w.head.bn1);
    // layer1: 2 blocks, first stride=2 with shortcut; second stride=1 no shortcut (in==planes).
    w.head.layer1.resize(2);
    ok &= load_res_block(tmp, "campplus/head/layer1/0", /*has_shortcut=*/true,  32, 32, 2, w.head.layer1[0]);
    ok &= load_res_block(tmp, "campplus/head/layer1/1", /*has_shortcut=*/false, 32, 32, 1, w.head.layer1[1]);
    w.head.layer2.resize(2);
    ok &= load_res_block(tmp, "campplus/head/layer2/0", /*has_shortcut=*/true,  32, 32, 2, w.head.layer2[0]);
    ok &= load_res_block(tmp, "campplus/head/layer2/1", /*has_shortcut=*/false, 32, 32, 1, w.head.layer2[1]);
    ok &= load_conv2d(tmp, "campplus/head/conv2/weight", 3, 3, 32, 32, 2, 1, 1, 1, w.head.conv2);
    ok &= load_bn    (tmp, "campplus/head/bn2", w.head.bn2);

    // FCM output channels: 32 * (80 / 8) = 320, then tdnn.
    const int fcm_out_ch = 32 * (w.feat_dim / 8);
    ok &= load_conv1d(tmp, "campplus/xvector/tdnn/linear/weight", "",
                      /*k=*/5, fcm_out_ch, init_channels, /*s=*/2, /*p=*/2, /*d=*/1, w.tdnn_linear);
    ok &= load_bn(tmp, "campplus/xvector/tdnn/nonlinear/batchnorm", w.tdnn_bn);

    ok &= load_cam_block(tmp, "campplus/xvector/block1", b1_layers, k_size, b1_dil,
                         /*init_C_in=*/init_channels, growth_rate, bn_channels, w.block1);
    const int after_b1_ch = init_channels + b1_layers * growth_rate;  // 128 + 12*32 = 512
    ok &= load_bn(tmp, "campplus/xvector/transit1/nonlinear/batchnorm", w.transit1.bn);
    ok &= load_conv1d(tmp, "campplus/xvector/transit1/linear/weight", "",
                      1, after_b1_ch, after_b1_ch / 2, 1, 0, 1, w.transit1.linear);

    const int b2_in_ch = after_b1_ch / 2;  // 256
    ok &= load_cam_block(tmp, "campplus/xvector/block2", b2_layers, k_size, b2_dil,
                         b2_in_ch, growth_rate, bn_channels, w.block2);
    const int after_b2_ch = b2_in_ch + b2_layers * growth_rate;  // 256 + 24*32 = 1024
    ok &= load_bn(tmp, "campplus/xvector/transit2/nonlinear/batchnorm", w.transit2.bn);
    ok &= load_conv1d(tmp, "campplus/xvector/transit2/linear/weight", "",
                      1, after_b2_ch, after_b2_ch / 2, 1, 0, 1, w.transit2.linear);

    const int b3_in_ch = after_b2_ch / 2;  // 512
    ok &= load_cam_block(tmp, "campplus/xvector/block3", b3_layers, k_size, b3_dil,
                         b3_in_ch, growth_rate, bn_channels, w.block3);
    const int after_b3_ch = b3_in_ch + b3_layers * growth_rate;  // 512 + 16*32 = 1024
    ok &= load_bn(tmp, "campplus/xvector/transit3/nonlinear/batchnorm", w.transit3.bn);
    ok &= load_conv1d(tmp, "campplus/xvector/transit3/linear/weight", "",
                      1, after_b3_ch, after_b3_ch / 2, 1, 0, 1, w.transit3.linear);

    const int final_ch = after_b3_ch / 2;  // 512
    ok &= load_bn(tmp, "campplus/xvector/out_nonlinear/batchnorm", w.out_nonlinear_bn);

    ok &= load_conv1d(tmp, "campplus/xvector/dense/linear/weight", "",
                      1, /*C_in=*/final_ch * 2, /*C_out=*/w.embedding_size, 1, 0, 1, w.dense_linear);
    ok &= load_bn(tmp, "campplus/xvector/dense/nonlinear/batchnorm", w.dense_bn);

    gguf_free(g); if (tmp) ggml_free(tmp);
    return ok;
}

// =============================================================================
// Core ops
// =============================================================================
//
// Memory layout convention for this file:
//   1-D feature map x:   row-major (C, T)  -- channel-major, time innermost
//                        access:  x[c * T + t]
//   2-D feature map:     row-major (C, H, W)
//                        access:  x[c * H * W + h * W + w]
//
// This differs from ggml's [W, H, C, B] but simplifies the C++ loops since
// everything we touch is single-batch.

// out[c, t] = x[c, t] * scale[c] + shift[c]
static inline void bn_apply(float * x, const float * scale, const float * shift,
                            int C, int T)
{
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        const float s = scale[c], b = shift[c];
        float * row = x + (size_t)c * T;
        for (int t = 0; t < T; ++t) row[t] = row[t] * s + b;
    }
}

static inline void relu_inplace(float * x, size_t n) {
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)n; ++i) if (x[i] < 0.0f) x[i] = 0.0f;
}

static inline void sigmoid_inplace(float * x, size_t n) {
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)n; ++i) x[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

// Conv1d:  y[co, to] = bias[co] + sum_{ci, k} w[co, ci, k] * x[ci, to*stride + k*dilation - pad]
// PyTorch weight layout (numpy order): w is (C_out, C_in, k), stored row-major.
static void conv1d(const float * x, int C_in, int T_in,
                   const float * w, const float * bias,
                   int C_out, int k, int stride, int pad, int dilation,
                   float * y, int T_out)
{
    #pragma omp parallel for
    for (int co = 0; co < C_out; ++co) {
        const float bias_v = bias ? bias[co] : 0.0f;
        const float * w_co = w + (size_t)co * C_in * k;
        float * y_row = y + (size_t)co * T_out;
        for (int to = 0; to < T_out; ++to) {
            float acc = bias_v;
            const int base_t = to * stride - pad;
            for (int ci = 0; ci < C_in; ++ci) {
                const float * x_row = x + (size_t)ci * T_in;
                const float * w_row = w_co + (size_t)ci * k;
                for (int kk = 0; kk < k; ++kk) {
                    const int ti = base_t + kk * dilation;
                    if (ti >= 0 && ti < T_in) acc += w_row[kk] * x_row[ti];
                }
            }
            y_row[to] = acc;
        }
    }
}

// Conv2d: input (C_in, H, W), output (C_out, H_out, W_out).
// Weight: (C_out, C_in, kH, kW) stored row-major.
static void conv2d(const float * x, int C_in, int H, int W,
                   const float * w, const float * bias,
                   int C_out, int kH, int kW,
                   int sH, int sW, int pH, int pW,
                   float * y, int H_out, int W_out)
{
    #pragma omp parallel for
    for (int co = 0; co < C_out; ++co) {
        const float bias_v = bias ? bias[co] : 0.0f;
        const float * w_co = w + (size_t)co * C_in * kH * kW;
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                float acc = bias_v;
                const int base_h = ho * sH - pH;
                const int base_w = wo * sW - pW;
                for (int ci = 0; ci < C_in; ++ci) {
                    const float * x_c = x + (size_t)ci * H * W;
                    const float * w_c = w_co + (size_t)ci * kH * kW;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int hi = base_h + kh;
                        if (hi < 0 || hi >= H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int wi = base_w + kw;
                            if (wi < 0 || wi >= W) continue;
                            acc += w_c[kh * kW + kw] * x_c[hi * W + wi];
                        }
                    }
                }
                y[(size_t)co * H_out * W_out + ho * W_out + wo] = acc;
            }
        }
    }
}

static inline int conv_out_len(int L_in, int k, int stride, int pad, int dilation) {
    return (L_in + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
}

// =============================================================================
// High-level module forwards
// =============================================================================

// ---- FCM ----

static void fcm_basic_resblock(const campplus_res_block & blk,
                               const std::vector<float> & x_in,
                               int C_in, int H, int W,
                               std::vector<float> & x_out, int & H_out, int & W_out)
{
    const int planes = blk.conv1.C_out;
    const int sH = blk.stride_h;
    const int sW = 1;  // FCM always uses stride_w=1 (only H downsamples)
    H_out = conv_out_len(H, 3, sH, 1, 1);
    W_out = conv_out_len(W, 3, sW, 1, 1);

    std::vector<float> t1((size_t)planes * H_out * W_out);
    conv2d(x_in.data(), C_in, H, W,
           blk.conv1.w.data(), nullptr,
           planes, 3, 3, sH, sW, 1, 1,
           t1.data(), H_out, W_out);
    bn_apply(t1.data(), blk.bn1.scale.data(), blk.bn1.shift.data(), planes, H_out * W_out);
    relu_inplace(t1.data(), t1.size());

    std::vector<float> t2((size_t)planes * H_out * W_out);
    conv2d(t1.data(), planes, H_out, W_out,
           blk.conv2.w.data(), nullptr,
           planes, 3, 3, 1, 1, 1, 1,
           t2.data(), H_out, W_out);
    bn_apply(t2.data(), blk.bn2.scale.data(), blk.bn2.shift.data(), planes, H_out * W_out);

    // Shortcut.
    std::vector<float> sc;
    if (!blk.shortcut_conv.w.empty()) {
        sc.resize((size_t)planes * H_out * W_out);
        conv2d(x_in.data(), C_in, H, W,
               blk.shortcut_conv.w.data(), nullptr,
               planes, 1, 1, sH, sW, 0, 0,
               sc.data(), H_out, W_out);
        bn_apply(sc.data(), blk.shortcut_bn.scale.data(), blk.shortcut_bn.shift.data(),
                 planes, H_out * W_out);
    }

    x_out.resize((size_t)planes * H_out * W_out);
    if (sc.empty()) {
        for (size_t i = 0; i < x_out.size(); ++i) x_out[i] = t2[i] + x_in[i];
    } else {
        for (size_t i = 0; i < x_out.size(); ++i) x_out[i] = t2[i] + sc[i];
    }
    relu_inplace(x_out.data(), x_out.size());
}

// FCM takes (80, T) fbank → outputs (320, T) after 3x H-downsample and reshape.
static void fcm_forward(const campplus_fcm & fcm,
                        const float * fbank_80_T, int T,
                        std::vector<float> & out, int & T_out)
{
    const int F = 80;
    // conv1 input is (C=1, H=80, W=T), i.e. we add a channel dim of 1.
    // Since C_in=1, the "channel-major" layout is just fbank_80_T.
    int H = F, W = T;
    int H2, W2;
    // conv1: (1 → 32, k=3, s=1, p=1) → H=H, W=T
    H2 = conv_out_len(H, 3, 1, 1, 1);
    W2 = conv_out_len(W, 3, 1, 1, 1);
    std::vector<float> x((size_t)32 * H2 * W2);
    conv2d(fbank_80_T, 1, H, W, fcm.conv1.w.data(), nullptr,
           32, 3, 3, 1, 1, 1, 1, x.data(), H2, W2);
    bn_apply(x.data(), fcm.bn1.scale.data(), fcm.bn1.shift.data(), 32, H2 * W2);
    relu_inplace(x.data(), x.size());
    H = H2; W = W2;

    auto run_layer = [&](const std::vector<campplus_res_block> & blocks) {
        for (size_t i = 0; i < blocks.size(); ++i) {
            std::vector<float> y;
            int Hn, Wn;
            fcm_basic_resblock(blocks[i], x, 32, H, W, y, Hn, Wn);
            x = std::move(y);
            H = Hn; W = Wn;
        }
    };
    run_layer(fcm.layer1);  // H: 80 → 40
    run_layer(fcm.layer2);  // H: 40 → 20

    // conv2: (32 → 32, k=3, s=(2,1), p=1) → H: 20 → 10
    H2 = conv_out_len(H, 3, 2, 1, 1);
    W2 = conv_out_len(W, 3, 1, 1, 1);
    std::vector<float> y((size_t)32 * H2 * W2);
    conv2d(x.data(), 32, H, W, fcm.conv2.w.data(), nullptr,
           32, 3, 3, 2, 1, 1, 1, y.data(), H2, W2);
    bn_apply(y.data(), fcm.bn2.scale.data(), fcm.bn2.shift.data(), 32, H2 * W2);
    relu_inplace(y.data(), y.size());
    H = H2; W = W2;

    // Reshape (32, 10, T) → (320, T) — channel-major layout means we just
    // re-interpret 32 × 10 × T as 320 × T with data in place.
    out = std::move(y);
    T_out = W;
}

// ---- CAMDenseTDNNLayer ----
//
// Python forward:
//     x_in   : (C_in, T)
//     BN1 + ReLU → Conv1x1 (C_in → 128) → BN2 + ReLU → CAMLayer
//     CAMLayer(y):
//        y = linear_local(y)                              # (C_in → growth, k, dil)
//        context = y.mean(-1, keepdim=True) + seg_pool(y) # over input y (post BN2+ReLU)
//        context = relu(linear1(context))
//        gate    = sigmoid(linear2(context))
//        return  y * gate
//     output = concat(x_in, CAMLayer output) along channel axis

// seg_pool: average-pool with kernel=seg_len, stride=seg_len, ceil_mode=True,
// then repeat-interleave each segment value `seg_len` times and truncate to T.
static void seg_pool_expand(const float * x, int C, int T, int seg_len,
                            float * out)  // out has shape (C, T)
{
    const int S = (T + seg_len - 1) / seg_len;  // ceil(T/seg_len)
    std::vector<float> pooled((size_t)C * S);
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        const float * row = x + (size_t)c * T;
        for (int s = 0; s < S; ++s) {
            int t0 = s * seg_len;
            int t1 = std::min(T, t0 + seg_len);
            int n  = t1 - t0;  // avg_pool1d with ceil_mode uses PyTorch semantics:
                               // denominator = kernel_size regardless of truncation
                               // (count_include_pad=True by default).  BUT for the
                               // LAST ceil-mode bin, PyTorch divides by the actual
                               // count of valid elements.  For voice-encoder lengths
                               // (seg_len=100, T~500), this only matters when
                               // T % 100 != 0, so use the true-count form.
            float acc = 0.0f;
            for (int t = t0; t < t1; ++t) acc += row[t];
            pooled[(size_t)c * S + s] = acc / std::max(n, 1);
        }
    }
    // Expand: each pooled[c, s] is tiled across seg_len consecutive t's,
    // then truncated at T.
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        float * dst = out + (size_t)c * T;
        for (int s = 0; s < S; ++s) {
            const float v = pooled[(size_t)c * S + s];
            const int t0 = s * seg_len;
            const int t1 = std::min(T, t0 + seg_len);
            for (int t = t0; t < t1; ++t) dst[t] = v;
        }
    }
}

static void cam_layer_forward(const campplus_cam_dense_tdnn_layer & L,
                              const float * x_in, int C_bn, int T,
                              int growth, int kernel_size, int dilation,
                              int seg_pool_len,
                              float * out /* (growth, T) */)
{
    // linear_local: (C_bn → growth, k, dilation)
    const int pad = (kernel_size - 1) / 2 * dilation;
    std::vector<float> y_local((size_t)growth * T);
    conv1d(x_in, C_bn, T,
           L.cam_linear_local.w.data(), nullptr,
           growth, kernel_size, 1, pad, dilation,
           y_local.data(), T);

    // Global mean over T (per channel).
    std::vector<float> mean_ctx(C_bn);
    #pragma omp parallel for
    for (int c = 0; c < C_bn; ++c) {
        const float * row = x_in + (size_t)c * T;
        double acc = 0.0;
        for (int t = 0; t < T; ++t) acc += row[t];
        mean_ctx[c] = (float)(acc / T);
    }

    // Segment pooling, expanded to (C_bn, T).
    std::vector<float> seg_ctx((size_t)C_bn * T);
    seg_pool_expand(x_in, C_bn, T, seg_pool_len, seg_ctx.data());

    // context[c, t] = mean_ctx[c] + seg_ctx[c, t].
    for (int c = 0; c < C_bn; ++c) {
        float * row = seg_ctx.data() + (size_t)c * T;
        const float m = mean_ctx[c];
        for (int t = 0; t < T; ++t) row[t] += m;
    }

    // linear1: (C_bn → C_bn/2) 1x1 + bias, then ReLU.
    const int mid = L.cam_linear1.C_out;
    std::vector<float> h1((size_t)mid * T);
    conv1d(seg_ctx.data(), C_bn, T,
           L.cam_linear1.w.data(), L.cam_linear1.b.empty() ? nullptr : L.cam_linear1.b.data(),
           mid, 1, 1, 0, 1, h1.data(), T);
    relu_inplace(h1.data(), h1.size());

    // linear2: (C_bn/2 → growth) 1x1 + bias, then sigmoid.
    std::vector<float> gate((size_t)growth * T);
    conv1d(h1.data(), mid, T,
           L.cam_linear2.w.data(), L.cam_linear2.b.empty() ? nullptr : L.cam_linear2.b.data(),
           growth, 1, 1, 0, 1, gate.data(), T);
    sigmoid_inplace(gate.data(), gate.size());

    // out = y_local * gate.
    for (size_t i = 0; i < y_local.size(); ++i) out[i] = y_local[i] * gate[i];
}

// Runs one CAMDenseTDNNLayer on x (C_in, T), appends the growth-d output to x,
// and resizes it to (C_in + growth, T) in-place.  Returns new C_in.
static int cam_dense_tdnn_layer_forward(const campplus_cam_dense_tdnn_layer & L,
                                        std::vector<float> & x, int C_in, int T,
                                        int growth, int bn_channels,
                                        int kernel_size, int dilation,
                                        int seg_pool_len)
{
    // nonlinear1 = BN + ReLU on (C_in, T).
    std::vector<float> y = x;
    bn_apply(y.data(), L.bn1.scale.data(), L.bn1.shift.data(), C_in, T);
    relu_inplace(y.data(), y.size());

    // linear1: Conv1x1 C_in → bn_channels.
    std::vector<float> z((size_t)bn_channels * T);
    conv1d(y.data(), C_in, T,
           L.linear1.w.data(), nullptr,
           bn_channels, 1, 1, 0, 1, z.data(), T);

    // nonlinear2 = BN + ReLU on (bn_channels, T).
    bn_apply(z.data(), L.bn2.scale.data(), L.bn2.shift.data(), bn_channels, T);
    relu_inplace(z.data(), z.size());

    // cam_layer → (growth, T).
    std::vector<float> cam_out((size_t)growth * T);
    cam_layer_forward(L, z.data(), bn_channels, T, growth, kernel_size, dilation,
                      seg_pool_len, cam_out.data());

    // Concat along channel axis: x_new = [x; cam_out].
    const int C_new = C_in + growth;
    x.resize((size_t)C_new * T);
    std::memcpy(x.data() + (size_t)C_in * T, cam_out.data(), cam_out.size() * sizeof(float));
    return C_new;
}

// Stats pooling: (C, T) → (2C,). Concatenates mean + unbiased std along channel.
static void stats_pool(const float * x, int C, int T, float * out)
{
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        const float * row = x + (size_t)c * T;
        double sum = 0.0, sq = 0.0;
        for (int t = 0; t < T; ++t) sum += row[t];
        double mean = sum / T;
        for (int t = 0; t < T; ++t) {
            double d = row[t] - mean;
            sq += d * d;
        }
        double var = sq / std::max(1, T - 1);   // unbiased=True
        out[c]     = (float)mean;
        out[C + c] = (float)std::sqrt(var);
    }
}

// =============================================================================
// Top-level forward
// =============================================================================

// Legacy scalar CPU forward.  Kept for the CPU test harness and as a fallback
// when the caller explicitly passes backend == nullptr.  The ggml-backed
// public entry point follows it at the bottom of the file.
static bool campplus_embed_cpu(const std::vector<float> & fbank_t_by_c, int T,
                               const campplus_weights & w, std::vector<float> & out)
{
    if ((int64_t)fbank_t_by_c.size() != (int64_t)T * w.feat_dim) {
        fprintf(stderr, "campplus_embed: fbank has %zu elts, expected %d*%d\n",
                fbank_t_by_c.size(), T, w.feat_dim);
        return false;
    }

    // 1. Transpose (T, 80) → (80, T) (channel-major).
    std::vector<float> fbank_ct((size_t)w.feat_dim * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < w.feat_dim; ++c)
            fbank_ct[(size_t)c * T + t] = fbank_t_by_c[(size_t)t * w.feat_dim + c];

    // 2. FCM → (320, T).
    std::vector<float> x;
    int T_after_fcm = 0;
    fcm_forward(w.head, fbank_ct.data(), T, x, T_after_fcm);

    const int fcm_out_ch = 32 * (w.feat_dim / 8);  // 320

    // 3. tdnn: Conv1d(320, 128, k=5, s=2, p=2) + BN + ReLU.
    const int init_C = w.tdnn_linear.C_out;  // 128
    const int T_tdnn = conv_out_len(T_after_fcm, 5, 2, 2, 1);
    std::vector<float> y((size_t)init_C * T_tdnn);
    conv1d(x.data(), fcm_out_ch, T_after_fcm,
           w.tdnn_linear.w.data(), nullptr,
           init_C, 5, 2, 2, 1, y.data(), T_tdnn);
    bn_apply(y.data(), w.tdnn_bn.scale.data(), w.tdnn_bn.shift.data(), init_C, T_tdnn);
    relu_inplace(y.data(), y.size());
    x = std::move(y);
    int C_cur = init_C;
    int T_cur = T_tdnn;

    // 4. Helper that runs a block + transit.
    auto run_block = [&](const campplus_cam_block & blk,
                         const campplus_transit & trans,
                         int growth, int bn_channels, int seg_pool_len) {
        for (int i = 0; i < blk.num_layers; ++i) {
            C_cur = cam_dense_tdnn_layer_forward(blk.layers[i], x, C_cur, T_cur,
                                                 growth, bn_channels,
                                                 blk.kernel_size, blk.dilation,
                                                 seg_pool_len);
        }
        // transit = BN + ReLU + Conv1x1 (halves channels).
        bn_apply(x.data(), trans.bn.scale.data(), trans.bn.shift.data(), C_cur, T_cur);
        relu_inplace(x.data(), x.size());
        const int C_out = trans.linear.C_out;
        std::vector<float> y((size_t)C_out * T_cur);
        conv1d(x.data(), C_cur, T_cur,
               trans.linear.w.data(), nullptr,
               C_out, 1, 1, 0, 1, y.data(), T_cur);
        x = std::move(y);
        C_cur = C_out;
    };

    const int growth = 32;
    const int bn_channels = 128;
    run_block(w.block1, w.transit1, growth, bn_channels, w.seg_pool_len);
    run_block(w.block2, w.transit2, growth, bn_channels, w.seg_pool_len);
    run_block(w.block3, w.transit3, growth, bn_channels, w.seg_pool_len);

    // 5. out_nonlinear: BN + ReLU on (C_cur=512, T).
    bn_apply(x.data(), w.out_nonlinear_bn.scale.data(), w.out_nonlinear_bn.shift.data(),
             C_cur, T_cur);
    relu_inplace(x.data(), x.size());

    // 6. stats_pool → (2*C_cur = 1024,).
    std::vector<float> stats(2 * (size_t)C_cur);
    stats_pool(x.data(), C_cur, T_cur, stats.data());

    // 7. dense: Conv1x1 (1024 → 192) + BN(affine=False).
    //    Input shape is (1024, 1) — single-frame; we can just do a matmul.
    const int E = w.embedding_size;
    std::vector<float> emb((size_t)E, 0.0f);
    conv1d(stats.data(), 2 * C_cur, 1,
           w.dense_linear.w.data(), nullptr,
           E, 1, 1, 0, 1, emb.data(), 1);
    bn_apply(emb.data(), w.dense_bn.scale.data(), w.dense_bn.shift.data(), E, 1);

    out = std::move(emb);
    return true;
}

// =============================================================================
// Public API: route through ggml graph when a backend is supplied, fall back
// to the scalar CPU path otherwise (test harnesses, legacy callers).
// =============================================================================

#include "campplus_forward.inc"

bool campplus_embed(const std::vector<float> & fbank_t_by_c, int T,
                    const campplus_weights & w,
                    ggml_backend_t backend,
                    std::vector<float> & out)
{
    if (backend) return campplus_embed_ggml(fbank_t_by_c, T, w, backend, out);
    return campplus_embed_cpu(fbank_t_by_c, T, w, out);
}
