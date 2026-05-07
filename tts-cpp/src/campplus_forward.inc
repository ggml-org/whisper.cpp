// =============================================================================
// CAMPPlus forward via a GGML graph (Metal / Vulkan / CUDA / ggml-cpu).
// =============================================================================
//
// Replaces the scalar C++ forward below.  The scalar version is kept in a
// separate _cpu variant for correctness validation and the test harnesses.
//
// Tensor layout inside the graph:
//   - 2-D feature maps use ggml's (W, H, C, B) convention
//     (ne[0]=W, ne[1]=H, ne[2]=C, ne[3]=B=1).
//   - 1-D feature maps use (T, C)   (ne[0]=T, ne[1]=C).
//   - BN is pre-fused in the loader to per-channel (scale, shift); we store
//     each as a row vector [1, C] so ggml's broadcast against (T, C) works
//     without an explicit reshape.
//
// Data flow:
//   fbank    (T, 80)                       host upload
//   FCM head (W=T, H=80, C=1) → (T, 10, 32)  Conv2d + BN + ReLU + res blocks
//   flatten  (T, 320)                       (32 ch × 10 frames per col)
//   tdnn     (T/2, 128)                      Conv1d k=5 s=2 p=2 + BN + ReLU
//   block1   (T/2, 512)                       12 CAMDenseTDNN layers, dil=1
//   transit1 (T/2, 256)                       BN + ReLU + Conv1x1
//   block2   (T/2, 1024)                      24 CAMDenseTDNN layers, dil=2
//   transit2 (T/2, 512)
//   block3   (T/2, 1024)                      16 CAMDenseTDNN layers, dil=2
//   transit3 (T/2, 512)                       BN + ReLU + Conv1x1
//   out_bn   (T/2, 512)                       BN + ReLU
//   stats    (1, 1024)                        concat(mean_T(x), std_T(x))
//   dense    (1, 192)                         Conv1x1 + BN (affine-less)
//   → flatten to (192,) output
//
// =============================================================================

struct ve_weights_t {
    ggml_tensor * w = nullptr;   // weight (conv1d / conv2d / linear)
    ggml_tensor * b = nullptr;   // optional bias (nullptr if absent)
};

struct ve_bn_t {
    ggml_tensor * scale = nullptr;
    ggml_tensor * shift = nullptr;
};

struct ve_resblock_t {
    ve_weights_t conv1;  ve_bn_t bn1;
    ve_weights_t conv2;  ve_bn_t bn2;
    ve_weights_t sc;     ve_bn_t sc_bn;   // optional shortcut (conv1x1 + BN)
    int          stride_h = 1;
    bool         has_shortcut = false;
};

struct ve_cam_layer_t {
    ve_bn_t      bn1;
    ve_weights_t lin1;
    ve_bn_t      bn2;
    ve_weights_t loc;     // linear_local (no bias)
    ve_weights_t lin1_cam; // cam/linear1 (has bias)
    ve_weights_t lin2_cam; // cam/linear2 (has bias)
};

struct ve_cam_block_t {
    int num_layers = 0;
    int kernel_size = 3;
    int dilation = 1;
    int growth = 32;
    int bn_channels = 128;
    int C_in = 0;              // channels entering layer 0
    std::vector<ve_cam_layer_t> layers;
};

struct ve_transit_t {
    ve_bn_t      bn;
    ve_weights_t lin;
};

struct campplus_gctx {
    ggml_backend_t        backend      = nullptr;
    bool                  owns_backend = false;
    ggml_context        * weights_ctx  = nullptr;
    ggml_backend_buffer_t weights_buf  = nullptr;
    ggml_gallocr_t        allocr       = nullptr;

    // Geometry snapshot (cached for the graph builder).
    int feat_dim      = 80;
    int init_channels = 128;
    int growth_rate   = 32;
    int bn_channels   = 128;
    int fcm_out_ch    = 320;
    int seg_pool_len  = 100;

    // FCM
    ve_weights_t fcm_conv1;  ve_bn_t fcm_bn1;
    ve_resblock_t fcm_layer1[2];
    ve_resblock_t fcm_layer2[2];
    ve_weights_t fcm_conv2;  ve_bn_t fcm_bn2;

    // TDNN
    ve_weights_t tdnn;       ve_bn_t tdnn_bn;

    // 3 CAM blocks + 3 transits
    ve_cam_block_t block1, block2, block3;
    ve_transit_t   transit1, transit2, transit3;

    // Out head
    ve_bn_t      out_bn;
    ve_weights_t dense;
    ve_bn_t      dense_bn;
};

static void campplus_gctx_free(campplus_gctx & g) {
    if (g.allocr)      { ggml_gallocr_free(g.allocr);              g.allocr = nullptr; }
    if (g.weights_buf) { ggml_backend_buffer_free(g.weights_buf);  g.weights_buf = nullptr; }
    if (g.weights_ctx) { ggml_free(g.weights_ctx);                 g.weights_ctx = nullptr; }
    if (g.backend && g.owns_backend) ggml_backend_free(g.backend);
    g.backend = nullptr;
}

// ---------- weight allocators ----------

static ggml_tensor * new_f32(ggml_context * ctx, int ne0, int ne1 = 0, int ne2 = 0, int ne3 = 0) {
    if (ne3 > 0) return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    if (ne2 > 0) return ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    if (ne1 > 0) return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
}

// Allocate a BN pair (shape [1, C]) so broadcasting against (T, C) works directly.
static void decl_bn(ggml_context * ctx, ve_bn_t & b, int C, const std::string & name) {
    b.scale = new_f32(ctx, 1, C);  ggml_set_name(b.scale, (name + "/s").c_str());
    b.shift = new_f32(ctx, 1, C);  ggml_set_name(b.shift, (name + "/b").c_str());
}

// Allocate a Conv1d weight.  ggml convention: [k, C_in, C_out].
static void decl_conv1d(ggml_context * ctx, ve_weights_t & w, int k, int C_in, int C_out,
                        const std::string & name, bool has_bias = false)
{
    w.w = new_f32(ctx, k, C_in, C_out);  ggml_set_name(w.w, (name + "/w").c_str());
    if (has_bias) {
        // Bias broadcast layout: [1, C_out] so (T, C_out) + bias works.
        w.b = new_f32(ctx, 1, C_out);  ggml_set_name(w.b, (name + "/b").c_str());
    }
}

// Allocate a Conv2d weight.  ggml convention: [kW, kH, C_in, C_out].
static void decl_conv2d(ggml_context * ctx, ve_weights_t & w, int kW, int kH, int C_in, int C_out,
                        const std::string & name)
{
    w.w = new_f32(ctx, kW, kH, C_in, C_out);
    ggml_set_name(w.w, (name + "/w").c_str());
}

// ---------- host → backend upload ----------

static void upload(ggml_tensor * t, const void * src, size_t bytes) {
    ggml_backend_tensor_set(t, src, 0, bytes);
}

static void upload_bn(const ve_bn_t & bn, const campplus_bn & src) {
    upload(bn.scale, src.scale.data(), src.scale.size() * sizeof(float));
    upload(bn.shift, src.shift.data(), src.shift.size() * sizeof(float));
}

// Both Conv1d and Conv2d weights in the CAMPPlus GGUF match PyTorch's
// row-major (C_out, C_in, kH, kW) or (C_out, C_in, k) layout.  GGML wants
// [k, C_in, C_out] for Conv1d and [kW, kH, C_in, C_out] for Conv2d.  For
// Conv1d (k, C_in, C_out) and PyTorch (C_out, C_in, k) — the flat memory is
// *exactly* the transpose we need to flip axes 0 and 2; with contiguous
// storage and 3D dims, the PyTorch layout (C_out, C_in, k) flattened is
// c_out · (C_in*k) + c_in · k + i, while ggml wants (k, C_in, C_out) flattened
// as c_out · (C_in*k) + c_in · k + i.  Same index order — no physical
// transpose needed.  Same story for 2D: PyTorch (C_out, C_in, kH, kW) ==
// ggml (kW, kH, C_in, C_out) as flat memory.  Confirmed against the existing
// s3tokenizer conv1d_f32 helper, which feeds identical PyTorch-loaded weights
// directly into ggml.
static bool init_weight(const ve_weights_t & w, const campplus_conv & src,
                        bool expect_bias = false)
{
    const size_t want = src.w.size() * sizeof(float);
    if (ggml_nbytes(w.w) != want) {
        fprintf(stderr, "campplus_ggml: weight size mismatch for %s: have %zu, want %zu\n",
                ggml_get_name(w.w), ggml_nbytes(w.w), want);
        return false;
    }
    upload(w.w, src.w.data(), want);
    if (expect_bias) {
        if (!w.b || src.b.empty()) {
            fprintf(stderr, "campplus_ggml: missing bias for %s\n", ggml_get_name(w.w));
            return false;
        }
        upload(w.b, src.b.data(), src.b.size() * sizeof(float));
    }
    return true;
}

// ---------- graph-building primitives ----------

// Apply BN (scale/shift stored as [1, C]) to (T, C).  Broadcast is native
// with ggml_mul / ggml_add.  Also handles (W, H, C, 1) layout if the caller
// reshapes scale/shift to [1, 1, C, 1] first — see bn_apply_2d.
static ggml_tensor * bn_apply_1d(ggml_context * ctx, ggml_tensor * x, const ve_bn_t & bn) {
    x = ggml_mul(ctx, x, bn.scale);
    x = ggml_add(ctx, x, bn.shift);
    return x;
}

// 2D variant: scale/shift stored as [1, C] in weights context need to be
// reshaped to [1, 1, C, 1] to broadcast across (W, H, C, B).
static ggml_tensor * bn_apply_2d(ggml_context * ctx, ggml_tensor * x, const ve_bn_t & bn) {
    const int C = (int) bn.scale->ne[1];
    ggml_tensor * s = ggml_reshape_4d(ctx, bn.scale, 1, 1, C, 1);
    ggml_tensor * b = ggml_reshape_4d(ctx, bn.shift, 1, 1, C, 1);
    x = ggml_mul(ctx, x, s);
    x = ggml_add(ctx, x, b);
    return x;
}

// F32 Conv1d via im2col + mul_mat.  Copied from s3tokenizer's helper to avoid
// the F16-kernel assertion in ggml_conv_1d.
static ggml_tensor * conv1d_f32(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation)
{
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input,
                                       stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, r, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

// Same, for 2D.  Returns (W_out, H_out, C_out, B).
static ggml_tensor * conv2d_f32(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int sW, int sH, int pW, int pH)
{
    // ggml_conv_2d does im2col + mat-mul internally and handles F32 kernels.
    return ggml_conv_2d(ctx, kernel, input, sW, sH, pW, pH, 1, 1);
}

static ggml_tensor * fcm_res_block(ggml_context * ctx, ggml_tensor * x,
                                   const ve_resblock_t & blk)
{
    ggml_tensor * t1 = conv2d_f32(ctx, blk.conv1.w, x, 1, blk.stride_h, 1, 1);
    t1 = bn_apply_2d(ctx, t1, blk.bn1);
    t1 = ggml_relu(ctx, t1);

    ggml_tensor * t2 = conv2d_f32(ctx, blk.conv2.w, t1, 1, 1, 1, 1);
    t2 = bn_apply_2d(ctx, t2, blk.bn2);

    ggml_tensor * sc = x;
    if (blk.has_shortcut) {
        sc = conv2d_f32(ctx, blk.sc.w, x, 1, blk.stride_h, 0, 0);
        sc = bn_apply_2d(ctx, sc, blk.sc_bn);
    }
    ggml_tensor * y = ggml_add(ctx, t2, sc);
    y = ggml_relu(ctx, y);
    return y;
}

// CAMDenseTDNN layer.  Takes x (T, C_in) and returns concat[x, cam_out] along
// the channel axis → (T, C_in + growth).
//
// `seg_scale` is a precomputed (1, S, 1) tensor of per-bin 1/count factors
// for the seg-pool step.  Rather than using ggml_pool_1d (which lacks a
// Vulkan shader and gives wrong results via the scheduler's CPU fallback),
// we do pooling via reshape + sum_rows + broadcast multiply, which stays in
// pure elementwise / reduction ops supported on every backend.
static ggml_tensor * cam_dense_tdnn_layer(ggml_context * ctx, ggml_tensor * x,
                                          const ve_cam_layer_t & L,
                                          int kernel_size, int dilation, int seg_pool_len,
                                          int growth, int bn_channels,
                                          ggml_tensor * seg_scale,
                                          int T_pad, int S, int T)
{
    // nonlinear1 = BN + ReLU on (T, C_in)
    ggml_tensor * y = bn_apply_1d(ctx, x, L.bn1);
    y = ggml_relu(ctx, y);

    // linear1: 1x1 conv (C_in → bn_channels)
    ggml_tensor * z = conv1d_f32(ctx, L.lin1.w, y, 1, 0, 1);  // (T, bn_channels)

    // nonlinear2 = BN + ReLU on (T, bn_channels)
    z = bn_apply_1d(ctx, z, L.bn2);
    z = ggml_relu(ctx, z);

    // CAMLayer:
    //   y_local = linear_local(z)                      (T, growth)
    //   mean_ctx = mean_T(z)                           (1, bn_channels)
    //   seg_ctx  = seg_pool_expand(z, seg_pool_len)    (T, bn_channels)
    //   context  = mean_ctx + seg_ctx                  (T, bn_channels) [broadcast]
    //   context  = relu(linear1(context))              (T, bn_channels/2)
    //   gate     = sigmoid(linear2(context))           (T, growth)
    //   cam_out  = y_local * gate                      (T, growth)

    const int pad = (kernel_size - 1) / 2 * dilation;
    ggml_tensor * y_local = conv1d_f32(ctx, L.loc.w, z, 1, pad, dilation);  // (T, growth)

    // mean_ctx (1, bn_channels) — ggml_mean reduces ne[0]=T
    ggml_tensor * mean_ctx = ggml_mean(ctx, z);

    // seg_pool_expand, Vulkan-safe reformulation:
    //
    //   1. Pad z along the time axis so T becomes T_pad = S * seg_len
    //      (S = ceil(T / seg_len)).  Requires zero-pad.
    //   2. Reshape (T_pad, C) → (seg_len, S, C).
    //   3. Sum across ne[0]=seg_len via ggml_sum_rows  → (1, S, C).
    //   4. Divide by per-bin count via a (1, S, 1) scale tensor
    //      precomputed by the host: [1/seg_len, ..., 1/last_bin_size].
    //   5. Broadcast back to (seg_len, S, C) via ggml_repeat.
    //   6. Reshape (seg_len, S, C) → (T_pad, C), view first T entries.
    //
    // Uses only reshape / pad / sum_rows / mul / repeat, all of which have
    // native Vulkan kernels.  Avoids ggml_pool_1d (no Vulkan shader; the
    // scheduler's CPU-fallback copy mangled the output, cos_sim ~ 0.13).
    const int64_t C_bn = z->ne[1];
    ggml_tensor * z_pad = z;
    if (T_pad > T) {
        z_pad = ggml_pad(ctx, z, T_pad - T, 0, 0, 0);   // [T_pad, C, 1]
    }
    ggml_tensor * z_r     = ggml_reshape_3d(ctx, z_pad, seg_pool_len, S, C_bn);
    ggml_tensor * seg_sum = ggml_sum_rows (ctx, z_r);                         // [1, S, C]
    ggml_tensor * seg_mean = ggml_mul(ctx, seg_sum, seg_scale);                // [1, S, C] broadcast
    ggml_tensor * seg_bc  = ggml_repeat (ctx, seg_mean, z_r);                  // [seg_len, S, C]
    ggml_tensor * seg_flat = ggml_reshape_2d(ctx, seg_bc, T_pad, C_bn);
    ggml_tensor * seg_ctx;
    if (T_pad > T) {
        seg_ctx = ggml_view_2d(ctx, seg_flat, T, C_bn, seg_flat->nb[1], 0);
    } else {
        seg_ctx = seg_flat;
    }

    // context = mean_ctx (1, bn_channels) + seg_ctx (T, bn_channels)
    ggml_tensor * context = ggml_add(ctx, seg_ctx, mean_ctx);

    // linear1: conv1x1 bn_channels → bn_channels/2, with bias
    ggml_tensor * h1 = conv1d_f32(ctx, L.lin1_cam.w, context, 1, 0, 1);
    h1 = ggml_add(ctx, h1, L.lin1_cam.b);
    h1 = ggml_relu(ctx, h1);

    // linear2: conv1x1 bn_channels/2 → growth, with bias
    ggml_tensor * gate = conv1d_f32(ctx, L.lin2_cam.w, h1, 1, 0, 1);
    gate = ggml_add(ctx, gate, L.lin2_cam.b);
    gate = ggml_sigmoid(ctx, gate);

    ggml_tensor * cam_out = ggml_mul(ctx, y_local, gate);          // (T, growth)

    // Dense concat: [x, cam_out] along the channel axis (ne[1]).
    return ggml_concat(ctx, x, cam_out, /*dim=*/1);
}

static ggml_tensor * build_cam_block(ggml_context * ctx, ggml_tensor * x,
                                     const ve_cam_block_t & blk, int seg_pool_len,
                                     ggml_tensor * seg_scale,
                                     int T_pad, int S, int T)
{
    for (int i = 0; i < blk.num_layers; ++i) {
        x = cam_dense_tdnn_layer(ctx, x, blk.layers[i],
                                  blk.kernel_size, blk.dilation, seg_pool_len,
                                  blk.growth, blk.bn_channels,
                                  seg_scale, T_pad, S, T);
    }
    return x;
}

static ggml_tensor * build_transit(ggml_context * ctx, ggml_tensor * x,
                                   const ve_transit_t & t)
{
    x = bn_apply_1d(ctx, x, t.bn);
    x = ggml_relu(ctx, x);
    x = conv1d_f32(ctx, t.lin.w, x, 1, 0, 1);
    return x;
}

// stats_pool: (T, C) → (1, 2C) = concat(mean, std) along the channel axis.
// PyTorch uses unbiased std (divisor T-1).  For typical T≥100 the difference
// from biased std is ≲1% which is far below model-level tolerance, and we
// avoid a fresh custom op by using ggml_mean as the reducer for both passes.
static ggml_tensor * build_stats_pool(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * mean = ggml_mean(ctx, x);                        // (1, C)
    ggml_tensor * diff = ggml_sub(ctx, x, mean);                   // (T, C) broadcast
    ggml_tensor * sq   = ggml_mul(ctx, diff, diff);
    ggml_tensor * var  = ggml_mean(ctx, sq);                       // (1, C) biased var
    ggml_tensor * std_ = ggml_sqrt(ctx, var);
    ggml_tensor * cat  = ggml_concat(ctx, mean, std_, /*dim=*/1);  // (1, 2C)
    return cat;
}

// ---------- graph builder ----------

static ggml_cgraph * build_campplus_graph(const campplus_gctx & g, int T) {
    const int max_nodes = 4096;
    const size_t buf_size = ggml_tensor_overhead() * max_nodes +
                            ggml_graph_overhead_custom(max_nodes, false);
    static std::vector<uint8_t> buf;
    buf.resize(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf   = ggml_new_graph_custom(ctx, max_nodes, false);

    // Input: fbank (W=T, H=80, C=1, B=1).  Uploaded from (T, 80) row-major.
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, g.feat_dim, 1, 1);
    ggml_set_name(x, "fbank"); ggml_set_input(x);

    // FCM conv1: (1 → 32, k=3x3, s=1, p=1)
    ggml_tensor * y = conv2d_f32(ctx, g.fcm_conv1.w, x, 1, 1, 1, 1);
    y = bn_apply_2d(ctx, y, g.fcm_bn1);
    y = ggml_relu(ctx, y);

    for (int i = 0; i < 2; ++i) y = fcm_res_block(ctx, y, g.fcm_layer1[i]);
    for (int i = 0; i < 2; ++i) y = fcm_res_block(ctx, y, g.fcm_layer2[i]);

    // FCM conv2: (32 → 32, k=3x3, s=(sW=1, sH=2), p=1)
    y = conv2d_f32(ctx, g.fcm_conv2.w, y, 1, 2, 1, 1);
    y = bn_apply_2d(ctx, y, g.fcm_bn2);
    y = ggml_relu(ctx, y);

    // y is now (W=T, H=10, C=32, 1) after 3 H-downsamples (80→40→20→10).
    // Flatten H×C into the channel axis: (T, 320).
    //
    // PyTorch's .view(C, H, W) → (320, W) reshape is *bitwise* the same as
    // ggml's (W, H, C) → (W, 320) when the memory is row-major in the ne[0]
    // innermost ordering — which ggml tensors always are for contiguous
    // results of conv2d.  Verified: the 320 channels spell out [h0c0, h0c1,
    // ..., h0c31, h1c0, h1c1, ...] in both cases.
    ggml_tensor * y_cont = ggml_cont(ctx, y);
    ggml_tensor * x1 = ggml_reshape_2d(ctx, y_cont, y_cont->ne[0],
                                        y_cont->ne[1] * y_cont->ne[2]);
    // (W=T_after_fcm, C=320)
    // Cast to 3D for conv1d (it expects [L, C_in, batch]).
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);

    // TDNN: Conv1d(320→128, k=5, s=2, p=2) + BN + ReLU
    x1 = conv1d_f32(ctx, g.tdnn.w, x1, 2, 2, 1);
    // conv1d_f32 returns 3D (T', C_out, batch); squeeze batch for BN broadcast.
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d(ctx, x1, g.tdnn_bn);
    x1 = ggml_relu(ctx, x1);

    // Segment-pool geometry: after TDNN (stride 2 along time), T is fixed for
    // the remainder of the network, so the same seg_scale tensor feeds all 52
    // CAMDenseTDNN layers across block1 / block2 / block3.
    const int T_cam   = (int) x1->ne[0];
    const int S       = (T_cam + g.seg_pool_len - 1) / g.seg_pool_len;
    const int T_pad   = S * g.seg_pool_len;

    // seg_scale: shape (1, S, 1).  Values are 1/bin_size per bin — full
    // seg_len for the first S-1 bins and the remainder for the last bin.
    ggml_tensor * seg_scale = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, S, 1);
    ggml_set_name(seg_scale, "seg_scale"); ggml_set_input(seg_scale);

    // Block1 + transit1
    // cam_dense_tdnn_layer expects (T, C).  conv1d_f32 returns 3D; we reshape
    // back to 2D before each block to keep shapes consistent.  The block
    // builder consumes 2D, emits 2D.
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);
    // Block/transit builders expect 3D (T, C, 1) to feed conv1d_f32 and then
    // squeeze to 2D for BN.  We keep it 3D throughout blocks.
    x1 = build_cam_block (ctx, x1, g.block1, g.seg_pool_len, seg_scale, T_pad, S, T_cam);
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.transit1.bn);
    x1 = ggml_relu      (ctx, x1);
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);
    x1 = conv1d_f32     (ctx, g.transit1.lin.w, x1, 1, 0, 1);

    x1 = build_cam_block (ctx, x1, g.block2, g.seg_pool_len, seg_scale, T_pad, S, T_cam);
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.transit2.bn);
    x1 = ggml_relu      (ctx, x1);
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);
    x1 = conv1d_f32     (ctx, g.transit2.lin.w, x1, 1, 0, 1);

    x1 = build_cam_block (ctx, x1, g.block3, g.seg_pool_len, seg_scale, T_pad, S, T_cam);
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.transit3.bn);
    x1 = ggml_relu      (ctx, x1);
    x1 = ggml_reshape_3d(ctx, x1, x1->ne[0], x1->ne[1], 1);
    x1 = conv1d_f32     (ctx, g.transit3.lin.w, x1, 1, 0, 1);

    // out_nonlinear: BN + ReLU
    x1 = ggml_reshape_2d(ctx, x1, x1->ne[0], x1->ne[1]);
    x1 = bn_apply_1d    (ctx, x1, g.out_bn);
    x1 = ggml_relu      (ctx, x1);

    // stats_pool → (1, 2C)
    ggml_tensor * stats = build_stats_pool(ctx, x1);

    // dense: Conv1x1 (2C → embedding) followed by BN (affine=False; scale is
    // 1/sqrt(var+eps)).  Since T=1 after stats_pool, just use mul_mat.
    //   stats       (1, 2C, 1, 1)
    //   dense_lin.w (1, 2C, E) — Conv1d kernel at k=1
    // Flatten so mul_mat sees (2C) × (E, 2C) = (E).
    ggml_tensor * stats_row = ggml_reshape_2d(ctx, stats, stats->ne[1], 1);  // (2C, 1)
    // dense_lin.w as [k=1, 2C, E] → flatten to [2C, E] for matmul.
    ggml_tensor * dense_w2 = ggml_reshape_2d(ctx, g.dense.w,
                                               g.dense.w->ne[1], g.dense.w->ne[2]);
    ggml_tensor * emb = ggml_mul_mat(ctx, dense_w2, stats_row);              // (E, 1)
    emb = ggml_reshape_2d(ctx, emb, 1, emb->ne[0]);                           // (1, E)
    emb = bn_apply_1d(ctx, emb, g.dense_bn);

    ggml_set_name(emb, "emb"); ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_free(ctx);
    return gf;
}

// ---------- weight upload ----------

static bool upload_cam_layer(const ve_cam_layer_t & gl, const campplus_cam_dense_tdnn_layer & src) {
    upload_bn(gl.bn1, src.bn1);
    if (!init_weight(gl.lin1,  src.linear1))                    return false;
    upload_bn(gl.bn2, src.bn2);
    if (!init_weight(gl.loc,   src.cam_linear_local))           return false;
    if (!init_weight(gl.lin1_cam, src.cam_linear1, true))       return false;
    if (!init_weight(gl.lin2_cam, src.cam_linear2, true))       return false;
    return true;
}

static bool upload_weights(campplus_gctx & g, const campplus_weights & w) {
    // FCM
    if (!init_weight(g.fcm_conv1, w.head.conv1)) return false;
    upload_bn(g.fcm_bn1, w.head.bn1);
    for (int i = 0; i < 2; ++i) {
        const auto & src = w.head.layer1[i];
        auto & dst = g.fcm_layer1[i];
        if (!init_weight(dst.conv1, src.conv1)) return false;
        upload_bn(dst.bn1, src.bn1);
        if (!init_weight(dst.conv2, src.conv2)) return false;
        upload_bn(dst.bn2, src.bn2);
        if (dst.has_shortcut) {
            if (!init_weight(dst.sc, src.shortcut_conv)) return false;
            upload_bn(dst.sc_bn, src.shortcut_bn);
        }
    }
    for (int i = 0; i < 2; ++i) {
        const auto & src = w.head.layer2[i];
        auto & dst = g.fcm_layer2[i];
        if (!init_weight(dst.conv1, src.conv1)) return false;
        upload_bn(dst.bn1, src.bn1);
        if (!init_weight(dst.conv2, src.conv2)) return false;
        upload_bn(dst.bn2, src.bn2);
        if (dst.has_shortcut) {
            if (!init_weight(dst.sc, src.shortcut_conv)) return false;
            upload_bn(dst.sc_bn, src.shortcut_bn);
        }
    }
    if (!init_weight(g.fcm_conv2, w.head.conv2)) return false;
    upload_bn(g.fcm_bn2, w.head.bn2);

    // TDNN
    if (!init_weight(g.tdnn, w.tdnn_linear)) return false;
    upload_bn(g.tdnn_bn, w.tdnn_bn);

    // Blocks
    auto upload_block = [&](ve_cam_block_t & gb, const campplus_cam_block & sb) -> bool {
        for (int i = 0; i < gb.num_layers; ++i) {
            if (!upload_cam_layer(gb.layers[i], sb.layers[i])) return false;
        }
        return true;
    };
    if (!upload_block(g.block1, w.block1)) return false;
    upload_bn(g.transit1.bn, w.transit1.bn);
    if (!init_weight(g.transit1.lin, w.transit1.linear)) return false;
    if (!upload_block(g.block2, w.block2)) return false;
    upload_bn(g.transit2.bn, w.transit2.bn);
    if (!init_weight(g.transit2.lin, w.transit2.linear)) return false;
    if (!upload_block(g.block3, w.block3)) return false;
    upload_bn(g.transit3.bn, w.transit3.bn);
    if (!init_weight(g.transit3.lin, w.transit3.linear)) return false;

    upload_bn(g.out_bn, w.out_nonlinear_bn);
    if (!init_weight(g.dense, w.dense_linear)) return false;
    upload_bn(g.dense_bn, w.dense_bn);
    return true;
}

// ---------- weight tensor declarations ----------

static bool declare_weights(campplus_gctx & g, const campplus_weights & w) {
    // Count of weight tensors we need in the context.  The ggml_init pool
    // must be large enough for all tensor structs we allocate there.
    //   FCM: 1 conv + 1 bn*2 + 4 resblocks × (2 convs + 2 bn*2 + possible sc conv + sc bn*2) + 1 conv + 1 bn*2
    //   TDNN: 1 conv + 1 bn*2
    //   3 blocks × up to 24 layers × (2 bn*2 + 4 convs incl. 2 biased) = up to 24×(4+4+2)=240
    //   3 transits × (1 bn*2 + 1 conv) = 9
    //   1 out_bn + 1 dense conv + 1 dense_bn = 5
    // Rounding up:
    const int n_tensors = 2000;
    ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    g.weights_ctx = ggml_init(ip);
    if (!g.weights_ctx) {
        fprintf(stderr, "campplus_ggml: ggml_init failed\n"); return false;
    }

    g.feat_dim      = w.feat_dim;
    g.seg_pool_len  = w.seg_pool_len;
    g.fcm_out_ch    = 32 * (w.feat_dim / 8);
    g.init_channels = w.tdnn_linear.C_out;
    g.growth_rate   = 32;
    g.bn_channels   = 128;

    // FCM
    decl_conv2d(g.weights_ctx, g.fcm_conv1, 3, 3, 1, 32, "fcm_conv1");
    decl_bn    (g.weights_ctx, g.fcm_bn1, 32, "fcm_bn1");

    auto setup_rb = [&](ve_resblock_t & rb, const campplus_res_block & src,
                        int in_ch, int planes, int stride, const std::string & name) {
        rb.stride_h    = stride;
        rb.has_shortcut = !src.shortcut_conv.w.empty();
        decl_conv2d(g.weights_ctx, rb.conv1, 3, 3, in_ch, planes, name + "_conv1");
        decl_bn    (g.weights_ctx, rb.bn1, planes, name + "_bn1");
        decl_conv2d(g.weights_ctx, rb.conv2, 3, 3, planes, planes, name + "_conv2");
        decl_bn    (g.weights_ctx, rb.bn2, planes, name + "_bn2");
        if (rb.has_shortcut) {
            decl_conv2d(g.weights_ctx, rb.sc, 1, 1, in_ch, planes, name + "_sc");
            decl_bn    (g.weights_ctx, rb.sc_bn, planes, name + "_scbn");
        }
    };
    setup_rb(g.fcm_layer1[0], w.head.layer1[0], 32, 32, 2, "fcm_l10");
    setup_rb(g.fcm_layer1[1], w.head.layer1[1], 32, 32, 1, "fcm_l11");
    setup_rb(g.fcm_layer2[0], w.head.layer2[0], 32, 32, 2, "fcm_l20");
    setup_rb(g.fcm_layer2[1], w.head.layer2[1], 32, 32, 1, "fcm_l21");

    decl_conv2d(g.weights_ctx, g.fcm_conv2, 3, 3, 32, 32, "fcm_conv2");
    decl_bn    (g.weights_ctx, g.fcm_bn2, 32, "fcm_bn2");

    // TDNN
    decl_conv1d(g.weights_ctx, g.tdnn, /*k=*/5, g.fcm_out_ch, g.init_channels, "tdnn");
    decl_bn    (g.weights_ctx, g.tdnn_bn, g.init_channels, "tdnn_bn");

    // Blocks
    auto setup_block = [&](ve_cam_block_t & gb, const campplus_cam_block & sb,
                           int C_in, const std::string & prefix) {
        gb.num_layers  = sb.num_layers;
        gb.kernel_size = sb.kernel_size;
        gb.dilation    = sb.dilation;
        gb.growth      = g.growth_rate;
        gb.bn_channels = g.bn_channels;
        gb.C_in        = C_in;
        gb.layers.resize(gb.num_layers);
        for (int i = 0; i < gb.num_layers; ++i) {
            const int layer_C_in = C_in + i * g.growth_rate;
            auto & L = gb.layers[i];
            const std::string name = prefix + "_l" + std::to_string(i);
            decl_bn    (g.weights_ctx, L.bn1, layer_C_in, name + "_bn1");
            decl_conv1d(g.weights_ctx, L.lin1, 1, layer_C_in, g.bn_channels, name + "_lin1");
            decl_bn    (g.weights_ctx, L.bn2, g.bn_channels, name + "_bn2");
            decl_conv1d(g.weights_ctx, L.loc, gb.kernel_size, g.bn_channels, g.growth_rate, name + "_loc");
            decl_conv1d(g.weights_ctx, L.lin1_cam, 1, g.bn_channels, g.bn_channels / 2, name + "_cam1", /*bias=*/true);
            decl_conv1d(g.weights_ctx, L.lin2_cam, 1, g.bn_channels / 2, g.growth_rate,  name + "_cam2", /*bias=*/true);
        }
    };
    setup_block(g.block1, w.block1, g.init_channels, "b1");
    const int after_b1 = g.init_channels + g.block1.num_layers * g.growth_rate;
    decl_bn    (g.weights_ctx, g.transit1.bn, after_b1, "tr1_bn");
    decl_conv1d(g.weights_ctx, g.transit1.lin, 1, after_b1, after_b1 / 2, "tr1_lin");

    const int b2_in = after_b1 / 2;
    setup_block(g.block2, w.block2, b2_in, "b2");
    const int after_b2 = b2_in + g.block2.num_layers * g.growth_rate;
    decl_bn    (g.weights_ctx, g.transit2.bn, after_b2, "tr2_bn");
    decl_conv1d(g.weights_ctx, g.transit2.lin, 1, after_b2, after_b2 / 2, "tr2_lin");

    const int b3_in = after_b2 / 2;
    setup_block(g.block3, w.block3, b3_in, "b3");
    const int after_b3 = b3_in + g.block3.num_layers * g.growth_rate;
    decl_bn    (g.weights_ctx, g.transit3.bn, after_b3, "tr3_bn");
    decl_conv1d(g.weights_ctx, g.transit3.lin, 1, after_b3, after_b3 / 2, "tr3_lin");

    const int final_ch = after_b3 / 2;
    decl_bn    (g.weights_ctx, g.out_bn, final_ch, "out_bn");
    decl_conv1d(g.weights_ctx, g.dense, 1, final_ch * 2, w.embedding_size, "dense");
    decl_bn    (g.weights_ctx, g.dense_bn, w.embedding_size, "dense_bn");

    g.weights_buf = ggml_backend_alloc_ctx_tensors(g.weights_ctx, g.backend);
    if (!g.weights_buf) {
        fprintf(stderr, "campplus_ggml: weight buffer alloc failed\n"); return false;
    }
    return true;
}

// ---------- public entry point (ggml path) ----------

static bool campplus_embed_ggml(const std::vector<float> & fbank_t_by_c, int T,
                                const campplus_weights & w, ggml_backend_t backend,
                                std::vector<float> & out)
{
    if ((int64_t) fbank_t_by_c.size() != (int64_t) T * w.feat_dim) {
        fprintf(stderr, "campplus_embed_ggml: fbank has %zu elts, expected %d*%d\n",
                fbank_t_by_c.size(), T, w.feat_dim);
        return false;
    }

    // CAMPPlus can now run on any backend: the CAM seg_pool step was
    // reformulated via reshape + sum_rows + broadcast multiply so no
    // `ggml_pool_1d` dispatch is needed (that op has no Vulkan shader and
    // the scheduler's CPU fallback gave cos_sim ~ 0.13 end-to-end).
    campplus_gctx g;
    g.backend = backend;
    if (!g.backend) {
        g.backend      = ggml_backend_cpu_init();
        g.owns_backend = true;
        if (!g.backend) { fprintf(stderr, "campplus_ggml: cpu init failed\n"); return false; }
    }

    if (!declare_weights(g, w)) { campplus_gctx_free(g); return false; }
    if (!upload_weights (g, w)) { campplus_gctx_free(g); return false; }

    ggml_cgraph * gf = build_campplus_graph(g, T);

    g.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g.backend));
    if (!g.allocr || !ggml_gallocr_reserve(g.allocr, gf)) {
        fprintf(stderr, "campplus_ggml: gallocr reserve failed\n");
        campplus_gctx_free(g); return false;
    }
    ggml_gallocr_alloc_graph(g.allocr, gf);

    // Upload fbank: we stored the declaration as (W=T, H=80, C=1, B=1) so the
    // flat memory ordering matches the input we were given (T, 80) row-major.
    ggml_tensor * fb = ggml_graph_get_tensor(gf, "fbank");
    upload(fb, fbank_t_by_c.data(), fbank_t_by_c.size() * sizeof(float));

    // Upload seg_scale for the CAM seg_pool.  T_cam is the post-TDNN time
    // dimension (T/2 since TDNN stride=2).  We rederive it here — staying
    // consistent with the graph builder — and emit one 1/bin_size value per
    // bin, with the last bin possibly partial.
    {
        const int T_cam = (T + 1) / 2;                       // TDNN stride 2, pad 2
        const int S     = (T_cam + w.seg_pool_len - 1) / w.seg_pool_len;
        const int T_pad = S * w.seg_pool_len;
        std::vector<float> scale((size_t) S);
        for (int s = 0; s < S - 1; ++s) scale[s] = 1.0f / (float) w.seg_pool_len;
        scale[S - 1] = 1.0f / (float) (T_cam - (S - 1) * w.seg_pool_len);
        (void) T_pad;
        ggml_tensor * ss = ggml_graph_get_tensor(gf, "seg_scale");
        if (ss) upload(ss, scale.data(), scale.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(g.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "campplus_ggml: graph_compute failed\n");
        campplus_gctx_free(g); return false;
    }

    ggml_tensor * emb = ggml_graph_get_tensor(gf, "emb");
    out.assign((size_t) w.embedding_size, 0.0f);
    ggml_backend_tensor_get(emb, out.data(), 0, out.size() * sizeof(float));

    campplus_gctx_free(g);
    return true;
}
