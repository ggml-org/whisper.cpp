// Forward-parity check for the CAMPPlus backward module.
//
// The gradcheck self-test (test_campplus_backward.cpp) validates the analytic
// backward against finite differences of the SAME double forward. That proves
// the backward is the exact derivative of `CampplusBackward::forward`, but not
// that this forward matches the model CAMPPlus actually runs. This test closes
// that gap: it feeds identical synthetic weights and the same fbank to the
// production scalar forward (`campplus_embed` with backend==nullptr, i.e.
// `campplus_embed_cpu`) and to `CampplusBackward::forward`, and asserts the two
// 192-d embeddings agree. Any drift in layout, dilation schedule, seg-pool
// geometry, stats-pool variance convention or per-channel scaling would surface
// here, so the gradcheck's relevance is anchored to the real forward.
//
// `campplus_embed_cpu` hardcodes growth=32 and bn_channels=128, so the synthetic
// topology below uses those values.
//
// Trust chain: the scalar CPU forward is what every `campplus_embed` caller in
// the repo actually uses (production `main.cpp`, `test-campplus`,
// `test-voice-embedding` all pass backend==nullptr), and `test-campplus` /
// `test-voice-embedding` validate it against the Python reference embedding. So
// anchoring this parity to `campplus_embed_cpu` ties the analytic forward (and
// therefore the gradchecked backward) to the real model: Python -> CPU forward
// -> analytic forward -> backward. The `campplus_embed_ggml` graph path is not
// exercised by any caller today; if it is wired up later it gets its own
// fixture parity against the CPU/Python path.
//
// Built via CMake (links campplus.cpp -> ggml). Runs in the `unit` ctest tier.

#include "campplus.h"
#include "campplus_backward.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace tts_cpp::cp_grad;

namespace {

int g_failures = 0;

double sample(int i, double phase) { return std::sin(i * 0.7 + phase) * 0.5; }

std::vector<float> gen_f(int n, double phase) {
    std::vector<float> v((std::size_t) n);
    for (int i = 0; i < n; ++i) v[i] = (float) sample(i, phase);
    return v;
}

std::vector<double> widen(const std::vector<float> & v) {
    return std::vector<double>(v.begin(), v.end());
}

// --- synthetic float weight builders ----------------------------------------

// Small weights so the deep ReLU stack stays numerically bounded (real CAMPPlus
// weights are BN-normalized; unscaled synthetic weights would blow activations
// up exponentially across the ~10 conv layers and overflow float).
constexpr double kWeightScale = 0.1;

campplus_conv mk_conv2d(int Co, int Ci, int kH, int kW, int sH, int sW, int pH, int pW, double phase) {
    campplus_conv c;
    c.w = gen_f(Co * Ci * kH * kW, phase);
    for (float & v : c.w) v *= (float) kWeightScale;
    c.C_out = Co; c.C_in = Ci; c.kH = kH; c.kW = kW;
    c.stride_h = sH; c.stride_w = sW; c.pad_h = pH; c.pad_w = pW;
    c.dilation_h = 1; c.dilation_w = 1; c.is_2d = true;
    return c;
}

campplus_conv mk_conv1d(int Co, int Ci, int k, int stride, int pad, int dil, double phase, bool bias) {
    campplus_conv c;
    c.w = gen_f(Co * Ci * k, phase);
    for (float & v : c.w) v *= (float) kWeightScale;
    if (bias) c.b = gen_f(Co, phase + 0.5);
    c.C_out = Co; c.C_in = Ci; c.k = k;
    c.stride_w = stride; c.pad_w = pad; c.dilation_w = dil; c.is_2d = false;
    return c;
}

// Positive-biased scale/shift so signal propagates through the ReLU stack and
// the embedding is non-degenerate (a zero-mean BN would let the ReLUs collapse
// everything to the final bias, making the parity comparison vacuous).
campplus_bn mk_bn(int C, double phase) {
    campplus_bn bn;
    bn.scale.resize((std::size_t) C);
    bn.shift.resize((std::size_t) C);
    for (int c = 0; c < C; ++c) {
        bn.scale[(std::size_t) c] = (float) (0.5 + 0.3 * std::fabs(sample(c, phase)));
        bn.shift[(std::size_t) c] = (float) (0.6 + 0.3 * sample(c, phase + 1.0));
    }
    return bn;
}

campplus_res_block mk_resblock(int Ci, int stride, bool shortcut, double phase) {
    campplus_res_block b;
    b.stride_h = stride;
    b.conv1 = mk_conv2d(Ci, Ci, 3, 3, stride, 1, 1, 1, phase);
    b.bn1 = mk_bn(Ci, phase + 0.1);
    b.conv2 = mk_conv2d(Ci, Ci, 3, 3, 1, 1, 1, 1, phase + 0.2);
    b.bn2 = mk_bn(Ci, phase + 0.3);
    if (shortcut) {
        b.shortcut_conv = mk_conv2d(Ci, Ci, 1, 1, stride, 1, 0, 0, phase + 0.4);
        b.shortcut_bn = mk_bn(Ci, phase + 0.5);
    }
    return b;
}

campplus_cam_block mk_cam_block(int num_layers, int kernel_size, int dilation, int C_in, int growth,
                                int bn_channels, double phase) {
    campplus_cam_block blk;
    blk.num_layers = num_layers;
    blk.kernel_size = kernel_size;
    blk.dilation = dilation;
    blk.layers.resize((std::size_t) num_layers);
    const int pad = (kernel_size - 1) / 2 * dilation;
    for (int i = 0; i < num_layers; ++i) {
        const int lc = C_in + i * growth;
        campplus_cam_dense_tdnn_layer & L = blk.layers[(std::size_t) i];
        const double p = phase + i;
        L.bn1 = mk_bn(lc, p + 0.1);
        L.linear1 = mk_conv1d(bn_channels, lc, 1, 1, 0, 1, p + 0.2, false);
        L.bn2 = mk_bn(bn_channels, p + 0.3);
        L.cam_linear_local = mk_conv1d(growth, bn_channels, kernel_size, 1, pad, dilation, p + 0.4, false);
        L.cam_linear1 = mk_conv1d(bn_channels / 2, bn_channels, 1, 1, 0, 1, p + 0.5, true);
        L.cam_linear2 = mk_conv1d(growth, bn_channels / 2, 1, 1, 0, 1, p + 0.6, true);
    }
    return blk;
}

campplus_transit mk_transit(int C_in, double phase) {
    campplus_transit t;
    t.bn = mk_bn(C_in, phase);
    t.linear = mk_conv1d(C_in / 2, C_in, 1, 1, 0, 1, phase + 0.5, false);
    return t;
}

struct Topo {
    // campplus_embed_cpu's fcm_forward hardcodes F=80 (H: 80->40->20->10), so the
    // production CPU path is only self-consistent at feat_dim=80; the parity check
    // must use it. fcm_out = 32 * (80/8) = 320.
    int feat_dim = 80;
    int init_C = 32;
    int growth = 32;           // hardcoded in campplus_embed_cpu
    int bn_channels = 128;     // hardcoded in campplus_embed_cpu
    int kernel_size = 3;
    int embedding = 8;
    int seg_pool_len = 5;
};

campplus_weights build_weights(const Topo & d) {
    campplus_weights w;
    w.feat_dim = d.feat_dim;
    w.embedding_size = d.embedding;
    w.seg_pool_len = d.seg_pool_len;
    w.sample_rate = 16000;

    w.head.conv1 = mk_conv2d(32, 1, 3, 3, 1, 1, 1, 1, 0.1);
    w.head.bn1 = mk_bn(32, 0.2);
    w.head.layer1 = {mk_resblock(32, 2, true, 1.0), mk_resblock(32, 1, false, 2.0)};
    w.head.layer2 = {mk_resblock(32, 2, true, 3.0), mk_resblock(32, 1, false, 4.0)};
    w.head.conv2 = mk_conv2d(32, 32, 3, 3, 2, 1, 1, 1, 5.0);
    w.head.bn2 = mk_bn(32, 5.5);

    const int fcm_out = 32 * (d.feat_dim / 8);  // 320 at feat_dim=80
    w.tdnn_linear = mk_conv1d(d.init_C, fcm_out, 5, 2, 2, 1, 6.0, false);
    w.tdnn_bn = mk_bn(d.init_C, 6.5);

    // Multi-layer CAM blocks (2/3/2) so the dense-concat accumulation (layer i
    // enters with C_in + i*growth) is anchored to production, not only to the
    // self-referential full-chain gradcheck.
    const int b1_layers = 2, b2_layers = 3, b3_layers = 2;
    w.block1 = mk_cam_block(b1_layers, d.kernel_size, 1, d.init_C, d.growth, d.bn_channels, 10.0);
    const int after_b1 = d.init_C + b1_layers * d.growth;
    w.transit1 = mk_transit(after_b1, 20.0);

    const int b2_in = after_b1 / 2;
    w.block2 = mk_cam_block(b2_layers, d.kernel_size, 2, b2_in, d.growth, d.bn_channels, 30.0);
    const int after_b2 = b2_in + b2_layers * d.growth;
    w.transit2 = mk_transit(after_b2, 40.0);

    const int b3_in = after_b2 / 2;
    w.block3 = mk_cam_block(b3_layers, d.kernel_size, 2, b3_in, d.growth, d.bn_channels, 50.0);
    const int after_b3 = b3_in + b3_layers * d.growth;
    w.transit3 = mk_transit(after_b3, 60.0);

    const int final_ch = after_b3 / 2;
    w.out_nonlinear_bn = mk_bn(final_ch, 70.0);
    w.dense_linear = mk_conv1d(d.embedding, final_ch * 2, 1, 1, 0, 1, 80.0, false);
    w.dense_bn = mk_bn(d.embedding, 85.0);
    return w;
}

// --- float -> double weight conversion (campplus_weights -> CpWeights) -------

CpConv to_cp_conv(const campplus_conv & c) {
    CpConv o;
    o.w = widen(c.w);
    o.b = widen(c.b);
    o.C_out = c.C_out; o.C_in = c.C_in; o.k = c.k;
    o.kH = c.kH; o.kW = c.kW;
    o.stride = c.stride_w; o.pad = c.pad_w; o.dilation = c.dilation_w;
    o.stride_h = c.stride_h; o.stride_w = c.stride_w; o.pad_h = c.pad_h; o.pad_w = c.pad_w;
    return o;
}

CpBn to_cp_bn(const campplus_bn & b) {
    CpBn o;
    o.scale = widen(b.scale);
    o.shift = widen(b.shift);
    return o;
}

CpResBlock to_cp_resblock(const campplus_res_block & b) {
    CpResBlock o;
    o.conv1 = to_cp_conv(b.conv1); o.bn1 = to_cp_bn(b.bn1);
    o.conv2 = to_cp_conv(b.conv2); o.bn2 = to_cp_bn(b.bn2);
    o.has_shortcut = !b.shortcut_conv.w.empty();
    if (o.has_shortcut) { o.sc = to_cp_conv(b.shortcut_conv); o.sc_bn = to_cp_bn(b.shortcut_bn); }
    o.stride_h = b.stride_h;
    return o;
}

CpCamBlock to_cp_block(const campplus_cam_block & b, int C_in, int growth, int bn_channels) {
    CpCamBlock o;
    o.num_layers = b.num_layers;
    o.kernel_size = b.kernel_size;
    o.dilation = b.dilation;
    o.growth = growth;
    o.bn_channels = bn_channels;
    o.C_in = C_in;
    o.layers.resize(b.layers.size());
    for (std::size_t i = 0; i < b.layers.size(); ++i) {
        const campplus_cam_dense_tdnn_layer & L = b.layers[i];
        CpCamLayer & d = o.layers[i];
        d.bn1 = to_cp_bn(L.bn1);
        d.linear1 = to_cp_conv(L.linear1);
        d.bn2 = to_cp_bn(L.bn2);
        d.loc = to_cp_conv(L.cam_linear_local);
        d.cam1 = to_cp_conv(L.cam_linear1);
        d.cam2 = to_cp_conv(L.cam_linear2);
    }
    return o;
}

CpTransit to_cp_transit(const campplus_transit & t) {
    CpTransit o;
    o.bn = to_cp_bn(t.bn);
    o.linear = to_cp_conv(t.linear);
    return o;
}

CpWeights to_cp_weights(const campplus_weights & w, const Topo & d) {
    CpWeights o;
    o.feat_dim = w.feat_dim;
    o.embedding_size = w.embedding_size;
    o.seg_pool_len = w.seg_pool_len;

    o.head.conv1 = to_cp_conv(w.head.conv1); o.head.bn1 = to_cp_bn(w.head.bn1);
    for (const auto & b : w.head.layer1) o.head.layer1.push_back(to_cp_resblock(b));
    for (const auto & b : w.head.layer2) o.head.layer2.push_back(to_cp_resblock(b));
    o.head.conv2 = to_cp_conv(w.head.conv2); o.head.bn2 = to_cp_bn(w.head.bn2);

    o.tdnn = to_cp_conv(w.tdnn_linear); o.tdnn_bn = to_cp_bn(w.tdnn_bn);

    const int after_b1 = d.init_C + w.block1.num_layers * d.growth;
    const int b2_in = after_b1 / 2;
    const int after_b2 = b2_in + w.block2.num_layers * d.growth;
    const int b3_in = after_b2 / 2;
    o.block1 = to_cp_block(w.block1, d.init_C, d.growth, d.bn_channels);
    o.transit1 = to_cp_transit(w.transit1);
    o.block2 = to_cp_block(w.block2, b2_in, d.growth, d.bn_channels);
    o.transit2 = to_cp_transit(w.transit2);
    o.block3 = to_cp_block(w.block3, b3_in, d.growth, d.bn_channels);
    o.transit3 = to_cp_transit(w.transit3);

    o.out_bn = to_cp_bn(w.out_nonlinear_bn);
    o.dense = to_cp_conv(w.dense_linear);
    o.dense_bn = to_cp_bn(w.dense_bn);
    return o;
}

}  // namespace

int main() {
    const Topo d;
    const int T = 16;

    const campplus_weights w_f = build_weights(d);
    const CpWeights w_d = to_cp_weights(w_f, d);

    const std::vector<float> fbank_f = gen_f(T * d.feat_dim, 0.3);

    std::vector<float> prod;
    const bool ok = campplus_embed(fbank_f, T, w_f, /*backend=*/nullptr, prod);
    if (!ok) {
        fprintf(stderr, "FAIL campplus_embed (cpu path) returned false\n");
        return 1;
    }

    CampplusBackward backward(w_d);
    const std::vector<double> ref = backward.forward(widen(fbank_f), T);

    if (prod.size() != ref.size() || (int) prod.size() != d.embedding) {
        fprintf(stderr, "FAIL embedding size mismatch: prod=%zu ref=%zu expected=%d\n", prod.size(),
                ref.size(), d.embedding);
        return 1;
    }

    double max_abs = 0.0, max_rel = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double a = (double) prod[i];
        const double b = ref[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++g_failures;
            fprintf(stderr, "FAIL non-finite embedding at %zu: prod=%g ref=%g\n", i, a, b);
            continue;
        }
        const double abs_err = std::fabs(a - b);
        const double rel_err = abs_err / (std::fabs(b) + 1e-6);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
    }

    // float production vs double reference: the only difference is float rounding
    // accumulated through the chain (bn_channels=128 reductions dominate). The
    // measured error is ~3e-8; 1e-4 leaves ample float-accumulation margin while
    // still catching any real layout / convention / wiring drift (which shows up
    // orders of magnitude larger).
    constexpr double kAbsTol = 1e-4;
    if (max_abs > kAbsTol) {
        ++g_failures;
        fprintf(stderr, "FAIL forward parity exceeded tolerance: max_abs=%.3e max_rel=%.3e\n", max_abs,
                max_rel);
    }

    fprintf(stderr, "%s: forward parity max_abs=%.3e max_rel=%.3e (emb[0]=%.6f ref[0]=%.6f)\n",
            g_failures == 0 ? "PASS" : "FAIL", max_abs, max_rel, (double) prod[0], ref[0]);
    return g_failures == 0 ? 0 : 1;
}
