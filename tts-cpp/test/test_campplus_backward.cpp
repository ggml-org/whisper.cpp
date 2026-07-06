// Gradcheck self-tests for the CAMPPlus speaker-encoder backward (voice-clone
// ticket "GGML backward pass: CAMPPlus speaker encoder"). Pure host
// logic, model-free: every analytic input-gradient is checked component-wise
// against a central finite-difference numeric gradient of the matching forward,
// using the Task 2 gradcheck harness.  Runs in the always-on `unit` ctest tier.
//
// Standalone build (single line):
//   g++ -std=c++17 -I src test/test_campplus_backward.cpp src/campplus_backward.cpp src/voiceclone_gradcheck.cpp -o /tmp/t && /tmp/t

#include "campplus_backward.h"
#include "voiceclone_gradcheck.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace tts_cpp::cp_grad;
using tts_cpp::voiceclone::ScalarLossFn;
using tts_cpp::voiceclone::compare_gradients;
using tts_cpp::voiceclone::finite_diff_gradient;
using tts_cpp::voiceclone::GradcheckReport;

// Friend accessor: exposes CampplusBackward's private primitives to the tests.
namespace tts_cpp {
namespace cp_grad {
struct CampplusBackwardTester {
    using CB = CampplusBackward;

    static std::vector<double> bn_forward(const std::vector<double> & x, int C, int T,
                                          const std::vector<double> & s, const std::vector<double> & b) {
        return CB::bn_forward(x, C, T, s, b);
    }
    static std::vector<double> bn_backward_input(const std::vector<double> & d, int C, int T,
                                                 const std::vector<double> & s) {
        return CB::bn_backward_input(d, C, T, s);
    }
    static std::vector<double> relu_forward(const std::vector<double> & x) { return CB::relu_forward(x); }
    static std::vector<double> relu_backward(const std::vector<double> & in, const std::vector<double> & d) {
        return CB::relu_backward(in, d);
    }
    static std::vector<double> sigmoid_backward(const std::vector<double> & s,
                                                const std::vector<double> & d) {
        return CB::sigmoid_backward(s, d);
    }
    static std::vector<double> conv1d_forward(const std::vector<double> & x, int Ci, int Ti,
                                              const std::vector<double> & w, const std::vector<double> & b,
                                              int Co, int k, int s, int p, int dl, int To) {
        return CB::conv1d_forward(x, Ci, Ti, w, b, Co, k, s, p, dl, To);
    }
    static std::vector<double> conv1d_backward_input(const std::vector<double> & d, int Ci, int Ti,
                                                     const std::vector<double> & w, int Co, int k, int s,
                                                     int p, int dl, int To) {
        return CB::conv1d_backward_input(d, Ci, Ti, w, Co, k, s, p, dl, To);
    }
    static std::vector<double> conv2d_forward(const std::vector<double> & x, int Ci, int H, int W,
                                              const std::vector<double> & w, const std::vector<double> & b,
                                              int Co, int kH, int kW, int sH, int sW, int pH, int pW,
                                              int Ho, int Wo) {
        return CB::conv2d_forward(x, Ci, H, W, w, b, Co, kH, kW, sH, sW, pH, pW, Ho, Wo);
    }
    static std::vector<double> conv2d_backward_input(const std::vector<double> & d, int Ci, int H, int W,
                                                     const std::vector<double> & w, int Co, int kH, int kW,
                                                     int sH, int sW, int pH, int pW, int Ho, int Wo) {
        return CB::conv2d_backward_input(d, Ci, H, W, w, Co, kH, kW, sH, sW, pH, pW, Ho, Wo);
    }
    static std::vector<double> mean_T_forward(const std::vector<double> & x, int C, int T) {
        return CB::mean_T_forward(x, C, T);
    }
    static std::vector<double> mean_T_backward(const std::vector<double> & d, int C, int T) {
        return CB::mean_T_backward(d, C, T);
    }
    static std::vector<double> seg_pool_forward(const std::vector<double> & x, int C, int T, int sl) {
        return CB::seg_pool_forward(x, C, T, sl);
    }
    static std::vector<double> seg_pool_backward(const std::vector<double> & d, int C, int T, int sl) {
        return CB::seg_pool_backward(d, C, T, sl);
    }
    static std::vector<double> stats_pool_forward(const std::vector<double> & x, int C, int T,
                                                  std::vector<double> & m, std::vector<double> & sd) {
        return CB::stats_pool_forward(x, C, T, m, sd);
    }
    static std::vector<double> stats_pool_backward_input(const std::vector<double> & d,
                                                         const std::vector<double> & x, int C, int T,
                                                         const std::vector<double> & m,
                                                         const std::vector<double> & sd) {
        return CB::stats_pool_backward_input(d, x, C, T, m, sd);
    }
    static std::vector<double> resblock_forward(const CB & cb, const CpResBlock & blk,
                                                const std::vector<double> & x, int Ci, int H, int W,
                                                int & Ho, int & Wo, CpResBlockActs & a) {
        return cb.fcm_resblock_forward(blk, x, Ci, H, W, Ho, Wo, a);
    }
    static std::vector<double> resblock_backward(const CB & cb, const CpResBlock & blk,
                                                 const CpResBlockActs & a, const std::vector<double> & d,
                                                 int Ci) {
        return cb.fcm_resblock_backward(blk, a, d, Ci);
    }
    static std::vector<double> cam_layer_forward(const CB & cb, const CpCamLayer & L,
                                                 const std::vector<double> & x, int Ci, int T, int g, int k,
                                                 int dl, int bn, int sl, CpCamLayerActs & a) {
        return cb.cam_layer_forward(L, x, Ci, T, g, k, dl, bn, sl, a);
    }
    static std::vector<double> cam_layer_backward(const CB & cb, const CpCamLayer & L,
                                                  const CpCamLayerActs & a, const std::vector<double> & d,
                                                  int Ci, int T, int g, int k, int dl, int bn, int sl) {
        return cb.cam_layer_backward(L, a, d, Ci, T, g, k, dl, bn, sl);
    }
};
}  // namespace cp_grad
}  // namespace tts_cpp

namespace {

using Tester = tts_cpp::cp_grad::CampplusBackwardTester;

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, ...) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
        ++g_failures;                                                    \
        fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);            \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                          \
    }                                                                    \
} while (0)

double sample(int i, double phase) { return std::sin(i * 0.9 + phase) * 0.8; }

std::vector<double> make_vector(int n, double phase) {
    std::vector<double> v((std::size_t) n);
    for (int i = 0; i < n; ++i) v[i] = sample(i, phase);
    return v;
}

// ReLU input kept away from the kink: |v| >= 0.3 so a +-eps perturbation never
// flips its sign and the central difference matches the analytic mask.
std::vector<double> make_relu_input(int n, double phase) {
    std::vector<double> v((std::size_t) n);
    for (int i = 0; i < n; ++i) {
        const double s = sample(i, phase);
        v[i] = std::copysign(0.3 + std::fabs(s), s == 0.0 ? 1.0 : s);
    }
    return v;
}

double dot(const std::vector<double> & a, const std::vector<double> & b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
    return acc;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

void report_check(const char * name, const GradcheckReport & r) {
    CHECK(r.passed, "%s: gradcheck failed (max_abs=%.3e max_rel=%.3e worst=%zu)", name, r.max_abs_err,
          r.max_rel_err, r.worst_index);
}

// --- primitive gradchecks ---------------------------------------------------

void test_bn_backward() {
    const int C = 4, T = 5;
    const std::vector<double> scale = make_vector(C, 0.2);
    const std::vector<double> shift = make_vector(C, 1.0);
    const std::vector<double> coeffs = make_vector(C * T, 2.0);
    const std::vector<double> x0 = make_vector(C * T, 0.7);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, Tester::bn_forward(x, C, T, scale, shift));
    };
    report_check("bn_backward_input", compare_gradients(finite_diff_gradient(f, x0),
                                                        Tester::bn_backward_input(coeffs, C, T, scale)));
}

void test_relu_backward() {
    const int n = 20;
    const std::vector<double> coeffs = make_vector(n, 1.3);
    const std::vector<double> x0 = make_relu_input(n, 0.25);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, Tester::relu_forward(x));
    };
    report_check("relu_backward", compare_gradients(finite_diff_gradient(f, x0),
                                                    Tester::relu_backward(x0, coeffs)));
}

void test_sigmoid_backward() {
    const int n = 16;
    const std::vector<double> coeffs = make_vector(n, 1.1);
    const std::vector<double> x0 = make_vector(n, 0.4);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        std::vector<double> y(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) y[i] = sigmoid(x[i]);
        return dot(coeffs, y);
    };
    std::vector<double> s0(x0.size());
    for (std::size_t i = 0; i < x0.size(); ++i) s0[i] = sigmoid(x0[i]);
    report_check("sigmoid_backward", compare_gradients(finite_diff_gradient(f, x0),
                                                       Tester::sigmoid_backward(s0, coeffs)));
}

void test_conv1d_backward() {
    const int Ci = 3, Ti = 7, Co = 4, k = 3, stride = 2, pad = 1, dilation = 2;
    const int To = (Ti + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    const std::vector<double> w = make_vector(Co * Ci * k, 0.3);
    const std::vector<double> b = make_vector(Co, 1.1);
    const std::vector<double> coeffs = make_vector(Co * To, 2.0);
    const std::vector<double> x0 = make_vector(Ci * Ti, 0.7);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, Tester::conv1d_forward(x, Ci, Ti, w, b, Co, k, stride, pad, dilation, To));
    };
    report_check("conv1d_backward_input",
                 compare_gradients(finite_diff_gradient(f, x0),
                                   Tester::conv1d_backward_input(coeffs, Ci, Ti, w, Co, k, stride, pad,
                                                                 dilation, To)));
}

void test_conv2d_backward() {
    const int Ci = 2, H = 5, W = 4, Co = 3, kH = 3, kW = 3, sH = 2, sW = 1, pH = 1, pW = 1;
    const int Ho = (H + 2 * pH - (kH - 1) - 1) / sH + 1;
    const int Wo = (W + 2 * pW - (kW - 1) - 1) / sW + 1;
    const std::vector<double> w = make_vector(Co * Ci * kH * kW, 0.3);
    const std::vector<double> b = make_vector(Co, 0.9);
    const std::vector<double> coeffs = make_vector(Co * Ho * Wo, 1.4);
    const std::vector<double> x0 = make_vector(Ci * H * W, 0.5);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, Tester::conv2d_forward(x, Ci, H, W, w, b, Co, kH, kW, sH, sW, pH, pW, Ho, Wo));
    };
    report_check("conv2d_backward_input",
                 compare_gradients(finite_diff_gradient(f, x0),
                                   Tester::conv2d_backward_input(coeffs, Ci, H, W, w, Co, kH, kW, sH, sW,
                                                                 pH, pW, Ho, Wo)));
}

void test_mean_T_backward() {
    const int C = 4, T = 6;
    const std::vector<double> coeffs = make_vector(C, 1.7);
    const std::vector<double> x0 = make_vector(C * T, 0.3);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, Tester::mean_T_forward(x, C, T));
    };
    report_check("mean_T_backward", compare_gradients(finite_diff_gradient(f, x0),
                                                      Tester::mean_T_backward(coeffs, C, T)));
}

void test_seg_pool_backward() {
    const int C = 3, T = 7, seg = 3;  // S = 3 bins: 3, 3, 1 (partial last)
    const std::vector<double> coeffs = make_vector(C * T, 1.2);
    const std::vector<double> x0 = make_vector(C * T, 0.6);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, Tester::seg_pool_forward(x, C, T, seg));
    };
    report_check("seg_pool_backward", compare_gradients(finite_diff_gradient(f, x0),
                                                        Tester::seg_pool_backward(coeffs, C, T, seg)));
}

void test_stats_pool_backward() {
    const int C = 4, T = 6;
    const std::vector<double> coeffs = make_vector(2 * C, 1.5);
    const std::vector<double> x0 = make_vector(C * T, 0.4);
    std::vector<double> mean, std_;
    Tester::stats_pool_forward(x0, C, T, mean, std_);
    const ScalarLossFn f = [&](const std::vector<double> & x) {
        std::vector<double> m, s;
        return dot(coeffs, Tester::stats_pool_forward(x, C, T, m, s));
    };
    report_check("stats_pool_backward_input",
                 compare_gradients(finite_diff_gradient(f, x0),
                                   Tester::stats_pool_backward_input(coeffs, x0, C, T, mean, std_)));
}

// --- module gradchecks ------------------------------------------------------

// BN with a firmly positive shift so the downstream ReLU stays in its active
// (locally linear) region at the evaluation point: a +-eps finite-difference
// step never crosses the kink, so the central difference matches the analytic
// mask. The ReLU mask=0 branch is covered by the dedicated relu unit test.
CpBn make_bn(int C, double phase) {
    CpBn bn;
    bn.scale.resize((std::size_t) C);
    bn.shift.resize((std::size_t) C);
    for (int c = 0; c < C; ++c) {
        bn.scale[(std::size_t) c] = 0.5 + 0.3 * std::fabs(sample(c, phase));  // positive scale
        bn.shift[(std::size_t) c] = 1.2 + 0.2 * sample(c, phase + 1.0);       // firmly positive bias
    }
    return bn;
}

CpConv make_conv1d(int Co, int Ci, int k, int stride, int pad, int dil, double phase, bool bias) {
    CpConv c;
    c.C_out = Co; c.C_in = Ci; c.k = k; c.stride = stride; c.pad = pad; c.dilation = dil;
    c.w = make_vector(Co * Ci * k, phase);
    for (double & v : c.w) v *= 0.1;  // small weights keep BN shift dominant -> ReLU active
    if (bias) c.b = make_vector(Co, phase + 0.5);
    return c;
}

CpConv make_conv2d(int Co, int Ci, int kH, int kW, int sH, int sW, int pH, int pW, double phase) {
    CpConv c;
    c.C_out = Co; c.C_in = Ci; c.kH = kH; c.kW = kW;
    c.stride_h = sH; c.stride_w = sW; c.pad_h = pH; c.pad_w = pW;
    c.w = make_vector(Co * Ci * kH * kW, phase);
    for (double & v : c.w) v *= 0.1;
    return c;
}

void test_resblock_backward(bool shortcut) {
    const int Ci = 4, H = 6, W = 5;
    const int stride = shortcut ? 2 : 1;
    CpResBlock blk;
    blk.stride_h = stride;
    blk.has_shortcut = shortcut;
    blk.conv1 = make_conv2d(Ci, Ci, 3, 3, stride, 1, 1, 1, 0.2);
    blk.bn1 = make_bn(Ci, 0.3);
    blk.conv2 = make_conv2d(Ci, Ci, 3, 3, 1, 1, 1, 1, 0.4);
    blk.bn2 = make_bn(Ci, 0.5);
    if (shortcut) {
        blk.sc = make_conv2d(Ci, Ci, 1, 1, stride, 1, 0, 0, 0.6);
        blk.sc_bn = make_bn(Ci, 0.7);
    }
    const CampplusBackward cb{CpWeights{}};
    const std::vector<double> x0 = make_vector(Ci * H * W, 0.35);

    int Ho = 0, Wo = 0;
    CpResBlockActs acts;
    Tester::resblock_forward(cb, blk, x0, Ci, H, W, Ho, Wo, acts);
    const std::vector<double> coeffs = make_vector(Ci * Ho * Wo, 1.1);
    const std::vector<double> analytic = Tester::resblock_backward(cb, blk, acts, coeffs, Ci);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        int h, w;
        CpResBlockActs a;
        return dot(coeffs, Tester::resblock_forward(cb, blk, x, Ci, H, W, h, w, a));
    };
    report_check(shortcut ? "fcm_resblock(shortcut) d_x" : "fcm_resblock(identity) d_x",
                 compare_gradients(finite_diff_gradient(f, x0), analytic));
}

void test_cam_layer_backward() {
    const int Ci = 6, T = 9, growth = 4, k = 3, dil = 2, bn = 8, seg = 4;
    CpCamLayer L;
    L.bn1 = make_bn(Ci, 0.2);
    L.linear1 = make_conv1d(bn, Ci, 1, 1, 0, 1, 0.3, false);
    L.bn2 = make_bn(bn, 0.4);
    L.loc = make_conv1d(growth, bn, k, 1, (k - 1) / 2 * dil, dil, 0.5, false);
    L.cam1 = make_conv1d(bn / 2, bn, 1, 1, 0, 1, 0.6, true);
    L.cam2 = make_conv1d(growth, bn / 2, 1, 1, 0, 1, 0.7, true);

    const CampplusBackward cb{CpWeights{}};
    const std::vector<double> x0 = make_vector(Ci * T, 0.3);

    CpCamLayerActs acts;
    Tester::cam_layer_forward(cb, L, x0, Ci, T, growth, k, dil, bn, seg, acts);
    const std::vector<double> coeffs = make_vector((Ci + growth) * T, 1.0);
    const std::vector<double> analytic =
        Tester::cam_layer_backward(cb, L, acts, coeffs, Ci, T, growth, k, dil, bn, seg);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        CpCamLayerActs a;
        return dot(coeffs, Tester::cam_layer_forward(cb, L, x, Ci, T, growth, k, dil, bn, seg, a));
    };
    report_check("cam_dense_tdnn_layer d_x", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

// --- full chain -------------------------------------------------------------

CpCamBlock make_block(int num_layers, int dilation, int C_in, int growth, int bn_channels, int k,
                      double phase) {
    CpCamBlock blk;
    blk.num_layers = num_layers;
    blk.kernel_size = k;
    blk.dilation = dilation;
    blk.growth = growth;
    blk.bn_channels = bn_channels;
    blk.C_in = C_in;
    blk.layers.resize((std::size_t) num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const int lc = C_in + i * growth;
        CpCamLayer & L = blk.layers[(std::size_t) i];
        const double p = phase + i;
        L.bn1 = make_bn(lc, p + 0.1);
        L.linear1 = make_conv1d(bn_channels, lc, 1, 1, 0, 1, p + 0.2, false);
        L.bn2 = make_bn(bn_channels, p + 0.3);
        L.loc = make_conv1d(growth, bn_channels, k, 1, (k - 1) / 2 * dilation, dilation, p + 0.4, false);
        L.cam1 = make_conv1d(bn_channels / 2, bn_channels, 1, 1, 0, 1, p + 0.5, true);
        L.cam2 = make_conv1d(growth, bn_channels / 2, 1, 1, 0, 1, p + 0.6, true);
    }
    return blk;
}

CpTransit make_transit(int C_in, double phase) {
    CpTransit t;
    t.bn = make_bn(C_in, phase);
    t.linear = make_conv1d(C_in / 2, C_in, 1, 1, 0, 1, phase + 0.5, false);
    return t;
}

CpResBlock make_resblock(int Ci, int stride, bool shortcut, double phase) {
    CpResBlock blk;
    blk.stride_h = stride;
    blk.has_shortcut = shortcut;
    blk.conv1 = make_conv2d(Ci, Ci, 3, 3, stride, 1, 1, 1, phase);
    blk.bn1 = make_bn(Ci, phase + 0.1);
    blk.conv2 = make_conv2d(Ci, Ci, 3, 3, 1, 1, 1, 1, phase + 0.2);
    blk.bn2 = make_bn(Ci, phase + 0.3);
    if (shortcut) {
        blk.sc = make_conv2d(Ci, Ci, 1, 1, stride, 1, 0, 0, phase + 0.4);
        blk.sc_bn = make_bn(Ci, phase + 0.5);
    }
    return blk;
}

CpWeights make_tiny_weights() {
    const int feat_dim = 8;       // FCM downsamples H by 8 -> H_after = 1
    const int growth = 4;
    const int bn_channels = 8;
    const int init_C = 8;
    const int k = 3;

    CpWeights w;
    w.feat_dim = feat_dim;
    w.seg_pool_len = 4;
    w.embedding_size = 4;

    // FCM
    w.head.conv1 = make_conv2d(32, 1, 3, 3, 1, 1, 1, 1, 0.1);
    w.head.bn1 = make_bn(32, 0.2);
    w.head.layer1 = {make_resblock(32, 2, true, 1.0), make_resblock(32, 1, false, 2.0)};
    w.head.layer2 = {make_resblock(32, 2, true, 3.0), make_resblock(32, 1, false, 4.0)};
    w.head.conv2 = make_conv2d(32, 32, 3, 3, 2, 1, 1, 1, 5.0);
    w.head.bn2 = make_bn(32, 5.5);

    const int fcm_out = 32;  // 32 * H_after(1)
    w.tdnn = make_conv1d(init_C, fcm_out, 5, 2, 2, 1, 6.0, false);
    w.tdnn_bn = make_bn(init_C, 6.5);

    w.block1 = make_block(2, 1, init_C, growth, bn_channels, k, 10.0);
    const int after_b1 = init_C + 2 * growth;
    w.transit1 = make_transit(after_b1, 20.0);

    const int b2_in = after_b1 / 2;
    w.block2 = make_block(2, 2, b2_in, growth, bn_channels, k, 30.0);
    const int after_b2 = b2_in + 2 * growth;
    w.transit2 = make_transit(after_b2, 40.0);

    const int b3_in = after_b2 / 2;
    w.block3 = make_block(1, 2, b3_in, growth, bn_channels, k, 50.0);
    const int after_b3 = b3_in + 1 * growth;
    w.transit3 = make_transit(after_b3, 60.0);

    const int final_ch = after_b3 / 2;
    w.out_bn = make_bn(final_ch, 70.0);
    w.dense = make_conv1d(w.embedding_size, final_ch * 2, 1, 1, 0, 1, 80.0, false);
    w.dense_bn = make_bn(w.embedding_size, 85.0);
    return w;
}

void test_full_chain_backward() {
    const int T = 12;
    const CpWeights w = make_tiny_weights();
    CampplusBackward cb{w};

    const std::vector<double> fbank0 = make_vector(T * w.feat_dim, 0.3);
    const std::vector<double> emb = cb.forward(fbank0, T);
    const std::vector<double> coeffs = make_vector((int) emb.size(), 1.0);
    const std::vector<double> analytic = cb.backward(coeffs);

    const ScalarLossFn f = [&](const std::vector<double> & fb) {
        CampplusBackward local{w};
        return dot(coeffs, local.forward(fb, T));
    };
    report_check("campplus full-chain d_fbank",
                 compare_gradients(finite_diff_gradient(f, fbank0), analytic));
}

}  // namespace

int main() {
    try {
        test_bn_backward();
        test_relu_backward();
        test_sigmoid_backward();
        test_conv1d_backward();
        test_conv2d_backward();
        test_mean_T_backward();
        test_seg_pool_backward();
        test_stats_pool_backward();
        test_resblock_backward(/*shortcut=*/false);
        test_resblock_backward(/*shortcut=*/true);
        test_cam_layer_backward();
        test_full_chain_backward();
    } catch (const std::exception & e) {
        ++g_failures;
        fprintf(stderr, "FAIL uncaught exception: %s\n", e.what());
    }
    fprintf(stderr, "\n%s: %d/%d checks passed\n", g_failures == 0 ? "PASS" : "FAIL",
            g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
