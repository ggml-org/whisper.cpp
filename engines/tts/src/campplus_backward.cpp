#include "campplus_backward.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tts_cpp {
namespace cp_grad {

namespace {

int conv_out_len(int L_in, int k, int stride, int pad, int dilation) {
    return (L_in + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
}

// Per-channel sum over the time axis of a channel-major (C, T) buffer.
std::vector<double> row_sum_T(const std::vector<double> & x, int C, int T) {
    std::vector<double> s((std::size_t) C, 0.0);
    for (int c = 0; c < C; ++c) {
        const double * row = x.data() + (std::size_t) c * T;
        double acc = 0.0;
        for (int t = 0; t < T; ++t) acc += row[t];
        s[(std::size_t) c] = acc;
    }
    return s;
}

void add_in_place(std::vector<double> & a, const std::vector<double> & b) {
    for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

}  // namespace

CampplusBackward::CampplusBackward(CpWeights weights) : weights_(std::move(weights)) {}

// --- elementwise / pooling primitives ---------------------------------------

std::vector<double> CampplusBackward::bn_forward(const std::vector<double> & x, int C, int T,
                                                 const std::vector<double> & scale,
                                                 const std::vector<double> & shift) {
    std::vector<double> y(x.size());
    for (int c = 0; c < C; ++c) {
        const double s = scale[(std::size_t) c];
        const double b = shift[(std::size_t) c];
        const std::size_t base = (std::size_t) c * T;
        for (int t = 0; t < T; ++t) y[base + t] = x[base + t] * s + b;
    }
    return y;
}

std::vector<double> CampplusBackward::bn_backward_input(const std::vector<double> & d_y, int C, int T,
                                                        const std::vector<double> & scale) {
    std::vector<double> d_x(d_y.size());
    for (int c = 0; c < C; ++c) {
        const double s = scale[(std::size_t) c];
        const std::size_t base = (std::size_t) c * T;
        for (int t = 0; t < T; ++t) d_x[base + t] = d_y[base + t] * s;
    }
    return d_x;
}

std::vector<double> CampplusBackward::relu_forward(const std::vector<double> & x) {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] > 0.0 ? x[i] : 0.0;
    return y;
}

std::vector<double> CampplusBackward::relu_backward(const std::vector<double> & relu_in,
                                                    const std::vector<double> & d_y) {
    std::vector<double> d_x(d_y.size());
    for (std::size_t i = 0; i < d_y.size(); ++i) d_x[i] = relu_in[i] > 0.0 ? d_y[i] : 0.0;
    return d_x;
}

std::vector<double> CampplusBackward::sigmoid_backward(const std::vector<double> & s,
                                                       const std::vector<double> & d_y) {
    std::vector<double> d_x(d_y.size());
    for (std::size_t i = 0; i < d_y.size(); ++i) d_x[i] = d_y[i] * s[i] * (1.0 - s[i]);
    return d_x;
}

// --- conv1d -----------------------------------------------------------------
// y[co, to] = b[co] + sum_{ci, kk} w[(co*C_in+ci)*k + kk] * x[ci, to*stride + kk*dilation - pad]
// (valid taps only; zero padding). Channel-major (C, T).

std::vector<double> CampplusBackward::conv1d_forward(const std::vector<double> & x, int C_in, int T_in,
                                                     const std::vector<double> & w,
                                                     const std::vector<double> & b, int C_out, int k,
                                                     int stride, int pad, int dilation, int T_out) {
    std::vector<double> y((std::size_t) C_out * T_out);
    const bool has_bias = !b.empty();
    for (int co = 0; co < C_out; ++co) {
        const double bias = has_bias ? b[(std::size_t) co] : 0.0;
        const std::size_t w_co = (std::size_t) co * C_in * k;
        double * y_row = y.data() + (std::size_t) co * T_out;
        for (int to = 0; to < T_out; ++to) {
            double acc = bias;
            const int base_t = to * stride - pad;
            for (int ci = 0; ci < C_in; ++ci) {
                const double * x_row = x.data() + (std::size_t) ci * T_in;
                const double * w_row = w.data() + w_co + (std::size_t) ci * k;
                for (int kk = 0; kk < k; ++kk) {
                    const int ti = base_t + kk * dilation;
                    if (ti >= 0 && ti < T_in) acc += w_row[kk] * x_row[ti];
                }
            }
            y_row[to] = acc;
        }
    }
    return y;
}

std::vector<double> CampplusBackward::conv1d_backward_input(const std::vector<double> & d_y, int C_in,
                                                            int T_in, const std::vector<double> & w,
                                                            int C_out, int k, int stride, int pad,
                                                            int dilation, int T_out) {
    std::vector<double> d_x((std::size_t) C_in * T_in, 0.0);
    for (int co = 0; co < C_out; ++co) {
        const std::size_t w_co = (std::size_t) co * C_in * k;
        const double * d_row = d_y.data() + (std::size_t) co * T_out;
        for (int to = 0; to < T_out; ++to) {
            const double g = d_row[to];
            if (g == 0.0) continue;
            const int base_t = to * stride - pad;
            for (int ci = 0; ci < C_in; ++ci) {
                double * dx_row = d_x.data() + (std::size_t) ci * T_in;
                const double * w_row = w.data() + w_co + (std::size_t) ci * k;
                for (int kk = 0; kk < k; ++kk) {
                    const int ti = base_t + kk * dilation;
                    if (ti >= 0 && ti < T_in) dx_row[ti] += g * w_row[kk];
                }
            }
        }
    }
    return d_x;
}

// --- conv2d -----------------------------------------------------------------
// Channel-major (C, H, W); weight (C_out, C_in, kH, kW) row-major.

std::vector<double> CampplusBackward::conv2d_forward(const std::vector<double> & x, int C_in, int H, int W,
                                                     const std::vector<double> & w,
                                                     const std::vector<double> & b, int C_out, int kH,
                                                     int kW, int sH, int sW, int pH, int pW, int H_out,
                                                     int W_out) {
    std::vector<double> y((std::size_t) C_out * H_out * W_out);
    const bool has_bias = !b.empty();
    for (int co = 0; co < C_out; ++co) {
        const double bias = has_bias ? b[(std::size_t) co] : 0.0;
        const std::size_t w_co = (std::size_t) co * C_in * kH * kW;
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                double acc = bias;
                const int base_h = ho * sH - pH;
                const int base_w = wo * sW - pW;
                for (int ci = 0; ci < C_in; ++ci) {
                    const double * x_c = x.data() + (std::size_t) ci * H * W;
                    const double * w_c = w.data() + w_co + (std::size_t) ci * kH * kW;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int hi = base_h + kh;
                        if (hi < 0 || hi >= H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int wi = base_w + kw;
                            if (wi < 0 || wi >= W) continue;
                            acc += w_c[(std::size_t) kh * kW + kw] * x_c[(std::size_t) hi * W + wi];
                        }
                    }
                }
                y[(std::size_t) co * H_out * W_out + (std::size_t) ho * W_out + wo] = acc;
            }
        }
    }
    return y;
}

std::vector<double> CampplusBackward::conv2d_backward_input(const std::vector<double> & d_y, int C_in, int H,
                                                            int W, const std::vector<double> & w, int C_out,
                                                            int kH, int kW, int sH, int sW, int pH, int pW,
                                                            int H_out, int W_out) {
    std::vector<double> d_x((std::size_t) C_in * H * W, 0.0);
    for (int co = 0; co < C_out; ++co) {
        const std::size_t w_co = (std::size_t) co * C_in * kH * kW;
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                const double g = d_y[(std::size_t) co * H_out * W_out + (std::size_t) ho * W_out + wo];
                if (g == 0.0) continue;
                const int base_h = ho * sH - pH;
                const int base_w = wo * sW - pW;
                for (int ci = 0; ci < C_in; ++ci) {
                    double * dx_c = d_x.data() + (std::size_t) ci * H * W;
                    const double * w_c = w.data() + w_co + (std::size_t) ci * kH * kW;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int hi = base_h + kh;
                        if (hi < 0 || hi >= H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int wi = base_w + kw;
                            if (wi < 0 || wi >= W) continue;
                            dx_c[(std::size_t) hi * W + wi] += g * w_c[(std::size_t) kh * kW + kw];
                        }
                    }
                }
            }
        }
    }
    return d_x;
}

// --- mean over time ---------------------------------------------------------

std::vector<double> CampplusBackward::mean_T_forward(const std::vector<double> & x, int C, int T) {
    std::vector<double> m((std::size_t) C);
    for (int c = 0; c < C; ++c) {
        const double * row = x.data() + (std::size_t) c * T;
        double acc = 0.0;
        for (int t = 0; t < T; ++t) acc += row[t];
        m[(std::size_t) c] = acc / (double) T;
    }
    return m;
}

std::vector<double> CampplusBackward::mean_T_backward(const std::vector<double> & d_m, int C, int T) {
    std::vector<double> d_x((std::size_t) C * T);
    const double inv = 1.0 / (double) T;
    for (int c = 0; c < C; ++c) {
        const double g = d_m[(std::size_t) c] * inv;
        double * row = d_x.data() + (std::size_t) c * T;
        for (int t = 0; t < T; ++t) row[t] = g;
    }
    return d_x;
}

// --- segment pooling (ceil mode, true-count last bin), expanded to (C, T) ----

std::vector<double> CampplusBackward::seg_pool_forward(const std::vector<double> & x, int C, int T,
                                                       int seg_len) {
    std::vector<double> out((std::size_t) C * T);
    const int S = (T + seg_len - 1) / seg_len;
    for (int c = 0; c < C; ++c) {
        const double * row = x.data() + (std::size_t) c * T;
        double * dst = out.data() + (std::size_t) c * T;
        for (int s = 0; s < S; ++s) {
            const int t0 = s * seg_len;
            const int t1 = (T < t0 + seg_len) ? T : t0 + seg_len;
            const int n = t1 - t0;
            double acc = 0.0;
            for (int t = t0; t < t1; ++t) acc += row[t];
            const double avg = acc / (double) (n > 0 ? n : 1);
            for (int t = t0; t < t1; ++t) dst[t] = avg;
        }
    }
    return out;
}

std::vector<double> CampplusBackward::seg_pool_backward(const std::vector<double> & d_out, int C, int T,
                                                        int seg_len) {
    std::vector<double> d_x((std::size_t) C * T);
    const int S = (T + seg_len - 1) / seg_len;
    for (int c = 0; c < C; ++c) {
        const double * d_row = d_out.data() + (std::size_t) c * T;
        double * dx_row = d_x.data() + (std::size_t) c * T;
        for (int s = 0; s < S; ++s) {
            const int t0 = s * seg_len;
            const int t1 = (T < t0 + seg_len) ? T : t0 + seg_len;
            const int n = t1 - t0;
            double acc = 0.0;
            for (int t = t0; t < t1; ++t) acc += d_row[t];
            const double g = acc / (double) (n > 0 ? n : 1);
            for (int t = t0; t < t1; ++t) dx_row[t] = g;
        }
    }
    return d_x;
}

// --- statistics pooling (mean + unbiased std) -------------------------------

std::vector<double> CampplusBackward::stats_pool_forward(const std::vector<double> & x, int C, int T,
                                                         std::vector<double> & mean_out,
                                                         std::vector<double> & std_out) {
    std::vector<double> out((std::size_t) 2 * C);
    mean_out.assign((std::size_t) C, 0.0);
    std_out.assign((std::size_t) C, 0.0);
    const double denom = (double) (T > 1 ? T - 1 : 1);
    for (int c = 0; c < C; ++c) {
        const double * row = x.data() + (std::size_t) c * T;
        double sum = 0.0;
        for (int t = 0; t < T; ++t) sum += row[t];
        const double mean = sum / (double) T;
        double sq = 0.0;
        for (int t = 0; t < T; ++t) {
            const double d = row[t] - mean;
            sq += d * d;
        }
        const double sd = std::sqrt(sq / denom);
        mean_out[(std::size_t) c] = mean;
        std_out[(std::size_t) c] = sd;
        out[(std::size_t) c] = mean;
        out[(std::size_t) C + c] = sd;
    }
    return out;
}

std::vector<double> CampplusBackward::stats_pool_backward_input(const std::vector<double> & d_out,
                                                               const std::vector<double> & x, int C, int T,
                                                               const std::vector<double> & mean,
                                                               const std::vector<double> & std_) {
    std::vector<double> d_x((std::size_t) C * T);
    const double inv_T = 1.0 / (double) T;
    const double denom = (double) (T > 1 ? T - 1 : 1);
    for (int c = 0; c < C; ++c) {
        const double d_mean = d_out[(std::size_t) c];
        const double d_std = d_out[(std::size_t) C + c];
        const double sd = std_[(std::size_t) c];
        const double m = mean[(std::size_t) c];
        // d_std/d_x[c,t] = d_std * (x - mean) / ((T-1) * std); the mean-coupling
        // term vanishes because sum_t (x - mean) = 0.
        const double std_coeff = sd > 0.0 ? d_std / (denom * sd) : 0.0;
        const double * row = x.data() + (std::size_t) c * T;
        double * dx_row = d_x.data() + (std::size_t) c * T;
        for (int t = 0; t < T; ++t) {
            dx_row[t] = d_mean * inv_T + std_coeff * (row[t] - m);
        }
    }
    return d_x;
}

// --- FCM residual block ------------------------------------------------------

std::vector<double> CampplusBackward::fcm_resblock_forward(const CpResBlock & blk,
                                                           const std::vector<double> & x, int C_in, int H,
                                                           int W, int & H_out, int & W_out,
                                                           CpResBlockActs & acts) const {
    const int planes = blk.conv1.C_out;
    const int sH = blk.stride_h;
    H_out = conv_out_len(H, 3, sH, 1, 1);
    W_out = conv_out_len(W, 3, 1, 1, 1);
    acts.H_in = H; acts.W_in = W; acts.H_out = H_out; acts.W_out = W_out;

    std::vector<double> t1 = conv2d_forward(x, C_in, H, W, blk.conv1.w, {}, planes, 3, 3, sH, 1, 1, 1,
                                            H_out, W_out);
    t1 = bn_forward(t1, planes, H_out * W_out, blk.bn1.scale, blk.bn1.shift);
    acts.relu1_in = t1;
    t1 = relu_forward(t1);

    std::vector<double> t2 = conv2d_forward(t1, planes, H_out, W_out, blk.conv2.w, {}, planes, 3, 3, 1, 1,
                                            1, 1, H_out, W_out);
    t2 = bn_forward(t2, planes, H_out * W_out, blk.bn2.scale, blk.bn2.shift);

    std::vector<double> sc;
    if (blk.has_shortcut) {
        sc = conv2d_forward(x, C_in, H, W, blk.sc.w, {}, planes, 1, 1, sH, 1, 0, 0, H_out, W_out);
        sc = bn_forward(sc, planes, H_out * W_out, blk.sc_bn.scale, blk.sc_bn.shift);
    }

    std::vector<double> y((std::size_t) planes * H_out * W_out);
    if (sc.empty()) {
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = t2[i] + x[i];
    } else {
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = t2[i] + sc[i];
    }
    acts.relu_out_in = y;
    return relu_forward(y);
}

std::vector<double> CampplusBackward::fcm_resblock_backward(const CpResBlock & blk,
                                                            const CpResBlockActs & acts,
                                                            const std::vector<double> & d_out,
                                                            int C_in) const {
    const int planes = blk.conv1.C_out;
    const int sH = blk.stride_h;
    const int H = acts.H_in, W = acts.W_in, Ho = acts.H_out, Wo = acts.W_out;

    // y = relu(t2 + sc)
    std::vector<double> d_pre = relu_backward(acts.relu_out_in, d_out);  // d(t2+sc)

    // t2 = bn2(conv2(t1))
    std::vector<double> d_t2_bn = bn_backward_input(d_pre, planes, Ho * Wo, blk.bn2.scale);
    std::vector<double> d_t1 = conv2d_backward_input(d_t2_bn, planes, Ho, Wo, blk.conv2.w, planes, 3, 3, 1,
                                                     1, 1, 1, Ho, Wo);
    // t1 = relu(bn1(conv1(x)))
    std::vector<double> d_t1_relu = relu_backward(acts.relu1_in, d_t1);
    std::vector<double> d_t1_bn = bn_backward_input(d_t1_relu, planes, Ho * Wo, blk.bn1.scale);
    std::vector<double> d_x = conv2d_backward_input(d_t1_bn, C_in, H, W, blk.conv1.w, planes, 3, 3, sH, 1,
                                                    1, 1, Ho, Wo);

    // shortcut path
    if (blk.has_shortcut) {
        std::vector<double> d_sc_bn = bn_backward_input(d_pre, planes, Ho * Wo, blk.sc_bn.scale);
        std::vector<double> d_x_sc = conv2d_backward_input(d_sc_bn, C_in, H, W, blk.sc.w, planes, 1, 1, sH,
                                                           1, 0, 0, Ho, Wo);
        add_in_place(d_x, d_x_sc);
    } else {
        // identity shortcut: y += x (shape-preserving block)
        add_in_place(d_x, d_pre);
    }
    return d_x;
}

// --- FCM ---------------------------------------------------------------------

std::vector<double> CampplusBackward::fcm_forward(const std::vector<double> & fbank_ct, int T, int & T_out,
                                                  CpFcmActs & acts) const {
    const CpFcm & f = weights_.head;
    const int F = weights_.feat_dim;
    acts.T = T;

    // conv1: (1 -> 32, k3, s1, p1)
    int H = conv_out_len(F, 3, 1, 1, 1);
    int W = conv_out_len(T, 3, 1, 1, 1);
    std::vector<double> x = conv2d_forward(fbank_ct, 1, F, T, f.conv1.w, {}, 32, 3, 3, 1, 1, 1, 1, H, W);
    x = bn_forward(x, 32, H * W, f.bn1.scale, f.bn1.shift);
    acts.conv1_relu_in = x;
    x = relu_forward(x);

    acts.layer1.assign(f.layer1.size(), CpResBlockActs{});
    for (std::size_t i = 0; i < f.layer1.size(); ++i) {
        int Hn, Wn;
        x = fcm_resblock_forward(f.layer1[i], x, 32, H, W, Hn, Wn, acts.layer1[i]);
        H = Hn; W = Wn;
    }
    acts.layer2.assign(f.layer2.size(), CpResBlockActs{});
    for (std::size_t i = 0; i < f.layer2.size(); ++i) {
        int Hn, Wn;
        x = fcm_resblock_forward(f.layer2[i], x, 32, H, W, Hn, Wn, acts.layer2[i]);
        H = Hn; W = Wn;
    }

    // conv2: (32 -> 32, k3, s(sH=2, sW=1), p1)
    const int H2 = conv_out_len(H, 3, 2, 1, 1);
    const int W2 = conv_out_len(W, 3, 1, 1, 1);
    std::vector<double> y = conv2d_forward(x, 32, H, W, f.conv2.w, {}, 32, 3, 3, 2, 1, 1, 1, H2, W2);
    y = bn_forward(y, 32, H2 * W2, f.bn2.scale, f.bn2.shift);
    acts.conv2_relu_in = y;
    y = relu_forward(y);

    acts.H_after = H2;
    T_out = W2;  // == T (sW=1 throughout)
    // (32, H2, W2) reinterpreted as (32*H2, W2) channel-major — identical memory.
    return y;
}

std::vector<double> CampplusBackward::fcm_backward(const std::vector<double> & d_out,
                                                   const CpFcmActs & acts) const {
    const CpFcm & f = weights_.head;
    const int F = weights_.feat_dim;
    const int T = acts.T;
    const int H_after = acts.H_after;

    // conv2: input (32, H_l2, T), output (32, H_after, T)
    const int H_l2 = acts.layer2.back().H_out;
    std::vector<double> d = relu_backward(acts.conv2_relu_in, d_out);  // (32, H_after*T)
    d = bn_backward_input(d, 32, H_after * T, f.bn2.scale);
    d = conv2d_backward_input(d, 32, H_l2, T, f.conv2.w, 32, 3, 3, 2, 1, 1, 1, H_after, T);

    for (std::size_t i = f.layer2.size(); i-- > 0;) {
        d = fcm_resblock_backward(f.layer2[i], acts.layer2[i], d, 32);
    }
    for (std::size_t i = f.layer1.size(); i-- > 0;) {
        d = fcm_resblock_backward(f.layer1[i], acts.layer1[i], d, 32);
    }

    // conv1: input (1, F, T), output (32, F, T)
    d = relu_backward(acts.conv1_relu_in, d);
    d = bn_backward_input(d, 32, F * T, f.bn1.scale);
    return conv2d_backward_input(d, 1, F, T, f.conv1.w, 32, 3, 3, 1, 1, 1, 1, F, T);  // (F, T)
}

// --- CAMDenseTDNN layer ------------------------------------------------------

std::vector<double> CampplusBackward::cam_layer_forward(const CpCamLayer & L,
                                                        const std::vector<double> & x_in, int C_in, int T,
                                                        int growth, int kernel_size, int dilation,
                                                        int bn_channels, int seg_pool_len,
                                                        CpCamLayerActs & acts) const {
    acts.C_in = C_in;

    // nonlinear1 = BN + ReLU on x_in
    std::vector<double> y = bn_forward(x_in, C_in, T, L.bn1.scale, L.bn1.shift);
    acts.relu1_in = y;
    y = relu_forward(y);

    // linear1: 1x1 conv (C_in -> bn_channels)
    std::vector<double> z = conv1d_forward(y, C_in, T, L.linear1.w, {}, bn_channels, 1, 1, 0, 1, T);

    // nonlinear2 = BN + ReLU
    z = bn_forward(z, bn_channels, T, L.bn2.scale, L.bn2.shift);
    acts.relu2_in = z;
    z = relu_forward(z);  // CAMLayer input

    // linear_local
    const int pad = (kernel_size - 1) / 2 * dilation;
    acts.y_local = conv1d_forward(z, bn_channels, T, L.loc.w, {}, growth, kernel_size, 1, pad, dilation, T);

    // context = mean_T(z) + seg_pool(z)
    const std::vector<double> mean_ctx = mean_T_forward(z, bn_channels, T);
    std::vector<double> context = seg_pool_forward(z, bn_channels, T, seg_pool_len);
    for (int c = 0; c < bn_channels; ++c) {
        const double m = mean_ctx[(std::size_t) c];
        double * row = context.data() + (std::size_t) c * T;
        for (int t = 0; t < T; ++t) row[t] += m;
    }

    // cam linear1: 1x1 (bn_channels -> bn_channels/2) + bias, ReLU
    const int mid = L.cam1.C_out;
    std::vector<double> h1 = conv1d_forward(context, bn_channels, T, L.cam1.w, L.cam1.b, mid, 1, 1, 0, 1, T);
    acts.h1_in = h1;
    h1 = relu_forward(h1);

    // cam linear2: 1x1 (bn_channels/2 -> growth) + bias, sigmoid
    std::vector<double> gate = conv1d_forward(h1, mid, T, L.cam2.w, L.cam2.b, growth, 1, 1, 0, 1, T);
    for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = 1.0 / (1.0 + std::exp(-gate[i]));
    acts.gate = gate;

    // cam_out = y_local * gate, then dense concat [x_in; cam_out]
    std::vector<double> out((std::size_t) (C_in + growth) * T);
    for (std::size_t i = 0; i < (std::size_t) C_in * T; ++i) out[i] = x_in[i];
    double * cam_dst = out.data() + (std::size_t) C_in * T;
    for (std::size_t i = 0; i < acts.y_local.size(); ++i) cam_dst[i] = acts.y_local[i] * gate[i];
    return out;
}

std::vector<double> CampplusBackward::cam_layer_backward(const CpCamLayer & L, const CpCamLayerActs & acts,
                                                         const std::vector<double> & d_out, int C_in, int T,
                                                         int growth, int kernel_size, int dilation,
                                                         int bn_channels, int seg_pool_len) const {
    const int mid = L.cam1.C_out;
    const int pad = (kernel_size - 1) / 2 * dilation;

    // split dense concat
    std::vector<double> d_x((std::size_t) C_in * T);
    for (std::size_t i = 0; i < d_x.size(); ++i) d_x[i] = d_out[i];  // direct identity path
    const double * d_cam = d_out.data() + (std::size_t) C_in * T;    // (growth, T)

    // cam_out = y_local * gate
    std::vector<double> d_y_local((std::size_t) growth * T);
    std::vector<double> d_gate((std::size_t) growth * T);
    for (std::size_t i = 0; i < d_y_local.size(); ++i) {
        d_y_local[i] = d_cam[i] * acts.gate[i];
        d_gate[i] = d_cam[i] * acts.y_local[i];
    }

    // gate = sigmoid(conv2(h1)+b)
    std::vector<double> d_g_pre = sigmoid_backward(acts.gate, d_gate);
    std::vector<double> d_h1 = conv1d_backward_input(d_g_pre, mid, T, L.cam2.w, growth, 1, 1, 0, 1, T);
    // h1 = relu(cam1(context)+b)
    std::vector<double> d_h1_pre = relu_backward(acts.h1_in, d_h1);
    std::vector<double> d_context = conv1d_backward_input(d_h1_pre, bn_channels, T, L.cam1.w, mid, 1, 1, 0,
                                                          1, T);

    // context = seg_pool(z) + mean_T(z) (broadcast)
    std::vector<double> d_mean = row_sum_T(d_context, bn_channels, T);
    std::vector<double> d_z = seg_pool_backward(d_context, bn_channels, T, seg_pool_len);
    add_in_place(d_z, mean_T_backward(d_mean, bn_channels, T));

    // y_local = conv_local(z)
    add_in_place(d_z, conv1d_backward_input(d_y_local, bn_channels, T, L.loc.w, growth, kernel_size, 1, pad,
                                            dilation, T));

    // z = relu(bn2(linear1(relu(bn1(x_in)))))
    std::vector<double> d_z_relu = relu_backward(acts.relu2_in, d_z);
    std::vector<double> d_lin1 = bn_backward_input(d_z_relu, bn_channels, T, L.bn2.scale);
    std::vector<double> d_y = conv1d_backward_input(d_lin1, C_in, T, L.linear1.w, bn_channels, 1, 1, 0, 1, T);
    std::vector<double> d_y_relu = relu_backward(acts.relu1_in, d_y);
    std::vector<double> d_x_branch = bn_backward_input(d_y_relu, C_in, T, L.bn1.scale);

    add_in_place(d_x, d_x_branch);
    return d_x;
}

// --- full chain --------------------------------------------------------------

std::vector<double> CampplusBackward::forward(const std::vector<double> & fbank_t_by_c, int T) {
    const CpWeights & w = weights_;
    const int F = w.feat_dim;
    acts_ = CpActs{};
    acts_.T = T;

    // transpose (T, F) -> (F, T) channel-major
    std::vector<double> fbank_ct((std::size_t) F * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < F; ++c)
            fbank_ct[(std::size_t) c * T + t] = fbank_t_by_c[(std::size_t) t * F + c];

    int T_after_fcm = 0;
    std::vector<double> x = fcm_forward(fbank_ct, T, T_after_fcm, acts_.fcm);
    const int fcm_out_ch = 32 * acts_.fcm.H_after;

    // tdnn: Conv1d(fcm_out -> init_channels, k5, s2, p2) + BN + ReLU
    const int init_C = w.tdnn.C_out;
    const int T_cam = conv_out_len(T_after_fcm, 5, 2, 2, 1);
    acts_.T_cam = T_cam;
    x = conv1d_forward(x, fcm_out_ch, T_after_fcm, w.tdnn.w, {}, init_C, 5, 2, 2, 1, T_cam);
    x = bn_forward(x, init_C, T_cam, w.tdnn_bn.scale, w.tdnn_bn.shift);
    acts_.tdnn_relu_in = x;
    x = relu_forward(x);

    int C_cur = init_C;

    auto run_block = [&](const CpCamBlock & blk, const CpTransit & tr, std::vector<CpCamLayerActs> & bacts,
                         std::vector<double> & tr_relu_in, int & tr_Cin) {
        bacts.assign(blk.layers.size(), CpCamLayerActs{});
        for (std::size_t i = 0; i < blk.layers.size(); ++i) {
            x = cam_layer_forward(blk.layers[i], x, C_cur, T_cam, blk.growth, blk.kernel_size, blk.dilation,
                                  blk.bn_channels, w.seg_pool_len, bacts[i]);
            C_cur += blk.growth;
        }
        // transit: BN + ReLU + 1x1 conv (halves channels)
        tr_Cin = C_cur;
        x = bn_forward(x, C_cur, T_cam, tr.bn.scale, tr.bn.shift);
        tr_relu_in = x;
        x = relu_forward(x);
        const int C_out = tr.linear.C_out;
        x = conv1d_forward(x, C_cur, T_cam, tr.linear.w, {}, C_out, 1, 1, 0, 1, T_cam);
        C_cur = C_out;
    };

    run_block(w.block1, w.transit1, acts_.block1, acts_.tr1_relu_in, acts_.tr1_Cin);
    run_block(w.block2, w.transit2, acts_.block2, acts_.tr2_relu_in, acts_.tr2_Cin);
    run_block(w.block3, w.transit3, acts_.block3, acts_.tr3_relu_in, acts_.tr3_Cin);

    acts_.final_ch = C_cur;

    // out_nonlinear: BN + ReLU
    x = bn_forward(x, C_cur, T_cam, w.out_bn.scale, w.out_bn.shift);
    acts_.out_relu_in = x;
    x = relu_forward(x);
    acts_.stats_x = x;

    // stats pool -> (2*final)
    std::vector<double> stats = stats_pool_forward(x, C_cur, T_cam, acts_.stats_mean, acts_.stats_std);

    // dense: 1x1 conv (2*final -> E) + BN(affine-less)
    const int E = w.embedding_size;
    std::vector<double> emb = conv1d_forward(stats, 2 * C_cur, 1, w.dense.w, {}, E, 1, 1, 0, 1, 1);
    emb = bn_forward(emb, E, 1, w.dense_bn.scale, w.dense_bn.shift);
    return emb;
}

std::vector<double> CampplusBackward::backward(const std::vector<double> & d_emb) const {
    if (acts_.stats_x.empty()) {
        throw std::logic_error("CampplusBackward::backward called before forward (no cached activations)");
    }
    const CpWeights & w = weights_;
    const int T = acts_.T;
    const int T_cam = acts_.T_cam;
    const int final_ch = acts_.final_ch;
    const int E = w.embedding_size;

    // dense: emb = bn(conv1d(stats))
    std::vector<double> d = bn_backward_input(d_emb, E, 1, w.dense_bn.scale);
    std::vector<double> d_stats = conv1d_backward_input(d, 2 * final_ch, 1, w.dense.w, E, 1, 1, 0, 1, 1);

    // stats pool
    std::vector<double> d_x = stats_pool_backward_input(d_stats, acts_.stats_x, final_ch, T_cam,
                                                        acts_.stats_mean, acts_.stats_std);

    // out_nonlinear: relu(bn(prev))
    d_x = relu_backward(acts_.out_relu_in, d_x);
    d_x = bn_backward_input(d_x, final_ch, T_cam, w.out_bn.scale);

    auto run_block_backward = [&](const CpCamBlock & blk, const CpTransit & tr,
                                  const std::vector<CpCamLayerActs> & bacts,
                                  const std::vector<double> & tr_relu_in, int tr_Cin) {
        // transit: x = conv1d(relu(bn(prev)))
        d_x = conv1d_backward_input(d_x, tr_Cin, T_cam, tr.linear.w, tr.linear.C_out, 1, 1, 0, 1, T_cam);
        d_x = relu_backward(tr_relu_in, d_x);
        d_x = bn_backward_input(d_x, tr_Cin, T_cam, tr.bn.scale);
        // block layers in reverse
        int C_in = tr_Cin;
        for (std::size_t i = blk.layers.size(); i-- > 0;) {
            C_in -= blk.growth;
            d_x = cam_layer_backward(blk.layers[i], bacts[i], d_x, C_in, T_cam, blk.growth, blk.kernel_size,
                                     blk.dilation, blk.bn_channels, w.seg_pool_len);
        }
    };

    run_block_backward(w.block3, w.transit3, acts_.block3, acts_.tr3_relu_in, acts_.tr3_Cin);
    run_block_backward(w.block2, w.transit2, acts_.block2, acts_.tr2_relu_in, acts_.tr2_Cin);
    run_block_backward(w.block1, w.transit1, acts_.block1, acts_.tr1_relu_in, acts_.tr1_Cin);

    // tdnn: relu(bn(conv1d(fcm_out)))
    const int init_C = w.tdnn.C_out;
    const int fcm_out_ch = 32 * acts_.fcm.H_after;
    d_x = relu_backward(acts_.tdnn_relu_in, d_x);
    d_x = bn_backward_input(d_x, init_C, T_cam, w.tdnn_bn.scale);
    std::vector<double> d_fcm_out = conv1d_backward_input(d_x, fcm_out_ch, T, w.tdnn.w, init_C, 5, 2, 2, 1,
                                                          T_cam);

    // fcm -> d_fbank_ct (F, T)
    std::vector<double> d_fbank_ct = fcm_backward(d_fcm_out, acts_.fcm);

    // transpose (F, T) -> (T, F)
    const int F = w.feat_dim;
    std::vector<double> d_fbank((std::size_t) T * F);
    for (int c = 0; c < F; ++c)
        for (int t = 0; t < T; ++t)
            d_fbank[(std::size_t) t * F + c] = d_fbank_ct[(std::size_t) c * T + t];
    return d_fbank;
}

}  // namespace cp_grad
}  // namespace tts_cpp
