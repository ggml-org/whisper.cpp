#include "supertonic_vocoder_backward.h"

#include "supertonic_vector_estimator_backward.h"  // shared conv1x1 / layer_norm / gelu

#include <cmath>
#include <cstddef>
#include <utility>

namespace tts_cpp {
namespace voc_grad {

namespace {

inline int causal_src_index(int t, int k, int dilation, int pad_left) {
    int st = t + k * dilation - pad_left;
    return st < 0 ? 0 : st;  // replicate ("causal") left padding
}

}  // namespace

VocoderBackward::VocoderBackward(VocoderWeights weights) : weights_(std::move(weights)) {}

// --- denorm -----------------------------------------------------------------

std::vector<double> VocoderBackward::denorm_forward(const std::vector<double> & x, int L, int C,
                                                    double normalizer_scale, const std::vector<double> & std,
                                                    const std::vector<double> & mean) const {
    std::vector<double> y((std::size_t) L * C);
    const double inv = 1.0 / normalizer_scale;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            const std::size_t i = (std::size_t) t * C + c;
            y[i] = x[i] * inv * std[(std::size_t) c] + mean[(std::size_t) c];
        }
    }
    return y;
}

std::vector<double> VocoderBackward::denorm_backward_input(const std::vector<double> & d_y, int L, int C,
                                                           double normalizer_scale,
                                                           const std::vector<double> & std) const {
    std::vector<double> d_x((std::size_t) L * C);
    const double inv = 1.0 / normalizer_scale;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            const std::size_t i = (std::size_t) t * C + c;
            d_x[i] = d_y[i] * std[(std::size_t) c] * inv;
        }
    }
    return d_x;
}

// --- full causal conv1d -----------------------------------------------------

std::vector<double> VocoderBackward::conv1d_causal_forward(const std::vector<double> & x, int L, int IC,
                                                           int OC, int K, const std::vector<double> & w,
                                                           const std::vector<double> & b) const {
    std::vector<double> y((std::size_t) L * OC);
    const int pad_left = K - 1;
    const bool has_bias = !b.empty();
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            double sum = has_bias ? b[(std::size_t) oc] : 0.0;
            for (int ic = 0; ic < IC; ++ic) {
                const std::size_t wbase = ((std::size_t) oc * IC + ic) * K;
                for (int k = 0; k < K; ++k) {
                    const int st = causal_src_index(t, k, 1, pad_left);
                    sum += w[wbase + k] * x[(std::size_t) st * IC + ic];
                }
            }
            y[(std::size_t) t * OC + oc] = sum;
        }
    }
    return y;
}

std::vector<double> VocoderBackward::conv1d_causal_backward_input(const std::vector<double> & d_y, int L,
                                                                  int IC, int OC, int K,
                                                                  const std::vector<double> & w) const {
    std::vector<double> d_x((std::size_t) L * IC, 0.0);
    const int pad_left = K - 1;
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            const double g = d_y[(std::size_t) t * OC + oc];
            for (int ic = 0; ic < IC; ++ic) {
                const std::size_t wbase = ((std::size_t) oc * IC + ic) * K;
                for (int k = 0; k < K; ++k) {
                    const int st = causal_src_index(t, k, 1, pad_left);
                    d_x[(std::size_t) st * IC + ic] += g * w[wbase + k];
                }
            }
        }
    }
    return d_x;
}

// --- causal depthwise conv1d ------------------------------------------------

std::vector<double> VocoderBackward::depthwise_causal_forward(const std::vector<double> & x, int L, int C,
                                                              int K, int dilation,
                                                              const std::vector<double> & w,
                                                              const std::vector<double> & b) const {
    std::vector<double> y((std::size_t) L * C);
    const int pad_left = (K - 1) * dilation;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            double sum = b[(std::size_t) c];
            const std::size_t wbase = (std::size_t) c * K;
            for (int k = 0; k < K; ++k) {
                const int st = causal_src_index(t, k, dilation, pad_left);
                sum += w[wbase + k] * x[(std::size_t) st * C + c];
            }
            y[(std::size_t) t * C + c] = sum;
        }
    }
    return y;
}

std::vector<double> VocoderBackward::depthwise_causal_backward_input(const std::vector<double> & d_y, int L,
                                                                     int C, int K, int dilation,
                                                                     const std::vector<double> & w) const {
    std::vector<double> d_x((std::size_t) L * C, 0.0);
    const int pad_left = (K - 1) * dilation;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            const double g = d_y[(std::size_t) t * C + c];
            const std::size_t wbase = (std::size_t) c * K;
            for (int k = 0; k < K; ++k) {
                const int st = causal_src_index(t, k, dilation, pad_left);
                d_x[(std::size_t) st * C + c] += g * w[wbase + k];
            }
        }
    }
    return d_x;
}

// --- affine batch norm (inference) ------------------------------------------

std::vector<double> VocoderBackward::batch_norm_forward(const std::vector<double> & x, int L, int C,
                                                        const std::vector<double> & gamma,
                                                        const std::vector<double> & beta,
                                                        const std::vector<double> & running_mean,
                                                        const std::vector<double> & running_var,
                                                        double eps) const {
    std::vector<double> y((std::size_t) L * C);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            const std::size_t i = (std::size_t) t * C + c;
            const double inv = 1.0 / std::sqrt(running_var[(std::size_t) c] + eps);
            y[i] = (x[i] - running_mean[(std::size_t) c]) * inv * gamma[(std::size_t) c] + beta[(std::size_t) c];
        }
    }
    return y;
}

std::vector<double> VocoderBackward::batch_norm_backward_input(const std::vector<double> & d_y, int L, int C,
                                                               const std::vector<double> & gamma,
                                                               const std::vector<double> & running_var,
                                                               double eps) const {
    std::vector<double> d_x((std::size_t) L * C);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            const std::size_t i = (std::size_t) t * C + c;
            const double inv = 1.0 / std::sqrt(running_var[(std::size_t) c] + eps);
            d_x[i] = d_y[i] * gamma[(std::size_t) c] * inv;
        }
    }
    return d_x;
}

// --- leaky relu / prelu -----------------------------------------------------

std::vector<double> VocoderBackward::leaky_relu_forward(const std::vector<double> & x, double slope) const {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] >= 0.0 ? x[i] : slope * x[i];
    return y;
}

std::vector<double> VocoderBackward::leaky_relu_backward(const std::vector<double> & x,
                                                         const std::vector<double> & d_y, double slope) const {
    std::vector<double> d_x(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) d_x[i] = d_y[i] * (x[i] >= 0.0 ? 1.0 : slope);
    return d_x;
}

// --- latent unpack ----------------------------------------------------------

std::vector<double> VocoderBackward::latent_unpack_forward(const std::vector<double> & latent, int latent_len,
                                                           int C_latent, int factor) const {
    const int T0 = latent_len * factor;
    std::vector<double> x((std::size_t) T0 * C_latent);
    for (int c = 0; c < C_latent; ++c) {
        for (int t = 0; t < latent_len; ++t) {
            for (int r = 0; r < factor; ++r) {
                const int src_c = c * factor + r;
                x[(std::size_t) (t * factor + r) * C_latent + c] =
                    latent[(std::size_t) src_c * latent_len + t];
            }
        }
    }
    return x;
}

std::vector<double> VocoderBackward::latent_unpack_backward(const std::vector<double> & d_x, int latent_len,
                                                            int C_latent, int factor) const {
    const int latent_channels = C_latent * factor;
    std::vector<double> d_latent((std::size_t) latent_channels * latent_len);
    for (int c = 0; c < C_latent; ++c) {
        for (int t = 0; t < latent_len; ++t) {
            for (int r = 0; r < factor; ++r) {
                const int src_c = c * factor + r;
                d_latent[(std::size_t) src_c * latent_len + t] =
                    d_x[(std::size_t) (t * factor + r) * C_latent + c];
            }
        }
    }
    return d_latent;
}

// --- vocoder ConvNeXt block -------------------------------------------------

std::vector<double> VocoderBackward::convnext_forward(const VocConvNextWeights & w,
                                                      const std::vector<double> & x, int L,
                                                      VocConvNextActivations & acts) const {
    acts.dw_out = depthwise_causal_forward(x, L, w.C, w.K, w.dilation, w.dw_w, w.dw_b);
    const std::vector<double> ln = ve_grad::layer_norm_forward(acts.dw_out, L, w.C, w.ln_gamma, w.ln_beta);
    acts.z1 = ve_grad::conv1x1_forward(ln, L, w.C, w.hidden, w.pw1_w, w.pw1_b);
    const std::vector<double> g  = ve_grad::gelu_forward(acts.z1);
    const std::vector<double> z2 = ve_grad::conv1x1_forward(g, L, w.hidden, w.C, w.pw2_w, w.pw2_b);

    std::vector<double> out((std::size_t) L * w.C);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = x[i] + w.gamma * z2[i];
    return out;
}

std::vector<double> VocoderBackward::convnext_backward_input(const VocConvNextWeights & w,
                                                             const VocConvNextActivations & acts,
                                                             const std::vector<double> & d_out, int L) const {
    std::vector<double> d_z2((std::size_t) L * w.C);
    for (std::size_t i = 0; i < d_z2.size(); ++i) d_z2[i] = w.gamma * d_out[i];

    const std::vector<double> d_g  = ve_grad::conv1x1_backward_input(d_z2, L, w.hidden, w.C, w.pw2_w);
    const std::vector<double> d_z1 = ve_grad::gelu_backward(acts.z1, d_g);
    const std::vector<double> d_ln = ve_grad::conv1x1_backward_input(d_z1, L, w.C, w.hidden, w.pw1_w);
    const std::vector<double> d_dw = ve_grad::layer_norm_backward_input(acts.dw_out, L, w.C, w.ln_gamma, d_ln);
    const std::vector<double> d_x_dw =
        depthwise_causal_backward_input(d_dw, L, w.C, w.K, w.dilation, w.dw_w);

    std::vector<double> d_x((std::size_t) L * w.C);
    for (std::size_t i = 0; i < d_x.size(); ++i) d_x[i] = d_out[i] + d_x_dw[i];  // residual path
    return d_x;
}

// --- full vocoder -----------------------------------------------------------

std::vector<double> VocoderBackward::forward(const std::vector<double> & latent) {
    const VocoderWeights & w = weights_;
    const int T0 = w.latent_len * w.factor;

    std::vector<double> x = latent_unpack_forward(latent, w.latent_len, w.C_latent, w.factor);
    x = denorm_forward(x, T0, w.C_latent, w.normalizer_scale, w.latent_std, w.latent_mean);
    x = conv1d_causal_forward(x, T0, w.C_latent, w.C, w.K_embed, w.embed_w, w.embed_b);

    block_acts_.assign(w.convnext.size(), VocConvNextActivations{});
    for (std::size_t i = 0; i < w.convnext.size(); ++i) {
        x = convnext_forward(w.convnext[i], x, T0, block_acts_[i]);
    }

    x = batch_norm_forward(x, T0, w.C, w.bn_gamma, w.bn_beta, w.bn_running_mean, w.bn_running_var);

    head1_out_ = conv1d_causal_forward(x, T0, w.C, w.Hh, w.K_head1, w.head1_w, w.head1_b);
    const std::vector<double> p = leaky_relu_forward(head1_out_, w.prelu_slope);
    return ve_grad::conv1x1_forward(p, T0, w.Hh, w.OUT, w.head2_w, /*b=*/{});
}

std::vector<double> VocoderBackward::backward(const std::vector<double> & d_wav) const {
    const VocoderWeights & w = weights_;
    const int T0 = w.latent_len * w.factor;

    std::vector<double> d_p = ve_grad::conv1x1_backward_input(d_wav, T0, w.Hh, w.OUT, w.head2_w);
    std::vector<double> d_x = leaky_relu_backward(head1_out_, d_p, w.prelu_slope);
    d_x = conv1d_causal_backward_input(d_x, T0, w.C, w.Hh, w.K_head1, w.head1_w);
    d_x = batch_norm_backward_input(d_x, T0, w.C, w.bn_gamma, w.bn_running_var);

    for (std::size_t i = w.convnext.size(); i-- > 0;) {
        d_x = convnext_backward_input(w.convnext[i], block_acts_[i], d_x, T0);
    }

    d_x = conv1d_causal_backward_input(d_x, T0, w.C_latent, w.C, w.K_embed, w.embed_w);
    d_x = denorm_backward_input(d_x, T0, w.C_latent, w.normalizer_scale, w.latent_std);
    return latent_unpack_backward(d_x, w.latent_len, w.C_latent, w.factor);
}

}  // namespace voc_grad
}  // namespace tts_cpp
