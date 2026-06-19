#pragma once

// Analytic backward pass for the Supertonic vocoder — voice-clone roadmap,
// ticket "GGML backward pass: vocoder" (QVAC-20983).
//
// Scope: the vocoder maps a latent (the CFM output) to a waveform. For
// gradient-based voice cloning only `style_ttl` is optimized, so the gradient
// this class produces is `d(loss)/d(latent)` — the signal the on-device
// enrollment loop backprops from the audio loss down through the vocoder into
// the latent, which then flows back through the vector estimator and text
// encoder to `style_ttl`. Model weights are frozen, so the backward returns the
// input gradient only.
//
// Why analytic (not ggml autodiff): the vocoder forward uses ops whose backward
// is not implemented in the vendored ggml (`ggml_norm` layer norm, `ggml_gelu_*`,
// the custom causal depthwise op, and the leaky-relu/prelu lowering). The
// "transposed convolution" the upsampling notionally needs is realized as a
// fixed reshape+permute (the latent unpack), so its backward is a pure
// permutation rather than a conv-transpose kernel.
//
// `VocoderBackward` owns the frozen weights and caches the per-call activations
// as state: `forward(latent)` runs the chain and stores the activations needed
// by `backward(d_wav)`. The math is computed in double for a well-conditioned
// reference and validated component-wise against central finite differences by
// the voiceclone gradcheck harness (Task 2 / QVAC-20979). The class has no
// dependency on `supertonic_model`; a thin adapter binds the real GGUF weights
// into `VocoderWeights` elsewhere.
//
// Pointwise (1x1) convs, channel layer norm and erf-GELU are shared with the
// vector-estimator backward (`tts_cpp::ve_grad`) since the math is identical;
// the class adds the vocoder-specific causal convs, affine batch norm,
// leaky-relu, scalar-gamma ConvNeXt block, latent unpack and the full chain.

#include <vector>

namespace tts_cpp {
namespace voc_grad {

// --- Plain data holders ------------------------------------------------------

// Vocoder ConvNeXt block weights (matches `convnext_block`). Differs from the
// vector-estimator block: depthwise is *causal* (left pad), and `gamma` is a
// single scalar residual scale (not per-channel).
struct VocConvNextWeights {
    std::vector<double> dw_w;      // depthwise [C * K]
    std::vector<double> dw_b;      // [C]
    std::vector<double> ln_gamma;  // [C]
    std::vector<double> ln_beta;   // [C]
    std::vector<double> pw1_w;     // [hidden * C]
    std::vector<double> pw1_b;     // [hidden]
    std::vector<double> pw2_w;     // [C * hidden]
    std::vector<double> pw2_b;     // [C]
    double gamma = 0.0;            // scalar residual scale
    int C        = 0;
    int hidden   = 0;
    int K        = 0;
    int dilation = 1;
};

// Activations cached by a ConvNeXt forward for reuse in its backward.
struct VocConvNextActivations {
    std::vector<double> dw_out;  // [L, C], depthwise output (input to layer norm)
    std::vector<double> z1;      // [L, hidden], pwconv1 output (input to gelu)
};

// Full vocoder weights (matches `supertonic_vocoder_forward_cpu`).
struct VocoderWeights {
    int latent_len    = 0;  // L in the packed latent
    int C_latent      = 0;  // unpacked latent channels
    int factor        = 0;  // time upsample factor (latent_channels = C_latent * factor)
    int C             = 0;  // embed output channels (ConvNeXt width)
    double normalizer_scale = 1.0;
    std::vector<double> latent_mean;  // [C_latent]
    std::vector<double> latent_std;   // [C_latent]

    std::vector<double> embed_w;  // [C * C_latent * K_embed] (OC=C, IC=C_latent)
    std::vector<double> embed_b;  // [C]
    int K_embed = 0;

    std::vector<VocConvNextWeights> convnext;  // ConvNeXt chain (carries dilations)

    std::vector<double> bn_gamma;         // [C]
    std::vector<double> bn_beta;          // [C]
    std::vector<double> bn_running_mean;  // [C]
    std::vector<double> bn_running_var;   // [C]

    std::vector<double> head1_w;  // [Hh * C * K_head1]
    std::vector<double> head1_b;  // [Hh]
    int K_head1 = 0;
    int Hh      = 0;  // head1 output channels
    double prelu_slope = 0.0;

    std::vector<double> head2_w;  // [OUT * Hh] pointwise (K=1), no bias
    int OUT = 0;                  // waveform output channels
};

// --- Vocoder backward --------------------------------------------------------
//
// Stateful: construct with the frozen weights, call `forward(latent)` (which
// caches the activations), then `backward(d_wav)` (which consumes them). Only
// the construction + forward/backward surface is public; the stateless math
// primitives are private implementation details. The gradcheck self-tests reach
// them through a `friend` so each primitive is still validated individually
// against finite differences without widening the public API.
class VocoderBackward {
public:
    explicit VocoderBackward(VocoderWeights weights);

    const VocoderWeights & weights() const { return weights_; }

    // Forward: `latent` is channel-major [latent_channels, latent_len]. Runs the
    // chain, caches the activations as state and returns the waveform, time-major
    // [T0, OUT] with T0 = latent_len * factor.
    std::vector<double> forward(const std::vector<double> & latent);

    // Backward: from d_wav [T0, OUT] return d_latent in the channel-major
    // [latent_channels, latent_len] layout the forward consumes. Uses the
    // activations cached by the most recent `forward`.
    std::vector<double> backward(const std::vector<double> & d_wav) const;

private:
    // Grants the gradcheck self-tests access to the private primitives below.
    friend struct VocoderBackwardTester;

    // The primitives below are pure: they read no member state, so they are
    // marked `const` (callable from `backward`). They are private helpers of the
    // forward/backward chain, individually gradchecked through the friend tester.

    // --- Latent denormalization (matches the vocoder "denorm" stage) ---------
    // y[t, c] = (x[t, c] / normalizer_scale) * std[c] + mean[c]. `x` is
    // time-major [L, C]; `std`/`mean` are per-channel [C].
    std::vector<double> denorm_forward(const std::vector<double> & x, int L, int C, double normalizer_scale,
                                       const std::vector<double> & std,
                                       const std::vector<double> & mean) const;

    std::vector<double> denorm_backward_input(const std::vector<double> & d_y, int L, int C,
                                              double normalizer_scale, const std::vector<double> & std) const;

    // --- Full causal conv1d, IC -> OC, K taps (matches `conv1d_causal`) ------
    // Left replicate ("causal") padding by K-1. Weight is ONNX row-major
    // [OC, IC, K] with raw index ((oc * IC + ic) * K + k); bias is optional
    // ([OC] or empty). `x` is time-major [L, IC]; output is [L, OC].
    std::vector<double> conv1d_causal_forward(const std::vector<double> & x, int L, int IC, int OC, int K,
                                              const std::vector<double> & w,
                                              const std::vector<double> & b) const;

    std::vector<double> conv1d_causal_backward_input(const std::vector<double> & d_y, int L, int IC, int OC,
                                                     int K, const std::vector<double> & w) const;

    // --- Causal depthwise conv1d (matches `depthwise_conv1d_causal`) ---------
    // Left replicate padding by (K-1)*dilation. Weight is [C, K] with raw index
    // c * K + k; bias is per-channel [C]. `x` is time-major [L, C].
    std::vector<double> depthwise_causal_forward(const std::vector<double> & x, int L, int C, int K,
                                                 int dilation, const std::vector<double> & w,
                                                 const std::vector<double> & b) const;

    std::vector<double> depthwise_causal_backward_input(const std::vector<double> & d_y, int L, int C, int K,
                                                        int dilation, const std::vector<double> & w) const;

    // --- Affine batch norm at inference (matches `batch_norm_channel`) -------
    // y[t, c] = (x[t, c] - running_mean[c]) / sqrt(running_var[c] + eps) *
    //           gamma[c] + beta[c]. Per-channel [C]; constants at inference, so
    // the backward into the input is a per-channel scale.
    std::vector<double> batch_norm_forward(const std::vector<double> & x, int L, int C,
                                           const std::vector<double> & gamma, const std::vector<double> & beta,
                                           const std::vector<double> & running_mean,
                                           const std::vector<double> & running_var, double eps = 1e-5) const;

    std::vector<double> batch_norm_backward_input(const std::vector<double> & d_y, int L, int C,
                                                  const std::vector<double> & gamma,
                                                  const std::vector<double> & running_var,
                                                  double eps = 1e-5) const;

    // --- Leaky-relu / prelu with a scalar negative slope (head prelu) --------
    // y = x >= 0 ? x : slope * x, elementwise.
    std::vector<double> leaky_relu_forward(const std::vector<double> & x, double slope) const;

    std::vector<double> leaky_relu_backward(const std::vector<double> & x, const std::vector<double> & d_y,
                                            double slope) const;

    // --- Latent unpack (the notional "transposed conv" upsampling) -----------
    // The latent ships channel-major [latent_channels, latent_len] (raw index
    // (c*factor + r) * latent_len + t) and is unpacked to the time-major
    // activation [T0, C_latent] with T0 = latent_len * factor and
    // latent_channels = C_latent * factor, via
    //   x[(t*factor + r) * C_latent + c] = latent[(c*factor + r)*latent_len + t].
    // This is a pure permutation; its backward is the transpose gather.
    std::vector<double> latent_unpack_forward(const std::vector<double> & latent, int latent_len, int C_latent,
                                              int factor) const;

    std::vector<double> latent_unpack_backward(const std::vector<double> & d_x, int latent_len, int C_latent,
                                               int factor) const;

    // --- Vocoder ConvNeXt block (matches `convnext_block`) -------------------
    // out = x + gamma * pwconv2(gelu(pwconv1(layer_norm(depthwise_causal(x))))).
    // pwconv1/pwconv2 are 1x1 out-major-weight convs (shared `ve_grad::conv1x1`).
    std::vector<double> convnext_forward(const VocConvNextWeights & w, const std::vector<double> & x, int L,
                                         VocConvNextActivations & acts) const;

    std::vector<double> convnext_backward_input(const VocConvNextWeights & w,
                                                const VocConvNextActivations & acts,
                                                const std::vector<double> & d_out, int L) const;

    VocoderWeights weights_;
    std::vector<VocConvNextActivations> block_acts_;  // per ConvNeXt block
    std::vector<double> head1_out_;                   // [T0, Hh], the prelu input
};

}  // namespace voc_grad
}  // namespace tts_cpp
