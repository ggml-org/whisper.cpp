#pragma once

// Analytic backward pass for the style-dependent tail of the Supertonic text
// encoder (voice-clone roadmap, ticket "GGML backward pass: Supertonic text
// encoder").
//
// Why only the tail: in the forward pass `style_ttl` enters the text encoder in
// exactly one place — the speech-prompted attention (two layers) plus the final
// layer norm.  Everything upstream (embedding, convnext, relpos/ffn stack) is
// independent of `style_ttl`, so its gradient w.r.t. the style vector is zero by
// construction.  Voice cloning optimizes only `style_ttl` (the model weights are
// frozen), so the gradient we need is d(text_emb_out)/d(style_ttl), which flows
// solely through this tail.
//
// Why analytic (not ggml autodiff): the forward tail uses ops whose backward is
// not implemented in the vendored ggml (`ggml_norm`, `ggml_gelu_*`,
// `flash_attn_ext`).  The math here is the standard attention / layer-norm
// backward, computed in double for a well-conditioned reference, and validated
// component-wise against finite differences by the voiceclone gradcheck harness.
//
// All functions are pure: they operate on flat std::vector<double> buffers and
// explicit dimensions, with no dependency on `supertonic_model`.  A thin adapter
// binds the real model weights into these buffers elsewhere.

#include <array>
#include <vector>

namespace tts_cpp {
namespace supertonic_grad {

// Dense per-time-step linear, ONNX MatMul weight layout: y[t, oc] = b[oc] +
// sum_ic x[t, ic] * w[ic * out_dim + oc].  `x` is time-major [L, in_dim].
struct LinearWeights {
    std::vector<double> w;  // [in_dim * out_dim], row-major over (in_dim, out_dim)
    std::vector<double> b;  // [out_dim]
    int in_dim  = 0;
    int out_dim = 0;
};

// Forward: returns y as time-major [L, out_dim].
std::vector<double> dense_time_forward(const std::vector<double> & x, int L,
                                       const LinearWeights & weights);

// Backward into the input only (weights are frozen): returns d_x [L, in_dim]
// from the upstream gradient d_y [L, out_dim].
std::vector<double> dense_time_backward_input(const std::vector<double> & d_y, int L,
                                              const LinearWeights & weights);

// Channel-wise layer norm with affine gamma/beta, applied independently per time
// step over the C channels (matches `layer_norm_channel` in the forward).
std::vector<double> layer_norm_channel_forward(const std::vector<double> & x_lc, int L, int C,
                                               const std::vector<double> & gamma,
                                               const std::vector<double> & beta,
                                               double eps = 1e-6);

// Backward into the input only (gamma/beta frozen): returns d_x [L, C].  Takes
// the original input `x_lc` to recompute the per-row statistics.
std::vector<double> layer_norm_channel_backward(const std::vector<double> & x_lc, int L, int C,
                                                const std::vector<double> & gamma,
                                                const std::vector<double> & d_y,
                                                double eps = 1e-6);

// Speech-prompted attention (one layer).  Q is projected from the text features
// `x_lc`, K is the constant per-head `tanh_k`, V is projected from `style`.  The
// channels split into `heads` contiguous head blocks of size C/heads.
struct SpeechAttentionDims {
    int L     = 0;   // text length (query positions)
    int Lctx  = 0;   // style context length (key/value positions)
    int C     = 0;   // channels
    int heads = 0;   // attention heads; head_dim = C / heads
    double scale = 0.0;
};

struct SpeechAttentionWeights {
    LinearWeights q;             // [C -> C]
    LinearWeights v;             // [C -> C]
    LinearWeights o;             // [C -> C], output projection
    std::vector<double> tanh_k;  // constant K, [heads * head_dim * Lctx], k[(h*head_dim+d)*Lctx + j]
};

// Activations cached by the forward for reuse in the backward.
struct SpeechAttentionActivations {
    std::vector<double> attn;  // softmax weights, [heads, L, Lctx]: attn[(h*L + t)*Lctx + j]
    std::vector<double> v;     // value projection of style, [Lctx, C]
};

// Forward: returns out [L, C] and fills `acts`.
std::vector<double> speech_attention_forward(const SpeechAttentionDims & dims,
                                             const SpeechAttentionWeights & weights,
                                             const std::vector<double> & x_lc,
                                             const std::vector<double> & style,
                                             SpeechAttentionActivations & acts);

// Backward: from d_out [L, C] produce gradients into the query input and the
// style input.  Attention weights depend on the query (not on style), so the
// softmax backward is required to reach `d_x`.
struct SpeechAttentionGrads {
    std::vector<double> d_x;      // [L, C]
    std::vector<double> d_style;  // [Lctx, C]
};

SpeechAttentionGrads speech_attention_backward(const SpeechAttentionDims & dims,
                                               const SpeechAttentionWeights & weights,
                                               const SpeechAttentionActivations & acts,
                                               const std::vector<double> & d_out);

// Style-dependent tail of the text encoder: the two speech-prompted attention
// layers (each added to the shared, style-independent stack output) followed by
// the final layer norm.  This is the full path through which `style_ttl` reaches
// the text-encoder output, and therefore the complete text-encoder backward
// needed for voice cloning.
//
// Forward:
//   attn0   = speech_attention(stack_out, style)
//   x1      = stack_out + attn0
//   attn1   = speech_attention(x1, style)
//   x_final = stack_out + attn1
//   out     = layer_norm(x_final)
struct SpeechTailWeights {
    std::array<SpeechAttentionWeights, 2> spa;
    std::vector<double> ln_gamma;  // [C]
    std::vector<double> ln_beta;   // [C]
};

struct SpeechTailActivations {
    std::array<SpeechAttentionActivations, 2> spa;
    std::vector<double> x_final;  // [L, C], input to the final layer norm
};

// `stack_out` is the style-independent text stack output [L, C]; `style` is
// [Lctx, C].  Returns the text-encoder output [L, C] and fills `acts`.
std::vector<double> speech_tail_forward(const SpeechAttentionDims & dims,
                                        const SpeechTailWeights & weights,
                                        const std::vector<double> & stack_out,
                                        const std::vector<double> & style,
                                        SpeechTailActivations & acts);

// From the upstream output gradient d_out [L, C], returns d(loss)/d(style)
// [Lctx, C].  The gradient into `stack_out` is omitted: it is style-independent,
// so it contributes nothing to the cloning optimization.
std::vector<double> speech_tail_backward(const SpeechAttentionDims & dims,
                                         const SpeechTailWeights & weights,
                                         const SpeechTailActivations & acts,
                                         const std::vector<double> & d_out);

}  // namespace supertonic_grad
}  // namespace tts_cpp
