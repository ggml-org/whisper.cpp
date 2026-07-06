#pragma once

// Analytic backward pass for the Supertonic vector estimator (conditional
// flow-matching / CFM) — voice-clone roadmap, ticket "GGML backward pass:
// vector estimator (CFM)".
//
// Scope (full per-step): one CFM step of the vector field is differentiated
// w.r.t. BOTH the optimized style value (`style_ttl`) and the latent input
// (`noisy_latent`).  The latent-input gradient is required so the on-device
// enrollment loop can backprop through the multi-step Euler solver (each step's
// output feeds the next step's input).
//
// Why analytic (not ggml autodiff): the CFM forward uses ops whose backward is
// not implemented in the vendored ggml (`ggml_norm`, `ggml_gelu_*`, the custom
// depthwise op, and the attention path).  The math here is the standard
// conv / layer-norm / attention backward, computed in double for a
// well-conditioned reference, validated component-wise and per-step against
// finite differences by the voiceclone gradcheck harness (Task 2).
//
// All functions are pure: they operate on flat std::vector<double> buffers and
// explicit dimensions, with no dependency on `supertonic_model`.  A thin adapter
// binds the real GGUF weights into these buffers elsewhere.

#include <vector>

namespace tts_cpp {
namespace ve_grad {

// --- Pointwise (1x1) conv, "out-major" weight layout (matches `conv1x1`) ----
// y[t, oc] = b[oc] + sum_ic w[oc * IC + ic] * x[t, ic].  `x` is time-major
// [L, IC]; weight is [OC * IC]; bias is optional ([OC] or empty).
std::vector<double> conv1x1_forward(const std::vector<double> & x, int L, int IC, int OC,
                                    const std::vector<double> & w, const std::vector<double> & b);

// Backward into the input only (weights frozen): d_x [L, IC] from d_y [L, OC].
std::vector<double> conv1x1_backward_input(const std::vector<double> & d_y, int L, int IC, int OC,
                                           const std::vector<double> & w);

// --- Per-time linear, "in-major" ONNX MatMul layout (matches dense_matmul_time)
// y[t, oc] = b[oc] + sum_ic x[t, ic] * w[ic * OC + oc].
std::vector<double> dense_time_forward(const std::vector<double> & x, int L, int IC, int OC,
                                       const std::vector<double> & w, const std::vector<double> & b);

std::vector<double> dense_time_backward_input(const std::vector<double> & d_y, int L, int IC, int OC,
                                              const std::vector<double> & w);

// --- Depthwise "same" conv with edge-clamp padding (matches `depthwise_same`)
// y[t, c] = b[c] + sum_k w[c * K + k] * x[clamp(t + k*dilation - pad_left), c],
// pad_left = ((K - 1) * dilation) / 2.  `x` is time-major [L, C].
std::vector<double> depthwise_same_forward(const std::vector<double> & x, int L, int C, int K,
                                           int dilation, const std::vector<double> & w,
                                           const std::vector<double> & b);

std::vector<double> depthwise_same_backward_input(const std::vector<double> & d_y, int L, int C,
                                                  int K, int dilation, const std::vector<double> & w);

// --- Channel-wise layer norm with affine gamma/beta, per time step over C ----
std::vector<double> layer_norm_forward(const std::vector<double> & x, int L, int C,
                                       const std::vector<double> & gamma,
                                       const std::vector<double> & beta, double eps = 1e-6);

// Backward into the input only (gamma/beta frozen): d_x [L, C].  Takes the
// original input to recompute per-row statistics.
std::vector<double> layer_norm_backward_input(const std::vector<double> & x, int L, int C,
                                              const std::vector<double> & gamma,
                                              const std::vector<double> & d_y, double eps = 1e-6);

// --- erf-GELU, elementwise (matches `gelu`) ---------------------------------
std::vector<double> gelu_forward(const std::vector<double> & x);

// d_x = d_y * gelu'(x), elementwise.
std::vector<double> gelu_backward(const std::vector<double> & x, const std::vector<double> & d_y);

// --- ConvNeXt block (matches `convnext`) ------------------------------------
// out = x + gamma (.) pwconv2(gelu(pwconv1(layer_norm(depthwise(x))))).
// pwconv1 expands C -> hidden, pwconv2 contracts hidden -> C, both 1x1
// out-major-weight convs.  Weights are frozen; only the input gradient is
// produced.
struct ConvNextWeights {
    std::vector<double> dw_w;      // depthwise [C * K]
    std::vector<double> dw_b;      // [C]
    std::vector<double> ln_gamma;  // [C]
    std::vector<double> ln_beta;   // [C]
    std::vector<double> pw1_w;     // [hidden * C]
    std::vector<double> pw1_b;     // [hidden]
    std::vector<double> pw2_w;     // [C * hidden]
    std::vector<double> pw2_b;     // [C]
    std::vector<double> gamma;     // [C], per-channel residual scale
    int C        = 0;
    int hidden   = 0;
    int K        = 0;
    int dilation = 1;
};

// Activations cached by the forward for reuse in the backward.
struct ConvNextActivations {
    std::vector<double> dw_out;  // [L, C], depthwise output (input to layer norm)
    std::vector<double> z1;      // [L, hidden], pwconv1 output (input to gelu)
};

std::vector<double> convnext_forward(const ConvNextWeights & w, const std::vector<double> & x, int L,
                                     ConvNextActivations & acts);

std::vector<double> convnext_backward_input(const ConvNextWeights & w, const ConvNextActivations & acts,
                                            const std::vector<double> & d_out, int L);

// --- Cross-attention -------------------------------------------------------
// Shared geometry for both attention flavours.  Internal attention width is
// A = H * D.  `Ckv` is the channel dim of the key/value source.
struct CrossAttnDims {
    int L     = 0;   // query positions
    int Lk    = 0;   // key/value positions
    int C     = 0;   // query input channels (== output channels)
    int Ckv   = 0;   // key/value source channels
    int H     = 0;   // heads
    int D     = 0;   // head dim
    double scale = 0.0;
};

// Linear projections common to both flavours (in-major ONNX MatMul layout).
struct CrossAttnWeights {
    std::vector<double> wq, bq;  // [C * A], [A]
    std::vector<double> wk, bk;  // [Ckv * A]
    std::vector<double> wv, bv;  // [Ckv * A]
    std::vector<double> wo, bo;  // [A * C]
};

// --- RoPE text cross-attention (matches `rope_attn`) -----------------------
// Q comes from `x` (rotary applied), K/V come from the (constant) text features
// `text_lc` [Lk, Ckv] (rotary applied to K).  Only the query input gradient is
// produced (text features are not optimized).
struct RopeAttnActivations {
    std::vector<double> k_rope;  // [Lk, A], post-rotary key
    std::vector<double> v;       // [Lk, A], value
    std::vector<double> prob;    // [H, L, Lk] softmax weights
};

std::vector<double> rope_attn_forward(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                      const std::vector<double> & theta, const std::vector<double> & x,
                                      const std::vector<double> & text_lc, RopeAttnActivations & acts);

std::vector<double> rope_attn_backward_input(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                             const std::vector<double> & theta,
                                             const RopeAttnActivations & acts,
                                             const std::vector<double> & d_out);

// --- Style cross-attention (matches `style_attn`) --------------------------
// Q comes from `x`; V comes from the optimized style value `style_v` [Lk, Ckv];
// K comes from a constant key source (tanh applied).  Produces gradients into
// both the query input and the style value.
struct StyleAttnActivations {
    std::vector<double> k;     // [Lk, A], post-tanh key (constant)
    std::vector<double> v;     // [Lk, A], value projection of style
    std::vector<double> prob;  // [H, L, Lk] softmax weights
};

struct StyleAttnGrads {
    std::vector<double> d_x;      // [L, C]
    std::vector<double> d_style;  // [Lk, Ckv]
};

std::vector<double> style_attn_forward(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                       const std::vector<double> & x, const std::vector<double> & style_v,
                                       const std::vector<double> & k_const, StyleAttnActivations & acts);

StyleAttnGrads style_attn_backward(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                   const StyleAttnActivations & acts, const std::vector<double> & d_out);

// --- Vector field (one conditional pass, matches `run_field`) ----------------
// proj_in -> [4 groups] -> last_convnext -> proj_out.  Each group:
//   4 convnext (dil 1/2/4/8) -> [time bias: constant, omitted] -> convnext ->
//   rope text-attn (+residual) -> layer norm -> convnext ->
//   style attn (+residual) -> layer norm.
// The time-bias add is a per-step constant (independent of latent and style),
// so it carries no gradient and is omitted from this pure forward/backward.
struct VectorFieldGroupWeights {
    ConvNextWeights convnext_a[4];   // leading 4 convnext, dilations 1/2/4/8
    ConvNextWeights convnext_b;      // single convnext after the time bias
    CrossAttnWeights rope;           // text cross-attention
    std::vector<double> theta;       // [D/2] rotary frequencies
    std::vector<double> ln1_gamma, ln1_beta;
    ConvNextWeights convnext_c;      // single convnext before style attention
    CrossAttnWeights style;          // style cross-attention
    std::vector<double> ln2_gamma, ln2_beta;
};

struct VectorFieldWeights {
    std::vector<double> proj_in_w;   // [C * Cin] out-major 1x1 conv, no bias
    std::vector<double> proj_out_w;  // [Cin * C] out-major 1x1 conv, no bias
    VectorFieldGroupWeights groups[4];
    ConvNextWeights last_convnext[4];
    int Cin = 0;
    int C   = 0;
    CrossAttnDims rope_dims;         // shared shape across groups
    CrossAttnDims style_dims;        // shared shape across groups
    std::vector<double> mask;        // [L], per-time latent mask applied after proj_in
    std::vector<double> text_lc;     // [Lk_text, Ckv] constant text features
    std::vector<double> k_const;     // [Lk_style, Ckv] constant style-key source
};

struct VectorFieldGroupActivations {
    ConvNextActivations cna[4];
    ConvNextActivations cnb;
    RopeAttnActivations rope;
    std::vector<double> ln1_in;
    ConvNextActivations cnc;
    StyleAttnActivations style;
    std::vector<double> ln2_in;
};

struct VectorFieldActivations {
    VectorFieldGroupActivations groups[4];
    ConvNextActivations last[4];
};

struct VectorFieldGrads {
    std::vector<double> d_in;     // [L, Cin]
    std::vector<double> d_style;  // [Lk_style, Ckv], accumulated over the 4 groups
};

// Forward: `in` is time-major [L, Cin]; `style_v` is [Lk_style, Ckv] (shared by
// every group).  Returns the velocity field, time-major [L, Cin].
std::vector<double> vector_field_forward(const VectorFieldWeights & w, const std::vector<double> & in,
                                         const std::vector<double> & style_v, int L,
                                         VectorFieldActivations & acts);

// Backward: from d_out [L, Cin] return d_in [L, Cin] and d_style [Lk_style, Ckv].
VectorFieldGrads vector_field_backward(const VectorFieldWeights & w, const VectorFieldActivations & acts,
                                       const std::vector<double> & d_out, int L);

// --- One CFM Euler step (no classifier-free guidance) -----------------------
// next[c, t] = noisy[c, t] + mask[t] * field(in)[t, c] / total_steps, where
// in[t, c] = noisy[c, t].  `noisy` and the result are channel-major [Cin, L].
// CFG (two field passes combined linearly) is a thin wrapper over this and is
// handled by the enrollment adapter.
struct VectorStepGrads {
    std::vector<double> d_noisy;  // [Cin, L]
    std::vector<double> d_style;  // [Lk_style, Ckv]
};

std::vector<double> vector_step_forward(const VectorFieldWeights & w, const std::vector<double> & noisy,
                                        const std::vector<double> & style_v, int L, int total_steps,
                                        VectorFieldActivations & acts);

VectorStepGrads vector_step_backward(const VectorFieldWeights & w, const VectorFieldActivations & acts,
                                     const std::vector<double> & d_next, int L, int total_steps);

}  // namespace ve_grad
}  // namespace tts_cpp
