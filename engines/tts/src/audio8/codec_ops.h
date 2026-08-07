#pragma once
// Building blocks shared by the Audio8 codec's encoder and decoder.
//
// Every activation here is [channels, length] with channels inner-most, which
// is torch's [B, L, C]. The transformers, the ConvNeXt blocks and the codebook
// lookups all want that layout already, so keeping the convolutions in it too
// means the codec never transposes.
//
// A convolution is therefore a stack of per-tap [in, out] matrices applied to
// shifted views of the signal rather than an im2col patch matrix: the kernels
// arrive tap-major from the converter, so each tap is a contiguous slice and
// nothing has to materialise K copies of the input.
//
// All of them are causal. The reference left-pads by the whole receptive field
// and right-pads only enough to reach a whole number of strides.

#include "ggml.h"

#include <vector>

#include "audio8/internal.h"

namespace tts_cpp {
namespace audio8 {
namespace detail {

ggml_tensor * causal_conv(ggml_context * ctx, const conv_weights & conv, ggml_tensor * x,
                          int stride, int dilation);
ggml_tensor * causal_depthwise_conv(ggml_context * ctx, const conv_weights & conv,
                                    ggml_tensor * x);
// Requires the kernel width to be a whole number of strides, which holds for
// every transposed convolution in the codec.
ggml_tensor * causal_conv_transpose(ggml_context * ctx, const conv_weights & conv,
                                    ggml_tensor * x, int stride);

// A 1x1 convolution, which is what the quantizer projections are.
ggml_tensor * project(ggml_context * ctx, const conv_weights & conv, ggml_tensor * x);

ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float epsilon);
ggml_tensor * convnext_block(ggml_context * ctx, const convnext_weights & block,
                             ggml_tensor * x, float eps);

// The three dilated residual units every DAC stage wraps around its resampling
// convolution.
ggml_tensor * residual_stack(ggml_context * ctx, const std::vector<residual_unit> & units,
                             const std::vector<int> & dilations, ggml_tensor * x,
                             float snake_epsilon);

// Runs the whole sequence through a window transformer; x is [dim, length] and
// comes back the same shape.
ggml_tensor * window_forward(ggml_context * ctx, const window_transformer & transformer,
                             ggml_tensor * x, ggml_tensor * mask);

// Fills a [length, length] mask for a transformer that sees its whole sequence
// at once.
std::vector<float> window_mask(const transformer_spec & spec, int length);

// Kernel widths, which the two layouts spell differently: a full kernel is
// [in, out, taps] and a depthwise one [channels, taps].
int conv_taps(const conv_weights & conv);
int depthwise_taps(const conv_weights & conv);

// How far back one output column of a causal module reaches, walked from the
// output towards the input: `span` goes in as the number of output columns and
// comes out as the number of input columns they need.
int span_through_conv(int span, int taps, int stride, int dilation);
int span_through_transpose(int span, int taps, int stride);
int span_through_convnext(int span, const convnext_weights & block);
int span_through_units(int span, const std::vector<residual_unit> & units,
                       const std::vector<int> & dilations);

// Asks the hook whether to stop, and writes the cancellation error when it
// says so. Both directions call it between blocks.
bool cancelled(const cancel_hook & cancel, std::string * error);

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
