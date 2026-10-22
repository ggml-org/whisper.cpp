#pragma once

// Graph forms for backends without a transposed convolution (ggml-hexagon has
// no CONV_TRANSPOSE_1D): one GEMM into columns, then ggml_col2im_1d scatters
// them back into the signal.

#include "ggml.h"

namespace tts_cpp::detail {

// Same result as ggml_conv_transpose_1d(kernel, x, stride, 0, 1) with
// `padding` samples cropped from both ends (PyTorch ConvTranspose1d).
// kernel ne=[K, OC, IC], x ne=[T, IC]; returns ne=[(T - 1) * stride + K - 2 * padding, OC].
ggml_tensor * conv_transpose_1d_columns(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * x,
                                        int stride, int padding);

} // namespace tts_cpp::detail
