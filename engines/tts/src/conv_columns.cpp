#include "conv_columns.h"

namespace tts_cpp::detail {

ggml_tensor * conv_transpose_1d_columns(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * x,
                                        int stride, int padding) {
    const int64_t taps            = kernel->ne[0];
    const int64_t out_channels    = kernel->ne[1];
    const int64_t in_channels     = kernel->ne[2];
    ggml_tensor * kernel_rows     = ggml_reshape_2d(ctx, kernel, taps * out_channels, in_channels);
    ggml_tensor * kernel_by_input = ggml_cont(ctx, ggml_transpose(ctx, kernel_rows));
    ggml_tensor * signal          = ggml_reshape_2d(ctx, x, x->ne[0], in_channels);
    ggml_tensor * signal_by_input = ggml_cont(ctx, ggml_transpose(ctx, signal));
    ggml_tensor * columns         = ggml_mul_mat(ctx, kernel_by_input, signal_by_input);
    return ggml_col2im_1d(ctx, columns, stride, (int) out_channels, padding);
}

} // namespace tts_cpp::detail
