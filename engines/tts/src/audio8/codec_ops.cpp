#include "audio8/codec_ops.h"

#include "audio8/graph.h"

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

const size_t FLOAT = sizeof(float);

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

int length_of(const ggml_tensor * x) {
    return static_cast<int>(x->ne[1]);
}

ggml_tensor * pad_history(ggml_context * ctx, ggml_tensor * x, int left, int right) {
    if (left == 0 && right == 0) return x;
    return ggml_pad_ext(ctx, x, 0, 0, left, right, 0, 0, 0, 0);
}

// The left pad covers the receptive field; the right pad only rounds the
// length up to a whole number of strides, which is what _extra_padding does.
ggml_tensor * pad_for_conv(ggml_context * ctx, ggml_tensor * x, int span, int stride) {
    const int length = length_of(x);
    return pad_history(ctx, x, span - stride, ceil_div(length, stride) * stride - length);
}

// The [in, out] matrix for one tap, a contiguous slice of the tap-major kernel.
ggml_tensor * tap_matrix(ggml_context * ctx, ggml_tensor * kernel, int tap) {
    return ggml_view_2d(ctx, kernel, kernel->ne[0], kernel->ne[1], kernel->nb[1],
                        static_cast<size_t>(tap) * kernel->nb[2]);
}

// The window of the padded signal that tap `tap` multiplies. Stride 1 leaves a
// contiguous slab; a longer stride skips columns, so the matmul gets a copy.
ggml_tensor * tap_window(ggml_context * ctx, ggml_tensor * padded, int tap, int stride,
                         int dilation, int length) {
    const size_t column = static_cast<size_t>(padded->ne[0]) * FLOAT;
    ggml_tensor * window = ggml_view_2d(ctx, padded, padded->ne[0], length,
                                        column * stride,
                                        column * static_cast<size_t>(tap) * dilation);
    return stride == 1 ? window : ggml_cont(ctx, window);
}

ggml_tensor * accumulate(ggml_context * ctx, ggml_tensor * total, ggml_tensor * term) {
    return total ? ggml_add(ctx, total, term) : term;
}

// The optional mixed-precision experiment is confined to large synthesis
// projections. Small/channel-tail products retain the original F32 request.
void synthesis_precision(ggml_tensor * product, const codec_conv_options & options) {
    const auto * weights = product->src[0];
    const bool eligible = weights->type == GGML_TYPE_F16 &&
                          weights->ne[0] >= 64 && weights->ne[0] % 32 == 0 &&
                          weights->ne[1] >= 64 && weights->ne[1] % 32 == 0;
    ggml_mul_mat_set_prec(product, eligible ? options.precision : GGML_PREC_F32);
}

ggml_tensor * sum_taps(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * padded,
                       int taps, int stride, int dilation, int length, const codec_conv_options & options) {
    if (stride == 1 && options.tiled) {
        if (auto * out = options.tiled(ctx, options.backend, kernel, padded, dilation, length)) return out;
    }
    ggml_tensor * total = nullptr;
    for (int tap = 0; tap < taps; ++tap) {
        ggml_tensor * term = ggml_mul_mat(ctx, tap_matrix(ctx, kernel, tap),
                                          tap_window(ctx, padded, tap, stride, dilation,
                                                     length));
        synthesis_precision(term, options);
        total = accumulate(ctx, total, term);
    }
    return total;
}

ggml_tensor * sum_depthwise_taps(ggml_context * ctx, ggml_tensor * kernel,
                                 ggml_tensor * padded, int taps, int length) {
    const size_t column = static_cast<size_t>(padded->ne[0]) * FLOAT;
    ggml_tensor * total = nullptr;
    for (int tap = 0; tap < taps; ++tap) {
        ggml_tensor * lane = ggml_view_1d(ctx, kernel, kernel->ne[0],
                                          static_cast<size_t>(tap) * kernel->nb[1]);
        ggml_tensor * window = ggml_view_2d(ctx, padded, padded->ne[0], length, column,
                                            column * static_cast<size_t>(tap));
        total = accumulate(ctx, total, ggml_mul(ctx, window, lane));
    }
    return total;
}

// One stride's worth of a transposed kernel's taps, flattened so a single
// matmul covers every output phase that draws on the same input column.
ggml_tensor * tap_group(ggml_context * ctx, ggml_tensor * kernel, int stride, int group) {
    const int64_t channels_in = kernel->ne[0];
    const int64_t channels_out = kernel->ne[1];
    ggml_tensor * flat = ggml_reshape_2d(ctx, kernel, channels_in,
                                         channels_out * kernel->ne[2]);
    return ggml_view_2d(ctx, flat, channels_in, channels_out * stride, flat->nb[1],
                        static_cast<size_t>(group) * stride * channels_out * flat->nb[1]);
}

ggml_tensor * delayed(ggml_context * ctx, ggml_tensor * x, int columns) {
    if (columns == 0) return x;
    ggml_tensor * padded = pad_history(ctx, x, columns, /*right=*/0);
    return ggml_view_2d(ctx, padded, padded->ne[0], x->ne[1], padded->nb[1], 0);
}

ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    return bias ? ggml_add(ctx, x, bias) : x;
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight,
                         ggml_tensor * bias, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight), bias);
}

ggml_tensor * scaled_residual(ggml_context * ctx, ggml_tensor * x, ggml_tensor * branch,
                              ggml_tensor * scale) {
    return ggml_add(ctx, x, ggml_mul(ctx, branch, scale));
}

ggml_tensor * attend_block(ggml_context * ctx, const scaled_block_weights & block,
                           ggml_tensor * x, const rope_planes & rope,
                           const attention_shape & shape, ggml_tensor * mask, float eps) {
    ggml_tensor * normed = rms_norm(ctx, x, block.attn.attn_norm, eps);
    ggml_tensor * branch = windowed_attention(ctx, block.attn, normed, rope, shape, mask);
    return scaled_residual(ctx, x, branch, block.attn_scale);
}

ggml_tensor * feed_forward_block(ggml_context * ctx, const scaled_block_weights & block,
                                 ggml_tensor * x, float eps) {
    ggml_tensor * normed = rms_norm(ctx, x, block.ffn_norm, eps);
    ggml_tensor * branch = swiglu(ctx, block.w1, block.w2, block.w3, normed);
    return scaled_residual(ctx, x, branch, block.ffn_scale);
}

attention_shape shape_for(const transformer_spec & spec, int length,
                          bool precise_attention_values) {
    attention_shape shape;
    shape.n_head = spec.n_head;
    shape.n_kv = spec.n_kv;
    shape.head_dim = spec.head_dim;
    shape.width = length;
    shape.precise_values = precise_attention_values;
    return shape;
}

ggml_tensor * residual_block(ggml_context * ctx, const residual_unit & unit, ggml_tensor * x,
                             int dilation, float snake_epsilon, const codec_conv_options & options) {
    ggml_tensor * branch = snake(ctx, x, unit.alpha1, snake_epsilon);
    branch = causal_conv(ctx, unit.conv1, branch, /*stride=*/1, dilation, options);
    branch = snake(ctx, branch, unit.alpha2, snake_epsilon);
    branch = causal_conv(ctx, unit.conv2, branch, /*stride=*/1, /*dilation=*/1, options);
    return ggml_add(ctx, x, branch);
}

}  // namespace

int conv_taps(const conv_weights & conv) {
    return static_cast<int>(conv.w->ne[2]);
}

int depthwise_taps(const conv_weights & conv) {
    return static_cast<int>(conv.w->ne[1]);
}

int span_through_conv(int span, int taps, int stride, int dilation) {
    return (span - 1) * stride + (taps - 1) * dilation + 1;
}

// Output column i * stride + phase reads input column i - g for every whole
// stride g the kernel spans, so a run of output columns covers that many input
// columns once the run's own two ends are rounded outwards.
int span_through_transpose(int span, int taps, int stride) {
    return ceil_div(span, stride) + taps / stride;
}

int span_through_convnext(int span, const convnext_weights & block) {
    span = span_through_conv(span, conv_taps(block.pwconv2), 1, 1);
    span = span_through_conv(span, conv_taps(block.pwconv1), 1, 1);
    return span_through_conv(span, depthwise_taps(block.dwconv), 1, 1);
}

int span_through_units(int span, const std::vector<residual_unit> & units,
                       const std::vector<int> & dilations) {
    for (size_t index = units.size(); index-- > 0;) {
        span = span_through_conv(span, conv_taps(units[index].conv2), 1, 1);
        span = span_through_conv(span, conv_taps(units[index].conv1), 1, dilations[index]);
    }
    return span;
}

ggml_tensor * causal_conv(ggml_context * ctx, const conv_weights & conv, ggml_tensor * x,
                          int stride, int dilation, const codec_conv_options & options) {
    const int taps = conv_taps(conv);
    const int span = (taps - 1) * dilation + 1;
    const int length = ceil_div(length_of(x), stride);
    ggml_tensor * padded = pad_for_conv(ctx, x, span, stride);
    return add_bias(ctx, sum_taps(ctx, conv.w, padded, taps, stride, dilation, length, options),
                    conv.b);
}

// One kernel per channel, so a tap is an element-wise scale rather than a
// contraction.
ggml_tensor * causal_depthwise_conv(ggml_context * ctx, const conv_weights & conv,
                                    ggml_tensor * x) {
    const int taps = depthwise_taps(conv);
    const int length = length_of(x);
    ggml_tensor * padded = pad_history(ctx, x, taps - 1, 0);
    return add_bias(ctx, sum_depthwise_taps(ctx, conv.w, padded, taps, length), conv.b);
}

// Output column i * stride + phase draws on input column i - g for each whole
// stride g the kernel spans, so the transposed convolution is one matmul per
// span over a progressively delayed signal. The reference's trailing crop is
// what leaves exactly length * stride columns.
ggml_tensor * causal_conv_transpose(ggml_context * ctx, const conv_weights & conv,
                                    ggml_tensor * x, int stride, const codec_conv_options & options) {
    const int taps = conv_taps(conv);
    GGML_ASSERT(taps % stride == 0);
    if (options.tiled && taps / stride > 1) {
        auto * weights = ggml_reshape_3d(ctx, conv.w, conv.w->ne[0],
                                         conv.w->ne[1] * stride, taps / stride);
        auto * input = pad_history(ctx, x, taps / stride - 1, 0);
        if (auto * out = options.tiled(ctx, options.backend, weights, input, -1, length_of(x))) {
            return add_bias(ctx, ggml_reshape_2d(ctx, out, conv.w->ne[1], stride * x->ne[1]), conv.b);
        }
    }
    ggml_tensor * total = nullptr;
    for (int group = 0; group < taps / stride; ++group) {
        ggml_tensor * term = ggml_mul_mat(ctx, tap_group(ctx, conv.w, stride, group),
                                          delayed(ctx, x, group));
        synthesis_precision(term, options);
        total = accumulate(ctx, total, term);
    }
    return add_bias(ctx, ggml_reshape_2d(ctx, total, conv.w->ne[1], stride * x->ne[1]),
                    conv.b);
}

ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float epsilon) {
    ggml_tensor * wave = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, x, alpha)));
    ggml_tensor * floored = ggml_scale_bias(ctx, alpha, 1.0f, epsilon);
    return ggml_add(ctx, x, ggml_div(ctx, wave, floored));
}

ggml_tensor * project(ggml_context * ctx, const conv_weights & conv, ggml_tensor * x) {
    return causal_conv(ctx, conv, x, /*stride=*/1, /*dilation=*/1);
}

ggml_tensor * residual_stack(ggml_context * ctx, const std::vector<residual_unit> & units,
                             const std::vector<int> & dilations, ggml_tensor * x,
                             float snake_epsilon, const codec_conv_options & options) {
    for (size_t index = 0; index < units.size(); ++index) {
        x = residual_block(ctx, units[index], x, dilations[index], snake_epsilon, options);
    }
    return x;
}

ggml_tensor * convnext_block(ggml_context * ctx, const convnext_weights & block,
                             ggml_tensor * x, float eps, const codec_conv_options & options) {
    ggml_tensor * branch = causal_depthwise_conv(ctx, block.dwconv, x);
    branch = layer_norm(ctx, branch, block.norm_w, block.norm_b, eps);
    const auto pointwise = [&](const conv_weights & conv, ggml_tensor * input) {
        auto * product = ggml_mul_mat(ctx, conv.w, input);
        synthesis_precision(product, options);
        return add_bias(ctx, product, conv.b);
    };
    branch = pointwise(block.pwconv1, branch);
    branch = ggml_gelu_erf(ctx, branch);
    branch = pointwise(block.pwconv2, branch);
    return ggml_add(ctx, x, ggml_mul(ctx, branch, block.gamma));
}

ggml_tensor * window_forward(ggml_context * ctx, const window_transformer & transformer,
                             ggml_tensor * x, ggml_tensor * mask,
                             bool precise_attention_values) {
    const transformer_spec & spec = transformer.spec;
    const rope_planes rope = rope_window(ctx, transformer.rope_cos, transformer.rope_sin,
                                         /*first=*/0, length_of(x));
    const attention_shape shape =
        shape_for(spec, length_of(x), precise_attention_values);
    for (const scaled_block_weights & block : transformer.blocks) {
        x = attend_block(ctx, block, x, rope, shape, mask, spec.norm_eps);
        x = feed_forward_block(ctx, block, x, spec.norm_eps);
    }
    return rms_norm(ctx, x, transformer.norm, spec.norm_eps);
}

namespace {

ggml_tensor * attend_block_after(ggml_context * ctx, const scaled_block_weights & block,
                                 ggml_tensor * x, const rope_planes & rope,
                                 const attention_shape & shape, ggml_tensor * mask, float eps,
                                 ggml_tensor * held_keys, ggml_tensor * held_values,
                                 windowed_step & step) {
    ggml_tensor * normed = rms_norm(ctx, x, block.attn.attn_norm, eps);
    step = windowed_attention_after(ctx, block.attn, normed, rope, shape, held_keys,
                                    held_values, mask);
    return scaled_residual(ctx, x, step.out, block.attn_scale);
}

ggml_tensor * held_at(const std::vector<ggml_tensor *> & held, size_t layer) {
    return layer < held.size() ? held[layer] : nullptr;
}

}  // namespace

ggml_tensor * window_forward_after(ggml_context * ctx, const window_transformer & transformer,
                                   ggml_tensor * x, ggml_tensor * mask, int first,
                                   const window_history & held, window_history & reached,
                                   bool precise_attention_values) {
    const transformer_spec & spec = transformer.spec;
    const rope_planes rope = rope_window(ctx, transformer.rope_cos, transformer.rope_sin,
                                         first, length_of(x));
    const attention_shape shape = shape_for(spec, length_of(x), precise_attention_values);
    reached.keys.assign(transformer.blocks.size(), nullptr);
    reached.values.assign(transformer.blocks.size(), nullptr);
    for (size_t layer = 0; layer < transformer.blocks.size(); ++layer) {
        windowed_step step;
        x = attend_block_after(ctx, transformer.blocks[layer], x, rope, shape, mask,
                               spec.norm_eps, held_at(held.keys, layer),
                               held_at(held.values, layer), step);
        x = feed_forward_block(ctx, transformer.blocks[layer], x, spec.norm_eps);
        reached.keys[layer] = step.keys;
        reached.values[layer] = step.values;
    }
    return rms_norm(ctx, x, transformer.norm, spec.norm_eps);
}

std::vector<float> window_mask(const transformer_spec & spec, int length) {
    std::vector<float> mask(static_cast<size_t>(length) * length);
    fill_causal_mask(mask.data(), length, length, /*first_query=*/0, spec.window);
    return mask;
}

std::vector<float> window_mask_after(const transformer_spec & spec, int held, int length) {
    std::vector<float> mask(static_cast<size_t>(held + length) * length);
    fill_causal_mask(mask.data(), held + length, length, /*first_query=*/held, spec.window);
    return mask;
}

bool cancelled(const cancel_hook & cancel, std::string * error) {
    if (!cancel || !cancel()) return false;
    if (error) *error = CANCELLED;
    return true;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
