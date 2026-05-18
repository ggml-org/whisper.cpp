#include "supertonic_internal.h"

#include "ggml-alloc.h"

#if defined(TTS_CPP_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(TTS_CPP_USE_CBLAS)
#include <cblas.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tts_cpp::supertonic::detail {
namespace {

struct f32_tensor { std::vector<float> data; int64_t ne[4] = {1,1,1,1}; };

f32_tensor read_f32(const supertonic_model & m, const std::string & source_name) {
    ggml_tensor * t = require_source_tensor(m, source_name);
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f)); }
inline float mish(float x) { return x * std::tanh(std::log1pf(std::exp(x))); }

bool vector_profile_enabled() {
    static const bool enabled = std::getenv("SUPERTONIC_VECTOR_PROFILE") != nullptr;
    return enabled;
}

struct vector_profile_state {
    int step = -1;
    std::chrono::steady_clock::time_point step_start{};
    std::chrono::steady_clock::time_point last{};
};

vector_profile_state & vector_profile() {
    thread_local vector_profile_state state;
    return state;
}

void profile_vector_step_begin(int step) {
    if (!vector_profile_enabled()) return;
    auto & state = vector_profile();
    state.step = step;
    state.step_start = std::chrono::steady_clock::now();
    state.last = state.step_start;
}

void profile_vector_compute(const supertonic_model & model,
                            ggml_cgraph * graph,
                            int step,
                            const char * island) {
    if (!vector_profile_enabled()) {
        supertonic_graph_compute(model, graph);
        return;
    }
    auto & state = vector_profile();
    const auto t0 = std::chrono::steady_clock::now();
    const double pre_ms = std::chrono::duration<double, std::milli>(t0 - state.last).count();
    supertonic_graph_compute(model, graph);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    state.last = t1;
    std::fprintf(stderr, "supertonic_vector_profile step=%d island=%s pre_ms=%.3f compute_ms=%.3f\n",
                 step, island, pre_ms, ms);
}

void profile_vector_step_end(int step) {
    if (!vector_profile_enabled()) return;
    auto & state = vector_profile();
    const auto now = std::chrono::steady_clock::now();
    const double post_ms = std::chrono::duration<double, std::milli>(now - state.last).count();
    const double total_ms = std::chrono::duration<double, std::milli>(now - state.step_start).count();
    std::fprintf(stderr, "supertonic_vector_profile step=%d island=step_end post_ms=%.3f total_ms=%.3f\n",
                 step, post_ms, total_ms);
}

void dense(const std::vector<float> & x, const f32_tensor & w, const f32_tensor & b,
           int IC, int OC, std::vector<float> & y) {
    y.assign(OC, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        float sum = b.data[oc];
        for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t) oc * IC + ic] * x[ic];
        y[oc] = sum;
    }
}

void dense_matmul_vec(const std::vector<float> & x, const f32_tensor & w, const f32_tensor & b,
                      int IC, int OC, std::vector<float> & y) {
    y.assign(OC, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        float sum = b.data[oc];
        for (int ic = 0; ic < IC; ++ic) sum += x[ic] * w.data[(size_t)ic * OC + oc];
        y[oc] = sum;
    }
}

void dense_matmul_time(const std::vector<float> & x, int L, int IC,
                       const f32_tensor & w, const f32_tensor & b,
                       int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b.data[oc];
            for (int ic = 0; ic < IC; ++ic) sum += x[(size_t)t*IC + ic] * w.data[(size_t)ic*OC + oc];
            y[(size_t)t*OC + oc] = sum;
        }
    }
}

void conv1x1(const std::vector<float> & x, int L, int IC,
             const f32_tensor & w, const f32_tensor * b, int OC,
             std::vector<float> & y) {
    y.assign((size_t)L*OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t)oc*IC + ic] * x[(size_t)t*IC + ic];
            y[(size_t)t*OC + oc] = sum;
        }
    }
}

ggml_tensor * repeat_like(ggml_context * ctx, ggml_tensor * v, ggml_tensor * like) {
    if (ggml_n_dims(v) == 1 && ggml_n_dims(like) >= 2) {
        if (like->ne[0] == v->ne[0]) v = ggml_reshape_2d(ctx, v, v->ne[0], 1);
        else if (like->ne[1] == v->ne[0]) v = ggml_reshape_2d(ctx, v, 1, v->ne[0]);
    }
    if (!ggml_can_repeat(v, like)) {
        throw std::runtime_error(
            "cannot repeat tensor [" + std::to_string(v->ne[0]) + "," + std::to_string(v->ne[1]) + "," +
            std::to_string(v->ne[2]) + "," + std::to_string(v->ne[3]) + "] to [" +
            std::to_string(like->ne[0]) + "," + std::to_string(like->ne[1]) + "," +
            std::to_string(like->ne[2]) + "," + std::to_string(like->ne[3]) + "]");
    }
    return ggml_repeat(ctx, v, like);
}

ggml_tensor * conv1d_f32(ggml_context * ctx,
                         ggml_tensor * kernel,
                         ggml_tensor * input,
                         int stride,
                         int padding,
                         int dilation) {
#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
    if (kernel->ne[0] == 1 && stride == 1 && padding == 0 && dilation == 1 &&
        input->type == GGML_TYPE_F32 && kernel->type == GGML_TYPE_F32 &&
        input->ne[2] == 1 && input->ne[3] == 1) {
        auto pointwise_op = [](ggml_tensor * dst, int ith, int nth, void *) {
            const ggml_tensor * x = dst->src[0];
            const ggml_tensor * w = dst->src[1];
            const int L = (int)x->ne[0];
            const int IC = (int)x->ne[1];
            const int OC = (int)w->ne[2];
            const int oc0 = (OC * ith) / nth;
            const int oc1 = (OC * (ith + 1)) / nth;
            if (oc0 >= oc1) return;
            const float * x_data = static_cast<const float *>(x->data);
            const float * w_data = static_cast<const float *>(w->data);
            float * y_data = static_cast<float *>(dst->data);
            const int lda = (int)(x->nb[1] / sizeof(float));
            const int ldb = (int)(w->nb[2] / sizeof(float));
            const int ldc = (int)(dst->nb[1] / sizeof(float));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        L, oc1 - oc0, IC,
                        1.0f,
                        x_data, lda,
                        w_data + (size_t)oc0 * ldb, ldb,
                        0.0f,
                        y_data + (size_t)oc0 * ldc, ldc);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        };
        ggml_tensor * args[] = { input, kernel };
        return ggml_custom_4d(ctx, GGML_TYPE_F32,
                              input->ne[0], kernel->ne[2], input->ne[2], input->ne[3],
                              args, 2,
                              pointwise_op,
                              GGML_N_TASKS_MAX,
                              nullptr);
    }
#endif
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

ggml_tensor * edge_clamp_pad_1d(ggml_context * ctx, ggml_tensor * x, int pad_left, int pad_right) {
    const int64_t L = x->ne[0];
    const int64_t C = x->ne[1];
    ggml_tensor * out = x;
    if (pad_left > 0) {
        ggml_tensor * first = ggml_view_2d(ctx, x, 1, C, x->nb[1], 0);
        ggml_tensor * rep = ggml_repeat_4d(ctx, first, pad_left, C, 1, 1);
        out = ggml_concat(ctx, rep, out, 0);
    }
    if (pad_right > 0) {
        ggml_tensor * last = ggml_view_2d(ctx, x, 1, C, x->nb[1], (size_t)(L - 1) * x->nb[0]);
        ggml_tensor * rep = ggml_repeat_4d(ctx, last, pad_right, C, 1, 1);
        out = ggml_concat(ctx, out, rep, 0);
    }
    return out;
}

struct depthwise_same_op_config {
    int dilation = 1;
};

const depthwise_same_op_config * depthwise_same_config(int dilation) {
    static const depthwise_same_op_config d1{1};
    static const depthwise_same_op_config d2{2};
    static const depthwise_same_op_config d4{4};
    static const depthwise_same_op_config d8{8};
    switch (dilation) {
        case 1: return &d1;
        case 2: return &d2;
        case 4: return &d4;
        case 8: return &d8;
        default: return nullptr;
    }
}

void depthwise_same_custom_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * cfg = static_cast<const depthwise_same_op_config *>(userdata);
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const ggml_tensor * b = dst->src[2];
    const int L = (int)x->ne[0];
    const int C = (int)x->ne[1];
    const int K = (int)w->ne[0];
    const int dilation = cfg ? cfg->dilation : 1;
    const int pad_left = ((K - 1) * dilation) / 2;
    const int c0 = (C * ith) / nth;
    const int c1 = (C * (ith + 1)) / nth;

    const auto * x_base = static_cast<const uint8_t *>(x->data);
    const auto * w_base = static_cast<const uint8_t *>(w->data);
    const auto * b_base = static_cast<const uint8_t *>(b->data);
    auto * dst_base = static_cast<uint8_t *>(dst->data);

    for (int c = c0; c < c1; ++c) {
        const float bias = *reinterpret_cast<const float *>(b_base + (size_t)c * b->nb[0]);
        if (K == 5) {
            const float w0 = *reinterpret_cast<const float *>(w_base + (size_t)c * w->nb[2]);
            const float w1 = *reinterpret_cast<const float *>(w_base + w->nb[0] + (size_t)c * w->nb[2]);
            const float w2 = *reinterpret_cast<const float *>(w_base + 2 * w->nb[0] + (size_t)c * w->nb[2]);
            const float w3 = *reinterpret_cast<const float *>(w_base + 3 * w->nb[0] + (size_t)c * w->nb[2]);
            const float w4 = *reinterpret_cast<const float *>(w_base + 4 * w->nb[0] + (size_t)c * w->nb[2]);
            for (int t = 0; t < L; ++t) {
                const int s0 = std::max(0, t - 2 * dilation);
                const int s1 = std::max(0, t - dilation);
                const int s2 = t;
                const int s3 = std::min(L - 1, t + dilation);
                const int s4 = std::min(L - 1, t + 2 * dilation);
                const float x0 = *reinterpret_cast<const float *>(x_base + (size_t)s0 * x->nb[0] + (size_t)c * x->nb[1]);
                const float x1 = *reinterpret_cast<const float *>(x_base + (size_t)s1 * x->nb[0] + (size_t)c * x->nb[1]);
                const float x2 = *reinterpret_cast<const float *>(x_base + (size_t)s2 * x->nb[0] + (size_t)c * x->nb[1]);
                const float x3 = *reinterpret_cast<const float *>(x_base + (size_t)s3 * x->nb[0] + (size_t)c * x->nb[1]);
                const float x4 = *reinterpret_cast<const float *>(x_base + (size_t)s4 * x->nb[0] + (size_t)c * x->nb[1]);
                const float sum = bias + x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4;
                *reinterpret_cast<float *>(dst_base + (size_t)t * dst->nb[0] + (size_t)c * dst->nb[1]) = sum;
            }
            continue;
        }
        for (int t = 0; t < L; ++t) {
            float sum = bias;
            for (int k = 0; k < K; ++k) {
                int st = t + k * dilation - pad_left;
                st = std::max(0, std::min(L - 1, st));
                const float xv = *reinterpret_cast<const float *>(x_base + (size_t)st * x->nb[0] + (size_t)c * x->nb[1]);
                const float wv = *reinterpret_cast<const float *>(w_base + (size_t)k * w->nb[0] + (size_t)c * w->nb[2]);
                sum += xv * wv;
            }
            *reinterpret_cast<float *>(dst_base + (size_t)t * dst->nb[0] + (size_t)c * dst->nb[1]) = sum;
        }
    }
}

ggml_tensor * depthwise_same_custom_ggml(ggml_context * ctx,
                                         ggml_tensor * x,
                                         ggml_tensor * w,
                                         ggml_tensor * b,
                                         int dilation) {
    const depthwise_same_op_config * cfg = depthwise_same_config(dilation);
    if (!cfg || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return nullptr;
    }
    ggml_tensor * args[] = { x, w, b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32,
                          x->ne[0], x->ne[1], x->ne[2], x->ne[3],
                          args, 3,
                          depthwise_same_custom_op,
                          GGML_N_TASKS_MAX,
                          const_cast<depthwise_same_op_config *>(cfg));
}

ggml_tensor * depthwise_same_ggml(ggml_context * ctx,
                                  ggml_tensor * x,
                                  ggml_tensor * w,
                                  ggml_tensor * b,
                                  int dilation) {
    if (ggml_tensor * custom = depthwise_same_custom_ggml(ctx, x, w, b, dilation)) {
        return custom;
    }
    const int K = (int) w->ne[0];
    const int pad_left = ((K - 1) * dilation) / 2;
    const int pad_right = (K - 1) * dilation - pad_left;
    ggml_tensor * padded = edge_clamp_pad_1d(ctx, x, pad_left, pad_right);
    ggml_tensor * new_b = ggml_reshape_4d(ctx, padded, padded->ne[0], 1, padded->ne[1], padded->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, w, new_b, 1, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, im2col, w);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);
    return ggml_add(ctx, y, repeat_like(ctx, b, y));
}

ggml_tensor * layer_norm_ggml(ggml_context * ctx,
                              ggml_tensor * x,
                              ggml_tensor * g,
                              ggml_tensor * b) {
    if (x->type == GGML_TYPE_F32 && g->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1) {
        auto layer_norm_op = [](ggml_tensor * dst, int ith, int nth, void *) {
            const ggml_tensor * src = dst->src[0];
            const ggml_tensor * gamma = dst->src[1];
            const ggml_tensor * beta = dst->src[2];
            const int L = (int)src->ne[0];
            const int C = (int)src->ne[1];
            const int t0 = (L * ith) / nth;
            const int t1 = (L * (ith + 1)) / nth;
            const auto * src_base = static_cast<const uint8_t *>(src->data);
            const auto * gamma_base = static_cast<const uint8_t *>(gamma->data);
            const auto * beta_base = static_cast<const uint8_t *>(beta->data);
            auto * dst_base = static_cast<uint8_t *>(dst->data);
            for (int t = t0; t < t1; ++t) {
                double mean = 0.0;
                for (int c = 0; c < C; ++c) {
                    mean += *reinterpret_cast<const float *>(src_base + (size_t)t * src->nb[0] + (size_t)c * src->nb[1]);
                }
                mean /= (double)C;
                double var = 0.0;
                for (int c = 0; c < C; ++c) {
                    const float v = *reinterpret_cast<const float *>(src_base + (size_t)t * src->nb[0] + (size_t)c * src->nb[1]);
                    const double d = (double)v - mean;
                    var += d * d;
                }
                const float inv = 1.0f / std::sqrt((float)(var / (double)C) + 1e-6f);
                for (int c = 0; c < C; ++c) {
                    const float v = *reinterpret_cast<const float *>(src_base + (size_t)t * src->nb[0] + (size_t)c * src->nb[1]);
                    const float gv = *reinterpret_cast<const float *>(gamma_base + (size_t)c * gamma->nb[0]);
                    const float bv = *reinterpret_cast<const float *>(beta_base + (size_t)c * beta->nb[0]);
                    const float y = ((v - (float)mean) * inv) * gv + bv;
                    *reinterpret_cast<float *>(dst_base + (size_t)t * dst->nb[0] + (size_t)c * dst->nb[1]) = y;
                }
            }
        };
        ggml_tensor * args[] = { x, g, b };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3],
                              args, 3, layer_norm_op, GGML_N_TASKS_MAX, nullptr);
    }
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    xt = ggml_norm(ctx, xt, 1e-6f);
    xt = ggml_mul(ctx, xt, repeat_like(ctx, g, xt));
    xt = ggml_add(ctx, xt, repeat_like(ctx, b, xt));
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}

ggml_tensor * dense_matmul_time_ggml(ggml_context * ctx,
                                     ggml_tensor * x,
                                     ggml_tensor * w,
                                     ggml_tensor * b) {
#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
    if (x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && (!b || b->type == GGML_TYPE_F32) &&
        x->ne[2] == 1 && x->ne[3] == 1 && w->ne[1] == x->ne[1]) {
        auto dense_op = [](ggml_tensor * dst, int ith, int nth, void *) {
            const ggml_tensor * src = dst->src[0];
            const ggml_tensor * weight = dst->src[1];
            const ggml_tensor * bias = dst->src[2];
            const int L = (int)src->ne[0];
            const int IC = (int)src->ne[1];
            const int OC = (int)weight->ne[0];
            const int oc0 = (OC * ith) / nth;
            const int oc1 = (OC * (ith + 1)) / nth;
            if (oc0 >= oc1) return;
            const float * src_data = static_cast<const float *>(src->data);
            const float * weight_data = static_cast<const float *>(weight->data);
            float * dst_data = static_cast<float *>(dst->data);
            const int lda = (int)(src->nb[1] / sizeof(float));
            const int ldb = (int)(weight->nb[1] / sizeof(float));
            const int ldc = (int)(dst->nb[1] / sizeof(float));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        L, oc1 - oc0, IC,
                        1.0f,
                        src_data, lda,
                        weight_data + oc0, ldb,
                        0.0f,
                        dst_data + (size_t)oc0 * ldc, ldc);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
            if (bias) {
                const auto * bias_base = static_cast<const uint8_t *>(bias->data);
                for (int oc = oc0; oc < oc1; ++oc) {
                    const float bv = *reinterpret_cast<const float *>(bias_base + (size_t)oc * bias->nb[0]);
                    float * col = dst_data + (size_t)oc * ldc;
                    for (int t = 0; t < L; ++t) col[t] += bv;
                }
            }
        };
        ggml_tensor * args_with_bias[] = { x, w, b };
        ggml_tensor * args_no_bias[] = { x, w };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], w->ne[0], x->ne[2], x->ne[3],
                              b ? args_with_bias : args_no_bias,
                              b ? 3 : 2,
                              dense_op,
                              GGML_N_TASKS_MAX,
                              nullptr);
    }
#endif
    // Raw ONNX MatMul weights are [IC, OC] in row-major order, while GGML
    // tensors are loaded as ne=[OC, IC].  Make that transpose contiguous, then
    // view it as a Conv1d kernel [K=1, IC, OC] so it can consume the repo's
    // standard time-major activation layout [T, IC].
    ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, w));
    ggml_tensor * kernel = ggml_reshape_3d(ctx, wt, 1, w->ne[1], w->ne[0]);
    ggml_tensor * y = conv1d_f32(ctx, kernel, x, 1, 0, 1);
    if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
    return y;
}

ggml_tensor * bias_gelu_ggml(ggml_context * ctx, ggml_tensor * x, ggml_tensor * b) {
    if (x->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && x->ne[2] == 1 && x->ne[3] == 1) {
        auto op = [](ggml_tensor * dst, int ith, int nth, void *) {
            const ggml_tensor * src = dst->src[0];
            const ggml_tensor * bias = dst->src[1];
            const int L = (int)src->ne[0];
            const int C = (int)src->ne[1];
            const int c0 = (C * ith) / nth;
            const int c1 = (C * (ith + 1)) / nth;
            const auto * src_base = static_cast<const uint8_t *>(src->data);
            const auto * bias_base = static_cast<const uint8_t *>(bias->data);
            auto * dst_base = static_cast<uint8_t *>(dst->data);
            for (int c = c0; c < c1; ++c) {
                const float bv = *reinterpret_cast<const float *>(bias_base + (size_t)c * bias->nb[0]);
                for (int t = 0; t < L; ++t) {
                    const float v = *reinterpret_cast<const float *>(src_base + (size_t)t * src->nb[0] + (size_t)c * src->nb[1]) + bv;
                    const float y = 0.5f * v * (1.0f + std::erff(v * 0.7071067811865475f));
                    *reinterpret_cast<float *>(dst_base + (size_t)t * dst->nb[0] + (size_t)c * dst->nb[1]) = y;
                }
            }
        };
        ggml_tensor * args[] = { x, b };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3],
                              args, 2, op, GGML_N_TASKS_MAX, nullptr);
    }
    return ggml_gelu_erf(ctx, ggml_add(ctx, x, repeat_like(ctx, b, x)));
}

ggml_tensor * pw2_residual_ggml(ggml_context * ctx,
                                ggml_tensor * residual,
                                ggml_tensor * x,
                                ggml_tensor * b,
                                ggml_tensor * gamma) {
    if (residual->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 &&
        b->type == GGML_TYPE_F32 && gamma->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1) {
        auto op = [](ggml_tensor * dst, int ith, int nth, void *) {
            const ggml_tensor * residual = dst->src[0];
            const ggml_tensor * src = dst->src[1];
            const ggml_tensor * bias = dst->src[2];
            const ggml_tensor * gamma = dst->src[3];
            const int L = (int)src->ne[0];
            const int C = (int)src->ne[1];
            const int c0 = (C * ith) / nth;
            const int c1 = (C * (ith + 1)) / nth;
            const auto * res_base = static_cast<const uint8_t *>(residual->data);
            const auto * src_base = static_cast<const uint8_t *>(src->data);
            const auto * bias_base = static_cast<const uint8_t *>(bias->data);
            const auto * gamma_base = static_cast<const uint8_t *>(gamma->data);
            auto * dst_base = static_cast<uint8_t *>(dst->data);
            for (int c = c0; c < c1; ++c) {
                const float bv = *reinterpret_cast<const float *>(bias_base + (size_t)c * bias->nb[0]);
                const float gv = *reinterpret_cast<const float *>(gamma_base + (size_t)c * gamma->nb[0]);
                for (int t = 0; t < L; ++t) {
                    const float rv = *reinterpret_cast<const float *>(res_base + (size_t)t * residual->nb[0] + (size_t)c * residual->nb[1]);
                    const float xv = *reinterpret_cast<const float *>(src_base + (size_t)t * src->nb[0] + (size_t)c * src->nb[1]);
                    const float y = rv + gv * (xv + bv);
                    *reinterpret_cast<float *>(dst_base + (size_t)t * dst->nb[0] + (size_t)c * dst->nb[1]) = y;
                }
            }
        };
        ggml_tensor * args[] = { residual, x, b, gamma };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3],
                              args, 4, op, GGML_N_TASKS_MAX, nullptr);
    }
    x = ggml_add(ctx, x, repeat_like(ctx, b, x));
    x = ggml_mul(ctx, x, repeat_like(ctx, gamma, x));
    return ggml_add(ctx, residual, x);
}

ggml_tensor * vector_convnext_ggml(ggml_context * ctx,
                                   const supertonic_model & model,
                                   const std::string & p,
                                   ggml_tensor * x,
                                   int dilation) {
    ggml_tensor * residual = x;
    ggml_tensor * y = depthwise_same_ggml(ctx, x,
        require_source_tensor(model, p + ".dwconv.weight"),
        require_source_tensor(model, p + ".dwconv.bias"),
        dilation);
    y = layer_norm_ggml(ctx, y,
        require_source_tensor(model, p + ".norm.norm.weight"),
        require_source_tensor(model, p + ".norm.norm.bias"));
    y = conv1d_f32(ctx, require_source_tensor(model, p + ".pwconv1.weight"), y, 1, 0, 1);
    y = bias_gelu_ggml(ctx, y, require_source_tensor(model, p + ".pwconv1.bias"));
    y = conv1d_f32(ctx, require_source_tensor(model, p + ".pwconv2.weight"), y, 1, 0, 1);
    return pw2_residual_ggml(ctx, residual, y,
        require_source_tensor(model, p + ".pwconv2.bias"),
        require_source_tensor(model, p + ".gamma"));
}

std::vector<float> tensor_to_time_channel(ggml_tensor * t) {
    const int L = (int) t->ne[0];
    const int C = (int) t->ne[1];
    std::vector<float> raw((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
    std::vector<float> out((size_t) L * C);
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < L; ++i) {
            out[(size_t) i * C + c] = raw[(size_t) c * L + i];
        }
    }
    return out;
}

std::vector<float> tensor_raw_f32(ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

std::vector<float> pack_time_channel_for_ggml(const std::vector<float> & x, int L, int C) {
    std::vector<float> out((size_t)L * C);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            out[(size_t)c * L + t] = x[(size_t)t * C + c];
        }
    }
    return out;
}

struct vector_static_layout_cache {
    const float * text_emb = nullptr;
    int text_len = 0;
    uint64_t text_generation_id = 0;
    std::vector<float> text_lc_host;

    const float * style_ttl = nullptr;
    const supertonic_model * model = nullptr;
    uint64_t style_generation_id = 0;
    std::vector<float> style_v_raw;
    std::vector<float> kctx_raw;
};

void cached_style_layouts(const supertonic_model & model,
                          const float * style_ttl,
                          const std::vector<float> *& style_v_raw,
                          const std::vector<float> *& kctx_raw) {
    thread_local vector_static_layout_cache cache;
    if (cache.style_ttl != style_ttl || cache.model != &model ||
        cache.style_generation_id != model.generation_id) {
        cache.style_ttl = style_ttl;
        cache.model = &model;
        cache.style_generation_id = model.generation_id;
        cache.style_v_raw.assign((size_t)50 * 256, 0.0f);
        cache.kctx_raw.assign((size_t)50 * 256, 0.0f);
        auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
        for (int c = 0; c < 256; ++c) {
            for (int t = 0; t < 50; ++t) {
                cache.style_v_raw[(size_t)c * 50 + t] = style_ttl[(size_t)t * 256 + c];
                cache.kctx_raw[(size_t)c * 50 + t] = kconst.data[(size_t)t * 256 + c];
            }
        }
    }
    style_v_raw = &cache.style_v_raw;
    kctx_raw = &cache.kctx_raw;
}

struct vector_text_attention_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int q_len = 0;
    int kv_len = 0;
    int n_heads = 0;
    int head_dim = 0;
    std::string out_w_source;
    std::string out_b_source;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * q_tc_in = nullptr;
    ggml_tensor * k_tc_in = nullptr;
    ggml_tensor * v_tc_in = nullptr;
};

void free_text_attention_cache(vector_text_attention_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

void build_text_attention_cache(vector_text_attention_cache & cache,
                                const supertonic_model & model,
                                int q_len,
                                int kv_len,
                                int n_heads,
                                int head_dim,
                                const std::string & out_w_source,
                                const std::string & out_b_source) {
    free_text_attention_cache(cache);
    cache.model = &model;
    cache.generation_id = model.generation_id;
    cache.q_len = q_len;
    cache.kv_len = kv_len;
    cache.n_heads = n_heads;
    cache.head_dim = head_dim;
    cache.out_w_source = out_w_source;
    cache.out_b_source = out_b_source;

    constexpr int NODES = 256;
    const size_t buf_size = ggml_tensor_overhead() * NODES + ggml_graph_overhead_custom(NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params p = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(p);
    cache.gf = ggml_new_graph_custom(cache.ctx, NODES, false);

    const int width = n_heads * head_dim;
    cache.q_tc_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, width, q_len);
    ggml_set_name(cache.q_tc_in, "vector_attn_q_tc"); ggml_set_input(cache.q_tc_in);
    cache.k_tc_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, width, kv_len);
    ggml_set_name(cache.k_tc_in, "vector_attn_k_tc"); ggml_set_input(cache.k_tc_in);
    cache.v_tc_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, width, kv_len);
    ggml_set_name(cache.v_tc_in, "vector_attn_v_tc"); ggml_set_input(cache.v_tc_in);

    const size_t time_stride = (size_t)width * sizeof(float);
    const size_t head_stride = (size_t)head_dim * sizeof(float);
    ggml_tensor * q_in = ggml_view_3d(cache.ctx, cache.q_tc_in,
        head_dim, q_len, n_heads, time_stride, head_stride, 0);
    ggml_tensor * k_in = ggml_view_3d(cache.ctx, cache.k_tc_in,
        head_dim, kv_len, n_heads, time_stride, head_stride, 0);
    ggml_tensor * v_in = ggml_view_3d(cache.ctx, cache.v_tc_in,
        head_dim, kv_len, n_heads, time_stride, head_stride, 0);

    ggml_tensor * attn = ggml_flash_attn_ext(cache.ctx, q_in, k_in, v_in,
                                             nullptr, 1.0f/16.0f, 0.0f, 0.0f);
    attn = ggml_reshape_2d(cache.ctx, attn, n_heads * head_dim, q_len);
    ggml_tensor * ctx_tc = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, attn));
    ggml_set_name(ctx_tc, "vector_attn_ctx"); ggml_set_output(ctx_tc);
    ggml_build_forward_expand(cache.gf, ctx_tc);

    ggml_tensor * out = dense_matmul_time_ggml(cache.ctx, ctx_tc,
        require_source_tensor(model, out_w_source),
        require_source_tensor(model, out_b_source));
    ggml_set_name(out, "vector_attn_out"); ggml_set_output(out);
    ggml_build_forward_expand(cache.gf, out);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new vector text attention cache failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve vector text attention cache failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

std::vector<float> run_text_attention_cache(vector_text_attention_cache & cache,
                                            const supertonic_model & model,
                                            const std::vector<float> & q_tc,
                                            const std::vector<float> & k_tc,
                                            const std::vector<float> & v_tc,
                                            int q_len,
                                            int kv_len,
                                            int n_heads,
                                            int head_dim,
                                            const std::string & out_w_source,
                                            const std::string & out_b_source,
                                            int current_step,
                                            const char * island,
                                            std::vector<float> * ctx_trace) {
    if (cache.model != &model || cache.generation_id != model.generation_id ||
        cache.q_len != q_len || cache.kv_len != kv_len ||
        cache.n_heads != n_heads || cache.head_dim != head_dim ||
        cache.out_w_source != out_w_source || cache.out_b_source != out_b_source) {
        build_text_attention_cache(cache, model, q_len, kv_len, n_heads, head_dim, out_w_source, out_b_source);
    }
    ggml_backend_tensor_set(cache.q_tc_in, q_tc.data(), 0, q_tc.size()*sizeof(float));
    ggml_backend_tensor_set(cache.k_tc_in, k_tc.data(), 0, k_tc.size()*sizeof(float));
    ggml_backend_tensor_set(cache.v_tc_in, v_tc.data(), 0, v_tc.size()*sizeof(float));
    profile_vector_compute(model, cache.gf, current_step, island);
    if (ctx_trace) *ctx_trace = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "vector_attn_ctx"));
    return tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "vector_attn_out"));
}

void push_trace(std::vector<supertonic_trace_tensor> & trace,
                const std::string & name,
                int L,
                int C,
                const std::vector<float> & data);

struct vector_group_graph_result {
    std::vector<float> post;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
};

struct vector_group_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int L = 0;
    int C = 0;
    int text_len = 0;
    int group = 0;
    int conv_block = 0;
    int linear_block = 0;
    int post_block = 0;
    bool trace_outputs = false;
    std::string matmul_source;
    std::string q_matmul_source;
    std::string k_matmul_source;
    std::string v_matmul_source;
    std::string q_name;
    std::string k_name;
    std::string v_name;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * temb_in = nullptr;
    ggml_tensor * text_in = nullptr;
};

void free_group_graph_cache(vector_group_graph_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

std::string vector_main_block(int index) {
    return "vector_estimator:tts.ttl.vector_field.main_blocks." + std::to_string(index);
}

void build_group_graph_cache(vector_group_graph_cache & cache,
                             const supertonic_model & model,
                             int L,
                             int C,
                             int group,
                             int conv_block,
                             int linear_block,
                             const std::string & matmul_source,
                             int post_block,
                             int text_len,
                             const std::string & q_matmul_source,
                             const std::string & k_matmul_source,
                             const std::string & v_matmul_source,
                             const std::string & q_name,
                             const std::string & k_name,
                             const std::string & v_name,
                             bool trace_outputs) {
    free_group_graph_cache(cache);
    cache.model = &model;
    cache.generation_id = model.generation_id;
    cache.L = L;
    cache.C = C;
    cache.text_len = text_len;
    cache.group = group;
    cache.conv_block = conv_block;
    cache.linear_block = linear_block;
    cache.post_block = post_block;
    cache.trace_outputs = trace_outputs;
    cache.matmul_source = matmul_source;
    cache.q_matmul_source = q_matmul_source;
    cache.k_matmul_source = k_matmul_source;
    cache.v_matmul_source = v_matmul_source;
    cache.q_name = q_name;
    cache.k_name = k_name;
    cache.v_name = v_name;

    constexpr int NODES = 512;
    const size_t buf_size = ggml_tensor_overhead() * NODES + ggml_graph_overhead_custom(NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params p = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(p);
    cache.gf = ggml_new_graph_custom(cache.ctx, NODES, false);

    cache.x_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(cache.x_in, "vector_group_in"); ggml_set_input(cache.x_in);
    cache.temb_in = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, 64);
    ggml_set_name(cache.temb_in, "vector_group_temb"); ggml_set_input(cache.temb_in);
    cache.text_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, text_len, 256);
    ggml_set_name(cache.text_in, "vector_group_text"); ggml_set_input(cache.text_in);

    ggml_tensor * cur = cache.x_in;
    int dils[4] = {1, 2, 4, 8};
    for (int j = 0; j < 4; ++j) {
        cur = vector_convnext_ggml(cache.ctx, model,
            vector_main_block(conv_block) + ".convnext." + std::to_string(j),
            cur, dils[j]);
        if (trace_outputs) {
            const std::string name = "ve_group" + std::to_string(group) + "_convnext" + std::to_string(j);
            ggml_set_name(cur, name.c_str()); ggml_set_output(cur);
            ggml_build_forward_expand(cache.gf, cur);
        }
    }
    ggml_tensor * t_proj = ggml_mul_mat(cache.ctx,
        ggml_cont(cache.ctx, ggml_transpose(cache.ctx, require_source_tensor(model, matmul_source))),
        ggml_reshape_2d(cache.ctx, cache.temb_in, 64, 1));
    t_proj = ggml_add(cache.ctx, t_proj,
        ggml_reshape_2d(cache.ctx,
            require_source_tensor(model, vector_main_block(linear_block) + ".linear.linear.bias"),
            C, 1));
    cur = ggml_add(cache.ctx, cur, repeat_like(cache.ctx, t_proj, cur));
    if (trace_outputs) {
        const std::string time_name = "ve_group" + std::to_string(group) + "_time_add";
        ggml_set_name(cur, time_name.c_str()); ggml_set_output(cur);
        ggml_build_forward_expand(cache.gf, cur);
    }
    cur = vector_convnext_ggml(cache.ctx, model,
        vector_main_block(post_block) + ".convnext.0",
        cur, 1);
    const std::string post_name = "ve_group" + std::to_string(group) + "_block" +
        std::to_string(post_block) + "_convnext0";
    ggml_set_name(cur, post_name.c_str()); ggml_set_output(cur);
    ggml_build_forward_expand(cache.gf, cur);

    const std::string attn_prefix = vector_main_block(post_block + 1) + ".attn.";
    ggml_tensor * q = dense_matmul_time_ggml(cache.ctx, cur,
        require_source_tensor(model, q_matmul_source),
        require_source_tensor(model, attn_prefix + "W_query.linear.bias"));
    ggml_tensor * k = dense_matmul_time_ggml(cache.ctx, cache.text_in,
        require_source_tensor(model, k_matmul_source),
        require_source_tensor(model, attn_prefix + "W_key.linear.bias"));
    ggml_tensor * v = dense_matmul_time_ggml(cache.ctx, cache.text_in,
        require_source_tensor(model, v_matmul_source),
        require_source_tensor(model, attn_prefix + "W_value.linear.bias"));
    ggml_set_name(q, q_name.c_str()); ggml_set_output(q); ggml_build_forward_expand(cache.gf, q);
    ggml_set_name(k, k_name.c_str()); ggml_set_output(k); ggml_build_forward_expand(cache.gf, k);
    ggml_set_name(v, v_name.c_str()); ggml_set_output(v); ggml_build_forward_expand(cache.gf, v);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new vector group cache failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve vector group cache failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

vector_group_graph_result run_group_graph_cache(vector_group_graph_cache & cache,
                                                const supertonic_model & model,
                                                const std::vector<float> & x_tc,
                                                int L,
                                                int C,
                                                const std::vector<float> & temb,
                                                const float * text_lc_host,
                                                int text_len,
                                                int current_step,
                                                int group,
                                                int conv_block,
                                                int linear_block,
                                                const std::string & matmul_source,
                                                int post_block,
                                                const std::string & q_matmul_source,
                                                const std::string & k_matmul_source,
                                                const std::string & v_matmul_source,
                                                const std::string & q_name,
                                                const std::string & k_name,
                                                const std::string & v_name,
                                                const char * island,
                                                std::vector<supertonic_trace_tensor> * trace) {
    if (cache.model != &model || cache.generation_id != model.generation_id ||
        cache.L != L || cache.C != C || cache.text_len != text_len ||
        cache.group != group || cache.conv_block != conv_block ||
        cache.linear_block != linear_block || cache.post_block != post_block ||
        cache.trace_outputs != (trace != nullptr) ||
        cache.matmul_source != matmul_source ||
        cache.q_matmul_source != q_matmul_source || cache.k_matmul_source != k_matmul_source ||
        cache.v_matmul_source != v_matmul_source) {
        build_group_graph_cache(cache, model, L, C, group, conv_block, linear_block, matmul_source, post_block,
                                text_len, q_matmul_source, k_matmul_source, v_matmul_source,
                                q_name, k_name, v_name,
                                trace != nullptr);
    }
    std::vector<float> x_raw = pack_time_channel_for_ggml(x_tc, L, C);
    ggml_backend_tensor_set(cache.x_in, x_raw.data(), 0, x_raw.size()*sizeof(float));
    ggml_backend_tensor_set(cache.temb_in, temb.data(), 0, temb.size()*sizeof(float));
    ggml_backend_tensor_set(cache.text_in, text_lc_host, 0, (size_t) text_len * 256 * sizeof(float));
    profile_vector_compute(model, cache.gf, current_step, island);
    if (trace) {
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_group" + std::to_string(group) + "_convnext" + std::to_string(j);
            push_trace(*trace, name, L, C, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, name.c_str())));
        }
        const std::string time_name = "ve_group" + std::to_string(group) + "_time_add";
        push_trace(*trace, time_name, L, C, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, time_name.c_str())));
    }
    const std::string post_name = "ve_group" + std::to_string(group) + "_block" +
        std::to_string(post_block) + "_convnext0";
    vector_group_graph_result out;
    out.post = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, post_name.c_str()));
    out.q = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, q_name.c_str()));
    out.k = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, k_name.c_str()));
    out.v = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, v_name.c_str()));
    if (trace) {
        push_trace(*trace, post_name, L, C, out.post);
        push_trace(*trace, q_name, L, 256, out.q);
        push_trace(*trace, k_name, text_len, 256, out.k);
        push_trace(*trace, v_name, text_len, 256, out.v);
    }
    return out;
}

struct vector_res_style_qkv_result {
    std::vector<float> post;
    std::vector<float> sq;
    std::vector<float> sk;
    std::vector<float> sv;
};

struct vector_res_style_qkv_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int L = 0;
    int C = 0;
    int norm_block = 0;
    int post_block = 0;
    int style_block = 0;
    bool trace_outputs = false;
    std::string q_matmul_source;
    std::string k_matmul_source;
    std::string v_matmul_source;
    std::string residual_name;
    std::string norm_name;
    std::string post_name;
    std::string q_name;
    std::string k_name;
    std::string v_name;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * lhs_in = nullptr;
    ggml_tensor * rhs_in = nullptr;
    ggml_tensor * style_v_in = nullptr;
    ggml_tensor * kctx_in = nullptr;
};

void free_res_style_qkv_cache(vector_res_style_qkv_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

void build_res_style_qkv_cache(vector_res_style_qkv_cache & cache,
                               const supertonic_model & model,
                               int L,
                               int C,
                               int norm_block,
                               int post_block,
                               int style_block,
                               const std::string & q_matmul_source,
                               const std::string & k_matmul_source,
                               const std::string & v_matmul_source,
                               const std::string & residual_name,
                               const std::string & norm_name,
                               const std::string & post_name,
                               const std::string & q_name,
                               const std::string & k_name,
                               const std::string & v_name,
                               bool trace_outputs) {
    free_res_style_qkv_cache(cache);
    cache.model = &model;
    cache.generation_id = model.generation_id;
    cache.L = L;
    cache.C = C;
    cache.norm_block = norm_block;
    cache.post_block = post_block;
    cache.style_block = style_block;
    cache.trace_outputs = trace_outputs;
    cache.q_matmul_source = q_matmul_source;
    cache.k_matmul_source = k_matmul_source;
    cache.v_matmul_source = v_matmul_source;
    cache.residual_name = residual_name;
    cache.norm_name = norm_name;
    cache.post_name = post_name;
    cache.q_name = q_name;
    cache.k_name = k_name;
    cache.v_name = v_name;

    constexpr int NODES = 512;
    const size_t buf_size = ggml_tensor_overhead() * NODES + ggml_graph_overhead_custom(NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params p = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(p);
    cache.gf = ggml_new_graph_custom(cache.ctx, NODES, false);

    cache.lhs_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(cache.lhs_in, "res_style_lhs"); ggml_set_input(cache.lhs_in);
    cache.rhs_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(cache.rhs_in, "res_style_rhs"); ggml_set_input(cache.rhs_in);
    cache.style_v_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, 50, 256);
    ggml_set_name(cache.style_v_in, "res_style_ttl_lc"); ggml_set_input(cache.style_v_in);
    cache.kctx_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, 50, 256);
    ggml_set_name(cache.kctx_in, "res_style_kctx_lc"); ggml_set_input(cache.kctx_in);

    ggml_tensor * res = ggml_add(cache.ctx, cache.lhs_in, cache.rhs_in);
    ggml_set_name(res, residual_name.c_str());
    if (trace_outputs) {
        ggml_set_output(res);
        ggml_build_forward_expand(cache.gf, res);
    }
    ggml_tensor * norm = layer_norm_ggml(cache.ctx, res,
        require_source_tensor(model, vector_main_block(norm_block) + ".norm.norm.weight"),
        require_source_tensor(model, vector_main_block(norm_block) + ".norm.norm.bias"));
    ggml_set_name(norm, norm_name.c_str());
    if (trace_outputs) {
        ggml_set_output(norm);
        ggml_build_forward_expand(cache.gf, norm);
    }
    ggml_tensor * post = vector_convnext_ggml(cache.ctx, model,
        vector_main_block(post_block) + ".convnext.0",
        norm, 1);
    ggml_set_name(post, post_name.c_str()); ggml_set_output(post);
    ggml_build_forward_expand(cache.gf, post);

    const std::string style_prefix = vector_main_block(style_block) + ".attention.";
    ggml_tensor * sq = dense_matmul_time_ggml(cache.ctx, post,
        require_source_tensor(model, q_matmul_source),
        require_source_tensor(model, style_prefix + "W_query.linear.bias"));
    ggml_tensor * sk = dense_matmul_time_ggml(cache.ctx, cache.kctx_in,
        require_source_tensor(model, k_matmul_source),
        require_source_tensor(model, style_prefix + "W_key.linear.bias"));
    sk = ggml_tanh(cache.ctx, sk);
    ggml_tensor * sv = dense_matmul_time_ggml(cache.ctx, cache.style_v_in,
        require_source_tensor(model, v_matmul_source),
        require_source_tensor(model, style_prefix + "W_value.linear.bias"));
    ggml_set_name(sq, q_name.c_str()); ggml_set_output(sq); ggml_build_forward_expand(cache.gf, sq);
    ggml_set_name(sk, k_name.c_str()); ggml_set_output(sk); ggml_build_forward_expand(cache.gf, sk);
    ggml_set_name(sv, v_name.c_str()); ggml_set_output(sv); ggml_build_forward_expand(cache.gf, sv);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new res-style-qkv failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve res-style-qkv failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

vector_res_style_qkv_result run_res_style_qkv_cache(vector_res_style_qkv_cache & cache,
                                                    const supertonic_model & model,
                                                    const std::vector<float> & lhs_tc,
                                                    const std::vector<float> & rhs_tc,
                                                    int L,
                                                    int C,
                                                    const std::vector<float> & style_v_raw,
                                                    const std::vector<float> & kctx_raw,
                                                    int current_step,
                                                    int norm_block,
                                                    int post_block,
                                                    int style_block,
                                                    const std::string & q_matmul_source,
                                                    const std::string & k_matmul_source,
                                                    const std::string & v_matmul_source,
                                                    const std::string & residual_name,
                                                    const std::string & norm_name,
                                                    const std::string & post_name,
                                                    const std::string & q_name,
                                                    const std::string & k_name,
                                                    const std::string & v_name,
                                                    const char * island,
                                                    std::vector<supertonic_trace_tensor> * trace) {
    const bool want_trace = trace != nullptr;
    if (cache.model != &model || cache.generation_id != model.generation_id ||
        cache.L != L || cache.C != C ||
        cache.norm_block != norm_block || cache.post_block != post_block ||
        cache.style_block != style_block || cache.trace_outputs != want_trace ||
        cache.q_matmul_source != q_matmul_source || cache.k_matmul_source != k_matmul_source ||
        cache.v_matmul_source != v_matmul_source) {
        build_res_style_qkv_cache(cache, model, L, C, norm_block, post_block, style_block,
                                  q_matmul_source, k_matmul_source, v_matmul_source,
                                  residual_name, norm_name, post_name, q_name, k_name, v_name,
                                  want_trace);
    }
    std::vector<float> lhs_raw = pack_time_channel_for_ggml(lhs_tc, L, C);
    std::vector<float> rhs_raw = pack_time_channel_for_ggml(rhs_tc, L, C);
    ggml_backend_tensor_set(cache.lhs_in, lhs_raw.data(), 0, lhs_raw.size() * sizeof(float));
    ggml_backend_tensor_set(cache.rhs_in, rhs_raw.data(), 0, rhs_raw.size() * sizeof(float));
    ggml_backend_tensor_set(cache.style_v_in, style_v_raw.data(), 0, style_v_raw.size() * sizeof(float));
    ggml_backend_tensor_set(cache.kctx_in, kctx_raw.data(), 0, kctx_raw.size() * sizeof(float));
    profile_vector_compute(model, cache.gf, current_step, island);
    if (trace) {
        push_trace(*trace, residual_name, L, C, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, residual_name.c_str())));
        push_trace(*trace, norm_name, L, C, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, norm_name.c_str())));
    }
    vector_res_style_qkv_result out;
    out.post = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, post_name.c_str()));
    out.sq = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, q_name.c_str()));
    out.sk = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, k_name.c_str()));
    out.sv = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, v_name.c_str()));
    if (trace) {
        push_trace(*trace, post_name, L, C, out.post);
        push_trace(*trace, q_name, L, 256, out.sq);
        push_trace(*trace, k_name, 50, 256, out.sk);
        push_trace(*trace, v_name, 50, 256, out.sv);
    }
    return out;
}

struct vector_tail_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int L = 0;
    int C = 0;
    int Cin = 0;
    int total_steps = 0;
    bool trace_outputs = false;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * tail_in = nullptr;
    ggml_tensor * tail_mask = nullptr;
    ggml_tensor * tail_noise = nullptr;
};

#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
void tail_update_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) return;
    const int total_steps = *static_cast<const int *>(userdata);
    const ggml_tensor * tail = dst->src[0];
    const ggml_tensor * mask = dst->src[1];
    const ggml_tensor * noise = dst->src[2];
    const ggml_tensor * weight = dst->src[3];
    const int L = (int)tail->ne[0];
    const int IC = (int)tail->ne[1];
    const int OC = (int)weight->ne[2];
    const float * tail_data = static_cast<const float *>(tail->data);
    const float * weight_data = static_cast<const float *>(weight->data);
    float * dst_data = static_cast<float *>(dst->data);
    const int lda = (int)(tail->nb[1] / sizeof(float));
    const int ldb = (int)(weight->nb[2] / sizeof(float));
    const int ldc = (int)(dst->nb[1] / sizeof(float));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                L, OC, IC,
                1.0f,
                tail_data, lda,
                weight_data, ldb,
                0.0f,
                dst_data, ldc);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    const auto * mask_base = static_cast<const uint8_t *>(mask->data);
    const auto * noise_base = static_cast<const uint8_t *>(noise->data);
    const float step_scale = 1.0f / (float)total_steps;
    for (int c = 0; c < OC; ++c) {
        float * out_col = dst_data + (size_t)c * ldc;
        for (int t = 0; t < L; ++t) {
            const float mv = *reinterpret_cast<const float *>(mask_base + (size_t)t * mask->nb[0]);
            const float nv = *reinterpret_cast<const float *>(noise_base + (size_t)t * noise->nb[0] + (size_t)c * noise->nb[1]);
            out_col[t] = nv + out_col[t] * mv * step_scale;
        }
    }
}
#endif

void free_tail_graph_cache(vector_tail_graph_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

void build_tail_graph_cache(vector_tail_graph_cache & cache,
                            const supertonic_model & model,
                            int L,
                            int C,
                            int Cin,
                            int total_steps,
                            bool trace_outputs) {
    free_tail_graph_cache(cache);
    cache.model = &model;
    cache.generation_id = model.generation_id;
    cache.L = L;
    cache.C = C;
    cache.Cin = Cin;
    cache.total_steps = total_steps;
    cache.trace_outputs = trace_outputs;

    constexpr int NODES = 512;
    const size_t buf_size = ggml_tensor_overhead() * NODES + ggml_graph_overhead_custom(NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params p = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(p);
    cache.gf = ggml_new_graph_custom(cache.ctx, NODES, false);

    cache.tail_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(cache.tail_in, "tail_in"); ggml_set_input(cache.tail_in);
    cache.tail_mask = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, L);
    ggml_set_name(cache.tail_mask, "tail_mask"); ggml_set_input(cache.tail_mask);
    cache.tail_noise = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, Cin);
    ggml_set_name(cache.tail_noise, "tail_noise"); ggml_set_input(cache.tail_noise);
    ggml_tensor * tail = cache.tail_in;
    for (int j = 0; j < 4; ++j) {
        tail = vector_convnext_ggml(cache.ctx, model,
            "vector_estimator:tts.ttl.vector_field.last_convnext.convnext." + std::to_string(j),
            tail, 1);
        if (trace_outputs) {
            const std::string name = "ve_last_convnext" + std::to_string(j);
            ggml_set_name(tail, name.c_str()); ggml_set_output(tail);
            ggml_build_forward_expand(cache.gf, tail);
        }
    }
    ggml_tensor * velocity_t = nullptr;
#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
    if (!trace_outputs) {
        ggml_tensor * args[] = {
            tail,
            cache.tail_mask,
            cache.tail_noise,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_out.net.weight")
        };
        ggml_tensor * next = ggml_custom_4d(cache.ctx, GGML_TYPE_F32, L, Cin, 1, 1,
                                           args, 4, tail_update_op, 1, &cache.total_steps);
        ggml_set_name(next, "ve_next_latent_tc"); ggml_set_output(next);
        ggml_build_forward_expand(cache.gf, next);
    } else
#endif
    {
        velocity_t = conv1d_f32(cache.ctx,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),
            tail, 1, 0, 1);
        velocity_t = ggml_mul(cache.ctx, velocity_t, repeat_like(cache.ctx, cache.tail_mask, velocity_t));
        ggml_set_name(velocity_t, "ve_proj_out"); ggml_set_output(velocity_t);
        ggml_build_forward_expand(cache.gf, velocity_t);
        ggml_tensor * next = ggml_add(cache.ctx, cache.tail_noise,
            ggml_scale(cache.ctx, velocity_t, 1.0f/(float) total_steps));
        ggml_set_name(next, "ve_next_latent_tc"); ggml_set_output(next);
        ggml_build_forward_expand(cache.gf, next);
    }

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new vector tail cache failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve vector tail cache failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

std::vector<float> run_tail_graph_cache(vector_tail_graph_cache & cache,
                                        const supertonic_model & model,
                                        const std::vector<float> & x_tc,
                                        const float * noisy_latent,
                                        const float * latent_mask,
                                        int L,
                                        int C,
                                        int Cin,
                                        int current_step,
                                        int total_steps,
                                        std::vector<supertonic_trace_tensor> * trace) {
    if (cache.model != &model || cache.generation_id != model.generation_id ||
        cache.L != L || cache.C != C ||
        cache.Cin != Cin || cache.total_steps != total_steps ||
        cache.trace_outputs != (trace != nullptr)) {
        build_tail_graph_cache(cache, model, L, C, Cin, total_steps, trace != nullptr);
    }
    std::vector<float> tail_in_raw = pack_time_channel_for_ggml(x_tc, L, C);
    std::vector<float> noise_tc((size_t)L*Cin);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < Cin; ++c) {
            noise_tc[(size_t)t*Cin+c] = noisy_latent[(size_t)c*L+t];
        }
    }
    std::vector<float> noise_raw = pack_time_channel_for_ggml(noise_tc, L, Cin);
    ggml_backend_tensor_set(cache.tail_in, tail_in_raw.data(), 0, tail_in_raw.size()*sizeof(float));
    ggml_backend_tensor_set(cache.tail_mask, latent_mask, 0, (size_t)L*sizeof(float));
    ggml_backend_tensor_set(cache.tail_noise, noise_raw.data(), 0, noise_raw.size()*sizeof(float));
    profile_vector_compute(model, cache.gf, current_step, "tail");
    if (trace) {
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_last_convnext" + std::to_string(j);
            push_trace(*trace, name, L, C, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, name.c_str())));
        }
        push_trace(*trace, "ve_proj_out", L, Cin, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "ve_proj_out")));
    }
    std::vector<float> next_latent_tc = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "ve_next_latent_tc"));
    if (trace) push_trace(*trace, "ve_next_latent_tc", L, Cin, next_latent_tc);
    return next_latent_tc;
}

void push_trace(std::vector<supertonic_trace_tensor> & trace,
                const std::string & name,
                int L,
                int C,
                const std::vector<float> & data) {
    trace.push_back({name, {L, C}, data});
}

void depthwise_same(const std::vector<float> & x, int L, int C, const f32_tensor & w,
                    const f32_tensor & b, int K, int dilation, std::vector<float> & y) {
    y.assign((size_t)L*C, 0.0f);
    int pad_left = ((K - 1) * dilation) / 2;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b.data[c];
            for (int k = 0; k < K; ++k) {
                int st = t + k*dilation - pad_left;
                st = std::max(0, std::min(L - 1, st));
                sum += w.data[(size_t)c*K + k] * x[(size_t)st*C + c];
            }
            y[(size_t)t*C + c] = sum;
        }
    }
}

void layer_norm(std::vector<float> & x, int L, int C, const f32_tensor & g, const f32_tensor & b) {
    for (int t = 0; t < L; ++t) {
        float mean = 0;
        for (int c = 0; c < C; ++c) mean += x[(size_t)t*C+c];
        mean /= (float)C;
        float var = 0;
        for (int c = 0; c < C; ++c) { float d=x[(size_t)t*C+c]-mean; var += d*d; }
        float inv = 1.0f/std::sqrt(var/(float)C + 1e-6f);
        for (int c = 0; c < C; ++c) x[(size_t)t*C+c] = (x[(size_t)t*C+c]-mean)*inv*g.data[c]+b.data[c];
    }
}

void convnext(const supertonic_model & m, const std::string & p, std::vector<float> & x, int L, int C, int dilation) {
    auto dw_w=read_f32(m,p+".dwconv.weight"), dw_b=read_f32(m,p+".dwconv.bias");
    auto ln_g=read_f32(m,p+".norm.norm.weight"), ln_b=read_f32(m,p+".norm.norm.bias");
    auto pw1_w=read_f32(m,p+".pwconv1.weight"), pw1_b=read_f32(m,p+".pwconv1.bias");
    auto pw2_w=read_f32(m,p+".pwconv2.weight"), pw2_b=read_f32(m,p+".pwconv2.bias");
    auto gamma=read_f32(m,p+".gamma");
    std::vector<float> residual=x,y,z;
    depthwise_same(x,L,C,dw_w,dw_b,(int)dw_w.ne[0],dilation,y);
    layer_norm(y,L,C,ln_g,ln_b);
    conv1x1(y,L,C,pw1_w,&pw1_b,(int)pw1_w.ne[2],z);
    for(float &v:z) v=gelu(v);
    conv1x1(z,L,(int)pw1_w.ne[2],pw2_w,&pw2_b,C,y);
    for(size_t i=0;i<x.size();++i){ int c=(int)(i%C); x[i]=residual[i]+gamma.data[c]*y[i]; }
}

std::vector<float> time_embedding(const supertonic_model & m, int current, int total) {
    const int D=64, half=32;
    float t = (float)current / (float)std::max(1,total);
    std::vector<float> emb(D);
    float denom = std::log(10000.0f)/(float)(half-1);
    for(int i=0;i<half;++i){ float f=std::exp((float)i * -denom); float a=t*1000.0f*f; emb[i]=std::sin(a); emb[half+i]=std::cos(a); }
    std::vector<float> h,o;
    dense(emb, read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.0.linear.weight"),
          read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.0.linear.bias"),64,256,h);
    for(float &v:h) v=mish(v);
    dense(h, read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.2.linear.weight"),
          read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.2.linear.bias"),256,64,o);
    return o;
}

void apply_rope(const float * theta, std::vector<float> & x, int L, int H, int D) {
    int half = D/2;
    for(int h=0;h<H;++h) for(int t=0;t<L;++t) for(int d=0;d<half;++d) {
        float angle = ((float)t/(float)L)*theta[d];
        float cs=std::cos(angle), sn=std::sin(angle);
        size_t i1=((size_t)t*H+h)*D+d, i2=((size_t)t*H+h)*D+half+d;
        float a=x[i1], b=x[i2];
        x[i1]=a*cs-b*sn; x[i2]=b*cs+a*sn;
    }
}

void rope_attn(const supertonic_model & m, int group, std::vector<float> & x, int L,
               const float * text_emb, int LT, std::vector<float> & out) {
    static const int qids[4]={3101,3146,3191,3236}, kids[4]={3102,3147,3192,3237}, vids[4]={3103,3148,3193,3238}, oids[4]={3110,3155,3200,3245};
    int C=512, A=256, H=4, D=64;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+3)+".attn.";
    std::vector<float> q,k,v;
    dense_matmul_time(x,L,C,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(qids[group])),read_f32(m,base+"W_query.linear.bias"),A,q);
    std::vector<float> text_lc((size_t)LT*256);
    for(int t=0;t<LT;++t) for(int c=0;c<256;++c) text_lc[(size_t)t*256+c]=text_emb[(size_t)c*LT+t];
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(kids[group])),read_f32(m,base+"W_key.linear.bias"),A,k);
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(vids[group])),read_f32(m,base+"W_value.linear.bias"),A,v);
    auto theta_t = read_f32(m,"vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
    apply_rope(theta_t.data.data(),q,L,H,D); apply_rope(theta_t.data.data(),k,LT,H,D);
    std::vector<float> attn_out((size_t)L*A,0), scores(LT), probs(LT);
    float scale=1.0f/16.0f;
    for(int h=0;h<H;++h) for(int qi=0;qi<L;++qi){
        float mx=-INFINITY;
        for(int kj=0;kj<LT;++kj){ float s=0; for(int d=0;d<D;++d) s+=q[((size_t)qi*H+h)*D+d]*k[((size_t)kj*H+h)*D+d]*scale; scores[kj]=s; mx=std::max(mx,s); }
        float den=0; for(int kj=0;kj<LT;++kj){ probs[kj]=std::exp(scores[kj]-mx); den+=probs[kj]; }
        for(int d=0;d<D;++d){ float sum=0; for(int kj=0;kj<LT;++kj) sum+=(probs[kj]/den)*v[((size_t)kj*H+h)*D+d]; attn_out[(size_t)qi*A+h*D+d]=sum; }
    }
    dense_matmul_time(attn_out,L,A,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(oids[group])),read_f32(m,base+"out_fc.linear.bias"),C,out);
}

void style_attn(const supertonic_model & m, int group, std::vector<float> & x, int L, const float * style_ttl, std::vector<float> & out) {
    static const int qids[4]={3116,3161,3206,3251}, kids[4]={3117,3162,3207,3252}, vids[4]={3118,3163,3208,3253}, oids[4]={3119,3164,3209,3254};
    int C=512,A=256,H=2,D=128,LC=50;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+5)+".attention.";
    std::vector<float> q,k,v,ctx((size_t)LC*256),kctx((size_t)LC*256);
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) ctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
    auto kconst=read_f32(m,"vector_estimator:/Expand_output_0");
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) kctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
    dense_matmul_time(x,L,C,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(qids[group])),read_f32(m,base+"W_query.linear.bias"),A,q);
    dense_matmul_time(kctx,LC,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(kids[group])),read_f32(m,base+"W_key.linear.bias"),A,k);
    for(float &vv:k) vv=std::tanh(vv);
    dense_matmul_time(ctx,LC,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(vids[group])),read_f32(m,base+"W_value.linear.bias"),A,v);
    std::vector<float> merged((size_t)L*A,0), scores(LC), probs(LC); float scale=1.0f/16.0f;
    for(int h=0;h<H;++h) for(int qi=0;qi<L;++qi){
        float mx=-INFINITY;
        for(int kj=0;kj<LC;++kj){ float s=0; for(int d=0;d<D;++d) s+=q[(size_t)qi*A+h*D+d]*k[(size_t)kj*A+h*D+d]*scale; scores[kj]=s; mx=std::max(mx,s); }
        float den=0; for(int kj=0;kj<LC;++kj){ probs[kj]=std::exp(scores[kj]-mx); den+=probs[kj]; }
        for(int d=0;d<D;++d){ float sum=0; for(int kj=0;kj<LC;++kj) sum+=(probs[kj]/den)*v[(size_t)kj*A+h*D+d]; merged[(size_t)qi*A+h*D+d]=sum; }
    }
    dense_matmul_time(merged,L,A,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(oids[group])),read_f32(m,base+"out_fc.linear.bias"),C,out);
}

} // namespace

bool supertonic_vector_step_cpu(const supertonic_model & model, const float * noisy_latent,
                                int latent_len, const float * text_emb, int text_len,
                                const float * style_ttl, const float * latent_mask,
                                int current_step, int total_steps,
                                std::vector<float> & next_latent_out, std::string * error) {
    try {
        int L=latent_len,Cin=144,C=512;
        std::vector<float> in((size_t)L*Cin);
        for(int t=0;t<L;++t) for(int c=0;c<Cin;++c) in[(size_t)t*Cin+c]=noisy_latent[(size_t)c*L+t];
        std::vector<float> x;
        conv1x1(in,L,Cin,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_in.net.weight"),nullptr,C,x);
        for(int t=0;t<L;++t) for(int c=0;c<C;++c) x[(size_t)t*C+c]*=latent_mask[t];
        std::vector<float> te=time_embedding(model,current_step,total_steps);
        static const int time_ids[4]={3095,3140,3185,3230};
        for(int group=0;group<4;++group){
            int ob=group*6;
            int dils[4]={1,2,4,8};
            for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob)+".convnext."+std::to_string(j),x,L,C,dils[j]);
            std::vector<float> tb;
            dense_matmul_vec(te,read_f32(model,"vector_estimator:onnx::MatMul_"+std::to_string(time_ids[group])),
                             read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+1)+".linear.linear.bias"),64,C,tb);
            for(int t=0;t<L;++t) for(int c=0;c<C;++c) x[(size_t)t*C+c]+=tb[c];
            convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+2)+".convnext.0",x,L,C,1);
            std::vector<float> a; rope_attn(model,group,x,L,text_emb,text_len,a);
            for(size_t i=0;i<x.size();++i) x[i]+=a[i];
            layer_norm(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.bias"));
            convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+4)+".convnext.0",x,L,C,1);
            style_attn(model,group,x,L,style_ttl,a);
            for(size_t i=0;i<x.size();++i) x[i]+=a[i];
            layer_norm(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.bias"));
        }
        for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.last_convnext.convnext."+std::to_string(j),x,L,C,1);
        std::vector<float> v;
        conv1x1(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),nullptr,Cin,v);
        next_latent_out.assign((size_t)Cin*L,0.0f);
        for(int c=0;c<Cin;++c) for(int t=0;t<L;++t) {
            float vel=v[(size_t)t*Cin+c]*latent_mask[t];
            next_latent_out[(size_t)c*L+t]=noisy_latent[(size_t)c*L+t]+vel/(float)total_steps;
        }
        if(error) error->clear(); return true;
    } catch(const std::exception &e){ if(error)*error=e.what(); return false; }
}

bool supertonic_vector_trace_proj_ggml(const supertonic_model & model,
                                       const float * noisy_latent,
                                       const float * text_emb,
                                       int text_len,
                                       const float * style_ttl,
                                       const float * latent_mask,
                                       int latent_len,
                                       int current_step,
                                       int total_steps,
                                       std::vector<supertonic_trace_tensor> & scalar_trace,
                                       std::vector<supertonic_trace_tensor> & ggml_trace,
                                       std::string * error,
                                       bool include_scalar_trace,
                                       bool include_ggml_trace,
                                       std::vector<float> * next_latent_tc_out) {
    try {
        scalar_trace.clear();
        ggml_trace.clear();
        const int L = latent_len;
        const int Cin = model.hparams.latent_channels;
        const int C = 512;
#define PUSH_GGML_TRACE(...) do { if (include_ggml_trace) ggml_trace.push_back(supertonic_trace_tensor __VA_ARGS__); } while (0)
        profile_vector_step_begin(current_step);
        std::vector<float> in((size_t) L * Cin);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < Cin; ++c) {
                in[(size_t) t * Cin + c] = noisy_latent[(size_t) c * L + t];
            }
        }

        if (include_scalar_trace) {
            push_trace(scalar_trace, "ve_latent_tc", L, Cin, in);

            std::vector<float> proj;
            f32_tensor proj_w = read_f32(model, "vector_estimator:tts.ttl.vector_field.proj_in.net.weight");
            conv1x1(in, L, Cin, proj_w, nullptr, C, proj);
            for (int t = 0; t < L; ++t) {
                for (int c = 0; c < C; ++c) {
                    proj[(size_t) t * C + c] *= latent_mask[t];
                }
            }
            push_trace(scalar_trace, "ve_masked", L, C, proj);

            std::vector<float> block = proj;
            int dils[4] = {1, 2, 4, 8};
            for (int j = 0; j < 4; ++j) {
                convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext." + std::to_string(j),
                         block, L, C, dils[j]);
                push_trace(scalar_trace, "ve_block0_convnext" + std::to_string(j), L, C, block);
            }

            std::vector<float> te = time_embedding(model, current_step, total_steps);
            std::vector<float> tb;
            dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3095"),
                             read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.1.linear.linear.bias"),
                             64, C, tb);
            for (int t = 0; t < L; ++t) {
                for (int c = 0; c < C; ++c) block[(size_t)t*C+c] += tb[c];
            }
            push_trace(scalar_trace, "ve_time_add0", L, C, block);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.2.convnext.0", block, L, C, 1);
            push_trace(scalar_trace, "ve_block2_convnext0", L, C, block);

            const int A = 256;
            std::string base = "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.";
            std::vector<float> q, k, v;
            dense_matmul_time(block, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3101"),
                              read_f32(model, base + "W_query.linear.bias"), A, q);
            std::vector<float> text_lc((size_t) text_len * 256);
            for (int t = 0; t < text_len; ++t) {
                for (int c = 0; c < 256; ++c) {
                    text_lc[(size_t)t * 256 + c] = text_emb[(size_t)c * text_len + t];
                }
            }
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3102"),
                              read_f32(model, base + "W_key.linear.bias"), A, k);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3103"),
                              read_f32(model, base + "W_value.linear.bias"), A, v);
            push_trace(scalar_trace, "ve_attn0_q", L, A, q);
            push_trace(scalar_trace, "ve_attn0_k", text_len, A, k);
            push_trace(scalar_trace, "ve_attn0_v", text_len, A, v);
            auto theta_t = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta_t.data.data(), q, L, 4, 64);
            apply_rope(theta_t.data.data(), k, text_len, 4, 64);
            push_trace(scalar_trace, "ve_attn0_q_rope", L, A, q);
            push_trace(scalar_trace, "ve_attn0_k_rope", text_len, A, k);

            std::vector<float> attn_ctx((size_t)L*A, 0.0f), scores(text_len), probs(text_len);
            const int H = 4, D = 64;
            const float scale = 1.0f / 16.0f;
            for (int h = 0; h < H; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        s += q[((size_t)qi*H+h)*D+d] * k[((size_t)kj*H+h)*D+d] * scale;
                    }
                    scores[kj] = s;
                    mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) {
                    probs[kj] = std::exp(scores[kj] - mx);
                    den += probs[kj];
                }
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) {
                        sum += (probs[kj] / den) * v[((size_t)kj*H+h)*D+d];
                    }
                    attn_ctx[(size_t)qi*A + h*D + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_attn0_ctx", L, A, attn_ctx);
            std::vector<float> attn_out;
            dense_matmul_time(attn_ctx, L, A, read_f32(model, "vector_estimator:onnx::MatMul_3110"),
                              read_f32(model, base + "out_fc.linear.bias"), C, attn_out);
            push_trace(scalar_trace, "ve_attn0_out", L, C, attn_out);
            std::vector<float> residual = block;
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += attn_out[i];
            push_trace(scalar_trace, "ve_attn0_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.norm.norm.bias"));
            push_trace(scalar_trace, "ve_attn0_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.4.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_block4_convnext0", L, C, residual);

        std::vector<float> style_attn_out;
        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) {
                for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            }
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3116"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3117"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3118"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_style0_q", L, 256, sq);
            push_trace(scalar_trace, "ve_style0_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_style0_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            const int SH=2, SD=128;
            for (int h = 0; h < SH; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < SD; ++d) {
                        score += sq[(size_t)qi*256 + h*SD + d] * sk[(size_t)kj*256 + h*SD + d] * (1.0f/16.0f);
                    }
                    sscores[kj] = score;
                    mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < SD; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*SD + d];
                    smerged[(size_t)qi*256 + h*SD + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_style0_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3119"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.bias"),
                              C, sout);
            push_trace(scalar_trace, "ve_style0_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_style0_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.bias"));
            push_trace(scalar_trace, "ve_style0_norm", L, C, residual);
        }
        (void) style_attn_out;

        int dils_g1[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.6.convnext." + std::to_string(j),
                     residual, L, C, dils_g1[j]);
            push_trace(scalar_trace, "ve_group1_convnext" + std::to_string(j), L, C, residual);
        }
        dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3140"),
                         read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.7.linear.linear.bias"),
                         64, C, tb);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < C; ++c) residual[(size_t)t*C+c] += tb[c];
        }
        push_trace(scalar_trace, "ve_group1_time_add", L, C, residual);
        convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.8.convnext.0", residual, L, C, 1);
        push_trace(scalar_trace, "ve_group1_block8_convnext0", L, C, residual);

        {
            const int A1 = 256;
            std::string base1 = "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.";
            std::vector<float> q1, k1, v1;
            dense_matmul_time(residual, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3146"),
                              read_f32(model, base1 + "W_query.linear.bias"), A1, q1);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3147"),
                              read_f32(model, base1 + "W_key.linear.bias"), A1, k1);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3148"),
                              read_f32(model, base1 + "W_value.linear.bias"), A1, v1);
            push_trace(scalar_trace, "ve_g1_attn_q", L, A1, q1);
            push_trace(scalar_trace, "ve_g1_attn_k", text_len, A1, k1);
            push_trace(scalar_trace, "ve_g1_attn_v", text_len, A1, v1);
            auto theta1 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta1.data.data(), q1, L, 4, 64);
            apply_rope(theta1.data.data(), k1, text_len, 4, 64);
            push_trace(scalar_trace, "ve_g1_attn_q_rope", L, A1, q1);
            push_trace(scalar_trace, "ve_g1_attn_k_rope", text_len, A1, k1);
            std::vector<float> ctx1((size_t)L*A1, 0.0f), scores1(text_len), probs1(text_len);
            for (int h = 0; h < 4; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q1[((size_t)qi*4+h)*64+d] * k1[((size_t)kj*4+h)*64+d] * (1.0f/16.0f);
                    scores1[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs1[kj] = std::exp(scores1[kj]-mx); den += probs1[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs1[kj]/den) * v1[((size_t)kj*4+h)*64+d];
                    ctx1[(size_t)qi*A1 + h*64 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g1_attn_ctx", L, A1, ctx1);
            std::vector<float> out1;
            dense_matmul_time(ctx1, L, A1, read_f32(model, "vector_estimator:onnx::MatMul_3155"),
                              read_f32(model, base1 + "out_fc.linear.bias"), C, out1);
            push_trace(scalar_trace, "ve_g1_attn_out", L, C, out1);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += out1[i];
            push_trace(scalar_trace, "ve_g1_attn_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g1_attn_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.10.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_g1_block10_convnext0", L, C, residual);
        }

        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3161"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3162"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3163"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_g1_style_q", L, 256, sq);
            push_trace(scalar_trace, "ve_g1_style_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_g1_style_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            for (int h = 0; h < 2; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < 128; ++d) score += sq[(size_t)qi*256 + h*128 + d] * sk[(size_t)kj*256 + h*128 + d] * (1.0f/16.0f);
                    sscores[kj] = score; mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < 128; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*128 + d];
                    smerged[(size_t)qi*256 + h*128 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g1_style_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3164"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.out_fc.linear.bias"), C, sout);
            push_trace(scalar_trace, "ve_g1_style_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_g1_style_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g1_style_norm", L, C, residual);
        }

        int dils_g2[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.12.convnext." + std::to_string(j),
                     residual, L, C, dils_g2[j]);
            push_trace(scalar_trace, "ve_group2_convnext" + std::to_string(j), L, C, residual);
        }
        dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3185"),
                         read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.13.linear.linear.bias"),
                         64, C, tb);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < C; ++c) residual[(size_t)t*C+c] += tb[c];
        }
        push_trace(scalar_trace, "ve_group2_time_add", L, C, residual);
        convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.14.convnext.0", residual, L, C, 1);
        push_trace(scalar_trace, "ve_group2_block14_convnext0", L, C, residual);

        {
            const int A2 = 256;
            std::string base2 = "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.";
            std::vector<float> q2, k2, v2;
            dense_matmul_time(residual, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3191"),
                              read_f32(model, base2 + "W_query.linear.bias"), A2, q2);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3192"),
                              read_f32(model, base2 + "W_key.linear.bias"), A2, k2);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3193"),
                              read_f32(model, base2 + "W_value.linear.bias"), A2, v2);
            push_trace(scalar_trace, "ve_g2_attn_q", L, A2, q2);
            push_trace(scalar_trace, "ve_g2_attn_k", text_len, A2, k2);
            push_trace(scalar_trace, "ve_g2_attn_v", text_len, A2, v2);
            auto theta2 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta2.data.data(), q2, L, 4, 64);
            apply_rope(theta2.data.data(), k2, text_len, 4, 64);
            push_trace(scalar_trace, "ve_g2_attn_q_rope", L, A2, q2);
            push_trace(scalar_trace, "ve_g2_attn_k_rope", text_len, A2, k2);
            std::vector<float> ctx2((size_t)L*A2, 0.0f), scores2(text_len), probs2(text_len);
            for (int h = 0; h < 4; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q2[((size_t)qi*4+h)*64+d] * k2[((size_t)kj*4+h)*64+d] * (1.0f/16.0f);
                    scores2[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs2[kj] = std::exp(scores2[kj]-mx); den += probs2[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs2[kj]/den) * v2[((size_t)kj*4+h)*64+d];
                    ctx2[(size_t)qi*A2 + h*64 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g2_attn_ctx", L, A2, ctx2);
            std::vector<float> out2;
            dense_matmul_time(ctx2, L, A2, read_f32(model, "vector_estimator:onnx::MatMul_3200"),
                              read_f32(model, base2 + "out_fc.linear.bias"), C, out2);
            push_trace(scalar_trace, "ve_g2_attn_out", L, C, out2);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += out2[i];
            push_trace(scalar_trace, "ve_g2_attn_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g2_attn_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.16.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_g2_block16_convnext0", L, C, residual);
        }

        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3206"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3207"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3208"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_g2_style_q", L, 256, sq);
            push_trace(scalar_trace, "ve_g2_style_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_g2_style_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            for (int h = 0; h < 2; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < 128; ++d) score += sq[(size_t)qi*256 + h*128 + d] * sk[(size_t)kj*256 + h*128 + d] * (1.0f/16.0f);
                    sscores[kj] = score; mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < 128; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*128 + d];
                    smerged[(size_t)qi*256 + h*128 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g2_style_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3209"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.out_fc.linear.bias"), C, sout);
            push_trace(scalar_trace, "ve_g2_style_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_g2_style_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g2_style_norm", L, C, residual);
        }

        int dils_g3[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.18.convnext." + std::to_string(j),
                     residual, L, C, dils_g3[j]);
            push_trace(scalar_trace, "ve_group3_convnext" + std::to_string(j), L, C, residual);
        }
        dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3230"),
                         read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.19.linear.linear.bias"),
                         64, C, tb);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < C; ++c) residual[(size_t)t*C+c] += tb[c];
        }
        push_trace(scalar_trace, "ve_group3_time_add", L, C, residual);
        convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.20.convnext.0", residual, L, C, 1);
        push_trace(scalar_trace, "ve_group3_block20_convnext0", L, C, residual);

        {
            const int A3 = 256;
            std::string base3 = "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.";
            std::vector<float> q3, k3, v3;
            dense_matmul_time(residual, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3236"),
                              read_f32(model, base3 + "W_query.linear.bias"), A3, q3);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3237"),
                              read_f32(model, base3 + "W_key.linear.bias"), A3, k3);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3238"),
                              read_f32(model, base3 + "W_value.linear.bias"), A3, v3);
            push_trace(scalar_trace, "ve_g3_attn_q", L, A3, q3);
            push_trace(scalar_trace, "ve_g3_attn_k", text_len, A3, k3);
            push_trace(scalar_trace, "ve_g3_attn_v", text_len, A3, v3);
            auto theta3 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta3.data.data(), q3, L, 4, 64);
            apply_rope(theta3.data.data(), k3, text_len, 4, 64);
            push_trace(scalar_trace, "ve_g3_attn_q_rope", L, A3, q3);
            push_trace(scalar_trace, "ve_g3_attn_k_rope", text_len, A3, k3);
            std::vector<float> ctx3((size_t)L*A3, 0.0f), scores3(text_len), probs3(text_len);
            for (int h = 0; h < 4; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q3[((size_t)qi*4+h)*64+d] * k3[((size_t)kj*4+h)*64+d] * (1.0f/16.0f);
                    scores3[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs3[kj] = std::exp(scores3[kj]-mx); den += probs3[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs3[kj]/den) * v3[((size_t)kj*4+h)*64+d];
                    ctx3[(size_t)qi*A3 + h*64 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g3_attn_ctx", L, A3, ctx3);
            std::vector<float> out3;
            dense_matmul_time(ctx3, L, A3, read_f32(model, "vector_estimator:onnx::MatMul_3245"),
                              read_f32(model, base3 + "out_fc.linear.bias"), C, out3);
            push_trace(scalar_trace, "ve_g3_attn_out", L, C, out3);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += out3[i];
            push_trace(scalar_trace, "ve_g3_attn_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g3_attn_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.22.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_g3_block22_convnext0", L, C, residual);
        }

        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3251"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3252"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3253"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_g3_style_q", L, 256, sq);
            push_trace(scalar_trace, "ve_g3_style_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_g3_style_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            for (int h = 0; h < 2; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < 128; ++d) score += sq[(size_t)qi*256 + h*128 + d] * sk[(size_t)kj*256 + h*128 + d] * (1.0f/16.0f);
                    sscores[kj] = score; mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < 128; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*128 + d];
                    smerged[(size_t)qi*256 + h*128 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g3_style_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3254"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.out_fc.linear.bias"), C, sout);
            push_trace(scalar_trace, "ve_g3_style_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_g3_style_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g3_style_norm", L, C, residual);
        }

        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.last_convnext.convnext." + std::to_string(j),
                     residual, L, C, 1);
            push_trace(scalar_trace, "ve_last_convnext" + std::to_string(j), L, C, residual);
        }
        std::vector<float> velocity;
        conv1x1(residual, L, C,
                read_f32(model, "vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),
                nullptr, Cin, velocity);
        push_trace(scalar_trace, "ve_proj_out", L, Cin, velocity);
        std::vector<float> next_latent((size_t)L * Cin);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < Cin; ++c) {
                float vel = velocity[(size_t)t*Cin+c] * latent_mask[t];
                next_latent[(size_t)t*Cin+c] = noisy_latent[(size_t)c*L+t] + vel / 5.0f;
            }
        }
        push_trace(scalar_trace, "ve_next_latent_tc", L, Cin, next_latent);
        }

        constexpr int MAX_NODES = 2048;
        static size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                                 ggml_graph_overhead_custom(MAX_NODES, false);
        thread_local std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, Cin);
        ggml_set_name(x, "ve_latent_tc");
        ggml_set_input(x);
        ggml_tensor * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, L);
        ggml_set_name(mask, "ve_latent_mask");
        ggml_set_input(mask);
        ggml_tensor * t_emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_set_name(t_emb, "ve_time_emb");
        ggml_set_input(t_emb);
        ggml_tensor * text_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, text_len, 256);
        ggml_set_name(text_in, "ve_text_lc");
        ggml_set_input(text_in);
        ggml_tensor * y = conv1d_f32(ctx, require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_in.net.weight"), x, 1, 0, 1);
        ggml_tensor * masked = ggml_mul(ctx, y, repeat_like(ctx, mask, y));
        ggml_set_name(masked, "ve_masked");
        if (include_ggml_trace) {
            ggml_set_output(masked);
            ggml_build_forward_expand(gf, masked);
        }

        ggml_tensor * cur = masked;
        int dils_ggml[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            cur = vector_convnext_ggml(ctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext." + std::to_string(j),
                cur, dils_ggml[j]);
            if (include_ggml_trace) {
                const std::string name = "ve_block0_convnext" + std::to_string(j);
                ggml_set_name(cur, name.c_str());
                ggml_set_output(cur);
                ggml_build_forward_expand(gf, cur);
            }
        }

        ggml_tensor * t_proj = ggml_mul_mat(ctx,
            ggml_cont(ctx, ggml_transpose(ctx, require_source_tensor(model, "vector_estimator:onnx::MatMul_3095"))),
            ggml_reshape_2d(ctx, t_emb, 64, 1));
        t_proj = ggml_add(ctx, t_proj,
            ggml_reshape_2d(ctx,
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.1.linear.linear.bias"),
                C, 1));
        cur = ggml_add(ctx, cur, repeat_like(ctx, t_proj, cur));
        ggml_set_name(cur, "ve_time_add0");
        if (include_ggml_trace) {
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
        }

        cur = vector_convnext_ggml(ctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.2.convnext.0",
            cur, 1);
        ggml_set_name(cur, "ve_block2_convnext0");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        ggml_tensor * q_t = dense_matmul_time_ggml(ctx, cur,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3101"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_query.linear.bias"));
        ggml_set_name(q_t, "ve_attn0_q");
        ggml_set_output(q_t);
        ggml_build_forward_expand(gf, q_t);
        ggml_tensor * k_t = dense_matmul_time_ggml(ctx, text_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3102"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_key.linear.bias"));
        ggml_set_name(k_t, "ve_attn0_k");
        ggml_set_output(k_t);
        ggml_build_forward_expand(gf, k_t);
        ggml_tensor * v_t = dense_matmul_time_ggml(ctx, text_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3103"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_value.linear.bias"));
        ggml_set_name(v_t, "ve_attn0_v");
        ggml_set_output(v_t);
        ggml_build_forward_expand(gf, v_t);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_new failed");
        }
        if (!ggml_gallocr_reserve(allocr, gf)) {
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_reserve failed");
        }
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(x, noisy_latent, 0, (size_t) L * Cin * sizeof(float));
        ggml_backend_tensor_set(mask, latent_mask, 0, (size_t) L * sizeof(float));
        std::vector<float> te_host = time_embedding(model, current_step, total_steps);
        ggml_backend_tensor_set(t_emb, te_host.data(), 0, te_host.size() * sizeof(float));
        // text_emb is already in (channel, time) layout so the cache that
        // used to wrap this set was a verbatim copy keyed on a pointer
        // that never matched twice.  Removed; set the tensor directly
        // from the caller-owned text_emb buffer.
        ggml_backend_tensor_set(text_in, text_emb, 0, (size_t) text_len * 256 * sizeof(float));
        profile_vector_compute(model, gf, current_step, "front_proj_attn0_qkv");

        PUSH_GGML_TRACE({"ve_latent_tc", {L, Cin}, in});
        PUSH_GGML_TRACE({"ve_masked", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_masked"))});
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_block0_convnext" + std::to_string(j);
            PUSH_GGML_TRACE({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, name.c_str()))});
        }
        PUSH_GGML_TRACE({"ve_time_add0", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_time_add0"))});
        std::vector<float> block2_ggml = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_block2_convnext0"));
        PUSH_GGML_TRACE({"ve_block2_convnext0", {L, C}, block2_ggml});
        std::vector<float> q_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_q"));
        std::vector<float> k_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_k"));
        std::vector<float> v_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_v"));
        PUSH_GGML_TRACE({"ve_attn0_q", {L, 256}, q_out});
        PUSH_GGML_TRACE({"ve_attn0_k", {text_len, 256}, k_out});
        PUSH_GGML_TRACE({"ve_attn0_v", {text_len, 256}, v_out});
        f32_tensor theta = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta.data.data(), q_out, L, 4, 64);
        apply_rope(theta.data.data(), k_out, text_len, 4, 64);
        thread_local vector_text_attention_cache att0_cache;
        std::vector<float> att0_ctx_trace;
        std::vector<float> attn_out_ggml = run_text_attention_cache(att0_cache, model, q_out, k_out, v_out,
            L, text_len, 4, 64,
            "vector_estimator:onnx::MatMul_3110",
            "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.out_fc.linear.bias",
            current_step, "attn0_flash",
            include_ggml_trace ? &att0_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_attn0_q_rope", {L, 256}, q_out});
        PUSH_GGML_TRACE({"ve_attn0_k_rope", {text_len, 256}, k_out});
        PUSH_GGML_TRACE({"ve_attn0_ctx", {L, 256}, att0_ctx_trace});
        PUSH_GGML_TRACE({"ve_attn0_out", {L, C}, attn_out_ggml});

        const std::vector<float> * style_v_raw = nullptr;
        const std::vector<float> * kctx_raw = nullptr;
        cached_style_layouts(model, style_ttl, style_v_raw, kctx_raw);
        thread_local vector_res_style_qkv_cache style0_res_qkv_cache;
        vector_res_style_qkv_result style0_res_qkv = run_res_style_qkv_cache(
            style0_res_qkv_cache, model, block2_ggml, attn_out_ggml, L, C,
            *style_v_raw, *kctx_raw, current_step,
            3, 4, 5,
            "vector_estimator:onnx::MatMul_3116",
            "vector_estimator:onnx::MatMul_3117",
            "vector_estimator:onnx::MatMul_3118",
            "ve_attn0_residual",
            "ve_attn0_norm",
            "ve_block4_convnext0",
            "ve_style0_q",
            "ve_style0_k_tanh",
            "ve_style0_v",
            "attn0_residual_style_qkv",
            include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> post_ggml = std::move(style0_res_qkv.post);
        std::vector<float> sq_out = std::move(style0_res_qkv.sq);
        std::vector<float> sk_out = std::move(style0_res_qkv.sk);
        std::vector<float> sv_out = std::move(style0_res_qkv.sv);
        thread_local vector_text_attention_cache style0_attn_cache;
        std::vector<float> style0_ctx_trace;
        std::vector<float> style_out_ggml = run_text_attention_cache(style0_attn_cache, model, sq_out, sk_out, sv_out,
            L, 50, 2, 128,
            "vector_estimator:onnx::MatMul_3119",
            "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.bias",
            current_step, "style0_flash",
            include_ggml_trace ? &style0_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_style0_ctx", {L, 256}, style0_ctx_trace});
        PUSH_GGML_TRACE({"ve_style0_out", {L, C}, style_out_ggml});
        constexpr int STYLE_RES_NODES = 128;
        static size_t style_res_buf_size = ggml_tensor_overhead() * STYLE_RES_NODES +
                                           ggml_graph_overhead_custom(STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> style_res_buf(style_res_buf_size);
        ggml_init_params srp = { style_res_buf_size, style_res_buf.data(), true };
        ggml_context * srctx = ggml_init(srp);
        ggml_cgraph * srgf = ggml_new_graph_custom(srctx, STYLE_RES_NODES, false);
        ggml_tensor * style_out_in = ggml_new_tensor_2d(srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(style_out_in, "style_out_in"); ggml_set_input(style_out_in);
        ggml_tensor * style_lhs_in = ggml_new_tensor_2d(srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(style_lhs_in, "style_lhs_in"); ggml_set_input(style_lhs_in);
        ggml_tensor * style_res = ggml_add(srctx, style_lhs_in, style_out_in);
        ggml_set_name(style_res, "ve_style0_residual");
        if (include_ggml_trace) {
            ggml_set_output(style_res);
            ggml_build_forward_expand(srgf, style_res);
        }
        ggml_tensor * style_norm = layer_norm_ggml(srctx, style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.bias"));
        ggml_set_name(style_norm, "ve_style0_norm"); ggml_set_output(style_norm);
        ggml_build_forward_expand(srgf, style_norm);
        ggml_gallocr_t srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!srallocr) {
            ggml_free(srctx);
            throw std::runtime_error("ggml_gallocr_new style residual failed");
        }
        if (!ggml_gallocr_reserve(srallocr, srgf)) {
            ggml_gallocr_free(srallocr);
            ggml_free(srctx);
            throw std::runtime_error("ggml_gallocr_reserve style residual failed");
        }
        ggml_gallocr_alloc_graph(srallocr, srgf);
        std::vector<float> style_out_raw = pack_time_channel_for_ggml(style_out_ggml, L, C);
        std::vector<float> style_lhs_raw = pack_time_channel_for_ggml(post_ggml, L, C);
        ggml_backend_tensor_set(style_out_in, style_out_raw.data(), 0, style_out_raw.size()*sizeof(float));
        ggml_backend_tensor_set(style_lhs_in, style_lhs_raw.data(), 0, style_lhs_raw.size()*sizeof(float));
        profile_vector_compute(model, srgf, current_step, "style0_residual");
        PUSH_GGML_TRACE({"ve_style0_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(srgf, "ve_style0_residual"))});
        std::vector<float> style_norm_ggml = tensor_to_time_channel(ggml_graph_get_tensor(srgf, "ve_style0_norm"));
        PUSH_GGML_TRACE({"ve_style0_norm", {L, C}, style_norm_ggml});
        ggml_gallocr_free(srallocr);
        ggml_free(srctx);

        thread_local vector_group_graph_cache g1_group_cache;
        vector_group_graph_result g1_group = run_group_graph_cache(g1_group_cache, model, style_norm_ggml,
            L, C, te_host, text_emb, text_len, current_step,
            1, 6, 7, "vector_estimator:onnx::MatMul_3140", 8,
            "vector_estimator:onnx::MatMul_3146",
            "vector_estimator:onnx::MatMul_3147",
            "vector_estimator:onnx::MatMul_3148",
            "ve_g1_attn_q", "ve_g1_attn_k", "ve_g1_attn_v",
            "group1_conv_attn_qkv", include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g1_block8 = std::move(g1_group.post);
        std::vector<float> g1q_out = std::move(g1_group.q);
        std::vector<float> g1k_out = std::move(g1_group.k);
        std::vector<float> g1v_out = std::move(g1_group.v);
        f32_tensor theta_g1 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta_g1.data.data(), g1q_out, L, 4, 64);
        apply_rope(theta_g1.data.data(), g1k_out, text_len, 4, 64);
        thread_local vector_text_attention_cache g1_attn_cache;
        std::vector<float> g1_attn_ctx_trace;
        std::vector<float> g1_attn_out = run_text_attention_cache(g1_attn_cache, model, g1q_out, g1k_out, g1v_out,
            L, text_len, 4, 64,
            "vector_estimator:onnx::MatMul_3155",
            "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.out_fc.linear.bias",
            current_step, "g1_attn_flash",
            include_ggml_trace ? &g1_attn_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g1_attn_q_rope", {L, 256}, g1q_out});
        PUSH_GGML_TRACE({"ve_g1_attn_k_rope", {text_len, 256}, g1k_out});
        PUSH_GGML_TRACE({"ve_g1_attn_ctx", {L, 256}, g1_attn_ctx_trace});
        PUSH_GGML_TRACE({"ve_g1_attn_out", {L, C}, g1_attn_out});

        thread_local vector_res_style_qkv_cache g1_res_qkv_cache;
        vector_res_style_qkv_result g1_res_qkv = run_res_style_qkv_cache(
            g1_res_qkv_cache, model, g1_block8, g1_attn_out, L, C,
            *style_v_raw, *kctx_raw, current_step,
            9, 10, 11,
            "vector_estimator:onnx::MatMul_3161",
            "vector_estimator:onnx::MatMul_3162",
            "vector_estimator:onnx::MatMul_3163",
            "ve_g1_attn_residual",
            "ve_g1_attn_norm",
            "ve_g1_block10_convnext0",
            "ve_g1_style_q",
            "ve_g1_style_k_tanh",
            "ve_g1_style_v",
            "g1_attn_residual_style_qkv",
            include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g1_block10 = std::move(g1_res_qkv.post);
        std::vector<float> g1sq_out = std::move(g1_res_qkv.sq);
        std::vector<float> g1sk_out = std::move(g1_res_qkv.sk);
        std::vector<float> g1sv_out = std::move(g1_res_qkv.sv);
        thread_local vector_text_attention_cache g1_style_attn_cache;
        std::vector<float> g1_style_ctx_trace;
        std::vector<float> g1_style_out = run_text_attention_cache(g1_style_attn_cache, model, g1sq_out, g1sk_out, g1sv_out,
            L, 50, 2, 128,
            "vector_estimator:onnx::MatMul_3164",
            "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.out_fc.linear.bias",
            current_step, "g1_style_flash",
            include_ggml_trace ? &g1_style_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g1_style_ctx", {L, 256}, g1_style_ctx_trace});
        PUSH_GGML_TRACE({"ve_g1_style_out", {L, C}, g1_style_out});

        constexpr int G1_STYLE_RES_NODES = 128;
        static size_t g1_style_res_buf_size = ggml_tensor_overhead() * G1_STYLE_RES_NODES +
                                              ggml_graph_overhead_custom(G1_STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> g1_style_res_buf(g1_style_res_buf_size);
        ggml_init_params g1srp = { g1_style_res_buf_size, g1_style_res_buf.data(), true };
        ggml_context * g1srctx = ggml_init(g1srp);
        ggml_cgraph * g1srgf = ggml_new_graph_custom(g1srctx, G1_STYLE_RES_NODES, false);
        ggml_tensor * g1_style_lhs = ggml_new_tensor_2d(g1srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_style_lhs, "g1_style_lhs"); ggml_set_input(g1_style_lhs);
        ggml_tensor * g1_style_out_in = ggml_new_tensor_2d(g1srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_style_out_in, "g1_style_out_in"); ggml_set_input(g1_style_out_in);
        ggml_tensor * g1_style_res = ggml_add(g1srctx, g1_style_lhs, g1_style_out_in);
        ggml_set_name(g1_style_res, "ve_g1_style_residual");
        if (include_ggml_trace) {
            ggml_set_output(g1_style_res);
            ggml_build_forward_expand(g1srgf, g1_style_res);
        }
        ggml_tensor * g1_style_norm = layer_norm_ggml(g1srctx, g1_style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.bias"));
        ggml_set_name(g1_style_norm, "ve_g1_style_norm"); ggml_set_output(g1_style_norm);
        ggml_build_forward_expand(g1srgf, g1_style_norm);
        ggml_gallocr_t g1srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g1srallocr) {
            ggml_free(g1srctx);
            throw std::runtime_error("ggml_gallocr_new group1 style residual failed");
        }
        if (!ggml_gallocr_reserve(g1srallocr, g1srgf)) {
            ggml_gallocr_free(g1srallocr);
            ggml_free(g1srctx);
            throw std::runtime_error("ggml_gallocr_reserve group1 style residual failed");
        }
        ggml_gallocr_alloc_graph(g1srallocr, g1srgf);
        std::vector<float> g1_style_lhs_raw = pack_time_channel_for_ggml(g1_block10, L, C);
        std::vector<float> g1_style_out_raw = pack_time_channel_for_ggml(g1_style_out, L, C);
        ggml_backend_tensor_set(g1_style_lhs, g1_style_lhs_raw.data(), 0, g1_style_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g1_style_out_in, g1_style_out_raw.data(), 0, g1_style_out_raw.size()*sizeof(float));
        profile_vector_compute(model, g1srgf, current_step, "g1_style_residual");
        PUSH_GGML_TRACE({"ve_g1_style_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g1srgf, "ve_g1_style_residual"))});
        std::vector<float> g1_style_norm_vec = tensor_to_time_channel(ggml_graph_get_tensor(g1srgf, "ve_g1_style_norm"));
        PUSH_GGML_TRACE({"ve_g1_style_norm", {L, C}, g1_style_norm_vec});
        ggml_gallocr_free(g1srallocr);
        ggml_free(g1srctx);

        thread_local vector_group_graph_cache g2_group_cache;
        vector_group_graph_result g2_group = run_group_graph_cache(g2_group_cache, model, g1_style_norm_vec,
            L, C, te_host, text_emb, text_len, current_step,
            2, 12, 13, "vector_estimator:onnx::MatMul_3185", 14,
            "vector_estimator:onnx::MatMul_3191",
            "vector_estimator:onnx::MatMul_3192",
            "vector_estimator:onnx::MatMul_3193",
            "ve_g2_attn_q", "ve_g2_attn_k", "ve_g2_attn_v",
            "group2_conv_attn_qkv", include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g2_block14 = std::move(g2_group.post);
        std::vector<float> g2q_out = std::move(g2_group.q);
        std::vector<float> g2k_out = std::move(g2_group.k);
        std::vector<float> g2v_out = std::move(g2_group.v);
        f32_tensor theta_g2 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta_g2.data.data(), g2q_out, L, 4, 64);
        apply_rope(theta_g2.data.data(), g2k_out, text_len, 4, 64);
        thread_local vector_text_attention_cache g2_attn_cache;
        std::vector<float> g2_attn_ctx_trace;
        std::vector<float> g2_attn_out = run_text_attention_cache(g2_attn_cache, model, g2q_out, g2k_out, g2v_out,
            L, text_len, 4, 64,
            "vector_estimator:onnx::MatMul_3200",
            "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.out_fc.linear.bias",
            current_step, "g2_attn_flash",
            include_ggml_trace ? &g2_attn_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g2_attn_q_rope", {L, 256}, g2q_out});
        PUSH_GGML_TRACE({"ve_g2_attn_k_rope", {text_len, 256}, g2k_out});
        PUSH_GGML_TRACE({"ve_g2_attn_ctx", {L, 256}, g2_attn_ctx_trace});
        PUSH_GGML_TRACE({"ve_g2_attn_out", {L, C}, g2_attn_out});

        thread_local vector_res_style_qkv_cache g2_res_qkv_cache;
        vector_res_style_qkv_result g2_res_qkv = run_res_style_qkv_cache(
            g2_res_qkv_cache, model, g2_block14, g2_attn_out, L, C,
            *style_v_raw, *kctx_raw, current_step,
            15, 16, 17,
            "vector_estimator:onnx::MatMul_3206",
            "vector_estimator:onnx::MatMul_3207",
            "vector_estimator:onnx::MatMul_3208",
            "ve_g2_attn_residual",
            "ve_g2_attn_norm",
            "ve_g2_block16_convnext0",
            "ve_g2_style_q",
            "ve_g2_style_k_tanh",
            "ve_g2_style_v",
            "g2_attn_residual_style_qkv",
            include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g2_block16 = std::move(g2_res_qkv.post);
        std::vector<float> g2sq_out = std::move(g2_res_qkv.sq);
        std::vector<float> g2sk_out = std::move(g2_res_qkv.sk);
        std::vector<float> g2sv_out = std::move(g2_res_qkv.sv);
        thread_local vector_text_attention_cache g2_style_attn_cache;
        std::vector<float> g2_style_ctx_trace;
        std::vector<float> g2_style_out = run_text_attention_cache(g2_style_attn_cache, model, g2sq_out, g2sk_out, g2sv_out,
            L, 50, 2, 128,
            "vector_estimator:onnx::MatMul_3209",
            "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.out_fc.linear.bias",
            current_step, "g2_style_flash",
            include_ggml_trace ? &g2_style_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g2_style_ctx", {L, 256}, g2_style_ctx_trace});
        PUSH_GGML_TRACE({"ve_g2_style_out", {L, C}, g2_style_out});

        constexpr int G2_STYLE_RES_NODES = 128;
        static size_t g2_style_res_buf_size = ggml_tensor_overhead() * G2_STYLE_RES_NODES +
                                              ggml_graph_overhead_custom(G2_STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> g2_style_res_buf(g2_style_res_buf_size);
        ggml_init_params g2srp = { g2_style_res_buf_size, g2_style_res_buf.data(), true };
        ggml_context * g2srctx = ggml_init(g2srp);
        ggml_cgraph * g2srgf = ggml_new_graph_custom(g2srctx, G2_STYLE_RES_NODES, false);
        ggml_tensor * g2_style_lhs = ggml_new_tensor_2d(g2srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_style_lhs, "g2_style_lhs"); ggml_set_input(g2_style_lhs);
        ggml_tensor * g2_style_out_in = ggml_new_tensor_2d(g2srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_style_out_in, "g2_style_out_in"); ggml_set_input(g2_style_out_in);
        ggml_tensor * g2_style_res = ggml_add(g2srctx, g2_style_lhs, g2_style_out_in);
        ggml_set_name(g2_style_res, "ve_g2_style_residual");
        if (include_ggml_trace) {
            ggml_set_output(g2_style_res);
            ggml_build_forward_expand(g2srgf, g2_style_res);
        }
        ggml_tensor * g2_style_norm = layer_norm_ggml(g2srctx, g2_style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.bias"));
        ggml_set_name(g2_style_norm, "ve_g2_style_norm"); ggml_set_output(g2_style_norm);
        ggml_build_forward_expand(g2srgf, g2_style_norm);
        ggml_gallocr_t g2srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g2srallocr) {
            ggml_free(g2srctx);
            throw std::runtime_error("ggml_gallocr_new group2 style residual failed");
        }
        if (!ggml_gallocr_reserve(g2srallocr, g2srgf)) {
            ggml_gallocr_free(g2srallocr);
            ggml_free(g2srctx);
            throw std::runtime_error("ggml_gallocr_reserve group2 style residual failed");
        }
        ggml_gallocr_alloc_graph(g2srallocr, g2srgf);
        std::vector<float> g2_style_lhs_raw = pack_time_channel_for_ggml(g2_block16, L, C);
        std::vector<float> g2_style_out_raw = pack_time_channel_for_ggml(g2_style_out, L, C);
        ggml_backend_tensor_set(g2_style_lhs, g2_style_lhs_raw.data(), 0, g2_style_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g2_style_out_in, g2_style_out_raw.data(), 0, g2_style_out_raw.size()*sizeof(float));
        profile_vector_compute(model, g2srgf, current_step, "g2_style_residual");
        PUSH_GGML_TRACE({"ve_g2_style_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g2srgf, "ve_g2_style_residual"))});
        std::vector<float> g2_style_norm_vec = tensor_to_time_channel(ggml_graph_get_tensor(g2srgf, "ve_g2_style_norm"));
        PUSH_GGML_TRACE({"ve_g2_style_norm", {L, C}, g2_style_norm_vec});
        ggml_gallocr_free(g2srallocr);
        ggml_free(g2srctx);

        thread_local vector_group_graph_cache g3_group_cache;
        vector_group_graph_result g3_group = run_group_graph_cache(g3_group_cache, model, g2_style_norm_vec,
            L, C, te_host, text_emb, text_len, current_step,
            3, 18, 19, "vector_estimator:onnx::MatMul_3230", 20,
            "vector_estimator:onnx::MatMul_3236",
            "vector_estimator:onnx::MatMul_3237",
            "vector_estimator:onnx::MatMul_3238",
            "ve_g3_attn_q", "ve_g3_attn_k", "ve_g3_attn_v",
            "group3_conv_attn_qkv", include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g3_block20 = std::move(g3_group.post);
        std::vector<float> g3q_out = std::move(g3_group.q);
        std::vector<float> g3k_out = std::move(g3_group.k);
        std::vector<float> g3v_out = std::move(g3_group.v);
        f32_tensor theta_g3 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta_g3.data.data(), g3q_out, L, 4, 64);
        apply_rope(theta_g3.data.data(), g3k_out, text_len, 4, 64);
        thread_local vector_text_attention_cache g3_attn_cache;
        std::vector<float> g3_attn_ctx_trace;
        std::vector<float> g3_attn_out = run_text_attention_cache(g3_attn_cache, model, g3q_out, g3k_out, g3v_out,
            L, text_len, 4, 64,
            "vector_estimator:onnx::MatMul_3245",
            "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.out_fc.linear.bias",
            current_step, "g3_attn_flash",
            include_ggml_trace ? &g3_attn_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g3_attn_q_rope", {L, 256}, g3q_out});
        PUSH_GGML_TRACE({"ve_g3_attn_k_rope", {text_len, 256}, g3k_out});
        PUSH_GGML_TRACE({"ve_g3_attn_ctx", {L, 256}, g3_attn_ctx_trace});
        PUSH_GGML_TRACE({"ve_g3_attn_out", {L, C}, g3_attn_out});

        thread_local vector_res_style_qkv_cache g3_res_qkv_cache;
        vector_res_style_qkv_result g3_res_qkv = run_res_style_qkv_cache(
            g3_res_qkv_cache, model, g3_block20, g3_attn_out, L, C,
            *style_v_raw, *kctx_raw, current_step,
            21, 22, 23,
            "vector_estimator:onnx::MatMul_3251",
            "vector_estimator:onnx::MatMul_3252",
            "vector_estimator:onnx::MatMul_3253",
            "ve_g3_attn_residual",
            "ve_g3_attn_norm",
            "ve_g3_block22_convnext0",
            "ve_g3_style_q",
            "ve_g3_style_k_tanh",
            "ve_g3_style_v",
            "g3_attn_residual_style_qkv",
            include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g3_block22 = std::move(g3_res_qkv.post);
        std::vector<float> g3sq_out = std::move(g3_res_qkv.sq);
        std::vector<float> g3sk_out = std::move(g3_res_qkv.sk);
        std::vector<float> g3sv_out = std::move(g3_res_qkv.sv);
        thread_local vector_text_attention_cache g3_style_attn_cache;
        std::vector<float> g3_style_ctx_trace;
        std::vector<float> g3_style_out = run_text_attention_cache(g3_style_attn_cache, model, g3sq_out, g3sk_out, g3sv_out,
            L, 50, 2, 128,
            "vector_estimator:onnx::MatMul_3254",
            "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.out_fc.linear.bias",
            current_step, "g3_style_flash",
            include_ggml_trace ? &g3_style_ctx_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g3_style_ctx", {L, 256}, g3_style_ctx_trace});
        PUSH_GGML_TRACE({"ve_g3_style_out", {L, C}, g3_style_out});

        constexpr int G3_STYLE_RES_NODES = 128;
        static size_t g3_style_res_buf_size = ggml_tensor_overhead() * G3_STYLE_RES_NODES +
                                              ggml_graph_overhead_custom(G3_STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> g3_style_res_buf(g3_style_res_buf_size);
        ggml_init_params g3srp = { g3_style_res_buf_size, g3_style_res_buf.data(), true };
        ggml_context * g3srctx = ggml_init(g3srp);
        ggml_cgraph * g3srgf = ggml_new_graph_custom(g3srctx, G3_STYLE_RES_NODES, false);
        ggml_tensor * g3_style_lhs = ggml_new_tensor_2d(g3srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_style_lhs, "g3_style_lhs"); ggml_set_input(g3_style_lhs);
        ggml_tensor * g3_style_out_in = ggml_new_tensor_2d(g3srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_style_out_in, "g3_style_out_in"); ggml_set_input(g3_style_out_in);
        ggml_tensor * g3_style_res = ggml_add(g3srctx, g3_style_lhs, g3_style_out_in);
        ggml_set_name(g3_style_res, "ve_g3_style_residual");
        if (include_ggml_trace) {
            ggml_set_output(g3_style_res);
            ggml_build_forward_expand(g3srgf, g3_style_res);
        }
        ggml_tensor * g3_style_norm = layer_norm_ggml(g3srctx, g3_style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.bias"));
        ggml_set_name(g3_style_norm, "ve_g3_style_norm"); ggml_set_output(g3_style_norm);
        ggml_build_forward_expand(g3srgf, g3_style_norm);
        ggml_gallocr_t g3srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g3srallocr) {
            ggml_free(g3srctx);
            throw std::runtime_error("ggml_gallocr_new group3 style residual failed");
        }
        if (!ggml_gallocr_reserve(g3srallocr, g3srgf)) {
            ggml_gallocr_free(g3srallocr);
            ggml_free(g3srctx);
            throw std::runtime_error("ggml_gallocr_reserve group3 style residual failed");
        }
        ggml_gallocr_alloc_graph(g3srallocr, g3srgf);
        std::vector<float> g3_style_lhs_raw = pack_time_channel_for_ggml(g3_block22, L, C);
        std::vector<float> g3_style_out_raw = pack_time_channel_for_ggml(g3_style_out, L, C);
        ggml_backend_tensor_set(g3_style_lhs, g3_style_lhs_raw.data(), 0, g3_style_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g3_style_out_in, g3_style_out_raw.data(), 0, g3_style_out_raw.size()*sizeof(float));
        profile_vector_compute(model, g3srgf, current_step, "g3_style_residual");
        PUSH_GGML_TRACE({"ve_g3_style_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g3srgf, "ve_g3_style_residual"))});
        std::vector<float> g3_style_norm_vec = tensor_to_time_channel(ggml_graph_get_tensor(g3srgf, "ve_g3_style_norm"));
        PUSH_GGML_TRACE({"ve_g3_style_norm", {L, C}, g3_style_norm_vec});
        ggml_gallocr_free(g3srallocr);
        ggml_free(g3srctx);

        thread_local vector_tail_graph_cache tail_cache;
        std::vector<float> next_latent_tc = run_tail_graph_cache(tail_cache, model, g3_style_norm_vec,
            noisy_latent, latent_mask, L, C, Cin, current_step, total_steps,
            include_ggml_trace ? &ggml_trace : nullptr);
        if (next_latent_tc_out) *next_latent_tc_out = next_latent_tc;

        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        profile_vector_step_end(current_step);
        if (error) error->clear();
#undef PUSH_GGML_TRACE
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_vector_step_ggml(const supertonic_model & model,
                                 const float * noisy_latent,
                                 int latent_len,
                                 const float * text_emb,
                                 int text_len,
                                 const float * style_ttl,
                                 const float * latent_mask,
                                 int current_step,
                                 int total_steps,
                                 std::vector<float> & next_latent_out,
                                 std::string * error) {
    try {
        std::vector<supertonic_trace_tensor> scalar_trace;
        std::vector<supertonic_trace_tensor> ggml_trace;
        std::vector<float> next_tc;
        if (!supertonic_vector_trace_proj_ggml(model, noisy_latent, text_emb, text_len,
                                               style_ttl, latent_mask, latent_len,
                                               current_step, total_steps,
                                               scalar_trace, ggml_trace, error,
                                               false, false, &next_tc)) {
            return false;
        }
        const int L = latent_len;
        const int C = model.hparams.latent_channels;
        if (next_tc.size() != (size_t)L*C) throw std::runtime_error("bad ve_next_latent_tc size");
        next_latent_out.assign((size_t)C*L, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < L; ++t) {
                next_latent_out[(size_t)c*L + t] = next_tc[(size_t)t*C + c];
            }
        }
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
