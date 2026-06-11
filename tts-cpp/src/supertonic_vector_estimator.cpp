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
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::supertonic::detail {
namespace {

// Per-thread registry of release-thunks for the thread_local graph
// caches that live in this TU.  Each cache use-site registers exactly
// once per thread (guarded by a static thread_local once-flag) so the
// engine destructor's `release_vector_estimator_thread_local_caches()`
// can drive every populated cache through its normal `free_*_cache`
// path against the still-live backend — instead of leaving the
// gallocr's internal hash tables / per-leaf records resident under
// the dead-backend skip in `supertonic_safe_gallocr_free`.
thread_local std::vector<std::function<void()>> g_tl_release_thunks;

inline void supertonic_register_tl_cache(std::function<void()> fn) {
    g_tl_release_thunks.push_back(std::move(fn));
}

// Sentinel: a `thread_local` instance of this struct constructed
// next to each `thread_local <cache_type> cache;` declaration runs
// its constructor exactly once per thread (at first reach of the
// host function), registering a thunk that drives `cache` through
// its normal `free_<type>_cache` path on engine teardown.
struct tl_register_once {
    template <typename F>
    explicit tl_register_once(F && f) { supertonic_register_tl_cache(std::forward<F>(f)); }
};

// One-line registration macro for the thread_local graph caches
// scattered through the vector-estimator pipeline.  The companion
// `tl_register_once` sentinel pushes the lambda onto `g_tl_release_thunks`
// exactly once per thread on first reach of the host function.
#define SUPERTONIC_REGISTER_TL_CACHE(cache_var, free_fn) \
    thread_local tl_register_once _tl_reg_##cache_var( \
        [&]() { free_fn(cache_var); })

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
                            const char * island,
                            bool use_sched = false) {
    // Callers pick the compute primitive by allocation strategy:
    //   use_sched == false : graph is bound to a per-cache
    //                        `ggml_gallocr_t` (HEAD's F8/F18/F19/...
    //                        caches).  Use `supertonic_graph_compute`
    //                        (direct backend compute) so the tensors'
    //                        gallocr-bound buffers are honoured.
    //                        Routing through `model.sched` would
    //                        force the graph through a scheduler that
    //                        doesn't know about the per-cache gallocr
    //                        and silently corrupt the output.
    //   use_sched == true  : graph is allocated by
    //                        `supertonic_sched_alloc` on the model
    //                        scheduler (QVAC-19254 fallback when the
    //                        primary backend doesn't support every
    //                        op).  Use `supertonic_sched_compute` so
    //                        the alloc + compute pair is consistent.
    auto dispatch = [&]() {
        if (use_sched) supertonic_sched_compute(model, graph);
        else           supertonic_graph_compute(model, graph);
    };
    const bool stderr_on = vector_profile_enabled();
    const bool csv_on    = supertonic_profile_csv_enabled();
    if (!stderr_on && !csv_on) {
        dispatch();
        return;
    }
    auto & state = vector_profile();
    const auto t0 = std::chrono::steady_clock::now();
    const double pre_ms = std::chrono::duration<double, std::milli>(t0 - state.last).count();
    dispatch();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    state.last = t1;
    if (stderr_on) {
        std::fprintf(stderr, "supertonic_vector_profile step=%d island=%s pre_ms=%.3f compute_ms=%.3f\n",
                     step, island, pre_ms, ms);
    }
    // Phase 2D: machine-readable timing for the post-mortem
    // analysis script.  Records every graph compute call with the
    // stage/island context the existing stderr line already
    // carries.  No-op when the CSV emitter isn't enabled.
    if (csv_on) {
        supertonic_profile_csv_record("vector", island, step, ms);
    }
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
    // Every call site in this file feeds the return value straight into
    // ggml_add / ggml_mul, both of which broadcast natively in ggml.  Skip
    // the explicit ggml_repeat node so the downstream op handles the
    // broadcast — saves ~282 REPEAT ops per consolidated per-step graph.
    // Override with SUPERTONIC_FORCE_EXPLICIT_REPEAT=1 if this regresses
    // on a backend that doesn't broadcast (none observed today).
    static const bool force_explicit_repeat =
        std::getenv("SUPERTONIC_FORCE_EXPLICIT_REPEAT") != nullptr;
    if (force_explicit_repeat) {
        return ggml_repeat(ctx, v, like);
    }
    return v;
}

ggml_tensor * conv1d_f32(ggml_context * ctx,
                         ggml_tensor * kernel,
                         ggml_tensor * input,
                         int stride,
                         int padding,
                         int dilation) {
#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
    // CPU-only fast path: see supertonic_op_dispatch_scope contract.
    if (supertonic_use_cpu_custom_ops() &&
        kernel->ne[0] == 1 && stride == 1 && padding == 0 && dilation == 1 &&
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
    if (pad_left == 0 && pad_right == 0) return x;
    // Fused fast path via supertonic_edge_pad_1d.  Same kernel handles
    // both sides; the legacy view + repeat_4d + concat chain (2 ops
    // per side) becomes 1 dispatch total.  Override:
    // SUPERTONIC_DISABLE_FUSED_EDGE_PAD=1.
    static const bool disable_fused_edge_pad =
        std::getenv("SUPERTONIC_DISABLE_FUSED_EDGE_PAD") != nullptr;
    if (!disable_fused_edge_pad && supertonic_use_fused_supertonic_ops() &&
        x->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        ggml_is_contiguous(x)) {
        return ggml_supertonic_edge_pad_1d(ctx, x, pad_left, pad_right);
    }
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
    // GPU backends reject GGML_OP_CUSTOM; fall through to the pure-GGML
    // im2col + mul_mat path in depthwise_same_ggml() below.
    if (!supertonic_use_cpu_custom_ops()) return nullptr;
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
    // Fused-op fast path (any backend that registers GGML_OP_SUPERTONIC_DEPTHWISE_1D
    // — Metal does via the local ggml port overlay; CPU's
    // ggml_compute_forward_supertonic_depthwise_1d is the parity backstop).
    // Replaces the edge_clamp_pad + im2col + mul_mat + add chain with one
    // dispatch.  Currently supports K in {3, 5}; the existing graph path is
    // the fallback for K outside that set.  Override with
    // SUPERTONIC_DISABLE_FUSED_DEPTHWISE=1 to force the stock-op chain.
    static const bool disable_fused =
        std::getenv("SUPERTONIC_DISABLE_FUSED_DEPTHWISE") != nullptr;
    if (!disable_fused && supertonic_use_fused_supertonic_ops() && (K == 3 || K == 5) &&
        x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
        b->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1 && w->ne[1] == 1 && w->ne[3] == 1 &&
        w->ne[2] == x->ne[1] && b->ne[0] == x->ne[1] &&
        ggml_is_contiguous(x) && ggml_is_contiguous(w) && ggml_is_contiguous(b)) {
        return ggml_supertonic_depthwise_1d(ctx, x, w, b, dilation);
    }
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
    // Fused-op fast path on non-CPU backends (Metal/Vulkan/CUDA/OpenCL):
    // GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL collapses the
    // permute + cont + ggml_norm + mul + add + permute + cont chain into
    // a single dispatch.  Override with SUPERTONIC_DISABLE_FUSED_LAYER_NORM=1.
    static const bool disable_fused_layer_norm =
        std::getenv("SUPERTONIC_DISABLE_FUSED_LAYER_NORM") != nullptr;
    if (!supertonic_use_cpu_custom_ops() && supertonic_use_fused_supertonic_ops() && !disable_fused_layer_norm &&
        x->type == GGML_TYPE_F32 && g->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        g->ne[0] == x->ne[1] && b->ne[0] == x->ne[1] &&
        ggml_is_contiguous(x) && ggml_is_contiguous(g) && ggml_is_contiguous(b)) {
        return ggml_supertonic_layer_norm_channel(ctx, x, g, b, 1e-6f);
    }
    // CPU-only direct row-wise layer-norm; falls through to permute +
    // ggml_norm on non-CPU backends so the graph stays GPU-executable.
    if (supertonic_use_cpu_custom_ops() &&
        x->type == GGML_TYPE_F32 && g->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
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
    // CPU-only direct dense-time matmul; the pure-GGML fallback below
    // expresses the same op via conv1d_f32(K=1) which is supported on
    // every backend.
    if (supertonic_use_cpu_custom_ops() &&
        x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && (!b || b->type == GGML_TYPE_F32) &&
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
    //
    // Tried replacing this conv1d_f32 wrapper with a direct ggml_mul_mat on
    // 2026-05-11 — it requires cont on BOTH operands to satisfy mul_mat's
    // !ggml_is_transposed(A) assertion, which yields the SAME dispatch count
    // (cont + cont + mul_mat + add) as the current conv1d path (cont +
    // im2col + mul_mat + add).  Net wash; keeping conv1d_f32 because it's
    // already battle-tested with the CPU fastpath.
    ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, w));
    ggml_tensor * kernel = ggml_reshape_3d(ctx, wt, 1, w->ne[1], w->ne[0]);
    ggml_tensor * y = conv1d_f32(ctx, kernel, x, 1, 0, 1);
    if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
    return y;
}

// Same as dense_matmul_time_ggml, but `model` is consulted for a pre-
// transposed copy of `w` (built at load time for `:onnx::MatMul_*` weights
// on non-CPU backends).  When available, the runtime `cont(transpose(w))`
// dispatch is skipped — the pre-transposed tensor already has the
// `[IC, OC]` layout that the conv1d_f32 K=1 kernel expects.  CPU callers
// fall through to the original path (the cblas pointwise fast path takes
// the loaded `[OC, IC]` weight directly).
// Forward decl — defined below.
ggml_tensor * dense_matmul_time_wt_pretransposed_ggml(ggml_context * ctx,
                                                      const supertonic_model & model,
                                                      ggml_tensor * x,
                                                      ggml_tensor * w,
                                                      ggml_tensor * b);

ggml_tensor * dense_matmul_time_pretransposed_ggml(ggml_context * ctx,
                                                   const supertonic_model & model,
                                                   ggml_tensor * x,
                                                   ggml_tensor * w,
                                                   ggml_tensor * b) {
    if (!supertonic_use_cpu_custom_ops()) {
        if (ggml_tensor * w_pre = try_pretransposed_weight(model, w)) {
            if (w_pre->type == GGML_TYPE_F32) {
                // f32 fast path: reshape w_pre into the conv1d kernel
                // [K=1, IC, OC] and dispatch via the existing wrapper.
                // mul_mat(im2col_f32, kernel_f32) hits the optimised
                // kernel_mul_mm_f32_f32.
                ggml_tensor * kernel = ggml_reshape_3d(ctx, w_pre, 1, w_pre->ne[0], w_pre->ne[1]);
                ggml_tensor * y = conv1d_f32(ctx, kernel, x, 1, 0, 1);
                if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
                return y;
            }
            // Quantized w_pre (q8_0): the f32 fast path's
            // mul_mat(im2col_f32, kernel_quant) would need a
            // kernel_mul_mm_f32_q8_0 variant which ggml-metal doesn't ship.
            // Route through the wt helper (kernel as src0 — dispatches
            // kernel_mul_mm_q8_0_f32) and transpose the [A, T] result back
            // to [T, A] so the caller's downstream code (residual adds,
            // [T, C]-shaped intermediate state) doesn't have to change.
            ggml_tensor * y_wt = dense_matmul_time_wt_pretransposed_ggml(
                ctx, model, x, w, b);
            return ggml_cont(ctx, ggml_transpose(ctx, y_wt));
        }
    }
    return dense_matmul_time_ggml(ctx, x, w, b);
}

// Phase B2 partial: like dense_matmul_time_pretransposed_ggml but emits
// the result in *width-major* `[OC, T]` layout instead of `[T, OC]`.
//
// The trick is to swap the `ggml_mul_mat` operand order from
// `mul_mat(im2col_[IC,T], kernel_[IC,OC]) -> [T, OC]` to
// `mul_mat(kernel_[IC,OC], im2col_[IC,T]) -> [OC, T]`.  Both operands
// stay non-transposed so the assertion on `a`/`b` is satisfied.  The
// kernel-as-`src0` ordering is also what `kernel_mul_mm_q8_0_f32`
// requires, so this single change *also* unlocks A3 step 2 (the
// optimized quantized matmul kernel will dispatch when `w_pre` is
// q8_0 — see the asymmetric load logic in supertonic_gguf.cpp).
//
// Used at the Q/K/V projection sites in the per-step graph: the
// downstream rope + flash_attn expect `[A, L]` layout, so the cont
// (transpose) that used to flip `[L, A]` -> `[A, L]` becomes dead
// code.  Eliminates ~24 cont dispatches per per-step graph × 5
// steps = ~120 ops per synth.
//
// Bias add: `b` (shape `[OC]`) broadcasts naturally against the
// new `[OC, T]` output via `repeat_like`'s 1-d → 2-d reshape on the
// `ne[0]` match.
//
// Falls through to the legacy path with a runtime cont(transpose)
// on the activation when no pretransposed weight is available
// (e.g. weight not on the `:onnx::MatMul_` allowlist).
ggml_tensor * dense_matmul_time_wt_pretransposed_ggml(ggml_context * ctx,
                                                      const supertonic_model & model,
                                                      ggml_tensor * x,
                                                      ggml_tensor * w,
                                                      ggml_tensor * b) {
    if (!supertonic_use_cpu_custom_ops()) {
        if (ggml_tensor * w_pre = try_pretransposed_weight(model, w)) {
            const int IC = (int) w_pre->ne[0];
            const int OC = (int) w_pre->ne[1];

            // ggml_im2col only reads the kernel's SHAPE (ne[0..3]); it never
            // touches the kernel data — the output buffer holds the
            // rearranged activation.  So for the SHAPE we can use:
            //   - a reshape of w_pre when w_pre is f32 (cheap, just metadata)
            //   - a tiny phantom f32 tensor allocated in the graph context
            //     when w_pre is quantized (because reshape_3d(q8_0, 1, IC, OC)
            //     would set ne[0]=1 < q8_0's 32-element block size and break
            //     the type's invariants).  The phantom is never read.
            ggml_tensor * shape_kernel;
            if (w_pre->type == GGML_TYPE_F32) {
                shape_kernel = ggml_reshape_3d(ctx, w_pre, 1, IC, OC);
            } else {
                shape_kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, IC, OC);
                // No data needs binding — im2col only consults ne[0..3].
            }

            ggml_tensor * im2col = ggml_im2col(ctx, shape_kernel, x, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
            // im2col has ne=[IC, T, 1, 1].  Reshape to 2D for mul_mat.
            ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col,
                                                      im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
            // Swapped order: w_pre first (src0 = the quantized/f32 weight),
            // im2col second (src1 = f32 activation).  Result is [M=OC, N=T].
            // For w_pre=q8_0 this dispatches kernel_mul_mm_q8_0_f32 — the
            // bandwidth-optimised quantized matmul kernel — which is the
            // A3 step 2 unlock.
            ggml_tensor * w_2d = ggml_reshape_2d(ctx, w_pre, IC, OC);
            ggml_tensor * y = ggml_mul_mat(ctx, w_2d, im2col_2d);
            // y has ne=[OC, T] — already the wt layout.
            if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
            return y;
        }
    }
    // Fallback: legacy [T, OC] matmul + explicit cont(transpose) to
    // produce [OC, T] for the caller.  CPU also lands here (and gets
    // the cblas fast path for free via dense_matmul_time_ggml).
    ggml_tensor * y_tc = dense_matmul_time_ggml(ctx, x, w, b);
    return ggml_cont(ctx, ggml_transpose(ctx, y_tc));
}

ggml_tensor * bias_gelu_ggml(ggml_context * ctx, ggml_tensor * x, ggml_tensor * b) {
    const bool use_cpu_custom = supertonic_use_cpu_custom_ops();
    // Fused-op fast path (any backend that registers
    // GGML_OP_SUPERTONIC_BIAS_GELU — Metal does via the local ggml port
    // overlay; CPU's ggml_compute_forward_supertonic_bias_gelu is the
    // parity backstop).  Replaces the add(bias) + gelu_erf chain
    // (2 dispatches on Metal) with one dispatch.  Override with
    // SUPERTONIC_DISABLE_FUSED_BIAS_GELU=1 to force the stock-op chain.
    // Skipped on CPU custom-op backends (cblas path below is faster).
    static const bool disable_fused_bias_gelu =
        std::getenv("SUPERTONIC_DISABLE_FUSED_BIAS_GELU") != nullptr;
    if (!use_cpu_custom && supertonic_use_fused_supertonic_ops() && !disable_fused_bias_gelu &&
        x->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        b->ne[0] == x->ne[1] &&
        ggml_is_contiguous(x) && ggml_is_contiguous(b)) {
        return ggml_supertonic_bias_gelu(ctx, x, b);
    }
    // CPU-only fused bias + GELU; falls back to gelu(add(x, b)) on GPU.
    if (use_cpu_custom &&
        x->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && x->ne[2] == 1 && x->ne[3] == 1) {
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
    const bool use_cpu_custom = supertonic_use_cpu_custom_ops();
    // Fused-op fast path (any backend that registers
    // GGML_OP_SUPERTONIC_PW2_RESIDUAL — Metal does via the local ggml port
    // overlay; CPU's ggml_compute_forward_supertonic_pw2_residual is the
    // parity backstop).  Replaces the add(bias) + mul(gamma) + add(residual)
    // chain with one dispatch.  Override with
    // SUPERTONIC_DISABLE_FUSED_PW2_RESIDUAL=1 to force the stock-op chain.
    // Skipped on CPU custom-op backends (cblas fast path below is faster).
    static const bool disable_fused_pw2_residual =
        std::getenv("SUPERTONIC_DISABLE_FUSED_PW2_RESIDUAL") != nullptr;
    if (!use_cpu_custom && supertonic_use_fused_supertonic_ops() && !disable_fused_pw2_residual &&
        residual->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 &&
        b->type == GGML_TYPE_F32 && gamma->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        residual->ne[0] == x->ne[0] && residual->ne[1] == x->ne[1] &&
        b->ne[0] == x->ne[1] && gamma->ne[0] == x->ne[1] &&
        ggml_is_contiguous(residual) && ggml_is_contiguous(x) &&
        ggml_is_contiguous(b) && ggml_is_contiguous(gamma)) {
        return ggml_supertonic_pw2_residual(ctx, residual, x, b, gamma);
    }
    // CPU-only fused (bias + gamma + residual); falls back to the
    // 3-step add/mul/add chain on GPU.
    if (use_cpu_custom &&
        residual->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 &&
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

// Phase B2 full: [C, T]-layout pointwise (K=1) Conv1d as a direct matmul.
//
// pwconv1/pwconv2 weights load as Conv1d kernels with ne=[K=1, IC, OC, 1].
// With activations already in [C, T] layout (IC inner-most), the K=1
// dimension is degenerate and the convolution is just:
//
//   y[OC, T] = sum_IC w[IC, OC] * x[IC, T]
//
// which is exactly `ggml_mul_mat(w_2d=[IC, OC], x_2d=[IC, T])` — no
// im2col, no transpose, no pretranspose-cache lookup needed.  Result is
// f32 contiguous and directly consumable by the next [C, T] op.
//
// CPU is intentionally NOT routed here: AMX cblas_sgemm in the legacy
// path is faster than the equivalent ggml_mul_mat dispatch on Apple
// CPUs.  Caller's `vector_convnext_ggml_ct` already roundtrips on CPU.
ggml_tensor * pointwise_matmul_ct(ggml_context * ctx,
                                  ggml_tensor * x_ct,   // [IC, T, 1, 1]
                                  ggml_tensor * w,      // [1, IC, OC, 1]  (Conv1d K=1)
                                  ggml_tensor * b) {
    GGML_ASSERT(w->ne[0] == 1);            // K=1
    GGML_ASSERT(w->ne[1] == x_ct->ne[0]);  // IC match
    GGML_ASSERT(ggml_is_contiguous(w));
    ggml_tensor * w_2d = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);
    ggml_tensor * x_2d = ggml_reshape_2d(ctx, x_ct, x_ct->ne[0], x_ct->ne[1]);
    ggml_tensor * y = ggml_mul_mat(ctx, w_2d, x_2d);  // [OC, T]
    if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
    return y;
}

// Phase B2 full: ConvNeXt block operating on `[C, T]` activations end-to-end.
// All five fused custom Metal kernels have layout-flag plumbing landed in
// port-version 13; this block strings their `_ct` variants together so the
// activation tensor never needs to flip layout mid-block.  Used by callers
// that fuse a chain of N convnext blocks with a single entry permute
// `[T, C] -> [C, T]` before the loop and a single exit permute after — net
// savings = (N - 1) intra-block transposes per chain × 5 CFM steps.
//
// Input  x:   [C, T, 1, 1]  f32 contiguous
// Output    : [C, T, 1, 1]  f32 contiguous
//
// CPU backends fall through to the legacy `[T, C]` path: the `_ct` ops have
// CPU forward implementations but they would force AMX-cblas off, so on
// CPU we permute in/out around the legacy block to keep AMX engaged.
ggml_tensor * vector_convnext_ggml_ct(ggml_context * ctx,
                                      const supertonic_model & model,
                                      const std::string & p,
                                      ggml_tensor * x_ct,
                                      int dilation) {
    // CPU rounds-trips to the legacy [T,C] block for the AMX-cblas fast path;
    // backends WITHOUT the fused _ct supertonic ops (Vulkan / OpenCL) must also
    // take the legacy path, whose helpers carry pure-GGML fallbacks — otherwise
    // the _ct custom ops below are silently skipped on those backends.
    if (model_prefers_cpu_kernels(model) || !supertonic_use_fused_supertonic_ops()) {
        // CPU: roundtrip to [T, C], run legacy block (AMX cblas fast path),
        // roundtrip back.  Cheap on CPU because the permute is just a copy.
        ggml_tensor * x_tc = ggml_cont(ctx, ggml_permute(ctx, x_ct, 1, 0, 2, 3));
        ggml_tensor * y_tc = vector_convnext_ggml(ctx, model, p, x_tc, dilation);
        return ggml_cont(ctx, ggml_permute(ctx, y_tc, 1, 0, 2, 3));
    }

    // Helper: flatten leading-1 dims so per-channel tensors come out as [C].
    // Supertonic GGUFs ship bias/gamma/norm parameters as [C, 1, 1, 1] or
    // [1, C, 1, 1] depending on which PyTorch broadcast view they were
    // exported from.  The `_ct` ctors all assert `param->ne[0] == C_dim`, so
    // unflattened tensors break them.  This is the same shape mismatch that
    // has been silently disabling the legacy `pw2_residual_ggml` fused path
    // for ConvNeXt blocks all along.
    auto flatten_1d = [&](ggml_tensor * t) -> ggml_tensor * {
        const int64_t n = ggml_nelements(t);
        // Skip reshape only when already a literal 1-d view with ne[0] == n
        // (`ggml_n_dims` is unreliable here — it ignores leading-1 dims and
        // would return 1 for a [1, C, 1, 1] tensor where ne[0] = 1).
        if (t->ne[0] == n && t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1) {
            return t;
        }
        return ggml_reshape_1d(ctx, t, n);
    };

    ggml_tensor * residual = x_ct;
    // depthwise_1d_ct: [C, T] -> [C, T]
    ggml_tensor * y = ggml_supertonic_depthwise_1d_ct(ctx, x_ct,
        require_source_tensor(model, p + ".dwconv.weight"),
        flatten_1d(require_source_tensor(model, p + ".dwconv.bias")),
        dilation);
    // layer_norm_channel_ct: [C, T] -> [C, T]
    y = ggml_supertonic_layer_norm_channel_ct(ctx, y,
        flatten_1d(require_source_tensor(model, p + ".norm.norm.weight")),
        flatten_1d(require_source_tensor(model, p + ".norm.norm.bias")),
        1e-6f);
    // pw1 matmul: [IC=C, T] -> [OC, T]
    y = pointwise_matmul_ct(ctx, y,
        require_source_tensor(model, p + ".pwconv1.weight"),
        nullptr);
    // bias_gelu_ct: [OC, T] -> [OC, T]
    y = ggml_supertonic_bias_gelu_ct(ctx, y,
        flatten_1d(require_source_tensor(model, p + ".pwconv1.bias")));
    // pw2 matmul: [IC=OC, T] -> [C, T]   (restores channel count)
    y = pointwise_matmul_ct(ctx, y,
        require_source_tensor(model, p + ".pwconv2.weight"),
        nullptr);
    // pw2_residual_ct: x[C, T] + bias[C] (×) gamma[C] + residual[C, T] -> [C, T]
    return ggml_supertonic_pw2_residual_ct(ctx, y,
        flatten_1d(require_source_tensor(model, p + ".pwconv2.bias")),
        flatten_1d(require_source_tensor(model, p + ".gamma")),
        residual);
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
    // QVAC-18605 round 4 — generalised cache key for the K/V
    // flash-attention dispatch dtype.  Replaces the round-1
    // boolean `f16_kv_attn` (kept the field name for grep
    // continuity in PROGRESS_SUPERTONIC.md / git history; the
    // semantics are now an enum carrying f32/f16/bf16/q8_0).
    // Rebuilding the graph when this flips matches the same
    // correctness contract as the (q_len, kv_len, n_heads,
    // head_dim) cache keys above.  See dispatch logic in
    // `build_text_attention_cache()`.
    kv_attn_dtype kv_attn_type = kv_attn_dtype::f32;
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
    // Reuse the cached graph when it already matches this shape AND was built on
    // the direct backend path (cache.allocr non-null). The scheduler path leaves
    // cache.allocr null, so it always rebuilds from a clean graph
    // (ggml_backend_sched_alloc_graph mutates node->src[]). Mirrors run_hift_decode.
    if (cache.ctx && cache.allocr && cache.generation_id == model.generation_id
        && cache.q_len == q_len && cache.kv_len == kv_len
        && cache.n_heads == n_heads && cache.head_dim == head_dim
        && cache.out_w_source == out_w_source && cache.out_b_source == out_b_source) {
        return;
    }
    free_text_attention_cache(cache);
    cache.model = &model;
    cache.generation_id = model.generation_id;
    cache.q_len = q_len;
    cache.kv_len = kv_len;
    cache.n_heads = n_heads;
    cache.head_dim = head_dim;
    cache.kv_attn_type = supertonic_kv_attn_type();
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

    // QVAC-18605 round 4 — multi-dtype K/V flash-attention
    // dispatch.  Generalises the round-1 F16-only path:
    //
    //   f32  → no cast (backend's F32 flash-attn kernel)
    //   f16  → cast K / V to F16 (OpenCL `flash_attn_f32_f16`,
    //          Vulkan `kernel_flash_attn_f32_f16_*`; chatterbox
    //          --cfm-f16-kv-attn equivalent)
    //   bf16 → cast K / V to BF16 (Vulkan coopmat2 — wider
    //          exponent range than F16 at identical bandwidth)
    //   q8_0 → cast K / V to Q8_0 (Vulkan + half the K/V upload
    //          bandwidth; row stride of 32 elements is exact for
    //          our `head_dim = 64` so block alignment is trivially
    //          satisfied)
    //
    // Q stays F32 in every case: cheaper to keep one operand at
    // the higher precision than to round-trip the post-attention
    // output back through F32 for the downstream dense projection.
    //
    // The decision lives in `model.kv_attn_type` (mirrored onto
    // the thread-local by `supertonic_op_dispatch_scope` and
    // captured into `cache.kv_attn_type` above as the cache key).
    // Probe-gated graceful fallback to f32 happens upstream in
    // `resolve_kv_attn_type` — by the time we reach this site the
    // chosen dtype is guaranteed to be one the backend accepts
    // for our (head_dim, n_heads) shape.
    ggml_type cast_target = GGML_TYPE_COUNT;  // sentinel "no cast"
    switch (cache.kv_attn_type) {
        case kv_attn_dtype::f32:                                   break;
        case kv_attn_dtype::f16:  cast_target = GGML_TYPE_F16;     break;
        case kv_attn_dtype::bf16: cast_target = GGML_TYPE_BF16;    break;
        case kv_attn_dtype::q8_0: cast_target = GGML_TYPE_Q8_0;    break;
        case kv_attn_dtype::autoselect:
            // Resolver never returns autoselect; defensive throw
            // so a future refactor that bypasses the resolver
            // can't silently take the F32 path.
            throw std::runtime_error(
                "vector_text_attention_cache: kv_attn_type=autoselect "
                "leaked into dispatch (resolver should have produced "
                "a concrete dtype)");
    }
    if (cast_target != GGML_TYPE_COUNT) {
        ggml_tensor * k_typed = ggml_new_tensor_3d(cache.ctx, cast_target, head_dim, kv_len, n_heads);
        ggml_tensor * v_typed = ggml_new_tensor_3d(cache.ctx, cast_target, head_dim, kv_len, n_heads);
        k_in = ggml_cpy(cache.ctx, k_in, k_typed);
        v_in = ggml_cpy(cache.ctx, v_in, v_typed);
    }

    ggml_tensor * attn = ggml_flash_attn_ext(cache.ctx, q_in, k_in, v_in,
                                             nullptr, 1.0f/16.0f, 0.0f, 0.0f);
    attn = ggml_reshape_2d(cache.ctx, attn, static_cast<int64_t>(n_heads) * head_dim, q_len);
    ggml_tensor * ctx_tc = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, attn));
    ggml_set_name(ctx_tc, "vector_attn_ctx"); ggml_set_output(ctx_tc);
    ggml_build_forward_expand(cache.gf, ctx_tc);

    ggml_tensor * out = dense_matmul_time_pretransposed_ggml(cache.ctx, model, ctx_tc,
        require_source_tensor(model, out_w_source),
        require_source_tensor(model, out_b_source));
    ggml_set_name(out, "vector_attn_out"); ggml_set_output(out);
    ggml_build_forward_expand(cache.gf, out);

    // Allocation is per-call via the model scheduler (supertonic_sched_alloc
    // in run), which routes GGML_OP_CUSTOM ops to CPU. No per-cache gallocr;
    // cache.allocr stays null (free_*_cache's safe_gallocr_free no-ops on it).
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
    // QVAC-18605 round 4 — cache-key check includes kv_attn_type so a
    // mid-run --kv-attn-type override rebuilds the graph with the new
    // dtype.  Rebuild only on key mismatch; preserve the shape-cached
    // graph on every other call.
    if (cache.model != &model || cache.generation_id != model.generation_id ||
        cache.q_len != q_len || cache.kv_len != kv_len ||
        cache.n_heads != n_heads || cache.head_dim != head_dim ||
        cache.kv_attn_type != supertonic_kv_attn_type() ||
        cache.out_w_source != out_w_source || cache.out_b_source != out_b_source) {
        build_text_attention_cache(cache, model, q_len, kv_len, n_heads, head_dim, out_w_source, out_b_source);
    }
    // QVAC-19254 — direct backend path when every node is supported by
    // the primary backend; route through the scheduler when an op must
    // run on CPU (GGML_OP_CUSTOM etc.).
    bool direct = true;
    const int n_nodes = ggml_graph_n_nodes(cache.gf);
    for (int i = 0; i < n_nodes; ++i) {
        if (!ggml_backend_supports_op(model.backend, ggml_graph_node(cache.gf, i))) { direct = false; break; }
    }
    if (direct) {
        if (!cache.allocr) {
            cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new supertonic text attention failed");
            if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
                throw std::runtime_error("ggml_gallocr_reserve supertonic text attention failed");
            }
        }
        ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    } else {
        supertonic_sched_alloc(model, cache.gf);
    }
    ggml_backend_tensor_set(cache.q_tc_in, q_tc.data(), 0, q_tc.size()*sizeof(float));
    ggml_backend_tensor_set(cache.k_tc_in, k_tc.data(), 0, k_tc.size()*sizeof(float));
    ggml_backend_tensor_set(cache.v_tc_in, v_tc.data(), 0, v_tc.size()*sizeof(float));
    if (direct) profile_vector_compute(model, cache.gf, current_step, island);
    else        profile_vector_compute(model, cache.gf, current_step, island, /*use_sched=*/true);
    if (ctx_trace) *ctx_trace = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "vector_attn_ctx"));
    return tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "vector_attn_out"));
}

// Audit follow-up #6 (2C-lite) — GPU-input fast path for
// `run_text_attention_cache`.  Equivalent to the host-vector
// overload above but replaces the three `ggml_backend_tensor_set`
// uploads with `ggml_backend_tensor_copy` (same-backend device→
// device blit) so Q / K / V never round-trip through the host
// between the producing graph (front-block / group-graph / res-
// style QKV cache) and this attention cache.
//
// Eliminates per call: 3 GPU→host downloads + 3 host→GPU uploads.
// Across the four attention sites × 5 denoise steps × Q/K/V =
// 120 sync points / synth on the production path (independent of
// trace-mode downloads, which still happen for parity harnesses
// when `include_ggml_trace` is set at the call site).
//
// `q_src` / `k_src` / `v_src` MUST point into a graph that has
// already been computed on the same `model.backend` and whose
// allocator is still alive.  The current call pattern (one
// `run_*_cache` per site, computed immediately before this
// attention call) satisfies both.
//
// Test contract: `test/test_supertonic_graph_to_graph_blit.cpp`
// — two minimal cached graphs sharing one backend, parity vs the
// download / upload pair across all five vector-estimator attn
// shapes (front+g1/g2/g3 Q at L=20, style K at kv=50, L=1 trip-
// wire).
std::vector<float> run_text_attention_cache_gpu(vector_text_attention_cache & cache,
                                                const supertonic_model & model,
                                                ggml_tensor * q_src,
                                                ggml_tensor * k_src,
                                                ggml_tensor * v_src,
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
        cache.kv_attn_type != supertonic_kv_attn_type() ||
        cache.out_w_source != out_w_source || cache.out_b_source != out_b_source) {
        build_text_attention_cache(cache, model, q_len, kv_len, n_heads, head_dim, out_w_source, out_b_source);
    }
    // QVAC-19254 — direct vs scheduler routing.  build_text_attention_cache
    // no longer creates a gallocr; the run paths (this GPU-bridge variant
    // + the host-vector overload above) must do it themselves, otherwise
    // `cache.q_tc_in` / `k_tc_in` / `v_tc_in` have null backend buffers and
    // the subsequent `ggml_backend_tensor_copy` aborts with
    // "tensor buffer not set".  Mirrors the direct/sched dispatch in
    // `run_text_attention_cache` above.
    bool direct = true;
    {
        const int n_nodes = ggml_graph_n_nodes(cache.gf);
        for (int i = 0; i < n_nodes; ++i) {
            if (!ggml_backend_supports_op(model.backend, ggml_graph_node(cache.gf, i))) { direct = false; break; }
        }
    }
    if (direct) {
        if (!cache.allocr) {
            cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new supertonic text attention (gpu bridge) failed");
            if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
                throw std::runtime_error("ggml_gallocr_reserve supertonic text attention (gpu bridge) failed");
            }
        }
        ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    } else {
        supertonic_sched_alloc(model, cache.gf);
    }
    // Same-backend device→device blits.  ggml_backend_tensor_copy
    // checks `ggml_nbytes(src) == ggml_nbytes(dst)` internally and
    // dispatches the backend's `cpy_tensor_async` path (CPU →
    // memcpy, OpenCL → clEnqueueCopyBuffer, etc.).  No host
    // synchronisation between the three copies; the next graph
    // compute happens-before-orders them via the same backend
    // queue.
    ggml_backend_tensor_copy(q_src, cache.q_tc_in);
    ggml_backend_tensor_copy(k_src, cache.k_tc_in);
    ggml_backend_tensor_copy(v_src, cache.v_tc_in);
    if (direct) profile_vector_compute(model, cache.gf, current_step, island);
    else        profile_vector_compute(model, cache.gf, current_step, island, /*use_sched=*/true);
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
    std::vector<float> q;        // pre-RoPE Q (kept for scalar-parity trace)
    std::vector<float> k;        // pre-RoPE K
    std::vector<float> v;
    // F23 — when the cache has `apply_rope = true` these hold the
    // post-RoPE Q/K downloaded from the in-graph rotation outputs
    // (`<q_name>_rope` / `<k_name>_rope`).  Call sites pass these
    // directly to `run_text_attention_cache` instead of calling
    // host-side `apply_rope(theta, …)` on q/k.  Empty when the
    // legacy fallback path is taken (model lacks `vector_rope_theta`).
    std::vector<float> q_rope;
    std::vector<float> k_rope;

    // Audit follow-up #6 (2C-lite) — GPU-side handles for the
    // post-RoPE Q/K and raw V tensors.  Pointers are valid as
    // long as the producing `vector_group_graph_cache` (or
    // `front_block_proj_cache` for the attn0 site) is still
    // alive and hasn't been rebuilt.  Call sites feed these
    // directly into `run_text_attention_cache_gpu` to skip the
    // download / upload pair.  Null when no graph executed (legacy
    // path with `apply_rope = false` falls back to the host-vector
    // members above).
    ggml_tensor * q_rope_gpu = nullptr;
    ggml_tensor * k_rope_gpu = nullptr;
    ggml_tensor * v_gpu      = nullptr;
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
    // QVAC-18605 round 12 #5 — host-pinned input scratchpad.
    // Holds ONLY `x_in` + `temb_in` (the two hot per-step inputs
    // uploaded fresh every denoise step).  On Vulkan, allocated
    // via `try_alloc_inputs_in_pinned_host_buffer` which returns
    // a buffer from `ggml_backend_vk_host_buffer_type()` — every
    // `ggml_backend_tensor_set(x_in, ...)` skips one staging-
    // buffer hop on the way to BAR-mapped GPU memory.  On CPU
    // / Metal / OpenCL (no host buffer type) the helper returns
    // nullptr and we fall back to allocating the same tensors
    // via `ggml_backend_alloc_ctx_tensors(input_ctx, backend)`
    // — same memory, just one staging hop per upload.
    //
    // `text_in` stays in the main `ctx` (gallocr handles it)
    // because it's upload-skipped by the round-10 tracker on
    // steps 1..N-1; the marginal staging-hop saving doesn't
    // amortise across the cold-miss / fast-path mix.
    std::vector<uint8_t> input_ctx_storage;
    ggml_context * input_ctx = nullptr;
    ggml_backend_buffer_t input_buf = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * temb_in = nullptr;
    ggml_tensor * text_in = nullptr;

    // Audit follow-up #5 / F23 — in-graph RoPE inputs.  Populated
    // at cache-build time and uploaded once (cos/sin only depend on
    // L / text_len / θ, all stable across the cache's lifetime).
    // When `apply_rope == false` (no `vector_rope_theta` available,
    // e.g. a malformed GGUF) the graph falls back to the historical
    // path: Q/K stay raw, host code still calls apply_rope.  See
    // `aiDocs/AUDIT_SUPERTONIC_OPENCL.md` F23.
    bool apply_rope = false;
    ggml_tensor * q_cos_in = nullptr;
    ggml_tensor * q_sin_in = nullptr;
    ggml_tensor * k_cos_in = nullptr;
    ggml_tensor * k_sin_in = nullptr;
    std::string q_rope_name; // == q_name + "_rope"
    std::string k_rope_name; // == k_name + "_rope"

    // QVAC-18605 round 10 — pointer-compare upload-skip tracker
    // for `text_in`.  `text_lc_host` is the same `text_emb`
    // pointer the front-block cache sees: stable within one
    // synth (5 calls × same pointer), potentially reused-at-same-
    // address across synths.  Caller resets at `current_step ==
    // 0` to invalidate the cache.  See upload_skip_tracker
    // contract in supertonic_internal.h.  Cache rebuild zeroes
    // this via `cache = {}` (effective reset).
    upload_skip_tracker text_in_skip;
};

void free_group_graph_cache(vector_group_graph_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    // QVAC-18605 round 12 #5 — tear down the host-pinned input
    // scratchpad.  Order matters: free the gallocr first (it
    // owns buffers for the main-ctx tensors), then the main
    // ctx (which holds the graph metadata referencing x_in /
    // temb_in pointers from `input_ctx`), then the input
    // buffer (drops the host-pinned pages), then the input
    // ctx (drops the tensor metadata).  Freeing input_ctx
    // BEFORE the gallocr would leave the gallocr with
    // dangling pointers to tensors that no longer exist.
    if (cache.ctx) ggml_free(cache.ctx);
    if (cache.input_buf) ggml_backend_buffer_free(cache.input_buf);
    if (cache.input_ctx) ggml_free(cache.input_ctx);
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
    // Reuse the cached graph when it already matches this shape AND was built on
    // the direct backend path (cache.allocr non-null). The scheduler path leaves
    // cache.allocr null, so it always rebuilds. Mirrors run_hift_decode.
    if (cache.ctx && cache.allocr && cache.generation_id == model.generation_id
        && cache.L == L && cache.C == C && cache.text_len == text_len
        && cache.group == group && cache.conv_block == conv_block
        && cache.linear_block == linear_block && cache.post_block == post_block
        && cache.trace_outputs == trace_outputs && cache.matmul_source == matmul_source
        && cache.q_matmul_source == q_matmul_source && cache.k_matmul_source == k_matmul_source
        && cache.v_matmul_source == v_matmul_source && cache.q_name == q_name
        && cache.k_name == k_name && cache.v_name == v_name) {
        return;
    }
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

    // F12: ingest the group graph's primary activation in
    // CPU-native `[C, L]` (channel-fast) layout so callers can
    // upload `x_tc` byte-for-byte without the per-call host
    // `pack_time_channel_for_ggml` loop.  The graph's first op
    // is an `ggml_cont(ggml_transpose(...))` that materialises
    // the `[L, C]` layout downstream `vector_convnext_ggml` /
    // `dense_matmul_time_ggml` builders already consume.  See
    // `supertonic_internal.h::transpose_time_channel_ggml` for
    // the bit-exact equivalence proof against the host pack.
    //
    // QVAC-18605 round 12 #5 — `x_in` + `temb_in` live in a
    // SEPARATE ggml_context (`cache.input_ctx`) so they can be
    // allocated from `ggml_backend_vk_host_buffer_type()` on
    // Vulkan and skip the staging-buffer hop on every per-step
    // `ggml_backend_tensor_set`.  Graph tensors in `cache.ctx`
    // reference these by pointer (ggml stores tensors as `void *`
    // in the graph regardless of which context allocated them);
    // gallocr's `ggml_gallocr_reserve` + `ggml_gallocr_alloc_graph`
    // skips tensors that already have a `tensor->buffer` set, so
    // pre-binding them in the host buffer doesn't interfere with
    // gallocr's allocation pass for the intermediates + outputs.
    //
    // `text_in` STAYS in `cache.ctx` because the round-10
    // upload-skip tracker means steps 1..N-1 don't upload at
    // all; the marginal staging-hop saving for the single cold-
    // miss step doesn't amortise.
    {
        // 8 tensor slots is well over what's needed (2 inputs);
        // padded so future round-12 follow-ups can add more
        // host-pinned inputs without re-tuning the size.
        const size_t INPUT_OVERHEAD = ggml_tensor_overhead() * 8;
        cache.input_ctx_storage.assign(INPUT_OVERHEAD, 0);
        ggml_init_params input_p = { INPUT_OVERHEAD, cache.input_ctx_storage.data(), /*no_alloc=*/true };
        cache.input_ctx = ggml_init(input_p);
        cache.x_in = ggml_new_tensor_2d(cache.input_ctx, GGML_TYPE_F32, C, L);
        ggml_set_name(cache.x_in, "vector_group_in_tc"); ggml_set_input(cache.x_in);
        cache.temb_in = ggml_new_tensor_1d(cache.input_ctx, GGML_TYPE_F32, 64);
        ggml_set_name(cache.temb_in, "vector_group_temb"); ggml_set_input(cache.temb_in);
        // QVAC-18605 round 13 #1 — consolidated allocator
        // (round-12 inlined the try-pinned-host + fallback
        // boilerplate at 4 sites; this round factors it out).
        cache.input_buf = alloc_input_scratchpad_or_throw(
            model, cache.input_ctx, "vector_group_graph_cache");
    }
    cache.text_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, text_len, 256);
    ggml_set_name(cache.text_in, "vector_group_text");
    // Same round-10 upload-skip pattern as the front-cache: `text_in`
    // is uploaded once per synth (`current_step == 0` resets, every
    // other step skips).  Mark INPUT + OUTPUT so the buffer survives
    // gallocr's free pass — without OUTPUT, step 0's compute frees
    // the buffer for intermediate reuse, and the step-1..N skipped
    // upload reads stale data.  See the matching note on
    // `front_cache.text_in_t` in `supertonic_vector_trace_proj_ggml`.
    ggml_set_input(cache.text_in);  ggml_set_output(cache.text_in);

    ggml_tensor * cur = transpose_time_channel_ggml(cache.ctx, cache.x_in);
    ggml_set_name(cur, "vector_group_in");
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
    // F6: pre-transposed companion lives in model.ctx_w under
    // `<matmul_source>__T` (populated at load).  Falls back to the
    // per-pointer `pretransposed_weights` map (Metal's broader Q/K/V
    // pretranspose roster), and finally to an in-graph
    // `ggml_cont(ggml_transpose(W))` rewrite if neither covers this
    // weight.
    ggml_tensor * t_proj;
    {
        auto pretrans_it = model.source_tensors.find(matmul_source + "__T");
        ggml_tensor * w_t = (pretrans_it != model.source_tensors.end()) ? pretrans_it->second : nullptr;
        if (!w_t) {
            ggml_tensor * t_proj_w_orig = require_source_tensor(model, matmul_source);
            w_t = try_pretransposed_weight(model, t_proj_w_orig);
            if (!w_t) {
                w_t = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, t_proj_w_orig));
            }
        }
        t_proj = ggml_mul_mat(cache.ctx, w_t,
            ggml_reshape_2d(cache.ctx, cache.temb_in, 64, 1));
    }
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
    ggml_tensor * q = dense_matmul_time_pretransposed_ggml(cache.ctx, model, cur,
        require_source_tensor(model, q_matmul_source),
        require_source_tensor(model, attn_prefix + "W_query.linear.bias"));
    ggml_tensor * k = dense_matmul_time_pretransposed_ggml(cache.ctx, model, cache.text_in,
        require_source_tensor(model, k_matmul_source),
        require_source_tensor(model, attn_prefix + "W_key.linear.bias"));
    // QVAC-18966 — pack V into the layout the downstream
    // `run_text_attention_cache_gpu` consumes via
    // `ggml_backend_tensor_copy(v_src, v_tc_in)`.  `v_tc_in` is
    // `ggml_new_tensor_2d(F32, A=HD, kv_len)` → ne=[HD, kv_len]
    // with natural strides nb=[elem, HD*elem] (time-major-flat
    // memory `data[c + t*HD]`).  `dense_matmul_time_(pre)ggml`
    // produces ne=[L_kv, HD] with channel-major-flat memory
    // (`data[t + c*L_kv]`) — the byte-for-byte transpose of what
    // the bridge expects.  `ggml_cont(ggml_transpose(...))` flips
    // the strides + materialises a contiguous fresh tensor with
    // the right layout.  Mirrors the head-of-pipeline transpose
    // inside `apply_rope_to_packed_qk` so Q-rope / K-rope / V all
    // land in `q_tc_in` / `k_tc_in` / `v_tc_in` bit-exactly.  See
    // the header doc on `apply_rope_to_packed_qk` in
    // `supertonic_internal.h` for the full layout reasoning.
    //
    // Note (Vulkan branch): master's
    // `dense_matmul_time_pretransposed_ggml` upgrade only pre-
    // transposes WEIGHTS, not the activation layout, so the
    // output ne=[T, OC] channel-major-flat stays identical to
    // the legacy `dense_matmul_time_ggml`.  The same
    // `ggml_cont(ggml_transpose(...))` head-of-V-pipeline fix
    // therefore lands the right bytes for both variants.
    //
    // Legacy host bridge: `tensor_raw_f32(v_gpu)` downloads the
    // post-transpose bytes (time-major-flat `out[t*HD + c]`) —
    // bit-identical to what scalar `apply_rope`'s reference loop
    // produces and what every legacy `push_trace`-consuming
    // harness expects (callers updated in lock-step).
    ggml_tensor * v_matmul = dense_matmul_time_pretransposed_ggml(cache.ctx, model, cache.text_in,
        require_source_tensor(model, v_matmul_source),
        require_source_tensor(model, attn_prefix + "W_value.linear.bias"));
    ggml_tensor * v = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, v_matmul));
    ggml_set_name(q, q_name.c_str()); ggml_set_output(q); ggml_build_forward_expand(cache.gf, q);
    ggml_set_name(k, k_name.c_str()); ggml_set_output(k); ggml_build_forward_expand(cache.gf, k);
    ggml_set_name(v, v_name.c_str()); ggml_set_output(v); ggml_build_forward_expand(cache.gf, v);

    // F23 — bake the RoPE rotation into the same graph that
    // produces Q/K, so the host path drops the per-step CPU
    // `apply_rope(theta, q_out, …)` round-trips entirely.  Q's
    // sequence length is `L` (latent_len) and K's is `text_len`;
    // each gets its own cos/sin table input (`ne=[half, L]` /
    // `ne=[half, text_len]`) populated once at build time.  The
    // post-rotation tensors are exposed under
    // `<q_name>_rope` / `<k_name>_rope` so trace harnesses can
    // download both the pre- and post-RoPE values for parity
    // checks against the scalar path.  Falls back to no-op when
    // the GGUF didn't ship a `vector_rope_theta` (cache.apply_rope
    // stays false; call sites then keep the legacy host
    // apply_rope call).
    const int H = model.hparams.vector_text_attn_heads;  // v1/v2=4, v3=8
    const int D = 64;
    const int half = D / 2;
    cache.apply_rope = (int) model.vector_rope_theta.size() == half;
    if (cache.apply_rope) {
        // RoPE cos/sin tables are constants for the cache's (L, text_len,
        // θ) key — uploaded once at build time and never per-call.  Mark
        // as both INPUT and OUTPUT so gallocr doesn't free the buffer
        // after the first compute pass (without OUTPUT, the leaf-input
        // buffer is released for intermediate reuse on the next compute,
        // silently corrupting the cos/sin data on the second call).
        cache.q_cos_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, half, L);
        ggml_set_name(cache.q_cos_in,
            ("vector_group_q_rope_cos_g" + std::to_string(group)).c_str());
        ggml_set_input(cache.q_cos_in);  ggml_set_output(cache.q_cos_in);
        cache.q_sin_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, half, L);
        ggml_set_name(cache.q_sin_in,
            ("vector_group_q_rope_sin_g" + std::to_string(group)).c_str());
        ggml_set_input(cache.q_sin_in);  ggml_set_output(cache.q_sin_in);
        cache.k_cos_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, half, text_len);
        ggml_set_name(cache.k_cos_in,
            ("vector_group_k_rope_cos_g" + std::to_string(group)).c_str());
        ggml_set_input(cache.k_cos_in);  ggml_set_output(cache.k_cos_in);
        cache.k_sin_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, half, text_len);
        ggml_set_name(cache.k_sin_in,
            ("vector_group_k_rope_sin_g" + std::to_string(group)).c_str());
        ggml_set_input(cache.k_sin_in);  ggml_set_output(cache.k_sin_in);

        ggml_tensor * q_rope = apply_rope_to_packed_qk(cache.ctx, q,
            cache.q_cos_in, cache.q_sin_in, H, D);
        ggml_tensor * k_rope = apply_rope_to_packed_qk(cache.ctx, k,
            cache.k_cos_in, cache.k_sin_in, H, D);
        cache.q_rope_name = q_name + "_rope";
        cache.k_rope_name = k_name + "_rope";
        ggml_set_name(q_rope, cache.q_rope_name.c_str());
        ggml_set_output(q_rope);
        ggml_build_forward_expand(cache.gf, q_rope);
        ggml_set_name(k_rope, cache.k_rope_name.c_str());
        ggml_set_output(k_rope);
        ggml_build_forward_expand(cache.gf, k_rope);
    }

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new vector group cache failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve vector group cache failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);

    // Upload the cos/sin tables — these inputs are stable for the
    // entire cache lifetime (cos/sin depend only on L / text_len /
    // θ, all encoded in the cache key + the model), so this is a
    // one-shot population.
    if (cache.apply_rope) {
        std::vector<float> q_cos, q_sin, k_cos, k_sin;
        make_rope_cos_sin_tables(model.vector_rope_theta.data(), L, half,
                                 q_cos, q_sin);
        make_rope_cos_sin_tables(model.vector_rope_theta.data(), text_len, half,
                                 k_cos, k_sin);
        ggml_backend_tensor_set(cache.q_cos_in, q_cos.data(), 0, q_cos.size() * sizeof(float));
        ggml_backend_tensor_set(cache.q_sin_in, q_sin.data(), 0, q_sin.size() * sizeof(float));
        ggml_backend_tensor_set(cache.k_cos_in, k_cos.data(), 0, k_cos.size() * sizeof(float));
        ggml_backend_tensor_set(cache.k_sin_in, k_sin.data(), 0, k_sin.size() * sizeof(float));
    }
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
    // QVAC-18605 — cache-key check (skip rebuild when shape/sources/
    // trace flag haven't changed).  Build is expensive on the hot
    // denoise-step path; the steady-state synth pays one rebuild on
    // the cold-miss step, zero on every subsequent step.
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
    // QVAC-19254 — direct vs scheduler routing: when every node is
    // supported by the primary backend, use the per-cache gallocr +
    // direct compute; when an op must run on CPU (GGML_OP_CUSTOM),
    // fall through to the model scheduler.
    //
    // HEAD's `build_group_graph_cache` already creates cache.allocr +
    // calls `ggml_gallocr_alloc_graph` AND uploads the cache-lifetime
    // RoPE cos/sin constants right after.  Re-calling alloc_graph
    // here would clobber those uploaded constants (gallocr rebinds
    // tensor offsets and the freshly-allocated buffer doesn't carry
    // build-time data forward).  So on direct path: only allocate
    // the gallocr lazily IF the build didn't (defensive — every
    // current build path does), and never re-alloc.
    bool direct = true;
    const int n_nodes = ggml_graph_n_nodes(cache.gf);
    for (int i = 0; i < n_nodes; ++i) {
        if (!ggml_backend_supports_op(model.backend, ggml_graph_node(cache.gf, i))) { direct = false; break; }
    }
    if (direct) {
        if (!cache.allocr) {
            cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new supertonic group graph failed");
            if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
                throw std::runtime_error("ggml_gallocr_reserve supertonic group graph failed");
            }
            ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
        }
    } else {
        supertonic_sched_alloc(model, cache.gf);
    }
    // F12: cache.x_in is now ne=[C, L] (CPU-native time-major).
    // Upload `x_tc` directly — the host pack loop is gone; the
    // graph runs `ggml_cont(ggml_transpose(...))` to recover the
    // [L, C] layout downstream ops expect.
    ggml_backend_tensor_set(cache.x_in, x_tc.data(), 0, x_tc.size()*sizeof(float));
    ggml_backend_tensor_set(cache.temb_in, temb.data(), 0, temb.size()*sizeof(float));
    // QVAC-18605 round 10 — text_lc_host upload-skip.  Same
    // `text_emb` pointer that the front-block cache sees: stable
    // within one synth (5 calls × same pointer), potentially
    // reused-at-same-address across synths.  Synth-boundary reset
    // on `current_step == 0` invalidates the cache so the next
    // synth's first step always uploads.  Per-synth wins:
    // 4 (skipped) × 3 (groups) × text_len × 256 × 4 bytes.  See
    // upload_skip_tracker contract in supertonic_internal.h.
    if (current_step == 0) cache.text_in_skip.reset();
    if (cache.text_in_skip.needs_upload(text_lc_host)) {
        ggml_backend_tensor_set(cache.text_in, text_lc_host, 0, (size_t) text_len * 256 * sizeof(float));
        cache.text_in_skip.mark_uploaded(text_lc_host);
    }
    if (direct) profile_vector_compute(model, cache.gf, current_step, island);
    else        profile_vector_compute(model, cache.gf, current_step, island, /*use_sched=*/true);
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
    // F23: on trace runs we still download the pre-RoPE Q/K so the
    // scalar-parity harness can compare them against its own scalar
    // `ve_g<n>_attn_q` reference.  Production runs don't push these
    // through PUSH_GGML_TRACE so the download is the only cost.
    // The post-RoPE Q/K (`q_rope` / `k_rope`) are what callers feed
    // into `run_text_attention_cache`, eliminating the per-step
    // host `apply_rope(theta, …)` round-trips entirely.
    // 2C-lite — expose the GPU-side handles so the attention
    // call site can `ggml_backend_tensor_copy` directly into its
    // own cache.  Pointers are valid until the next rebuild of
    // this cache (i.e., until L/C/text_len/group/... changes).
    // The host downloads of q_rope/k_rope/v_gpu are now gated on
    // `trace != nullptr` for the FAST path (apply_rope == true)
    // because the production path no longer reads `out.q_rope` /
    // `out.k_rope` / `out.v` — it consumes `*_gpu` instead via
    // `run_text_attention_cache_gpu`.  The LEGACY path
    // (apply_rope == false; e.g. malformed GGUF without
    // vector_rope_theta) still needs q/k/v on the host because it
    // calls scalar `apply_rope` and the host `run_text_attention_
    // cache` overload.
    if (cache.apply_rope) {
        out.q_rope_gpu = ggml_graph_get_tensor(cache.gf, cache.q_rope_name.c_str());
        out.k_rope_gpu = ggml_graph_get_tensor(cache.gf, cache.k_rope_name.c_str());
    }
    out.v_gpu = ggml_graph_get_tensor(cache.gf, v_name.c_str());

    const bool need_host_qkv = (trace != nullptr) || !cache.apply_rope;
    if (need_host_qkv) {
        // Trace harnesses want pre-RoPE Q/K + V for the
        // `push_trace` block below and the call-site
        // `PUSH_GGML_TRACE({"ve_g*_attn_v", …})` push.  The legacy
        // host-RoPE fallback consumes them directly.
        //
        // Q / K matmul outputs are UNCHANGED ne=[L, HD] / ne=[text_
        // len, HD] channel-major-flat memory, so `tensor_to_time_
        // channel` is the right call (decodes col=c, row=t at
        // `c*L + t` into out[t*HD + c]).
        out.q = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, q_name.c_str()));
        out.k = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, k_name.c_str()));
        // QVAC-18966 — V is now graph-packed to ne=[HD, text_len]
        // time-major-flat by the head-of-V transpose in
        // `build_group_graph_cache`.  `tensor_raw_f32` downloads
        // the bytes in the layout scalar `apply_rope` /
        // `flash_attention_qkv` host references expect
        // (`v[t*HD + c]`).  `tensor_to_time_channel` would now
        // mis-interpret the swapped ne (reading HD as L_var and
        // L as C_var) and silently feed wrong-orientation V into
        // the attention.  See the header doc on
        // `apply_rope_to_packed_qk` in `supertonic_internal.h`.
        out.v = tensor_raw_f32(ggml_graph_get_tensor(cache.gf, v_name.c_str()));
    }
    if (trace && cache.apply_rope) {
        // Trace-only extra downloads — post-RoPE Q/K mirrors the
        // call site's `PUSH_GGML_TRACE({"ve_g*_attn_q_rope", …})`.
        //
        // QVAC-18966 — post-fix layout contract:
        // `apply_rope_to_packed_qk` now produces ne=[HD, L] with
        // time-major-flat memory (`data[c + t*HD]`).  Those bytes
        // ARE the scalar `apply_rope`'s native flat layout
        // (`out[t*HD + c]`), so `tensor_raw_f32` downloads them
        // directly — no transpose needed.  `tensor_to_time_channel`
        // would mis-interpret the new ne shape (reading `HD` as
        // L_var and `L` as C_var) and produce the transpose of
        // the transpose.  See the header doc on
        // `apply_rope_to_packed_qk` in `supertonic_internal.h`.
        out.q_rope = tensor_raw_f32(
            ggml_graph_get_tensor(cache.gf, cache.q_rope_name.c_str()));
        out.k_rope = tensor_raw_f32(
            ggml_graph_get_tensor(cache.gf, cache.k_rope_name.c_str()));
    }
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

    // QVAC-18605 round 9 — GPU-side handles for the post-projection
    // style Q / K / V tensors so the next-stage style flash-attn
    // call site (`run_text_attention_cache_gpu`) can blit them
    // device→device instead of round-tripping through `sq` / `sk`
    // / `sv` host vectors.  Same lifetime + dispatch pattern as
    // `vector_group_graph_result::q_rope_gpu` / `v_gpu` (round-1
    // 2C-lite for text attention; rounds 8 + 9 extend to front-
    // block + style sites).
    //
    // Pointers are valid as long as the producing
    // `vector_res_style_qkv_cache` is alive and hasn't been
    // rebuilt (cache is `thread_local` at every call site;
    // rebuild only on shape / matmul-source change).
    //
    // Always populated by `run_res_style_qkv_cache` (cheap —
    // just `ggml_graph_get_tensor`); the host vectors above are
    // gated on `trace != nullptr` (production path skips the
    // download because it consumes `*_gpu` instead).  `post`
    // stays unconditional — consumed by the next-stage
    // `run_style_residual_cache` which still expects a host
    // vector (cross-stage GPU bridge for `post` is deferred —
    // see `aiDocs/PLAN_VULKAN_NEXT_ROUNDS.md`).
    ggml_tensor * sq_gpu = nullptr;
    ggml_tensor * sk_gpu = nullptr;
    ggml_tensor * sv_gpu = nullptr;
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

    // Audit F4 — skip the re-upload of `style_v_in` and `kctx_in`
    // when the caller hands us the same host vectors as the
    // previous call.  `cached_style_layouts` returns a stable
    // pointer keyed on (model.generation_id, style_ttl), so the
    // pointer comparison is a sound "same data" proxy.
    // Steady-state per synth: 4 caches × 5 steps = 20 invocations,
    // 1 cold-miss upload per cache, then ≥4 × (5−1) = 16 skipped.
    // Across synths with the same voice: zero uploads after the
    // first synth.  See AUDIT_SUPERTONIC_OPENCL.md F4.
    const std::vector<float> * last_style_v_raw_uploaded = nullptr;
    const std::vector<float> * last_kctx_raw_uploaded = nullptr;
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
    // Reuse the cached graph when it already matches this shape AND was built on
    // the direct backend path (cache.allocr non-null). The scheduler path leaves
    // cache.allocr null, so it always rebuilds. Mirrors run_hift_decode.
    if (cache.ctx && cache.allocr && cache.generation_id == model.generation_id
        && cache.L == L && cache.C == C && cache.norm_block == norm_block
        && cache.post_block == post_block && cache.style_block == style_block
        && cache.trace_outputs == trace_outputs && cache.q_matmul_source == q_matmul_source
        && cache.k_matmul_source == k_matmul_source && cache.v_matmul_source == v_matmul_source
        && cache.residual_name == residual_name && cache.norm_name == norm_name
        && cache.post_name == post_name && cache.q_name == q_name
        && cache.k_name == k_name && cache.v_name == v_name) {
        return;
    }
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

    // F12: lhs / rhs ingested in CPU-native `[C, L]` channel-fast
    // layout — `run_res_style_qkv_cache` uploads `lhs_tc` / `rhs_tc`
    // directly, no host pack.  `style_v_in` / `kctx_in` are already
    // shaped `[50, 256]` (i.e. `[ttl_len=L_ttl, C_style=256]`) and
    // come from `cached_style_layouts(...)`, which produces stable
    // c-major buffers shared across all 4 style residual sites —
    // those keep their existing layout to preserve the F4 pointer-
    // compare upload-skip optimization.
    cache.lhs_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, C, L);
    ggml_set_name(cache.lhs_in, "res_style_lhs_tc"); ggml_set_input(cache.lhs_in);
    cache.rhs_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, C, L);
    ggml_set_name(cache.rhs_in, "res_style_rhs_tc"); ggml_set_input(cache.rhs_in);
    // style_v_in / kctx_in use the F4 pointer-compare upload-skip — the
    // host pointer is stable across calls within one synth, so they're
    // uploaded only on cold miss / pointer change.  That assumption
    // requires the backend buffer to ALSO be stable.  gallocr frees
    // leaf inputs once their last consumer runs, releasing the buffer
    // for intermediate reuse on the next compute pass.  Mark INPUT +
    // OUTPUT so the buffer is kept alive and the skip-upload optimisation
    // actually preserves the uploaded data.
    cache.style_v_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, 50, 256);
    ggml_set_name(cache.style_v_in, "res_style_ttl_lc");
    ggml_set_input(cache.style_v_in);  ggml_set_output(cache.style_v_in);
    cache.kctx_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, 50, 256);
    ggml_set_name(cache.kctx_in, "res_style_kctx_lc");
    ggml_set_input(cache.kctx_in);  ggml_set_output(cache.kctx_in);

    ggml_tensor * lhs_lc = transpose_time_channel_ggml(cache.ctx, cache.lhs_in);
    ggml_tensor * rhs_lc = transpose_time_channel_ggml(cache.ctx, cache.rhs_in);
    ggml_set_name(lhs_lc, "res_style_lhs");
    ggml_set_name(rhs_lc, "res_style_rhs");
    ggml_tensor * res = ggml_add(cache.ctx, lhs_lc, rhs_lc);
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
    // Round 11 sq/sk/sv layout fix layered on top of master's
    // `dense_matmul_time_pretransposed_ggml` upgrade.  Same
    // reasoning as the front-block V site above: pretransposed
    // variant still produces ne=[T, OC] channel-major-flat
    // memory; the round-11 `ggml_cont(ggml_transpose(...))`
    // below this block remains required to land bytes in the
    // ne=[HD, L] time-major-flat layout `q_tc_in`/`k_tc_in`/
    // `v_tc_in` expect for the GPU-bridge blit.
    ggml_tensor * sq_matmul = dense_matmul_time_pretransposed_ggml(cache.ctx, model, post,
        require_source_tensor(model, q_matmul_source),
        require_source_tensor(model, style_prefix + "W_query.linear.bias"));
    ggml_tensor * sk_matmul = dense_matmul_time_pretransposed_ggml(cache.ctx, model, cache.kctx_in,
        require_source_tensor(model, k_matmul_source),
        require_source_tensor(model, style_prefix + "W_key.linear.bias"));
    sk_matmul = ggml_tanh(cache.ctx, sk_matmul);
    ggml_tensor * sv_matmul = dense_matmul_time_pretransposed_ggml(cache.ctx, model, cache.style_v_in,
        require_source_tensor(model, v_matmul_source),
        require_source_tensor(model, style_prefix + "W_value.linear.bias"));
    // QVAC-18605 follow-up — pack style Q/K/V into the time-major-
    // flat layout that `run_text_attention_cache_gpu` consumes via
    // `ggml_backend_tensor_copy`.  The style attention path has
    // no RoPE (cos/sin tables are absent for the style sites), so
    // the head-of-pipeline transpose inside
    // `apply_rope_to_packed_qk` doesn't run here — we open-code
    // it for each of the three matmul outputs.  Matmul output is
    // ne=[L_in, HD] channel-major-flat (`data[t + c*L_in]`);
    // `q_tc_in` / `k_tc_in` / `v_tc_in` in
    // `vector_text_attention_cache` are ne=[HD, L_in] time-major-
    // flat (`data[c + t*HD]`).  `ggml_cont(ggml_transpose(...))`
    // flips strides + materialises a contiguous fresh tensor
    // with the right layout.  See the header doc on
    // `apply_rope_to_packed_qk` in `supertonic_internal.h` for
    // the full reasoning.
    ggml_tensor * sq = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, sq_matmul));
    ggml_tensor * sk = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, sk_matmul));
    ggml_tensor * sv = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, sv_matmul));
    ggml_set_name(sq, q_name.c_str()); ggml_set_output(sq); ggml_build_forward_expand(cache.gf, sq);
    ggml_set_name(sk, k_name.c_str()); ggml_set_output(sk); ggml_build_forward_expand(cache.gf, sk);
    ggml_set_name(sv, v_name.c_str()); ggml_set_output(sv); ggml_build_forward_expand(cache.gf, sv);

    // Allocation is per-call via the model scheduler (supertonic_sched_alloc
    // in run), which routes GGML_OP_CUSTOM ops to CPU. No per-cache gallocr.
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
    // QVAC-18605 — cache-key check (skip rebuild on hot path).
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
    // QVAC-19254 — direct vs scheduler routing.
    bool direct = true;
    const int n_nodes = ggml_graph_n_nodes(cache.gf);
    for (int i = 0; i < n_nodes; ++i) {
        if (!ggml_backend_supports_op(model.backend, ggml_graph_node(cache.gf, i))) { direct = false; break; }
    }
    if (direct) {
        if (!cache.allocr) {
            cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new supertonic res style qkv failed");
            if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
                throw std::runtime_error("ggml_gallocr_reserve supertonic res style qkv failed");
            }
        }
        ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    } else {
        supertonic_sched_alloc(model, cache.gf);
    }
    // F12: direct upload of CPU-native `[L, C]` (time-major)
    // buffers — `cache.lhs_in` / `cache.rhs_in` are now `ne=[C, L]`
    // and the graph transposes them inside; no host pack.
    ggml_backend_tensor_set(cache.lhs_in, lhs_tc.data(), 0, lhs_tc.size() * sizeof(float));
    ggml_backend_tensor_set(cache.rhs_in, rhs_tc.data(), 0, rhs_tc.size() * sizeof(float));
    // F4: pointer-compare against the last successfully uploaded
    // host vector.  Cache rebuilds (above) reset last_*_uploaded
    // to nullptr via `cache = {}`, so the cold-miss path always
    // fires the upload regardless of pointer match.
    if (cache.last_style_v_raw_uploaded != &style_v_raw) {
        ggml_backend_tensor_set(cache.style_v_in, style_v_raw.data(), 0, style_v_raw.size() * sizeof(float));
        cache.last_style_v_raw_uploaded = &style_v_raw;
    }
    if (cache.last_kctx_raw_uploaded != &kctx_raw) {
        ggml_backend_tensor_set(cache.kctx_in, kctx_raw.data(), 0, kctx_raw.size() * sizeof(float));
        cache.last_kctx_raw_uploaded = &kctx_raw;
    }
    if (direct) profile_vector_compute(model, cache.gf, current_step, island);
    else        profile_vector_compute(model, cache.gf, current_step, island, /*use_sched=*/true);
    if (trace) {
        push_trace(*trace, residual_name, L, C, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, residual_name.c_str())));
        push_trace(*trace, norm_name, L, C, tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, norm_name.c_str())));
    }
    vector_res_style_qkv_result out;

    // QVAC-18605 round 9 — populate GPU handles for the post-
    // projection Q / K / V tensors unconditionally.  Cheap (no
    // GPU sync; just a name-to-pointer lookup in the cached
    // graph).  Lifetime contract documented on the struct.
    out.sq_gpu = ggml_graph_get_tensor(cache.gf, q_name.c_str());
    out.sk_gpu = ggml_graph_get_tensor(cache.gf, k_name.c_str());
    out.sv_gpu = ggml_graph_get_tensor(cache.gf, v_name.c_str());

    // `post` stays a host download — the next-stage
    // `run_style_residual_cache` still consumes a host vector.
    out.post = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, post_name.c_str()));

    // QVAC-18605 round 9 — gate `sq` / `sk` / `sv` host downloads
    // on trace mode.  Production path skips them because the
    // call site uses `out.sq_gpu` / `out.sk_gpu` / `out.sv_gpu`
    // via `run_text_attention_cache_gpu`.  Eliminates 3 sync
    // points per call × 4 sites × 5 denoise steps = 60 GPU→host
    // downloads / synth.  Mirrors the round-1 2C-lite
    // `need_host_qkv = (trace != nullptr)` gate on the group
    // graph cache.
    if (trace) {
        // QVAC-18605 follow-up — sq / sk / sv are now graph-packed
        // to ne=[HD, L] time-major-flat (see the matmul-output
        // transpose in `build_res_style_qkv_cache`).
        // `tensor_raw_f32` downloads the bytes in the layout
        // scalar reference and trace harnesses expect
        // (`out[t*256 + c]`).  See the header doc on
        // `apply_rope_to_packed_qk` in `supertonic_internal.h`.
        out.sq = tensor_raw_f32(out.sq_gpu);
        out.sk = tensor_raw_f32(out.sk_gpu);
        out.sv = tensor_raw_f32(out.sv_gpu);
        push_trace(*trace, post_name, L, C, out.post);
        push_trace(*trace, q_name, L, 256, out.sq);
        push_trace(*trace, k_name, 50, 256, out.sk);
        push_trace(*trace, v_name, 50, 256, out.sv);
    }
    return out;
}

// Audit finding F8 — cached "(add residual) + layer_norm" graph.
//
// The vector estimator's GGML production path runs four of these
// tiny graphs per step: one after each group's style-attention
// output to fold the style residual back into the main activation
// before the next group's convnext block runs.  Pre-audit, each
// call allocated a fresh `ggml_context`, `ggml_cgraph`, and
// `ggml_gallocr_t`, then freed them at the end.  Per synth that's
// 4 sites × 5 steps = 20 allocator churns; key is constant within
// a synth, so caching gets that down to 4 cold-miss rebuilds per
// model+L combination.
struct vector_style_residual_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int L = 0;
    int C = 0;
    int norm_block = 0;
    bool trace_outputs = false;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * lhs_in = nullptr;
    ggml_tensor * out_in = nullptr;
};

inline void free_style_residual_cache(vector_style_residual_graph_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

inline void build_style_residual_cache(vector_style_residual_graph_cache & cache,
                                       const supertonic_model & model,
                                       int L, int C, int norm_block, bool trace_outputs) {
    free_style_residual_cache(cache);
    cache.model = &model;
    cache.generation_id = model.generation_id;
    cache.L = L;
    cache.C = C;
    cache.norm_block = norm_block;
    cache.trace_outputs = trace_outputs;

    constexpr int NODES = 128;
    const size_t buf_size = ggml_tensor_overhead() * NODES +
                            ggml_graph_overhead_custom(NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params p = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(p);
    cache.gf = ggml_new_graph_custom(cache.ctx, NODES, false);

    // F12: ingest both residual operands in CPU-native `[C, L]`
    // layout — `run_style_residual_cache` uploads `lhs_tc` /
    // `out_tc` directly; the graph transposes both inside.
    cache.lhs_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, C, L);
    ggml_set_name(cache.lhs_in, "sr_lhs_in_tc"); ggml_set_input(cache.lhs_in);
    cache.out_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, C, L);
    ggml_set_name(cache.out_in, "sr_out_in_tc"); ggml_set_input(cache.out_in);

    ggml_tensor * lhs_lc = transpose_time_channel_ggml(cache.ctx, cache.lhs_in);
    ggml_tensor * out_lc = transpose_time_channel_ggml(cache.ctx, cache.out_in);
    ggml_set_name(lhs_lc, "sr_lhs");
    ggml_set_name(out_lc, "sr_out");
    ggml_tensor * res = ggml_add(cache.ctx, lhs_lc, out_lc);
    ggml_set_name(res, "sr_residual");
    if (trace_outputs) {
        ggml_set_output(res);
        ggml_build_forward_expand(cache.gf, res);
    }
    ggml_tensor * norm = layer_norm_ggml(cache.ctx, res,
        require_source_tensor(model, vector_main_block(norm_block) + ".norm.norm.weight"),
        require_source_tensor(model, vector_main_block(norm_block) + ".norm.norm.bias"));
    ggml_set_name(norm, "sr_norm"); ggml_set_output(norm);
    ggml_build_forward_expand(cache.gf, norm);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new style residual cache failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve style residual cache failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

inline std::vector<float> run_style_residual_cache(
    vector_style_residual_graph_cache & cache,
    const supertonic_model & model,
    const std::vector<float> & lhs_tc,
    const std::vector<float> & out_tc,
    int L, int C, int norm_block,
    int current_step, const char * island,
    std::vector<float> * residual_trace_out) {
    const bool want_trace = residual_trace_out != nullptr;
    if (cache.model != &model || cache.generation_id != model.generation_id ||
        cache.L != L || cache.C != C ||
        cache.norm_block != norm_block || cache.trace_outputs != want_trace) {
        build_style_residual_cache(cache, model, L, C, norm_block, want_trace);
    }
    // F12: direct upload — host pack loops eliminated.
    ggml_backend_tensor_set(cache.lhs_in, lhs_tc.data(), 0, lhs_tc.size()*sizeof(float));
    ggml_backend_tensor_set(cache.out_in, out_tc.data(), 0, out_tc.size()*sizeof(float));
    profile_vector_compute(model, cache.gf, current_step, island);
    if (residual_trace_out) {
        *residual_trace_out = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "sr_residual"));
    }
    return tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "sr_norm"));
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
    // Reuse the cached graph when it already matches this shape AND was built on
    // the direct backend path (cache.allocr non-null). The scheduler path leaves
    // cache.allocr null, so it always rebuilds. Mirrors run_hift_decode.
    if (cache.ctx && cache.allocr && cache.generation_id == model.generation_id
        && cache.L == L && cache.C == C && cache.Cin == Cin
        && cache.total_steps == total_steps && cache.trace_outputs == trace_outputs) {
        return;
    }
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

    // F12: ingest `tail_in` in CPU-native `[C, L]` channel-fast
    // layout — `run_tail_graph_cache` uploads `x_tc` directly; the
    // graph transposes it inside.  `tail_noise` stays at `[L, Cin]`
    // because the (non-CPU non-trace) tail update path adds it
    // directly to `velocity_t` (shape [L, Cin]); see the
    // accompanying redundancy fix in `run_tail_graph_cache` which
    // also skips two redundant CPU transposes on `noisy_latent`
    // that cancel each other out.
    cache.tail_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, C, L);
    ggml_set_name(cache.tail_in, "tail_in_tc"); ggml_set_input(cache.tail_in);
    cache.tail_mask = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, L);
    ggml_set_name(cache.tail_mask, "tail_mask"); ggml_set_input(cache.tail_mask);
    cache.tail_noise = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, Cin);
    ggml_set_name(cache.tail_noise, "tail_noise"); ggml_set_input(cache.tail_noise);
    ggml_tensor * tail = transpose_time_channel_ggml(cache.ctx, cache.tail_in);
    ggml_set_name(tail, "tail_in");
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
    // CPU-only fused tail-update op (BLAS matmul + mask + step scale +
    // residual add).  The `else` branch below is the pure-GGML
    // decomposition used on GPU backends and during trace runs.
    if (!trace_outputs && supertonic_use_cpu_custom_ops()) {
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

    // Allocation is per-call via the model scheduler (supertonic_sched_alloc
    // in run), which routes GGML_OP_CUSTOM ops to CPU. No per-cache gallocr.
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
    // QVAC-18605 — cache-key check (skip rebuild on hot path).
    if (cache.model != &model || cache.generation_id != model.generation_id ||
        cache.L != L || cache.C != C ||
        cache.Cin != Cin || cache.total_steps != total_steps ||
        cache.trace_outputs != (trace != nullptr)) {
        build_tail_graph_cache(cache, model, L, C, Cin, total_steps, trace != nullptr);
    }
    // QVAC-19254 — direct vs scheduler routing.
    bool direct = true;
    const int n_nodes = ggml_graph_n_nodes(cache.gf);
    for (int i = 0; i < n_nodes; ++i) {
        if (!ggml_backend_supports_op(model.backend, ggml_graph_node(cache.gf, i))) { direct = false; break; }
    }
    if (direct) {
        if (!cache.allocr) {
            cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new supertonic tail graph failed");
            if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
                throw std::runtime_error("ggml_gallocr_reserve supertonic tail graph failed");
            }
        }
        ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    } else {
        supertonic_sched_alloc(model, cache.gf);
    }
    // F12: direct upload of `x_tc` to `cache.tail_in` (now
    // `ne=[C, L]`).  Also eliminates an inadvertent CPU
    // double-transpose on `noisy_latent`: the old code unpacked
    // `noisy_latent[c*L+t]` → `noise_tc[t*Cin+c]` (CPU loop #1)
    // then packed `noise_tc[t*Cin+c]` → `noise_raw[c*L+t]` (CPU
    // loop #2), producing `noise_raw` byte-equivalent to
    // `noisy_latent`.  `noisy_latent` is already in the
    // channel-major memory layout `ne=[L, Cin]` (with natural
    // strides) wants — its element (c, t) at byte `c*L + t`
    // matches GGML's element (l=t, c=c) at memory byte `t + c*L`.
    // Uploading directly skips both loops.
    ggml_backend_tensor_set(cache.tail_in, x_tc.data(), 0, x_tc.size()*sizeof(float));
    ggml_backend_tensor_set(cache.tail_mask, latent_mask, 0, (size_t)L*sizeof(float));
    ggml_backend_tensor_set(cache.tail_noise, noisy_latent, 0, (size_t)L*Cin*sizeof(float));
    if (direct) profile_vector_compute(model, cache.gf, current_step, "tail");
    else        profile_vector_compute(model, cache.gf, current_step, "tail", /*use_sched=*/true);
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

// Audit F9 — cache `time_embedding(model, current, total)` outputs
// keyed by `(current, total)`.  Pure function over its key, so a
// stored entry is the byte-exact result the slow path would produce.
// Cache lives in `model.time_emb_cache` (mutable map); steady-state
// hit rate after the first synth is (total_steps − 1) / total_steps
// (only the cold-miss step on each new key triggers the underlying
// `time_embedding`).  Returns a copy by value (only 64 floats) so
// callers don't have to worry about cache mutation invalidating
// their reference across nested lookups.
inline uint64_t time_emb_cache_key(int current, int total) {
    return ((uint64_t)(uint32_t) current << 32) | (uint32_t) total;
}

} // namespace

std::array<float, 64> cached_time_embedding(const supertonic_model & model,
                                            int current_step,
                                            int total_steps) {
    const uint64_t key = time_emb_cache_key(current_step, total_steps);
    auto it = model.time_emb_cache.find(key);
    if (it != model.time_emb_cache.end()) {
        return it->second;
    }
    std::vector<float> raw = time_embedding(model, current_step, total_steps);
    std::array<float, 64> arr{};
    const size_t n = std::min((size_t) 64, raw.size());
    for (size_t i = 0; i < n; ++i) arr[i] = raw[i];
    auto ins = model.time_emb_cache.emplace(key, arr);
    return ins.first->second;
}

namespace {

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
    // Text cross-attention head count differs across families (v1/v2: 4 heads,
    // v3: 8 heads); head_dim stays 64 so the internal width A = H*64 grows
    // 256 -> 512 in v3.  Bound from GGUF metadata.
    const int D=64;
    const int H=m.hparams.vector_text_attn_heads;
    int C=512, A=H*D;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+3)+".attn.";
    std::vector<float> q,k,v;
    dense_matmul_time(x,L,C,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(qids[group])),read_f32(m,base+"W_query.linear.bias"),A,q);
    std::vector<float> text_lc((size_t)LT*256);
    for(int t=0;t<LT;++t) for(int c=0;c<256;++c) text_lc[(size_t)t*256+c]=text_emb[(size_t)c*LT+t];
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(kids[group])),read_f32(m,base+"W_key.linear.bias"),A,k);
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(vids[group])),read_f32(m,base+"W_value.linear.bias"),A,v);
    // F1: shared host-side cache; same data as
    // `read_f32(m, "...3.attn.theta")` but no per-call backend read.
    const float * theta_t = m.vector_rope_theta.data();
    apply_rope(theta_t,q,L,H,D); apply_rope(theta_t,k,LT,H,D);
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

// `style_ttl` is the style value input (V).  `style_key` overrides the key
// input (K); when null the conditional path's `/Expand_output_0` constant is
// used.  CFG's unconditional pass passes the learned `style_key_special_token`.
void style_attn(const supertonic_model & m, int group, std::vector<float> & x, int L,
                const float * style_ttl, const float * style_key, std::vector<float> & out) {
    static const int qids[4]={3116,3161,3206,3251}, kids[4]={3117,3162,3207,3252}, vids[4]={3118,3163,3208,3253}, oids[4]={3119,3164,3209,3254};
    int C=512,A=256,H=2,D=128,LC=50;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+5)+".attention.";
    std::vector<float> q,k,v,ctx((size_t)LC*256),kctx((size_t)LC*256);
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) ctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
    f32_tensor kconst;
    if(!style_key) kconst=read_f32(m,"vector_estimator:/Expand_output_0");
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) kctx[(size_t)t*256+c]= style_key ? style_key[(size_t)t*256+c] : kconst.data[(size_t)t*256+c];
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
        // F9: cached time-embedding (5 distinct keys per default schedule).
        auto te_arr = cached_time_embedding(model, current_step, total_steps);
        std::vector<float> te(te_arr.begin(), te_arr.end());
        static const int time_ids[4]={3095,3140,3185,3230};

        // One conditional/unconditional pass of the vector field.  `text_cond`
        // is [256, text_len] channel-major; `style_v` is the [50,256] style
        // value input; `style_k` overrides the style key input (null => the
        // conditional `/Expand_output_0` constant).  Returns the per-step
        // velocity-applied latent (channel-major [Cin, L]).
        auto run_field = [&](const float * text_cond, const float * style_v,
                             const float * style_k, std::vector<float> & out_cl) {
            std::vector<float> xf;
            conv1x1(in,L,Cin,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_in.net.weight"),nullptr,C,xf);
            for(int t=0;t<L;++t) for(int c=0;c<C;++c) xf[(size_t)t*C+c]*=latent_mask[t];
            for(int group=0;group<4;++group){
                int ob=group*6;
                int dils[4]={1,2,4,8};
                for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob)+".convnext."+std::to_string(j),xf,L,C,dils[j]);
                std::vector<float> tb;
                dense_matmul_vec(te,read_f32(model,"vector_estimator:onnx::MatMul_"+std::to_string(time_ids[group])),
                                 read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+1)+".linear.linear.bias"),64,C,tb);
                for(int t=0;t<L;++t) for(int c=0;c<C;++c) xf[(size_t)t*C+c]+=tb[c];
                convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+2)+".convnext.0",xf,L,C,1);
                std::vector<float> a; rope_attn(model,group,xf,L,text_cond,text_len,a);
                for(size_t i=0;i<xf.size();++i) xf[i]+=a[i];
                layer_norm(xf,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.bias"));
                convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+4)+".convnext.0",xf,L,C,1);
                style_attn(model,group,xf,L,style_v,style_k,a);
                for(size_t i=0;i<xf.size();++i) xf[i]+=a[i];
                layer_norm(xf,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.bias"));
            }
            for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.last_convnext.convnext."+std::to_string(j),xf,L,C,1);
            conv1x1(xf,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),nullptr,Cin,out_cl);
        };

        // Conditional pass (style value = style_ttl, key = /Expand_output_0).
        std::vector<float> v_cond;
        run_field(text_emb, style_ttl, nullptr, v_cond);

        next_latent_out.assign((size_t)Cin*L,0.0f);
        if (!model.hparams.cfg_enabled()) {
            for(int c=0;c<Cin;++c) for(int t=0;t<L;++t) {
                float vel=v_cond[(size_t)t*Cin+c]*latent_mask[t];
                next_latent_out[(size_t)c*L+t]=noisy_latent[(size_t)c*L+t]+vel/(float)total_steps;
            }
            if(error) error->clear(); return true;
        }

        // Unconditional pass — learned null tokens replace text / style K+V.
        // velocity = cond_scale*v_cond - uncond_scale*v_uncond; the noise term
        // cancels under `next = noise + velocity/N`, so we combine `next`s with
        // the same coefficients (cond_scale - uncond_scale == 1).
        f32_tensor tst = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.text_special_token");  // [256]
        f32_tensor skt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_key_special_token");   // [50,256]
        f32_tensor svt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_value_special_token"); // [50,256]
        std::vector<float> uncond_text((size_t)256*text_len);
        for(int c=0;c<256;++c) for(int t=0;t<text_len;++t) uncond_text[(size_t)c*text_len+t]=tst.data[c];
        std::vector<float> v_uncond;
        run_field(uncond_text.data(), svt.data.data(), skt.data.data(), v_uncond);

        const float cs = model.hparams.cfg_cond_scale, us = model.hparams.cfg_uncond_scale;
        for(int c=0;c<Cin;++c) for(int t=0;t<L;++t) {
            float vel=(cs*v_cond[(size_t)t*Cin+c] - us*v_uncond[(size_t)t*Cin+c])*latent_mask[t];
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
                                       std::vector<float> * next_latent_tc_out,
                                       const std::vector<float> * style_v_raw_override,
                                       const std::vector<float> * kctx_raw_override) {
    supertonic_op_dispatch_scope dispatch(model);
    try {
        scalar_trace.clear();
        ggml_trace.clear();
        const int L = latent_len;
        const int Cin = model.hparams.latent_channels;
        const int C = 512;
        // Text cross-attention head count: v1/v2=4, v3=8.  head_dim stays 64,
        // so the internal attention width A = H_text*64 (256 -> 512 in v3).
        const int H_text = model.hparams.vector_text_attn_heads;
        const int A_text = H_text * 64;
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

            // F9: cached time-embedding.
            auto te_arr = cached_time_embedding(model, current_step, total_steps);
            std::vector<float> te(te_arr.begin(), te_arr.end());
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

            const int A = A_text;
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
            // F1: theta lives in model.vector_rope_theta (populated at load).
            const float * theta_t = model.vector_rope_theta.data();
            apply_rope(theta_t, q, L, H_text, 64);
            apply_rope(theta_t, k, text_len, H_text, 64);
            push_trace(scalar_trace, "ve_attn0_q_rope", L, A, q);
            push_trace(scalar_trace, "ve_attn0_k_rope", text_len, A, k);

            std::vector<float> attn_ctx((size_t)L*A, 0.0f), scores(text_len), probs(text_len);
            const int H = H_text, D = 64;
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
            const int A1 = A_text;
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
            // F1: theta lives in model.vector_rope_theta (populated at load).
            const float * theta1 = model.vector_rope_theta.data();
            apply_rope(theta1, q1, L, H_text, 64);
            apply_rope(theta1, k1, text_len, H_text, 64);
            push_trace(scalar_trace, "ve_g1_attn_q_rope", L, A1, q1);
            push_trace(scalar_trace, "ve_g1_attn_k_rope", text_len, A1, k1);
            std::vector<float> ctx1((size_t)L*A1, 0.0f), scores1(text_len), probs1(text_len);
            for (int h = 0; h < H_text; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q1[((size_t)qi*H_text+h)*64+d] * k1[((size_t)kj*H_text+h)*64+d] * (1.0f/16.0f);
                    scores1[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs1[kj] = std::exp(scores1[kj]-mx); den += probs1[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs1[kj]/den) * v1[((size_t)kj*H_text+h)*64+d];
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
            const int A2 = A_text;
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
            // F1: theta lives in model.vector_rope_theta (populated at load).
            const float * theta2 = model.vector_rope_theta.data();
            apply_rope(theta2, q2, L, H_text, 64);
            apply_rope(theta2, k2, text_len, H_text, 64);
            push_trace(scalar_trace, "ve_g2_attn_q_rope", L, A2, q2);
            push_trace(scalar_trace, "ve_g2_attn_k_rope", text_len, A2, k2);
            std::vector<float> ctx2((size_t)L*A2, 0.0f), scores2(text_len), probs2(text_len);
            for (int h = 0; h < H_text; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q2[((size_t)qi*H_text+h)*64+d] * k2[((size_t)kj*H_text+h)*64+d] * (1.0f/16.0f);
                    scores2[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs2[kj] = std::exp(scores2[kj]-mx); den += probs2[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs2[kj]/den) * v2[((size_t)kj*H_text+h)*64+d];
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
            const int A3 = A_text;
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
            // F1: theta lives in model.vector_rope_theta (populated at load).
            const float * theta3 = model.vector_rope_theta.data();
            apply_rope(theta3, q3, L, H_text, 64);
            apply_rope(theta3, k3, text_len, H_text, 64);
            push_trace(scalar_trace, "ve_g3_attn_q_rope", L, A3, q3);
            push_trace(scalar_trace, "ve_g3_attn_k_rope", text_len, A3, k3);
            std::vector<float> ctx3((size_t)L*A3, 0.0f), scores3(text_len), probs3(text_len);
            for (int h = 0; h < H_text; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q3[((size_t)qi*H_text+h)*64+d] * k3[((size_t)kj*H_text+h)*64+d] * (1.0f/16.0f);
                    scores3[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs3[kj] = std::exp(scores3[kj]-mx); den += probs3[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs3[kj]/den) * v3[((size_t)kj*H_text+h)*64+d];
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

        // F19 — vector-estimator front-block graph cache.  Same
        // pattern as F8 / F11 / F14 / F18: build once per
        // (model, L, text_len, trace), survive across denoise
        // steps.  Pre-audit: 5 fresh alloc/free cycles per synth
        // (one per step); post-audit: 1 cold-miss rebuild on the
        // first step of the first synth, zero rebuilds thereafter
        // for fixed-shape prompts.
        //
        // `trace` is part of the key because the graph wires extra
        // `ggml_set_output` markers for the intermediate convnext
        // outputs in trace mode; rebuilding when the flag flips
        // keeps the gallocr's reserved buffer right-sized.
        struct ve_front_block_graph_cache {
            const supertonic_model * model = nullptr;
            uint64_t generation_id = 0;
            int L = 0;
            int text_len = 0;
            bool trace_outputs = false;
            std::vector<uint8_t> buf;
            ggml_context * ctx = nullptr;
            ggml_cgraph * gf = nullptr;
            ggml_gallocr_t allocr = nullptr;
            // QVAC-18605 round 12 #5 — host-pinned input scratchpad
            // for the three hot per-step inputs (x_in, mask_in,
            // t_emb_in).  Same dispatch pattern as
            // `vector_group_graph_cache`: helper returns nullptr on
            // CPU / non-Vulkan backends; we fall back to the
            // default backend buffer via
            // `ggml_backend_alloc_ctx_tensors(input_ctx, backend)`.
            // `text_in_t` stays in `ctx` (gallocr-allocated) — the
            // round-10 upload-skip tracker handles the per-step
            // upload elision so the staging-hop saving doesn't
            // amortise on the cold-miss-only path.
            std::vector<uint8_t> input_ctx_storage;
            ggml_context * input_ctx = nullptr;
            ggml_backend_buffer_t input_buf = nullptr;
            ggml_tensor * x_in = nullptr;
            ggml_tensor * mask_in = nullptr;
            ggml_tensor * t_emb_in = nullptr;
            ggml_tensor * text_in_t = nullptr;
            // F23 — in-graph RoPE inputs (cos/sin tables for Q's
            // sequence length L and K's sequence length text_len).
            // Stable for the cache's lifetime; uploaded once at
            // build time.  `apply_rope` is false when the GGUF
            // didn't ship vector_rope_theta, in which case the
            // legacy host apply_rope path is taken downstream.
            bool apply_rope = false;
            ggml_tensor * q_cos_in = nullptr;
            ggml_tensor * q_sin_in = nullptr;
            ggml_tensor * k_cos_in = nullptr;
            ggml_tensor * k_sin_in = nullptr;

            // QVAC-18605 round 10 — pointer-compare upload-skip
            // tracker for `text_in_t`.  `text_emb` is stable within
            // one synth (5 calls × same pointer) but the stack-
            // local `std::vector<float>` may be reallocated to the
            // SAME address across synths (allocator size-class
            // reuse).  Caller resets at `current_step == 0` to
            // avoid leaking synth-N data into synth-N+1.  See the
            // upload_skip_tracker contract in
            // supertonic_internal.h.
            //
            // Cache rebuild zeroes this via `front_cache = {}`
            // (the tracker's only field is a pointer that
            // zero-initialises to nullptr → effective reset).
            upload_skip_tracker text_in_skip;
        };
        thread_local ve_front_block_graph_cache front_cache;
        thread_local tl_register_once _tl_reg_front_cache([&]() {
            supertonic_safe_gallocr_free(front_cache.allocr, front_cache.generation_id);
            if (front_cache.ctx) ggml_free(front_cache.ctx);
            if (front_cache.input_buf) ggml_backend_buffer_free(front_cache.input_buf);
            if (front_cache.input_ctx) ggml_free(front_cache.input_ctx);
            front_cache = {};
        });
        if (front_cache.model != &model ||
            front_cache.generation_id != model.generation_id ||
            front_cache.L != L ||
            front_cache.text_len != text_len ||
            front_cache.trace_outputs != include_ggml_trace) {
            // Tear down stale state.  Round 12 #5 — same teardown
            // order as `free_group_graph_cache`: gallocr → main
            // ctx → input host buffer → input ctx.  Reversing
            // order would dangle gallocr pointers into freed
            // input-ctx tensor metadata.
            supertonic_safe_gallocr_free(front_cache.allocr, front_cache.generation_id);
            if (front_cache.ctx) ggml_free(front_cache.ctx);
            if (front_cache.input_buf) ggml_backend_buffer_free(front_cache.input_buf);
            if (front_cache.input_ctx) ggml_free(front_cache.input_ctx);
            front_cache = {};
            front_cache.model = &model;
            front_cache.generation_id = model.generation_id;
            front_cache.L = L;
            front_cache.text_len = text_len;
            front_cache.trace_outputs = include_ggml_trace;

            constexpr int MAX_NODES = 2048;
            const size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                                    ggml_graph_overhead_custom(MAX_NODES, false);
            front_cache.buf.assign(buf_size, 0);
            ggml_init_params p = { buf_size, front_cache.buf.data(), true };
            front_cache.ctx = ggml_init(p);
            front_cache.gf  = ggml_new_graph_custom(front_cache.ctx, MAX_NODES, false);

            // QVAC-18605 round 12 #5 — host-pinned scratchpad for
            // the 3 hot per-step inputs (x_in, mask_in, t_emb_in).
            // text_in_t stays in the main ctx (round-10 upload-skip
            // tracker elides per-step uploads; pinned-host doesn't
            // amortise on the cold-miss-only path).
            {
                const size_t INPUT_OVERHEAD = ggml_tensor_overhead() * 8;
                front_cache.input_ctx_storage.assign(INPUT_OVERHEAD, 0);
                ggml_init_params input_p = { INPUT_OVERHEAD, front_cache.input_ctx_storage.data(), /*no_alloc=*/true };
                front_cache.input_ctx = ggml_init(input_p);
                front_cache.x_in = ggml_new_tensor_2d(front_cache.input_ctx, GGML_TYPE_F32, L, Cin);
                ggml_set_name(front_cache.x_in, "ve_latent_tc");
                ggml_set_input(front_cache.x_in);
                front_cache.mask_in = ggml_new_tensor_1d(front_cache.input_ctx, GGML_TYPE_F32, L);
                ggml_set_name(front_cache.mask_in, "ve_latent_mask");
                ggml_set_input(front_cache.mask_in);
                front_cache.t_emb_in = ggml_new_tensor_1d(front_cache.input_ctx, GGML_TYPE_F32, 64);
                ggml_set_name(front_cache.t_emb_in, "ve_time_emb");
                ggml_set_input(front_cache.t_emb_in);
                // QVAC-18605 round 13 #1 — consolidated allocator
                // (round-12 inlined the try-pinned-host + fallback
                // boilerplate; this round factors it out via
                // `alloc_input_scratchpad_or_throw`).
                front_cache.input_buf = alloc_input_scratchpad_or_throw(
                    model, front_cache.input_ctx, "ve_front_block_graph_cache");
            }
            front_cache.text_in_t = ggml_new_tensor_2d(front_cache.ctx, GGML_TYPE_F32, text_len, 256);
            ggml_set_name(front_cache.text_in_t, "ve_text_lc");
            // text_in_t is uploaded once per synth (round-10 upload-skip
            // tracker — `current_step == 0` resets, every other step
            // skips the upload as the host pointer is stable).  Without
            // OUTPUT the gallocr-managed buffer is freed after step 0's
            // last consumer runs and aliased with step 1's intermediates,
            // silently corrupting the text embedding for steps 1..N-1.
            // INPUT alone protects the initial allocation but not the
            // buffer's lifetime across compute passes.  See the matching
            // notes on the relpos masks + RoPE cos/sin tables.
            ggml_set_input(front_cache.text_in_t);  ggml_set_output(front_cache.text_in_t);

            ggml_tensor * y_t = conv1d_f32(front_cache.ctx,
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_in.net.weight"),
                front_cache.x_in, 1, 0, 1);
            ggml_tensor * masked_t = ggml_mul(front_cache.ctx, y_t,
                repeat_like(front_cache.ctx, front_cache.mask_in, y_t));
            ggml_set_name(masked_t, "ve_masked");
            if (include_ggml_trace) {
                ggml_set_output(masked_t);
                ggml_build_forward_expand(front_cache.gf, masked_t);
            }
            ggml_tensor * cur_t = masked_t;
            int dils_ggml[4] = {1, 2, 4, 8};
            for (int j = 0; j < 4; ++j) {
                cur_t = vector_convnext_ggml(front_cache.ctx, model,
                    "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext." + std::to_string(j),
                    cur_t, dils_ggml[j]);
                if (include_ggml_trace) {
                    const std::string name = "ve_block0_convnext" + std::to_string(j);
                    ggml_set_name(cur_t, name.c_str());
                    ggml_set_output(cur_t);
                    ggml_build_forward_expand(front_cache.gf, cur_t);
                }
            }

            // F6 pre-transposed t_proj companion or fallback.
            ggml_tensor * t_proj_w_t;
            {
                auto pretrans_it = model.source_tensors.find("vector_estimator:onnx::MatMul_3095__T");
                t_proj_w_t = (pretrans_it != model.source_tensors.end()) ? pretrans_it->second : nullptr;
                if (!t_proj_w_t) {
                    t_proj_w_t = ggml_cont(front_cache.ctx, ggml_transpose(front_cache.ctx,
                        require_source_tensor(model, "vector_estimator:onnx::MatMul_3095")));
                }
            }
            ggml_tensor * t_proj = ggml_mul_mat(front_cache.ctx, t_proj_w_t,
                ggml_reshape_2d(front_cache.ctx, front_cache.t_emb_in, 64, 1));
            t_proj = ggml_add(front_cache.ctx, t_proj,
                ggml_reshape_2d(front_cache.ctx,
                    require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.1.linear.linear.bias"),
                    C, 1));
            cur_t = ggml_add(front_cache.ctx, cur_t, repeat_like(front_cache.ctx, t_proj, cur_t));
            ggml_set_name(cur_t, "ve_time_add0");
            if (include_ggml_trace) {
                ggml_set_output(cur_t);
                ggml_build_forward_expand(front_cache.gf, cur_t);
            }

            cur_t = vector_convnext_ggml(front_cache.ctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.2.convnext.0",
                cur_t, 1);
            ggml_set_name(cur_t, "ve_block2_convnext0");
            ggml_set_output(cur_t);
            ggml_build_forward_expand(front_cache.gf, cur_t);
            ggml_tensor * q_t = dense_matmul_time_ggml(front_cache.ctx, cur_t,
                require_source_tensor(model, "vector_estimator:onnx::MatMul_3101"),
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_query.linear.bias"));
            ggml_set_name(q_t, "ve_attn0_q");
            ggml_set_output(q_t);
            ggml_build_forward_expand(front_cache.gf, q_t);
            ggml_tensor * k_t = dense_matmul_time_ggml(front_cache.ctx, front_cache.text_in_t,
                require_source_tensor(model, "vector_estimator:onnx::MatMul_3102"),
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_key.linear.bias"));
            ggml_set_name(k_t, "ve_attn0_k");
            ggml_set_output(k_t);
            ggml_build_forward_expand(front_cache.gf, k_t);
            // QVAC-18966 — pack V into the layout
            // `run_text_attention_cache_gpu` consumes via
            // `ggml_backend_tensor_copy(v_src, v_tc_in)`.  See the
            // identical transpose in `build_group_graph_cache` +
            // the header doc on `apply_rope_to_packed_qk` in
            // `supertonic_internal.h`.  Matmul output is ne=[L_kv,
            // HD] channel-major-flat; v_tc_in expects ne=[HD,
            // L_kv] time-major-flat.  Legacy host bridge
            // downloads `ve_attn0_v` via `tensor_raw_f32` to get
            // bytes in the time-major-flat shape scalar
            // `apply_rope` / `flash_attention_qkv` references.
            ggml_tensor * v_matmul = dense_matmul_time_ggml(front_cache.ctx, front_cache.text_in_t,
                require_source_tensor(model, "vector_estimator:onnx::MatMul_3103"),
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_value.linear.bias"));
            ggml_tensor * v_t = ggml_cont(front_cache.ctx,
                ggml_transpose(front_cache.ctx, v_matmul));
            ggml_set_name(v_t, "ve_attn0_v");
            ggml_set_output(v_t);
            ggml_build_forward_expand(front_cache.gf, v_t);

            // F23 — same in-graph RoPE wiring as the per-group
            // graph cache: produce post-rotation
            // `ve_attn0_q_rope` / `ve_attn0_k_rope` outputs so the
            // call site below can drop the host `apply_rope`
            // round-trips.  Falls through to the legacy host
            // rotation path when the GGUF didn't ship theta.
            const int FRONT_H = H_text;
            const int FRONT_D = 64;
            const int FRONT_HALF = FRONT_D / 2;
            front_cache.apply_rope =
                (int) model.vector_rope_theta.size() == FRONT_HALF;
            if (front_cache.apply_rope) {
                // RoPE cos/sin tables are cache-lifetime constants
                // (depend only on L / text_len / θ).  Mark INPUT + OUTPUT
                // so gallocr keeps the buffers alive across compute
                // passes — see the matching note in build_group_graph_cache.
                front_cache.q_cos_in = ggml_new_tensor_2d(front_cache.ctx,
                    GGML_TYPE_F32, FRONT_HALF, L);
                ggml_set_name(front_cache.q_cos_in, "ve_attn0_q_rope_cos");
                ggml_set_input(front_cache.q_cos_in);  ggml_set_output(front_cache.q_cos_in);
                front_cache.q_sin_in = ggml_new_tensor_2d(front_cache.ctx,
                    GGML_TYPE_F32, FRONT_HALF, L);
                ggml_set_name(front_cache.q_sin_in, "ve_attn0_q_rope_sin");
                ggml_set_input(front_cache.q_sin_in);  ggml_set_output(front_cache.q_sin_in);
                front_cache.k_cos_in = ggml_new_tensor_2d(front_cache.ctx,
                    GGML_TYPE_F32, FRONT_HALF, text_len);
                ggml_set_name(front_cache.k_cos_in, "ve_attn0_k_rope_cos");
                ggml_set_input(front_cache.k_cos_in);  ggml_set_output(front_cache.k_cos_in);
                front_cache.k_sin_in = ggml_new_tensor_2d(front_cache.ctx,
                    GGML_TYPE_F32, FRONT_HALF, text_len);
                ggml_set_name(front_cache.k_sin_in, "ve_attn0_k_rope_sin");
                ggml_set_input(front_cache.k_sin_in);  ggml_set_output(front_cache.k_sin_in);
                ggml_tensor * q_rope = apply_rope_to_packed_qk(front_cache.ctx,
                    q_t, front_cache.q_cos_in, front_cache.q_sin_in,
                    FRONT_H, FRONT_D);
                ggml_set_name(q_rope, "ve_attn0_q_rope");
                ggml_set_output(q_rope);
                ggml_build_forward_expand(front_cache.gf, q_rope);
                ggml_tensor * k_rope = apply_rope_to_packed_qk(front_cache.ctx,
                    k_t, front_cache.k_cos_in, front_cache.k_sin_in,
                    FRONT_H, FRONT_D);
                ggml_set_name(k_rope, "ve_attn0_k_rope");
                ggml_set_output(k_rope);
                ggml_build_forward_expand(front_cache.gf, k_rope);
            }

            front_cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!front_cache.allocr) {
                ggml_free(front_cache.ctx);
                front_cache = {};
                throw std::runtime_error("ggml_gallocr_new failed");
            }
            if (!ggml_gallocr_reserve(front_cache.allocr, front_cache.gf)) {
                ggml_gallocr_free(front_cache.allocr);
                ggml_free(front_cache.ctx);
                front_cache = {};
                throw std::runtime_error("ggml_gallocr_reserve failed");
            }
            ggml_gallocr_alloc_graph(front_cache.allocr, front_cache.gf);

            // F23 — upload cos/sin tables for the in-graph RoPE
            // rotation.  These inputs depend only on (L, text_len,
            // theta), all stable for the cache's lifetime; the
            // upload is one-shot at build time.
            if (front_cache.apply_rope) {
                const int FRONT_HALF = 32;
                std::vector<float> q_cos, q_sin, k_cos, k_sin;
                make_rope_cos_sin_tables(model.vector_rope_theta.data(),
                                         L, FRONT_HALF, q_cos, q_sin);
                make_rope_cos_sin_tables(model.vector_rope_theta.data(),
                                         text_len, FRONT_HALF, k_cos, k_sin);
                ggml_backend_tensor_set(front_cache.q_cos_in, q_cos.data(),
                                        0, q_cos.size() * sizeof(float));
                ggml_backend_tensor_set(front_cache.q_sin_in, q_sin.data(),
                                        0, q_sin.size() * sizeof(float));
                ggml_backend_tensor_set(front_cache.k_cos_in, k_cos.data(),
                                        0, k_cos.size() * sizeof(float));
                ggml_backend_tensor_set(front_cache.k_sin_in, k_sin.data(),
                                        0, k_sin.size() * sizeof(float));
            }
        }
        // QVAC-18605 round 12 — reuse-or-rebuild done; expose the
        // cache's compute graph + input tensors under the variable
        // names the rest of this scope already uses.  HEAD's
        // front_cache builds these same nodes (ve_time_add0,
        // ve_block2_convnext0, ve_attn0_q/k/v, optional rope outputs)
        // ONCE at cache-build time and reuses them across the 5
        // denoise-step calls; master's inline-build path is the
        // non-cached equivalent that rebuilds every call.  We keep
        // the cache here; the post-`profile_vector_compute` GPU-
        // bridge path below still reads the same named tensors.
        ggml_cgraph * gf = front_cache.gf;
        ggml_tensor * x = front_cache.x_in;
        ggml_tensor * mask = front_cache.mask_in;
        ggml_tensor * t_emb = front_cache.t_emb_in;
        ggml_tensor * text_in = front_cache.text_in_t;
        (void) text_in;
        (void) mask; (void) t_emb;  // referenced via `front_cache.*` below

        ggml_backend_tensor_set(x, noisy_latent, 0, (size_t) L * Cin * sizeof(float));
        ggml_backend_tensor_set(mask, latent_mask, 0, (size_t) L * sizeof(float));
        // F9: cached time-embedding — second+ synth pays zero CPU cost
        // for this step and skips the underlying 2 weight downloads.
        // `te_host` stays a std::vector<float> because it's forwarded
        // to `run_group_graph_cache(..., const std::vector<float> & temb, …)`
        // three times below and changing that ABI would ripple into
        // the trace harnesses.  64-element copy is negligible vs the
        // GPU sync saved on the underlying read_f32 calls.
        auto te_arr = cached_time_embedding(model, current_step, total_steps);
        std::vector<float> te_host(te_arr.begin(), te_arr.end());
        ggml_backend_tensor_set(t_emb, te_host.data(), 0, te_host.size() * sizeof(float));
        // QVAC-18605 round 10 — text_emb upload-skip.  `text_emb`
        // is stable within one synth (5 calls × same pointer); skip
        // the upload on steps 1..N-1 if the pointer matches the
        // last successful upload's pointer.  Synth-boundary reset
        // (`current_step == 0`) invalidates the cache so the next
        // synth's first step always uploads — protects against
        // the stack-realloc-same-address hazard documented on
        // `upload_skip_tracker` in supertonic_internal.h.
        //
        // The earlier comment "the cache that used to wrap this
        // was a verbatim copy keyed on a pointer that never
        // matched twice" referred to a per-call wrapper that
        // forgot to use a stable cache instance — round 10 fixes
        // that by storing the tracker on the (thread_local)
        // front_cache instance, so consecutive `current_step`
        // values within the same synth see a populated tracker.
        if (current_step == 0) front_cache.text_in_skip.reset();
        if (front_cache.text_in_skip.needs_upload(text_emb)) {
            ggml_backend_tensor_set(text_in, text_emb, 0, (size_t) text_len * 256 * sizeof(float));
            front_cache.text_in_skip.mark_uploaded(text_emb);
        }
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
        // QVAC-18605 round 8 — front-block attn0 GPU bridge.
        //
        // PR #16's audit follow-up #6 (2C-lite) shipped the GPU
        // device→device blit infrastructure (`run_text_attention_cache_gpu`)
        // and wired g1 / g2 / g3 group attentions to use it.  The
        // front-block attn0 site was deferred because of cache-
        // lifetime concerns at the time; round 8 picks it up.
        //
        // The front_cache (`ve_front_block_graph_cache` in the
        // outer scope) is `thread_local` and stable across calls
        // (rebuilds only on shape change L / text_len /
        // trace_outputs).  After `profile_vector_compute` returns,
        // the named output tensors `ve_attn0_v` and (when
        // `apply_rope` is true) `ve_attn0_q_rope` /
        // `ve_attn0_k_rope` are valid GPU handles for the
        // duration of the next attention compute.  Same lifetime
        // guarantee as the g1/g2/g3 caches → safe to pass into
        // `run_text_attention_cache_gpu`.
        //
        // Eliminates per call: 3 GPU→host downloads + 3 host→GPU
        // uploads.  Across 5 denoise steps × Q/K/V = 30 sync
        // points / synth.  Production path only — trace mode
        // still takes the legacy host-bridge path so the trace
        // dump captures pre-attention Q/K/V host vectors.
        //
        // Note: the legacy host-bridge fallback below still uses
        // `tensor_to_time_channel(v_gpu_attn0)`; round 11's
        // QVAC-18966 layout fix re-patches that call site to
        // `tensor_raw_f32(...)` after `ve_attn0_v` becomes
        // `ggml_cont(ggml_transpose(...))`-shaped.
        ggml_tensor * v_gpu_attn0      = ggml_graph_get_tensor(gf, "ve_attn0_v");
        ggml_tensor * q_rope_gpu_attn0 = ggml_graph_get_tensor(gf, "ve_attn0_q_rope");
        ggml_tensor * k_rope_gpu_attn0 = ggml_graph_get_tensor(gf, "ve_attn0_k_rope");
        const bool front_in_graph_rope = (q_rope_gpu_attn0 != nullptr);
        const bool front_use_gpu_bridge = front_in_graph_rope && !include_ggml_trace
                                          && v_gpu_attn0 && k_rope_gpu_attn0;
        std::vector<float> q_out, k_out, q_rotated, k_rotated, v_out;
        thread_local vector_text_attention_cache att0_cache;
        SUPERTONIC_REGISTER_TL_CACHE(att0_cache, free_text_attention_cache);
        std::vector<float> att0_ctx_trace;
        std::vector<float> attn_out_ggml;
        if (front_use_gpu_bridge) {
            // Fast path: device→device blit, host never sees Q/K/V.
            // Mirrors the g1/g2/g3 dispatch at lines 2926-2933.
            attn_out_ggml = run_text_attention_cache_gpu(att0_cache, model,
                q_rope_gpu_attn0, k_rope_gpu_attn0, v_gpu_attn0,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3110",
                "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.out_fc.linear.bias",
                current_step, "attn0_flash",
                /*ctx_trace=*/ nullptr);
        } else {
            // Legacy / trace-mode host bridge.  Falls back to the
            // pre-round-8 download + rotate + upload pattern.
            //
            // QVAC-18605 follow-up — post-fix V graph layout:
            // `ve_attn0_v` is now `ggml_cont(ggml_transpose(...))`
            // of the matmul output (ne=[HD, text_len] time-major-
            // flat memory).  `tensor_raw_f32` downloads the bytes
            // directly in the layout scalar `apply_rope` /
            // `flash_attention_qkv` host references expect
            // (`v[t*HD + c]`).  Using `tensor_to_time_channel`
            // here would mis-interpret the swapped ne.  See the
            // header doc on `apply_rope_to_packed_qk` in
            // `supertonic_internal.h`.  Q/K matmul outputs are
            // UNCHANGED (still ne=[L, HD] channel-major-flat) so
            // `tensor_to_time_channel` is the right call there.
            v_out = tensor_raw_f32(v_gpu_attn0);
            if (include_ggml_trace) {
                q_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_q"));
                k_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_k"));
                PUSH_GGML_TRACE({"ve_attn0_q", {L, 256}, q_out});
                PUSH_GGML_TRACE({"ve_attn0_k", {text_len, 256}, k_out});
                PUSH_GGML_TRACE({"ve_attn0_v", {text_len, 256}, v_out});
            }
            // F23 — when the front-block graph has the in-graph
            // RoPE wired in (model carries `vector_rope_theta`),
            // feed `run_text_attention_cache` the already-rotated
            // Q/K from the `_rope` graph outputs.  Host
            // `apply_rope(theta, …)` is fully eliminated on the
            // in-graph-rope path.
            if (front_in_graph_rope) {
                // QVAC-18605 follow-up — post-fix layout contract:
                // `apply_rope_to_packed_qk` produces ne=[HD, L]
                // with time-major-flat memory (`data[c + t*HD]`),
                // which is bit-identical to scalar `apply_rope`'s
                // output buffer.  `tensor_raw_f32` downloads those
                // bytes directly — no transpose needed (and using
                // `tensor_to_time_channel` here would mis-interpret
                // the ne shape and produce the transpose of the
                // transpose, silently feeding wrong-orientation
                // Q/K into the attention).  See the header doc on
                // `apply_rope_to_packed_qk` in
                // `supertonic_internal.h`.
                q_rotated = tensor_raw_f32(q_rope_gpu_attn0);
                k_rotated = tensor_raw_f32(k_rope_gpu_attn0);
            } else {
                // Legacy GGUF path: rotate host-side.
                if (q_out.empty()) {
                    q_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_q"));
                    k_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_k"));
                }
                const float * theta = model.vector_rope_theta.data();
                apply_rope(theta, q_out, L, H_text, 64);
                apply_rope(theta, k_out, text_len, H_text, 64);
                q_rotated = std::move(q_out);
                k_rotated = std::move(k_out);
            }
            attn_out_ggml = run_text_attention_cache(att0_cache, model, q_rotated, k_rotated, v_out,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3110",
                "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.out_fc.linear.bias",
                current_step, "attn0_flash",
                include_ggml_trace ? &att0_ctx_trace : nullptr);
        }
        // Trace pushes — `q_rotated` / `k_rotated` are populated
        // by the legacy branch above; empty on the GPU-bridge
        // path (in which case `PUSH_GGML_TRACE` is a no-op
        // because `include_ggml_trace == false`).  Matches the
        // g1/g2/g3 trace-push pattern at lines 2955-2956.
        PUSH_GGML_TRACE({"ve_attn0_q_rope", {L, 256}, q_rotated});
        PUSH_GGML_TRACE({"ve_attn0_k_rope", {text_len, 256}, k_rotated});
        PUSH_GGML_TRACE({"ve_attn0_ctx", {L, 256}, att0_ctx_trace});
        PUSH_GGML_TRACE({"ve_attn0_out", {L, C}, attn_out_ggml});

        const std::vector<float> * style_v_raw = nullptr;
        const std::vector<float> * kctx_raw = nullptr;
        if (style_v_raw_override && kctx_raw_override) {
            // CFG unconditional pass: caller supplies the learned null-token
            // style V/K layouts (already in the channel-major [50,256] form
            // that `cached_style_layouts` produces for the conditional path).
            style_v_raw = style_v_raw_override;
            kctx_raw    = kctx_raw_override;
        } else {
            cached_style_layouts(model, style_ttl, style_v_raw, kctx_raw);
        }
        thread_local vector_res_style_qkv_cache style0_res_qkv_cache;
        SUPERTONIC_REGISTER_TL_CACHE(style0_res_qkv_cache, free_res_style_qkv_cache);
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
        // QVAC-18605 round 9 — style flash-attn GPU bridge for
        // style0 (front-block style residual).  Same dispatch
        // pattern as the round-8 front-block attn0 bridge:
        // production path uses `run_text_attention_cache_gpu`
        // with the GPU handles from the res-style-qkv cache,
        // trace mode falls back to the legacy host bridge so
        // the trace harness still gets the host vectors.
        thread_local vector_text_attention_cache style0_attn_cache;
        SUPERTONIC_REGISTER_TL_CACHE(style0_attn_cache, free_text_attention_cache);
        std::vector<float> style0_ctx_trace;
        std::vector<float> style_out_ggml;
        const bool style0_use_gpu_bridge = !include_ggml_trace
            && style0_res_qkv.sq_gpu && style0_res_qkv.sk_gpu && style0_res_qkv.sv_gpu;
        if (style0_use_gpu_bridge) {
            style_out_ggml = run_text_attention_cache_gpu(style0_attn_cache, model,
                style0_res_qkv.sq_gpu, style0_res_qkv.sk_gpu, style0_res_qkv.sv_gpu,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3119",
                "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.bias",
                current_step, "style0_flash",
                /*ctx_trace=*/ nullptr);
        } else {
            std::vector<float> sq_out = std::move(style0_res_qkv.sq);
            std::vector<float> sk_out = std::move(style0_res_qkv.sk);
            std::vector<float> sv_out = std::move(style0_res_qkv.sv);
            style_out_ggml = run_text_attention_cache(style0_attn_cache, model, sq_out, sk_out, sv_out,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3119",
                "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.bias",
                current_step, "style0_flash",
                include_ggml_trace ? &style0_ctx_trace : nullptr);
        }
        PUSH_GGML_TRACE({"ve_style0_ctx", {L, 256}, style0_ctx_trace});
        PUSH_GGML_TRACE({"ve_style0_out", {L, C}, style_out_ggml});
        // F8: cached style-residual graph (lhs + out → add → LN).
        // norm_block = 5 for the front-block style residual.
        // QVAC-18605 round 12 — `run_style_residual_cache` keeps a
        // thread_local graph across calls; master's inline-build
        // equivalent has been deliberately replaced by the cache.
        thread_local vector_style_residual_graph_cache style0_res_cache;
        SUPERTONIC_REGISTER_TL_CACHE(style0_res_cache, free_style_residual_cache);
        std::vector<float> style0_res_trace;
        std::vector<float> style_norm_ggml = run_style_residual_cache(
            style0_res_cache, model, post_ggml, style_out_ggml,
            L, C, /*norm_block=*/5, current_step, "style0_residual",
            include_ggml_trace ? &style0_res_trace : nullptr);
        PUSH_GGML_TRACE({"ve_style0_residual", {L, C}, style0_res_trace});
        PUSH_GGML_TRACE({"ve_style0_norm", {L, C}, style_norm_ggml});

        thread_local vector_group_graph_cache g1_group_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g1_group_cache, free_group_graph_cache);
        vector_group_graph_result g1_group = run_group_graph_cache(g1_group_cache, model, style_norm_ggml,
            L, C, te_host, text_emb, text_len, current_step,
            1, 6, 7, "vector_estimator:onnx::MatMul_3140", 8,
            "vector_estimator:onnx::MatMul_3146",
            "vector_estimator:onnx::MatMul_3147",
            "vector_estimator:onnx::MatMul_3148",
            "ve_g1_attn_q", "ve_g1_attn_k", "ve_g1_attn_v",
            "group1_conv_attn_qkv", include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g1_block8 = std::move(g1_group.post);
        // 2C-lite — production fast path: pass GPU tensor handles
        // straight from the group cache into the attention cache
        // via `ggml_backend_tensor_copy`.  Host vectors for
        // q/k/v/q_rope/k_rope are empty in production (gated on
        // `trace != nullptr` inside `run_group_graph_cache`), so
        // we MUST use the *_gpu pointers when present.  Falls
        // back to the legacy host rotation path when the cache
        // didn't wire RoPE in graph (e.g. malformed GGUF).
        thread_local vector_text_attention_cache g1_attn_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g1_attn_cache, free_text_attention_cache);
        std::vector<float> g1_attn_ctx_trace;
        std::vector<float> g1_attn_out;
        if (g1_group.q_rope_gpu && g1_group.k_rope_gpu && g1_group.v_gpu) {
            g1_attn_out = run_text_attention_cache_gpu(g1_attn_cache, model,
                g1_group.q_rope_gpu, g1_group.k_rope_gpu, g1_group.v_gpu,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3155",
                "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.out_fc.linear.bias",
                current_step, "g1_attn_flash",
                include_ggml_trace ? &g1_attn_ctx_trace : nullptr);
        } else {
            std::vector<float> g1q_out = std::move(g1_group.q);
            std::vector<float> g1k_out = std::move(g1_group.k);
            std::vector<float> g1v_out = std::move(g1_group.v);
            std::vector<float> g1q_rotated = g1q_out;
            std::vector<float> g1k_rotated = g1k_out;
            const float * theta_g1 = model.vector_rope_theta.data();
            apply_rope(theta_g1, g1q_rotated, L, H_text, 64);
            apply_rope(theta_g1, g1k_rotated, text_len, H_text, 64);
            g1_attn_out = run_text_attention_cache(g1_attn_cache, model,
                g1q_rotated, g1k_rotated, g1v_out,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3155",
                "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.out_fc.linear.bias",
                current_step, "g1_attn_flash",
                include_ggml_trace ? &g1_attn_ctx_trace : nullptr);
        }
        // Trace pushes — use the host vectors the group cache
        // downloaded under its `if (trace)` guard.  Empty when
        // include_ggml_trace is false (PUSH_GGML_TRACE is a no-op
        // in that case).
        PUSH_GGML_TRACE({"ve_g1_attn_q_rope", {L, 256}, g1_group.q_rope});
        PUSH_GGML_TRACE({"ve_g1_attn_k_rope", {text_len, 256}, g1_group.k_rope});
        PUSH_GGML_TRACE({"ve_g1_attn_ctx", {L, 256}, g1_attn_ctx_trace});
        PUSH_GGML_TRACE({"ve_g1_attn_out", {L, C}, g1_attn_out});

        thread_local vector_res_style_qkv_cache g1_res_qkv_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g1_res_qkv_cache, free_res_style_qkv_cache);
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
        // QVAC-18605 round 9 — style flash-attn GPU bridge for g1.
        thread_local vector_text_attention_cache g1_style_attn_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g1_style_attn_cache, free_text_attention_cache);
        std::vector<float> g1_style_ctx_trace;
        std::vector<float> g1_style_out;
        const bool g1_style_use_gpu_bridge = !include_ggml_trace
            && g1_res_qkv.sq_gpu && g1_res_qkv.sk_gpu && g1_res_qkv.sv_gpu;
        if (g1_style_use_gpu_bridge) {
            g1_style_out = run_text_attention_cache_gpu(g1_style_attn_cache, model,
                g1_res_qkv.sq_gpu, g1_res_qkv.sk_gpu, g1_res_qkv.sv_gpu,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3164",
                "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.out_fc.linear.bias",
                current_step, "g1_style_flash",
                /*ctx_trace=*/ nullptr);
        } else {
            std::vector<float> g1sq_out = std::move(g1_res_qkv.sq);
            std::vector<float> g1sk_out = std::move(g1_res_qkv.sk);
            std::vector<float> g1sv_out = std::move(g1_res_qkv.sv);
            g1_style_out = run_text_attention_cache(g1_style_attn_cache, model, g1sq_out, g1sk_out, g1sv_out,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3164",
                "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.out_fc.linear.bias",
                current_step, "g1_style_flash",
                include_ggml_trace ? &g1_style_ctx_trace : nullptr);
        }
        PUSH_GGML_TRACE({"ve_g1_style_ctx", {L, 256}, g1_style_ctx_trace});
        PUSH_GGML_TRACE({"ve_g1_style_out", {L, C}, g1_style_out});

        // F8: cached style-residual graph (norm_block = 11 for group 1).
        // Mirror of style0_residual block; HEAD's cache reused across
        // calls, master's inline-build equivalent dropped.
        thread_local vector_style_residual_graph_cache g1_style_res_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g1_style_res_cache, free_style_residual_cache);
        std::vector<float> g1_style_res_trace;
        std::vector<float> g1_style_norm_vec = run_style_residual_cache(
            g1_style_res_cache, model, g1_block10, g1_style_out,
            L, C, /*norm_block=*/11, current_step, "g1_style_residual",
            include_ggml_trace ? &g1_style_res_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g1_style_residual", {L, C}, g1_style_res_trace});
        PUSH_GGML_TRACE({"ve_g1_style_norm", {L, C}, g1_style_norm_vec});

        thread_local vector_group_graph_cache g2_group_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g2_group_cache, free_group_graph_cache);
        vector_group_graph_result g2_group = run_group_graph_cache(g2_group_cache, model, g1_style_norm_vec,
            L, C, te_host, text_emb, text_len, current_step,
            2, 12, 13, "vector_estimator:onnx::MatMul_3185", 14,
            "vector_estimator:onnx::MatMul_3191",
            "vector_estimator:onnx::MatMul_3192",
            "vector_estimator:onnx::MatMul_3193",
            "ve_g2_attn_q", "ve_g2_attn_k", "ve_g2_attn_v",
            "group2_conv_attn_qkv", include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g2_block14 = std::move(g2_group.post);
        // 2C-lite — same GPU fast-path / host-fallback pattern as g1.
        thread_local vector_text_attention_cache g2_attn_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g2_attn_cache, free_text_attention_cache);
        std::vector<float> g2_attn_ctx_trace;
        std::vector<float> g2_attn_out;
        if (g2_group.q_rope_gpu && g2_group.k_rope_gpu && g2_group.v_gpu) {
            g2_attn_out = run_text_attention_cache_gpu(g2_attn_cache, model,
                g2_group.q_rope_gpu, g2_group.k_rope_gpu, g2_group.v_gpu,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3200",
                "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.out_fc.linear.bias",
                current_step, "g2_attn_flash",
                include_ggml_trace ? &g2_attn_ctx_trace : nullptr);
        } else {
            std::vector<float> g2q_out = std::move(g2_group.q);
            std::vector<float> g2k_out = std::move(g2_group.k);
            std::vector<float> g2v_out = std::move(g2_group.v);
            std::vector<float> g2q_rotated = g2q_out;
            std::vector<float> g2k_rotated = g2k_out;
            const float * theta_g2 = model.vector_rope_theta.data();
            apply_rope(theta_g2, g2q_rotated, L, H_text, 64);
            apply_rope(theta_g2, g2k_rotated, text_len, H_text, 64);
            g2_attn_out = run_text_attention_cache(g2_attn_cache, model,
                g2q_rotated, g2k_rotated, g2v_out,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3200",
                "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.out_fc.linear.bias",
                current_step, "g2_attn_flash",
                include_ggml_trace ? &g2_attn_ctx_trace : nullptr);
        }
        PUSH_GGML_TRACE({"ve_g2_attn_q_rope", {L, 256}, g2_group.q_rope});
        PUSH_GGML_TRACE({"ve_g2_attn_k_rope", {text_len, 256}, g2_group.k_rope});
        PUSH_GGML_TRACE({"ve_g2_attn_ctx", {L, 256}, g2_attn_ctx_trace});
        PUSH_GGML_TRACE({"ve_g2_attn_out", {L, C}, g2_attn_out});

        thread_local vector_res_style_qkv_cache g2_res_qkv_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g2_res_qkv_cache, free_res_style_qkv_cache);
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
        // QVAC-18605 round 9 — style flash-attn GPU bridge for g2.
        thread_local vector_text_attention_cache g2_style_attn_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g2_style_attn_cache, free_text_attention_cache);
        std::vector<float> g2_style_ctx_trace;
        std::vector<float> g2_style_out;
        const bool g2_style_use_gpu_bridge = !include_ggml_trace
            && g2_res_qkv.sq_gpu && g2_res_qkv.sk_gpu && g2_res_qkv.sv_gpu;
        if (g2_style_use_gpu_bridge) {
            g2_style_out = run_text_attention_cache_gpu(g2_style_attn_cache, model,
                g2_res_qkv.sq_gpu, g2_res_qkv.sk_gpu, g2_res_qkv.sv_gpu,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3209",
                "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.out_fc.linear.bias",
                current_step, "g2_style_flash",
                /*ctx_trace=*/ nullptr);
        } else {
            std::vector<float> g2sq_out = std::move(g2_res_qkv.sq);
            std::vector<float> g2sk_out = std::move(g2_res_qkv.sk);
            std::vector<float> g2sv_out = std::move(g2_res_qkv.sv);
            g2_style_out = run_text_attention_cache(g2_style_attn_cache, model, g2sq_out, g2sk_out, g2sv_out,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3209",
                "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.out_fc.linear.bias",
                current_step, "g2_style_flash",
                include_ggml_trace ? &g2_style_ctx_trace : nullptr);
        }
        PUSH_GGML_TRACE({"ve_g2_style_ctx", {L, 256}, g2_style_ctx_trace});
        PUSH_GGML_TRACE({"ve_g2_style_out", {L, C}, g2_style_out});

        // F8: cached style-residual graph (norm_block = 17 for group 2).
        thread_local vector_style_residual_graph_cache g2_style_res_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g2_style_res_cache, free_style_residual_cache);
        std::vector<float> g2_style_res_trace;
        std::vector<float> g2_style_norm_vec = run_style_residual_cache(
            g2_style_res_cache, model, g2_block16, g2_style_out,
            L, C, /*norm_block=*/17, current_step, "g2_style_residual",
            include_ggml_trace ? &g2_style_res_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g2_style_residual", {L, C}, g2_style_res_trace});
        PUSH_GGML_TRACE({"ve_g2_style_norm", {L, C}, g2_style_norm_vec});

        thread_local vector_group_graph_cache g3_group_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g3_group_cache, free_group_graph_cache);
        vector_group_graph_result g3_group = run_group_graph_cache(g3_group_cache, model, g2_style_norm_vec,
            L, C, te_host, text_emb, text_len, current_step,
            3, 18, 19, "vector_estimator:onnx::MatMul_3230", 20,
            "vector_estimator:onnx::MatMul_3236",
            "vector_estimator:onnx::MatMul_3237",
            "vector_estimator:onnx::MatMul_3238",
            "ve_g3_attn_q", "ve_g3_attn_k", "ve_g3_attn_v",
            "group3_conv_attn_qkv", include_ggml_trace ? &ggml_trace : nullptr);
        std::vector<float> g3_block20 = std::move(g3_group.post);
        // 2C-lite — same GPU fast-path / host-fallback pattern as g1, g2.
        thread_local vector_text_attention_cache g3_attn_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g3_attn_cache, free_text_attention_cache);
        std::vector<float> g3_attn_ctx_trace;
        std::vector<float> g3_attn_out;
        if (g3_group.q_rope_gpu && g3_group.k_rope_gpu && g3_group.v_gpu) {
            g3_attn_out = run_text_attention_cache_gpu(g3_attn_cache, model,
                g3_group.q_rope_gpu, g3_group.k_rope_gpu, g3_group.v_gpu,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3245",
                "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.out_fc.linear.bias",
                current_step, "g3_attn_flash",
                include_ggml_trace ? &g3_attn_ctx_trace : nullptr);
        } else {
            std::vector<float> g3q_out = std::move(g3_group.q);
            std::vector<float> g3k_out = std::move(g3_group.k);
            std::vector<float> g3v_out = std::move(g3_group.v);
            std::vector<float> g3q_rotated = g3q_out;
            std::vector<float> g3k_rotated = g3k_out;
            const float * theta_g3 = model.vector_rope_theta.data();
            apply_rope(theta_g3, g3q_rotated, L, H_text, 64);
            apply_rope(theta_g3, g3k_rotated, text_len, H_text, 64);
            g3_attn_out = run_text_attention_cache(g3_attn_cache, model,
                g3q_rotated, g3k_rotated, g3v_out,
                L, text_len, H_text, 64,
                "vector_estimator:onnx::MatMul_3245",
                "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.out_fc.linear.bias",
                current_step, "g3_attn_flash",
                include_ggml_trace ? &g3_attn_ctx_trace : nullptr);
        }
        PUSH_GGML_TRACE({"ve_g3_attn_q_rope", {L, 256}, g3_group.q_rope});
        PUSH_GGML_TRACE({"ve_g3_attn_k_rope", {text_len, 256}, g3_group.k_rope});
        PUSH_GGML_TRACE({"ve_g3_attn_ctx", {L, 256}, g3_attn_ctx_trace});
        PUSH_GGML_TRACE({"ve_g3_attn_out", {L, C}, g3_attn_out});

        thread_local vector_res_style_qkv_cache g3_res_qkv_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g3_res_qkv_cache, free_res_style_qkv_cache);
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
        // QVAC-18605 round 9 — style flash-attn GPU bridge for g3.
        thread_local vector_text_attention_cache g3_style_attn_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g3_style_attn_cache, free_text_attention_cache);
        std::vector<float> g3_style_ctx_trace;
        std::vector<float> g3_style_out;
        const bool g3_style_use_gpu_bridge = !include_ggml_trace
            && g3_res_qkv.sq_gpu && g3_res_qkv.sk_gpu && g3_res_qkv.sv_gpu;
        if (g3_style_use_gpu_bridge) {
            g3_style_out = run_text_attention_cache_gpu(g3_style_attn_cache, model,
                g3_res_qkv.sq_gpu, g3_res_qkv.sk_gpu, g3_res_qkv.sv_gpu,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3254",
                "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.out_fc.linear.bias",
                current_step, "g3_style_flash",
                /*ctx_trace=*/ nullptr);
        } else {
            std::vector<float> g3sq_out = std::move(g3_res_qkv.sq);
            std::vector<float> g3sk_out = std::move(g3_res_qkv.sk);
            std::vector<float> g3sv_out = std::move(g3_res_qkv.sv);
            g3_style_out = run_text_attention_cache(g3_style_attn_cache, model, g3sq_out, g3sk_out, g3sv_out,
                L, 50, 2, 128,
                "vector_estimator:onnx::MatMul_3254",
                "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.out_fc.linear.bias",
                current_step, "g3_style_flash",
                include_ggml_trace ? &g3_style_ctx_trace : nullptr);
        }
        PUSH_GGML_TRACE({"ve_g3_style_ctx", {L, 256}, g3_style_ctx_trace});
        PUSH_GGML_TRACE({"ve_g3_style_out", {L, C}, g3_style_out});

        // F8: cached style-residual graph (norm_block = 23 for group 3).
        thread_local vector_style_residual_graph_cache g3_style_res_cache;
        SUPERTONIC_REGISTER_TL_CACHE(g3_style_res_cache, free_style_residual_cache);
        std::vector<float> g3_style_res_trace;
        std::vector<float> g3_style_norm_vec = run_style_residual_cache(
            g3_style_res_cache, model, g3_block22, g3_style_out,
            L, C, /*norm_block=*/23, current_step, "g3_style_residual",
            include_ggml_trace ? &g3_style_res_trace : nullptr);
        PUSH_GGML_TRACE({"ve_g3_style_residual", {L, C}, g3_style_res_trace});
        PUSH_GGML_TRACE({"ve_g3_style_norm", {L, C}, g3_style_norm_vec});

        thread_local vector_tail_graph_cache tail_cache;
        SUPERTONIC_REGISTER_TL_CACHE(tail_cache, free_tail_graph_cache);
        std::vector<float> next_latent_tc = run_tail_graph_cache(tail_cache, model, g3_style_norm_vec,
            noisy_latent, latent_mask, L, C, Cin, current_step, total_steps,
            include_ggml_trace ? &ggml_trace : nullptr);
        if (next_latent_tc_out) *next_latent_tc_out = next_latent_tc;

        // F19: front-block ctx + allocr live in `front_cache` and
        // survive across denoise steps; no per-call ctx to free.
        profile_vector_step_end(current_step);
        if (error) error->clear();
#undef PUSH_GGML_TRACE
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

// Apply Supertonic's non-standard RoPE in-graph.
// Supertonic uses angle = (t/L) * theta[d_half], where theta is loaded from
// the GGUF and L is the per-call sequence length.  ggml_rope_ext's formula
// expands to angle = (pos / freq_factors[d/2]) * freq_scale * freq_base^(-d/n_dims).
// Setting freq_base=1, freq_scale=1, freq_factors[d_half] = L / theta[d_half],
// positions = [0..L) reproduces the Supertonic formula exactly.  NEOX mode
// matches apply_rope's split-pairs layout (x[d] rotates with x[d+D/2]) at
// supertonic_vector_estimator.cpp:1416.
//
// x_tc must be a contiguous 2D tensor of shape ne=[H*D, q_len] (width-major).
// `positions` is int32 [q_len], `freq_factors` is f32 [D/2]; both are caller-
// owned input tensors set via ggml_backend_tensor_set before compute.
ggml_tensor * apply_supertonic_rope_ggml(ggml_context * ctx,
                                          ggml_tensor * x_tc,
                                          ggml_tensor * positions,
                                          ggml_tensor * freq_factors,
                                          int q_len,
                                          int H,
                                          int D) {
    GGML_ASSERT(x_tc->ne[0] == (int64_t)(H*D));
    GGML_ASSERT(x_tc->ne[1] == (int64_t)q_len);
    const size_t row_bytes = (size_t)(H*D) * sizeof(float);
    const size_t head_bytes = (size_t)D * sizeof(float);
    // View [H*D, q_len] as [D, H, q_len] so rope's outer dim is time.
    // Strides: nb1 = head step (D floats), nb2 = time step (H*D floats).
    // This view is naturally contiguous (nb[0]=elem_size, nb[1]=D*elem_size,
    // nb[2]=H*D*elem_size = ne[0]*ne[1]*elem_size) so we can skip the
    // ggml_cont copy that earlier versions inserted defensively.
    ggml_tensor * x_view = ggml_view_3d(ctx, x_tc, D, H, q_len,
                                         head_bytes, row_bytes, 0);
    ggml_tensor * roped = ggml_rope_ext(ctx, x_view, positions, freq_factors,
                                         D, GGML_ROPE_TYPE_NEOX, 0,
                                         /*freq_base=*/1.0f,
                                         /*freq_scale=*/1.0f,
                                         /*ext_factor=*/0.0f,
                                         /*attn_factor=*/1.0f,
                                         /*beta_fast=*/0.0f,
                                         /*beta_slow=*/0.0f);
    return ggml_reshape_2d(ctx, roped, (int64_t) H * D, q_len);
}

// Append a text-attention subgraph (Q, K, V flash-attention + out projection +
// bias add) to the parent (ctx, gf).  Mirrors build_text_attention_cache but
// composes into the caller's context instead of owning one.
//
// Inputs:
//   q_tc, k_tc, v_tc: contiguous [H*D, *_len] tensors
//   out_w_tensor: model tensor for the out projection weight
//   out_b_tensor: model tensor for the out projection bias
// Returns: out_tc tensor of shape [out_dim, q_len].
ggml_tensor * append_text_attention_subgraph(ggml_context * ctx,
                                              const supertonic_model & model,
                                              ggml_tensor * q_tc,
                                              ggml_tensor * k_tc,
                                              ggml_tensor * v_tc,
                                              int q_len, int kv_len,
                                              int n_heads, int head_dim,
                                              ggml_tensor * out_w_tensor,
                                              ggml_tensor * out_b_tensor,
                                              float scale) {
    const int width = n_heads * head_dim;
    const size_t time_stride = (size_t)width * sizeof(float);
    const size_t head_stride = (size_t)head_dim * sizeof(float);
    ggml_tensor * q_in = ggml_view_3d(ctx, q_tc,
        head_dim, q_len, n_heads, time_stride, head_stride, 0);
    ggml_tensor * k_in = ggml_view_3d(ctx, k_tc,
        head_dim, kv_len, n_heads, time_stride, head_stride, 0);
    ggml_tensor * v_in = ggml_view_3d(ctx, v_tc,
        head_dim, kv_len, n_heads, time_stride, head_stride, 0);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q_in, k_in, v_in,
                                              nullptr, scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, (int64_t) n_heads * head_dim, q_len);
    ggml_tensor * ctx_tc = ggml_cont(ctx, ggml_transpose(ctx, attn));
    return dense_matmul_time_pretransposed_ggml(ctx, model, ctx_tc, out_w_tensor, out_b_tensor);
}

// Per-group MatMul tensor name suffixes (groups 0..3).  See per-group source
// names in trace_proj_ggml; these tables centralise them for the consolidated
// path.
struct vector_step_group_names {
    int t_linear;    // time-linear (matmul for time embedding projection)
    int attn_q;
    int attn_k;
    int attn_v;
    int attn_out;
    int style_q;
    int style_k;
    int style_v;
    int style_out;
};

static const vector_step_group_names kGroupNames[4] = {
    {3095, 3101, 3102, 3103, 3110, 3116, 3117, 3118, 3119},
    {3140, 3146, 3147, 3148, 3155, 3161, 3162, 3163, 3164},
    {3185, 3191, 3192, 3193, 3200, 3206, 3207, 3208, 3209},
    {3230, 3236, 3237, 3238, 3245, 3251, 3252, 3253, 3254},
};

static std::string matmul_name(int suffix) {
    return "vector_estimator:onnx::MatMul_" + std::to_string(suffix);
}

// Bundle of input tensors a single CFM step subgraph needs.  Used both by
// the per-step cache (one step per ggml_cgraph) and by the
// 5-steps-unrolled-into-one-graph cache (Phase A1+A2).
//
// `x_in` / `noise_in` vary per step (x_in = latent for this step,
// noise_in is the "residual" we add the velocity to — for Supertonic's
// CFM equation `next = noise_in + velocity * (1 / total_steps)` they
// happen to be the same tensor for a single step but become DIFFERENT
// tensors when steps are chained: step N's x_in is step N-1's output,
// while noise_in is still the original noisy latent that step.  In the
// per-step path we bind them to the same external buffer; in the
// unrolled-loop path we wire them as graph edges between steps).
//
// `t_emb_in` varies per step (one time embedding per CFM step index).
// All other inputs are constant across the 5 CFM steps and bind to a
// single shared input tensor regardless of which path is used.
struct vector_step_inputs {
    ggml_tensor * x_in           = nullptr;  // ne=[L, Cin]    f32
    ggml_tensor * mask_in        = nullptr;  // ne=[L]         f32
    ggml_tensor * t_emb_in       = nullptr;  // ne=[64]        f32  (per-step)
    ggml_tensor * text_in        = nullptr;  // ne=[text_len, 256] f32
    ggml_tensor * style_v_raw_in = nullptr;  // ne=[50, 256]   f32
    ggml_tensor * style_kctx_in  = nullptr;  // ne=[50, 256]   f32
    ggml_tensor * noise_in       = nullptr;  // ne=[L, Cin]    f32  (per-step)
    ggml_tensor * pos_q          = nullptr;  // ne=[L]         i32
    ggml_tensor * pos_k          = nullptr;  // ne=[text_len]  i32
    ggml_tensor * freq_factors_q = nullptr;  // ne=[D/2]       f32
    ggml_tensor * freq_factors_k = nullptr;  // ne=[D/2]       f32
};

// Append one CFM step's subgraph (proj_in → 4 groups → tail → proj_out
// → velocity → next = noise + velocity / total_steps) to `gf`.  All
// inputs are pre-bound by the caller; this function only builds the
// dataflow and returns the `next` tensor (ne=[L, Cin]) so the caller
// can either set it as a graph output or feed it as the next step's
// `x_in`.  The function does NOT call `ggml_set_output` /
// `ggml_build_forward_expand` on the result — that's the caller's
// decision.
//
// `L`, `text_len` and `total_steps` are passed explicitly because they're
// used in several places.  CPU vs GPU dispatch lives on the thread-local
// `supertonic_use_cpu_custom_ops()` flag set by the outer
// `supertonic_op_dispatch_scope` at the public entry point.
ggml_tensor * append_supertonic_vector_step_subgraph(
        ggml_context * gctx,
        ggml_cgraph * gf,
        const supertonic_model & model,
        const vector_step_inputs & inputs,
        int L,
        int text_len,
        int total_steps);

// Consolidated per-step cache: one ctx, one cgraph, one gallocr for the entire
// per-step computation.  Replaces the ~17 sub-graph dispatches the trace_proj
// orchestrator emits with a single ggml_backend_graph_compute call.
struct vector_step_one_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int L = 0;
    int text_len = 0;
    int total_steps = 0;

    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Per-call inputs
    ggml_tensor * x_in = nullptr;          // noisy_latent (L, Cin) ggml-shape: ne=[L, Cin]
    ggml_tensor * mask_in = nullptr;       // [L]
    ggml_tensor * t_emb_in = nullptr;      // [64]
    ggml_tensor * text_in = nullptr;       // [text_len, 256]
    ggml_tensor * style_v_raw_in = nullptr; // [50, 256] (style_ttl repacked)
    ggml_tensor * style_kctx_in = nullptr;  // [50, 256] (model's /Expand_output_0)
    ggml_tensor * noise_in = nullptr;       // (L, Cin) (same data as x_in but indep slot for tail)
    // CFG unconditional-pass inputs (v3 only; null when cfg disabled): learned
    // null tokens replacing text / style-K / style-V.
    ggml_tensor * text_in_u = nullptr;       // [text_len, 256]
    ggml_tensor * style_v_raw_in_u = nullptr; // [50, 256]
    ggml_tensor * style_kctx_in_u = nullptr;  // [50, 256]

    // Per-build (rope) inputs
    ggml_tensor * pos_q = nullptr;          // int32 [L]
    ggml_tensor * pos_k = nullptr;          // int32 [text_len]
    ggml_tensor * freq_factors_q = nullptr; // f32 [32] (head_dim/2)
    ggml_tensor * freq_factors_k = nullptr; // f32 [32]

    // Output
    ggml_tensor * next_latent_out = nullptr; // ne=[L, Cin] in (t, c) order
};

void free_vector_step_one_graph_cache(vector_step_one_graph_cache & cache) {
    if (cache.allocr) {
        supertonic_safe_gallocr_free(cache.allocr, cache.model ? cache.model->generation_id : 0);
        cache.allocr = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.gf = nullptr;
    cache.buf.clear();
    cache.model = nullptr;
    cache.generation_id = 0;
    cache.L = 0;
    cache.text_len = 0;
    cache.total_steps = 0;
    cache.x_in = cache.mask_in = cache.t_emb_in = cache.text_in = nullptr;
    cache.style_v_raw_in = cache.style_kctx_in = cache.noise_in = nullptr;
    cache.text_in_u = cache.style_v_raw_in_u = cache.style_kctx_in_u = nullptr;
    cache.pos_q = cache.pos_k = cache.freq_factors_q = cache.freq_factors_k = nullptr;
    cache.next_latent_out = nullptr;
}

ggml_tensor * append_supertonic_vector_step_subgraph(
        ggml_context * gctx,
        ggml_cgraph * gf,
        const supertonic_model & model,
        const vector_step_inputs & inputs,
        int L,
        int text_len,
        int total_steps) {
    const bool use_cpu_custom = supertonic_use_cpu_custom_ops();
    // Shape constants that aren't dependent on L / text_len.  Mirror the
    // values from supertonic_vector_step_one_graph_ggml.
    const int C = 512;
    const int H = model.hparams.vector_text_attn_heads;  // text-attn heads: v1/v2=4, v3=8
    const int D = 64;       // text-attention head_dim (constant; width A = H*D)
    const int SH = 2;       // style-attention heads
    const int SD = 128;     // style-attention head_dim
    const int kv_style = 50; // fixed by /Expand_output_0
    (void)H; (void)D; (void)SH; (void)SD; (void)kv_style;

    // ===== PHASE 0: proj_in + mask =====
    ggml_tensor * cur = conv1d_f32(gctx,
        require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_in.net.weight"),
        inputs.x_in, 1, 0, 1);
    cur = ggml_mul(gctx, cur, repeat_like(gctx, inputs.mask_in, cur));

    // ===== PHASE 1: Group 0 prologue — ConvNeXt × 4 on main_blocks.0 + time_add (1) + ConvNeXt (2) =====
    int dils[4] = {1, 2, 4, 8};
    // Phase B2 full: permute to [C, T] once before the 4-block chain, run
    // the chain in [C, T] (which lets each block's two pointwise convs
    // become a direct ggml_mul_mat with no im2col), permute back to
    // [T, C] for the downstream time-add.  Saves 2 im2col dispatches per
    // block × 4 blocks × 5 steps − 2 permutes per chain × 5 steps =
    // 30 dispatches eliminated per synth.  Override:
    // SUPERTONIC_DISABLE_CT_CONVNEXT=1.
    static const bool disable_ct_convnext =
        std::getenv("SUPERTONIC_DISABLE_CT_CONVNEXT") != nullptr;
    const bool use_ct_convnext = !disable_ct_convnext && !use_cpu_custom;
    if (use_ct_convnext) {
        ggml_tensor * cur_ct = ggml_cont(gctx, ggml_permute(gctx, cur, 1, 0, 2, 3));
        for (int j = 0; j < 4; ++j) {
            cur_ct = vector_convnext_ggml_ct(gctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext." + std::to_string(j),
                cur_ct, dils[j]);
        }
        cur = ggml_cont(gctx, ggml_permute(gctx, cur_ct, 1, 0, 2, 3));
    } else {
        for (int j = 0; j < 4; ++j) {
            cur = vector_convnext_ggml(gctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext." + std::to_string(j),
                cur, dils[j]);
        }
    }
    // Time-add for group 0.
    {
        ggml_tensor * w = require_source_tensor(model, matmul_name(kGroupNames[0].t_linear));
        ggml_tensor * b = require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.1.linear.linear.bias");
        ggml_tensor * w_t = try_pretransposed_weight(model, w);
        if (!w_t) w_t = ggml_cont(gctx, ggml_transpose(gctx, w));
        ggml_tensor * t_proj = ggml_mul_mat(gctx, w_t, ggml_reshape_2d(gctx, inputs.t_emb_in, 64, 1));
        t_proj = ggml_add(gctx, t_proj, ggml_reshape_2d(gctx, b, C, 1));
        cur = ggml_add(gctx, cur, repeat_like(gctx, t_proj, cur));
    }
    cur = vector_convnext_ggml(gctx, model,
        "vector_estimator:tts.ttl.vector_field.main_blocks.2.convnext.0",
        cur, 1);
    ggml_tensor * block_pre_attn = cur;

    // Per-group attention block.
    auto run_group = [&](ggml_tensor * x, int group, ggml_tensor * x_pre_attn) -> ggml_tensor * {
        const auto & names = kGroupNames[group];
        const int attn_block = group * 6 + 3;
        const int post_attn_block = group * 6 + 4;
        const int style_block = group * 6 + 5;

        // Text attention QKV — output directly in [A, T] (width-major)
        // layout so the cont(transpose) before rope/flash_attn is gone.
        // The kernel-as-src0 ordering also dispatches the optimized
        // kernel_mul_mm_q8_0_f32 when weights are q8_0.
        ggml_tensor * q_wt = dense_matmul_time_wt_pretransposed_ggml(gctx, model, x_pre_attn,
            require_source_tensor(model, matmul_name(names.attn_q)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(attn_block) + ".attn.W_query.linear.bias"));
        ggml_tensor * k_wt = dense_matmul_time_wt_pretransposed_ggml(gctx, model, inputs.text_in,
            require_source_tensor(model, matmul_name(names.attn_k)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(attn_block) + ".attn.W_key.linear.bias"));
        ggml_tensor * v_wt = dense_matmul_time_wt_pretransposed_ggml(gctx, model, inputs.text_in,
            require_source_tensor(model, matmul_name(names.attn_v)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(attn_block) + ".attn.W_value.linear.bias"));

        q_wt = apply_supertonic_rope_ggml(gctx, q_wt, inputs.pos_q, inputs.freq_factors_q, L, H, D);
        k_wt = apply_supertonic_rope_ggml(gctx, k_wt, inputs.pos_k, inputs.freq_factors_k, text_len, H, D);

        ggml_tensor * attn_out = append_text_attention_subgraph(gctx, model,
            q_wt, k_wt, v_wt, L, text_len, H, D,
            require_source_tensor(model, matmul_name(names.attn_out)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(attn_block) + ".attn.out_fc.linear.bias"),
            1.0f / 16.0f);

        ggml_tensor * residual = ggml_add(gctx, x_pre_attn, attn_out);
        ggml_tensor * normed = layer_norm_ggml(gctx, residual,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(attn_block) + ".norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(attn_block) + ".norm.norm.bias"));

        ggml_tensor * post = vector_convnext_ggml(gctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks." +
            std::to_string(post_attn_block) + ".convnext.0",
            normed, 1);

        ggml_tensor * masked_post = ggml_mul(gctx, post, repeat_like(gctx, inputs.mask_in, post));

        // Style attention QKV — output directly in [A, T] layout.
        ggml_tensor * sq_wt = dense_matmul_time_wt_pretransposed_ggml(gctx, model, masked_post,
            require_source_tensor(model, matmul_name(names.style_q)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(style_block) + ".attention.W_query.linear.bias"));
        ggml_tensor * sk_wt = dense_matmul_time_wt_pretransposed_ggml(gctx, model, inputs.style_kctx_in,
            require_source_tensor(model, matmul_name(names.style_k)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(style_block) + ".attention.W_key.linear.bias"));
        sk_wt = ggml_tanh(gctx, sk_wt);
        ggml_tensor * sv_wt = dense_matmul_time_wt_pretransposed_ggml(gctx, model, inputs.style_v_raw_in,
            require_source_tensor(model, matmul_name(names.style_v)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(style_block) + ".attention.W_value.linear.bias"));

        ggml_tensor * style_out = append_text_attention_subgraph(gctx, model,
            sq_wt, sk_wt, sv_wt, L, kv_style, SH, SD,
            require_source_tensor(model, matmul_name(names.style_out)),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(style_block) + ".attention.out_fc.linear.bias"),
            1.0f / 16.0f);

        ggml_tensor * style_residual = ggml_add(gctx, post, style_out);
        ggml_tensor * style_normed = layer_norm_ggml(gctx, style_residual,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(style_block) + ".norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks." +
                                         std::to_string(style_block) + ".norm.norm.bias"));
        (void)x;
        return style_normed;
    };

    // Group prep for groups 1-3.
    auto group_prep = [&](ggml_tensor * x, int group) -> ggml_tensor * {
        const int conv_block = group * 6 + 0;
        const int linear_block = group * 6 + 1;
        const int post_block = group * 6 + 2;
        int dils2[4] = {1, 2, 4, 8};
        ggml_tensor * y = x;
        if (use_ct_convnext) {
            ggml_tensor * y_ct = ggml_cont(gctx, ggml_permute(gctx, y, 1, 0, 2, 3));
            for (int j = 0; j < 4; ++j) {
                y_ct = vector_convnext_ggml_ct(gctx, model,
                    "vector_estimator:tts.ttl.vector_field.main_blocks." +
                    std::to_string(conv_block) + ".convnext." + std::to_string(j),
                    y_ct, dils2[j]);
            }
            y = ggml_cont(gctx, ggml_permute(gctx, y_ct, 1, 0, 2, 3));
        } else {
            for (int j = 0; j < 4; ++j) {
                y = vector_convnext_ggml(gctx, model,
                    "vector_estimator:tts.ttl.vector_field.main_blocks." +
                    std::to_string(conv_block) + ".convnext." + std::to_string(j),
                    y, dils2[j]);
            }
        }
        ggml_tensor * w = require_source_tensor(model, matmul_name(kGroupNames[group].t_linear));
        ggml_tensor * b = require_source_tensor(model,
            "vector_estimator:tts.ttl.vector_field.main_blocks." +
            std::to_string(linear_block) + ".linear.linear.bias");
        ggml_tensor * w_t = try_pretransposed_weight(model, w);
        if (!w_t) w_t = ggml_cont(gctx, ggml_transpose(gctx, w));
        ggml_tensor * t_proj = ggml_mul_mat(gctx, w_t, ggml_reshape_2d(gctx, inputs.t_emb_in, 64, 1));
        t_proj = ggml_add(gctx, t_proj, ggml_reshape_2d(gctx, b, C, 1));
        y = ggml_add(gctx, y, repeat_like(gctx, t_proj, y));
        y = vector_convnext_ggml(gctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks." +
            std::to_string(post_block) + ".convnext.0",
            y, 1);
        return y;
    };

    ggml_tensor * x_after_g0 = run_group(cur, 0, block_pre_attn);
    ggml_tensor * x_pre_g1 = group_prep(x_after_g0, 1);
    ggml_tensor * x_after_g1 = run_group(x_after_g0, 1, x_pre_g1);
    ggml_tensor * x_pre_g2 = group_prep(x_after_g1, 2);
    ggml_tensor * x_after_g2 = run_group(x_after_g1, 2, x_pre_g2);
    ggml_tensor * x_pre_g3 = group_prep(x_after_g2, 3);
    ggml_tensor * x_after_g3 = run_group(x_after_g2, 3, x_pre_g3);

    // Tail: last_convnext × 4 + proj_out + mask + noise add.
    ggml_tensor * tail = x_after_g3;
    if (use_ct_convnext) {
        ggml_tensor * tail_ct = ggml_cont(gctx, ggml_permute(gctx, tail, 1, 0, 2, 3));
        for (int j = 0; j < 4; ++j) {
            tail_ct = vector_convnext_ggml_ct(gctx, model,
                "vector_estimator:tts.ttl.vector_field.last_convnext.convnext." + std::to_string(j),
                tail_ct, 1);
        }
        tail = ggml_cont(gctx, ggml_permute(gctx, tail_ct, 1, 0, 2, 3));
    } else {
        for (int j = 0; j < 4; ++j) {
            tail = vector_convnext_ggml(gctx, model,
                "vector_estimator:tts.ttl.vector_field.last_convnext.convnext." + std::to_string(j),
                tail, 1);
        }
    }
    ggml_tensor * velocity = conv1d_f32(gctx,
        require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),
        tail, 1, 0, 1);
    ggml_tensor * masked_velocity = ggml_mul(gctx, velocity, repeat_like(gctx, inputs.mask_in, velocity));
    ggml_tensor * scaled = ggml_scale(gctx, masked_velocity, 1.0f / (float)total_steps);
    ggml_tensor * next = ggml_add(gctx, inputs.noise_in, scaled);

    // Mark gf as used so the unused-parameter warning doesn't fire — the
    // graph build is via the tensors above which inherit gf via ctx.
    (void)gf;
    return next;
}


// Compute one CFM denoising step as ONE ggml graph.  Used only when the
// model's backend isn't CPU (Metal / CUDA / Vulkan / OpenCL).  Replaces the
// ~21 sub-graph dispatches the trace_proj orchestrator emits with a single
// ggml_backend_graph_compute call.
bool supertonic_vector_step_one_graph_ggml(const supertonic_model & model,
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
    // The outer entry point sets `supertonic_op_dispatch_scope`; this
    // function is only called on non-CPU backends, so the thread-local
    // `supertonic_use_cpu_custom_ops()` reads false inside the helpers.
    if (std::getenv("SUPERTONIC_VE_DEBUG"))
        std::fprintf(stderr, "[ve-cfg] one_graph entry: cfg_enabled=%d cond=%g uncond=%g\n",
            (int)model.hparams.cfg_enabled(), model.hparams.cfg_cond_scale, model.hparams.cfg_uncond_scale);
    try {
        const int L = latent_len;
        const int Cin = model.hparams.latent_channels;  // typically 16
        const int C = 512;
        const int text_C = 256;
        const int H = model.hparams.vector_text_attn_heads;  // v1/v2=4, v3=8
        const int D = 64;       // text-attention head_dim
        const int A = H * D;    // attention width: 256 (v1/v2) or 512 (v3)
        const int SH = 2;       // style-attention heads
        const int SD = 128;     // style-attention head_dim
        const int kv_style = 50; // style attention kv length (fixed by /Expand_output_0)

        thread_local vector_step_one_graph_cache cache;
        SUPERTONIC_REGISTER_TL_CACHE(cache, free_vector_step_one_graph_cache);
        const bool need_rebuild = cache.model != &model ||
                                  cache.generation_id != model.generation_id ||
                                  cache.L != L ||
                                  cache.text_len != text_len ||
                                  cache.total_steps != total_steps;
        if (need_rebuild) {
            free_vector_step_one_graph_cache(cache);
            cache.model = &model;
            cache.generation_id = model.generation_id;
            cache.L = L;
            cache.text_len = text_len;
            cache.total_steps = total_steps;

            // Memory budget for the consolidated graph.  The original
            // sub-graphs each used 128-512 nodes; the full per-step graph is
            // roughly the sum (4 groups x ~700 ops/group + tail + front).
            // Round up generously.
            // CFG (v3) builds the field twice (cond + uncond) in one graph.
            const int MAX_NODES = model.hparams.cfg_enabled() ? 16384 : 8192;
            const size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                                     ggml_graph_overhead_custom(MAX_NODES, false);
            cache.buf.assign(buf_size, 0);
            ggml_init_params p = { buf_size, cache.buf.data(), true };
            cache.ctx = ggml_init(p);
            cache.gf = ggml_new_graph_custom(cache.ctx, MAX_NODES, false);

            // --- Per-call inputs ---
            cache.x_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, Cin);
            ggml_set_name(cache.x_in, "step_x_in"); ggml_set_input(cache.x_in);
            cache.mask_in = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, L);
            ggml_set_name(cache.mask_in, "step_mask"); ggml_set_input(cache.mask_in);
            cache.t_emb_in = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, 64);
            ggml_set_name(cache.t_emb_in, "step_temb"); ggml_set_input(cache.t_emb_in);
            cache.text_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, text_len, text_C);
            ggml_set_name(cache.text_in, "step_text_in"); ggml_set_input(cache.text_in);
            cache.style_v_raw_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
            ggml_set_name(cache.style_v_raw_in, "step_style_v"); ggml_set_input(cache.style_v_raw_in);
            cache.style_kctx_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
            ggml_set_name(cache.style_kctx_in, "step_style_kctx"); ggml_set_input(cache.style_kctx_in);
            cache.noise_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, Cin);
            ggml_set_name(cache.noise_in, "step_noise_in"); ggml_set_input(cache.noise_in);
            if (model.hparams.cfg_enabled()) {
                cache.text_in_u = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, text_len, text_C);
                ggml_set_name(cache.text_in_u, "step_text_in_u"); ggml_set_input(cache.text_in_u);
                cache.style_v_raw_in_u = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
                ggml_set_name(cache.style_v_raw_in_u, "step_style_v_u"); ggml_set_input(cache.style_v_raw_in_u);
                cache.style_kctx_in_u = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
                ggml_set_name(cache.style_kctx_in_u, "step_style_kctx_u"); ggml_set_input(cache.style_kctx_in_u);
            }

            // --- RoPE inputs ---
            cache.pos_q = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_I32, L);
            ggml_set_name(cache.pos_q, "step_pos_q"); ggml_set_input(cache.pos_q);
            cache.pos_k = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_I32, text_len);
            ggml_set_name(cache.pos_k, "step_pos_k"); ggml_set_input(cache.pos_k);
            cache.freq_factors_q = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, D / 2);
            ggml_set_name(cache.freq_factors_q, "step_ff_q"); ggml_set_input(cache.freq_factors_q);
            cache.freq_factors_k = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, D / 2);
            ggml_set_name(cache.freq_factors_k, "step_ff_k"); ggml_set_input(cache.freq_factors_k);

            ggml_context * gctx = cache.ctx;
            ggml_cgraph * gf = cache.gf;

            vector_step_inputs inputs;
            inputs.x_in           = cache.x_in;
            inputs.mask_in        = cache.mask_in;
            inputs.t_emb_in       = cache.t_emb_in;
            inputs.text_in        = cache.text_in;
            inputs.style_v_raw_in = cache.style_v_raw_in;
            inputs.style_kctx_in  = cache.style_kctx_in;
            inputs.noise_in       = cache.noise_in;
            inputs.pos_q          = cache.pos_q;
            inputs.pos_k          = cache.pos_k;
            inputs.freq_factors_q = cache.freq_factors_q;
            inputs.freq_factors_k = cache.freq_factors_k;

            ggml_tensor * next = append_supertonic_vector_step_subgraph(
                gctx, gf, model, inputs, L, text_len, total_steps);

            if (model.hparams.cfg_enabled()) {
                // Unconditional pass shares everything but the conditioning.
                vector_step_inputs inputs_u = inputs;
                inputs_u.text_in        = cache.text_in_u;
                inputs_u.style_v_raw_in = cache.style_v_raw_in_u;
                inputs_u.style_kctx_in  = cache.style_kctx_in_u;
                ggml_tensor * next_u = append_supertonic_vector_step_subgraph(
                    gctx, gf, model, inputs_u, L, text_len, total_steps);
                // next = cond_scale*next_cond - uncond_scale*next_uncond
                // (noise term cancels because cond_scale - uncond_scale == 1).
                next = ggml_sub(gctx,
                    ggml_scale(gctx, next,   model.hparams.cfg_cond_scale),
                    ggml_scale(gctx, next_u, model.hparams.cfg_uncond_scale));
            }

            ggml_set_name(next, "step_next_latent");
            ggml_set_output(next);
            ggml_build_forward_expand(gf, next);
            cache.next_latent_out = next;


            // Allocate.
            cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new vector step one-graph failed");
            if (!ggml_gallocr_reserve(cache.allocr, gf)) {
                throw std::runtime_error("ggml_gallocr_reserve vector step one-graph failed");
            }
            ggml_gallocr_alloc_graph(cache.allocr, gf);
        }

        // ===== Per-call inputs =====
        // The existing trace_proj_ggml at lines 2143/2151 sets these tensors
        // DIRECTLY from the caller-provided channel-major buffers (no host
        // transpose), and the views downstream interpret memory accordingly.
        // Copy that pattern exactly — my earlier transpose loops were a bug
        // (correlation 0.003 vs CPU reference; root-caused 2026-05-11).
        ggml_backend_tensor_set(cache.x_in, noisy_latent, 0, (size_t)L * Cin * sizeof(float));
        ggml_backend_tensor_set(cache.noise_in, noisy_latent, 0, (size_t)L * Cin * sizeof(float));
        ggml_backend_tensor_set(cache.mask_in, latent_mask, 0, (size_t)L * sizeof(float));

        std::vector<float> te_host = time_embedding(model, current_step, total_steps);
        ggml_backend_tensor_set(cache.t_emb_in, te_host.data(), 0, te_host.size() * sizeof(float));

        // text_emb is in (C=256, text_len) channel-major; the tensor has
        // ne=[text_len, 256] which puts t_len fast in memory.  Same raw layout,
        // so direct memcpy (matches trace_proj_ggml).
        ggml_backend_tensor_set(cache.text_in, text_emb, 0, (size_t)text_len * 256 * sizeof(float));

        // Style inputs (cached host buffers from existing helper).
        const std::vector<float> * style_v_raw_ptr = nullptr;
        const std::vector<float> * kctx_raw_ptr = nullptr;
        cached_style_layouts(model, style_ttl, style_v_raw_ptr, kctx_raw_ptr);
        ggml_backend_tensor_set(cache.style_v_raw_in, style_v_raw_ptr->data(), 0, style_v_raw_ptr->size() * sizeof(float));
        ggml_backend_tensor_set(cache.style_kctx_in, kctx_raw_ptr->data(), 0, kctx_raw_ptr->size() * sizeof(float));

        if (model.hparams.cfg_enabled()) {
            // Learned null tokens, repacked to the same channel-major layouts
            // (text [256,text_len] c*Lt+t; style [256,50] c*50+t) as the cond
            // inputs above.
            f32_tensor tst = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.text_special_token");   // [256]
            f32_tensor skt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_key_special_token");   // [50,256] (t,c)
            f32_tensor svt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_value_special_token"); // [50,256] (t,c)
            std::vector<float> text_u((size_t)256 * text_len);
            for (int c = 0; c < 256; ++c) for (int t = 0; t < text_len; ++t) text_u[(size_t)c*text_len+t] = tst.data[c];
            std::vector<float> sv_u((size_t)256 * 50), sk_u((size_t)256 * 50);
            for (int c = 0; c < 256; ++c) for (int t = 0; t < 50; ++t) {
                sv_u[(size_t)c*50+t] = svt.data[(size_t)t*256+c];
                sk_u[(size_t)c*50+t] = skt.data[(size_t)t*256+c];
            }
            ggml_backend_tensor_set(cache.text_in_u, text_u.data(), 0, text_u.size() * sizeof(float));
            ggml_backend_tensor_set(cache.style_v_raw_in_u, sv_u.data(), 0, sv_u.size() * sizeof(float));
            ggml_backend_tensor_set(cache.style_kctx_in_u, sk_u.data(), 0, sk_u.size() * sizeof(float));
        }

        // RoPE positions + freq_factors.  theta is loaded from the model and
        // depends on L (sequence length); recompute per call.
        {
            std::vector<int32_t> pos_q_host(L);
            for (int i = 0; i < L; ++i) pos_q_host[i] = i;
            ggml_backend_tensor_set(cache.pos_q, pos_q_host.data(), 0, pos_q_host.size() * sizeof(int32_t));
            std::vector<int32_t> pos_k_host(text_len);
            for (int i = 0; i < text_len; ++i) pos_k_host[i] = i;
            ggml_backend_tensor_set(cache.pos_k, pos_k_host.data(), 0, pos_k_host.size() * sizeof(int32_t));

            const int half = 32;  // D/2 = 64/2
            f32_tensor theta = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            if ((int)theta.data.size() < half) {
                throw std::runtime_error("theta tensor has fewer than D/2 elements");
            }
            std::vector<float> ff_q(half), ff_k(half);
            for (int d = 0; d < half; ++d) {
                ff_q[d] = (float)L / theta.data[d];
                ff_k[d] = (float)text_len / theta.data[d];
            }
            ggml_backend_tensor_set(cache.freq_factors_q, ff_q.data(), 0, ff_q.size() * sizeof(float));
            ggml_backend_tensor_set(cache.freq_factors_k, ff_k.data(), 0, ff_k.size() * sizeof(float));
        }

        // ===== ONE compute call =====
        supertonic_graph_compute(model, cache.gf);

        // ===== Read output =====
        // The output tensor has ne=[L, Cin] with element (i=t, j=c) at offset
        // c*L+t — exactly the (c, t) channel-major layout the caller expects.
        // Direct memcpy, no transpose.
        next_latent_out.assign((size_t)Cin * L, 0.0f);
        ggml_backend_tensor_get(cache.next_latent_out, next_latent_out.data(), 0,
                                 (size_t)Cin * L * sizeof(float));
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

// =====================================================================
// Phase A1+A2 — single-graph CFM loop
// =====================================================================
//
// Unroll all `total_steps` CFM denoising steps into ONE ggml_cgraph and
// dispatch with a single ggml_backend_graph_compute call.  Each step's
// `x_in` and `noise_in` is the previous step's output node (no host
// round-trip), and only `t_emb_in` differs per step (N inputs, one
// per CFM step).  Replaces the engine's `for (step ...) {
// supertonic_vector_step_ggml(...) }` loop on non-CPU backends.
//
// CPU keeps the per-step path because its cblas fastpaths benefit from
// the cache-per-shape boundary and the host-side rope/style helpers in
// trace_proj_ggml expect to see per-step outputs.

struct vector_loop_one_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int L = 0;
    int text_len = 0;
    int total_steps = 0;

    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Shared inputs (constant across CFM steps).
    ggml_tensor * x0_in = nullptr;          // ne=[L, Cin]  initial noisy latent
    ggml_tensor * mask_in = nullptr;        // ne=[L]
    ggml_tensor * text_in = nullptr;        // ne=[text_len, 256]
    ggml_tensor * style_v_raw_in = nullptr; // ne=[50, 256]
    ggml_tensor * style_kctx_in = nullptr;  // ne=[50, 256]
    // CFG unconditional-pass inputs (v3 only; null when disabled).
    ggml_tensor * text_in_u = nullptr;
    ggml_tensor * style_v_raw_in_u = nullptr;
    ggml_tensor * style_kctx_in_u = nullptr;

    // RoPE inputs (constant across steps).
    ggml_tensor * pos_q = nullptr;
    ggml_tensor * pos_k = nullptr;
    ggml_tensor * freq_factors_q = nullptr;
    ggml_tensor * freq_factors_k = nullptr;

    // Per-step time embedding (one tensor per CFM step).
    std::vector<ggml_tensor *> t_emb_in;

    // Final output — last step's `next` tensor.
    ggml_tensor * final_latent_out = nullptr;
};

void free_vector_loop_one_graph_cache(vector_loop_one_graph_cache & cache) {
    if (cache.allocr) {
        supertonic_safe_gallocr_free(cache.allocr, cache.model ? cache.model->generation_id : 0);
        cache.allocr = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.gf = nullptr;
    cache.buf.clear();
    cache.model = nullptr;
    cache.generation_id = 0;
    cache.L = 0;
    cache.text_len = 0;
    cache.total_steps = 0;
    cache.x0_in = cache.mask_in = cache.text_in = nullptr;
    cache.style_v_raw_in = cache.style_kctx_in = nullptr;
    cache.text_in_u = cache.style_v_raw_in_u = cache.style_kctx_in_u = nullptr;
    cache.pos_q = cache.pos_k = cache.freq_factors_q = cache.freq_factors_k = nullptr;
    cache.t_emb_in.clear();
    cache.final_latent_out = nullptr;
}

bool supertonic_vector_loop_one_graph_ggml(const supertonic_model & model,
                                            const float * initial_noisy_latent,
                                            int latent_len,
                                            const float * text_emb,
                                            int text_len,
                                            const float * style_ttl,
                                            const float * latent_mask,
                                            int total_steps,
                                            std::vector<float> & final_latent_out,
                                            std::string * error) {
    // Public entry point — set the thread-local dispatch flag so the
    // helpers' `supertonic_use_cpu_custom_ops()` reads consistently
    // (false on non-CPU backends, true on CPU + accelerate/cblas).
    supertonic_op_dispatch_scope dispatch(model);
    try {
        const int L = latent_len;
        const int Cin = model.hparams.latent_channels;
        const int text_C = 256;
        const int D = 64;
        const int kv_style = 50;

        thread_local vector_loop_one_graph_cache cache;
        SUPERTONIC_REGISTER_TL_CACHE(cache, free_vector_loop_one_graph_cache);
        const bool need_rebuild = cache.model != &model ||
                                  cache.generation_id != model.generation_id ||
                                  cache.L != L ||
                                  cache.text_len != text_len ||
                                  cache.total_steps != total_steps;
        if (need_rebuild) {
            free_vector_loop_one_graph_cache(cache);
            cache.model = &model;
            cache.generation_id = model.generation_id;
            cache.L = L;
            cache.text_len = text_len;
            cache.total_steps = total_steps;

            // ~5x the per-step node budget.  Each per-step build registered ~1056
            // ggml nodes pre-Tier-2; post-Tier-2 it's ~928.  Round up to 8192/step
            // × total_steps = ~40k.  Plus the shared inputs (a few dozen) +
            // per-step temb input tensors.
            const int per_step_nodes = model.hparams.cfg_enabled() ? 16384 : 8192;
            const int MAX_NODES = per_step_nodes * std::max(1, total_steps) + 256;
            const size_t buf_size = ggml_tensor_overhead() * (size_t) MAX_NODES +
                                     ggml_graph_overhead_custom(MAX_NODES, false);
            cache.buf.assign(buf_size, 0);
            ggml_init_params p = { buf_size, cache.buf.data(), true };
            cache.ctx = ggml_init(p);
            cache.gf = ggml_new_graph_custom(cache.ctx, MAX_NODES, false);

            // --- Shared inputs ---
            cache.x0_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, Cin);
            ggml_set_name(cache.x0_in, "loop_x0_in"); ggml_set_input(cache.x0_in);
            cache.mask_in = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, L);
            ggml_set_name(cache.mask_in, "loop_mask"); ggml_set_input(cache.mask_in);
            cache.text_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, text_len, text_C);
            ggml_set_name(cache.text_in, "loop_text_in"); ggml_set_input(cache.text_in);
            cache.style_v_raw_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
            ggml_set_name(cache.style_v_raw_in, "loop_style_v"); ggml_set_input(cache.style_v_raw_in);
            cache.style_kctx_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
            ggml_set_name(cache.style_kctx_in, "loop_style_kctx"); ggml_set_input(cache.style_kctx_in);
            if (model.hparams.cfg_enabled()) {
                cache.text_in_u = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, text_len, text_C);
                ggml_set_name(cache.text_in_u, "loop_text_in_u"); ggml_set_input(cache.text_in_u);
                cache.style_v_raw_in_u = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
                ggml_set_name(cache.style_v_raw_in_u, "loop_style_v_u"); ggml_set_input(cache.style_v_raw_in_u);
                cache.style_kctx_in_u = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, kv_style, text_C);
                ggml_set_name(cache.style_kctx_in_u, "loop_style_kctx_u"); ggml_set_input(cache.style_kctx_in_u);
            }

            cache.pos_q = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_I32, L);
            ggml_set_name(cache.pos_q, "loop_pos_q"); ggml_set_input(cache.pos_q);
            cache.pos_k = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_I32, text_len);
            ggml_set_name(cache.pos_k, "loop_pos_k"); ggml_set_input(cache.pos_k);
            cache.freq_factors_q = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, D / 2);
            ggml_set_name(cache.freq_factors_q, "loop_ff_q"); ggml_set_input(cache.freq_factors_q);
            cache.freq_factors_k = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, D / 2);
            ggml_set_name(cache.freq_factors_k, "loop_ff_k"); ggml_set_input(cache.freq_factors_k);

            cache.t_emb_in.resize(total_steps, nullptr);
            for (int s = 0; s < total_steps; ++s) {
                cache.t_emb_in[s] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, 64);
                const std::string name_te = "loop_temb_" + std::to_string(s);
                ggml_set_name(cache.t_emb_in[s], name_te.c_str());
                ggml_set_input(cache.t_emb_in[s]);
            }

            // --- Chain N CFM steps together ---
            ggml_tensor * cur_latent = cache.x0_in;
            for (int s = 0; s < total_steps; ++s) {
                vector_step_inputs inputs;
                inputs.x_in           = cur_latent;       // previous step's output
                inputs.mask_in        = cache.mask_in;
                inputs.t_emb_in       = cache.t_emb_in[s];
                inputs.text_in        = cache.text_in;
                inputs.style_v_raw_in = cache.style_v_raw_in;
                inputs.style_kctx_in  = cache.style_kctx_in;
                inputs.noise_in       = cur_latent;       // CFM: next = noise_in + v/N
                inputs.pos_q          = cache.pos_q;
                inputs.pos_k          = cache.pos_k;
                inputs.freq_factors_q = cache.freq_factors_q;
                inputs.freq_factors_k = cache.freq_factors_k;

                ggml_tensor * next = append_supertonic_vector_step_subgraph(
                    cache.ctx, cache.gf, model, inputs, L, text_len, total_steps);
                if (model.hparams.cfg_enabled()) {
                    vector_step_inputs inputs_u = inputs;
                    inputs_u.text_in        = cache.text_in_u;
                    inputs_u.style_v_raw_in = cache.style_v_raw_in_u;
                    inputs_u.style_kctx_in  = cache.style_kctx_in_u;
                    ggml_tensor * next_u = append_supertonic_vector_step_subgraph(
                        cache.ctx, cache.gf, model, inputs_u, L, text_len, total_steps);
                    next = ggml_sub(cache.ctx,
                        ggml_scale(cache.ctx, next,   model.hparams.cfg_cond_scale),
                        ggml_scale(cache.ctx, next_u, model.hparams.cfg_uncond_scale));
                }
                const std::string step_name = "loop_next_" + std::to_string(s);
                ggml_set_name(next, step_name.c_str());
                cur_latent = next;
            }
            ggml_set_output(cur_latent);
            ggml_build_forward_expand(cache.gf, cur_latent);
            cache.final_latent_out = cur_latent;

            cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new vector loop one-graph failed");
            if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
                throw std::runtime_error("ggml_gallocr_reserve vector loop one-graph failed");
            }
            ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
        }

        // --- Per-call inputs (constants across CFM steps) ---
        ggml_backend_tensor_set(cache.x0_in, initial_noisy_latent, 0,
                                 (size_t) L * Cin * sizeof(float));
        ggml_backend_tensor_set(cache.mask_in, latent_mask, 0, (size_t) L * sizeof(float));
        ggml_backend_tensor_set(cache.text_in, text_emb, 0, (size_t) text_len * 256 * sizeof(float));

        const std::vector<float> * style_v_raw_ptr = nullptr;
        const std::vector<float> * kctx_raw_ptr = nullptr;
        cached_style_layouts(model, style_ttl, style_v_raw_ptr, kctx_raw_ptr);
        ggml_backend_tensor_set(cache.style_v_raw_in, style_v_raw_ptr->data(), 0,
                                 style_v_raw_ptr->size() * sizeof(float));
        ggml_backend_tensor_set(cache.style_kctx_in, kctx_raw_ptr->data(), 0,
                                 kctx_raw_ptr->size() * sizeof(float));

        if (model.hparams.cfg_enabled()) {
            f32_tensor tst = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.text_special_token");
            f32_tensor skt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_key_special_token");
            f32_tensor svt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_value_special_token");
            std::vector<float> text_u((size_t)256 * text_len);
            for (int c = 0; c < 256; ++c) for (int t = 0; t < text_len; ++t) text_u[(size_t)c*text_len+t] = tst.data[c];
            std::vector<float> sv_u((size_t)256 * 50), sk_u((size_t)256 * 50);
            for (int c = 0; c < 256; ++c) for (int t = 0; t < 50; ++t) {
                sv_u[(size_t)c*50+t] = svt.data[(size_t)t*256+c];
                sk_u[(size_t)c*50+t] = skt.data[(size_t)t*256+c];
            }
            ggml_backend_tensor_set(cache.text_in_u, text_u.data(), 0, text_u.size() * sizeof(float));
            ggml_backend_tensor_set(cache.style_v_raw_in_u, sv_u.data(), 0, sv_u.size() * sizeof(float));
            ggml_backend_tensor_set(cache.style_kctx_in_u, sk_u.data(), 0, sk_u.size() * sizeof(float));
        }

        {
            std::vector<int32_t> pos_q_host(L);
            for (int i = 0; i < L; ++i) pos_q_host[i] = i;
            ggml_backend_tensor_set(cache.pos_q, pos_q_host.data(), 0,
                                     pos_q_host.size() * sizeof(int32_t));
            std::vector<int32_t> pos_k_host(text_len);
            for (int i = 0; i < text_len; ++i) pos_k_host[i] = i;
            ggml_backend_tensor_set(cache.pos_k, pos_k_host.data(), 0,
                                     pos_k_host.size() * sizeof(int32_t));

            const int half = 32;
            f32_tensor theta = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            if ((int) theta.data.size() < half) {
                throw std::runtime_error("theta tensor has fewer than D/2 elements");
            }
            std::vector<float> ff_q(half), ff_k(half);
            for (int d = 0; d < half; ++d) {
                ff_q[d] = (float) L / theta.data[d];
                ff_k[d] = (float) text_len / theta.data[d];
            }
            ggml_backend_tensor_set(cache.freq_factors_q, ff_q.data(), 0,
                                     ff_q.size() * sizeof(float));
            ggml_backend_tensor_set(cache.freq_factors_k, ff_k.data(), 0,
                                     ff_k.size() * sizeof(float));
        }

        // --- Per-step time embeddings ---
        for (int s = 0; s < total_steps; ++s) {
            std::vector<float> te = time_embedding(model, s, total_steps);
            ggml_backend_tensor_set(cache.t_emb_in[s], te.data(), 0,
                                     te.size() * sizeof(float));
        }

        // --- ONE compute call for ALL CFM steps ---
        supertonic_graph_compute(model, cache.gf);

        // --- Read final output ---
        final_latent_out.assign((size_t) Cin * L, 0.0f);
        ggml_backend_tensor_get(cache.final_latent_out, final_latent_out.data(), 0,
                                 (size_t) Cin * L * sizeof(float));
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

// Public-ish driver: dispatches to the unrolled-loop path on non-CPU
// backends, falls back to the per-step `supertonic_vector_step_ggml`
// loop on CPU.  Gate the unrolled path off with
// SUPERTONIC_DISABLE_LOOP_GRAPH=1 to A/B against the per-step path on
// the same backend.
bool supertonic_vector_loop_ggml(const supertonic_model & model,
                                  const float * initial_noisy_latent,
                                  int latent_len,
                                  const float * text_emb,
                                  int text_len,
                                  const float * style_ttl,
                                  const float * latent_mask,
                                  int total_steps,
                                  std::vector<float> & final_latent_out,
                                  std::string * error) {
    const bool disable_loop =
        std::getenv("SUPERTONIC_DISABLE_LOOP_GRAPH") != nullptr;
    if (!disable_loop && !model_prefers_cpu_kernels(model)) {
        return supertonic_vector_loop_one_graph_ggml(
            model, initial_noisy_latent, latent_len, text_emb, text_len,
            style_ttl, latent_mask, total_steps, final_latent_out, error);
    }
    // CPU / disabled path: run the per-step loop in the addon's existing way.
    try {
        std::vector<float> latent((size_t) model.hparams.latent_channels * latent_len);
        std::memcpy(latent.data(), initial_noisy_latent, latent.size() * sizeof(float));
        std::vector<float> next;
        for (int step = 0; step < total_steps; ++step) {
            if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                              text_emb, text_len,
                                              style_ttl, latent_mask,
                                              step, total_steps, next, error)) {
                return false;
            }
            latent.swap(next);
        }
        final_latent_out = std::move(latent);
        if (error) error->clear();
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
    supertonic_op_dispatch_scope dispatch(model);
    // Metal / CUDA / Vulkan / OpenCL: use the consolidated one-graph path
    // (one ggml_backend_graph_compute call per CFM step instead of ~21).
    // CPU: keep the multi-cache trace_proj path — its CPU fast-paths and
    // thread_local sub-graph caches stay competitive on CPU and trace mode
    // relies on the per-stage outputs.  Set SUPERTONIC_DISABLE_ONE_GRAPH=1
    // to fall back to the multi-cache path on GPU backends if needed.
    const bool disable_one_graph = std::getenv("SUPERTONIC_DISABLE_ONE_GRAPH") != nullptr;
    if (!disable_one_graph && !model_prefers_cpu_kernels(model)) {
        return supertonic_vector_step_one_graph_ggml(model, noisy_latent, latent_len,
                                                      text_emb, text_len, style_ttl,
                                                      latent_mask, current_step,
                                                      total_steps, next_latent_out, error);
    }
    try {
        std::vector<supertonic_trace_tensor> scalar_trace;
        std::vector<supertonic_trace_tensor> ggml_trace;
        const int L = latent_len;
        const int C = model.hparams.latent_channels;

        // Conditional pass (text = real embeddings, style derived from
        // `style_ttl` + `/Expand_output_0`).
        std::vector<float> next_tc;
        if (!supertonic_vector_trace_proj_ggml(model, noisy_latent, text_emb, text_len,
                                               style_ttl, latent_mask, latent_len,
                                               current_step, total_steps,
                                               scalar_trace, ggml_trace, error,
                                               false, false, &next_tc)) {
            return false;
        }
        if (next_tc.size() != (size_t)L*C) throw std::runtime_error("bad ve_next_latent_tc size");

        if (!model.hparams.cfg_enabled()) {
            next_latent_out.assign((size_t)C*L, 0.0f);
            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < L; ++t) {
                    next_latent_out[(size_t)c*L + t] = next_tc[(size_t)t*C + c];
                }
            }
            if (error) error->clear();
            return true;
        }

        // Classifier-Free Guidance (Supertonic 3) — second, unconditional
        // pass using the learned null text / style tokens, then combine.
        // `next = noise + velocity/N` is affine in the velocity, and the
        // exported guidance uses cond_scale - uncond_scale == 1, so combining
        // the two integrated latents with the same coefficients reproduces
        // `velocity = cs*v_cond - us*v_uncond` exactly (the residual
        // `(1 - cs + us)*noise` term below covers any non-unit gap).
        f32_tensor tst = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.text_special_token");   // [256]
        f32_tensor skt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_key_special_token");   // [50,256] (t,c)
        f32_tensor svt = read_f32(model, "vector_estimator:tts.ttl.uncond_masker.style_value_special_token"); // [50,256] (t,c)
        std::vector<float> text_u((size_t)256 * text_len);
        for (int c = 0; c < 256; ++c) for (int t = 0; t < text_len; ++t) text_u[(size_t)c*text_len+t] = tst.data[c];
        std::vector<float> sv_u((size_t)256 * 50), sk_u((size_t)256 * 50);
        for (int c = 0; c < 256; ++c) for (int t = 0; t < 50; ++t) {
            sv_u[(size_t)c*50+t] = svt.data[(size_t)t*256+c];
            sk_u[(size_t)c*50+t] = skt.data[(size_t)t*256+c];
        }
        std::vector<float> next_tc_u;
        if (!supertonic_vector_trace_proj_ggml(model, noisy_latent, text_u.data(), text_len,
                                               /*style_ttl=*/nullptr, latent_mask, latent_len,
                                               current_step, total_steps,
                                               scalar_trace, ggml_trace, error,
                                               false, false, &next_tc_u,
                                               &sv_u, &sk_u)) {
            return false;
        }
        if (next_tc_u.size() != (size_t)L*C) throw std::runtime_error("bad ve_next_latent_tc (uncond) size");

        const float cs = model.hparams.cfg_cond_scale, us = model.hparams.cfg_uncond_scale;
        const float kn = 1.0f - cs + us; // == 0 when cond_scale - uncond_scale == 1
        next_latent_out.assign((size_t)C*L, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < L; ++t) {
                float combined = cs * next_tc[(size_t)t*C + c] - us * next_tc_u[(size_t)t*C + c];
                if (kn != 0.0f) combined += kn * noisy_latent[(size_t)c*L + t];
                next_latent_out[(size_t)c*L + t] = combined;
            }
        }
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

void release_vector_estimator_thread_local_caches() {
    // Walk every registered release thunk on the calling thread.  DO NOT
    // `clear()` the registry afterwards — each `thread_local tl_register_once`
    // sentinel only constructs once per thread (on first reach of its host
    // function), so a clear here would leave the registry empty for every
    // subsequent engine cycle, re-leaking the gallocrs.  The thunks
    // themselves are cheap (one `std::function` slot each, ~64 bytes) and
    // safe to re-invoke: each `free_*_cache` is idempotent (cache = {}
    // after the first call, and the gallocr/ctx are nullptr-guarded), so
    // a second invocation on the same registry entry from a later engine
    // cycle's destructor finds an already-empty cache and no-ops.
    for (auto & fn : g_tl_release_thunks) {
        if (fn) fn();
    }
}

} // namespace tts_cpp::supertonic::detail
