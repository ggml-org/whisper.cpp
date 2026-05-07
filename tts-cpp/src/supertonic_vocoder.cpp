#include "supertonic_internal.h"

#include "ggml-alloc.h"

#if defined(TTS_CPP_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(TTS_CPP_USE_CBLAS)
#include <cblas.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tts_cpp::supertonic::detail {
namespace {

struct f32_tensor {
    std::vector<float> data;
    int64_t ne[4] = {1, 1, 1, 1}; // ggml order; ONNX row-major is reversed
};

f32_tensor read_f32_tensor(ggml_tensor * t) {
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

f32_tensor read_f32(const supertonic_model & m, const std::string & source_name) {
    return read_f32_tensor(require_source_tensor(m, source_name));
}

float scalar_f32_tensor(ggml_tensor * tensor) {
    f32_tensor t = read_f32_tensor(tensor);
    if (t.data.empty()) throw std::runtime_error("empty scalar tensor");
    return t.data[0];
}

float scalar_f32(const supertonic_model & m, const std::string & source_name) {
    return scalar_f32_tensor(require_source_tensor(m, source_name));
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f));
}

bool vocoder_profile_enabled() {
    static const bool enabled = std::getenv("SUPERTONIC_VOCODER_PROFILE") != nullptr;
    return enabled;
}

void profile_vocoder_checkpoint(const char * label,
                                std::chrono::steady_clock::time_point & last) {
    if (!vocoder_profile_enabled()) return;
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
    std::fprintf(stderr, "supertonic_vocoder_profile island=%s ms=%.3f\n", label, ms);
}

ggml_tensor * repeat_like(ggml_context * ctx, ggml_tensor * v, ggml_tensor * like) {
    if (ggml_n_dims(v) == 1 && ggml_n_dims(like) >= 2) {
        if (like->ne[0] == v->ne[0]) v = ggml_reshape_2d(ctx, v, v->ne[0], 1);
        else if (like->ne[1] == v->ne[0]) v = ggml_reshape_2d(ctx, v, 1, v->ne[0]);
    } else if (v->ne[0] == 1 && v->ne[1] > 1 && v->ne[2] == 1) {
        if (like->ne[0] == v->ne[1]) v = ggml_reshape_2d(ctx, v, v->ne[1], 1);
        else if (like->ne[1] == v->ne[1]) v = ggml_reshape_2d(ctx, v, 1, v->ne[1]);
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

ggml_tensor * causal_replicate_pad_1d(ggml_context * ctx, ggml_tensor * x, int pad_left) {
    if (pad_left <= 0) return x;
    const int64_t C = x->ne[1];
    ggml_tensor * first = ggml_view_2d(ctx, x, 1, C, x->nb[1], 0);
    ggml_tensor * rep = ggml_repeat_4d(ctx, first, pad_left, C, 1, 1);
    return ggml_concat(ctx, rep, x, 0);
}

ggml_tensor * conv1d_causal_ggml(ggml_context * ctx,
                                 ggml_tensor * x,
                                 ggml_tensor * w,
                                 ggml_tensor * b,
                                 int dilation = 1) {
    const int K = (int) w->ne[0];
#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
    if (K == 1 && dilation == 1 &&
        x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
        (!b || b->type == GGML_TYPE_F32) &&
        x->ne[2] == 1 && x->ne[3] == 1) {
        auto pointwise_op = [](ggml_tensor * dst, int ith, int, void *) {
            if (ith != 0) return;
            const ggml_tensor * src = dst->src[0];
            const ggml_tensor * weight = dst->src[1];
            const ggml_tensor * bias = dst->src[2];
            const int L = (int)src->ne[0];
            const int IC = (int)src->ne[1];
            const int OC = (int)weight->ne[2];
            const float * src_data = static_cast<const float *>(src->data);
            const float * weight_data = static_cast<const float *>(weight->data);
            float * dst_data = static_cast<float *>(dst->data);
            const int lda = (int)(src->nb[1] / sizeof(float));
            const int ldb = (int)(weight->nb[2] / sizeof(float));
            const int ldc = (int)(dst->nb[1] / sizeof(float));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        L, OC, IC,
                        1.0f,
                        src_data, lda,
                        weight_data, ldb,
                        0.0f,
                        dst_data, ldc);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
            if (bias) {
                const auto * bias_base = static_cast<const uint8_t *>(bias->data);
                for (int oc = 0; oc < OC; ++oc) {
                    const float bv = *reinterpret_cast<const float *>(bias_base + (size_t)oc * bias->nb[0]);
                    float * col = dst_data + (size_t)oc * ldc;
                    for (int t = 0; t < L; ++t) col[t] += bv;
                }
            }
        };
        ggml_tensor * args_with_bias[] = { x, w, b };
        ggml_tensor * args_no_bias[] = { x, w };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], w->ne[2], x->ne[2], x->ne[3],
                              b ? args_with_bias : args_no_bias,
                              b ? 3 : 2,
                              pointwise_op,
                              1,
                              nullptr);
    }
    if (K > 1 && dilation == 1 &&
        x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
        (!b || b->type == GGML_TYPE_F32) &&
        x->ne[2] == 1 && x->ne[3] == 1) {
        auto conv_op = [](ggml_tensor * dst, int ith, int, void *) {
            if (ith != 0) return;
            const ggml_tensor * src = dst->src[0];
            const ggml_tensor * weight = dst->src[1];
            const ggml_tensor * bias = dst->src[2];
            const int L = (int)src->ne[0];
            const int IC = (int)src->ne[1];
            const int K = (int)weight->ne[0];
            const int OC = (int)weight->ne[2];
            const int KC = K * IC;
            const int pad_left = K - 1;
            const auto * src_base = static_cast<const uint8_t *>(src->data);
            const auto * weight_data = static_cast<const float *>(weight->data);
            auto * dst_data = static_cast<float *>(dst->data);
            const int ldb = (int)(weight->nb[2] / sizeof(float));
            const int ldc = (int)(dst->nb[1] / sizeof(float));

            std::vector<float> cols((size_t)L * KC);
            for (int ic = 0; ic < IC; ++ic) {
                for (int k = 0; k < K; ++k) {
                    const int col = ic * K + k;
                    float * col_ptr = cols.data() + (size_t)col * L;
                    for (int t = 0; t < L; ++t) {
                        int st = t + k - pad_left;
                        if (st < 0) st = 0;
                        col_ptr[t] = *reinterpret_cast<const float *>(
                            src_base + (size_t)st * src->nb[0] + (size_t)ic * src->nb[1]);
                    }
                }
            }

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        L, OC, KC,
                        1.0f,
                        cols.data(), L,
                        weight_data, ldb,
                        0.0f,
                        dst_data, ldc);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
            if (bias) {
                const auto * bias_base = static_cast<const uint8_t *>(bias->data);
                for (int oc = 0; oc < OC; ++oc) {
                    const float bv = *reinterpret_cast<const float *>(bias_base + (size_t)oc * bias->nb[0]);
                    float * col = dst_data + (size_t)oc * ldc;
                    for (int t = 0; t < L; ++t) col[t] += bv;
                }
            }
        };
        ggml_tensor * args_with_bias[] = { x, w, b };
        ggml_tensor * args_no_bias[] = { x, w };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], w->ne[2], x->ne[2], x->ne[3],
                              b ? args_with_bias : args_no_bias,
                              b ? 3 : 2,
                              conv_op,
                              1,
                              nullptr);
    }
#endif
    ggml_tensor * padded = causal_replicate_pad_1d(ctx, x, (K - 1) * dilation);
    ggml_tensor * im2col = ggml_im2col(ctx, w, padded, 1, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
    y = ggml_reshape_3d(ctx, y, im2col->ne[1], w->ne[2], im2col->ne[2]);
    if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
    return y;
}

struct depthwise_causal_op_config {
    int dilation = 1;
};

const depthwise_causal_op_config * depthwise_causal_config(int dilation) {
    static const depthwise_causal_op_config d1{1};
    static const depthwise_causal_op_config d2{2};
    static const depthwise_causal_op_config d4{4};
    switch (dilation) {
        case 1: return &d1;
        case 2: return &d2;
        case 4: return &d4;
        default: return nullptr;
    }
}

void depthwise_causal_custom_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * cfg = static_cast<const depthwise_causal_op_config *>(userdata);
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const ggml_tensor * b = dst->src[2];
    const int L = (int)x->ne[0];
    const int C = (int)x->ne[1];
    const int K = (int)w->ne[0];
    const int dilation = cfg ? cfg->dilation : 1;
    const int pad_left = (K - 1) * dilation;
    const int c0 = (C * ith) / nth;
    const int c1 = (C * (ith + 1)) / nth;

    const auto * x_base = static_cast<const uint8_t *>(x->data);
    const auto * w_base = static_cast<const uint8_t *>(w->data);
    const auto * b_base = static_cast<const uint8_t *>(b->data);
    auto * dst_base = static_cast<uint8_t *>(dst->data);

    for (int c = c0; c < c1; ++c) {
        const float bias = *reinterpret_cast<const float *>(b_base + (size_t)c * b->nb[0]);
        for (int t = 0; t < L; ++t) {
            float sum = bias;
            for (int k = 0; k < K; ++k) {
                int st = t + k * dilation - pad_left;
                if (st < 0) st = 0;
                const float xv = *reinterpret_cast<const float *>(x_base + (size_t)st * x->nb[0] + (size_t)c * x->nb[1]);
                const float wv = *reinterpret_cast<const float *>(w_base + (size_t)k * w->nb[0] + (size_t)c * w->nb[2]);
                sum += xv * wv;
            }
            *reinterpret_cast<float *>(dst_base + (size_t)t * dst->nb[0] + (size_t)c * dst->nb[1]) = sum;
        }
    }
}

ggml_tensor * depthwise_causal_custom_ggml(ggml_context * ctx,
                                           ggml_tensor * x,
                                           ggml_tensor * w,
                                           ggml_tensor * b,
                                           int dilation) {
    const depthwise_causal_op_config * cfg = depthwise_causal_config(dilation);
    if (!cfg || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return nullptr;
    }
    ggml_tensor * args[] = { x, w, b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32,
                          x->ne[0], x->ne[1], x->ne[2], x->ne[3],
                          args, 3,
                          depthwise_causal_custom_op,
                          GGML_N_TASKS_MAX,
                          const_cast<depthwise_causal_op_config *>(cfg));
}

ggml_tensor * depthwise_conv1d_causal_ggml(ggml_context * ctx,
                                           ggml_tensor * x,
                                           ggml_tensor * w,
                                           ggml_tensor * b,
                                           int dilation) {
    if (ggml_tensor * custom = depthwise_causal_custom_ggml(ctx, x, w, b, dilation)) {
        return custom;
    }
    const int K = (int) w->ne[0];
    ggml_tensor * padded = causal_replicate_pad_1d(ctx, x, (K - 1) * dilation);
    ggml_tensor * new_b = ggml_reshape_4d(ctx, padded, padded->ne[0], 1, padded->ne[1], padded->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, w, new_b, 1, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, im2col, w);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);
    return ggml_add(ctx, y, repeat_like(ctx, b, y));
}

ggml_tensor * layer_norm_channel_ggml(ggml_context * ctx,
                                      ggml_tensor * x,
                                      ggml_tensor * gamma,
                                      ggml_tensor * beta,
                                      float eps = 1e-6f) {
    ggml_tensor * y = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    y = ggml_norm(ctx, y, eps);
    y = ggml_mul(ctx, y, repeat_like(ctx, gamma, y));
    y = ggml_add(ctx, y, repeat_like(ctx, beta, y));
    return ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
}

ggml_tensor * convnext_block_ggml(ggml_context * ctx,
                                  const supertonic_vocoder_convnext_weights & w,
                                  ggml_tensor * x,
                                  int idx) {
    static const int dilations[10] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
    ggml_tensor * residual = x;
    ggml_tensor * y = depthwise_conv1d_causal_ggml(ctx, x, w.dw_w, w.dw_b, dilations[idx]);
    y = layer_norm_channel_ggml(ctx, y, w.norm_g, w.norm_b);
    y = conv1d_causal_ggml(ctx, y, w.pw1_w, w.pw1_b);
    y = ggml_gelu_erf(ctx, y);
    y = conv1d_causal_ggml(ctx, y, w.pw2_w, w.pw2_b);
    y = ggml_mul(ctx, y, repeat_like(ctx, w.gamma, y));
    return ggml_add(ctx, residual, y);
}

struct vocoder_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int latent_len = 0;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * bn_scale = nullptr;
    ggml_tensor * bn_shift = nullptr;
    ggml_tensor * wav = nullptr;
};

// Guards ggml_gallocr_free against a backend that has already been torn
// down (e.g. host destroyed engine_a then immediately invoked synthesize
// on engine_b on the same thread; the cache miss-key triggers this free
// path against the dangling allocr).  See supertonic_internal.h for the
// full alive-registry rationale.  Skipping the gallocr_free leaks the
// gallocr bookkeeping (~80 bytes) but the underlying GPU buffers were
// already released when the model's backend was freed.
void free_vocoder_cache(vocoder_graph_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

void build_supertonic_vocoder_cache(vocoder_graph_cache & cache,
                                    const supertonic_model & model,
                                    int latent_len) {
    free_vocoder_cache(cache);
    cache.model = &model;
    cache.generation_id = model.generation_id;
    cache.latent_len = latent_len;
    const int C_latent = model.hparams.latent_dim;
    const int T0 = latent_len * model.hparams.ttl_chunk_compress_factor;
    constexpr int MAX_NODES = 4096;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                            ggml_graph_overhead_custom(MAX_NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params p = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(p);
    cache.gf = ggml_new_graph_custom(cache.ctx, MAX_NODES, false);

    ggml_tensor * x = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, T0, C_latent);
    cache.x_in = x;
    ggml_set_name(cache.x_in, "vocoder_in");
    ggml_set_input(cache.x_in);

    cache.bn_scale = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, 512);
    ggml_set_name(cache.bn_scale, "vocoder_bn_scale");
    ggml_set_input(cache.bn_scale);
    cache.bn_shift = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, 512);
    ggml_set_name(cache.bn_shift, "vocoder_bn_shift");
    ggml_set_input(cache.bn_shift);

    const float normalizer_scale = scalar_f32_tensor(model.vocoder.normalizer_scale);
    x = ggml_scale(cache.ctx, x, 1.0f / normalizer_scale);
    x = ggml_mul(cache.ctx, x, repeat_like(cache.ctx, model.vocoder.latent_std, x));
    x = ggml_add(cache.ctx, x, repeat_like(cache.ctx, model.vocoder.latent_mean, x));
    ggml_set_name(x, "vocoder_denorm");

    x = conv1d_causal_ggml(cache.ctx, x, model.vocoder.embed_w, model.vocoder.embed_b);
    ggml_set_name(x, "vocoder_embed");
    for (int i = 0; i < 10; ++i) {
        x = convnext_block_ggml(cache.ctx, model.vocoder.convnext[(size_t) i], x, i);
        ggml_set_name(x, ("vocoder_convnext_" + std::to_string(i)).c_str());
    }

    x = ggml_mul(cache.ctx, x, repeat_like(cache.ctx, cache.bn_scale, x));
    x = ggml_add(cache.ctx, x, repeat_like(cache.ctx, cache.bn_shift, x));
    ggml_set_name(x, "vocoder_final_norm");

    x = conv1d_causal_ggml(cache.ctx, x, model.vocoder.head1_w, model.vocoder.head1_b);
    ggml_set_name(x, "vocoder_head1");
    const float prelu = scalar_f32_tensor(model.vocoder.head_prelu);
    x = ggml_leaky_relu(cache.ctx, x, prelu, false);
    ggml_set_name(x, "vocoder_prelu");
    x = conv1d_causal_ggml(cache.ctx, x, model.vocoder.head2_w, nullptr);
    ggml_set_name(x, "wav");
    ggml_set_output(x);
    ggml_build_forward_expand(cache.gf, x);
    cache.wav = x;

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new vocoder cache failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve vocoder cache failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

void linear1x1(const std::vector<float> & x, int L, int IC,
               const f32_tensor & w, const f32_tensor * b,
               int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    // ONNX Conv weight is row-major [OC, IC, 1]; raw index ((oc*IC + ic)*1).
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            const size_t woff = (size_t) oc * IC;
            for (int ic = 0; ic < IC; ++ic) {
                sum += w.data[woff + ic] * x[(size_t) t * IC + ic];
            }
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void conv1d_causal(const std::vector<float> & x, int L, int IC,
                   const f32_tensor & w, const f32_tensor * b,
                   int K, int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    const int pad_left = K - 1;
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            for (int ic = 0; ic < IC; ++ic) {
                const size_t wbase = ((size_t) oc * IC + ic) * K;
                for (int k = 0; k < K; ++k) {
                    int src_t = t + k - pad_left;
                    if (src_t < 0) src_t = 0; // replicate pad
                    sum += w.data[wbase + k] * x[(size_t) src_t * IC + ic];
                }
            }
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void depthwise_conv1d_causal(const std::vector<float> & x, int L, int C,
                             const f32_tensor & w, const f32_tensor & b,
                             int K, int dilation, std::vector<float> & y) {
    y.assign((size_t) L * C, 0.0f);
    const int pad_left = (K - 1) * dilation;
    // ONNX depthwise Conv weight is [C, 1, K].
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b.data[c];
            const size_t wbase = (size_t) c * K;
            for (int k = 0; k < K; ++k) {
                int src_t = t + k * dilation - pad_left;
                if (src_t < 0) src_t = 0;
                sum += w.data[wbase + k] * x[(size_t) src_t * C + c];
            }
            y[(size_t) t * C + c] = sum;
        }
    }
}

void layer_norm_channel(std::vector<float> & x, int L, int C,
                        const f32_tensor & gamma, const f32_tensor & beta,
                        float eps = 1e-6f) {
    for (int t = 0; t < L; ++t) {
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) mean += x[(size_t) t * C + c];
        mean /= (float) C;
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float d = x[(size_t) t * C + c] - mean;
            var += d * d;
        }
        float inv = 1.0f / std::sqrt(var / (float) C + eps);
        for (int c = 0; c < C; ++c) {
            float v = (x[(size_t) t * C + c] - mean) * inv;
            x[(size_t) t * C + c] = v * gamma.data[c] + beta.data[c];
        }
    }
}

void batch_norm_channel(std::vector<float> & x, int L, int C,
                        const f32_tensor & gamma, const f32_tensor & beta,
                        const f32_tensor & running_mean, const f32_tensor & running_var,
                        float eps = 1e-5f) {
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float v = (x[(size_t) t * C + c] - running_mean.data[c]) /
                      std::sqrt(running_var.data[c] + eps);
            x[(size_t) t * C + c] = v * gamma.data[c] + beta.data[c];
        }
    }
}

void push_trace(std::vector<supertonic_trace_tensor> & trace,
                const std::string & name,
                int L,
                int C,
                const std::vector<float> & data) {
    trace.push_back({name, {L, C}, data});
}

std::vector<float> unpack_latent_scalar(const supertonic_model & model,
                                        const float * latent,
                                        int latent_len) {
    const int C_latent = model.hparams.latent_dim;
    const int factor = model.hparams.ttl_chunk_compress_factor;
    const int T0 = latent_len * factor;
    std::vector<float> x((size_t) T0 * C_latent);
    for (int c = 0; c < C_latent; ++c) {
        for (int t = 0; t < latent_len; ++t) {
            for (int r = 0; r < factor; ++r) {
                int src_c = c * factor + r;
                x[(size_t) (t * factor + r) * C_latent + c] =
                    latent[(size_t) src_c * latent_len + t];
            }
        }
    }
    return x;
}

std::vector<float> unpack_latent_ggml_layout(const supertonic_model & model,
                                             const float * latent,
                                             int latent_len) {
    const int C_latent = model.hparams.latent_dim;
    const int factor = model.hparams.ttl_chunk_compress_factor;
    const int T0 = latent_len * factor;
    std::vector<float> x((size_t) T0 * C_latent);
    for (int c = 0; c < C_latent; ++c) {
        for (int t = 0; t < latent_len; ++t) {
            for (int r = 0; r < factor; ++r) {
                int src_c = c * factor + r;
                x[(size_t) c * T0 + (t * factor + r)] =
                    latent[(size_t) src_c * latent_len + t];
            }
        }
    }
    return x;
}

std::vector<float> ggml_tensor_to_time_channel(ggml_tensor * t) {
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

void convnext_block(const supertonic_model & m, int idx,
                    std::vector<float> & x, int L, int C) {
    const auto & cw = m.vocoder.convnext[(size_t) idx];
    f32_tensor dw_w = read_f32_tensor(cw.dw_w);
    f32_tensor dw_b = read_f32_tensor(cw.dw_b);
    f32_tensor ln_g = read_f32_tensor(cw.norm_g);
    f32_tensor ln_b = read_f32_tensor(cw.norm_b);
    f32_tensor pw1_w = read_f32_tensor(cw.pw1_w);
    f32_tensor pw1_b = read_f32_tensor(cw.pw1_b);
    f32_tensor pw2_w = read_f32_tensor(cw.pw2_w);
    f32_tensor pw2_b = read_f32_tensor(cw.pw2_b);
    f32_tensor gamma = read_f32_tensor(cw.gamma);

    std::vector<float> residual = x;
    std::vector<float> y;
    const int K = (int) dw_w.ne[0];
    static const int dilations[10] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
    depthwise_conv1d_causal(x, L, C, dw_w, dw_b, K, dilations[idx], y);
    layer_norm_channel(y, L, C, ln_g, ln_b);

    std::vector<float> z;
    const int hidden = (int) pw1_w.ne[2];
    linear1x1(y, L, C, pw1_w, &pw1_b, hidden, z);
    for (float & v : z) v = gelu(v);
    linear1x1(z, L, hidden, pw2_w, &pw2_b, C, y);

    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            x[(size_t) t * C + c] = residual[(size_t) t * C + c] +
                                    gamma.data[c] * y[(size_t) t * C + c];
        }
    }
}

} // namespace

bool supertonic_vocoder_forward_cpu(const supertonic_model & model,
                                    const float * latent,
                                    int latent_len,
                                    std::vector<float> & wav_out,
                                    std::string * error) {
    try {
        const int C_latent = model.hparams.latent_dim;       // 24
        const int factor = model.hparams.ttl_chunk_compress_factor; // 6
        const int latent_channels = model.hparams.latent_channels;  // 144
        if (latent_len <= 0) throw std::runtime_error("latent_len must be positive");

        // Input latent is NumPy/PyTorch row-major [1, 144, L].  Vocoder unpacks
        // it as [1, 24, 6, L] -> [1, 24, L, 6] -> [1, 24, L*6].
        const int T0 = latent_len * factor;
        std::vector<float> x((size_t) T0 * C_latent);
        for (int c = 0; c < C_latent; ++c) {
            for (int t = 0; t < latent_len; ++t) {
                for (int r = 0; r < factor; ++r) {
                    int src_c = c * factor + r;
                    x[(size_t) (t * factor + r) * C_latent + c] =
                        latent[(size_t) src_c * latent_len + t];
                }
            }
        }

        float normalizer_scale = scalar_f32_tensor(model.vocoder.normalizer_scale);
        f32_tensor mean = read_f32_tensor(model.vocoder.latent_mean);
        f32_tensor std = read_f32_tensor(model.vocoder.latent_std);
        for (int t = 0; t < T0; ++t) {
            for (int c = 0; c < C_latent; ++c) {
                float v = x[(size_t) t * C_latent + c] / normalizer_scale;
                x[(size_t) t * C_latent + c] = v * std.data[c] + mean.data[c];
            }
        }

        f32_tensor embed_w = read_f32_tensor(model.vocoder.embed_w);
        f32_tensor embed_b = read_f32_tensor(model.vocoder.embed_b);
        std::vector<float> y;
        conv1d_causal(x, T0, C_latent, embed_w, &embed_b,
                      (int) embed_w.ne[0], (int) embed_w.ne[2], y);
        x.swap(y);
        const int C = (int) embed_w.ne[2]; // 512

        for (int i = 0; i < 10; ++i) {
            convnext_block(model, i, x, T0, C);
        }

        batch_norm_channel(
            x, T0, C,
            read_f32_tensor(model.vocoder.final_norm_g),
            read_f32_tensor(model.vocoder.final_norm_b),
            read_f32_tensor(model.vocoder.final_norm_running_mean),
            read_f32_tensor(model.vocoder.final_norm_running_var));

        f32_tensor h1_w = read_f32_tensor(model.vocoder.head1_w);
        f32_tensor h1_b = read_f32_tensor(model.vocoder.head1_b);
        conv1d_causal(x, T0, C, h1_w, &h1_b, (int) h1_w.ne[0], (int) h1_w.ne[2], y);
        float prelu = scalar_f32_tensor(model.vocoder.head_prelu);
        for (float & v : y) {
            if (v < 0.0f) v *= prelu;
        }

        f32_tensor h2_w = read_f32_tensor(model.vocoder.head2_w);
        std::vector<float> z;
        linear1x1(y, T0, (int) h1_w.ne[2], h2_w, nullptr, (int) h2_w.ne[2], z);

        wav_out = std::move(z);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_vocoder_forward_ggml(const supertonic_model & model,
                                     const float * latent,
                                     int latent_len,
                                     std::vector<float> & wav_out,
                                     std::string * error) {
    try {
        auto profile_last = std::chrono::steady_clock::now();
        const int C_latent = model.hparams.latent_dim;
        const int factor = model.hparams.ttl_chunk_compress_factor;
        const int T0 = latent_len * factor;
        if (latent_len <= 0) throw std::runtime_error("latent_len must be positive");

        std::vector<float> x_in((size_t) T0 * C_latent);
        for (int c = 0; c < C_latent; ++c) {
            for (int t = 0; t < latent_len; ++t) {
                for (int r = 0; r < factor; ++r) {
                    int src_c = c * factor + r;
                    x_in[(size_t) c * T0 + (t * factor + r)] =
                        latent[(size_t) src_c * latent_len + t];
                }
            }
        }
        profile_vocoder_checkpoint("unpack", profile_last);

        f32_tensor gamma = read_f32_tensor(model.vocoder.final_norm_g);
        f32_tensor beta = read_f32_tensor(model.vocoder.final_norm_b);
        f32_tensor mean = read_f32_tensor(model.vocoder.final_norm_running_mean);
        f32_tensor var = read_f32_tensor(model.vocoder.final_norm_running_var);
        std::vector<float> bn_scale(512), bn_shift(512);
        for (int c = 0; c < 512; ++c) {
            bn_scale[c] = gamma.data[c] / std::sqrt(var.data[c] + 1e-5f);
            bn_shift[c] = beta.data[c] - mean.data[c] * bn_scale[c];
        }
        profile_vocoder_checkpoint("bn_params", profile_last);

        thread_local vocoder_graph_cache cache;
        if (cache.model != &model || cache.generation_id != model.generation_id ||
            cache.latent_len != latent_len) {
            build_supertonic_vocoder_cache(cache, model, latent_len);
        }
        profile_vocoder_checkpoint("graph_cache", profile_last);

        ggml_backend_tensor_set(cache.x_in, x_in.data(), 0, x_in.size() * sizeof(float));
        ggml_backend_tensor_set(cache.bn_scale, bn_scale.data(), 0, bn_scale.size() * sizeof(float));
        ggml_backend_tensor_set(cache.bn_shift, bn_shift.data(), 0, bn_shift.size() * sizeof(float));
        profile_vocoder_checkpoint("set_inputs", profile_last);

        supertonic_graph_compute(model, cache.gf);
        profile_vocoder_checkpoint("compute", profile_last);
        wav_out = ggml_tensor_to_time_channel(cache.wav);
        profile_vocoder_checkpoint("readback", profile_last);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_vocoder_trace_scalar(const supertonic_model & model,
                                     const float * latent,
                                     int latent_len,
                                     std::vector<supertonic_trace_tensor> & trace_out,
                                     std::string * error) {
    try {
        trace_out.clear();
        const int C_latent = model.hparams.latent_dim;
        const int factor = model.hparams.ttl_chunk_compress_factor;
        const int T0 = latent_len * factor;
        std::vector<float> x = unpack_latent_scalar(model, latent, latent_len);
        push_trace(trace_out, "unpack", T0, C_latent, x);

        float normalizer_scale = scalar_f32_tensor(model.vocoder.normalizer_scale);
        f32_tensor mean = read_f32_tensor(model.vocoder.latent_mean);
        f32_tensor std = read_f32_tensor(model.vocoder.latent_std);
        for (int t = 0; t < T0; ++t) {
            for (int c = 0; c < C_latent; ++c) {
                float v = x[(size_t) t * C_latent + c] / normalizer_scale;
                x[(size_t) t * C_latent + c] = v * std.data[c] + mean.data[c];
            }
        }
        push_trace(trace_out, "denorm", T0, C_latent, x);

        f32_tensor embed_w = read_f32_tensor(model.vocoder.embed_w);
        f32_tensor embed_b = read_f32_tensor(model.vocoder.embed_b);
        std::vector<float> y;
        conv1d_causal(x, T0, C_latent, embed_w, &embed_b,
                      (int) embed_w.ne[0], (int) embed_w.ne[2], y);
        push_trace(trace_out, "embed", T0, (int) embed_w.ne[2], y);

        const int C = (int) embed_w.ne[2];
        const auto & cw = model.vocoder.convnext[0];
        f32_tensor dw_w = read_f32_tensor(cw.dw_w);
        f32_tensor dw_b = read_f32_tensor(cw.dw_b);
        f32_tensor ln_g = read_f32_tensor(cw.norm_g);
        f32_tensor ln_b = read_f32_tensor(cw.norm_b);
        f32_tensor pw1_w = read_f32_tensor(cw.pw1_w);
        f32_tensor pw1_b = read_f32_tensor(cw.pw1_b);
        f32_tensor pw2_w = read_f32_tensor(cw.pw2_w);
        f32_tensor pw2_b = read_f32_tensor(cw.pw2_b);
        f32_tensor gamma = read_f32_tensor(cw.gamma);
        std::vector<float> residual = y;
        std::vector<float> z;
        depthwise_conv1d_causal(y, T0, C, dw_w, dw_b, (int) dw_w.ne[0], 1, z);
        push_trace(trace_out, "block0_dw", T0, C, z);
        layer_norm_channel(z, T0, C, ln_g, ln_b);
        push_trace(trace_out, "block0_norm", T0, C, z);
        std::vector<float> h;
        linear1x1(z, T0, C, pw1_w, &pw1_b, (int) pw1_w.ne[2], h);
        push_trace(trace_out, "block0_pw1", T0, (int) pw1_w.ne[2], h);
        for (float & v : h) v = gelu(v);
        push_trace(trace_out, "block0_gelu", T0, (int) pw1_w.ne[2], h);
        linear1x1(h, T0, (int) pw1_w.ne[2], pw2_w, &pw2_b, C, z);
        push_trace(trace_out, "block0_pw2", T0, C, z);
        for (size_t i = 0; i < z.size(); ++i) {
            int c = (int)(i % (size_t) C);
            z[i] = residual[i] + gamma.data[c] * z[i];
        }
        push_trace(trace_out, "block0_out", T0, C, z);

        for (int i = 1; i < 10; ++i) {
            convnext_block(model, i, z, T0, C);
            push_trace(trace_out, "block" + std::to_string(i) + "_out", T0, C, z);
        }

        batch_norm_channel(
            z, T0, C,
            read_f32_tensor(model.vocoder.final_norm_g),
            read_f32_tensor(model.vocoder.final_norm_b),
            read_f32_tensor(model.vocoder.final_norm_running_mean),
            read_f32_tensor(model.vocoder.final_norm_running_var));
        push_trace(trace_out, "final_norm", T0, C, z);

        f32_tensor h1_w = read_f32_tensor(model.vocoder.head1_w);
        f32_tensor h1_b = read_f32_tensor(model.vocoder.head1_b);
        conv1d_causal(z, T0, C, h1_w, &h1_b, (int) h1_w.ne[0], (int) h1_w.ne[2], h);
        push_trace(trace_out, "head1", T0, (int) h1_w.ne[2], h);
        float prelu = scalar_f32_tensor(model.vocoder.head_prelu);
        for (float & v : h) if (v < 0.0f) v *= prelu;
        push_trace(trace_out, "prelu", T0, (int) h1_w.ne[2], h);
        f32_tensor h2_w = read_f32_tensor(model.vocoder.head2_w);
        linear1x1(h, T0, (int) h1_w.ne[2], h2_w, nullptr, (int) h2_w.ne[2], z);
        push_trace(trace_out, "wav", T0, (int) h2_w.ne[2], z);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_vocoder_trace_ggml(const supertonic_model & model,
                                   const float * latent,
                                   int latent_len,
                                   std::vector<supertonic_trace_tensor> & trace_out,
                                   std::string * error) {
    try {
        trace_out.clear();
        const int C_latent = model.hparams.latent_dim;
        const int factor = model.hparams.ttl_chunk_compress_factor;
        const int T0 = latent_len * factor;
        constexpr int MAX_NODES = 512;
        static size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                                 ggml_graph_overhead_custom(MAX_NODES, false);
        thread_local std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

        ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T0, C_latent);
        ggml_set_name(x_in, "unpack");
        ggml_set_input(x_in);

        const float normalizer_scale = scalar_f32_tensor(model.vocoder.normalizer_scale);
        ggml_tensor * x = ggml_scale(ctx, x_in, 1.0f / normalizer_scale);
        x = ggml_mul(ctx, x, repeat_like(ctx, model.vocoder.latent_std, x));
        x = ggml_add(ctx, x, repeat_like(ctx, model.vocoder.latent_mean, x));
        ggml_set_name(x, "denorm");
        ggml_set_output(x);
        ggml_build_forward_expand(gf, x);

        ggml_tensor * embed = conv1d_causal_ggml(ctx, x, model.vocoder.embed_w, model.vocoder.embed_b);
        ggml_set_name(embed, "embed");
        ggml_set_output(embed);
        ggml_build_forward_expand(gf, embed);

        const auto & cw = model.vocoder.convnext[0];
        ggml_tensor * dw = depthwise_conv1d_causal_ggml(ctx, embed, cw.dw_w, cw.dw_b, 1);
        ggml_set_name(dw, "block0_dw");
        ggml_set_output(dw);
        ggml_build_forward_expand(gf, dw);
        ggml_tensor * norm = layer_norm_channel_ggml(ctx, dw, cw.norm_g, cw.norm_b);
        ggml_set_name(norm, "block0_norm");
        ggml_set_output(norm);
        ggml_build_forward_expand(gf, norm);
        ggml_tensor * pw1 = conv1d_causal_ggml(ctx, norm, cw.pw1_w, cw.pw1_b);
        ggml_set_name(pw1, "block0_pw1");
        ggml_set_output(pw1);
        ggml_build_forward_expand(gf, pw1);
        ggml_tensor * gelu0 = ggml_gelu_erf(ctx, pw1);
        ggml_set_name(gelu0, "block0_gelu");
        ggml_set_output(gelu0);
        ggml_build_forward_expand(gf, gelu0);
        ggml_tensor * pw2 = conv1d_causal_ggml(ctx, gelu0, cw.pw2_w, cw.pw2_b);
        ggml_set_name(pw2, "block0_pw2");
        ggml_set_output(pw2);
        ggml_build_forward_expand(gf, pw2);
        ggml_tensor * out0 = ggml_add(ctx, embed, ggml_mul(ctx, pw2, repeat_like(ctx, cw.gamma, pw2)));
        ggml_set_name(out0, "block0_out");
        ggml_set_output(out0);
        ggml_build_forward_expand(gf, out0);

        ggml_tensor * cur = out0;
        for (int i = 1; i < 10; ++i) {
            cur = convnext_block_ggml(ctx, model.vocoder.convnext[(size_t) i], cur, i);
            ggml_set_name(cur, ("block" + std::to_string(i) + "_out").c_str());
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
        }

        ggml_tensor * bn_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        ggml_set_name(bn_scale, "trace_bn_scale");
        ggml_set_input(bn_scale);
        ggml_tensor * bn_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        ggml_set_name(bn_shift, "trace_bn_shift");
        ggml_set_input(bn_shift);
        cur = ggml_mul(ctx, cur, repeat_like(ctx, bn_scale, cur));
        cur = ggml_add(ctx, cur, repeat_like(ctx, bn_shift, cur));
        ggml_set_name(cur, "final_norm");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        cur = conv1d_causal_ggml(ctx, cur, model.vocoder.head1_w, model.vocoder.head1_b);
        ggml_set_name(cur, "head1");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        cur = ggml_leaky_relu(ctx, cur, scalar_f32_tensor(model.vocoder.head_prelu), false);
        ggml_set_name(cur, "prelu");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        cur = conv1d_causal_ggml(ctx, cur, model.vocoder.head2_w, nullptr);
        ggml_set_name(cur, "wav");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

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

        std::vector<float> x_host = unpack_latent_ggml_layout(model, latent, latent_len);
        ggml_backend_tensor_set(x_in, x_host.data(), 0, x_host.size() * sizeof(float));
        f32_tensor gamma = read_f32_tensor(model.vocoder.final_norm_g);
        f32_tensor beta = read_f32_tensor(model.vocoder.final_norm_b);
        f32_tensor mean = read_f32_tensor(model.vocoder.final_norm_running_mean);
        f32_tensor var = read_f32_tensor(model.vocoder.final_norm_running_var);
        std::vector<float> bn_scale_host(512), bn_shift_host(512);
        for (int c = 0; c < 512; ++c) {
            bn_scale_host[c] = gamma.data[c] / std::sqrt(var.data[c] + 1e-5f);
            bn_shift_host[c] = beta.data[c] - mean.data[c] * bn_scale_host[c];
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "trace_bn_scale"), bn_scale_host.data(), 0, bn_scale_host.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "trace_bn_shift"), bn_shift_host.data(), 0, bn_shift_host.size() * sizeof(float));
        supertonic_graph_compute(model, gf);

        trace_out.push_back({"unpack", {T0, C_latent}, unpack_latent_scalar(model, latent, latent_len)});
        trace_out.push_back({"denorm", {T0, C_latent}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "denorm"))});
        trace_out.push_back({"embed", {T0, (int) model.vocoder.embed_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "embed"))});
        trace_out.push_back({"block0_dw", {T0, (int) model.vocoder.embed_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "block0_dw"))});
        trace_out.push_back({"block0_norm", {T0, (int) model.vocoder.embed_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "block0_norm"))});
        trace_out.push_back({"block0_pw1", {T0, (int) model.vocoder.convnext[0].pw1_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "block0_pw1"))});
        trace_out.push_back({"block0_gelu", {T0, (int) model.vocoder.convnext[0].pw1_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "block0_gelu"))});
        trace_out.push_back({"block0_pw2", {T0, (int) model.vocoder.embed_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "block0_pw2"))});
        trace_out.push_back({"block0_out", {T0, (int) model.vocoder.embed_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "block0_out"))});
        for (int i = 1; i < 10; ++i) {
            trace_out.push_back({"block" + std::to_string(i) + "_out", {T0, (int) model.vocoder.embed_w->ne[2]},
                                 ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, ("block" + std::to_string(i) + "_out").c_str()))});
        }
        trace_out.push_back({"final_norm", {T0, (int) model.vocoder.embed_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "final_norm"))});
        trace_out.push_back({"head1", {T0, (int) model.vocoder.head1_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "head1"))});
        trace_out.push_back({"prelu", {T0, (int) model.vocoder.head1_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "prelu"))});
        trace_out.push_back({"wav", {T0, (int) model.vocoder.head2_w->ne[2]}, ggml_tensor_to_time_channel(ggml_graph_get_tensor(gf, "wav"))});
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
