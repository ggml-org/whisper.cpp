#include "supertonic_internal.h"

#include "ggml-alloc.h"

#include <algorithm>
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
    int64_t ne[4] = {1, 1, 1, 1};
};

f32_tensor read_f32(const supertonic_model & m, const std::string & source_name) {
    ggml_tensor * t = require_source_tensor(m, source_name);
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f)); }
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

bool text_profile_enabled() {
    static const bool enabled = std::getenv("SUPERTONIC_TEXT_PROFILE") != nullptr;
    return enabled;
}

struct text_profile_state {
    std::chrono::steady_clock::time_point start{};
    std::chrono::steady_clock::time_point last{};
};

text_profile_state & text_profile() {
    thread_local text_profile_state state;
    return state;
}

void profile_text_begin() {
    if (!text_profile_enabled()) return;
    auto & state = text_profile();
    state.start = std::chrono::steady_clock::now();
    state.last = state.start;
}

void profile_text_compute(const supertonic_model & model, ggml_cgraph * graph, const char * island) {
    if (!text_profile_enabled()) {
        supertonic_graph_compute(model, graph);
        return;
    }
    auto & state = text_profile();
    const auto t0 = std::chrono::steady_clock::now();
    const double pre_ms = std::chrono::duration<double, std::milli>(t0 - state.last).count();
    supertonic_graph_compute(model, graph);
    const auto t1 = std::chrono::steady_clock::now();
    const double compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    state.last = t1;
    std::fprintf(stderr, "supertonic_text_profile island=%s pre_ms=%.3f compute_ms=%.3f\n",
                 island, pre_ms, compute_ms);
}

void profile_text_checkpoint(const char * island) {
    if (!text_profile_enabled()) return;
    auto & state = text_profile();
    const auto now = std::chrono::steady_clock::now();
    const double host_ms = std::chrono::duration<double, std::milli>(now - state.last).count();
    state.last = now;
    std::fprintf(stderr, "supertonic_text_profile island=%s host_ms=%.3f\n", island, host_ms);
}

void profile_text_end() {
    if (!text_profile_enabled()) return;
    auto & state = text_profile();
    const auto now = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(now - state.start).count();
    std::fprintf(stderr, "supertonic_text_profile island=total total_ms=%.3f\n", total_ms);
}

void linear1x1(const std::vector<float> & x, int L, int IC,
               const f32_tensor & w, const f32_tensor * b,
               int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            const size_t woff = (size_t) oc * IC;
            for (int ic = 0; ic < IC; ++ic) sum += w.data[woff + ic] * x[(size_t) t * IC + ic];
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

ggml_tensor * repeat_like(ggml_context * ctx, ggml_tensor * v, ggml_tensor * like) {
    if (ggml_n_dims(v) == 1 && ggml_n_dims(like) >= 2) {
        if (like->ne[0] == v->ne[0]) v = ggml_reshape_2d(ctx, v, v->ne[0], 1);
        else if (like->ne[1] == v->ne[0]) v = ggml_reshape_2d(ctx, v, 1, v->ne[0]);
    }
    if (!ggml_can_repeat(v, like)) throw std::runtime_error("cannot repeat tensor in text encoder graph");
    return ggml_repeat(ctx, v, like);
}

ggml_tensor * conv1d_f32(ggml_context * ctx,
                         ggml_tensor * kernel,
                         ggml_tensor * input,
                         int stride,
                         int padding,
                         int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

ggml_tensor * edge_clamp_pad_1d(ggml_context * ctx, ggml_tensor * x, int pad_left, int pad_right) {
    const int64_t L = x->ne[0], C = x->ne[1];
    ggml_tensor * out = x;
    if (pad_left > 0) {
        ggml_tensor * first = ggml_view_2d(ctx, x, 1, C, x->nb[1], 0);
        out = ggml_concat(ctx, ggml_repeat_4d(ctx, first, pad_left, C, 1, 1), out, 0);
    }
    if (pad_right > 0) {
        ggml_tensor * last = ggml_view_2d(ctx, x, 1, C, x->nb[1], (size_t)(L - 1) * x->nb[0]);
        out = ggml_concat(ctx, out, ggml_repeat_4d(ctx, last, pad_right, C, 1, 1), 0);
    }
    return out;
}

ggml_tensor * depthwise_same_ggml(ggml_context * ctx,
                                  ggml_tensor * x,
                                  ggml_tensor * w,
                                  ggml_tensor * b) {
    const int K = (int)w->ne[0];
    const int pad_left = (K - 1) / 2;
    const int pad_right = (K - 1) - pad_left;
    ggml_tensor * padded = edge_clamp_pad_1d(ctx, x, pad_left, pad_right);
    ggml_tensor * new_b = ggml_reshape_4d(ctx, padded, padded->ne[0], 1, padded->ne[1], padded->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, w, new_b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, im2col, w);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);
    return ggml_add(ctx, y, repeat_like(ctx, b, y));
}

ggml_tensor * layer_norm_ggml(ggml_context * ctx, ggml_tensor * x, ggml_tensor * g, ggml_tensor * b) {
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
    ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, w));
    ggml_tensor * kernel = ggml_reshape_3d(ctx, wt, 1, w->ne[1], w->ne[0]);
    ggml_tensor * y = conv1d_f32(ctx, kernel, x, 1, 0, 1);
    if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
    return y;
}

ggml_tensor * conv1d_k1_channel_time_ggml(ggml_context * ctx,
                                          ggml_tensor * kernel,
                                          ggml_tensor * x_lc,
                                          ggml_tensor * bias) {
    ggml_tensor * x_cl = ggml_cont(ctx, ggml_transpose(ctx, x_lc));
    ggml_tensor * w2d = ggml_reshape_2d(ctx, kernel, kernel->ne[1], kernel->ne[2]);
    ggml_tensor * y = ggml_mul_mat(ctx, w2d, x_cl);
    if (bias) y = ggml_add(ctx, y, repeat_like(ctx, bias, y));
    return y;
}

ggml_tensor * text_convnext_ggml(ggml_context * ctx,
                                const supertonic_model & model,
                                const std::string & p,
                                ggml_tensor * x) {
    ggml_tensor * residual = x;
    ggml_tensor * y = depthwise_same_ggml(ctx, x,
        require_source_tensor(model, p + ".dwconv.weight"),
        require_source_tensor(model, p + ".dwconv.bias"));
    y = layer_norm_ggml(ctx, y,
        require_source_tensor(model, p + ".norm.norm.weight"),
        require_source_tensor(model, p + ".norm.norm.bias"));
    y = conv1d_f32(ctx, require_source_tensor(model, p + ".pwconv1.weight"), y, 1, 0, 1);
    y = ggml_add(ctx, y, repeat_like(ctx, require_source_tensor(model, p + ".pwconv1.bias"), y));
    y = ggml_gelu_erf(ctx, y);
    y = conv1d_f32(ctx, require_source_tensor(model, p + ".pwconv2.weight"), y, 1, 0, 1);
    y = ggml_add(ctx, y, repeat_like(ctx, require_source_tensor(model, p + ".pwconv2.bias"), y));
    y = ggml_mul(ctx, y, repeat_like(ctx, require_source_tensor(model, p + ".gamma"), y));
    return ggml_add(ctx, residual, y);
}

std::vector<float> tensor_to_time_channel(ggml_tensor * t) {
    const int L = (int)t->ne[0], C = (int)t->ne[1];
    std::vector<float> raw((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
    std::vector<float> out((size_t)L*C);
    for (int c = 0; c < C; ++c) for (int i = 0; i < L; ++i) out[(size_t)i*C+c] = raw[(size_t)c*L+i];
    return out;
}

std::vector<float> pack_time_channel_for_ggml(const std::vector<float> & x, int L, int C) {
    std::vector<float> out((size_t)L*C);
    for (int t = 0; t < L; ++t) for (int c = 0; c < C; ++c) out[(size_t)c*L+t] = x[(size_t)t*C+c];
    return out;
}

void push_trace(std::vector<supertonic_trace_tensor> & trace,
                const std::string & name,
                int L,
                int C,
                const std::vector<float> & data) {
    trace.push_back({name, {L, C}, data});
}

void dense_time(const std::vector<float> & x, int L, int IC,
                const f32_tensor & w, const f32_tensor & b,
                int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b.data[oc];
            for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t) oc * IC + ic] * x[(size_t) t * IC + ic];
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void dense_time_matmul(const std::vector<float> & x, int L, int IC,
                       const f32_tensor & w, const f32_tensor & b,
                       int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    // ONNX MatMul constants are row-major [IC, OC]; PyTorch Linear weights
    // transposed these at load time, but here we consume the raw ONNX tensor.
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b.data[oc];
            for (int ic = 0; ic < IC; ++ic) sum += x[(size_t) t * IC + ic] * w.data[(size_t) ic * OC + oc];
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void depthwise_conv1d_same(const std::vector<float> & x, int L, int C,
                           const f32_tensor & w, const f32_tensor & b,
                           int K, int dilation, std::vector<float> & y) {
    y.assign((size_t) L * C, 0.0f);
    const int total_pad = (K - 1) * dilation;
    const int pad_left = total_pad / 2;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b.data[c];
            const size_t wbase = (size_t) c * K;
            for (int k = 0; k < K; ++k) {
                int src_t = t + k * dilation - pad_left;
                src_t = std::max(0, std::min(L - 1, src_t));
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

void convnext_block(const supertonic_model & m, const std::string & p,
                    std::vector<float> & x, int L, int C) {
    f32_tensor dw_w = read_f32(m, p + ".dwconv.weight");
    f32_tensor dw_b = read_f32(m, p + ".dwconv.bias");
    f32_tensor ln_g = read_f32(m, p + ".norm.norm.weight");
    f32_tensor ln_b = read_f32(m, p + ".norm.norm.bias");
    f32_tensor pw1_w = read_f32(m, p + ".pwconv1.weight");
    f32_tensor pw1_b = read_f32(m, p + ".pwconv1.bias");
    f32_tensor pw2_w = read_f32(m, p + ".pwconv2.weight");
    f32_tensor pw2_b = read_f32(m, p + ".pwconv2.bias");
    f32_tensor gamma = read_f32(m, p + ".gamma");
    std::vector<float> residual = x;
    std::vector<float> y, z;
    depthwise_conv1d_same(x, L, C, dw_w, dw_b, (int) dw_w.ne[0], 1, y);
    layer_norm_channel(y, L, C, ln_g, ln_b);
    linear1x1(y, L, C, pw1_w, &pw1_b, (int) pw1_w.ne[2], z);
    for (float & v : z) v = gelu(v);
    linear1x1(z, L, (int) pw1_w.ne[2], pw2_w, &pw2_b, C, y);
    for (size_t i = 0; i < x.size(); ++i) {
        int c = (int) (i % C);
        x[i] = residual[i] + gamma.data[c] * y[i];
    }
}

void relpos_attention(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    const int H = 4;
    const int D = C / H;
    const float scale = 1.0f / std::sqrt((float) D);
    const std::string p = "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers." + std::to_string(idx);
    f32_tensor q_w = read_f32(m, p + ".conv_q.weight");
    f32_tensor q_b = read_f32(m, p + ".conv_q.bias");
    f32_tensor k_w = read_f32(m, p + ".conv_k.weight");
    f32_tensor k_b = read_f32(m, p + ".conv_k.bias");
    f32_tensor v_w = read_f32(m, p + ".conv_v.weight");
    f32_tensor v_b = read_f32(m, p + ".conv_v.bias");
    f32_tensor o_w = read_f32(m, p + ".conv_o.weight");
    f32_tensor o_b = read_f32(m, p + ".conv_o.bias");
    f32_tensor rel_k = read_f32(m, p + ".emb_rel_k");
    f32_tensor rel_v = read_f32(m, p + ".emb_rel_v");
    std::vector<float> q, k, v;
    linear1x1(x, L, C, q_w, &q_b, C, q);
    linear1x1(x, L, C, k_w, &k_b, C, k);
    linear1x1(x, L, C, v_w, &v_b, C, v);
    std::vector<float> out((size_t) L * C, 0.0f);
    std::vector<float> scores(L), probs(L);
    const int half_window = 4;
    for (int h = 0; h < H; ++h) {
        for (int qi = 0; qi < L; ++qi) {
            float max_score = -INFINITY;
            for (int kj = 0; kj < L; ++kj) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s += q[(size_t) qi * C + h * D + d] * scale * k[(size_t) kj * C + h * D + d];
                }
                int rel_pos = kj - qi;
                if (rel_pos >= -half_window && rel_pos <= half_window) {
                    int ridx = rel_pos + half_window;
                    for (int d = 0; d < D; ++d) {
                        s += q[(size_t) qi * C + h * D + d] * scale * rel_k.data[(size_t) ridx * D + d];
                    }
                }
                scores[kj] = s;
                max_score = std::max(max_score, s);
            }
            float denom = 0.0f;
            for (int kj = 0; kj < L; ++kj) {
                probs[kj] = std::exp(scores[kj] - max_score);
                denom += probs[kj];
            }
            for (int kj = 0; kj < L; ++kj) probs[kj] /= denom;
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int kj = 0; kj < L; ++kj) {
                    sum += probs[kj] * v[(size_t) kj * C + h * D + d];
                    int rel_pos = kj - qi;
                    if (rel_pos >= -half_window && rel_pos <= half_window) {
                        int ridx = rel_pos + half_window;
                        sum += probs[kj] * rel_v.data[(size_t) ridx * D + d];
                    }
                }
                out[(size_t) qi * C + h * D + d] = sum;
            }
        }
    }
    std::vector<float> proj;
    linear1x1(out, L, C, o_w, &o_b, C, proj);
    x.swap(proj);
}

struct text_relpos_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int idx = -1;
    int L = 0;
    int C = 0;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * masks[9] = {};
};

void free_relpos_cache(text_relpos_graph_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

void build_relpos_cache(text_relpos_graph_cache & cache,
                        const supertonic_model & m,
                        int idx,
                        int L,
                        int C) {
    free_relpos_cache(cache);
    cache.model = &m;
    cache.generation_id = m.generation_id;
    cache.idx = idx;
    cache.L = L;
    cache.C = C;
    const int H = 4;
    const int D = C / H;
    const float scale = 1.0f / std::sqrt((float)D);
    const std::string p = "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers." + std::to_string(idx);
    constexpr int N_MASKS = 9;
    constexpr int MAX_NODES = 768;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params gp = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(gp);
    cache.gf = ggml_new_graph_custom(cache.ctx, MAX_NODES, false);

    cache.x_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(cache.x_in, "relpos_x"); ggml_set_input(cache.x_in);
    for (int i = 0; i < N_MASKS; ++i) {
        cache.masks[i] = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, L, L, 1);
        const std::string name = "relpos_mask_" + std::to_string(i);
        ggml_set_name(cache.masks[i], name.c_str()); ggml_set_input(cache.masks[i]);
    }

    ggml_tensor * q = conv1d_k1_channel_time_ggml(cache.ctx,
        require_source_tensor(m, p + ".conv_q.weight"), cache.x_in,
        require_source_tensor(m, p + ".conv_q.bias"));
    ggml_tensor * k = conv1d_k1_channel_time_ggml(cache.ctx,
        require_source_tensor(m, p + ".conv_k.weight"), cache.x_in,
        require_source_tensor(m, p + ".conv_k.bias"));
    ggml_tensor * v = conv1d_k1_channel_time_ggml(cache.ctx,
        require_source_tensor(m, p + ".conv_v.weight"), cache.x_in,
        require_source_tensor(m, p + ".conv_v.bias"));

    const size_t time_stride = (size_t)C * sizeof(float);
    const size_t head_stride = (size_t)D * sizeof(float);
    ggml_tensor * q_dlh = ggml_view_3d(cache.ctx, q, D, L, H, time_stride, head_stride, 0);
    ggml_tensor * k_dlh = ggml_view_3d(cache.ctx, k, D, L, H, time_stride, head_stride, 0);
    ggml_tensor * v_dlh = ggml_view_3d(cache.ctx, v, D, L, H, time_stride, head_stride, 0);

    ggml_tensor * scores = ggml_scale(cache.ctx, ggml_mul_mat(cache.ctx, k_dlh, q_dlh), scale);
    ggml_tensor * rel_k = require_source_tensor(m, p + ".emb_rel_k");
    ggml_tensor * rel_scores = ggml_mul_mat(cache.ctx, rel_k, q_dlh);
    for (int ri = 0; ri < N_MASKS; ++ri) {
        ggml_tensor * rel_delta = ggml_view_3d(cache.ctx, rel_scores, 1, L, H,
                                               rel_scores->nb[1], rel_scores->nb[2],
                                               (size_t)ri * rel_scores->nb[0]);
        rel_delta = ggml_repeat(cache.ctx, rel_delta, scores);
        ggml_tensor * mask = ggml_repeat(cache.ctx, cache.masks[ri], scores);
        scores = ggml_add(cache.ctx, scores, ggml_mul(cache.ctx, ggml_scale(cache.ctx, rel_delta, scale), mask));
    }

    ggml_tensor * attn = ggml_soft_max(cache.ctx, scores);
    ggml_tensor * v_for_mm = ggml_cont(cache.ctx, ggml_permute(cache.ctx, v_dlh, 1, 0, 2, 3));
    ggml_tensor * out = ggml_mul_mat(cache.ctx, v_for_mm, attn);

    ggml_tensor * rel_v = require_source_tensor(m, p + ".emb_rel_v");
    ggml_tensor * rel_out = ggml_scale(cache.ctx, out, 0.0f);
    for (int ri = 0; ri < N_MASKS; ++ri) {
        ggml_tensor * mask = ggml_repeat(cache.ctx, cache.masks[ri], attn);
        ggml_tensor * p_delta = ggml_sum_rows(cache.ctx, ggml_mul(cache.ctx, attn, mask));
        p_delta = ggml_repeat(cache.ctx, p_delta, out);
        ggml_tensor * rv_delta = ggml_view_3d(cache.ctx, rel_v, D, 1, 1,
                                              rel_v->nb[1], rel_v->nb[2],
                                              (size_t)ri * rel_v->nb[1]);
        rv_delta = ggml_repeat(cache.ctx, rv_delta, out);
        rel_out = ggml_add(cache.ctx, rel_out, ggml_mul(cache.ctx, p_delta, rv_delta));
    }
    out = ggml_add(cache.ctx, out, rel_out);

    ggml_tensor * out_lc_t = ggml_cont(cache.ctx, ggml_permute(cache.ctx, out, 1, 0, 2, 3));
    out_lc_t = ggml_reshape_2d(cache.ctx, out_lc_t, L, C);
    out_lc_t = conv1d_f32(cache.ctx, require_source_tensor(m, p + ".conv_o.weight"), out_lc_t, 1, 0, 1);
    out_lc_t = ggml_add(cache.ctx, out_lc_t, repeat_like(cache.ctx, require_source_tensor(m, p + ".conv_o.bias"), out_lc_t));
    ggml_set_name(out_lc_t, "relpos_out"); ggml_set_output(out_lc_t);
    ggml_build_forward_expand(cache.gf, out_lc_t);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new text relpos failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve text relpos failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    for (int ri = 0; ri < N_MASKS; ++ri) {
        const int delta = ri - 4;
        std::vector<float> mask((size_t)L * L, 0.0f);
        for (int qi = 0; qi < L; ++qi) {
            const int kj = qi + delta;
            if (kj >= 0 && kj < L) mask[(size_t)kj + (size_t)L*qi] = 1.0f;
        }
        ggml_backend_tensor_set(cache.masks[ri], mask.data(), 0, mask.size()*sizeof(float));
    }
}

void relpos_attention_ggml(const supertonic_model & m, int idx,
                           const std::vector<float> & x_lc,
                           int L,
                           int C,
                           std::vector<float> & out_lc) {
    if (idx < 0 || idx >= 4) throw std::runtime_error("invalid text relpos layer index");
    thread_local text_relpos_graph_cache caches[4];
    text_relpos_graph_cache & cache = caches[idx];
    if (cache.model != &m || cache.generation_id != m.generation_id ||
        cache.idx != idx || cache.L != L || cache.C != C) {
        build_relpos_cache(cache, m, idx, L, C);
    }
    std::vector<float> x_raw = pack_time_channel_for_ggml(x_lc, L, C);
    ggml_backend_tensor_set(cache.x_in, x_raw.data(), 0, x_raw.size()*sizeof(float));
    std::string island = "relpos" + std::to_string(idx);
    profile_text_compute(m, cache.gf, island.c_str());
    out_lc = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "relpos_out"));
}

void ffn_block(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    const std::string p = "text_encoder:tts.ttl.text_encoder.attn_encoder.ffn_layers." + std::to_string(idx);
    f32_tensor w1 = read_f32(m, p + ".conv_1.weight");
    f32_tensor b1 = read_f32(m, p + ".conv_1.bias");
    f32_tensor w2 = read_f32(m, p + ".conv_2.weight");
    f32_tensor b2 = read_f32(m, p + ".conv_2.bias");
    std::vector<float> y;
    linear1x1(x, L, C, w1, &b1, (int) w1.ne[2], y);
    for (float & v : y) v = relu(v);
    linear1x1(y, L, (int) w1.ne[2], w2, &b2, C, x);
}

struct text_ffn_graph_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int idx = -1;
    int L = 0;
    int C = 0;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * x_in = nullptr;
};

void free_ffn_cache(text_ffn_graph_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

void build_ffn_cache(text_ffn_graph_cache & cache,
                     const supertonic_model & m,
                     int idx,
                     int L,
                     int C) {
    free_ffn_cache(cache);
    cache.model = &m;
    cache.generation_id = m.generation_id;
    cache.idx = idx;
    cache.L = L;
    cache.C = C;
    const std::string p = "text_encoder:tts.ttl.text_encoder.attn_encoder.ffn_layers." + std::to_string(idx);
    constexpr int MAX_NODES = 256;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params gp = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(gp);
    cache.gf = ggml_new_graph_custom(cache.ctx, MAX_NODES, false);
    cache.x_in = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(cache.x_in, "text_ffn_in"); ggml_set_input(cache.x_in);

    ggml_tensor * y = conv1d_f32(cache.ctx, require_source_tensor(m, p + ".conv_1.weight"), cache.x_in, 1, 0, 1);
    y = ggml_add(cache.ctx, y, repeat_like(cache.ctx, require_source_tensor(m, p + ".conv_1.bias"), y));
    y = ggml_relu(cache.ctx, y);
    y = conv1d_f32(cache.ctx, require_source_tensor(m, p + ".conv_2.weight"), y, 1, 0, 1);
    y = ggml_add(cache.ctx, y, repeat_like(cache.ctx, require_source_tensor(m, p + ".conv_2.bias"), y));
    ggml_set_name(y, "text_ffn_out"); ggml_set_output(y);
    ggml_build_forward_expand(cache.gf, y);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new text ffn failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve text ffn failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

void ffn_block_ggml(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    if (idx < 0 || idx >= 4) throw std::runtime_error("invalid text ffn layer index");
    thread_local text_ffn_graph_cache caches[4];
    text_ffn_graph_cache & cache = caches[idx];
    if (cache.model != &m || cache.generation_id != m.generation_id ||
        cache.idx != idx || cache.L != L || cache.C != C) {
        build_ffn_cache(cache, m, idx, L, C);
    }
    std::vector<float> raw = pack_time_channel_for_ggml(x, L, C);
    ggml_backend_tensor_set(cache.x_in, raw.data(), 0, raw.size()*sizeof(float));
    std::string island = "ffn" + std::to_string(idx);
    profile_text_compute(m, cache.gf, island.c_str());
    x = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "text_ffn_out"));
}

void speech_prompted_attention(const supertonic_model & m, int idx,
                               std::vector<float> & x_lc, int L,
                               const float * style_ttl,
                               std::vector<float> & out_lc) {
    const int C = 256;
    const int half = 128;
    const int Lctx = 50;
    const float scale = 1.0f / 16.0f;
    const int attn_num = idx + 1;
    const std::string p = "text_encoder:tts.ttl.speech_prompted_text_encoder.attention" + std::to_string(attn_num);
    f32_tensor q_w = read_f32(m, "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3678" : "onnx::MatMul_3682"));
    f32_tensor q_b = read_f32(m, p + ".W_query.linear.bias");
    f32_tensor kv_w = read_f32(m, "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3680" : "onnx::MatMul_3684"));
    f32_tensor kv_b = read_f32(m, p + ".W_value.linear.bias");
    f32_tensor out_w = read_f32(m, "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3681" : "onnx::MatMul_3685"));
    f32_tensor out_b = read_f32(m, p + ".out_fc.linear.bias");
    f32_tensor tanh_k = read_f32(m, "text_encoder:/speech_prompted_text_encoder/attention" + std::to_string(attn_num) + "/tanh/Tanh_output_0");

    std::vector<float> q;
    dense_time_matmul(x_lc, L, C, q_w, q_b, C, q);

    std::vector<float> style((size_t) Lctx * C);
    // style_ttl is NumPy row-major [1, 50, 256].
    for (int t = 0; t < Lctx; ++t) {
        for (int c = 0; c < C; ++c) style[(size_t) t * C + c] = style_ttl[(size_t) t * C + c];
    }
    std::vector<float> kv;
    dense_time_matmul(style, Lctx, C, kv_w, kv_b, C, kv);

    std::vector<float> attn((size_t) 2 * L * Lctx);
    std::vector<float> scores(Lctx);
    for (int part = 0; part < 2; ++part) {
        for (int t = 0; t < L; ++t) {
            float max_score = -INFINITY;
            for (int j = 0; j < Lctx; ++j) {
                float s = 0.0f;
                for (int d = 0; d < half; ++d) {
                    // tanh_k shape [2, 1, 128, 50] row-major.
                    float k = tanh_k.data[((size_t) part * half + d) * Lctx + j];
                    s += q[(size_t) t * C + part * half + d] * k * scale;
                }
                scores[j] = s;
                max_score = std::max(max_score, s);
            }
            float denom = 0.0f;
            for (int j = 0; j < Lctx; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }
            for (int j = 0; j < Lctx; ++j) attn[((size_t) part * L + t) * Lctx + j] = scores[j] / denom;
        }
    }

    std::vector<float> merged((size_t) L * C, 0.0f);
    for (int part = 0; part < 2; ++part) {
        for (int t = 0; t < L; ++t) {
            for (int d = 0; d < half; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < Lctx; ++j) {
                    // kv_stacked = [k, v]; part 0 uses k split, part 1 uses v split.
                    sum += attn[((size_t) part * L + t) * Lctx + j] * kv[(size_t) j * C + part * half + d];
                }
                merged[(size_t) t * C + part * half + d] = sum;
            }
        }
    }
    dense_time_matmul(merged, L, C, out_w, out_b, C, out_lc);
}

struct speech_attention_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int idx = -1;
    int L = 0;
    int Lctx = 0;
    std::string out_w_source;
    std::string out_b_source;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

void free_speech_attention_cache(speech_attention_cache & cache) {
    supertonic_safe_gallocr_free(cache.allocr, cache.generation_id);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = {};
}

void build_speech_attention_cache(speech_attention_cache & cache,
                                  const supertonic_model & m,
                                  int idx,
                                  int L,
                                  int Lctx,
                                  const std::string & out_w_source,
                                  const std::string & out_b_source) {
    free_speech_attention_cache(cache);
    cache.model = &m;
    cache.generation_id = m.generation_id;
    cache.idx = idx;
    cache.L = L;
    cache.Lctx = Lctx;
    cache.out_w_source = out_w_source;
    cache.out_b_source = out_b_source;
    constexpr int NODES = 256;
    const size_t buf_size = ggml_tensor_overhead() * NODES + ggml_graph_overhead_custom(NODES, false);
    cache.buf.assign(buf_size, 0);
    ggml_init_params gp = { buf_size, cache.buf.data(), true };
    cache.ctx = ggml_init(gp);
    cache.gf = ggml_new_graph_custom(cache.ctx, NODES, false);
    cache.q = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, 128, L, 2);
    ggml_set_name(cache.q, "speech_attn_q_dlh"); ggml_set_input(cache.q);
    cache.k = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, 128, Lctx, 2);
    ggml_set_name(cache.k, "speech_attn_k_dlh"); ggml_set_input(cache.k);
    cache.v = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, 128, Lctx, 2);
    ggml_set_name(cache.v, "speech_attn_v_dlh"); ggml_set_input(cache.v);
    ggml_tensor * attn = ggml_flash_attn_ext(cache.ctx, cache.q, cache.k, cache.v, nullptr, 1.0f / 16.0f, 0.0f, 0.0f);
    attn = ggml_reshape_2d(cache.ctx, attn, 256, L);
    ggml_tensor * ctx_tc = ggml_cont(cache.ctx, ggml_transpose(cache.ctx, attn));
    ggml_tensor * out = dense_matmul_time_ggml(cache.ctx, ctx_tc,
        require_source_tensor(m, out_w_source),
        require_source_tensor(m, out_b_source));
    ggml_set_name(out, "speech_attn_out"); ggml_set_output(out); ggml_build_forward_expand(cache.gf, out);
    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!cache.allocr) throw std::runtime_error("ggml_gallocr_new speech attention cache failed");
    if (!ggml_gallocr_reserve(cache.allocr, cache.gf)) {
        throw std::runtime_error("ggml_gallocr_reserve speech attention cache failed");
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
}

void speech_prompted_attention_ggml(const supertonic_model & m, int idx,
                                    const std::vector<float> & x_lc, int L,
                                    const float * style_ttl,
                                    std::vector<float> & out_lc) {
    const int C = 256;
    const int half = 128;
    const int Lctx = 50;
    const int attn_num = idx + 1;
    const std::string p = "text_encoder:tts.ttl.speech_prompted_text_encoder.attention" + std::to_string(attn_num);
    const std::string q_w = "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3678" : "onnx::MatMul_3682");
    const std::string v_w = "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3680" : "onnx::MatMul_3684");
    const std::string o_w = "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3681" : "onnx::MatMul_3685");

    constexpr int MAX_NODES = 256;
    static size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(x_in, "speech_attn_x"); ggml_set_input(x_in);
    ggml_tensor * style_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lctx, C);
    ggml_set_name(style_in, "speech_attn_style"); ggml_set_input(style_in);
    ggml_tensor * q = dense_matmul_time_ggml(ctx, x_in,
        require_source_tensor(m, q_w),
        require_source_tensor(m, p + ".W_query.linear.bias"));
    ggml_set_name(q, "speech_attn_q"); ggml_set_output(q); ggml_build_forward_expand(gf, q);
    ggml_tensor * v = dense_matmul_time_ggml(ctx, style_in,
        require_source_tensor(m, v_w),
        require_source_tensor(m, p + ".W_value.linear.bias"));
    ggml_set_name(v, "speech_attn_v"); ggml_set_output(v); ggml_build_forward_expand(gf, v);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!allocr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_gallocr_new speech text attention failed");
    }
    if (!ggml_gallocr_reserve(allocr, gf)) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        throw std::runtime_error("ggml_gallocr_reserve speech text attention failed");
    }
    ggml_gallocr_alloc_graph(allocr, gf);

    std::vector<float> x_raw = pack_time_channel_for_ggml(x_lc, L, C);
    std::vector<float> style_tc((size_t)Lctx*C);
    for (int t = 0; t < Lctx; ++t) for (int c = 0; c < C; ++c) style_tc[(size_t)t*C+c] = style_ttl[(size_t)t*C+c];
    std::vector<float> style_raw = pack_time_channel_for_ggml(style_tc, Lctx, C);
    ggml_backend_tensor_set(x_in, x_raw.data(), 0, x_raw.size()*sizeof(float));
    ggml_backend_tensor_set(style_in, style_raw.data(), 0, style_raw.size()*sizeof(float));
    std::string qkv_island = "speech" + std::to_string(idx) + "_qkv";
    profile_text_compute(m, gf, qkv_island.c_str());

    std::vector<float> q_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "speech_attn_q"));
    std::vector<float> v_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "speech_attn_v"));
    f32_tensor tanh_k = read_f32(m, "text_encoder:/speech_prompted_text_encoder/attention" + std::to_string(attn_num) + "/tanh/Tanh_output_0");
    std::vector<float> q_pack((size_t)half*L*2), k_pack((size_t)half*Lctx*2), v_pack((size_t)half*Lctx*2);
    for (int h = 0; h < 2; ++h) {
        for (int t = 0; t < L; ++t) {
            for (int d = 0; d < half; ++d) q_pack[(size_t)d + (size_t)half*((size_t)t + (size_t)L*h)] = q_out[(size_t)t*C + h*half + d];
        }
        for (int t = 0; t < Lctx; ++t) {
            for (int d = 0; d < half; ++d) {
                k_pack[(size_t)d + (size_t)half*((size_t)t + (size_t)Lctx*h)] = tanh_k.data[((size_t)h*half + d)*Lctx + t];
                v_pack[(size_t)d + (size_t)half*((size_t)t + (size_t)Lctx*h)] = v_out[(size_t)t*C + h*half + d];
            }
        }
    }
    thread_local speech_attention_cache caches[2];
    speech_attention_cache & cache = caches[idx];
    if (cache.model != &m || cache.generation_id != m.generation_id ||
        cache.idx != idx || cache.L != L || cache.Lctx != Lctx ||
        cache.out_w_source != o_w || cache.out_b_source != p + ".out_fc.linear.bias") {
        build_speech_attention_cache(cache, m, idx, L, Lctx, o_w, p + ".out_fc.linear.bias");
    }
    ggml_backend_tensor_set(cache.q, q_pack.data(), 0, q_pack.size()*sizeof(float));
    ggml_backend_tensor_set(cache.k, k_pack.data(), 0, k_pack.size()*sizeof(float));
    ggml_backend_tensor_set(cache.v, v_pack.data(), 0, v_pack.size()*sizeof(float));
    std::string flash_island = "speech" + std::to_string(idx) + "_flash";
    profile_text_compute(m, cache.gf, flash_island.c_str());
    out_lc = tensor_to_time_channel(ggml_graph_get_tensor(cache.gf, "speech_attn_out"));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}

} // namespace

bool supertonic_text_encoder_forward_cpu(const supertonic_model & model,
                                         const int64_t * text_ids,
                                         int text_len,
                                         const float * style_ttl,
                                         std::vector<float> & text_emb_out,
                                         std::string * error) {
    try {
        const int C = 256;
        const int L = text_len;
        f32_tensor emb = read_f32(model, "text_encoder:tts.ttl.text_encoder.text_embedder.char_embedder.weight");
        std::vector<float> x((size_t) L * C);
        for (int t = 0; t < L; ++t) {
            int64_t id = text_ids[t];
            if (id < 0 || id >= emb.ne[1]) throw std::runtime_error("text id out of range");
            for (int c = 0; c < C; ++c) x[(size_t) t * C + c] = emb.data[(size_t) id * C + c];
        }

        for (int i = 0; i < 6; ++i) {
            convnext_block(model, "text_encoder:tts.ttl.text_encoder.convnext.convnext." + std::to_string(i), x, L, C);
        }
        std::vector<float> convnext_out = x;

        for (int i = 0; i < 4; ++i) {
            std::vector<float> residual = x;
            relpos_attention(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.bias"));
            residual = x;
            ffn_block(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.bias"));
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += convnext_out[i];

        std::vector<float> shared_residual = x; // [L, C]
        std::vector<float> attn_out;
        speech_prompted_attention(model, 0, x, L, style_ttl, attn_out);
        for (size_t i = 0; i < x.size(); ++i) x[i] = shared_residual[i] + attn_out[i];
        speech_prompted_attention(model, 1, x, L, style_ttl, attn_out);
        for (size_t i = 0; i < x.size(); ++i) x[i] = shared_residual[i] + attn_out[i];

        layer_norm_channel(
            x, L, C,
            read_f32(model, "text_encoder:tts.ttl.speech_prompted_text_encoder.norm.norm.weight"),
            read_f32(model, "text_encoder:tts.ttl.speech_prompted_text_encoder.norm.norm.bias"));

        // Return in ONNX/PyTorch shape [1, 256, L] row-major.
        text_emb_out.assign((size_t) C * L, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < L; ++t) text_emb_out[(size_t) c * L + t] = x[(size_t) t * C + c];
        }
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_text_encoder_forward_ggml(const supertonic_model & model,
                                          const int64_t * text_ids,
                                          int text_len,
                                          const float * style_ttl,
                                          std::vector<float> & text_emb_out,
                                          std::string * error) {
    try {
        profile_text_begin();
        const int C = 256;
        const int L = text_len;
        f32_tensor emb = read_f32(model, "text_encoder:tts.ttl.text_encoder.text_embedder.char_embedder.weight");
        std::vector<float> x((size_t)L*C);
        for (int t = 0; t < L; ++t) {
            int64_t id = text_ids[t];
            if (id < 0 || id >= emb.ne[1]) throw std::runtime_error("text id out of range");
            for (int c = 0; c < C; ++c) x[(size_t)t*C+c] = emb.data[(size_t)id*C+c];
        }

        constexpr int MAX_NODES = 640;
        static size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
        thread_local std::vector<uint8_t> buf(buf_size);
        ggml_init_params gp = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);
        ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
        ggml_set_name(in, "text_encoder_embed"); ggml_set_input(in);
        ggml_tensor * y = in;
        for (int i = 0; i < 6; ++i) {
            y = text_convnext_ggml(ctx, model, "text_encoder:tts.ttl.text_encoder.convnext.convnext." + std::to_string(i), y);
        }
        ggml_set_name(y, "text_encoder_convnext5"); ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_new text encoder failed");
        }
        if (!ggml_gallocr_reserve(allocr, gf)) {
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_reserve text encoder failed");
        }
        ggml_gallocr_alloc_graph(allocr, gf);
        std::vector<float> raw = pack_time_channel_for_ggml(x, L, C);
        ggml_backend_tensor_set(in, raw.data(), 0, raw.size()*sizeof(float));
        profile_text_compute(model, gf, "convnext_front");
        x = tensor_to_time_channel(ggml_graph_get_tensor(gf, "text_encoder_convnext5"));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        profile_text_checkpoint("convnext_readback");

        // The text encoder's relative-position and speech-prompted attention
        // layers are custom scalar continuations for now; the ConvNeXt front
        // half above is already run as a GGML graph.
        std::vector<float> convnext_out = x;
        for (int i = 0; i < 4; ++i) {
            std::vector<float> residual = x;
            relpos_attention_ggml(model, i, x, L, C, x);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.bias"));
            std::string attn_post = "relpos" + std::to_string(i) + "_res_norm";
            profile_text_checkpoint(attn_post.c_str());
            residual = x;
            ffn_block_ggml(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.bias"));
            std::string ffn_post = "ffn" + std::to_string(i) + "_res_norm";
            profile_text_checkpoint(ffn_post.c_str());
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += convnext_out[i];
        profile_text_checkpoint("convnext_skip_add");

        std::vector<float> shared_residual = x;
        std::vector<float> attn_out;
        speech_prompted_attention_ggml(model, 0, x, L, style_ttl, attn_out);
        for (size_t i = 0; i < x.size(); ++i) x[i] = shared_residual[i] + attn_out[i];
        profile_text_checkpoint("speech0_residual");
        speech_prompted_attention_ggml(model, 1, x, L, style_ttl, attn_out);
        for (size_t i = 0; i < x.size(); ++i) x[i] = shared_residual[i] + attn_out[i];
        profile_text_checkpoint("speech1_residual");
        layer_norm_channel(
            x, L, C,
            read_f32(model, "text_encoder:tts.ttl.speech_prompted_text_encoder.norm.norm.weight"),
            read_f32(model, "text_encoder:tts.ttl.speech_prompted_text_encoder.norm.norm.bias"));
        profile_text_checkpoint("speech_norm");

        text_emb_out.assign((size_t) C * L, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < L; ++t) text_emb_out[(size_t)c*L+t] = x[(size_t)t*C+c];
        }
        profile_text_end();
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_text_encoder_trace_ggml(const supertonic_model & model,
                                        const int64_t * text_ids,
                                        int text_len,
                                        std::vector<supertonic_trace_tensor> & scalar_trace,
                                        std::vector<supertonic_trace_tensor> & ggml_trace,
                                        std::string * error) {
    try {
        scalar_trace.clear();
        ggml_trace.clear();
        const int C = 256;
        const int L = text_len;
        f32_tensor emb = read_f32(model, "text_encoder:tts.ttl.text_encoder.text_embedder.char_embedder.weight");
        std::vector<float> x((size_t)L*C);
        for (int t = 0; t < L; ++t) {
            int64_t id = text_ids[t];
            if (id < 0 || id >= emb.ne[1]) throw std::runtime_error("text id out of range");
            for (int c = 0; c < C; ++c) x[(size_t)t*C+c] = emb.data[(size_t)id*C+c];
        }
        push_trace(scalar_trace, "text_encoder_embed", L, C, x);
        std::vector<float> cur = x;
        for (int i = 0; i < 6; ++i) {
            convnext_block(model, "text_encoder:tts.ttl.text_encoder.convnext.convnext." + std::to_string(i), cur, L, C);
            push_trace(scalar_trace, "text_encoder_convnext" + std::to_string(i), L, C, cur);
        }
        std::vector<float> q0, k0, v0;
        f32_tensor q_w = read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_q.weight");
        f32_tensor q_b = read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_q.bias");
        f32_tensor k_w = read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_k.weight");
        f32_tensor k_b = read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_k.bias");
        f32_tensor v_w = read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_v.weight");
        f32_tensor v_b = read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_v.bias");
        linear1x1(cur, L, C, q_w, &q_b, C, q0);
        linear1x1(cur, L, C, k_w, &k_b, C, k0);
        linear1x1(cur, L, C, v_w, &v_b, C, v0);
        push_trace(scalar_trace, "text_encoder_attn0_q", L, C, q0);
        push_trace(scalar_trace, "text_encoder_attn0_k", L, C, k0);
        push_trace(scalar_trace, "text_encoder_attn0_v", L, C, v0);

        constexpr int MAX_NODES = 768;
        static size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
        thread_local std::vector<uint8_t> buf(buf_size);
        ggml_init_params gp = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);
        ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
        ggml_set_name(in, "text_encoder_embed"); ggml_set_input(in);
        ggml_tensor * y = in;
        for (int i = 0; i < 6; ++i) {
            y = text_convnext_ggml(ctx, model, "text_encoder:tts.ttl.text_encoder.convnext.convnext." + std::to_string(i), y);
            const std::string name = "text_encoder_convnext" + std::to_string(i);
            ggml_set_name(y, name.c_str()); ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        ggml_tensor * q = conv1d_f32(ctx, require_source_tensor(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_q.weight"), y, 1, 0, 1);
        q = ggml_add(ctx, q, repeat_like(ctx, require_source_tensor(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_q.bias"), q));
        ggml_set_name(q, "text_encoder_attn0_q"); ggml_set_output(q); ggml_build_forward_expand(gf, q);
        ggml_tensor * k = conv1d_f32(ctx, require_source_tensor(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_k.weight"), y, 1, 0, 1);
        k = ggml_add(ctx, k, repeat_like(ctx, require_source_tensor(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_k.bias"), k));
        ggml_set_name(k, "text_encoder_attn0_k"); ggml_set_output(k); ggml_build_forward_expand(gf, k);
        ggml_tensor * v = conv1d_f32(ctx, require_source_tensor(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_v.weight"), y, 1, 0, 1);
        v = ggml_add(ctx, v, repeat_like(ctx, require_source_tensor(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_v.bias"), v));
        ggml_set_name(v, "text_encoder_attn0_v"); ggml_set_output(v); ggml_build_forward_expand(gf, v);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_new text encoder failed");
        }
        if (!ggml_gallocr_reserve(allocr, gf)) {
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_reserve text encoder failed");
        }
        ggml_gallocr_alloc_graph(allocr, gf);
        std::vector<float> raw = pack_time_channel_for_ggml(x, L, C);
        ggml_backend_tensor_set(in, raw.data(), 0, raw.size()*sizeof(float));
        supertonic_graph_compute(model, gf);
        ggml_trace.push_back({"text_encoder_embed", {L, C}, x});
        for (int i = 0; i < 6; ++i) {
            const std::string name = "text_encoder_convnext" + std::to_string(i);
            ggml_trace.push_back({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, name.c_str()))});
        }
        ggml_trace.push_back({"text_encoder_attn0_q", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "text_encoder_attn0_q"))});
        ggml_trace.push_back({"text_encoder_attn0_k", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "text_encoder_attn0_k"))});
        ggml_trace.push_back({"text_encoder_attn0_v", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "text_encoder_attn0_v"))});
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        auto vit = model.voices.find(model.hparams.default_voice);
        if (vit != model.voices.end()) {
            std::vector<float> style_ttl((size_t)ggml_nelements(vit->second.ttl));
            ggml_backend_tensor_get(vit->second.ttl, style_ttl.data(), 0, ggml_nbytes(vit->second.ttl));
            std::vector<float> scalar_final, ggml_final;
            std::string nested_error;
            if (supertonic_text_encoder_forward_cpu(model, text_ids, text_len, style_ttl.data(), scalar_final, &nested_error) &&
                supertonic_text_encoder_forward_ggml(model, text_ids, text_len, style_ttl.data(), ggml_final, &nested_error) &&
                scalar_final.size() == (size_t)C*L && ggml_final.size() == (size_t)C*L) {
                std::vector<float> scalar_lc((size_t)L*C), ggml_lc((size_t)L*C);
                for (int c = 0; c < C; ++c) {
                    for (int t = 0; t < L; ++t) {
                        scalar_lc[(size_t)t*C+c] = scalar_final[(size_t)c*L+t];
                        ggml_lc[(size_t)t*C+c] = ggml_final[(size_t)c*L+t];
                    }
                }
                scalar_trace.push_back({"text_encoder_final", {L, C}, scalar_lc});
                ggml_trace.push_back({"text_encoder_final", {L, C}, ggml_lc});
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
