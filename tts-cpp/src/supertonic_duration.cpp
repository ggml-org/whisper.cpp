#include "supertonic_internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
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

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f)); }

void linear1x1(const std::vector<float> & x, int L, int IC,
               const f32_tensor & w, const f32_tensor * b,
               int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
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

ggml_tensor * repeat_like(ggml_context * ctx, ggml_tensor * v, ggml_tensor * like) {
    if (ggml_n_dims(v) == 1 && ggml_n_dims(like) >= 2) {
        if (like->ne[0] == v->ne[0]) v = ggml_reshape_2d(ctx, v, v->ne[0], 1);
        else if (like->ne[1] == v->ne[0]) v = ggml_reshape_2d(ctx, v, 1, v->ne[0]);
    }
    if (!ggml_can_repeat(v, like)) {
        throw std::runtime_error("cannot repeat tensor in duration graph");
    }
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

ggml_tensor * depthwise_same_ggml(ggml_context * ctx,
                                  ggml_tensor * x,
                                  ggml_tensor * w,
                                  ggml_tensor * b,
                                  int dilation) {
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

ggml_tensor * layer_norm_ggml(ggml_context * ctx, ggml_tensor * x, ggml_tensor * g, ggml_tensor * b) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    xt = ggml_norm(ctx, xt, 1e-6f);
    xt = ggml_mul(ctx, xt, repeat_like(ctx, g, xt));
    xt = ggml_add(ctx, xt, repeat_like(ctx, b, xt));
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}

ggml_tensor * duration_convnext_ggml(ggml_context * ctx,
                                     const supertonic_model & model,
                                     const std::string & p,
                                     ggml_tensor * x) {
    ggml_tensor * residual = x;
    ggml_tensor * y = depthwise_same_ggml(ctx, x,
        require_source_tensor(model, p + ".dwconv.weight"),
        require_source_tensor(model, p + ".dwconv.bias"),
        1);
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
    const int L = (int)t->ne[0];
    const int C = (int)t->ne[1];
    std::vector<float> raw((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
    std::vector<float> out((size_t)L*C);
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < L; ++i) out[(size_t)i*C+c] = raw[(size_t)c*L+i];
    }
    return out;
}

std::vector<float> pack_time_channel_for_ggml(const std::vector<float> & x, int L, int C) {
    std::vector<float> out((size_t)L*C);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) out[(size_t)c*L+t] = x[(size_t)t*C+c];
    }
    return out;
}

void push_trace(std::vector<supertonic_trace_tensor> & trace,
                const std::string & name,
                int L,
                int C,
                const std::vector<float> & data) {
    trace.push_back({name, {L, C}, data});
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
                src_t = std::max(0, std::min(L - 1, src_t)); // replicate pad
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
                    std::vector<float> & x, int L, int C, int dilation = 1) {
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
    std::vector<float> y;
    depthwise_conv1d_same(x, L, C, dw_w, dw_b, (int) dw_w.ne[0], dilation, y);
    layer_norm_channel(y, L, C, ln_g, ln_b);

    std::vector<float> z;
    int hidden = (int) pw1_w.ne[2];
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

void self_attention(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    const int H = 2;
    const int D = C / H;
    const float scale = 1.0f / std::sqrt((float) D);
    const std::string p = "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers." + std::to_string(idx);

    f32_tensor q_w = read_f32(m, p + ".conv_q.weight");
    f32_tensor q_b = read_f32(m, p + ".conv_q.bias");
    f32_tensor k_w = read_f32(m, p + ".conv_k.weight");
    f32_tensor k_b = read_f32(m, p + ".conv_k.bias");
    f32_tensor v_w = read_f32(m, p + ".conv_v.weight");
    f32_tensor v_b = read_f32(m, p + ".conv_v.bias");
    f32_tensor o_w = read_f32(m, p + ".conv_o.weight");
    f32_tensor o_b = read_f32(m, p + ".conv_o.bias");
    f32_tensor rel_k = read_f32(m, p + ".emb_rel_k"); // [1, 9, D]
    f32_tensor rel_v = read_f32(m, p + ".emb_rel_v");

    std::vector<float> q, k, v;
    linear1x1(x, L, C, q_w, &q_b, C, q);
    linear1x1(x, L, C, k_w, &k_b, C, k);
    linear1x1(x, L, C, v_w, &v_b, C, v);

    std::vector<float> out((size_t) L * C, 0.0f);
    std::vector<float> scores(L);
    std::vector<float> probs(L);
    const int half_window = 4;

    for (int h = 0; h < H; ++h) {
        for (int qi = 0; qi < L; ++qi) {
            float max_score = -INFINITY;
            for (int kj = 0; kj < L; ++kj) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s += q[(size_t) qi * C + h * D + d] * scale *
                         k[(size_t) kj * C + h * D + d];
                }
                int rel_pos = kj - qi;
                if (rel_pos >= -half_window && rel_pos <= half_window) {
                    int ridx = rel_pos + half_window;
                    for (int d = 0; d < D; ++d) {
                        s += q[(size_t) qi * C + h * D + d] * scale *
                             rel_k.data[(size_t) ridx * D + d];
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

void ffn_block(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    const std::string p = "duration:tts.dp.sentence_encoder.attn_encoder.ffn_layers." + std::to_string(idx);
    f32_tensor w1 = read_f32(m, p + ".conv_1.weight");
    f32_tensor b1 = read_f32(m, p + ".conv_1.bias");
    f32_tensor w2 = read_f32(m, p + ".conv_2.weight");
    f32_tensor b2 = read_f32(m, p + ".conv_2.bias");
    std::vector<float> y;
    linear1x1(x, L, C, w1, &b1, (int) w1.ne[2], y);
    for (float & v : y) v = relu(v);
    linear1x1(y, L, (int) w1.ne[2], w2, &b2, C, x);
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

} // namespace

bool supertonic_duration_forward_cpu(const supertonic_model & model,
                                     const int64_t * text_ids,
                                     int text_len,
                                     const float * style_dp,
                                     float & duration_out,
                                     std::string * error) {
    try {
        const int C = 64;
        const int L = text_len + 1;
        f32_tensor emb = read_f32(model, "duration:tts.dp.sentence_encoder.text_embedder.char_embedder.weight");
        f32_tensor sentence = read_f32(model, "duration:tts.dp.sentence_encoder.sentence_token");

        std::vector<float> x((size_t) L * C, 0.0f);
        for (int c = 0; c < C; ++c) x[c] = sentence.data[c];
        for (int t = 0; t < text_len; ++t) {
            int64_t id = text_ids[t];
            if (id < 0 || id >= emb.ne[1]) throw std::runtime_error("text id out of range");
            for (int c = 0; c < C; ++c) {
                x[(size_t) (t + 1) * C + c] = emb.data[(size_t) id * C + c];
            }
        }

        for (int i = 0; i < 6; ++i) {
            const std::string p = "duration:tts.dp.sentence_encoder.convnext.convnext." + std::to_string(i);
            convnext_block(model, p, x, L, C);
        }
        std::vector<float> convnext_out = x;

        for (int i = 0; i < 2; ++i) {
            std::vector<float> residual = x;
            self_attention(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.bias"));

            residual = x;
            ffn_block(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.bias"));
        }

        for (size_t i = 0; i < x.size(); ++i) x[i] += convnext_out[i];

        std::vector<float> sentence_repr(C);
        for (int c = 0; c < C; ++c) sentence_repr[c] = x[c];
        std::vector<float> projected;
        f32_tensor proj_w = read_f32(model, "duration:tts.dp.sentence_encoder.proj_out.net.weight");
        linear1x1(sentence_repr, 1, C, proj_w, nullptr, C, projected);

        std::vector<float> combined(192);
        for (int c = 0; c < C; ++c) combined[c] = projected[c];
        for (int i = 0; i < 128; ++i) combined[C + i] = style_dp[i];

        std::vector<float> h;
        dense(combined,
              read_f32(model, "duration:tts.dp.predictor.layers.0.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.0.bias"),
              192, 128, h);
        float prelu = read_f32(model, "duration:tts.dp.predictor.activation.weight").data[0];
        for (float & v : h) if (v < 0.0f) v *= prelu;
        std::vector<float> out;
        dense(h,
              read_f32(model, "duration:tts.dp.predictor.layers.1.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.1.bias"),
              128, 1, out);
        duration_out = std::exp(out[0]);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

static bool duration_sentence_proj_ggml_impl(const supertonic_model & model,
                                             const int64_t * text_ids,
                                             int text_len,
                                             std::vector<supertonic_trace_tensor> * scalar_trace,
                                             std::vector<supertonic_trace_tensor> * ggml_trace,
                                             std::string * error,
                                             bool include_scalar_trace,
                                             bool include_ggml_trace,
                                             std::vector<float> * sentence_proj_out) {
    try {
        if (scalar_trace) scalar_trace->clear();
        if (ggml_trace) ggml_trace->clear();
        const int C = 64;
        const int L = text_len + 1;
#define PUSH_DURATION_GGML(...) do { if (include_ggml_trace && ggml_trace) ggml_trace->push_back(supertonic_trace_tensor __VA_ARGS__); } while (0)
        f32_tensor emb = read_f32(model, "duration:tts.dp.sentence_encoder.text_embedder.char_embedder.weight");
        f32_tensor sentence = read_f32(model, "duration:tts.dp.sentence_encoder.sentence_token");

        std::vector<float> x((size_t)L*C, 0.0f);
        for (int c = 0; c < C; ++c) x[c] = sentence.data[c];
        for (int t = 0; t < text_len; ++t) {
            int64_t id = text_ids[t];
            if (id < 0 || id >= emb.ne[1]) throw std::runtime_error("text id out of range");
            for (int c = 0; c < C; ++c) x[(size_t)(t + 1)*C + c] = emb.data[(size_t)id*C + c];
        }
        if (include_scalar_trace && scalar_trace) {
        push_trace(*scalar_trace, "duration_embed", L, C, x);

        std::vector<float> cur = x;
        for (int i = 0; i < 6; ++i) {
            const std::string p = "duration:tts.dp.sentence_encoder.convnext.convnext." + std::to_string(i);
            convnext_block(model, p, cur, L, C);
            push_trace(*scalar_trace, "duration_convnext" + std::to_string(i), L, C, cur);
        }
        std::vector<float> q0, k0, v0;
        f32_tensor q0_w = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_q.weight");
        f32_tensor q0_b = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_q.bias");
        f32_tensor k0_w = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_k.weight");
        f32_tensor k0_b = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_k.bias");
        f32_tensor v0_w = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_v.weight");
        f32_tensor v0_b = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_v.bias");
        linear1x1(cur, L, C, q0_w, &q0_b, C, q0);
        linear1x1(cur, L, C, k0_w, &k0_b, C, k0);
        linear1x1(cur, L, C, v0_w, &v0_b, C, v0);
        push_trace(*scalar_trace, "duration_attn0_q", L, C, q0);
        push_trace(*scalar_trace, "duration_attn0_k", L, C, k0);
        push_trace(*scalar_trace, "duration_attn0_v", L, C, v0);
        std::vector<float> attn0_out = cur;
        self_attention(model, 0, attn0_out, L, C);
        push_trace(*scalar_trace, "duration_attn0_out", L, C, attn0_out);
        for (size_t i = 0; i < attn0_out.size(); ++i) attn0_out[i] += cur[i];
        push_trace(*scalar_trace, "duration_attn0_residual", L, C, attn0_out);
        layer_norm_channel(
            attn0_out, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.0.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.0.norm.bias"));
        push_trace(*scalar_trace, "duration_attn0_norm", L, C, attn0_out);
        std::vector<float> ffn0 = attn0_out;
        ffn_block(model, 0, ffn0, L, C);
        push_trace(*scalar_trace, "duration_ffn0_out", L, C, ffn0);
        for (size_t i = 0; i < ffn0.size(); ++i) ffn0[i] += attn0_out[i];
        push_trace(*scalar_trace, "duration_ffn0_residual", L, C, ffn0);
        layer_norm_channel(
            ffn0, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.0.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.0.norm.bias"));
        push_trace(*scalar_trace, "duration_ffn0_norm", L, C, ffn0);

        std::vector<float> attn1_out = ffn0;
        self_attention(model, 1, attn1_out, L, C);
        push_trace(*scalar_trace, "duration_attn1_out", L, C, attn1_out);
        for (size_t i = 0; i < attn1_out.size(); ++i) attn1_out[i] += ffn0[i];
        push_trace(*scalar_trace, "duration_attn1_residual", L, C, attn1_out);
        layer_norm_channel(
            attn1_out, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.1.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.1.norm.bias"));
        push_trace(*scalar_trace, "duration_attn1_norm", L, C, attn1_out);
        std::vector<float> ffn1 = attn1_out;
        ffn_block(model, 1, ffn1, L, C);
        push_trace(*scalar_trace, "duration_ffn1_out", L, C, ffn1);
        for (size_t i = 0; i < ffn1.size(); ++i) ffn1[i] += attn1_out[i];
        push_trace(*scalar_trace, "duration_ffn1_residual", L, C, ffn1);
        layer_norm_channel(
            ffn1, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.1.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.1.norm.bias"));
        push_trace(*scalar_trace, "duration_ffn1_norm", L, C, ffn1);
        for (size_t i = 0; i < ffn1.size(); ++i) ffn1[i] += cur[i];
        push_trace(*scalar_trace, "duration_encoder_out", L, C, ffn1);
        std::vector<float> sentence_repr(C);
        for (int c = 0; c < C; ++c) sentence_repr[c] = ffn1[c];
        std::vector<float> projected;
        linear1x1(sentence_repr, 1, C,
                  read_f32(model, "duration:tts.dp.sentence_encoder.proj_out.net.weight"),
                  nullptr, C, projected);
        push_trace(*scalar_trace, "duration_sentence_proj", 1, C, projected);
        std::vector<float> combined(192);
        for (int c = 0; c < C; ++c) combined[c] = projected[c];
        for (int i = 0; i < 128; ++i) combined[C + i] = 0.0f;
        std::vector<float> h;
        dense(combined,
              read_f32(model, "duration:tts.dp.predictor.layers.0.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.0.bias"),
              192, 128, h);
        push_trace(*scalar_trace, "duration_pred0_no_style", 1, 128, h);
        }

        constexpr int MAX_NODES = 512;
        static size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                                 ggml_graph_overhead_custom(MAX_NODES, false);
        thread_local std::vector<uint8_t> buf(buf_size);
        ggml_init_params gp = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

        ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
        ggml_set_name(in, "duration_embed"); ggml_set_input(in);
        ggml_tensor * y = in;
        for (int i = 0; i < 6; ++i) {
            const std::string p = "duration:tts.dp.sentence_encoder.convnext.convnext." + std::to_string(i);
            y = duration_convnext_ggml(ctx, model, p, y);
            const std::string name = "duration_convnext" + std::to_string(i);
            ggml_set_name(y, name.c_str()); ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        ggml_tensor * q = conv1d_f32(ctx, require_source_tensor(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_q.weight"), y, 1, 0, 1);
        q = ggml_add(ctx, q, repeat_like(ctx, require_source_tensor(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_q.bias"), q));
        ggml_set_name(q, "duration_attn0_q"); ggml_set_output(q); ggml_build_forward_expand(gf, q);
        ggml_tensor * k = conv1d_f32(ctx, require_source_tensor(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_k.weight"), y, 1, 0, 1);
        k = ggml_add(ctx, k, repeat_like(ctx, require_source_tensor(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_k.bias"), k));
        ggml_set_name(k, "duration_attn0_k"); ggml_set_output(k); ggml_build_forward_expand(gf, k);
        ggml_tensor * v = conv1d_f32(ctx, require_source_tensor(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_v.weight"), y, 1, 0, 1);
        v = ggml_add(ctx, v, repeat_like(ctx, require_source_tensor(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_v.bias"), v));
        ggml_set_name(v, "duration_attn0_v"); ggml_set_output(v); ggml_build_forward_expand(gf, v);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_new duration failed");
        }
        if (!ggml_gallocr_reserve(allocr, gf)) {
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            throw std::runtime_error("ggml_gallocr_reserve duration failed");
        }
        ggml_gallocr_alloc_graph(allocr, gf);
        std::vector<float> x_raw = pack_time_channel_for_ggml(x, L, C);
        ggml_backend_tensor_set(in, x_raw.data(), 0, x_raw.size()*sizeof(float));
        supertonic_graph_compute(model, gf);

        PUSH_DURATION_GGML({"duration_embed", {L, C}, x});
        for (int i = 0; i < 6; ++i) {
            const std::string name = "duration_convnext" + std::to_string(i);
            PUSH_DURATION_GGML({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, name.c_str()))});
        }
        PUSH_DURATION_GGML({"duration_attn0_q", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "duration_attn0_q"))});
        PUSH_DURATION_GGML({"duration_attn0_k", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "duration_attn0_k"))});
        PUSH_DURATION_GGML({"duration_attn0_v", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "duration_attn0_v"))});

        // Relative-position attention is not plain flash attention because it
        // adds learned relative key scores and relative value outputs. Keep it
        // in a scalar continuation for now while the surrounding Conv/Linear
        // graph boundaries are validated.
        std::vector<float> q0_g = tensor_to_time_channel(ggml_graph_get_tensor(gf, "duration_attn0_q"));
        std::vector<float> k0_g = tensor_to_time_channel(ggml_graph_get_tensor(gf, "duration_attn0_k"));
        std::vector<float> v0_g = tensor_to_time_channel(ggml_graph_get_tensor(gf, "duration_attn0_v"));
        const int H = 2, D = C / H, half_window = 4;
        const float scale = 1.0f / std::sqrt((float)D);
        f32_tensor rel_k = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.emb_rel_k");
        f32_tensor rel_v = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.emb_rel_v");
        std::vector<float> out((size_t)L*C, 0.0f), scores(L), probs(L);
        for (int h = 0; h < H; ++h) {
            for (int qi = 0; qi < L; ++qi) {
                float max_score = -INFINITY;
                for (int kj = 0; kj < L; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < D; ++d) s += q0_g[(size_t)qi*C+h*D+d] * scale * k0_g[(size_t)kj*C+h*D+d];
                    int rel_pos = kj - qi;
                    if (rel_pos >= -half_window && rel_pos <= half_window) {
                        int ridx = rel_pos + half_window;
                        for (int d = 0; d < D; ++d) s += q0_g[(size_t)qi*C+h*D+d] * scale * rel_k.data[(size_t)ridx*D+d];
                    }
                    scores[kj] = s;
                    max_score = std::max(max_score, s);
                }
                float denom = 0.0f;
                for (int kj = 0; kj < L; ++kj) {
                    probs[kj] = std::exp(scores[kj] - max_score);
                    denom += probs[kj];
                }
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < L; ++kj) {
                        const float p = probs[kj] / denom;
                        sum += p * v0_g[(size_t)kj*C+h*D+d];
                        int rel_pos = kj - qi;
                        if (rel_pos >= -half_window && rel_pos <= half_window) {
                            int ridx = rel_pos + half_window;
                            sum += p * rel_v.data[(size_t)ridx*D+d];
                        }
                    }
                    out[(size_t)qi*C+h*D+d] = sum;
                }
            }
        }
        f32_tensor o_w = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_o.weight");
        f32_tensor o_b = read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_o.bias");
        std::vector<float> proj;
        linear1x1(out, L, C, o_w, &o_b, C, proj);
        PUSH_DURATION_GGML({"duration_attn0_out", {L, C}, proj});
        std::vector<float> conv_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "duration_convnext5"));
        std::vector<float> attn_res = proj;
        for (size_t i = 0; i < attn_res.size(); ++i) attn_res[i] += conv_out[i];
        PUSH_DURATION_GGML({"duration_attn0_residual", {L, C}, attn_res});
        layer_norm_channel(
            attn_res, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.0.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.0.norm.bias"));
        PUSH_DURATION_GGML({"duration_attn0_norm", {L, C}, attn_res});
        std::vector<float> ffn0_g = attn_res;
        ffn_block(model, 0, ffn0_g, L, C);
        PUSH_DURATION_GGML({"duration_ffn0_out", {L, C}, ffn0_g});
        for (size_t i = 0; i < ffn0_g.size(); ++i) ffn0_g[i] += attn_res[i];
        PUSH_DURATION_GGML({"duration_ffn0_residual", {L, C}, ffn0_g});
        layer_norm_channel(
            ffn0_g, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.0.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.0.norm.bias"));
        PUSH_DURATION_GGML({"duration_ffn0_norm", {L, C}, ffn0_g});

        std::vector<float> attn1_g = ffn0_g;
        self_attention(model, 1, attn1_g, L, C);
        PUSH_DURATION_GGML({"duration_attn1_out", {L, C}, attn1_g});
        for (size_t i = 0; i < attn1_g.size(); ++i) attn1_g[i] += ffn0_g[i];
        PUSH_DURATION_GGML({"duration_attn1_residual", {L, C}, attn1_g});
        layer_norm_channel(
            attn1_g, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.1.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.1.norm.bias"));
        PUSH_DURATION_GGML({"duration_attn1_norm", {L, C}, attn1_g});
        std::vector<float> ffn1_g = attn1_g;
        ffn_block(model, 1, ffn1_g, L, C);
        PUSH_DURATION_GGML({"duration_ffn1_out", {L, C}, ffn1_g});
        for (size_t i = 0; i < ffn1_g.size(); ++i) ffn1_g[i] += attn1_g[i];
        PUSH_DURATION_GGML({"duration_ffn1_residual", {L, C}, ffn1_g});
        layer_norm_channel(
            ffn1_g, L, C,
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.1.norm.weight"),
            read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2.1.norm.bias"));
        PUSH_DURATION_GGML({"duration_ffn1_norm", {L, C}, ffn1_g});
        for (size_t i = 0; i < ffn1_g.size(); ++i) ffn1_g[i] += conv_out[i];
        PUSH_DURATION_GGML({"duration_encoder_out", {L, C}, ffn1_g});
        std::vector<float> sentence_repr_g(C);
        for (int c = 0; c < C; ++c) sentence_repr_g[c] = ffn1_g[c];
        std::vector<float> projected_g;
        linear1x1(sentence_repr_g, 1, C,
                  read_f32(model, "duration:tts.dp.sentence_encoder.proj_out.net.weight"),
                  nullptr, C, projected_g);
        if (sentence_proj_out) *sentence_proj_out = projected_g;
        PUSH_DURATION_GGML({"duration_sentence_proj", {1, C}, projected_g});
        std::vector<float> combined_g(192);
        for (int c = 0; c < C; ++c) combined_g[c] = projected_g[c];
        for (int i = 0; i < 128; ++i) combined_g[C + i] = 0.0f;
        std::vector<float> h_g;
        dense(combined_g,
              read_f32(model, "duration:tts.dp.predictor.layers.0.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.0.bias"),
              192, 128, h_g);
        PUSH_DURATION_GGML({"duration_pred0_no_style", {1, 128}, h_g});
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        if (error) error->clear();
#undef PUSH_DURATION_GGML
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_duration_trace_ggml(const supertonic_model & model,
                                    const int64_t * text_ids,
                                    int text_len,
                                    std::vector<supertonic_trace_tensor> & scalar_trace,
                                    std::vector<supertonic_trace_tensor> & ggml_trace,
                                    std::string * error,
                                    bool include_scalar_trace,
                                    bool include_ggml_trace,
                                    std::vector<float> * sentence_proj_out) {
    return duration_sentence_proj_ggml_impl(model, text_ids, text_len, &scalar_trace, &ggml_trace,
                                           error, include_scalar_trace, include_ggml_trace,
                                           sentence_proj_out);
}

bool supertonic_duration_forward_ggml(const supertonic_model & model,
                                      const int64_t * text_ids,
                                      int text_len,
                                      const float * style_dp,
                                      float & duration_out,
                                      std::string * error) {
    try {
        std::vector<supertonic_trace_tensor> scalar;
        std::vector<supertonic_trace_tensor> ggml;
        std::vector<float> projected;
        if (!duration_sentence_proj_ggml_impl(model, text_ids, text_len, nullptr, nullptr, error,
                                              false, false, &projected)) return false;
        if (projected.size() != 64) throw std::runtime_error("missing duration sentence projection");
        std::vector<float> combined(192);
        for (int c = 0; c < 64; ++c) combined[c] = projected[c];
        for (int i = 0; i < 128; ++i) combined[64 + i] = style_dp[i];
        std::vector<float> h;
        dense(combined,
              read_f32(model, "duration:tts.dp.predictor.layers.0.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.0.bias"),
              192, 128, h);
        float prelu = read_f32(model, "duration:tts.dp.predictor.activation.weight").data[0];
        for (float & v : h) if (v < 0.0f) v *= prelu;
        std::vector<float> out;
        dense(h,
              read_f32(model, "duration:tts.dp.predictor.layers.1.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.1.bias"),
              128, 1, out);
        duration_out = std::exp(out[0]);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
