#include "supertonic_text_encoder_backward.h"

#include <cmath>
#include <cstddef>

namespace tts_cpp {
namespace supertonic_grad {

namespace {

// --- dense per-time-step linear --------------------------------------------

// One output channel of one time step: dot of the input row with column `oc` of
// the ONNX-layout weight (w[ic * out_dim + oc]).
double linear_output(const double * x_row, const std::vector<double> & w,
                     int oc, int in_dim, int out_dim) {
    double acc = 0.0;
    for (int ic = 0; ic < in_dim; ++ic) {
        acc += x_row[ic] * w[(std::size_t) ic * out_dim + oc];
    }
    return acc;
}

// One time step of the forward: fills `out_row` (out_dim values).
void dense_time_forward_row(double * out_row, const double * x_row,
                            const LinearWeights & weights) {
    for (int oc = 0; oc < weights.out_dim; ++oc) {
        out_row[oc] = weights.b[oc] + linear_output(x_row, weights.w, oc,
                                                     weights.in_dim, weights.out_dim);
    }
}

// One input channel gradient of one time step: dot of the upstream-grad row with
// row `ic` of the weight (w[ic * out_dim + oc] over oc).
double linear_input_grad(const double * d_y_row, const std::vector<double> & w,
                         int ic, int in_dim, int out_dim) {
    (void) in_dim;
    double acc = 0.0;
    for (int oc = 0; oc < out_dim; ++oc) {
        acc += d_y_row[oc] * w[(std::size_t) ic * out_dim + oc];
    }
    return acc;
}

// One time step of the input-gradient: fills `d_x_row` (in_dim values).
void dense_time_backward_row(double * d_x_row, const double * d_y_row,
                             const LinearWeights & weights) {
    for (int ic = 0; ic < weights.in_dim; ++ic) {
        d_x_row[ic] = linear_input_grad(d_y_row, weights.w, ic,
                                        weights.in_dim, weights.out_dim);
    }
}

// --- channel-wise layer norm helpers ---------------------------------------

double row_mean(const double * row, int C) {
    double acc = 0.0;
    for (int c = 0; c < C; ++c) {
        acc += row[c];
    }
    return acc / (double) C;
}

double row_variance(const double * row, double mean, int C) {
    double acc = 0.0;
    for (int c = 0; c < C; ++c) {
        const double d = row[c] - mean;
        acc += d * d;
    }
    return acc / (double) C;
}

// Normalized + affine output of one row: y[c] = (x[c] - mean) * inv * g[c] + b[c].
void layer_norm_apply_row(double * out_row, const double * x_row, double mean, double inv,
                          const std::vector<double> & gamma, const std::vector<double> & beta,
                          int C) {
    for (int c = 0; c < C; ++c) {
        out_row[c] = (x_row[c] - mean) * inv * gamma[c] + beta[c];
    }
}

// Centered, scaled row: xhat[c] = (x[c] - mean) * inv.
std::vector<double> centered_scaled_row(const double * x_row, double mean, double inv, int C) {
    std::vector<double> xhat((std::size_t) C);
    for (int c = 0; c < C; ++c) {
        xhat[c] = (x_row[c] - mean) * inv;
    }
    return xhat;
}

// d_xhat[c] = d_y[c] * gamma[c].
std::vector<double> scale_by_gamma(const double * d_y_row, const std::vector<double> & gamma, int C) {
    std::vector<double> out((std::size_t) C);
    for (int c = 0; c < C; ++c) {
        out[c] = d_y_row[c] * gamma[c];
    }
    return out;
}

double mean_of_product(const std::vector<double> & a, const std::vector<double> & b, int C) {
    double acc = 0.0;
    for (int c = 0; c < C; ++c) {
        acc += a[c] * b[c];
    }
    return acc / (double) C;
}

// Final layer-norm backward combine for one row:
//   d_x[c] = inv * (d_xhat[c] - mean(d_xhat) - xhat[c] * mean(d_xhat * xhat)).
void ln_backward_combine_row(double * d_x_row, const std::vector<double> & d_xhat,
                             const std::vector<double> & xhat,
                             double inv, double mean_dxhat, double mean_dxhat_xhat, int C) {
    for (int c = 0; c < C; ++c) {
        d_x_row[c] = inv * (d_xhat[c] - mean_dxhat - xhat[c] * mean_dxhat_xhat);
    }
}

void ln_backward_row(double * d_x_row, const double * x_row,
                     const std::vector<double> & gamma, const double * d_y_row,
                     int C, double eps) {
    const double mean = row_mean(x_row, C);
    const double inv  = 1.0 / std::sqrt(row_variance(x_row, mean, C) + eps);
    const std::vector<double> xhat   = centered_scaled_row(x_row, mean, inv, C);
    const std::vector<double> d_xhat = scale_by_gamma(d_y_row, gamma, C);
    const double mean_dxhat      = row_mean(d_xhat.data(), C);
    const double mean_dxhat_xhat = mean_of_product(d_xhat, xhat, C);
    ln_backward_combine_row(d_x_row, d_xhat, xhat, inv, mean_dxhat, mean_dxhat_xhat, C);
}

}  // namespace

std::vector<double> dense_time_forward(const std::vector<double> & x, int L,
                                       const LinearWeights & weights) {
    std::vector<double> y((std::size_t) L * weights.out_dim);
    for (int t = 0; t < L; ++t) {
        dense_time_forward_row(&y[(std::size_t) t * weights.out_dim],
                               &x[(std::size_t) t * weights.in_dim], weights);
    }
    return y;
}

std::vector<double> dense_time_backward_input(const std::vector<double> & d_y, int L,
                                              const LinearWeights & weights) {
    std::vector<double> d_x((std::size_t) L * weights.in_dim);
    for (int t = 0; t < L; ++t) {
        dense_time_backward_row(&d_x[(std::size_t) t * weights.in_dim],
                                &d_y[(std::size_t) t * weights.out_dim], weights);
    }
    return d_x;
}

std::vector<double> layer_norm_channel_forward(const std::vector<double> & x_lc, int L, int C,
                                               const std::vector<double> & gamma,
                                               const std::vector<double> & beta,
                                               double eps) {
    std::vector<double> y((std::size_t) L * C);
    for (int t = 0; t < L; ++t) {
        const double * x_row = &x_lc[(std::size_t) t * C];
        const double mean = row_mean(x_row, C);
        const double inv  = 1.0 / std::sqrt(row_variance(x_row, mean, C) + eps);
        layer_norm_apply_row(&y[(std::size_t) t * C], x_row, mean, inv, gamma, beta, C);
    }
    return y;
}

std::vector<double> layer_norm_channel_backward(const std::vector<double> & x_lc, int L, int C,
                                                const std::vector<double> & gamma,
                                                const std::vector<double> & d_y,
                                                double eps) {
    std::vector<double> d_x((std::size_t) L * C);
    for (int t = 0; t < L; ++t) {
        ln_backward_row(&d_x[(std::size_t) t * C], &x_lc[(std::size_t) t * C],
                        gamma, &d_y[(std::size_t) t * C], C, eps);
    }
    return d_x;
}

namespace {

// --- speech-prompted attention indexing ------------------------------------
//
// Channels split into `heads` contiguous blocks of head_dim = C / heads.
//   q / merged : time-major [L, C],  element (t, head h, d) at t*C + h*head_dim + d
//   v          : time-major [Lctx, C], element (j, h, d) at j*C + h*head_dim + d
//   K (tanh_k) : [heads, head_dim, Lctx], element (h, d, j) at (h*head_dim+d)*Lctx + j
//   attn       : [heads, L, Lctx], element (h, t, j) at (h*L + t)*Lctx + j

int head_dim(const SpeechAttentionDims & dims) { return dims.C / dims.heads; }

// --- forward helpers --------------------------------------------------------

// Pre-softmax score for one (head, query t, key j): scale * <q_head, K_head_j>.
double attention_score(const double * q, const std::vector<double> & K,
                       const SpeechAttentionDims & dims, int h, int t, int j) {
    const int hd = head_dim(dims);
    double acc = 0.0;
    for (int d = 0; d < hd; ++d) {
        const int ch = h * hd + d;
        acc += q[(std::size_t) t * dims.C + ch] * K[(std::size_t) ch * dims.Lctx + j];
    }
    return acc * dims.scale;
}

void attention_scores_row(double * scores, const double * q, const std::vector<double> & K,
                          const SpeechAttentionDims & dims, int h, int t) {
    for (int j = 0; j < dims.Lctx; ++j) {
        scores[j] = attention_score(q, K, dims, h, t, j);
    }
}

double max_value(const double * v, int n) {
    double m = v[0];
    for (int i = 1; i < n; ++i) {
        if (v[i] > m) m = v[i];
    }
    return m;
}

// Replaces v[i] with exp(v[i] - max_v) and returns their sum.
double exp_shift_sum(double * v, int n, double max_v) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        v[i] = std::exp(v[i] - max_v);
        s += v[i];
    }
    return s;
}

void scale_inplace(double * v, int n, double f) {
    for (int i = 0; i < n; ++i) {
        v[i] *= f;
    }
}

void softmax_inplace(double * v, int n) {
    const double s = exp_shift_sum(v, n, max_value(v, n));
    scale_inplace(v, n, 1.0 / s);
}

// One head's attention weights: score each query row then softmax over keys.
void attention_head_forward(double * attn, const double * q, const std::vector<double> & K,
                            const SpeechAttentionDims & dims, int h) {
    for (int t = 0; t < dims.L; ++t) {
        double * row = &attn[((std::size_t) h * dims.L + t) * dims.Lctx];
        attention_scores_row(row, q, K, dims, h, t);
        softmax_inplace(row, dims.Lctx);
    }
}

void compute_attention(std::vector<double> & attn, const double * q,
                       const std::vector<double> & K, const SpeechAttentionDims & dims) {
    for (int h = 0; h < dims.heads; ++h) {
        attention_head_forward(attn.data(), q, K, dims, h);
    }
}

// Context value for one (head, t, d): sum_j attn[h,t,j] * v[j, head h, d].
double merged_element(const std::vector<double> & attn, const std::vector<double> & v,
                      const SpeechAttentionDims & dims, int h, int t, int d) {
    const int ch = h * head_dim(dims) + d;
    const double * arow = &attn[((std::size_t) h * dims.L + t) * dims.Lctx];
    double acc = 0.0;
    for (int j = 0; j < dims.Lctx; ++j) {
        acc += arow[j] * v[(std::size_t) j * dims.C + ch];
    }
    return acc;
}

void merged_row(double * merged, const std::vector<double> & attn, const std::vector<double> & v,
                const SpeechAttentionDims & dims, int h, int t) {
    const int hd = head_dim(dims);
    for (int d = 0; d < hd; ++d) {
        const int ch = h * hd + d;
        merged[(std::size_t) t * dims.C + ch] = merged_element(attn, v, dims, h, t, d);
    }
}

void merged_head(double * merged, const std::vector<double> & attn, const std::vector<double> & v,
                 const SpeechAttentionDims & dims, int h) {
    for (int t = 0; t < dims.L; ++t) {
        merged_row(merged, attn, v, dims, h, t);
    }
}

void compute_merged(std::vector<double> & merged, const std::vector<double> & attn,
                    const std::vector<double> & v, const SpeechAttentionDims & dims) {
    for (int h = 0; h < dims.heads; ++h) {
        merged_head(merged.data(), attn, v, dims, h);
    }
}

// --- backward helpers -------------------------------------------------------

// d_attn[h,t,j] = sum_d d_merged[t, head h, d] * v[j, head h, d].
double attn_grad_element(const std::vector<double> & d_merged, const std::vector<double> & v,
                         const SpeechAttentionDims & dims, int h, int t, int j) {
    const int hd = head_dim(dims);
    double acc = 0.0;
    for (int d = 0; d < hd; ++d) {
        const int ch = h * hd + d;
        acc += d_merged[(std::size_t) t * dims.C + ch] * v[(std::size_t) j * dims.C + ch];
    }
    return acc;
}

void attn_grad_row(double * d_attn_row, const std::vector<double> & d_merged,
                   const std::vector<double> & v, const SpeechAttentionDims & dims, int h, int t) {
    for (int j = 0; j < dims.Lctx; ++j) {
        d_attn_row[j] = attn_grad_element(d_merged, v, dims, h, t, j);
    }
}

void attn_grad_head(std::vector<double> & d_attn, const std::vector<double> & d_merged,
                    const std::vector<double> & v, const SpeechAttentionDims & dims, int h) {
    for (int t = 0; t < dims.L; ++t) {
        attn_grad_row(&d_attn[((std::size_t) h * dims.L + t) * dims.Lctx], d_merged, v, dims, h, t);
    }
}

void compute_attn_grad(std::vector<double> & d_attn, const std::vector<double> & d_merged,
                       const std::vector<double> & v, const SpeechAttentionDims & dims) {
    for (int h = 0; h < dims.heads; ++h) {
        attn_grad_head(d_attn, d_merged, v, dims, h);
    }
}

// d_v[j, head h, d] = sum_t d_merged[t, head h, d] * attn[h,t,j].
double value_grad_element(const std::vector<double> & d_merged, const std::vector<double> & attn,
                          const SpeechAttentionDims & dims, int h, int j, int d) {
    const int ch = h * head_dim(dims) + d;
    double acc = 0.0;
    for (int t = 0; t < dims.L; ++t) {
        acc += d_merged[(std::size_t) t * dims.C + ch]
             * attn[((std::size_t) h * dims.L + t) * dims.Lctx + j];
    }
    return acc;
}

void value_grad_key(std::vector<double> & d_v, const std::vector<double> & d_merged,
                    const std::vector<double> & attn, const SpeechAttentionDims & dims,
                    int h, int j) {
    const int hd = head_dim(dims);
    for (int d = 0; d < hd; ++d) {
        const int ch = h * hd + d;
        d_v[(std::size_t) j * dims.C + ch] = value_grad_element(d_merged, attn, dims, h, j, d);
    }
}

void value_grad_head(std::vector<double> & d_v, const std::vector<double> & d_merged,
                     const std::vector<double> & attn, const SpeechAttentionDims & dims, int h) {
    for (int j = 0; j < dims.Lctx; ++j) {
        value_grad_key(d_v, d_merged, attn, dims, h, j);
    }
}

void compute_value_grad(std::vector<double> & d_v, const std::vector<double> & d_merged,
                        const std::vector<double> & attn, const SpeechAttentionDims & dims) {
    for (int h = 0; h < dims.heads; ++h) {
        value_grad_head(d_v, d_merged, attn, dims, h);
    }
}

double weighted_sum(const double * attn_row, const double * d_attn_row, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += attn_row[i] * d_attn_row[i];
    }
    return acc;
}

// Softmax Jacobian-vector product for one row: d_scores = attn ⊙ (d_attn - <attn, d_attn>).
void softmax_backward_row(double * d_scores_row, const double * attn_row,
                          const double * d_attn_row, int n) {
    const double s = weighted_sum(attn_row, d_attn_row, n);
    for (int j = 0; j < n; ++j) {
        d_scores_row[j] = attn_row[j] * (d_attn_row[j] - s);
    }
}

void softmax_backward_head(std::vector<double> & d_scores, const std::vector<double> & attn,
                           const std::vector<double> & d_attn, const SpeechAttentionDims & dims,
                           int h) {
    for (int t = 0; t < dims.L; ++t) {
        const std::size_t off = ((std::size_t) h * dims.L + t) * dims.Lctx;
        softmax_backward_row(&d_scores[off], &attn[off], &d_attn[off], dims.Lctx);
    }
}

void compute_softmax_backward(std::vector<double> & d_scores, const std::vector<double> & attn,
                              const std::vector<double> & d_attn, const SpeechAttentionDims & dims) {
    for (int h = 0; h < dims.heads; ++h) {
        softmax_backward_head(d_scores, attn, d_attn, dims, h);
    }
}

// d_q[t, head h, d] = scale * sum_j d_scores[h,t,j] * K[head h, d, j].
double query_grad_element(const std::vector<double> & d_scores, const std::vector<double> & K,
                          const SpeechAttentionDims & dims, int h, int t, int d) {
    const int ch = h * head_dim(dims) + d;
    const double * srow = &d_scores[((std::size_t) h * dims.L + t) * dims.Lctx];
    double acc = 0.0;
    for (int j = 0; j < dims.Lctx; ++j) {
        acc += srow[j] * K[(std::size_t) ch * dims.Lctx + j];
    }
    return acc * dims.scale;
}

void query_grad_row(std::vector<double> & d_q, const std::vector<double> & d_scores,
                    const std::vector<double> & K, const SpeechAttentionDims & dims, int h, int t) {
    const int hd = head_dim(dims);
    for (int d = 0; d < hd; ++d) {
        const int ch = h * hd + d;
        d_q[(std::size_t) t * dims.C + ch] = query_grad_element(d_scores, K, dims, h, t, d);
    }
}

void query_grad_head(std::vector<double> & d_q, const std::vector<double> & d_scores,
                     const std::vector<double> & K, const SpeechAttentionDims & dims, int h) {
    for (int t = 0; t < dims.L; ++t) {
        query_grad_row(d_q, d_scores, K, dims, h, t);
    }
}

void compute_query_grad(std::vector<double> & d_q, const std::vector<double> & d_scores,
                        const std::vector<double> & K, const SpeechAttentionDims & dims) {
    for (int h = 0; h < dims.heads; ++h) {
        query_grad_head(d_q, d_scores, K, dims, h);
    }
}

}  // namespace

std::vector<double> speech_attention_forward(const SpeechAttentionDims & dims,
                                             const SpeechAttentionWeights & weights,
                                             const std::vector<double> & x_lc,
                                             const std::vector<double> & style,
                                             SpeechAttentionActivations & acts) {
    const std::vector<double> q = dense_time_forward(x_lc, dims.L, weights.q);
    acts.v = dense_time_forward(style, dims.Lctx, weights.v);
    acts.attn.assign((std::size_t) dims.heads * dims.L * dims.Lctx, 0.0);
    compute_attention(acts.attn, q.data(), weights.tanh_k, dims);
    std::vector<double> merged((std::size_t) dims.L * dims.C, 0.0);
    compute_merged(merged, acts.attn, acts.v, dims);
    return dense_time_forward(merged, dims.L, weights.o);
}

SpeechAttentionGrads speech_attention_backward(const SpeechAttentionDims & dims,
                                               const SpeechAttentionWeights & weights,
                                               const SpeechAttentionActivations & acts,
                                               const std::vector<double> & d_out) {
    const std::vector<double> d_merged = dense_time_backward_input(d_out, dims.L, weights.o);

    std::vector<double> d_attn((std::size_t) dims.heads * dims.L * dims.Lctx, 0.0);
    compute_attn_grad(d_attn, d_merged, acts.v, dims);

    std::vector<double> d_v((std::size_t) dims.Lctx * dims.C, 0.0);
    compute_value_grad(d_v, d_merged, acts.attn, dims);

    std::vector<double> d_scores((std::size_t) dims.heads * dims.L * dims.Lctx, 0.0);
    compute_softmax_backward(d_scores, acts.attn, d_attn, dims);

    std::vector<double> d_q((std::size_t) dims.L * dims.C, 0.0);
    compute_query_grad(d_q, d_scores, weights.tanh_k, dims);

    SpeechAttentionGrads grads;
    grads.d_x     = dense_time_backward_input(d_q, dims.L, weights.q);
    grads.d_style = dense_time_backward_input(d_v, dims.Lctx, weights.v);
    return grads;
}

namespace {

// Elementwise sum of two equal-length vectors.
std::vector<double> add_vectors(const std::vector<double> & a, const std::vector<double> & b) {
    std::vector<double> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

}  // namespace

std::vector<double> speech_tail_forward(const SpeechAttentionDims & dims,
                                        const SpeechTailWeights & weights,
                                        const std::vector<double> & stack_out,
                                        const std::vector<double> & style,
                                        SpeechTailActivations & acts) {
    const std::vector<double> attn0 =
        speech_attention_forward(dims, weights.spa[0], stack_out, style, acts.spa[0]);
    const std::vector<double> x1 = add_vectors(stack_out, attn0);
    const std::vector<double> attn1 =
        speech_attention_forward(dims, weights.spa[1], x1, style, acts.spa[1]);
    acts.x_final = add_vectors(stack_out, attn1);
    return layer_norm_channel_forward(acts.x_final, dims.L, dims.C,
                                      weights.ln_gamma, weights.ln_beta);
}

std::vector<double> speech_tail_backward(const SpeechAttentionDims & dims,
                                         const SpeechTailWeights & weights,
                                         const SpeechTailActivations & acts,
                                         const std::vector<double> & d_out) {
    const std::vector<double> d_x_final = layer_norm_channel_backward(
        acts.x_final, dims.L, dims.C, weights.ln_gamma, d_out);
    // x_final = stack_out + attn1 (stack_out is style-independent), so the
    // gradient flows unchanged into the second attention layer's output.
    const SpeechAttentionGrads g1 =
        speech_attention_backward(dims, weights.spa[1], acts.spa[1], d_x_final);
    // x1 = stack_out + attn0, so g1.d_x is exactly the gradient into attn0.
    const SpeechAttentionGrads g0 =
        speech_attention_backward(dims, weights.spa[0], acts.spa[0], g1.d_x);
    return add_vectors(g1.d_style, g0.d_style);
}

}  // namespace supertonic_grad
}  // namespace tts_cpp
