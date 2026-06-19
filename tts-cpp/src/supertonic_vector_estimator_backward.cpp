#include "supertonic_vector_estimator_backward.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace tts_cpp {
namespace ve_grad {

namespace {

constexpr double kInvSqrt2    = 0.7071067811865476;   // 1 / sqrt(2)
constexpr double kInvSqrt2Pi  = 0.3989422804014327;   // 1 / sqrt(2 * pi)

double dot(const double * a, const double * b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

double bias_or_zero(const std::vector<double> & b, int i) {
    return b.empty() ? 0.0 : b[(std::size_t) i];
}

// --- conv1x1 (out-major weight) ---------------------------------------------

double conv1x1_out(const double * x_row, const std::vector<double> & w,
                   const std::vector<double> & b, int IC, int oc) {
    return bias_or_zero(b, oc) + dot(x_row, &w[(std::size_t) oc * IC], IC);
}

void conv1x1_forward_row(const double * x_row, const std::vector<double> & w,
                         const std::vector<double> & b, int IC, int OC, double * y_row) {
    for (int oc = 0; oc < OC; ++oc) y_row[oc] = conv1x1_out(x_row, w, b, IC, oc);
}

double conv1x1_input_grad_elem(const double * d_y_row, const std::vector<double> & w,
                               int IC, int OC, int ic) {
    double s = 0.0;
    for (int oc = 0; oc < OC; ++oc) s += d_y_row[oc] * w[(std::size_t) oc * IC + ic];
    return s;
}

void conv1x1_input_grad_row(const double * d_y_row, const std::vector<double> & w,
                            int IC, int OC, double * d_x_row) {
    for (int ic = 0; ic < IC; ++ic) d_x_row[ic] = conv1x1_input_grad_elem(d_y_row, w, IC, OC, ic);
}

// --- dense_time (in-major ONNX MatMul weight) -------------------------------

double dense_time_out(const double * x_row, const std::vector<double> & w,
                      const std::vector<double> & b, int IC, int OC, int oc) {
    double s = bias_or_zero(b, oc);
    for (int ic = 0; ic < IC; ++ic) s += x_row[ic] * w[(std::size_t) ic * OC + oc];
    return s;
}

void dense_time_forward_row(const double * x_row, const std::vector<double> & w,
                            const std::vector<double> & b, int IC, int OC, double * y_row) {
    for (int oc = 0; oc < OC; ++oc) y_row[oc] = dense_time_out(x_row, w, b, IC, OC, oc);
}

double dense_time_input_grad_elem(const double * d_y_row, const std::vector<double> & w,
                                  int OC, int ic) {
    double s = 0.0;
    for (int oc = 0; oc < OC; ++oc) s += d_y_row[oc] * w[(std::size_t) ic * OC + oc];
    return s;
}

void dense_time_input_grad_row(const double * d_y_row, const std::vector<double> & w,
                               int IC, int OC, double * d_x_row) {
    for (int ic = 0; ic < IC; ++ic) d_x_row[ic] = dense_time_input_grad_elem(d_y_row, w, OC, ic);
}

// --- depthwise "same" conv (edge-clamp padding) -----------------------------

int clamp_index(int idx, int lo, int hi) {
    return std::max(lo, std::min(hi, idx));
}

double depthwise_out(const std::vector<double> & x, int L, int C, const std::vector<double> & w,
                     int K, int dilation, int pad_left, int t, int c) {
    double s = 0.0;
    for (int k = 0; k < K; ++k) {
        const int st = clamp_index(t + k * dilation - pad_left, 0, L - 1);
        s += w[(std::size_t) c * K + k] * x[(std::size_t) st * C + c];
    }
    return s;
}

void depthwise_forward_row(const std::vector<double> & x, int L, int C, const std::vector<double> & w,
                           const std::vector<double> & b, int K, int dilation, int pad_left,
                           int t, double * y_row) {
    for (int c = 0; c < C; ++c) y_row[c] = bias_or_zero(b, c) + depthwise_out(x, L, C, w, K, dilation, pad_left, t, c);
}

void depthwise_scatter(std::vector<double> & d_x, double d_y_tc, int L, int C,
                       const std::vector<double> & w, int K, int dilation, int pad_left, int t, int c) {
    for (int k = 0; k < K; ++k) {
        const int st = clamp_index(t + k * dilation - pad_left, 0, L - 1);
        d_x[(std::size_t) st * C + c] += d_y_tc * w[(std::size_t) c * K + k];
    }
}

void depthwise_scatter_row(std::vector<double> & d_x, const std::vector<double> & d_y, int L, int C,
                           const std::vector<double> & w, int K, int dilation, int pad_left, int t) {
    for (int c = 0; c < C; ++c) {
        depthwise_scatter(d_x, d_y[(std::size_t) t * C + c], L, C, w, K, dilation, pad_left, t, c);
    }
}

// --- layer norm (per time step over C channels) -----------------------------

double row_mean(const double * x_row, int C) {
    double s = 0.0;
    for (int c = 0; c < C; ++c) s += x_row[c];
    return s / (double) C;
}

double row_variance(const double * x_row, int C, double mean) {
    double s = 0.0;
    for (int c = 0; c < C; ++c) { const double d = x_row[c] - mean; s += d * d; }
    return s / (double) C;
}

void layer_norm_apply_row(const double * x_row, int C, double mean, double inv,
                          const std::vector<double> & gamma, const std::vector<double> & beta,
                          double * y_row) {
    for (int c = 0; c < C; ++c) y_row[c] = (x_row[c] - mean) * inv * gamma[(std::size_t) c] + beta[(std::size_t) c];
}

// Returns normalized row n_c = (x_c - mean) * inv into `norm`.
void normalized_row(const double * x_row, int C, double mean, double inv, double * norm) {
    for (int c = 0; c < C; ++c) norm[c] = (x_row[c] - mean) * inv;
}

// d_norm_c = d_y_c * gamma_c.
void scaled_by_gamma(const double * d_y_row, const std::vector<double> & gamma, int C, double * d_norm) {
    for (int c = 0; c < C; ++c) d_norm[c] = d_y_row[c] * gamma[(std::size_t) c];
}

double mean_of(const double * v, int C) {
    double s = 0.0;
    for (int c = 0; c < C; ++c) s += v[c];
    return s / (double) C;
}

double mean_of_product(const double * a, const double * b, int C) {
    double s = 0.0;
    for (int c = 0; c < C; ++c) s += a[c] * b[c];
    return s / (double) C;
}

void layer_norm_input_grad_row(const double * d_norm, const double * norm, int C, double inv,
                               double mean_dn, double mean_dn_n, double * d_x_row) {
    for (int c = 0; c < C; ++c) d_x_row[c] = inv * (d_norm[c] - mean_dn - norm[c] * mean_dn_n);
}

// --- gelu (erf form) --------------------------------------------------------

double gelu_scalar(double x) {
    return 0.5 * x * (1.0 + std::erf(x * kInvSqrt2));
}

double gelu_derivative(double x) {
    const double cdf = 0.5 * (1.0 + std::erf(x * kInvSqrt2));
    const double pdf = kInvSqrt2Pi * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

// --- ConvNeXt helpers -------------------------------------------------------

// out[t, c] = x[t, c] + gamma[c] * y2[t, c]  (residual with per-channel scale).
void residual_gamma_row(const double * x_row, const double * y2_row,
                        const std::vector<double> & gamma, int C, double * out_row) {
    for (int c = 0; c < C; ++c) out_row[c] = x_row[c] + gamma[(std::size_t) c] * y2_row[c];
}

// d_y2[t, c] = d_out[t, c] * gamma[c]  (gradient through the per-channel scale).
void scale_by_gamma_row(const double * d_out_row, const std::vector<double> & gamma, int C,
                        double * d_y2_row) {
    for (int c = 0; c < C; ++c) d_y2_row[c] = d_out_row[c] * gamma[(std::size_t) c];
}

void add_in_place(std::vector<double> & a, const std::vector<double> & b) {
    for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

// --- Cross-attention core (q, k, v already projected; A = H * D) -------------

double max_of(const double * a, int n) {
    double mx = a[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, a[i]);
    return mx;
}

double exp_shift_sum(double * a, int n, double mx) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { a[i] = std::exp(a[i] - mx); sum += a[i]; }
    return sum;
}

void scale_vec(double * a, int n, double f) {
    for (int i = 0; i < n; ++i) a[i] *= f;
}

void softmax_inplace(double * a, int n) {
    const double sum = exp_shift_sum(a, n, max_of(a, n));
    scale_vec(a, n, 1.0 / sum);
}

void attn_scores_row(const std::vector<double> & q, const std::vector<double> & k, int qi, int h,
                     int Lk, int A, int D, double scale, double * scores) {
    const std::size_t bq = (std::size_t) qi * A + (std::size_t) h * D;
    for (int kj = 0; kj < Lk; ++kj) {
        scores[kj] = scale * dot(&q[bq], &k[(std::size_t) kj * A + (std::size_t) h * D], D);
    }
}

double weighted_value(const double * prob, const std::vector<double> & v, int h, int Lk, int A, int D,
                      int d) {
    double s = 0.0;
    for (int kj = 0; kj < Lk; ++kj) s += prob[kj] * v[(std::size_t) kj * A + (std::size_t) h * D + d];
    return s;
}

void attn_out_row(const double * prob, const std::vector<double> & v, int qi, int h, int Lk, int A,
                  int D, double * out) {
    const std::size_t bq = (std::size_t) qi * A + (std::size_t) h * D;
    for (int d = 0; d < D; ++d) out[bq + d] = weighted_value(prob, v, h, Lk, A, D, d);
}

void cross_attn_head_forward(const std::vector<double> & q, const std::vector<double> & k,
                             const std::vector<double> & v, int h, int L, int Lk, int A, int D,
                             double scale, double * prob, double * out) {
    for (int qi = 0; qi < L; ++qi) {
        double * pr = &prob[(std::size_t) (h * L + qi) * Lk];
        attn_scores_row(q, k, qi, h, Lk, A, D, scale, pr);
        softmax_inplace(pr, Lk);
        attn_out_row(pr, v, qi, h, Lk, A, D, out);
    }
}

std::vector<double> cross_attn_core_forward(const std::vector<double> & q, const std::vector<double> & k,
                                            const std::vector<double> & v, int L, int Lk, int H, int D,
                                            double scale, std::vector<double> & prob) {
    const int A = H * D;
    std::vector<double> out((std::size_t) L * A, 0.0);
    prob.assign((std::size_t) H * L * Lk, 0.0);
    for (int h = 0; h < H; ++h) {
        cross_attn_head_forward(q, k, v, h, L, Lk, A, D, scale, prob.data(), out.data());
    }
    return out;
}

double d_prob_elem(const double * d_attn, const std::vector<double> & v, int qi, int h, int kj, int A,
                   int D) {
    const std::size_t bq = (std::size_t) qi * A + (std::size_t) h * D;
    const std::size_t bk = (std::size_t) kj * A + (std::size_t) h * D;
    double s = 0.0;
    for (int d = 0; d < D; ++d) s += d_attn[bq + d] * v[bk + d];
    return s;
}

void d_prob_row(const double * d_attn, const std::vector<double> & v, int qi, int h, int Lk, int A,
                int D, double * d_prob) {
    for (int kj = 0; kj < Lk; ++kj) d_prob[kj] = d_prob_elem(d_attn, v, qi, h, kj, A, D);
}

void accumulate_dv_elem(std::vector<double> & d_v, const double * d_attn, double prob_kj, int qi, int h,
                        int kj, int A, int D) {
    const std::size_t bq = (std::size_t) qi * A + (std::size_t) h * D;
    const std::size_t bk = (std::size_t) kj * A + (std::size_t) h * D;
    for (int d = 0; d < D; ++d) d_v[bk + d] += prob_kj * d_attn[bq + d];
}

void accumulate_dv_row(std::vector<double> & d_v, const double * prob, const double * d_attn, int qi,
                       int h, int Lk, int A, int D) {
    for (int kj = 0; kj < Lk; ++kj) accumulate_dv_elem(d_v, d_attn, prob[kj], qi, h, kj, A, D);
}

double softmax_self_dot(const double * prob, const double * d_prob, int Lk) {
    double s = 0.0;
    for (int kj = 0; kj < Lk; ++kj) s += prob[kj] * d_prob[kj];
    return s;
}

void d_scores_row(const double * prob, const double * d_prob, double self_dot, int Lk, double * d_sc) {
    for (int kj = 0; kj < Lk; ++kj) d_sc[kj] = prob[kj] * (d_prob[kj] - self_dot);
}

double d_query_elem(const double * d_sc, const std::vector<double> & k, int h, int Lk, int A, int D,
                    int d, double scale) {
    double s = 0.0;
    for (int kj = 0; kj < Lk; ++kj) s += d_sc[kj] * k[(std::size_t) kj * A + (std::size_t) h * D + d];
    return scale * s;
}

void d_query_row(const double * d_sc, const std::vector<double> & k, int qi, int h, int Lk, int A,
                 int D, double scale, double * d_q) {
    const std::size_t bq = (std::size_t) qi * A + (std::size_t) h * D;
    for (int d = 0; d < D; ++d) d_q[bq + d] = d_query_elem(d_sc, k, h, Lk, A, D, d, scale);
}

void cross_attn_head_backward(const std::vector<double> & d_attn, const std::vector<double> & k,
                              const std::vector<double> & v, const double * prob, int h, int L, int Lk,
                              int A, int D, double scale, bool want_dv, std::vector<double> & d_q,
                              std::vector<double> & d_v) {
    std::vector<double> d_prob((std::size_t) Lk);
    std::vector<double> d_sc((std::size_t) Lk);
    for (int qi = 0; qi < L; ++qi) {
        const double * pr = &prob[(std::size_t) (h * L + qi) * Lk];
        d_prob_row(d_attn.data(), v, qi, h, Lk, A, D, d_prob.data());
        if (want_dv) accumulate_dv_row(d_v, pr, d_attn.data(), qi, h, Lk, A, D);
        const double self_dot = softmax_self_dot(pr, d_prob.data(), Lk);
        d_scores_row(pr, d_prob.data(), self_dot, Lk, d_sc.data());
        d_query_row(d_sc.data(), k, qi, h, Lk, A, D, scale, d_q.data());
    }
}

// Returns d_q [L, A]; fills d_v [Lk, A] when want_dv (key gradient is never
// needed — the key source is constant in both attention flavours).
std::vector<double> cross_attn_core_backward(const std::vector<double> & d_attn,
                                             const std::vector<double> & k, const std::vector<double> & v,
                                             const std::vector<double> & prob, int L, int Lk, int H,
                                             int D, double scale, bool want_dv,
                                             std::vector<double> & d_v) {
    const int A = H * D;
    std::vector<double> d_q((std::size_t) L * A, 0.0);
    if (want_dv) d_v.assign((std::size_t) Lk * A, 0.0);
    for (int h = 0; h < H; ++h) {
        cross_attn_head_backward(d_attn, k, v, prob.data(), h, L, Lk, A, D, scale, want_dv, d_q, d_v);
    }
    return d_q;
}

// --- Rotary position embedding (matches `apply_rope`) -----------------------

void rope_rotate_elem(double * x, int t, int h, int d, int half, int L, int H, int D,
                      const std::vector<double> & theta) {
    const double angle = ((double) t / (double) L) * theta[(std::size_t) d];
    const double cs = std::cos(angle), sn = std::sin(angle);
    const std::size_t i1 = ((std::size_t) t * H + h) * D + d;
    const std::size_t i2 = ((std::size_t) t * H + h) * D + half + d;
    const double a = x[i1], b = x[i2];
    x[i1] = a * cs - b * sn;
    x[i2] = b * cs + a * sn;
}

void rope_unrotate_elem(double * d_x, int t, int h, int d, int half, int L, int H, int D,
                        const std::vector<double> & theta) {
    const double angle = ((double) t / (double) L) * theta[(std::size_t) d];
    const double cs = std::cos(angle), sn = std::sin(angle);
    const std::size_t i1 = ((std::size_t) t * H + h) * D + d;
    const std::size_t i2 = ((std::size_t) t * H + h) * D + half + d;
    const double g1 = d_x[i1], g2 = d_x[i2];
    d_x[i1] = g1 * cs + g2 * sn;   // transpose of the forward rotation
    d_x[i2] = -g1 * sn + g2 * cs;
}

void rope_rotate_position(double * x, int t, int h, int half, int L, int H, int D,
                          const std::vector<double> & theta) {
    for (int d = 0; d < half; ++d) rope_rotate_elem(x, t, h, d, half, L, H, D, theta);
}

void rope_rotate_head(double * x, int h, int half, int L, int H, int D,
                      const std::vector<double> & theta) {
    for (int t = 0; t < L; ++t) rope_rotate_position(x, t, h, half, L, H, D, theta);
}

void rope_apply(std::vector<double> & x, int L, int H, int D, const std::vector<double> & theta) {
    const int half = D / 2;
    for (int h = 0; h < H; ++h) rope_rotate_head(x.data(), h, half, L, H, D, theta);
}

void rope_unrotate_position(double * d_x, int t, int h, int half, int L, int H, int D,
                            const std::vector<double> & theta) {
    for (int d = 0; d < half; ++d) rope_unrotate_elem(d_x, t, h, d, half, L, H, D, theta);
}

void rope_unrotate_head(double * d_x, int h, int half, int L, int H, int D,
                        const std::vector<double> & theta) {
    for (int t = 0; t < L; ++t) rope_unrotate_position(d_x, t, h, half, L, H, D, theta);
}

void rope_backward(std::vector<double> & d_x, int L, int H, int D, const std::vector<double> & theta) {
    const int half = D / 2;
    for (int h = 0; h < H; ++h) rope_unrotate_head(d_x.data(), h, half, L, H, D, theta);
}

void tanh_inplace(std::vector<double> & x) {
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::tanh(x[i]);
}

// --- vector-field assembly helpers ------------------------------------------

void mask_row(double * x_row, int C, double m) {
    for (int c = 0; c < C; ++c) x_row[c] *= m;
}

void mask_rows_in_place(std::vector<double> & x, int L, int C, const std::vector<double> & mask) {
    for (int t = 0; t < L; ++t) mask_row(&x[(std::size_t) t * C], C, mask[(std::size_t) t]);
}

void transpose_cl_column(const std::vector<double> & src, int Cin, int L, int c, double * dst) {
    for (int t = 0; t < L; ++t) dst[(std::size_t) t * Cin + c] = src[(std::size_t) c * L + t];
}

// Channel-major [Cin, L] -> time-major [L, Cin].
void transpose_cl_to_lc(const std::vector<double> & src, int Cin, int L, std::vector<double> & dst) {
    dst.assign((std::size_t) L * Cin, 0.0);
    for (int c = 0; c < Cin; ++c) transpose_cl_column(src, Cin, L, c, dst.data());
}

void accumulate_lc_column(const std::vector<double> & src_lc, int Cin, int L, int c, double * dst_cl) {
    for (int t = 0; t < L; ++t) dst_cl[(std::size_t) c * L + t] += src_lc[(std::size_t) t * Cin + c];
}

// Accumulate a time-major [L, Cin] gradient into a channel-major [Cin, L] buffer.
void accumulate_lc_into_cl(const std::vector<double> & src_lc, int Cin, int L,
                           std::vector<double> & dst_cl) {
    for (int c = 0; c < Cin; ++c) accumulate_lc_column(src_lc, Cin, L, c, dst_cl.data());
}

}  // namespace

namespace {

std::vector<double> field_convnext_a_forward(const VectorFieldGroupWeights & g, std::vector<double> xf,
                                             int L, VectorFieldGroupActivations & ga) {
    for (int j = 0; j < 4; ++j) xf = convnext_forward(g.convnext_a[j], xf, L, ga.cna[j]);
    return xf;
}

std::vector<double> field_group_forward(const VectorFieldWeights & w, const VectorFieldGroupWeights & g,
                                        std::vector<double> xf, const std::vector<double> & style_v,
                                        int L, VectorFieldGroupActivations & ga) {
    xf = field_convnext_a_forward(g, std::move(xf), L, ga);
    xf = convnext_forward(g.convnext_b, xf, L, ga.cnb);
    add_in_place(xf, rope_attn_forward(w.rope_dims, g.rope, g.theta, xf, w.text_lc, ga.rope));
    ga.ln1_in = xf;
    xf = layer_norm_forward(xf, L, w.C, g.ln1_gamma, g.ln1_beta);
    xf = convnext_forward(g.convnext_c, xf, L, ga.cnc);
    add_in_place(xf, style_attn_forward(w.style_dims, g.style, xf, style_v, w.k_const, ga.style));
    ga.ln2_in = xf;
    xf = layer_norm_forward(xf, L, w.C, g.ln2_gamma, g.ln2_beta);
    return xf;
}

std::vector<double> field_groups_forward(const VectorFieldWeights & w, std::vector<double> xf,
                                         const std::vector<double> & style_v, int L,
                                         VectorFieldActivations & acts) {
    for (int gi = 0; gi < 4; ++gi) {
        xf = field_group_forward(w, w.groups[gi], std::move(xf), style_v, L, acts.groups[gi]);
    }
    return xf;
}

std::vector<double> field_last_forward(const VectorFieldWeights & w, std::vector<double> xf, int L,
                                       VectorFieldActivations & acts) {
    for (int j = 0; j < 4; ++j) xf = convnext_forward(w.last_convnext[j], xf, L, acts.last[j]);
    return xf;
}

std::vector<double> field_convnext_a_backward(const VectorFieldGroupWeights & g,
                                              const VectorFieldGroupActivations & ga,
                                              std::vector<double> d, int L) {
    for (int j = 3; j >= 0; --j) d = convnext_backward_input(g.convnext_a[j], ga.cna[j], d, L);
    return d;
}

std::vector<double> field_group_backward(const VectorFieldWeights & w, const VectorFieldGroupWeights & g,
                                         const VectorFieldGroupActivations & ga, std::vector<double> d,
                                         int L, std::vector<double> & d_style_accum) {
    d = layer_norm_backward_input(ga.ln2_in, L, w.C, g.ln2_gamma, d);
    const StyleAttnGrads sg = style_attn_backward(w.style_dims, g.style, ga.style, d);
    add_in_place(d, sg.d_x);             // residual + attention path
    add_in_place(d_style_accum, sg.d_style);
    d = convnext_backward_input(g.convnext_c, ga.cnc, d, L);
    d = layer_norm_backward_input(ga.ln1_in, L, w.C, g.ln1_gamma, d);
    add_in_place(d, rope_attn_backward_input(w.rope_dims, g.rope, g.theta, ga.rope, d));
    d = convnext_backward_input(g.convnext_b, ga.cnb, d, L);
    d = field_convnext_a_backward(g, ga, d, L);
    return d;
}

std::vector<double> field_groups_backward(const VectorFieldWeights & w, const VectorFieldActivations & acts,
                                          std::vector<double> d, int L,
                                          std::vector<double> & d_style_accum) {
    for (int gi = 3; gi >= 0; --gi) {
        d = field_group_backward(w, w.groups[gi], acts.groups[gi], std::move(d), L, d_style_accum);
    }
    return d;
}

std::vector<double> field_last_backward(const VectorFieldWeights & w, const VectorFieldActivations & acts,
                                        std::vector<double> d, int L) {
    for (int j = 3; j >= 0; --j) d = convnext_backward_input(w.last_convnext[j], acts.last[j], d, L);
    return d;
}

// next[c, t] += mask[t] * v[t, c] / total_steps   (velocity Euler update).
void add_velocity_column(std::vector<double> & next, const std::vector<double> & v,
                         const std::vector<double> & mask, int Cin, int L, int c, double inv_n) {
    for (int t = 0; t < L; ++t) {
        next[(std::size_t) c * L + t] += mask[(std::size_t) t] * v[(std::size_t) t * Cin + c] * inv_n;
    }
}

void add_velocity(std::vector<double> & next, const std::vector<double> & v,
                  const std::vector<double> & mask, int Cin, int L, double inv_n) {
    for (int c = 0; c < Cin; ++c) add_velocity_column(next, v, mask, Cin, L, c, inv_n);
}

// d_v[t, c] = d_next[c, t] * mask[t] / total_steps  (backward of the update).
void build_dv_column(const std::vector<double> & d_next, const std::vector<double> & mask, int Cin,
                     int L, int c, double inv_n, double * d_v) {
    for (int t = 0; t < L; ++t) {
        d_v[(std::size_t) t * Cin + c] = d_next[(std::size_t) c * L + t] * mask[(std::size_t) t] * inv_n;
    }
}

void build_dv_from_dnext(const std::vector<double> & d_next, const std::vector<double> & mask, int Cin,
                         int L, double inv_n, std::vector<double> & d_v) {
    d_v.assign((std::size_t) L * Cin, 0.0);
    for (int c = 0; c < Cin; ++c) build_dv_column(d_next, mask, Cin, L, c, inv_n, d_v.data());
}

}  // namespace

// ---------------------------------------------------------------------------

std::vector<double> conv1x1_forward(const std::vector<double> & x, int L, int IC, int OC,
                                    const std::vector<double> & w, const std::vector<double> & b) {
    std::vector<double> y((std::size_t) L * OC);
    for (int t = 0; t < L; ++t) {
        conv1x1_forward_row(&x[(std::size_t) t * IC], w, b, IC, OC, &y[(std::size_t) t * OC]);
    }
    return y;
}

std::vector<double> conv1x1_backward_input(const std::vector<double> & d_y, int L, int IC, int OC,
                                           const std::vector<double> & w) {
    std::vector<double> d_x((std::size_t) L * IC);
    for (int t = 0; t < L; ++t) {
        conv1x1_input_grad_row(&d_y[(std::size_t) t * OC], w, IC, OC, &d_x[(std::size_t) t * IC]);
    }
    return d_x;
}

std::vector<double> dense_time_forward(const std::vector<double> & x, int L, int IC, int OC,
                                       const std::vector<double> & w, const std::vector<double> & b) {
    std::vector<double> y((std::size_t) L * OC);
    for (int t = 0; t < L; ++t) {
        dense_time_forward_row(&x[(std::size_t) t * IC], w, b, IC, OC, &y[(std::size_t) t * OC]);
    }
    return y;
}

std::vector<double> dense_time_backward_input(const std::vector<double> & d_y, int L, int IC, int OC,
                                              const std::vector<double> & w) {
    std::vector<double> d_x((std::size_t) L * IC);
    for (int t = 0; t < L; ++t) {
        dense_time_input_grad_row(&d_y[(std::size_t) t * OC], w, IC, OC, &d_x[(std::size_t) t * IC]);
    }
    return d_x;
}

std::vector<double> depthwise_same_forward(const std::vector<double> & x, int L, int C, int K,
                                           int dilation, const std::vector<double> & w,
                                           const std::vector<double> & b) {
    const int pad_left = ((K - 1) * dilation) / 2;
    std::vector<double> y((std::size_t) L * C);
    for (int t = 0; t < L; ++t) {
        depthwise_forward_row(x, L, C, w, b, K, dilation, pad_left, t, &y[(std::size_t) t * C]);
    }
    return y;
}

std::vector<double> depthwise_same_backward_input(const std::vector<double> & d_y, int L, int C,
                                                  int K, int dilation, const std::vector<double> & w) {
    const int pad_left = ((K - 1) * dilation) / 2;
    std::vector<double> d_x((std::size_t) L * C, 0.0);
    for (int t = 0; t < L; ++t) {
        depthwise_scatter_row(d_x, d_y, L, C, w, K, dilation, pad_left, t);
    }
    return d_x;
}

std::vector<double> layer_norm_forward(const std::vector<double> & x, int L, int C,
                                       const std::vector<double> & gamma,
                                       const std::vector<double> & beta, double eps) {
    std::vector<double> y((std::size_t) L * C);
    for (int t = 0; t < L; ++t) {
        const double * x_row = &x[(std::size_t) t * C];
        const double mean = row_mean(x_row, C);
        const double inv = 1.0 / std::sqrt(row_variance(x_row, C, mean) + eps);
        layer_norm_apply_row(x_row, C, mean, inv, gamma, beta, &y[(std::size_t) t * C]);
    }
    return y;
}

std::vector<double> layer_norm_backward_input(const std::vector<double> & x, int L, int C,
                                              const std::vector<double> & gamma,
                                              const std::vector<double> & d_y, double eps) {
    std::vector<double> d_x((std::size_t) L * C);
    std::vector<double> norm((std::size_t) C);
    std::vector<double> d_norm((std::size_t) C);
    for (int t = 0; t < L; ++t) {
        const double * x_row = &x[(std::size_t) t * C];
        const double mean = row_mean(x_row, C);
        const double inv = 1.0 / std::sqrt(row_variance(x_row, C, mean) + eps);
        normalized_row(x_row, C, mean, inv, norm.data());
        scaled_by_gamma(&d_y[(std::size_t) t * C], gamma, C, d_norm.data());
        const double mean_dn = mean_of(d_norm.data(), C);
        const double mean_dn_n = mean_of_product(d_norm.data(), norm.data(), C);
        layer_norm_input_grad_row(d_norm.data(), norm.data(), C, inv, mean_dn, mean_dn_n,
                                  &d_x[(std::size_t) t * C]);
    }
    return d_x;
}

std::vector<double> gelu_forward(const std::vector<double> & x) {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = gelu_scalar(x[i]);
    return y;
}

std::vector<double> gelu_backward(const std::vector<double> & x, const std::vector<double> & d_y) {
    std::vector<double> d_x(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) d_x[i] = d_y[i] * gelu_derivative(x[i]);
    return d_x;
}

std::vector<double> convnext_forward(const ConvNextWeights & w, const std::vector<double> & x, int L,
                                     ConvNextActivations & acts) {
    acts.dw_out = depthwise_same_forward(x, L, w.C, w.K, w.dilation, w.dw_w, w.dw_b);
    const std::vector<double> ln_out = layer_norm_forward(acts.dw_out, L, w.C, w.ln_gamma, w.ln_beta);
    acts.z1 = conv1x1_forward(ln_out, L, w.C, w.hidden, w.pw1_w, w.pw1_b);
    const std::vector<double> z2 = gelu_forward(acts.z1);
    const std::vector<double> y2 = conv1x1_forward(z2, L, w.hidden, w.C, w.pw2_w, w.pw2_b);
    std::vector<double> out((std::size_t) L * w.C);
    for (int t = 0; t < L; ++t) {
        residual_gamma_row(&x[(std::size_t) t * w.C], &y2[(std::size_t) t * w.C], w.gamma, w.C,
                           &out[(std::size_t) t * w.C]);
    }
    return out;
}

std::vector<double> convnext_backward_input(const ConvNextWeights & w, const ConvNextActivations & acts,
                                            const std::vector<double> & d_out, int L) {
    std::vector<double> d_y2((std::size_t) L * w.C);
    for (int t = 0; t < L; ++t) {
        scale_by_gamma_row(&d_out[(std::size_t) t * w.C], w.gamma, w.C, &d_y2[(std::size_t) t * w.C]);
    }
    const std::vector<double> d_z2 = conv1x1_backward_input(d_y2, L, w.hidden, w.C, w.pw2_w);
    const std::vector<double> d_z1 = gelu_backward(acts.z1, d_z2);
    const std::vector<double> d_lnout = conv1x1_backward_input(d_z1, L, w.C, w.hidden, w.pw1_w);
    const std::vector<double> d_dwout = layer_norm_backward_input(acts.dw_out, L, w.C, w.ln_gamma, d_lnout);
    std::vector<double> d_x = depthwise_same_backward_input(d_dwout, L, w.C, w.K, w.dilation, w.dw_w);
    add_in_place(d_x, d_out);  // residual path
    return d_x;
}

std::vector<double> rope_attn_forward(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                      const std::vector<double> & theta, const std::vector<double> & x,
                                      const std::vector<double> & text_lc, RopeAttnActivations & acts) {
    const int A = dims.H * dims.D;
    const std::vector<double> q = dense_time_forward(x, dims.L, dims.C, A, w.wq, w.bq);
    std::vector<double> q_rope = q;
    rope_apply(q_rope, dims.L, dims.H, dims.D, theta);
    acts.k_rope = dense_time_forward(text_lc, dims.Lk, dims.Ckv, A, w.wk, w.bk);
    rope_apply(acts.k_rope, dims.Lk, dims.H, dims.D, theta);
    acts.v = dense_time_forward(text_lc, dims.Lk, dims.Ckv, A, w.wv, w.bv);
    const std::vector<double> attn =
        cross_attn_core_forward(q_rope, acts.k_rope, acts.v, dims.L, dims.Lk, dims.H, dims.D, dims.scale,
                                acts.prob);
    return dense_time_forward(attn, dims.L, A, dims.C, w.wo, w.bo);
}

std::vector<double> rope_attn_backward_input(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                             const std::vector<double> & theta,
                                             const RopeAttnActivations & acts,
                                             const std::vector<double> & d_out) {
    const int A = dims.H * dims.D;
    const std::vector<double> d_attn = dense_time_backward_input(d_out, dims.L, A, dims.C, w.wo);
    std::vector<double> unused_dv;
    std::vector<double> d_q_rope =
        cross_attn_core_backward(d_attn, acts.k_rope, acts.v, acts.prob, dims.L, dims.Lk, dims.H,
                                 dims.D, dims.scale, /*want_dv=*/false, unused_dv);
    rope_backward(d_q_rope, dims.L, dims.H, dims.D, theta);
    return dense_time_backward_input(d_q_rope, dims.L, dims.C, A, w.wq);
}

std::vector<double> style_attn_forward(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                       const std::vector<double> & x, const std::vector<double> & style_v,
                                       const std::vector<double> & k_const, StyleAttnActivations & acts) {
    const int A = dims.H * dims.D;
    const std::vector<double> q = dense_time_forward(x, dims.L, dims.C, A, w.wq, w.bq);
    acts.k = dense_time_forward(k_const, dims.Lk, dims.Ckv, A, w.wk, w.bk);
    tanh_inplace(acts.k);
    acts.v = dense_time_forward(style_v, dims.Lk, dims.Ckv, A, w.wv, w.bv);
    const std::vector<double> attn =
        cross_attn_core_forward(q, acts.k, acts.v, dims.L, dims.Lk, dims.H, dims.D, dims.scale, acts.prob);
    return dense_time_forward(attn, dims.L, A, dims.C, w.wo, w.bo);
}

StyleAttnGrads style_attn_backward(const CrossAttnDims & dims, const CrossAttnWeights & w,
                                   const StyleAttnActivations & acts, const std::vector<double> & d_out) {
    const int A = dims.H * dims.D;
    const std::vector<double> d_attn = dense_time_backward_input(d_out, dims.L, A, dims.C, w.wo);
    StyleAttnGrads grads;
    std::vector<double> d_v;
    const std::vector<double> d_q =
        cross_attn_core_backward(d_attn, acts.k, acts.v, acts.prob, dims.L, dims.Lk, dims.H, dims.D,
                                 dims.scale, /*want_dv=*/true, d_v);
    grads.d_x = dense_time_backward_input(d_q, dims.L, dims.C, A, w.wq);
    // V = dense_time(style_v); K is a constant key source (tanh) → no style grad through K.
    grads.d_style = dense_time_backward_input(d_v, dims.Lk, dims.Ckv, A, w.wv);
    return grads;
}

std::vector<double> vector_field_forward(const VectorFieldWeights & w, const std::vector<double> & in,
                                         const std::vector<double> & style_v, int L,
                                         VectorFieldActivations & acts) {
    std::vector<double> xf = conv1x1_forward(in, L, w.Cin, w.C, w.proj_in_w, {});
    mask_rows_in_place(xf, L, w.C, w.mask);
    xf = field_groups_forward(w, std::move(xf), style_v, L, acts);
    xf = field_last_forward(w, std::move(xf), L, acts);
    return conv1x1_forward(xf, L, w.C, w.Cin, w.proj_out_w, {});
}

VectorFieldGrads vector_field_backward(const VectorFieldWeights & w, const VectorFieldActivations & acts,
                                       const std::vector<double> & d_out, int L) {
    std::vector<double> d = conv1x1_backward_input(d_out, L, w.C, w.Cin, w.proj_out_w);
    d = field_last_backward(w, acts, std::move(d), L);
    VectorFieldGrads grads;
    grads.d_style.assign((std::size_t) w.style_dims.Lk * w.style_dims.Ckv, 0.0);
    d = field_groups_backward(w, acts, std::move(d), L, grads.d_style);
    mask_rows_in_place(d, L, w.C, w.mask);
    grads.d_in = conv1x1_backward_input(d, L, w.Cin, w.C, w.proj_in_w);
    return grads;
}

std::vector<double> vector_step_forward(const VectorFieldWeights & w, const std::vector<double> & noisy,
                                        const std::vector<double> & style_v, int L, int total_steps,
                                        VectorFieldActivations & acts) {
    std::vector<double> in;
    transpose_cl_to_lc(noisy, w.Cin, L, in);
    const std::vector<double> v = vector_field_forward(w, in, style_v, L, acts);
    std::vector<double> next = noisy;  // identity term of the Euler update
    add_velocity(next, v, w.mask, w.Cin, L, 1.0 / (double) total_steps);
    return next;
}

VectorStepGrads vector_step_backward(const VectorFieldWeights & w, const VectorFieldActivations & acts,
                                     const std::vector<double> & d_next, int L, int total_steps) {
    std::vector<double> d_v;
    build_dv_from_dnext(d_next, w.mask, w.Cin, L, 1.0 / (double) total_steps, d_v);
    const VectorFieldGrads fg = vector_field_backward(w, acts, d_v, L);
    VectorStepGrads sg;
    sg.d_style = fg.d_style;
    sg.d_noisy = d_next;  // identity term
    accumulate_lc_into_cl(fg.d_in, w.Cin, L, sg.d_noisy);
    return sg;
}

}  // namespace ve_grad
}  // namespace tts_cpp
