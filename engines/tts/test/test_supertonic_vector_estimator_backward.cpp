// Gradcheck self-tests for the Supertonic vector-estimator (CFM) backward
// (voice-clone ticket "GGML backward pass: vector estimator (CFM)").  Pure host
// logic, model-free: every analytic gradient is checked component-wise against a
// central finite-difference numeric gradient of the matching forward.  Runs in
// the always-on `unit` ctest tier.
//
// Standalone build (single line):
//   g++ -std=c++17 -I src test/test_supertonic_vector_estimator_backward.cpp src/supertonic_vector_estimator_backward.cpp src/voiceclone_gradcheck.cpp -o /tmp/t && /tmp/t

#include "supertonic_vector_estimator_backward.h"
#include "voiceclone_gradcheck.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace tts_cpp::ve_grad;
using tts_cpp::voiceclone::ScalarLossFn;
using tts_cpp::voiceclone::compare_gradients;
using tts_cpp::voiceclone::finite_diff_gradient;
using tts_cpp::voiceclone::GradcheckReport;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, ...) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
        ++g_failures;                                                    \
        fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);            \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                          \
    }                                                                    \
} while (0)

// Deterministic pseudo-random fill so tests are reproducible without a PRNG
// dependency; values land in roughly [-1, 1] with no special structure.
double sample(int i, double phase) {
    return std::sin(i * 0.9 + phase) * 0.8;
}

std::vector<double> make_vector(int n, double phase) {
    std::vector<double> v((std::size_t) n);
    for (int i = 0; i < n; ++i) v[i] = sample(i, phase);
    return v;
}

double dot(const std::vector<double> & a, const std::vector<double> & b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
    return acc;
}

void report_check(const char * name, const GradcheckReport & r) {
    CHECK(r.passed, "%s: gradcheck failed (max_abs=%.3e max_rel=%.3e worst=%zu)",
          name, r.max_abs_err, r.max_rel_err, r.worst_index);
}

// ---------------------------------------------------------------------------

void test_conv1x1_backward() {
    const int L = 4, IC = 5, OC = 6;
    const std::vector<double> w = make_vector(OC * IC, 0.3);
    const std::vector<double> b = make_vector(OC, 1.1);
    const std::vector<double> coeffs = make_vector(L * OC, 2.0);
    const std::vector<double> x0 = make_vector(L * IC, 0.7);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, conv1x1_forward(x, L, IC, OC, w, b));
    };
    const std::vector<double> analytic = conv1x1_backward_input(coeffs, L, IC, OC, w);
    report_check("conv1x1_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

void test_dense_time_backward() {
    const int L = 3, IC = 4, OC = 5;
    const std::vector<double> w = make_vector(IC * OC, 0.3);
    const std::vector<double> b = make_vector(OC, 1.1);
    const std::vector<double> coeffs = make_vector(L * OC, 2.0);
    const std::vector<double> x0 = make_vector(L * IC, 0.7);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, dense_time_forward(x, L, IC, OC, w, b));
    };
    const std::vector<double> analytic = dense_time_backward_input(coeffs, L, IC, OC, w);
    report_check("dense_time_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

void test_depthwise_backward() {
    const int L = 7, C = 4, K = 3, dilation = 2;
    const std::vector<double> w = make_vector(C * K, 0.4);
    const std::vector<double> b = make_vector(C, 0.9);
    const std::vector<double> coeffs = make_vector(L * C, 1.6);
    const std::vector<double> x0 = make_vector(L * C, 0.5);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, depthwise_same_forward(x, L, C, K, dilation, w, b));
    };
    const std::vector<double> analytic = depthwise_same_backward_input(coeffs, L, C, K, dilation, w);
    report_check("depthwise_same_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

void test_layer_norm_backward() {
    const int L = 3, C = 6;
    const std::vector<double> gamma = make_vector(C, 0.5);
    const std::vector<double> beta  = make_vector(C, 1.7);
    const std::vector<double> coeffs = make_vector(L * C, 2.3);
    const std::vector<double> x0 = make_vector(L * C, 0.2);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, layer_norm_forward(x, L, C, gamma, beta));
    };
    const std::vector<double> analytic = layer_norm_backward_input(x0, L, C, gamma, coeffs);
    report_check("layer_norm_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

void test_gelu_backward() {
    const int n = 16;
    const std::vector<double> coeffs = make_vector(n, 1.3);
    const std::vector<double> x0 = make_vector(n, 0.25);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, gelu_forward(x));
    };
    const std::vector<double> analytic = gelu_backward(x0, coeffs);
    report_check("gelu_backward", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

ConvNextWeights make_convnext(int C, int hidden, int K, int dilation) {
    ConvNextWeights w;
    w.C = C;
    w.hidden = hidden;
    w.K = K;
    w.dilation = dilation;
    w.dw_w = make_vector(C * K, 0.2);
    w.dw_b = make_vector(C, 0.4);
    w.ln_gamma = make_vector(C, 0.6);
    w.ln_beta = make_vector(C, 0.8);
    w.pw1_w = make_vector(hidden * C, 1.0);
    w.pw1_b = make_vector(hidden, 1.2);
    w.pw2_w = make_vector(C * hidden, 1.4);
    w.pw2_b = make_vector(C, 1.6);
    w.gamma = make_vector(C, 1.8);
    return w;
}

void test_convnext_backward() {
    const int L = 6, C = 5, hidden = 9, K = 3, dilation = 2;
    const ConvNextWeights w = make_convnext(C, hidden, K, dilation);
    const std::vector<double> coeffs = make_vector(L * C, 1.5);
    const std::vector<double> x0 = make_vector(L * C, 0.3);

    ConvNextActivations acts;
    convnext_forward(w, x0, L, acts);
    const std::vector<double> analytic = convnext_backward_input(w, acts, coeffs, L);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        ConvNextActivations a;
        return dot(coeffs, convnext_forward(w, x, L, a));
    };
    report_check("convnext_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

CrossAttnWeights make_cross_weights(int C, int Ckv, int A) {
    CrossAttnWeights w;
    w.wq = make_vector(C * A, 0.2);
    w.bq = make_vector(A, 0.3);
    w.wk = make_vector(Ckv * A, 0.4);
    w.bk = make_vector(A, 0.5);
    w.wv = make_vector(Ckv * A, 0.6);
    w.bv = make_vector(A, 0.7);
    w.wo = make_vector(A * C, 0.8);
    w.bo = make_vector(C, 0.9);
    return w;
}

void test_rope_attn_backward() {
    CrossAttnDims dims;
    dims.L = 5;
    dims.Lk = 4;
    dims.C = 8;
    dims.Ckv = 6;
    dims.H = 2;
    dims.D = 4;
    dims.scale = 1.0 / 16.0;
    const int A = dims.H * dims.D;
    const CrossAttnWeights w = make_cross_weights(dims.C, dims.Ckv, A);
    const std::vector<double> theta = make_vector(dims.D / 2, 1.0);
    const std::vector<double> text_lc = make_vector(dims.Lk * dims.Ckv, 0.5);
    const std::vector<double> coeffs = make_vector(dims.L * dims.C, 1.3);
    const std::vector<double> x0 = make_vector(dims.L * dims.C, 0.2);

    RopeAttnActivations acts;
    rope_attn_forward(dims, w, theta, x0, text_lc, acts);
    const std::vector<double> analytic = rope_attn_backward_input(dims, w, theta, acts, coeffs);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        RopeAttnActivations a;
        return dot(coeffs, rope_attn_forward(dims, w, theta, x, text_lc, a));
    };
    report_check("rope_attn d_x", compare_gradients(finite_diff_gradient(f, x0), analytic));
}

void test_style_attn_backward() {
    CrossAttnDims dims;
    dims.L = 5;
    dims.Lk = 4;
    dims.C = 8;
    dims.Ckv = 6;
    dims.H = 2;
    dims.D = 4;
    dims.scale = 1.0 / 16.0;
    const int A = dims.H * dims.D;
    const CrossAttnWeights w = make_cross_weights(dims.C, dims.Ckv, A);
    const std::vector<double> k_const = make_vector(dims.Lk * dims.Ckv, 0.45);
    const std::vector<double> coeffs = make_vector(dims.L * dims.C, 1.3);
    const std::vector<double> x0 = make_vector(dims.L * dims.C, 0.2);
    const std::vector<double> style0 = make_vector(dims.Lk * dims.Ckv, 0.35);

    StyleAttnActivations acts;
    style_attn_forward(dims, w, x0, style0, k_const, acts);
    const StyleAttnGrads grads = style_attn_backward(dims, w, acts, coeffs);

    const ScalarLossFn f_x = [&](const std::vector<double> & x) {
        StyleAttnActivations a;
        return dot(coeffs, style_attn_forward(dims, w, x, style0, k_const, a));
    };
    report_check("style_attn d_x", compare_gradients(finite_diff_gradient(f_x, x0), grads.d_x));

    const ScalarLossFn f_style = [&](const std::vector<double> & style) {
        StyleAttnActivations a;
        return dot(coeffs, style_attn_forward(dims, w, x0, style, k_const, a));
    };
    report_check("style_attn d_style", compare_gradients(finite_diff_gradient(f_style, style0), grads.d_style));
}

// --- full vector field / CFM step -------------------------------------------

struct FieldDims {
    int Cin = 5, C = 8, hidden = 10, K = 3;
    int H = 2, D = 4, Ckv = 6;
    int L = 5, Lk_text = 4, Lk_style = 3;
    double scale = 1.0 / 16.0;
};

ConvNextWeights make_convnext_dil(const FieldDims & d, int dilation, double phase) {
    ConvNextWeights w = make_convnext(d.C, d.hidden, d.K, dilation);
    for (double & v : w.dw_w) v += phase * 0.01;  // perturb so blocks differ
    return w;
}

CrossAttnDims make_attn_dims(const FieldDims & d, int Lk) {
    CrossAttnDims dims;
    dims.L = d.L;
    dims.Lk = Lk;
    dims.C = d.C;
    dims.Ckv = d.Ckv;
    dims.H = d.H;
    dims.D = d.D;
    dims.scale = d.scale;
    return dims;
}

VectorFieldGroupWeights make_group(const FieldDims & d, double phase) {
    VectorFieldGroupWeights g;
    const int dils[4] = {1, 2, 4, 8};
    for (int j = 0; j < 4; ++j) g.convnext_a[j] = make_convnext_dil(d, dils[j], phase + j);
    g.convnext_b = make_convnext_dil(d, 1, phase + 4);
    g.convnext_c = make_convnext_dil(d, 1, phase + 5);
    const int A = d.H * d.D;
    g.rope = make_cross_weights(d.C, d.Ckv, A);
    g.style = make_cross_weights(d.C, d.Ckv, A);
    g.theta = make_vector(d.D / 2, phase + 6);
    g.ln1_gamma = make_vector(d.C, phase + 7);
    g.ln1_beta = make_vector(d.C, phase + 8);
    g.ln2_gamma = make_vector(d.C, phase + 9);
    g.ln2_beta = make_vector(d.C, phase + 10);
    return g;
}

VectorFieldWeights make_field(const FieldDims & d) {
    VectorFieldWeights w;
    w.Cin = d.Cin;
    w.C = d.C;
    w.proj_in_w = make_vector(d.C * d.Cin, 0.2);
    w.proj_out_w = make_vector(d.Cin * d.C, 0.3);
    for (int g = 0; g < 4; ++g) w.groups[g] = make_group(d, 11.0 * (g + 1));
    for (int j = 0; j < 4; ++j) w.last_convnext[j] = make_convnext_dil(d, 1, 80.0 + j);
    w.rope_dims = make_attn_dims(d, d.Lk_text);
    w.style_dims = make_attn_dims(d, d.Lk_style);
    w.mask = make_vector(d.L, 0.15);
    for (double & m : w.mask) m = 0.6 + 0.3 * m;  // keep mask positive, varied
    w.text_lc = make_vector(d.Lk_text * d.Ckv, 0.5);
    w.k_const = make_vector(d.Lk_style * d.Ckv, 0.45);
    return w;
}

void test_vector_field_backward() {
    const FieldDims d;
    const VectorFieldWeights w = make_field(d);
    const std::vector<double> in0 = make_vector(d.L * d.Cin, 0.2);
    const std::vector<double> style0 = make_vector(d.Lk_style * d.Ckv, 0.35);
    const std::vector<double> coeffs = make_vector(d.L * d.Cin, 1.1);

    VectorFieldActivations acts;
    vector_field_forward(w, in0, style0, d.L, acts);
    const VectorFieldGrads grads = vector_field_backward(w, acts, coeffs, d.L);

    const ScalarLossFn f_in = [&](const std::vector<double> & in) {
        VectorFieldActivations a;
        return dot(coeffs, vector_field_forward(w, in, style0, d.L, a));
    };
    report_check("vector_field d_in", compare_gradients(finite_diff_gradient(f_in, in0), grads.d_in));

    const ScalarLossFn f_style = [&](const std::vector<double> & style) {
        VectorFieldActivations a;
        return dot(coeffs, vector_field_forward(w, in0, style, d.L, a));
    };
    report_check("vector_field d_style", compare_gradients(finite_diff_gradient(f_style, style0), grads.d_style));
}

void test_vector_step_backward() {
    const FieldDims d;
    const int total_steps = 4;
    const VectorFieldWeights w = make_field(d);
    const std::vector<double> noisy0 = make_vector(d.Cin * d.L, 0.25);
    const std::vector<double> style0 = make_vector(d.Lk_style * d.Ckv, 0.35);
    const std::vector<double> coeffs = make_vector(d.Cin * d.L, 1.2);

    VectorFieldActivations acts;
    vector_step_forward(w, noisy0, style0, d.L, total_steps, acts);
    const VectorStepGrads grads = vector_step_backward(w, acts, coeffs, d.L, total_steps);

    const ScalarLossFn f_noisy = [&](const std::vector<double> & noisy) {
        VectorFieldActivations a;
        return dot(coeffs, vector_step_forward(w, noisy, style0, d.L, total_steps, a));
    };
    report_check("vector_step d_noisy", compare_gradients(finite_diff_gradient(f_noisy, noisy0), grads.d_noisy));

    const ScalarLossFn f_style = [&](const std::vector<double> & style) {
        VectorFieldActivations a;
        return dot(coeffs, vector_step_forward(w, noisy0, style, d.L, total_steps, a));
    };
    report_check("vector_step d_style", compare_gradients(finite_diff_gradient(f_style, style0), grads.d_style));
}

}  // namespace

int main() {
    try {
        test_conv1x1_backward();
        test_dense_time_backward();
        test_depthwise_backward();
        test_layer_norm_backward();
        test_gelu_backward();
        test_convnext_backward();
        test_rope_attn_backward();
        test_style_attn_backward();
        test_vector_field_backward();
        test_vector_step_backward();
    } catch (const std::exception & e) {
        ++g_failures;
        fprintf(stderr, "FAIL uncaught exception: %s\n", e.what());
    }
    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
