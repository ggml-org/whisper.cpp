// Gradcheck self-tests for the Supertonic text-encoder backward (voice-clone
// ticket "GGML backward pass: Supertonic text encoder").  Pure host logic,
// model-free: every analytic gradient is checked component-wise against a
// central finite-difference numeric gradient of the matching forward.  Runs in
// the always-on `unit` ctest tier.
//
// Standalone build (single line):
//   g++ -std=c++17 -I src test/test_supertonic_text_encoder_backward.cpp src/supertonic_text_encoder_backward.cpp src/voiceclone_gradcheck.cpp -o /tmp/t && /tmp/t

#include "supertonic_text_encoder_backward.h"
#include "voiceclone_gradcheck.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace tts_cpp::supertonic_grad;
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

void test_dense_time_backward() {
    const int L = 3, in_dim = 4, out_dim = 5;
    LinearWeights w;
    w.in_dim = in_dim;
    w.out_dim = out_dim;
    w.w = make_vector(in_dim * out_dim, 0.3);
    w.b = make_vector(out_dim, 1.1);

    const std::vector<double> coeffs = make_vector(L * out_dim, 2.0);
    const std::vector<double> x0 = make_vector(L * in_dim, 0.7);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, dense_time_forward(x, L, w));
    };
    const std::vector<double> analytic = dense_time_backward_input(coeffs, L, w);
    const std::vector<double> numeric  = finite_diff_gradient(f, x0);
    report_check("dense_time_backward_input", compare_gradients(numeric, analytic));
}

void test_layer_norm_backward() {
    const int L = 3, C = 6;
    const std::vector<double> gamma = make_vector(C, 0.5);
    const std::vector<double> beta  = make_vector(C, 1.7);
    const std::vector<double> coeffs = make_vector(L * C, 2.3);
    const std::vector<double> x0 = make_vector(L * C, 0.2);

    const ScalarLossFn f = [&](const std::vector<double> & x) {
        return dot(coeffs, layer_norm_channel_forward(x, L, C, gamma, beta));
    };
    const std::vector<double> analytic = layer_norm_channel_backward(x0, L, C, gamma, coeffs);
    const std::vector<double> numeric  = finite_diff_gradient(f, x0);
    report_check("layer_norm_channel_backward", compare_gradients(numeric, analytic));
}

LinearWeights make_linear(int in_dim, int out_dim, double phase) {
    LinearWeights w;
    w.in_dim = in_dim;
    w.out_dim = out_dim;
    w.w = make_vector(in_dim * out_dim, phase);
    w.b = make_vector(out_dim, phase + 0.5);
    return w;
}

SpeechAttentionWeights make_attention_weights(const SpeechAttentionDims & dims) {
    SpeechAttentionWeights w;
    w.q = make_linear(dims.C, dims.C, 0.3);
    w.v = make_linear(dims.C, dims.C, 0.6);
    w.o = make_linear(dims.C, dims.C, 0.9);
    w.tanh_k = make_vector(dims.heads * (dims.C / dims.heads) * dims.Lctx, 1.2);
    return w;
}

void test_speech_attention_backward() {
    SpeechAttentionDims dims;
    dims.L = 4;
    dims.Lctx = 3;
    dims.C = 6;
    dims.heads = 2;
    dims.scale = 0.25;
    const SpeechAttentionWeights w = make_attention_weights(dims);

    const std::vector<double> x0     = make_vector(dims.L * dims.C, 0.2);
    const std::vector<double> style0 = make_vector(dims.Lctx * dims.C, 0.4);
    const std::vector<double> coeffs = make_vector(dims.L * dims.C, 1.5);

    SpeechAttentionActivations acts;
    speech_attention_forward(dims, w, x0, style0, acts);
    const SpeechAttentionGrads grads = speech_attention_backward(dims, w, acts, coeffs);

    const ScalarLossFn f_style = [&](const std::vector<double> & style) {
        SpeechAttentionActivations a;
        return dot(coeffs, speech_attention_forward(dims, w, x0, style, a));
    };
    report_check("speech_attention d_style",
                 compare_gradients(finite_diff_gradient(f_style, style0), grads.d_style));

    const ScalarLossFn f_x = [&](const std::vector<double> & x) {
        SpeechAttentionActivations a;
        return dot(coeffs, speech_attention_forward(dims, w, x, style0, a));
    };
    report_check("speech_attention d_x",
                 compare_gradients(finite_diff_gradient(f_x, x0), grads.d_x));
}

void test_speech_tail_backward() {
    SpeechAttentionDims dims;
    dims.L = 4;
    dims.Lctx = 3;
    dims.C = 6;
    dims.heads = 2;
    dims.scale = 0.25;

    SpeechTailWeights w;
    w.spa[0] = make_attention_weights(dims);
    w.spa[1] = make_attention_weights(dims);
    // Distinguish the second layer's weights so a bug confined to one layer is visible.
    for (double & v : w.spa[1].q.w) v *= 0.7;
    for (double & v : w.spa[1].v.w) v *= 1.3;
    w.ln_gamma = make_vector(dims.C, 0.5);
    w.ln_beta  = make_vector(dims.C, 1.1);

    const std::vector<double> stack_out = make_vector(dims.L * dims.C, 0.15);
    const std::vector<double> style0    = make_vector(dims.Lctx * dims.C, 0.35);
    const std::vector<double> coeffs    = make_vector(dims.L * dims.C, 1.4);

    SpeechTailActivations acts;
    speech_tail_forward(dims, w, stack_out, style0, acts);
    const std::vector<double> analytic = speech_tail_backward(dims, w, acts, coeffs);

    const ScalarLossFn f_style = [&](const std::vector<double> & style) {
        SpeechTailActivations a;
        return dot(coeffs, speech_tail_forward(dims, w, stack_out, style, a));
    };
    report_check("speech_tail d_style",
                 compare_gradients(finite_diff_gradient(f_style, style0), analytic));
}

}  // namespace

int main() {
    try {
        test_dense_time_backward();
        test_layer_norm_backward();
        test_speech_attention_backward();
        test_speech_tail_backward();
    } catch (const std::exception & e) {
        ++g_failures;
        fprintf(stderr, "FAIL uncaught exception: %s\n", e.what());
    }
    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
