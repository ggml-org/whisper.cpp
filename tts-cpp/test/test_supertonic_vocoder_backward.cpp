// Gradcheck self-tests for the Supertonic vocoder backward (voice-clone ticket
// "GGML backward pass: vocoder", QVAC-20983). Pure host logic, model-free: every
// analytic gradient is checked component-wise against a central finite-difference
// numeric gradient of the matching forward. Runs in the always-on `unit` ctest
// tier.
//
// Standalone build (single line):
//   g++ -std=c++17 -I src test/test_supertonic_vocoder_backward.cpp \
//       src/supertonic_vocoder_backward.cpp \
//       src/supertonic_vector_estimator_backward.cpp \
//       src/voiceclone_gradcheck.cpp -o /tmp/t && /tmp/t

#include "supertonic_vocoder_backward.h"
#include "voiceclone_gradcheck.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

using namespace tts_cpp::voc_grad;
using tts_cpp::voiceclone::compare_gradients;
using tts_cpp::voiceclone::finite_diff_gradient;
using tts_cpp::voiceclone::GradcheckReport;
using tts_cpp::voiceclone::ScalarLossFn;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, ...) do {                                  \
    ++g_checks;                                                \
    if (!(cond)) {                                             \
        ++g_failures;                                          \
        fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
    }                                                          \
} while (0)

double sample(int i, double phase) {
    return std::sin(i * 0.9 + phase) * 0.8;
}

std::vector<double> make_vector(int n, double phase) {
    std::vector<double> v((std::size_t) n);
    for (int i = 0; i < n; ++i) v[i] = sample(i, phase);
    return v;
}

// Strictly positive samples for variance-like quantities (running_var).
std::vector<double> make_positive(int n, double phase) {
    std::vector<double> v((std::size_t) n);
    for (int i = 0; i < n; ++i) v[i] = 0.5 + 0.4 * (sample(i, phase) + 1.0);
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

VocConvNextWeights make_block(int C, int hidden, int K, int dilation, double phase) {
    VocConvNextWeights w;
    w.C        = C;
    w.hidden   = hidden;
    w.K        = K;
    w.dilation = dilation;
    w.dw_w     = make_vector(C * K, phase + 0.1);
    w.dw_b     = make_vector(C, phase + 0.2);
    w.ln_gamma = make_vector(C, phase + 0.3);
    w.ln_beta  = make_vector(C, phase + 0.4);
    w.pw1_w    = make_vector(hidden * C, phase + 0.5);
    w.pw1_b    = make_vector(hidden, phase + 0.6);
    w.pw2_w    = make_vector(C * hidden, phase + 0.7);
    w.pw2_b    = make_vector(C, phase + 0.8);
    w.gamma    = 0.6;
    return w;
}

VocoderWeights make_vocoder() {
    VocoderWeights w;
    w.latent_len = 2;
    w.C_latent   = 3;
    w.factor     = 2;  // latent_channels = 6, T0 = 4
    w.C          = 5;
    w.normalizer_scale = 1.7;
    w.latent_mean = make_vector(w.C_latent, 0.2);
    w.latent_std  = make_vector(w.C_latent, 0.5);

    w.K_embed = 3;
    w.embed_w = make_vector(w.C * w.C_latent * w.K_embed, 0.9);
    w.embed_b = make_vector(w.C, 1.0);

    // Three ConvNeXt blocks exercising the production dilation variety (1/2/4).
    const int hidden = 7, Kdw = 3;
    w.convnext.push_back(make_block(w.C, hidden, Kdw, 1, 1.0));
    w.convnext.push_back(make_block(w.C, hidden, Kdw, 2, 2.0));
    w.convnext.push_back(make_block(w.C, hidden, Kdw, 4, 3.0));

    w.bn_gamma        = make_vector(w.C, 0.4);
    w.bn_beta         = make_vector(w.C, 0.7);
    w.bn_running_mean = make_vector(w.C, 0.1);
    w.bn_running_var  = make_positive(w.C, 0.5);

    w.Hh      = 6;
    w.K_head1 = 3;
    w.head1_w = make_vector(w.Hh * w.C * w.K_head1, 0.3);
    w.head1_b = make_vector(w.Hh, 0.45);
    w.prelu_slope = 0.1;

    w.OUT     = 1;
    w.head2_w = make_vector(w.OUT * w.Hh, 0.65);
    return w;
}

}  // namespace

namespace tts_cpp {
namespace voc_grad {

// Friend of VocoderBackward: validates the private math primitives individually
// (and the full forward/backward chain) against finite differences. Declared a
// friend so the gradchecks reach the primitives without widening the public API.
struct VocoderBackwardTester {
    // The primitives are pure (ignore weights), so a default-constructed instance
    // is all the friend tester needs to exercise them.
    static VocoderBackward op() { return VocoderBackward{VocoderWeights{}}; }

    static void test_denorm_backward() {
        const int L = 4, C = 3;
        const double normalizer_scale = 1.7;
        const std::vector<double> std    = make_vector(C, 0.4);
        const std::vector<double> mean   = make_vector(C, 1.2);
        const std::vector<double> coeffs = make_vector(L * C, 2.0);
        const std::vector<double> x0     = make_vector(L * C, 0.7);

        const VocoderBackward vb = op();
        const ScalarLossFn f = [&](const std::vector<double> & x) {
            return dot(coeffs, vb.denorm_forward(x, L, C, normalizer_scale, std, mean));
        };
        const std::vector<double> analytic = vb.denorm_backward_input(coeffs, L, C, normalizer_scale, std);
        report_check("denorm_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
    }

    static void test_conv1d_causal_backward() {
        const int L = 6, IC = 4, OC = 5, K = 3;
        const std::vector<double> w      = make_vector(OC * IC * K, 0.3);
        const std::vector<double> b      = make_vector(OC, 1.1);
        const std::vector<double> coeffs = make_vector(L * OC, 2.0);
        const std::vector<double> x0     = make_vector(L * IC, 0.7);

        const VocoderBackward vb = op();
        const ScalarLossFn f = [&](const std::vector<double> & x) {
            return dot(coeffs, vb.conv1d_causal_forward(x, L, IC, OC, K, w, b));
        };
        const std::vector<double> analytic = vb.conv1d_causal_backward_input(coeffs, L, IC, OC, K, w);
        report_check("conv1d_causal_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
    }

    static void test_depthwise_causal_backward() {
        const int L = 7, C = 4, K = 3, dilation = 2;
        const std::vector<double> w      = make_vector(C * K, 0.4);
        const std::vector<double> b      = make_vector(C, 0.9);
        const std::vector<double> coeffs = make_vector(L * C, 1.6);
        const std::vector<double> x0     = make_vector(L * C, 0.5);

        const VocoderBackward vb = op();
        const ScalarLossFn f = [&](const std::vector<double> & x) {
            return dot(coeffs, vb.depthwise_causal_forward(x, L, C, K, dilation, w, b));
        };
        const std::vector<double> analytic = vb.depthwise_causal_backward_input(coeffs, L, C, K, dilation, w);
        report_check("depthwise_causal_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
    }

    static void test_batch_norm_backward() {
        const int L = 5, C = 4;
        const std::vector<double> gamma = make_vector(C, 0.5);
        const std::vector<double> beta  = make_vector(C, 1.7);
        const std::vector<double> rmean = make_vector(C, 0.3);
        const std::vector<double> rvar  = make_positive(C, 0.8);
        const std::vector<double> coeffs = make_vector(L * C, 2.3);
        const std::vector<double> x0     = make_vector(L * C, 0.2);

        const VocoderBackward vb = op();
        const ScalarLossFn f = [&](const std::vector<double> & x) {
            return dot(coeffs, vb.batch_norm_forward(x, L, C, gamma, beta, rmean, rvar));
        };
        const std::vector<double> analytic = vb.batch_norm_backward_input(coeffs, L, C, gamma, rvar);
        report_check("batch_norm_backward_input", compare_gradients(finite_diff_gradient(f, x0), analytic));
    }

    static void test_leaky_relu_backward() {
        const int n = 16;
        const double slope = 0.1;
        const std::vector<double> coeffs = make_vector(n, 1.3);
        const std::vector<double> x0     = make_vector(n, 0.6);

        const VocoderBackward vb = op();
        const ScalarLossFn f = [&](const std::vector<double> & x) {
            return dot(coeffs, vb.leaky_relu_forward(x, slope));
        };
        const std::vector<double> analytic = vb.leaky_relu_backward(x0, coeffs, slope);
        // The kink at 0 is measure-zero for these samples; central diff is exact away from it.
        report_check("leaky_relu_backward", compare_gradients(finite_diff_gradient(f, x0), analytic));
    }

    static void test_latent_unpack_backward() {
        const int latent_len = 3, C_latent = 4, factor = 2;
        const int T0 = latent_len * factor;
        const int latent_channels = C_latent * factor;
        const std::vector<double> coeffs = make_vector(T0 * C_latent, 1.4);
        const std::vector<double> latent0 = make_vector(latent_channels * latent_len, 0.5);

        const VocoderBackward vb = op();
        const ScalarLossFn f = [&](const std::vector<double> & latent) {
            return dot(coeffs, vb.latent_unpack_forward(latent, latent_len, C_latent, factor));
        };
        const std::vector<double> analytic = vb.latent_unpack_backward(coeffs, latent_len, C_latent, factor);
        report_check("latent_unpack_backward", compare_gradients(finite_diff_gradient(f, latent0), analytic));
    }

    static void test_convnext_backward() {
        const int L = 6, C = 5, hidden = 7, K = 3, dilation = 2;
        const VocConvNextWeights w = make_block(C, hidden, K, dilation, 0.0);
        const std::vector<double> coeffs = make_vector(L * C, 1.1);
        const std::vector<double> x0     = make_vector(L * C, 0.3);

        const VocoderBackward vb = op();
        VocConvNextActivations acts;
        vb.convnext_forward(w, x0, L, acts);
        const std::vector<double> analytic = vb.convnext_backward_input(w, acts, coeffs, L);

        const ScalarLossFn f = [&](const std::vector<double> & x) {
            VocConvNextActivations a;
            return dot(coeffs, vb.convnext_forward(w, x, L, a));
        };
        report_check("voc_convnext d_x", compare_gradients(finite_diff_gradient(f, x0), analytic));
    }

    static void test_vocoder_backward() {
        const VocoderWeights w = make_vocoder();
        const int T0 = w.latent_len * w.factor;
        const int latent_channels = w.C_latent * w.factor;
        const std::vector<double> coeffs  = make_vector(T0 * w.OUT, 1.2);
        const std::vector<double> latent0 = make_vector(latent_channels * w.latent_len, 0.4);

        VocoderBackward vb(w);
        vb.forward(latent0);
        const std::vector<double> analytic = vb.backward(coeffs);

        const ScalarLossFn f = [&](const std::vector<double> & latent) {
            VocoderBackward local(w);
            return dot(coeffs, local.forward(latent));
        };
        report_check("vocoder d_latent", compare_gradients(finite_diff_gradient(f, latent0), analytic));
    }
};

}  // namespace voc_grad
}  // namespace tts_cpp

int main() {
    using tts_cpp::voc_grad::VocoderBackwardTester;
    try {
        VocoderBackwardTester::test_denorm_backward();
        VocoderBackwardTester::test_conv1d_causal_backward();
        VocoderBackwardTester::test_depthwise_causal_backward();
        VocoderBackwardTester::test_batch_norm_backward();
        VocoderBackwardTester::test_leaky_relu_backward();
        VocoderBackwardTester::test_latent_unpack_backward();
        VocoderBackwardTester::test_convnext_backward();
        VocoderBackwardTester::test_vocoder_backward();
    } catch (const std::exception & e) {
        ++g_failures;
        fprintf(stderr, "FAIL uncaught exception: %s\n", e.what());
    }
    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
