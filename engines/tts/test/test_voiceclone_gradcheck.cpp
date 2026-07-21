// Self-tests for the voice-clone finite-difference gradient checker.
// Pure host logic — no model, no fixture — so it builds and runs
// standalone and in the always-on `unit` ctest tier:
//
//   g++ -std=c++17 -I src test/test_voiceclone_gradcheck.cpp src/voiceclone_gradcheck.cpp -o /tmp/t
//   /tmp/t
//
// The harness is itself a test tool, so these self-tests verify it on BOTH
// sides: it must accept a known-correct analytic gradient AND reject a wrong
// one.  A checker that only ever returns "pass" would be worse than useless —
// it would green-light a broken backward pass in a later cloning task.
//
// Coverage:
//   - finite_diff_gradient matches closed-form gradients of quadratic and
//     transcendental losses to central-difference accuracy.
//   - gradcheck passes the correct gradient and fails a perturbed / wrong /
//     mis-sized one.
//   - tolerances and the worst-component report behave as documented.

#include "voiceclone_gradcheck.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace tts_cpp::voiceclone;

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

// f(x) = sum_i a_i * x_i^2 + b_i * x_i  ->  df/dx_i = 2 a_i x_i + b_i
// std::array (not std::vector) so the namespace-scope constants need no heap
// allocation and cannot throw during static initialization.
constexpr std::array<double, 4> kA = {1.5, -0.5, 3.0, 0.25};
constexpr std::array<double, 4> kB = {0.0, 2.0, -1.0, 4.0};

double quadratic(const std::vector<double> & x) {
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        s += kA[i] * x[i] * x[i] + kB[i] * x[i];
    }
    return s;
}

std::vector<double> quadratic_grad(const std::vector<double> & x) {
    std::vector<double> g(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        g[i] = 2.0 * kA[i] * x[i] + kB[i];
    }
    return g;
}

// A transcendental loss to exercise non-polynomial curvature.
// f(x) = sum_i sin(x_i) * exp(0.1 x_i)
//   df/dx_i = cos(x_i) exp(0.1 x_i) + 0.1 sin(x_i) exp(0.1 x_i)
double transcendental(const std::vector<double> & x) {
    double s = 0.0;
    for (double v : x) s += std::sin(v) * std::exp(0.1 * v);
    return s;
}

std::vector<double> transcendental_grad(const std::vector<double> & x) {
    std::vector<double> g(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const double e = std::exp(0.1 * x[i]);
        g[i] = std::cos(x[i]) * e + 0.1 * std::sin(x[i]) * e;
    }
    return g;
}

void test_finite_diff_matches_closed_form() {
    const std::vector<double> x = {0.3, -1.2, 0.7, 2.1};
    const std::vector<double> num = finite_diff_gradient(quadratic, x);
    const std::vector<double> ana = quadratic_grad(x);
    for (size_t i = 0; i < x.size(); ++i) {
        CHECK(std::fabs(num[i] - ana[i]) < 1e-6,
              "quadratic fd grad[%zu]: num=%.9f ana=%.9f", i, num[i], ana[i]);
    }
}

void test_gradcheck_accepts_correct_gradient() {
    const std::vector<double> x = {0.3, -1.2, 0.7, 2.1};
    const GradcheckReport r = gradcheck(quadratic, x, quadratic_grad(x));
    CHECK(r.passed, "gradcheck should accept the exact quadratic gradient");
    CHECK(r.n == x.size(), "report.n");
    CHECK(r.max_rel_err < 1e-4, "max_rel_err small for correct grad: %.3e", r.max_rel_err);
}

void test_gradcheck_accepts_transcendental_gradient() {
    const std::vector<double> x = {0.1, 0.9, -0.4, 1.7};
    const GradcheckReport r = gradcheck(transcendental, x, transcendental_grad(x));
    CHECK(r.passed, "gradcheck should accept the transcendental gradient (max_rel=%.3e)",
          r.max_rel_err);
}

void test_gradcheck_rejects_wrong_gradient() {
    const std::vector<double> x = {0.3, -1.2, 0.7, 2.1};
    // Flip the sign of the whole gradient: clearly wrong, must be rejected.
    std::vector<double> wrong = quadratic_grad(x);
    for (double & g : wrong) g = -g;
    const GradcheckReport r = gradcheck(quadratic, x, wrong);
    CHECK(!r.passed, "gradcheck must reject a sign-flipped gradient");
}

void test_gradcheck_rejects_single_bad_component() {
    const std::vector<double> x = {0.3, -1.2, 0.7, 2.1};
    std::vector<double> almost = quadratic_grad(x);
    almost[2] += 0.5;  // one component off by a wide margin
    const GradcheckReport r = gradcheck(quadratic, x, almost);
    CHECK(!r.passed, "gradcheck must reject a gradient with one bad component");
    CHECK(r.worst_index == 2, "worst component should be index 2, got %zu", r.worst_index);
}

void test_gradcheck_rejects_size_mismatch() {
    const std::vector<double> x = {0.3, -1.2, 0.7, 2.1};
    const std::vector<double> wrong_size = {0.0, 0.0};
    const GradcheckReport r = gradcheck(quadratic, x, wrong_size);
    CHECK(!r.passed, "gradcheck must reject a mis-sized gradient");
}

void test_gradcheck_tolerance_is_meaningful() {
    // A tiny perturbation within tolerance passes; a large one outside fails.
    const std::vector<double> x = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> tiny = quadratic_grad(x);
    for (double & g : tiny) g += 1e-9;
    CHECK(gradcheck(quadratic, x, tiny).passed,
          "a 1e-9 perturbation should pass the default tolerance");

    std::vector<double> big = quadratic_grad(x);
    for (double & g : big) g += 1e-1;
    CHECK(!gradcheck(quadratic, x, big).passed,
          "a 1e-1 perturbation should fail the default tolerance");
}

}  // namespace

int main() {
    test_finite_diff_matches_closed_form();
    test_gradcheck_accepts_correct_gradient();
    test_gradcheck_accepts_transcendental_gradient();
    test_gradcheck_rejects_wrong_gradient();
    test_gradcheck_rejects_single_bad_component();
    test_gradcheck_rejects_size_mismatch();
    test_gradcheck_tolerance_is_meaningful();

    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL",
            g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
