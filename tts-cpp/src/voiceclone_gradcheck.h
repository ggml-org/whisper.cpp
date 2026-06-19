#pragma once

// Finite-difference gradient checker for the voice-clone inverse-optimization
// stack (QVAC-20979).
//
// The voice-cloning method we are building extracts a Supertonic style vector
// (`style_ttl`, see the QVAC voice-clone design / Kim, "Extracting Voice Styles
// from Frozen TTS Models via Gradient-Based Inverse Optimization") by running
// gradient descent on the style vector through the *frozen* TTS pipeline.  That
// requires an analytic backward pass for every Supertonic stage we differentiate
// through.  This module is the validation tool those future analytic gradients
// are checked against: it estimates the gradient of a scalar loss numerically
// (central differences) and compares it component-wise to the analytic gradient.
//
// It deliberately knows nothing about Supertonic, ggml, or any model: it takes a
// scalar loss callback and a candidate gradient, so the same harness validates a
// single stage, a fused sub-graph, or the whole end-to-end pipeline.

#include <cstddef>
#include <functional>
#include <vector>

namespace tts_cpp {
namespace voiceclone {

// Scalar loss over a flat parameter vector.  Returning double (not float) keeps
// the central-difference subtraction well-conditioned for small `eps`.
using ScalarLossFn = std::function<double(const std::vector<double> &)>;

// Central-difference numerical gradient: g[i] = (f(x + eps e_i) - f(x - eps e_i))
// / (2 eps).  Central differences have O(eps^2) truncation error vs the O(eps)
// of a one-sided difference, so the same `eps` yields a much tighter reference.
std::vector<double> finite_diff_gradient(const ScalarLossFn & f,
                                         const std::vector<double> & x,
                                         double eps = 1e-4);

struct GradcheckReport {
    std::size_t n           = 0;      // number of parameters checked
    std::size_t worst_index = 0;      // component with the largest relative error
    double max_abs_err      = 0.0;    // max_i |analytic_i - numeric_i|
    double max_rel_err      = 0.0;    // max_i |analytic_i - numeric_i| / (|numeric_i| + abs_tol)
    bool   passed           = false;  // every component within (abs_tol + rel_tol*|numeric_i|)
};

// Compare an analytic gradient against a precomputed numeric gradient.
//
// A component passes when |analytic_i - numeric_i| <= abs_tol + rel_tol*|numeric_i|.
// The mixed absolute/relative tolerance keeps near-zero components (where a pure
// relative test explodes) and large components (where a pure absolute test is
// too strict) both meaningful.  `passed` is the AND over all components.
GradcheckReport compare_gradients(const std::vector<double> & numeric,
                                  const std::vector<double> & analytic,
                                  double rel_tol = 1e-4,
                                  double abs_tol = 1e-6);

// Convenience: estimate the numeric gradient of `f` at `x` and compare the
// analytic gradient against it in one call.
GradcheckReport gradcheck(const ScalarLossFn & f,
                          const std::vector<double> & x,
                          const std::vector<double> & analytic_grad,
                          double eps     = 1e-4,
                          double rel_tol = 1e-4,
                          double abs_tol = 1e-6);

}  // namespace voiceclone
}  // namespace tts_cpp
