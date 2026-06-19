#include "voiceclone_gradcheck.h"

#include <cmath>
#include <limits>

namespace tts_cpp {
namespace voiceclone {

namespace {

// Central finite difference of f along coordinate i: (f(x+eps*e_i) - f(x-eps*e_i)) / 2*eps.
double partial_derivative(const ScalarLossFn & f, const std::vector<double> & x,
                          std::size_t i, double eps) {
    std::vector<double> probe = x;
    probe[i] = x[i] + eps;
    const double f_plus = f(probe);
    probe[i] = x[i] - eps;
    const double f_minus = f(probe);
    return (f_plus - f_minus) / (2.0 * eps);
}

// Folds one component's error into the running report: refreshes the running
// maxima (tracking the worst relative component) and clears `passed` when the
// component falls outside abs_tol + rel_tol*|numeric|.
void update_report_with_component(GradcheckReport & report, std::size_t i,
                                  double numeric, double analytic,
                                  double rel_tol, double abs_tol) {
    const double abs_err = std::fabs(analytic - numeric);
    const double rel_err = abs_err / (std::fabs(numeric) + abs_tol);
    if (abs_err > report.max_abs_err) {
        report.max_abs_err = abs_err;
    }
    if (rel_err > report.max_rel_err) {
        report.max_rel_err = rel_err;
        report.worst_index = i;
    }
    if (abs_err > abs_tol + rel_tol * std::fabs(numeric)) {
        report.passed = false;
    }
}

// Folds every component's error into the report (the per-element traversal).
void accumulate_component_errors(GradcheckReport & report,
                                 const std::vector<double> & numeric,
                                 const std::vector<double> & analytic,
                                 double rel_tol, double abs_tol) {
    for (std::size_t i = 0; i < numeric.size(); ++i) {
        update_report_with_component(report, i, numeric[i], analytic[i], rel_tol, abs_tol);
    }
}

}  // namespace

std::vector<double> finite_diff_gradient(const ScalarLossFn & f,
                                         const std::vector<double> & x,
                                         double eps) {
    std::vector<double> grad(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        grad[i] = partial_derivative(f, x, i, eps);
    }
    return grad;
}

GradcheckReport compare_gradients(const std::vector<double> & numeric,
                                  const std::vector<double> & analytic,
                                  double rel_tol,
                                  double abs_tol) {
    GradcheckReport report;
    report.n = numeric.size();
    if (analytic.size() != numeric.size()) {
        // Size mismatch is a hard failure: the caller wired the wrong gradient.
        report.passed = false;
        report.max_abs_err = std::numeric_limits<double>::infinity();
        report.max_rel_err = std::numeric_limits<double>::infinity();
        return report;
    }

    report.passed = true;
    accumulate_component_errors(report, numeric, analytic, rel_tol, abs_tol);
    return report;
}

GradcheckReport gradcheck(const ScalarLossFn & f,
                          const std::vector<double> & x,
                          const std::vector<double> & analytic_grad,
                          double eps,
                          double rel_tol,
                          double abs_tol) {
    return compare_gradients(finite_diff_gradient(f, x, eps), analytic_grad, rel_tol, abs_tol);
}

}  // namespace voiceclone
}  // namespace tts_cpp
