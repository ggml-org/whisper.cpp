#include "voiceclone_metrics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace tts_cpp {
namespace voiceclone {

namespace {

// --- small numeric primitives ----------------------------------------------

// Dot product accumulated in double regardless of the float32 inputs.
double dot_product(const std::vector<float> & a, const std::vector<float> & b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return acc;
}

double l2_norm(const std::vector<float> & v) {
    return std::sqrt(dot_product(v, v));
}

// Sum of squared component differences of two equal-length vectors.
double sum_squared_diff(const std::vector<double> & a, const std::vector<double> & b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

// --- time-averaged statistics helpers --------------------------------------

const float * row_ptr(const std::vector<float> & H, int t, int D) {
    return H.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(D);
}

// Adds one frame's D features into the running per-feature accumulator.
void accumulate_row(std::vector<double> & acc, const float * row, int D) {
    for (int d = 0; d < D; ++d) {
        acc[d] += static_cast<double>(row[d]);
    }
}

// Adds one frame's centered squared deviations (row[d] - mu[d])^2 into acc.
void accumulate_centered_sq(std::vector<double> & acc, const float * row,
                            const std::vector<double> & mu, int D) {
    for (int d = 0; d < D; ++d) {
        const double diff = static_cast<double>(row[d]) - mu[d];
        acc[d] += diff * diff;
    }
}

// Divides every element by `denom` in place.
void divide_in_place(std::vector<double> & v, double denom) {
    for (double & x : v) {
        x /= denom;
    }
}

// Replaces every element with sqrt(element / T) in place (mean -> std step).
void sqrt_of_mean_in_place(std::vector<double> & v, int T) {
    for (double & x : v) {
        x = std::sqrt(x / T);
    }
}

// Per-feature sum over the T frames.
std::vector<double> column_sum(const std::vector<float> & H, int T, int D) {
    std::vector<double> acc(static_cast<std::size_t>(D), 0.0);
    for (int t = 0; t < T; ++t) {
        accumulate_row(acc, row_ptr(H, t, D), D);
    }
    return acc;
}

// Per-feature sum of centered squared deviations over the T frames.
std::vector<double> column_centered_sq_sum(const std::vector<float> & H, int T, int D,
                                           const std::vector<double> & mu) {
    std::vector<double> acc(static_cast<std::size_t>(D), 0.0);
    for (int t = 0; t < T; ++t) {
        accumulate_centered_sq(acc, row_ptr(H, t, D), mu, D);
    }
    return acc;
}

// Per-feature mean over the T frames.
std::vector<double> mean_over_time(const std::vector<float> & H, int T, int D) {
    std::vector<double> mu = column_sum(H, T, D);
    divide_in_place(mu, T);
    return mu;
}

// Per-feature population std (1/T): sqrt(mean_t (H[t,d] - mu[d])^2).
std::vector<double> std_over_time(const std::vector<float> & H, int T, int D,
                                  const std::vector<double> & mu) {
    std::vector<double> sigma = column_centered_sq_sum(H, T, D, mu);
    sqrt_of_mean_in_place(sigma, T);
    return sigma;
}

// --- WER helpers ------------------------------------------------------------

// Cost of the best edit reaching cell (i, j) given its three predecessors.
int best_edit_cost(int diagonal, int up, int left, bool tokens_match) {
    const int sub = diagonal + (tokens_match ? 0 : 1);
    const int del = up + 1;
    const int ins = left + 1;
    return std::min(sub, std::min(del, ins));
}

// Initializes row 0: inserting j tokens into an empty string costs j edits, so
// dp[0][j] = j.
void init_distance_base_row(std::vector<int> & row) {
    for (std::size_t j = 0; j < row.size(); ++j) {
        row[j] = static_cast<int>(j);
    }
}

// Initializes column 0: turning a string into an empty one costs one deletion
// per token, so dp[i][0] = i.
void init_distance_base_column(std::vector<std::vector<int>> & dp) {
    for (std::size_t i = 0; i < dp.size(); ++i) {
        dp[i][0] = static_cast<int>(i);
    }
}

// Fills one DP row left to right; each cell reads the cell to its left, so this
// must run in order (which a sequential for guarantees).
void fill_distance_row(std::vector<std::vector<int>> & dp, int i,
                       const std::vector<std::string> & ref,
                       const std::vector<std::string> & hyp) {
    const int m = static_cast<int>(hyp.size());
    for (int j = 1; j <= m; ++j) {
        dp[i][j] = best_edit_cost(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1],
                                  ref[i - 1] == hyp[j - 1]);
    }
}

// Standard Levenshtein DP table: dp[i][j] = edit distance between ref[0..i) and
// hyp[0..j).
std::vector<std::vector<int>> build_edit_distance_table(
        const std::vector<std::string> & ref,
        const std::vector<std::string> & hyp) {
    const int n = static_cast<int>(ref.size());
    const int m = static_cast<int>(hyp.size());
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    init_distance_base_row(dp[0]);
    init_distance_base_column(dp);
    for (int i = 1; i <= n; ++i) {
        fill_distance_row(dp, i, ref, hyp);
    }
    return dp;
}

// Backtrack the finished DP table to attribute each edit to S / D / I.  Ties are
// broken substitution > deletion > insertion, which keeps the reported counts
// deterministic.
void attribute_edits(const std::vector<std::vector<int>> & dp,
                     const std::vector<std::string> & ref,
                     const std::vector<std::string> & hyp,
                     WerResult & out) {
    int i = static_cast<int>(ref.size());
    int j = static_cast<int>(hyp.size());
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0) {
            const int cost = (ref[i - 1] == hyp[j - 1]) ? 0 : 1;
            if (dp[i][j] == dp[i - 1][j - 1] + cost) {
                if (cost == 1) ++out.substitutions;
                --i; --j;
                continue;
            }
        }
        if (i > 0 && dp[i][j] == dp[i - 1][j] + 1) {
            ++out.deletions;
            --i;
            continue;
        }
        ++out.insertions;
        --j;
    }
}

}  // namespace

double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("cosine_similarity: size mismatch");
    }
    if (a.empty()) {
        throw std::invalid_argument("cosine_similarity: empty vectors");
    }
    const double denom = l2_norm(a) * l2_norm(b);
    if (denom <= 0.0) {
        return 0.0;
    }
    return dot_product(a, b) / denom;
}

TimeAvgStats time_avg_stats(const std::vector<float> & H_t_by_d, int T, int D) {
    if (T <= 0 || D <= 0) {
        throw std::invalid_argument("time_avg_stats: T and D must be positive");
    }
    if (H_t_by_d.size() != static_cast<std::size_t>(T) * static_cast<std::size_t>(D)) {
        throw std::invalid_argument("time_avg_stats: data size != T*D");
    }
    TimeAvgStats s;
    s.mu = mean_over_time(H_t_by_d, T, D);
    s.sigma = std_over_time(H_t_by_d, T, D, s.mu);
    return s;
}

double wavlm_style_loss(const TimeAvgStats & a, const TimeAvgStats & b) {
    if (a.mu.size() != b.mu.size() || a.sigma.size() != b.sigma.size()) {
        throw std::invalid_argument("wavlm_style_loss: stat dimensionality mismatch");
    }
    return sum_squared_diff(a.mu, b.mu) + sum_squared_diff(a.sigma, b.sigma);
}

double wavlm_style_loss(const std::vector<float> & Ha, int Ta,
                        const std::vector<float> & Hb, int Tb,
                        int D) {
    return wavlm_style_loss(time_avg_stats(Ha, Ta, D), time_avg_stats(Hb, Tb, D));
}

WerResult word_error_rate(const std::vector<std::string> & ref,
                          const std::vector<std::string> & hyp) {
    WerResult result;
    result.ref_len = static_cast<int>(ref.size());

    if (ref.empty()) {
        result.insertions = static_cast<int>(hyp.size());
        result.wer = hyp.empty() ? 0.0 : static_cast<double>(hyp.size());
        return result;
    }

    const std::vector<std::vector<int>> dp = build_edit_distance_table(ref, hyp);
    attribute_edits(dp, ref, hyp, result);
    result.wer = static_cast<double>(result.substitutions + result.deletions + result.insertions)
               / static_cast<double>(result.ref_len);
    return result;
}

std::vector<std::string> split_whitespace(const std::string & s) {
    // operator>> on a string stream skips leading whitespace and stops at the
    // next whitespace run, so reading one token per iteration yields exactly the
    // non-empty tokens regardless of how they are spaced.
    std::vector<std::string> tokens;
    std::istringstream stream(s);
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

WerResult word_error_rate(const std::string & ref, const std::string & hyp) {
    return word_error_rate(split_whitespace(ref), split_whitespace(hyp));
}

}  // namespace voiceclone
}  // namespace tts_cpp
