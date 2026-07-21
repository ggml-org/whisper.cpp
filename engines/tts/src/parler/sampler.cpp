#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

int32_t sample_row(const float * row, int vocab, const parler_sampling_params & p,
                   std::mt19937 & rng) {
    if (p.greedy) {
        int best = 0;
        for (int i = 1; i < vocab; ++i) {
            if (row[i] > row[best]) best = i;
        }
        return best;
    }

    std::vector<float> l(row, row + vocab);
    if (p.temperature > 0.0f && p.temperature != 1.0f) {
        for (float & v : l) v /= p.temperature;
    }
    if (p.top_k > 0 && p.top_k < vocab) {
        std::vector<float> sorted(l);
        std::nth_element(sorted.begin(), sorted.begin() + (p.top_k - 1), sorted.end(),
                         std::greater<float>());
        const float thresh = sorted[p.top_k - 1];
        for (float & v : l) {
            if (v < thresh) v = -std::numeric_limits<float>::infinity();
        }
    }
    if (p.top_p < 1.0f) {
        std::vector<int> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        float max_l = l[idx[0]];
        double denom = 0.0;
        std::vector<double> probs(vocab);
        for (int i = 0; i < vocab; ++i) {
            probs[i] = std::exp((double) l[idx[i]] - max_l);
            denom += probs[i];
        }
        double cum = 0.0;
        for (int i = 0; i < vocab; ++i) {
            cum += probs[i] / denom;
            if (cum > p.top_p) {
                for (int j = i + 1; j < vocab; ++j) {
                    l[idx[j]] = -std::numeric_limits<float>::infinity();
                }
                break;
            }
        }
    }

    // softmax + multinomial
    float max_l = -std::numeric_limits<float>::infinity();
    for (float v : l) max_l = std::max(max_l, v);
    std::vector<double> probs(vocab, 0.0);
    double denom = 0.0;
    for (int i = 0; i < vocab; ++i) {
        if (std::isinf(l[i]) && l[i] < 0) continue;
        probs[i] = std::exp((double) l[i] - max_l);
        denom += probs[i];
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng) * denom;
    double cum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return vocab - 1;
}

} // namespace

std::vector<int32_t> parler_sample_frame(const float * logits, int n_codebooks, int vocab,
                                         const parler_sampling_params & params,
                                         std::mt19937 & rng) {
    std::vector<int32_t> frame((size_t) n_codebooks);
    for (int k = 0; k < n_codebooks; ++k) {
        frame[k] = sample_row(logits + (size_t) k * vocab, vocab, params, rng);
    }
    return frame;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
