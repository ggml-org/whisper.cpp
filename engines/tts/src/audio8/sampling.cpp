#include "audio8/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

constexpr float MIN_TEMPERATURE = 1e-5f;
// Keeps -log(u) finite when the draw lands on zero.
constexpr float MIN_UNIFORM = 1e-20f;
const float REJECTED = -std::numeric_limits<float>::infinity();

std::vector<int> order_by_logit(const std::vector<float> & logits) {
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int left, int right) { return logits[left] > logits[right]; });
    return order;
}

std::vector<float> softmax_in_rank_order(const std::vector<float> & logits,
                                         const std::vector<int> & order) {
    std::vector<float> probabilities(order.size());
    const float top = logits[order.front()];
    float total = 0.0f;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        probabilities[rank] = std::exp(logits[order[rank]] - top);
        total += probabilities[rank];
    }
    for (float & value : probabilities) value /= total;
    return probabilities;
}

// How many of the ranked candidates survive top-k and top-p. A candidate goes
// when the mass up to and including it passes top_p, so the one that crosses
// the threshold is dropped; the highest-ranked candidate always stays.
size_t surviving_rank_count(const std::vector<float> & probabilities,
                            const sampling_params & params) {
    const size_t limit = params.top_k > 0
                             ? std::min<size_t>(params.top_k, probabilities.size())
                             : probabilities.size();
    float mass = 0.0f;
    for (size_t rank = 0; rank < limit; ++rank) {
        mass += probabilities[rank];
        if (mass > params.top_p) return std::max<size_t>(rank, 1);
    }
    return std::max<size_t>(limit, 1);
}

// Gumbel-max: the argmax of p / -log(u) is a draw from p. The weights stay
// unnormalised because a common positive factor cannot move that argmax.
int draw(const std::vector<float> & scores, std::mt19937 & rng) {
    const float top = *std::max_element(scores.begin(), scores.end());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    int best = 0;
    float best_score = -1.0f;
    for (size_t index = 0; index < scores.size(); ++index) {
        const float weight = std::exp(scores[index] - top);
        if (weight == 0.0f) continue;
        const float noise = -std::log(std::max(uniform(rng), MIN_UNIFORM));
        if (weight / noise <= best_score) continue;
        best_score = weight / noise;
        best = static_cast<int>(index);
    }
    return best;
}

}  // namespace

std::vector<float> filter_scores(const std::vector<float> & logits,
                                 const sampling_params & params) {
    const std::vector<int> order = order_by_logit(logits);
    const std::vector<float> probabilities = softmax_in_rank_order(logits, order);
    const size_t kept = surviving_rank_count(probabilities, params);
    const float scale = 1.0f / std::max(params.temperature, MIN_TEMPERATURE);
    std::vector<float> scores(logits.size(), REJECTED);
    for (size_t rank = 0; rank < kept; ++rank) {
        scores[order[rank]] = logits[order[rank]] * scale;
    }
    return scores;
}

// Argmax survives the filter untouched -- it is always kept, and the
// temperature is a positive scale -- so greedy reads the raw logits.
int sample_token(const std::vector<float> & logits, const sampling_params & params,
                 std::mt19937 & rng) {
    if (params.greedy) {
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) -
                                logits.begin());
    }
    return draw(filter_scores(logits, params), rng);
}

RepetitionAwareSampler::RepetitionAwareSampler(int window, int semantic_begin,
                                               int semantic_end,
                                               const sampling_params & narrow)
    : window_(window), semantic_begin_(semantic_begin), semantic_end_(semantic_end),
      narrow_(narrow) {}

bool RepetitionAwareSampler::is_repeat(int token) const {
    if (token < semantic_begin_ || token > semantic_end_) return false;
    return std::find(recent_.begin(), recent_.end(), token) != recent_.end();
}

void RepetitionAwareSampler::record(int token) {
    if (static_cast<int>(recent_.size()) == window_) recent_.pop_front();
    recent_.push_back(token);
}

void RepetitionAwareSampler::remember(int token) {
    if (window_ <= 0) return;
    if (!opened_) {
        opened_ = true;
        return;
    }
    record(token);
}

int RepetitionAwareSampler::pick(const std::vector<float> & logits,
                                 const sampling_params & params, std::mt19937 & rng) {
    int chosen = sample_token(logits, params, rng);
    if (!params.greedy && window_ > 0 && is_repeat(chosen)) {
        sampling_params retry = params;
        retry.temperature = narrow_.temperature;
        retry.top_p = narrow_.top_p;
        chosen = sample_token(logits, retry, rng);
    }
    remember(chosen);
    return chosen;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
