#include "moss/generation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace tts_cpp::moss::detail {
namespace {

constexpr int64_t DRAIN_DISARMED = std::numeric_limits<int64_t>::max();
constexpr float MASKED = -std::numeric_limits<float>::infinity();

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss generation: " + message);
}

float finite_maximum(const std::vector<float> & logits) {
    float best = MASKED;
    for (float logit : logits) {
        if (std::isfinite(logit) && logit > best) {
            best = logit;
        }
    }
    return best;
}

float exponentiate_finite(const std::vector<float> & logits, float best, std::vector<float> & probs) {
    float total = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (std::isfinite(logits[i])) {
            probs[i] = std::exp(logits[i] - best);
            total += probs[i];
        }
    }
    return total;
}

void normalize(std::vector<float> & probs, float total) {
    for (float & p : probs) {
        p /= total;
    }
}

std::vector<float> softmax(const std::vector<float> & logits) {
    const float best = finite_maximum(logits);
    std::vector<float> probs(logits.size(), 0.0f);
    if (!std::isfinite(best)) {
        probs[0] = 1.0f;
        return probs;
    }
    normalize(probs, exponentiate_finite(logits, best, probs));
    return probs;
}

size_t multinomial(const std::vector<float> & probs, std::mt19937 & rng) {
    const float draw = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (!(cumulative < draw)) {
            return i;
        }
    }
    return probs.size() - 1;
}

bool past_nucleus(size_t rank, float cumulative, float top_p) {
    return rank > 0 && cumulative > top_p;
}

void mask_beyond_top_p(std::vector<float> & probs, std::vector<size_t> & order, float top_p) {
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
    float cumulative = 0.0f;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        if (past_nucleus(rank, cumulative, top_p)) {
            probs[order[rank]] = 0.0f;
        }
        cumulative += probs[order[rank]];
    }
}

void keep_top_p(std::vector<float> & probs, float top_p) {
    if (top_p >= 1.0f) {
        return;
    }
    std::vector<size_t> order(probs.size());
    std::iota(order.begin(), order.end(), 0);
    mask_beyond_top_p(probs, order, top_p);
    normalize(probs, std::accumulate(probs.begin(), probs.end(), 0.0f));
}

int32_t argmax(const std::vector<float> & logits) {
    return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
}

struct Candidate {
    float logit;
    int32_t index;
};

bool ranks_higher(const Candidate & a, const Candidate & b) {
    return a.logit > b.logit || (a.logit == b.logit && a.index < b.index);
}

void offer_candidate(std::vector<Candidate> & heap, size_t k, Candidate candidate) {
    if (heap.size() < k) {
        heap.push_back(candidate);
        std::push_heap(heap.begin(), heap.end(), ranks_higher);
        return;
    }
    if (ranks_higher(candidate, heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), ranks_higher);
        heap.back() = candidate;
        std::push_heap(heap.begin(), heap.end(), ranks_higher);
    }
}

std::vector<Candidate> top_k_candidates(const std::vector<float> & logits, size_t k) {
    std::vector<Candidate> heap;
    heap.reserve(k);
    for (size_t i = 0; i < logits.size(); ++i) {
        offer_candidate(heap, k, {logits[i], (int32_t) i});
    }
    std::sort_heap(heap.begin(), heap.end(), ranks_higher);
    return heap;
}

std::vector<float> candidate_logits(const std::vector<Candidate> & candidates) {
    std::vector<float> subset(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        subset[i] = candidates[i].logit;
    }
    return subset;
}

int32_t sample_top_k(const std::vector<float> & logits, float top_p, int top_k, std::mt19937 & rng) {
    const size_t k = std::min<size_t>((size_t) top_k, logits.size());
    const std::vector<Candidate> candidates = top_k_candidates(logits, k);
    std::vector<float> probs = softmax(candidate_logits(candidates));
    keep_top_p(probs, top_p);
    return candidates[multinomial(probs, rng)].index;
}

int32_t sample_top_p(const std::vector<float> & logits, float top_p, std::mt19937 & rng) {
    std::vector<float> probs = softmax(logits);
    keep_top_p(probs, top_p);
    return (int32_t) multinomial(probs, rng);
}

void scale_by_temperature(std::vector<float> & logits, float temperature) {
    if (temperature <= 0.0f) {
        return;
    }
    for (float & logit : logits) {
        logit /= temperature;
    }
}

} // namespace

std::vector<int32_t> apply_delay_pattern(const std::vector<int32_t> & codes, int n_frames,
                                         int n_vq, int pad_code) {
    const int delayed_frames = n_frames + n_vq - 1;
    std::vector<int32_t> delayed((size_t) delayed_frames * n_vq, pad_code);
    for (int t = 0; t < n_frames; ++t) {
        for (int c = 0; c < n_vq; ++c) {
            delayed[(size_t) (c + t) * n_vq + c] = codes[(size_t) t * n_vq + c];
        }
    }
    return delayed;
}

std::vector<int32_t> apply_de_delay_pattern(const std::vector<int32_t> & delayed, int delayed_frames,
                                            int n_vq, int pad_code) {
    if (delayed_frames + 1 <= n_vq) {
        return {};
    }
    const int n_frames = delayed_frames - n_vq + 1;
    std::vector<int32_t> codes((size_t) n_frames * n_vq, pad_code);
    for (int t = 0; t < n_frames; ++t) {
        for (int c = 0; c < n_vq; ++c) {
            codes[(size_t) t * n_vq + c] = delayed[(size_t) (c + t) * n_vq + c];
        }
    }
    return codes;
}

void apply_repetition_penalty(std::vector<float> & logits, const std::unordered_set<int32_t> & seen,
                              float penalty) {
    if (penalty == 1.0f || seen.empty()) {
        return;
    }
    for (int32_t token : seen) {
        if (token < 0 || (size_t) token >= logits.size()) {
            continue;
        }
        float & logit = logits[(size_t) token];
        logit = logit > 0.0f ? logit / penalty : logit * penalty;
    }
}

void apply_repetition_penalty(std::vector<float> & logits, const std::vector<int32_t> & history,
                              float penalty) {
    apply_repetition_penalty(logits,
            std::unordered_set<int32_t>(history.begin(), history.end()), penalty);
}

int32_t sample_row(std::vector<float> logits, float top_p, int top_k, bool do_sample,
                   std::mt19937 & rng) {
    if (!do_sample) {
        return argmax(logits);
    }
    if (top_k > 0) {
        return sample_top_k(logits, top_p, top_k, rng);
    }
    return sample_top_p(logits, top_p, rng);
}

bool frame_is_pad(const std::vector<int32_t> & codes, int frame, int n_vq, int pad_code) {
    const size_t first = (size_t) frame * (size_t) n_vq;
    return std::all_of(codes.begin() + (std::ptrdiff_t) first,
            codes.begin() + (std::ptrdiff_t) (first + (size_t) n_vq),
            [pad_code](int32_t code) { return code == pad_code; });
}

AudioSegments extract_audio_segments(const std::vector<int32_t> & codes, int n_frames,
                                     int n_vq, int pad_code) {
    AudioSegments segments;
    bool in_segment = false;
    for (int t = 0; t < n_frames; ++t) {
        const bool pad = frame_is_pad(codes, t, n_vq, pad_code);
        if (!pad && !in_segment) {
            segments.emplace_back();
        }
        in_segment = !pad;
        if (!pad) {
            segments.back().insert(segments.back().end(),
                    codes.begin() + (std::ptrdiff_t) ((size_t) t * n_vq),
                    codes.begin() + (std::ptrdiff_t) ((size_t) (t + 1) * n_vq));
        }
    }
    return segments;
}

DelayState::DelayState(const DelayConfig & config, const std::vector<DelayRow> & prompt_rows,
                       int32_t pad_token_id, int32_t im_end_token_id)
    : config_(config), pad_token_id_(pad_token_id), im_end_token_id_(im_end_token_id),
      delayed_length_(DRAIN_DISARMED) {
    if (prompt_rows.empty()) {
        fail("empty prompt");
    }
    text_history_.reserve(prompt_rows.size());
    audio_history_.reserve(prompt_rows.size() * (size_t) config_.n_vq);
    channel_seen_.resize((size_t) config_.n_vq);
    for (const DelayRow & row : prompt_rows) {
        record_row(row);
    }
    const int32_t last = text_history_.back();
    if (last == config_.audio_start_token_id || last == config_.audio_assistant_gen_slot_token_id) {
        int64_t start_index = -1;
        for (int64_t i = (int64_t) text_history_.size() - 1; i >= 0; --i) {
            if (text_history_[i] == config_.audio_start_token_id) {
                start_index = i;
                break;
            }
        }
        if (start_index >= 0) {
            audio_length_ = (int32_t) ((int64_t) text_history_.size() - start_index);
            is_audio_ = true;
        }
    }
}

bool DelayState::channel_is_sampled(int channel) const {
    const bool pre_audio = channel < std::max<int32_t>(audio_length_, 0);
    const bool post_audio = delayed_length_ == DRAIN_DISARMED ||
            (int64_t) channel > std::max<int64_t>(delayed_length_ - 1, -1);
    return pre_audio && post_audio;
}

TextLogits DelayState::text_logits() const {
    if (stopping_ || delayed_length_ <= config_.n_vq) {
        return TextLogits::None;
    }
    return is_audio_ ? TextLogits::Slots : TextLogits::Full;
}

int32_t DelayState::next_text_token(const DelayLogits & logits, const SamplingConfig & sampling,
                                    std::mt19937 & rng) const {
    if (delayed_length_ < config_.n_vq) {
        return config_.audio_assistant_delay_slot_token_id;
    }
    if (delayed_length_ == config_.n_vq) {
        return config_.audio_end_token_id;
    }
    std::vector<float> row = logits.text;
    scale_by_temperature(row, sampling.text_temperature);
    if (!is_audio_) {
        row[(size_t) pad_token_id_] = MASKED;
        row[(size_t) config_.audio_assistant_gen_slot_token_id] = MASKED;
        row[(size_t) config_.audio_assistant_delay_slot_token_id] = MASKED;
        row[(size_t) config_.audio_end_token_id] = MASKED;
    } else {
        std::vector<float> masked(row.size(), MASKED);
        masked[(size_t) config_.audio_assistant_gen_slot_token_id] =
                row[(size_t) config_.audio_assistant_gen_slot_token_id];
        masked[(size_t) config_.audio_assistant_delay_slot_token_id] =
                row[(size_t) config_.audio_assistant_delay_slot_token_id];
        row = std::move(masked);
    }
    if (time_step_ == 0) {
        row[(size_t) config_.audio_assistant_delay_slot_token_id] = MASKED;
    }
    if (time_step_ <= config_.n_vq) {
        row[(size_t) im_end_token_id_] = MASKED;
    }
    return sample_row(std::move(row), sampling.text_top_p, sampling.text_top_k,
            sampling.text_temperature > 0.0f, rng);
}

int32_t DelayState::sample_audio_channel(const DelayLogits & logits, int channel,
                                         const std::unordered_set<int32_t> & seen,
                                         const SamplingConfig & sampling, std::mt19937 & rng) const {
    std::vector<float> row = logits.audio[(size_t) channel];
    scale_by_temperature(row, sampling.audio_temperature);
    if (config_.audio_pad_code >= 0 && (size_t) config_.audio_pad_code < row.size()) {
        row[(size_t) config_.audio_pad_code] = MASKED;
    }
    apply_repetition_penalty(row, seen, sampling.audio_repetition_penalty);
    return sample_row(std::move(row), sampling.audio_top_p, sampling.audio_top_k,
            sampling.audio_temperature > 0.0f, rng);
}

std::vector<int> DelayState::sampled_rest_channels() const {
    std::vector<int> channels;
    for (int channel = 1; channel < config_.n_vq; ++channel) {
        if (channel_is_sampled(channel)) {
            channels.push_back(channel);
        }
    }
    return channels;
}

std::unordered_set<int32_t> DelayState::merged_channel_seen(const std::vector<int> & channels) const {
    std::unordered_set<int32_t> seen;
    for (int channel : channels) {
        seen.insert(channel_seen_[(size_t) channel].begin(),
                channel_seen_[(size_t) channel].end());
    }
    return seen;
}

std::vector<int32_t> DelayState::next_audio_codes(const DelayLogits & logits,
                                                  const SamplingConfig & sampling,
                                                  std::mt19937 & rng) const {
    std::vector<int32_t> codes((size_t) config_.n_vq, config_.audio_pad_code);
    if (channel_is_sampled(0)) {
        codes[0] = sample_audio_channel(logits, 0, channel_seen_[0], sampling, rng);
    }
    const std::vector<int> rest = sampled_rest_channels();
    if (!rest.empty()) {
        sample_rest_channels(codes, rest, logits, sampling, rng);
    }
    return codes;
}

void DelayState::sample_rest_channels(std::vector<int32_t> & codes, const std::vector<int> & rest,
                                      const DelayLogits & logits, const SamplingConfig & sampling,
                                      std::mt19937 & rng) const {
    const std::unordered_set<int32_t> seen = merged_channel_seen(rest);
    for (int channel : rest) {
        codes[(size_t) channel] = sample_audio_channel(logits, channel, seen, sampling, rng);
    }
}

void DelayState::record_row(const DelayRow & row) {
    text_history_.push_back(row.text);
    audio_history_.insert(audio_history_.end(), row.audio.begin(), row.audio.end());
    for (size_t channel = 0; channel < row.audio.size(); ++channel) {
        channel_seen_[channel].insert(row.audio[channel]);
    }
}

void DelayState::advance_counters(int32_t text_token) {
    if (text_token == config_.audio_start_token_id ||
        text_token == config_.audio_assistant_gen_slot_token_id ||
        text_token == config_.audio_assistant_delay_slot_token_id) {
        audio_length_ += 1;
    }
    if (text_token == config_.audio_end_token_id) {
        audio_length_ = 0;
    }
    if (delayed_length_ == DRAIN_DISARMED &&
        text_token == config_.audio_assistant_delay_slot_token_id) {
        delayed_length_ = 0;
    }
    if (delayed_length_ != DRAIN_DISARMED) {
        delayed_length_ += 1;
    }
    if (delayed_length_ != DRAIN_DISARMED && delayed_length_ > config_.n_vq) {
        delayed_length_ = DRAIN_DISARMED;
    }
    time_step_ += 1;
}

DelayRow DelayState::step(const DelayLogits & logits, const SamplingConfig & sampling,
                          std::mt19937 & rng) {
    DelayRow row;
    if (stopping_) {
        row.text = pad_token_id_;
        row.audio.assign((size_t) config_.n_vq, config_.audio_pad_code);
        return row;
    }
    row.text = next_text_token(logits, sampling, rng);
    if (row.text == config_.audio_start_token_id) {
        is_audio_ = true;
    }
    if (row.text == im_end_token_id_) {
        stopping_ = true;
    }
    row.audio = next_audio_codes(logits, sampling, rng);
    if (row.text == config_.audio_end_token_id) {
        is_audio_ = false;
    }
    advance_counters(row.text);
    record_row(row);
    return row;
}

int DelayState::available_frames(int base_row) const {
    const int64_t rows = (int64_t) (audio_history_.size() / (size_t) config_.n_vq) - base_row;
    return (int) std::max<int64_t>(0, rows - (config_.n_vq - 1));
}

std::vector<int32_t> DelayState::frame_codes(int base_row, int frame) const {
    std::vector<int32_t> codes((size_t) config_.n_vq, config_.audio_pad_code);
    for (int channel = 0; channel < config_.n_vq; ++channel) {
        const size_t row = (size_t) (base_row + frame + channel);
        codes[(size_t) channel] = audio_history_[row * (size_t) config_.n_vq + (size_t) channel];
    }
    return codes;
}

AudioSegments DelayState::generated_audio(int base_row) const {
    const size_t begin = (size_t) base_row * config_.n_vq;
    if (begin >= audio_history_.size()) {
        return {};
    }
    const std::vector<int32_t> delayed(audio_history_.begin() + begin, audio_history_.end());
    const int delayed_frames = (int) (delayed.size() / (size_t) config_.n_vq);
    const std::vector<int32_t> codes = apply_de_delay_pattern(delayed, delayed_frames,
            config_.n_vq, config_.audio_pad_code);
    return extract_audio_segments(codes, (int) (codes.size() / (size_t) config_.n_vq),
            config_.n_vq, config_.audio_pad_code);
}

} // namespace tts_cpp::moss::detail
