#pragma once

#include "moss/delay_lm.h"

#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace tts_cpp::moss::detail {

struct SamplingConfig {
    float text_temperature  = 1.5f;
    float text_top_p        = 1.0f;
    int   text_top_k        = 50;
    float audio_temperature = 1.7f;
    float audio_top_p       = 0.8f;
    int   audio_top_k       = 25;
    float audio_repetition_penalty = 1.0f;
};

std::vector<int32_t> apply_delay_pattern(const std::vector<int32_t> & codes, int n_frames,
                                         int n_vq, int pad_code);
std::vector<int32_t> apply_de_delay_pattern(const std::vector<int32_t> & delayed, int delayed_frames,
                                            int n_vq, int pad_code);

void apply_repetition_penalty(std::vector<float> & logits, const std::unordered_set<int32_t> & seen,
                              float penalty);
void apply_repetition_penalty(std::vector<float> & logits, const std::vector<int32_t> & history,
                              float penalty);
int32_t sample_row(std::vector<float> logits, float top_p, int top_k, bool do_sample,
                   std::mt19937 & rng);

using AudioSegments = std::vector<std::vector<int32_t>>;

bool frame_is_pad(const std::vector<int32_t> & codes, int frame, int n_vq, int pad_code);
AudioSegments extract_audio_segments(const std::vector<int32_t> & codes, int n_frames,
                                     int n_vq, int pad_code);

class DelayState {
public:
    DelayState(const DelayConfig & config, const std::vector<DelayRow> & prompt_rows,
               int32_t pad_token_id, int32_t im_end_token_id);

    bool stopping() const { return stopping_; }
    TextLogits text_logits() const;
    const std::vector<int32_t> & audio_history() const { return audio_history_; }

    DelayRow step(const DelayLogits & logits, const SamplingConfig & sampling, std::mt19937 & rng);

    AudioSegments generated_audio(int base_row) const;
    int available_frames(int base_row) const;
    std::vector<int32_t> frame_codes(int base_row, int frame) const;

private:
    bool channel_is_sampled(int channel) const;
    int32_t next_text_token(const DelayLogits & logits, const SamplingConfig & sampling,
                            std::mt19937 & rng) const;
    std::vector<int32_t> next_audio_codes(const DelayLogits & logits, const SamplingConfig & sampling,
                                          std::mt19937 & rng) const;
    int32_t sample_audio_channel(const DelayLogits & logits, int channel,
                                 const std::unordered_set<int32_t> & seen,
                                 const SamplingConfig & sampling, std::mt19937 & rng) const;
    std::vector<int> sampled_rest_channels() const;
    void sample_rest_channels(std::vector<int32_t> & codes, const std::vector<int> & rest,
                              const DelayLogits & logits, const SamplingConfig & sampling,
                              std::mt19937 & rng) const;
    std::unordered_set<int32_t> merged_channel_seen(const std::vector<int> & channels) const;
    void record_row(const DelayRow & row);
    void advance_counters(int32_t text_token);

    DelayConfig config_;
    int32_t pad_token_id_;
    int32_t im_end_token_id_;
    int32_t audio_length_ = 0;
    int64_t delayed_length_;
    bool is_audio_ = false;
    bool stopping_ = false;
    int64_t time_step_ = 0;
    std::vector<int32_t> text_history_;
    std::vector<int32_t> audio_history_;
    std::vector<std::unordered_set<int32_t>> channel_seen_;
};

} // namespace tts_cpp::moss::detail
