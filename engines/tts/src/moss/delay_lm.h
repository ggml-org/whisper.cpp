#pragma once

#include "moss/backend_request.h"
#include "tts-cpp/fit.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct DelayConfig {
    int n_layers      = 0;
    int n_embd        = 0;
    int n_heads       = 0;
    int n_kv_heads    = 0;
    int head_dim      = 0;
    int n_ff          = 0;
    int n_ctx_train   = 0;
    float rope_base   = 10000.0f;
    float rms_eps     = 1e-6f;
    int text_vocab    = 0;
    int audio_vocab   = 0;
    int n_vq          = 0;
    int audio_pad_code = 0;
    int audio_start_token_id = 0;
    int audio_end_token_id   = 0;
    int audio_user_slot_token_id           = 0;
    int audio_assistant_gen_slot_token_id  = 0;
    int audio_assistant_delay_slot_token_id = 0;
    int sampling_rate = 0;
};

struct DelayRow {
    int32_t text = 0;
    std::vector<int32_t> audio;
};

struct DelayLogits {
    std::vector<float> text;
    std::vector<std::vector<float>> audio;
};

enum class TextLogits { None, Slots, Full };

struct DelayLayout {
    TablePlacement tables = TablePlacement::Automatic;
    AttentionLayout attention = AttentionLayout::Automatic;
};

class DelayLM {
public:
    DelayLM(const std::string & path, const BackendRequest & backend, int n_threads, int n_ctx,
            bool measure_only = false, const DelayLayout & layout = {});
    ~DelayLM();
    DelayLM(const DelayLM &) = delete;
    DelayLM & operator=(const DelayLM &) = delete;

    const DelayConfig & config() const;
    const char * backend_name() const;
    bool tables_on_host() const;
    bool fused_attention() const;
    int position() const;
    int context() const;
    FitResult measure(int prompt_rows, int generated_rows);
    void reset();

    std::vector<std::string> tokenizer_tokens() const;
    std::vector<std::string> tokenizer_merges() const;
    int32_t token_id(const std::string & token) const;

    DelayLogits prefill(const std::vector<DelayRow> & rows, TextLogits text = TextLogits::Full);
    DelayLogits step(const DelayRow & row, TextLogits text = TextLogits::Full);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
