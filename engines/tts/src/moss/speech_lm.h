#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct SpeechTokens {
    int32_t text_placeholder = 0;
    int32_t audio_pad = 0;
    int32_t speech_start = 0;
    int32_t speech_end = 0;
    int32_t im_start = 0;
    int32_t im_end = 0;
    int32_t pad = 0;
};

struct SpeechLmConfig {
    int n_shared_layers = 0;
    int n_modality_layers = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int n_ctx_train = 0;
    float rope_base = 1000000.0f;
    float rms_eps = 1e-6f;
    int text_vocab = 0;
    int audio_vocab = 0;
    SpeechTokens tokens;
    std::string audio_system_prompt;
    std::string text_system_prompt;
};

struct SpeechRow {
    int32_t text = 0;
    int32_t audio = 0;
};

using SpeechStop = std::function<bool()>;

struct SpeechHeads {
    bool text = true;
    bool audio = true;
};

struct SpeechLogits {
    std::vector<float> text;
    std::vector<float> audio;
};

class SpeechLM {
public:
    SpeechLM(const std::string & path, bool use_gpu, int n_threads);
    ~SpeechLM();
    SpeechLM(const SpeechLM &) = delete;
    SpeechLM & operator=(const SpeechLM &) = delete;

    const SpeechLmConfig & config() const;
    const char * backend_name() const;
    std::vector<std::string> tokenizer_tokens() const;
    std::vector<std::string> tokenizer_merges() const;
    std::vector<int32_t> tokenizer_types() const;

    void begin(int n_ctx);
    // Drop per-request KV and graph buffers, retaining weights/tokenizer data.
    // A subsequent begin() starts a fresh request on the same model.
    void release_generation();
    // Also release device weights between LM and codec stages. Metadata and
    // backend identity remain valid; begin() reloads weights on the next turn.
    void release_weights();
    int position() const;
    int context() const;
    SpeechLogits prefill(const std::vector<SpeechRow> & rows, int batch_tokens, const SpeechStop & stop = {});
    SpeechLogits step(const SpeechRow & row, SpeechHeads heads = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
