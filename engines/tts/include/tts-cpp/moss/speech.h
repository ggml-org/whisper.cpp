#pragma once
#include "tts-cpp/export.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss {

struct SpeechOptions {
    std::string model_path;
    std::string codec_path;
    int n_threads = 4;
    bool use_gpu = false;
    std::string backends_dir;
    std::string backend;
};

enum class SpeechRole { System, User, Assistant };

struct SpeechMessage {
    SpeechRole role = SpeechRole::User;
    std::string text;
    std::vector<float> audio;
    int sample_rate = 0;
};

struct SpeechRequest {
    std::vector<SpeechMessage> messages;
    std::vector<float> voice;
    int voice_sample_rate = 0;
    bool text_reply = false;
    float max_reply_seconds = 0.0f;
    int max_new_tokens = 0;
    bool greedy = false;
    float temperature = 0.7f;
    float top_p = 0.95f;
    int top_k = 20;
    uint32_t seed = 0;
};

struct SpeechResult {
    std::vector<float> pcm;
    int sample_rate = 0;
    std::string text;
    int reply_tokens = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    bool truncated = false;
    bool cancelled = false;
    double encode_ms = 0;
    double prefill_ms = 0;
    double generate_ms = 0;
    double decode_ms = 0;
    std::string tokenizer_backend;
};

using SpeechProgress = std::function<bool(int generated_tokens, int max_new_tokens)>;

class TTS_CPP_API SpeechEngine {
public:
    explicit SpeechEngine(const SpeechOptions & options);
    ~SpeechEngine();
    SpeechEngine(const SpeechEngine &) = delete;
    SpeechEngine & operator=(const SpeechEngine &) = delete;
    SpeechResult respond(const SpeechRequest & request, const SpeechProgress & progress = {});
    void cancel() noexcept;
    int sample_rate() const noexcept;
    float tokens_per_second() const noexcept;
    const char * backend_name() const noexcept;
    bool tokenizer_on_coreml() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss
