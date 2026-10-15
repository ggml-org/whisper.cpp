#pragma once
#include "tts-cpp/export.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss {

struct EngineOptions {
    std::string backbone_path;
    std::string decoder_path;
    std::string encoder_path;
    std::string reference_audio_path;
    std::vector<std::string> dialogue_reference_paths;
    std::string language = "zh";
    int duration_tokens = 0;
    int n_threads = 4;
    int context = 4096;
    int max_new_tokens = 2048;
    bool use_gpu = false;
    std::string backend;
    float text_temperature  = 1.5f;
    float text_top_p        = 1.0f;
    int   text_top_k        = 50;
    float audio_temperature = 1.7f;
    float audio_top_p       = 0.8f;
    int   audio_top_k       = 25;
    float audio_repetition_penalty = 1.0f;
    uint32_t seed = 1234;
    int stream_chunk_frames = 25;
    std::string backends_dir;
};

struct SynthesisResult {
    std::vector<float> pcm;
    int sample_rate = 0;
    bool cancelled = false;
    int generated_frames = 0;
    double generation_ms = 0;
    double decode_ms = 0;
    double first_audio_ms = 0;
    std::string codec_backend;
};

using AudioCallback = std::function<bool(const float *, size_t, int)>;

class TTS_CPP_API Engine {
public:
    explicit Engine(const EngineOptions & options);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    SynthesisResult synthesize(const std::string & text);
    SynthesisResult synthesize_stream(const std::string & text, const AudioCallback & callback);
    void cancel() noexcept;
    int sample_rate() const noexcept;
    const char * backend_name() const noexcept;
    bool codec_on_coreml() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss
