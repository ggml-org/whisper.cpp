#pragma once
#include "tts-cpp/export.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::pocket {
struct EngineOptions {
    std::string flow_lm_path, mimi_path, frontend_path;
    // Supply exactly one voice source. Prepared voices are checkpoint-specific.
    std::string voice_path;
    std::string reference_audio_path;
    int n_threads = 1;  // per worker; FlowLM and Mimi have separate workers
    int output_sample_rate = 24000; // 8000..192000 Hz, streaming sinc resampling
    int context = 2048;
    int max_tokens = 50;
    int steps = 1;
    float temperature = 0.3f;
    float noise_clamp = 0;  // 0 disables truncated-normal sampling
    float eos_threshold = -4;
    int frames_after_eos = -1;  // -1: checkpoint/text default; 0..100: explicit tail
    // Context validation reserves the complete EOS tail before emitting PCM.
    uint32_t seed = 1234;
    // CPU is the default. GPU requests fail if no supported GPU is available.
    bool use_gpu = false;
    // Empty/auto follows use_gpu; cpu/opencl/hexagon (HTP0) explicitly selects the backend.
    std::string backend;
    std::string backends_dir;
};
struct SynthesisResult {
    std::vector<float> pcm; // populated by synthesize(); streaming uses the callback
    int sample_rate = 24000;
    bool cancelled = false;
    int generated_frames = 0;
    double first_audio_ms = 0, generation_ms = 0;
};
using AudioCallback = std::function<bool(const float *, size_t, int)>;

// Persistent native CPU, GPU or Hexagon engine. One synthesis at a time per instance;
// overlapping/reentrant calls throw. cancel() is safe from another thread.
// Streaming callbacks run on the calling thread, with valid PCM until return.
// Returning false cancels the request. Each new request resets the voice,
// codec and RNG. The seed is portable within Fabric, not PyTorch-equivalent.
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
    const char * flow_lm_backend_name() const noexcept;
    const char * mimi_backend_name() const noexcept;
    const char * device_name() const noexcept;
    std::string model_source_sha256() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
