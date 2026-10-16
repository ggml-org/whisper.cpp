#pragma once
#include "tts-cpp/export.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss {

struct SoundEffectOptions {
    std::string model_path;
    int n_threads = 4;
    bool use_gpu = false;
    std::string backends_dir;
    std::string backend;
};

struct SoundEffectRequest {
    std::string prompt;
    std::string negative_prompt;
    double seconds = 10.0;
    int steps = 0;
    float guidance = 0.0f;
    float shift = 0.0f;
    uint32_t seed = 0;
};

struct SoundEffectResult {
    std::vector<float> pcm;
    int sample_rate = 0;
    bool cancelled = false;
    double text_ms = 0;
    double diffusion_ms = 0;
    double decode_ms = 0;
    std::string dit_backend;
    std::string vae_backend;
};

using SoundEffectProgress = std::function<bool(int step, int total)>;

class TTS_CPP_API SoundEffectEngine {
public:
    explicit SoundEffectEngine(const SoundEffectOptions & options);
    ~SoundEffectEngine();
    SoundEffectEngine(const SoundEffectEngine &) = delete;
    SoundEffectEngine & operator=(const SoundEffectEngine &) = delete;
    SoundEffectResult generate(const SoundEffectRequest & request, const SoundEffectProgress & progress = {});
    void cancel() noexcept;
    int sample_rate() const noexcept;
    float max_seconds() const noexcept;
    const char * backend_name() const noexcept;
    bool dit_on_coreml() const noexcept;
    bool vae_on_coreml() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss
