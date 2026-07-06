#pragma once

// Public LavaSR enhancer API.
//
// Opt-in neural speech enhancement applied to a synthesized PCM signal:
// bandwidth-extends the engine output to 48 kHz using the LavaSR Vocos
// enhancer (ConvNeXt backbone + ISTFT spec head), converted to a single GGUF
// and run on the CPU/GGML path.  The denoiser stage is a planned follow-up.
//
// Usage (e.g. from the tts-ggml addon, after engine->synthesize()):
//
//     auto enh = tts_cpp::lavasr::Enhancer::load("lavasr-enhancer.gguf");
//     result.pcm = enh->enhance(result.pcm, result.sample_rate);
//     result.sample_rate = enh->output_sample_rate();   // 48000
//
// The Enhancer is immutable after load and safe to share across threads for
// concurrent enhance() calls (it holds only const weights).

#include "tts-cpp/export.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

class TTS_CPP_API Enhancer {
public:
    // Load the enhancer GGUF.  Throws std::runtime_error on failure (file
    // missing, wrong architecture, missing tensors).
    static std::unique_ptr<Enhancer> load(const std::string & gguf_path);

    ~Enhancer();
    Enhancer(const Enhancer &)             = delete;
    Enhancer & operator=(const Enhancer &) = delete;

    // Enhance mono float32 PCM at `sr_in` Hz (the engine's native rate) to a
    // 48 kHz enhanced signal.  Returns empty for empty input.
    std::vector<float> enhance(const std::vector<float> & pcm_in, int sr_in) const;

    // Output sample rate of enhance() (48 kHz).
    int output_sample_rate() const;

private:
    Enhancer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::lavasr
