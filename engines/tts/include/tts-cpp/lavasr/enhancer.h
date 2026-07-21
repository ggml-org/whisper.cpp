#pragma once

// Public LavaSR enhancer API.
//
// Opt-in neural speech enhancement applied to a synthesized PCM signal:
// bandwidth-extends the engine output to 48 kHz using the LavaSR Vocos
// enhancer (ConvNeXt backbone + ISTFT spec head), converted to a single GGUF.
// The ConvNeXt backbone + spec head run through a ggml compute graph: on the
// ggml-CPU backend by default, or on a GPU backend (Vulkan / Metal / CUDA /
// OpenCL) when requested via EnhancerOptions.  A scalar CPU core is kept as the
// correctness oracle and as the fallback if graph creation fails.
// The companion denoiser stage (tts-cpp/lavasr/denoiser.h) runs before this and
// stays on the CPU (see that header for the rationale).
//
// Usage (e.g. from the tts-ggml addon, after engine->synthesize()):
//
//     tts_cpp::lavasr::EnhancerOptions opts;
//     opts.use_gpu = true;                 // route the network onto the GPU
//     auto enh = tts_cpp::lavasr::Enhancer::load("lavasr-enhancer.gguf", opts);
//     result.pcm = enh->enhance(result.pcm, result.sample_rate);
//     result.sample_rate = enh->output_sample_rate();   // 48000
//
// The Enhancer is immutable after load.  enhance() serialises concurrent calls
// on a single instance (the reusable ggml graph allocator is mutated per call),
// so create one Enhancer per thread for parallel enhancement; the scalar
// fallback path (no graph) is reentrant.

#include "tts-cpp/backend.h" // BackendDevice
#include "tts-cpp/export.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

// GPU / backend selection for the enhancer network forward.  The default runs
// the network on the ggml-CPU backend (a scalar core is kept as the fallback).
struct EnhancerOptions {
    // Run the ConvNeXt backbone + spec head on a GPU backend (Vulkan on
    // Windows/Linux, Metal on Apple, CUDA, OpenCL).  Falls back to the ggml-CPU
    // backend (then the scalar core) when no GPU backend is available.
    bool use_gpu = false;
    // Which Vulkan adapter to prefer when several are visible (see
    // backend_selection.h::init_gpu_backend): 0 = first, N = Nth (0-indexed),
    // -1 = auto-pick by free VRAM.  Ignored for non-Vulkan backends.
    int vulkan_device = 0;
    // Log backend selection to stderr.
    bool verbose = false;
};

class TTS_CPP_API Enhancer {
public:
    // Load the enhancer GGUF.  With opts.use_gpu the network runs on a GPU
    // backend when one is available, else on the ggml-CPU backend (the default).
    // Throws std::runtime_error on failure (file missing, wrong architecture,
    // missing tensors).
    static std::unique_ptr<Enhancer> load(const std::string & gguf_path,
                                          const EnhancerOptions & opts = {});

    ~Enhancer();
    Enhancer(const Enhancer &)             = delete;
    Enhancer & operator=(const Enhancer &) = delete;

    // Enhance mono float32 PCM at `sr_in` Hz (the engine's native rate) to a
    // 48 kHz enhanced signal.  Returns empty for empty input.
    std::vector<float> enhance(const std::vector<float> & pcm_in, int sr_in) const;

    // Output sample rate of enhance() (48 kHz).
    int output_sample_rate() const;

    // Registered name of the backend the network forward runs on ("Vulkan0",
    // "CUDA0", "Metal", "CPU", ...).  Mirrors chatterbox/supertonic
    // Engine::backend_name() so a host can confirm the GPU port actually
    // engaged.  "CPU" for the ggml-CPU backend (and the scalar fallback).
    std::string backend_name() const;

    // Resolved compute device: GPU when the ggml network graph runs on a GPU
    // backend (opts.use_gpu honoured and a backend was available), otherwise CPU
    // (the ggml-CPU graph — the default — or the scalar fallback).  Stable for
    // the lifetime of the Enhancer.
    BackendDevice backend_device() const;

private:
    Enhancer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::lavasr
