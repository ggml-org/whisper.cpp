#pragma once

#include "audiogen-cpp/export.h"
#include "audiogen-cpp/gpu_fallback.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::minimax {

struct EngineOptions {
    std::string model_dir;
    std::string lm_model_path;
    std::string synth_model_path;
    int n_threads = 0;
    std::string backends_dir;
    // Compute device: "cpu" (default), "gpu" (require a GPU; engine creation
    // fails when none is usable) or "auto" (GPU when available, CPU fallback
    // otherwise). An empty string defers to the MM3_DEVICE environment
    // variable, then "cpu". GPU coverage comes from whichever ggml backends
    // are packaged (CUDA, Vulkan, Metal, ...); the CPU backend always backs
    // unsupported ops.
    std::string device;
    // Explicit compiled Core ML conditioner projection (.mlmodelc), exported
    // from this synth GGUF. Uses CPU + ANE; the other stages use `device`.
    // Empty keeps the native conditioner. The model stays resident in Engine.
    std::string coreml_conditioner_path;
};

struct GenerateParams {
    std::string caption;
    std::string lyrics;
    int64_t max_frames = 300;
    int64_t seed = -1;
    int inference_steps = 0;  // 0 = engine default
    float cfg_scale = 0.0f;
};

struct GenerateResult {
    std::vector<float> pcm;
    int sample_rate = 0;
    int channels = 2;
    int64_t emitted_frames = 0;
    double ar_ms = 0.0;
    double condition_ms = 0.0;
    double flow_ms = 0.0;
    double vocoder_ms = 0.0;
    double total_ms = 0.0;
};

using ProgressFn = std::function<bool(const std::string & stage, int64_t current, int64_t total)>;

class AUDIOGEN_API Engine {
public:
    static std::unique_ptr<Engine> create(const EngineOptions & options);

    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    GenerateResult generate(const GenerateParams & params, const ProgressFn & progress = {}) const;
    void cancel() const;
    int sample_rate() const;
    std::string backend_name() const;

    // Why a run asked for a GPU and got the CPU. `not_requested` when
    // device=cpu, `none` when a GPU backend was acquired.
    GpuFallbackReason gpu_fallback_reason() const;

private:
    Engine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
