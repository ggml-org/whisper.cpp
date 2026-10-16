#pragma once

// Native Pocket TTS numerical core, returning continuous latents.
#include "pocket/memory.h"
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace tts_cpp::pocket::detail {

struct FlowConfig {
    int dim = 0;
    int heads = 0;
    int layers = 0;
    int ff_dim = 0;
    int latent_dim = 0;
    int flow_dim = 0;
    int flow_depth = 0;
    int vocab_size = 0;
    float rope_base = 10000.0f;
    bool bos_before_voice = false;
};

struct FlowMemory : MemoryMeasure { FlowConfig config; };

struct FrameCondition {
    std::vector<float> hidden;
    float eos_logit = 0;
};
struct VoiceState {
    std::string source_hash;
    int frames = 0;
    std::vector<std::vector<float>> keys, values;
};

// One instance owns weights, its backend, and a bounded linear KV cache.
// Calls must be serialized by the caller. reset() starts an independent
// utterance; prefill and advance check capacity before changing the cache.
class FlowLM {
public:
    explicit FlowLM(const std::string & gguf, int context = 2048, int threads = 2,
                    ggml_backend_device * device = nullptr);
    ~FlowLM();
    FlowLM(const FlowLM &) = delete;
    FlowLM & operator=(const FlowLM &) = delete;

    // Shares the loader and builders, without payload reads or device buffers.
    static FlowMemory measure(const std::string & path, int context,
                                 int prefill_frames, int threads, ggml_backend_device * device = nullptr);
    static int measure_voice(const std::string & path, const FlowMemory & model, int context);
    const FlowConfig & config() const;
    const char * backend_name() const noexcept;
    int position() const;
    void reset();
    std::string source_hash() const;
    VoiceState read_voice(const std::string & path) const;
    VoiceState capture_voice() const;
    void restore_voice(const VoiceState & voice);
    std::vector<float> voice_embeddings(const std::vector<float> & latents) const;

    // Row-major [frames, dim], including any checkpoint-specific voice/BOS
    // conditioning prepared by the frontend. Returns all normalized rows.
    std::vector<float> prefill(const std::vector<float> & embeddings);
    std::vector<float> text_embeddings(const std::vector<int> & token_ids) const;
    // Empty latent selects the learned audio BOS. Otherwise takes one
    // normalized continuous latent (the output of sample()).
    FrameCondition advance(const std::vector<float> & latent = {});
    // Noise is supplied by the caller to permit exact reference replay.
    // Temperature / truncation are applied to noise before this call.
    std::vector<float> sample(const std::vector<float> & hidden,
                              const std::vector<float> & noise, int steps);
    std::vector<float> denormalize(const std::vector<float> & latent) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::pocket::detail
