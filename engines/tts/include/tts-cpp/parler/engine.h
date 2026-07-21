#pragma once

// Parler-TTS engine: description-conditioned text-to-speech.
//
//   prompt      — the transcript to speak.
//   description — a natural-language voice description (speaker, pacing,
//                 acoustics); encoded by the built-in T5 encoder and fed to
//                 the decoder through cross-attention.
//
// Pipeline: T5 encoder -> delay-pattern decoder LM (9 DAC codebooks) ->
// DAC codec decode -> 44.1 kHz mono PCM. CPU-validated; the backend plumbing
// mirrors the other engines so GPU backends can be enabled later.
//
// Streaming synthesis is intentionally not offered yet: the delay pattern
// completes a frame only 8 steps late and DAC decode is whole-sequence.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {

struct EngineOptions {
    // Required.
    std::string model_gguf_path;

    // Used by the one-argument synthesize(); the two-argument overload
    // takes the description per call. Empty + one-argument call throws.
    std::string default_description;

    int n_threads = 0;   // 0 => min(hardware_concurrency, 4)

    // Sampling. Parler's HF defaults (do_sample, temperature 1.0, top_k 50)
    // are stored in the GGUF; zero values defer to them. greedy forces
    // deterministic argmax decoding regardless of the GGUF default.
    bool  greedy      = false;
    int   seed        = 42;
    float temperature = 0.0f;  // 0 => GGUF default
    int   top_k       = 0;     // 0 => GGUF default
    float top_p       = 1.0f;

    // Generation length cap in delayed decoder steps (~86 frames/s of audio).
    // 0 => the GGUF's max_length (2580 ~= 30 s). Lower values bound both
    // latency and the KV-cache footprint of a synthesize() call.
    int max_frames = 0;

    // -1 => GGUF default (mini: 10; large: disabled).
    int min_new_tokens = -1;

    // Expand digits in the prompt to English words before tokenization
    // (parler-v1 voices raw digits badly). Deliberate divergence from HF.
    bool normalize_numbers = true;

    // Directory for dynamically-loaded ggml backends (GGML_BACKEND_DL
    // builds); empty keeps the default search behaviour.
    std::string backends_dir;
};

struct SynthesisResult {
    std::vector<float> pcm;          // mono float32
    int   sample_rate = 44100;
    float duration_s  = 0.0f;
};

// Persistent engine. Loads the GGUF once at construction; synthesize()
// reuses the resident model. The T5 encoding of the most recent
// description is cached, so consecutive calls with the same description
// only run the decoder + codec.
class TTS_CPP_API Engine {
public:
    // Throws std::runtime_error on any hard failure.
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize with options().default_description. Throws on failure or
    // when no default description was configured. Not thread-safe per
    // instance.
    SynthesisResult synthesize(const std::string & prompt);

    // Synthesize `prompt` spoken as described by `description`.
    SynthesisResult synthesize(const std::string & prompt,
                               const std::string & description);

    // Best-effort cancel of an in-flight synthesize() on another thread;
    // takes effect at the next decoder step.
    void cancel();

    const EngineOptions & options() const;
    std::string backend_name() const;
    BackendDevice backend_device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// One-shot convenience wrapper (pays a full model load per call).
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts,
                                       const std::string & prompt,
                                       const std::string & description);

} // namespace parler
} // namespace tts_cpp
