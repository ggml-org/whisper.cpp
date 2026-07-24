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
// Streaming: the synthesize(...on_chunk) overload emits audio in chunks by
// re-decoding the growing DAC-code prefix, holding back a right margin for seams.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
#include <functional>
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

    // GPU layers to offload (Metal/Vulkan/...); 0 = CPU. >0 selects the best
    // available GPU and falls back to CPU when none is present.
    int n_gpu_layers = 0;

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

    // Streaming: the synthesize(...on_chunk) overload emits audio every
    // stream_chunk_frames decoder steps (~86/s). 0 = whole-utterance (no stream).
    int stream_chunk_frames       = 0;   // steady cadence (e.g. 86 ~= 1 s)
    int stream_first_chunk_frames = 0;   // 0 => stream_chunk_frames (e.g. 43 ~= 0.5 s)

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

// Per-chunk PCM callback: consecutive float32 mono samples (engine-owned, do not
// retain). chunk_index 0-based +1 each call; is_last true only on the last chunk.
using StreamCallback = std::function<void(const float * pcm, std::size_t samples,
                                          int chunk_index, bool is_last)>;

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

    // Streaming variant: delivers audio in chunks via `on_chunk` as decoding
    // proceeds. result.pcm is still fully populated (== concat of the chunks).
    SynthesisResult synthesize(const std::string & prompt,
                               const std::string & description,
                               const StreamCallback & on_chunk);

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
