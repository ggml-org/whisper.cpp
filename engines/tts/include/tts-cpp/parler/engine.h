#pragma once

// Parler-TTS engine: description-conditioned text-to-speech.
//
//   prompt      — the transcript to speak.
//   description — a natural-language voice description (speaker, pacing,
//                 acoustics); encoded by the built-in T5 encoder and fed to
//                 the decoder through cross-attention.
//
// Pipeline: T5 encoder -> delay-pattern decoder LM (9 DAC codebooks) ->
// DAC codec decode -> 44.1 kHz mono PCM. Validated backends are CPU, Metal,
// Vulkan, and OpenCL; unsupported GPU backends fall back to CPU. The Hexagon
// NPU runs only on explicit request (EngineOptions::backend).
//
// Streaming: the synthesize(...on_chunk) overload emits audio in chunks by
// re-decoding the growing DAC-code prefix, holding back a right margin for seams.
//
// Core ML (Apple, TTS_CPP_COREML builds): when the DAC sidecar sits next to the
// GGUF (`parler-mini-v1-q8_0.gguf` -> `parler-mini-v1-dac.mlmodelc`, the quant
// tag dropped), the DAC decode runs on it on the compute units Core ML picks
// (the Neural Engine on Apple silicon), whatever the `backend` option; the T5
// encoder and decoder LM stay on ggml by default. coreml_decoder opts into
// decoder steps on a separate -decoder.mlmodelc sidecar (macOS 15 / iOS 18);
// T5 and prompt prefill still run on ggml.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
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

    // Explicit compute device. Empty or "auto" (default) keeps the
    // n_gpu_layers policy walk. "cpu", "opencl", "hexagon" (the Snapdragon
    // HTP0 NPU) or an exact ggml device name select that device and override
    // n_gpu_layers; there is no fallback, so the constructor throws when it
    // is missing. The automatic walk never picks Hexagon.
    std::string backend;

    // Opt in to an experimental Core ML decoder-step sidecar. T5 and prefill
    // stay on ggml. Missing/incompatible/failing sidecars fall back to ggml;
    // coreml_strict makes a requested sidecar failure fatal.
    // PARLER_COREML_DISABLE disables this along with the DAC sidecar.
    bool coreml_decoder = false;

    // Sampling. Parler's HF defaults (do_sample, temperature 1.0, top_k 50)
    // are stored in the GGUF; zero values defer to them. greedy and top_k = 1
    // are accepted but repaired to sampling with a warning on stderr -- argmax
    // decoding never terminates for this architecture. Use seed for
    // reproducibility. A very small top_p is left as-is but narrows similarly.
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

    // Per-instance policy for BOTH DAC and decoder sidecars, resolved once
    // at construction without modifying process environment variables.
    // Empty inherits PARLER_COREML_COMPUTE_UNITS (default "all"). Explicit
    // values must be all, cpu_and_ane, cpu_and_gpu or cpu_only; invalid values
    // throw even in builds without Core ML. This does not enable the decoder.
    std::string coreml_compute_units;
    // Unset inherits PARLER_COREML_STRICT (presence means true, even "0").
    // Explicit false permits fallback even when that environment variable is
    // set. PARLER_COREML_DISABLE overrides both stages at construction.
    std::optional<bool> coreml_strict;
};

struct SynthesisResult {
    std::vector<float> pcm;          // mono float32
    int   sample_rate = 44100;
    float duration_s  = 0.0f;

    // Where this synthesis's DAC decode ran: "ggml" (on backend_name()) or the
    // Core ML sidecar's compute label ("coreml-all", "coreml-ane", ...), or
    // "mixed" when a streamed call's chunks ran on both. Unlike
    // Engine::dac_on_coreml(), this is per call: a sidecar that fails is
    // retired and the decode falls back to ggml.
    std::string dac_backend = "ggml";

    // Decoder step execution for this call, including replay after a failure.
    // Prefill currently runs on ggml. "mixed" means successful Core ML steps
    // were followed by ggml after a sidecar failure.
    std::string decoder_backend = "ggml";
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

    // True while the DAC decode runs on the Core ML sidecar (TTS_CPP_COREML
    // builds; see the header comment and docs/parler.md). Becomes false if a
    // failing sidecar is retired.
    bool dac_on_coreml() const;
    bool decoder_on_coreml() const;

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
