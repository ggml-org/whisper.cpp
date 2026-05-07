#pragma once

// Persistent Chatterbox engine.
//
// Loads the T3 GGUF once, preloads the S3Gen GGUF, optionally bakes a
// voice-cloning profile from a reference wav (or loads a pre-baked one
// from a --ref-dir directory), and keeps all of that resident so that
// subsequent calls to `synthesize()` pay only the T3 autoregressive
// decode + S3Gen + HiFT synthesis cost.
//
// Usage:
//
//     using tts_cpp::chatterbox::Engine;
//     using tts_cpp::chatterbox::EngineOptions;
//
//     EngineOptions opts;
//     opts.t3_gguf_path    = "models/chatterbox-t3-turbo.gguf";
//     opts.s3gen_gguf_path = "models/chatterbox-s3gen.gguf";
//     opts.n_gpu_layers    = 99;                          // Metal/CUDA/Vulkan/OpenCL
//     opts.reference_audio = "voices/alice.wav";          // optional
//
//     Engine engine(opts);
//     for (const auto & line : lines) {
//         auto result = engine.synthesize(line);
//         write_wav(result.pcm, result.sample_rate);
//     }
//
// Threading model:
//   - `synthesize()` on the same `Engine` instance is **not** safe to
//     call concurrently — the T3 KV cache, gallocr, and CFM rng are
//     per-instance shared state.
//   - `synthesize()` on **different** `Engine` instances from
//     different threads is safe.  Internal scratch buffers used by the
//     T3 graph builders are `thread_local`, so two threads do not race
//     on a shared graph workspace.
//   - `cancel()` is safe to call from any thread.
//
// Implemented in src/chatterbox_engine.cpp on top of the library-internal
// helpers in src/chatterbox_t3_internal.h.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::chatterbox {

struct EngineOptions {
    // Required: T3 GGUF and S3Gen GGUF paths (as produced by the Python
    // converters under scripts/).
    std::string t3_gguf_path;
    std::string s3gen_gguf_path;

    // Optional voice cloning.  Either (or both) of these two paths may be
    // set; if both are empty the engine uses the built-in reference voice
    // embedded in the S3Gen GGUF.
    //
    //   reference_audio: path to a mono wav >= 5 s.  The engine computes
    //                    speaker_emb + prompt_feat + prompt_token +
    //                    embedding + cond_prompt_speech_tokens in C++
    //                    once (at construction) and reuses them across
    //                    every synthesize() call.
    //
    //   voice_dir:       path to a directory of pre-baked .npy tensors
    //                    (the layout produced by `tts-cli --save-voice`).
    //                    Faster to load than reference_audio.  When both
    //                    are provided, reference_audio takes precedence for
    //                    any tensor missing from voice_dir.
    std::string reference_audio;
    std::string voice_dir;

    // Backend selection.  n_gpu_layers > 0 enables the first available
    // GPU backend (CUDA → Metal → Vulkan → OpenCL in build order), falling
    // back to the CPU backend when none is compiled in or initialisation fails.
    // The exact per-layer split is not used today; any positive value
    // moves the whole model to the GPU.
    int n_gpu_layers = 0;

    // 0 = std::thread::hardware_concurrency() (capped at 4 by default).
    int n_threads = 0;

    // RNG seed for T3 sampling + CFM initial noise.  A fixed seed gives
    // reproducible output across runs of the same text; bump it to get a
    // different "take" of the same line.
    int seed = 42;

    // Maximum speech tokens to decode per synthesize() call.
    int n_predict = 1000;

    // Optional T3 context-size cap (0 = use the GGUF's value).
    int n_ctx = 0;

    // Sampling knobs (mirrors the CLI defaults, which match Python's
    // ChatterboxTurboTTS.generate()).
    int   top_k          = 1000;
    float top_p          = 0.95f;
    float temperature    = 0.8f;
    float repeat_penalty = 1.2f;

    // ---------------- Multilingual variant only ---------------------
    //
    // Honoured when the loaded T3 GGUF reports
    // `chatterbox.variant = "t3_mtl"`; ignored on Turbo.  The engine
    // dispatches automatically based on the variant metadata - callers
    // don't need to look at the GGUF to know which set of fields apply.
    //
    //   language     ISO 639-1 code (e.g. "en", "es", "fr"); the
    //                multilingual tokenizer wraps the input text in a
    //                <lang>...</lang> prefix per the language.  Must
    //                be one of mtl_tokenizer::supported_languages();
    //                the engine throws on unsupported inputs.
    //
    //   exaggeration Per-utterance prosody knob; >0 = more expressive,
    //                <0 = flatter.  Default 0.5 matches Python's
    //                ChatterboxMultilingualTTS.generate().
    //
    //   cfg_weight   Classifier-free guidance scale.  0 disables CFG
    //                (single forward pass per token).  >0 enables CFG
    //                (cond+uncond batched into one B=2 forward).
    //                Default 0.5 (matches the Python reference).
    //
    //   min_p        Min-p sampling threshold; 0 disables, ~0.05 is a
    //                reasonable starting point for tighter sampling
    //                without sacrificing diversity.
    std::string language     = "en";
    float       exaggeration = 0.5f;
    float       cfg_weight   = 0.5f;
    float       min_p        = 0.0f;

    // S3Gen side.  0 = library default (2-step meanflow).
    int cfm_steps = 0;

    // ---------------- Streaming synthesis ----------------------------
    //
    // When `stream_chunk_tokens > 0` AND the caller passes a non-empty
    // chunk callback to `synthesize()`, the engine runs the chunked
    // S3Gen+HiFT loop and invokes the callback with each chunk's 24 kHz
    // float32 PCM as it's produced.  The callback is called synchronously
    // from the same thread as `synthesize()`.
    //
    //   stream_chunk_tokens        Number of T3 speech tokens per chunk
    //                              (25 ~= 1 s of audio, 50 ~= 2 s).
    //                              0 = non-streaming (batch).
    //
    //   stream_first_chunk_tokens  Override for the *first* chunk so first
    //                              audio lands early while later chunks
    //                              stay large and keep overall RTF low.
    //                              0 = same as stream_chunk_tokens.
    //
    //   stream_cfm_steps           CFM Euler step count for streaming
    //                              chunks.  0 = library default (2).
    //                              Setting 1 halves CFM cost with a small
    //                              quality penalty on Turbo's meanflow
    //                              sampler.
    int stream_chunk_tokens       = 0;
    int stream_first_chunk_tokens = 0;
    int stream_cfm_steps          = 0;

    // Pass-through of the CLI's --verbose behaviour: per-stage wall times
    // on stderr.  Errors always go through regardless of this flag.
    bool verbose = false;
};

// Per-chunk PCM callback.  Receives a pointer to `samples` consecutive
// 24 kHz float32 mono samples.  The buffer is owned by the engine and
// must not be retained past the callback; copy out if you need the data.
//   `chunk_index`  0-based index of the chunk within the current utterance.
//   `is_last`      true on the final chunk (after which synthesize() returns).
// Throwing from this callback aborts synthesis (the exception propagates
// out of synthesize()).
using StreamCallback = std::function<void(
    const float * pcm, std::size_t samples, int chunk_index, bool is_last)>;

struct SynthesisResult {
    // 24 kHz mono PCM, float32 in [-1, 1].
    std::vector<float> pcm;
    int sample_rate = 24000;

    // Stats, for parity with the CLI's `BENCH:` lines.
    double t3_ms         = 0.0;
    double s3gen_ms      = 0.0;
    int    t3_tokens     = 0;
    int    audio_samples = 0;
};

class TTS_CPP_API Engine {
public:
    // Loads T3, kicks off the S3Gen preload, bakes voice conditioning if
    // reference_audio / voice_dir is set.  Throws std::runtime_error on
    // any hard failure (GGUF not found, GGUF malformed, reference wav
    // invalid, voice dir missing tensors, etc.).
    explicit Engine(const EngineOptions & opts);

    // Frees the backend + all ggml contexts.
    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;

    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize `text` into PCM.  Returns a non-empty `pcm` on success;
    // throws std::runtime_error on failure.  Empty `text` is rejected.
    //
    // Not safe to call concurrently on the same Engine instance.
    SynthesisResult synthesize(const std::string & text);

    // Same as above, but when `options().stream_chunk_tokens > 0` and
    // `on_chunk` is non-empty, runs the chunked S3Gen+HiFT loop and
    // invokes `on_chunk` with each chunk's PCM in order.  The returned
    // SynthesisResult.pcm still contains the concatenated audio (the
    // callback is an *addition*, not a replacement).  Falls through to
    // the batch path when streaming is disabled.
    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk);

    // Best-effort cancel of an in-flight synthesize() call on another
    // thread.  Safe to call concurrently with synthesize() or from any
    // thread.  Setting the flag is all this does; actual termination
    // happens at the next cancellation checkpoint inside the pipeline:
    //
    //   - between iterations of the T3 decode loop (every token);
    //   - immediately before run_encoder;
    //   - between CFM steps (top of each iteration of the loop in
    //     s3gen_synthesize_to_wav);
    //   - between run_stft and run_hift_decode;
    //   - immediately before run_hift_decode.
    //
    // ggml_backend_graph_compute calls cannot be preempted, so the
    // worst-case cancel latency is one in-flight graph submission -
    // bounded by the longest single graph among encoder, CFM step,
    // STFT, and HiFT decode.  For chatterbox today, HiFT decode is
    // the dominant one (a few hundred ms on M3 Ultra Metal at
    // HiFT-realistic shapes; longer on CPU).  CFM steps are ~80 ms
    // each on Metal but can dominate on CPU when many are queued.
    // Streaming-mode cancels apply at the same checkpoints inside
    // each chunk.
    void cancel();

    // Return the options the engine was constructed with (convenience for
    // callers that want to introspect the resolved n_gpu_layers / n_ctx).
    const EngineOptions & options() const;

    // Return the registered name of the backend the engine actually
    // resolved to during construction (e.g. "Metal", "Vulkan", "CUDA0",
    // "CPU").  Useful for telemetry / debug logs after the load-time
    // backend cascade picks one of the candidates compiled in.  Returns
    // "(unknown)" when the backend is unset (e.g. a partial construction
    // path that hasn't run init_backend yet).
    std::string backend_name() const;

    // Resolved compute device for this engine.  CPU when the build has
    // no GPU backend compiled in, when no GPU was requested
    // (n_gpu_layers <= 0), or when the requested GPU backend refused
    // to initialise.  GPU otherwise (covers Metal, CUDA, Vulkan, OpenCL
    // and any other ggml ACCEL/GPU device).  Stable for the lifetime
    // of the Engine.
    BackendDevice backend_device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace tts_cpp::chatterbox
