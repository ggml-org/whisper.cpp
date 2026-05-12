#pragma once

// Persistent Supertonic engine.
//
// Loads the Supertonic GGUF once, validates the requested voice / language /
// step count, and keeps the model resident so subsequent calls to
// `synthesize()` only pay the per-call preprocess + duration + text-encoder +
// vector-estimator + vocoder cost - not the GGUF tensor load.
//
// Usage:
//
//     using tts_cpp::supertonic::Engine;
//     using tts_cpp::supertonic::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_gguf_path = "models/supertonic.gguf";
//     opts.n_gpu_layers    = 0;                      // 0 = CPU; >0 enables Metal
//                                                    // on macOS / CUDA / Vulkan /
//                                                    // OpenCL when compiled in.
//                                                    // Metal on Apple silicon is the
//                                                    // fastest backend as of 2026-05-12
//                                                    // (~35× realtime on M2, beats
//                                                    // ggml-CPU, ONNX-CPU and ONNX-CoreML
//                                                    // on every stage that matters).
//                                                    // See PROGRESS_SUPERTONIC.md.
//
//     Engine engine(opts);
//     for (const auto & line : lines) {
//         auto result = engine.synthesize(line);
//         write_wav(result.pcm, result.sample_rate);
//     }
//
// Threading model:
//   - synthesize() on the same Engine instance is NOT safe to call
//     concurrently - the per-stage thread_local caches and the seeded
//     RNG are per-instance shared state.
//   - synthesize() on different Engine instances from different
//     threads is safe.  The supertonic generation_id (set per Engine
//     ctor) keys the stage-internal caches so two Engines don't collide.
//   - cancel() is safe from any thread.
//
// Implemented in src/supertonic_engine.cpp on top of the library-internal
// helpers in src/supertonic_internal.h.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::supertonic {

// Compute precision for matmul weights inside the model buffer.  Selects
// how the GGUF's stored q8_0 weights are loaded into the resident model:
//   - F32  (default): expand q8_0 to f32 at load time.  CPU path uses
//          cblas/AMX f32 matmul.  Metal path uses kernel_mul_mat_f32_f32.
//          Highest accuracy + simplest, but on Metal misses the 4×
//          weight-bandwidth win of running the native q8_0 matmul kernel.
//   - F16  (Phase B1): expand q8_0 to f16 at load time, run f16 matmul
//          with f32 accumulator.  ~2× less activation bandwidth on Metal,
//          may drift slightly across the 5 CFM steps (parity tolerance
//          relaxed to ~1e-2 L_inf).
//   - Q8_0 (Phase A3): keep weights as q8_0 in the model buffer, let
//          ggml's quantized matmul kernels dispatch directly.  Metal-only
//          (Phase A3 makes the load logic asymmetric: q8_0 on Metal, f32
//          on CPU).
enum class Precision {
    F32,
    F16,
    Q8_0,
};

struct EngineOptions {
    // Required.
    std::string model_gguf_path;

    // Empty / zero values use the defaults stored in the GGUF metadata.
    std::string voice;
    std::string language = "en";
    int   steps    = 0;
    float speed    = 0.0f;
    int   seed     = 42;
    int   n_threads     = 0;
    int   n_gpu_layers  = 0;

    // Compute precision for matmul weights — see Precision enum above.
    // Default F32 is the current behaviour (load q8_0 GGUF, expand to f32).
    // F16 / Q8_0 are non-default GPU paths (Metal-validated).
    Precision precision = Precision::F32;

    // F16 K/V flash-attention in the vector estimator.  When -1, the
    // engine auto-enables this on GPU backends (non-CPU) and disables
    // it on CPU; pass 1 / 0 to force the setting regardless of the
    // resolved backend.  Triggers the OpenCL `flash_attn_f32_f16`
    // path on Adreno; mirrors chatterbox's `--cfm-f16-kv-attn`.  No
    // effect on CPU (the cblas attention path is already efficient).
    // On Vulkan dispatches `kernel_flash_attn_f32_f16_*` (head_dim=64
    // satisfies the `HSK % 8 == 0` supports_op gate; see
    // `ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT`).
    int f16_attn = -1;

    // QVAC-18605 — Vulkan adapter index.  Passed verbatim to
    // `ggml_backend_vk_init(idx)` when the build is compiled with
    // `GGML_VULKAN=ON` and `n_gpu_layers > 0`.  Range-checked
    // against `ggml_backend_vk_get_device_count()` at load; an
    // out-of-range value throws (no silent CPU fallback — that
    // would mask CLI typos / wrong-machine config).  Default 0
    // (the historical hard-coded value).  Negative values are
    // reserved for a future "auto-pick best device" policy.
    int vulkan_device = 0;

    // F16 storage type for the audit-identified hot matmul /
    // pointwise-conv weights (vector-estimator attention W_*,
    // pwconv1/pwconv2 across every convnext block, vocoder
    // head linear, text-encoder linears, …).  Same -1/0/1 tri-state
    // as `f16_attn`: -1 auto (on for GPU, off for CPU); 0 or 1 force.
    // Halves the GPU read bandwidth into those ops with a small
    // (≤ 2e-3 abs / 5e-3 cosine) numerical drift on the end-to-end
    // synth.  Mirrors chatterbox's CHATTERBOX_F16_CFM gate.
    // Orthogonal to `precision`: this is a per-op runtime selector for
    // the OpenCL hot-weight materialisation, while `precision` decides
    // the storage type of all matmul weights uniformly.
    int f16_weights = -1;

    // Optional path to a .npy file containing the initial noise tensor of
    // shape [1, latent_channels, latent_len] (float32).  When provided,
    // latent_len is taken from the npy file (overriding the duration-
    // predicted length) and the seeded RNG is bypassed.  Useful for
    // byte-exact reproduction of an ONNX/PyTorch reference run.
    std::string noise_npy_path;

    // ---------------- Streaming synthesis ----------------------------
    //
    // When `stream_chunk_tokens > 0` AND a non-empty callback is passed
    // to synthesize(), the engine splits `text` into chunks of roughly
    // `stream_chunk_tokens` Unicode code points (Supertonic's text-token
    // grain — see supertonic_text_to_ids), runs the full pipeline per
    // chunk, and invokes the callback with each chunk's PCM as it's
    // produced.  The returned SynthesisResult.pcm still contains the
    // concatenated audio (the callback is an *addition*, not a
    // replacement).  Streaming is disabled when stream_chunk_tokens == 0
    // OR the callback is empty — both paths fall through to the batch
    // path with no per-chunk overhead.
    //
    //   stream_chunk_tokens         Target chunk size in text tokens.
    //                               ~50 ≈ 1-3 s English audio; CJK
    //                               languages are denser so a lower
    //                               target (~25-30) tends to feel
    //                               better.  0 disables streaming.
    //
    //   stream_first_chunk_tokens   Override for the *first* chunk so
    //                               first audio lands early while later
    //                               chunks stay at the larger target
    //                               for steady-state throughput.
    //                               0 = same as stream_chunk_tokens.
    //
    //   stream_chunk_tolerance_pct  Boundary-snap window for CLAUSE and
    //                               WHITESPACE fallbacks (±N% of target).
    //                               Sentence-end is searched on a much
    //                               wider implicit window (target/2 to
    //                               3× target) because sentence-aligned
    //                               chunks let the per-chunk duration
    //                               predictor and attention phrase
    //                               naturally; mid-clause cuts work
    //                               (continuation flag in preprocess
    //                               avoids the artificial trailing
    //                               period that would otherwise make
    //                               the model speak the stub as a
    //                               complete sentence) but produce
    //                               audible pauses + rate shifts at
    //                               seams since the model is not
    //                               streaming-trained.  Default 20.
    //
    //   stream_min_chunk_tokens     Hard floor on every chunk's size.
    //                               Effective targets are
    //                               max(target, min) — below the floor
    //                               the model glitches on stub input
    //                               (dropped / muddled phonemes,
    //                               verified empirically).  Trailing
    //                               chunks shorter than the floor are
    //                               merged into the previous chunk.
    //                               Default 30.
    int stream_chunk_tokens        = 0;
    int stream_first_chunk_tokens  = 0;
    int stream_chunk_tolerance_pct = 20;
    int stream_min_chunk_tokens    = 30;
};

// Per-chunk PCM callback for streaming synthesis.  Receives a pointer to
// `samples` consecutive float32 mono samples at SynthesisResult::sample_rate
// (typically 44.1 kHz — read from model metadata, not hard-coded).  The
// buffer is owned by the engine and must not be retained past the
// callback; copy out if you need the data.
//   `chunk_index`  0-based index of the chunk within the current synth.
//   `is_last`      true on the final chunk (after which synthesize() returns).
// Throwing from this callback aborts synthesis (the exception propagates
// out of synthesize()).
using StreamCallback = std::function<void(
    const float * pcm, std::size_t samples, int chunk_index, bool is_last)>;

struct SynthesisResult {
    std::vector<float> pcm;
    int   sample_rate = 44100;
    float duration_s  = 0.0f;
};

// Persistent engine.  Loads the GGUF once at construction; subsequent
// synthesize() calls reuse the resident model.
class TTS_CPP_API Engine {
public:
    // Loads the Supertonic GGUF, initialises the backend, validates
    // opts.voice / opts.language up front.  Throws std::runtime_error
    // on any hard failure (GGUF not found, GGUF malformed, unsupported
    // voice).
    explicit Engine(const EngineOptions & opts);

    // Frees the backend + all ggml contexts.
    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;

    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize `text` into PCM (44.1 kHz mono float32 by default;
    // see SynthesisResult::sample_rate).  Throws std::runtime_error
    // on failure.  Empty `text` is rejected.
    //
    // Not safe to call concurrently on the same Engine instance.
    SynthesisResult synthesize(const std::string & text);

    // Same as above, but when `options().stream_chunk_tokens > 0` and
    // `on_chunk` is non-empty, runs the chunked pipeline and invokes
    // `on_chunk` with each chunk's PCM in order.  The returned
    // SynthesisResult.pcm still contains the concatenated audio (the
    // callback is an *addition*, not a replacement).  Falls through to
    // the batch path when either condition is false.
    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk);

    // Best-effort cancel of an in-flight synthesize() call on another
    // thread.  Setting the flag is all this does; actual termination
    // happens at the next cancellation check inside the vector-
    // estimator loop (one step is the worst-case cancel latency).
    void cancel();

    // Return the options the engine was constructed with (convenience
    // for callers that want to introspect the resolved n_gpu_layers /
    // n_threads after defaults are applied).
    const EngineOptions & options() const;

    // Return the registered name of the backend the engine actually
    // resolved to during construction (e.g. "CPU", "Metal").  Returns
    // "(unknown)" when the backend is unset.
    std::string backend_name() const;

    // Resolved compute device.  CPU when the build has no GPU backend
    // compiled in, when no GPU was requested (n_gpu_layers <= 0), or
    // when the requested GPU backend refused to initialise.  GPU
    // otherwise.  Stable for the lifetime of the Engine.
    BackendDevice backend_device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience one-shot wrapper around Engine.  Equivalent to:
//   Engine e(opts); return e.synthesize(text);
// Use the Engine class directly for any host that synthesizes more
// than once - this wrapper pays the full GGUF load + free per call.
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::supertonic
