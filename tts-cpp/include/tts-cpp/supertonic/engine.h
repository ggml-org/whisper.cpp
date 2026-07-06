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
#include <map>
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

    // external (cloned) voice injection. A Supertonic voice
    // is just two small float tensors (`style_ttl` timbre + `style_dp`
    // pacing); the baked-in presets are these tensors stored in the GGUF.
    // When an external voice is supplied here it overrides the `voice`
    // name lookup, letting the engine synthesize an arbitrary (e.g.
    // cloned) voice while synthesis stays fully on-device and
    // format-agnostic.  Resolution precedence, highest first:
    //
    //   1. voice_style_ttl + voice_style_dp  (both non-empty: in-memory tensors)
    //   2. voice_json_path                   (load a voice JSON from disk)
    //   3. voice                             (baked-in preset name)
    //
    // The flattened element order of the tensors / JSON `data` arrays is
    // the same row-major layout the synthesis pipeline consumes, so
    // injecting a preset's tensors is bit-for-bit identical to selecting
    // that preset by name.  Element counts are validated against the
    // model's baked voice tensors at construction time (a mismatch throws).
    //
    // Path to a voice JSON:
    //   { "style_ttl": { "data": [...] }, "style_dp": { "data": [...] },
    //     "metadata": { ... } }
    std::string voice_json_path;
    // Or inject the style tensors directly (flattened, row-major).  Both
    // must be non-empty to take effect; takes precedence over
    // voice_json_path.
    std::vector<float> voice_style_ttl;
    std::vector<float> voice_style_dp;

    std::string language = "en";
    int   steps    = 0;
    float speed    = 0.0f;
    int   seed     = 42;
    int   n_threads     = 0;
    int   n_gpu_layers  = 0;

    // desired output sample rate in Hz. Supertonic natively
    // emits at the model's metadata rate (typically 44.1 kHz); when this is a
    // positive rate other than the native one the engine resamples the final
    // PCM (Kaiser-windowed sinc) and reports it on
    // SynthesisResult::sample_rate.  Honoured on both the batch and streaming
    // paths.  0 keeps the native rate (default; zero behaviour change).
    // Validated at construction to 0 or [8000, 192000] Hz.
    int   output_sample_rate = 0;

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

    // Vulkan adapter index. Passed verbatim to
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

    // round 6 — extra deny-list for F16 weight
    // materialization, layered ON TOP of the curated allow-list
    // in `should_materialise_f16_weight()`.  Each entry is a
    // substring; if ANY non-empty entry is found inside a
    // tensor's source name, that tensor stays at its native
    // storage type (typically F32) even when `f16_weights` is
    // on.  Empty strings are skipped (no-op) so a stray empty
    // entry from a config-file typo doesn't silently disable F16
    // weights for the whole model.
    //
    // Use cases:
    //   - A/B testing a specific tensor pattern without recompiling.
    //   - Force-keeping a tensor as F32 if drift on a particular
    //     adapter / driver / shape is observed.
    //   - Safety net for new tensor patterns added in future
    //     GGUFs that the curated allow-list inadvertently scoops in.
    //
    // Default empty (zero behaviour change for every existing
    // operator config).  No effect when `f16_weights == 0`.
    std::vector<std::string> f16_weights_deny_list;

    // round 4 — multi-dtype K/V flash-attention dispatch
    // for the vector estimator's attention sites.  Generalises the
    // round-1 `f16_attn` boolean (F16 vs F32 only) to:
    //
    //   -1 → auto (default — falls back to `f16_attn`'s value;
    //              identical behaviour to round 1 / 2 / 3 / 5 / 6
    //              for every existing operator config)
    //    0 → f32  (force F32 K/V — useful for parity-harness runs
    //              and for triaging a perf cliff caused by F16
    //              underflow on a specific model + adapter combo)
    //    1 → f16  (same as `f16_attn=1`; OpenCL adreno fast path,
    //              Vulkan `kernel_flash_attn_f32_f16_*`)
    //    2 → bf16 (Vulkan coopmat2 — wider exponent range than F16,
    //              same precision; identical bandwidth to F16, no
    //              underflow on small attention scores; falls back
    //              to f32 on adapters without coopmat2)
    //    3 → q8_0 (Vulkan + half the K/V upload bandwidth on
    //              workloads that are upload-bound; falls back to
    //              f32 on backends without Q8_0 K/V flash-attn)
    //
    // Probe-gated graceful fallback to F32 on adapters that don't
    // support the requested dtype — same advisory-probe semantics
    // as `f16_attn`'s round-1 auto-policy, so an operator config
    // setting `--kv-attn-type bf16` works on both NVIDIA Ampere+
    // and Intel ARC (BF16 effective on the former; silent F32 on
    // the latter) without crashing.  Out-of-range values throw
    // loudly to surface CLI typos.
    //
    // When the resolved value is non-f32, the legacy
    // `model.use_f16_attn` boolean is ALSO updated to
    // `(resolved == f16)` so any code path still keying on the
    // boolean (text-encoder / duration / vocoder; not the vector
    // estimator) sees the historically-correct value.
    int kv_attn_type = -1;

    // Directory to scan for dynamically-loaded ggml backends
    // (`libspeech-ggml-vulkan.so`, `libspeech-ggml-opencl.so`,
    // `libspeech-ggml-cpu-android_armv8.2_1.so`, ...). Forwarded to
    // `ggml_backend_load_all_from_path()` on the first Engine
    // construction in the process; subsequent constructions reuse the
    // already-populated registry.
    //
    // Leave empty to fall back to ggml's default search path
    // (`ggml_backend_load_all()`). Embedded host applications built
    // with `GGML_BACKEND_DL=ON` (the Android / Linux non-Apple
    // default; see CMakeLists.txt) should pass an explicit dir so the
    // .so files ship next to the host's binary in a per-module
    // folder rather than relying on `LD_LIBRARY_PATH` / `dlopen()`
    // heuristics. No-op on `GGML_BACKEND_DL=OFF` (static-link)
    // builds.
    std::string backends_dir;

    // Sets `$GGML_OPENCL_CACHE_DIR` before the first backend init so
    // ggml-opencl persists `clCreateProgramWithBinary` blobs across
    // process restarts. Strongly recommended on Android where the
    // cold `clBuildProgram` cost dominates first-utterance latency;
    // pass a writable per-app directory (typically the app's
    // `cacheDir` from the host platform).
    //
    // Honoured only on `__ANDROID__` builds; ignored elsewhere.
    std::string opencl_cache_dir;

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

    // follow-up — first-synth-latency pre-warming.
    //
    // When non-empty, the Engine ctor invokes `warm_up(prewarm_text)`
    // immediately after the GGUF load + voice validation, running one
    // throwaway synth on the supplied text.  On Vulkan / OpenCL this
    // forces the GPU shader pipelines for every Supertonic stage to
    // compile up-front (the in-tree thread_local graph caches handle
    // every subsequent call but can't avoid the first pipeline-compile
    // cost — measured ~hundreds of ms on first synth on Adreno + RADV
    // in chatterbox PROGRESS.md), so the operator-visible first synth
    // call sees ~steady-state latency.  No effect on CPU (no shader
    // compilation cost; warm_up returns immediately on
    // `model.backend_is_cpu`).
    //
    // Pre-warm text should be similar in length to representative
    // production input — the per-stage graph caches are keyed on
    // (text_len, latent_len) tuples, so a too-short pre-warm leaves
    // a graph-rebuild on the first real call (still saves the
    // shader-compile cost; only the cgraph allocation is repeated).
    // Default empty (no pre-warming).
    std::string prewarm_text;

    // round 7 — Vulkan env-var passthrough.
    //
    // Applied to the process environment via `set_env_if_unset`
    // semantics just before `init_supertonic_backend()` runs.
    // Each key MUST start with `GGML_VK_` (operator-config typo
    // guard — invalid keys throw at engine-construction time, no
    // partial-application).
    //
    // Operator-set env vars (already present in the environment
    // when the Engine ctor runs) WIN over these overrides — lets
    // a debugging operator force-disable a setting from the shell
    // without recompiling, while still letting an EngineOptions
    // configuration set the same knob in production.
    //
    // Example use cases (the round-7 CLI flags map onto these):
    //   {"GGML_VK_PREFER_HOST_MEMORY",      "1"}  // --vulkan-prefer-host-memory
    //   {"GGML_VK_DISABLE_COOPMAT2",        "1"}  // --vulkan-disable-coopmat2
    //   {"GGML_VK_DISABLE_BFLOAT16",        "1"}  // --vulkan-disable-bfloat16
    //   {"GGML_VK_PERF_LOGGER",             "1"}  // --vulkan-perf-logger
    //   {"GGML_VK_ASYNC_USE_TRANSFER_QUEUE","1"}  // --vulkan-async-transfer
    //
    // Default empty (zero behaviour change for every existing
    // operator config).
    std::map<std::string, std::string> vulkan_env_overrides;
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

    // follow-up — first-synth-latency pre-warming.
    //
    // Runs one throwaway synth on `text` to force every per-stage
    // GPU graph cache to populate and every Vulkan / OpenCL shader
    // pipeline to compile up-front.  The PCM result is discarded.
    // Subsequent `synthesize()` calls hit the warmed caches +
    // pre-compiled pipelines, so the operator-visible first synth
    // sees steady-state latency.
    //
    // No-op on CPU backends (no pipeline cache to warm).  Auto-
    // invoked by the ctor when `EngineOptions::prewarm_text` is
    // non-empty; callers can also invoke explicitly mid-life when
    // they need to warm a different shape (e.g. switching from a
    // short-prompt to a long-prompt workload).
    //
    // Throws on the same conditions as `synthesize()` — if the
    // throwaway synth fails for any reason, the failure surfaces
    // here rather than being swallowed.
    void warm_up(const std::string & text);

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

    // True when a GPU device was present but unusable (outside the validated
    // allowlist), so we fell back to CPU. Always false when backend_device() == GPU.
    bool gpu_unsupported() const;

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
