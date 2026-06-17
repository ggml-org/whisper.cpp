#pragma once

// Loaded GGUF inference: transcribe, stream, diarize, and backend metadata behind one Engine class.
//
// Loads weights once; subsequent calls pay mel + encoder + decode only. Model kind (CTC, TDT,
// EOU, Sortformer) comes from GGUF metadata.
//
// Transcription:
//   - transcribe / transcribe_samples — one-shot wav or PCM to text.
//   - transcribe_stream — full audio up front; segments via callback (offline encoder, chunked output).
//   - stream_start — push PCM over time with left/right context windows (live streaming).
//
// Diarization (Sortformer GGUFs):
//   - diarize / diarize_samples — offline segments + speaker_probs.
//   - diarize_start — sliding-history streaming diarization (push PCM).
//
// Combined ASR + diarization: transcribe_with_speakers in <parakeet/attributed.h>.
//
// Usage (transcription):
//
//     #include <parakeet/engine.h>
//     using parakeet::Engine;
//     using parakeet::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_gguf_path = "models/parakeet-tdt-0.6b-v3.q8_0.gguf";
//     opts.n_threads       = 8;
//
//     Engine engine(opts);
//     for (const auto & wav_path : wavs) {
//         auto result = engine.transcribe(wav_path);
//         std::puts(result.text.c_str());
//     }
//
// Threading model:
//
//   - Concurrent `transcribe()` / `diarize()` / `transcribe_stream()`
//     calls on the same instance are not supported (the encoder's graph
//     allocator is shared mutable state). Wrap an Engine in your own
//     mutex if you need that, or hold one Engine per worker.
//
//   - `cancel()` is safe to call from any thread while another thread
//     is inside `transcribe*` / `diarize*`. It causes the running call
//     to bail out at the next chunk boundary and return.
//
//   - Each new call to `transcribe*` / `diarize*` resets the cancel
//     flag at entry, so a `cancel()` racing with a subsequent
//     `transcribe()` from the *same* thread will be lost. If you need
//     to hard-stop and not start a new call, gate the next entry on
//     your own application-level flag.
//
//   - `~Engine()` does NOT wait for in-flight calls; destroying an
//     Engine while another thread is inside a `transcribe*` call is
//     undefined behaviour. Call `cancel()` and join the working thread
//     before destruction.
//
//   - `~StreamSession()` and `~SortformerStreamSession()` cancel the
//     session; they do NOT call `finalize()`. If you let a session
//     destruct without an explicit `finalize()` call, any audio that
//     hadn't yet rolled into a chunk is dropped, the synthetic
//     `is_final=true` terminator is not emitted (Sortformer), and the
//     final partial-chunk tail segment is not emitted (CTC/TDT
//     Mode 3). Always call `finalize()` if you care about those.

#include "export.h"
#include "streaming.h"
#include "diarization.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace parakeet {

struct EngineOptions {
    std::string model_gguf_path;

    int n_gpu_layers = 0;
    int n_threads    = 0;

    bool verbose     = false;

    // Directory to scan for dynamically-loaded ggml backends
    // (`libspeech-ggml-vulkan.so`, `libspeech-ggml-opencl.so`,
    // `libspeech-ggml-cpu-android_armv8.2_1.so`, ...). Forwarded to
    // `ggml_backend_load_all_from_path()` on the first Engine
    // construction in the process; subsequent constructions reuse the
    // already-populated registry.
    //
    // Leave empty to fall back to ggml's default search path
    // (`ggml_backend_load_all()`), which walks compile-time defaults
    // (`$EXE_DIR`, `LD_LIBRARY_PATH`, ...). Embedded host applications
    // built with `GGML_BACKEND_DL=ON` (the Android / Linux non-Apple
    // default; see CMakeLists.txt) should pass an explicit dir
    // because the .so files ship next to the host's binary in a
    // platform-specific subfolder rather than on the system loader's
    // path.
    //
    // No-op on builds where ggml is statically linked
    // (`GGML_BACKEND_DL=OFF`, e.g. desktop dev cmake builds and the
    // Apple xcframework). On those, every backend is registered at
    // constructor time from inside libggml and no filesystem scan
    // takes place.
    std::string backends_dir;

    // Sets `$GGML_OPENCL_CACHE_DIR` before the first backend init so
    // ggml-opencl persists `clCreateProgramWithBinary` blobs across
    // process restarts (see the program-binary-cache patch on
    // qvac-ext-ggml@speech). Strongly recommended on Android where
    // the cold `clBuildProgram` cost dominates first-utterance
    // latency; pass a writable per-app directory (typically the
    // app's `cacheDir` from the host platform).
    //
    // Honoured only on `__ANDROID__` builds; ignored elsewhere
    // (desktop OpenCL platforms don't ship the binary-cache patch
    // and would otherwise pollute the user's tmpdir).
    //
    // Leave empty to keep the existing `$GGML_OPENCL_CACHE_DIR` env
    // value (or no cache at all). Wrapper scripts that already
    // export the env take precedence.
    std::string opencl_cache_dir;

    // Opt-in cold-start mitigation.
    //
    // When `prewarm == true`, the Engine constructor runs one
    // synthetic forward pass through the encoder (and, on TDT
    // GGUFs, through the per-step LSTM/joint graphs too) using a
    // `prewarm_audio_seconds`-long all-zero mel input. The
    // intent is to amortise the *first-call* cold cost into
    // construction:
    //
    //   * Metal:   triggers the MSL → MTLPipelineState compile.
    //   * OpenCL:  triggers `clBuildProgram` for every kernel
    //              variant the encoder graph touches; binaries
    //              get cached via the program-binary-cache patch 
    //              when GGML_OPENCL_CACHE_DIR is set.
    //   * Vulkan:  triggers vkCreateGraphicsPipelines.
    //   * CUDA:    triggers cuGraphInstantiate.
    //   * CPU:     pre-builds the ggml graph nodes + scratch.
    //
    // Default off (back-compat: callers who wanted the old
    // first-call-pays-cold behaviour keep getting it). Adds the
    // cold-start cost to construction time instead of first
    // transcribe; useful for embedded / interactive UX where
    // first-utterance latency is the user-perceived metric.
    bool  prewarm                = false;
    float prewarm_audio_seconds  = 1.0f;
};

// Resolved compute device the Engine is actually running on, after the
// load-time backend cascade (CUDA / Metal / Vulkan / OpenCL) and any
// fallbacks (Adreno-tier policy, OpenCL extension probe, missing GPU
// build, kernel-init failure). This is the *post-fallback* truth and
// will not match the user's `EngineOptions::n_gpu_layers` request when
// a fallback occurred.
enum class BackendDevice : int {
    CPU = 0,
    GPU = 1,
};

struct EngineResult {
    std::string text;
    std::vector<int32_t> token_ids;

    double preprocess_ms = 0.0;
    double encoder_ms    = 0.0;
    double decode_ms     = 0.0;
    double total_ms      = 0.0;

    int audio_samples    = 0;
    int sample_rate      = 16000;
    int mel_frames       = 0;
    int encoder_frames   = 0;
};

class PARAKEET_API Engine {
public:
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    EngineResult transcribe(const std::string & wav_path);

    EngineResult transcribe_samples(const float * samples,
                                    int n_samples,
                                    int sample_rate);

    EngineResult transcribe_stream(const std::string & wav_path,
                                   const StreamingOptions & opts,
                                   StreamingCallback on_segment);

    EngineResult transcribe_samples_stream(const float * samples,
                                           int n_samples,
                                           int sample_rate,
                                           const StreamingOptions & opts,
                                           StreamingCallback on_segment);

    std::unique_ptr<StreamSession> stream_start(const StreamingOptions & opts,
                                                StreamingCallback on_segment);

    // Diarization (Sortformer models only). Throws if loaded GGUF
    // is a transcription model.
    DiarizationResult diarize(const std::string & wav_path,
                              const DiarizationOptions & opts = {});

    DiarizationResult diarize_samples(const float * samples,
                                      int n_samples,
                                      int sample_rate,
                                      const DiarizationOptions & opts = {});

    // Live Sortformer session (push PCM). See SortformerStreamSession for how speaker
    // IDs behave across overlapping chunk passes.
    std::unique_ptr<SortformerStreamSession> diarize_start(
        const SortformerStreamingOptions & opts,
        SortformerSegmentCallback on_segment);

    void cancel();

    // Used by transcribe_with_speakers to slice audio per Sortformer
    // segment and feed each slice through the ASR engine. Public so
    // downstream callers can inspect the underlying model_type and
    // route accordingly.
    bool is_diarization_model() const;
    bool is_transcription_model() const;

    const EngineOptions & options() const;

    // "ctc", "tdt", "eou", or "sortformer", reflecting the
    // parakeet.model.type metadata of the loaded GGUF.
    std::string model_type() const;

    // Resolved compute device for this Engine's loaded model. CPU when
    // the build has no GPU backend compiled in, when no GPU was
    // requested (n_gpu_layers <= 0), or when the requested GPU backend
    // refused to initialise (e.g. Adreno-6xx forced to CPU,
    // GGML_OPENCL_ALLOW_UNKNOWN_GPU=1 but the device lacks the
    // required subgroup-size extension). GPU otherwise.
    BackendDevice backend_device() const;

    // Human-readable name of the active backend, e.g. "CUDA0", "Metal",
    // "Vulkan0", "OpenCL", "CPU". Sourced from `ggml_backend_name()`
    // when a GPU backend is active; literal "CPU" otherwise. Stable for
    // the lifetime of the Engine.
    std::string backend_name() const;

    // True when a GPU was detected but the engine fell back to CPU because it is
    // a known-bad backend (Mali). A CPU backend with this set is expected, not a
    // regression.
    bool gpu_unsupported() const;

    struct Impl;

private:
    std::unique_ptr<Impl> pimpl_;
};

}
