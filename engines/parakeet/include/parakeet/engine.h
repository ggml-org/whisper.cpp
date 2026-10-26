#pragma once

// Loaded GGUF inference: transcribe, stream, diarize, and backend metadata behind one Engine class.
//
// Loads weights once; subsequent calls pay mel + encoder + decode only. Model kind (CTC, RNN-T, TDT,
// EOU, Nemotron, Sortformer) comes from GGUF metadata.
//
// Transcription:
//   - transcribe / transcribe_samples — one-shot wav or PCM to text.
//   - transcribe_stream — full audio up front; segments via callback. CTC/RNN-T/TDT/EOU
//     encode once then slice output; Nemotron uses its cache-aware encoder.
//   - stream_start — push PCM over time. CTC/RNN-T/TDT/EOU re-encode sliding windows;
//     Nemotron keeps bounded attention and convolution caches.
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
//     `compute_mel()` / `compute_mel_wav()` are the exception and may
//     run on a worker thread during a `transcribe*` call; see
//     PrecomputedMel.
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
//     final partial-chunk tail segment is not emitted (CTC/RNN-T/TDT/EOU/
//     Nemotron Mode 3). Always call `finalize()` if you care about those.

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
    // built with `GGML_BACKEND_DL=ON` (including Android prebuilds)
    // should pass an explicit dir
    // because the .so files ship next to the host's binary in a
    // platform-specific subfolder rather than on the system loader's
    // path.
    // On Android this directory is also prepended once to FastRPC's
    // DSP_LIBRARY_PATH before discovery, for bundled Hexagon DSP libraries.
    // Existing DSP paths (or legacy ADSP paths) and vendor defaults remain.
    //
    // No-op on builds where ggml is statically linked
    // (`GGML_BACKEND_DL=OFF`, e.g. desktop dev cmake builds and the
    // Apple xcframework). On those, every backend is registered at
    // constructor time from inside libggml and no filesystem scan
    // takes place.
    std::string backends_dir;

    // Sets `$GGML_OPENCL_CACHE_DIR` before the first backend init so
    // ggml-opencl persists `clCreateProgramWithBinary` blobs across
    // process restarts (implemented by qvac-ext-ggml@speech). Strongly
    // recommended on Android where
    // the cold `clBuildProgram` cost dominates first-utterance
    // latency; pass a writable per-app directory (typically the
    // app's `cacheDir` from the host platform).
    //
    // Honoured only on `__ANDROID__` builds; ignored elsewhere
    // (desktop OpenCL platforms don't enable that binary-cache behavior
    // and would otherwise pollute the user's tmpdir).
    //
    // Leave empty to keep the existing `$GGML_OPENCL_CACHE_DIR` env
    // value (or no cache at all). Wrapper scripts that already
    // export the env take precedence.
    std::string opencl_cache_dir;

    // Opt-in cold-start mitigation.
    //
    // When `prewarm == true`, the Engine constructor runs one
    // synthetic encoder-only forward pass using a
    // `prewarm_audio_seconds`-long all-zero mel input. Decoder
    // predictor/joint graphs are not executed. Nemotron 3
    // Diarization instead runs one offline pass of that length and
    // one chunk at the default live-stream geometry, so both its
    // offline and streaming kernels are built. The
    // intent is to amortise the *first-call* cold cost into
    // construction:
    //
    //   * Metal:   triggers the MSL → MTLPipelineState compile.
    //   * OpenCL:  triggers `clBuildProgram` for every kernel
    //              variant the encoder graph touches; binaries
    //              get cached by the qvac-ext-ggml@speech backend
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

    // ── Long-form offline transcription (bounded-memory encoder) ──────────
    //
    // transcribe_samples() / transcribe_samples_stream() run the conformer
    // encoder over the whole input in a single graph. Self-attention is
    // O(T_enc^2) in encoder frames, so multi-hour inputs build tens-of-GB
    // score tensors and OOM the process (a ~90 min file needs ~100 GB). When
    // the input would exceed `long_form_window_frames` encoder frames, the
    // encoder is instead slid over the audio in overlapping windows, the
    // shared context is trimmed at the interior seams, and the per-window
    // encoder outputs are concatenated into a single buffer the decoder
    // consumes exactly as before. The mel-spectrogram is still computed once
    // over the whole input, so per-feature CMVN statistics stay global; only the
    // encoder's self-attention becomes window-local, so a windowed run's
    // committed frames closely approximate -- but are not bit-identical to -- a
    // full single-pass encode. Inputs that already fit in one window skip
    // windowing entirely and keep the bit-identical single-pass path, so
    // short/typical files are unaffected.
    //
    // long_form_window_frames -- requested encoder-frame ceiling per window.
    // Whatever the source, the effective window is floored to a small minimum
    // and never exceeds the encoder's trained positional range
    // (pos_emb_max_len), which is a hard ceiling:
    //     > 0  explicit request (still clamped to pos_emb_max_len).
    //     = 0  auto: the window the model recommends when its GGUF carries
    //          one (TranscribePsy: 750 frames with 94 of context, the 60 s /
    //          25 % setting validated on phones), otherwise at most
    //          min(pos_emb_max_len, 3750), which keeps each window inside the
    //          model's trained positional range and caps the attention score
    //          tensor at a few hundred MB; the input is then split into the
    //          fewest such windows and they share it equally, so a recording
    //          just past one window gets two half windows.
    //     < 0  disabled: always single-pass (legacy behaviour; OOMs on long
    //          inputs -- for A/B testing only).
    //
    // Nemotron 3 Diarization reads long_form_window_frames with the same
    // signs: inputs longer than the window (auto: min(pos_emb_max_len, 1125),
    // 90 s) are diarized by the model's cached long-form path, 30 s chunks
    // with 1040 ms of left and right context, instead of one graph. A single
    // graph loses speaker identity past about two minutes; < 0 keeps the
    // single graph, which fails past pos_emb_max_len.
    //
    // long_form_context_frames -- shared left/right context each window carries
    // so its committed centre frames see enough neighbourhood to match the
    // single-pass encoder. Trimmed off after the encoder runs, so it never
    // appears twice in the output. 0 => auto (min(256, window/4) each side).
    int long_form_window_frames  = 0;
    int long_form_context_frames = 0;

    // Model-family-specific language selection.
    // - Nemotron: locale alias or "auto"; empty resolves to "auto".
    // - Multilingual CTC: required language id (e.g. "hi", "ta") when the
    //   GGUF advertises parakeet.ctc.lang_* ranges.
    // - Monolingual CTC and existing transducer families: ignored.
    // Unsupported Nemotron locales and multilingual CTC language IDs throw.
    std::string language;

    // "auto" (or empty) preserves n_gpu_layers and runtime GPU tiering.
    // "cpu", "cuda", "vulkan", "opencl", "webgpu", "metal" and "hexagon" explicitly
    // select a backend; "hexagon" selects HTP0. An exact ggml device name (e.g. HTP0)
    // is also accepted. Explicit selection overrides n_gpu_layers and
    // fails construction if unavailable or initialization fails. Individual
    // unsupported operations can still use the existing CPU scheduler.
    std::string backend = "auto";
};

// Resolved compute device the Engine is actually running on, after
// registry-based runtime tiering and any fallbacks (Adreno policy,
// OpenCL extension probe, missing GPU
// build, kernel-init failure). This is the *post-fallback* truth and
// will not match the user's `EngineOptions::n_gpu_layers` request when
// a fallback occurred.
enum class BackendDevice : int {
    CPU = 0,
    GPU = 1,
    NPU = 2,
};

// A mel spectrogram detached from the Engine that produced it, so the mel
// stage can run on one thread while the encoder runs on another.
//
// Intended for a batch of separate utterances. Each one's mel depends only on
// its own audio (CMVN is a whole-utterance statistic over that utterance), so
// utterance i+1's mel can be built while utterance i is still in the encoder.
// For a single recording there is nothing to overlap: its mel must complete
// before the first encoder dispatch.
struct PrecomputedMel {
    std::vector<float> mel;
    int    n_frames      = 0;
    int    n_mels        = 0;
    int    audio_samples = 0;
    int    sample_rate   = 16000;
    double preprocess_ms = 0.0;

    bool empty() const { return n_frames <= 0; }
};

struct EngineResult {
    std::string text;
    std::vector<int32_t> token_ids;

    // True only when every encoder invocation used Core ML successfully for
    // this transcription. Unlike Engine::encoder_on_coreml(), this reports
    // actual per-call routing rather than sidecar availability.
    bool encoder_used_coreml = false;

    double preprocess_ms = 0.0;
    double encoder_ms    = 0.0;
    double decode_ms     = 0.0;
    // Only non-zero on the Core ML independent-segment path, where together
    // with the three above they account for `total_ms`: pause-cut planning,
    // then per-segment encoder-output concatenation, lane setup and stitching.
    double segment_plan_ms = 0.0;
    double assemble_ms     = 0.0;
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

    // Mel ahead of time, encoder later. See PrecomputedMel.
    //
    // `compute_mel*` is the one member that IS safe to call concurrently with
    // a `transcribe*` call on the same Engine: it reads only the mel config,
    // which is immutable after load, and allocates its own scratch rather than
    // touching the Engine's. Everything else in the "no concurrent calls" rule
    // above still applies; in particular two concurrent `transcribe_mel`
    // calls are no more legal than two concurrent `transcribe_samples` ones.
    //
    // `transcribe_mel(compute_mel(...))` returns exactly what
    // `transcribe_samples` returns on the same audio.
    PrecomputedMel compute_mel(const float * samples,
                               int n_samples,
                               int sample_rate) const;

    PrecomputedMel compute_mel_wav(const std::string & wav_path) const;

    EngineResult transcribe_mel(PrecomputedMel mel);

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

    // "ctc", "rnnt", "tdt", "eou", "nemotron", or "sortformer", reflecting
    // the parakeet.model.type metadata of the loaded GGUF.
    std::string model_type() const;

    // Resolved primary compute device: CPU, GPU, or NPU (Hexagon).
    // In auto mode n_gpu_layers and GPU availability determine CPU fallback.
    // Explicit backend requests fail construction if unavailable.
    BackendDevice backend_device() const;

    // Human-readable name of the active backend, e.g. "CUDA0", "Metal",
    // "Vulkan0", "OpenCL", "HTP0", "CPU". Sourced from `ggml_backend_name()`
    // when an accelerator backend is active; literal "CPU" otherwise. Stable for
    // the lifetime of the Engine.
    std::string backend_name() const;

    // Reports whether an Apple Core ML encoder or diarizer sidecar loaded for
    // this model. Returns the sidecar's compute label (for example
    // "coreml-all") when loaded; otherwise identical to backend_name().
    // Encoder sidecars cover CTC/IndicConformer, Unified RNN-T, TDT, EOU,
    // Nemotron, and Sortformer v2.1; their decoders, and the Sortformer speaker
    // head without a diarizer sidecar, run on the ggml backend reported by
    // backend_name(). Diarizer sidecars run the whole Nemotron 3 Diarization
    // or Sortformer v2.1 network after feature stacking or subsampling.
    std::string encoder_backend() const;

    // True when an Apple Core ML encoder or diarizer sidecar was found. This
    // is a load-status query, not a guarantee that every call shape uses Core
    // ML; the per-call truth is EngineResult::encoder_used_coreml.
    // CTC/IndicConformer, Unified RNN-T, and TDT pad shorter inputs to the
    // compiled capacity and window longer offline inputs. EOU and Nemotron use
    // Core ML only at the exact compiled shape: shorter calls are never padded,
    // longer EOU inputs run on ggml, and longer Nemotron inputs take the
    // cache-aware ggml path. Diarizer sidecars take any call up to their
    // largest compiled capacity and leave larger calls on ggml.
    // Always false on non-Apple builds and when no sidecar is present.
    bool encoder_on_coreml() const;

    // True when a GPU was detected but the engine fell back to CPU because the
    // available path is known-bad (currently Adreno 6xx OpenCL without the
    // explicit override). A CPU backend with this set is expected.
    bool gpu_unsupported() const;

    struct Impl;

private:
    // Everything from the encoder onwards, shared by every transcribe entry
    // point. `mel` is consumed. `prior_ms` is wall time already spent before
    // entry (the mel, and the wav read when there was one) so `total_ms`
    // stays end-to-end whichever entry point built the mel.
    EngineResult transcribe_mel_impl(PrecomputedMel && mel, double prior_ms);

    std::unique_ptr<Impl> pimpl_;
};

}
