#pragma once

// Memory-fit preflight: project whether a Parakeet GGUF fits the device
// memory available right now, without reading any weight data.
//
// The fitter opens the GGUF metadata-only (`no_alloc`), wires the model's
// tensor handles exactly as a real load would, and then *measures* the
// peak-resident allocation set of one offline transcribe (or diarize) on a
// fresh Engine -- the weight buffers (including the CPU-repack extra buffers
// on CPU runs), every encoder compute graph the runtime's graph cache will
// hold at once (a windowed long-form pass keeps up to three window graphs
// resident, not one), and the decoder-side state and fixed-shape graphs --
// using ggml's size-only allocator APIs. Nothing is allocated on the device
// and no graph is executed, so a fit call is cheap (milliseconds + one
// metadata read) and safe to run before committing to a full Engine load.
//
// Scope: the projection covers the offline paths. Streaming sessions for the
// legacy families (CTC/RNN-T/TDT/EOU) build smaller per-chunk graphs but
// rotate through the same 3-slot graph cache with session-dependent keys, so
// the offline projection is a good guide but not a proven upper bound for
// every streaming configuration. Nemotron is the exception: its native
// cache-aware streaming path is modelled explicitly -- the projection covers
// one offline transcribe (including the full-length locale-prompt projection
// graph, which scales with audio_seconds because the prompt conditioning runs
// over the whole stitched encoder output, not per window) PLUS one live
// streaming session at the FitOptions::nemotron_chunk_ms operating point (its
// step graph, per-chunk pre-encode graph, and host-resident caches).
// Nemotron 3 Diarization runs offline inputs up to the long-form window as
// one graph and longer ones through its cached long-form path; every graph
// shares the model scheduler, so its compute projection is the largest of
// the single-pass graph (bounded by the window), one chunk at the default
// live geometry (1040 ms chunk, 80 ms right context, 264 speaker-cache and
// 80 FIFO rows), and one 30 s long-form chunk when the workload needs it.
// On Apple builds both diarization families also charge the Core ML
// diarizer sidecars those calls can load (FitResult::coreml_bytes).
//
// Status semantics follow the SDK's @qvac/model-fit contract:
//   Success -- a projection was made and it fits (result.fits == true).
//   Failure -- a projection was made and it does NOT fit. This is a valid
//              answer, not an error.
//   Error   -- no projection could be made (unreadable model, no backend
//              device registered, measurement failure). An empty device
//              registry is always Error, never Success: a projection made
//              against a machine the fitter cannot see is worse than none.

#include "export.h"

#include <cstdint>
#include <string>

namespace parakeet {

enum class FitStatus : int {
    Success = 0,
    Failure = 1,
    Error   = 2,
};

PARAKEET_API const char * fit_status_name(FitStatus status);

struct FitOptions {
    std::string model_gguf_path;

    // Same semantics as EngineOptions: > 0 requests the GPU backend (with the
    // same runtime tiering and known-bad-device fallbacks a real load
    // applies); <= 0 projects for the CPU backend.
    int n_gpu_layers = 0;
    int n_threads    = 0;

    // Same semantics as EngineOptions::backends_dir: directory scanned for
    // dynamically-loaded ggml backends on the first registry init in the
    // process. Leave empty for ggml's default search path.
    std::string backends_dir;

    bool verbose = false;

    // ── Workload ──────────────────────────────────────────────────────────
    // Longest single transcribe() input the projection must accommodate.
    // Device-side encoder memory is bounded by the long-form window (the
    // engine slides the encoder over longer inputs), so the device projection
    // saturates once audio_seconds exceeds one window; the host-side buffers
    // (full-input mel, encoder-output slab, ...) keep growing with it.
    // Exceptions that keep growing on the device too: Sortformer (no
    // windowing), Nemotron (its locale-prompt projection graph runs over
    // the full stitched encoder output). Nemotron 3 Diarization saturates at
    // its long-form window; with long-form disabled, inputs past its
    // positional limit are Error/"workload-too-large".
    float audio_seconds = 300.0f;

    // Same semantics as the EngineOptions fields of the same name (0 = auto).
    // Pass the values the real Engine will run with so the projected window
    // matches.
    int long_form_window_frames  = 0;
    int long_form_context_frames = 0;

    // Nemotron only (ignored for every other model type): the cache-aware
    // streaming operating point (StreamingOptions::chunk_ms) whose live
    // session the projection must also accommodate -- the step graph and the
    // per-layer caches scale with the operating point's right context. Must
    // be one of the GGUF's parakeet.nemotron.allowed_chunk_ms values
    // (80/160/320/560/1120 on the shipped checkpoint); anything else is
    // Error/"invalid-arguments", mirroring Engine::stream_start. 0 (default)
    // projects the largest allowed operating point -- the biggest resident
    // set, so the default verdict is safe for any chunk_ms the host later
    // picks.
    int nemotron_chunk_ms = 0;

    // Free-memory headroom that must remain on the device for the projection
    // to count as fitting.
    uint64_t margin_bytes = 256ull * 1024 * 1024;

    // Same semantics as EngineOptions::backend: "auto" (or empty) follows
    // n_gpu_layers; "cpu", "opencl", "hexagon" or an exact ggml device name
    // projects for that backend and reports "no-backend-device" when it is
    // unavailable.
    std::string backend = "auto";
};

// Projected bytes on the resolved compute device.
struct FitBreakdown {
    uint64_t weights_bytes         = 0;  // weight buffers; on CPU runs includes the repack extra buffers
    uint64_t encoder_compute_bytes = 0;  // worst-case single-window encoder graph
    uint64_t decoder_state_bytes   = 0;  // persistent decoder state (transducer h/c/pred/enc_proj, ...)
    uint64_t decoder_compute_bytes = 0;  // fixed-shape decoder graphs
    uint64_t total_bytes           = 0;  // sum of the above
};

struct FitResult {
    FitStatus   status = FitStatus::Error;
    bool        fits   = false;  // status == Success
    // "fits" | "does-not-fit" | "model-unreadable" | "no-backend-device" |
    // "measurement-failed" | "invalid-arguments" | "workload-too-large"
    // ("model-type-not-supported" is retired: every model type the loader
    // accepts is now modelled, Nemotron included)
    std::string reason;

    std::string model_type;     // "ctc" | "rnnt" | "tdt" | "eou" | "nemotron" | "sortformer" | "nemotron-diarization" | "moss-transcribe"
    std::string model_variant;  // GGUF parakeet.model_variant, may be empty

    // Resolved compute device, after the same runtime tiering and fallbacks a
    // real load applies (Adreno policy, Mali routing, missing GPU build, ...).
    std::string device_name;
    bool        device_is_cpu = false;
    // True when the device's memory pool IS system RAM (the CPU backend,
    // integrated GPUs, Apple Metal unified memory): host_bytes then competes
    // with `device` for the same physical memory and the verdict charges
    // both against the free figure. False only for discrete-VRAM devices.
    bool        device_shares_host_memory = false;
    uint64_t    device_free_bytes  = 0;
    uint64_t    device_total_bytes = 0;

    FitBreakdown device;

    // Host-side memory the workload needs in addition to `device`: the
    // full-input mel spectrogram and encoder-output slabs (these scale with
    // audio_seconds, unlike the windowed device buffers), and the
    // host-dequantised decoder weights on CPU decode paths. When
    // device_shares_host_memory these compete with `device` for the same
    // physical RAM and the fit test accounts for both.
    uint64_t host_bytes = 0;

    // Apple builds only: projected system RAM of the Core ML diarizer
    // sidecars (`<model>-diarizer-<frames>.mlmodelc`) the workload can load.
    // Nemotron 3 Diarization charges the bundle for its offline input, or
    // every bundle up to its 30 s long-form block, plus the bundle for one
    // default live chunk; Sortformer v2.1 charges the bundle for its offline
    // input. Each bundle costs twice its weights file, plus a fixed Core ML
    // runtime cost once, as measured on an M4. The ggml weights stay loaded
    // for the fallback, so this adds to `device`, and with
    // device_shares_host_memory the verdict counts it. Zero without
    // sidecars, without a covering capacity, or with
    // PARAKEET_COREML_DISABLE / PARAKEET_COREML_DIARIZER_DISABLE set.
    uint64_t coreml_bytes = 0;

    // Human-readable projection table (multi-line, suitable for logging).
    std::string report;
};

// Project the model + workload in `opts` against the device memory available
// right now. Reads only GGUF metadata, never weight data; builds and measures
// graphs but never allocates or executes them. Never throws for a
// "does not fit" outcome (that is a valid Failure result) or an unreadable
// model (Error); see FitStatus.
PARAKEET_API FitResult fit_params(const FitOptions & opts);

}  // namespace parakeet
