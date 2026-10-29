#pragma once

// Memory-fit preflight types shared by the tts-cpp engines (audio8,
// chatterbox): project whether a model set + workload fits the device memory
// available right now, WITHOUT reading any weight data.
//
// A fitter opens each GGUF metadata-only (`no_alloc`), wires the tensor
// handles exactly as a real load would, and prices every allocation the real
// engine makes -- weight buffers, persistent caches (KV slabs), and the
// compute graphs of one synthesis -- through ggml's size-only allocator APIs
// (ggml_backend_alloc_ctx_tensors_from_buft_size, ggml_gallocr_reserve_n_size,
// ggml_backend_sched_reserve_size). Nothing is allocated on the device and no
// graph runs, so a fit call is cheap (milliseconds + metadata reads) and safe
// to run before committing to a full Engine load.
//
// Status semantics follow the SDK's @qvac/model-fit contract:
//   Success -- a projection was made and it fits (result.fits == true).
//   Failure -- a projection was made and it does NOT fit. This is a valid
//              answer, not an error.
//   Error   -- no projection could be made (unreadable model, no backend
//              device registered, measurement failure). An empty device
//              registry is always Error, never Success: a projection made
//              against a machine the fitter cannot see is worse than none.
//
// Per-engine entry points live in tts-cpp/audio8/fit.h and
// tts-cpp/chatterbox/fit.h, tts-cpp/supertonic/fit.h, and tts-cpp/pocket/fit.h.

#include "tts-cpp/export.h"

#include <cstdint>
#include <string>

namespace tts_cpp {

enum class FitStatus : int {
    Success = 0,
    Failure = 1,
    Error   = 2,
};

TTS_CPP_API const char * fit_status_name(FitStatus status);

// Projected bytes on the resolved compute device.
struct FitBreakdown {
    uint64_t weights_bytes       = 0;  // weight buffers of every sub-model the pipeline loads
    uint64_t state_bytes         = 0;  // persistent state: KV-cache slabs, conditioning buffers
    uint64_t lm_compute_bytes    = 0;  // language-model / acoustic-model graphs
    uint64_t codec_compute_bytes = 0;  // codec / vocoder graphs
    uint64_t total_bytes         = 0;  // sum of the above
};

struct FitResult {
    FitStatus   status = FitStatus::Error;
    bool        fits   = false;  // status == Success
    // "fits" | "does-not-fit" | "model-unreadable" | "no-backend-device" |
    // "measurement-failed" | "invalid-arguments" | "workload-too-large"
    std::string reason;

    // Which pipeline the projection is for (e.g. "audio8",
    // "chatterbox-t3-mtl", "chatterbox-t3-turbo").
    std::string model_variant;

    // Resolved compute device, after the same registry walk and runtime
    // fallbacks a real load applies (validated-backend allowlists, Adreno
    // tier policy, missing GPU build, ...). Its ggml device name, as
    // ggml_backend_dev_by_name finds it: "CPU", "CUDA0", "MTL0", "GPUOpenCL".
    std::string device_name;
    bool        device_is_cpu = false;
    // True when the device's memory pool IS system RAM (the CPU backend,
    // integrated GPUs, Apple Metal unified memory, Hexagon, Android's OpenCL
    // GPUs): host_bytes then competes with `device` for the same physical
    // memory and the verdict charges both against the free figure. False only
    // for discrete-VRAM devices.
    bool        device_shares_host_memory = false;
    // The device's free and total memory. A device that shares host memory
    // and reports none of its own carries the CPU's figures.
    uint64_t    device_free_bytes  = 0;
    uint64_t    device_total_bytes = 0;

    FitBreakdown device;

    // Host-side memory the workload needs in addition to `device`: waveform
    // and latent slabs that scale with the synthesis length, staging buffers,
    // and the CPU-fallback portion of scheduler-dispatched graphs. When
    // device_shares_host_memory these compete with `device` for the same
    // physical RAM and the fit test accounts for both.
    uint64_t host_bytes = 0;

    // Human-readable projection table (multi-line, suitable for logging).
    std::string report;
};

}  // namespace tts_cpp
