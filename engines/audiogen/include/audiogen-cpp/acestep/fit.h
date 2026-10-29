#pragma once

// Memory-fit preflight for the ACE-Step music-generation pipeline: project
// whether the four stage GGUFs + a generation workload fit the device memory
// available right now, without reading any weight data.
//
// The fitter resolves the same backends and per-stage placement a real
// Engine::create would (engine_backends.h / stage_placement.h), opens each
// stage GGUF metadata-only, wires its tensors exactly as the real loaders do,
// and prices every allocation through ggml's size-only APIs
// (ggml_backend_alloc_ctx_tensors_from_buft_size for weight/KV buffers,
// ggml_gallocr_reserve_n_size for the real compute graphs at the workload's
// shapes). Nothing is allocated on any device
// and no graph is executed, so a fit call is cheap and safe to run before
// committing to a full Engine load.
//
// Residency: by default the engine time-shares the six stages (each is loaded
// right before its step and freed right after -- QVAC-22955), so the default
// projection is the PEAK PHASE: the largest per-pool footprint any single
// pipeline phase holds at once (the text/cond encoders overlap in one phase;
// everything else is one stage plus its compute). With ACESTEP_KEEP_STAGES (or
// FitOptions::keep_stages = 1) every stage stays resident and the projection
// is the SUM of all weights/KV/persistent graphs plus the largest ephemeral
// compute. The verdict per memory pool (every device a stage is placed on, and
// host RAM for CPU-placed stages and host-side buffers) must hold for the fit
// to pass; devices that share host memory (unified memory, or an NPU that
// reports no memory of its own) add their requirements to host RAM's.
//
// Status semantics follow the SDK's @qvac/model-fit contract:
//   Success -- a projection was made and it fits (result.fits == true).
//   Failure -- a projection was made and it does NOT fit. This is a valid
//              answer, not an error.
//   Error   -- no projection could be made (unreadable model, no backend
//              device registered, measurement failure). An empty device
//              registry is always Error, never Success.

#include "audiogen-cpp/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

enum class FitStatus : int {
    Success = 0,
    Failure = 1,
    Error   = 2,
};

AUDIOGEN_API const char * fit_status_name(FitStatus status);

struct FitOptions {
    // Same path resolution as EngineOptions: point models_dir at the directory
    // holding the four GGUFs, or set explicit per-stage paths (explicit wins).
    std::string models_dir;
    std::string text_enc_model_path;  // Qwen3-Embedding-*.gguf
    std::string lm_model_path;        // acestep-5Hz-lm-*.gguf
    std::string dit_model_path;       // acestep-v15-*.gguf
    std::string vae_model_path;       // vae-*.gguf

    int  n_threads    = 0;  // 0 = hardware concurrency (backend resolution only)
    int  n_gpu_layers = 0;  // > 0 requests the GPU stack, same fallbacks as a real load
    bool verbose      = false;
    std::string backend = "auto";  // same semantics as EngineOptions::backend
    std::string lm_backend = "auto";  // same semantics as EngineOptions::lm_backend

    // Same semantics as EngineOptions::backends_dir: directory scanned for
    // dynamically-loaded ggml backends on the first registry init.
    std::string backends_dir;

    // ── Workload ──────────────────────────────────────────────────────────
    // Longest single generate() the projection must accommodate. Drives the
    // LM code budget (duration * 5 codes + 100, the pipeline's own cap), the
    // DiT temporal length (25 latent frames/s, rounded up to the patch size)
    // and the VAE output length. The DiT compute graph grows quadratically
    // with this (sliding-window self-attention mask), so it dominates long
    // requests; the VAE decode is windowed and saturates.
    float duration_seconds = 60.0f;

    // Prompt-side token budgets (tokenised lengths, template overhead
    // included). text_tokens feeds the text encoder and the cond text
    // projector; lyric_tokens the embed lookup and the cond lyric encoder
    // (whose sliding-window attention is quadratic in it).
    int text_tokens  = 160;
    int lyric_tokens = 512;

    // LM prompt length for the prefill graph. 0 = derive as
    // text_tokens + lyric_tokens + 64 (prompt template overhead).
    int lm_prompt_tokens = 0;

    // Cap on LM Phase-2 decode length. 0 = derive like the pipeline:
    // duration * 5 + 100. The KV budget itself is fixed (the engine loads the
    // LM with max_seq_len 2048, 2 KV sets); the decode-step graph is projected
    // at the full KV window either way (strict worst case).
    int lm_max_new_tokens = 0;

    // LM classifier-free guidance scale (GenerateParams::lm_cfg_scale,
    // default 2.0). > 1 projects the CFG decode path: the batched 2-stream
    // decode graph where the backend supports it, cond+uncond prefills.
    float lm_cfg_scale = 2.0f;

    // DiT guidance (GenerateParams::guidance_scale). 0 = auto (turbo 1.0 = no
    // CFG, base/sft 7.0 = CFG). > 1 adds the APG host-side buffers; the DiT
    // graph itself is shared between the cond/uncond forwards (N stays 1).
    float guidance_scale = 0.0f;

    // Project a request that supplies source/reference audio (cover / lego /
    // repaint / timbre reference): adds the VAE-encoder phase (encoder weights
    // + one encode-window graph) to the projection.
    bool with_source_audio = false;

    // Stage residency to project: -1 = mirror the engine (ACESTEP_KEEP_STAGES
    // env, the default), 0 = force the lazy/low-memory projection, 1 = force
    // the everything-resident projection.
    int keep_stages = -1;

    // Free-memory headroom that must remain on every pool for the projection
    // to count as fitting.
    uint64_t margin_bytes = 256ull * 1024 * 1024;
};

// One pipeline stage's projected footprint.
struct FitStageProjection {
    std::string name;         // "textenc" | "lm" | "cond" | "detok" | "dit" | "vae"
    std::string device_name;  // ggml device the stage loads on (after placement)
    bool        on_gpu = false;

    uint64_t weights_bytes      = 0;  // backend weight buffer (LM: + compact tied head)
    uint64_t weights_mmap_bytes = 0;  // CPU map-in-place portion (file-backed host pages)
    uint64_t state_bytes        = 0;  // LM KV cache
    uint64_t compute_bytes      = 0;  // largest compute arena the stage holds at once
    uint64_t host_bytes         = 0;  // stage-phase host-RAM buffers (masks, latents, PCM, ...)
};

struct FitDevicePool {
    std::string name;
    bool        shares_host_memory = false;
    uint64_t    free_bytes         = 0;
    uint64_t    total_bytes        = 0;
    uint64_t    peak_bytes         = 0;
};

struct FitResult {
    FitStatus   status = FitStatus::Error;
    bool        fits   = false;  // status == Success
    // "fits" | "does-not-fit" | "model-unreadable" | "no-backend-device" |
    // "measurement-failed" | "invalid-arguments" | "workload-too-large"
    std::string reason;

    std::string model_name;  // DiT GGUF general.name
    bool        is_turbo = false;

    // Primary compute device (DiT/VAE; the per-stage rows carry their own).
    // Its ggml device name, as ggml_backend_dev_by_name finds it: "CPU",
    // "CUDA0", "MTL0", "GPUOpenCL".
    std::string device_name;
    bool        device_is_cpu = false;
    // True when the primary device's memory pool IS system RAM (the CPU
    // backend, integrated GPUs, Apple Metal unified memory, Android's OpenCL
    // GPUs, a device that reports no memory of its own): the host pool and the
    // device pool are then the same physical memory and the verdict charges
    // both against it. False only for discrete-VRAM devices.
    bool        device_shares_host_memory = false;
    // The device's free and total memory. A device that reports none of its
    // own carries the CPU's figures.
    uint64_t    device_free_bytes  = 0;
    uint64_t    device_total_bytes = 0;
    // Host pool (the CPU backend device): CPU-placed stages, mmapped weights,
    // and host-side workload buffers live here. Equals the device pool on
    // unified memory.
    uint64_t    host_free_bytes  = 0;
    uint64_t    host_total_bytes = 0;

    bool stages_resident = false;  // residency mode the projection used

    std::vector<FitStageProjection> stages;

    // Peak projected bytes per pool under the projected residency mode.
    uint64_t peak_device_bytes = 0;  // on the primary device pool
    uint64_t peak_host_bytes   = 0;  // on the host pool (CPU stages + host buffers)

    std::vector<FitDevicePool> extra_devices;

    // Human-readable projection table (multi-line, suitable for logging).
    std::string report;
};

// Project the models + workload in `opts` against the memory available right
// now. Reads only GGUF metadata, never weight data; builds and sizes the real
// compute graphs but never allocates or executes them. Never throws for a
// "does not fit" outcome (that is a valid Failure result) or an unreadable
// model (Error); see FitStatus.
AUDIOGEN_API FitResult fit_params(const FitOptions & opts);

}  // namespace tts_cpp::acestep
