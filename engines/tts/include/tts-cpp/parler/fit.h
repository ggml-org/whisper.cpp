#pragma once

// Parler memory-fit preflight: project whether the Parler GGUF (T5 encoder +
// delay-pattern decoder LM + DAC codec, one file) fits the device memory
// available right now, without reading any weight data.
//
// The projection mirrors one Engine load plus one synthesize() of the
// configured workload BY CONSTRUCTION: the same loader runs metadata-only
// (same backend policy, FA probe and KV-type resolution, GPU fused-weight
// stack sized through the same type/shape guards), and the same graph
// builders are priced through ggml's size-only APIs.  What is priced:
//
//   weights  -- the whole GGUF's tensor set on the resolved backend, plus
//               the GPU-only fused-weight buffer (per-layer qkv stacks and
//               the stacked LM heads) where a real load would build it
//   state    -- the decoder self-attention KV slab at the resolved kv type
//               and the GGUF's n_ctx, plus the per-description cross-
//               attention K/V buffer at the projected description length
//   lm       -- the resident decoder arena (max of the description-encode,
//               prefill, and deepest decode-step graphs -- one gallocr
//               serves all three, exactly as the Engine dispatches them)
//   codec    -- the DAC window arena at the projected frame count (the
//               decode is windowed, so this saturates once the sequence
//               exceeds one window plus its convolution context)
//
// Scope: the projection covers the batch synthesize() path.  Streaming
// re-decodes the growing DAC-code prefix per chunk through the same bounded
// window arena, so the batch projection at the same frame count is also the
// streaming peak (same shapes, priced once).
//
// On the CPU backend a real load maps the GGUF weights in place (mmap)
// instead of allocating a buffer; the projection prices the allocate-and-
// stream fallback, which bounds the fully-touched mapping from above --
// wrong only in the strict direction.
//
// Core ML: when the Engine would run the DAC on its Core ML sidecar (the
// GGUF's name minus its quant tag plus `-dac.mlmodelc`, next to it, in a
// TTS_CPP_COREML build, not disabled by PARLER_COREML_DISABLE), the result
// is Error with reason
// "coreml-memory-estimate-unavailable": resident Core ML models and their
// prediction buffers cannot be projected by the native fitter. The same
// refusal applies when coreml_decoder is requested in a Core ML build.
//
// See tts-cpp/fit.h for the status contract and result layout.

#include "tts-cpp/export.h"
#include "tts-cpp/fit.h"

#include <cstdint>
#include <string>

namespace tts_cpp {
namespace parler {

struct FitOptions {
    // Required; same role as EngineOptions::model_gguf_path.
    std::string model_gguf_path;

    // Same semantics as the EngineOptions fields of the same name.
    int         n_gpu_layers = 0;
    std::string backend;
    std::string backends_dir;
    bool coreml_decoder = false; // mirror EngineOptions; Core ML memory is not projectable

    // ── Workload ──────────────────────────────────────────────────────────
    // Token count of the voice description fed to the T5 encoder (the
    // per-description cross-attention K/V and the encoder graph follow it).
    int description_tokens = 50;

    // Token count of the transcript prompt (the decoder prefill width is
    // prompt_tokens + 1: the BOS start frame is appended).
    int prompt_tokens = 128;

    // Same semantics as EngineOptions::max_frames: generation length cap in
    // delayed decoder steps.  0 = the GGUF's own max_length.  The decode
    // depth and the DAC frame count follow it.
    int max_frames = 0;

    // Free-memory headroom that must remain on the device for the projection
    // to count as fitting.
    uint64_t margin_bytes = 256ull * 1024 * 1024;
};

// Project the model + workload in `opts` against the device memory available
// right now.  Reads only GGUF metadata, never weight data; builds and
// measures graphs but never allocates or executes them.  Never throws for a
// "does not fit" outcome (a valid Failure result) or an unreadable model
// (Error); see FitStatus in tts-cpp/fit.h.
TTS_CPP_API FitResult fit_params(const FitOptions & opts);

}  // namespace parler
}  // namespace tts_cpp

// CLI front-end over parler::fit_params (the parler-fit-params tool); lives
// in the library so hosts can link it directly.  Exit code == fit status:
// 0 fits, 1 does not fit, 2 error.
extern "C" TTS_CPP_API int parler_fit_cli_main(int argc, char ** argv);
