#pragma once

// Chatterbox memory-fit preflight: project whether the T3 + S3Gen GGUF pair
// fits the device memory available right now, without reading any weight
// data.
//
// The projection mirrors one Engine load plus one synthesize() of the
// configured workload BY CONSTRUCTION: the same loaders run metadata-only
// (same variant peek, backend policy and KV-type resolution -- including the
// per-backend downgrades a real load applies), and the same graph builders
// are priced through ggml's size-only APIs.  What is priced:
//
//   weights  -- the T3 buffer (Turbo GPT-2 or MTL Llama; the MTL fused wqkv
//               stack on GPU backends) plus the whole S3Gen GGUF (flow
//               encoder + CFM + HiFT + s3tokenizer + campplus + built-in
//               voice), exactly the resident set an Engine holds
//   state    -- the T3 KV slab at the resolved kv_cache_type and n_ctx
//               (the MTL variant packs the CFG cond+uncond pair, 2x)
//   lm       -- the worst of the T3 prompt graph and the deepest decode-step
//               graph (they share one allocator)
//   codec    -- ALL FIVE resident S3Gen stage arenas (conformer encoder, CFM
//               estimator -- B=2 CFG where the runtime batches it, F0, STFT,
//               HiFT) at the shapes one full generation produces; these
//               caches stay alive after the first synthesis
//
// Scope: the projection covers steady-state synthesis with the built-in
// voice (reference-audio conditioning is capped to the same prompt lengths,
// so the steady state matches).  The TRANSIENT peak of baking a reference
// voice (the S3Tokenizer encoder is temporarily allocated next to the T3
// weights) is NOT yet included -- an explicit follow-up.
//
// See tts-cpp/fit.h for the status contract and result layout.

#include "tts-cpp/export.h"
#include "tts-cpp/fit.h"

#include <cstdint>
#include <string>

namespace tts_cpp {
namespace chatterbox {

struct FitOptions {
    // Same roles as EngineOptions: both GGUFs are required; the T3 variant
    // (turbo vs mtl) is picked from GGUF metadata exactly as a real load
    // picks it.
    std::string t3_gguf_path;
    std::string s3gen_gguf_path;

    // Same semantics as the EngineOptions fields of the same name.
    int         n_gpu_layers = 0;
    int         n_ctx        = 0;   // 0 = the GGUF's own context length
    std::string kv_cache_type;      // "", "f32", "f16", "q8_0" -- resolved
                                    // against the backend like a real load
    std::string backends_dir;

    // ── Workload ──────────────────────────────────────────────────────────
    // Text tokens of the longest single synthesis segment.
    int text_tokens = 128;

    // Same semantics as EngineOptions::n_predict: the speech-token budget of
    // one generation.  The T3 decode depth and every S3Gen stage shape
    // follow it.
    int n_predict = 1000;

    // Free-memory headroom that must remain on the device for the projection
    // to count as fitting.
    uint64_t margin_bytes = 256ull * 1024 * 1024;

    // Same semantics as EngineOptions::backend; appended last so existing
    // positional initializers stay valid.
    std::string backend;
};

// Project the model pair + workload in `opts` against the device memory
// available right now.  Reads only GGUF metadata, never weight data; builds
// and measures graphs but never allocates or executes them.  Never throws
// for a "does not fit" outcome (a valid Failure result) or an unreadable
// model (Error); see FitStatus in tts-cpp/fit.h.
TTS_CPP_API FitResult fit_params(const FitOptions & opts);

}  // namespace chatterbox
}  // namespace tts_cpp

// CLI front-end over chatterbox::fit_params (the chatterbox-fit-params
// tool); lives in the library so hosts can link it directly.  Exit code ==
// fit status: 0 fits, 1 does not fit, 2 error.
extern "C" TTS_CPP_API int chatterbox_fit_cli_main(int argc, char ** argv);
