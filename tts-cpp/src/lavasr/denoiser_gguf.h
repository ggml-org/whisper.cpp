#pragma once

// Load LavaSR denoiser weights from the GGUF produced by
// scripts/convert-lavasr-denoiser-to-gguf.py into a DenoiserWeights struct.
// Mirrors enhancer_gguf.h — this will be the only denoiser translation unit that
// depends on ggml/gguf; the forward math (denoiser_core) and the pipeline
// (denoiser) stay pure C++.
//
// SCAFFOLD: not yet implemented (returns false with an explanatory error).

#include "denoiser_core.h"

#include <string>

namespace tts_cpp::lavasr {

// Returns true on success.  On failure, sets *err (if non-null) and returns
// false (e.g. file missing, wrong architecture, missing tensors).
bool load_denoiser_gguf(const std::string & path, DenoiserWeights & out,
                        std::string * err = nullptr);

} // namespace tts_cpp::lavasr
