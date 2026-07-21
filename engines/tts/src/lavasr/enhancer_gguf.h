#pragma once

// Load the LavaSR enhancer weights from the GGUF produced by
// scripts/convert-lavasr-enhancer-to-gguf.py into an EnhancerWeights struct
// (host float32, ready for the scalar CPU forward in enhancer_core).
//
// This is the only LavaSR translation unit that depends on ggml/gguf; the
// forward math (enhancer_core) and the pipeline (enhancer) are pure C++.

#include "enhancer_core.h"

#include <string>

namespace tts_cpp::lavasr {

// Returns true on success.  On failure, sets *err (if non-null) and returns
// false (e.g. file missing, wrong architecture, missing tensors).
bool load_enhancer_gguf(const std::string & path, EnhancerWeights & out,
                        std::string * err = nullptr);

} // namespace tts_cpp::lavasr
