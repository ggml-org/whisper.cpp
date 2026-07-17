#pragma once

// ACE-Step FSQ detokenizer — ggml compute engine (QVAC-21921).
//
// Bridges the LM audio semantic codes to the DiT context latents:
//   codes [T_5Hz] --FSQ decode--> [6, T_5Hz] --project_out--> [2048, T_5Hz]
//   per 5Hz token: embed + special_tokens broadcast (5 frames) + 2L Qwen3
//   encoder + norm + proj_out --> [64, 5]  => context latents [64, T_25Hz]
//   with T_25Hz = T_5Hz * 5.
//
// Weights live in the DiT GGUF under "tokenizer.quantizer.*" and
// "detokenizer.*". The 2-layer encoder is the shared Qwen3 backbone
// (qwen3_block.h). Ported from acestep.cpp/src/fsq-detok.h. No new ggml op.

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct DetokModel;  // opaque

// Load the detokenizer weights from the DiT GGUF onto `backend` (borrowed).
DetokModel * detok_model_load(const std::string & path, ggml_backend_t backend, bool verbose);
void         detok_model_free(DetokModel * m);
size_t       detok_model_weight_bytes(const DetokModel * m);

// Decode T_5Hz LM codes into context latents. context_out is filled with
// [64, T_25Hz] (frame t of channel c at index t*64 + c), T_25Hz = T_5Hz * 5.
// Returns T_25Hz on success, -1 on failure. Caller sizes context_out to
// 64 * T_5Hz * 5 floats.
int detok_model_decode(DetokModel * m, const int * codes, int T_5Hz, float * context_out);

} // namespace tts_cpp::acestep
