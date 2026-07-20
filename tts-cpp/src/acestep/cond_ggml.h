#pragma once

// ACE-Step condition encoder — ggml compute engine (QVAC-21921).
//
// Builds the DiT's cross-attention states `enc_hidden` [2048, S_total] from:
//   - text_hidden  [1024, S_text]  : Qwen3-Embedding text-encoder output
//   - lyric_embed  [1024, S_lyric] : CPU vocab lookup of lyric tokens
//   - timbre_feats [64,   S_ref]   : reference-audio features (optional)
// via a lyric encoder (8L Qwen3, bidirectional), a timbre encoder (4L Qwen3),
// and a text projector (Linear 1024->2048). Packed as [lyric | timbre[0:1] |
// text_proj]. Weights live in the DiT GGUF under the `encoder.` prefix, plus the
// `null_condition_emb` used by classifier-free guidance. Ported from
// acestep.cpp/src/cond-enc.h. No new ggml op.

#include "ggml-backend.h"

#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct CondModel;  // opaque

CondModel * cond_model_load(const std::string & dit_gguf_path, ggml_backend_t backend, bool verbose);
void        cond_model_free(CondModel * m);
size_t      cond_model_weight_bytes(const CondModel * m);

// null_condition_emb [2048] (F32), used by CFG on non-turbo runs.
const std::vector<float> & cond_model_null_emb(const CondModel * m);

// First frame [64] of the DiT GGUF's silence_latent. Fed to the timbre encoder
// as the text2music timbre input (empty if the GGUF lacks silence_latent).
const std::vector<float> & cond_model_silence_frame(const CondModel * m);

// Encode conditioning into `enc_hidden` [2048, S_total] (2048 contiguous per
// token). Set timbre_feats=nullptr / S_ref=0 to skip timbre (text2music path).
// Writes S_total to *out_enc_S. Returns false on failure.
bool cond_model_forward(CondModel *          m,
                        const float *        text_hidden,
                        int                  S_text,
                        const float *        lyric_embed,
                        int                  S_lyric,
                        const float *        timbre_feats,
                        int                  S_ref,
                        std::vector<float> & enc_hidden,
                        int *                out_enc_S);

} // namespace tts_cpp::acestep
