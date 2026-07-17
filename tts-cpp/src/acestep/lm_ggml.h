#pragma once

// ACE-Step LM (ace-lm, Qwen3 0.6B) — ggml compute engine (QVAC-21921).
//
// Autoregressive Qwen3 causal LM with a persistent KV cache: prefill the prompt
// then decode audio-code / text tokens one step at a time. Same Qwen3 backbone
// as the encoders (shared qwen3_block.h) plus KV-cache read/write via set_rows
// and a tied LM head (logits = embed_tokens^T @ hidden). Ported from
// acestep.cpp/src/qwen3-lm.h. No new ggml op.
//
// This is the model core (CPU, single KV set, F32 attention). The BPE tokenizer,
// metadata FSM and top-k/p sampling live above it in the pipeline.

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct LMConfig {
    int   vocab_size   = 0;
    int   hidden_size  = 0;
    int   n_heads      = 0;
    int   n_kv_heads   = 0;
    int   head_dim     = 0;
    int   n_layers     = 0;
    float rope_theta   = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    int   max_seq_len  = 4096;  // KV cache capacity
};

struct LMModel;  // opaque

// Load ace-lm GGUF onto `backend` (borrowed). Config is derived from tensor
// shapes (H, V, layer count, head counts). `n_kv_sets` independent KV caches are
// allocated (>=2 enables classifier-free guidance: cond=0, uncond=1). Returns
// nullptr on failure.
LMModel *        lm_model_load(const std::string & path, ggml_backend_t backend, int max_seq_len, bool verbose,
                               int n_kv_sets = 1);
void             lm_model_free(LMModel * m);
const LMConfig & lm_model_config(const LMModel * m);
size_t           lm_model_weight_bytes(const LMModel * m);
int              lm_num_kv_sets(const LMModel * m);

// Reset the KV cache for one set (start a new sequence).
void lm_reset(LMModel * m, int set = 0);
int  lm_kv_pos(const LMModel * m, int set = 0);

// Run one forward over `n_tokens` (prefill: n_tokens>1, decode: 1) at the
// current KV position of `set`, appending to that cache. Writes the last token's
// logits [vocab_size] to `logits_out`. Returns false on failure.
bool lm_model_forward(LMModel * m, const int32_t * token_ids, int n_tokens, std::vector<float> & logits_out,
                      int set = 0);

} // namespace tts_cpp::acestep
