#pragma once

// Stage-builder entry points for the Chatterbox multilingual T3 forward
// pass.  These exist so src/test_t3_mtl_stages.cpp can exercise individual
// sub-stages with Python-injected inputs (bottom-up parity, matching the
// staged-validation pattern used in scripts/dump-s3gen-reference.py).

#include "chatterbox_t3_internal.h"
#include "ggml.h"
#include "ggml-backend.h"

namespace tts_cpp::chatterbox::detail {

// Phase 15: drop a (buffer_stack, ctx_stack) pair from the process-wide
// atexit registry. Called from main()'s free_t3() lambda on error-path
// early-returns so we don't double-free at process exit.
void t3_stack_unregister(ggml_backend_buffer_t buf, ggml_context * ctx);

// Each builder returns a ggml_cgraph*; the caller uses ggml_gallocr_reserve +
// alloc_graph and sets input tensors by name before compute.

// Stage 1: cond_emb = spkr_enc(spk) + perceiver(cond_prompt + speech_pos)
//   Inputs : "exaggeration"         F32  (1,1)
//            "cond_prompt_pos_ids"  I32  (cond_prompt_len,)
//   Output : "cond_emb"              F32  (n_embd, 34)
ggml_cgraph * build_stage_cond_emb_graph(const chatterbox_model & m);

// Stage 2: text_emb + learned text_pos_emb (single batch).
//   Inputs : "text_tokens"   I32 (T,)
//            "text_pos_ids"  I32 (T,)
//   Output : "text_emb_with_pos"  F32 (n_embd, T)
ggml_cgraph * build_stage_text_emb_graph(const chatterbox_model & m, int T_text);

// Stage 3: full input assembly for cond OR uncond pass.
//   Inputs : "text_tokens", "text_pos_ids", "speech_bos", "speech_pos0",
//            "cond_prompt_pos_ids", "exaggeration"
//   Output : "inputs_embeds"  F32 (n_embd, len_cond + T_text + 2)
ggml_cgraph * build_stage_inputs_graph(const chatterbox_model & m, int T_text,
                                       bool is_uncond);

// Stage 4: run `n_layers` Llama blocks starting from a caller-provided
// inputs_embeds tensor.  Skips the input assembly; this is ideal for
// injecting Python's `inputs_embeds_initial.npy` and isolating the
// transformer math.
//   Inputs : "inputs_embeds"  F32 (n_embd, N)
//            "pos_ids"        I32 (N,)
//            "kq_mask"        F16 (N, N) causal
//   Output : "layers_out"     F32 (n_embd, N)
// Writes KV cache positions [0, N) in memory_k / memory_v (cond) or
// memory_k_uncond / memory_v_uncond (uncond) depending on is_uncond.
ggml_cgraph * build_stage_layers_graph(const chatterbox_model & m, int N,
                                       int n_layers, bool is_uncond);

// Stage 5: final RMSNorm + speech_head over the layers_out.
//   Inputs : "inputs_embeds"  F32 (n_embd, N)  [post-layer hidden state]
//   Output : "logits"         F32 (n_speech_vocab, 1)  [last position]
ggml_cgraph * build_stage_head_graph(const chatterbox_model & m, int N);

} // namespace tts_cpp::chatterbox::detail
