#pragma once

// ACE-Step LM pipeline (QVAC-21921): prompt/CoT building, top-k/p sampling and
// the Phase-2 audio-code decode loop layered on top of the LM core (lm_ggml) and
// the BPE tokenizer (bpe_tokenizer). Ported from acestep.cpp (prompt.h,
// sampling.h, pipeline-lm.cpp).
//
// This first cut targets the turbo text2music path with a fully-specified prompt
// (caption + lyrics + all metadata), so Phase 1 (CoT/lyric generation + metadata
// FSM) can be skipped and only Phase 2 (audio semantic codes) runs. CFG is not
// used here (single KV set); it can be added once the core grows multi-set KV.

#include "bpe_tokenizer.h"
#include "lm_ggml.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

// User-facing music prompt (mirrors acestep.cpp AcePrompt).
struct AcePrompt {
    std::string caption;
    std::string lyrics;
    float       duration      = 0.0f;  // seconds
    int         bpm           = 0;
    std::string keyscale;
    std::string timesignature;
    std::string vocal_language;
};

struct LmSampleParams {
    float    temperature    = 0.85f;
    float    top_p          = 0.9f;
    int      top_k          = 0;      // 0 = disabled (top_p only), matches upstream
    float    cfg_scale      = 2.0f;   // >1 enables CFG (needs a 2nd KV set)
    uint32_t seed           = 0;
    int      max_new_tokens = 0;      // 0 => derive from duration (dur*5 + 100)
};

// Temperature -> top_k -> top_p -> softmax -> multinomial. Mutates `logits`.
// Ported verbatim from acestep.cpp/src/sampling.h.
int sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng);

// Build the Qwen3 chat prompt with an injected CoT metadata block (Phase 2).
// The assistant turn stays open so the LM emits audio codes then <|im_end|>.
std::vector<int> build_lm_prompt_with_cot(const BpeTokenizer & bpe, const AcePrompt & prompt);

// Phase 1: auto-generate missing metadata (bpm/keyscale/duration/timesignature/
// language) — and lyrics for a bare caption — via FSM-constrained decoding.
// Mutates `prompt` in place (gap-fill: only empty fields are overwritten).
// use_fsm=false lets the LM free-run (lower reliability). Returns false on failure.
bool lm_generate_phase1(LMModel * m, const BpeTokenizer & bpe, AcePrompt & prompt, const LmSampleParams & params,
                        bool use_fsm = true, bool use_cot_caption = true);

// Phase 2: generate FSQ audio semantic codes for a fully-specified prompt.
// Single sequence, no CFG. Returns the raw codes (already offset-subtracted,
// i.e. token - AUDIO_CODE_BASE). Returns false on failure.
bool lm_generate_codes(LMModel *              m,
                       const BpeTokenizer &   bpe,
                       const AcePrompt &      prompt,
                       const LmSampleParams & params,
                       std::vector<int> &     codes_out);

} // namespace tts_cpp::acestep
