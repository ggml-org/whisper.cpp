#pragma once

// CosyVoice3 conditioning resolution, split out from the engine so it carries
// no ggml dependency and can be pinned by a model-free test.

#include "tts-cpp/cosyvoice/engine.h"

#include <string>

namespace tts_cpp::cosyvoice::detail {

// Resolves `controls` to the bare CosyVoice3 instruction that sits between the
// prompt wrapper and <|endofprompt|>.  Returns "" when no channel is engaged,
// which selects the unchanged zero-shot path.
//
// Throws std::invalid_argument when a value is outside the engine's supported
// set (checked first, on every channel), or when two channels are engaged.
std::string resolve_instruct(const VoiceControls & controls);

// Same checks, discarding the instruction: for the engine constructor, so a bad
// default fails at load rather than at the first synthesis.
void validate_controls(const VoiceControls & controls);

// The two LM prompt templates, deliberately NOT unified: zero-shot has no space
// after "assistant." and puts the transcript after <|endofprompt|>; instruct
// mode has the space and puts the instruction before it.  Both are what the
// model was trained on, so aligning them would move the prompt off-distribution.
std::string build_lm_prompt_instruct(const std::string & instruct);
std::string build_lm_prompt_zero_shot(const std::string & transcript);

} // namespace tts_cpp::cosyvoice::detail
