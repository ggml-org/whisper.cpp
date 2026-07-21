#pragma once
// Number normalization for TTS prompts: expands ASCII digit runs to English
// words, or (multilingual variant) to the surrounding script's native digits.

#include <string>

namespace tts_cpp {
namespace parler {
namespace detail {

// Every byte outside a recognized number token passes through unchanged
// (UTF-8 safe); output contains no ASCII digits, so the pass is idempotent.
std::string normalize_numbers_en(const std::string & text);

// Multilingual variant for models with an Indic-script prompt tokenizer:
// ASCII digit runs whose nearest letter context is an Indic script become
// that script's native digits; Latin-context runs fall through to
// normalize_numbers_en. Native numerals always pass through untouched.
std::string normalize_numbers_indic(const std::string & text);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
