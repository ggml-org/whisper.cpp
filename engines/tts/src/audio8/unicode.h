// UTF-8 and Unicode support for the Audio8 tokenizer: codepoint transcoding,
// the \p{L} / \p{N} classes the Qwen2 pre-tokenizer splits on, and NFC.
//
// Only what the tokenizer needs is here. The tables live in the generated
// unicode_tables.inc; see scripts/gen-audio8-unicode-tables.py for what they
// cover and why they are needed at all.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp {
namespace audio8 {

using codepoints = std::vector<uint32_t>;

void append_utf8(uint32_t codepoint, std::string & out);
std::string to_utf8(const codepoints & text);
codepoints to_codepoints(const std::string & text);

bool is_letter(uint32_t codepoint);
bool is_number(uint32_t codepoint);
bool is_whitespace(uint32_t codepoint);
bool is_newline(uint32_t codepoint);

codepoints normalize_nfc(const codepoints & text);

}  // namespace audio8
}  // namespace tts_cpp
