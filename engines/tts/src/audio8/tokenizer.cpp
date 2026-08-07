#include "audio8/tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>

#include "audio8/unicode.h"

namespace tts_cpp {
namespace audio8 {
namespace {

constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

const char * const kSystemHeader = "<|im_start|>system\n";
const char * const kPlainInstruction = "convert the provided text to speech";
const char * const kReferenceInstruction =
    "convert the provided text to speech reference to the following:\n\nText:\n";
const char * const kReferenceTail = "\n\nSpeech:\n";
const char * const kTurnEnd = "<|im_end|>\n";
const char * const kUserHeader = "<|im_start|>user\n";
const char * const kAssistantHeader = "<|im_start|>assistant\n<|voice|>";
const char * const kDefaultSpeaker = "<|speaker:0|>";

// GPT-2's bytes_to_unicode: every byte becomes a printable codepoint so the
// BPE vocabulary can be plain text.
const std::array<std::string, 256> & byte_symbols() {
    static const std::array<std::string, 256> table = [] {
        std::array<std::string, 256> symbols;
        uint32_t spare = 0;
        for (uint32_t byte = 0; byte < 256; ++byte) {
            const bool printable = (byte >= 0x21 && byte <= 0x7E) ||
                                   (byte >= 0xA1 && byte <= 0xAC) ||
                                   (byte >= 0xAE && byte <= 0xFF);
            append_utf8(printable ? byte : 256 + spare++, symbols[byte]);
        }
        return symbols;
    }();
    return table;
}

bool is_contraction_start(const codepoints & text, size_t at) {
    return text[at] == '\'' && at + 1 < text.size();
}

uint32_t lowered(const codepoints & text, size_t at) {
    const uint32_t value = at < text.size() ? text[at] : 0;
    return (value >= 'A' && value <= 'Z') ? value + 32 : value;
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d)
size_t contraction_length(const codepoints & text, size_t at) {
    if (!is_contraction_start(text, at)) {
        return 0;
    }
    const uint32_t first = lowered(text, at + 1);
    if (first == 's' || first == 't' || first == 'm' || first == 'd') {
        return 2;
    }
    const uint32_t second = lowered(text, at + 2);
    const bool two_letter = (first == 'r' && second == 'e') ||
                            (first == 'v' && second == 'e') ||
                            (first == 'l' && second == 'l');
    return two_letter ? 3 : 0;
}

size_t run_of(const codepoints & text, size_t at, bool (*predicate)(uint32_t)) {
    size_t end = at;
    while (end < text.size() && predicate(text[end])) {
        ++end;
    }
    return end;
}

bool is_symbol(uint32_t value) {
    return !is_whitespace(value) && !is_letter(value) && !is_number(value);
}

// [^\r\n\p{L}\p{N}]?\p{L}+
size_t letter_length(const codepoints & text, size_t at) {
    size_t start = at;
    if (!is_letter(text[at])) {
        if (is_newline(text[at]) || is_number(text[at]) || at + 1 >= text.size() ||
            !is_letter(text[at + 1])) {
            return 0;
        }
        ++start;
    }
    return run_of(text, start, is_letter) - at;
}

// ' ?[^\s\p{L}\p{N}]+[\r\n]*
size_t symbol_length(const codepoints & text, size_t at) {
    size_t start = at;
    if (text[at] == ' ' && at + 1 < text.size() && is_symbol(text[at + 1])) {
        ++start;
    }
    if (start >= text.size() || !is_symbol(text[start])) {
        return 0;
    }
    const size_t symbols = run_of(text, start, is_symbol);
    return run_of(text, symbols, is_newline) - at;
}

// \s*[\r\n]+ -- a run of whitespace that ends in newlines is taken whole.
size_t newline_run_length(const codepoints & text, size_t at) {
    const size_t spaces = run_of(text, at, is_whitespace);
    size_t last_newline = at;
    for (size_t index = at; index < spaces; ++index) {
        if (is_newline(text[index])) {
            last_newline = index + 1;
        }
    }
    return last_newline > at ? last_newline - at : 0;
}

// \s+(?!\S) | \s+ -- a whitespace run keeps its last character only when
// nothing follows it, so the trailing space stays attached to the next word.
size_t whitespace_length(const codepoints & text, size_t at) {
    const size_t end = run_of(text, at, is_whitespace);
    if (end == at) {
        return 0;
    }
    const bool followed = end < text.size();
    const size_t length = end - at;
    return (followed && length > 1) ? length - 1 : length;
}

// \p{N} -- digits are taken one at a time, so "2024" is four pieces.
size_t number_length(const codepoints & text, size_t at) {
    return is_number(text[at]) ? 1 : 0;
}

// The alternation, in the order the pre-tokenizer tries it. Every codepoint
// is a letter, a number, whitespace or a symbol, so one rule always fires;
// the trailing 1 only keeps a malformed table from spinning the caller.
size_t piece_length(const codepoints & text, size_t at) {
    for (auto rule : {contraction_length, letter_length, number_length, symbol_length,
                      newline_run_length, whitespace_length}) {
        const size_t length = rule(text, at);
        if (length > 0) {
            return length;
        }
    }
    return 1;
}

std::vector<std::string> pretokenize(const std::string & span) {
    const codepoints text = normalize_nfc(to_codepoints(span));
    std::vector<std::string> pieces;
    size_t at = 0;
    while (at < text.size()) {
        const size_t length = piece_length(text, at);
        pieces.push_back(to_utf8(codepoints(text.begin() + at, text.begin() + at + length)));
        at += length;
    }
    return pieces;
}

std::string join_pair(const std::string & left, const std::string & right) {
    return left + " " + right;
}

}  // namespace

Tokenizer::Tokenizer(const TokenizerData & data) : texts_(data.tokens) {
    ids_.reserve(data.tokens.size());
    for (size_t index = 0; index < data.tokens.size(); ++index) {
        ids_.emplace(data.tokens[index], static_cast<int32_t>(index));
    }
    merge_ranks_.reserve(data.merges.size());
    for (size_t rank = 0; rank < data.merges.size(); ++rank) {
        merge_ranks_.emplace(data.merges[rank], static_cast<int32_t>(rank));
    }
    added_.reserve(data.added_token_ids.size());
    for (int32_t id : data.added_token_ids) {
        added_.emplace_back(data.tokens.at(static_cast<size_t>(id)), id);
    }
    std::sort(added_.begin(), added_.end(), [](const auto & left, const auto & right) {
        return left.first.size() > right.first.size();
    });
}

int32_t Tokenizer::token_id(const std::string & token) const {
    const auto found = ids_.find(token);
    if (found == ids_.end()) {
        throw std::runtime_error("audio8 tokenizer: unknown token '" + token + "'");
    }
    return found->second;
}

const std::string & Tokenizer::token_text(int32_t id) const {
    return texts_.at(static_cast<size_t>(id));
}

std::vector<std::string> Tokenizer::merge(const std::string & piece) const {
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (unsigned char byte : piece) {
        symbols.push_back(byte_symbols()[byte]);
    }
    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_at = kNotFound;
        for (size_t at = 0; at + 1 < symbols.size(); ++at) {
            const auto found = merge_ranks_.find(join_pair(symbols[at], symbols[at + 1]));
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_at = at;
            }
        }
        if (best_at == kNotFound) {
            break;
        }
        symbols[best_at] += symbols[best_at + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_at) + 1);
    }
    return symbols;
}

void Tokenizer::encode_span(const std::string & span, std::vector<int32_t> & out) const {
    for (const std::string & piece : pretokenize(span)) {
        for (const std::string & symbol : merge(piece)) {
            // A complete byte-level vocabulary can spell anything, so a miss
            // means the GGUF's token list is truncated or from another model.
            // Dropping it would quietly change what the model is asked to say.
            out.push_back(token_id(symbol));
        }
    }
}

bool Tokenizer::next_added_token(const std::string & text, size_t from, size_t & at,
                                 size_t & length, int32_t & id) const {
    at = kNotFound;
    for (const auto & [token, value] : added_) {
        const size_t found = text.find(token, from);
        if (found != std::string::npos && (at == kNotFound || found < at)) {
            at = found;
            length = token.size();
            id = value;
        }
    }
    return at != kNotFound;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    size_t position = 0;
    while (position < text.size()) {
        size_t at = 0;
        size_t length = 0;
        int32_t id = 0;
        const bool found = next_added_token(text, position, at, length, id);
        const size_t end = found ? at : text.size();
        if (end > position) {
            encode_span(text.substr(position, end - position), ids);
        }
        if (!found) {
            break;
        }
        ids.push_back(id);
        position = at + length;
    }
    return ids;
}

namespace {

// Python's str.split() splits on White_Space plus the four ASCII separators,
// and the processor cleans text with " ".join(text.split()).
bool splits_a_word(uint32_t value) {
    return is_whitespace(value) || (value >= 0x1C && value <= 0x1F);
}

void append_word(const codepoints & word, std::string & out) {
    if (word.empty()) {
        return;
    }
    if (!out.empty()) {
        out += ' ';
    }
    out += to_utf8(word);
}

}  // namespace

std::string collapse_whitespace(const std::string & text) {
    std::string out;
    codepoints word;
    for (uint32_t value : to_codepoints(text)) {
        if (splits_a_word(value)) {
            append_word(word, out);
            word.clear();
            continue;
        }
        word.push_back(value);
    }
    append_word(word, out);
    return out;
}

namespace {

void append_encoded(const Tokenizer & tokenizer, const std::string & segment,
                    std::vector<int32_t> & out) {
    const std::vector<int32_t> ids = tokenizer.encode(segment);
    out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int32_t> encode_segments(const Tokenizer & tokenizer,
                                     const std::vector<std::string> & segments) {
    std::vector<int32_t> ids;
    for (const std::string & segment : segments) {
        append_encoded(tokenizer, segment, ids);
    }
    return ids;
}

bool names_a_speaker(const std::string & text) {
    const std::string marker = "<|speaker:";
    const size_t at = text.find(marker);
    if (at == std::string::npos) {
        return false;
    }
    size_t index = at + marker.size();
    const size_t first_digit = index;
    while (index < text.size() && std::isdigit(static_cast<unsigned char>(text[index])) != 0) {
        ++index;
    }
    return index > first_digit && text.compare(index, 2, "|>") == 0;
}

std::string with_speaker(const std::string & reference_text) {
    const std::string cleaned = collapse_whitespace(reference_text);
    return names_a_speaker(cleaned) ? cleaned : kDefaultSpeaker + cleaned;
}

std::vector<int32_t> plain_prompt(const Tokenizer & tokenizer, const std::string & target) {
    return encode_segments(tokenizer, {
        kSystemHeader, kPlainInstruction, kTurnEnd,
        kUserHeader, target, kTurnEnd, kAssistantHeader,
    });
}

std::vector<int32_t> reference_prefix(const Tokenizer & tokenizer,
                                      const std::string & reference_text) {
    return encode_segments(tokenizer, {
        kSystemHeader, kReferenceInstruction, with_speaker(reference_text), kReferenceTail,
    });
}

std::vector<int32_t> reference_suffix(const Tokenizer & tokenizer, const std::string & target) {
    return encode_segments(tokenizer, {
        kTurnEnd, kUserHeader, target, kTurnEnd, kAssistantHeader,
    });
}

}  // namespace

PromptSegments build_prompt(const Tokenizer & tokenizer, const std::string & text,
                            const std::string & reference_text, bool has_reference) {
    const std::string target = collapse_whitespace(text);
    if (target.empty()) {
        throw std::runtime_error("audio8: text must not be empty");
    }
    if (!has_reference) {
        return {plain_prompt(tokenizer, target), {}};
    }
    if (reference_text.empty()) {
        throw std::runtime_error(
            "audio8: reference_text is required when a reference voice is provided");
    }
    return {reference_prefix(tokenizer, reference_text), reference_suffix(tokenizer, target)};
}

}  // namespace audio8
}  // namespace tts_cpp
