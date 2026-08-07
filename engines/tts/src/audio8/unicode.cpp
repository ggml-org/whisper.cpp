#include "audio8/unicode.h"

#include <algorithm>
#include <array>

namespace tts_cpp {
namespace audio8 {
namespace {

#include "audio8/unicode_tables.inc"

constexpr uint32_t kReplacement = 0xFFFD;
constexpr uint32_t kSurrogateFirst = 0xD800;
constexpr uint32_t kSurrogateLast = 0xDFFF;
constexpr uint32_t kMaxCodepoint = 0x10FFFF;

constexpr uint32_t kHangulBase = 0xAC00;
constexpr uint32_t kHangulLeadBase = 0x1100;
constexpr uint32_t kHangulVowelBase = 0x1161;
constexpr uint32_t kHangulTrailBase = 0x11A7;
constexpr uint32_t kHangulLeadCount = 19;
constexpr uint32_t kHangulVowelCount = 21;
constexpr uint32_t kHangulTrailCount = 28;
constexpr uint32_t kHangulVowelSpan = kHangulVowelCount * kHangulTrailCount;
constexpr uint32_t kHangulCount = kHangulLeadCount * kHangulVowelSpan;

template <typename Table>
bool in_ranges(const Table & table, uint32_t codepoint) {
    const auto * end = std::end(table);
    const auto * hit = std::upper_bound(
        std::begin(table), end, codepoint,
        [](uint32_t value, const audio8_range & range) { return value < range.first; });
    return hit != std::begin(table) && codepoint <= (hit - 1)->last;
}

uint8_t combining_class(uint32_t codepoint) {
    const auto * end = std::end(k_audio8_combining);
    const auto * hit = std::lower_bound(
        std::begin(k_audio8_combining), end, codepoint,
        [](const audio8_combining & entry, uint32_t value) { return entry.cp < value; });
    return (hit != end && hit->cp == codepoint) ? hit->ccc : 0;
}

const audio8_decomposition * find_decomposition(uint32_t codepoint) {
    const auto * end = std::end(k_audio8_decompositions);
    const auto * hit = std::lower_bound(
        std::begin(k_audio8_decompositions), end, codepoint,
        [](const audio8_decomposition & entry, uint32_t value) { return entry.cp < value; });
    return (hit != end && hit->cp == codepoint) ? hit : nullptr;
}

uint32_t compose_pair(uint32_t first, uint32_t second) {
    const auto * end = std::end(k_audio8_compositions);
    const auto * hit = std::lower_bound(
        std::begin(k_audio8_compositions), end, first,
        [](const audio8_composition & entry, uint32_t value) { return entry.first < value; });
    for (; hit != end && hit->first == first; ++hit) {
        if (hit->second == second) {
            return hit->composed;
        }
    }
    return 0;
}

bool is_hangul_syllable(uint32_t codepoint) {
    return codepoint >= kHangulBase && codepoint < kHangulBase + kHangulCount;
}

void decompose_hangul(uint32_t codepoint, codepoints & out) {
    const uint32_t index = codepoint - kHangulBase;
    out.push_back(kHangulLeadBase + index / kHangulVowelSpan);
    out.push_back(kHangulVowelBase + (index % kHangulVowelSpan) / kHangulTrailCount);
    const uint32_t trail = index % kHangulTrailCount;
    if (trail != 0) {
        out.push_back(kHangulTrailBase + trail);
    }
}

uint32_t compose_hangul(uint32_t first, uint32_t second) {
    const uint32_t lead = first - kHangulLeadBase;
    const uint32_t vowel = second - kHangulVowelBase;
    if (lead < kHangulLeadCount && vowel < kHangulVowelCount) {
        return kHangulBase + (lead * kHangulVowelCount + vowel) * kHangulTrailCount;
    }
    const uint32_t syllable = first - kHangulBase;
    const uint32_t trail = second - kHangulTrailBase;
    if (is_hangul_syllable(first) && syllable % kHangulTrailCount == 0 &&
        trail > 0 && trail < kHangulTrailCount) {
        return first + trail;
    }
    return 0;
}

void decompose_into(uint32_t codepoint, codepoints & out) {
    if (is_hangul_syllable(codepoint)) {
        decompose_hangul(codepoint, out);
        return;
    }
    const audio8_decomposition * entry = find_decomposition(codepoint);
    if (entry == nullptr) {
        out.push_back(codepoint);
        return;
    }
    const uint32_t * values = k_audio8_decomposition_data + entry->offset;
    out.insert(out.end(), values, values + entry->length);
}

codepoints decompose(const codepoints & text) {
    codepoints out;
    out.reserve(text.size());
    for (uint32_t codepoint : text) {
        decompose_into(codepoint, out);
    }
    return out;
}

// Combining marks are sorted by class within each run, leaving starters where
// they are. Unicode Annex 15 defines this as a stable bubble sort over
// adjacent pairs, and it has to stay stable: marks of equal class are ordered
// by appearance.
void canonical_order(codepoints & text) {
    for (size_t index = 1; index < text.size(); ++index) {
        const uint8_t current = combining_class(text[index]);
        if (current == 0) {
            continue;
        }
        size_t position = index;
        while (position > 0) {
            const uint8_t previous = combining_class(text[position - 1]);
            if (previous == 0 || previous <= current) {
                break;
            }
            std::swap(text[position], text[position - 1]);
            --position;
        }
    }
}

uint32_t compose(uint32_t first, uint32_t second) {
    const uint32_t hangul = compose_hangul(first, second);
    return hangul != 0 ? hangul : compose_pair(first, second);
}

// A mark can only combine with the last starter if nothing between them
// blocks it -- that is, if every intervening mark has a strictly lower
// combining class.
bool is_blocked(uint8_t last_class, uint8_t current_class, bool adjacent) {
    return !adjacent && last_class >= current_class;
}

codepoints recompose(const codepoints & text) {
    codepoints out;
    out.reserve(text.size());
    size_t starter = std::string::npos;
    uint8_t last_class = 0;
    for (uint32_t codepoint : text) {
        const uint8_t current_class = combining_class(codepoint);
        const bool adjacent = starter != std::string::npos && starter + 1 == out.size();
        if (starter != std::string::npos &&
            !is_blocked(last_class, current_class, adjacent)) {
            const uint32_t composed = compose(out[starter], codepoint);
            if (composed != 0) {
                out[starter] = composed;
                continue;
            }
        }
        if (current_class == 0) {
            starter = out.size();
        }
        last_class = current_class;
        out.push_back(codepoint);
    }
    return out;
}

}  // namespace

void append_utf8(uint32_t codepoint, std::string & out) {
    if (codepoint < 0x80) {
        out += static_cast<char>(codepoint);
    } else if (codepoint < 0x800) {
        out += static_cast<char>(0xC0 | (codepoint >> 6));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
        out += static_cast<char>(0xE0 | (codepoint >> 12));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (codepoint >> 18));
        out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
}

std::string to_utf8(const codepoints & text) {
    std::string out;
    out.reserve(text.size());
    for (uint32_t codepoint : text) {
        append_utf8(codepoint, out);
    }
    return out;
}

namespace {

int sequence_length(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 0;
}

uint32_t lead_bits(unsigned char lead, int length) {
    static const std::array<uint32_t, 5> masks = {0, 0x7F, 0x1F, 0x0F, 0x07};
    return lead & masks[length];
}

bool is_continuation(unsigned char byte) {
    return (byte & 0xC0) == 0x80;
}

bool is_valid(uint32_t codepoint, int length) {
    static const std::array<uint32_t, 5> minimum = {0, 0, 0x80, 0x800, 0x10000};
    return codepoint >= minimum[length] && codepoint <= kMaxCodepoint &&
           (codepoint < kSurrogateFirst || codepoint > kSurrogateLast);
}

}  // namespace

// Malformed input becomes U+FFFD one byte at a time rather than throwing: the
// text comes from a caller, not from a model file, and a mangled byte should
// not take down a synthesis.
codepoints to_codepoints(const std::string & text) {
    codepoints out;
    out.reserve(text.size());
    size_t index = 0;
    while (index < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[index]);
        const int length = sequence_length(lead);
        if (length == 0 || index + length > text.size()) {
            out.push_back(kReplacement);
            ++index;
            continue;
        }
        uint32_t codepoint = lead_bits(lead, length);
        bool valid = true;
        for (int offset = 1; offset < length; ++offset) {
            const unsigned char byte = static_cast<unsigned char>(text[index + offset]);
            if (!is_continuation(byte)) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (byte & 0x3F);
        }
        if (!valid || !is_valid(codepoint, length)) {
            out.push_back(kReplacement);
            ++index;
            continue;
        }
        out.push_back(codepoint);
        index += length;
    }
    return out;
}

bool is_letter(uint32_t codepoint) {
    return in_ranges(k_audio8_letters, codepoint);
}

bool is_number(uint32_t codepoint) {
    return in_ranges(k_audio8_numbers, codepoint);
}

bool is_whitespace(uint32_t codepoint) {
    return in_ranges(k_audio8_whitespace, codepoint);
}

bool is_newline(uint32_t codepoint) {
    return codepoint == '\r' || codepoint == '\n';
}

codepoints normalize_nfc(const codepoints & text) {
    codepoints decomposed = decompose(text);
    canonical_order(decomposed);
    return recompose(decomposed);
}

}  // namespace audio8
}  // namespace tts_cpp
