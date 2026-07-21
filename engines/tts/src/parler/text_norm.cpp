#include "text_norm.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

const char * kOnes[20] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"};
const char * kTens[10] = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety"};
const char * kScales[6] = {"", "thousand", "million", "billion", "trillion",
    "quadrillion"};

bool is_digit(char c)       { return c >= '0' && c <= '9'; }
bool is_ascii_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
char lower(char c)          { return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c; }

std::string three_digit_words(int v) { // 1..999
    std::string out;
    if (v >= 100) {
        out += kOnes[v / 100];
        out += " hundred";
        v %= 100;
        if (v) out += ' ';
    }
    if (v >= 20) {
        out += kTens[v / 10];
        if (v % 10) { out += '-'; out += kOnes[v % 10]; }
    } else if (v > 0) {
        out += kOnes[v];
    }
    return out;
}

std::string cardinal_words(unsigned long long v) {
    if (v == 0) return kOnes[0];
    int groups[6] = {0};
    int n = 0;
    while (v > 0) { groups[n++] = (int)(v % 1000); v /= 1000; }
    std::string out;
    for (int g = n - 1; g >= 0; --g) {
        if (!groups[g]) continue;
        if (!out.empty()) out += ' ';
        out += three_digit_words(groups[g]);
        if (g) { out += ' '; out += kScales[g]; }
    }
    return out;
}

std::string digit_by_digit(const std::string & digits) {
    std::string out;
    for (char c : digits) {
        if (!out.empty()) out += ' ';
        out += kOnes[c - '0'];
    }
    return out;
}

// "twenty-one" -> "twenty-first": ordinalize the final word.
std::string ordinal_words(std::string words) {
    const size_t cut = words.find_last_of(" -");
    const size_t at  = cut == std::string::npos ? 0 : cut + 1;
    const std::string last = words.substr(at);
    static const struct { const char * card; const char * ord; } kSpecial[] = {
        {"one", "first"}, {"two", "second"}, {"three", "third"},
        {"five", "fifth"}, {"eight", "eighth"}, {"nine", "ninth"},
        {"twelve", "twelfth"},
    };
    for (const auto & s : kSpecial) {
        if (last == s.card) {
            words.replace(at, last.size(), s.ord);
            return words;
        }
    }
    if (!last.empty() && last.back() == 'y') {
        words.replace(words.size() - 1, 1, "ieth"); // twenty -> twentieth
    } else {
        words += "th"; // four / six / hundred / thousand -> fourth / ...
    }
    return words;
}

struct cp_span {
    uint32_t v;
    size_t   off;
    size_t   len;
};

std::vector<cp_span> decode_utf8(const std::string & s) {
    std::vector<cp_span> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t b = (uint8_t) s[i];
        size_t len = b < 0x80 ? 1 : (b & 0xe0) == 0xc0 ? 2
                   : (b & 0xf0) == 0xe0 ? 3 : (b & 0xf8) == 0xf0 ? 4 : 1;
        if (i + len > s.size()) len = 1;
        uint32_t v = b;
        if (len > 1) {
            v = b & (0x7f >> len);
            for (size_t k = 1; k < len; ++k) {
                const uint8_t cb = (uint8_t) s[i + k];
                if ((cb & 0xc0) != 0x80) { len = 1; v = b; break; }  // malformed: lone byte
                v = (v << 6) | (cb & 0x3f);
            }
        }
        // malformed bytes carry no script/letter meaning (bytes still pass
        // through verbatim via their offsets)
        if (len == 1 && v >= 0x80) v = 0xFFFD;
        out.push_back({v, i, len});
        i += len;
    }
    return out;
}

void append_utf8(std::string & out, uint32_t c) {
    if (c < 0x80) {
        out += (char) c;
    } else if (c < 0x800) {
        out += (char) (0xc0 | (c >> 6));
        out += (char) (0x80 | (c & 0x3f));
    } else if (c < 0x10000) {
        out += (char) (0xe0 | (c >> 12));
        out += (char) (0x80 | ((c >> 6) & 0x3f));
        out += (char) (0x80 | (c & 0x3f));
    } else {
        out += (char) (0xf0 | (c >> 18));
        out += (char) (0x80 | ((c >> 12) & 0x3f));
        out += (char) (0x80 | ((c >> 6) & 0x3f));
        out += (char) (0x80 | (c & 0x3f));
    }
}

// Unicode block of a scripted codepoint -> the script's zero digit (0 = none).
// Digits sit inside their script's block, so native-digit neighbours also
// resolve context. Arabic script maps to the extended Arabic-Indic digits
// (the Urdu/Sindhi convention).
uint32_t script_digit_base(uint32_t c) {
    // danda/double danda are encoded in the Devanagari block but are shared
    // sentence punctuation across Indic scripts — they identify no script
    if (c == 0x0964 || c == 0x0965) return 0;
    static const struct { uint32_t lo, hi, zero; } kBlocks[] = {
        {0x0600, 0x06FF, 0x06F0},  // Arabic (Urdu, Sindhi, Kashmiri)
        {0x0750, 0x077F, 0x06F0},  // Arabic Supplement
        {0x0900, 0x097F, 0x0966},  // Devanagari
        {0x0980, 0x09FF, 0x09E6},  // Bengali / Assamese
        {0x0A00, 0x0A7F, 0x0A66},  // Gurmukhi
        {0x0A80, 0x0AFF, 0x0AE6},  // Gujarati
        {0x0B00, 0x0B7F, 0x0B66},  // Odia
        {0x0B80, 0x0BFF, 0x0BE6},  // Tamil
        {0x0C00, 0x0C7F, 0x0C66},  // Telugu
        {0x0C80, 0x0CFF, 0x0CE6},  // Kannada
        {0x0D00, 0x0D7F, 0x0D66},  // Malayalam
        {0x1C50, 0x1C7F, 0x1C50},  // Ol Chiki (Santali)
        {0xABC0, 0xABFF, 0xABF0},  // Meetei Mayek (Manipuri)
    };
    for (const auto & blk : kBlocks) {
        if (c >= blk.lo && c <= blk.hi) return blk.zero;
    }
    return 0;
}

bool is_latin_letter(uint32_t c) {
    if (c == 0x00D7 || c == 0x00F7) return false;  // multiplication/division signs
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= 0x00C0 && c <= 0x024F);  // Latin-1 Supplement .. Latin Extended-B
}

// nearest letter context of cps[run_begin, run_end): left first, then right;
// an Indic script wins with its digit base, a Latin letter wins with 0
uint32_t run_context_base(const std::vector<cp_span> & cps,
                          size_t run_begin, size_t run_end) {
    for (size_t k = run_begin; k-- > 0;) {
        const uint32_t base = script_digit_base(cps[k].v);
        if (base) return base;
        if (is_latin_letter(cps[k].v)) return 0;
    }
    for (size_t k = run_end; k < cps.size(); ++k) {
        const uint32_t base = script_digit_base(cps[k].v);
        if (base) return base;
        if (is_latin_letter(cps[k].v)) return 0;
    }
    return 0;
}

} // namespace

std::string normalize_numbers_en(const std::string & text) {
    const size_t n = text.size();
    std::string out;
    out.reserve(n);

    size_t i = 0;
    while (i < n) {
        if (!is_digit(text[i])) {
            out += text[i++];
            continue;
        }

        size_t j = i;
        while (j < n && is_digit(text[j])) ++j;
        std::string ds = text.substr(i, j - i);

        // ",ddd" thousands separators: only after a 1-3 digit lead group,
        // and only for full groups not followed by a fourth digit
        if (ds.size() <= 3) {
            while (j + 4 <= n && text[j] == ',' &&
                   is_digit(text[j + 1]) && is_digit(text[j + 2]) &&
                   is_digit(text[j + 3]) &&
                   (j + 4 == n || !is_digit(text[j + 4]))) {
                ds.append(text, j + 1, 3);
                j += 4;
            }
        }

        std::string frac;
        if (j + 1 < n && text[j] == '.' && is_digit(text[j + 1])) {
            size_t k = j + 1;
            while (k < n && is_digit(text[k])) ++k;
            frac.assign(text, j + 1, k - j - 1);
            j = k;
        }

        bool ordinal = false, plural = false;
        if (frac.empty() && j + 1 < n) {
            const char a = lower(text[j]), b = lower(text[j + 1]);
            const bool sfx = (a == 's' && b == 't') || (a == 'n' && b == 'd') ||
                             (a == 'r' && b == 'd') || (a == 't' && b == 'h');
            if (sfx) {
                size_t k = j + 2;
                const bool pl = k < n && lower(text[k]) == 's';
                if (pl) ++k;
                if (k == n || (!is_ascii_alpha(text[k]) && !is_digit(text[k]))) {
                    ordinal = true;
                    plural  = pl;
                    j       = k;
                }
            }
        }

        // leading zeros and >15-digit runs read digit-by-digit
        const bool dbd = ds.size() > 15 || (ds.size() > 1 && ds[0] == '0');
        std::string words = dbd ? digit_by_digit(ds)
                                : cardinal_words(std::stoull(ds));
        if (!frac.empty()) {
            words += " point ";
            words += digit_by_digit(frac);
        } else if (ordinal) {
            words = ordinal_words(std::move(words));
            if (plural) words += 's';
        }

        // keep words separated from abutting letters (and any UTF-8 bytes)
        if (!out.empty() &&
            (is_ascii_alpha(out.back()) || (unsigned char) out.back() >= 0x80)) {
            out += ' ';
        }
        out += words;
        if (j < n &&
            (is_ascii_alpha(text[j]) || (unsigned char) text[j] >= 0x80)) {
            out += ' ';
        }
        i = j;
    }
    return out;
}

std::string normalize_numbers_indic(const std::string & text) {
    const std::vector<cp_span> cps = decode_utf8(text);
    std::string pass1;
    pass1.reserve(text.size());

    size_t i = 0;
    while (i < cps.size()) {
        const bool digit = cps[i].len == 1 && cps[i].v >= '0' && cps[i].v <= '9';
        if (!digit) {
            pass1.append(text, cps[i].off, cps[i].len);
            ++i;
            continue;
        }
        size_t j = i;
        while (j < cps.size() && cps[j].len == 1 && cps[j].v >= '0' && cps[j].v <= '9') ++j;
        const uint32_t base = run_context_base(cps, i, j);
        if (base) {
            for (size_t k = i; k < j; ++k) {
                append_utf8(pass1, base + (cps[k].v - '0'));
            }
        } else {
            pass1.append(text, cps[i].off,
                         cps[j - 1].off + cps[j - 1].len - cps[i].off);
        }
        i = j;
    }
    // Latin-context (and context-free) runs still read as English words
    return normalize_numbers_en(pass1);
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
