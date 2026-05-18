#include "supertonic_internal.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "mtl_unicode_tables.inc"

namespace tts_cpp::supertonic::detail {
namespace {

bool utf8_decode(const char * s, size_t len, size_t & pos, uint32_t & cp) {
    if (pos >= len) return false;
    uint8_t b0 = (uint8_t) s[pos];
    if (b0 < 0x80) {
        cp = b0;
        pos += 1;
        return true;
    }
    int extra = 0;
    if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; extra = 1; }
    else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; extra = 2; }
    else if ((b0 & 0xF8) == 0xF0) { cp = b0 & 0x07; extra = 3; }
    else { cp = 0xFFFD; pos += 1; return true; }
    if (pos + 1 + extra > len) { cp = 0xFFFD; pos += 1; return true; }
    for (int i = 0; i < extra; ++i) {
        uint8_t b = (uint8_t) s[pos + 1 + i];
        if ((b & 0xC0) != 0x80) { cp = 0xFFFD; pos += 1; return true; }
        cp = (cp << 6) | (b & 0x3F);
    }
    pos += 1 + extra;
    return true;
}

void utf8_append(uint32_t cp, std::string & out) {
    if (cp < 0x80) {
        out.push_back((char) cp);
    } else if (cp < 0x800) {
        out.push_back((char) (0xC0 | (cp >> 6)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char) (0xE0 | (cp >> 12)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char) (0xF0 | (cp >> 18)));
        out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    }
}

std::vector<uint32_t> utf8_to_cps(const std::string & s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    size_t pos = 0;
    uint32_t cp = 0;
    while (utf8_decode(s.data(), s.size(), pos, cp)) out.push_back(cp);
    return out;
}

std::string cps_to_utf8(const std::vector<uint32_t> & cps) {
    std::string out;
    out.reserve(cps.size());
    for (uint32_t cp : cps) utf8_append(cp, out);
    return out;
}

const mtl_unicode_entry * find_entry(const mtl_unicode_entry * table, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        uint32_t c = table[mid].cp;
        if (c == cp) return &table[mid];
        if (c < cp) lo = mid + 1;
        else hi = mid;
    }
    return nullptr;
}

void append_nfkd_cp(uint32_t cp, std::vector<uint32_t> & out) {
    if (cp >= 0xAC00 && cp <= 0xD7A3) {
        uint32_t base = cp - 0xAC00;
        uint32_t l = 0x1100 + base / (21 * 28);
        uint32_t v = 0x1161 + (base % (21 * 28)) / 28;
        uint32_t t = base % 28;
        out.push_back(l);
        out.push_back(v);
        if (t != 0) out.push_back(0x11A7 + t);
        return;
    }
    const auto * e = find_entry(k_mtl_nfkd_table, k_mtl_nfkd_table_len, cp);
    if (e) {
        for (uint32_t i = 0; i < e->length; ++i) out.push_back(k_mtl_nfkd_data[e->offset + i]);
    } else {
        out.push_back(cp);
    }
}

bool is_emoji_or_symbol(uint32_t cp) {
    return (cp >= 0x1F600 && cp <= 0x1F64F) ||
           (cp >= 0x1F300 && cp <= 0x1F5FF) ||
           (cp >= 0x1F680 && cp <= 0x1F6FF) ||
           (cp >= 0x1F700 && cp <= 0x1F77F) ||
           (cp >= 0x1F780 && cp <= 0x1F7FF) ||
           (cp >= 0x1F800 && cp <= 0x1F8FF) ||
           (cp >= 0x1F900 && cp <= 0x1F9FF) ||
           (cp >= 0x1FA00 && cp <= 0x1FA6F) ||
           (cp >= 0x1FA70 && cp <= 0x1FAFF) ||
           (cp >= 0x2600 && cp <= 0x26FF) ||
           (cp >= 0x2700 && cp <= 0x27BF) ||
           (cp >= 0x1F1E6 && cp <= 0x1F1FF);
}

bool is_space_cp(uint32_t cp) {
    return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
           cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string collapse_spaces(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;
    for (unsigned char ch : s) {
        bool sp = std::isspace(ch) != 0;
        if (sp) {
            if (!prev_space) out.push_back(' ');
            prev_space = true;
        } else {
            out.push_back((char) ch);
            prev_space = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

bool has_terminal_punct(const std::string & s) {
    if (s.empty()) return false;
    std::vector<uint32_t> cps = utf8_to_cps(s);
    if (cps.empty()) return false;
    uint32_t cp = cps.back();
    switch (cp) {
        case '.': case '!': case '?': case ';': case ':': case ',': case '\'':
        case '"': case ')': case ']': case '}': case 0x2026: case 0x3002:
        case 0x300D: case 0x300F: case 0x3011: case 0x3009: case 0x300B:
        case 0x203A: case 0x00BB:
            return true;
        default:
            return false;
    }
}

bool is_supported_language(const std::string & language) {
    return language == "en" || language == "ko" || language == "es" ||
           language == "pt" || language == "fr";
}

} // namespace

std::string supertonic_preprocess_text(const std::string & text,
                                       const std::string & language,
                                       const std::string & language_wrap_mode) {
    if (!is_supported_language(language)) {
        throw std::runtime_error("invalid Supertonic language: " + language);
    }

    std::vector<uint32_t> nfkd;
    for (uint32_t cp : utf8_to_cps(text)) append_nfkd_cp(cp, nfkd);

    std::vector<uint32_t> filtered;
    filtered.reserve(nfkd.size());
    for (uint32_t cp : nfkd) {
        if (is_emoji_or_symbol(cp)) continue;
        if (cp == 0x2013 || cp == 0x2011 || cp == 0x2014) cp = '-';
        else if (cp == '_' || cp == '[' || cp == ']' || cp == '|' || cp == '/' ||
                 cp == '#' || cp == 0x2192 || cp == 0x2190) cp = ' ';
        else if (cp == 0x201C || cp == 0x201D) cp = '"';
        else if (cp == 0x2018 || cp == 0x2019 || cp == 0x00B4 || cp == '`') cp = '\'';
        if (cp == 0x2665 || cp == 0x2606 || cp == 0x2661 || cp == 0x00A9 || cp == '\\') continue;
        if (is_space_cp(cp)) cp = ' ';
        filtered.push_back(cp);
    }

    std::string s = cps_to_utf8(filtered);
    replace_all(s, "@", " at ");
    replace_all(s, "e.g.,", "for example, ");
    replace_all(s, "i.e.,", "that is, ");

    replace_all(s, " ,", ",");
    replace_all(s, " .", ".");
    replace_all(s, " !", "!");
    replace_all(s, " ?", "?");
    replace_all(s, " ;", ";");
    replace_all(s, " :", ":");
    replace_all(s, " '", "'");

    while (s.find("\"\"") != std::string::npos) replace_all(s, "\"\"", "\"");
    while (s.find("''") != std::string::npos) replace_all(s, "''", "'");
    while (s.find("``") != std::string::npos) replace_all(s, "``", "`");

    s = collapse_spaces(s);
    if (!has_terminal_punct(s)) s += ".";
    if (language_wrap_mode == "none") return s;
    if (language_wrap_mode == "prefix") return "<" + language + ">" + s + " ";
    if (language_wrap_mode == "open_close") return "<" + language + ">" + s + "</" + language + ">";
    throw std::runtime_error("invalid Supertonic language_wrap_mode: " + language_wrap_mode);
}

bool supertonic_text_to_ids(const supertonic_model & model,
                            const std::string & text,
                            const std::string & language,
                            std::vector<int32_t> & ids,
                            std::string * normalized_text,
                            std::string * error) {
    try {
        std::string normalized = supertonic_preprocess_text(text, language, model.hparams.language_wrap_mode);
        std::vector<uint32_t> cps = utf8_to_cps(normalized);
        ids.clear();
        ids.reserve(cps.size());
        for (uint32_t cp : cps) {
            if (cp >= model.unicode_indexer.size()) {
                throw std::runtime_error("unsupported character outside Unicode indexer");
            }
            int32_t id = model.unicode_indexer[cp];
            if (id < 0) {
                throw std::runtime_error("unsupported character U+" + std::to_string(cp));
            }
            ids.push_back(id);
        }
        if (normalized_text) *normalized_text = normalized;
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
