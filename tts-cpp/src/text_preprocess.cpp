#include "text_preprocess.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef TTS_CPP_HAS_MECAB
#include <mecab.h>
#endif

namespace tts_cpp::chatterbox::text_preprocess {

namespace {

constexpr uint32_t kHangulSyllableBase = 0xAC00;
constexpr uint32_t kHangulSyllableEnd  = 0xD7A3;
constexpr int      kJamoMedialCount    = 21;
constexpr int      kJamoFinalCount     = 28;
constexpr uint32_t kJamoInitialBase    = 0x1100;
constexpr uint32_t kJamoMedialBase     = 0x1161;
constexpr uint32_t kJamoFinalBase      = 0x11A8;

constexpr uint32_t kKatakanaStart      = 0x30A1;
constexpr uint32_t kKatakanaEnd        = 0x30F6;
constexpr uint32_t kKatakanaToHiragana = 0x60;

constexpr int kIpadicReadingFieldIndex = 7;

bool is_hangul_syllable(uint32_t cp) {
    return cp >= kHangulSyllableBase && cp <= kHangulSyllableEnd;
}

bool is_katakana(uint32_t cp) {
    return cp >= kKatakanaStart && cp <= kKatakanaEnd;
}

bool is_cjk_ideograph(uint32_t cp) {
    return (cp >= 0x4E00  && cp <= 0x9FFF)  ||
           (cp >= 0x3400  && cp <= 0x4DBF)  ||
           (cp >= 0xF900  && cp <= 0xFAFF)  ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0x2CEB0 && cp <= 0x2EBEF) ||
           (cp >= 0x30000 && cp <= 0x3134F);
}

int detect_utf8_seq_len(unsigned char b) {
    if (b < 0x80)               return 1;
    if ((b & 0xE0) == 0xC0)     return 2;
    if ((b & 0xF0) == 0xE0)     return 3;
    if ((b & 0xF8) == 0xF0)     return 4;
    return 0;
}

uint32_t extract_leading_bits(unsigned char b, int seq_len) {
    switch (seq_len) {
        case 1: return b;
        case 2: return b & 0x1Fu;
        case 3: return b & 0x0Fu;
        case 4: return b & 0x07u;
        default: return 0;
    }
}

uint32_t decode_single_codepoint(const unsigned char * bytes, size_t total, size_t & pos) {
    const int len = detect_utf8_seq_len(bytes[pos]);
    if (len == 0) { ++pos; return 0xFFFD; }

    uint32_t cp = extract_leading_bits(bytes[pos], len);
    for (int k = 1; k < len && (pos + k) < total; ++k) {
        cp = (cp << 6) | (bytes[pos + k] & 0x3Fu);
    }
    pos += len;
    return cp;
}

void append_hangul_jamo(uint32_t cp, std::string & out) {
    const int idx     = static_cast<int>(cp - kHangulSyllableBase);
    const int initial = idx / (kJamoMedialCount * kJamoFinalCount);
    const int medial  = (idx % (kJamoMedialCount * kJamoFinalCount)) / kJamoFinalCount;
    const int final_  = idx %  kJamoFinalCount;

    out += encode_codepoint(kJamoInitialBase + initial);
    out += encode_codepoint(kJamoMedialBase  + medial);
    if (final_ > 0) {
        out += encode_codepoint(kJamoFinalBase + final_ - 1);
    }
}

void decompose_or_passthrough(uint32_t cp, std::string & out) {
    if (is_hangul_syllable(cp)) {
        append_hangul_jamo(cp, out);
    } else {
        out += encode_codepoint(cp);
    }
}

uint32_t katakana_to_hiragana_cp(uint32_t cp) {
    if (is_katakana(cp)) return cp - kKatakanaToHiragana;
    return cp;
}

std::string strip_trailing_tsv_columns(const std::string & line, size_t first_tab) {
    const auto second_tab = line.find('\t', first_tab + 1);
    if (second_tab == std::string::npos) return line.substr(first_tab + 1);
    return line.substr(first_tab + 1, second_tab - first_tab - 1);
}

struct TsvRow {
    uint32_t    codepoint;
    std::string code;
};

bool parse_tsv_line(const std::string & line, TsvRow & row) {
    if (line.empty()) return false;

    const auto tab = line.find('\t');
    if (tab == std::string::npos) return false;

    auto cps = decode_utf8(line.substr(0, tab));
    if (cps.size() != 1) return false;

    row.codepoint = cps[0];
    row.code = strip_trailing_tsv_columns(line, tab);
    return true;
}

#ifdef TTS_CPP_HAS_MECAB
std::string extract_csv_field(const char * feature, int field_idx) {
    int idx = 0;
    const char * p = feature;
    while (*p) {
        if (idx == field_idx) {
            const char * q = p;
            while (*q && *q != ',') ++q;
            return std::string(p, q);
        }
        if (*p == ',') ++idx;
        ++p;
    }
    return {};
}

bool has_valid_reading(const std::string & reading) {
    return !reading.empty() && reading != "*";
}

void append_node_reading(const mecab_node_t * node, std::string & out) {
    const std::string reading = extract_csv_field(node->feature, kIpadicReadingFieldIndex);
    if (has_valid_reading(reading)) {
        out += convert_katakana_to_hiragana(reading);
    } else {
        out.append(node->surface, node->length);
    }
}

std::string collect_readings_from_nodes(const mecab_node_t * node) {
    std::string out;
    for (; node; node = node->next) {
        if (node->stat == MECAB_BOS_NODE || node->stat == MECAB_EOS_NODE) continue;
        append_node_reading(node, out);
    }
    return out;
}

mecab_t * create_mecab_tagger(const std::string & dic_str) {
    std::string a0 = "mecab";
    std::string a1 = "-d";
    std::string a2 = dic_str;
    std::string a3 = "-r";
    std::string a4 = (std::filesystem::path(dic_str) / "mecabrc").string();
    char * argv[] = { a0.data(), a1.data(), a2.data(), a3.data(), a4.data() };
    return mecab_new(5, argv);
}
#endif

}  // namespace


std::vector<uint32_t> decode_utf8(const std::string & text) {
    std::vector<uint32_t> out;
    out.reserve(text.size());
    const auto * bytes = reinterpret_cast<const unsigned char *>(text.data());
    size_t pos = 0;
    while (pos < text.size()) {
        out.push_back(decode_single_codepoint(bytes, text.size(), pos));
    }
    return out;
}

std::string encode_codepoint(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0u | (cp >> 6));
        out += static_cast<char>(0x80u | (cp & 0x3Fu));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0u | (cp >> 12));
        out += static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
        out += static_cast<char>(0x80u | (cp & 0x3Fu));
    } else {
        out += static_cast<char>(0xF0u | (cp >> 18));
        out += static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu));
        out += static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
        out += static_cast<char>(0x80u | (cp & 0x3Fu));
    }
    return out;
}

std::string decompose_korean_to_jamo(const std::string & text) {
    auto cps = decode_utf8(text);
    std::string out;
    out.reserve(text.size() * 2);
    for (uint32_t cp : cps) {
        decompose_or_passthrough(cp, out);
    }
    return out;
}

std::string convert_katakana_to_hiragana(const std::string & text) {
    auto cps = decode_utf8(text);
    std::string out;
    out.reserve(cps.size());
    for (uint32_t cp : cps) {
        out += encode_codepoint(katakana_to_hiragana_cp(cp));
    }
    return out;
}


void CangjieTable::emit_cangjie_tokens(const Entry & entry, std::string & out) {
    for (size_t i = 0; i < entry.code.size(); ++i) {
        out += "[cj_";
        out += entry.code[i];
        out += "]";
    }
    if (!entry.suffix.empty()) {
        out += "[cj_";
        out += entry.suffix;
        out += "]";
    }
    out += "[cj_.]";
}

void CangjieTable::read_tsv_entries(std::istream & f,
                                    std::unordered_map<std::string, std::vector<uint32_t>> & code_to_cps) {
    std::string line;
    while (std::getline(f, line)) {
        TsvRow row;
        if (!parse_tsv_line(line, row)) continue;
        if (m_table.count(row.codepoint)) continue;

        code_to_cps[row.code].push_back(row.codepoint);
        m_table[row.codepoint] = Entry{row.code, ""};
    }
}

void CangjieTable::assign_disambiguation_suffixes(
        const std::unordered_map<std::string, std::vector<uint32_t>> & code_to_cps) {
    for (auto & [cp, entry] : m_table) {
        const auto & siblings = code_to_cps.at(entry.code);
        for (size_t i = 0; i < siblings.size(); ++i) {
            if (siblings[i] == cp && i > 0) {
                entry.suffix = std::to_string(i);
                break;
            }
        }
    }
}

void CangjieTable::load(const std::filesystem::path & tsv_path) {
    m_table.clear();
    std::ifstream f(tsv_path);
    if (!f) {
        throw std::runtime_error("CangjieTable::load: cannot open " + tsv_path.string());
    }

    std::unordered_map<std::string, std::vector<uint32_t>> code_to_cps;
    read_tsv_entries(f, code_to_cps);
    assign_disambiguation_suffixes(code_to_cps);
}

std::string CangjieTable::convert(const std::string & text) const {
    auto cps = decode_utf8(text);
    std::string out;
    out.reserve(text.size() * 8);
    for (uint32_t cp : cps) {
        if (is_cjk_ideograph(cp)) {
            auto it = m_table.find(cp);
            if (it != m_table.end()) {
                emit_cangjie_tokens(it->second, out);
                continue;
            }
        }
        out += encode_codepoint(cp);
    }
    return out;
}


#ifdef TTS_CPP_HAS_MECAB

struct MeCabTagger::Impl {
    mecab_t * tagger = nullptr;
    ~Impl() {
        if (tagger) {
            mecab_destroy(tagger);
            tagger = nullptr;
        }
    }
};

MeCabTagger::MeCabTagger()  : m_impl(std::make_unique<Impl>()) {}
MeCabTagger::~MeCabTagger() = default;
MeCabTagger::MeCabTagger(MeCabTagger &&) noexcept = default;
MeCabTagger & MeCabTagger::operator=(MeCabTagger &&) noexcept = default;

bool MeCabTagger::loaded() const { return m_impl && m_impl->tagger != nullptr; }

void MeCabTagger::load(const std::filesystem::path & dic_dir) {
    const std::string dic_str = dic_dir.string();
    mecab_t * t = create_mecab_tagger(dic_str);
    if (!t) {
        const char * err = mecab_strerror(nullptr);
        throw std::runtime_error(std::string("MeCabTagger::load failed: ") +
                                 (err ? err : "unknown") + " (dic=" + dic_str + ")");
    }
    if (m_impl->tagger) mecab_destroy(m_impl->tagger);
    m_impl->tagger = t;
}

std::string MeCabTagger::convert(const std::string & text) const {
    if (!loaded()) return convert_katakana_to_hiragana(text);

    const mecab_node_t * node = mecab_sparse_tonode(m_impl->tagger, text.c_str());
    if (!node) return convert_katakana_to_hiragana(text);

    return collect_readings_from_nodes(node);
}

#endif

}  // namespace tts_cpp::chatterbox::text_preprocess
