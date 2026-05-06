#include "mtl_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include "mtl_unicode_tables.inc"

namespace tts_cpp::chatterbox::detail {

namespace {

// ---- UTF-8 helpers ---------------------------------------------------------

static bool utf8_decode(const char * s, size_t len, size_t & pos, uint32_t & cp) {
    if (pos >= len) {
        return false;
    }
    uint8_t b0 = (uint8_t) s[pos];
    if (b0 < 0x80) {
        cp = b0;
        pos += 1;
        return true;
    }
    int extra;
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

static void utf8_append(uint32_t cp, std::string & out) {
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

static std::vector<uint32_t> utf8_to_cps(const std::string & s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    size_t pos = 0;
    uint32_t cp;
    while (utf8_decode(s.data(), s.size(), pos, cp)) {
        out.push_back(cp);
    }
    return out;
}

static std::string cps_to_utf8(const std::vector<uint32_t> & cps) {
    std::string out;
    out.reserve(cps.size() * 2);
    for (auto cp : cps) utf8_append(cp, out);
    return out;
}

// ---- Unicode classification (whitespace/word) ------------------------------

static bool cp_is_space(uint32_t cp) {
    if (cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D || cp == 0x20) return true;
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680) return true;
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000) return true;
    return false;
}

// Best-effort \w classifier matching HF tokenizers' Whitespace pre-tokenizer
// regex `\w+|[^\w\s]+`. Covers letters, marks, digits, underscore for the
// scripts that appear in tier-1 languages after NFKD decomposition.
static bool cp_is_word(uint32_t cp) {
    if (cp == 0x5F) return true;
    if (cp >= 0x30 && cp <= 0x39) return true;
    if (cp >= 0x41 && cp <= 0x5A) return true;
    if (cp >= 0x61 && cp <= 0x7A) return true;
    if (cp < 0x80) return false;
    if (cp >= 0xC0 && cp <= 0xFF) return cp != 0xD7 && cp != 0xF7;
    if (cp >= 0x0100 && cp <= 0x024F) return true;
    if (cp >= 0x0250 && cp <= 0x02AF) return true;
    if (cp >= 0x02B0 && cp <= 0x02FF) return true;
    if (cp >= 0x0300 && cp <= 0x036F) return true;
    if (cp >= 0x0370 && cp <= 0x03FF) {
        if (cp == 0x0374 || cp == 0x0375 || cp == 0x037E) return false;
        if (cp == 0x0384 || cp == 0x0385 || cp == 0x0387) return false;
        return true;
    }
    if (cp >= 0x0400 && cp <= 0x052F) return true;
    if (cp >= 0x0531 && cp <= 0x058F) return true;
    if (cp >= 0x0590 && cp <= 0x05FF) return true;
    if (cp >= 0x0600 && cp <= 0x06FF) {
        if (cp == 0x060C || cp == 0x061B || cp == 0x061F) return false;
        if (cp >= 0x066A && cp <= 0x066D) return false;
        return true;
    }
    if (cp >= 0x0700 && cp <= 0x074F) return true;
    if (cp >= 0x0900 && cp <= 0x097F) return true;
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    if (cp >= 0x1D00 && cp <= 0x1DBF) return true;
    if (cp >= 0x1E00 && cp <= 0x1FFF) return true;
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    if (cp >= 0x3130 && cp <= 0x318F) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0xA960 && cp <= 0xA97F) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0xFB00 && cp <= 0xFB4F) return true;
    if (cp >= 0xFB50 && cp <= 0xFDFF) return true;
    if (cp >= 0xFE70 && cp <= 0xFEFF) return true;
    if (cp >= 0xFF00 && cp <= 0xFFEF) {
        if (cp >= 0xFF01 && cp <= 0xFF0F) return false;
        if (cp >= 0xFF1A && cp <= 0xFF20) return false;
        if (cp >= 0xFF3B && cp <= 0xFF40) return false;
        if (cp >= 0xFF5B && cp <= 0xFF65) return false;
        return true;
    }
    return false;
}

// ---- NFKD + lowercase tables (binary search) -------------------------------

static const mtl_unicode_entry * find_entry(const mtl_unicode_entry * table, size_t n, uint32_t cp) {
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

static void append_lowercase_cp(uint32_t cp, std::vector<uint32_t> & out) {
    if (cp >= 0x41 && cp <= 0x5A) {
        out.push_back(cp + 0x20);
        return;
    }
    const auto * e = find_entry(k_mtl_lower_table, k_mtl_lower_table_len, cp);
    if (e) {
        for (uint32_t i = 0; i < e->length; ++i) out.push_back(k_mtl_lower_data[e->offset + i]);
    } else {
        out.push_back(cp);
    }
}

static void append_nfkd_cp(uint32_t cp, std::vector<uint32_t> & out) {
    // Hangul syllable decomposition (algorithmic).
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
        // NFKD decompositions in the table may themselves require further
        // decomposition (unicodedata.normalize already produces the fully
        // decomposed form, so this is typically one level).
        for (uint32_t i = 0; i < e->length; ++i) {
            out.push_back(k_mtl_nfkd_data[e->offset + i]);
        }
    } else {
        out.push_back(cp);
    }
}

// Canonical reordering of combining marks by combining class.
// For tier-1 scripts the common combining classes are 230 (above), 220 (below),
// 202 (attached below), 216 (attached above).  Looks up `cp` in the
// generated `k_mtl_ccc_table` (sorted by codepoint, full Unicode coverage)
// via binary search; matches `unicodedata.combining(chr(cp))` exactly so
// `canonical_reorder` agrees with Python's NFKD on every codepoint, not
// just tier-1.  The table is regenerated by scripts/gen-nfkd-table.py.
static int combining_class(uint32_t cp) {
    if (cp < 0x0300) return 0;
    size_t lo = 0, hi = k_mtl_ccc_table_len;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint32_t mcp = k_mtl_ccc_table[mid].cp;
        if (mcp == cp) return (int) k_mtl_ccc_table[mid].ccc;
        if (mcp < cp) lo = mid + 1;
        else          hi = mid;
    }
    return 0;
}

static void canonical_reorder(std::vector<uint32_t> & cps) {
    for (size_t i = 0; i < cps.size();) {
        size_t start = i;
        while (start < cps.size() && combining_class(cps[start]) == 0) ++start;
        size_t end = start;
        while (end < cps.size() && combining_class(cps[end]) != 0) ++end;
        if (end > start + 1) {
            std::stable_sort(cps.begin() + start, cps.begin() + end,
                [](uint32_t a, uint32_t b) { return combining_class(a) < combining_class(b); });
        }
        if (end == start) { ++i; }
        else { i = end; }
    }
}

static std::string nfkd_normalize(const std::string & s) {
    auto cps = utf8_to_cps(s);
    std::vector<uint32_t> out;
    out.reserve(cps.size() * 2);
    for (auto cp : cps) append_nfkd_cp(cp, out);
    canonical_reorder(out);
    return cps_to_utf8(out);
}

static std::string utf8_lowercase(const std::string & s) {
    auto cps = utf8_to_cps(s);
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    for (auto cp : cps) append_lowercase_cp(cp, out);
    return cps_to_utf8(out);
}

// Korean syllable decomposition (replicates python reference's `korean_normalize`).
// Decomposes Hangul syllables into initial+medial+final Jamo (U+1100 base ranges).
// Already what NFKD does algorithmically, but the reference also trims trailing
// whitespace after decomposition.
static std::string korean_normalize(const std::string & s) {
    auto cps = utf8_to_cps(s);
    std::vector<uint32_t> out;
    out.reserve(cps.size() * 3);
    for (auto cp : cps) {
        if (cp >= 0xAC00 && cp <= 0xD7AF) {
            uint32_t base = cp - 0xAC00;
            out.push_back(0x1100 + base / (21 * 28));
            out.push_back(0x1161 + (base % (21 * 28)) / 28);
            uint32_t t = base % 28;
            if (t != 0) out.push_back(0x11A7 + t);
        } else {
            out.push_back(cp);
        }
    }
    // Python strip(): remove leading/trailing whitespace.
    size_t lo = 0, hi = out.size();
    while (lo < hi && cp_is_space(out[lo])) ++lo;
    while (hi > lo && cp_is_space(out[hi - 1])) --hi;
    return cps_to_utf8(std::vector<uint32_t>(out.begin() + lo, out.begin() + hi));
}

// ---- Minimal JSON parser ---------------------------------------------------

struct json_value {
    enum kind_t { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } kind = J_NULL;
    bool                                               boolean = false;
    double                                             number = 0;
    std::string                                        str;
    std::vector<json_value>                            arr;
    std::map<std::string, json_value>                  obj;

    const json_value * find(const std::string & k) const {
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
};

struct json_parser {
    const char * p;
    const char * end;

    void skip_ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }

    [[noreturn]] void fail(const char * msg) { throw std::runtime_error(std::string("JSON: ") + msg); }

    std::string parse_string() {
        if (p >= end || *p != '"') fail("expected string");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p >= end) fail("bad escape");
                char c = *p++;
                switch (c) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        // Helper: try to parse 4 hex digits at p[0..3] into *out.
                        // Returns true iff all four are valid hex.  Caller is
                        // responsible for ensuring p + 4 <= end before calling.
                        auto try_parse_hex4 = [](const char * q, uint32_t & out_cp) -> bool {
                            uint32_t v = 0;
                            for (int i = 0; i < 4; ++i) {
                                char h = q[i];
                                v <<= 4;
                                if      (h >= '0' && h <= '9') v |= (uint32_t)(h - '0');
                                else if (h >= 'a' && h <= 'f') v |= (uint32_t)(h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') v |= (uint32_t)(h - 'A' + 10);
                                else return false;
                            }
                            out_cp = v;
                            return true;
                        };

                        if (p + 4 > end) fail("short \\u");
                        uint32_t cp = 0;
                        if (!try_parse_hex4(p, cp)) fail("bad hex");
                        p += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // Need \uXXXX continuation = 6 more bytes.  Be
                            // lenient: any malformation (truncated, missing
                            // \u marker, non-hex digits) emits U+FFFD instead
                            // of throwing, matching the truncated-input path
                            // and what unicodedata / WHATWG-style parsers do
                            // on lone surrogates.  Treats embedded tokenizer
                            // JSON as untrusted input.
                            uint32_t cp2 = 0;
                            const bool have_continuation =
                                p + 6 <= end &&
                                p[0] == '\\' && p[1] == 'u' &&
                                try_parse_hex4(p + 2, cp2) &&
                                cp2 >= 0xDC00 && cp2 <= 0xDFFF;
                            if (!have_continuation) {
                                utf8_append(0xFFFD, out);
                                break;
                            }
                            p += 6;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            // Lone low surrogate without preceding high.
                            utf8_append(0xFFFD, out);
                            break;
                        }
                        utf8_append(cp, out);
                        break;
                    }
                    default: fail("bad escape char");
                }
            } else {
                out.push_back(*p++);
            }
        }
        if (p >= end) fail("unterminated string");
        ++p;
        return out;
    }

    json_value parse_value() {
        skip_ws();
        if (p >= end) fail("unexpected eof");
        json_value v;
        char c = *p;
        if (c == '"') {
            v.kind = json_value::J_STR;
            v.str = parse_string();
        } else if (c == '{') {
            v.kind = json_value::J_OBJ;
            ++p;
            skip_ws();
            if (p < end && *p == '}') { ++p; return v; }
            while (p < end) {
                skip_ws();
                std::string key = parse_string();
                skip_ws();
                if (p >= end || *p != ':') fail("expected :");
                ++p;
                v.obj.emplace(std::move(key), parse_value());
                skip_ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; break; }
                fail("expected , or }");
            }
        } else if (c == '[') {
            v.kind = json_value::J_ARR;
            ++p;
            skip_ws();
            if (p < end && *p == ']') { ++p; return v; }
            while (p < end) {
                v.arr.push_back(parse_value());
                skip_ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; break; }
                fail("expected , or ]");
            }
        } else if (c == 't' || c == 'f') {
            if (end - p >= 4 && std::memcmp(p, "true", 4) == 0) { v.kind = json_value::J_BOOL; v.boolean = true; p += 4; }
            else if (end - p >= 5 && std::memcmp(p, "false", 5) == 0) { v.kind = json_value::J_BOOL; v.boolean = false; p += 5; }
            else fail("bad literal");
        } else if (c == 'n') {
            if (end - p >= 4 && std::memcmp(p, "null", 4) == 0) { v.kind = json_value::J_NULL; p += 4; }
            else fail("bad null");
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            v.kind = json_value::J_NUM;
            const char * s = p;
            if (*p == '-') ++p;
            while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
            v.number = std::strtod(std::string(s, p - s).c_str(), nullptr);
        } else {
            fail("bad token");
        }
        return v;
    }
};

// ---- Tier-1 language lists -------------------------------------------------

} // namespace

// Tier-1 / all-known language tables.  Held inside Meyers-style function-local
// statics rather than namespace-scope globals so they're built lazily on first
// use (no static-init-order coupling) and benefit from C++11's thread-safe
// local-static initialisation.
const std::vector<std::string> & mtl_tokenizer::supported_languages() {
    static const std::vector<std::string> k_supported = {
        "en","es","fr","de","it","pt","nl","pl","tr","sv","da","fi","no","el","ms","sw","ar","ko"
    };
    return k_supported;
}

const std::vector<std::string> & mtl_tokenizer::all_known_languages() {
    static const std::vector<std::string> k_all_known = {
        "en","es","fr","de","it","pt","nl","pl","tr","sv","da","fi","no","el",
        "ms","sw","ar","ko","ja","he","ru","zh","hi"
    };
    return k_all_known;
}

bool mtl_tokenizer::is_language_supported(const std::string & lang) const {
    for (const auto & s : supported_languages()) if (s == lang) return true;
    return false;
}

int32_t mtl_tokenizer::sot_id() const { return m_sot_id; }
int32_t mtl_tokenizer::eot_id() const { return m_eot_id; }
int32_t mtl_tokenizer::unk_id() const { return m_unk_id; }
int32_t mtl_tokenizer::vocab_size() const { return (int32_t) m_id_to_token.size(); }

void mtl_tokenizer::index_vocab() {
    int32_t max_id = -1;
    for (const auto & kv : m_vocab) if (kv.second > max_id) max_id = kv.second;
    for (const auto & t : m_added_tokens) if (t.id > max_id) max_id = t.id;
    m_id_to_token.assign(max_id + 1, std::string());
    for (const auto & kv : m_vocab) {
        if (kv.second >= 0 && kv.second <= max_id) m_id_to_token[kv.second] = kv.first;
    }
    for (const auto & t : m_added_tokens) {
        if (t.id >= 0 && t.id <= max_id) m_id_to_token[t.id] = t.content;
    }
}

bool mtl_tokenizer::load_from_json(const std::string & json_blob) {
    json_parser jp{json_blob.data(), json_blob.data() + json_blob.size()};
    json_value root;
    try {
        root = jp.parse_value();
    } catch (const std::exception & e) {
        fprintf(stderr, "mtl_tokenizer: failed to parse JSON: %s\n", e.what());
        return false;
    }

    if (root.kind != json_value::J_OBJ) return false;

    const auto * model = root.find("model");
    if (!model || model->kind != json_value::J_OBJ) return false;
    const auto * type = model->find("type");
    if (!type || type->kind != json_value::J_STR || type->str != "BPE") {
        fprintf(stderr, "mtl_tokenizer: unsupported model type '%s' (expected BPE)\n",
                type ? type->str.c_str() : "<missing>");
        return false;
    }

    const auto * vocab = model->find("vocab");
    if (!vocab || vocab->kind != json_value::J_OBJ) return false;
    m_vocab.clear();
    m_vocab.reserve(vocab->obj.size());
    for (const auto & kv : vocab->obj) {
        if (kv.second.kind != json_value::J_NUM) continue;
        m_vocab[kv.first] = (int32_t) kv.second.number;
    }

    const auto * merges = model->find("merges");
    if (!merges || merges->kind != json_value::J_ARR) return false;
    m_bpe_ranks.clear();
    m_bpe_ranks.reserve(merges->arr.size());
    for (size_t i = 0; i < merges->arr.size(); ++i) {
        if (merges->arr[i].kind != json_value::J_STR) continue;
        m_bpe_ranks[merges->arr[i].str] = (int32_t) i;
    }

    const auto * unk = model->find("unk_token");
    if (unk && unk->kind == json_value::J_STR) m_unk_token = unk->str;

    m_added_tokens.clear();
    const auto * added = root.find("added_tokens");
    if (added && added->kind == json_value::J_ARR) {
        for (const auto & a : added->arr) {
            if (a.kind != json_value::J_OBJ) continue;
            const auto * c = a.find("content");
            const auto * idv = a.find("id");
            if (!c || c->kind != json_value::J_STR) continue;
            if (!idv || idv->kind != json_value::J_NUM) continue;
            m_added_tokens.push_back({c->str, (int32_t) idv->number});
        }
    }

    // Sort added tokens by descending length so longer literals match first
    // (e.g. "[SPACE]" before any shorter prefix).
    std::sort(m_added_tokens.begin(), m_added_tokens.end(),
        [](const added_token & a, const added_token & b) { return a.content.size() > b.content.size(); });

    auto find_added_id = [&](const std::string & c) -> int32_t {
        for (const auto & t : m_added_tokens) if (t.content == c) return t.id;
        auto it = m_vocab.find(c);
        return it == m_vocab.end() ? -1 : it->second;
    };

    m_sot_id = find_added_id("[START]");
    m_eot_id = find_added_id("[STOP]");
    m_unk_id = find_added_id(m_unk_token);
    m_space_id = find_added_id("[SPACE]");

    if (m_sot_id < 0 || m_eot_id < 0 || m_unk_id < 0) {
        fprintf(stderr, "mtl_tokenizer: missing required special tokens (sot=%d eot=%d unk=%d)\n",
                m_sot_id, m_eot_id, m_unk_id);
        return false;
    }

    index_vocab();
    return true;
}

bool mtl_tokenizer::load_from_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "mtl_tokenizer: failed to open %s\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return load_from_json(ss.str());
}

// ---- BPE on a pre-token ----------------------------------------------------

// Naive O(N^2) merge loop, where N is the *pre-token* length in UTF-8
// codepoints (not the full input — the encoder splits on whitespace and
// special tokens before getting here).  Real-world pre-tokens are
// short: 5-15 chars for European words, ~30-40 for Korean/Turkish/
// Finnish agglutinations, occasionally ~100 for URL fragments.  Even at
// 100 chars the inner loop tops out at ~10k hash lookups (microseconds),
// so the encoder is not the bottleneck on the synthesis hot path
// (T3 forward pass per token: ~30ms; tokenising a paragraph: <2 ms total).
//
// HuggingFace's reference Python `tokenizers` library uses the same
// O(N^2) shape; switching to a heap-based incremental merge would be a
// 5x code-complexity bump for 0% speedup at typical pre-token lengths.
// If a future workload regresses on this (very long agglutinated input
// seen in the encoder profile), revisit with the heap pattern.
void mtl_tokenizer::bpe_word(const std::string & word, std::vector<int32_t> & out) const {
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < word.size()) {
        size_t start = pos;
        uint32_t cp;
        utf8_decode(word.data(), word.size(), pos, cp);
        parts.emplace_back(word.data() + start, pos - start);
    }
    if (parts.empty()) return;

    while (parts.size() >= 2) {
        int best_rank = INT32_MAX;
        size_t best_idx = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            auto it = m_bpe_ranks.find(parts[i] + " " + parts[i + 1]);
            if (it != m_bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        std::vector<std::string> merged;
        merged.reserve(parts.size() - 1);
        for (size_t i = 0; i < parts.size();) {
            if (i == best_idx) {
                merged.push_back(parts[i] + parts[i + 1]);
                i += 2;
            } else {
                merged.push_back(parts[i]);
                i += 1;
            }
        }
        parts = std::move(merged);
    }

    for (const auto & t : parts) {
        auto it = m_vocab.find(t);
        if (it == m_vocab.end()) {
            out.push_back(m_unk_id);
        } else {
            out.push_back(it->second);
        }
    }
}

// ---- Encode ----------------------------------------------------------------

std::vector<int32_t> mtl_tokenizer::encode(const std::string & text,
                                            const std::string & language_id) const {
    std::string txt = text;

    if (!language_id.empty()) {
        if (language_id == "ja" || language_id == "he" || language_id == "ru" ||
            language_id == "zh" || language_id == "hi") {
            throw std::runtime_error(
                "mtl_tokenizer: language '" + language_id + "' requires preprocessing not "
                "included in this build (pykakasi / dicta / russian_text_stresser / "
                "Cangjie mapping). Pre-process the text externally before passing it in.");
        }
        if (!is_language_supported(language_id)) {
            throw std::runtime_error("mtl_tokenizer: unsupported language '" + language_id + "'");
        }
    }

    txt = utf8_lowercase(txt);
    txt = nfkd_normalize(txt);

    if (language_id == "ko") {
        txt = korean_normalize(txt);
    }

    if (!language_id.empty()) {
        txt = std::string("[") + language_id + "]" + txt;
    }

    {
        std::string r;
        r.reserve(txt.size() + 16);
        for (size_t i = 0; i < txt.size(); ++i) {
            if (txt[i] == ' ') r += "[SPACE]";
            else r.push_back(txt[i]);
        }
        txt = std::move(r);
    }

    // Split on added tokens (longest-first match).
    std::vector<std::pair<std::string, int32_t>> segments;
    {
        size_t i = 0;
        std::string buf;
        while (i < txt.size()) {
            bool matched = false;
            for (const auto & tok : m_added_tokens) {
                if (tok.content.empty()) continue;
                if (txt.compare(i, tok.content.size(), tok.content) == 0) {
                    if (!buf.empty()) {
                        segments.emplace_back(std::move(buf), -1);
                        buf.clear();
                    }
                    segments.emplace_back(tok.content, tok.id);
                    i += tok.content.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                buf.push_back(txt[i]);
                ++i;
            }
        }
        if (!buf.empty()) segments.emplace_back(std::move(buf), -1);
    }

    std::vector<int32_t> out;
    out.reserve(txt.size());

    for (const auto & seg : segments) {
        if (seg.second >= 0) {
            out.push_back(seg.second);
            continue;
        }
        // Whitespace pre-tokenizer: emit runs of word chars or runs of non-word non-space chars.
        const std::string & s = seg.first;
        auto cps = utf8_to_cps(s);
        size_t i = 0;
        while (i < cps.size()) {
            while (i < cps.size() && cp_is_space(cps[i])) ++i;
            if (i >= cps.size()) break;
            size_t start = i;
            bool word = cp_is_word(cps[i]);
            while (i < cps.size() && !cp_is_space(cps[i]) && cp_is_word(cps[i]) == word) ++i;
            std::vector<uint32_t> chunk(cps.begin() + start, cps.begin() + i);
            std::string chunk_utf8 = cps_to_utf8(chunk);
            bpe_word(chunk_utf8, out);
        }
    }

    return out;
}

// Best-effort decoder for debugging.  Joins regular tokens with a space and
// expands [SPACE] tokens.  The `!out.empty()` guard avoids a leading space
// on the first emitted regular token even when the first id(s) are skipped
// special markers ([START], [PAD], …) — matching the Python reference's
// `decode`, which strips leading whitespace.  Real synthesis only
// round-trips through `encode()`; this path is debug-only.
std::string mtl_tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string out;
    out.reserve(ids.size() * 2);
    for (size_t k = 0; k < ids.size(); ++k) {
        int32_t id = ids[k];
        if (id < 0 || id >= (int32_t) m_id_to_token.size()) continue;
        const std::string & tok = m_id_to_token[id];
        if (tok == "[SPACE]") {
            out.push_back(' ');
            continue;
        }
        if (tok == "[START]" || tok == "[STOP]" || tok == "[UNK]" || tok == "[PAD]") continue;
        if (!out.empty()) out.push_back(' ');
        out += tok;
    }
    return out;
}

} // namespace tts_cpp::chatterbox::detail
