#include "gpt2_bpe.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <queue>
#include <regex>
#include <sstream>
#include <unordered_set>

// ---- byte-level encoding (GPT-2 convention) --------------------------------

static std::unordered_map<uint8_t, std::string> build_byte_to_unicode() {
    std::unordered_map<uint8_t, std::string> m;
    auto cpt_to_utf8 = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    };
    for (int c = 0x21; c <= 0x7E; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    for (int c = 0xA1; c <= 0xAC; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    for (int c = 0xAE; c <= 0xFF; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    int n = 0;
    for (int c = 0; c < 256; ++c) {
        if (m.find((uint8_t)c) == m.end()) {
            m[(uint8_t)c] = cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return m;
}

static const std::unordered_map<uint8_t, std::string> & byte_to_unicode() {
    static auto m = build_byte_to_unicode();
    return m;
}

static std::string bytes_to_unicode_str(const std::string & raw) {
    std::string out;
    auto & b2u = byte_to_unicode();
    for (unsigned char c : raw) out += b2u.at(c);
    return out;
}

// ---- load tokenizer from arrays embedded in GGUF metadata -----------------

bool gpt2_bpe::load_from_arrays(const std::vector<std::string> & tokens,
                                const std::vector<std::string> & merges) {
    if (tokens.empty()) return false;

    id_to_token = tokens;
    token_to_id.clear();
    token_to_id.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id[tokens[i]] = (int32_t) i;
    }

    bpe_ranks.clear();
    bpe_ranks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        bpe_ranks[merges[i]] = (int) i;
    }
    return true;
}

// ---- GPT-2 pre-tokenization regex ------------------------------------------
// Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

static std::vector<std::string> gpt2_regex_split(const std::string & text) {
    static const std::regex re(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)",
        std::regex::optimize);

    std::vector<std::string> words;
    auto begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        words.push_back(it->str());
    }
    return words;
}

// ---- BPE merge algorithm ---------------------------------------------------

static int find_rank(const std::unordered_map<std::string, int> & ranks,
                     const std::string & left, const std::string & right) {
    auto it = ranks.find(left + " " + right);
    return it != ranks.end() ? it->second : -1;
}

static std::vector<std::string> bpe_merge(const std::string & token,
                                          const std::unordered_map<std::string, int> & ranks) {
    // split token into individual UTF-8 characters
    std::vector<std::string> parts;
    for (size_t i = 0; i < token.size(); ) {
        size_t len = 1;
        unsigned char c = (unsigned char)token[i];
        if      ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        parts.push_back(token.substr(i, len));
        i += len;
    }

    while (parts.size() >= 2) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            int r = find_rank(ranks, parts[i], parts[i+1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_rank == INT32_MAX) break;
        parts[best_i] = parts[best_i] + parts[best_i + 1];
        parts.erase(parts.begin() + (int)best_i + 1);
    }
    return parts;
}

// ---- tokenize --------------------------------------------------------------

std::vector<int32_t> gpt2_bpe::tokenize(const std::string & text) const {
    std::vector<int32_t> ids;
    if (text.empty()) return ids;

    // check added tokens first (paralinguistic tags like [laugh])
    struct added_span { size_t start; size_t len; int32_t id; };
    std::vector<added_span> spans;
    for (auto & [tok, id] : token_to_id) {
        if (id < 50257) continue; // only added tokens
        size_t pos = 0;
        while ((pos = text.find(tok, pos)) != std::string::npos) {
            spans.push_back({pos, tok.size(), id});
            pos += tok.size();
        }
    }
    std::sort(spans.begin(), spans.end(), [](const added_span & a, const added_span & b) { return a.start < b.start; });

    // remove overlapping spans (keep earliest)
    std::vector<added_span> clean;
    size_t last_end = 0;
    for (auto & sp : spans) {
        if (sp.start >= last_end) { clean.push_back(sp); last_end = sp.start + sp.len; }
    }

    auto tokenize_fragment = [&](const std::string & frag) {
        auto words = gpt2_regex_split(frag);
        for (auto & word : words) {
            std::string bpe_input = bytes_to_unicode_str(word);
            auto parts = bpe_merge(bpe_input, bpe_ranks);
            for (auto & part : parts) {
                auto it = token_to_id.find(part);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                } else {
                    for (unsigned char c : word) {
                        auto & b2u = byte_to_unicode();
                        auto jt = token_to_id.find(b2u.at(c));
                        if (jt != token_to_id.end()) ids.push_back(jt->second);
                    }
                }
            }
        }
    };

    size_t cursor = 0;
    for (auto & sp : clean) {
        if (sp.start > cursor) tokenize_fragment(text.substr(cursor, sp.start - cursor));
        ids.push_back(sp.id);
        cursor = sp.start + sp.len;
    }
    if (cursor < text.size()) tokenize_fragment(text.substr(cursor));

    return ids;
}

// ---- punc_norm (matches Python tts_turbo.py) -------------------------------

std::string gpt2_bpe::punc_norm(const std::string & text) {
    if (text.empty()) return "You need to add some text for me to talk.";

    std::string t = text;

    // capitalize first letter
    if (t[0] >= 'a' && t[0] <= 'z') t[0] = t[0] - 'a' + 'A';

    // collapse multiple spaces
    {
        std::string r;
        bool prev_space = false;
        for (char c : t) {
            if (c == ' ') { if (!prev_space) r += c; prev_space = true; }
            else { r += c; prev_space = false; }
        }
        t = r;
    }

    // replace uncommon punctuation
    auto replace_all = [](std::string & s, const std::string & from, const std::string & to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(t, "\xe2\x80\xa6", ", ");   // …
    replace_all(t, ":", ",");
    replace_all(t, "\xe2\x80\x94", "-");     // —
    replace_all(t, "\xe2\x80\x93", "-");     // –
    replace_all(t, " ,", ",");
    replace_all(t, "\xe2\x80\x9c", "\"");    // "
    replace_all(t, "\xe2\x80\x9d", "\"");    // "
    replace_all(t, "\xe2\x80\x98", "'");     // '
    replace_all(t, "\xe2\x80\x99", "'");     // '

    // strip trailing whitespace (spaces, tabs, newlines).  Sentences
    // coming from the live tail-follow reader typically end in '\n' or
    // ' ' because that's the terminator pop_sentence used; leaving that
    // in place confuses the "add period if no ending punctuation" step
    // below (it would append '.' after the newline, producing ".\n.").
    while (!t.empty()) {
        char b = t.back();
        if (b == ' ' || b == '\t' || b == '\n' || b == '\r') t.pop_back();
        else break;
    }

    // add period if no ending punctuation
    if (!t.empty()) {
        char last = t.back();
        if (last != '.' && last != '!' && last != '?' && last != '-' && last != ',')
            t += '.';
    }

    return t;
}
