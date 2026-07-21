// SentencePiece-unigram tokenizer replicating the HF fast tokenizer of
// parler-tts-mini-v1 (T5). Verified token-exact against the HF corpus fixture
// in artifacts/parler-ref/tokenizer_corpus.json. Charsmap walk follows the
// XCDA scheme used by sentencepiece / llama.cpp's ugm tokenizer.

#include "tokenizer.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>

namespace tts_cpp {
namespace parler {
namespace detail {

// SentencePiece whitespace marker U+2581 (lower one eighth block)
static const char k_marker[] = "\xe2\x96\x81";

static size_t utf8_len(uint8_t c) {
    if (c < 0x80)          return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 0;  // invalid lead byte
}

// length of one well-formed UTF-8 sequence at input[offset], 0 if malformed
static size_t utf8_valid_len(const std::string & input, size_t offset) {
    size_t len = utf8_len((uint8_t) input[offset]);
    if (len == 0 || offset + len > input.size()) {
        return 0;
    }
    for (size_t i = 1; i < len; ++i) {
        if (((uint8_t) input[offset + i] & 0xc0) != 0x80) {
            return 0;
        }
    }
    return len;
}

bool parler_tokenizer::load(std::vector<std::string> pieces, std::vector<float> scores,
                            std::vector<uint8_t> charsmap, int unk_id, int eos_id, bool add_eos) {
    if (pieces.empty() || pieces.size() != scores.size()) {
        fprintf(stderr, "parler_tokenizer: bad vocab (%zu pieces, %zu scores)\n",
                pieces.size(), scores.size());
        return false;
    }
    if (unk_id < 0 || (size_t) unk_id >= pieces.size() ||
        eos_id < 0 || (size_t) eos_id >= pieces.size()) {
        fprintf(stderr, "parler_tokenizer: unk/eos id out of range\n");
        return false;
    }

    // charsmap blob: u32 LE byte size of the XCDA section, then the XCDA
    // entries (u32 each), then the NUL-terminated replacement-string pool
    m_charsmap    = std::move(charsmap);
    m_xcda_words  = 0;
    m_repl_offset = 0;
    m_repl_size   = 0;
    if (!m_charsmap.empty()) {
        uint32_t xcda_bytes = 0;
        if (m_charsmap.size() < sizeof(xcda_bytes)) {
            fprintf(stderr, "parler_tokenizer: charsmap too small (%zu bytes)\n", m_charsmap.size());
            return false;
        }
        memcpy(&xcda_bytes, m_charsmap.data(), sizeof(xcda_bytes));
        if (sizeof(xcda_bytes) + (size_t) xcda_bytes > m_charsmap.size()) {
            fprintf(stderr, "parler_tokenizer: charsmap XCDA size out of bounds\n");
            return false;
        }
        m_xcda_words  = xcda_bytes / sizeof(uint32_t);
        m_repl_offset = sizeof(xcda_bytes) + xcda_bytes;
        m_repl_size   = m_charsmap.size() - m_repl_offset;
    }

    m_scores = std::move(scores);
    float min_score = FLT_MAX;
    for (float s : m_scores) {
        min_score = std::min(min_score, s);
    }
    m_unk_score = (double) min_score - 10.0;

    m_trie.clear();
    m_trie.emplace_back();
    for (size_t id = 0; id < pieces.size(); ++id) {
        const std::string & piece = pieces[id];
        if (piece.empty()) {
            continue;
        }
        size_t node = 0;
        for (char ch : piece) {
            auto it = m_trie[node].next.find((uint8_t) ch);
            if (it != m_trie[node].next.end()) {
                node = (size_t) it->second;
            } else {
                size_t child = m_trie.size();
                m_trie[node].next.emplace((uint8_t) ch, (int32_t) child);
                m_trie.emplace_back();
                node = child;
            }
        }
        if (m_trie[node].token_id < 0) {
            m_trie[node].token_id = (int32_t) id;
        }
    }

    m_unk_id  = unk_id;
    m_eos_id  = eos_id;
    m_add_eos = add_eos;
    return true;
}

uint32_t parler_tokenizer::xcda_node(size_t index) const {
    if (index >= m_xcda_words) {
        return 0;  // LCHECK 0 never matches a non-NUL byte, so the walk stops
    }
    uint32_t v;
    memcpy(&v, m_charsmap.data() + sizeof(uint32_t) + index * sizeof(uint32_t), sizeof(v));
    return v;
}

// XCDA entry packing: BASE in bits 10-30 (shifted left by 8 more when bit 9
// is set), LCHECK in bits 0-7 (bit 31 marks value nodes and poisons LCHECK),
// LEAF flag in bit 8; value nodes store the replacement offset in bits 0-30.
static inline uint32_t xcda_base(uint32_t packed)   { return (packed >> 10) << ((packed & (1u << 9)) >> 6); }
static inline uint32_t xcda_lcheck(uint32_t packed) { return packed & ((1u << 31) | 0xff); }
static inline bool     xcda_leaf(uint32_t packed)   { return (packed >> 8) & 1; }
static inline uint32_t xcda_value(uint32_t packed)  { return packed & ((1u << 31) - 1); }

parler_tokenizer::norm_span parler_tokenizer::normalize_prefix(const std::string & input,
                                                               size_t offset) const {
    size_t   best_len = 0;
    uint32_t best_val = 0;

    if (m_xcda_words > 0) {
        // longest-prefix walk: child index is BASE[s] ^ c, valid only when
        // LCHECK[child] == c; a LEAF child's BASE points at the value node
        // holding the replacement-pool offset for the prefix matched so far
        uint32_t node = xcda_base(xcda_node(0));
        for (size_t p = offset; p < input.size(); ++p) {
            uint8_t c = (uint8_t) input[p];
            if (c == 0) {
                break;
            }
            node ^= c;
            uint32_t packed = xcda_node(node);
            if (xcda_lcheck(packed) != c) {
                break;
            }
            bool leaf = xcda_leaf(packed);
            node ^= xcda_base(packed);
            if (leaf) {
                best_len = p - offset + 1;
                best_val = xcda_value(xcda_node(node));
            }
        }
    }

    if (best_len > 0 && best_val < m_repl_size) {
        const char * repl = (const char *) m_charsmap.data() + m_repl_offset + best_val;
        size_t max_len = m_repl_size - best_val;
        size_t repl_len = 0;
        while (repl_len < max_len && repl[repl_len] != '\0') {
            repl_len++;
        }
        if (repl_len < max_len) {
            return { repl, repl_len, best_len };
        }
        // unterminated pool entry: fall through to passthrough
    }

    size_t len = utf8_valid_len(input, offset);
    if (len > 0) {
        return { input.data() + offset, len, len };  // unmatched codepoint passes through
    }
    return { "\xef\xbf\xbd", 3, 1 };  // stray byte -> U+FFFD
}

std::string parler_tokenizer::normalize(const std::string & text) const {
    std::string mapped;
    mapped.reserve(text.size());
    for (size_t offset = 0; offset < text.size(); ) {
        norm_span span = normalize_prefix(text, offset);
        mapped.append(span.data, span.len);
        offset += span.consumed;
    }

    // second normalizer stage of the HF pipeline: Replace " {2,}" -> " "
    // (runs of ASCII spaces collapse to one; no leading/trailing strip)
    std::string out;
    out.reserve(mapped.size());
    for (char c : mapped) {
        if (c == ' ' && !out.empty() && out.back() == ' ') {
            continue;
        }
        out.push_back(c);
    }
    return out;
}

void parler_tokenizer::viterbi(const char * s, size_t n, std::vector<int32_t> & out) const {
    if (n == 0) {
        return;
    }

    struct best_end {
        int32_t id;     // token ending at this byte position
        size_t  start;  // where that token starts
        double  score;
    };
    std::vector<best_end> res(n + 1, { m_unk_id, 0, -DBL_MAX });
    res[0].score = 0.0;

    for (size_t offset = 0; offset < n; ) {
        size_t cpt = utf8_len((uint8_t) s[offset]);
        if (cpt == 0) {
            cpt = 1;
        }
        cpt = std::min(cpt, n - offset);

        const best_end cur = res[offset];
        bool   single_cpt_token = false;
        size_t node = 0;
        for (size_t p = offset; p < n; ) {
            auto it = m_trie[node].next.find((uint8_t) s[p]);
            if (it == m_trie[node].next.end()) {
                break;
            }
            node = (size_t) it->second;
            p++;
            int32_t tid = m_trie[node].token_id;
            if (tid >= 0) {
                if (p - offset == cpt) {
                    single_cpt_token = true;
                }
                // double sums keep results identical to the HF tokenizer
                double sc = cur.score + (double) m_scores[tid];
                if (sc > res[p].score) {
                    res[p] = { tid, offset, sc };
                }
            }
        }

        // no piece covers exactly this codepoint: offer unk over it
        if (!single_cpt_token) {
            double sc = cur.score + m_unk_score;
            if (sc > res[offset + cpt].score) {
                res[offset + cpt] = { m_unk_id, offset, sc };
            }
        }
        offset += cpt;
    }

    // backtrack; consecutive unk tokens fuse into one (sentencepiece fuse_unk)
    size_t first = out.size();
    bool prev_unk = false;
    for (const best_end * t = &res[n]; ; t = &res[t->start]) {
        bool is_unk = (t->id == m_unk_id);
        if (!(prev_unk && is_unk)) {
            out.push_back(t->id);
        }
        if (t->start == 0) {
            break;
        }
        prev_unk = is_unk;
    }
    std::reverse(out.begin() + first, out.end());
}

std::vector<int32_t> parler_tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;

    const std::string norm = normalize(text);
    if (!norm.empty()) {
        // Metaspace: every ' ' becomes the marker; prepend one when the
        // replaced string would not already start with it (prepend_scheme=always)
        std::string marked;
        marked.reserve(norm.size() + sizeof(k_marker));
        if (norm[0] != ' ' && norm.compare(0, 3, k_marker) != 0) {
            marked += k_marker;
        }
        for (char c : norm) {
            if (c == ' ') {
                marked += k_marker;
            } else {
                marked.push_back(c);
            }
        }

        // split=true: Viterbi runs per pre-token; every marker starts one
        size_t start = 0;
        while (start < marked.size()) {
            size_t end = marked.find(k_marker, start + 1);
            if (end == std::string::npos) {
                end = marked.size();
            }
            viterbi(marked.data() + start, end - start, ids);
            start = end;
        }
    }

    if (m_add_eos) {
        ids.push_back(m_eos_id);
    }
    return ids;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
