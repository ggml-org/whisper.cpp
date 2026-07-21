#include "bpe_tokenizer.h"

#include <cstdio>
#include <queue>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

constexpr const char * kSpace = "\xe2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK

size_t utf8_len(uint8_t lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xe0) == 0xc0) return 2;
    if ((lead & 0xf0) == 0xe0) return 3;
    if ((lead & 0xf8) == 0xf0) return 4;
    return 1;  // invalid lead byte: treat as a lone byte, byte fallback covers it
}

} // namespace

bool parler_bpe_tokenizer::load(const std::vector<std::string> & pieces,
                                const std::vector<std::string> & merges,
                                int unk_id, int bos_id, bool add_bos) {
    if (pieces.empty()) return false;
    m_vocab.clear();
    m_vocab.reserve(pieces.size() * 2);
    for (size_t i = 0; i < pieces.size(); ++i) {
        m_vocab.emplace(pieces[i], (int32_t) i);
    }
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        auto it = m_vocab.find(name);
        m_byte_id[b] = it == m_vocab.end() ? -1 : it->second;
    }
    m_merges.clear();
    m_merges.reserve(merges.size() * 2);
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string & m = merges[rank];
        const size_t sp = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos) {
            std::fprintf(stderr, "parler bpe: malformed merge '%s'\n", m.c_str());
            return false;
        }
        const std::string left = m.substr(0, sp), right = m.substr(sp + 1);
        auto l = m_vocab.find(left), r = m_vocab.find(right), o = m_vocab.find(left + right);
        if (l == m_vocab.end() || r == m_vocab.end() || o == m_vocab.end()) {
            std::fprintf(stderr, "parler bpe: merge references unknown piece '%s'\n", m.c_str());
            return false;
        }
        const uint64_t key = ((uint64_t) (uint32_t) l->second << 32) | (uint32_t) r->second;
        m_merges.emplace(key, ((uint64_t) rank << 32) | (uint32_t) o->second);
    }
    m_unk_id  = unk_id;
    m_bos_id  = bos_id;
    m_add_bos = add_bos;
    return true;
}

std::vector<int32_t> parler_bpe_tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> out;
    if (m_add_bos) out.push_back(m_bos_id);
    if (text.empty()) return out;

    // Metaspace(prepend_scheme=first, split=false): every ' ' becomes the
    // marker FIRST, then the marker is prepended only if not already leading.
    std::string norm;
    norm.reserve(text.size() + 3);
    for (char c : text) {
        if (c == ' ') norm += kSpace; else norm += c;
    }
    if (norm.compare(0, 3, kSpace) != 0) norm.insert(0, kSpace);

    // initial symbols: one per codepoint if in vocab, else its <0xXX> bytes;
    // consecutive unknowns fuse (HF fuse_unk — unreachable with byte tokens)
    struct sym { int32_t id; int prev, next; };
    std::vector<sym> syms;
    syms.reserve(norm.size() + 1);
    bool prev_unk = false;
    for (size_t i = 0; i < norm.size();) {
        const size_t n = std::min(utf8_len((uint8_t) norm[i]), norm.size() - i);
        const auto it = m_vocab.find(norm.substr(i, n));
        if (it != m_vocab.end()) {
            syms.push_back({it->second, 0, 0});
            prev_unk = false;
        } else {
            bool bytes_ok = true;
            for (size_t b = i; b < i + n; ++b) {
                if (m_byte_id[(uint8_t) norm[b]] < 0) { bytes_ok = false; break; }
            }
            if (bytes_ok) {
                for (size_t b = i; b < i + n; ++b) {
                    syms.push_back({m_byte_id[(uint8_t) norm[b]], 0, 0});
                }
                prev_unk = false;
            } else if (!prev_unk) {
                syms.push_back({m_unk_id, 0, 0});
                prev_unk = true;
            }
        }
        i += n;
    }
    for (int i = 0; i < (int) syms.size(); ++i) {
        syms[i].prev = i - 1;
        syms[i].next = i + 1 < (int) syms.size() ? i + 1 : -1;
    }

    // rank-ordered merges; ties break on leftmost pair (HF heap order)
    struct cand { uint32_t rank; int left; int32_t merged; };
    auto worse = [](const cand & a, const cand & b) {
        return a.rank != b.rank ? a.rank > b.rank : a.left > b.left;
    };
    std::priority_queue<cand, std::vector<cand>, decltype(worse)> pq(worse);
    auto pair_key = [&](int left, int right) {
        return ((uint64_t) (uint32_t) syms[left].id << 32) | (uint32_t) syms[right].id;
    };
    auto push = [&](int left) {
        const int right = syms[left].next;
        if (right < 0) return;
        const auto it = m_merges.find(pair_key(left, right));
        if (it != m_merges.end()) {
            pq.push({(uint32_t) (it->second >> 32), left, (int32_t) (uint32_t) it->second});
        }
    };
    for (int i = 0; i + 1 < (int) syms.size(); ++i) push(i);
    while (!pq.empty()) {
        const cand c = pq.top();
        pq.pop();
        const int right = syms[c.left].next;
        if (syms[c.left].id < 0 || right < 0) continue;
        // staleness check as HF does it: the CURRENT pair must still merge
        // into the queued output id (not necessarily via the queued pair)
        const auto it = m_merges.find(pair_key(c.left, right));
        if (it == m_merges.end() || (int32_t) (uint32_t) it->second != c.merged) continue;
        syms[c.left].id   = c.merged;
        syms[c.left].next = syms[right].next;
        if (syms[right].next >= 0) syms[syms[right].next].prev = c.left;
        syms[right].id = -1;
        if (syms[c.left].prev >= 0) push(syms[c.left].prev);
        push(c.left);
    }

    for (int i = 0; i >= 0 && i < (int) syms.size(); i = syms[i].next) {
        out.push_back(syms[i].id);
    }
    return out;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
