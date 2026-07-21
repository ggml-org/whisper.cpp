#pragma once
// SentencePiece-unigram tokenizer for the Parler-TTS T5 text encoder.
// Mirrors the HF fast tokenizer: Precompiled charsmap normalization,
// " {2,}" -> " " collapse, Metaspace(prepend always, split), unigram Viterbi.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

class parler_tokenizer {
  public:
    // pieces/scores: from GGUF tokenizer.ggml.tokens / .scores; pieces carry
    // the SentencePiece "\xe2\x96\x81" marker verbatim. charsmap: raw
    // precompiled_charsmap bytes (may be empty => no normalization).
    bool load(std::vector<std::string> pieces, std::vector<float> scores,
              std::vector<uint8_t> charsmap, int unk_id, int eos_id, bool add_eos);

    // full pipeline: normalize -> metaspace pretokenize -> unigram viterbi -> ids (+eos)
    std::vector<int32_t> encode(const std::string & text) const;

  private:
    struct trie_node {
        std::unordered_map<uint8_t, int32_t> next;  // child node index by byte
        int32_t token_id = -1;                      // piece ending here, or -1
    };
    struct norm_span {
        const char * data;
        size_t       len;       // bytes of normalized output
        size_t       consumed;  // input bytes consumed
    };

    uint32_t    xcda_node(size_t index) const;
    norm_span   normalize_prefix(const std::string & input, size_t offset) const;
    std::string normalize(const std::string & text) const;
    void        viterbi(const char * s, size_t n, std::vector<int32_t> & out) const;

    std::vector<trie_node> m_trie;    // piece matcher, root at index 0
    std::vector<float>     m_scores;

    std::vector<uint8_t> m_charsmap;  // raw blob; offsets below index into it
    size_t m_xcda_words  = 0;         // 32-bit XCDA entries starting at byte 4
    size_t m_repl_offset = 0;         // NUL-terminated replacement-string pool
    size_t m_repl_size   = 0;

    int32_t m_unk_id    = 2;
    int32_t m_eos_id    = 1;
    bool    m_add_eos   = true;
    double  m_unk_score = 0.0;        // min piece score - 10
};

} // namespace detail
} // namespace parler
} // namespace tts_cpp
