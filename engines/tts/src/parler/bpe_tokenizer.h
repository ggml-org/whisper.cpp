#pragma once
// SentencePiece-BPE prompt tokenizer for indic-class Parler-TTS checkpoints.
// Mirrors the HF fast tokenizer: Metaspace(prepend first, no split),
// per-codepoint vocab lookup with byte fallback, merge-rank BPE, BOS prepend.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

class parler_bpe_tokenizer {
  public:
    // pieces: from GGUF parler.prompt_tokenizer.tokens (index = id); merges:
    // "left right" strings in rank order. Fails if a merge references a piece
    // (or its concatenation) missing from the vocab.
    bool load(const std::vector<std::string> & pieces,
              const std::vector<std::string> & merges,
              int unk_id, int bos_id, bool add_bos);

    // full pipeline: metaspace -> codepoint symbols (+byte fallback) ->
    // rank-ordered merges -> ids (+bos)
    std::vector<int32_t> encode(const std::string & text) const;

  private:
    std::unordered_map<std::string, int32_t> m_vocab;   // piece -> id
    // (left_id << 32 | right_id) -> (rank << 32 | merged_id)
    std::unordered_map<uint64_t, uint64_t> m_merges;
    int32_t m_byte_id[256];                             // <0xXX> ids, -1 if absent
    int32_t m_unk_id  = 0;
    int32_t m_bos_id  = 1;
    bool    m_add_bos = true;
};

} // namespace detail
} // namespace parler
} // namespace tts_cpp
