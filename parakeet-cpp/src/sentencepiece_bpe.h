#pragma once

// SentencePiece-style BPE detokenizer: maps token IDs from GGUF vocab tables to UTF-8 text.
//
// Consumes piece strings and special IDs loaded from GGUF (llama.cpp-compatible tokenizer blocks).

#include <cstdint>
#include <string>
#include <vector>

namespace parakeet {

struct BpeVocab {
    std::vector<std::string> pieces;
    std::vector<float>        scores;
    std::vector<int8_t>       piece_types;

    int32_t blank_id = -1;
    int32_t unk_id   = -1;
    int32_t bos_id   = -1;
    int32_t eos_id   = -1;
    int32_t pad_id   = -1;
};

std::string detokenize(const BpeVocab & vocab,
                       const std::vector<int32_t> & token_ids);

// True when the token's piece (in `vocab.pieces`) begins with the
// SentencePiece word-boundary marker `▁` (U+2581, encoded as the 3-byte
// sequence 0xE2 0x96 0x81 in UTF-8). Used by the streaming sessions to
// stamp `StreamingSegment::starts_word` so consumers can distinguish a
// chunk-boundary wordpiece continuation ("ctuation" after "pun") from
// a fresh word ("if" after "see") without re-implementing the BPE
// detokenizer rules.
//
// Returns false for out-of-range, blank/bos/eos/pad ids, and pieces
// whose first byte does not start the U+2581 marker (e.g. punctuation
// pieces like ",", "." that should still be glued onto the previous
// word without an inserted space).
bool token_is_word_start(const BpeVocab & vocab, int32_t token_id);

}
