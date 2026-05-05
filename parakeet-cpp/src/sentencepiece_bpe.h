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

}
