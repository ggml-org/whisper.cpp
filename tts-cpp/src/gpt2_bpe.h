#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct gpt2_bpe {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string>                 id_to_token;
    std::unordered_map<std::string, int>     bpe_ranks;

    // Populate from arrays read out of GGUF metadata
    // (tokens indexed by id; merges are lines of form "left right").
    // Returns false if `tokens` is empty.
    bool load_from_arrays(const std::vector<std::string> & tokens,
                          const std::vector<std::string> & merges);

    std::vector<int32_t> tokenize(const std::string & text) const;

    static std::string punc_norm(const std::string & text);
};
