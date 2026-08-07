// Qwen2.5 byte-level BPE for Audio8, plus the ChatML prompt the processor
// builds around it.
//
// Everything the tokenizer needs travels in the GGUF, so nothing here reads
// the original tokenizer.json at run time.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp {
namespace audio8 {

struct TokenizerData {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> added_token_ids;
};

class Tokenizer {
public:
    explicit Tokenizer(const TokenizerData & data);

    std::vector<int32_t> encode(const std::string & text) const;
    int32_t token_id(const std::string & token) const;
    const std::string & token_text(int32_t id) const;

private:
    std::vector<std::string> texts_;
    std::unordered_map<std::string, int32_t> ids_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::vector<std::pair<std::string, int32_t>> added_;

    std::vector<std::string> merge(const std::string & piece) const;
    void encode_span(const std::string & span, std::vector<int32_t> & out) const;
    bool next_added_token(const std::string & text, size_t from, size_t & at,
                          size_t & length, int32_t & id) const;
};

// The prompt the reference processor produces. Text is whitespace-collapsed
// and each segment is encoded on its own, because BPE merges must not run
// across a segment boundary.
struct PromptSegments {
    std::vector<int32_t> prefix;
    std::vector<int32_t> suffix;
};

PromptSegments build_prompt(const Tokenizer & tokenizer, const std::string & text,
                            const std::string & reference_text, bool has_reference);

std::string collapse_whitespace(const std::string & text);

}  // namespace audio8
}  // namespace tts_cpp
