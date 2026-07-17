#pragma once

// ACE-Step BPE tokenizer (Qwen3/GPT-2 byte-level BPE) — QVAC-21921.
//
// Loads vocab + merges from the LM GGUF KV (tokenizer.ggml.tokens /
// tokenizer.ggml.merges), runs the GPT-2 byte-level pre-tokenizer + BPE merges.
// CPU-only, no ggml dependency. Ported from acestep.cpp/src/bpe.h (+ bpe_decode
// from sampling.h). Used by the LM pipeline to turn prompts/lyrics into token
// IDs and decoded text back out.

#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp::acestep {

// Qwen3 special token IDs (ACE-Step LM vocabulary).
constexpr int TOKEN_IM_START   = 151644;
constexpr int TOKEN_IM_END     = 151645;
constexpr int TOKEN_THINK      = 151667;
constexpr int TOKEN_THINK_END  = 151668;
constexpr int AUDIO_CODE_BASE  = 151669;
constexpr int AUDIO_CODE_COUNT = 65535;

struct BpeTokenizer {
    std::unordered_map<std::string, int> vocab;          // token_str -> id
    std::unordered_map<std::string, int> merges;         // "a b" -> rank
    std::string                          byte2str[256];  // byte -> GPT-2 UTF-8 string
    int                                  eos_id = 151643;
    int                                  n_vocab = 0;
    std::vector<std::string>             id_to_str;      // id -> token_str
};

// Load from the LM GGUF. Returns false if the tokenizer KV is missing.
bool bpe_load_from_gguf(BpeTokenizer & tok, const std::string & gguf_path);

// Encode text -> token IDs (byte-level BPE). add_eos appends <|endoftext|>.
std::vector<int> bpe_encode(const BpeTokenizer & tok, const std::string & text, bool add_eos = false);

// Decode token IDs -> text (skips audio-code + im_start/end, expands think tags).
std::string bpe_decode(const BpeTokenizer & tok, const std::vector<int> & ids);

// UTF-8 codepoint decode (advances by the byte width). Exposed for the metadata
// FSM's per-token text decoding.
int bpe_utf8_codepoint(const char * s, int * advance);

} // namespace tts_cpp::acestep
