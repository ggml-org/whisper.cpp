#pragma once
// Per-codebook sampling for the Parler decoder: greedy argmax, or the HF
// warper chain temperature -> top-k -> top-p, then softmax + multinomial.

#include <cstdint>
#include <random>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

struct parler_sampling_params {
    bool  greedy      = false;
    float temperature = 1.0f;
    int   top_k       = 50;
    float top_p       = 1.0f;
};

// logits: [n_codebooks, vocab] row-major (already logits-processed);
// returns one token id per codebook.
std::vector<int32_t> parler_sample_frame(const float * logits, int n_codebooks, int vocab,
                                         const parler_sampling_params & params,
                                         std::mt19937 & rng);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
