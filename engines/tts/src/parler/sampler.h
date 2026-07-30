#pragma once
// Per-codebook sampling for the Parler decoder: greedy argmax, or the HF
// warper chain temperature -> top-k -> top-p, then softmax + multinomial.

#include <cstdint>
#include <random>
#include <string>
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

// The model's own generation defaults, read from the GGUF's parler.gen.* keys.
struct parler_gen_defaults {
    bool  do_sample   = true;
    float temperature = 1.0f;
    int   top_k       = 50;
};

// What the caller asked for; 0 (or <= 0) defers to the model default.
struct parler_sampling_request {
    bool  greedy      = false;
    float temperature = 0.0f;
    int   top_k       = 0;
    float top_p       = 1.0f;
};

// Resolve a request against the model defaults.  Argmax decoding never
// terminates for this architecture, so a request that is statically argmax is
// repaired to sampling and the trigger is reported through `repaired` (cleared
// when nothing was changed).  Only the argmax-forcing knob is repaired.
parler_sampling_params parler_resolve_sampling(const parler_sampling_request & req,
                                               const parler_gen_defaults & def,
                                               std::string * repaired = nullptr);

// logits: [n_codebooks, vocab] row-major (already logits-processed);
// returns one token id per codebook.
std::vector<int32_t> parler_sample_frame(const float * logits, int n_codebooks, int vocab,
                                         const parler_sampling_params & params,
                                         std::mt19937 & rng);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
