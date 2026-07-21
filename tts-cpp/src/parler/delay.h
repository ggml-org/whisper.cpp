#pragma once
// Parler-TTS delay-pattern state machine (MusicGen-style), model-free.
// Mirrors HF parler_tts: build_delay_pattern_mask / apply_delay_pattern_mask,
// ParlerTTSLogitsProcessor, MinNewTokensLengthLogitsProcessor and the
// _sample stopping semantics, all in the delayed token domain.

#include <cstdint>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

struct delay_config {
    int n_codebooks    = 9;
    int bos_id         = 1025;
    int eos_id         = 1024;   // == pad in v1 checkpoints
    int pad_id         = 1024;
    int max_length     = 2580;   // delayed-domain length incl. the start frame
    int min_new_tokens = 10;     // 0 => no min-new-tokens EOS suppression
};

// mask[k*max_length + j]: -1 = model predicts here, else the forced token id.
// Generation case only (start ids = one column of BOS).
std::vector<int32_t> build_delay_mask(const delay_config & cfg);

class delay_state {
  public:
    explicit delay_state(const delay_config & cfg);

    // ids the model must be fed for the next step (delay mask applied to the
    // newest column). size n_codebooks. First call returns the BOS start frame.
    std::vector<int32_t> input_frame() const;

    // HF logits-processor chain, in place, on raw logits [n_codebooks, vocab]
    // (row-major). Call BEFORE sampling each step.
    void process_logits(float * logits, int vocab);

    // record the sampled frame (n_codebooks ids). Rows already finished are
    // forced to pad_id exactly like HF _sample.
    void append(const std::vector<int32_t> & frame);

    bool finished() const;
    int  cur_len() const { return m_len; }

    // Un-delay: frame t, codebook k = seq[k][t+k+1]; frames containing any
    // id >= codebook_size are dropped (HF decode_sequentially rule).
    // Returns [n_codebooks, n_frames] row-major.
    std::vector<int32_t> undelay(int codebook_size, int * n_frames_out) const;

    // introspection for tests
    int first_unfinished() const { return m_first_unfinished; }
    const std::vector<int32_t> & sequence() const { return m_seq; }

  private:
    delay_config m_cfg;
    std::vector<int32_t> m_mask;      // [n_cb, max_length]
    std::vector<int32_t> m_seq;       // accumulated sampled ids [n_cb, m_len]
    std::vector<uint8_t> m_done;      // per codebook row
    int m_len = 0;
    int m_first_unfinished = 0;
};

} // namespace detail
} // namespace parler
} // namespace tts_cpp
