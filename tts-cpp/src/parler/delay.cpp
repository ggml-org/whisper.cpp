#include "delay.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace tts_cpp {
namespace parler {
namespace detail {

// HF build_delay_pattern_mask, generation case (start ids = one BOS column):
// row k, position j is a forced BOS iff j <= k (tril incl. diagonal), a forced
// PAD iff j >= max_length - n_codebooks + 1 + k (triu with diagonal
// max_length - n_codebooks + 1), else -1 (model prediction slot).
std::vector<int32_t> build_delay_mask(const delay_config & cfg) {
    const int n = cfg.n_codebooks;
    const int L = cfg.max_length;
    std::vector<int32_t> mask((size_t) n * L, -1);
    if (L < 2 * n - 1) {
        return mask; // HF returns the all -1 mask below this length
    }
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < L; ++j) {
            int32_t v = -1;
            if (j <= k) {
                v = cfg.bos_id;
            } else if (j >= L - n + 1 + k) {
                v = cfg.pad_id;
            }
            mask[(size_t) k * L + j] = v;
        }
    }
    return mask;
}

delay_state::delay_state(const delay_config & cfg)
    : m_cfg(cfg), m_mask(build_delay_mask(cfg)),
      m_done((size_t) cfg.n_codebooks, 0) {
    // start frame: one column of BOS (decoder_start_token_id == bos in v1)
    m_seq.assign((size_t) cfg.n_codebooks, cfg.bos_id);
    m_len = 1;
}

std::vector<int32_t> delay_state::input_frame() const {
    const int n = m_cfg.n_codebooks;
    std::vector<int32_t> frame((size_t) n);
    const int j = m_len - 1;
    for (int k = 0; k < n; ++k) {
        const int32_t forced = m_mask[(size_t) k * m_cfg.max_length + j];
        frame[k] = forced == -1 ? m_seq[(size_t) k * m_len + j] : forced;
    }
    return frame;
}

void delay_state::process_logits(float * logits, int vocab) {
    const int n = m_cfg.n_codebooks;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // MinNewTokensLengthLogitsProcessor: prompt_length_to_skip = 1 (the BOS
    // start frame); forbid EOS while generated length < min_new_tokens.
    if (m_cfg.min_new_tokens > 0 && m_len - 1 < m_cfg.min_new_tokens) {
        for (int k = 0; k < n; ++k) {
            logits[(size_t) k * vocab + m_cfg.eos_id] = neg_inf;
        }
    }

    // ParlerTTSLogitsProcessor: advance the pointer by at most one per call if
    // the tracked codebook row has an EOS anywhere in its history, then forbid
    // EOS on every codebook above the pointer.
    bool tracked_has_eos = false;
    for (int j = 0; j < m_len; ++j) {
        if (m_seq[(size_t) m_first_unfinished * m_len + j] == m_cfg.eos_id) {
            tracked_has_eos = true;
            break;
        }
    }
    if (tracked_has_eos && m_first_unfinished < n - 1) {
        m_first_unfinished++;
    }
    for (int k = m_first_unfinished + 1; k < n; ++k) {
        logits[(size_t) k * vocab + m_cfg.eos_id] = neg_inf;
    }
}

void delay_state::append(const std::vector<int32_t> & frame) {
    assert((int) frame.size() == m_cfg.n_codebooks);
    assert(m_len < m_cfg.max_length);
    const int n = m_cfg.n_codebooks;
    // re-layout [n, m_len] -> [n, m_len+1]
    std::vector<int32_t> next((size_t) n * (m_len + 1));
    for (int k = 0; k < n; ++k) {
        std::copy(m_seq.begin() + (size_t) k * m_len,
                  m_seq.begin() + (size_t) (k + 1) * m_len,
                  next.begin() + (size_t) k * (m_len + 1));
        // finished rows keep emitting pad, exactly like HF _sample
        int32_t tok = m_done[k] ? m_cfg.pad_id : frame[k];
        next[(size_t) k * (m_len + 1) + m_len] = tok;
        if (!m_done[k] && tok == m_cfg.eos_id) {
            m_done[k] = 1;
        }
    }
    m_seq.swap(next);
    m_len++;
}

bool delay_state::finished() const {
    if (m_len >= m_cfg.max_length) {
        return true;
    }
    for (uint8_t d : m_done) {
        if (!d) return false;
    }
    return true;
}

std::vector<int32_t> delay_state::undelay(int codebook_size, int * n_frames_out) const {
    const int n = m_cfg.n_codebooks;
    const int L = m_len;
    const int n_slots = L - n; // kept positions per row (HF keep-mask count)
    std::vector<int32_t> codes;
    int n_frames = 0;
    if (n_slots > 0) {
        std::vector<int32_t> frame((size_t) n);
        codes.reserve((size_t) n * n_slots);
        std::vector<std::vector<int32_t>> rows((size_t) n);
        for (int t = 0; t < n_slots; ++t) {
            bool ok = true;
            for (int k = 0; k < n; ++k) {
                frame[k] = m_seq[(size_t) k * L + (t + k + 1)];
                if (frame[k] < 0 || frame[k] >= codebook_size) {
                    ok = false;
                }
            }
            if (!ok) continue; // HF decode_sequentially drops such frames
            for (int k = 0; k < n; ++k) {
                rows[k].push_back(frame[k]);
            }
            n_frames++;
        }
        for (int k = 0; k < n; ++k) {
            codes.insert(codes.end(), rows[k].begin(), rows[k].end());
        }
    }
    if (n_frames_out) *n_frames_out = n_frames;
    return codes;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
