#pragma once

#include <cstdint>

// Leading decode-logits rows to drop before the vocab matmul: >0 only when the requested-logits
// flags form a trailing suffix (readback then maps src_row = i - offset); 0 otherwise, e.g. 1,0,1.
static inline int whisper_logits_suffix_offset(const int8_t * logits, int n_tokens) {
    int i0 = 0;
    while (i0 < n_tokens && logits[i0] == 0) {
        i0++;
    }
    bool suffix = i0 < n_tokens;
    for (int i = i0; i < n_tokens; ++i) {
        suffix = suffix && logits[i] != 0;
    }
    return (suffix && i0 > 0) ? i0 : 0;
}
