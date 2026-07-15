#include "whisper-logits-slice.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Pins the decoder vocab-matmul row-slice contract (whisper.cpp): the readback maps
// src_row = i - offset, so the offset must be non-zero ONLY for a contiguous trailing suffix.
static void expect_offset(const std::vector<int8_t> & logits, int expected) {
    const int actual = whisper_logits_suffix_offset(logits.data(), (int) logits.size());
    if (actual != expected) {
        fprintf(stderr, "logits suffix offset: flags size %zu expected %d, got %d\n",
                logits.size(), expected, actual);
        std::abort();
    }
}

int main() {
    // prompt eval: only the last token requests logits -> drop the leading rows
    expect_offset({0, 0, 1}, 2);
    // trailing suffix of length 2
    expect_offset({0, 1, 1}, 1);
    // every token requested -> no leading zeros to drop, src_row = i
    expect_offset({1, 1, 1}, 0);
    // non-suffix 1,0,1 -> must NOT slice; readback falls back to src_row = i
    expect_offset({1, 0, 1}, 0);
    // non-suffix (a trailing zero) -> no slice
    expect_offset({1, 1, 0}, 0);
    // hole inside a longer run -> no slice
    expect_offset({0, 1, 0, 1}, 0);
    // no token requests logits -> no slice
    expect_offset({0, 0, 0}, 0);
    // single token
    expect_offset({1}, 0);

    return 0;
}
