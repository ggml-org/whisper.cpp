#include "coreml/parakeet_coreml_shape.h"

#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace parakeet {
namespace {

bool shape_is_concrete(const std::vector<int64_t> & dims) {
    if (dims.size() < 2) {
        return false;
    }
    for (int64_t d : dims) {
        if (d <= 0) {
            return false;
        }
    }
    return true;
}

}  // namespace

CoremlTrailingMatch coreml_match_trailing_dims(const std::vector<int64_t> & dims,
                                               int64_t rows, int64_t cols) {
    const std::size_t n = dims.size();
    if (n < 2) {
        return {false, false};
    }
    const int64_t outer = dims[n - 2];
    const int64_t inner = dims[n - 1];
    if (outer == rows && inner == cols) {
        return {true, false};
    }
    if (outer == cols && inner == rows) {
        return {true, true};
    }
    return {false, false};
}

CoremlTrailingMatch coreml_match_trailing_capacity(
        const std::vector<int64_t> & dims,
        int64_t required_rows,
        int64_t cols) {
    const std::size_t n = dims.size();
    if (n < 2 || required_rows <= 0 || cols <= 0) {
        return {false, false};
    }

    const int64_t outer = dims[n - 2];
    const int64_t inner = dims[n - 1];
    if (inner == cols && outer >= required_rows) {
        return {true, false};
    }
    if (outer == cols && inner >= required_rows) {
        return {true, true};
    }

    return {false, false};
}

CoremlInputDims coreml_resolve_input_dims(const std::vector<int64_t> & declared,
                                          int64_t n_mel_frames, int64_t n_mels) {
    if (shape_is_concrete(declared)) {
        const CoremlTrailingMatch exact =
            coreml_match_trailing_dims(declared, n_mel_frames, n_mels);
        if (exact.matched) {
            return {declared, exact.transpose};
        }
        const std::size_t   n    = declared.size();
        std::vector<int64_t> dims = declared;
        if (declared[n - 1] == n_mels) {
            dims[n - 2] = n_mel_frames;
            dims[n - 1] = n_mels;
            return {dims, false};
        }
        if (declared[n - 2] == n_mels) {
            dims[n - 2] = n_mels;
            dims[n - 1] = n_mel_frames;
            return {dims, true};
        }
    }
    return {{1, n_mels, n_mel_frames}, true};
}

constexpr int kOfflineSubsamplingStages = 3;

int64_t coreml_offline_encoder_frames(int64_t mel_frames) {
    int64_t frames = mel_frames;
    for (int stage = 0; stage < kOfflineSubsamplingStages; ++stage) {
        frames = frames <= 0 ? 0 : (frames - 1) / 2 + 1;
    }
    return frames;
}

CoremlAneValidity coreml_ane_valid_lengths(int64_t n_mel_frames, int64_t n_mel_valid,
                                           int64_t n_enc_frames) {
    const bool whole_slab = n_mel_valid <= 0 || n_mel_valid >= n_mel_frames;
    const int64_t mel_frames = whole_slab ? n_mel_frames : n_mel_valid;
    const int64_t enc_frames = std::min(n_enc_frames, coreml_offline_encoder_frames(mel_frames));
    return CoremlAneValidity{mel_frames, enc_frames};
}

}  // namespace parakeet
