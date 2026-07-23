#include "coreml/parakeet_coreml_shape.h"

#include <cstddef>

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

}  // namespace parakeet
