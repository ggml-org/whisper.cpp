#pragma once

#include <cstdint>
#include <vector>

namespace parakeet {

// Result of matching the trailing two dims of a Core ML tensor shape against an
// unordered (rows, cols) pair. `transpose` is false for a [.., rows, cols] layout
// and true for [.., cols, rows]; `matched` is false when neither order fits.
struct CoremlTrailingMatch {
    bool matched;
    bool transpose;
};

CoremlTrailingMatch coreml_match_trailing_dims(const std::vector<int64_t> & dims,
                                               int64_t rows, int64_t cols);

// Concrete Core ML input dims to allocate, plus whether the row-major
// (n_mel_frames, n_mels) mel must be transposed to features-major on copy.
struct CoremlInputDims {
    std::vector<int64_t> dims;
    bool                 transpose;
};

// Resolves the input shape the wrapper feeds a Core ML encoder from the model's
// declared input dims (empty or non-concrete when unknown/flexible) and the target
// (n_mel_frames, n_mels). A concrete model already sized to this input is honoured
// verbatim; otherwise the shape is rebuilt at the requested mel length in the
// model's declared orientation so a flexible export serves variable-length
// utterances; with no usable concrete shape it defaults to the NeMo-natural
// features-major [1, n_mels, n_mel_frames].
CoremlInputDims coreml_resolve_input_dims(const std::vector<int64_t> & declared,
                                          int64_t n_mel_frames, int64_t n_mels);

}  // namespace parakeet
