#pragma once

// Pause-cut segmentation for independently decoded windows.
//
// The batched Core ML TDT decoder steps several windows per dispatch, which
// needs windows with no dependency between them: each segment is encoded on
// its own and decoded from a fresh prediction state. Cutting at the quietest
// nearby mel frame puts the state reset in a pause rather than mid-word.
//
// Pure, dependency-free helpers so they can be unit-tested without a model
// (test/test_independent_segments.cpp).

#include <algorithm>
#include <utility>
#include <vector>

namespace parakeet {

// Mean log-mel energy per mel frame, used as a pause detector. Works from the
// mel because the source audio may already have been released.
inline std::vector<float> mel_frame_energy(const float * mel, int n_mel_frames, int n_mels) {
    std::vector<float> energy((size_t) std::max(0, n_mel_frames), 0.0f);
    if (n_mels <= 0) return energy;
    for (int t = 0; t < n_mel_frames; ++t) {
        const float * row = mel + (size_t) t * (size_t) n_mels;
        double acc = 0.0;
        for (int m = 0; m < n_mels; ++m) acc += row[m];
        energy[(size_t) t] = (float) (acc / n_mels);
    }
    return energy;
}

// Mel-frame range [begin, end) encoded and decoded on its own.
struct SegmentRange {
    int begin = 0;
    int end   = 0;
};

// The quietest frame in [lo, hi]. The cap `hi` wins a tie, so a flat stretch
// keeps the full segment; among earlier frames a tie goes to the earliest.
inline int quietest_frame(const std::vector<float> & energy, int lo, int hi) {
    int   best        = hi;
    float best_energy = energy[(size_t) hi];
    for (int t = lo; t < hi; ++t) {
        if (energy[(size_t) t] < best_energy) {
            best_energy = energy[(size_t) t];
            best        = t;
        }
    }
    return best;
}

// Tiles [0, n_mel_frames) gap-free into segments of at most `segment_frames`
// mel frames. Each cut is the quietest frame within `search_frames` before the
// cap. A segment that starts mid-phrase can decode to nothing, so a wider
// search makes a real pause more likely at the cost of shorter segments and
// more encoder dispatches. Searching backwards only, never past the cap,
// keeps every segment inside a single encoder dispatch.
inline std::vector<SegmentRange>
plan_independent_segments(const std::vector<float> & energy, int n_mel_frames,
                          int segment_frames, int search_frames) {
    std::vector<SegmentRange> out;
    if (n_mel_frames <= 0 || segment_frames <= 0) return out;

    int begin = 0;
    while (begin < n_mel_frames) {
        const int end = begin + segment_frames;
        if (end >= n_mel_frames) {
            out.push_back({begin, n_mel_frames});
            break;
        }
        const int lo  = std::max(begin + 1, end - std::max(0, search_frames));
        const int cut = quietest_frame(energy, lo, end);
        out.push_back({begin, cut});
        begin = cut;
    }
    return out;
}

}  // namespace parakeet
