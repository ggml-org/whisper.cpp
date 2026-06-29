#pragma once

#include <vector>

namespace tts_cpp::lavasr::dsp {

// FastLR spectral crossover merge.  Combines the low frequencies of the
// `original` (engine-rate, upsampled) signal with the high frequencies of
// the neurally `enhanced` signal, using a cubic-Hermite smoothstep mask
// across a transition band centred on `cutoff_hz`.  Both inputs must share
// the same length and sample rate.
class FastLRMerge {
public:
    static std::vector<float> merge(const std::vector<float> & enhanced,
                                    const std::vector<float> & original,
                                    int sample_rate     = 48000,
                                    int cutoff_hz       = 4000,
                                    int transition_bins = 256);
};

} // namespace tts_cpp::lavasr::dsp
