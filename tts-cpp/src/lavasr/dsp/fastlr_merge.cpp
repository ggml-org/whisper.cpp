#include "fastlr_merge.h"

#include "stft_processor.h"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>

namespace tts_cpp::lavasr::dsp {

std::vector<float> FastLRMerge::merge(const std::vector<float> & enhanced,
                                      const std::vector<float> & original,
                                      int sample_rate, int cutoff_hz,
                                      int transition_bins) {
    const int N = static_cast<int>(enhanced.size());
    const int M = static_cast<int>(original.size());
    if (N == 0) {
        return {};
    }
    if (M == 0) {
        return enhanced;
    }
    if (N != M) {
        throw std::invalid_argument("FastLRMerge: enhanced (" + std::to_string(N) +
                                    ") and original (" + std::to_string(M) +
                                    ") must have equal length");
    }

    int n_pow2 = 1;
    while (n_pow2 < std::max(N, M)) {
        n_pow2 <<= 1;
    }

    ComplexVec spec1(n_pow2, {0.0f, 0.0f});
    ComplexVec spec2(n_pow2, {0.0f, 0.0f});
    for (int i = 0; i < N; i++) {
        spec1[i] = {enhanced[i], 0.0f};
    }
    for (int i = 0; i < M; i++) {
        spec2[i] = {original[i], 0.0f};
    }

    StftProcessor::fft(spec1, false);
    StftProcessor::fft(spec2, false);

    const int n_bins = n_pow2 / 2 + 1;
    const int cutoff_bin =
        static_cast<int>(cutoff_hz / (sample_rate / 2.0f) * n_bins);
    const int half  = transition_bins / 2;
    const int start = std::max(0, cutoff_bin - half);
    const int end   = std::min(n_bins, cutoff_bin + half);

    // Crossover mask: 0 below cutoff (use original), 1 above (use enhanced);
    // cubic-Hermite smoothstep across the transition band.
    std::vector<float> mask(n_bins, 1.0f);
    for (int i = 0; i < start; i++) {
        mask[i] = 0.0f;
    }
    if (end - start > 1) {
        for (int i = start; i < end; i++) {
            const float x =
                -1.0f + 2.0f * (i - start) / static_cast<float>(end - start - 1);
            const float t = (x + 1.0f) / 2.0f;
            mask[i]       = 3.0f * t * t - 2.0f * t * t * t;
        }
    } else if (end == start + 1) {
        mask[start] = 0.5f;
    }

    // Blend spectra: original low-freq + enhanced high-freq, keeping the
    // result Hermitian-symmetric so the inverse FFT is real.
    for (int i = 0; i < n_bins; i++) {
        spec2[i] = spec2[i] + (spec1[i] - spec2[i]) * mask[i];
        if (i > 0 && i < n_pow2 / 2) {
            spec2[n_pow2 - i] = std::conj(spec2[i]);
        }
    }
    spec2[n_pow2 / 2] = {spec2[n_pow2 / 2].real(), 0.0f};

    StftProcessor::fft(spec2, true);

    std::vector<float> out(N);
    for (int i = 0; i < N; i++) {
        out[i] = spec2[i].real();
    }
    return out;
}

} // namespace tts_cpp::lavasr::dsp
