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

    // Two-for-one real FFT: pack the two real signals as (enhanced + i*original), transform
    // once, and recover the individual spectra from the combined transform's Hermitian symmetry.
    ComplexVec comb(n_pow2, {0.0f, 0.0f});
    for (int i = 0; i < N; i++) {
        comb[i] = {enhanced[i], original[i]};
    }
    StftProcessor::fft(comb, false);

    auto spec1_at = [&](int k) {           // FFT(enhanced)[k]
        const int km = (n_pow2 - k) % n_pow2;
        return (comb[k] + std::conj(comb[km])) * 0.5f;
    };
    auto spec2_at = [&](int k) {           // FFT(original)[k]
        const int km = (n_pow2 - k) % n_pow2;
        return (comb[k] - std::conj(comb[km])) * std::complex<float>(0.0f, -0.5f);
    };

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
    ComplexVec blended(n_pow2, {0.0f, 0.0f});
    for (int i = 0; i < n_bins; i++) {
        const std::complex<float> s1 = spec1_at(i);
        const std::complex<float> s2 = spec2_at(i);
        blended[i] = s2 + (s1 - s2) * mask[i];
        if (i > 0 && i < n_pow2 / 2) {
            blended[n_pow2 - i] = std::conj(blended[i]);
        }
    }
    blended[n_pow2 / 2] = {blended[n_pow2 / 2].real(), 0.0f};

    StftProcessor::fft(blended, true);

    std::vector<float> out(N);
    for (int i = 0; i < N; i++) {
        out[i] = blended[i].real();
    }
    return out;
}

} // namespace tts_cpp::lavasr::dsp
