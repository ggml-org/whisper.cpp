#include "fastlr_merge.h"

#include "dsp_constants.h"
#include "stft_processor.h"

#include <algorithm>
#include <cmath>
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

    // Blend: original low-freq + enhanced high-freq.  Hermitian-symmetric (real
    // signal) so only bins 0..N/2 are kept; DC and Nyquist forced real explicitly.
    const int n_half = n_pow2 / 2;
    ComplexVec blended(n_bins);
    for (int i = 0; i < n_bins; i++) {
        const std::complex<float> s1 = spec1_at(i);
        const std::complex<float> s2 = spec2_at(i);
        blended[i] = s2 + (s1 - s2) * mask[i];
    }
    blended[0]      = {blended[0].real(), 0.0f};
    blended[n_half] = {blended[n_half].real(), 0.0f};
    if (n_pow2 == 1) {
        return {blended[0].real()};  // size-1 iFFT is the identity
    }

    // Inverse two-for-one: Hermitian length-N B inverts via one length-N/2 iFFT.
    // E=(B[k]+conj(B[N/2-k]))/2, O=(B[k]-conj(B[N/2-k]))/2*W^k (W=e^{2pi i/N}); z=iFFT(E+iO), out[2m]=Re z[m], out[2m+1]=Im z[m].
    ComplexVec packed(n_half);
    const float step_re = std::cos(2.0f * static_cast<float>(PI) / n_pow2);
    const float step_im = std::sin(2.0f * static_cast<float>(PI) / n_pow2);
    // Fixed-size chunks, each restarting its twiddle recurrence from an exactly
    // computed W^c: element values are independent of thread count/schedule.
    constexpr int chunk = 512;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < n_half; c += chunk) {
        const double ang = 2.0 * PI * (static_cast<double>(c) / n_pow2);
        float w_re = static_cast<float>(std::cos(ang));
        float w_im = static_cast<float>(std::sin(ang));
        const int c_end = std::min(c + chunk, n_half);
        for (int k = c; k < c_end; k++) {
            const std::complex<float> bk = blended[k];
            const std::complex<float> bm = blended[n_half - k];
            const float e_re = 0.5f * (bk.real() + bm.real()); // E = (bk + conj(bm))/2
            const float e_im = 0.5f * (bk.imag() - bm.imag());
            const float d_re = 0.5f * (bk.real() - bm.real()); // D = (bk - conj(bm))/2
            const float d_im = 0.5f * (bk.imag() + bm.imag());
            const float o_re = d_re * w_re - d_im * w_im;      // O = D * w
            const float o_im = d_re * w_im + d_im * w_re;
            packed[k] = {e_re - o_im, e_im + o_re};            // E + i*O
            const float nw_re = w_re * step_re - w_im * step_im; // w *= step
            w_im              = w_re * step_im + w_im * step_re;
            w_re              = nw_re;
        }
    }
    StftProcessor::fft(packed, true);

    std::vector<float> out(N);
    for (int i = 0; i < N; i++) {
        const std::complex<float> & z = packed[i >> 1];
        out[i] = (i & 1) ? z.imag() : z.real();
    }
    return out;
}

} // namespace tts_cpp::lavasr::dsp
