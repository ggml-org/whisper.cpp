// Smoke-test for resample_sinc. Generates a broadband test signal (multi-tone)
// in memory, round-trips it through 24 kHz -> 48 kHz -> 24 kHz, and reports
// SNR in the middle of the buffer (well past the filter transient).

#include "voice_features.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    (void)argc; (void)argv;

    // 4 seconds of a multi-tone signal at 24 kHz.
    const int sr = 24000;
    const int N = 4 * sr;
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i) {
        double t = (double)i / sr;
        // Frequencies well below Nyquist so the resampler shouldn't have to
        // attenuate them.
        double s = 0.25 * std::sin(2 * M_PI *  220.0 * t)
                 + 0.25 * std::sin(2 * M_PI *  880.0 * t)
                 + 0.25 * std::sin(2 * M_PI * 2200.0 * t)
                 + 0.25 * std::sin(2 * M_PI * 4400.0 * t);
        in[i] = (float)s;
    }

    auto up   = resample_sinc(in, 24000, 48000);
    auto back = resample_sinc(up,  48000, 24000);
    printf("in:   samples=%zu sr=24000\n", in.size());
    printf("up:   samples=%zu sr=48000\n", up.size());
    printf("back: samples=%zu sr=24000 (expected ~%zu)\n", back.size(), in.size());

    // Compare middle region (skip the half-filter-length boundary).
    const size_t N_ = std::min(in.size(), back.size());
    const size_t skip = 64;
    float in_rms = 0, diff_rms = 0, diff_max = 0;
    for (size_t i = skip; i < N_ - skip; ++i) {
        in_rms   += in[i] * in[i];
        float d   = in[i] - back[i];
        diff_rms += d * d;
        diff_max  = std::max(diff_max, std::fabs(d));
    }
    size_t M = N_ - 2 * skip;
    in_rms   = std::sqrt(in_rms   / (float)M);
    diff_rms = std::sqrt(diff_rms / (float)M);
    double snr = 20.0 * std::log10(std::max((double)in_rms, 1e-12) /
                                    std::max((double)diff_rms, 1e-12));
    printf("\nround-trip 24k -> 48k -> 24k on a 4-tone test signal:\n"
           "  input RMS  = %.4e\n"
           "  diff RMS   = %.4e\n"
           "  diff max   = %.4e\n"
           "  SNR        = %.2f dB\n",
           in_rms, diff_rms, diff_max, snr);

    return snr >= 60.0 ? 0 : 1;
}
