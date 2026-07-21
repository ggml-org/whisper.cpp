#include "resampler.h"

#include "dsp_constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace tts_cpp::lavasr::dsp {

namespace {
constexpr int LANCZOS_A = 5;

float lanczos_weight(double x) {
    if (x == 0.0) return 1.0f;
    const double pi_x = PI * x;
    return static_cast<float>(std::sin(pi_x) * std::sin(pi_x / LANCZOS_A) /
                              (pi_x * pi_x / LANCZOS_A));
}
} // namespace

std::vector<float> Resampler::resample(const std::vector<float> & input,
                                       int sr_in, int sr_out) {
    if (sr_in == sr_out || input.empty()) {
        return input;
    }

    const double  ratio   = static_cast<double>(sr_out) / sr_in;
    const int64_t out_len  = static_cast<int64_t>(std::round(input.size() * ratio));
    const int64_t in_size  = static_cast<int64_t>(input.size());
    std::vector<float> output(static_cast<size_t>(out_len), 0.0f);
    const double  scale   = std::min(1.0, ratio);

    // Polyphase Lanczos: center = i*p/q is periodic in i with period q, so the per-tap weights
    // take only q distinct values — precompute them once instead of recomputing sin() per tap.
    const int p = sr_in / std::gcd(sr_in, sr_out);
    const int q = sr_out / std::gcd(sr_in, sr_out);
    struct Phase { int dl, dr; std::vector<float> w; float wsum; };
    std::vector<Phase> ph(q);
    for (int k = 0; k < q; k++) {
        const double frac = static_cast<double>(k) / q;
        Phase & P = ph[k];
        P.dl   = static_cast<int>(std::floor(frac - LANCZOS_A / scale));
        P.dr   = static_cast<int>(std::floor(frac + LANCZOS_A / scale));
        P.wsum = 0.0f;
        P.w.resize(static_cast<size_t>(P.dr - P.dl + 1));
        for (int d = P.dl; d <= P.dr; d++) {
            const float w = lanczos_weight((frac - d) * scale);
            P.w[static_cast<size_t>(d - P.dl)] = w;
            P.wsum += w;
        }
    }

    for (int64_t i = 0; i < out_len; i++) {
        const int64_t ip = i * p;
        const Phase & P  = ph[static_cast<size_t>(ip % q)];
        const int64_t ci = ip / q;                          // floor(center)
        float sum = 0.0f;
        if (ci + P.dl >= 0 && ci + P.dr < in_size) {        // interior: full kernel
            for (int d = P.dl; d <= P.dr; d++) {
                sum += input[static_cast<size_t>(ci + d)] * P.w[static_cast<size_t>(d - P.dl)];
            }
            output[static_cast<size_t>(i)] = sum / P.wsum;
        } else {                                            // edge: clip + renormalise
            float wsum = 0.0f;
            for (int d = P.dl; d <= P.dr; d++) {
                const int64_t j = ci + d;
                if (j < 0 || j >= in_size) continue;
                const float w = P.w[static_cast<size_t>(d - P.dl)];
                sum += input[static_cast<size_t>(j)] * w;
                wsum += w;
            }
            output[static_cast<size_t>(i)] = (wsum > 0.0f) ? sum / wsum : 0.0f;
        }
    }

    return output;
}

} // namespace tts_cpp::lavasr::dsp
