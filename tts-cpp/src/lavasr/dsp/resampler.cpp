#include "resampler.h"

#include "dsp_constants.h"

#include <algorithm>
#include <cmath>

namespace tts_cpp::lavasr::dsp {

namespace {
constexpr int LANCZOS_A = 5;
} // namespace

std::vector<float> Resampler::resample(const std::vector<float> & input,
                                       int sr_in, int sr_out) {
    if (sr_in == sr_out || input.empty()) {
        return input;
    }

    const double ratio  = static_cast<double>(sr_out) / sr_in;
    const auto   out_len = static_cast<size_t>(std::round(input.size() * ratio));
    std::vector<float> output(out_len, 0.0f);
    const double scale = std::min(1.0, ratio);

    for (size_t i = 0; i < out_len; i++) {
        const double center = i / ratio;
        const auto   left   = static_cast<int>(
            std::max(0.0, std::floor(center - LANCZOS_A / scale)));
        const auto right = static_cast<int>(
            std::min(static_cast<double>(input.size()) - 1,
                     std::floor(center + LANCZOS_A / scale)));

        float sum       = 0.0f;
        float weight_sum = 0.0f;

        for (int j = left; j <= right; j++) {
            const double x      = (center - j) * scale;
            double       weight = 1.0;
            if (x != 0.0) {
                const double pi_x = PI * x;
                weight = std::sin(pi_x) * std::sin(pi_x / LANCZOS_A) /
                         (pi_x * pi_x / LANCZOS_A);
            }
            sum        += input[j] * static_cast<float>(weight);
            weight_sum += static_cast<float>(weight);
        }

        output[i] = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
    }

    return output;
}

} // namespace tts_cpp::lavasr::dsp
