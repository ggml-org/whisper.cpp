#include "minimax/conditioner-layout.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Compare tiled, strided Core ML IO against an independent full convolution.
// Unequal channels and nonzero bias catch axis swaps and missing boundary taps.
int main() {
    constexpr int H = 3, D = 2, payload = 150, window = payload + 2;
    constexpr int ts = 2, cs = window * ts + 7;
    for (int frames : {1, 2, 149, 150, 151, 299, 300, 301}) {
        std::vector<float> input(frames * H), weights(D * H * 3), expected(frames * D), actual(frames * D);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin((float) i * 0.3f);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = (float) ((int) (i % 7) - 3) * 0.125f;
        for (int t = 0; t < frames; ++t) {
            for (int d = 0; d < D; ++d) {
                float y = 0.25f * (d + 1);
                for (int c = 0; c < H; ++c) {
                    for (int k = 0; k < 3; ++k) {
                        const int source = t + k - 1;
                        if (source >= 0 && source < frames) y += input[source * H + c] * weights[(d * H + c) * 3 + k];
                    }
                }
                expected[t * D + d] = y;
            }
        }
        std::vector<float> packed(H * cs, -999.0f), output(D * cs, -999.0f);
        for (int first = 0; first < frames; first += payload) {
            tts_cpp::minimax::detail::pack_conditioner_window(
                input.data(), frames, H, first, window, packed.data(), cs, ts);
            for (int d = 0; d < D; ++d) {
                for (int t = 0; t < window; ++t) {
                    float y = 0.25f * (d + 1);
                    for (int c = 0; c < H; ++c) {
                        for (int k = 0; k < 3; ++k) {
                            const int source = t + k - 1;
                            if (source >= 0 && source < window) y += packed[c * cs + source * ts] * weights[(d * H + c) * 3 + k];
                        }
                    }
                    output[d * cs + t * ts] = y;
                }
            }
            tts_cpp::minimax::detail::unpack_conditioner_window(
                output.data(), D, std::min(payload, frames - first), cs, ts, actual.data() + first * D);
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(expected[i] - actual[i]) > 1e-6f) {
                std::fprintf(stderr, "Convolution differs at frames=%d element=%zu: %g vs %g\n", frames, i, expected[i], actual[i]);
                return 1;
            }
        }
    }
    std::puts("Conditioner strided IO and convolution boundaries passed");
}
