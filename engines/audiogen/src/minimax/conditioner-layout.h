#pragma once

#include <algorithm>
#include <cstdint>

namespace tts_cpp::minimax::detail {

// The sidecar projects a three-tap, padding-one convolution. The first and last
// slots are halos, never emitted; padding is only applied at the real boundary.
inline void pack_conditioner_window(const float * mixed, int64_t frames, int64_t hidden,
                                    int64_t first, int64_t window, float * destination,
                                    int64_t channel_stride, int64_t time_stride) {
    for (int64_t c = 0; c < hidden; ++c) {
        for (int64_t t = 0; t < window; ++t) {
            const int64_t source = first + t - 1;
            destination[c * channel_stride + t * time_stride] =
                source >= 0 && source < frames ? mixed[source * hidden + c] : 0.0f;
        }
    }
}

inline void unpack_conditioner_window(const float * source, int64_t output, int64_t count,
                                      int64_t channel_stride, int64_t time_stride,
                                      float * destination) {
    for (int64_t t = 0; t < count; ++t) {
        for (int64_t c = 0; c < output; ++c) {
            destination[t * output + c] = source[c * channel_stride + (t + 1) * time_stride];
        }
    }
}

}
