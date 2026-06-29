#include "stft_processor.h"

#include "dsp_constants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tts_cpp::lavasr::dsp {

StftProcessor::StftProcessor(int n_fft, int hop_length, int win_length,
                             bool center_pad)
    : n_fft_(n_fft), hop_length_(hop_length), win_length_(win_length),
      center_pad_(center_pad), window_(hann_periodic(win_length)) {}

void StftProcessor::fft(ComplexVec & x, bool inverse) {
    const int N = static_cast<int>(x.size());
    if (N <= 1) {
        return;
    }
    // Radix-2 only: a non-power-of-two N would silently corrupt the result.
    // n_fft/win are validated at GGUF load; this guards every other caller
    // (and FastLRMerge, which always pads to a power of two).
    if ((N & (N - 1)) != 0) {
        throw std::invalid_argument(
            "StftProcessor::fft: size must be a power of two (got " +
            std::to_string(N) + ")");
    }

    // Bit-reversal permutation.
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }

    for (int len = 2; len <= N; len <<= 1) {
        const float angle =
            2.0f * static_cast<float>(PI) / len * (inverse ? 1.0f : -1.0f);
        const std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < N; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; j++) {
                std::complex<float> u = x[i + j];
                std::complex<float> v = x[i + j + len / 2] * w;
                x[i + j]             = u + v;
                x[i + j + len / 2]   = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (int i = 0; i < N; i++) {
            x[i] /= static_cast<float>(N);
        }
    }
}

std::vector<float> StftProcessor::hann_periodic(int length) {
    std::vector<float> w(length);
    for (int i = 0; i < length; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(PI) * i / length));
    }
    return w;
}

std::vector<float> StftProcessor::pad_reflect(const std::vector<float> & x,
                                              int pad_left, int pad_right) {
    const int N = static_cast<int>(x.size());
    if (N <= 1) {
        return std::vector<float>(N + pad_left + pad_right, N == 1 ? x[0] : 0.0f);
    }
    std::vector<float> y(N + pad_left + pad_right);
    for (int i = -pad_left; i < N + pad_right; i++) {
        int idx = i;
        while (idx < 0 || idx >= N) {
            if (idx < 0) {
                idx = -idx;
            }
            if (idx >= N) {
                idx = 2 * N - 2 - idx;
            }
        }
        y[i + pad_left] = x[idx];
    }
    return y;
}

Spectrogram StftProcessor::stft(const std::vector<float> & signal) const {
    const int pad = center_pad_ ? (n_fft_ / 2) : ((win_length_ - hop_length_) / 2);
    std::vector<float> xpad = pad_reflect(signal, pad, pad);
    if (static_cast<int>(xpad.size()) < win_length_) {
        xpad.resize(win_length_, 0.0f);
    }

    const int num_frames =
        (static_cast<int>(xpad.size()) - win_length_) / hop_length_ + 1;
    const int freq_bins = n_fft_ / 2 + 1;

    Spectrogram spec(num_frames, std::vector<std::complex<float>>(freq_bins));
    ComplexVec  frame(n_fft_);

    for (int t = 0; t < num_frames; t++) {
        std::fill(frame.begin(), frame.end(), std::complex<float>{0.0f, 0.0f});
        for (int i = 0; i < win_length_; i++) {
            frame[i] = {xpad[t * hop_length_ + i] * window_[i], 0.0f};
        }
        fft(frame, false);
        for (int f = 0; f < freq_bins; f++) {
            spec[t][f] = frame[f];
        }
    }

    return spec;
}

std::vector<float> StftProcessor::istft(const Spectrogram & spec,
                                        int target_len) const {
    const int pad = center_pad_ ? (n_fft_ / 2) : ((win_length_ - hop_length_) / 2);
    const int T   = static_cast<int>(spec.size());
    const int output_size = (T - 1) * hop_length_ + win_length_;

    std::vector<float> y(output_size, 0.0f);
    std::vector<float> wenv(output_size, 0.0f);
    ComplexVec         frame(n_fft_);

    for (int t = 0; t < T; t++) {
        std::fill(frame.begin(), frame.end(), std::complex<float>{0.0f, 0.0f});
        for (int f = 0; f <= n_fft_ / 2; f++) {
            frame[f] = spec[t][f];
            if (f > 0 && f < n_fft_ / 2) {
                frame[n_fft_ - f] = std::conj(spec[t][f]);
            }
        }
        fft(frame, true);

        for (int i = 0; i < win_length_; i++) {
            y[t * hop_length_ + i]    += frame[i].real() * window_[i];
            wenv[t * hop_length_ + i] += window_[i] * window_[i];
        }
    }

    std::vector<float> out;
    out.reserve(output_size - 2 * pad);
    for (int i = pad; i < output_size - pad; i++) {
        out.push_back(y[i] / std::max(wenv[i], 1e-8f));
    }

    if (target_len > 0) {
        out.resize(target_len, 0.0f);
    }

    return out;
}

} // namespace tts_cpp::lavasr::dsp
