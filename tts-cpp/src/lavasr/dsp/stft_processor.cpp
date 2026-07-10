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

    // Butterflies on raw float storage: std::complex operator* carries C99 inf/nan-recovery
    // overhead that never triggers for finite FFT data, so raw arithmetic is bit-identical, faster.
    float * f = reinterpret_cast<float *>(x.data()); // [re0,im0,re1,im1,...]
    for (int len = 2; len <= N; len <<= 1) {
        const float angle =
            2.0f * static_cast<float>(PI) / len * (inverse ? 1.0f : -1.0f);
        const float wlen_re = std::cos(angle);
        const float wlen_im = std::sin(angle);
        for (int i = 0; i < N; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const int a = 2 * (i + j);
                const int b = 2 * (i + j + len / 2);
                const float u_re = f[a],     u_im = f[a + 1];
                const float x_re = f[b],     x_im = f[b + 1];
                const float v_re = x_re * w_re - x_im * w_im; // v = x * w
                const float v_im = x_re * w_im + x_im * w_re;
                f[a]     = u_re + v_re; f[a + 1] = u_im + v_im;
                f[b]     = u_re - v_re; f[b + 1] = u_im - v_im;
                const float nw_re = w_re * wlen_re - w_im * wlen_im; // w *= wlen
                w_im              = w_re * wlen_im + w_im * wlen_re;
                w_re              = nw_re;
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

    // Two-for-one: pack frame t (real) + frame t+1 (imag) into one complex FFT
    // and recover both spectra from the Hermitian symmetry of the transform.
    int t = 0;
    for (; t + 1 < num_frames; t += 2) {
        std::fill(frame.begin(), frame.end(), std::complex<float>{0.0f, 0.0f});
        for (int i = 0; i < win_length_; i++) {
            frame[i] = {xpad[t * hop_length_ + i] * window_[i],
                        xpad[(t + 1) * hop_length_ + i] * window_[i]};
        }
        fft(frame, false);
        for (int f = 0; f < freq_bins; f++) {
            const int fm = (n_fft_ - f) % n_fft_;
            const std::complex<float> cf = frame[f];
            const std::complex<float> cm = std::conj(frame[fm]);
            spec[t][f]     = (cf + cm) * 0.5f;                                  // FFT(frame t)
            spec[t + 1][f] = (cf - cm) * std::complex<float>(0.0f, -0.5f);      // FFT(frame t+1)
        }
    }
    if (t < num_frames) {  // trailing odd frame
        std::fill(frame.begin(), frame.end(), std::complex<float>{0.0f, 0.0f});
        for (int i = 0; i < win_length_; i++) {
            frame[i] = {xpad[t * hop_length_ + i] * window_[i], 0.0f};
        }
        fft(frame, false);
        for (int f = 0; f < freq_bins; f++) spec[t][f] = frame[f];
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

    // Overlap-add one real time-frame (real or imag part of `frame`) at index t.
    auto add_frame = [&](int t, bool use_imag) {
        for (int i = 0; i < win_length_; i++) {
            const float v = use_imag ? frame[i].imag() : frame[i].real();
            y[t * hop_length_ + i]    += v * window_[i];
            wenv[t * hop_length_ + i] += window_[i] * window_[i];
        }
    };

    // Two-for-one: IFFT(A + iB) = a + i·b for real signals a,b (Hermitian A,B),
    // so a frame pair is recovered from one inverse FFT (real→t, imag→t+1).
    int t = 0;
    for (; t + 1 < T; t += 2) {
        for (int k = 0; k < n_fft_; k++) {
            const int kk = (k <= n_fft_ / 2) ? k : n_fft_ - k;
            std::complex<float> a = spec[t][kk];
            std::complex<float> b = spec[t + 1][kk];
            if (k > n_fft_ / 2) { a = std::conj(a); b = std::conj(b); }
            // DC and Nyquist must be real for each packed spectrum to be Hermitian, else
            // frame t+1's DC/Nyquist imag leaks into frame t (per-frame istft drops it via .real()).
            if (k == 0 || k == n_fft_ / 2) {
                a = std::complex<float>(a.real(), 0.0f);
                b = std::complex<float>(b.real(), 0.0f);
            }
            frame[k] = a + std::complex<float>(0.0f, 1.0f) * b;
        }
        fft(frame, true);
        add_frame(t, false);
        add_frame(t + 1, true);
    }
    if (t < T) {  // trailing odd frame
        std::fill(frame.begin(), frame.end(), std::complex<float>{0.0f, 0.0f});
        for (int f = 0; f <= n_fft_ / 2; f++) {
            frame[f] = spec[t][f];
            if (f > 0 && f < n_fft_ / 2) frame[n_fft_ - f] = std::conj(spec[t][f]);
        }
        fft(frame, true);
        add_frame(t, false);
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
