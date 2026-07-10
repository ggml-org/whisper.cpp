#include "mel_filterbank.h"

#include <algorithm>
#include <cmath>

namespace tts_cpp::lavasr::dsp {

float MelFilterbank::hz_to_mel_slaney(float f) {
    if (f >= 1000.0f) {
        return 15.0f + std::log(f / 1000.0f) / (std::log(6.4f) / 27.0f);
    }
    return f / (200.0f / 3.0f);
}

float MelFilterbank::mel_to_hz_slaney(float m) {
    if (m >= 15.0f) {
        return 1000.0f * std::exp((std::log(6.4f) / 27.0f) * (m - 15.0f));
    }
    return (200.0f / 3.0f) * m;
}

MelFilterbank::MelFilterbank(int sample_rate, int n_fft, int n_mels, float f_min,
                             float f_max)
    : sample_rate_(sample_rate), n_fft_(n_fft), n_mels_(n_mels), f_min_(f_min),
      f_max_(f_max) {
    const int n_freqs = n_fft / 2 + 1;

    std::vector<float> fftfreqs(n_freqs);
    for (int i = 0; i < n_freqs; i++) {
        fftfreqs[i] = static_cast<float>(i) * sample_rate / n_fft;
    }

    const float m_min = hz_to_mel_slaney(f_min);
    const float m_max = hz_to_mel_slaney(f_max);
    std::vector<float> f_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        f_pts[i] = mel_to_hz_slaney(m_min + i * (m_max - m_min) / (n_mels + 1));
    }

    filters_.resize(n_mels, std::vector<float>(n_freqs, 0.0f));
    for (int i = 0; i < n_mels; i++) {
        const float fdiff_left  = std::max(f_pts[i + 1] - f_pts[i], 1e-12f);
        const float fdiff_right = std::max(f_pts[i + 2] - f_pts[i + 1], 1e-12f);
        const float enorm       = 2.0f / std::max(f_pts[i + 2] - f_pts[i], 1e-12f);

        for (int j = 0; j < n_freqs; j++) {
            const float lower = (fftfreqs[j] - f_pts[i]) / fdiff_left;
            const float upper = (f_pts[i + 2] - fftfreqs[j]) / fdiff_right;
            filters_[i][j]    = std::max(0.0f, std::min(lower, upper)) * enorm;
        }
    }
}

std::vector<std::vector<float>>
MelFilterbank::mel_spectrogram(const std::vector<float> & wav,
                               int hop_length) const {
    if (!cached_stft_ || cached_hop_length_ != hop_length) {
        cached_stft_ =
            std::make_unique<StftProcessor>(n_fft_, hop_length, n_fft_, false);
        cached_hop_length_ = hop_length;
    }
    const Spectrogram spec = cached_stft_->stft(wav);

    const int T       = static_cast<int>(spec.size());
    const int n_freqs = n_fft_ / 2 + 1;
    std::vector<std::vector<float>> mel(n_mels_, std::vector<float>(T, 0.0f));

    // Magnitude spectrogram (not power) — matches Vocos/LavaSR.  Compute |spec| once per (t,f)
    // into a scratch row, then apply the mel filters (avoids the redundant per-filter sqrt).
    std::vector<float> mag(n_freqs);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_freqs; f++) {
            mag[f] = std::abs(spec[t][f]);
        }
        for (int m = 0; m < n_mels_; m++) {
            float sum = 0.0f;
            for (int f = 0; f < n_freqs; f++) {
                sum += filters_[m][f] * mag[f];
            }
            mel[m][t] = std::log(std::max(sum, 1e-5f));
        }
    }

    return mel;
}

} // namespace tts_cpp::lavasr::dsp
