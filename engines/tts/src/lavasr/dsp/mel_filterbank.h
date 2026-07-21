#pragma once

#include "stft_processor.h"

#include <memory>
#include <vector>

namespace tts_cpp::lavasr::dsp {

// Slaney-scale mel filterbank producing a log-magnitude mel spectrogram.
// The enhancer feeds an 80-bin log-mel (n_fft=2048, hop=512, fmin=0,
// fmax=8000, Slaney mel computed at a 44100 Hz reference rate) into the
// ConvNeXt backbone, matching upstream LavaSR / Vocos training.
class MelFilterbank {
public:
    MelFilterbank(int sample_rate = 44100, int n_fft = 2048, int n_mels = 80,
                  float f_min = 0.0f, float f_max = 8000.0f);

    // Returns an [n_mels][T] log-mel spectrogram from raw mono audio.
    std::vector<std::vector<float>> mel_spectrogram(const std::vector<float> & wav,
                                                    int hop_length) const;

    int n_mels() const { return n_mels_; }
    int n_fft() const { return n_fft_; }

private:
    static float hz_to_mel_slaney(float f);
    static float mel_to_hz_slaney(float m);

    int   sample_rate_;
    int   n_fft_;
    int   n_mels_;
    float f_min_;
    float f_max_;
    // [n_mels][n_freqs] triangular filter matrix, precomputed at construction.
    std::vector<std::vector<float>> filters_;

    mutable std::unique_ptr<StftProcessor> cached_stft_;
    mutable int                            cached_hop_length_ = 0;
};

} // namespace tts_cpp::lavasr::dsp
