#pragma once

#include <complex>
#include <vector>

namespace tts_cpp::lavasr::dsp {

using ComplexVec   = std::vector<std::complex<float>>;
using Spectrogram  = std::vector<std::vector<std::complex<float>>>; // [T][freq]

// Short-time Fourier transform with a periodic Hann window and reflect
// padding, plus a radix-2 in-place FFT.  Ported verbatim (algorithmically)
// from the @qvac/tts-onnx StftProcessor so the GGML enhancer produces
// bit-comparable spectrograms.
//
// Two configurations are used by LavaSR:
//   - enhancer ISTFT: n_fft=2048, hop=512, win=2048, center_pad=false
//   - (denoiser, future) STFT: n_fft=512, hop=256, win=512, center_pad=true
class StftProcessor {
public:
    StftProcessor(int n_fft, int hop_length, int win_length, bool center_pad);

    Spectrogram        stft(const std::vector<float> & signal) const;
    std::vector<float> istft(const Spectrogram & spec, int target_len = 0) const;

    // In-place radix-2 Cooley-Tukey FFT.  `inverse=true` divides by N.
    // Public + static because FastLRMerge reuses it directly.
    static void fft(ComplexVec & x, bool inverse);

    int n_fft() const { return n_fft_; }
    int hop_length() const { return hop_length_; }
    int win_length() const { return win_length_; }

private:
    static std::vector<float> hann_periodic(int length);
    static std::vector<float> pad_reflect(const std::vector<float> & x,
                                          int pad_left, int pad_right);

    int                n_fft_;
    int                hop_length_;
    int                win_length_;
    bool               center_pad_;
    std::vector<float> window_;
};

} // namespace tts_cpp::lavasr::dsp
