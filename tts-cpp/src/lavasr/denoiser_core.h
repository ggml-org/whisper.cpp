#pragma once

// LavaSR denoiser — scalar CPU forward (UL-UNAS).  SCAFFOLD.
//
// Mirrors enhancer_core.h: a pure-C++ scalar forward (no ggml compute; ggml is
// used only to parse the GGUF in denoiser_gguf).  NOT yet implemented — this
// header pins down the weights container + entry point so the port can fill in
// denoiser_forward() without reshuffling the file/CMake structure.
//
// Architecture to port — UL-UNAS (Ultra-Lightweight U-Net via Network
// Architecture Search, arXiv:2503.00340), the denoiser used by LavaSR:
//   * input:   STFT of the (mono) signal — real/imag [2,F,T] (or magnitude)
//   * encoder: downsampling stack of efficient depthwise-separable conv blocks
//              with the affine-PReLU activation
//   * bottleneck + decoder: upsampling stack with U-Net skip connections
//   * cTFA (causal time-frequency attention) per block:
//       - TA branch: GRU  -> FC -> Sigmoid            (temporal; causal in time)
//       - FA branch: BiGRU over folded frequency (R=4) -> FC -> Sigmoid (spectral)
//   * output:  a (complex-ratio) mask applied to the input STFT -> ISTFT
//
// Causal along time -> streaming-friendly (carry GRU hidden state across chunks).
//
// Weights are held orientation-matched to the GGUF produced by
// scripts/convert-lavasr-denoiser-to-gguf.py:
//   conv kernels:  [out, in/groups, K]   (ONNX/numpy order)
//   linear/GRU W:  [out, in]             (so y[o] = sum_i W[o,i] x[i] + b[o])
// TODO(QVAC-16579 follow-up): finalise dims/tensor names against the reference
// ONNX once the NAS-selected architecture is mapped.

#include <map>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

struct DnTensor {
    std::vector<float> data;
    std::vector<int>   shape; // C/numpy order
};

struct DenoiserWeights {
    // STFT / rate params (read from GGUF metadata; placeholders until confirmed).
    int   n_fft            = 512;
    int   hop              = 256;
    int   win              = 512;
    int   spec_bins        = 257;
    int   work_sample_rate = 48000; // rate the UL-UNAS net operates at
    // cTFA frequency-folding group size R (fixed to 4 in the UL-UNAS paper).
    int   freq_fold_r      = 4;
    float ln_eps           = 1e-5f;

    std::map<std::string, DnTensor> t;

    bool             has(const std::string & name) const { return t.count(name) != 0; }
    const DnTensor & get(const std::string & name) const;
};

// UL-UNAS forward: mono float32 `pcm` at the network work rate -> cleaned pcm at
// the same rate.  SCAFFOLD: throws std::runtime_error until implemented.
std::vector<float> denoiser_forward(const DenoiserWeights & w,
                                    const std::vector<float> & pcm);

} // namespace tts_cpp::lavasr
