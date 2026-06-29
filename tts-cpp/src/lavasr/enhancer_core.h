#pragma once

// LavaSR enhancer — scalar CPU forward (no ggml dependency).
//
// Implements the Vocos bandwidth-extension network ported from the
// @qvac/tts-onnx enhancer (validated bit-comparable against onnxruntime):
//   backbone:  embed Conv1d(80->512,k7) -> LayerNorm
//              8x ConvNeXt block(dwconv k7 g512, LayerNorm, pwconv1 512->1536,
//                                erf-GELU, pwconv2 1536->512, *gamma, +residual)
//              final LayerNorm  -> hidden[T,512]
//   spec head: Linear(512->2050) -> split log-mag / phase ->
//              mag=clip(exp(.),max=clip_max); real=mag*cos(phase); imag=mag*sin(phase)
//
// The enhancer is tiny (512-dim, 8 blocks) so the scalar path is already very
// fast — this matches the "CPU only, lightweight" target for QVAC-16579 and
// mirrors the supertonic vector-estimator's scalar reference path.
//
// Weights are held orientation-matched to the GGUF produced by
// scripts/convert-lavasr-enhancer-to-gguf.py:
//   conv kernels:  [out, in/groups, K]   (numpy/C order)
//   linear (pw*):  [out, in]             (so y[o] = sum_i W[o,i] x[i] + b[o])

#include <map>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

struct EnhTensor {
    std::vector<float> data;
    std::vector<int>   shape; // C/numpy order
};

struct EnhancerWeights {
    int   dim       = 512;
    int   ffn_dim   = 1536;
    int   n_blocks  = 8;
    int   n_mels    = 80;
    int   kernel    = 7;
    int   n_fft     = 2048;
    int   hop       = 512;
    int   win       = 2048;
    int   spec_bins = 1025;
    // Working sample rate the network operates at (enhance() output rate) and
    // the Slaney-mel reference rate, both read from GGUF metadata so a future
    // converter change to these constants stays in sync with the C++.
    int   work_sample_rate    = 48000;
    int   mel_ref_sample_rate = 44100;
    float clip_max  = 1000.0f;
    float ln_eps    = 1e-6f;

    std::map<std::string, EnhTensor> t;

    bool                has(const std::string & name) const { return t.count(name) != 0; }
    const EnhTensor &   get(const std::string & name) const;
};

// Backbone + spec-head forward.
//   mel:  [n_mels * T] row-major, mel[c * T + t]   (log-mel, as produced by
//         MelFilterbank::mel_spectrogram flattened channel-major).
//   real, imag (out): [spec_bins * T] row-major, x[f * T + t].
void enhancer_spec_forward(const EnhancerWeights & w,
                           const std::vector<float> & mel, int T,
                           std::vector<float> & real, std::vector<float> & imag);

} // namespace tts_cpp::lavasr
