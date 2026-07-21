#pragma once

// LavaSR denoiser — scalar CPU forward (UL-UNAS).
//
// Mirrors enhancer_core.h: a pure-C++ scalar forward (no ggml compute; ggml is
// used only to parse the GGUF in denoiser_gguf).  This is the network part of
// the pipeline — the STFT/ISTFT + chunked overlap-add + resampling live in
// denoiser.cpp, exactly as enhancer.cpp wraps enhancer_core.cpp.
//
// Architecture — UL-UNAS (Ultra-Lightweight U-Net via Network Architecture
// Search, arXiv:2503.00340, github.com/Xiaobin-Rong/ul-unas), the GTCRN-family
// denoiser used by LavaSR.  The default `ULUNAS()` config (matching the
// Topping1/LavaSRcpp `denoiser_core_legacy_fixed63.onnx` weights) is:
//   * input:   STFT real/imag [2,T,257] @ 16 kHz -> log-magnitude feature [1,T,257]
//   * ERB band-merge:  257 -> 129 (keep 65 linear bins + erb_fc(192->64))
//   * encoder: 5 blocks (XConv / XMB / XDWS), freq 129 ->(s2) 65 ->(s2) 33,
//              channels [12,24,24,32,16]; each block =
//                grouped 1x1 pconv + depthwise conv + BatchNorm + affine-PReLU
//                + cTFA (causal time-freq attention) + channel shuffle (g==2)
//   * bottleneck: 2x DPGRNN (dual-path grouped RNN: intra BiGRU over freq +
//              inter GRU over time, + FC + LayerNorm + residual)
//   * decoder: 5 blocks mirroring the encoder (ConvTranspose upsampling) with
//              additive U-Net skips, then sigmoid -> real ratio mask [1,T,129]
//   * ERB band-split: 129 -> 257, then spec_enh = spec * mask (both re/im)
//
// cTFA per block:
//   TA: mean over freq -> GRU(C->2C, causal) -> FC(2C->C) -> Sigmoid  (temporal)
//   FA: mean over chan -> fold freq by R=4 -> BiGRU(4->4) -> FC(8->4) -> Sigmoid
// Causal along time -> streaming-friendly (each fixed chunk runs zero-state).
//
// Weights are orientation-matched to the GGUF from
// scripts/convert-lavasr-denoiser-to-gguf.py (ONNX initializers, `model.`
// prefix stripped):
//   conv kernels:  [out, in/groups, kt, kf]   (ONNX/PyTorch order)
//   linear/FC W:   [out, in]                   (y[o] = sum_i W[o,i] x[i] + b[o])
//   GRU gates:     weight_ih/hh_l0 [3H,*] (+ _reverse for BiGRU), order r,z,n

#include <map>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

struct DnTensor {
    std::vector<float> data;
    std::vector<int>   shape; // C/numpy order
};

struct DenoiserWeights {
    // STFT / rate params (read from GGUF metadata; defaults = the shipped model).
    int   n_fft            = 512;
    int   hop              = 256;
    int   win              = 512;
    int   spec_bins        = 257;
    int   work_sample_rate = 16000; // rate the UL-UNAS net operates at
    // ERB band-merge split: keep `erb_low` linear bins, compress the rest to
    // `erb_high` bands (erb_low + erb_high == encoder input width, 129).
    int   erb_low          = 65;
    int   erb_high         = 64;
    // cTFA frequency-folding group size R (FA r=4 in the UL-UNAS paper).
    int   freq_comp_ratio  = 4;
    // Chunked inference (fixed-length ONNX export): 63-frame windows, hop 21.
    int   chunk_frames     = 63;
    int   chunk_hop        = 21;
    float bn_eps           = 1e-5f;
    float ln_eps           = 1e-8f;

    std::map<std::string, DnTensor> t;

    bool             has(const std::string & name) const { return t.count(name) != 0; }
    const DnTensor & get(const std::string & name) const;
};

// UL-UNAS network forward on ONE zero-state chunk of `T` frames.  The complex
// spectrogram is passed as real/imag planes, each [T * spec_bins] row-major
// (index t*spec_bins + f).  Writes the enhanced (masked) real/imag planes in the
// same layout.  This is the direct C++ analogue of the ONNX graph
// (spec_ri -> spec_enh_ri) and is what the chunked pipeline in denoiser.cpp
// drives.  Throws std::runtime_error if a required weight tensor is missing.
void denoiser_net_forward(const DenoiserWeights & w,
                          const std::vector<float> & real_in,
                          const std::vector<float> & imag_in, int T,
                          std::vector<float> & real_out,
                          std::vector<float> & imag_out);

} // namespace tts_cpp::lavasr
