#pragma once

// Public LavaSR denoiser API — follow-up (denoiser stage).
//
// Mirrors the shipped enhancer (tts-cpp/lavasr/enhancer.h): a pure-CPU scalar
// forward of the UL-UNAS neural denoiser, with ggml used only to parse the GGUF.
//
// LavaSR is a two-stage pipeline:
//   (1) UL-UNAS denoiser (this file) removes noise from the input, then
//   (2) the Vocos bandwidth-extension enhancer upsamples to 48 kHz.
//
// The network is UL-UNAS (arXiv:2503.00340, github.com/Xiaobin-Rong/ul-unas), a
// GTCRN-family TF-domain U-Net operating on 16 kHz STFT frames (ERB band-merge ->
// grouped depthwise-separable conv encoder with affine-PReLU + causal
// time-frequency attention -> 2x dual-path grouped RNN -> decoder -> ratio mask).
// The weights come from Topping1/LavaSRcpp (`denoiser_core_legacy_fixed63.onnx`)
// — the same release family the enhancer GGUF came from — via
// scripts/convert-lavasr-denoiser-to-gguf.py.
//
// Usage (denoise BEFORE enhance):
//
//     auto dn = tts_cpp::lavasr::Denoiser::load("lavasr-denoiser.gguf");
//     pcm = dn->denoise(pcm, sr);          // cleaned, SAME sample rate
//     // ...then optionally Enhancer::enhance() bandwidth-extends to 48 kHz.
//
// Like Enhancer, the Denoiser is immutable after load and safe to share across
// threads for concurrent denoise() calls (it holds only const weights).

#include "tts-cpp/export.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

class TTS_CPP_API Denoiser {
public:
    // Load the denoiser GGUF.  Throws std::runtime_error on failure (file
    // missing, wrong architecture, missing/mis-shaped tensors).  n_gpu_layers: 0 = scalar
    // CPU (default); >0 = ggml GPU graph (Adreno OpenCL); <0 = ggml-CPU (the GPU graph's twin).
    static std::unique_ptr<Denoiser> load(const std::string & gguf_path, int n_gpu_layers = 0);

    ~Denoiser();
    Denoiser(const Denoiser &)             = delete;
    Denoiser & operator=(const Denoiser &) = delete;

    // Denoise mono float32 PCM at `sr_in` Hz.  Returns a cleaned signal at the
    // SAME sample rate (contrast with Enhancer::enhance(), which resamples to
    // 48 kHz).  Returns empty for empty input.
    std::vector<float> denoise(const std::vector<float> & pcm_in, int sr_in) const;

    // Internal working sample rate the UL-UNAS network operates at (read from
    // GGUF metadata).  Informational — denoise() itself is rate-preserving.
    int native_sample_rate() const;

    // Compute backend of the neural core, mirroring Enhancer::backend_name():
    // the ggml backend name (e.g. "OpenCL", "CPU") when a ggml engine is
    // active, "scalar" for the pure-scalar path.
    std::string backend_name() const;

private:
    Denoiser();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::lavasr
