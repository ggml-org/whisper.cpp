#pragma once

// Public LavaSR denoiser API — QVAC-16579 follow-up (deferred denoiser stage).
//
// SCAFFOLD / SKELETON ONLY.  This lays down the file structure, the public
// contract and the GGUF/converter layout for the LavaSR denoiser, mirroring the
// shipped enhancer (tts-cpp/lavasr/enhancer.h).  The forward math is NOT
// implemented yet: Denoiser::load() throws std::runtime_error until the UL-UNAS
// port lands.  This PR exists to land the structure/API so the implementation
// can follow in focused commits (see the module docstrings + the converter).
//
// LavaSR is a two-stage pipeline:
//   (1) UL-UNAS denoiser (this file) removes noise from the input, then
//   (2) the Vocos bandwidth-extension enhancer upsamples to 48 kHz.
// Only the enhancer shipped in QVAC-16579; this is the explicitly deferred stage
// ("The denoiser stage is a planned follow-up" — enhancer.h).
//
// Reference: UL-UNAS (arXiv:2503.00340, github.com/Xiaobin-Rong/ul-unas) as used
// by LavaSR (github.com/ysharma3501/LavaSR; ONNX family Topping1/LavaSRcpp) —
// the same release family the enhancer GGUF came from.
//
// Intended usage (denoise BEFORE enhance):
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
    // missing, wrong architecture, missing tensors).
    //
    // SCAFFOLD: currently always throws "not yet implemented" until the
    // UL-UNAS GGUF loader (denoiser_gguf) is ported.
    static std::unique_ptr<Denoiser> load(const std::string & gguf_path);

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

private:
    Denoiser();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::lavasr
