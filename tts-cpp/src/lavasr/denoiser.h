#pragma once

// LavaSR denoiser — high-level post-processing entry point.
//
// Pipeline (mirrors the UL-UNAS denoiser as used by LavaSR / @qvac/tts-onnx):
//   pcm_in @ sr_in  --Lanczos-->  work_sr (16 kHz)
//                   --STFT (n_fft, hop, win, center)-->  complex spectrogram
//                   --UL-UNAS U-Net (denoiser_core), fixed-chunk overlap-add-->
//                     ratio mask -> cleaned spec
//                   --ISTFT-->  cleaned @ work_sr
//                   --Lanczos-->  cleaned @ sr_in (length preserved)
//
// Runs BEFORE the enhancer.  Unlike enhance() (which always returns 48 kHz),
// denoise() is rate-preserving: the output sample rate equals `sr_in`.

#include "denoiser_core.h"

#include <string>
#include <vector>

namespace tts_cpp::lavasr {

// Denoise `pcm_in` (mono float32 at `sr_in` Hz) and return a cleaned signal at
// the same rate (same length).  Returns empty on empty input.  Throws
// std::runtime_error if a required weight tensor is missing.
std::vector<float> denoise(const DenoiserWeights & w,
                           const std::vector<float> & pcm_in, int sr_in);

// Internal working sample rate the denoiser network operates at (from GGUF
// metadata; 16 kHz for the shipped UL-UNAS model).
int denoiser_work_sample_rate(const DenoiserWeights & w);

} // namespace tts_cpp::lavasr
