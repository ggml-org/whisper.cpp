#pragma once

// LavaSR denoiser — high-level post-processing entry point.  SCAFFOLD.
//
// Pipeline (mirrors the UL-UNAS denoiser as used by LavaSR / @qvac/tts-onnx):
//   pcm_in @ sr_in  --Lanczos-->  work_sr
//                   --STFT (n_fft, hop, win)-->  complex spectrogram
//                   --UL-UNAS U-Net (denoiser_core)-->  mask -> cleaned spec
//                   --ISTFT-->  cleaned @ work_sr
//                   --Lanczos-->  cleaned @ sr_in
//
// Runs BEFORE the enhancer.  Unlike enhance() (which always returns 48 kHz),
// denoise() is rate-preserving: the output sample rate equals `sr_in`.

#include "denoiser_core.h"

#include <string>
#include <vector>

namespace tts_cpp::lavasr {

// Denoise `pcm_in` (mono float32 at `sr_in` Hz) and return a cleaned signal at
// the same rate.  Returns empty on empty input.  Throws std::runtime_error if a
// required weight tensor is missing (or, in this scaffold, always — the forward
// is not implemented yet).
std::vector<float> denoise(const DenoiserWeights & w,
                           const std::vector<float> & pcm_in, int sr_in);

// Internal working sample rate the denoiser network operates at (from GGUF
// metadata; default 48 kHz placeholder).
int denoiser_work_sample_rate(const DenoiserWeights & w);

} // namespace tts_cpp::lavasr
