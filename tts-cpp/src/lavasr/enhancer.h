#pragma once

// LavaSR enhancer — high-level post-processing entry point.
//
// Pipeline (mirrors @qvac/tts-onnx LavaSREnhancer):
//   pcm_in @ sr_in  --Lanczos-->  48 kHz
//                   --Slaney log-mel (44.1k ref, n_fft=2048, hop=512, 80 bins)-->
//                   --ConvNeXt backbone + ISTFT spec head (enhancer_core)-->  real/imag
//                   --ISTFT (n_fft=2048, hop=512, center=false)-->  enhanced @ 48 kHz
//                   --FastLR crossover (cutoff = sr_in/2)-->  merged @ 48 kHz
//
// Neural bandwidth extension: the low band comes from the original (upsampled)
// signal, the synthesised high band from the network.  Output is always 48 kHz;
// the caller resamples to its target output rate afterwards.

#include "enhancer_core.h"

#include <string>
#include <vector>

namespace tts_cpp::lavasr {

// Enhance `pcm_in` (mono float32 at `sr_in` Hz, the engine's native rate) to a
// 48 kHz enhanced signal.  Returns empty on empty input.  Throws
// std::runtime_error if a required weight tensor is missing.
std::vector<float> enhance(const EnhancerWeights & w,
                           const std::vector<float> & pcm_in, int sr_in);

// Working sample rate the enhancer network operates at (48 kHz).
int enhancer_work_sample_rate(const EnhancerWeights & w);

} // namespace tts_cpp::lavasr
