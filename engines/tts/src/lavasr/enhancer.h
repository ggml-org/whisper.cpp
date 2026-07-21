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

#include <functional>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

// Neural core signature: mel[n_mels*T] (mel[c*T+t]) -> real/imag[spec_bins*T]
// (x[f*T+t]).  Lets the DSP pipeline run either the scalar core
// (enhancer_spec_forward) or the GGML/GPU core (enhancer_ggml_spec_forward)
// without duplicating the resample/mel/ISTFT/FastLR stages.
using SpecForwardFn = std::function<void(const std::vector<float> & mel, int T,
                                         std::vector<float> & real,
                                         std::vector<float> & imag)>;

// Shared enhance() pipeline: upsample -> log-mel -> `spec_fwd` (neural core) ->
// ISTFT -> FastLR crossover.  Returns empty on empty input.  This TU has no
// ggml dependency; the GPU core is injected via `spec_fwd`.
std::vector<float> enhance_with(const EnhancerWeights & w,
                                const std::vector<float> & pcm_in, int sr_in,
                                const SpecForwardFn & spec_fwd);

// Enhance `pcm_in` (mono float32 at `sr_in` Hz, the engine's native rate) to a
// 48 kHz enhanced signal.  Uses the scalar CPU core.  Returns empty on empty
// input.  Throws std::runtime_error if a required weight tensor is missing.
std::vector<float> enhance(const EnhancerWeights & w,
                           const std::vector<float> & pcm_in, int sr_in);

// Working sample rate the enhancer network operates at (48 kHz).
int enhancer_work_sample_rate(const EnhancerWeights & w);

} // namespace tts_cpp::lavasr
