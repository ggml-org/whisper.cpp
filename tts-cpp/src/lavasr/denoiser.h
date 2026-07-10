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

#include <functional>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

// The per-chunk neural core (same contract as denoiser_net_forward): real/imag [L*spec_bins]
// in, masked real/imag out.  Lets the pipeline run either the scalar or the ggml (GPU) core.
using DenoiseChunkCore = std::function<void(const std::vector<float> & real_in,
                                            const std::vector<float> & imag_in, int L,
                                            std::vector<float> & real_out, std::vector<float> & imag_out)>;

// Denoise `pcm_in` driving the chunk pipeline with an injected neural core.
std::vector<float> denoise_with_core(const DenoiserWeights & w, const std::vector<float> & pcm_in,
                                     int sr_in, const DenoiseChunkCore & core);

// Batched neural core: all n_chunks zero-state chunks stacked as [n_chunks][L*spec_bins]
// in, masked planes out (same layout).  Lets a GPU core run one graph for the whole clip.
using DenoiseBatchCore = std::function<void(const std::vector<float> & real_in,
                                            const std::vector<float> & imag_in, int L, int n_chunks,
                                            std::vector<float> & real_out, std::vector<float> & imag_out)>;

// Denoise `pcm_in` extracting ALL chunks up front, running the batch core once,
// then overlap-add.  Same result as denoise_with_core; fewer core invocations.
std::vector<float> denoise_with_batch_core(const DenoiserWeights & w, const std::vector<float> & pcm_in,
                                           int sr_in, const DenoiseBatchCore & core);

// Denoise `pcm_in` (mono float32 at `sr_in` Hz) and return a cleaned signal at
// the same rate (same length).  Returns empty on empty input.  Throws
// std::runtime_error if a required weight tensor is missing.  Uses the scalar core.
std::vector<float> denoise(const DenoiserWeights & w,
                           const std::vector<float> & pcm_in, int sr_in);

// Internal working sample rate the denoiser network operates at (from GGUF
// metadata; 16 kHz for the shipped UL-UNAS model).
int denoiser_work_sample_rate(const DenoiserWeights & w);

} // namespace tts_cpp::lavasr
