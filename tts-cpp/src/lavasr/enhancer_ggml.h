#pragma once

// LavaSR enhancer — GGML compute-graph forward (GPU: Vulkan / Metal / CUDA /
// OpenCL, and ggml-cpu).  This is the GPU-capable counterpart of the scalar
// enhancer_core forward; the two are kept bit-comparable (within f32 tolerance)
// so the scalar path stays as the correctness oracle and the CPU fallback.
//
// The enhancer is a feed-forward ConvNeXt backbone + a trigonometric ISTFT
// spec head, which maps cleanly onto ggml ops (im2col + mul_mat convolutions,
// ggml_norm layer-norm, gelu_erf, exp/clamp/sin/cos).  It therefore runs
// entirely on the selected backend with no host round-trips inside the graph.
//
// Layout convention inside the graph mirrors the scalar core's channel-major
// activations x[c * T + t]: a ggml tensor with ne = [T, C] (ne0 = time,
// ne1 = channel) holds exactly that flat memory, matching the
// supertonic_vocoder / campplus [L, C] 1-D convolution convention.

#include "enhancer_core.h"

#include "ggml-backend.h"

#include <vector>

namespace tts_cpp::lavasr {

// Opaque persistent state: the backend weight buffer (uploaded once at load),
// the declared weight tensor handles and a reusable graph allocator.  Built
// once via enhancer_ggml_create() and reused across enhance() calls.
struct EnhancerGgml;

// Declare + upload the enhancer weights to `backend` and build the reusable
// context.  `backend` is borrowed (not freed by enhancer_ggml_free); the
// caller owns its lifetime.  Returns nullptr on failure (e.g. a missing weight
// tensor or a backend buffer allocation failure) so the caller can fall back
// to the scalar path.
EnhancerGgml * enhancer_ggml_create(const EnhancerWeights & w, ggml_backend_t backend);

void enhancer_ggml_free(EnhancerGgml * g);

// GGML-graph equivalent of enhancer_spec_forward():
//   mel:  [n_mels * T] row-major, mel[c * T + t].
//   real, imag (out): [spec_bins * T] row-major, x[f * T + t].
// Not thread-safe: the reusable graph allocator is mutated per call, so the
// caller serialises concurrent invocations (Enhancer::enhance holds a mutex).
// Returns false on failure so the caller can fall back to the scalar core.
bool enhancer_ggml_spec_forward(EnhancerGgml * g,
                                const std::vector<float> & mel, int T,
                                std::vector<float> & real, std::vector<float> & imag);

// Full enhance() pipeline routed through the GPU/GGML neural core.  Shares the
// scalar DSP pipeline (resample -> mel -> [core] -> ISTFT -> FastLR); only the
// backbone + spec head run on `gpu`.  Falls back to the scalar core for the
// spec forward if the graph compute fails at runtime.  With gpu == nullptr this
// is identical to the scalar enhance().
std::vector<float> enhance(const EnhancerWeights & w, EnhancerGgml * gpu,
                           const std::vector<float> & pcm_in, int sr_in);

} // namespace tts_cpp::lavasr
