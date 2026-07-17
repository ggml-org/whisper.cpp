#pragma once

// ACE-Step DiT (Diffusion Transformer) — ggml compute engine (QVAC-21921).
//
// The generative core: a 24-layer transformer (AdaLN, GQA self-attention with
// per-layer sliding-window / full alternation, cross-attention to the text
// encoder states, SwiGLU MLP) run as a ggml graph, one forward per flow-matching
// step. Ported from acestep.cpp/src/dit*.h. Every op it needs already exists in
// the ggml-speech fork (rms_norm, mul_mat, rope_ext, flash_attn_ext /
// soft_max_ext, swiglu[_split], timestep_embedding, conv_transpose_1d) — no new
// custom op, unlike the VAE. On CPU, attention uses the F32 soft_max_ext path
// (flash_attn_ext accumulates in F16 and drifts over 24 layers x 8 steps).
//
// Layout: math [S, H] == ggml ne[0]=H, ne[1]=S. Latents are channel-major per
// frame: latent[t*C + c]. The sampler (Euler loop) lives in the engine glue.

#include "ggml-backend.h"

#include <string>
#include <vector>

namespace tts_cpp::acestep {

// Populated from GGUF metadata (acestep-dit.* / acestep.*).
struct DitConfig {
    int   hidden_size       = 0;
    int   intermediate_size = 0;
    int   n_heads           = 0;
    int   n_kv_heads        = 0;
    int   head_dim          = 0;
    int   n_layers          = 0;
    int   in_channels       = 0;
    int   out_channels      = 0;
    int   patch_size        = 0;
    int   sliding_window    = 0;
    float rope_theta        = 0.0f;
    float rms_norm_eps      = 0.0f;
    int   enc_hidden_size   = 0;     // condition_embedder input dim (from weight shape)
    bool  is_turbo          = true;  // acestep.is_turbo: turbo=8 steps/shift 3; base/sft=50/shift 1
};

struct DitModel;  // opaque: fused weight tensors + backend weight buffer

// Load DiT weights from `path` onto `backend` (borrowed). Returns nullptr on
// failure. Reads config from GGUF metadata; fuses QKV / gate-up when tensor
// types match and pre-permutes proj_in/proj_out convs at load time.
DitModel *        dit_model_load(const std::string & path, ggml_backend_t backend, bool verbose);
void              dit_model_free(DitModel * m);
const DitConfig & dit_model_config(const DitModel * m);
size_t            dit_model_weight_bytes(const DitModel * m);

// Inputs for one DiT forward (velocity prediction). N = batch (CFG uses N=2).
struct DitForwardInputs {
    const float * input_latents = nullptr;  // [in_channels, T, N], channel-major per frame
    int           T             = 0;        // temporal length (multiple of patch_size)
    int           N             = 1;        // batch size

    const float * enc_hidden = nullptr;     // [H_enc, enc_S, N] text-encoder states (cond-embedded input)
    int           enc_S      = 0;
    int           H_enc      = 0;

    float t   = 0.0f;   // flow-matching timestep
    float t_r = 0.0f;   // reference timestep (turbo: t_r == t)

    // Attention masks (F16), row-major [KV, Q, 1, N]; null = unmasked.
    const void * sa_mask_sw = nullptr;  // [S, S, 1, N] self-attn sliding window
    const void * ca_mask    = nullptr;  // [enc_S, S, 1, N] cross-attn encoder padding
};

// Run one forward pass. Writes velocity [out_channels, T, N] (channel-major per
// frame) to `velocity_out`. Returns false on failure.
bool dit_model_forward(DitModel * m, const DitForwardInputs & in, std::vector<float> & velocity_out);

// Flow-matching timestep schedule (ACE-Step default):
//   t_i = shift * t / (1 + (shift-1)*t),  t = 1 - i/num_steps.
// Turbo: shift=3.0, num_steps=8. Base/SFT: shift=1.0, num_steps=50.
void dit_build_schedule(float shift, int num_steps, std::vector<float> & schedule_out);

// One full flow-matching denoise (Euler, no CFG — turbo runs guidance=1.0).
struct DitSampleParams {
    const float * noise           = nullptr;  // [out_channels, T, N] initial x_T
    const float * context_latents = nullptr;  // [in_channels-out_channels, T, N] conditioning
    const float * enc_hidden      = nullptr;  // [H_enc, enc_S, N] text-encoder states
    int           enc_S           = 0;
    int           H_enc           = 0;
    int           T               = 0;
    int           N               = 1;
    const float * schedule        = nullptr;  // [num_steps] descending timesteps
    int           num_steps       = 0;
    const int *   real_enc_S      = nullptr;  // [N] valid encoder lengths; null = all enc_S
};

// Writes the denoised latent [out_channels, T, N] to `latent_out`. Rebuilds the
// DiT graph per step (bring-up simplicity); correctness first, fusion later.
bool dit_sample(DitModel * m, const DitSampleParams & p, std::vector<float> & latent_out);

} // namespace tts_cpp::acestep
