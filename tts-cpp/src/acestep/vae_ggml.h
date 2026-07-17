#pragma once

// ACE-Step Oobleck VAE — ggml compute engine (QVAC-21921).
//
// Encoder (audio -> 64-ch latent) and decoder (latent -> 48 kHz stereo audio)
// as ggml graphs. The decoder's transposed convolutions use ggml_col2im_1d and
// every stage uses ggml_snake -- the two custom ops landed in the ggml-speech
// fork. Everything else is stock ggml. F32 activations, bf16 weights fused to F16.
//
// Downsample / upsample factor is 1920 (= 2*4*4*6*10). Memory scales with the
// audio length, so callers should decode/encode bounded windows on CPU.

#include "ggml-backend.h"

#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct VaeModel;  // opaque: weight tensors + backend weight buffer

// Load VAE weights from `path` onto `backend` (borrowed). Loads the decoder
// always; the encoder only when `with_encoder` (needed for the reconstruction
// roundtrip, not for pure DiT-latent decode). Returns nullptr on failure.
VaeModel * vae_model_load(const std::string & path, ggml_backend_t backend, bool with_encoder, bool verbose);
void       vae_model_free(VaeModel * m);
bool       vae_model_has_encoder(const VaeModel * m);
size_t     vae_model_weight_bytes(const VaeModel * m);

// Decode a 64-ch latent (time-major, latent[t*64 + c]) into interleaved stereo
// 48 kHz PCM. Returns T_audio frames (= T_latent * 1920) or -1 on failure.
int vae_model_decode(VaeModel * m, const float * latent, int T_latent, std::vector<float> & pcm_out);

// Encode interleaved stereo PCM (frames*2 samples, 48 kHz) into the 64-ch mean
// latent (time-major, out[t*64 + c]). Returns T_latent or -1 on failure.
int vae_model_encode(VaeModel * m, const float * pcm, int frames, std::vector<float> & latent_out);

} // namespace tts_cpp::acestep
