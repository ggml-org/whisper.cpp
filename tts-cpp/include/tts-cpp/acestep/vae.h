#pragma once

// Public ACE-Step Oobleck VAE API (QVAC-21921).
//
// The audio autoencoder stage of ACE-Step 1.5 (music generation): a 48 kHz
// stereo AutoencoderOobleck. The decoder turns the 64-channel acoustic latent
// (produced by the DiT stage) into audio; the encoder does the inverse and is
// used for the reconstruction roundtrip. Both run through a ggml compute graph
// on the ggml-CPU backend, using the two custom ops landed in the ggml-speech
// fork (ggml_col2im_1d for the decoder's transposed convs, ggml_snake for the
// snake activations).
//
// Latents are laid out time-major: latent[t * 64 + c] for frame t, channel c.
// Audio is interleaved stereo: pcm[t * 2 + ch]. Upsample factor is 1920, so
// T_audio = T_latent * 1920 at 48 kHz.
//
// Usage:
//     auto vae = tts_cpp::acestep::Vae::load("vae-BF16.gguf");
//     auto pcm = vae->decode(latent, T_latent);       // -> interleaved 48 kHz stereo
//
// Memory scales with the decoded length; decode/encode bounded windows on CPU.

#include "tts-cpp/export.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct VaeOptions {
    bool verbose      = false;
    bool with_encoder = true;  // load the encoder too (needed for encode()/roundtrip)
    int  n_threads    = 0;     // 0 = hardware concurrency
};

class TTS_CPP_API Vae {
public:
    // Load the VAE GGUF (from acestep.cpp's convert.py). Throws std::runtime_error
    // on failure (file missing, wrong tensors, backend/alloc failure).
    static std::unique_ptr<Vae> load(const std::string & gguf_path, const VaeOptions & opts = {});

    ~Vae();
    Vae(const Vae &)             = delete;
    Vae & operator=(const Vae &) = delete;

    // Decode a 64-channel latent (time-major, latent[t*64 + c]) into interleaved
    // stereo 48 kHz PCM (2 * T_latent * 1920 samples). Empty on failure.
    std::vector<float> decode(const std::vector<float> & latent, int T_latent) const;

    // Encode interleaved stereo 48 kHz PCM (frames*2 samples) into the 64-channel
    // mean latent (time-major). Sets *T_latent_out. Empty on failure or if the
    // encoder was not loaded (see VaeOptions::with_encoder).
    std::vector<float> encode(const std::vector<float> & pcm_interleaved, int frames, int * T_latent_out) const;

    bool        has_encoder() const;
    int         sample_rate() const;   // 48000
    int         upsample_factor() const;  // 1920
    std::string backend_name() const;

private:
    Vae();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::acestep
