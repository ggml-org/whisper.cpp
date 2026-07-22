#pragma once

// Public ACE-Step Oobleck VAE API.
//
// The audio autoencoder stage of ACE-Step 1.5 (music generation): a 48 kHz
// stereo AutoencoderOobleck. The decoder turns the 64-channel acoustic latent
// (produced by the DiT stage) into audio; the encoder does the inverse and is
// used for the reconstruction roundtrip. Both run through a ggml compute graph
// (CPU by default, or a GPU backend via VaeOptions::n_gpu_layers), using the two
// custom ops landed in the ggml-speech fork (ggml_col2im_1d for the decoder's
// transposed convs, ggml_snake for the snake activations) which now have both
// CPU and Metal kernels.
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

#include "audiogen-cpp/export.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct VaeOptions {
    bool verbose      = false;
    bool with_encoder = true;  // load the encoder too (needed for encode()/roundtrip)
    int  n_threads    = 0;     // 0 = hardware concurrency
    int  n_gpu_layers = 0;     // >0 = run the decode/encode graph on a GPU backend
                               // (Metal/CUDA/...); the custom snake / col2im_1d ops
                               // have GPU kernels in the ggml-speech fork. Falls
                               // back to CPU when no GPU backend is available.
};

class AUDIOGEN_API Vae {
public:
    // Load the VAE GGUF (from acestep.cpp's convert.py). Throws std::runtime_error
    // on failure (file missing, wrong tensors, backend/alloc failure).
    static std::unique_ptr<Vae> load(const std::string & gguf_path, const VaeOptions & opts = {});

    ~Vae();
    Vae(const Vae &)             = delete;
    Vae & operator=(const Vae &) = delete;

    // Decode a 64-channel latent (time-major, latent[t*64 + c]) into interleaved
    // stereo 48 kHz PCM (2 * T_latent * 1920 samples). Empty on failure.
    //
    // `on_progress` (optional) is invoked as the decode graph executes, once per
    // computed node, with (done, total) node counts -> fine-grained progress for
    // the otherwise-opaque VAE stage. Return false to cancel the decode (yields
    // an empty result). Throttled internally to ~1% steps.
    using ProgressCb = std::function<bool(int done, int total)>;
    std::vector<float> decode(const std::vector<float> & latent, int T_latent,
                              const ProgressCb & on_progress = {}) const;

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
