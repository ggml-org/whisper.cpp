#include "tts-cpp/lavasr/denoiser.h"

#include "denoiser.h"        // internal: denoise(DenoiserWeights, ...)
#include "denoiser_core.h"   // internal: DenoiserWeights
#include "denoiser_ggml.h"   // internal: DenoiserGgml (ggml/GPU neural core)
#include "denoiser_gguf.h"   // internal: load_denoiser_gguf
#include "coreml.h"

#include <stdexcept>

namespace tts_cpp::lavasr {

struct Denoiser::Impl {
    DenoiserWeights               weights;
    std::unique_ptr<CoreMLNetwork> coreml;
    std::unique_ptr<DenoiserGgml> ggml; // non-null when a ggml backend is used
};

Denoiser::Denoiser() : impl_(std::make_unique<Impl>()) {}
Denoiser::~Denoiser() = default;

std::unique_ptr<Denoiser> Denoiser::load(const std::string & gguf_path, int n_gpu_layers) {
    DenoiserOptions options;
    options.n_gpu_layers = n_gpu_layers;
    return load(gguf_path, options);
}

std::unique_ptr<Denoiser> Denoiser::load(const std::string & gguf_path, const DenoiserOptions & options) {
    const bool use_coreml = coreml_requested(options.coreml);
    if (use_coreml && options.n_gpu_layers != 0) {
        throw std::invalid_argument("LavaSR Core ML compute_units and n_gpu_layers cannot be combined");
    }
    std::unique_ptr<Denoiser> d(new Denoiser());
    std::string err;
    if (!load_denoiser_gguf(gguf_path, d->impl_->weights, &err)) {
        throw std::runtime_error(err.empty() ? "lavasr: failed to load denoiser GGUF"
                                             : err);
    }
    if (use_coreml) {
        d->impl_->coreml = CoreMLNetwork::load(options.coreml, d->impl_->weights);
    } else if (options.n_gpu_layers != 0) {
        d->impl_->ggml = DenoiserGgml::create(d->impl_->weights, options.n_gpu_layers);
    }
    return d;
}

std::vector<float> Denoiser::denoise(const std::vector<float> & pcm_in,
                                     int sr_in) const {
    if (impl_->coreml) {
        return tts_cpp::lavasr::denoise_with_batch_core(impl_->weights, pcm_in, sr_in,
            [this](const std::vector<float> & real, const std::vector<float> & imag, int frames, int chunks,
                   std::vector<float> & real_out, std::vector<float> & imag_out) {
                impl_->coreml->denoise_batch(real, imag, frames, chunks, real_out, imag_out);
            });
    }
    if (impl_->ggml) {
        DenoiserGgml * g = impl_->ggml.get();
        return tts_cpp::lavasr::denoise_with_batch_core(
            impl_->weights, pcm_in, sr_in,
            [g](const std::vector<float> & re, const std::vector<float> & im, int L, int nc,
                std::vector<float> & orr, std::vector<float> & oii) {
                g->batch_forward(re, im, L, nc, orr, oii);
            });
    }
    return tts_cpp::lavasr::denoise(impl_->weights, pcm_in, sr_in);
}

int Denoiser::native_sample_rate() const {
    return denoiser_work_sample_rate(impl_->weights);
}

std::string Denoiser::backend_name() const {
    if (impl_->coreml) return impl_->coreml->backend_name();
    return impl_->ggml ? impl_->ggml->backend_name() : "scalar";
}

} // namespace tts_cpp::lavasr
