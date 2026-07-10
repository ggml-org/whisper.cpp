#include "tts-cpp/lavasr/denoiser.h"

#include "denoiser.h"        // internal: denoise(DenoiserWeights, ...)
#include "denoiser_core.h"   // internal: DenoiserWeights
#include "denoiser_ggml.h"   // internal: DenoiserGgml (ggml/GPU neural core)
#include "denoiser_gguf.h"   // internal: load_denoiser_gguf

#include <stdexcept>

namespace tts_cpp::lavasr {

struct Denoiser::Impl {
    DenoiserWeights               weights;
    std::unique_ptr<DenoiserGgml> ggml; // non-null when a ggml backend is used
};

Denoiser::Denoiser() : impl_(std::make_unique<Impl>()) {}
Denoiser::~Denoiser() = default;

std::unique_ptr<Denoiser> Denoiser::load(const std::string & gguf_path, int n_gpu_layers) {
    std::unique_ptr<Denoiser> d(new Denoiser());
    std::string err;
    if (!load_denoiser_gguf(gguf_path, d->impl_->weights, &err)) {
        throw std::runtime_error(err.empty() ? "lavasr: failed to load denoiser GGUF"
                                             : err);
    }
    if (n_gpu_layers != 0) {
        d->impl_->ggml = DenoiserGgml::create(d->impl_->weights, n_gpu_layers);
    }
    return d;
}

std::vector<float> Denoiser::denoise(const std::vector<float> & pcm_in,
                                     int sr_in) const {
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
    return impl_->ggml ? impl_->ggml->backend_name() : "scalar";
}

} // namespace tts_cpp::lavasr
