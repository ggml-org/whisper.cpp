#include "tts-cpp/lavasr/denoiser.h"

#include "denoiser.h"        // internal: denoise(DenoiserWeights, ...)
#include "denoiser_core.h"   // internal: DenoiserWeights
#include "denoiser_gguf.h"   // internal: load_denoiser_gguf

#include <stdexcept>

namespace tts_cpp::lavasr {

struct Denoiser::Impl {
    DenoiserWeights weights;
};

Denoiser::Denoiser() : impl_(std::make_unique<Impl>()) {}
Denoiser::~Denoiser() = default;

std::unique_ptr<Denoiser> Denoiser::load(const std::string & gguf_path) {
    std::unique_ptr<Denoiser> d(new Denoiser());
    std::string err;
    if (!load_denoiser_gguf(gguf_path, d->impl_->weights, &err)) {
        throw std::runtime_error(err.empty() ? "lavasr: failed to load denoiser GGUF"
                                             : err);
    }
    return d;
}

std::vector<float> Denoiser::denoise(const std::vector<float> & pcm_in,
                                     int sr_in) const {
    return tts_cpp::lavasr::denoise(impl_->weights, pcm_in, sr_in);
}

int Denoiser::native_sample_rate() const {
    return denoiser_work_sample_rate(impl_->weights);
}

} // namespace tts_cpp::lavasr
