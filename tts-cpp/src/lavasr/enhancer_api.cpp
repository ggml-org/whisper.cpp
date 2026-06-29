#include "tts-cpp/lavasr/enhancer.h"

#include "enhancer.h"        // internal: enhance(EnhancerWeights, ...)
#include "enhancer_core.h"   // internal: EnhancerWeights
#include "enhancer_gguf.h"   // internal: load_enhancer_gguf

#include <stdexcept>

namespace tts_cpp::lavasr {

struct Enhancer::Impl {
    EnhancerWeights weights;
};

Enhancer::Enhancer() : impl_(std::make_unique<Impl>()) {}
Enhancer::~Enhancer() = default;

std::unique_ptr<Enhancer> Enhancer::load(const std::string & gguf_path) {
    std::unique_ptr<Enhancer> e(new Enhancer());
    std::string err;
    if (!load_enhancer_gguf(gguf_path, e->impl_->weights, &err)) {
        throw std::runtime_error(err.empty() ? "lavasr: failed to load enhancer GGUF"
                                             : err);
    }
    return e;
}

std::vector<float> Enhancer::enhance(const std::vector<float> & pcm_in,
                                     int sr_in) const {
    return tts_cpp::lavasr::enhance(impl_->weights, pcm_in, sr_in);
}

int Enhancer::output_sample_rate() const {
    return enhancer_work_sample_rate(impl_->weights);
}

} // namespace tts_cpp::lavasr
