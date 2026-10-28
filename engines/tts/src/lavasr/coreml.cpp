#include "coreml.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

#include <algorithm>
#include <array>
#include <climits>
#include <mutex>
#include <stdexcept>

namespace tts_cpp::lavasr {
namespace {

const char * compute_policy(const std::string & units) {
    if (units == "CPU_ONLY") return "cpu_only";
    if (units == "CPU_AND_GPU") return "cpu_and_gpu";
    if (units == "CPU_AND_NE") return "cpu_and_ane";
    if (units == "ALL") return "all";
    throw std::invalid_argument("LavaSR Core ML requires CPU_ONLY, CPU_AND_GPU, CPU_AND_NE or ALL");
}

void gather_window(const std::vector<float> & source, int channels, int frames, int start,
                    int filled, int capacity, std::vector<float> & target) {
    std::fill(target.begin(), target.end(), 0.0f);
    for (int channel = 0; channel < channels; ++channel) {
        std::copy_n(source.data() + static_cast<size_t>(channel) * frames + start, filled,
                    target.data() + static_cast<size_t>(channel) * capacity);
    }
}

void copy_window(const std::vector<float> & source, int channels, int capacity, int skip,
                  int count, int offset, int frames, std::vector<float> & target) {
    for (int channel = 0; channel < channels; ++channel) {
        std::copy_n(source.data() + static_cast<size_t>(channel) * capacity + skip, count,
                    target.data() + static_cast<size_t>(channel) * frames + offset);
    }
}

}

bool coreml_requested(const CoreMLOptions & options) {
    if (options.model_path.empty() && options.compute_units.empty()) return false;
    if (options.model_path.empty() || options.compute_units.empty()) {
        throw std::invalid_argument("LavaSR Core ML requires both model_path and compute_units");
    }
    compute_policy(options.compute_units);
    return true;
}

struct CoreMLNetwork::Impl {
#ifdef TTS_CPP_USE_COREML
    std::unique_ptr<moss_coreml_model, decltype(&moss_coreml_free)> model{nullptr, moss_coreml_free};
#endif
    std::mutex mutex;
    std::string label;
    int frames = 0;
    int channels = 0;
    int bins = 0;
    int context = 0;
    std::vector<float> input;
    std::vector<float> second;
    std::vector<float> real;
    std::vector<float> imag;

    std::vector<int64_t> dimensions(const char * name) const {
#ifdef TTS_CPP_USE_COREML
        constexpr int MAX_RANK = 4;
        std::array<int64_t, MAX_RANK> shape{};
        const int rank = moss_coreml_feature_dims(model.get(), name, shape.data(), MAX_RANK);
        if (rank <= 0 || rank > MAX_RANK) throw std::runtime_error("LavaSR Core ML: missing tensor " + std::string(name));
        return {shape.begin(), shape.begin() + rank};
#else
        (void) name;
        throw std::runtime_error("LavaSR Core ML is unavailable in this build");
#endif
    }

    void require_shape(const char * name, const std::vector<int64_t> & expected) const {
        if (dimensions(name) != expected) throw std::runtime_error("LavaSR Core ML: incompatible tensor " + std::string(name));
    }

    void predict(const char * first, const char * other) {
#ifdef TTS_CPP_USE_COREML
        constexpr int TENSOR_COUNT = 2;
        const char * names[] = {first, other};
        const float * inputs[] = {input.data(), second.data()};
        const char * outputs[] = {"real_out", "imag_out"};
        float * values[] = {real.data(), imag.data()};
        if (moss_coreml_predict(model.get(), names, inputs, TENSOR_COUNT, outputs, values, TENSOR_COUNT) != 0) {
            throw std::runtime_error("LavaSR Core ML prediction failed");
        }
#else
        (void) first;
        (void) other;
        throw std::runtime_error("LavaSR Core ML is unavailable in this build");
#endif
    }
};

CoreMLNetwork::CoreMLNetwork(const CoreMLOptions & options) : impl_(std::make_unique<Impl>()) {
    if (!coreml_requested(options)) throw std::invalid_argument("LavaSR Core ML model was not specified");
#ifdef TTS_CPP_USE_COREML
    impl_->model.reset(moss_coreml_load(options.model_path.c_str(), compute_policy(options.compute_units)));
    if (!impl_->model) throw std::runtime_error("LavaSR Core ML could not load " + options.model_path);
    if (moss_coreml_has_state(impl_->model.get())) throw std::runtime_error("LavaSR Core ML requires a stateless model");
    impl_->label = moss_coreml_label(impl_->model.get());
#else
    throw std::runtime_error("LavaSR Core ML requires an Apple build with TTS_CPP_COREML=ON");
#endif
}

CoreMLNetwork::~CoreMLNetwork() = default;

std::unique_ptr<CoreMLNetwork> CoreMLNetwork::load(const CoreMLOptions & options, const EnhancerWeights & weights) {
    auto network = std::unique_ptr<CoreMLNetwork>(new CoreMLNetwork(options));
    auto & model = *network->impl_;
    const auto shape = model.dimensions("mel");
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != weights.n_mels || shape[2] <= 0 || shape[2] > INT_MAX) {
        throw std::runtime_error("LavaSR Core ML enhancer has incompatible mel dimensions");
    }
    model.frames = static_cast<int>(shape[2]);
    model.channels = weights.n_mels;
    model.bins = weights.spec_bins;
    const int64_t context = ((static_cast<int64_t>(weights.kernel) - 1) / 2) *
                            (static_cast<int64_t>(weights.n_blocks) + 1);
    if (context < 0 || model.frames <= 2 * context) {
        throw std::runtime_error("LavaSR Core ML enhancer window cannot contain its context");
    }
    model.context = static_cast<int>(context);
    model.require_shape("valid", {1, 1, model.frames});
    model.require_shape("real_out", {1, model.bins, model.frames});
    model.require_shape("imag_out", {1, model.bins, model.frames});
    model.input.resize(static_cast<size_t>(model.channels) * model.frames);
    model.second.resize(model.frames);
    model.real.resize(static_cast<size_t>(model.bins) * model.frames);
    model.imag.resize(model.real.size());
    return network;
}

std::unique_ptr<CoreMLNetwork> CoreMLNetwork::load(const CoreMLOptions & options, const DenoiserWeights & weights) {
    auto network = std::unique_ptr<CoreMLNetwork>(new CoreMLNetwork(options));
    auto & model = *network->impl_;
    const std::vector<int64_t> shape{1, 1, weights.chunk_frames, weights.spec_bins};
    model.require_shape("real", shape);
    model.require_shape("imag", shape);
    model.require_shape("real_out", shape);
    model.require_shape("imag_out", shape);
    model.frames = weights.chunk_frames;
    model.bins = weights.spec_bins;
    model.input.resize(static_cast<size_t>(model.frames) * model.bins);
    model.second.resize(model.input.size());
    model.real.resize(model.input.size());
    model.imag.resize(model.input.size());
    return network;
}

std::string CoreMLNetwork::backend_name() const { return impl_->label; }

void CoreMLNetwork::enhance(const std::vector<float> & mel, int frames,
                           std::vector<float> & real, std::vector<float> & imag) {
    auto & model = *impl_;
    std::lock_guard<std::mutex> lock(model.mutex);
    if (frames < 0 || mel.size() != static_cast<size_t>(model.channels) * frames) {
        throw std::invalid_argument("LavaSR Core ML enhancer input dimensions do not match");
    }
    real.resize(static_cast<size_t>(model.bins) * frames);
    imag.resize(real.size());
    for (int offset = 0; offset < frames;) {
        const int start = std::min(std::max(0, offset - model.context), std::max(0, frames - model.frames));
        const int filled = std::min(model.frames, frames - start);
        const int right_context = start + filled == frames ? 0 : model.context;
        const int count = start + filled - right_context - offset;
        gather_window(mel, model.channels, frames, start, filled, model.frames, model.input);
        std::fill(model.second.begin(), model.second.end(), 0.0f);
        std::fill_n(model.second.begin(), filled, 1.0f);
        model.predict("mel", "valid");
        copy_window(model.real, model.bins, model.frames, offset - start, count, offset, frames, real);
        copy_window(model.imag, model.bins, model.frames, offset - start, count, offset, frames, imag);
        offset += count;
    }
}

void CoreMLNetwork::denoise(const std::vector<float> & real, const std::vector<float> & imag, int frames,
                           std::vector<float> & real_out, std::vector<float> & imag_out) {
    auto & model = *impl_;
    std::lock_guard<std::mutex> lock(model.mutex);
    if (frames != model.frames || real.size() != model.input.size() || imag.size() != real.size()) {
        throw std::invalid_argument("LavaSR Core ML denoiser input dimensions do not match");
    }
    std::copy(real.begin(), real.end(), model.input.begin());
    std::copy(imag.begin(), imag.end(), model.second.begin());
    model.predict("real", "imag");
    real_out = model.real;
    imag_out = model.imag;
}

void CoreMLNetwork::denoise_batch(const std::vector<float> & real, const std::vector<float> & imag,
                                 int frames, int chunks, std::vector<float> & real_out,
                                 std::vector<float> & imag_out) {
    auto & model = *impl_;
    std::lock_guard<std::mutex> lock(model.mutex);
    const size_t chunk_size = model.input.size();
    if (frames != model.frames || chunks <= 0 || chunk_size == 0 || real.size() % chunk_size != 0 ||
        real.size() / chunk_size != static_cast<size_t>(chunks) || imag.size() != real.size()) {
        throw std::invalid_argument("LavaSR Core ML denoiser batch dimensions do not match");
    }
#ifdef TTS_CPP_USE_COREML
    constexpr int TENSOR_COUNT = 2;
    const char * names[] = {"real", "imag"};
    const float * inputs[] = {real.data(), imag.data()};
    const char * outputs[] = {"real_out", "imag_out"};
    real_out.resize(real.size());
    imag_out.resize(imag.size());
    float * values[] = {real_out.data(), imag_out.data()};
    if (moss_coreml_predict_batch(model.model.get(), names, inputs, TENSOR_COUNT,
                                 outputs, values, TENSOR_COUNT, chunks) != 0) {
        throw std::runtime_error("LavaSR Core ML batch prediction failed");
    }
#else
    (void) real_out;
    (void) imag_out;
    throw std::runtime_error("LavaSR Core ML is unavailable in this build");
#endif
}

}
