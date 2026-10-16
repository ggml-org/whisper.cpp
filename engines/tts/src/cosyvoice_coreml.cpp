#include "cosyvoice_coreml.h"

#include "coreml_sidecar_path.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

namespace {
constexpr int max_feature_dimensions = 8;

void require_shape(const cosyvoice_coreml & model, const char * name, const std::vector<int64_t> & shape) {
    if (!model.matches(name, shape)) throw std::runtime_error(std::string("cosyvoice: incompatible Core ML feature ") + name);
}

size_t elements(const std::vector<int64_t> & shape) {
    size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > SIZE_MAX / result)
            throw std::runtime_error("cosyvoice: invalid Core ML feature dimensions");
        result *= static_cast<size_t>(dimension);
    }
    return result;
}

void require_finite(const float * values, const std::vector<int64_t> & shape) {
    const auto count = elements(shape);
    if (!values || !std::all_of(values, values + count, [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("cosyvoice: non-finite Core ML feature");
}
}

bool cosyvoice_coreml::has_sidecar(const std::string & gguf, const std::string & suffix) {
#ifdef TTS_CPP_USE_COREML
    const char * units = std::getenv("COSYVOICE_COREML_COMPUTE_UNITS");
    if (units && std::string(units) == "disabled") return false;
    return tts_cpp::detail::coreml_sidecar_exists(tts_cpp::detail::coreml_sidecar_path(gguf, suffix));
#else
    (void) gguf;
    (void) suffix;
    return false;
#endif
}

std::shared_ptr<cosyvoice_coreml> cosyvoice_coreml::load(const std::string & gguf, const std::string & suffix) {
#ifdef TTS_CPP_USE_COREML
    const char * units = std::getenv("COSYVOICE_COREML_COMPUTE_UNITS");
    if (units && std::string(units) == "disabled") return {};
    if (units && std::string(units) != "cpu_only" && std::string(units) != "cpu_and_ane" &&
        std::string(units) != "cpu_and_gpu" && std::string(units) != "all")
        throw std::invalid_argument("cosyvoice: invalid COSYVOICE_COREML_COMPUTE_UNITS");
    const auto path = tts_cpp::detail::coreml_sidecar_path(gguf, suffix);
    if (!has_sidecar(gguf, suffix)) return {};
    auto * model = moss_coreml_load(path.c_str(), units ? units : "cpu_and_ane");
    if (!model) {
        std::fprintf(stderr, "cosyvoice: Core ML model unavailable, retaining ggml: %s\n", path.c_str());
        return {};
    }
    auto result = std::shared_ptr<cosyvoice_coreml>(new cosyvoice_coreml());
    result->model_ = std::shared_ptr<void>(model, [](void * p) { moss_coreml_free(static_cast<moss_coreml_model *>(p)); });
    result->label_ = moss_coreml_label(model);
    return result;
#else
    (void) gguf;
    (void) suffix;
    return {};
#endif
}

std::vector<int64_t> cosyvoice_coreml::dimensions(const char * name) const {
#ifdef TTS_CPP_USE_COREML
    int64_t shape[max_feature_dimensions]{};
    const auto count = moss_coreml_feature_dims(static_cast<moss_coreml_model *>(model_.get()), name,
                                               shape, max_feature_dimensions);
    if (count > 0 && count <= max_feature_dimensions) return {shape, shape + count};
#else
    (void) name;
#endif
    return {};
}

bool cosyvoice_coreml::matches(const char * name, const std::vector<int64_t> & shape) const {
    return !shape.empty() && dimensions(name) == shape;
}

bool cosyvoice_coreml::stateful() const {
#ifdef TTS_CPP_USE_COREML
    return moss_coreml_has_state(static_cast<moss_coreml_model *>(model_.get())) != 0;
#else
    return false;
#endif
}

void cosyvoice_coreml::reset() {
#ifdef TTS_CPP_USE_COREML
    if (stateful() && moss_coreml_reset_state(static_cast<moss_coreml_model *>(model_.get())) != 0)
        throw std::runtime_error("cosyvoice: cannot reset Core ML state");
#endif
    predictions_ = 0;
}

void cosyvoice_coreml::predict(std::initializer_list<cosyvoice_coreml_input> inputs,
                              std::initializer_list<cosyvoice_coreml_output> outputs) {
    std::vector<const char *> input_names, output_names;
    std::vector<const float *> input_data;
    std::vector<float *> output_data;
    for (const auto & input : inputs) {
        require_shape(*this, input.name, input.shape);
        if (!input.data) throw std::runtime_error("cosyvoice: missing Core ML input buffer");
        if (!input.finite_already_checked) require_finite(input.data, input.shape);
        input_names.push_back(input.name);
        input_data.push_back(input.data);
    }
    for (const auto & output : outputs) {
        require_shape(*this, output.name, output.shape);
        if (!output.data) throw std::runtime_error("cosyvoice: missing Core ML output buffer");
        output_names.push_back(output.name);
        output_data.push_back(output.data);
    }
#ifdef TTS_CPP_USE_COREML
    if (moss_coreml_predict(static_cast<moss_coreml_model *>(model_.get()),
                            input_names.data(), input_data.data(), static_cast<int>(input_names.size()),
                            output_names.data(), output_data.data(), static_cast<int>(output_names.size())) != 0)
        throw std::runtime_error("cosyvoice: Core ML prediction failed");
#else
    throw std::runtime_error("cosyvoice: Core ML support was not compiled");
#endif
    for (const auto & output : outputs) require_finite(output.data, output.shape);
    if (!predictions_) std::fprintf(stderr, "cosyvoice: Core ML predictions active (%s)\n", label_.c_str());
    ++predictions_;
}
