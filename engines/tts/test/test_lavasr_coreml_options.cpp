#include "lavasr/coreml.h"
#include "tts-cpp/lavasr/denoiser.h"
#include "tts-cpp/lavasr/enhancer.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void require(bool value, const char * message) {
    if (value) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

template <typename Function>
void invalid(Function function, const char * message) {
    try {
        function();
        require(false, message);
    } catch (const std::invalid_argument &) {
    } catch (...) {
        require(false, message);
    }
}

void check_policies() {
    using tts_cpp::lavasr::coreml_requested;
    require(!coreml_requested({}), "an existing call does not select Core ML");
    for (const char * policy : {"CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"}) {
        require(coreml_requested({"model.mlmodelc", policy}), "the Apple compute policy is accepted");
    }
    invalid([] { coreml_requested({"model.mlmodelc", ""}); }, "a model requires an explicit policy");
    invalid([] { coreml_requested({"", "CPU_ONLY"}); }, "a policy requires an explicit model");
    invalid([] { coreml_requested({"model.mlmodelc", "invalid"}); }, "an unknown policy is rejected");
}

void check_public_options() {
    tts_cpp::lavasr::EnhancerOptions enhancer;
    enhancer.coreml = {"model.mlmodelc", "CPU_ONLY"};
    enhancer.use_gpu = true;
    invalid([&] { tts_cpp::lavasr::Enhancer::load("missing.gguf", enhancer); },
            "conflicting enhancer devices are rejected before model loading");
    tts_cpp::lavasr::DenoiserOptions denoiser;
    denoiser.coreml = {"model.mlmodelc", "CPU_AND_NE"};
    denoiser.n_gpu_layers = 1;
    invalid([&] { tts_cpp::lavasr::Denoiser::load("missing.gguf", denoiser); },
            "conflicting denoiser devices are rejected before model loading");
    require(static_cast<int>(tts_cpp::BackendDevice::CPU) == 0, "the CPU enum value is preserved");
    require(static_cast<int>(tts_cpp::BackendDevice::GPU) == 1, "the GPU enum value is preserved");
    require(tts_cpp::BackendDevice::CoreML != tts_cpp::BackendDevice::GPU,
            "managed Core ML placement does not claim the GPU");
}

}

int main() {
    check_policies();
    check_public_options();
    return failures == 0 ? 0 : 1;
}
