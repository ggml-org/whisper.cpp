#include "tts-cpp/lavasr/denoiser.h"
#include "tts-cpp/lavasr/enhancer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

constexpr int SAMPLE_RATE = 16000;
constexpr double MAX_NRMSE = 1e-4;
constexpr double REPEAT_NRMSE = 1e-6;

std::vector<float> signal(int count) {
    std::vector<float> result(count);
    for (int i = 0; i < count; ++i) {
        result[i] = static_cast<float>(0.15 * std::sin(i * 0.071) + 0.04 * std::cos(i * 0.173));
    }
    return result;
}

bool close(const std::vector<float> & actual, const std::vector<float> & expected, double tolerance) {
    if (actual.size() != expected.size()) return false;
    double error = 0.0, energy = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return false;
        const double delta = static_cast<double>(actual[i]) - expected[i];
        error += delta * delta;
        energy += static_cast<double>(expected[i]) * expected[i];
    }
    const double nrmse = energy > 0 ? std::sqrt(error / energy) : std::sqrt(error);
    std::printf("samples=%zu nrmse=%.8g limit=%.8g\n", actual.size(), nrmse, tolerance);
    return nrmse <= tolerance;
}

bool rejects_wrong_stage(const char * gguf, const char * wrong_model) {
    try {
        tts_cpp::lavasr::DenoiserOptions options;
        options.coreml = {wrong_model, "CPU_ONLY"};
        tts_cpp::lavasr::Denoiser::load(gguf, options);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

int check(const char * denoiser_gguf, const char * enhancer_gguf,
          const char * denoiser_coreml, const char * enhancer_coreml) {
    using namespace tts_cpp::lavasr;
    DenoiserOptions dn_options;
    dn_options.coreml = {denoiser_coreml, "CPU_ONLY"};
    EnhancerOptions enh_options;
    enh_options.coreml = {enhancer_coreml, "CPU_ONLY"};
    auto dn = Denoiser::load(denoiser_gguf, dn_options);
    auto enh = Enhancer::load(enhancer_gguf, enh_options);
    auto reference_dn = Denoiser::load(denoiser_gguf, -1);
    auto reference_enh = Enhancer::load(enhancer_gguf);
    if (dn->backend_name() != "coreml-cpu" || enh->backend_name() != "coreml-cpu" ||
        enh->backend_device() != tts_cpp::BackendDevice::CoreML) return 1;
    int failures = 0;
    for (int count : {0, 31, SAMPLE_RATE, 6 * SAMPLE_RATE}) {
        const auto input = signal(count);
        const auto expected_dn = reference_dn->denoise(input, SAMPLE_RATE);
        const auto actual_dn = dn->denoise(input, SAMPLE_RATE);
        failures += !close(actual_dn, expected_dn, MAX_NRMSE);
        const auto expected_enh = reference_enh->enhance(expected_dn, SAMPLE_RATE);
        const auto actual_enh = enh->enhance(actual_dn, SAMPLE_RATE);
        failures += !close(actual_enh, expected_enh, MAX_NRMSE);
        failures += !close(dn->denoise(input, SAMPLE_RATE), actual_dn, REPEAT_NRMSE);
        failures += !close(enh->enhance(actual_dn, SAMPLE_RATE), actual_enh, REPEAT_NRMSE);
        failures += actual_dn.size() != input.size();
        failures += actual_enh.size() != static_cast<size_t>(count) * enh->output_sample_rate() / SAMPLE_RATE;
    }
    failures += !rejects_wrong_stage(denoiser_gguf, enhancer_coreml);
    return failures == 0 ? 0 : 1;
}

}

int main(int argc, char ** argv) {
#ifndef TTS_CPP_USE_COREML
    (void) argc;
    (void) argv;
    return 77;
#else
    if (argc != 5) return 2;
    try {
        return check(argv[1], argv[2], argv[3], argv[4]);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
#endif
}
