#include "lavasr/coreml.h"
#include "lavasr/denoiser_gguf.h"

#include <cmath>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float MAX_ABSOLUTE_ERROR = 1e-6f;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(const std::vector<float> & actual, const std::vector<float> & expected) {
    require(actual.size() == expected.size(), "batch output length differs from serial output");
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "batch output is not finite");
        require(std::abs(actual[i] - expected[i]) <= MAX_ABSOLUTE_ERROR, "batch output differs from serial output");
    }
}

std::vector<float> input_plane(size_t chunk_size, int chunks, float phase) {
    std::vector<float> values(chunk_size * chunks);
    for (size_t i = 0; i < values.size(); ++i) {
        const float marker = static_cast<float>(i / chunk_size + 1);
        values[i] = 0.02f * marker + 0.15f * std::sin(static_cast<float>(i) * 0.071f + phase);
    }
    return values;
}

void check_chunk(tts_cpp::lavasr::CoreMLNetwork & model, int frames, int chunk,
                  const std::vector<float> & real, const std::vector<float> & imag,
                  const std::vector<float> & real_out, const std::vector<float> & imag_out,
                  size_t chunk_size) {
    const auto offset = static_cast<std::ptrdiff_t>(chunk * chunk_size);
    const auto count = static_cast<std::ptrdiff_t>(chunk_size);
    std::vector<float> expected_real, expected_imag;
    model.denoise({real.begin() + offset, real.begin() + offset + count},
                  {imag.begin() + offset, imag.begin() + offset + count},
                  frames, expected_real, expected_imag);
    require_close({real_out.begin() + offset, real_out.begin() + offset + count}, expected_real);
    require_close({imag_out.begin() + offset, imag_out.begin() + offset + count}, expected_imag);
}

void check_batch(tts_cpp::lavasr::CoreMLNetwork & model, int frames, int bins, int chunks) {
    const size_t chunk_size = static_cast<size_t>(frames) * bins;
    const auto real = input_plane(chunk_size, chunks, 0.0f);
    const auto imag = input_plane(chunk_size, chunks, 0.3f);
    std::vector<float> real_out, imag_out;
    model.denoise_batch(real, imag, frames, chunks, real_out, imag_out);
    require(real_out.size() == real.size() && imag_out.size() == imag.size(), "wrong batch output shape");
    for (int chunk = 0; chunk < chunks; ++chunk) {
        check_chunk(model, frames, chunk, real, imag, real_out, imag_out, chunk_size);
    }
    std::printf("chunks=%d: batch matches serial\n", chunks);
}

template<typename Function>
void rejects(Function function) {
    try {
        function();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid batch dimensions were accepted");
}

void check_invalid(tts_cpp::lavasr::CoreMLNetwork & model, int frames, int bins) {
    const auto real = input_plane(static_cast<size_t>(frames) * bins, 2, 0.0f);
    const std::vector<float> empty;
    std::vector<float> real_out, imag_out;
    rejects([&] { model.denoise_batch(real, real, frames, 0, real_out, imag_out); });
    rejects([&] { model.denoise_batch(real, real, frames, -1, real_out, imag_out); });
    rejects([&] { model.denoise_batch(real, real, frames, 1, real_out, imag_out); });
    rejects([&] { model.denoise_batch(real, empty, frames, 2, real_out, imag_out); });
    rejects([&] { model.denoise_batch(empty, empty, frames, 1, real_out, imag_out); });
    rejects([&] { model.denoise_batch(real, real, frames + 1, 2, real_out, imag_out); });
}

int check(const char * gguf, const char * sidecar, const char * policy) {
    tts_cpp::lavasr::DenoiserWeights weights;
    std::string error;
    const bool loaded = tts_cpp::lavasr::load_denoiser_gguf(gguf, weights, &error);
    require(loaded, error.c_str());
    auto model = tts_cpp::lavasr::CoreMLNetwork::load({sidecar, policy}, weights);
    for (int chunks : {1, 2, 3, 16, 1}) {
        check_batch(*model, weights.chunk_frames, weights.spec_bins, chunks);
    }
    check_invalid(*model, weights.chunk_frames, weights.spec_bins);
    auto concurrent = std::async(std::launch::async, [&] {
        check_batch(*model, weights.chunk_frames, weights.spec_bins, 2);
    });
    check_batch(*model, weights.chunk_frames, weights.spec_bins, 3);
    concurrent.get();
    return 0;
}

}

int main(int argc, char ** argv) {
#ifndef TTS_CPP_USE_COREML
    (void) argc;
    (void) argv;
    return 77;
#else
    if (argc != 4) return 2;
    try {
        return check(argv[1], argv[2], argv[3]);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
#endif
}
