// Fixture-backed public API test. Each CTest invocation owns one Engine/process.
#include "parakeet/engine.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// Ignore case/punctuation, preserving every word. Apostrophes are elided.
static std::string normalized(const std::string & text) {
    std::string out;
    for (unsigned char ch : text) {
        if (std::isalnum(ch)) out += char(std::toupper(ch));
        else if (ch != '\'' && !out.empty() && out.back() != ' ') out += ' ';
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: test-mosaic-inference MODEL WAV EXPECTED SAMPLES native|coreml cpu|MTL0|CUDA0|ROCm0\n");
        return 1;
    }
    try {
        const std::string mode = argv[5], backend = argv[6];
        require(mode == "native" || mode == "coreml", "invalid test mode");
        require(backend == "cpu" || backend == "MTL0" || backend == "CUDA0" || backend == "ROCm0", "invalid test backend");
        const bool coreml = mode == "coreml";
        env("PARAKEET_AFFINE_FUSED", nullptr); // Mosaic uses stock GGML blocks.
        env("PARAKEET_COREML_DISABLE", coreml ? nullptr : "1");
        env("PARAKEET_COREML_COMPUTE_UNITS", "all");
        parakeet::EngineOptions opts;
        opts.model_gguf_path = argv[1];
        opts.backend = backend;
        opts.n_threads = 4;
        opts.n_gpu_layers = backend == "cpu" ? 0 : 999;
        parakeet::Engine engine(opts);
        require(engine.model_type() == "tdt", "Mosaic did not load as TDT");
        require(engine.backend_device() == (backend == "cpu" ? parakeet::BackendDevice::CPU : parakeet::BackendDevice::GPU),
                "requested backend fell back");
        require(backend == "cpu" ? engine.backend_name() == "CPU" : engine.backend_name() == backend,
                "unexpected decoder backend");
        require(engine.encoder_on_coreml() == coreml, "sidecar load/disable contract failed");
        if (coreml) require(engine.encoder_backend() == "coreml-all", "unexpected Core ML compute mode");
        const std::string expected = normalized(argv[3]);
        std::vector<int32_t> tokens;
        std::string first_text;
        for (int run = 0; run < 2; ++run) {
            const auto result = engine.transcribe(argv[2]);
            std::printf("run=%d mode=%s backend=%s encoder=%s coreml_used=%d text=%s\n",
                run, mode.c_str(), engine.backend_name().c_str(), engine.encoder_backend().c_str(),
                int(result.encoder_used_coreml), result.text.c_str());
            require(result.encoder_used_coreml == coreml, "encoder routing/fallback contract failed");
            require(result.sample_rate == 16000 && result.audio_samples == std::stoi(argv[4]),
                    "audio was truncated or resampled");
            require(!result.token_ids.empty() && !result.text.empty(), "empty transcription");
            require(result.encoder_frames > 0 && result.mel_frames > 0, "encoder did not execute");
            if (coreml) require(result.mel_frames <= 512, "fixture exceeds masked512 capacity");
            require(normalized(result.text) == expected, "transcript differs from committed reference");
            if (run) require(result.token_ids == tokens && result.text == first_text, "repeat transcription drift");
            tokens = result.token_ids;
            first_text = result.text;
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Mosaic integration FAIL: %s\n", error.what());
        return 1;
    }
}
