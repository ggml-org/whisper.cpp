#include "tts-cpp/chatterbox/engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int    SEED         = 42;
constexpr int    GREEDY_TOP_K = 1;
constexpr int    GPU_LAYERS   = 99;

// T3 sampling is autoregressive, so one differing argmax re-rolls the rest of
// the sequence and the two backends legitimately emit different token counts.
// What must hold is that the GPU produces an utterance of the same order as
// the CPU: truncation, runaway generation and silent failure all break this
// while ordinary numerical divergence does not.
constexpr double MAX_DURATION_RATIO = 1.35;

const char * const PHRASES[] = {
    "The quick brown fox jumps over the lazy dog.",
    "Autumn leaves drift past the window of the evening train.",
    "She counted every star above the quiet harbour.",
};

struct run_result {
    int    tokens;
    int    samples;
    size_t pcm_size;
};

struct device_choice {
    int         n_gpu_layers = 0;
    std::string backend;
};

run_result synthesize(const std::string & t3, const std::string & s3gen,
                      const std::string & language, const std::string & text,
                      const device_choice & device) {
    tts_cpp::chatterbox::EngineOptions options;
    options.t3_gguf_path    = t3;
    options.s3gen_gguf_path = s3gen;
    options.n_gpu_layers    = device.n_gpu_layers;
    options.backend         = device.backend;
    options.seed            = SEED;
    options.top_k           = GREEDY_TOP_K;
    if (!language.empty()) {
        options.language = language;
    }

    tts_cpp::chatterbox::Engine engine(options);
    const auto result = engine.synthesize(text);
    return { result.t3_tokens, result.audio_samples, result.pcm.size() };
}

bool compare_one(const std::string & t3, const std::string & s3gen,
                 const std::string & language, const std::string & text,
                 const device_choice & device) {
    const run_result cpu = synthesize(t3, s3gen, language, text, device_choice{0, "cpu"});
    const run_result gpu = synthesize(t3, s3gen, language, text, device);

    if (cpu.pcm_size == 0 || gpu.pcm_size == 0) {
        fprintf(stderr, "  FAIL a backend produced no audio\n");
        return false;
    }

    const double longer  = cpu.samples > gpu.samples ? cpu.samples : gpu.samples;
    const double shorter = cpu.samples > gpu.samples ? gpu.samples : cpu.samples;
    const double ratio   = longer / shorter;

    fprintf(stderr, "  cpu tokens=%d samples=%d | gpu tokens=%d samples=%d -> ratio %.3f\n",
            cpu.tokens, cpu.samples, gpu.tokens, gpu.samples, ratio);

    if (ratio > MAX_DURATION_RATIO) {
        fprintf(stderr, "  FAIL duration ratio %.3f exceeds %.3f\n", ratio, MAX_DURATION_RATIO);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s T3.gguf S3GEN.gguf [language] [--backend NAME]\n", argv[0]);
        return 2;
    }
    const std::string t3    = argv[1];
    const std::string s3gen = argv[2];
    std::string   language;
    device_choice device{GPU_LAYERS, {}};
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            device = device_choice{0, argv[++i]};
        } else {
            language = arg;
        }
    }

    int failures = 0;
    for (const char * text : PHRASES) {
        fprintf(stderr, "\n== %s ==\n", text);
        if (!compare_one(t3, s3gen, language, text, device)) {
            ++failures;
        }
    }

    if (failures) {
        fprintf(stderr, "\n%d phrase(s) failed CPU-vs-GPU comparison\n", failures);
        return 1;
    }
    fprintf(stderr, "\nPASS\n");
    return 0;
}
