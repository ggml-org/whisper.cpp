#include "tts-cpp/supertonic/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model supertonic2.gguf --text TEXT --out out.wav\n"
        "          [--language en] [--voice NAME] [--steps N] [--speed X]\n"
        "          (voice/steps/speed default to GGUF metadata when omitted)\n"
        "          [--seed 42] [--threads N] [--n-gpu-layers N]\n"
        "          [--noise-npy /path/to/noise.npy]\n",
        argv0);
}

void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output wav: " + path);
    uint32_t n = (uint32_t) wav.size();
    uint32_t byte_rate = (uint32_t) sr * 2;
    uint32_t data_size = n * 2;
    uint32_t chunk_size = 36 + data_size;
    uint16_t fmt = 1, channels = 1, align = 2, bps = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f); std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, f); std::fwrite(&fmt, 2, 1, f);
    std::fwrite(&channels, 2, 1, f); std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f); std::fwrite(&align, 2, 1, f);
    std::fwrite(&bps, 2, 1, f); std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float c = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t) std::lrintf(c * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

} // namespace

int main(int argc, char ** argv) {
    tts_cpp::supertonic::EngineOptions opts;
    std::string text;
    std::string out;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
            return argv[++i];
        };
        if (arg == "--model") opts.model_gguf_path = next("--model");
        else if (arg == "--text") text = next("--text");
        else if (arg == "--out") out = next("--out");
        else if (arg == "--language") opts.language = next("--language");
        else if (arg == "--voice") opts.voice = next("--voice");
        else if (arg == "--steps") opts.steps = std::stoi(next("--steps"));
        else if (arg == "--speed") opts.speed = std::stof(next("--speed"));
        else if (arg == "--seed") opts.seed = std::stoi(next("--seed"));
        else if (arg == "--threads") opts.n_threads = std::stoi(next("--threads"));
        else if (arg == "--n-gpu-layers") opts.n_gpu_layers = std::stoi(next("--n-gpu-layers"));
        else if (arg == "--noise-npy") opts.noise_npy_path = next("--noise-npy");
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", arg.c_str()); usage(argv[0]); return 2; }
    }
    if (opts.model_gguf_path.empty() || text.empty() || out.empty()) {
        usage(argv[0]);
        return 2;
    }
    try {
        auto result = tts_cpp::supertonic::synthesize(opts, text);
        write_wav(out, result.pcm, result.sample_rate);
        fprintf(stderr, "wrote %s (%.2fs @ %d Hz, %zu samples)\n",
                out.c_str(), result.duration_s, result.sample_rate, result.pcm.size());
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
