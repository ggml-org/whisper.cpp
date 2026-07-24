#include "tts-cpp/parler/description.h"
#include "tts-cpp/parler/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model parler.gguf --text TEXT --out out.wav\n"
        "          [--description DESC] (full voice description; mutually\n"
        "                      exclusive with the template flags below)\n"
        "          [--voice NAME] [--emotion NAME] (one of the 12 speaking\n"
        "                      styles, e.g. happy, sad, anger, news)\n"
        "          [--pitch low|moderate|high] [--pace slow|moderate|fast]\n"
        "          [--expressivity monotone|\"slightly expressive\"|expressive]\n"
        "          [--noise clear|noisy] [--reverb close|distant]\n"
        "          [--quality basic|high|\"very high\"]\n"
        "          [--seed 42] [--threads N]\n"
        "          [--n-gpu-layers N] (offload to GPU: Metal/Vulkan/...; 0 = CPU)\n"
        "          [--greedy] (deterministic argmax decoding; default is the\n"
        "                      model's sampled decoding: temp 1.0, top-k 50)\n"
        "          [--temperature X] [--top-k N] [--top-p X]\n"
        "          [--max-frames N] (cap generation length in decoder steps,\n"
        "                      ~86 steps per second of audio; 0 = model default ~30 s)\n"
        "          [--min-new-tokens N] (-1 = model default)\n"
        "          [--no-normalize-numbers] (keep raw digits in the prompt;\n"
        "                      by default digits are expanded to English words)\n"
        "          [--backends-dir DIR]\n",
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
    tts_cpp::parler::EngineOptions opts;
    tts_cpp::parler::DescriptionSpec spec;
    std::string text, out;
    bool have_description = false;
    bool have_template_flag = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                usage(argv[0]);
                exit(2);
            }
            return argv[++i];
        };
        if      (a == "--model")          opts.model_gguf_path = next();
        else if (a == "--text")           text = next();
        else if (a == "--description")  { opts.default_description = next(); have_description = true; }
        else if (a == "--voice")        { spec.voice        = next(); have_template_flag = true; }
        else if (a == "--emotion")      { spec.emotion      = next(); have_template_flag = true; }
        else if (a == "--pitch")        { spec.pitch        = next(); have_template_flag = true; }
        else if (a == "--pace")         { spec.pace         = next(); have_template_flag = true; }
        else if (a == "--expressivity") { spec.expressivity = next(); have_template_flag = true; }
        else if (a == "--noise")        { spec.noise        = next(); have_template_flag = true; }
        else if (a == "--reverb")       { spec.reverb       = next(); have_template_flag = true; }
        else if (a == "--quality")      { spec.quality      = next(); have_template_flag = true; }
        else if (a == "--out")            out = next();
        else if (a == "--seed")           opts.seed = atoi(next());
        else if (a == "--threads")        opts.n_threads = atoi(next());
        else if (a == "--n-gpu-layers")   opts.n_gpu_layers = atoi(next());
        else if (a == "--greedy")         opts.greedy = true;
        else if (a == "--temperature")    opts.temperature = (float) atof(next());
        else if (a == "--top-k")          opts.top_k = atoi(next());
        else if (a == "--top-p")          opts.top_p = (float) atof(next());
        else if (a == "--max-frames")     opts.max_frames = atoi(next());
        else if (a == "--min-new-tokens") opts.min_new_tokens = atoi(next());
        else if (a == "--no-normalize-numbers") opts.normalize_numbers = false;
        else if (a == "--backends-dir")   opts.backends_dir = next();
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (opts.model_gguf_path.empty() || text.empty() || out.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (have_description && have_template_flag) {
        fprintf(stderr, "--description is mutually exclusive with the template "
                        "flags (--voice/--emotion/...)\n");
        return 2;
    }
    if (have_description && opts.default_description.empty()) {
        fprintf(stderr, "--description must not be empty (omit it for the "
                        "default template)\n");
        return 2;
    }
    if (!have_description) {
        try {
            opts.default_description = tts_cpp::parler::build_description(spec);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s\n", e.what());
            return 2;
        }
        fprintf(stderr, "parler-cli: description: %s\n",
                opts.default_description.c_str());
    }

    try {
        tts_cpp::parler::Engine engine(opts);
        fprintf(stderr, "parler-cli: backend %s\n", engine.backend_name().c_str());
        tts_cpp::parler::SynthesisResult res = engine.synthesize(text);
        write_wav(out, res.pcm, res.sample_rate);
        fprintf(stderr, "parler-cli: wrote %s (%.2f s @ %d Hz)\n",
                out.c_str(), res.duration_s, res.sample_rate);
    } catch (const std::exception & e) {
        fprintf(stderr, "parler-cli: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
