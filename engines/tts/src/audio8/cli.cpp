#include "audio8/cli.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace tts_cpp::audio8::cli {

namespace {

// Flags that stand alone; everything else in apply_flag consumes the next
// argument.
bool apply_switch(options & opts, const std::string & flag) {
    if (flag == "--greedy") opts.greedy = true;
    else if (flag == "--stream") opts.stream = true;
    else if (flag == "--verbose") opts.verbose = true;
    else return false;
    return true;
}

bool apply_flag(options & opts, const std::string & flag, const char * value) {
    if (flag == "--lm") opts.lm = value;
    else if (flag == "--codec-decoder") opts.codec_decoder = value;
    else if (flag == "--codec-encoder") opts.codec_encoder = value;
    else if (flag == "--ref-audio") opts.ref_audio = value;
    else if (flag == "--ref-text") opts.ref_text = value;
    else if (flag == "--text") opts.text = value;
    else if (flag == "--out") opts.out = value;
    else if (flag == "--dump-codes") opts.codes_out = value;
    else if (flag == "--backends-dir") opts.backends_dir = value;
    else if (flag == "--backend") opts.backend = value;
    else if (flag == "--seed") opts.seed = std::atoi(value);
    else if (flag == "--threads" || flag == "-t") opts.threads = std::atoi(value);
    else if (flag == "--n-gpu-layers" || flag == "-ngl") opts.n_gpu_layers = std::atoi(value);
    else if (flag == "--max-frames") opts.max_frames = std::atoi(value);
    else if (flag == "--top-k") opts.top_k = std::atoi(value);
    else if (flag == "--output-sample-rate") opts.output_sample_rate = std::atoi(value);
    else if (flag == "--temperature") opts.temperature = static_cast<float>(std::atof(value));
    else if (flag == "--top-p") opts.top_p = static_cast<float>(std::atof(value));
    else return false;
    return true;
}

void write_frame(std::FILE * file, const int * frame, int books) {
    for (int book = 0; book < books; ++book) {
        std::fprintf(file, book == 0 ? "%d" : ",%d", frame[book]);
    }
    std::fputc('\n', file);
}

}  // namespace

void print_usage(const char * program) {
    std::fprintf(stderr,
                 "usage: %s --lm LM.gguf --codec-decoder DEC.gguf [--text ...] "
                 "[--out out.wav]\n"
                 "          [--codec-encoder ENC.gguf --ref-audio ref.wav --ref-text "
                 "\"...\"]\n"
                 "          [--seed N] [--greedy] [--temperature F] [--top-k N] "
                 "[--top-p F]\n"
                 "          [--max-frames N] [--threads N] [--output-sample-rate N]\n"
                 "          [--n-gpu-layers N] [--dump-codes codes.txt]\n"
                 "          [--backend NAME] [--backends-dir DIR] [--verbose] [--stream]\n"
                 "\n"
                 "  --backend NAME   explicit backend: auto (default), cpu,\n"
                 "                   hexagon (Snapdragon HTP0), or an exact\n"
                 "                   ggml device name. Overrides --n-gpu-layers\n"
                 "                   when non-auto.\n",
                 program);
}

bool parse_args(int argc, const char * const * argv, options & opts) {
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (apply_switch(opts, flag)) continue;
        if (index + 1 >= argc) return false;
        if (!apply_flag(opts, flag, argv[++index])) return false;
    }
    return !opts.lm.empty() && !opts.codec_decoder.empty();
}

bool write_codes(const std::string & path, const std::vector<int> & codes, int frames) {
    if (frames <= 0) return false;
    const int books = static_cast<int>(codes.size()) / frames;
    std::FILE * file = std::fopen(path.c_str(), "w");
    if (!file) return false;
    for (int frame = 0; frame < frames; ++frame) {
        write_frame(file, codes.data() + static_cast<size_t>(frame) * books, books);
    }
    std::fclose(file);
    return true;
}

}  // namespace tts_cpp::audio8::cli
