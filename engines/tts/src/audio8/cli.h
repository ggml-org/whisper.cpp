#pragma once

// The audio8-cli argument surface and its code dump, apart from the CLI itself
// so both can be exercised without loading a model.

#include <string>
#include <vector>

namespace tts_cpp::audio8::cli {

struct options {
    std::string lm;
    std::string codec_decoder;
    std::string codec_encoder;
    std::string ref_audio;
    std::string ref_text;
    std::string text = "Hello from a fully on-device C plus plus pipeline.";
    std::string out = "audio8_out.wav";
    std::string codes_out;
    std::string backends_dir;
    std::string backend;              // "" (auto), "cpu", "hexagon", or an exact ggml device name
    int seed = 42;
    int threads = 0;
    int n_gpu_layers = 0;
    int max_frames = 0;
    int top_k = 50;
    int output_sample_rate = 0;
    float temperature = 0.7f;
    float top_p = 0.9f;
    bool greedy = false;
    bool verbose = false;
    bool stream = false;
};

void print_usage(const char * program);

// False on an unknown flag, on a value-taking flag that ends the arguments, and
// when either required GGUF path is missing.
bool parse_args(int argc, const char * const * argv, options & opts);

// One line per frame, its codebook values comma-separated. Plain text so two
// runs can be diffed directly to see whether a backend or a quantisation tier
// changed the discrete trajectory.
bool write_codes(const std::string & path, const std::vector<int> & codes, int frames);

}  // namespace tts_cpp::audio8::cli
