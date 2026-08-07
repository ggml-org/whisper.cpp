// audio8-cli: text -> 44.1 kHz WAV through the public tts_cpp::audio8::Engine,
// the same API the SDK binding drives.
//
//   audio8-cli --lm audio8-lm-q8_0.gguf --codec-decoder audio8-codec-decoder-f16.gguf \
//       --text "Hello from a fully on-device C++ pipeline." --out out.wav
//
// Cloning is in-process: add the encoder GGUF and a reference recording with
// its transcript, and the voice comes from the recording.
//
//   audio8-cli ... --codec-encoder audio8-codec-encoder-f16.gguf \
//       --ref-audio voice.wav --ref-text "What the recording says." --out out.wav

#include "tts-cpp/audio8/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int BITS_PER_SAMPLE = 16;
constexpr int CHANNELS = 1;
constexpr float FULL_SCALE = 32767.0f;
constexpr uint32_t PCM_FORMAT_CHUNK_BYTES = 16;
constexpr uint16_t PCM_FORMAT_TAG = 1;
constexpr uint32_t HEADER_BYTES_AFTER_SIZE = 36;

struct options {
    std::string lm;
    std::string codec_decoder;
    std::string codec_encoder;
    std::string ref_audio;
    std::string ref_text;
    std::string text = "Hello from a fully on-device C plus plus pipeline.";
    std::string out = "audio8_out.wav";
    std::string backends_dir;
    int seed = 42;
    int threads = 0;
    int max_frames = 0;
    int top_k = 50;
    int output_sample_rate = 0;
    float temperature = 0.7f;
    float top_p = 0.9f;
    bool greedy = false;
};

void print_usage(const char * program) {
    std::fprintf(stderr,
                 "usage: %s --lm LM.gguf --codec-decoder DEC.gguf [--text ...] "
                 "[--out out.wav]\n"
                 "          [--codec-encoder ENC.gguf --ref-audio ref.wav --ref-text "
                 "\"...\"]\n"
                 "          [--seed N] [--greedy] [--temperature F] [--top-k N] "
                 "[--top-p F]\n"
                 "          [--max-frames N] [--threads N] [--output-sample-rate N]\n"
                 "          [--backends-dir DIR]\n",
                 program);
}

bool takes_value(const std::string & flag) {
    return flag != "--greedy";
}

bool apply_flag(options & opts, const std::string & flag, const char * value) {
    if (flag == "--lm") opts.lm = value;
    else if (flag == "--codec-decoder") opts.codec_decoder = value;
    else if (flag == "--codec-encoder") opts.codec_encoder = value;
    else if (flag == "--ref-audio") opts.ref_audio = value;
    else if (flag == "--ref-text") opts.ref_text = value;
    else if (flag == "--text") opts.text = value;
    else if (flag == "--out") opts.out = value;
    else if (flag == "--backends-dir") opts.backends_dir = value;
    else if (flag == "--seed") opts.seed = std::atoi(value);
    else if (flag == "--threads" || flag == "-t") opts.threads = std::atoi(value);
    else if (flag == "--max-frames") opts.max_frames = std::atoi(value);
    else if (flag == "--top-k") opts.top_k = std::atoi(value);
    else if (flag == "--output-sample-rate") opts.output_sample_rate = std::atoi(value);
    else if (flag == "--temperature") opts.temperature = static_cast<float>(std::atof(value));
    else if (flag == "--top-p") opts.top_p = static_cast<float>(std::atof(value));
    else return false;
    return true;
}

bool parse_args(int argc, char ** argv, options & opts) {
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "--greedy") {
            opts.greedy = true;
            continue;
        }
        if (!takes_value(flag) || index + 1 >= argc) return false;
        if (!apply_flag(opts, flag, argv[++index])) return false;
    }
    return !opts.lm.empty() && !opts.codec_decoder.empty();
}

void write_u32(std::FILE * file, uint32_t value) {
    std::fwrite(&value, sizeof(value), 1, file);
}

void write_u16(std::FILE * file, uint16_t value) {
    std::fwrite(&value, sizeof(value), 1, file);
}

void write_tag(std::FILE * file, const char * tag) {
    std::fwrite(tag, 1, 4, file);
}

void write_header(std::FILE * file, size_t n_samples, int sample_rate) {
    const uint32_t data_bytes =
        static_cast<uint32_t>(n_samples * (BITS_PER_SAMPLE / 8) * CHANNELS);
    write_tag(file, "RIFF");
    write_u32(file, HEADER_BYTES_AFTER_SIZE + data_bytes);
    write_tag(file, "WAVE");
    write_tag(file, "fmt ");
    write_u32(file, PCM_FORMAT_CHUNK_BYTES);
    write_u16(file, PCM_FORMAT_TAG);
    write_u16(file, CHANNELS);
    write_u32(file, static_cast<uint32_t>(sample_rate));
    write_u32(file,
              static_cast<uint32_t>(sample_rate * CHANNELS * (BITS_PER_SAMPLE / 8)));
    write_u16(file, CHANNELS * (BITS_PER_SAMPLE / 8));
    write_u16(file, BITS_PER_SAMPLE);
    write_tag(file, "data");
    write_u32(file, data_bytes);
}

int16_t to_pcm16(float sample) {
    const float clipped = std::max(-1.0f, std::min(1.0f, sample));
    return static_cast<int16_t>(std::lrintf(clipped * FULL_SCALE));
}

void write_samples(std::FILE * file, const std::vector<float> & pcm) {
    for (float sample : pcm) {
        const int16_t value = to_pcm16(sample);
        std::fwrite(&value, sizeof(value), 1, file);
    }
}

bool write_wav(const std::string & path, const std::vector<float> & pcm, int sample_rate) {
    std::FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    write_header(file, pcm.size(), sample_rate);
    write_samples(file, pcm);
    std::fclose(file);
    return true;
}

tts_cpp::audio8::VoicePrompt load_voice(const options & opts) {
    if (opts.ref_audio.empty()) return {};
    return tts_cpp::audio8::load_voice_prompt(opts.ref_audio, opts.ref_text);
}

tts_cpp::audio8::EngineOptions to_engine_options(const options & opts) {
    tts_cpp::audio8::EngineOptions engine;
    engine.lm_gguf_path = opts.lm;
    engine.codec_decoder_gguf_path = opts.codec_decoder;
    engine.codec_encoder_gguf_path = opts.codec_encoder;
    engine.n_threads = opts.threads;
    engine.greedy = opts.greedy;
    engine.seed = opts.seed;
    engine.temperature = opts.temperature;
    engine.top_k = opts.top_k;
    engine.top_p = opts.top_p;
    engine.max_frames = opts.max_frames;
    engine.output_sample_rate = opts.output_sample_rate;
    engine.backends_dir = opts.backends_dir;
    return engine;
}

tts_cpp::audio8::SynthesisResult speak(tts_cpp::audio8::Engine & engine,
                                       const options & opts,
                                       const tts_cpp::audio8::VoicePrompt & voice) {
    return voice.empty() ? engine.synthesize(opts.text)
                         : engine.synthesize(opts.text, voice);
}

}  // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!opts.ref_audio.empty() && opts.codec_encoder.empty()) {
        std::fprintf(stderr, "--ref-audio needs --codec-encoder\n");
        return 1;
    }

    try {
        const tts_cpp::audio8::VoicePrompt voice = load_voice(opts);
        tts_cpp::audio8::Engine engine(to_engine_options(opts));
        const tts_cpp::audio8::SynthesisResult result = speak(engine, opts, voice);
        if (!write_wav(opts.out, result.pcm, result.sample_rate)) {
            std::fprintf(stderr, "cannot write %s\n", opts.out.c_str());
            return 1;
        }
        std::printf("%s: %d frames, %.2f s at %d Hz on %s\n", opts.out.c_str(),
                    result.frames, result.duration_s, result.sample_rate,
                    engine.backend_name().c_str());
    } catch (const std::exception & failure) {
        std::fprintf(stderr, "%s\n", failure.what());
        return 1;
    }
    return 0;
}
