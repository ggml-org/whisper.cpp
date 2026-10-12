#include "moss/transcribe_cli.h"

#include "moss/transcribe_text.h"

#include "mel_preprocess.h"
#include "parakeet/cli.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace parakeet::moss::cli {
namespace {

constexpr const char * PROGRAM = "moss-transcribe";

std::string json_escape(const std::string & text) {
    std::string out;
    for (unsigned char ch : text) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char code[8];
                    std::snprintf(code, sizeof(code), "\\u%04x", ch);
                    out += code;
                } else {
                    out += (char) ch;
                }
        }
    }
    return out;
}

std::string quoted(const std::string & text) {
    return "\"" + json_escape(text) + "\"";
}

std::string number(double value) {
    return detail::format_decimal(value);
}

std::string segment_json(const TranscriptSegment & segment) {
    return "{\"start\": " + number(segment.start_s) + ", \"end\": " + number(segment.end_s) +
           ", \"speaker\": " + quoted(segment.speaker) + ", \"text\": " + quoted(segment.text) + "}";
}

std::string segments_json(const std::vector<TranscriptSegment> & segments) {
    std::string out = "[";
    for (size_t i = 0; i < segments.size(); ++i) {
        out += (i == 0 ? "\n    " : ",\n    ") + segment_json(segments[i]);
    }
    return out + (segments.empty() ? "]" : "\n  ]");
}

void print_segment_lines(const std::vector<TranscriptSegment> & segments) {
    for (const auto & segment : segments) {
        std::printf("[%.2f-%.2f] %s: %s\n", segment.start_s, segment.end_s, segment.speaker.c_str(),
                segment.text.c_str());
    }
}

void print_segments(const TranscribeResult & result) {
    print_segment_lines(result.segments);
    if (result.segments.empty()) {
        std::printf("%s\n", result.text.c_str());
    }
}

bool save_transcript(const std::string & path, const TranscribeResult & result) {
    std::ofstream output(path, std::ios::binary);
    output << transcript_json(result) << "\n";
    return (bool) output;
}

void report_timings(const TranscribeResult & result, size_t samples, int sample_rate) {
    std::fprintf(stderr,
        "[%s] %.1fs audio, %d audio tokens, %d generated (encode %.0f ms on %s, prefill %.0f ms, decode %.0f ms)\n",
        PROGRAM, (double) samples / sample_rate, result.audio_tokens, result.generated_tokens,
        result.encode_ms, result.encoder_backend.c_str(), result.prefill_ms, result.decode_ms);
}

const char * encoder_placement(const TranscribeEngine & engine) {
    return engine.encoder_on_coreml() ? "Core ML sidecar" : "ggml";
}

int transcribe_file(const TranscribeCliArgs & args) {
    std::vector<float> pcm;
    int sample_rate = 0;
    if (load_wav_mono_f32(args.audio_path, pcm, sample_rate) != 0) {
        std::fprintf(stderr, "[%s] cannot read WAV: %s\n", PROGRAM, args.audio_path.c_str());
        return 1;
    }
    TranscribeEngine engine(args.options);
    std::fprintf(stderr, "[%s] backend: %s, audio encoder: %s\n", PROGRAM, engine.backend_name(),
            encoder_placement(engine));
    const TranscribeResult result = engine.transcribe(pcm.data(), pcm.size(), sample_rate, args.request);
    if (result.cancelled) {
        std::fprintf(stderr, "[%s] transcription cancelled\n", PROGRAM);
        return 1;
    }
    print_segments(result);
    if (!args.out_path.empty() && !save_transcript(args.out_path, result)) {
        std::fprintf(stderr, "[%s] cannot write %s\n", PROGRAM, args.out_path.c_str());
        return 1;
    }
    report_timings(result, pcm.size(), sample_rate);
    return 0;
}

std::vector<std::string> split_hotwords(const std::string & list) {
    std::vector<std::string> hotwords;
    size_t start = 0;
    for (size_t comma = list.find(','); comma != std::string::npos; comma = list.find(',', start)) {
        hotwords.push_back(list.substr(start, comma - start));
        start = comma + 1;
    }
    hotwords.push_back(list.substr(start));
    return hotwords;
}

} // namespace

void print_usage() {
    std::fprintf(stderr,
        "usage: %s --model moss-transcribe.gguf --audio speech.wav [--out transcript.json]\n"
        "       [--prompt \"...\" | --hotwords \"QVAC,vcpkg,Parakeet\"] [--max-new-tokens N]\n"
        "       [--threads 4] [--gpu] [--backend auto|cpu|cuda|vulkan|CUDA0] [--backends-dir dir]\n"
        "           16 kHz WAV in; prints one [start-end] Sxx: text line per segment\n", PROGRAM);
}

bool parse_args(int argc, const char * const * argv, TranscribeCliArgs & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (flag == "--model")               args.options.model_path = next();
        else if (flag == "--audio")          args.audio_path = next();
        else if (flag == "--out")            args.out_path = next();
        else if (flag == "--prompt")         args.request.prompt = next();
        else if (flag == "--hotwords")       args.request.hotwords = split_hotwords(next());
        else if (flag == "--max-new-tokens") args.request.max_new_tokens = std::atoi(next());
        else if (flag == "--threads")        args.options.n_threads = std::atoi(next());
        else if (flag == "--gpu")            args.options.use_gpu = true;
        else if (flag == "--backend")        args.options.backend = next();
        else if (flag == "--backends-dir")   args.options.backends_dir = next();
        else {
            std::fprintf(stderr, "unknown flag: %s\n", flag.c_str());
            return false;
        }
    }
    return !args.options.model_path.empty() && !args.audio_path.empty();
}

int run(const TranscribeCliArgs & args) {
    try {
        return transcribe_file(args);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[%s] error: %s\n", PROGRAM, error.what());
        return 1;
    }
}

std::string transcript_json(const TranscribeResult & result) {
    return "{\n  \"text\": " + quoted(result.text) +
           ",\n  \"segments\": " + segments_json(result.segments) +
           ",\n  \"audio_tokens\": " + std::to_string(result.audio_tokens) +
           ",\n  \"prompt_tokens\": " + std::to_string(result.prompt_tokens) +
           ",\n  \"generated_tokens\": " + std::to_string(result.generated_tokens) +
           ",\n  \"encode_ms\": " + number(result.encode_ms) +
           ",\n  \"encoder_backend\": " + quoted(result.encoder_backend) +
           ",\n  \"prefill_ms\": " + number(result.prefill_ms) +
           ",\n  \"decode_ms\": " + number(result.decode_ms) + "\n}";
}

} // namespace parakeet::moss::cli

extern "C" int parakeet_moss_transcribe_cli_main(int argc, char ** argv) {
    parakeet::moss::cli::TranscribeCliArgs args;
    try {
        if (!parakeet::moss::cli::parse_args(argc, argv, args)) {
            parakeet::moss::cli::print_usage();
            return 2;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        parakeet::moss::cli::print_usage();
        return 2;
    }
    return parakeet::moss::cli::run(args);
}
