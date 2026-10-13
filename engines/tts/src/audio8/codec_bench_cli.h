#pragma once

#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace audio8_codec_bench {
struct options {
    std::string encoder, decoder, input, output;
    std::string mode = "roundtrip";
    std::string codes, backend, output_f32;
    std::string codes_format = "csv";
    int frames = 0;
    int codebooks = 0;
    int warmup = 1;
    int repeat = 5;
    int threads = 4;
    int gpu_layers = 0;
    bool help = false;
};

inline int integer(const std::string & value, const std::string & flag) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(flag + " requires a nonnegative integer");
    size_t used = 0;
    const unsigned long n = std::stoul(value, &used);
    if (used != value.size() || n > INT_MAX)
        throw std::runtime_error(flag + " is out of range");
    return static_cast<int>(n);
}

inline options parse(int argc, char ** argv) {
    options opts;
    bool mode_set = false, gpu_layers_set = false, decode_options = false;
    bool backend_set = false, warmup_set = false, repeat_set = false, codebooks_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help" || flag == "-h") { opts.help = true; continue; }
        if (flag != "--codec-encoder" && flag != "--codec-decoder" &&
            flag != "--in" && flag != "--out" && flag != "--threads" &&
            flag != "--n-gpu-layers" && flag != "--mode" && flag != "--codes" &&
            flag != "--codec-backend" && flag != "--backend" && flag != "--frames" &&
            flag != "--codebooks" && flag != "--warmup" && flag != "--warmups" &&
            flag != "--repeat" && flag != "--repeats" && flag != "--out-f32" && flag != "--codes-format")
            throw std::runtime_error("unknown argument: " + flag);
        if (++i == argc || std::string(argv[i]).empty() || std::string(argv[i]).rfind("--", 0) == 0)
            throw std::runtime_error("missing value for " + flag);
        const std::string value(argv[i]);
        if (flag == "--codec-encoder") opts.encoder = value;
        else if (flag == "--codec-decoder") opts.decoder = value;
        else if (flag == "--in") opts.input = value;
        else if (flag == "--out") opts.output = value;
        else if (flag == "--threads") opts.threads = integer(value, flag);
        else if (flag == "--n-gpu-layers") { opts.gpu_layers = integer(value, flag); gpu_layers_set = true; }
        else if (flag == "--mode") {
            if (mode_set) throw std::runtime_error("duplicate --mode");
            opts.mode = value; mode_set = true;
        } else {
            decode_options = true;
            if (flag == "--codes") opts.codes = value;
            else if (flag == "--out-f32") opts.output_f32 = value;
            else if (flag == "--frames") opts.frames = integer(value, flag);
            else if (flag == "--codebooks") { opts.codebooks = integer(value, flag); codebooks_set = true; }
            else if (flag == "--codes-format") opts.codes_format = value;
            else if (flag == "--codec-backend" || flag == "--backend") {
                if (backend_set) throw std::runtime_error("duplicate codec backend option");
                opts.backend = value; backend_set = true;
            } else if (flag == "--warmup" || flag == "--warmups") {
                if (warmup_set) throw std::runtime_error("duplicate warmup option");
                opts.warmup = integer(value, flag); warmup_set = true;
            } else {
                if (repeat_set) throw std::runtime_error("duplicate repeat option");
                opts.repeat = integer(value, flag); repeat_set = true;
            }
        }
    }
    if (opts.help) return opts;
    if (!mode_set && !opts.codes.empty()) opts.mode = "decode";
    if (opts.mode != "roundtrip" && opts.mode != "decode")
        throw std::runtime_error("--mode must be roundtrip or decode");
    if (opts.threads == 0) throw std::runtime_error("--threads must be positive");
    if (opts.mode == "roundtrip") {
        if (decode_options) throw std::runtime_error("decoder benchmark options require --mode decode or --codes");
        if (opts.encoder.empty() || opts.decoder.empty() || opts.input.empty() || opts.output.empty())
            throw std::runtime_error("--codec-encoder, --codec-decoder, --in and --out are required");
    } else {
        if (!opts.encoder.empty() || !opts.input.empty() || gpu_layers_set)
            throw std::runtime_error("decode mode cannot use --codec-encoder, --in or --n-gpu-layers");
        if (opts.decoder.empty() || opts.codes.empty() || opts.backend.empty())
            throw std::runtime_error("decode mode requires --codec-decoder, --codes and --codec-backend");
        if (opts.backend != "cpu" && opts.backend != "hexagon" && opts.backend != "opencl")
            throw std::runtime_error("--codec-backend must be cpu, hexagon or opencl");
        if (opts.codes_format != "csv" && opts.codes_format != "i32")
            throw std::runtime_error("--codes-format must be csv or i32");
        if (opts.frames <= 0 || (codebooks_set && opts.codebooks <= 0) || opts.repeat <= 0)
            throw std::runtime_error("--frames, --codebooks and --repeat must be positive");
        if (opts.output.empty() && opts.output_f32.empty())
            throw std::runtime_error("decode mode requires --out and/or --out-f32");
    }
    for (const auto & output : {opts.output, opts.output_f32}) {
        if (output.empty()) continue;
        for (const auto & input : {opts.input, opts.encoder, opts.decoder, opts.codes})
            if (!input.empty() && input == output)
                throw std::runtime_error("output must differ from input and model paths");
    }
    if (!opts.output.empty() && opts.output == opts.output_f32)
        throw std::runtime_error("WAV and F32 output paths must differ");
    return opts;
}

inline std::filesystem::path normalized_output_path(const std::filesystem::path & path) {
    auto normalized = std::filesystem::weakly_canonical(path);
    // weakly_canonical leaves a dangling final symlink unresolved. Its target
    // can be another output which does not exist until the first write.
    for (int depth = 0; depth < 40; ++depth) {
        std::error_code ec;
        if (!std::filesystem::is_symlink(std::filesystem::symlink_status(normalized, ec)))
            return normalized;
        const auto target = std::filesystem::read_symlink(normalized);
        normalized = std::filesystem::weakly_canonical(
            target.is_absolute() ? target : normalized.parent_path() / target);
    }
    throw std::runtime_error("output path contains a symlink cycle or excessive symlink depth");
}

inline bool output_paths_case_equal(const std::filesystem::path & a,
                                    const std::filesystem::path & b) {
    const auto & left = a.native();
    const auto & right = b.native();
#ifdef _WIN32
    if (left.size() > INT_MAX || right.size() > INT_MAX)
        throw std::runtime_error("output path is too long");
    return CompareStringOrdinal(left.data(), static_cast<int>(left.size()),
                                right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
#else
    if (left.size() != right.size()) return false;
    auto fold = [](unsigned char c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; };
    for (size_t i = 0; i < left.size(); ++i)
        if (fold(static_cast<unsigned char>(left[i])) != fold(static_cast<unsigned char>(right[i])))
            return false;
    return true;
#endif
}

// Conservatively reject case aliases even before outputs exist. Windows uses
// Unicode ordinal comparison; other hosts also reject ASCII case aliases so
// the same invocation is safe on both case-sensitive and insensitive volumes.
// Filesystem identities still catch hard links and existing Unicode aliases.
inline void validate_output_paths(const options & opts) {
    std::vector<std::filesystem::path> outputs;
    for (const auto & output : {opts.output, opts.output_f32}) {
        if (output.empty()) continue;
        const auto normalized = normalized_output_path(output);
        for (const auto & input : {opts.input, opts.encoder, opts.decoder, opts.codes}) {
            if (input.empty()) continue;
            std::error_code ec;
            if (output_paths_case_equal(normalized, std::filesystem::weakly_canonical(input)) ||
                std::filesystem::equivalent(input, output, ec))
                throw std::runtime_error("output aliases an input or model file");
        }
        for (const auto & prior : outputs) {
            std::error_code ec;
            if (output_paths_case_equal(normalized, prior) || std::filesystem::equivalent(prior, output, ec))
                throw std::runtime_error("WAV and F32 outputs alias the same file");
        }
        outputs.push_back(normalized);
    }
}

inline size_t code_count(int frames, int codebooks) {
    if (frames <= 0 || codebooks <= 0 ||
        static_cast<size_t>(frames) > std::numeric_limits<size_t>::max() / sizeof(int32_t) / codebooks)
        throw std::runtime_error("invalid or overflowing code dimensions");
    return static_cast<size_t>(frames) * codebooks;
}

// The file has no header: exactly frames * codebooks little-endian int32
// tokens, ordered frame first, then semantic codebook followed by residuals.
inline std::vector<int32_t> read_i32_codes(const std::string & path, int frames, int codebooks) {
    const size_t count = code_count(frames, codebooks);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open codes file: " + path);
    const auto bytes = file.tellg();
    if (bytes < 0 || static_cast<uint64_t>(bytes) != static_cast<uint64_t>(count) * sizeof(int32_t))
        throw std::runtime_error("codes file size must equal frames * codebooks * 4 bytes");
    file.seekg(0);
    std::vector<int32_t> codes(count);
    for (size_t i = 0; i < count; ++i) {
        unsigned char token[4];
        if (!file.read(reinterpret_cast<char *>(token), sizeof(token)))
            throw std::runtime_error("failed reading codes file: " + path);
        const uint32_t value = static_cast<uint32_t>(token[0]) | (static_cast<uint32_t>(token[1]) << 8) |
                               (static_cast<uint32_t>(token[2]) << 16) | (static_cast<uint32_t>(token[3]) << 24);
        // Decode negative values without relying on unsigned-to-signed overflow.
        codes[i] = value <= INT32_MAX ? static_cast<int32_t>(value) :
                   static_cast<int32_t>(static_cast<int64_t>(value) - (int64_t{1} << 32));
    }
    return codes;
}

// CSV matches audio8-cli --dump-codes: one frame per line, one token per
// codebook. Reject partial rows, blank rows, extra frames and noninteger tokens.
inline std::vector<int32_t> parse_code_row(const std::string & row, int codebooks) {
    if (codebooks <= 0) throw std::runtime_error("invalid codebook count");
    std::vector<int32_t> codes;
    size_t start = 0;
    while (true) {
        const size_t end = row.find(',', start);
        std::string token = row.substr(start, end == std::string::npos ? end : end - start);
        const size_t first = token.find_first_not_of(" \t\r");
        const size_t last = token.find_last_not_of(" \t\r");
        if (first == std::string::npos) throw std::runtime_error("empty CSV code token");
        token = token.substr(first, last - first + 1);
        const size_t digits = token[0] == '-' ? 1 : 0;
        if (digits == token.size() || token.find_first_not_of("0123456789", digits) != std::string::npos)
            throw std::runtime_error("CSV code token must be an int32 integer");
        const int64_t value = std::stoll(token);
        if (value < INT32_MIN || value > INT32_MAX)
            throw std::runtime_error("CSV code token is out of int32 range");
        codes.push_back(static_cast<int32_t>(value));
        if (codes.size() > static_cast<size_t>(codebooks))
            throw std::runtime_error("CSV row has too many codebooks");
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (codes.size() != static_cast<size_t>(codebooks))
        throw std::runtime_error("CSV row does not match codec codebook count");
    return codes;
}

inline std::vector<int32_t> read_codes(const std::string & path, int frames, int codebooks,
                                     const std::string & format = "csv") {
    if (format == "i32") return read_i32_codes(path, frames, codebooks);
    if (format != "csv") throw std::runtime_error("unknown codes format");
    const size_t count = code_count(frames, codebooks);
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open codes file: " + path);
    std::vector<int32_t> codes;
    codes.reserve(count);
    std::string row;
    while (std::getline(file, row)) {
        if (codes.size() == count) throw std::runtime_error("CSV has more than --frames rows");
        const auto values = parse_code_row(row, codebooks);
        codes.insert(codes.end(), values.begin(), values.end());
    }
    if (!file.eof()) throw std::runtime_error("failed reading codes file: " + path);
    if (codes.size() != count) throw std::runtime_error("CSV frame count does not match --frames");
    return codes;
}

inline void validate_codes(const std::vector<int32_t> & codes, int frames, int codebooks,
                           int semantic_size, int residual_size) {
    if (codes.size() != code_count(frames, codebooks))
        throw std::runtime_error("code count does not match frames * codebooks");
    if (semantic_size <= 0 || residual_size <= 0)
        throw std::runtime_error("invalid codec token ranges");
    for (size_t i = 0; i < codes.size(); ++i) {
        const int limit = i % codebooks == 0 ? semantic_size : residual_size;
        if (codes[i] < 0 || codes[i] >= limit)
            throw std::runtime_error("code token out of range at frame " + std::to_string(i / codebooks) +
                                     ", codebook " + std::to_string(i % codebooks));
    }
}

// Production decode_codes consumes codebook-major rows, while CLI dumps are
// frame-major. Convert once before warmup or timing.
inline std::vector<int32_t> decoder_rows(const std::vector<int32_t> & frames, int n_frames, int codebooks) {
    if (frames.size() != code_count(n_frames, codebooks))
        throw std::runtime_error("code count does not match decoder dimensions");
    std::vector<int32_t> rows(frames.size());
    for (int frame = 0; frame < n_frames; ++frame)
        for (int book = 0; book < codebooks; ++book)
            rows[static_cast<size_t>(book) * n_frames + frame] = frames[static_cast<size_t>(frame) * codebooks + book];
    return rows;
}
} // namespace audio8_codec_bench
