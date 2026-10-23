// Native Whisper correctness/measurement driver. JSONL goes to stdout; library
// diagnostics (including total stage timings) go to stderr. No device suspend,
// model download, resampling, or SDK integration is performed here.
#include "whisper.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {
using clock_type = std::chrono::steady_clock;

struct failure : std::runtime_error {
    int code;
    failure(int code_, const std::string & message) : std::runtime_error(message), code(code_) {}
};

struct options {
    std::string model, audio, backends_dir, language = "en";
    int gpu_device = -1, threads = 4, warmup = 1, runs = 3;
    bool cpu = false, translate = false, flash_attn = true, help = false;
    double duration_seconds = 0;
};

void usage() {
    std::puts(
        "usage: whisper-hexagon --model FILE --audio FILE (--device N | --no-gpu) [options]\n"
        "  --audio FILE          little-endian mono 16000 Hz PCM16/IEEE float32 WAV\n"
        "  --device N            filtered GPU/IGPU ordinal, as listed by whisper-cli\n"
        "  --no-gpu              require a CPU/host BLAS reference backend\n"
        "  --threads N           CPU thread count (default 4)\n"
        "  --language CODE       fixed Whisper language (default en; no auto-detection)\n"
        "  --translate           translate speech into English (default transcribe)\n"
        "  --warmup N            untabulated warmup calls (default 1)\n"
        "  --runs N              timed measurement calls (default 3)\n"
        "  --duration-seconds S  then repeat calls for at least S seconds (default 0)\n"
        "  --no-flash-attn        disable flash attention in every phase\n"
        "  --backends-dir DIR    load dynamic GGML backends from this directory\n"
        "  --help                print help\n"
        "One model load; deterministic greedy decoding, seed 0, no prompt carry-over.\n"
        "Only whisper_full is timed. Run records include warmup/measure/sustained phases.\n"
        "Stage *_mean_s values are per-invocation API means, NOT transcription totals.\n"
        "Peak RSS is the process lifetime maximum, not per-call working memory.\n"
        "Primary backend identity is checked; graph placement needs separate diagnostics.\n"
        "Exit codes: 0 success, 2 arguments, 3 WAV, 4 backend, 5 model load,\n"
        "            7 whisper_full failure, 8 internal/runtime exception.");
}

int integer(const std::string & value, const std::string & flag, int minimum) {
    int result = 0;
    if (value.empty()) throw failure(2, flag + " requires an integer");
    for (char c : value) {
        if (c < '0' || c > '9' || result > (std::numeric_limits<int>::max() - (c - '0')) / 10) {
            throw failure(2, flag + ": invalid integer '" + value + "'");
        }
        result = result * 10 + c - '0';
    }
    if (result < minimum) throw failure(2, flag + ": value is below " + std::to_string(minimum));
    return result;
}

options parse(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") { opts.help = true; return opts; }
        if (flag == "--no-gpu") { opts.cpu = true; continue; }
        if (flag == "--translate") { opts.translate = true; continue; }
        if (flag == "--no-flash-attn") { opts.flash_attn = false; continue; }
        if (flag != "--model" && flag != "--audio" && flag != "--device" && flag != "--threads" &&
            flag != "--language" && flag != "--warmup" && flag != "--runs" &&
            flag != "--duration-seconds" && flag != "--backends-dir") {
            throw failure(2, "unknown argument: " + flag);
        }
        if (++i == argc) throw failure(2, flag + " requires a value");
        const std::string value = argv[i];
        if      (flag == "--model") opts.model = value;
        else if (flag == "--audio") opts.audio = value;
        else if (flag == "--backends-dir") opts.backends_dir = value;
        else if (flag == "--language") opts.language = value;
        else if (flag == "--device") opts.gpu_device = integer(value, flag, 0);
        else if (flag == "--threads") opts.threads = integer(value, flag, 1);
        else if (flag == "--warmup") opts.warmup = integer(value, flag, 0);
        else if (flag == "--runs") opts.runs = integer(value, flag, 1);
        else {
            char * end = nullptr;
            errno = 0;
            opts.duration_seconds = std::strtod(value.c_str(), &end);
            if (value.empty() || end == value.c_str() || *end || errno == ERANGE || !std::isfinite(opts.duration_seconds) ||
                opts.duration_seconds < 0 || std::isspace((unsigned char) value[0])) {
                throw failure(2, flag + " requires a finite non-negative number");
            }
        }
    }
    if (opts.model.empty() || opts.audio.empty()) throw failure(2, "--model and --audio are required");
    if (opts.cpu == (opts.gpu_device >= 0)) throw failure(2, "choose exactly one of --device N and --no-gpu");
    if (opts.language == "auto" || whisper_lang_id(opts.language.c_str()) < 0) {
        throw failure(2, "--language must be a fixed supported language code");
    }
    return opts;
}

// JSON strings preserve valid UTF-8, replacing malformed byte sequences with
// U+FFFD so an unusual filename cannot make the entire JSONL stream invalid.
std::string quote(const std::string & value) {
    std::string out = "\"";
    const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < value.size();) {
        const unsigned char c = (unsigned char) value[i];
        if (c < 0x80) {
            if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
            else if (c < 0x20) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
            else out += (char) c;
            ++i;
            continue;
        }
        size_t n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        bool valid = n != 0 && n <= value.size() - i;
        for (size_t j = 1; valid && j < n; ++j) {
            const unsigned char d = (unsigned char) value[i + j];
            valid = d >= 0x80 && d <= 0xbf &&
                !(j == 1 && ((c == 0xe0 && d < 0xa0) || (c == 0xed && d > 0x9f) ||
                            (c == 0xf0 && d < 0x90) || (c == 0xf4 && d > 0x8f)));
        }
        if (valid) { out.append(value, i, n); i += n; }
        else { out += "\\ufffd"; ++i; }
    }
    return out + '"';
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

const char * boolean(bool value) { return value ? "true" : "false"; }

std::string environment() {
    std::string out = "{";
    for (const char * name : {"GGML_HEXAGON_NDEV", "GGML_HEXAGON_HOSTBUF", "GGML_HEXAGON_BF16",
                              "GGML_HEXAGON_PROFILE", "GGML_HEXAGON_OPTRACE", "GGML_SCHED_DEBUG"}) {
        if (out.size() > 1) out += ',';
        const char * value = std::getenv(name);
        out += quote(name) + ':' + (value ? quote(value) : "null");
    }
    return out + '}';
}

void emit(const std::string & json) {
    if (std::printf("%s\n", json.c_str()) < 0 || std::fflush(stdout) != 0) {
        throw failure(8, "cannot write JSONL output");
    }
}

uint16_t le16(const unsigned char * p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
uint32_t le32(const unsigned char * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

void read_exact(std::ifstream & in, unsigned char * data, size_t size) {
    if (!in.read(reinterpret_cast<char *>(data), (std::streamsize) size)) throw failure(3, "truncated WAV data");
}

std::vector<float> read_wav(const std::string & path) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE float32 required");
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw failure(3, "cannot open WAV: " + path);
    const std::streamoff file_size = in.tellg();
    if (file_size < 12) throw failure(3, "WAV is shorter than its RIFF header");
    in.seekg(0);
    unsigned char header[16];
    read_exact(in, header, 12);
    if (std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4)) {
        throw failure(3, "only little-endian RIFF/WAVE audio is supported");
    }
    const uint64_t riff_end = uint64_t(le32(header + 4)) + 8;
    if (riff_end < 12 || riff_end > uint64_t(file_size)) throw failure(3, "invalid RIFF size");
    bool have_fmt = false, have_data = false;
    uint16_t format = 0, bits = 0, block_align = 0;
    uint64_t data_offset = 0, data_size = 0, pos = 12;
    while (pos + 8 <= riff_end) {
        in.seekg((std::streamoff) pos);
        read_exact(in, header, 8);
        const uint64_t size = le32(header + 4), payload = pos + 8;
        const uint64_t padded = size + (size & 1);
        if (padded > riff_end - payload) throw failure(3, "WAV chunk exceeds RIFF size");
        if (!std::memcmp(header, "fmt ", 4)) {
            if (have_fmt || size < 16) throw failure(3, "invalid or duplicate WAV fmt chunk");
            read_exact(in, header, 16);
            format = le16(header); bits = le16(header + 14); block_align = le16(header + 12);
            if (le16(header + 2) != 1 || le32(header + 4) != WHISPER_SAMPLE_RATE ||
                !((format == 1 && bits == 16) || (format == 3 && bits == 32)) ||
                block_align != bits / 8 || le32(header + 8) != WHISPER_SAMPLE_RATE * block_align) {
                throw failure(3, "WAV must be mono 16000 Hz PCM16 (format 1) or float32 (format 3)");
            }
            have_fmt = true;
        } else if (!std::memcmp(header, "data", 4)) {
            if (have_data) throw failure(3, "multiple WAV data chunks are unsupported");
            data_offset = payload; data_size = size; have_data = true;
        }
        pos = payload + padded;
    }
    if (pos != riff_end || !have_fmt || !have_data || !data_size || data_size % block_align) {
        throw failure(3, "missing, empty or misaligned WAV audio");
    }
    const uint64_t count = data_size / block_align;
    if (count > uint64_t(std::numeric_limits<int>::max())) throw failure(3, "WAV sample count exceeds whisper_full's int limit");
    std::vector<float> samples((size_t) count);
    in.seekg((std::streamoff) data_offset);
    std::array<unsigned char, 4096> buffer;
    size_t output = 0;
    for (uint64_t remaining = data_size; remaining;) {
        const size_t bytes = (size_t) std::min<uint64_t>(remaining, buffer.size());
        read_exact(in, buffer.data(), bytes);
        for (size_t i = 0; i < bytes; i += block_align) {
            float sample;
            if (format == 1) {
                const uint16_t value = le16(buffer.data() + i);
                sample = (value < 32768 ? int(value) : int(value) - 65536) / 32768.0f;
            } else {
                const uint32_t value = le32(buffer.data() + i);
                std::memcpy(&sample, &value, sizeof(sample));
                if (!std::isfinite(sample)) throw failure(3, "WAV contains a non-finite float sample");
            }
            samples[output++] = sample;
        }
        remaining -= bytes;
    }
    return samples;
}

std::string peak_rss_bytes() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return "null";
#if defined(__APPLE__)
    return std::to_string(uint64_t(usage.ru_maxrss));
#else
    return std::to_string(uint64_t(usage.ru_maxrss) * 1024);
#endif
#else
    return "null";
#endif
}

double elapsed(clock_type::time_point from) { return std::chrono::duration<double>(clock_type::now() - from).count(); }

ggml_backend_dev_t requested_device(int ordinal) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            if (ordinal-- == 0) return dev;
        }
    }
    throw failure(4, "requested GPU/IGPU ordinal is unavailable; check whisper-cli --list-devices");
}

int benchmark(const options & opts) {
    const auto process_start = clock_type::now();
    const std::vector<float> samples = read_wav(opts.audio);
    const double audio_seconds = double(samples.size()) / WHISPER_SAMPLE_RATE;
    if (opts.backends_dir.empty()) ggml_backend_load_all();
    else ggml_backend_load_all_from_path(opts.backends_dir.c_str());
    const std::string requested = opts.cpu ? "CPU" : ggml_backend_dev_name(requested_device(opts.gpu_device));
    emit("{\"kind\":\"config\",\"schema_version\":1,\"whisper_version\":" + quote(whisper_version()) +
         ",\"model\":" + quote(opts.model) + ",\"audio\":" + quote(opts.audio) +
         ",\"samples\":" + std::to_string(samples.size()) + ",\"audio_s\":" + number(audio_seconds) +
         ",\"requested_backend\":" + quote(requested) + ",\"requested_gpu_ordinal\":" +
         (opts.cpu ? "null" : std::to_string(opts.gpu_device)) + ",\"threads\":" + std::to_string(opts.threads) +
         ",\"language\":" + quote(opts.language) + ",\"translate\":" + boolean(opts.translate) +
         ",\"flash_attn\":" + boolean(opts.flash_attn) + ",\"no_context\":true,\"temperature\":0,\"temperature_inc\":0,\"seed\":0," +
         "\"greedy_best_of\":1,\"warmup_calls\":" + std::to_string(opts.warmup) + ",\"measurement_calls\":" +
         std::to_string(opts.runs) + ",\"sustained_seconds\":" + number(opts.duration_seconds) +
         ",\"environment\":" + environment() + ",\"backend_execution_confirmed\":false}");

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = !opts.cpu;
    cparams.gpu_device = opts.cpu ? 0 : opts.gpu_device;
    cparams.flash_attn = opts.flash_attn;
    const auto load_start = clock_type::now();
    std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx(
        whisper_init_from_file_with_params(opts.model.c_str(), cparams), whisper_free);
    const double load_seconds = elapsed(load_start);
    if (!ctx) throw failure(5, "Whisper model/context loading failed");
    const char * actual = whisper_get_primary_backend_name(ctx.get());
    const std::string actual_name = actual ? actual : "";
    ggml_backend_dev_t actual_dev = actual ? ggml_backend_dev_by_name(actual) : nullptr;
    ggml_backend_reg_t actual_reg = actual_dev ? ggml_backend_dev_backend_reg(actual_dev) : nullptr;
    const char * actual_registry = actual_reg ? ggml_backend_reg_name(actual_reg) : nullptr;
    const bool cpu_device = actual_dev && ggml_backend_dev_type(actual_dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    // Host BLAS is an ACCEL backend and may be primary on macOS. Permit only
    // that known host registry; an arbitrary ACCEL is not a CPU reference.
    const bool host_blas = actual_dev && ggml_backend_dev_type(actual_dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL &&
                           actual_registry && std::strcmp(actual_registry, "BLAS") == 0;
    const bool primary_matches = opts.cpu ? cpu_device || host_blas : actual_name == requested;
    emit("{\"kind\":\"load\",\"load_s\":" + number(load_seconds) + ",\"actual_primary_backend\":" +
         (actual ? quote(actual_name) : "null") + ",\"primary_backend_matches_request\":" + boolean(primary_matches) +
         ",\"actual_primary_registry\":" + (actual_registry ? quote(actual_registry) : "null") +
         ",\"cpu_reference_kind\":" + (cpu_device ? quote("cpu") : host_blas ? quote("host-blas") : "null") +
         ",\"backend_execution_confirmed\":false,\"peak_rss_bytes\":" + peak_rss_bytes() + "}");
    if (!primary_matches) throw failure(4, "initialized primary backend does not match request: " + actual_name + " vs " + requested);

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = opts.threads;
    params.language = opts.language.c_str();
    params.translate = opts.translate;
    params.no_context = true;
    params.temperature = 0;
    params.temperature_inc = 0;
    params.greedy.best_of = 1;
    params.seed = 0;
    params.print_progress = params.print_realtime = params.print_timestamps = params.print_special = false;
    std::string first_transcript;
    bool have_first = false;
    uint64_t total_calls = 0, sustained_calls = 0;
    auto run = [&](const char * phase, uint64_t index, clock_type::time_point phase_start) {
        const double phase_begin = elapsed(phase_start);
        std::fprintf(stderr, "\nwhisper-hexagon: begin phase=%s index=%llu\n", phase, (unsigned long long) index);
        whisper_reset_timings(ctx.get());
        const auto start = clock_type::now();
        const int rc = whisper_full(ctx.get(), params, samples.data(), (int) samples.size());
        const double inference_seconds = elapsed(start);
        const double phase_end = elapsed(phase_start);
        std::string transcript;
        if (rc == 0) {
            for (int i = 0; i < whisper_full_n_segments(ctx.get()); ++i) transcript += whisper_full_get_segment_text(ctx.get(), i);
        }
        // This API returns a new heap object and per-invocation means. The
        // existing stderr timing report additionally supplies total stage times.
        std::unique_ptr<whisper_timings> timings(whisper_get_timings(ctx.get()));
        whisper_print_timings(ctx.get());
        const double unavailable = std::numeric_limits<double>::quiet_NaN();
        emit("{\"kind\":\"run\",\"phase\":" + quote(phase) + ",\"index\":" + std::to_string(index) +
             ",\"call\":" + std::to_string(++total_calls) + ",\"whisper_full_rc\":" + std::to_string(rc) +
             ",\"samples\":" + std::to_string(samples.size()) + ",\"audio_s\":" + number(audio_seconds) +
             ",\"inference_s\":" + number(inference_seconds) + ",\"rtf\":" + number(inference_seconds / audio_seconds) +
             ",\"encode_mean_s\":" + number(timings ? timings->encode_ms / 1000.0 : unavailable) +
             ",\"decode_mean_s\":" + number(timings ? timings->decode_ms / 1000.0 : unavailable) +
             ",\"prompt_mean_s\":" + number(timings ? timings->prompt_ms / 1000.0 : unavailable) +
             ",\"batch_decode_mean_s\":" + number(timings ? timings->batchd_ms / 1000.0 : unavailable) +
             ",\"sample_mean_s\":" + number(timings ? timings->sample_ms / 1000.0 : unavailable) +
             ",\"phase_begin_s\":" + number(phase_begin) + ",\"phase_end_s\":" + number(phase_end) +
             ",\"peak_rss_bytes\":" + peak_rss_bytes() + ",\"actual_primary_backend\":" + quote(actual_name) +
             ",\"transcript\":" + quote(transcript) + ",\"matches_first_transcript\":" +
             (rc == 0 && have_first ? boolean(transcript == first_transcript) : "null") + "}");
        std::fprintf(stderr, "whisper-hexagon: end phase=%s index=%llu whisper_full_rc=%d\n", phase, (unsigned long long) index, rc);
        if (rc != 0) throw failure(7, "whisper_full failed with code " + std::to_string(rc));
        if (!have_first) { first_transcript = transcript; have_first = true; }
    };
    auto phase_start = clock_type::now();
    for (int i = 0; i < opts.warmup; ++i) run("warmup", uint64_t(i) + 1, phase_start);
    phase_start = clock_type::now();
    for (int i = 0; i < opts.runs; ++i) run("measure", uint64_t(i) + 1, phase_start);
    phase_start = clock_type::now();
    while (elapsed(phase_start) < opts.duration_seconds) run("sustained", ++sustained_calls, phase_start);
    emit("{\"kind\":\"end\",\"exit_code\":0,\"total_calls\":" + std::to_string(total_calls) +
         ",\"sustained_calls\":" + std::to_string(sustained_calls) + ",\"driver_elapsed_s\":" + number(elapsed(process_start)) +
         ",\"peak_rss_bytes\":" + peak_rss_bytes() + "}");
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse(argc, argv);
        if (opts.help) { usage(); return 0; }
        return benchmark(opts);
    } catch (const failure & error) {
        std::fprintf(stderr, "whisper-hexagon: %s\n", error.what());
        std::printf("{\"kind\":\"error\",\"exit_code\":%d,\"message\":%s}\n", error.code, quote(error.what()).c_str());
        return error.code;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "whisper-hexagon: %s\n", error.what());
        std::printf("{\"kind\":\"error\",\"exit_code\":8,\"message\":%s}\n", quote(error.what()).c_str());
        return 8;
    }
}
