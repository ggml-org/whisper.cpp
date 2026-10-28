// Installed-public-API Chatterbox measurement consumer. JSONL is written to
// stdout; library diagnostics stay on stderr. Backend selection is not proof
// of DSP execution: collect the backend's diagnostics in a separate run.
#include <tts-cpp/chatterbox/engine.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {
using clock_type = std::chrono::steady_clock;
using tts_cpp::chatterbox::Engine;
using tts_cpp::chatterbox::EngineOptions;
using tts_cpp::chatterbox::SynthesisResult;

struct failure : std::runtime_error {
    int code;
    failure(int code_, const std::string & message) : std::runtime_error(message), code(code_) {}
};

struct options {
    EngineOptions engine;
    std::string text, text_file, out;
    int warmup = 1, runs = 3;
    double duration_seconds = 0;
    bool help = false;
    options() {
        engine.n_threads = 4;
        engine.seed = 42;
        engine.max_sentence_chars = 180;
        engine.kv_cache_type = "f32";
    }
};

void usage() {
    std::puts(
        "usage: chatterbox-hexagon --model T3.gguf --s3gen-gguf S3GEN.gguf\n"
        "       --backend cpu|opencl|hexagon (--text TEXT | --text-file FILE) --out FILE [options]\n"
        "  --language CODE                    en, fr or es (default en)\n"
        "  --reference-audio FILE              reference voice WAV\n"
        "  --voice-dir DIR                     precomputed reference voice\n"
        "  --threads N                         CPU threads (default 4)\n"
        "  --seed N                            fixed non-negative seed (default 42)\n"
        "  --n-ctx N                           T3 context cap, 0=model default\n"
        "  --n-predict N                       speech token limit (default 1000)\n"
        "  --kv-cache-type f32|f16|q8_0         requested KV representation (default f32)\n"
        "  --max-sentence-chars N              sentence splitting (default 180; 0 disables)\n"
        "  --cfm-steps N                       0=variant default\n"
        "  --stream-chunk-tokens N             0=batch (default), positive=streaming\n"
        "  --stream-first-chunk-tokens N       0=normal chunk size\n"
        "  --stream-left-context-tokens N      0=full cumulative context\n"
        "  --stream-cfm-steps N                0=variant streaming default\n"
        "  --warmup N                          extra excluded warmups (default 1)\n"
        "  --runs N                            steady measurement calls (default 3)\n"
        "  --duration-seconds S                additional sustained native inference seconds\n"
        "  --backends-dir DIR                  dynamic GGML backend directory\n"
        "  --opencl-cache-dir DIR              Android OpenCL binary cache\n"
        "  --verbose                           library stage diagnostics to stderr\n"
        "  --help                              print this help\n"
        "One excluded first_call always completes any lazy S3Gen preload. Constructor\n"
        "time is not full model-load time. Only Engine::synthesize is timed; callback\n"
        "validation/copies are part of that measured call. WAV writes and final copy\n"
        "are outside it. Each call writes FILE.PHASE-INDEX.wav; FILE is the final\n"
        "completed call's PCM16 WAV. Existing output files are refused.\n"
        "Sustained duration and phase_begin_s/end_s sum native calls, excluding I/O.\n"
        "Stop-token status is not exposed by the public API; token limits are only\n"
        "flagged as suspected. Process peak RSS excludes some device/shared memory.\n"
        "Exit codes: 0 complete, 2 arguments, 4 backend mismatch, 5 construction,\n"
        "            6 audio/output contract, 7 synthesis, 8 driver/runtime error.");
}

int integer(const std::string & value, const std::string & flag, int minimum) {
    int result = 0;
    if (value.empty()) throw failure(2, flag + " requires an integer");
    for (char c : value) {
        if (c < '0' || c > '9' || result > (std::numeric_limits<int>::max() - (c - '0')) / 10)
            throw failure(2, flag + " requires a non-negative integer within int range");
        result = result * 10 + (c - '0');
    }
    if (result < minimum) throw failure(2, flag + " must be at least " + std::to_string(minimum));
    return result;
}

options parse(int argc, char ** argv) {
    options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") { o.help = true; return o; }
        if (!seen.insert(flag).second) throw failure(2, "duplicate argument: " + flag);
        if (flag == "--verbose") { o.engine.verbose = true; continue; }
        if (flag != "--model" && flag != "--s3gen-gguf" && flag != "--backend" &&
            flag != "--text" && flag != "--text-file" && flag != "--out" &&
            flag != "--language" && flag != "--reference-audio" && flag != "--voice-dir" &&
            flag != "--threads" && flag != "--seed" && flag != "--n-ctx" &&
            flag != "--n-predict" && flag != "--kv-cache-type" && flag != "--max-sentence-chars" &&
            flag != "--cfm-steps" && flag != "--stream-chunk-tokens" &&
            flag != "--stream-first-chunk-tokens" && flag != "--stream-left-context-tokens" &&
            flag != "--stream-cfm-steps" && flag != "--warmup" && flag != "--runs" &&
            flag != "--duration-seconds" && flag != "--backends-dir" && flag != "--opencl-cache-dir")
            throw failure(2, "unknown argument: " + flag);
        if (++i == argc) throw failure(2, flag + " requires a value");
        const std::string v = argv[i];
        if      (flag == "--model") o.engine.t3_gguf_path = v;
        else if (flag == "--s3gen-gguf") o.engine.s3gen_gguf_path = v;
        else if (flag == "--backend") o.engine.backend = v;
        else if (flag == "--text") o.text = v;
        else if (flag == "--text-file") o.text_file = v;
        else if (flag == "--out") o.out = v;
        else if (flag == "--language") o.engine.language = v;
        else if (flag == "--reference-audio") o.engine.reference_audio = v;
        else if (flag == "--voice-dir") o.engine.voice_dir = v;
        else if (flag == "--backends-dir") o.engine.backends_dir = v;
        else if (flag == "--opencl-cache-dir") o.engine.opencl_cache_dir = v;
        else if (flag == "--kv-cache-type") o.engine.kv_cache_type = v;
        else if (flag == "--threads") o.engine.n_threads = integer(v, flag, 1);
        else if (flag == "--seed") o.engine.seed = integer(v, flag, 0);
        else if (flag == "--n-ctx") o.engine.n_ctx = integer(v, flag, 0);
        else if (flag == "--n-predict") o.engine.n_predict = integer(v, flag, 1);
        else if (flag == "--max-sentence-chars") o.engine.max_sentence_chars = integer(v, flag, 0);
        else if (flag == "--cfm-steps") o.engine.cfm_steps = integer(v, flag, 0);
        else if (flag == "--stream-chunk-tokens") o.engine.stream_chunk_tokens = integer(v, flag, 0);
        else if (flag == "--stream-first-chunk-tokens") o.engine.stream_first_chunk_tokens = integer(v, flag, 0);
        else if (flag == "--stream-left-context-tokens") o.engine.stream_left_context_tokens = integer(v, flag, 0);
        else if (flag == "--stream-cfm-steps") o.engine.stream_cfm_steps = integer(v, flag, 0);
        else if (flag == "--warmup") o.warmup = integer(v, flag, 0);
        else if (flag == "--runs") o.runs = integer(v, flag, 1);
        else {
            char * end = nullptr;
            errno = 0;
            o.duration_seconds = std::strtod(v.c_str(), &end);
            if (v.empty() || end == v.c_str() || *end || errno == ERANGE ||
                !std::isfinite(o.duration_seconds) || o.duration_seconds < 0 ||
                v.find_first_of(" \t\r\n") != std::string::npos)
                throw failure(2, "--duration-seconds requires a finite non-negative number");
        }
    }
    if (o.engine.t3_gguf_path.empty() || o.engine.s3gen_gguf_path.empty() || o.out.empty())
        throw failure(2, "--model, --s3gen-gguf and --out are required");
    if (o.engine.backend != "cpu" && o.engine.backend != "opencl" && o.engine.backend != "hexagon")
        throw failure(2, "--backend must explicitly name cpu, opencl or hexagon");
    if (seen.count("--text") == seen.count("--text-file"))
        throw failure(2, "choose exactly one of --text and --text-file");
    if (o.engine.language != "en" && o.engine.language != "fr" && o.engine.language != "es")
        throw failure(2, "--language must be en, fr or es");
    if (o.engine.kv_cache_type != "f32" && o.engine.kv_cache_type != "f16" && o.engine.kv_cache_type != "q8_0")
        throw failure(2, "--kv-cache-type must be f32, f16 or q8_0");
    if (!o.engine.stream_chunk_tokens && (o.engine.stream_first_chunk_tokens ||
        o.engine.stream_left_context_tokens || o.engine.stream_cfm_steps))
        throw failure(2, "stream options require positive --stream-chunk-tokens");
    if (seen.count("--text-file")) {
        std::ifstream in(o.text_file, std::ios::binary | std::ios::ate);
        const std::streamoff size = in ? in.tellg() : std::streampos(-1);
        if (size < 0 || size > 1024 * 1024) throw failure(2, "cannot read text file, or file exceeds 1 MiB");
        o.text.resize(static_cast<size_t>(size));
        in.seekg(0);
        if (size && !in.read(&o.text[0], size)) throw failure(2, "cannot read complete text file");
    }
    if (o.text.empty() || o.text.find_first_not_of(" \t\r\n") == std::string::npos ||
        o.text.find('\0') != std::string::npos)
        throw failure(2, "synthesis text must be non-empty and contain no NUL");
    o.engine.n_gpu_layers = o.engine.backend == "cpu" ? 0 : 99;
    return o;
}

// Preserve valid UTF-8; malformed filename bytes become U+FFFD in JSON only.
std::string quote(const std::string & value) {
    std::string out = "\"";
    const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < value.size();) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c < 0x80) {
            if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
            else if (c < 0x20) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
            else out += static_cast<char>(c);
            ++i;
            continue;
        }
        const size_t n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        bool valid = n && n <= value.size() - i;
        for (size_t j = 1; valid && j < n; ++j) {
            const unsigned char d = static_cast<unsigned char>(value[i + j]);
            valid = d >= 0x80 && d <= 0xbf && !(j == 1 &&
                ((c == 0xe0 && d < 0xa0) || (c == 0xed && d > 0x9f) ||
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
    out.imbue(std::locale::classic());
    out << std::setprecision(15) << value;
    return out.str();
}

const char * boolean(bool v) { return v ? "true" : "false"; }
double elapsed(clock_type::time_point start) { return std::chrono::duration<double>(clock_type::now() - start).count(); }

void emit(const std::string & line) {
    if (std::printf("%s\n", line.c_str()) < 0 || std::fflush(stdout) != 0)
        throw failure(8, "cannot write JSONL output");
}

std::string peak_rss_bytes() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) || usage.ru_maxrss < 0) return "null";
    const uint64_t value = static_cast<uint64_t>(usage.ru_maxrss);
#if defined(__APPLE__)
    return std::to_string(value);
#else
    return value <= UINT64_MAX / 1024 ? std::to_string(value * 1024) : "null";
#endif
#else
    return "null";
#endif
}

std::string environment() {
    std::string out = "{";
    for (const char * name : {"GGML_HEXAGON_NDEV", "GGML_HEXAGON_HOSTBUF", "GGML_HEXAGON_BF16",
         "GGML_HEXAGON_OPPOLL", "GGML_HEXAGON_OPSTAGE", "GGML_HEXAGON_PROFILE",
         "GGML_HEXAGON_OPTRACE", "GGML_SCHED_DEBUG", "TTS_CPP_FORCE_SCHED", "TTS_CPP_GPU_BACKEND"}) {
        if (out.size() > 1) out += ',';
        const char * value = std::getenv(name);
        out += quote(name) + ':' + (value ? quote(value) : "null");
    }
    return out + '}';
}

std::string settings(const options & o) {
    const EngineOptions & e = o.engine;
    return "{\"language\":" + quote(e.language) + ",\"threads\":" + std::to_string(e.n_threads) +
        ",\"seed\":" + std::to_string(e.seed) + ",\"n_ctx\":" + std::to_string(e.n_ctx) +
        ",\"n_predict\":" + std::to_string(e.n_predict) + ",\"kv_cache_type\":" + quote(e.kv_cache_type) +
        ",\"max_sentence_chars\":" + std::to_string(e.max_sentence_chars) + ",\"cfm_steps\":" + std::to_string(e.cfm_steps) +
        ",\"stream_chunk_tokens\":" + std::to_string(e.stream_chunk_tokens) +
        ",\"stream_first_chunk_tokens\":" + std::to_string(e.stream_first_chunk_tokens) +
        ",\"stream_left_context_tokens\":" + std::to_string(e.stream_left_context_tokens) +
        ",\"stream_cfm_steps\":" + std::to_string(e.stream_cfm_steps) +
        ",\"top_k\":" + std::to_string(e.top_k) + ",\"top_p\":" + number(e.top_p) +
        ",\"temperature\":" + number(e.temperature) + ",\"repeat_penalty\":" + number(e.repeat_penalty) +
        ",\"cfg_weight\":" + number(e.cfg_weight) + ",\"exaggeration\":" + number(e.exaggeration) +
        ",\"s3gen_cfg_rate\":" + number(e.s3gen_cfg_rate) + ",\"crossfade_ms\":" + std::to_string(e.crossfade_ms) +
        ",\"n_gpu_layers\":" + std::to_string(e.n_gpu_layers) + ",\"warmup_calls\":" + std::to_string(o.warmup) +
        ",\"measurement_calls\":" + std::to_string(o.runs) + ",\"first_call_calls\":1,\"sustained_seconds\":" +
        number(o.duration_seconds) + '}';
}

void ensure_new_output(const std::string & path) {
    const std::filesystem::path p(path);
    const auto parent = p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
    const auto object_exists = [](const std::filesystem::path & candidate) {
        std::error_code error;
        const auto status = std::filesystem::symlink_status(candidate, error);
        if (error && error != std::errc::no_such_file_or_directory)
            throw failure(6, "cannot inspect output path: " + candidate.string());
        // A dangling symlink is still an existing output object, never a
        // fresh destination into which this tool may follow and write.
        return status.type() != std::filesystem::file_type::not_found;
    };
    std::error_code error;
    if (object_exists(p) || object_exists(path + ".partial"))
        throw failure(6, "refusing an existing output or partial file: " + path);
    if (!std::filesystem::is_directory(parent, error) || error)
        throw failure(6, "output parent directory is missing or inaccessible: " + parent.string());
}

void put16(std::ostream & out, uint16_t value) {
    const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8)};
    out.write(bytes, 2);
}
void put32(std::ostream & out, uint32_t value) {
    const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8),
                          static_cast<char>(value >> 16), static_cast<char>(value >> 24)};
    out.write(bytes, 4);
}

void write_wav(const std::string & path, const SynthesisResult & result) {
    ensure_new_output(path);
    if (result.pcm.size() > (UINT32_MAX - 36) / 2 || result.sample_rate <= 0 ||
        static_cast<uint64_t>(result.sample_rate) > UINT32_MAX / 2)
        throw failure(6, "audio exceeds PCM16 RIFF/WAVE limits");
    const std::string partial = path + ".partial";
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    if (!out) throw failure(6, "cannot create output WAV: " + partial);
    const uint32_t bytes = static_cast<uint32_t>(result.pcm.size() * 2);
    out.write("RIFF", 4); put32(out, 36 + bytes); out.write("WAVEfmt ", 8);
    put32(out, 16); put16(out, 1); put16(out, 1);
    put32(out, static_cast<uint32_t>(result.sample_rate)); put32(out, static_cast<uint32_t>(result.sample_rate) * 2);
    put16(out, 2); put16(out, 16); out.write("data", 4); put32(out, bytes);
    for (float sample : result.pcm) {
        const double bounded = std::max(-1.0, std::min(1.0, static_cast<double>(sample)));
        const auto encoded = static_cast<int16_t>(std::lround(bounded * 32767.0));
        put16(out, static_cast<uint16_t>(encoded));
    }
    out.close();
    if (!out) throw failure(6, "cannot complete output WAV: " + partial);
    std::error_code error;
    std::filesystem::rename(partial, path, error);
    if (error) throw failure(6, "cannot publish output WAV: " + path + ": " + error.message());
}

struct audio_stats {
    size_t non_finite = 0, clipped = 0;
    double peak = 0, rms = 0;
};

audio_stats inspect_audio(const std::vector<float> & pcm) {
    audio_stats s;
    double squares = 0;
    for (float v : pcm) {
        if (!std::isfinite(v)) { ++s.non_finite; continue; }
        const double a = std::fabs(static_cast<double>(v));
        s.peak = std::max(s.peak, a);
        if (a > 1) ++s.clipped;
        squares += static_cast<double>(v) * v;
    }
    s.rms = pcm.empty() ? 0 : std::sqrt(squares / pcm.size());
    return s;
}

bool primary_matches(const options & o, const Engine & engine, const std::string & actual) {
    if (actual.empty() || actual == "(unknown)") return false;
    if (o.engine.backend == "cpu") return engine.backend_device() == tts_cpp::BackendDevice::CPU;
    if (o.engine.backend == "hexagon") return engine.backend_device() == tts_cpp::BackendDevice::GPU && actual == "HTP0";
    return engine.backend_device() == tts_cpp::BackendDevice::GPU && actual.compare(0, 6, "OpenCL") == 0;
}

int benchmark(const options & o) {
    const auto process_start = clock_type::now();
    ensure_new_output(o.out);
    emit("{\"kind\":\"config\",\"schema_version\":1,\"model\":" + quote(o.engine.t3_gguf_path) +
         ",\"s3gen_gguf\":" + quote(o.engine.s3gen_gguf_path) + ",\"requested_backend\":" + quote(o.engine.backend) +
         ",\"text\":" + quote(o.text) + ",\"text_file\":" + quote(o.text_file) + ",\"output_path\":" + quote(o.out) +
         ",\"reference_audio\":" + quote(o.engine.reference_audio) + ",\"voice_dir\":" + quote(o.engine.voice_dir) +
         ",\"backends_dir\":" + quote(o.engine.backends_dir) + ",\"settings\":" + settings(o) +
         ",\"environment\":" + environment() + ",\"backend_execution_confirmed\":false," +
         "\"time_basis\":\"steady_clock_native_call\",\"output_encoding\":\"pcm_s16le\"}");
    std::unique_ptr<Engine> engine;
    const auto load_start = clock_type::now();
    try { engine = std::make_unique<Engine>(o.engine); }
    catch (const std::exception & e) { throw failure(5, std::string("Engine construction failed: ") + e.what()); }
    const double constructor_s = elapsed(load_start);
    const std::string actual = engine->backend_name();
    const bool matches = primary_matches(o, *engine, actual);
    emit("{\"kind\":\"load\",\"constructor_s\":" + number(constructor_s) +
         ",\"preload_s\":null,\"constructor_includes_all_model_loading\":false," +
         "\"preload_completion\":\"first_call_boundary\",\"actual_primary_backend\":" + quote(actual) +
         ",\"primary_backend_matches_request\":" + boolean(matches) +
         ",\"backend_execution_confirmed\":false,\"peak_rss_bytes\":" + peak_rss_bytes() + '}');
    if (!matches) throw failure(4, "reported primary backend does not match explicit request: " + actual);

    const bool streaming = o.engine.stream_chunk_tokens > 0;
    uint64_t total_calls = 0, sustained_calls = 0;
    double sustained_native_s = 0;
    std::string last_output;
    auto run = [&](const char * phase, uint64_t index, clock_type::time_point phase_start, double & phase_native_s) {
        const uint64_t call = ++total_calls;
        const double native_begin = phase_native_s;
        const double wall_begin = elapsed(phase_start);
        std::vector<float> callback_pcm;
        int chunks = 0, last_chunks = 0;
        bool callback_order_ok = true, saw_last = false;
        double first_audio_s = -1;
        SynthesisResult result;
        std::fprintf(stderr, "chatterbox-hexagon: begin phase=%s index=%llu\n", phase, static_cast<unsigned long long>(index));
        const auto start = clock_type::now();
        try {
            if (streaming) {
                result = engine->synthesize(o.text, [&](const float * pcm, size_t samples, int chunk_index, bool is_last) {
                    if (samples && first_audio_s < 0) first_audio_s = elapsed(start);
                    if (chunk_index != chunks || saw_last || (samples && !pcm)) callback_order_ok = false;
                    if (samples && !pcm) throw failure(6, "stream callback has a null PCM buffer");
                    if (samples > callback_pcm.max_size() - callback_pcm.size()) throw failure(6, "stream PCM size overflows");
                    if (samples) callback_pcm.insert(callback_pcm.end(), pcm, pcm + samples);
                    ++chunks;
                    if (is_last) { ++last_chunks; saw_last = true; }
                });
            } else result = engine->synthesize(o.text);
        } catch (const std::exception & e) {
            const double seconds = elapsed(start);
            phase_native_s += seconds;
            emit("{\"kind\":\"run\",\"status\":\"error\",\"phase\":" + quote(phase) +
                 ",\"index\":" + std::to_string(index) + ",\"call\":" + std::to_string(call) +
                 ",\"native_s\":" + number(seconds) + ",\"phase_begin_s\":" + number(native_begin) +
                 ",\"phase_end_s\":" + number(phase_native_s) + ",\"message\":" + quote(e.what()) + '}');
            throw failure(7, std::string("Engine synthesis failed: ") + e.what());
        }
        const double native_s = elapsed(start);
        phase_native_s += native_s;
        const double wall_end = elapsed(phase_start);
        const audio_stats audio = inspect_audio(result.pcm);
        const bool stream_matches = !streaming || callback_pcm == result.pcm;
        const bool stream_ok = !streaming || (chunks > 0 && last_chunks == 1 && saw_last && callback_order_ok &&
                                              stream_matches && first_audio_s >= 0 && first_audio_s <= native_s);
        const bool audio_ok = !result.pcm.empty() && audio.non_finite == 0 && result.sample_rate > 0 &&
                              result.audio_samples >= 0 && static_cast<size_t>(result.audio_samples) == result.pcm.size() &&
                              std::isfinite(result.t3_ms) && result.t3_ms >= 0 &&
                              std::isfinite(result.s3gen_ms) && result.s3gen_ms >= 0 && result.t3_tokens > 0 && native_s > 0;
        const double audio_s = result.sample_rate > 0 ? static_cast<double>(result.pcm.size()) / result.sample_rate : 0;
        const std::string path = o.out + '.' + phase + '-' + std::to_string(index) + ".wav";
        // A failed numerical/audio contract never publishes a success WAV.
        if (audio_ok && stream_ok) write_wav(path, result);
        emit("{\"kind\":\"run\",\"status\":" + quote(audio_ok && stream_ok ? "ok" : "error") +
             ",\"phase\":" + quote(phase) + ",\"index\":" + std::to_string(index) + ",\"call\":" + std::to_string(call) +
             ",\"native_s\":" + number(native_s) + ",\"t3_s\":" + number(result.t3_ms / 1000.0) +
             ",\"s3gen_s\":" + number(result.s3gen_ms / 1000.0) + ",\"samples\":" + std::to_string(result.pcm.size()) +
             ",\"sample_rate\":" + std::to_string(result.sample_rate) + ",\"audio_s\":" + number(audio_s) +
             ",\"rtf\":" + number(audio_s > 0 ? native_s / audio_s : std::numeric_limits<double>::quiet_NaN()) +
             ",\"audio_seconds_per_native_second\":" + number(audio_s / native_s) +
             ",\"first_audio_s\":" + (streaming ? number(first_audio_s) : "null") +
             ",\"streaming\":" + boolean(streaming) + ",\"stream_chunks\":" + std::to_string(chunks) +
             ",\"last_chunks\":" + std::to_string(last_chunks) + ",\"stream_order_ok\":" + boolean(callback_order_ok) +
             ",\"stream_pcm_matches\":" + boolean(stream_matches) + ",\"stream_contract_ok\":" + boolean(stream_ok) +
             ",\"audio_contract_ok\":" + boolean(audio_ok) + ",\"non_finite_samples\":" + std::to_string(audio.non_finite) +
             ",\"clipped_samples\":" + std::to_string(audio.clipped) + ",\"peak_abs\":" + number(audio.peak) +
             ",\"rms\":" + number(audio.rms) + ",\"t3_tokens\":" + std::to_string(result.t3_tokens) +
             ",\"termination_status\":\"not_exposed_by_public_api\",\"token_limit_suspected\":" +
             boolean(result.t3_tokens >= o.engine.n_predict) +
             ",\"may_include_lazy_loading\":" + boolean(std::strcmp(phase, "first_call") == 0) +
             ",\"phase_begin_s\":" + number(native_begin) + ",\"phase_end_s\":" + number(phase_native_s) +
             ",\"phase_wall_begin_s\":" + number(wall_begin) + ",\"phase_wall_end_s\":" + number(wall_end) +
             ",\"output_path\":" + (audio_ok && stream_ok ? quote(path) : "null") +
             ",\"actual_primary_backend\":" + quote(actual) + ",\"backend_execution_confirmed\":false," +
             "\"peak_rss_bytes\":" + peak_rss_bytes() + '}');
        if (!audio_ok || !stream_ok) throw failure(6, "audio or streaming callback contract failed");
        last_output = path;
        std::fprintf(stderr, "chatterbox-hexagon: end phase=%s index=%llu native_s=%.9f\n", phase,
                     static_cast<unsigned long long>(index), native_s);
    };

    double phase_native_s = 0;
    auto phase_start = clock_type::now();
    run("first_call", 1, phase_start, phase_native_s);
    phase_native_s = 0; phase_start = clock_type::now();
    for (int i = 0; i < o.warmup; ++i) run("warmup", static_cast<uint64_t>(i) + 1, phase_start, phase_native_s);
    phase_native_s = 0; phase_start = clock_type::now();
    for (int i = 0; i < o.runs; ++i) run("measure", static_cast<uint64_t>(i) + 1, phase_start, phase_native_s);
    phase_start = clock_type::now();
    while (sustained_native_s < o.duration_seconds)
        run("sustained", ++sustained_calls, phase_start, sustained_native_s);
    ensure_new_output(o.out);
    std::error_code copy_error;
    if (!std::filesystem::copy_file(last_output, o.out, copy_error) || copy_error)
        throw failure(6, "cannot publish final output WAV: " + copy_error.message());
    emit("{\"kind\":\"end\",\"complete\":true,\"exit_code\":0,\"total_calls\":" + std::to_string(total_calls) +
         ",\"first_call_calls\":1,\"warmup_calls\":" + std::to_string(o.warmup) + ",\"measurement_calls\":" + std::to_string(o.runs) +
         ",\"sustained_calls\":" + std::to_string(sustained_calls) + ",\"sustained_native_s\":" + number(sustained_native_s) +
         ",\"output_path\":" + quote(o.out) + ",\"final_run_output_path\":" + quote(last_output) +
         ",\"driver_elapsed_s\":" + number(elapsed(process_start)) + ",\"peak_rss_bytes\":" + peak_rss_bytes() + '}');
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const options o = parse(argc, argv);
        if (o.help) { usage(); return 0; }
        return benchmark(o);
    } catch (const failure & e) {
        std::fprintf(stderr, "chatterbox-hexagon: %s\n", e.what());
        std::printf("{\"kind\":\"error\",\"complete\":false,\"exit_code\":%d,\"message\":%s}\n", e.code, quote(e.what()).c_str());
        return e.code;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "chatterbox-hexagon: %s\n", e.what());
        std::printf("{\"kind\":\"error\",\"complete\":false,\"exit_code\":8,\"message\":%s}\n", quote(e.what()).c_str());
        return 8;
    }
}
