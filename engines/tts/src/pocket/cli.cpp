#include "tts-cpp/pocket/engine.h"
#include "tts-cpp/pocket/fit.h"
#include "fit_util.h"
#define DR_WAV_IMPLEMENTATION
#define DRWAV_API static
#define DRWAV_PRIVATE static
#include "dr_wav.h"
#include "json.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace {
using namespace tts_cpp::pocket;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
struct Arguments {
    std::map<std::string, std::string> values;
    bool has(const std::string & key) const { return values.count(key) != 0; }
    std::string get(const std::string & key, const std::string & fallback) const {
        const auto it = values.find(key); return it == values.end() ? fallback : it->second;
    }
    int integer(const std::string & key, int fallback) const {
        const auto s = get(key, std::to_string(fallback)); size_t end;
        const int n = std::stoi(s, &end);
        if (end != s.size()) throw std::runtime_error("invalid integer: "+key);
        return n;
    }
    uint64_t mib(const std::string & key, int fallback) const {
        const int value = integer(key, fallback);
        if (value < 0) throw std::runtime_error("memory size must be nonnegative: "+key);
        return tts_cpp::fitutil::margin_mib_to_bytes(uint64_t(value));
    }
};
Arguments parse_arguments(int argc, char ** argv) {
    Arguments args;
    const std::set<std::string> accepted = {"--model-dir","--flow-lm","--mimi","--frontend","--voice","--reference-audio",
        "--text","--output","--threads","--steps","--temperature","--seed","--runs","--report","--context","--max-tokens",
        "--frames-after-eos","--fit-budget-mib","--fit-margin-mib","--backends-dir","--backend"};
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { args.values[key] = "true"; return args; }
        if (key == "--fit" || key == "--gpu") {
            if (args.has(key)) throw std::runtime_error("duplicate option: "+key);
            args.values[key] = "true"; continue;
        }
        if (!accepted.count(key) || i+1 == argc || args.has(key)) throw std::runtime_error("invalid or duplicate option: "+key);
        args.values[key] = argv[++i];
    }
    return args;
}
void print_help() {
    std::puts("pocket-cli --model-dir DIR --text TEXT --output FILE.wav [--voice FILE.gguf | --reference-audio FILE.wav]\n"
              "           [--threads 1 --steps 1 --temperature 0.3 --seed 1234 --runs 1 --report FILE.json]\n"
              "           [--gpu] [--backends-dir DIR] [--frames-after-eos -1]\n"
              "--gpu selects Metal, CUDA, Vulkan or OpenCL for FlowLM and Mimi; CPU is used when --gpu is omitted.\n"
              "--backend cpu|opencl explicitly selects a backend.\n"
              "--frames-after-eos -1 uses the checkpoint/text default; explicit range 0..100.\n"
              "DIR contains flow-lm.gguf, mimi.gguf, frontend.json, voice.gguf. Individual asset paths can override DIR.\n"
              "--runs greater than 1 includes one untimed warmup and records each warm run.\n"
              "--fit estimates memory on the selected backend without loading weights; optional --fit-budget-mib and --fit-margin-mib.\n"
              "Fit exit status: 0 fits, 1 does not fit, 2 measurement error.");
}
EngineOptions engine_options(const Arguments & args) {
    const auto dir = args.get("--model-dir", ".");
    EngineOptions opts;
    opts.use_gpu = args.has("--gpu");
    opts.backends_dir = args.get("--backends-dir", "");
    opts.backend = args.get("--backend", "");
    opts.flow_lm_path = args.get("--flow-lm", dir+"/flow-lm.gguf");
    opts.mimi_path = args.get("--mimi", dir+"/mimi.gguf");
    opts.frontend_path = args.get("--frontend", dir+"/frontend.json");
    opts.reference_audio_path = args.get("--reference-audio", "");
    opts.voice_path = args.get("--voice", opts.reference_audio_path.empty() ? dir+"/voice.gguf" : "");
    opts.n_threads = args.integer("--threads", opts.n_threads); opts.steps = args.integer("--steps", opts.steps);
    opts.context = args.integer("--context", opts.context); opts.max_tokens = args.integer("--max-tokens", opts.max_tokens);
    opts.frames_after_eos = args.integer("--frames-after-eos", opts.frames_after_eos);
    const int seed = args.integer("--seed", opts.seed);
    if (seed < 0) throw std::runtime_error("seed must be nonnegative");
    opts.seed = seed;
    size_t end; const auto temperature = args.get("--temperature", std::to_string(opts.temperature));
    opts.temperature = std::stof(temperature, &end);
    if (end != temperature.size()) throw std::runtime_error("invalid temperature");
    return opts;
}
void write_report(const std::string & path, const Json & report) {
    if (path.empty()) return;
    std::ofstream file(path); file << report.dump(2) << '\n';
    if (!file) throw std::runtime_error("cannot write report");
}
int run_fit(const Arguments & args, const EngineOptions & opts, const std::string & text) {
    FitOptions fit; fit.engine = opts; fit.text = text;
    fit.margin_bytes = args.mib("--fit-margin-mib", 256);
    fit.memory_budget_bytes = args.mib("--fit-budget-mib", 0);
    const auto result = fit_params(fit);
    std::puts(result.report.c_str());
    write_report(args.get("--report", ""), {
        {"status", int(result.status)}, {"fits", result.fits}, {"reason", result.reason},
        {"backend", result.device_name}, {"device_is_cpu", result.device_is_cpu},
        {"device_shares_host_memory", result.device_shares_host_memory},
        {"weights_bytes", result.device.weights_bytes}, {"state_bytes", result.device.state_bytes},
        {"lm_compute_bytes", result.device.lm_compute_bytes}, {"codec_compute_bytes", result.device.codec_compute_bytes},
        {"device_bytes", result.device.total_bytes}, {"host_bytes", result.host_bytes},
        {"available_bytes", result.device_free_bytes}, {"total_memory_bytes", result.device_total_bytes},
        {"margin_bytes", fit.margin_bytes}, {"report", result.report}});
    return int(result.status);
}
void write_wav(const std::string & path, const SynthesisResult & result) {
    drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_IEEE_FLOAT, 1, drwav_uint32(result.sample_rate), 32};
    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) throw std::runtime_error("cannot open output WAV");
    const auto written = drwav_write_pcm_frames(&wav, result.pcm.size(), result.pcm.data());
    drwav_uninit(&wav);
    if (written != result.pcm.size()) throw std::runtime_error("incomplete WAV write");
}
double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
Json benchmark_once(Engine & engine, const std::string & text, const std::string & output) {
    const auto start = Clock::now();
    std::vector<float> pcm; double first_audio = 0;
    auto result = engine.synthesize_stream(text, [&](const float * chunk, size_t n, int) {
        if (pcm.empty()) first_audio = seconds_since(start);
        pcm.insert(pcm.end(), chunk, chunk+n); return true;
    });
    result.pcm = std::move(pcm);
    const double elapsed = seconds_since(start);
    const double duration = double(result.pcm.size())/result.sample_rate;
    Json measurement = {{"seconds",elapsed},{"first_audio_seconds",first_audio},
                        {"audio_seconds",duration},{"real_time_factor",elapsed/duration}};
    std::puts(measurement.dump().c_str());
    if (!output.empty()) write_wav(output, result);
    return measurement;
}
Json benchmark_runs(Engine & engine, const std::string & text, const std::string & output, int runs) {
    Json measurements = Json::array();
    for (int i = 0; i < runs; ++i) measurements.push_back(benchmark_once(engine, text, i == 0 ? output : ""));
    return measurements;
}
int run_synthesis(const Arguments & args, const EngineOptions & opts, const std::string & text) {
    const int runs = args.integer("--runs", 1);
    if (runs < 1 || runs > 100) throw std::runtime_error("runs must be 1..100");
    const auto start = Clock::now(); Engine engine(opts);
    const double load = seconds_since(start);
    std::fprintf(stderr, "Pocket FlowLM: %s; Mimi: %s; device: %s\n", engine.flow_lm_backend_name(), engine.mimi_backend_name(), engine.device_name());
    if (runs > 1) engine.synthesize("A short warm up.");
    const auto measurements = benchmark_runs(engine, text, args.get("--output", "pocket-output.wav"), runs);
    write_report(args.get("--report", ""), {
        {"implementation","Fabric native ggml"},{"backend",engine.backend_name()},
        {"flow_lm_backend",engine.flow_lm_backend_name()},{"mimi_backend",engine.mimi_backend_name()},
        {"device",engine.device_name()},
        {"threads_per_worker",opts.n_threads},{"worker_threads",2},{"steps",opts.steps},{"temperature",opts.temperature},
        {"frames_after_eos",opts.frames_after_eos},
        {"seed",opts.seed},{"text",text},{"load_seconds",load},{"warmups",runs>1?1:0},{"runs",measurements},
        {"source_sha256",engine.model_source_sha256()},{"sample_rate",engine.sample_rate()},
        {"compute_precision","float32"},{"rng","Fabric MT19937 Box-Muller (not PyTorch seed-equivalent)"},
        {"assets",{{"flow_lm",opts.flow_lm_path},{"mimi",opts.mimi_path},{"frontend",opts.frontend_path},{"voice",opts.voice_path}}}});
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        if (args.has("--help")) { print_help(); return 0; }
        const auto opts = engine_options(args);
        const auto text = args.get("--text", "");
        if (text.empty()) throw std::runtime_error("--text is required");
        return args.has("--fit") ? run_fit(args, opts, text) : run_synthesis(args, opts, text);
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
