#include "minimax/backend.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"
#include "minimax/mm3-replay-io.h"
#include "minimax/request-utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct CliOptions {
    std::string models;
    std::string out_dir;
    std::string mode;
    std::string device;
    std::string coreml_conditioner;
#ifdef MM3_ANE_VALIDATION
    std::string ane_validation;
    std::string ane_compute_units;
#endif
    std::string caption;
    std::string lyrics;
    std::string tokens_path;
    std::string semantic_path;
    std::string acoustic_path;
    std::vector<std::string> noise_paths;
    std::string dump_dir;
    uint64_t seed = 0;
    int steps = 0;
    int64_t max_frames = 0;
    int threads = 0;
    int64_t dump_iters = 0;
};

const char * arg_value(int argc, char ** argv, const char * name, const char * fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<std::string> collect_noise_paths(int argc, char ** argv) {
    std::vector<std::string> paths;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--noise") == 0) {
            paths.emplace_back(argv[i + 1]);
        }
    }
    return paths;
}

CliOptions parse_options(int argc, char ** argv) {
    CliOptions options;
    options.models        = arg_value(argc, argv, "--models", "");
    options.out_dir       = arg_value(argc, argv, "--out", ".");
    options.mode          = arg_value(argc, argv, "--mode", "full");
    options.device        = arg_value(argc, argv, "--device", "");
    options.coreml_conditioner = arg_value(argc, argv, "--coreml-cond", "");
#ifdef MM3_ANE_VALIDATION
    options.ane_validation = arg_value(argc, argv, "--ane-validation", "");
    options.ane_compute_units = arg_value(argc, argv, "--ane-compute-units", "CPU_AND_NE");
    if (options.ane_compute_units != "CPU_ONLY" && options.ane_compute_units != "CPU_AND_GPU" &&
        options.ane_compute_units != "CPU_AND_NE" && options.ane_compute_units != "ALL") {
        throw std::invalid_argument("--ane-compute-units requires CPU_ONLY, CPU_AND_GPU, CPU_AND_NE or ALL");
    }
#endif
    options.caption       = arg_value(argc, argv, "--caption", "");
    options.lyrics        = arg_value(argc, argv, "--lyrics", "");
    options.tokens_path   = arg_value(argc, argv, "--tokens", "");
    options.semantic_path = arg_value(argc, argv, "--semantic", "");
    options.acoustic_path = arg_value(argc, argv, "--acoustic", "");
    options.noise_paths   = collect_noise_paths(argc, argv);
    options.seed          = (uint64_t) atoll(arg_value(argc, argv, "--seed", "42"));
    const std::string default_steps = std::to_string(tts_cpp::minimax::detail::kDefaultFlowSteps);
    options.steps         = atoi(arg_value(argc, argv, "--steps", default_steps.c_str()));
    options.max_frames    = atoll(arg_value(argc, argv, "--max-frames", "300"));
    options.threads       = atoi(arg_value(argc, argv, "--threads", "0"));
    options.dump_iters    = atoll(arg_value(argc, argv, "--dump-iters", "0"));
    options.dump_dir      = arg_value(argc, argv, "--dump-dir", options.out_dir.c_str());
    return options;
}

void print_usage() {
    fprintf(stderr,
            "usage: mm3-replay --models <dir> --out <dir> --mode replay|full|condcheck\n"
            "  replay:    --tokens <i32 file> --semantic <i32 file> --acoustic <i32 file>\n"
            "             [--noise <f32 file>]... (per window)\n"
            "  full:      --caption <text> [--lyrics <text>]\n"
            "  condcheck: verify the DiT emits byte-identical velocities across\n"
            "             repeated computes with interleaved CFG branches\n"
            "  common:    [--seed N] [--steps N] [--max-frames N] [--threads N]\n"
            "             [--device cpu|gpu|auto]\n"
            "             [--coreml-cond <conditioner.mlmodelc>] CPU + ANE projection\n"
            "             [--dump-iters N [--dump-dir <dir>]] write per-iteration AR\n"
            "             probes (semantic logits, guided logits, hiddens, feedback)\n"
            "             for the first N iterations as raw f32 files\n");
#ifdef MM3_ANE_VALIDATION
    fprintf(stderr,"  experimental: --ane-validation <profile> [--ane-compute-units CPU_ONLY|CPU_AND_GPU|CPU_AND_NE|ALL]\n");
#endif
}

template <typename T>
std::vector<T> read_raw_or_exit(const std::string & path) {
    std::vector<T> data;
    if (!mm3_replay_read_raw(path, data)) {
        fprintf(stderr, "cannot read %s (missing, empty, or not element-aligned)\n", path.c_str());
        exit(1);
    }
    return data;
}

bool load_model(const CliOptions & options, MM3Model & model, std::string * error) {
    tts_cpp::minimax::detail::ModelPair pair;
    try {
        pair = tts_cpp::minimax::detail::resolve_model_pair(options.models, "", "");
    } catch (const std::exception & e) {
        *error = e.what();
        return false;
    }
    model.models_dir = options.models;
    model.coreml_conditioner_path = options.coreml_conditioner;
#ifdef MM3_ANE_VALIDATION
    model.ane_validation_directory = options.ane_validation;
    model.ane_validation_compute_units = options.ane_compute_units;
#endif
    mm3_probe_file(pair.lm, &model.lm_file, &model.lm_cfg, nullptr, &model.meta_errors);
    mm3_probe_file(pair.synth, &model.synth_file, nullptr, &model.synth_cfg, &model.meta_errors);
    if (!model.meta_errors.empty()) {
        *error = model.meta_errors.front();
        return false;
    }
    return mm3_load(&model, error);
}

struct VelocityDivergence {
    double cosine = 0.0;
    double max_abs = 0.0;
};

VelocityDivergence velocity_divergence(const std::vector<float> & first, const std::vector<float> & repeat) {
    double dot = 0.0, first_norm = 0.0, repeat_norm = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < first.size(); ++i) {
        dot += (double) first[i] * repeat[i];
        first_norm += (double) first[i] * first[i];
        repeat_norm += (double) repeat[i] * repeat[i];
        max_abs = std::max(max_abs, (double) std::fabs(first[i] - repeat[i]));
    }
    return {dot / std::sqrt(first_norm * repeat_norm), max_abs};
}

// The DiT must emit byte-identical velocities for identical inputs across
// repeated graph computes: run the production CFG pair, an interleaved pass
// with the branch gates swapped, then the CFG pair again, and require run 3 ==
// run 1 for both branches. This regressed once when the flow loop relied on a
// resident condition upload that the graph allocator had recycled between
// computes.
int run_condcheck(const MM3Model & model) {
    const int64_t L = 128;
    const int64_t N = (int64_t) model.synth_cfg.dit.in_channels * L;
    const int64_t CN = (int64_t) model.synth_cfg.dit.condition_dim * L;
    std::vector<float> latents, condition;
    tts_cpp::minimax::detail::fill_noise(1234, 0, latents, N);
    tts_cpp::minimax::detail::fill_noise(1234, 1, condition, CN);
    std::vector<float> first((size_t) N), first_unconditional((size_t) N);
    std::vector<float> swapped((size_t) N), swapped_unconditional((size_t) N);
    std::vector<float> repeat((size_t) N), repeat_unconditional((size_t) N);
    const std::vector<MM3DitBranch> swapped_branches = {
        { MM3_DIT_UNCONDITIONED_GATE, swapped_unconditional.data() },
        { MM3_DIT_CONDITIONED_GATE, swapped.data() },
    };
    std::string error;
    if (!mm3_dit_prepare(model, &g_mm3_dit, &error) ||
        !mm3_dit_run_cfg(model, &g_mm3_dit, latents.data(), condition.data(), 0.5f, L, first.data(),
                         first_unconditional.data(), &error) ||
        !mm3_dit_run(model, &g_mm3_dit, latents.data(), condition.data(), swapped_branches, 0.5f, L, &error) ||
        !mm3_dit_run_cfg(model, &g_mm3_dit, latents.data(), condition.data(), 0.5f, L, repeat.data(),
                         repeat_unconditional.data(), &error)) {
        fprintf(stderr, "condcheck error: %s\n", error.c_str());
        return 1;
    }
    const VelocityDivergence divergence = velocity_divergence(first, repeat);
    const VelocityDivergence unconditional_divergence =
        velocity_divergence(first_unconditional, repeat_unconditional);
    fprintf(stderr, "[condcheck] cos(first,repeat)=%.9f max_abs_diff=%.6g, unconditional max_abs_diff=%.6g\n",
            divergence.cosine, divergence.max_abs, unconditional_divergence.max_abs);
    return divergence.max_abs < 1e-4 && unconditional_divergence.max_abs < 1e-4 ? 0 : 2;
}

MM3GenRequest build_base_request(const CliOptions & options, const MM3Model & model) {
    MM3GenRequest request;
    request.seed = options.seed;
    request.steps = options.steps;
    request.max_frames = options.max_frames;
    request.cfg_flow = model.synth_cfg.flow.cfg_scale > 0 ? model.synth_cfg.flow.cfg_scale : 1.7f;
    request.keep_window_latents = true;
    request.dump_iters = options.dump_iters;
    return request;
}

void load_forced_noise(const std::vector<std::string> & paths, MM3GenRequest & request) {
    for (const std::string & path : paths) {
        request.forced_noise.push_back(read_raw_or_exit<float>(path));
    }
}

MM3GenRequest build_replay_request(const CliOptions & options, const MM3Model & model) {
    MM3GenRequest request = build_base_request(options, model);
    request.ids_cond = read_raw_or_exit<int32_t>(options.tokens_path);
    request.forced_semantic = read_raw_or_exit<int32_t>(options.semantic_path);
    request.forced_acoustic = read_raw_or_exit<int32_t>(options.acoustic_path);
    request.max_frames = (int64_t) request.forced_semantic.size();
    load_forced_noise(options.noise_paths, request);
    fprintf(stderr, "[replay] %zu prompt tokens, %zu frames, %zu noise windows\n",
            request.ids_cond.size(), request.forced_semantic.size(), request.forced_noise.size());
    return request;
}

MM3GenRequest build_full_request(const CliOptions & options, const MM3Model & model) {
    MM3GenRequest request = build_base_request(options, model);
    request.prompt = tts_cpp::minimax::detail::build_prompt(options.caption, options.lyrics);
    fprintf(stderr, "[full] prompt: %s\n", request.prompt.c_str());
    return request;
}

void log_progress(const MM3GenProgress & progress) {
    fprintf(stderr, "[progress] %s %lld/%lld window %lld/%lld\n", progress.stage,
            (long long) progress.step, (long long) progress.n_steps, (long long) progress.window,
            (long long) progress.n_windows);
}

bool write_window_latents(const std::string & out_dir, const std::vector<std::vector<float>> & latents) {
    for (size_t window = 0; window < latents.size(); ++window) {
        if (!mm3_replay_write_raw(out_dir + "/window-" + std::to_string(window) + ".f32",
                                  latents[window].data(), latents[window].size())) {
            return false;
        }
    }
    return true;
}

bool write_artifacts(const std::string & out_dir, const MM3GenResult & result) {
    return mm3_replay_write_wav(out_dir + "/audio.wav", result.audio, result.n_samples,
                                result.sample_rate) &&
           write_window_latents(out_dir, result.window_latents) &&
           mm3_replay_write_raw(out_dir + "/tokens.i32", result.ids_cond_used.data(),
                                result.ids_cond_used.size()) &&
           mm3_replay_write_raw(out_dir + "/frame-hiddens.f32", result.ar.frame_hiddens.data(),
                                result.ar.frame_hiddens.size()) &&
           mm3_replay_write_raw(out_dir + "/semantic.i32", result.ar.semantic_all.data(),
                                result.ar.semantic_all.size()) &&
           mm3_replay_write_raw(out_dir + "/acoustic.i32", result.ar.acoustic_all.data(),
                                result.ar.acoustic_all.size());
}

bool write_ar_dump(const std::string & dir, size_t index, const MM3ArDump & dump) {
    const std::string base = dir + "/ar-iter-" + std::to_string(index) + "-";
    return mm3_replay_write_raw(base + "last-hidden.f32", dump.last_hidden.data(), dump.last_hidden.size()) &&
           mm3_replay_write_raw(base + "sem-logits.f32", dump.sem_logits.data(), dump.sem_logits.size()) &&
           mm3_replay_write_raw(base + "guided.f32", dump.guided.data(), dump.guided.size()) &&
           mm3_replay_write_raw(base + "feedback.f32", dump.feedback.data(), dump.feedback.size()) &&
           mm3_replay_write_raw(base + "depth-hidden.f32", dump.depth_hidden.data(), dump.depth_hidden.size());
}

bool write_ar_dumps(const CliOptions & options, const std::vector<MM3ArDump> & dumps) {
    if (options.dump_iters <= 0) {
        return true;
    }
    std::string error;
    if (options.dump_dir != options.out_dir &&
        !mm3_replay_prepare_output_dir(options.dump_dir, &error)) {
        fprintf(stderr, "dump error: %s\n", error.c_str());
        return false;
    }
    for (size_t index = 0; index < dumps.size(); ++index) {
        if (!write_ar_dump(options.dump_dir, index, dumps[index])) {
            return false;
        }
    }
    fprintf(stderr, "[dump] %zu AR iterations under %s\n", dumps.size(), options.dump_dir.c_str());
    return true;
}

void report_done(const MM3GenResult & result) {
    fprintf(stderr,
            "[done] frames=%lld windows=%lld samples=%lld peak=%.3f rms=%.4f "
            "ar=%.0fms cond=%.0fms flow=%.0fms voc=%.0fms total=%.0fms\n",
            (long long) result.frames, (long long) result.n_windows, (long long) result.n_samples,
            result.peak, result.rms, result.ar_ms, result.cond_ms, result.flow_ms, result.voc_ms,
            result.total_ms);
}

int run_generation(const CliOptions & options, const MM3Model & model) {
    const MM3GenRequest request = options.mode == "replay" ? build_replay_request(options, model)
                                                           : build_full_request(options, model);
    MM3Tokenizer tokenizer;
    MM3GenResult result;
    std::string error;
    if (!mm3_generate(model, request, &tokenizer, log_progress, &result, &error)) {
        fprintf(stderr, "generate error: %s\n", error.c_str());
        return 1;
    }
    if (!write_artifacts(options.out_dir, result)) {
        fprintf(stderr, "output error: cannot write artifacts under %s\n", options.out_dir.c_str());
        return 1;
    }
    if (!write_ar_dumps(options, result.ar.dumps)) {
        return 1;
    }
    report_done(result);
#ifdef MM3_ANE_VALIDATION
    if (model.ane_validation) {
        std::ofstream timings(options.out_dir+"/ane-timings.json");
        timings << "{\n  \"compute_units\": \"" << model.ane_validation_compute_units << "\",\n"
                << "  \"load_ms\": " << model.load_ms << ",\n"
                << "  \"generation_ms\": " << result.total_ms << ",\n"
                << "  \"ar_ms\": " << result.ar_ms << ",\n"
                << "  \"condition_ms\": " << result.cond_ms << ",\n"
                << "  \"flow_ms\": " << result.flow_ms << ",\n"
                << "  \"vocoder_ms\": " << result.voc_ms << ",\n"
                << "  \"samples\": " << result.n_samples << ",\n"
                << "  \"frames\": " << result.frames << "\n}\n";
    }
#endif
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef MM3_ANE_VALIDATION
    try {
#endif
    const CliOptions options = parse_options(argc, argv);
    if (options.models.empty()) {
        print_usage();
        return 1;
    }
    if (!mm3_replay_mode_is_supported(options.mode)) {
        fprintf(stderr, "unsupported mode '%s'\n", options.mode.c_str());
        print_usage();
        return 1;
    }
    std::string error;
    if (options.mode != "condcheck" && !mm3_replay_prepare_output_dir(options.out_dir, &error)) {
        fprintf(stderr, "output error: %s\n", error.c_str());
        return 1;
    }

    backend_configure_cpu(options.threads > 0 ? options.threads
                                              : (int) std::thread::hardware_concurrency(),
                          "");
    backend_configure_device(options.device);

    MM3Model model;
    if (!load_model(options, model, &error)) {
        fprintf(stderr, "load error: %s\n", error.c_str());
        return 1;
    }
    if (options.mode == "condcheck") {
        return run_condcheck(model);
    }
    return run_generation(options, model);
#ifdef MM3_ANE_VALIDATION
    } catch (const std::exception & error) {
        fprintf(stderr,"ANE validation error: %s\n",error.what());
        return 1;
    }
#endif
}
