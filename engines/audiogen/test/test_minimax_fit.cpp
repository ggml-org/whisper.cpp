#include "minimax_fit_parity.h"
#include "minimax_tiny_model.h"

#include <cstdio>
#include <limits>
#include <string>

namespace {

using tts_cpp::minimax::EngineOptions;
using tts_cpp::minimax::FitResult;
using tts_cpp::minimax::FitStatus;
using tts_cpp::minimax::FitWorkload;

constexpr int64_t     kFrames            = 300;
constexpr int64_t     kPrompt            = 40;
constexpr int64_t     kLongFrames        = 600;
constexpr int64_t     kLongPrompt        = 400;
constexpr size_t      kStageCount        = 5;
constexpr const char * kStages[]         = { "lm", "depth", "cond", "dit", "vocoder" };

int failures = 0;
int checks   = 0;

void check(bool condition, const char * expression, int line) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL test_minimax_fit.cpp:%d %s\n", line, expression);
    }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

EngineOptions tiny_options(const minimax_tiny::Pair & pair) {
    EngineOptions options;
    options.lm_model_path    = pair.lm.string();
    options.synth_model_path = pair.synth.string();
    options.n_threads        = 1;
    options.device           = "cpu";
    return options;
}

FitWorkload workload(int64_t frames, int64_t prompt, uint64_t margin = 0) {
    FitWorkload request;
    request.max_frames    = frames;
    request.prompt_tokens = prompt;
    request.margin_bytes  = margin;
    return request;
}

const tts_cpp::minimax::FitStageProjection & stage(const FitResult & fit, const char * name) {
    static const tts_cpp::minimax::FitStageProjection missing;
    const auto * found = minimax_fit_parity::find_stage(fit, name);
    return found ? *found : missing;
}

void check_stage_rows(const FitResult & fit) {
    for (const char * name : kStages) {
        CHECK(minimax_fit_parity::find_stage(fit, name) != nullptr);
        CHECK(stage(fit, name).weights_bytes > 0);
        CHECK(stage(fit, name).compute_bytes > 0);
    }
}

void test_projects_a_readable_pair(const EngineOptions & options) {
    const FitResult fit = tts_cpp::minimax::fit_params(options, workload(kFrames, kPrompt));
    CHECK(fit.status == FitStatus::Success);
    CHECK(fit.fits);
    CHECK(fit.reason == "fits");
    CHECK(fit.device_is_cpu);
    CHECK(fit.device_shares_host_memory);
    CHECK(fit.stages_resident);
    CHECK(fit.stages.size() == kStageCount);
    CHECK(fit.model_name == "tiny MiniMax-Music3 LM");
    CHECK(fit.peak_device_bytes == fit.peak_host_bytes);
    CHECK(fit.device_total_bytes > 0);
    CHECK(!fit.report.empty());
    check_stage_rows(fit);
    CHECK(stage(fit, "lm").state_bytes > 0);
    CHECK(stage(fit, "depth").state_bytes > 0);
}

void check_matches_runtime_allocations(const EngineOptions & options, const FitResult & fit) {
    minimax_fit_parity::RealBytes real;
    std::string                   error;
    CHECK(minimax_fit_parity::measure_real(options, kPrompt, kFrames, real, error));
    CHECK(error.empty());
    CHECK(minimax_fit_parity::projected_weights(fit) == real.weights);
    CHECK(stage(fit, "lm").state_bytes == real.lm_state);
    CHECK(stage(fit, "lm").compute_bytes == real.lm_compute);
    CHECK(stage(fit, "depth").state_bytes == real.depth_state);
    CHECK(stage(fit, "depth").compute_bytes == real.depth_compute);
    CHECK(stage(fit, "depth").host_bytes == real.depth_host);
    CHECK(stage(fit, "cond").compute_bytes == real.cond_compute);
    CHECK(stage(fit, "cond").host_bytes == real.cond_host);
    CHECK(stage(fit, "dit").compute_bytes == real.dit_compute);
    CHECK(stage(fit, "vocoder").compute_bytes == real.vocoder_compute);
}

void test_matches_runtime_allocations(const EngineOptions & options) {
    check_matches_runtime_allocations(options, tts_cpp::minimax::fit_params(options, workload(kFrames, kPrompt)));
}

void test_workload_growth(const EngineOptions & options) {
    const FitResult base        = tts_cpp::minimax::fit_params(options, workload(kFrames, kPrompt));
    const FitResult more_frames = tts_cpp::minimax::fit_params(options, workload(kLongFrames, kPrompt));
    const FitResult more_prompt = tts_cpp::minimax::fit_params(options, workload(kFrames, kLongPrompt));
    CHECK(more_frames.peak_device_bytes > base.peak_device_bytes);
    CHECK(stage(more_frames, "lm").state_bytes > stage(base, "lm").state_bytes);
    CHECK(stage(more_frames, "vocoder").host_bytes > stage(base, "vocoder").host_bytes);
    CHECK(stage(more_prompt, "lm").state_bytes > stage(base, "lm").state_bytes);
    CHECK(stage(more_prompt, "lm").compute_bytes > stage(base, "lm").compute_bytes);
}

void test_frames_are_capped_like_generate(const EngineOptions & options) {
    const FitResult capped = tts_cpp::minimax::fit_params(options, workload(minimax_tiny::kMaxFrames, kPrompt));
    const FitResult beyond = tts_cpp::minimax::fit_params(options, workload(minimax_tiny::kMaxFrames * 2, kPrompt));
    CHECK(capped.status == FitStatus::Success);
    CHECK(beyond.peak_device_bytes == capped.peak_device_bytes);
}

void test_margin_turns_a_fit_into_a_failure(const EngineOptions & options) {
    const FitResult fit =
        tts_cpp::minimax::fit_params(options, workload(kFrames, kPrompt, std::numeric_limits<uint64_t>::max()));
    CHECK(fit.status == FitStatus::Failure);
    CHECK(!fit.fits);
    CHECK(fit.reason == "does-not-fit");
}

void check_rejection(const EngineOptions & options, const FitWorkload & request, const char * reason) {
    const FitResult fit = tts_cpp::minimax::fit_params(options, request);
    CHECK(fit.status == FitStatus::Error);
    CHECK(fit.reason == reason);
    CHECK(!fit.report.empty());
}

void test_rejections(const EngineOptions & options) {
    check_rejection(options, workload(0, kPrompt), "invalid-arguments");
    check_rejection(options, workload(kFrames, 0), "invalid-arguments");
    check_rejection(options, workload(kFrames, minimax_tiny::kMaxPromptTokens + 1), "workload-too-large");
    check_rejection(options, workload(minimax_tiny::kMaxFrames, minimax_tiny::kMaxPromptTokens), "workload-too-large");
    EngineOptions bad_device = options;
    bad_device.device        = "tpu";
    check_rejection(bad_device, workload(kFrames, kPrompt), "invalid-arguments");
    EngineOptions no_paths;
    no_paths.device = "cpu";
    check_rejection(no_paths, workload(kFrames, kPrompt), "invalid-arguments");
    EngineOptions missing    = options;
    missing.lm_model_path    = (std::filesystem::temp_directory_path() / "minimax-fit-missing" / "mm3-lm-f16.gguf").string();
    check_rejection(missing, workload(kFrames, kPrompt), "model-unreadable");
}

void test_gpu_request_matches_runtime_or_reports_no_device(const EngineOptions & options) {
    EngineOptions gpu = options;
    gpu.device        = "gpu";
    const FitResult fit = tts_cpp::minimax::fit_params(gpu, workload(kFrames, kPrompt));
    if (fit.status == FitStatus::Error) {
        CHECK(fit.reason == "no-backend-device");
        return;
    }
    CHECK(!fit.device_is_cpu);
    check_matches_runtime_allocations(gpu, fit);
}

void test_coreml_memory_estimate_is_explicitly_unavailable(const EngineOptions & options) {
    EngineOptions coreml = options;
    coreml.coreml_conditioner_path = "conditioner.mlmodelc";
    const FitResult fit = tts_cpp::minimax::fit_params(coreml, workload(kFrames, kPrompt));
    CHECK(fit.status == FitStatus::Error);
    CHECK(!fit.fits);
    CHECK(fit.reason == "coreml-memory-estimate-unavailable");
    CHECK(!fit.report.empty());
    CHECK(fit.stages.empty());
    CHECK(fit.peak_device_bytes == 0);
    CHECK(fit.peak_host_bytes == 0);
    coreml.coreml_conditioner_path.clear();
    coreml.coreml_profile_dir = "compiled-profile";
    coreml.coreml_compute_units = "CPU_AND_NE";
    const FitResult full = tts_cpp::minimax::fit_params(coreml, workload(kFrames, kPrompt));
    CHECK(full.status == FitStatus::Error);
    CHECK(!full.fits);
    CHECK(full.reason == "coreml-memory-estimate-unavailable");
    CHECK(full.stages.empty());
}

void test_weightless_description_projects_the_same(const EngineOptions & options,
                                                    const minimax_tiny::Pair & pair) {
    const FitResult full = tts_cpp::minimax::fit_params(options, workload(kFrames, kPrompt));
    minimax_tiny::strip_payload(pair.lm);
    minimax_tiny::strip_payload(pair.synth);
    const FitResult stripped = tts_cpp::minimax::fit_params(options, workload(kFrames, kPrompt));
    CHECK(stripped.status == full.status);
    CHECK(stripped.peak_device_bytes == full.peak_device_bytes);
    CHECK(stripped.peak_host_bytes == full.peak_host_bytes);
    CHECK(minimax_fit_parity::projected_weights(stripped) == minimax_fit_parity::projected_weights(full));
}

}

int main() {
    const minimax_tiny::Pair pair("fit");
    CHECK(pair.written);
    if (!pair.written) {
        return 1;
    }
    const EngineOptions options = tiny_options(pair);
    test_projects_a_readable_pair(options);
    test_matches_runtime_allocations(options);
    test_workload_growth(options);
    test_frames_are_capped_like_generate(options);
    test_margin_turns_a_fit_into_a_failure(options);
    test_rejections(options);
    test_gpu_request_matches_runtime_or_reports_no_device(options);
    test_coreml_memory_estimate_is_explicitly_unavailable(options);
    test_weightless_description_projects_the_same(options, pair);
    std::printf("[test-minimax-fit] %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
