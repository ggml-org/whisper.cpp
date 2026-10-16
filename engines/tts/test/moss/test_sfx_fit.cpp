#include "sfx_fixtures.h"
#include "tts-cpp/moss/sound_effect_fit.h"
#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace moss_sfx_fixtures;
using namespace tts_cpp;

namespace {
constexpr uint32_t MODEL_SEED = 47;
constexpr double SHORT_SECONDS = 0.1;
constexpr double LONG_SECONDS = 0.2;
int failures = 0;

void check(bool condition, const char * label) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", label); ++failures; }
}

struct Model {
    std::filesystem::path path = write_sfx_model("sfx-fit", MODEL_SEED);
    ~Model() { std::filesystem::remove(path); }
    moss::SoundEffectOptions options() const {
        moss::SoundEffectOptions options;
        options.model_path = path.string();
        options.n_threads = 1;
        return options;
    }
};

moss::SoundEffectRequest request(double seconds = SHORT_SECONDS) {
    moss::SoundEffectRequest result;
    result.prompt = "rain";
    result.seconds = seconds;
    return result;
}

void strip_weights(const std::filesystem::path & path) {
    ggml_context * metadata = nullptr;
    auto * file = gguf_init_from_file(path.string().c_str(), {true, &metadata});
    const auto bytes = gguf_get_meta_size(file);
    ggml_free(metadata);
    gguf_free(file);
    std::filesystem::resize_file(path, bytes);
}

void test_weightless() {
    Model model;
    const auto full = moss::fit_params(model.options(), request(), 0);
    check(full.status == FitStatus::Success, "small CPU projection fits");
    check(full.model_variant == "moss-sfx", "sound-effect variant");
    check(full.device_is_cpu && full.device_shares_host_memory, "CPU charges host memory");
    check(full.device.weights_bytes > 0 && full.device.lm_compute_bytes > 0, "weights and graphs measured");
    check(full.host_bytes >= ggml_graph_overhead_custom(16384, false), "graph descriptors counted");
    check(full.device.total_bytes == full.device.weights_bytes + full.device.state_bytes +
        full.device.lm_compute_bytes + full.device.codec_compute_bytes, "breakdown sums to total");
    strip_weights(model.path);
    const auto stub = moss::fit_params(model.options(), request(), 0);
    check(stub.status == full.status && stub.device.total_bytes == full.device.total_bytes,
          "metadata without a tensor payload produces the same device projection");
    check(stub.host_bytes == full.host_bytes, "weightless host projection is identical");
    check(moss::fit_params(model.options(), request(), UINT64_MAX).status == FitStatus::Failure,
          "overflowing margin cannot fit");
}

void test_workloads() {
    Model model;
    const auto short_fit = moss::fit_params(model.options(), request(), 0);
    const auto long_fit = moss::fit_params(model.options(), request(LONG_SECONDS), 0);
    check(long_fit.status == FitStatus::Success, "longer decode projects");
    check(long_fit.device.lm_compute_bytes == short_fit.device.lm_compute_bytes,
          "diffusion still covers the fixed model duration");
    check(long_fit.host_bytes >= short_fit.host_bytes, "longer output does not reduce host requirement");
    auto unguided = request();
    unguided.guidance = 1.0f;
    const auto single = moss::fit_params(model.options(), unguided, 0);
    check(single.status == FitStatus::Success && single.host_bytes <= short_fit.host_bytes,
          "unguided diffusion does not retain extra velocity copies");
    auto steps = request();
    steps.steps = STEPS + 1;
    check(moss::fit_params(model.options(), steps, 0).host_bytes >= short_fit.host_bytes,
          "step schedule is included");
}

void test_errors() {
    Model model;
    auto invalid = request();
    invalid.seconds = std::numeric_limits<double>::quiet_NaN();
    check(moss::fit_params(model.options(), invalid, 0).reason == "invalid-arguments", "NaN duration rejected");
    invalid = request();
    invalid.seconds = LONG_SECONDS * 2;
    check(moss::fit_params(model.options(), invalid, 0).reason == "invalid-arguments", "runtime duration range reused");
    invalid = request();
    invalid.prompt = "  ";
    check(moss::fit_params(model.options(), invalid, 0).reason == "invalid-arguments", "empty cleaned prompt rejected");
    invalid = request();
    invalid.steps = -1;
    check(moss::fit_params(model.options(), invalid, 0).reason == "invalid-arguments", "negative steps rejected");
    invalid = request();
    invalid.guidance = 0.5f;
    check(moss::fit_params(model.options(), invalid, 0).reason == "invalid-arguments", "runtime guidance range reused");
    auto missing = model.options();
    missing.model_path = "/nonexistent/moss-sfx.gguf";
    check(moss::fit_params(missing, request(), 0).reason == "model-unreadable", "missing model is an error outcome");
}

void test_explicit_backend() {
    Model model;
    auto options = model.options();
    options.backend = "cpu";
    const auto cpu = moss::fit_params(options, request(), 0);
    check(cpu.status == FitStatus::Success && cpu.device_is_cpu, "an explicit cpu backend projects on the CPU");
    options.backend = "no-such-device";
    const auto missing = moss::fit_params(options, request(), 0);
    check(missing.status == FitStatus::Error && missing.reason == "no-backend-device",
          "an unavailable explicit backend reports no-backend-device");
}

void test_generation_after_fit() {
    Model model;
    check(moss::fit_params(model.options(), request(), 0).status == FitStatus::Success, "fit before synthesis");
    moss::SoundEffectEngine engine(model.options());
    const auto audio = engine.generate(request());
    check(!audio.cancelled && audio.pcm.size() == size_t(SAMPLE_RATE * SHORT_SECONDS), "generation length unchanged");
    check(std::all_of(audio.pcm.begin(), audio.pcm.end(), [](float v) { return std::isfinite(v); }),
          "generation remains finite after metadata measurement");
}

void test_real_model() {
    const char * path = std::getenv("MOSS_SFX_MODEL");
    if (!path || !*path) return;
    moss::SoundEffectOptions options;
    options.model_path = path;
    options.n_threads = 1;
    options.use_gpu = std::getenv("MOSS_SFX_GPU") != nullptr;
    const auto projection = moss::fit_params(options, request(8), 0);
    check(projection.status != FitStatus::Error, "real model graphs project");
    if (options.use_gpu) check(!projection.device_is_cpu, "real GPU projection uses a GPU");
    check(projection.device.weights_bytes > 0 && projection.host_bytes > 0, "real model memory measured");
    std::puts(projection.report.c_str());
}
}

int main() {
    test_weightless();
    test_workloads();
    test_errors();
    test_explicit_backend();
    test_generation_after_fit();
    test_real_model();
    if (!failures) std::puts("MOSS sound-effect fit tests passed");
    return failures ? 1 : 0;
}
