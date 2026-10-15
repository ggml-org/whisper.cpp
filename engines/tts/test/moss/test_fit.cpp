#include "gguf_fixtures.h"
#include "tts-cpp/moss/fit.h"
#include "moss/codec.h"
#include "moss/delay_lm.h"
#include <cstdio>
#include <limits>

using namespace moss_fixtures;
using namespace tts_cpp;

namespace {
int failures = 0;
constexpr int CONTEXT = 64;
constexpr int PROMPT_ROWS = 8;
constexpr int GENERATION_ROWS = 24;
constexpr int REFERENCE_SAMPLES = 64;

void check(bool condition, const char * label) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", label); ++failures; }
}

struct Models {
    std::filesystem::path backbone = write_backbone("fit-backbone", [](gguf_context *) {});
    std::filesystem::path decoder = write_decoder("fit-decoder", N_VQ, 3 * D_MODEL, [](gguf_context *) {});
    std::filesystem::path encoder = write_encoder("fit-encoder");
    ~Models() {
        std::filesystem::remove(backbone);
        std::filesystem::remove(decoder);
        std::filesystem::remove(encoder);
    }
    moss::EngineOptions options() const {
        moss::EngineOptions result;
        result.backbone_path = backbone.string();
        result.decoder_path = decoder.string();
        result.context = CONTEXT;
        result.max_new_tokens = GENERATION_ROWS;
        result.n_threads = 1;
        return result;
    }
};

void strip_payload(const std::filesystem::path & path) {
    ggml_context * metadata = nullptr;
    auto * file = gguf_init_from_file(path.string().c_str(), {true, &metadata});
    const size_t bytes = gguf_get_meta_size(file);
    ggml_free(metadata);
    gguf_free(file);
    std::filesystem::resize_file(path, bytes);
}

void test_metadata_only() {
    Models models;
    const auto options = models.options();
    const moss::FitWorkload workload(PROMPT_ROWS, 0, false, 0);
    const auto full = moss::fit_params(options, workload);
    check(full.status == FitStatus::Success, "small CPU model fits");
    check(full.device_is_cpu && full.device_shares_host_memory, "CPU shares system RAM");
    check(full.device.weights_bytes > 0 && full.device.state_bytes > 0, "weights and KV priced");
    check(full.device.lm_compute_bytes > 0 && full.device.codec_compute_bytes > 0, "both graph arenas priced");
    const uint64_t descriptor_floor = ggml_graph_overhead_custom(8192, false) +
        ggml_graph_overhead_custom(16384, false);
    check(full.host_bytes >= descriptor_floor, "host requirement includes both live graph descriptor arenas");
    check(full.device.total_bytes == full.device.weights_bytes + full.device.state_bytes +
        full.device.lm_compute_bytes + full.device.codec_compute_bytes, "breakdown sums to total");
    strip_payload(models.backbone);
    strip_payload(models.decoder);
    const auto stripped = moss::fit_params(options, workload);
    check(stripped.status == full.status && stripped.device.total_bytes == full.device.total_bytes,
          "weightless metadata produces the same projection");
    check(stripped.host_bytes == full.host_bytes, "weightless host projection stays identical");
}

void test_workloads() {
    Models models;
    auto options = models.options();
    const auto batch = moss::fit_params(options, {PROMPT_ROWS, 0, false, 0});
    const auto stream = moss::fit_params(options, {PROMPT_ROWS, 0, true, 0});
    check(stream.status == FitStatus::Success, "streaming projects");
    check(stream.device.state_bytes > batch.device.state_bytes, "streaming includes decoder cache");
    options.encoder_path = models.encoder.string();
    const auto cloning = moss::fit_params(options, {PROMPT_ROWS, REFERENCE_SAMPLES, false, 0});
    check(cloning.status == FitStatus::Success, "reference encoder projects");
    check(cloning.device.weights_bytes > batch.device.weights_bytes, "encoder weights included");
    check(cloning.device.codec_compute_bytes > batch.device.codec_compute_bytes, "encoder graph included");
    options.dialogue_reference_paths = {"speaker-one.wav", "speaker-two.wav"};
    const auto dialogue = moss::fit_params(options, {PROMPT_ROWS, REFERENCE_SAMPLES, false, 0});
    check(dialogue.status == FitStatus::Success, "dialogue projects without opening reference recordings");
    check(dialogue.device.total_bytes == cloning.device.total_bytes, "same dialogue shapes have same requirement");
    options.context = CONTEXT * 2;
    const auto larger_cache = moss::fit_params(options, {PROMPT_ROWS, 0, false, 0});
    check(larger_cache.device.state_bytes > batch.device.state_bytes, "context sizes KV allocation");
    const auto margin = moss::fit_params(options, {PROMPT_ROWS, 0, false, UINT64_MAX});
    check(margin.status == FitStatus::Failure && !margin.fits, "overflowing margin cannot fit");
}

void test_errors() {
    Models models;
    auto options = models.options();
    check(moss::fit_params(options, {0, 0, false, 0}).reason == "invalid-arguments", "empty prompt rejected");
    check(moss::fit_params(options, {PROMPT_ROWS, -1, false, 0}).reason == "invalid-arguments", "negative reference rejected");
    check(moss::fit_params(options, {PROMPT_ROWS, REFERENCE_SAMPLES, false, 0}).reason == "invalid-arguments",
          "reference requires encoder");
    check(moss::fit_params(options, {std::numeric_limits<int>::max(), 0, false, 0}).reason == "workload-too-large",
          "oversized prompt does not overflow");
    options.duration_tokens = GENERATION_ROWS;
    check(moss::fit_params(options, {PROMPT_ROWS, 0, false, 0}).reason == "workload-too-large", "duration drain matches runtime");
    options.duration_tokens = 0;
    options.decoder_path = models.encoder.string();
    check(moss::fit_params(options, {PROMPT_ROWS, 0, false, 0}).reason == "model-unreadable", "wrong decoder role rejected");
    options = models.options();
    options.backbone_path = "/nonexistent/moss.gguf";
    check(moss::fit_params(options, {PROMPT_ROWS, 0, false, 0}).reason == "model-unreadable", "unreadable model is an outcome");
}

void test_explicit_backend() {
    Models models;
    auto options = models.options();
    options.backend = "cpu";
    const auto cpu = moss::fit_params(options, {PROMPT_ROWS, 0, false, 0});
    check(cpu.status == FitStatus::Success && cpu.device_is_cpu, "an explicit cpu backend projects on the CPU");
    options.backend = "no-such-device";
    const auto missing = moss::fit_params(options, {PROMPT_ROWS, 0, false, 0});
    check(missing.status == FitStatus::Error && missing.reason == "no-backend-device",
          "an unavailable explicit backend reports no-backend-device");
}

void test_runtime_after_measurement() {
    Models models;
    const auto options = models.options();
    check(moss::fit_params(options, {PROMPT_ROWS, 0, true, 0}).status == FitStatus::Success, "measure before runtime");
    moss::detail::DelayLM backbone(options.backbone_path, {}, 1, CONTEXT);
    moss::detail::DelayRow row;
    row.audio.assign(N_VQ, 0);
    check(backbone.prefill({row}).text.size() == TEXT_VOCAB, "runtime LM still computes after measure");
    moss::detail::Codec decoder(options.decoder_path, {}, 1);
    check(decoder.decode(std::vector<int32_t>(N_VQ * 4), N_VQ).size() == OUT_DIM * 4,
          "runtime decoder still computes after measure");
}
}

int main() {
    test_metadata_only();
    test_workloads();
    test_errors();
    test_explicit_backend();
    test_runtime_after_measurement();
    if (!failures) std::puts("MOSS fit tests passed");
    return failures ? 1 : 0;
}
