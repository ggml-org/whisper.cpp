#include "moss/transcribe_audio.h"
#include "moss/transcribe_fit.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_request.h"
#include "parakeet/moss_transcribe_fit.h"
#include "transcribe_fixtures.h"

#include <cstdlib>
#include <iostream>
#include <limits>

using namespace moss_transcribe_fixtures;
using namespace parakeet;
using namespace parakeet::moss;
using namespace parakeet::moss::detail;

namespace {

constexpr uint32_t SEED = 26493;
constexpr double SHORT_SECONDS = 0.02;
constexpr double LONG_SECONDS = 0.8;
constexpr int MANY_THREADS = 1024;
constexpr uint64_t ONE_GIB = uint64_t(1) << 30;

void check(bool passed, const char * message) {
    if (!passed) throw std::runtime_error(message);
}

FitResult project(const std::filesystem::path & path, double seconds = SHORT_SECONDS,
                   const TranscribeRequest & request = {}, uint64_t margin = 0) {
    TranscribeOptions options;
    options.model_path = path.string();
    return parakeet::moss::fit_params(options, request, seconds, margin);
}

void check_projection(const FitResult & fit) {
    check(fit.status != FitStatus::Error, fit.report.c_str());
    check(fit.model_type == "moss-transcribe", "wrong model type");
    check(fit.device.weights_bytes > 0, "no weights projected");
    check(fit.device.decoder_state_bytes > 0, "no KV projected");
    check(fit.device.encoder_compute_bytes > 0, "no encoder graph projected");
    check(fit.host_bytes > 0, "no host payload projected");
    check(fit.device.total_bytes == fit.device.weights_bytes + fit.device.encoder_compute_bytes +
        fit.device.decoder_state_bytes + fit.device.decoder_compute_bytes, "component sum disagrees");
}

void check_weightless(const std::filesystem::path & path, const FitResult & full) {
    ggml_context * tensors = nullptr;
    auto * file = gguf_init_from_file(path.string().c_str(), {true, &tensors});
    const auto header = temp_gguf("transcribe-fit-header");
    check(gguf_write_to_file(file, header.string().c_str(), true), "cannot write metadata-only fixture");
    gguf_free(file);
    ggml_free(tensors);
    const auto fit = project(header);
    check_projection(fit);
    check(fit.device.total_bytes == full.device.total_bytes, "weightless projection differs");
    check(fit.host_bytes == full.host_bytes, "weightless host projection differs");
    std::filesystem::remove(header);
}

void check_invalid(const std::filesystem::path & path) {
    for (double seconds : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        check(project(path, seconds).reason == "invalid-arguments", "invalid duration accepted");
    }
    check(project(path, std::numeric_limits<double>::max()).reason == "workload-too-large", "huge duration accepted");
    TranscribeRequest request;
    request.max_new_tokens = -1;
    check(project(path, SHORT_SECONDS, request).reason == "invalid-arguments", "negative tokens accepted");
    request.max_new_tokens = TEXT_CTX;
    check(project(path, SHORT_SECONDS, request).reason == "workload-too-large", "context overflow accepted");
    request = {};
    request.prompt.assign(TRANSCRIBE_MAX_PROMPT_BYTES + 1, 'a');
    check(project(path, SHORT_SECONDS, request).reason == "invalid-arguments", "oversized prompt accepted");
    request.prompt = "custom";
    request.hotwords = {"QVAC"};
    check(project(path, SHORT_SECONDS, request).reason == "invalid-arguments", "prompt and hotwords accepted");
    request.prompt.clear();
    request.hotwords.assign(MAX_HOTWORDS + 1, "QVAC");
    check(project(path, SHORT_SECONDS, request).reason == "invalid-arguments", "oversized hotwords accepted");
    check(project("/nonexistent/moss-transcribe.gguf").reason == "model-unreadable", "missing model not reported");
    const auto margin = project(path, SHORT_SECONDS, {}, std::numeric_limits<uint64_t>::max());
    check(margin.status == FitStatus::Failure && !margin.fits, "saturated margin incorrectly fits");
}

void check_scaling(const std::filesystem::path & path, const FitResult & short_fit) {
    const auto long_fit = project(path, LONG_SECONDS);
    check_projection(long_fit);
    check(long_fit.device.weights_bytes == short_fit.device.weights_bytes, "weights depend on duration");
    check(long_fit.device.decoder_state_bytes > short_fit.device.decoder_state_bytes, "KV did not grow");
    check(long_fit.host_bytes > short_fit.host_bytes, "host did not grow");
    TranscribeRequest request;
    request.max_new_tokens = 1024;
    const auto decode = project(path, SHORT_SECONDS, request);
    check_projection(decode);
    check(decode.device.decoder_state_bytes > short_fit.device.decoder_state_bytes, "token allowance did not grow KV");
    request.max_new_tokens = 0;
    request.hotwords = {"QVAC", "Tether"};
    check_projection(project(path, SHORT_SECONDS, request));
}

void check_runtime_parity(const std::filesystem::path & path, const FitResult & fit) {
    TranscribeModel model(path.string(), false, 4);
    check(model.weight_bytes() == fit.device.weights_bytes, "weight allocation differs from projection");
    TranscribeTokenizer tokenizer(model);
    const size_t samples = (size_t) (SHORT_SECONDS * SAMPLE_RATE);
    const int audio_tokens = transcribe_audio_tokens(model.config(), samples);
    TranscribeMel mel(model.config().audio, read_mel_filters(model));
    std::vector<float> pcm(samples, 0.0f);
    const auto encoding = encode_audio_chunk(model, mel.chunk(pcm.data(), samples), audio_tokens, false);
    check(model.allocated_memory().device_bytes == fit.device.encoder_compute_bytes, "encoder allocation differs");
    check(model.allocated_memory().host_bytes > ONE_GIB,
          "scheduler host allocation is undercounted");
    check(model.allocated_memory().cpu_work_bytes > 0,
          "encoder CPU scratch omitted");
    check(fit.host_bytes >= model.allocated_memory().host_bytes +
                                model.allocated_memory().cpu_work_bytes,
          "encoder host allocation exceeds projection");
    const auto prompt = transcribe_prompt(model.config(), tokenizer, audio_tokens, "");
    TranscribeDecoder decoder(model, (int) prompt.size() + DEFAULT_MAX_NEW_TOKENS);
    auto logits = decoder.prefill(prompt, encoding.embeddings, TRANSCRIBE_PREFILL_BATCH_TOKENS);
    for (int i = 1; i < DEFAULT_MAX_NEW_TOKENS; ++i) logits = decoder.step(byte_token('a'));
    check(model.allocated_memory().device_bytes <= fit.device.encoder_compute_bytes + fit.device.decoder_compute_bytes,
        "decode allocation exceeds projection");
    check(fit.host_bytes >= model.allocated_memory().host_bytes +
                                model.allocated_memory().cpu_work_bytes,
          "decoder host allocation exceeds projection");
    TranscribeModel metadata(path.string(), false, 4, "auto", true);
    TranscribeGraph graph(8);
    bool guarded = false;
    try { metadata.compute(graph); } catch (const std::runtime_error &) { guarded = true; }
    check(guarded, "metadata-only compute was allowed");
}

void check_host_budget(const FitResult &fit) {
  auto limited = fit;
  limited.device_free_bytes = ONE_GIB;
  limited.device_total_bytes = ONE_GIB;
  limited.device_shares_host_memory = true;
  finish_transcribe_fit(limited, 0);
  check(limited.status == FitStatus::Failure && !limited.fits &&
            limited.reason == "does-not-fit",
        "scheduler host allocations falsely fit within 1 GiB");
}

void check_thread_scaling(const std::filesystem::path &path) {
  TranscribeOptions options;
  options.model_path = path.string();
  const auto ordinary =
      parakeet::moss::fit_params(options, {}, LONG_SECONDS, 0);
  options.n_threads = MANY_THREADS;
  const auto many = parakeet::moss::fit_params(options, {}, LONG_SECONDS, 0);
  check_projection(ordinary);
  check_projection(many);
  check(many.host_bytes > ordinary.host_bytes,
        "CPU scratch does not reflect configured threads");
  check(many.device.total_bytes == ordinary.device.total_bytes,
        "CPU scratch charged to tensor buffers");
}

void check_real_model() {
    const char * path = std::getenv("MOSS_TRANSCRIBE_MODEL");
    if (!path || !*path) return;
    TranscribeOptions options;
    options.model_path = path;
    options.use_gpu = std::getenv("MOSS_TRANSCRIBE_GPU") != nullptr;
    const auto fit = parakeet::moss::fit_params(options, {}, 30, 0);
    check_projection(fit);
    if (options.use_gpu) check(!fit.device_is_cpu, "GPU test fell back to CPU");
    std::cout << fit.report;
}

}

int main() try {
    const auto path = write_transcribe_model("transcribe-fit", SEED);
    const auto fit = project(path);
    check_projection(fit);
    check_weightless(path, fit);
    check_invalid(path);
    check_scaling(path, fit);
    check_runtime_parity(path, fit);
    check_host_budget(fit);
    check_thread_scaling(path);
    std::filesystem::remove(path);
    check_real_model();
    std::cout << "test-moss-transcribe-fit: all checks passed\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
