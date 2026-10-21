// Core ML diarizer failure handling. A bundle that fails to load or to run is
// quarantined for the model's lifetime: the call that hit it runs on the
// next covering capacity or on ggml, and later calls never retry it. The
// wrapper refuses a call that leaves a model input unsupplied, so a bundle
// with other input names falls back instead of diarizing zeros. Stages
// symlinks to the real Nemotron 3 Diarization and Sortformer v2.1 bundles in a
// temporary directory; exits 77 when they are missing.

#include "coreml/coreml_diarizers.h"
#include "mel_preprocess.h"
#include "parakeet_coreml_path.h"
#include "parakeet_ctc.h"
#include "parakeet_diarization_v3.h"
#include "parakeet_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;

constexpr int kSkip = 77;
constexpr int kThreads = 4;
constexpr int kGpuLayers = 99;
constexpr int kSampleRate = 16000;
constexpr int kShortSeconds = 4;
constexpr int kFailingCapacity = 100;
constexpr int kSortformerSpeakers = 4;
constexpr int kSortformerWidth = 512;
constexpr int kCalls = 2;
constexpr float kInputValue = 0.1f;
constexpr double kSameGraphError = 1.0e-5;
constexpr double kMaximumProbabilityError = 0.02;
constexpr double kMaximumRelativeError = 0.003;
constexpr const char * kStemFile = "model.gguf";
constexpr const char * kDisableEnv = "PARAKEET_COREML_DISABLE";
constexpr const char * kPreEncode = "pre_encode";
constexpr const char * kRenamedPreEncode = "pre_encode_renamed";
constexpr const char * kProbabilities = "probabilities";
constexpr const char * kLoadWarning = "failed to load Core ML diarizer";
constexpr const char * kRunWarning = "quarantining it";

int g_load_warnings = 0;
int g_run_warnings = 0;

void count_warnings(ggml_log_level level, const char * text, void *) {
    if (level != GGML_LOG_LEVEL_WARN || text == nullptr) return;
    if (std::strstr(text, kLoadWarning) != nullptr) ++g_load_warnings;
    if (std::strstr(text, kRunWarning) != nullptr) ++g_run_warnings;
}

void reset_warnings() {
    g_load_warnings = 0;
    g_run_warnings = 0;
}

bool check(bool condition, const std::string & label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label.c_str());
    return condition;
}

struct Comparison {
    double maximum_error = 0.0;
    double relative_error = 0.0;
};

Comparison compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    Comparison result;
    if (actual.size() != expected.size() || expected.empty()) {
        result.maximum_error = INFINITY;
        return result;
    }
    double difference = 0.0;
    double energy = 0.0;
    for (size_t index = 0; index < expected.size(); ++index) {
        const double delta = static_cast<double>(actual[index]) - expected[index];
        result.maximum_error = std::max(result.maximum_error, std::fabs(delta));
        difference += delta * delta;
        energy += static_cast<double>(expected[index]) * expected[index];
    }
    result.relative_error = std::sqrt(difference / std::max(energy, 1.0e-30));
    return result;
}

bool load_model(const std::string & path, bool disable_coreml, parakeet::ParakeetCtcModel & model) {
    if (disable_coreml) setenv(kDisableEnv, "1", 1);
    const int status = parakeet::load_from_gguf(path, model, kThreads, kGpuLayers, false);
    if (disable_coreml) unsetenv(kDisableEnv);
    return status == 0;
}

std::string first_bundle(const std::string & gguf) {
    const std::vector<parakeet::CoremlDiarizerBundle> bundles =
        parakeet::list_coreml_diarizer_bundles(gguf);
    return bundles.empty() ? std::string()
                           : parakeet::coreml_diarizer_sidecar_path(gguf, bundles.front().capacity);
}

int first_capacity(const std::string & gguf) {
    const std::vector<parakeet::CoremlDiarizerBundle> bundles =
        parakeet::list_coreml_diarizer_bundles(gguf);
    return bundles.empty() ? 0 : bundles.front().capacity;
}

fs::path staged_model(const fs::path & root, const char * name, const std::string & gguf) {
    const fs::path directory = root / name;
    fs::create_directories(directory);
    fs::create_symlink(fs::absolute(gguf), directory / kStemFile);
    return directory / kStemFile;
}

std::string bundle_path(const fs::path & model, int capacity) {
    return parakeet::coreml_diarizer_sidecar_path(model.string(), capacity);
}

void link_bundle(const fs::path & model, int capacity, const std::string & bundle) {
    fs::create_directory_symlink(fs::absolute(bundle), bundle_path(model, capacity));
}

void broken_bundle(const fs::path & model, int capacity) {
    fs::create_directories(bundle_path(model, capacity));
}

std::vector<float> short_mel(const parakeet::ParakeetCtcModel & model, const char * wav,
                             int & mel_frames) {
    std::vector<float> samples;
    int sample_rate = 0;
    std::vector<float> mel;
    mel_frames = 0;
    if (parakeet::load_wav_mono_f32(wav, samples, sample_rate) != 0 || sample_rate != kSampleRate) {
        return mel;
    }
    samples.resize(std::min<size_t>(samples.size(), static_cast<size_t>(kShortSeconds) * kSampleRate));
    parakeet::compute_log_mel(samples.data(), static_cast<int>(samples.size()), model.mel_cfg, mel,
                              mel_frames);
    return mel;
}

struct Calls {
    bool ran = true;
    bool coreml = true;
    bool ggml = true;
    Comparison worst;
};

Calls run_calls(const parakeet::ParakeetCtcModel & model, const std::vector<float> & mel,
                int mel_frames, const parakeet::NemotronDiarizationChunk & expected) {
    Calls calls;
    for (int call = 0; call < kCalls; ++call) {
        parakeet::NemotronDiarizationChunk chunk;
        calls.ran = parakeet::run_nemotron_diarization_chunk(model, mel.data(), mel_frames,
                                                             nullptr, 0, chunk) == 0 && calls.ran;
        calls.coreml = chunk.used_coreml && calls.coreml;
        calls.ggml = !chunk.used_coreml && calls.ggml;
        const Comparison comparison = compare(chunk.probabilities, expected.probabilities);
        calls.worst.maximum_error = std::max(calls.worst.maximum_error, comparison.maximum_error);
        calls.worst.relative_error = std::max(calls.worst.relative_error, comparison.relative_error);
    }
    return calls;
}

bool quarantined(const parakeet::ParakeetCtcModel & model, int frames) {
    parakeet::CoremlDiarizers * diarizers = parakeet::model_coreml_diarizers(model);
    return diarizers != nullptr && !diarizers->covers(frames);
}

bool load_failure_uses_next_capacity(const fs::path & root, const std::string & gguf,
                                     const std::vector<float> & mel, int mel_frames,
                                     const parakeet::NemotronDiarizationChunk & expected) {
    const fs::path model_path = staged_model(root, "next-capacity", gguf);
    broken_bundle(model_path, kFailingCapacity);
    link_bundle(model_path, first_capacity(gguf), first_bundle(gguf));
    parakeet::ParakeetCtcModel model;
    if (!check(load_model(model_path.string(), false, model), "next-capacity: model loads")) return false;
    reset_warnings();
    const Calls calls = run_calls(model, mel, mel_frames, expected);
    bool passed = check(calls.ran && calls.coreml, "next-capacity: every call runs on the covering bundle");
    passed = check(calls.worst.maximum_error <= kMaximumProbabilityError &&
                   calls.worst.relative_error <= kMaximumRelativeError,
                   "next-capacity: probabilities match ggml") && passed;
    return check(g_load_warnings == 1, "next-capacity: the broken bundle is tried once") && passed;
}

bool load_failure_uses_ggml(const fs::path & root, const std::string & gguf,
                            const std::vector<float> & mel, int mel_frames, int frames,
                            const parakeet::NemotronDiarizationChunk & expected) {
    const fs::path model_path = staged_model(root, "load-failure", gguf);
    broken_bundle(model_path, first_capacity(gguf));
    parakeet::ParakeetCtcModel model;
    if (!check(load_model(model_path.string(), false, model), "load-failure: model loads")) return false;
    reset_warnings();
    const Calls calls = run_calls(model, mel, mel_frames, expected);
    bool passed = check(calls.ran && calls.ggml, "load-failure: every call runs on ggml");
    passed = check(calls.worst.maximum_error <= kSameGraphError, "load-failure: probabilities equal ggml") && passed;
    passed = check(g_load_warnings == 1, "load-failure: the bundle is tried once") && passed;
    return check(quarantined(model, frames), "load-failure: the capacity stays quarantined") && passed;
}

bool run_failure_uses_ggml(const fs::path & root, const std::string & gguf,
                                  const std::string & foreign_bundle, int foreign_capacity,
                                  const std::vector<float> & mel, int mel_frames, int frames,
                                  const parakeet::NemotronDiarizationChunk & expected) {
    const fs::path model_path = staged_model(root, "run-failure", gguf);
    link_bundle(model_path, foreign_capacity, foreign_bundle);
    parakeet::ParakeetCtcModel model;
    if (!check(load_model(model_path.string(), false, model), "run-failure: model loads")) return false;
    reset_warnings();
    const Calls calls = run_calls(model, mel, mel_frames, expected);
    bool passed = check(calls.ran && calls.ggml, "run-failure: every call runs on ggml");
    passed = check(calls.worst.maximum_error <= kSameGraphError,
                   "run-failure: probabilities equal ggml") && passed;
    passed = check(g_load_warnings == 0 && g_run_warnings == 1,
                   "run-failure: the bundle loads and fails once") && passed;
    return check(quarantined(model, frames), "run-failure: the capacity stays quarantined") && passed;
}

int run_with_input(parakeet_coreml_diarizer * context, int capacity, const char * name,
                   const std::vector<float> & rows, std::vector<float> & probabilities) {
    const parakeet_coreml_rows input{name, 0, capacity, kSortformerWidth, rows.data()};
    const parakeet_coreml_rows_out output{kProbabilities, 0, capacity, kSortformerSpeakers,
                                          probabilities.data()};
    return parakeet_coreml_diarizer_run(context, capacity, name ? &input : nullptr, name ? 1 : 0,
                                        &output, 1);
}

bool wrapper_refuses_missing_inputs(const std::string & bundle, int capacity) {
    parakeet_coreml_diarizer * context =
        parakeet_coreml_diarizer_init(bundle.c_str(), parakeet::kCoremlDiarizerUnitsEnv);
    if (!check(context != nullptr, "wrapper: the Sortformer bundle loads")) return false;
    const std::vector<float> rows(static_cast<size_t>(capacity) * kSortformerWidth, kInputValue);
    std::vector<float> probabilities(static_cast<size_t>(capacity) * kSortformerSpeakers);
    bool passed = check(run_with_input(context, capacity, kPreEncode, rows, probabilities) == 0,
                        "wrapper: a call that supplies every input runs");
    passed = check(run_with_input(context, capacity, kRenamedPreEncode, rows, probabilities) != 0,
                   "wrapper: a renamed input is refused") && passed;
    passed = check(run_with_input(context, capacity, nullptr, rows, probabilities) != 0,
                   "wrapper: a call without inputs is refused") && passed;
    parakeet_coreml_diarizer_free(context);
    return passed;
}

fs::path temporary_root() {
    const fs::path root = fs::temp_directory_path() /
        ("parakeet-diarizer-fallback-" + std::to_string(getpid()));
    std::error_code ignored;
    fs::remove_all(root, ignored);
    fs::create_directories(root);
    return root;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s NEMOTRON.gguf SORTFORMER.gguf AUDIO.wav\n", argv[0]);
        return 2;
    }
    const std::string nemotron = argv[1];
    const std::string sortformer_bundle = first_bundle(argv[2]);
    const int sortformer_capacity = first_capacity(argv[2]);
    if (first_bundle(nemotron).empty() || sortformer_bundle.empty()) {
        std::printf("SKIP: the Nemotron and Sortformer diarizer bundles are not staged\n");
        return kSkip;
    }
    parakeet::log_set_callback(count_warnings, nullptr);
    parakeet::ParakeetCtcModel reference;
    if (!load_model(nemotron, true, reference)) return 1;
    int mel_frames = 0;
    const std::vector<float> mel = short_mel(reference, argv[3], mel_frames);
    const int frames = parakeet::nemotron_diarization_encoder_frames(reference, mel_frames);
    parakeet::NemotronDiarizationChunk expected;
    if (mel.empty() || frames > kFailingCapacity ||
        parakeet::run_nemotron_diarization_chunk(reference, mel.data(), mel_frames, nullptr, 0,
                                                 expected) != 0) {
        std::fprintf(stderr, "cannot prepare the %d s reference chunk\n", kShortSeconds);
        return 1;
    }
    const fs::path root = temporary_root();
    bool passed = wrapper_refuses_missing_inputs(sortformer_bundle, sortformer_capacity);
    passed = load_failure_uses_next_capacity(root, nemotron, mel, mel_frames, expected) && passed;
    passed = load_failure_uses_ggml(root, nemotron, mel, mel_frames, frames, expected) && passed;
    passed = run_failure_uses_ggml(root, nemotron, sortformer_bundle, sortformer_capacity,
                                          mel, mel_frames, frames, expected) && passed;
    std::error_code ignored;
    fs::remove_all(root, ignored);
    std::printf("[coreml-diarizer-fallback] %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
