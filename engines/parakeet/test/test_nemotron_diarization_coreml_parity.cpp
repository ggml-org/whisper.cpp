#include "mel_preprocess.h"
#include "parakeet_ctc.h"
#include "parakeet_diarization_v3.h"

#ifdef PARAKEET_USE_COREML
#include "coreml/coreml_diarizers.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kSkip = 77;
constexpr int kThreads = 4;
constexpr int kGpuLayers = 99;
constexpr int kLiveMelFrames = 112;
constexpr int kLongFormCacheFrames = 300;
constexpr double kMaximumProbabilityError = 0.02;
constexpr double kMaximumProbabilityRelativeError = 0.003;
constexpr double kMaximumEmbeddingRelativeError = 0.01;
constexpr float kGainFloor = 1.0e-3f;
constexpr const char * kDisableEnv = "PARAKEET_COREML_DISABLE";

struct Comparison {
    double maximum_error = 0.0;
    double relative_error = 0.0;
};

struct Case {
    std::string name;
    int mel_frames = 0;
    int cache_frames = 0;
    bool expects_coreml = true;
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

float peak_of(const std::vector<float> & samples) {
    float peak = 0.0f;
    for (float sample : samples) peak = std::max(peak, std::fabs(sample));
    return peak;
}

void normalize(std::vector<float> & samples) {
    const float gain = 1.0f / (peak_of(samples) + kGainFloor);
    std::transform(samples.begin(), samples.end(), samples.begin(),
        [gain](float sample) { return sample * gain; });
}

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

bool load_model(const char * path, bool disable_coreml, parakeet::ParakeetCtcModel & model) {
    if (disable_coreml) set_env(kDisableEnv, "1");
    const int status = parakeet::load_from_gguf(path, model, kThreads, kGpuLayers, false);
    if (disable_coreml) unset_env(kDisableEnv);
    return status == 0;
}

bool run_chunk(const parakeet::ParakeetCtcModel & model, const std::vector<float> & mel,
               int mel_frames, const std::vector<float> & cache, int cache_frames,
               parakeet::NemotronDiarizationChunk & chunk) {
    return parakeet::run_nemotron_diarization_chunk(model, mel.data(), mel_frames,
        cache_frames > 0 ? cache.data() : nullptr, cache_frames, chunk) == 0;
}

bool within_tolerance(const Comparison & probabilities, const Comparison & embeddings) {
    return probabilities.maximum_error <= kMaximumProbabilityError &&
           probabilities.relative_error <= kMaximumProbabilityRelativeError &&
           embeddings.relative_error <= kMaximumEmbeddingRelativeError;
}

bool run_case(const parakeet::ParakeetCtcModel & accelerated, const parakeet::ParakeetCtcModel & reference,
              const std::vector<float> & mel, const std::vector<float> & cache, const Case & test) {
    parakeet::NemotronDiarizationChunk actual;
    parakeet::NemotronDiarizationChunk expected;
    if (!run_chunk(accelerated, mel, test.mel_frames, cache, test.cache_frames, actual) ||
        !run_chunk(reference, mel, test.mel_frames, cache, test.cache_frames, expected)) {
        std::printf("[FAIL] %s: inference failed\n", test.name.c_str());
        return false;
    }
    const Comparison probabilities = compare(actual.probabilities, expected.probabilities);
    const Comparison embeddings = compare(actual.embeddings, expected.embeddings);
    const bool routed = actual.used_coreml == test.expects_coreml && !expected.used_coreml;
    const bool passed = routed && within_tolerance(probabilities, embeddings);
    std::printf("[%s] %s: cache=%d chunk=%d coreml=%d max=%.5f rel=%.6f embedding_rel=%.6f\n",
        passed ? "ok  " : "FAIL", test.name.c_str(), test.cache_frames, actual.chunk_frames,
        actual.used_coreml ? 1 : 0, probabilities.maximum_error, probabilities.relative_error,
        embeddings.relative_error);
    return passed;
}

int largest_capacity(const parakeet::ParakeetCtcModel & model) {
#ifdef PARAKEET_USE_COREML
    const std::vector<int> capacities = parakeet::model_coreml_diarizers(model)->capacities();
    return capacities.empty() ? 0 : capacities.back();
#else
    (void) model;
    return 0;
#endif
}

bool has_sidecars(const parakeet::ParakeetCtcModel & model) {
#ifdef PARAKEET_USE_COREML
    return parakeet::model_coreml_diarizers(model) != nullptr;
#else
    (void) model;
    return false;
#endif
}

std::vector<float> tile_rows(const std::vector<float> & rows, int width, int count) {
    const size_t available = rows.size() / width;
    std::vector<float> tiled(static_cast<size_t>(count) * width);
    for (int row = 0; row < count; ++row) {
        std::copy_n(rows.begin() + (row % available) * width, width, tiled.begin() + row * width);
    }
    return tiled;
}

std::vector<Case> cases(const parakeet::ParakeetCtcModel & model, int mel_frames) {
    const int capacity = largest_capacity(model);
    const int offline_frames = parakeet::nemotron_diarization_encoder_frames(model, mel_frames);
    const int live_mel = std::min(mel_frames, kLiveMelFrames);
    const int live_cache = parakeet::nemotron_diarization_stream_state_frames();
    const int live_frames = live_cache + parakeet::nemotron_diarization_encoder_frames(model, live_mel);
    return {
        {"offline", mel_frames, 0, offline_frames <= capacity},
        {"live-chunk", live_mel, live_cache, live_frames <= capacity},
        {"long-form-block", mel_frames, kLongFormCacheFrames,
            kLongFormCacheFrames + offline_frames <= capacity},
        {"above-capacity", live_mel, capacity, false},
    };
}

int cache_rows_needed(const parakeet::ParakeetCtcModel & model) {
    return std::max({largest_capacity(model), kLongFormCacheFrames,
                     parakeet::nemotron_diarization_stream_state_frames()});
}

bool run_cases(const parakeet::ParakeetCtcModel & accelerated, const parakeet::ParakeetCtcModel & reference,
               const std::vector<float> & mel, const std::vector<float> & cache,
               const std::vector<Case> & tests) {
    bool passed = true;
    for (const Case & test : tests) passed = run_case(accelerated, reference, mel, cache, test) && passed;
    return passed;
}

}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf AUDIO.wav\n", argv[0]);
        return 2;
    }
    parakeet::ParakeetCtcModel accelerated;
    parakeet::ParakeetCtcModel reference;
    if (!load_model(argv[1], false, accelerated)) return 1;
    if (!has_sidecars(accelerated)) {
        std::printf("SKIP: no Core ML diarizer beside %s\n", argv[1]);
        return kSkip;
    }
    if (!load_model(argv[1], true, reference)) return 1;
    std::vector<float> samples;
    int sample_rate = 0;
    if (parakeet::load_wav_mono_f32(argv[2], samples, sample_rate) != 0) return 1;
    normalize(samples);
    std::vector<float> mel;
    int mel_frames = 0;
    if (parakeet::compute_log_mel(samples.data(), static_cast<int>(samples.size()),
            reference.mel_cfg, mel, mel_frames) != 0) return 1;
    parakeet::NemotronDiarizationChunk offline;
    if (!run_chunk(reference, mel, mel_frames, {}, 0, offline)) return 1;
    const std::vector<float> cache = tile_rows(offline.embeddings,
        reference.nemotron_diarization_cfg.encoder_width, cache_rows_needed(accelerated));
    const bool passed = run_cases(accelerated, reference, mel, cache, cases(accelerated, mel_frames));
    std::printf("[nemotron-diarization-coreml-parity] %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
