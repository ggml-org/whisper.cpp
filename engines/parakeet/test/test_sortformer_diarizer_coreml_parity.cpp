#include "mel_preprocess.h"
#include "parakeet_ctc.h"
#include "parakeet_sortformer.h"

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
constexpr int kAoscFrames = 409;
constexpr double kMaximumProbabilityError = 0.02;
constexpr double kMaximumRelativeError = 0.003;
constexpr float kGainFloor = 1.0e-3f;
constexpr const char * kDisableEnv = "PARAKEET_COREML_DISABLE";
constexpr const char * kDiarizerDisableEnv = "PARAKEET_COREML_DIARIZER_DISABLE";
constexpr const char * kUnitsEnv = "PARAKEET_COREML_COMPUTE_UNITS";
constexpr const char * kNeuralEngineLabel = "coreml-ane";

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

float peak_of(const std::vector<float> & samples) {
    float peak = 0.0f;
    for (float sample : samples) peak = std::max(peak, std::fabs(sample));
    return peak;
}

void normalize(std::vector<float> & samples) {
    const float peak = peak_of(samples);
    if (peak <= kGainFloor) return;
    std::transform(samples.begin(), samples.end(), samples.begin(),
        [peak](float sample) { return sample / peak; });
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

int largest_capacity(const parakeet::ParakeetCtcModel & model) {
#ifdef PARAKEET_USE_COREML
    parakeet::CoremlDiarizers * diarizers = parakeet::model_coreml_diarizers(model);
    const std::vector<int> capacities = diarizers ? diarizers->capacities() : std::vector<int>();
    return capacities.empty() ? 0 : capacities.back();
#else
    (void) model;
    return 0;
#endif
}

std::string loaded_label(const parakeet::ParakeetCtcModel & model) {
#ifdef PARAKEET_USE_COREML
    parakeet::CoremlDiarizers * diarizers = parakeet::model_coreml_diarizers(model);
    return diarizers ? diarizers->label() : std::string();
#else
    (void) model;
    return std::string();
#endif
}

bool defaults_to_neural_engine(const parakeet::ParakeetCtcModel & model) {
    if (std::getenv(kUnitsEnv) != nullptr) return true;
    const std::string label = loaded_label(model);
    const bool ane = label == kNeuralEngineLabel;
    std::printf("[%s] default-units: label=%s\n", ane ? "ok  " : "FAIL", label.c_str());
    return ane;
}

bool diarizer_env_hides_sidecars(const char * path) {
    set_env(kDiarizerDisableEnv, "1");
    parakeet::ParakeetCtcModel model;
    const bool loaded = parakeet::load_from_gguf(path, model, kThreads, kGpuLayers, false) == 0;
    unset_env(kDiarizerDisableEnv);
    const bool hidden = loaded && largest_capacity(model) == 0;
    std::printf("[%s] diarizer-disable-env: coreml=%d\n", hidden ? "ok  " : "FAIL", hidden ? 0 : 1);
    return hidden;
}

bool ggml_diarize(parakeet::ParakeetCtcModel & model, const std::vector<float> & pre_encode, int frames,
                  parakeet::SortformerDiarizationResult & result) {
    parakeet::EncoderOutputs encoded;
    const int width = model.encoder_cfg.sortformer_fc_d_model;
    return parakeet::run_encoder_bypass_pre_encode(model, pre_encode.data(), frames, width, encoded) == 0 &&
           parakeet::sortformer_diarize_ggml(model, encoded.encoder_out.data(), frames, width, {}, result) == 0;
}

std::vector<float> tile_rows(const std::vector<float> & rows, int width, int count) {
    const size_t available = rows.size() / width;
    std::vector<float> tiled(static_cast<size_t>(count) * width);
    for (int row = 0; row < count; ++row) {
        std::copy_n(rows.begin() + (row % available) * width, width, tiled.begin() + row * width);
    }
    return tiled;
}

bool report(const char * name, int frames, bool routed, const Comparison & comparison) {
    const bool passed = routed && comparison.maximum_error <= kMaximumProbabilityError &&
                        comparison.relative_error <= kMaximumRelativeError;
    std::printf("[%s] %s: frames=%d coreml=%d max=%.5f rel=%.6f\n", passed ? "ok  " : "FAIL",
        name, frames, routed ? 1 : 0, comparison.maximum_error, comparison.relative_error);
    return passed;
}

bool batch_case(parakeet::ParakeetCtcModel & accelerated, parakeet::ParakeetCtcModel & reference,
                const std::vector<float> & mel, int mel_frames) {
    parakeet::SortformerDiarizationResult actual;
    const bool routed = parakeet::sortformer_diarize_mel_coreml(accelerated, mel.data(), mel_frames, {}, actual);
    parakeet::EncoderOutputs encoded;
    parakeet::SortformerDiarizationResult expected;
    if (parakeet::run_encoder(reference, mel.data(), mel_frames, reference.mel_cfg.n_mels, encoded,
                              -1, false, false) != 0 ||
        parakeet::sortformer_diarize_ggml(reference, encoded.encoder_out.data(), encoded.n_enc_frames,
                                          encoded.d_model, {}, expected) != 0) {
        return false;
    }
    return report("batch", expected.n_frames, routed, compare(actual.speaker_probs, expected.speaker_probs));
}

bool slab_case(parakeet::ParakeetCtcModel & accelerated, parakeet::ParakeetCtcModel & reference,
               const std::vector<float> & slab, int frames, bool expects_coreml, const char * name) {
    parakeet::SortformerDiarizationResult actual;
    const int width = accelerated.encoder_cfg.sortformer_fc_d_model;
    const bool routed = parakeet::sortformer_diarize_coreml(accelerated, slab.data(), frames, width, {}, actual);
    parakeet::SortformerDiarizationResult expected;
    if (!ggml_diarize(reference, slab, frames, expected)) return false;
    if (!expects_coreml) {
        std::printf("[%s] %s: frames=%d coreml=%d\n", routed ? "FAIL" : "ok  ", name, frames, routed ? 1 : 0);
        return !routed;
    }
    return report(name, frames, routed, compare(actual.speaker_probs, expected.speaker_probs));
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
    const int capacity = largest_capacity(accelerated);
    if (capacity <= 0) {
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
    std::vector<float> pre_encode;
    int frames = 0;
    if (parakeet::run_subsampling(reference, mel.data(), mel_frames, reference.mel_cfg.n_mels,
            pre_encode, frames) != 0) return 1;
    const int width = reference.encoder_cfg.sortformer_fc_d_model;
    const int slab_frames = std::min(kAoscFrames, capacity);
    bool passed = frames > capacity || batch_case(accelerated, reference, mel, mel_frames);
    passed = slab_case(accelerated, reference, tile_rows(pre_encode, width, slab_frames), slab_frames,
                       true, "aosc-slab") && passed;
    passed = slab_case(accelerated, reference, tile_rows(pre_encode, width, capacity + 1), capacity + 1,
                       false, "above-capacity") && passed;
    passed = defaults_to_neural_engine(accelerated) && passed;
    passed = diarizer_env_hides_sidecars(argv[1]) && passed;
    std::printf("[sortformer-diarizer-coreml-parity] %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
