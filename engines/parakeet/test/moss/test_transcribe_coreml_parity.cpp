#include "moss/transcribe_audio.h"
#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_text.h"

#include "mel_preprocess.h"
#include "parakeet/moss_transcribe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parakeet::moss::detail;

namespace {

constexpr int SKIP = 77;
constexpr double WINDOW_COSINE = 0.999;
constexpr double TOKEN_COSINE = 0.99;
constexpr double TRANSCRIPT_AGREEMENT = 0.95;
constexpr int PARITY_THREADS = 8;
constexpr double SHORT_WINDOW_SECONDS = 3.0;
constexpr const char * MODEL_ENV = "MOSS_TRANSCRIBE_COREML_MODEL";
constexpr const char * GPU_ENV = "MOSS_TRANSCRIBE_GPU";

int failures = 0;

void check(bool condition, const std::string & label) {
    std::printf("%s: %s\n", condition ? "ok  " : "FAIL", label.c_str());
    if (!condition) {
        failures++;
    }
}

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void clear_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

double cosine(const float * a, const float * b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

double worst_token_cosine(const std::vector<float> & a, const std::vector<float> & b, int width) {
    double worst = 1.0;
    for (size_t row = 0; row * width < a.size(); ++row) {
        worst = std::min(worst, cosine(a.data() + row * width, b.data() + row * width, (size_t) width));
    }
    return worst;
}

double token_agreement(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t count = std::min(a.size(), b.size());
    size_t same = 0;
    while (same < count && a[same] == b[same]) {
        same++;
    }
    return (double) same / (double) std::max<size_t>(std::max(a.size(), b.size()), 1);
}

std::vector<float> read_wav(const std::string & path) {
    std::vector<float> pcm;
    int rate = 0;
    if (parakeet::load_wav_mono_f32(path, pcm, rate) != 0) {
        throw std::runtime_error("cannot read " + path);
    }
    return pcm;
}

std::filesystem::path scratch_dir(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
        ("moss-coreml-" + std::string(tag) + "-" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
    return dir;
}

std::filesystem::path linked_model(const std::filesystem::path & dir, const std::string & model_path) {
    const auto link = dir / std::filesystem::path(model_path).filename();
    std::filesystem::create_symlink(std::filesystem::absolute(model_path), link);
    return link;
}

parakeet::moss::TranscribeOptions engine_options(const std::string & model_path) {
    parakeet::moss::TranscribeOptions options;
    options.model_path = model_path;
    options.n_threads = PARITY_THREADS;
    options.use_gpu = std::getenv(GPU_ENV) != nullptr;
    return options;
}

void compare_window(TranscribeModel & model, TranscribeEncoderSidecar & sidecar, const std::vector<float> & audio,
                    size_t samples, const std::string & label) {
    const TranscribeConfig & config = model.config();
    const TranscribeMel mel(config.audio, read_mel_filters(model));
    const size_t count = std::min(samples, audio.size());
    const std::vector<float> window = mel.chunk(audio.data(), count);
    const int tokens = transcribe_chunk_tokens(config, count);
    const std::vector<float> ggml = encode_audio_chunk(model, window, tokens, false).embeddings;
    std::vector<float> rows;
    check(sidecar.encode(window, rows), label + ": the sidecar predicts");
    rows.resize(ggml.size());
    const double whole = cosine(rows.data(), ggml.data(), ggml.size());
    const double worst = worst_token_cosine(rows, ggml, config.text.n_embd);
    std::printf("%s: %d tokens, cosine %.6f, worst token %.6f\n", label.c_str(), tokens, whole, worst);
    check(whole >= WINDOW_COSINE, label + ": embeddings match the ggml encoder");
    check(worst >= TOKEN_COSINE, label + ": every token matches the ggml encoder");
}

void test_windows(const std::string & model_path, const std::vector<float> & audio) {
    TranscribeModel model(model_path, std::getenv(GPU_ENV) != nullptr, PARITY_THREADS);
    auto sidecar = open_transcribe_encoder_sidecar(model_path, model.config(), {});
    check(sidecar != nullptr, "the staged sidecar loads and matches the model window");
    if (!sidecar) {
        return;
    }
    const auto & audio_config = model.config().audio;
    compare_window(model, *sidecar, audio, (size_t) audio_config.chunk_samples, "first window");
    compare_window(model, *sidecar, audio, (size_t) (SHORT_WINDOW_SECONDS * audio_config.sample_rate),
                   "zero-padded window");
}

parakeet::moss::TranscribeResult transcribe(parakeet::moss::TranscribeEngine & engine,
                                            const std::vector<float> & audio) {
    return engine.transcribe(audio.data(), audio.size(), engine.sample_rate());
}

void test_transcripts(const std::string & model_path, const std::vector<float> & audio) {
    set_env(COREML_STRICT_ENV, "1");
    parakeet::moss::TranscribeEngine coreml(engine_options(model_path));
    const auto accelerated = transcribe(coreml, audio);
    clear_env(COREML_STRICT_ENV);
    set_env(COREML_DISABLE_ENV, "1");
    parakeet::moss::TranscribeEngine ggml(engine_options(model_path));
    const auto reference = transcribe(ggml, audio);
    clear_env(COREML_DISABLE_ENV);
    check(coreml.encoder_on_coreml() && accelerated.encoder_backend.rfind("coreml", 0) == 0,
          "the engine encodes on the sidecar and reports its label (" + accelerated.encoder_backend + ")");
    check(!ggml.encoder_on_coreml() && reference.encoder_backend == GGML_ENCODER_BACKEND,
          "MOSS_COREML_DISABLE keeps the ggml encoder");
    std::printf("encode: Core ML %.0f ms, ggml %.0f ms\n", accelerated.encode_ms, reference.encode_ms);
    std::printf("transcript match: %s\n", accelerated.text == reference.text ? "exact" : "differs");
    TranscribeModel model(model_path, false, 1, "auto", true);
    const TranscribeTokenizer tokenizer(model);
    const double agreement = token_agreement(tokenizer.encode(accelerated.text), tokenizer.encode(reference.text));
    std::printf("greedy token prefix agreement: %.4f\n", agreement);
    check(agreement >= TRANSCRIPT_AGREEMENT, "the Core ML transcript follows the ggml one");
}

void test_strict_without_sidecar(const std::string & model_path, const std::vector<float> & audio) {
    const auto dir = scratch_dir("strict");
    const auto bare = linked_model(dir, model_path);
    set_env(COREML_STRICT_ENV, "1");
    bool failed = false;
    try {
        parakeet::moss::TranscribeEngine engine(engine_options(bare.string()));
        transcribe(engine, audio);
    } catch (const std::exception & e) {
        failed = std::string(e.what()).find(COREML_STRICT_ENV) != std::string::npos;
    }
    clear_env(COREML_STRICT_ENV);
    check(failed, "MOSS_COREML_STRICT fails a transcription without a sidecar");
    std::filesystem::remove_all(dir);
}

void test_invalid_sidecar(const std::string & model_path, const std::vector<float> & audio) {
    const auto dir = scratch_dir("invalid");
    const auto staged = linked_model(dir, model_path);
    std::filesystem::create_directories(transcribe_encoder_sidecar_path(staged.string()));
    parakeet::moss::TranscribeEngine engine(engine_options(staged.string()));
    check(!engine.encoder_on_coreml(), "a sidecar directory that is not a model is not attached");
    check(transcribe(engine, audio).encoder_backend == GGML_ENCODER_BACKEND,
          "an engine with an invalid sidecar transcribes on ggml");
    std::filesystem::remove_all(dir);
}

} // namespace

int main(int argc, char ** argv) {
#ifndef PARAKEET_USE_COREML
    std::printf("SKIP: built without PARAKEET_COREML\n");
    return SKIP;
#endif
    const char * model_path = std::getenv(MODEL_ENV);
    if (!model_path || argc < 2) {
        std::printf("SKIP: set %s to a GGUF with its -encoder.mlmodelc beside it, and pass a 16 kHz WAV\n",
                    MODEL_ENV);
        return SKIP;
    }
    try {
        const std::vector<float> audio = read_wav(argv[1]);
        test_windows(model_path, audio);
        test_transcripts(model_path, audio);
        test_strict_without_sidecar(model_path, audio);
        test_invalid_sidecar(model_path, audio);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss transcribe Core ML parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss transcribe Core ML parity: %d failures\n", failures);
    return 1;
}
