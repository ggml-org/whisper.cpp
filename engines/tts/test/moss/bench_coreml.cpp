#include "moss/codec.h"
#include "moss/coreml_sidecar.h"
#include "moss/sfx_coreml.h"
#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_request.h"
#include "moss/sfx_sampler.h"
#include "moss/sfx_tokenizer.h"
#include "moss/speech_tokenizer.h"

#include "../test_env_portable.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;

namespace {

constexpr int SKIP = 77;
constexpr int BENCH_THREADS = 8;
constexpr int RUNS = 5;
constexpr double PARITY_COSINE = 0.999;
constexpr uint32_t SEED = 7;
constexpr int VAE_FRAMES_KEPT = 1500;
constexpr int CODEC_FRAMES = 250;
constexpr int CODEC_CODEBOOK = 1024;
constexpr int CODEC_STREAM_CHUNK_FRAMES = 25;
constexpr float PROBE_SIGMA = 0.9f;
constexpr const char * SFX_PROMPT = "Heavy rain falling on a tin roof with distant thunder. duration: 10.0s";
constexpr const char * SPEECH_CODEC_ENV = "MOSS_SPEECH_COREML_CODEC";
constexpr const char * SFX_MODEL_ENV = "MOSS_SFX_COREML_MODEL";
constexpr const char * CODEC_DECODER_ENV = "MOSS_CODEC_COREML_DECODER";

int failures = 0;

struct Row {
    std::string stage;
    std::string workload;
    double ggml_ms = 0;
    double coreml_ms = 0;
    double agreement = 0;
    std::string label;
};

std::vector<Row> rows;

double median_ms(const std::function<void()> & run) {
    run();
    std::vector<double> times;
    for (int i = 0; i < RUNS; ++i) {
        const auto start = std::chrono::steady_clock::now();
        run();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return -2.0;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

double code_agreement(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    size_t same = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        same += a[i] == b[i] ? 1 : 0;
    }
    return a.size() == b.size() && !a.empty() ? (double) same / (double) a.size() : 0.0;
}

void gate(const Row & row, double floor) {
    rows.push_back(row);
    if (row.agreement < floor) {
        std::fprintf(stderr, "FAIL: %s %s agreement %.6f below %.3f\n", row.stage.c_str(), row.workload.c_str(),
                     row.agreement, floor);
        failures++;
    }
}

void set_env(const char * name) {
    setenv(name, "1", 1);
}

void clear_env(const char * name) {
    unsetenv(name);
}

std::vector<float> speech_like(size_t samples) {
    std::mt19937 rng(SEED);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    std::vector<float> pcm(samples);
    for (size_t i = 0; i < samples; ++i) {
        pcm[i] = 0.3f * std::sin(0.03f * (float) i) * std::sin(0.0004f * (float) i) + noise(rng);
    }
    return pcm;
}

void bench_tokenizer_segment(SpeechTokenizer & coreml, SpeechTokenizer & ggml, size_t samples,
                             const std::string & workload) {
    const std::vector<float> pcm = speech_like(samples);
    Row row{"speech tokenizer", workload};
    std::vector<int32_t> fast;
    std::vector<int32_t> reference;
    row.coreml_ms = median_ms([&] { fast = coreml.encode(pcm); });
    row.ggml_ms = median_ms([&] { reference = ggml.encode(pcm); });
    coreml.begin_run();
    coreml.encode(pcm);
    row.label = coreml.run_backend();
    row.agreement = code_agreement(fast, reference);
    rows.push_back(row);
}

void bench_tokenizer(const std::string & codec_path) {
    set_env(COREML_STRICT_ENV);
    SpeechTokenizer coreml(codec_path, {"", true}, BENCH_THREADS);
    clear_env(COREML_STRICT_ENV);
    set_env(COREML_DISABLE_ENV);
    SpeechTokenizer ggml(codec_path, {"", true}, BENCH_THREADS);
    clear_env(COREML_DISABLE_ENV);
    bench_tokenizer_segment(coreml, ggml, (size_t) coreml.config().chunk_samples, "30 s segment");
}

void bench_dit(SfxModel & model, const std::string & model_path) {
    auto sidecar = open_sfx_dit_sidecar(model_path, model.config(), {});
    if (!sidecar) {
        std::fprintf(stderr, "FAIL: no DiT sidecar\n");
        failures++;
        return;
    }
    const SfxConfig & config = model.config();
    const SfxTokenizer tokenizer(model);
    const std::vector<float> context = encode_text(model, tokenizer.encode(SFX_PROMPT));
    const std::vector<float> latents = gaussian_noise((size_t) config.latent_frames() * config.dit.in_channels, SEED);
    const float timestep = PROBE_SIGMA * (float) config.train_timesteps;
    const std::vector<float> sinusoid = timestep_sinusoid(timestep, config.dit.freq_dim);
    SfxDitSession session(model, config.latent_frames());
    Row row{"SoundEffect DiT", "one velocity, 30 s latent"};
    std::vector<float> fast;
    std::vector<float> reference;
    row.ggml_ms = median_ms([&] { reference = session.velocity(latents, timestep, context); });
    row.coreml_ms = median_ms([&] { sidecar->velocity(latents, sinusoid, context, fast); });
    row.label = sidecar->label();
    row.agreement = cosine(fast, reference);
    gate(row, PARITY_COSINE);
}

void bench_vae(SfxModel & model, const std::string & model_path) {
    auto sidecar = open_sfx_vae_sidecar(model_path, model.config(), {});
    if (!sidecar) {
        std::fprintf(stderr, "FAIL: no VAE sidecar\n");
        failures++;
        return;
    }
    const SfxConfig & config = model.config();
    const int frames = config.latent_frames();
    const std::vector<float> latents = gaussian_noise((size_t) frames * config.vae.latent_dim, SEED);
    Row row{"SoundEffect VAE", "decode 30 s"};
    std::vector<float> fast;
    std::vector<float> reference;
    row.ggml_ms = median_ms([&] {
        reference = decode_latents(model, latents, frames, VAE_FRAMES_KEPT, SFX_DECODE_WINDOW_FRAMES, {});
    });
    row.coreml_ms = median_ms([&] {
        decode_latents_on_sidecar(*sidecar, config.vae, latents, frames, VAE_FRAMES_KEPT, {}, fast);
    });
    row.label = sidecar->label();
    row.agreement = cosine(fast, reference);
    gate(row, PARITY_COSINE);
}

void bench_sound_effect(const std::string & model_path) {
    SfxModel model(model_path, {"", true}, BENCH_THREADS);
    bench_dit(model, model_path);
    bench_vae(model, model_path);
}

std::vector<int32_t> random_codes(int frames, int channels) {
    std::mt19937 rng(SEED);
    std::uniform_int_distribution<int32_t> draw(0, CODEC_CODEBOOK - 1);
    std::vector<int32_t> codes((size_t) frames * channels);
    for (int32_t & code : codes) {
        code = draw(rng);
    }
    return codes;
}

std::vector<float> stream_decode(Codec & codec, const std::vector<int32_t> & codes, int chunk_frames) {
    const int channels = codec.num_quantizers();
    const int frames = (int) (codes.size() / (size_t) channels);
    std::vector<float> pcm;
    codec.begin_decode_stream(0, chunk_frames);
    for (int first = 0; first < frames; first += chunk_frames) {
        const int count = std::min(chunk_frames, frames - first);
        const std::vector<float> piece = codec.decode_stream(std::vector<int32_t>(
                codes.begin() + (std::ptrdiff_t) first * channels, codes.begin() + (std::ptrdiff_t) (first + count) * channels));
        pcm.insert(pcm.end(), piece.begin(), piece.end());
    }
    return pcm;
}

void bench_codec_stream(Codec & coreml, Codec & ggml, const std::vector<int32_t> & codes) {
    Row row{"TTS codec decoder", "stream 20 s in 25-frame chunks"};
    std::vector<float> fast;
    std::vector<float> reference;
    row.coreml_ms = median_ms([&] { fast = stream_decode(coreml, codes, CODEC_STREAM_CHUNK_FRAMES); });
    row.ggml_ms = median_ms([&] { reference = stream_decode(ggml, codes, CODEC_STREAM_CHUNK_FRAMES); });
    coreml.begin_run();
    stream_decode(coreml, codes, CODEC_STREAM_CHUNK_FRAMES);
    row.label = coreml.run_backend();
    row.agreement = cosine(fast, reference);
    gate(row, PARITY_COSINE);
}

void bench_codec(const std::string & decoder_path) {
    set_env(COREML_STRICT_ENV);
    Codec coreml(decoder_path, {"", true}, BENCH_THREADS);
    clear_env(COREML_STRICT_ENV);
    set_env(COREML_DISABLE_ENV);
    Codec ggml(decoder_path, {"", true}, BENCH_THREADS);
    clear_env(COREML_DISABLE_ENV);
    bench_codec_stream(coreml, ggml, random_codes(CODEC_FRAMES, coreml.num_quantizers()));
}

void print_table() {
    std::printf("| stage | workload | ggml GPU | Core ML | speedup | agreement | placement |\n");
    std::printf("|---|---|---:|---:|---:|---:|---|\n");
    for (const Row & row : rows) {
        std::printf("| %s | %s | %.1f ms | %.1f ms | %.2fx | %.6f | %s |\n", row.stage.c_str(), row.workload.c_str(),
                    row.ggml_ms, row.coreml_ms, row.ggml_ms / row.coreml_ms, row.agreement, row.label.c_str());
    }
}

bool run_stage(const char * env, const std::function<void(const std::string &)> & bench) {
    const char * path = std::getenv(env);
    if (path == nullptr) {
        return false;
    }
    bench(path);
    return true;
}

} // namespace

int main() {
#ifndef TTS_CPP_USE_COREML
    std::printf("SKIP: built without TTS_CPP_COREML\n");
    return SKIP;
#endif
    bool ran = false;
    try {
        ran = run_stage(SPEECH_CODEC_ENV, bench_tokenizer) || ran;
        ran = run_stage(SFX_MODEL_ENV, bench_sound_effect) || ran;
        ran = run_stage(CODEC_DECODER_ENV, bench_codec) || ran;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (!ran) {
        std::printf("SKIP: set %s, %s or %s\n", SPEECH_CODEC_ENV, SFX_MODEL_ENV, CODEC_DECODER_ENV);
        return SKIP;
    }
    print_table();
    return failures == 0 ? 0 : 1;
}
