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

#include "tts-cpp/moss/sound_effect.h"
#include "voice_features.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;

namespace {

constexpr int SKIP = 77;
constexpr int PARITY_THREADS = 8;
constexpr double CODE_AGREEMENT = 0.95;
constexpr double STAGE_COSINE = 0.999;
constexpr size_t FULL_SEGMENTS = 2;
constexpr double SHORT_CLIP_SECONDS = 2.0;
constexpr uint32_t NOISE_SEED = 7;
constexpr uint32_t CODE_SEED = 11;
constexpr float PROBE_SIGMA = 0.9f;
constexpr int VAE_KEEP_FRAMES = 500;
constexpr int CODEC_FRAMES = 160;
constexpr int CODEC_CODEBOOK = 1024;
constexpr int CODEC_STREAM_PIECE_FRAMES = 25;
constexpr const char * SFX_PROMPT = "Heavy rain falling on a tin roof with distant thunder. duration: 10.0s";
const std::vector<int> CODEC_PIECES = {25, 7, 13, 1, 30, 25, 4};
constexpr const char * SPEECH_CODEC_ENV = "MOSS_SPEECH_COREML_CODEC";
constexpr const char * SFX_MODEL_ENV = "MOSS_SFX_COREML_MODEL";
constexpr const char * CODEC_DECODER_ENV = "MOSS_CODEC_COREML_DECODER";
constexpr const char * GPU_ENV = "MOSS_COREML_PARITY_GPU";
constexpr const char * GPU_PLACEMENT_LABEL = "coreml-gpu";

int failures = 0;

void check(bool condition, const std::string & label) {
    std::printf("%s: %s\n", condition ? "ok  " : "FAIL", label.c_str());
    if (!condition) {
        failures++;
    }
}

void set_env(const char * name) {
    setenv(name, "1", 1);
}

void clear_env(const char * name) {
    unsetenv(name);
}

bool use_gpu() {
    return std::getenv(GPU_ENV) != nullptr;
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

std::filesystem::path scratch_dir(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
        ("moss-coreml-" + std::string(tag) + "-" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
    return dir;
}

std::filesystem::path linked_file(const std::filesystem::path & dir, const std::string & path) {
    const auto link = dir / std::filesystem::path(path).filename();
    std::filesystem::create_symlink(std::filesystem::absolute(path), link);
    return link;
}

std::vector<float> read_wav(const std::string & path) {
    std::vector<float> pcm;
    int rate = 0;
    if (!wav_load(path, pcm, rate)) {
        throw std::runtime_error("cannot read " + path);
    }
    return pcm;
}

std::vector<float> looped(const std::vector<float> & pcm, size_t samples) {
    std::vector<float> out;
    out.reserve(samples);
    while (out.size() < samples) {
        const size_t take = std::min(pcm.size(), samples - out.size());
        out.insert(out.end(), pcm.begin(), pcm.begin() + (std::ptrdiff_t) take);
    }
    return out;
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

void report_cosine(const std::string & label, double value) {
    std::printf("%s: cosine %.6f\n", label.c_str(), value);
    check(value >= STAGE_COSINE, label + " matches ggml");
}

double code_agreement(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    size_t same = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        same += a[i] == b[i] ? 1 : 0;
    }
    return (double) same / (double) a.size();
}

void compare_codes(SpeechTokenizer & coreml, SpeechTokenizer & ggml, const std::vector<float> & audio,
                   const std::string & label) {
    coreml.begin_run();
    auto start = std::chrono::steady_clock::now();
    const std::vector<int32_t> accelerated = coreml.encode(audio);
    const double coreml_ms = elapsed_ms(start);
    start = std::chrono::steady_clock::now();
    const std::vector<int32_t> reference = ggml.encode(audio);
    const double ggml_ms = elapsed_ms(start);
    const double agreement = code_agreement(accelerated, reference);
    std::printf("%s: %zu codes, agreement %.4f, Core ML %.0f ms, ggml %.0f ms\n", label.c_str(), reference.size(),
                agreement, coreml_ms, ggml_ms);
    check(coreml.run_backend().rfind("coreml", 0) == 0, label + ": encoded on the sidecar (" + coreml.run_backend() + ")");
    check(agreement >= CODE_AGREEMENT, label + ": codes follow the ggml tokenizer");
}

void check_default_tokenizer_placement(const SpeechTokenizer & coreml) {
    if (std::getenv(COREML_UNITS_ENV) == nullptr) {
        check(coreml.run_backend() == GPU_PLACEMENT_LABEL, "the tokenizer defaults to the Core ML GPU placement");
    }
}

void check_partial_segment_on_ggml(SpeechTokenizer & coreml, SpeechTokenizer & ggml, const std::vector<float> & clip) {
    coreml.begin_run();
    const bool same = coreml.encode(clip) == ggml.encode(clip);
    check(same && coreml.run_backend() == GGML_STAGE_BACKEND, "a partial segment stays on the ggml tokenizer");
}

void test_tokenizer_codes(const std::string & codec_path, const std::vector<float> & audio) {
    set_env(COREML_STRICT_ENV);
    SpeechTokenizer coreml(codec_path, use_gpu(), PARITY_THREADS);
    clear_env(COREML_STRICT_ENV);
    set_env(COREML_DISABLE_ENV);
    SpeechTokenizer ggml(codec_path, use_gpu(), PARITY_THREADS);
    clear_env(COREML_DISABLE_ENV);
    check(coreml.on_coreml() && !ggml.on_coreml(), "the staged tokenizer sidecar attaches; MOSS_COREML_DISABLE skips it");
    if (!coreml.on_coreml()) {
        return;
    }
    const size_t segment = (size_t) coreml.config().chunk_samples;
    compare_codes(coreml, ggml, looped(audio, segment), "one segment");
    compare_codes(coreml, ggml, looped(audio, FULL_SEGMENTS * segment), "two segments");
    check_default_tokenizer_placement(coreml);
    const size_t short_clip = (size_t) (SHORT_CLIP_SECONDS * coreml.config().sample_rate);
    check_partial_segment_on_ggml(coreml, ggml, looped(audio, short_clip));
}

void test_tokenizer_strict_without_sidecar(const std::string & codec_path, const std::vector<float> & audio) {
    const auto dir = scratch_dir("speech-strict");
    const auto bare = linked_file(dir, codec_path);
    set_env(COREML_STRICT_ENV);
    bool failed = false;
    try {
        SpeechTokenizer tokenizer(bare.string(), use_gpu(), PARITY_THREADS);
        tokenizer.encode(audio);
    } catch (const std::exception & e) {
        failed = std::string(e.what()).find(COREML_STRICT_ENV) != std::string::npos;
    }
    clear_env(COREML_STRICT_ENV);
    check(failed, "MOSS_COREML_STRICT fails a tokenizer without a sidecar");
    std::filesystem::remove_all(dir);
}

void test_tokenizer_invalid_sidecar(const std::string & codec_path, const std::vector<float> & audio) {
    const auto dir = scratch_dir("speech-invalid");
    const auto staged = linked_file(dir, codec_path);
    std::filesystem::create_directories(speech_tokenizer_sidecar_path(staged.string()));
    SpeechTokenizer tokenizer(staged.string(), use_gpu(), PARITY_THREADS);
    tokenizer.begin_run();
    tokenizer.encode(audio);
    check(!tokenizer.on_coreml() && tokenizer.run_backend() == GGML_STAGE_BACKEND,
          "a tokenizer sidecar directory that is not a model leaves the tokenizer on ggml");
    std::filesystem::remove_all(dir);
}

bool run_speech_tokenizer(const std::vector<float> & audio) {
    const char * codec_path = std::getenv(SPEECH_CODEC_ENV);
    if (codec_path == nullptr) {
        std::printf("speech tokenizer: skipped (set %s)\n", SPEECH_CODEC_ENV);
        return false;
    }
    test_tokenizer_codes(codec_path, audio);
    test_tokenizer_strict_without_sidecar(codec_path, audio);
    test_tokenizer_invalid_sidecar(codec_path, audio);
    return true;
}

void test_sfx_dit(tts_cpp::moss::detail::SfxModel & model, const std::string & model_path) {
    auto sidecar = open_sfx_dit_sidecar(model_path, model.config(), {});
    check(sidecar != nullptr, "the staged DiT sidecar loads and matches the model");
    if (!sidecar) {
        return;
    }
    const SfxConfig & config = model.config();
    const SfxTokenizer tokenizer(model);
    const std::vector<float> context = encode_text(model, tokenizer.encode(SFX_PROMPT));
    const std::vector<float> latents = gaussian_noise((size_t) config.latent_frames() * config.dit.in_channels, NOISE_SEED);
    const float timestep = PROBE_SIGMA * (float) config.train_timesteps;
    auto start = std::chrono::steady_clock::now();
    SfxDitSession session(model, config.latent_frames());
    const std::vector<float> ggml = session.velocity(latents, timestep, context);
    const double ggml_ms = elapsed_ms(start);
    std::vector<float> accelerated;
    sidecar->velocity(latents, timestep_sinusoid(timestep, config.dit.freq_dim), context, accelerated);
    start = std::chrono::steady_clock::now();
    check(sidecar->velocity(latents, timestep_sinusoid(timestep, config.dit.freq_dim), context, accelerated),
          "the DiT sidecar predicts");
    std::printf("DiT velocity: Core ML %.0f ms (warm), ggml %.0f ms\n", elapsed_ms(start), ggml_ms);
    report_cosine("DiT velocity", cosine(accelerated, ggml));
}

void test_sfx_vae(tts_cpp::moss::detail::SfxModel & model, const std::string & model_path) {
    auto sidecar = open_sfx_vae_sidecar(model_path, model.config(), {});
    check(sidecar != nullptr, "the staged VAE sidecar loads and matches the model");
    if (!sidecar) {
        return;
    }
    const SfxConfig & config = model.config();
    const int frames = config.latent_frames();
    const std::vector<float> latents = gaussian_noise((size_t) frames * config.vae.latent_dim, NOISE_SEED);
    auto start = std::chrono::steady_clock::now();
    const std::vector<float> ggml = decode_latents(model, latents, frames, VAE_KEEP_FRAMES, SFX_DECODE_WINDOW_FRAMES, {});
    const double ggml_ms = elapsed_ms(start);
    std::vector<float> accelerated;
    start = std::chrono::steady_clock::now();
    const SidecarDecode outcome = decode_latents_on_sidecar(*sidecar, config.vae, latents, frames, VAE_KEEP_FRAMES,
            {}, accelerated);
    std::printf("VAE decode of %d frames: Core ML %.0f ms (window %d), ggml %.0f ms\n", VAE_KEEP_FRAMES,
                elapsed_ms(start), sidecar->window(), ggml_ms);
    check(outcome == SidecarDecode::Done, "the VAE sidecar decodes every window");
    report_cosine("VAE waveform", cosine(accelerated, ggml));
}

void test_sfx_engine(const std::string & model_path) {
    tts_cpp::moss::SoundEffectOptions options;
    options.model_path = model_path;
    options.n_threads = PARITY_THREADS;
    options.use_gpu = use_gpu();
    set_env(COREML_STRICT_ENV);
    tts_cpp::moss::SoundEffectEngine engine(options);
    clear_env(COREML_STRICT_ENV);
    check(engine.dit_on_coreml() && engine.vae_on_coreml(), "the engine attaches both SoundEffect sidecars");
}

void test_sfx_invalid_sidecars(const std::string & model_path) {
    const auto dir = scratch_dir("sfx-invalid");
    const auto staged = linked_file(dir, model_path);
    std::filesystem::create_directories(sfx_dit_sidecar_path(staged.string()));
    std::filesystem::create_directories(sfx_vae_sidecar_path(staged.string()));
    tts_cpp::moss::SoundEffectOptions options;
    options.model_path = staged.string();
    options.n_threads = PARITY_THREADS;
    options.use_gpu = use_gpu();
    tts_cpp::moss::SoundEffectEngine engine(options);
    check(!engine.dit_on_coreml() && !engine.vae_on_coreml(),
          "SoundEffect sidecar directories that are not models leave both stages on ggml");
    std::filesystem::remove_all(dir);
}

bool run_sound_effect() {
    const char * model_path = std::getenv(SFX_MODEL_ENV);
    if (model_path == nullptr) {
        std::printf("sound effect: skipped (set %s)\n", SFX_MODEL_ENV);
        return false;
    }
    {
        tts_cpp::moss::detail::SfxModel model(model_path, use_gpu(), PARITY_THREADS);
        test_sfx_dit(model, model_path);
        test_sfx_vae(model, model_path);
    }
    test_sfx_engine(model_path);
    test_sfx_invalid_sidecars(model_path);
    return true;
}

std::vector<int32_t> random_codes(int frames, int channels, int codebook) {
    std::mt19937 rng(CODE_SEED);
    std::uniform_int_distribution<int32_t> draw(0, codebook - 1);
    std::vector<int32_t> codes((size_t) frames * channels);
    for (int32_t & code : codes) {
        code = draw(rng);
    }
    return codes;
}

std::vector<float> stream_codes(Codec & codec, const std::vector<int32_t> & codes, int channels) {
    std::vector<float> pcm;
    codec.begin_decode_stream(0, CODEC_STREAM_PIECE_FRAMES);
    const int frames = (int) (codes.size() / (size_t) channels);
    size_t turn = 0;
    for (int first = 0; first < frames; ++turn) {
        const int count = std::min(CODEC_PIECES[turn % CODEC_PIECES.size()], frames - first);
        const std::vector<float> piece = codec.decode_stream(std::vector<int32_t>(
                codes.begin() + (std::ptrdiff_t) first * channels, codes.begin() + (std::ptrdiff_t) (first + count) * channels));
        pcm.insert(pcm.end(), piece.begin(), piece.end());
        first += count;
    }
    return pcm;
}

void test_codec_decode(const std::string & decoder_path) {
    set_env(COREML_STRICT_ENV);
    Codec coreml(decoder_path, {"", use_gpu()}, PARITY_THREADS);
    clear_env(COREML_STRICT_ENV);
    set_env(COREML_DISABLE_ENV);
    Codec ggml(decoder_path, {"", use_gpu()}, PARITY_THREADS);
    clear_env(COREML_DISABLE_ENV);
    check(coreml.on_coreml() && !ggml.on_coreml(), "the staged codec sidecar attaches; MOSS_COREML_DISABLE skips it");
    if (!coreml.on_coreml()) {
        return;
    }
    const int channels = coreml.num_quantizers();
    const std::vector<int32_t> codes = random_codes(CODEC_FRAMES, channels, CODEC_CODEBOOK);
    coreml.begin_run();
    const std::vector<float> streamed = stream_codes(coreml, codes, channels);
    check(coreml.run_backend().rfind("coreml", 0) == 0,
          "a stream of chunk-sized pieces decodes on the sidecar (" + coreml.run_backend() + ")");
    report_cosine("codec streamed decode", cosine(streamed, stream_codes(ggml, codes, channels)));
    coreml.begin_run();
    coreml.decode(codes);
    check(coreml.run_backend() == GGML_STAGE_BACKEND, "a batch decode stays on the ggml codec");
}

void test_codec_invalid_sidecar(const std::string & decoder_path) {
    const auto dir = scratch_dir("codec-invalid");
    const auto staged = linked_file(dir, decoder_path);
    std::filesystem::create_directories(codec_decoder_sidecar_path(staged.string()));
    Codec codec(staged.string(), {"", use_gpu()}, PARITY_THREADS);
    check(!codec.on_coreml(), "a codec sidecar directory that is not a model leaves the codec on ggml");
    std::filesystem::remove_all(dir);
}

bool run_codec() {
    const char * decoder_path = std::getenv(CODEC_DECODER_ENV);
    if (decoder_path == nullptr) {
        std::printf("codec: skipped (set %s)\n", CODEC_DECODER_ENV);
        return false;
    }
    test_codec_decode(decoder_path);
    test_codec_invalid_sidecar(decoder_path);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
#ifndef TTS_CPP_USE_COREML
    std::printf("SKIP: built without TTS_CPP_COREML\n");
    return SKIP;
#endif
    if (argc < 2) {
        std::printf("SKIP: pass a 16 kHz speech WAV\n");
        return SKIP;
    }
    bool ran = false;
    try {
        const std::vector<float> speech = read_wav(argv[1]);
        ran = run_speech_tokenizer(speech) || ran;
        ran = run_sound_effect() || ran;
        ran = run_codec() || ran;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (!ran) {
        std::printf("SKIP: no MOSS Core ML sidecar staged\n");
        return SKIP;
    }
    if (failures == 0) {
        std::printf("moss Core ML parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss Core ML parity: %d failures\n", failures);
    return 1;
}
