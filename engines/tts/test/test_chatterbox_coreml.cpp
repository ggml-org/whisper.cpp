// Chatterbox S3Gen Core ML sidecars.
//
//   test-chatterbox-coreml                  policy, naming, function buckets,
//                                           HiFT window plan, fit refusal
//   test-chatterbox-coreml --stages-fixture each sidecar against ggml on the
//                                           same tokens, one stage at a time,
//                                           then all three; routing past the
//                                           widest function and strict mode
//   test-chatterbox-coreml --shape-fixture  every feature shape a sidecar
//                                           declares is checked before
//                                           predicting
//   test-chatterbox-coreml --engine-fixture Engine sentence overlap against
//                                           in-order sequential synthesis,
//                                           cancellation, worker errors
//
// The stages fixture reads CHATTERBOX_COREML_TEST_S3GEN (an S3Gen GGUF with
// the exported `<stem>-{encoder,cfm,hift}.mlmodelc` next to it); the engine
// fixture also reads CHATTERBOX_COREML_TEST_T3 (the matching T3 GGUF); the
// shape fixture reads CHATTERBOX_COREML_TEST_SHAPES (the directory
// prepare_chatterbox_coreml_shape_fixture.py writes). Each reports a skip
// (77) without its inputs or without Core ML.

#include "chatterbox_coreml.h"
#include "coreml_sidecar_path.h"
#include "npy.h"
#include "test_env_portable.h"
#include "text_preprocess.h"
#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/fit.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cb = tts_cpp::chatterbox::detail;
namespace fs = std::filesystem;
using tts_cpp::chatterbox::Engine;
using tts_cpp::chatterbox::EngineOptions;
using tts_cpp::chatterbox::SynthesisResult;

namespace {

constexpr int SKIP = 77;

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

struct env_guard {
    std::string name;
    bool present;
    std::string value;
    explicit env_guard(const char * n) : name(n), present(std::getenv(n) != nullptr),
                                          value(present ? std::getenv(n) : "") {}
    ~env_guard() {
        if (present) setenv(name.c_str(), value.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

struct temp_dir {
    fs::path path;
    explicit temp_dir(const std::string & tag)
        : path(fs::path(test_tmpdir()) / ("chatterbox-coreml-" + tag + "-" + test_process_tag())) {
        std::error_code error;
        fs::remove_all(path, error);
        fs::create_directories(path);
    }
    ~temp_dir() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

void check_policy() {
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    auto policy = cb::resolve_coreml_policy();
    require(policy.compute_units == "cpu_and_ane" && !policy.strict && !policy.disabled, "default policy");

    setenv(cb::COREML_UNITS_ENV, "all", 1);
    setenv(cb::COREML_STRICT_ENV, "0", 1);  // presence means strict, as for Parler
    policy = cb::resolve_coreml_policy();
    require(policy.compute_units == "all" && policy.strict, "inherited policy");
    policy = cb::resolve_coreml_policy("cpu_only", false);
    require(policy.compute_units == "cpu_only" && !policy.strict, "explicit policy overrides the environment");

    setenv(cb::COREML_UNITS_ENV, "gpu_please", 1);
    require(cb::resolve_coreml_policy().compute_units == "cpu_and_ane", "invalid inherited units keep the default");
    bool threw = false;
    try {
        cb::resolve_coreml_policy("gpu_please");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "invalid explicit units throw");

    setenv(cb::COREML_DISABLE_ENV, "1", 1);
    require(cb::resolve_coreml_policy().disabled, "CHATTERBOX_COREML_DISABLE");
}

void check_naming() {
    require(cb::coreml_stage_path("models/chatterbox-s3gen-q8_0.gguf", cb::COREML_CFM_SUFFIX) ==
                "models/chatterbox-s3gen-cfm.mlmodelc", "turbo CFM sidecar name");
    require(cb::coreml_stage_path("m/chatterbox-s3gen-mtl-f32.gguf", cb::COREML_HIFT_SUFFIX) ==
                "m/chatterbox-s3gen-mtl-hift.mlmodelc", "multilingual HiFT sidecar name");
    require(cb::coreml_stage_path("chatterbox-s3gen.gguf", cb::COREML_ENCODER_SUFFIX) ==
                "chatterbox-s3gen-encoder.mlmodelc", "untagged encoder sidecar name");
}

void check_buckets() {
    const auto frames = cb::coreml_buckets("frames_640,tokens_256,frames_512,frames_,frames_x1,frames_640,main",
                                           cb::COREML_CFM_PREFIX);
    require(frames == std::vector<int>{512, 640}, "frame buckets");
    require(cb::coreml_buckets("tokens_256,tokens_320", cb::COREML_ENCODER_PREFIX) == std::vector<int>{256, 320},
            "token buckets");
    require(cb::coreml_pick_bucket(frames, 1) == 512 && cb::coreml_pick_bucket(frames, 512) == 512 &&
                cb::coreml_pick_bucket(frames, 513) == 640 && cb::coreml_pick_bucket(frames, 641) == -1,
            "bucket selection");
}

void check_windows() {
    const int window = 128, context = cb::HIFT_CONTEXT_FRAMES;
    require(cb::coreml_hift_windows(10, 2 * context, context).empty(), "a window without room for context");
    for (int frames = 1; frames <= 700; ++frames) {
        const auto plan = cb::coreml_hift_windows(frames, window, context);
        int covered = 0;
        for (const auto & w : plan) {
            require(w.keep_begin == covered, "kept spans tile the utterance");
            require(w.keep_end > w.keep_begin && w.start <= w.keep_begin, "non-empty kept span inside its window");
            require(w.keep_end - w.start <= window, "kept span inside the window");
            require(w.start == 0 || w.keep_begin - w.start >= context, "left context");
            require(w.keep_end == frames || w.start + window - w.keep_end >= context, "right context");
            covered = w.keep_end;
        }
        require(covered == frames, "every frame kept once");
        require(frames > window || plan.size() == 1, "one call when the utterance fits");
    }
}

void check_warm_estimates() {
    require(cb::coreml_expected_speech_tokens(0) == 0 && cb::coreml_expected_speech_tokens(100) == 115,
            "speech-token estimate");
    // 60 tokens streamed in 25s, first chunk 10: cumulative windows, or
    // capped by 25 tokens of left context.
    require(cb::coreml_stream_windows(60, 10, 25, 0) == std::vector<int>{10, 35, 60}, "cumulative windows");
    require(cb::coreml_stream_windows(60, 10, 25, 25) == std::vector<int>{10, 35, 50}, "left-context windows");
    require(cb::coreml_stream_windows(0, 10, 25, 0).empty() && cb::coreml_stream_windows(5, 0, 0, 0).empty(),
            "no windows");
}

void check_fit_and_detection() {
    env_guard disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    temp_dir dir("fit");
    tts_cpp::chatterbox::FitOptions options;
    options.t3_gguf_path = (dir.path / "chatterbox-t3-turbo-q8_0.gguf").string();
    options.s3gen_gguf_path = (dir.path / "chatterbox-s3gen-q8_0.gguf").string();
    options.backend = "cpu";
    const auto plain = tts_cpp::chatterbox::fit_params(options);
    require(plain.reason != "coreml-memory-estimate-unavailable", "no sidecar, no Core ML refusal");
    require(!cb::coreml_sidecars_enabled(options.s3gen_gguf_path, cb::resolve_coreml_policy()), "nothing to detect");

    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        const std::string sidecar = cb::coreml_stage_path(options.s3gen_gguf_path, suffix);
        fs::create_directory(sidecar);
        const auto result = tts_cpp::chatterbox::fit_params(options);
#ifdef TTS_CPP_USE_COREML
        require(cb::coreml_sidecars_enabled(options.s3gen_gguf_path, cb::resolve_coreml_policy()), "sidecar detected");
        require(result.status == tts_cpp::FitStatus::Error && !result.fits &&
                    result.reason == "coreml-memory-estimate-unavailable" && !result.report.empty(),
                std::string("fit refuses with an enabled ") + suffix + " sidecar");
#else
        require(result.reason != "coreml-memory-estimate-unavailable", "no Core ML refusal without Core ML");
#endif
        setenv(cb::COREML_DISABLE_ENV, "1", 1);
        require(tts_cpp::chatterbox::fit_params(options).reason != "coreml-memory-estimate-unavailable",
                "CHATTERBOX_COREML_DISABLE restores the native projection");
        unsetenv(cb::COREML_DISABLE_ENV);
        fs::remove(sidecar);
    }
}

// ---------------------------------------------------------------- fixture --

struct run_result {
    std::vector<float> wav;
    std::vector<float> mu;
    std::vector<float> mel;
    std::string backend;
};

std::vector<float> load_npy(const std::string & path) {
    npy_array a = npy_load(path);
    const float * data = reinterpret_cast<const float *>(a.data.data());
    return std::vector<float>(data, data + a.n_elements());
}

run_result synthesize(const std::string & gguf, const std::vector<int32_t> & tokens, const fs::path & scratch,
                      bool disabled, int strict) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.n_gpu_layers = 99;
    opts.seed = 42;
    opts.coreml_disabled = disabled;
    opts.coreml_strict = strict;
    run_result r;
    opts.pcm_out = &r.wav;
    opts.s3gen_backend_out = &r.backend;
    opts.dump_mel_path = (scratch / "dump.npy").string();
    if (s3gen_synthesize_to_wav(tokens, opts) != 0) throw std::runtime_error("s3gen_synthesize_to_wav failed");
    r.mu = load_npy((scratch / "dump_mu.npy").string());
    r.mel = load_npy((scratch / "dump.npy").string());
    return r;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size() && !a.empty(), "compared tensors differ in size");
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / (std::sqrt(aa * bb) + 1e-30);
}

double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

// Directory holding a link to the GGUF and links to the named sidecars only.
fs::path stage_dir(const fs::path & root, const std::string & gguf, const std::vector<const char *> & suffixes,
                   const std::string & tag) {
    const fs::path dir = root / tag;
    fs::create_directories(dir);
    const fs::path link = dir / fs::path(gguf).filename();
    fs::create_symlink(fs::absolute(gguf), link);
    for (const char * suffix : suffixes) {
        const fs::path sidecar = cb::coreml_stage_path(gguf, suffix);
        fs::create_directory_symlink(fs::absolute(sidecar), cb::coreml_stage_path(link.string(), suffix));
    }
    return dir;
}

std::vector<int32_t> fixture_tokens(int count) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> token(0, 6560);
    std::vector<int32_t> tokens((size_t) count);
    for (auto & t : tokens) t = token(rng);
    return tokens;
}

int stages_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_S3GEN");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_S3GEN is not set");
        return SKIP;
    }
    const std::string gguf = env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(cb::coreml_stage_path(gguf, suffix))) {
            std::printf("SKIP: %s is missing\n", cb::coreml_stage_path(gguf, suffix).c_str());
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    temp_dir root("stages");
    const std::vector<int32_t> tokens = fixture_tokens(90);

    const fs::path base = stage_dir(root.path, gguf, {}, "ggml");
    const std::string base_gguf = (base / fs::path(gguf).filename()).string();
    const run_result ref = synthesize(base_gguf, tokens, base, true, -1);
    require(ref.backend == cb::GGML_STAGE_BACKEND, "ggml reference reports ggml");

    auto stage = [&](const char * suffix, const char * name) {
        const fs::path dir = stage_dir(root.path, gguf, {suffix}, name);
        return synthesize((dir / fs::path(gguf).filename()).string(), tokens, dir, false, 1);
    };
    const run_result enc = stage(cb::COREML_ENCODER_SUFFIX, "encoder");
    const double enc_cos = cosine(enc.mu, ref.mu);
    std::printf("[encoder] backend=%s mu cosine=%.6f max_abs=%.3e\n", enc.backend.c_str(), enc_cos,
                max_abs(enc.mu, ref.mu));
    require(enc.backend == cb::MIXED_STAGE_BACKEND && enc_cos > 0.9999, "encoder parity");

    const run_result cfm = stage(cb::COREML_CFM_SUFFIX, "cfm");
    const double cfm_mu = cosine(cfm.mu, ref.mu), cfm_cos = cosine(cfm.mel, ref.mel);
    std::printf("[cfm] mel cosine=%.6f max_abs=%.3e\n", cfm_cos, max_abs(cfm.mel, ref.mel));
    require(cfm_mu > 0.999999 && cfm_cos > 0.999, "CFM parity");

    const run_result hift = stage(cb::COREML_HIFT_SUFFIX, "hift");
    const double hift_cos = cosine(hift.wav, ref.wav);
    std::printf("[hift] waveform cosine=%.6f max_abs=%.3e\n", hift_cos, max_abs(hift.wav, ref.wav));
    require(cosine(hift.mel, ref.mel) > 0.999999 && hift_cos > 0.995, "HiFT parity");

    const fs::path all_dir = stage_dir(root.path, gguf,
                                       {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}, "all");
    const std::string all_gguf = (all_dir / fs::path(gguf).filename()).string();
    const run_result all = synthesize(all_gguf, tokens, all_dir, false, 1);
    const double all_cos = cosine(all.mel, ref.mel);
    std::printf("[all] backend=%s mel cosine=%.6f waveform cosine=%.6f\n", all.backend.c_str(), all_cos,
                cosine(all.wav, ref.wav));
    require(all.backend == "coreml-ane" && all_cos > 0.999 && all.wav.size() == ref.wav.size(), "all stages");

    // Past the widest encoder/CFM function those stages stay on ggml (HiFT
    // windows any length); strict turns that fallback into an error.
    const std::vector<int32_t> long_tokens = fixture_tokens(1200);
    const run_result past = synthesize(all_gguf, long_tokens, all_dir, false, 0);
    std::printf("[routing] %zu tokens: backend=%s\n", long_tokens.size(), past.backend.c_str());
    require(past.backend == cb::MIXED_STAGE_BACKEND, "fallback past the widest function");
    bool threw = false;
    try {
        synthesize(all_gguf, long_tokens, all_dir, false, 1);
    } catch (const std::runtime_error & e) {
        threw = std::string(e.what()).find("cannot serve") != std::string::npos;
    }
    require(threw, "strict refuses an input past the widest function");
    s3gen_unload();
    return 0;
#endif
}

// ---------------------------------------------------------- shape fixture --

// The features each stage packs, as prepare_chatterbox_coreml_shape_fixture.py
// writes them: good sidecars declare exactly these (encoder tokens_8, CFM
// frames_16, HiFT window 32); `<stage>-<feature>` declares one of them with a
// doubled leading dimension.
const std::vector<std::pair<std::string, std::vector<std::string>>> SHAPE_FEATURES = {
    {"encoder", {"embedding", "valid", "mu"}},
    {"cfm", {"x", "mu", "cond", "spks", "t_emb", "valid", "velocity"}},
    {"hift", {"mel", "source", "valid", "spectrum"}},
};

// One call of `stage` on inputs narrower than its function; true when served.
bool serve(cb::s3gen_coreml & sidecars, const std::string & stage) {
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    auto random = [&](size_t n) {
        std::vector<float> v(n);
        for (float & x : v) x = noise(rng);
        return v;
    };
    if (stage == "encoder") {
        constexpr int tokens = 6;
        std::vector<float> mu;
        const bool ok = sidecars.encode(random((size_t) tokens * 512), tokens, 2, mu);
        require(!ok || mu.size() == (size_t) tokens * 2 * 80, "encoder output size");
        return ok;
    }
    if (stage == "cfm") {
        constexpr int frames = 12;
        const auto x = random(80 * frames), mu = random(80 * frames), cond = random(80 * frames);
        const auto spks = random(80), t_emb = random(1024);
        std::vector<std::vector<float>> out;
        // Like the pipeline: the CFG pair, of which a batch-1 sidecar uses the first.
        const bool ok = sidecars.velocity(1, frames, {x.data(), x.data()}, {mu.data(), mu.data()},
                                          {cond.data(), cond.data()}, {spks.data(), spks.data()},
                                          {t_emb.data(), t_emb.data()}, out);
        require(!ok || (out.size() == 1 && out[0].size() == (size_t) 80 * frames), "CFM output size");
        return ok;
    }
    constexpr int frames = 40;  // two HiFT windows
    const size_t stft = (size_t) frames * cb::HIFT_STFT_PER_FRAME + 1;
    std::vector<float> spectrum;
    const bool ok = sidecars.vocode(random((size_t) 80 * frames), frames, random(cb::HIFT_STFT_BINS * stft), spectrum);
    require(!ok || spectrum.size() == cb::HIFT_STFT_BINS * stft, "HiFT output size");
    return ok;
}

int shape_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_SHAPES");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_SHAPES is not set");
        return SKIP;
    }
    const fs::path dir = env;
    {
        cb::s3gen_coreml good((dir / "good-s3gen.gguf").string(), "cpu_only");
        for (const auto & [stage, features] : SHAPE_FEATURES) {
            require(serve(good, stage), "the matching " + stage + " sidecar serves");
        }
    }
    // The shared runner copies each feature by its declared shape, so every
    // one must be rejected before predicting, not only the ones the old
    // checks covered.
    for (const auto & [stage, features] : SHAPE_FEATURES) {
        for (const std::string & feature : features) {
            const std::string stem = stage + "-" + feature + "-s3gen";
            require(fs::exists(dir / (stem + "-" + stage + ".mlmodelc")), "fixture has " + stem);
            cb::s3gen_coreml sidecars((dir / (stem + ".gguf")).string(), "cpu_only");
            std::string error;
            try {
                serve(sidecars, stage);
            } catch (const std::runtime_error & e) {
                error = e.what();
            }
            std::printf("[%s %s] %s\n", stage.c_str(), feature.c_str(), error.c_str());
            require(error.find("declares " + feature + " ") != std::string::npos,
                    stage + " rejects a sidecar declaring a larger " + feature);
        }
    }
    return 0;
#endif
}

// --------------------------------------------------------- engine fixture --

// Three segments at 60 bytes; with the CFM and HiFT sidecars present the
// Engine renders each segment's S3Gen on a worker while T3 decodes the next.
const char * OVERLAP_TEXT = "The quick brown fox jumps over the lazy dog. "
                            "Core ML renders this sentence while the next one decodes. "
                            "Every segment must come back in order.";
constexpr int OVERLAP_SENTENCE_CHARS = 60;

EngineOptions engine_options(const std::string & t3, const std::string & s3gen) {
    EngineOptions opts;
    opts.t3_gguf_path = t3;
    opts.s3gen_gguf_path = s3gen;
    opts.n_gpu_layers = 99;
    opts.seed = 42;
    opts.max_sentence_chars = OVERLAP_SENTENCE_CHARS;
    return opts;
}

bool same_pcm(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() && max_abs(a, b) == 0.0;
}

int engine_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * t3_env = std::getenv("CHATTERBOX_COREML_TEST_T3");
    const char * s3gen_env = std::getenv("CHATTERBOX_COREML_TEST_S3GEN");
    if (!t3_env || !*t3_env || !s3gen_env || !*s3gen_env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_T3 / CHATTERBOX_COREML_TEST_S3GEN are not set");
        return SKIP;
    }
    const std::string t3 = t3_env, s3gen = s3gen_env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(cb::coreml_stage_path(s3gen, suffix))) {
            std::printf("SKIP: %s is missing\n", cb::coreml_stage_path(s3gen, suffix).c_str());
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    temp_dir root("engine");
    const auto segments = tts_cpp::chatterbox::text_preprocess::split_text_for_tts(OVERLAP_TEXT,
                                                                                   OVERLAP_SENTENCE_CHARS);
    require(segments.size() == 3, "the overlap text splits into three segments");
    require(cb::coreml_offloads_s3gen(s3gen, cb::resolve_coreml_policy("", std::nullopt)),
            "the sidecars enable the sentence overlap");

    // Overlapped synthesis equals the segments synthesized one call at a time
    // (one segment never overlaps) and crossfaded in order, sample for sample:
    // T3 and the CFM noise are seeded per segment, so only order can differ.
    Engine engine(engine_options(t3, s3gen));
    const SynthesisResult overlapped = engine.synthesize(OVERLAP_TEXT);
    // Timed again once every function is compiled and loaded, so the cancel
    // points below fall inside a warm request.
    const auto t0 = std::chrono::steady_clock::now();
    require(same_pcm(engine.synthesize(OVERLAP_TEXT).pcm, overlapped.pcm), "overlapped synthesis is repeatable");
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    std::vector<float> sequential;
    std::set<std::string> backends;
    int tokens = 0;
    for (const std::string & segment : segments) {
        const SynthesisResult one = engine.synthesize(segment);
        tts_cpp::chatterbox::text_preprocess::append_pcm_crossfade(sequential, one.pcm, one.sample_rate,
                                                                   engine.options().crossfade_ms);
        backends.insert(one.s3gen_backend);
        tokens += one.t3_tokens;
    }
    std::printf("[overlap] %zu samples in %.2f s, backend=%s, max |overlapped - sequential| = %.3e\n",
                overlapped.pcm.size(), std::chrono::duration<double>(elapsed).count(),
                overlapped.s3gen_backend.c_str(),
                overlapped.pcm.size() == sequential.size() ? max_abs(overlapped.pcm, sequential) : -1.0);
    require(backends == std::set<std::string>{"coreml-ane"} && overlapped.s3gen_backend == "coreml-ane",
            "every segment and the aggregate report the Core ML backend");
    require(overlapped.t3_tokens == tokens, "overlapped T3 token count matches the segments'");
    require(same_pcm(overlapped.pcm, sequential), "overlapped PCM equals the in-order sequential PCM");

    // Engine::cancel() at points across the request: each run either finishes
    // or throws a cancellation (from T3, the S3Gen worker or the segment
    // boundary), never anything else, and the next request is unaffected.
    int cancelled = 0;
    for (double fraction : {0.2, 0.45, 0.7}) {
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed * fraction));
            engine.cancel();
        });
        std::string error;
        try {
            engine.synthesize(OVERLAP_TEXT);
        } catch (const std::runtime_error & e) {
            error = e.what();
        }
        canceller.join();
        std::printf("[cancel at %.0f%%] %s\n", fraction * 100, error.empty() ? "finished" : error.c_str());
        require(error.empty() || error.find("cancelled") != std::string::npos, "a cancel surfaces as a cancellation");
        cancelled += !error.empty();
    }
    require(cancelled > 0, "at least one cancel lands mid-request");
    require(same_pcm(engine.synthesize(OVERLAP_TEXT).pcm, overlapped.pcm), "a cancelled engine synthesizes again");

    // A CFM sidecar that cannot serve (here, a package without frames_*
    // functions) still enables the overlap, so its failures surface on the
    // worker: strict mode must rethrow the worker's error from synthesize(),
    // and without strict the segments fall back to ggml CFM and aggregate
    // as mixed.
    const fs::path broken = stage_dir(root.path, s3gen, {cb::COREML_ENCODER_SUFFIX, cb::COREML_HIFT_SUFFIX}, "broken");
    const std::string broken_gguf = (broken / fs::path(s3gen).filename()).string();
    fs::create_directory_symlink(fs::absolute(cb::coreml_stage_path(s3gen, cb::COREML_HIFT_SUFFIX)),
                                 cb::coreml_stage_path(broken_gguf, cb::COREML_CFM_SUFFIX));
    EngineOptions strict_opts = engine_options(t3, broken_gguf);
    strict_opts.coreml_strict = true;
    std::string error;
    try {
        Engine(strict_opts).synthesize(OVERLAP_TEXT);
    } catch (const std::runtime_error & e) {
        error = e.what();
    }
    std::printf("[worker error] %s\n", error.c_str());
    require(error.find("cannot serve") != std::string::npos, "the worker's strict-mode error reaches the caller");
    EngineOptions lenient_opts = engine_options(t3, broken_gguf);
    lenient_opts.coreml_strict = false;
    const SynthesisResult mixed = Engine(lenient_opts).synthesize(OVERLAP_TEXT);
    std::printf("[fallback] %zu samples, backend=%s\n", mixed.pcm.size(), mixed.s3gen_backend.c_str());
    require(!mixed.pcm.empty() && mixed.s3gen_backend == cb::MIXED_STAGE_BACKEND,
            "a CFM fallback on the worker aggregates as mixed");
    s3gen_unload();
    return 0;
#endif
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--stages-fixture") return stages_fixture();
        if (argc > 1 && std::string(argv[1]) == "--shape-fixture") return shape_fixture();
        if (argc > 1 && std::string(argv[1]) == "--engine-fixture") return engine_fixture();
        check_policy();
        check_naming();
        check_buckets();
        check_windows();
        check_warm_estimates();
        check_fit_and_detection();
        std::puts("test-chatterbox-coreml: ok");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-chatterbox-coreml: FAIL: %s\n", e.what());
        return 1;
    }
}
