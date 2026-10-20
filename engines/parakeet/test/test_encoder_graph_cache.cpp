// Fixture test for the ggml encoder graph cache limit and the windowed
// encoder's scoped use of it.
//
// Usage:
//   test-encoder-graph-cache --model <parakeet.gguf> --wav <input.wav>
//
// It asserts that:
//   1. distinct input shapes fill the cache up to its default of three graphs;
//   2. lowering the limit evicts the oldest graphs at once and reports the
//      previous limit, so a caller can restore it;
//   3. a windowed encoder pass holds a single graph and restores the previous
//      limit when it returns, whether it ran to completion or was cancelled
//      before the first window.
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet/engine.h"

#include "long_form.h"
#include "long_form_encoder.h"
#include "mel_preprocess.h"
#include "parakeet_ctc.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace parakeet;

namespace {

constexpr size_t kDefaultCacheLimit = 3;
constexpr int    kShapeStep         = 100;   // mel frames between the probed shapes
constexpr int    kTileCount         = 4;     // copies of the clip for the windowed pass
constexpr int    kWindowFrames      = 256;   // encoder frames per long-form window
constexpr int    kContextFrames     = 32;

int g_failures = 0;

void expect(bool cond, const std::string & what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// Reads the limit without changing it: set returns the previous value.
size_t current_cache_limit(ParakeetCtcModel & model) {
    const size_t limit = set_encoder_graph_cache_limit(model, kDefaultCacheLimit);
    set_encoder_graph_cache_limit(model, limit);
    return limit;
}

int encode_frames(ParakeetCtcModel & model, const std::vector<float> & mel, int n_frames) {
    EncoderOutputs out;
    return run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, out,
                       /*max_layers=*/-1, /*capture_intermediates=*/false);
}

void fill_cache_with_shapes(ParakeetCtcModel & model, const std::vector<float> & mel, int n_frames) {
    for (size_t i = 0; i < kDefaultCacheLimit; ++i) {
        const int frames = n_frames - (int) i * kShapeStep;
        expect(encode_frames(model, mel, frames) == 0,
               "run_encoder failed at " + std::to_string(frames) + " frames");
    }
}

std::vector<float> tile_mel(const std::vector<float> & mel, int copies) {
    std::vector<float> tiled;
    tiled.reserve(mel.size() * (size_t) copies);
    for (int i = 0; i < copies; ++i) tiled.insert(tiled.end(), mel.begin(), mel.end());
    return tiled;
}

void check_fill_and_evict(ParakeetCtcModel & model, const std::vector<float> & mel, int n_frames) {
    fill_cache_with_shapes(model, mel, n_frames);
    expect(encoder_graph_cache_size(model) == kDefaultCacheLimit,
           "three shapes should fill the three-graph cache");

    expect(set_encoder_graph_cache_limit(model, 1) == kDefaultCacheLimit,
           "lowering the limit must report the previous limit");
    expect(encoder_graph_cache_size(model) == 1, "lowering the limit to one must evict to one graph");
    expect(set_encoder_graph_cache_limit(model, kDefaultCacheLimit) == 1,
           "restoring the limit must report the lowered limit");
    expect(encoder_graph_cache_size(model) == 1, "raising the limit keeps the surviving graph");
    expect(set_encoder_graph_cache_limit(model, 0) == kDefaultCacheLimit &&
           current_cache_limit(model) == 1,
           "a limit below one is clamped to one");
    set_encoder_graph_cache_limit(model, kDefaultCacheLimit);
}

void check_windowed_pass(ParakeetCtcModel & model, const std::vector<float> & mel, int n_frames) {
    const std::vector<float> tiled = tile_mel(mel, kTileCount);
    const int n_tiled = n_frames * kTileCount;

    EngineOptions opts;
    opts.long_form_window_frames  = kWindowFrames;
    opts.long_form_context_frames = kContextFrames;
    const LongFormPlan plan = resolve_long_form_plan(model, opts, n_tiled);
    expect(plan.enabled, "the tiled clip must need long-form windows");

    fill_cache_with_shapes(model, mel, n_frames);
    expect(encoder_graph_cache_size(model) == kDefaultCacheLimit, "cache full before the windowed pass");

    std::atomic<bool> cancel{false};
    EncoderOutputs out;
    WindowedEncoderStats stats;
    expect(run_encoder_windowed(model, tiled.data(), n_tiled, model.mel_cfg.n_mels, plan,
                                cancel, out, stats) == 0,
           "windowed pass failed");
    expect(out.n_enc_frames > 0, "windowed pass produced no frames");
    expect(encoder_graph_cache_size(model) == 1,
           "a windowed pass over equal windows must leave exactly one graph");
    expect(current_cache_limit(model) == kDefaultCacheLimit,
           "the windowed pass must restore the previous cache limit");

    fill_cache_with_shapes(model, mel, n_frames);
    std::atomic<bool> cancelled{true};
    EncoderOutputs none;
    WindowedEncoderStats none_stats;
    expect(run_encoder_windowed(model, tiled.data(), n_tiled, model.mel_cfg.n_mels, plan,
                                cancelled, none, none_stats) == 0,
           "a cancelled windowed pass must still return success");
    expect(none.n_enc_frames == 0, "a cancelled windowed pass commits no frames");
    expect(current_cache_limit(model) == kDefaultCacheLimit,
           "an early return must restore the previous cache limit");
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path, wav_path;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--model") == 0) model_path = argv[i + 1];
        else if (std::strcmp(argv[i], "--wav") == 0) wav_path = argv[i + 1];
    }
    if (model_path.empty() || wav_path.empty()) {
        std::fprintf(stderr, "usage: test-encoder-graph-cache --model <gguf> --wav <wav>\n");
        return 2;
    }

    ParakeetCtcModel model;
    if (load_from_gguf(model_path, model, /*n_threads=*/0, /*n_gpu_layers=*/0, /*verbose=*/false) != 0) {
        std::fprintf(stderr, "FAIL: could not load %s\n", model_path.c_str());
        return 1;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (load_wav_mono_f32(wav_path, samples, sample_rate) != 0) {
        std::fprintf(stderr, "FAIL: could not load %s\n", wav_path.c_str());
        return 1;
    }
    std::vector<float> mel;
    int n_frames = 0;
    if (compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames) != 0 ||
        n_frames <= (int) kDefaultCacheLimit * kShapeStep) {
        std::fprintf(stderr, "FAIL: the clip is too short for the probed shapes\n");
        return 1;
    }

    check_fill_and_evict(model, mel, n_frames);
    check_windowed_pass(model, mel, n_frames);

    if (g_failures == 0) {
        std::printf("test-encoder-graph-cache: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-encoder-graph-cache: %d failure(s)\n", g_failures);
    return 1;
}
