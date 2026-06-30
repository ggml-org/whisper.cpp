// QVAC-21483 — end-to-end coverage of output-frequency selection on the
// supertonic::Engine API.  Sibling of test_output_sample_rate.cpp (which covers
// chatterbox::Engine) so both engines stay symmetric as the feature evolves.
// Gated on the Supertonic GGUF fixture; auto-disabled when it's absent.
//
// Verifies, against a short utterance synthesized on the CPU path
// (deterministic, fixed seed):
//   * default (output_sample_rate == 0) reports the model's native rate + audio;
//   * output_sample_rate == 16000 reports 16 kHz and the produced sample count
//     tracks the 16000/native ratio;
//   * an out-of-range output_sample_rate is rejected at construction;
//   * streaming keeps the documented `result.pcm == concat(chunks)` invariant,
//     reports the requested rate, and actually splits into multiple chunks;
//   * streaming a non-native rate equals resampling the streamed NATIVE audio in
//     one shot — the utterance-spanning OutputResampler batch-exact property.
//     This is the assertion that catches a regression to per-chunk resampling
//     (seams + length drift); test_resample proves the same property model-free.

#include "tts-cpp/supertonic/engine.h"
#include "voice_features.h"  // resample_for_output (QVAC-21483)

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

using tts_cpp::supertonic::Engine;
using tts_cpp::supertonic::EngineOptions;
using tts_cpp::supertonic::StreamCallback;
using tts_cpp::supertonic::SynthesisResult;

static int g_fail = 0;
#define CHECK(cond, msg) do {                                  \
        if (!(cond)) { std::printf("FAIL: %s\n", (msg)); ++g_fail; } \
        else         { std::printf("ok:   %s\n", (msg)); }           \
    } while (0)

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s SUPERTONIC.gguf\n", argv[0]);
        return 64;
    }

    EngineOptions base;
    base.model_gguf_path = argv[1];
    base.language        = "en";
    base.seed            = 42;
    base.n_gpu_layers    = 0;    // CPU: deterministic, backend-independent

    // Long enough (and multi-sentence) that the streaming chunker below splits
    // it into several chunks rather than one.
    const std::string text =
        "Hello world. This is the first sentence of a sample rate test. "
        "And here is a second, somewhat longer sentence for streaming.";

    // --- native (default 0 == model rate) --------------------------------
    int         native_sr = 0;
    std::size_t n_native   = 0;
    try {
        Engine eng(base);
        SynthesisResult r = eng.synthesize(text);
        CHECK(r.sample_rate > 0, "native reports a positive sample rate");
        CHECK(!r.pcm.empty(),    "native produced audio");
        native_sr = r.sample_rate;
        n_native  = r.pcm.size();
        std::printf("  native rate = %d Hz, %zu samples\n", native_sr, n_native);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "native synth threw: %s\n", e.what());
        return 1;
    }

    // --- 16 kHz batch -----------------------------------------------------
    try {
        EngineOptions o = base;
        o.output_sample_rate = 16000;
        Engine eng(o);
        SynthesisResult r = eng.synthesize(text);
        CHECK(r.sample_rate == 16000, "batch reports 16000 Hz");
        const double expect_ratio = native_sr ? 16000.0 / (double) native_sr : 0.0;
        const double ratio        = n_native ? (double) r.pcm.size() / (double) n_native : 0.0;
        std::printf("  16k/native sample ratio = %.4f (expect ~%.4f)\n", ratio, expect_ratio);
        CHECK(std::fabs(ratio - expect_ratio) < 0.02, "batch length tracks the 16000/native ratio");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "16k batch synth threw: %s\n", e.what());
        return 1;
    }

    // --- out-of-range rate rejected at construction -----------------------
    {
        EngineOptions o = base;
        o.output_sample_rate = 1234;   // below the 8000 floor
        bool threw = false;
        try { Engine eng(o); } catch (const std::exception &) { threw = true; }
        CHECK(threw, "out-of-range output_sample_rate rejected at construction");
    }

    // Small target + floor force several chunks so the inter-chunk seam path
    // is actually exercised (the per-chunk-resampling bug only shows with >1).
    EngineOptions stream_base = base;
    stream_base.stream_chunk_tokens     = 30;
    stream_base.stream_min_chunk_tokens = 12;

    // --- streaming native: reported rate + result.pcm == concat(chunks) ---
    std::vector<float> stream_native;
    int chunks_seen = 0;
    try {
        Engine eng(stream_base);
        StreamCallback cb = [&](const float * p, std::size_t n, int, bool) {
            stream_native.insert(stream_native.end(), p, p + n);
            ++chunks_seen;
        };
        SynthesisResult r = eng.synthesize(text, cb);
        CHECK(r.sample_rate == native_sr,           "streaming native reports the model rate");
        CHECK(stream_native.size() == r.pcm.size(), "streaming result.pcm == concat(chunks)");
        CHECK(!r.pcm.empty(),                       "streaming produced audio");
        std::printf("  streaming produced %d chunk(s)\n", chunks_seen);
        CHECK(chunks_seen >= 2, "streaming split into multiple chunks (seam path exercised)");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "streaming native synth threw: %s\n", e.what());
        return 1;
    }

    // --- streaming resampling is batch-exact (OutputResampler property) ---
    // streamed-16k must equal a whole-buffer resample of the streamed-NATIVE
    // run.  output_sample_rate only changes the final resample, so both runs
    // share the same native signal and the comparison is bit-exact.  Per-chunk
    // resampling (the bug fixed here) would inject seams + a length drift and
    // fail this check.
    try {
        EngineOptions o = stream_base;
        o.output_sample_rate = 16000;
        std::vector<float> stream_16k;
        Engine eng(o);
        StreamCallback cb = [&](const float * p, std::size_t n, int, bool) {
            stream_16k.insert(stream_16k.end(), p, p + n);
        };
        eng.synthesize(text, cb);

        const std::vector<float> expect =
            resample_for_output(stream_native, native_sr, 16000);
        bool exact = (stream_16k.size() == expect.size());
        for (std::size_t i = 0; exact && i < expect.size(); ++i) {
            if (stream_16k[i] != expect[i]) exact = false;
        }
        std::printf("  streamed native=%zu, streamed 16k=%zu, whole-buffer resample=%zu\n",
                    stream_native.size(), stream_16k.size(), expect.size());
        CHECK(exact,
              "streaming 16k == whole-buffer resample of streamed native (no per-chunk seams)");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "streaming-resample-invariant synth threw: %s\n", e.what());
        return 1;
    }

    std::printf("\n%s\n", g_fail ? "TEST FAILED" : "ALL CHECKS PASSED");
    return g_fail ? 1 : 0;
}
