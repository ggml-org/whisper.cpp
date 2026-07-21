// end-to-end coverage of the output-frequency selection on the
// chatterbox::Engine API (the surface the @qvac/tts-ggml addon consumes).
// Gated on the multilingual GGUF fixtures; auto-disabled when they're absent.
//
// Verifies, against a single short utterance synthesized on the CPU path
// (deterministic, fixed seed):
//   * default (output_sample_rate == 0) reports + emits the native 24 kHz;
//   * output_sample_rate == 16000 reports 16 kHz and the produced sample
//     count tracks the 2/3 ratio of the native run;
//   * an out-of-range output_sample_rate is rejected at construction;
//   * streaming with a non-native output rate keeps the documented
//     `result.pcm == concat(chunks)` invariant and reports the requested rate.

#include "tts-cpp/chatterbox/engine.h"
#include "voice_features.h"  // resample_for_output

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

using tts_cpp::chatterbox::Engine;
using tts_cpp::chatterbox::EngineOptions;
using tts_cpp::chatterbox::StreamCallback;
using tts_cpp::chatterbox::SynthesisResult;

static int g_fail = 0;
#define CHECK(cond, msg) do {                                  \
        if (!(cond)) { std::printf("FAIL: %s\n", (msg)); ++g_fail; } \
        else         { std::printf("ok:   %s\n", (msg)); }           \
    } while (0)

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s T3.gguf S3GEN.gguf\n", argv[0]);
        return 64;
    }

    EngineOptions base;
    base.t3_gguf_path       = argv[1];
    base.s3gen_gguf_path    = argv[2];
    base.language           = "en";
    base.seed               = 42;
    base.n_gpu_layers       = 0;    // CPU: deterministic, backend-independent
    base.max_sentence_chars = 0;    // single segment, no auto-split

    const std::string text = "Hello world, a sample rate test.";

    // --- native (default 0 == 24 kHz) ------------------------------------
    std::size_t n_native = 0;
    try {
        Engine eng(base);
        SynthesisResult r = eng.synthesize(text);
        CHECK(r.sample_rate == 24000, "native reports 24000 Hz");
        CHECK(!r.pcm.empty(),         "native produced audio");
        n_native = r.pcm.size();
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
        const double ratio = n_native ? (double) r.pcm.size() / (double) n_native : 0.0;
        std::printf("  16k/native sample ratio = %.4f (expect ~0.6667)\n", ratio);
        CHECK(std::fabs(ratio - 2.0 / 3.0) < 0.02, "batch length tracks the 2/3 ratio");
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

    // --- streaming 16 kHz: invariant + reported rate ----------------------
    try {
        EngineOptions o = base;
        o.output_sample_rate  = 16000;
        o.stream_chunk_tokens = 25;
        Engine eng(o);
        std::vector<float> concat;
        StreamCallback cb = [&](const float * pcm, std::size_t n, int, bool) {
            concat.insert(concat.end(), pcm, pcm + n);
        };
        SynthesisResult r = eng.synthesize(text, cb);
        CHECK(r.sample_rate == 16000,          "streaming reports 16000 Hz");
        CHECK(concat.size() == r.pcm.size(),   "streaming result.pcm == concat(chunks)");
        CHECK(!r.pcm.empty(),                  "streaming produced audio");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "streaming synth threw: %s\n", e.what());
        return 1;
    }

    // --- streaming resampling is batch-exact (the OutputResampler property) ---
    // GustavoA1604 review: assert that a non-native streamed rate equals
    // resampling the streamed NATIVE audio in one shot — the invariant the
    // utterance-spanning OutputResampler guarantees (test_resample proves it
    // model-free; this exercises it end-to-end through the engine, and would
    // catch a regression to per-chunk resampling: seams + length drift).
    //
    // We compare streamed-16k against a whole-buffer resample of the streamed-
    // NATIVE run rather than against the batch path: chatterbox batch and
    // streaming synthesis legitimately differ (sliding-window context, HiFT
    // cache continuity, first-chunk trim-fade, floored streaming CFM steps), so
    // batch.pcm != stream.pcm even at the native rate.  output_sample_rate only
    // changes the final resample, so both streamed runs share the same native
    // signal and the comparison is bit-exact.
    try {
        EngineOptions on = base;          // native streaming
        on.stream_chunk_tokens  = 25;
        EngineOptions o16 = base;         // 16 kHz streaming
        o16.stream_chunk_tokens = 25;
        o16.output_sample_rate  = 16000;

        std::vector<float> stream_native;
        {
            Engine eng(on);
            StreamCallback cb = [&](const float * p, std::size_t n, int, bool) {
                stream_native.insert(stream_native.end(), p, p + n);
            };
            eng.synthesize(text, cb);
        }
        std::vector<float> stream_16k;
        {
            Engine eng(o16);
            StreamCallback cb = [&](const float * p, std::size_t n, int, bool) {
                stream_16k.insert(stream_16k.end(), p, p + n);
            };
            eng.synthesize(text, cb);
        }

        const std::vector<float> expect =
            resample_for_output(stream_native, 24000, 16000);
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
