// Engine-level streaming-callback contract test for Parler-TTS.  Pins:
//   * chunk_index contiguous 0..n-1, is_last exactly once on the last chunk;
//   * result.pcm == concat(callback chunks);
//   * streamed pcm == whole-utterance decode -- BIT-IDENTICAL on CPU (the DAC is
//     local/convolutional so a prefix decode reproduces the interior exactly),
//     within tolerance on GPU (mul_mat kernel dispatch varies with the sequence
//     length, so interior samples can differ in the last ULP).
//
// A fixed seed makes the AR codes identical between the streamed and batch
// runs, so the only difference under test is windowed-vs-whole DAC decode.  Both
// legs run: n_gpu_layers=0 forces CPU; n_gpu_layers>0 uses the GPU where present
// and falls back to CPU otherwise (assertion picks itself via backend_device()).
//
// Gated on the parler GGUF (REQUIRES in CMakeLists); DISABLED in CI when absent.
// Usage: test-parler-engine-stream MODEL.gguf

#include "tts-cpp/parler/engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

using tts_cpp::BackendDevice;
using tts_cpp::parler::Engine;
using tts_cpp::parler::EngineOptions;
using tts_cpp::parler::SynthesisResult;

static int g_failures = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            ++g_failures;                                                 \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                                 \
    } while (0)

static void run_case(const std::string & model, int n_gpu_layers) {
    const char * prompt =
        "The quick brown fox jumps over the lazy dog. She sells seashells by the "
        "seashore. Peter Piper picked a peck of pickled peppers.";
    const char * desc = "A calm female voice.";

    EngineOptions opts;
    opts.model_gguf_path           = model;
    opts.seed                      = 42;     // deterministic AR -> identical codes
    opts.max_frames                = 400;    // a few seconds -> several chunks
    opts.n_threads                 = 4;
    opts.n_gpu_layers              = n_gpu_layers;
    opts.stream_first_chunk_frames = 40;     // ~0.5 s
    opts.stream_chunk_frames       = 70;     // ~0.8 s

    Engine eng(opts);
    const bool gpu = eng.backend_device() == BackendDevice::GPU;

    // Streaming run: collect the chunks and assert the callback contract.
    std::vector<int> indices;
    std::size_t total = 0;
    int last_count = 0, last_index = -1;
    SynthesisResult sres = eng.synthesize(
        prompt, desc, [&](const float * /*pcm*/, std::size_t n, int idx, bool is_last) {
            indices.push_back(idx);
            total += n;
            if (is_last) { ++last_count; last_index = idx; }
        });

    std::fprintf(stderr, "[%s] %zu chunks, %zu samples\n",
                 gpu ? "GPU" : "CPU", indices.size(), total);

    CHECK(!indices.empty(), "at least one chunk emitted");
    CHECK(indices.size() >= 2, "multiple chunks emitted");
    bool contiguous = true;
    for (std::size_t k = 0; k < indices.size(); ++k)
        if (indices[k] != (int) k) contiguous = false;
    CHECK(contiguous, "chunk_index contiguous 0..n-1");
    CHECK(last_count == 1, "is_last fires exactly once");
    CHECK(last_index == (int) indices.size() - 1, "is_last only on the final chunk");
    CHECK(sres.pcm.size() == total, "result.pcm == concat(callback chunks)");

    // Batch reference: same engine, no callback -> whole-utterance decode of the
    // same codes (same seed).
    SynthesisResult bres = eng.synthesize(prompt, desc);
    CHECK(bres.pcm.size() == sres.pcm.size(), "streamed vs batch length match");
    if (bres.pcm.size() == sres.pcm.size()) {
        double max_abs = 0.0;
        for (std::size_t i = 0; i < bres.pcm.size(); ++i)
            max_abs = std::max(max_abs, (double) std::fabs(bres.pcm[i] - sres.pcm[i]));
        std::fprintf(stderr, "  max|stream-batch| = %.3g\n", max_abs);
        if (gpu) {
            CHECK(max_abs < 1e-3, "GPU: streamed pcm within tolerance of whole-utterance decode");
        } else {
            CHECK(max_abs == 0.0, "CPU: streamed pcm BIT-IDENTICAL to whole-utterance decode");
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    run_case(model, 0);    // CPU: bit-identical
    run_case(model, 99);   // GPU where present (else CPU fallback): tolerance

    if (g_failures == 0) {
        std::printf("test_parler_engine_stream: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_parler_engine_stream: %d failure(s)\n", g_failures);
    return 1;
}
