// Engine-level streaming-callback contract test for the per-sentence
// segmentation path (Fix #2).  Exercises tts_cpp::chatterbox::Engine end to
// end (no in-repo coverage existed before) and pins the invariants the
// segmentation refactor must hold:
//
//   * chunk_index is a single monotonic 0..n-1 counter for the whole
//     synthesize() call -- it does NOT reset per sentence segment;
//   * is_last fires exactly once, on the very final chunk of the final
//     segment;
//   * result.pcm == concatenation of the callback chunk buffers
//     (the documented invariant the streaming path must preserve);
//   * result.audio_samples == result.pcm.size(); t3_tokens accumulates.
//
// Tested for both the single-segment path (auto-split off) and the
// multi-segment path (auto-split on over a multi-sentence paragraph).
//
// Gated on the chatterbox T3 + S3Gen GGUFs (REQUIRES in CMakeLists), so it is
// DISABLED in CI when those fixtures are absent.
//
// Usage: test-chatterbox-engine-stream T3.gguf S3GEN.gguf [language]

#include "tts-cpp/chatterbox/engine.h"

#include <cstdio>
#include <string>
#include <vector>

using tts_cpp::chatterbox::Engine;
using tts_cpp::chatterbox::EngineOptions;
using tts_cpp::chatterbox::SynthesisResult;

static int g_failures = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            ++g_failures;                                                 \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                                 \
    } while (0)

// Synthesize `text` with streaming and assert the callback contract.
// `expect_multi` requires at least 2 chunks (a stand-in for "multiple
// segments emitted a stream of chunks").
static void run_case(Engine & eng, const std::string & text,
                     const char * label, bool expect_multi) {
    std::vector<int> indices;
    std::size_t total_samples = 0;
    int is_last_count = 0;
    int is_last_index = -1;

    SynthesisResult res = eng.synthesize(
        text, [&](const float * /*pcm*/, std::size_t n, int idx, bool is_last) {
            indices.push_back(idx);
            total_samples += n;
            if (is_last) { ++is_last_count; is_last_index = idx; }
        });

    std::fprintf(stderr, "[%s] %zu chunks, %zu samples, t3_tokens=%d\n",
                 label, indices.size(), (size_t)res.audio_samples, res.t3_tokens);

    CHECK(!indices.empty(), "at least one chunk emitted");
    if (expect_multi) CHECK(indices.size() >= 2, "multiple chunks emitted");

    bool contiguous = true;
    for (std::size_t k = 0; k < indices.size(); ++k)
        if (indices[k] != (int)k) contiguous = false;
    CHECK(contiguous, "chunk_index is contiguous 0..n-1 (monotonic, no per-segment reset)");

    CHECK(is_last_count == 1, "is_last fires exactly once");
    CHECK(is_last_index == (int)indices.size() - 1, "is_last only on the final chunk");

    CHECK(res.pcm.size() == total_samples, "result.pcm == concat(callback chunks)");
    CHECK(res.audio_samples == (int)res.pcm.size(), "audio_samples == result.pcm.size()");
    CHECK(res.t3_tokens > 0, "t3_tokens accumulated > 0");
    CHECK(res.t3_ms > 0.0 && res.s3gen_ms > 0.0, "t3_ms / s3gen_ms accumulated");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s T3.gguf S3GEN.gguf [language]\n", argv[0]);
        return 2;
    }

    EngineOptions base;
    base.t3_gguf_path     = argv[1];
    base.s3gen_gguf_path  = argv[2];
    if (argc >= 4) base.language = argv[3];
    base.seed             = 0;
    base.n_threads        = 4;
    base.stream_chunk_tokens = 25;

    // Single-segment path: auto-split OFF (default).  One run_t3, one stream.
    {
        EngineOptions opts = base;
        opts.max_sentence_chars = 0;
        Engine eng(opts);
        run_case(eng, "The quick brown fox jumps over the lazy dog.",
                 "single-segment", /*expect_multi=*/false);
    }

    // Multi-segment path: auto-split ON over a 3-sentence paragraph.  Several
    // segments, but ONE utterance: chunk_index stays monotonic across the
    // segment boundaries and is_last fires only at the very end.
    {
        EngineOptions opts = base;
        opts.max_sentence_chars = 30;
        Engine eng(opts);
        run_case(eng,
                 "Hello there friend. The quick brown fox jumps over the lazy "
                 "dog today. Good night and good luck everyone.",
                 "multi-segment", /*expect_multi=*/true);
    }

    if (g_failures == 0) {
        std::printf("test_chatterbox_engine_stream: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_chatterbox_engine_stream: %d failure(s)\n", g_failures);
    return 1;
}
