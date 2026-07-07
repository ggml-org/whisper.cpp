// Equivalence test for the S3Gen dual-path dispatch after its migration
// onto src/sched_dispatch.{h,cpp}: the forced scheduler path must produce
// BIT-IDENTICAL audio to the direct single-backend path, and running the
// scheduler path must not poison any cached state for later direct-path
// calls (the HiFT graph cache rebuilds on every sched-routed call because
// ggml_backend_sched_alloc_graph rewrites node->src[] in place).
//
// Sequence (same model, same tokens, same seed each phase; the model and
// all synth caches stay loaded across phases — same-process reuse is the
// stronger poisoning check, and it mirrors test_t3_sched_equivalence.
// Per-phase s3gen_unload/reload cycles are deliberately avoided: they
// trip the PRE-EXISTING load-sensitive compute_time_mlp_cached segfault
// documented in docs/s3gen-time-mlp-cached-segfault.md, which is
// unrelated to the dispatch under test):
//   A  — direct path:      full s3gen_synthesize_to_wav
//   B  — TTS_CPP_FORCE_SCHED=1: same synth through the scheduler
//   A' — direct again:     must still match A
// PCM is compared byte-for-byte (memcmp); bit-exactness is the bar — do
// NOT relax this to a numeric tolerance.
//
// Uses the GGUF's built-in voice (empty ref_dir) and a fixed synthetic
// speech-token list, so the only fixture is the S3Gen GGUF itself.
//
// usage: test-s3gen-sched-equivalence MODEL.gguf [n_gpu_layers]

#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK_MSG(cond, ...) do {                                     \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); \
        std::fprintf(stderr, __VA_ARGS__);                            \
        std::fprintf(stderr, "\n");                                   \
    }                                                                 \
} while (0)

// Fixed, deterministic token list (valid S3 tokenizer ids, ~2 s of audio).
std::vector<int32_t> make_tokens() {
    std::vector<int32_t> toks;
    for (int i = 0; i < 50; ++i) {
        toks.push_back((i * 73 + 11) % 4200);
    }
    return toks;
}

bool run_phase(const std::string & gguf, int n_gpu_layers, std::vector<float> & pcm,
               const char * label) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.out_wav_path.clear();
    opts.pcm_out      = &pcm;
    opts.seed         = 42;
    opts.n_threads    = 2;
    opts.n_gpu_layers = n_gpu_layers;
    const int rc = s3gen_synthesize_to_wav(make_tokens(), opts);
    if (rc != 0) {
        std::fprintf(stderr, "phase %s: s3gen_synthesize_to_wav rc=%d\n", label, rc);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [n_gpu_layers]\n", argv[0]);
        return 2;
    }
    const std::string gguf  = argv[1];
    const int n_gpu_layers  = (argc > 2) ? std::atoi(argv[2]) : 0;

    std::vector<float> pcm_a, pcm_b, pcm_a2;

    unsetenv("TTS_CPP_FORCE_SCHED");
    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_a, "A (direct)"), "phase A failed");

    setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_b, "B (sched)"), "phase B failed");
    unsetenv("TTS_CPP_FORCE_SCHED");

    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_a2, "A' (direct)"), "phase A' failed");

    if (g_failures == 0) {
        // Non-degenerate: audible signal, no NaN.
        bool any_signal = false, any_nan = false;
        for (float v : pcm_a) {
            if (std::fabs(v) > 1e-4f) any_signal = true;
            if (std::isnan(v))        any_nan = true;
        }
        CHECK_MSG(!pcm_a.empty(), "phase A produced no PCM");
        CHECK_MSG(any_signal, "phase A PCM is all-zero/near-silent");
        CHECK_MSG(!any_nan, "phase A PCM contains NaN");

        CHECK_MSG(pcm_a.size() == pcm_b.size(),
                  "A vs B sample count %zu != %zu", pcm_a.size(), pcm_b.size());
        CHECK_MSG(pcm_a.size() == pcm_a2.size(),
                  "A vs A' sample count %zu != %zu", pcm_a.size(), pcm_a2.size());
        if (pcm_a.size() == pcm_b.size() && !pcm_a.empty()) {
            CHECK_MSG(std::memcmp(pcm_a.data(), pcm_b.data(),
                                  pcm_a.size() * sizeof(float)) == 0,
                      "direct vs forced-sched PCM NOT bit-identical");
        }
        if (pcm_a.size() == pcm_a2.size() && !pcm_a.empty()) {
            CHECK_MSG(std::memcmp(pcm_a.data(), pcm_a2.data(),
                                  pcm_a.size() * sizeof(float)) == 0,
                      "direct pre- vs post-sched PCM NOT bit-identical (state poisoned)");
        }
    }

    // Deterministic teardown before process exit (the cached model would
    // otherwise be freed by static destructors after the ggml-metal device
    // finalizer, tripping its resource-leak assert).
    s3gen_unload();

    std::fprintf(stderr, "%s: %d/%d checks passed\n",
                 argv[0], g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
