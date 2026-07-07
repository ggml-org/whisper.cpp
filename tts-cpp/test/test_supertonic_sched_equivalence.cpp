// Equivalence test for the Supertonic dual-path dispatch
// (src/sched_dispatch.{h,cpp}): the forced scheduler path must produce
// BIT-IDENTICAL audio to the direct path.
//
// This holds because every dual-path site REBUILDS its graph before each
// sched pass (sched graphs are single-use for allocation: alloc rewires
// node->src[] into sched-owned copies the next pass frees, and
// buffer/data bindings survive sched_reset — reusing a sched-allocated
// graph computes deterministic garbage on CPU and crashes on a
// [metal,cpu] sched; both were measured before the rebuild fix).  The
// *_gpu cross-graph handles are withheld from sched-run producers, so
// sched-route consumers take the host path with the same in-graph values.
//
//   A  — direct synth
//   B  — TTS_CPP_FORCE_SCHED=1: every dual-path graph through the
//        scheduler ([primary] on CPU; [metal, cpu] with n_gpu_layers>0 —
//        the multi-backend case is the one that used to SIGSEGV).  The
//        front-block and style-residual islands never consult the gate
//        and stay direct; the 6 dual-path graphs are exactly the
//        corruption surface this guards.
//   A' — direct again: no cross-phase state poisoning.
// A fresh Engine per phase (generation_id bump rebuilds every
// thread_local cache).  PCM compared byte-for-byte (memcmp); bit-exactness
// is the bar — do NOT relax this to a numeric tolerance.
//
// usage: test-supertonic-sched-equivalence MODEL.gguf [n_gpu_layers]

#include "tts-cpp/supertonic/engine.h"

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

const char * kText = "The quick brown fox jumps over the lazy dog.";

bool run_phase(const std::string & gguf, int n_gpu_layers, std::vector<float> & pcm,
               const char * label) {
    try {
        tts_cpp::supertonic::EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.seed            = 42;
        opts.n_threads       = 2;
        opts.n_gpu_layers    = n_gpu_layers;
        tts_cpp::supertonic::Engine engine(opts);
        pcm = engine.synthesize(kText).pcm;
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "phase %s: %s\n", label, e.what());
        return false;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [n_gpu_layers]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const int n_gpu_layers = (argc > 2) ? std::atoi(argv[2]) : 0;

    std::vector<float> pcm_a, pcm_b, pcm_a2;

    unsetenv("TTS_CPP_FORCE_SCHED");
    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_a, "A (direct)"), "phase A failed");

    setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_b, "B (sched)"), "phase B failed");
    unsetenv("TTS_CPP_FORCE_SCHED");

    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_a2, "A' (direct)"), "phase A' failed");

    if (g_failures == 0) {
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
            const bool ab_same = std::memcmp(pcm_a.data(), pcm_b.data(),
                                             pcm_a.size() * sizeof(float)) == 0;
            CHECK_MSG(ab_same, "direct vs forced-sched PCM NOT bit-identical "
                               "(a dual-path graph was reused through the scheduler?)");
            if (!ab_same) {
                size_t first = pcm_a.size();
                size_t n_diff = 0;
                float  max_abs = 0.0f;
                for (size_t i = 0; i < pcm_a.size(); ++i) {
                    if (pcm_a[i] != pcm_b[i]) {
                        if (first == pcm_a.size()) first = i;
                        ++n_diff;
                        const float d = std::fabs(pcm_a[i] - pcm_b[i]);
                        if (d > max_abs) max_abs = d;
                    }
                }
                std::fprintf(stderr,
                             "  A/B divergence: first at sample %zu/%zu, %zu samples differ, max |d|=%g\n",
                             first, pcm_a.size(), n_diff, (double) max_abs);
            }
        }
        if (pcm_a.size() == pcm_a2.size() && !pcm_a.empty()) {
            CHECK_MSG(std::memcmp(pcm_a.data(), pcm_a2.data(),
                                  pcm_a.size() * sizeof(float)) == 0,
                      "direct pre- vs post-sched PCM NOT bit-identical");
        }
    }

    std::fprintf(stderr, "%s: %d/%d checks passed\n",
                 argv[0], g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
