// Unit test for the MTL sampler's EOS-suppression option (Phase 2,
// improvement #1).  Pure logit math — no model load — but it links libtts-cpp
// because sample_next_token_mtl lives in the t3_mtl translation unit.
//
//   ./test-mtl-sampler         (no arguments; self-contained)

#include "chatterbox_t3_internal.h"

#include <cstdio>
#include <random>
#include <vector>

using namespace tts_cpp::chatterbox::detail;

namespace {
int g_failures = 0, g_checks = 0;
#define CHECK(cond, ...) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) { ++g_failures;                                         \
        fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);            \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); }          \
} while (0)
}

int main() {
    constexpr int V = 16;
    constexpr int32_t STOP = 9;

    // Logits where the stop token is the dominant choice.  The MTL sampler is
    // the CFG path and always indexes uncond, so pass a full (zero) uncond;
    // with cfg_weight = 0 the combined logits equal `cond`.
    std::vector<float> cond((size_t) V, 0.0f);
    cond[(size_t) STOP] = 12.0f;               // stop is the clear argmax
    const std::vector<float> uncond((size_t) V, 0.0f);
    const std::vector<int32_t> generated;

    chatterbox_sampling_params p;
    p.top_k = 0; p.top_p = 1.0f; p.temp = 0.8f;
    p.repeat_penalty = 1.0f; p.min_p = 0.0f; p.cfg_weight = 0.0f;

    // Without suppression: the dominant stop token should be drawn at least once.
    {
        std::mt19937 rng(123);
        bool got_stop = false;
        for (int i = 0; i < 64 && !got_stop; ++i)
            if (sample_next_token_mtl(cond, uncond, generated, p, rng, STOP, /*suppress=*/false) == STOP)
                got_stop = true;
        CHECK(got_stop, "without suppression the dominant stop token must be reachable");
    }

    // With suppression: the stop token must NEVER be drawn, even though it has
    // the highest logit (it is forced to -inf before sampling).
    {
        std::mt19937 rng(123);
        bool got_stop = false;
        for (int i = 0; i < 500; ++i)
            if (sample_next_token_mtl(cond, uncond, generated, p, rng, STOP, /*suppress=*/true) == STOP)
                got_stop = true;
        CHECK(!got_stop, "with suppression the stop token must never be sampled");
    }

    // Suppression must still return a valid in-vocab token (not -1 / OOB).
    {
        std::mt19937 rng(7);
        int32_t t = sample_next_token_mtl(cond, uncond, generated, p, rng, STOP, /*suppress=*/true);
        CHECK(t >= 0 && t < V && t != STOP, "suppressed draw is a valid non-stop token (got %d)", t);
    }

    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
