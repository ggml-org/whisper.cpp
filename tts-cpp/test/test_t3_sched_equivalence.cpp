// Equivalence + cache-integrity test for the T3 dual-path dispatch
// (src/sched_dispatch.{h,cpp}): the forced scheduler path must produce
// BIT-IDENTICAL logits to the direct single-backend path, and running
// the scheduler path must not poison the step-graph cache for later
// direct-path calls (ggml_backend_sched_alloc_graph rewrites
// node->src[] in place, so run_step_pass rebuilds fresh graphs on the
// sched path instead of feeding it a cached one).
//
// Sequence (same model, same tokens, same KV positions each phase):
//   A  — direct path:      eval_prompt_mtl + N eval_step_mtl
//   B  — TTS_CPP_FORCE_SCHED=1: same calls through the scheduler
//   A' — direct again:     must still match A (cache not poisoned)
// All logits are compared byte-for-byte (memcmp); bit-exactness is the
// bar — do NOT relax this to a numeric tolerance.
//
// KV state: each phase re-runs the prompt pass, which rewrites cache
// positions [0, prompt_len) and each step rewrites its own position, so
// phases are independent without an explicit KV reset.
//
// --step-cache: sets CHATTERBOX_T3_STEP_CACHE=1 before the first eval
// (the flag is latched in a static on first use, so it is a separate
// ctest registration, not a phase).  With the cache on, phase A fills
// it, phase B must BYPASS the cached graphs, and phase A' proves they
// were left intact.

#include "chatterbox_t3_internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace tts_cpp::chatterbox::detail;

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

constexpr int N_STEPS = 8;

struct phase_result {
    std::vector<float> prompt_cond, prompt_uncond;
    // per-step cond/uncond logits, concatenated
    std::vector<std::vector<float>> step_cond, step_uncond;
    int prompt_len = 0;
};

bool non_degenerate(const std::vector<float> & v) {
    bool any_nonzero = false;
    for (float x : v) {
        if (!std::isfinite(x)) return false;
        if (x != 0.0f) any_nonzero = true;
    }
    return any_nonzero;
}

// One full pass: prompt + N_STEPS greedy-fixed steps.  The step tokens are a
// fixed arbitrary sequence so both phases feed identical inputs.
bool run_phase(const chatterbox_model & model, ggml_gallocr_t allocr,
               int n_threads, const std::vector<int32_t> & text_tokens,
               phase_result & out) {
    if (!eval_prompt_mtl(model, allocr, n_threads, text_tokens, /*exaggeration=*/0.5f,
                         out.prompt_cond, out.prompt_uncond, out.prompt_len)) {
        std::fprintf(stderr, "eval_prompt_mtl failed\n");
        return false;
    }
    out.step_cond.resize(N_STEPS);
    out.step_uncond.resize(N_STEPS);
    for (int s = 0; s < N_STEPS; ++s) {
        const int32_t token = model.hparams.start_speech_token + s;
        if (!eval_step_mtl(model, allocr, n_threads, out.prompt_len + s, token,
                           out.step_cond[s], out.step_uncond[s])) {
            std::fprintf(stderr, "eval_step_mtl failed at step %d\n", s);
            return false;
        }
    }
    return true;
}

bool bytes_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void compare_phases(const char * label, const phase_result & a, const phase_result & b) {
    CHECK_MSG(a.prompt_len == b.prompt_len, "(%s) prompt_len %d vs %d",
              label, a.prompt_len, b.prompt_len);
    CHECK_MSG(bytes_equal(a.prompt_cond, b.prompt_cond),   "(%s) prompt cond logits differ", label);
    CHECK_MSG(bytes_equal(a.prompt_uncond, b.prompt_uncond), "(%s) prompt uncond logits differ", label);
    for (int s = 0; s < N_STEPS; ++s) {
        CHECK_MSG(bytes_equal(a.step_cond[s], b.step_cond[s]),
                  "(%s) step %d cond logits differ", label, s);
        CHECK_MSG(bytes_equal(a.step_uncond[s], b.step_uncond[s]),
                  "(%s) step %d uncond logits differ", label, s);
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s MODEL.gguf [--threads N] [--n-gpu-layers N] [--step-cache]\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    int  n_threads    = (int) std::thread::hardware_concurrency();
    int  n_gpu_layers = 0;
    bool step_cache   = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc)           n_threads    = atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) n_gpu_layers = atoi(argv[++i]);
        else if (a == "--step-cache")                   step_cache   = true;
    }
    if (n_threads <= 0) n_threads = 4;

    // Must be set before the FIRST eval call: the flag is latched in a
    // static on first use.
    if (step_cache) setenv("CHATTERBOX_T3_STEP_CACHE", "1", 1);
    unsetenv("TTS_CPP_FORCE_SCHED");

    std::fprintf(stderr, "test-t3-sched-equivalence: model=%s threads=%d gpu_layers=%d step_cache=%d\n",
                 model_path.c_str(), n_threads, n_gpu_layers, step_cache ? 1 : 0);

    chatterbox_model model;
    if (!load_model_gguf(model_path, model, /*requested_ctx=*/0, n_gpu_layers)) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    if (model.hparams.variant != CHBX_VARIANT_MTL) {
        std::fprintf(stderr, "model is not t3_mtl variant\n");
        return 1;
    }

    // Fixed arbitrary in-vocab text tokens: identical inputs across phases.
    std::vector<int32_t> text_tokens;
    for (int i = 0; i < 12; ++i) {
        text_tokens.push_back(1 + (i * 37) % (model.hparams.n_text_vocab - 1));
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    phase_result direct_a, forced_b, direct_a2;

    // A — direct path.
    if (!run_phase(model, allocr, n_threads, text_tokens, direct_a)) return 1;
    CHECK_MSG(non_degenerate(direct_a.prompt_cond),  "phase A prompt logits degenerate");
    CHECK_MSG(non_degenerate(direct_a.step_cond[0]), "phase A step0 logits degenerate");

    // B — forced scheduler path (per-op fallback machinery engaged even
    // though every op is supported: single-backend pass-through split).
    setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    const bool phase_b_ok = run_phase(model, allocr, n_threads, text_tokens, forced_b);
    unsetenv("TTS_CPP_FORCE_SCHED");
    if (!phase_b_ok) return 1;
    compare_phases("direct vs forced-sched", direct_a, forced_b);

    // A' — direct again: the sched phase must not have poisoned any cached
    // step graph (with --step-cache these calls hit the entries phase A
    // created and phase B was required to bypass).
    if (!run_phase(model, allocr, n_threads, text_tokens, direct_a2)) return 1;
    compare_phases("direct vs direct-after-sched", direct_a, direct_a2);

    ggml_gallocr_free(allocr);
    tts_cpp::chatterbox::detail::t3_release_caches();
    tts_cpp::detail::sched_fallback_free(model.sched_fb);

    std::fprintf(stderr, "test-t3-sched-equivalence: %d checks, %d failures\n",
                 g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
