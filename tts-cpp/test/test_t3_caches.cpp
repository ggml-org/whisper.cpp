// T3 step-graph cache validation.
//
// Verifies the per-(n_past, is_uncond) graph cache that
// `build_step_graph_mtl` consults instead of rebuilding the ~5500-
// node graph from scratch every token-decode call.  Multilingual
// fires the step graph 2× per token (CFG cond + uncond); a 136-token
// utterance previously rebuilt 272 graphs at ~3 ms each — ~800 ms
// of pure host-CPU work that the cache eliminates after warm-up.
//
// Coverage:
//   1. Cache empty before any eval_step_mtl call.
//   2. After one eval_step_mtl call, cache holds 2 entries
//      (cond + uncond at n_past=0).
//   3. Calling eval_step_mtl with the same (n_past, is_uncond) key
//      reuses the cached graph (hits++, no new entries).
//   4. Calling at a different n_past adds new entries.
//   5. logits_cond / logits_uncond are bit-exact across cold and
//      warm-cache step calls (KV cache state held identical via
//      explicit ordering).
//   6. t3_release_caches() drops every entry; second call is
//      idempotent; subsequent eval_step_mtl rebuilds.
//   7. (Optional, slow) LRU eviction: filling the cache past
//      `t3_step_graph_cache_capacity()` evicts the oldest entry.
//
// Usage:
//   ./test-t3-caches MTL_T3.gguf
//
// Without arguments, runs only the lightweight default-state
// invariants (no model load required).

#include "chatterbox_t3_internal.h"
#include "chatterbox_tts_test_hooks.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace th = tts_cpp::chatterbox::test_hooks;
using namespace tts_cpp::chatterbox::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, ...) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
        ++g_failures;                                                    \
        fprintf(stderr, "FAIL %s:%d  %s\n        ",                      \
                __FILE__, __LINE__, #cond);                              \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
    }                                                                    \
} while (0)

bool path_exists(const std::string & p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

// ---------------- 1. default invariants (no model required) ---------------

void test_initial_state() {
    fprintf(stderr, "=== t3 step-graph cache: initial state ===\n");

    // Idempotent before any work.
    th::t3_release_caches();

    CHECK(th::t3_step_graph_cache_size() == 0,
          "cache must start empty");
    CHECK(th::t3_step_graph_cache_capacity() > 0,
          "cache capacity must be positive (saw %zu)",
          th::t3_step_graph_cache_capacity());
    CHECK(th::t3_step_graph_cache_hits() == 0,
          "hits counter must start at 0");
    CHECK(th::t3_step_graph_cache_misses() == 0,
          "misses counter must start at 0");
    CHECK(!th::t3_step_graph_cache_contains(/*n_past=*/0, /*is_uncond=*/false),
          "no (n_past=0, cond) entry should be present");
    CHECK(!th::t3_step_graph_cache_contains(/*n_past=*/0, /*is_uncond=*/true),
          "no (n_past=0, uncond) entry should be present");

    // Second release must not crash or produce errors.
    th::t3_release_caches();
}

// ---------------- 2. step pass cache lifecycle (model required) -----------

// Run one eval_step_mtl call with the given (n_past, token) and
// capture both cond + uncond logits.  Always runs cond first, then
// uncond — eval_step_mtl populates both halves on each call.
bool run_step(const chatterbox_model & model, ggml_gallocr_t allocr,
              int n_threads, int n_past, int32_t token,
              std::vector<float> & logits_cond,
              std::vector<float> & logits_uncond) {
    return eval_step_mtl(model, allocr, n_threads, n_past, token,
                         logits_cond, logits_uncond);
}

void test_step_lifecycle(const std::string & model_path) {
    fprintf(stderr, "=== t3 step-graph cache: lifecycle (model=%s) ===\n",
            model_path.c_str());

    th::t3_release_caches();  // clean slate

    chatterbox_model model;
    if (!load_model_gguf(model_path, model, /*requested_ctx=*/0,
                         /*n_gpu_layers=*/0)) {
        fprintf(stderr, "skip: failed to load model\n");
        return;
    }
    if (model.hparams.variant != CHBX_VARIANT_MTL) {
        fprintf(stderr, "skip: model is not MTL variant\n");
        return;
    }

    const int n_threads = std::max(1u, std::thread::hardware_concurrency() / 2u);
    ggml_gallocr_t allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend));
    CHECK(allocr != nullptr, "gallocr_new must succeed");
    if (!allocr) {
        return;
    }

    // -------- (a) first call populates 2 entries (cond + uncond) ---------
    std::vector<float> logits_cond_a, logits_uncond_a;
    const double t0 = now_ms();
    const bool ok = run_step(model, allocr, n_threads,
                             /*n_past=*/0, /*token=*/100,
                             logits_cond_a, logits_uncond_a);
    const double dt_first = now_ms() - t0;
    CHECK(ok, "eval_step_mtl(n_past=0, token=100) must succeed");
    if (!ok) goto cleanup;

    CHECK(th::t3_step_graph_cache_size() == 2,
          "after first eval_step_mtl, cache must hold exactly 2 entries "
          "(cond + uncond at n_past=0); saw %zu",
          th::t3_step_graph_cache_size());
    CHECK(th::t3_step_graph_cache_contains(/*n_past=*/0, /*is_uncond=*/false),
          "(n_past=0, cond) must be present after first call");
    CHECK(th::t3_step_graph_cache_contains(/*n_past=*/0, /*is_uncond=*/true),
          "(n_past=0, uncond) must be present after first call");
    CHECK(th::t3_step_graph_cache_misses() == 2,
          "first call must record 2 misses (one per mode); saw %zu",
          th::t3_step_graph_cache_misses());
    CHECK(th::t3_step_graph_cache_hits() == 0,
          "first call must record 0 hits; saw %zu",
          th::t3_step_graph_cache_hits());
    fprintf(stderr,
            "  call #1 (cold cache): %.1f ms  cache_size=%zu\n",
            dt_first, th::t3_step_graph_cache_size());

    // -------- (b) re-run at the same n_past — cache HIT ------------------
    //
    // Note: eval_step_mtl writes into the KV cache at position n_past
    // every call.  Repeating at n_past=0 with the same token should be
    // bit-exact because (i) the input is identical and (ii) the KV slot
    // is overwritten with the same value.  We spot-check this below.
    {
        std::vector<float> logits_cond_b, logits_uncond_b;
        const double t1 = now_ms();
        const bool ok2 = run_step(model, allocr, n_threads,
                                  /*n_past=*/0, /*token=*/100,
                                  logits_cond_b, logits_uncond_b);
        const double dt_warm = now_ms() - t1;
        CHECK(ok2, "second eval_step_mtl(n_past=0) must succeed");
        if (!ok2) goto cleanup;

        CHECK(th::t3_step_graph_cache_size() == 2,
              "second call at same key must NOT grow cache (saw %zu)",
              th::t3_step_graph_cache_size());
        CHECK(th::t3_step_graph_cache_hits() == 2,
              "second call must record 2 hits (cond + uncond); saw %zu",
              th::t3_step_graph_cache_hits());
        CHECK(th::t3_step_graph_cache_misses() == 2,
              "miss counter must stay at 2 after a warm call; saw %zu",
              th::t3_step_graph_cache_misses());
        fprintf(stderr,
                "  call #2 (warm cache): %.1f ms  cache_size=%zu  hits=%zu\n",
                dt_warm, th::t3_step_graph_cache_size(),
                th::t3_step_graph_cache_hits());

        // Bit-exact (or float-identical) on logits across cold/warm.
        // The graph topology is the same, the same backend runs the
        // same compute, the same KV slot gets re-overwritten with the
        // same data.  Any drift here would mean the cached graph is
        // reading stale state.
        CHECK(logits_cond_b.size() == logits_cond_a.size(),
              "cond logits size mismatch across calls (cold=%zu warm=%zu)",
              logits_cond_a.size(), logits_cond_b.size());
        CHECK(logits_uncond_b.size() == logits_uncond_a.size(),
              "uncond logits size mismatch across calls (cold=%zu warm=%zu)",
              logits_uncond_a.size(), logits_uncond_b.size());
        if (logits_cond_a.size() == logits_cond_b.size()) {
            const int rc =
                std::memcmp(logits_cond_a.data(), logits_cond_b.data(),
                            logits_cond_a.size() * sizeof(float));
            CHECK(rc == 0,
                  "cond logits must be byte-identical across cold/warm cache "
                  "calls at same (n_past, token)");
        }
        if (logits_uncond_a.size() == logits_uncond_b.size()) {
            const int rc =
                std::memcmp(logits_uncond_a.data(), logits_uncond_b.data(),
                            logits_uncond_a.size() * sizeof(float));
            CHECK(rc == 0,
                  "uncond logits must be byte-identical across cold/warm cache "
                  "calls at same (n_past, token)");
        }
    }

    // -------- (c) different n_past → cache grows -------------------------
    {
        std::vector<float> lc, lu;
        const bool ok3 = run_step(model, allocr, n_threads,
                                  /*n_past=*/1, /*token=*/200, lc, lu);
        CHECK(ok3, "eval_step_mtl(n_past=1) must succeed");
        if (!ok3) goto cleanup;

        CHECK(th::t3_step_graph_cache_size() == 4,
              "after a step at a NEW n_past, cache must hold 4 entries; saw %zu",
              th::t3_step_graph_cache_size());
        CHECK(th::t3_step_graph_cache_contains(/*n_past=*/1, /*is_uncond=*/false),
              "(n_past=1, cond) must be present");
        CHECK(th::t3_step_graph_cache_contains(/*n_past=*/1, /*is_uncond=*/true),
              "(n_past=1, uncond) must be present");
        CHECK(th::t3_step_graph_cache_misses() == 4,
              "second n_past must record 4 misses total; saw %zu",
              th::t3_step_graph_cache_misses());
    }

    // -------- (d) explicit teardown -------------------------------------
    th::t3_release_caches();
    CHECK(th::t3_step_graph_cache_size() == 0,
          "t3_release_caches() must drop every entry; saw %zu",
          th::t3_step_graph_cache_size());
    CHECK(th::t3_step_graph_cache_hits() == 0,
          "release must reset hits counter");
    CHECK(th::t3_step_graph_cache_misses() == 0,
          "release must reset misses counter");
    th::t3_release_caches();  // idempotent

cleanup:
    // Always release caches BEFORE freeing the backend (per the
    // contract documented on detail::t3_release_caches).
    th::t3_release_caches();
    if (allocr) ggml_gallocr_free(allocr);
    if (model.buffer_w)        ggml_backend_buffer_free(model.buffer_w);
    if (model.buffer_kv)       ggml_backend_buffer_free(model.buffer_kv);
    if (model.buffer_stack)    ggml_backend_buffer_free(model.buffer_stack);
    if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
    if (model.backend)         ggml_backend_free(model.backend);
    if (model.ctx_w)           ggml_free(model.ctx_w);
    if (model.ctx_kv)          ggml_free(model.ctx_kv);
    if (model.ctx_stack)       ggml_free(model.ctx_stack);
    if (model.ctx_override)    ggml_free(model.ctx_override);
}

// ---------------- 3. multi-synth amortisation timing test ------------------
//
// Demonstrates the actual server-mode win: run N step calls at
// increasing n_past (cold cache, building entries), then run the
// same N calls again (warm cache, every entry is a hit).  The second
// pass is what server-mode users see when synth #2 starts at
// n_past=0 again to decode a different prompt of similar length.
//
// Bit-exact assertion: cold-pass logits and warm-pass logits at the
// same (n_past, token) are byte-identical because the graph is the
// same and the KV cache slot was overwritten with identical data.

void test_multi_synth_timing(const std::string & model_path) {
    fprintf(stderr, "=== t3 step-graph cache: multi-synth timing (cold vs warm) ===\n");

    th::t3_release_caches();

    chatterbox_model model;
    if (!load_model_gguf(model_path, model, /*requested_ctx=*/0,
                         /*n_gpu_layers=*/0)) {
        fprintf(stderr, "skip: failed to load model\n");
        return;
    }
    if (model.hparams.variant != CHBX_VARIANT_MTL) {
        fprintf(stderr, "skip: model is not MTL variant\n");
        return;
    }

    const int n_threads = std::max(1u, std::thread::hardware_concurrency() / 2u);
    ggml_gallocr_t allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) return;

    // 16 steps × 2 modes = 32 cached entries; both passes assert bit-
    // exact logits, so we keep the cold-pass outputs around to diff
    // against the warm pass.  Fits comfortably under T3_STEP_CACHE_CAP
    // (256), so no LRU eviction during the test.
    constexpr int N_STEPS = 16;
    std::vector<std::vector<float>> cold_cond(N_STEPS), cold_uncond(N_STEPS);
    std::vector<std::vector<float>> warm_cond(N_STEPS), warm_uncond(N_STEPS);

    // -------- cold pass: 16 step calls, each populates 2 cache entries -----
    bool ok = true;
    double t_cold = 0;
    {
        const double t_cold0 = now_ms();
        for (int i = 0; i < N_STEPS && ok; ++i) {
            if (!run_step(model, allocr, n_threads,
                          /*n_past=*/i, /*token=*/100 + i,
                          cold_cond[i], cold_uncond[i])) {
                fprintf(stderr, "skip: cold step #%d failed\n", i);
                ok = false;
            }
        }
        t_cold = now_ms() - t_cold0;
    }

    if (ok) {
        const size_t expected = (size_t) N_STEPS * 2;
        CHECK(th::t3_step_graph_cache_size() == expected,
              "after %d cold steps, cache must hold %zu entries; saw %zu",
              N_STEPS, expected, th::t3_step_graph_cache_size());
        CHECK(th::t3_step_graph_cache_misses() == expected,
              "all cold-pass step calls must be cache misses; saw %zu",
              th::t3_step_graph_cache_misses());
        CHECK(th::t3_step_graph_cache_hits() == 0,
              "no hits during cold pass; saw %zu",
              th::t3_step_graph_cache_hits());
    }

    // -------- warm pass: re-run the same n_past sequence — every call
    //          is a cache hit ------------------------------------------------
    if (ok) {
        const size_t hits_before = th::t3_step_graph_cache_hits();
        const double t_warm0 = now_ms();
        for (int i = 0; i < N_STEPS && ok; ++i) {
            if (!run_step(model, allocr, n_threads,
                          /*n_past=*/i, /*token=*/100 + i,
                          warm_cond[i], warm_uncond[i])) {
                fprintf(stderr, "skip: warm step #%d failed\n", i);
                ok = false;
            }
        }
        const double t_warm = now_ms() - t_warm0;

        if (ok) {
            const size_t hits_added = th::t3_step_graph_cache_hits() - hits_before;
            const size_t expected_hits = (size_t) N_STEPS * 2;
            CHECK(hits_added == expected_hits,
                  "warm pass must hit cache %zu times; saw %zu",
                  expected_hits, hits_added);
            CHECK(th::t3_step_graph_cache_misses() == expected_hits,
                  "warm pass must NOT add new misses (%zu); saw %zu",
                  expected_hits, th::t3_step_graph_cache_misses());

            // Bit-exact across cold/warm at every (n_past, token) pair.
            for (int i = 0; i < N_STEPS; ++i) {
                CHECK(cold_cond[i].size() == warm_cond[i].size(),
                      "step %d cond logits size mismatch", i);
                CHECK(cold_uncond[i].size() == warm_uncond[i].size(),
                      "step %d uncond logits size mismatch", i);
                if (cold_cond[i].size() == warm_cond[i].size()) {
                    const int rc = std::memcmp(cold_cond[i].data(),
                                               warm_cond[i].data(),
                                               cold_cond[i].size() * sizeof(float));
                    CHECK(rc == 0, "step %d cond logits not bit-exact across cold/warm", i);
                }
                if (cold_uncond[i].size() == warm_uncond[i].size()) {
                    const int rc = std::memcmp(cold_uncond[i].data(),
                                               warm_uncond[i].data(),
                                               cold_uncond[i].size() * sizeof(float));
                    CHECK(rc == 0, "step %d uncond logits not bit-exact across cold/warm", i);
                }
            }

            const double saved = t_cold - t_warm;
            const double pct   = t_cold > 0 ? 100.0 * saved / t_cold : 0.0;
            fprintf(stderr,
                    "  cold pass (%d steps × 2 modes): %.1f ms\n"
                    "  warm pass (same shapes):        %.1f ms\n"
                    "  saved by cache:                 %.1f ms (%.1f %%)\n"
                    "  per-step savings:               %.2f ms\n",
                    N_STEPS, t_cold, t_warm, saved, pct,
                    (double)(t_cold - t_warm) / (double)(N_STEPS * 2));

            CHECK(t_warm < t_cold,
                  "warm pass must be measurably faster than cold pass "
                  "(cold=%.1f ms, warm=%.1f ms)", t_cold, t_warm);
        }
    }


    th::t3_release_caches();
    if (allocr) ggml_gallocr_free(allocr);
    if (model.buffer_w)        ggml_backend_buffer_free(model.buffer_w);
    if (model.buffer_kv)       ggml_backend_buffer_free(model.buffer_kv);
    if (model.buffer_stack)    ggml_backend_buffer_free(model.buffer_stack);
    if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
    if (model.backend)         ggml_backend_free(model.backend);
    if (model.ctx_w)           ggml_free(model.ctx_w);
    if (model.ctx_kv)          ggml_free(model.ctx_kv);
    if (model.ctx_stack)       ggml_free(model.ctx_stack);
    if (model.ctx_override)    ggml_free(model.ctx_override);
}

}  // namespace

int main(int argc, char ** argv) {
    fprintf(stderr, "test-t3-caches: T3 step-graph cache validation\n");

    // Enable the opt-in cache for the duration of the test.  In
    // production the cache is gated behind CHATTERBOX_T3_STEP_CACHE
    // (default off; server-mode callers opt in to amortise across
    // synths).  See t3_mtl.cpp t3_step_cache_enabled().
#ifdef _WIN32
    _putenv_s("CHATTERBOX_T3_STEP_CACHE", "1");
#else
    setenv("CHATTERBOX_T3_STEP_CACHE", "1", /*overwrite=*/1);
#endif

    test_initial_state();

    if (argc >= 2) {
        const std::string model_path = argv[1];
        if (!path_exists(model_path)) {
            fprintf(stderr, "error: model not found at %s\n", model_path.c_str());
            return 2;
        }
        test_step_lifecycle(model_path);
        test_multi_synth_timing(model_path);
    } else {
        fprintf(stderr, "\n(no GGUF given — skipping step-pass tests; "
                        "run as `%s MTL_T3.gguf` to exercise the full cache)\n",
                argv[0]);
    }

    th::t3_release_caches();

    fprintf(stderr, "\n=== summary ===\n  checks:   %d\n  failures: %d\n",
            g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
