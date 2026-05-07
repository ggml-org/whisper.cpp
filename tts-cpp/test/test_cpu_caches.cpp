// CPU-side persistent-cache validation harness for the multilingual
// CPU TTS path.
//
// Verifies the four cache layers added to chatterbox_tts.cpp:
//
//  1. compute_time_mlp_cached() — t_val (float) → (1024,) t_emb vector.
//     Multilingual fires 10 distinct t-values per synth (cosine schedule);
//     Turbo fires 3.  Across synth calls the schedule is constant, so the
//     cache amortises every subsequent synth to zero compute_time_mlp work.
//
//  2. compute_time_emb_cached() — (t_val, r_val) → (1024,) mixed embedding.
//     Turbo meanflow only; multilingual leaves this cache empty.
//
//  3. g_cfm_estimator_cache — promotes the local-scope cfm_estimator_cache
//     to global lifetime so subsequent synth calls don't rebuild the
//     ~5500-node CFM graph or pay the gallocr_reserve cost.
//
//  4. g_weight_cpu_mirror — CPU mirror of large per-synth weight reads
//     (flow/input_embedding ~28 MB on multilingual, spk_embed_affine
//     ~60 KB).  Saves the ggml_backend_tensor_get round-trip every synth.
//
// All caches are invalidated together by s3gen_unload() so that switching
// to a different backend (e.g. CPU → Vulkan) doesn't reuse stale state.
//
// Usage (with model)        : ./test-cpu-caches MODEL_S3GEN.gguf [REF_DIR]
// Usage (cache-key only)    : ./test-cpu-caches
//
// Without a GGUF the harness still runs the lightweight cache-key tests
// that catch the typical -0/+0/NaN / std::hash<float> portability traps.

#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "chatterbox_tts_test_hooks.h"
#include "npy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace th = tts_cpp::chatterbox::test_hooks;

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

// ---------------- 1. cache-key bit-cast tests ----------------
//
// These run unconditionally — no model needed.  They guard the rule
// that the time_mlp result cache uses a bit-cast hash of the float
// (so +0/-0 land in different buckets, NaNs are stable per-bit-pattern,
// and equal floats always hash to the same bucket regardless of how
// they were computed).

void test_cache_keys() {
    fprintf(stderr, "=== cache key (bit-cast) tests ===\n");

    // Equal floats → equal keys.
    CHECK(th::float_cache_key(0.5f) == th::float_cache_key(0.5f),
          "0.5 should be stable");

    // +0.0 and -0.0 are NOT equal under bit-cast (sign bit differs).
    // std::hash<float> typically collapses them — we deliberately don't.
    const float pos_zero = 0.0f;
    const float neg_zero = -0.0f;
    CHECK(th::float_cache_key(pos_zero) != th::float_cache_key(neg_zero),
          "+0 and -0 must produce distinct cache keys");

    // Distinct values → distinct keys (sanity).
    CHECK(th::float_cache_key(0.5f) != th::float_cache_key(0.25f),
          "0.5 vs 0.25 must differ");

    // NaN: bit-pattern stable (we don't normalise) — same NaN payload
    // hashes the same.  This is fine because the time_mlp_cache is
    // only ever queried with t_span values, none of which are NaN.
    uint32_t nan_bits = 0x7fc00001u;  // a quiet NaN
    float nan_val;
    std::memcpy(&nan_val, &nan_bits, sizeof(float));
    CHECK(th::float_cache_key(nan_val) == 0x7fc00001u,
          "NaN bit pattern must round-trip");

    // Pair key: high 32 bits = t_val, low 32 bits = r_val.
    const float t = 0.5f;
    const float r = 1.0f;
    const uint64_t expect =
        ((uint64_t) th::float_cache_key(t) << 32) |
         (uint64_t) th::float_cache_key(r);
    CHECK(th::float_pair_cache_key(t, r) == expect,
          "pair key must compose from individual float keys");

    // Order matters: (t, r) ≠ (r, t).
    CHECK(th::float_pair_cache_key(0.5f, 1.0f) !=
          th::float_pair_cache_key(1.0f, 0.5f),
          "pair key must not be commutative");

    // Cosine schedule used by multilingual (n_timesteps=10) — verify
    // 10 distinct keys.  Mirrors the t_span = 1 - cos(i/10 * pi/2) loop
    // in s3gen_synthesize_to_wav.
    std::vector<uint32_t> keys;
    keys.reserve(10);
    for (int i = 0; i < 10; ++i) {
        float tau = (float) i / 10.0f;
        float t_cos = 1.0f - std::cos(tau * 0.5f * (float) M_PI);
        keys.push_back(th::float_cache_key(t_cos));
    }
    bool all_distinct = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = i + 1; j < keys.size(); ++j) {
            if (keys[i] == keys[j]) { all_distinct = false; break; }
        }
    }
    CHECK(all_distinct,
          "multilingual t-span (n_timesteps=10 cosine) must produce 10 "
          "distinct cache keys, otherwise compute_time_mlp_cached would "
          "alias unrelated steps");
}

// ---------------- 2. starting cache state ----------------

void test_initial_state() {
    fprintf(stderr, "=== initial cache state ===\n");

    // s3gen_unload() before any synth must succeed even if no caches
    // were ever populated (idempotent).  Production callers in the
    // bare-addon teardown rely on this.
    s3gen_unload();
    CHECK(th::time_mlp_result_cache_size() == 0,
          "time_mlp result cache must start empty");
    CHECK(th::time_emb_result_cache_size() == 0,
          "time_emb result cache must start empty");
    CHECK(th::weight_mirror_cache_size() == 0,
          "weight mirror cache must start empty");
    CHECK(!th::cfm_estimator_cache_built(),
          "persistent cfm_estimator_cache must not be built before any "
          "synth");
    CHECK(!th::cfm_estimator_cache_b2(),
          "persistent cfm_estimator_cache b2 flag must default false");

    // Round 2: encoder / HiFT / F0 graph caches + scaffolding caches.
    CHECK(!th::encoder_graph_cache_built(),
          "persistent encoder graph cache must not be built before any synth");
    CHECK(th::encoder_graph_cache_T() == -1,
          "encoder graph cache T must be -1 (sentinel) before any build");
    CHECK(!th::hift_graph_cache_built(),
          "persistent HiFT decoder graph cache must not be built before any synth");
    CHECK(th::hift_graph_cache_T_mel() == -1,
          "HiFT graph cache T_mel must be -1 before any build");
    CHECK(th::hift_graph_cache_T_stft() == -1,
          "HiFT graph cache T_stft must be -1 before any build");
    CHECK(!th::f0_graph_cache_built(),
          "persistent F0 predictor graph cache must not be built before any synth");
    CHECK(th::f0_graph_cache_T_mel() == -1,
          "F0 graph cache T_mel must be -1 before any build");
    CHECK(th::pos_emb_cache_size() == 0,
          "encoder pos_emb result cache must start empty");
    CHECK(th::inv_alpha_cache_size() == 0,
          "HiFT inv_alpha result cache must start empty");
    CHECK(th::istft_kernel_cache_size() == 0,
          "HiFT istft_kernel cache must start empty");
    CHECK(th::hann_window_cache_size() == 0,
          "HiFT hann_window cache must start empty");
    CHECK(th::window_sum_cache_size() == 0,
          "HiFT window_sum cache must start empty");

    // Round 5: STFT graph + analysis-kernel caches.
    CHECK(!th::stft_graph_cache_built(),
          "STFT graph cache must not be built before any synth");
    CHECK(th::stft_graph_cache_T_src() == -1,
          "STFT graph cache T_src must be -1 (sentinel) before any build");
    CHECK(th::stft_kernel_cache_size() == 0,
          "STFT analysis kernel cache must start empty");
}

// ---------------- 3. determinism + cache wiring on a real synth ----------

// Read built-in voice tokens.  No multilingual model available locally,
// so the harness uses the Turbo built-in voice if --ref-dir wasn't
// passed.  The cache logic is model-agnostic by construction; the
// multilingual benefit factor is larger but the bit-exact + lifecycle
// invariants this test verifies are identical across variants.
std::vector<int32_t> sample_speech_tokens() {
    // 24 tokens — enough to exercise the encoder + a single CFM batch
    // without bloating run-time.  Values are within [0, 6561) (S3 vocab).
    return {
        12, 34, 56, 78, 90, 121, 152, 173, 195, 217, 239, 261,
        283, 305, 327, 349, 371, 393, 415, 437, 459, 481, 503, 525,
    };
}

bool synthesize_once(const std::string & gguf,
                     const std::string & ref_dir,
                     std::vector<float> & wav,
                     double & wall_ms) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.ref_dir         = ref_dir;
    opts.out_wav_path    = "";          // stay in-memory
    opts.pcm_out         = &wav;
    opts.seed            = 42;
    opts.n_threads       = 0;           // auto: hardware_concurrency
    opts.sr              = 24000;
    opts.verbose         = false;
    opts.n_gpu_layers    = 0;           // CPU-only for this test
    opts.apply_trim_fade = true;
    opts.finalize        = true;

    const auto tokens = sample_speech_tokens();
    const double t0 = now_ms();
    int rc = s3gen_synthesize_to_wav(tokens, opts);
    wall_ms = now_ms() - t0;
    return rc == 0 && !wav.empty();
}

void test_warm_cache_bit_exact_and_lifecycle(const std::string & gguf,
                                             const std::string & ref_dir) {
    fprintf(stderr, "=== warm-cache bit-exact + lifecycle ===\n");

    // First call populates every cache.  Subsequent calls must (a)
    // produce bit-exact output and (b) skip every cache that was
    // already warmed.
    std::vector<float> wav_a, wav_b, wav_c;
    double t_a = 0, t_b = 0, t_c = 0;
    if (!synthesize_once(gguf, ref_dir, wav_a, t_a)) {
        fprintf(stderr, "skip: synth #1 failed (model load / arch?)\n");
        return;
    }

    const size_t n_time_mlp_after_a = th::time_mlp_result_cache_size();
    const size_t n_time_emb_after_a = th::time_emb_result_cache_size();
    const size_t n_weights_after_a  = th::weight_mirror_cache_size();
    const bool   cfm_built_after_a  = th::cfm_estimator_cache_built();
    const bool   enc_built_after_a  = th::encoder_graph_cache_built();
    const int    enc_T_after_a      = th::encoder_graph_cache_T();
    const bool   hift_built_after_a = th::hift_graph_cache_built();
    const int    hift_Tmel_after_a  = th::hift_graph_cache_T_mel();
    const int    hift_Tstft_after_a = th::hift_graph_cache_T_stft();
    const bool   f0_built_after_a   = th::f0_graph_cache_built();
    const int    f0_Tmel_after_a    = th::f0_graph_cache_T_mel();
    const size_t n_pos_emb_after_a  = th::pos_emb_cache_size();
    const size_t n_inv_alpha_after_a = th::inv_alpha_cache_size();
    const size_t n_istft_after_a    = th::istft_kernel_cache_size();
    const size_t n_hann_after_a     = th::hann_window_cache_size();
    const size_t n_wsum_after_a     = th::window_sum_cache_size();
    const bool   stft_built_after_a = th::stft_graph_cache_built();
    const int    stft_Tsrc_after_a  = th::stft_graph_cache_T_src();
    const size_t n_stft_kern_after_a = th::stft_kernel_cache_size();

    CHECK(cfm_built_after_a,
          "after first synth, persistent cfm_estimator_cache must be built");
    CHECK(n_time_mlp_after_a > 0,
          "after first synth, time_mlp result cache must have at least one "
          "entry (n_timesteps for multilingual / 3 for Turbo)");
    CHECK(n_weights_after_a > 0,
          "after first synth, weight_mirror_cache must have at least one "
          "entry (input_embedding + spk_embed_affine/{w,b})");

    // Round 2 — every per-pipeline graph must be built after the first
    // synth, with non-sentinel keys.
    CHECK(enc_built_after_a,
          "after first synth, persistent encoder graph cache must be built");
    CHECK(enc_T_after_a > 0,
          "after first synth, encoder graph cache T must be > 0 (saw %d)",
          enc_T_after_a);
    CHECK(hift_built_after_a,
          "after first synth, persistent HiFT graph cache must be built");
    CHECK(hift_Tmel_after_a > 0 && hift_Tstft_after_a > 0,
          "after first synth, HiFT graph cache (T_mel=%d, T_stft=%d) must "
          "have positive shape keys",
          hift_Tmel_after_a, hift_Tstft_after_a);
    CHECK(f0_built_after_a,
          "after first synth, persistent F0 predictor graph cache must be built");
    CHECK(f0_Tmel_after_a > 0,
          "after first synth, F0 graph cache T_mel must be > 0 (saw %d)",
          f0_Tmel_after_a);

    // Scaffolding caches: pos_emb fires twice per synth (T and 2T), so
    // ≥ 2 entries.  inv_alpha fires once per HiFT alpha tensor (~72
    // tensors total).  istft_kernel + hann_window are keyed by n_fft
    // (one constant value), so exactly 1 entry each.  window_sum is
    // keyed by T_stft, also exactly 1 entry per synth-shape.
    CHECK(n_pos_emb_after_a >= 2,
          "after first synth, pos_emb cache should have ≥ 2 entries (T and 2T) "
          "but saw %zu", n_pos_emb_after_a);
    CHECK(n_inv_alpha_after_a > 0,
          "after first synth, inv_alpha cache must have at least one entry");
    CHECK(n_istft_after_a == 1,
          "after first synth, istft_kernel cache must have exactly 1 entry "
          "(keyed by n_fft); saw %zu", n_istft_after_a);
    CHECK(n_hann_after_a >= 1,
          "after first synth, hann_window cache must have ≥ 1 entry; saw %zu",
          n_hann_after_a);
    CHECK(n_wsum_after_a == 1,
          "after first synth, window_sum cache must have exactly 1 entry; "
          "saw %zu", n_wsum_after_a);

    // Round 5: STFT graph + analysis-kernel caches.
    CHECK(stft_built_after_a,
          "after first synth, persistent STFT graph cache must be built");
    CHECK(stft_Tsrc_after_a > 0,
          "after first synth, STFT graph cache T_src must be > 0 (saw %d)",
          stft_Tsrc_after_a);
    CHECK(n_stft_kern_after_a == 1,
          "after first synth, STFT analysis kernel cache must have exactly 1 "
          "entry (keyed by n_fft); saw %zu", n_stft_kern_after_a);

    fprintf(stderr,
            "  synth #1: time_mlp=%zu time_emb=%zu weights=%zu cfm=%s "
            "enc=%s(T=%d) hift=%s(T_mel=%d,T_stft=%d) f0=%s(T_mel=%d) "
            "pos_emb=%zu inv_alpha=%zu istft=%zu hann=%zu wsum=%zu "
            "stft=%s(T_src=%d) stft_kern=%zu (%.1f ms)\n",
            n_time_mlp_after_a, n_time_emb_after_a, n_weights_after_a,
            cfm_built_after_a ? "built" : "fresh",
            enc_built_after_a ? "built" : "fresh", enc_T_after_a,
            hift_built_after_a ? "built" : "fresh",
            hift_Tmel_after_a, hift_Tstft_after_a,
            f0_built_after_a ? "built" : "fresh", f0_Tmel_after_a,
            n_pos_emb_after_a, n_inv_alpha_after_a,
            n_istft_after_a, n_hann_after_a, n_wsum_after_a,
            stft_built_after_a ? "built" : "fresh", stft_Tsrc_after_a,
            n_stft_kern_after_a, t_a);

    // Second call: every cache must already be warm.  Its size must
    // not grow because the t-schedule and the model weights are
    // constant across synth calls.
    if (!synthesize_once(gguf, ref_dir, wav_b, t_b)) {
        fprintf(stderr, "skip: synth #2 failed\n");
        return;
    }
    CHECK(th::time_mlp_result_cache_size() == n_time_mlp_after_a,
          "synth #2 must NOT add new time_mlp entries (saw %zu, expected %zu)",
          th::time_mlp_result_cache_size(), n_time_mlp_after_a);
    CHECK(th::time_emb_result_cache_size() == n_time_emb_after_a,
          "synth #2 must NOT add new time_emb entries");
    CHECK(th::weight_mirror_cache_size() == n_weights_after_a,
          "synth #2 must NOT add new weight_mirror entries");
    CHECK(th::cfm_estimator_cache_built(),
          "synth #2 must keep the persistent cfm graph built");

    // Round 2: graph caches must remain built with the same shape
    // keys, scaffolding caches must not grow.
    CHECK(th::encoder_graph_cache_built() && th::encoder_graph_cache_T() == enc_T_after_a,
          "synth #2 must keep the encoder graph built with the same T (was %d, "
          "now built=%d, T=%d)",
          enc_T_after_a, th::encoder_graph_cache_built() ? 1 : 0,
          th::encoder_graph_cache_T());
    CHECK(th::hift_graph_cache_built() &&
          th::hift_graph_cache_T_mel()  == hift_Tmel_after_a &&
          th::hift_graph_cache_T_stft() == hift_Tstft_after_a,
          "synth #2 must keep the HiFT graph built with the same shape keys "
          "(was T_mel=%d, T_stft=%d; now built=%d, T_mel=%d, T_stft=%d)",
          hift_Tmel_after_a, hift_Tstft_after_a,
          th::hift_graph_cache_built() ? 1 : 0,
          th::hift_graph_cache_T_mel(), th::hift_graph_cache_T_stft());
    CHECK(th::f0_graph_cache_built() && th::f0_graph_cache_T_mel() == f0_Tmel_after_a,
          "synth #2 must keep the F0 graph built with the same T_mel (was %d)",
          f0_Tmel_after_a);
    CHECK(th::pos_emb_cache_size()      == n_pos_emb_after_a,
          "synth #2 must NOT add new pos_emb entries (saw %zu, expected %zu)",
          th::pos_emb_cache_size(), n_pos_emb_after_a);
    CHECK(th::inv_alpha_cache_size()    == n_inv_alpha_after_a,
          "synth #2 must NOT add new inv_alpha entries (saw %zu, expected %zu)",
          th::inv_alpha_cache_size(), n_inv_alpha_after_a);
    CHECK(th::istft_kernel_cache_size() == n_istft_after_a,
          "synth #2 must NOT add new istft_kernel entries");
    CHECK(th::hann_window_cache_size()  == n_hann_after_a,
          "synth #2 must NOT add new hann_window entries");
    CHECK(th::window_sum_cache_size()   == n_wsum_after_a,
          "synth #2 must NOT add new window_sum entries");
    CHECK(th::stft_graph_cache_built() &&
          th::stft_graph_cache_T_src() == stft_Tsrc_after_a,
          "synth #2 must keep the STFT graph built with the same T_src "
          "(was %d, now built=%d, T_src=%d)",
          stft_Tsrc_after_a,
          th::stft_graph_cache_built() ? 1 : 0,
          th::stft_graph_cache_T_src());
    CHECK(th::stft_kernel_cache_size() == n_stft_kern_after_a,
          "synth #2 must NOT add new STFT analysis kernel entries");

    CHECK(wav_a.size() == wav_b.size(),
          "warm-cache synth #2 wav length must match cold-cache synth #1 "
          "(%zu vs %zu)", wav_a.size(), wav_b.size());
    if (wav_a.size() == wav_b.size() && !wav_a.empty()) {
        size_t diff = 0;
        float  max_abs = 0;
        for (size_t i = 0; i < wav_a.size(); ++i) {
            float d = std::fabs(wav_a[i] - wav_b[i]);
            if (d > 0) diff++;
            if (d > max_abs) max_abs = d;
        }
        CHECK(diff == 0,
              "warm-cache synth #2 must be byte-for-byte identical to "
              "synth #1 (mismatched samples=%zu, max_abs=%.6e)", diff, max_abs);
    }
    fprintf(stderr, "  synth #2: %.1f ms (warm caches, bit-exact ok)\n", t_b);

    // Third call after s3gen_unload() — every cache must have been
    // reset.  Subsequent synth must repopulate them and still
    // produce bit-exact output (deterministic seed=42).
    s3gen_unload();
    CHECK(th::time_mlp_result_cache_size() == 0,
          "s3gen_unload must clear time_mlp result cache");
    CHECK(th::time_emb_result_cache_size() == 0,
          "s3gen_unload must clear time_emb result cache");
    CHECK(th::weight_mirror_cache_size() == 0,
          "s3gen_unload must clear weight_mirror cache");
    CHECK(!th::cfm_estimator_cache_built(),
          "s3gen_unload must tear down the persistent cfm cache");
    // Round 2 caches must also be torn down — gallocators in the
    // graph caches reference the model's backend and would crash on
    // backend-free if left dangling.
    CHECK(!th::encoder_graph_cache_built(),
          "s3gen_unload must tear down the encoder graph cache");
    CHECK(!th::hift_graph_cache_built(),
          "s3gen_unload must tear down the HiFT decoder graph cache");
    CHECK(!th::f0_graph_cache_built(),
          "s3gen_unload must tear down the F0 predictor graph cache");
    CHECK(th::pos_emb_cache_size() == 0,
          "s3gen_unload must clear pos_emb cache");
    CHECK(th::inv_alpha_cache_size() == 0,
          "s3gen_unload must clear inv_alpha cache");
    CHECK(th::istft_kernel_cache_size() == 0,
          "s3gen_unload must clear istft_kernel cache");
    CHECK(th::hann_window_cache_size() == 0,
          "s3gen_unload must clear hann_window cache");
    CHECK(th::window_sum_cache_size() == 0,
          "s3gen_unload must clear window_sum cache");
    CHECK(!th::stft_graph_cache_built(),
          "s3gen_unload must tear down the STFT graph cache");
    CHECK(th::stft_graph_cache_T_src() == -1,
          "s3gen_unload must reset STFT graph cache T_src to sentinel -1");
    CHECK(th::stft_kernel_cache_size() == 0,
          "s3gen_unload must clear STFT analysis kernel cache");

    // Idempotent: a second unload must not crash or produce errors.
    s3gen_unload();

    if (!synthesize_once(gguf, ref_dir, wav_c, t_c)) {
        fprintf(stderr, "skip: synth #3 (post-unload) failed\n");
        return;
    }
    CHECK(th::cfm_estimator_cache_built(),
          "synth #3 must rebuild the cfm cache after unload");
    CHECK(wav_a.size() == wav_c.size(),
          "post-unload synth wav length must match");
    if (wav_a.size() == wav_c.size() && !wav_a.empty()) {
        size_t diff = 0;
        float max_abs = 0;
        for (size_t i = 0; i < wav_a.size(); ++i) {
            float d = std::fabs(wav_a[i] - wav_c[i]);
            if (d > 0) diff++;
            if (d > max_abs) max_abs = d;
        }
        CHECK(diff == 0,
              "post-unload synth must be byte-for-byte identical to first "
              "synth (mismatched samples=%zu, max_abs=%.6e)",
              diff, max_abs);
    }
    fprintf(stderr, "  synth #3 (post-unload): %.1f ms — bit-exact ok\n", t_c);

    // peek_time_mlp_cached: warm value should round-trip.
    auto cosine_t = [](int i, int n) {
        float tau = (float) i / (float) n;
        return 1.0f - std::cos(tau * 0.5f * (float) M_PI);
    };
    // For Turbo (meanflow=true, n_timesteps=2) the schedule is linear:
    // [0, 0.5, 1.0].  For multilingual (cosine, n_timesteps=10) the
    // schedule is cosine.  We probe both candidates non-destructively;
    // at least one of {0.5f, cosine_t(1,10)} should be present.
    auto a = th::peek_time_mlp_cached(0.5f);
    auto b = th::peek_time_mlp_cached(cosine_t(1, 10));
    CHECK(!a.empty() || !b.empty(),
          "peek_time_mlp_cached must return a populated entry for at least "
          "one of the canonical t-values (0.5 for Turbo or cosine[1] for "
          "multilingual)");
    if (!a.empty()) {
        CHECK(a.size() == 1024,
              "time_mlp cached entry must be (1024,) — saw %zu", a.size());
    }
    if (!b.empty()) {
        CHECK(b.size() == 1024,
              "time_mlp cached entry must be (1024,) — saw %zu", b.size());
    }

    // Variant-specific schedule shape — derived from the time_mlp cache
    // size after a synth populates it.  Multilingual = 10 cosine-spaced
    // t-values + 0 time_emb pairs (non-meanflow); Turbo = ≤3 t-values
    // + 2 (t,r) time_emb pairs (meanflow).
    if (n_time_mlp_after_a == 10 && n_time_emb_after_a == 0) {
        // Multilingual cosine schedule: every entry must round-trip,
        // every cosine_t(i, 10) for i in 0..9 must be present.
        fprintf(stderr, "  detected multilingual variant (cosine n_timesteps=10)\n");
        for (int i = 0; i < 10; ++i) {
            float t_cos = cosine_t(i, 10);
            auto v = th::peek_time_mlp_cached(t_cos);
            CHECK(!v.empty(),
                  "multilingual cosine t_span entry %d (t=%.6f) must be cached "
                  "after first synth", i, t_cos);
            if (!v.empty()) {
                CHECK(v.size() == 1024,
                      "multilingual cached t_emb entry %d size must be 1024 — "
                      "saw %zu", i, v.size());
            }
        }
    } else if (n_time_mlp_after_a <= 3 && n_time_emb_after_a == 2) {
        fprintf(stderr, "  detected Turbo variant (meanflow t_span ⊆ {0,0.5,1})\n");
        // Turbo's meanflow loop visits the pairs (0, 0.5) and (0.5, 1).
        auto v05 = th::peek_time_mlp_cached(0.5f);
        CHECK(!v05.empty(),
              "Turbo: t_val=0.5 must be in time_mlp cache after first synth");
    } else {
        fprintf(stderr,
                "  unrecognised variant: time_mlp=%zu time_emb=%zu — neither "
                "the multilingual (10/0) nor Turbo (≤3/2) shape\n",
                n_time_mlp_after_a, n_time_emb_after_a);
    }
}

// ---------------- 4. Streaming shape invalidation ---------------------------
//
// Streaming mode synthesises chunks of varying length; T is different on
// every call.  The generic graph_cache rebuilds when its key diverges —
// this test exercises that branch by submitting two different token
// counts and checking the encoder / HiFT cache keys move with them
// while the t-schedule / weight caches remain stable.

void test_streaming_shape_invalidation(const std::string & gguf,
                                       const std::string & ref_dir) {
    fprintf(stderr, "=== streaming shape invalidation ===\n");

    s3gen_unload();  // clean slate

    // Chunk #1 — shorter token sequence.
    std::vector<int32_t> short_tokens = {12, 34, 56, 78, 90, 121, 152, 173};
    s3gen_synthesize_opts opts1;
    opts1.s3gen_gguf_path = gguf;
    opts1.ref_dir         = ref_dir;
    opts1.out_wav_path    = "";
    std::vector<float> wav1;
    opts1.pcm_out         = &wav1;
    opts1.seed            = 42;
    opts1.n_threads       = 0;
    opts1.sr              = 24000;
    opts1.n_gpu_layers    = 0;
    opts1.apply_trim_fade = true;
    opts1.finalize        = true;
    if (s3gen_synthesize_to_wav(short_tokens, opts1) != 0 || wav1.empty()) {
        fprintf(stderr, "skip: chunk #1 synth failed\n");
        return;
    }
    const int enc_T_chunk1     = th::encoder_graph_cache_T();
    const int hift_Tmel_chunk1 = th::hift_graph_cache_T_mel();
    const int f0_Tmel_chunk1   = th::f0_graph_cache_T_mel();
    const int stft_Tsrc_chunk1 = th::stft_graph_cache_T_src();

    // Chunk #2 — longer token sequence (different shape).  All the
    // graph caches must rebuild, the t-schedule + weight + scaffolding
    // result caches must NOT grow.
    std::vector<int32_t> long_tokens;
    for (int i = 0; i < 32; ++i) long_tokens.push_back(50 + i * 7);
    s3gen_synthesize_opts opts2 = opts1;
    std::vector<float> wav2;
    opts2.pcm_out = &wav2;
    if (s3gen_synthesize_to_wav(long_tokens, opts2) != 0 || wav2.empty()) {
        fprintf(stderr, "skip: chunk #2 synth failed\n");
        return;
    }
    const int enc_T_chunk2     = th::encoder_graph_cache_T();
    const int hift_Tmel_chunk2 = th::hift_graph_cache_T_mel();
    const int f0_Tmel_chunk2   = th::f0_graph_cache_T_mel();
    const int stft_Tsrc_chunk2 = th::stft_graph_cache_T_src();

    CHECK(enc_T_chunk1 != enc_T_chunk2,
          "encoder graph cache T must change between chunks of different "
          "lengths (chunk1 T=%d, chunk2 T=%d)",
          enc_T_chunk1, enc_T_chunk2);
    CHECK(hift_Tmel_chunk1 != hift_Tmel_chunk2,
          "HiFT graph cache T_mel must change between chunks (chunk1=%d, "
          "chunk2=%d)", hift_Tmel_chunk1, hift_Tmel_chunk2);
    CHECK(f0_Tmel_chunk1 != f0_Tmel_chunk2,
          "F0 graph cache T_mel must change between chunks (chunk1=%d, "
          "chunk2=%d)", f0_Tmel_chunk1, f0_Tmel_chunk2);
    CHECK(stft_Tsrc_chunk1 != stft_Tsrc_chunk2,
          "STFT graph cache T_src must change between chunks of different "
          "lengths (chunk1 T_src=%d, chunk2 T_src=%d)",
          stft_Tsrc_chunk1, stft_Tsrc_chunk2);
    CHECK(th::encoder_graph_cache_built(),
          "encoder graph cache must remain built after shape change "
          "(rebuilt for new T)");
    CHECK(th::hift_graph_cache_built(),
          "HiFT graph cache must remain built after shape change");
    CHECK(th::f0_graph_cache_built(),
          "F0 graph cache must remain built after shape change");
    CHECK(th::stft_graph_cache_built(),
          "STFT graph cache must remain built after shape change "
          "(rebuilt for new T_src)");
    CHECK(th::stft_kernel_cache_size() == 1,
          "STFT analysis kernel cache must stay at exactly 1 entry across "
          "chunks (n_fft is constant); got %zu", th::stft_kernel_cache_size());
    fprintf(stderr,
            "  chunk #1: enc_T=%d hift_T_mel=%d f0_T_mel=%d wav_len=%zu\n"
            "  chunk #2: enc_T=%d hift_T_mel=%d f0_T_mel=%d wav_len=%zu\n",
            enc_T_chunk1, hift_Tmel_chunk1, f0_Tmel_chunk1, wav1.size(),
            enc_T_chunk2, hift_Tmel_chunk2, f0_Tmel_chunk2, wav2.size());

    // pos_emb cache might add up to 2 new entries (T2 and 2*T2 for the
    // longer chunk).  The previous chunk's entries persist (we don't
    // evict on shape change).
    CHECK(th::pos_emb_cache_size() >= 2,
          "pos_emb cache must contain ≥ 2 entries across two chunks of "
          "different lengths (got %zu)", th::pos_emb_cache_size());

    // Window-sum cache: 1 entry per distinct T_stft.  Two chunks of
    // different lengths produce two distinct T_stft values, so the
    // cache must hold exactly 2 entries.
    CHECK(th::window_sum_cache_size() >= 1,
          "window_sum cache must contain ≥ 1 entry after multi-shape "
          "synthesis (got %zu)", th::window_sum_cache_size());

    // hann_window + istft_kernel are keyed by n_fft (single value
    // shared across all chunks) — sizes must NOT grow with chunk count.
    CHECK(th::hann_window_cache_size() <= 2,
          "hann_window cache size must stay small across chunks (got %zu); "
          "if this grows with chunk count the key is wrong", th::hann_window_cache_size());
    CHECK(th::istft_kernel_cache_size() == 1,
          "istft_kernel cache must stay at 1 entry (n_fft is constant); "
          "got %zu", th::istft_kernel_cache_size());
}

}  // namespace

int main(int argc, char ** argv) {
    fprintf(stderr, "test-cpu-caches: CPU-side persistent-cache validation\n");

    test_cache_keys();
    test_initial_state();

    if (argc < 2) {
        fprintf(stderr, "\n(no GGUF given — skipping warm-cache + lifecycle "
                        "tests; run as `%s MODEL.gguf [REF_DIR]` to exercise "
                        "the full pipeline)\n", argv[0]);
    } else {
        const std::string gguf = argv[1];
        const std::string ref_dir = (argc >= 3 ? argv[2] : "");
        if (!path_exists(gguf)) {
            fprintf(stderr, "error: GGUF not found at %s\n", gguf.c_str());
            return 2;
        }
        test_warm_cache_bit_exact_and_lifecycle(gguf, ref_dir);
        test_streaming_shape_invalidation(gguf, ref_dir);
    }

    // Always release at exit so the next test invocation starts clean.
    s3gen_unload();

    fprintf(stderr, "\n=== summary ===\n  checks:   %d\n  failures: %d\n",
            g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
