// QVAC-18605 follow-up — CPU-only unit test for the process-wide
// backend-capability probe cache and the new probes added to it.
//
// Three optimizations are exercised here:
//
//   1. `cached_backend_capabilities(backend)` — process-wide cache of
//      the LEAKY_RELU + F16-K/V flash-attn + F16 mul_mat + Q8_0 K/V
//      flash-attn supports_op probes.  Engine + bench + load all hit
//      the cache instead of re-probing the same backend 2-3 times.
//
//   2. `supertonic_backend_supports_f16_mul_mat` — symmetric to the
//      F16-K/V probe.  Gates the `use_f16_weights` auto-policy in
//      `load_supertonic_gguf` so a partial-port backend that ships
//      F16 storage but rejects F16 mul_mat for the hot vector-
//      estimator attention shape stays on the F32 weight path
//      instead of crashing at first synth call.
//
//   3. `supertonic_backend_supports_q8_0_kv_flash_attn` — forward-
//      compat probe for an opt-in Q8_0 K/V dispatch (cuts K/V
//      upload bandwidth ~2× on memory-bandwidth-bound mobile GPUs).
//      The dispatch isn't yet wired but the probe primes the cache
//      so a follow-up patch can flip it without re-querying.
//
// Cache contract verified:
//   - Cold call advances the probe-call counter by exactly 1.
//   - Subsequent calls on the same backend handle don't advance
//     the counter (cache short-circuit).
//   - `supertonic_clear_capability_cache()` lets the next call
//     advance the counter again (test seam works).
//   - All three public forwarders return the same boolean across
//     repeated calls (idempotency).
//   - `nullptr` backend returns `false` from every forwarder.
//
// Probe-result correctness:
//   - On the GGML CPU backend: native LEAKY_RELU is true (CPU has
//     the fused builtin), F16 mul_mat is true (CPU's matmul kernel
//     accepts mixed F16/F32 inputs).  F16-K/V and Q8_0 K/V flash-
//     attn results depend on whether the CPU backend was built
//     with the flash-attn kernel; we don't pin those values here
//     (the smoke test in test_supertonic_vulkan_dispatch.cpp
//     already covers the F16-K/V branch).
//
// No GGUF / model file required.  Registered with `LABEL "unit"`
// in CMakeLists.txt so a fresh checkout's `ctest` exercises this
// without any fixture.

#include "supertonic_internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

// Test 1 — Null-backend safety.
//
// All three public forwarders must return `false` for a null
// backend handle (the engine + bench paths normally never pass
// null, but the test harness exercises this defensively).
void test_null_backend_returns_false() {
    supertonic_clear_capability_cache();
    CHECK(supertonic_backend_supports_f16_kv_flash_attn(nullptr)   == false);
    CHECK(supertonic_backend_supports_f16_mul_mat(nullptr)         == false);
    CHECK(supertonic_backend_supports_q8_0_kv_flash_attn(nullptr)  == false);
    // Round 3 — BF16 K/V probe must also handle null defensively.
    CHECK(supertonic_backend_supports_bf16_kv_flash_attn(nullptr)  == false);
    // Round 3 — pinned-host-buffer probe must also handle null
    // defensively (and is always false off Vulkan, even more so
    // for null).
    CHECK(supertonic_backend_supports_pinned_host_buffer(nullptr)  == false);
}

// Test 2 — Cache short-circuits on a hit.
//
// First call advances the probe-call counter by exactly 1
// (cold cache).  Five subsequent calls in any order on the same
// backend handle don't advance the counter (cache hits).
//
// The counter only counts uncached probe-set executions, not the
// public-forwarder call count — so the test asserts on the
// difference between "call set 1" and "call set 2" rather than
// the absolute value (other tests in this TU may have
// pre-populated the counter via shared cache).
void test_cache_short_circuits_on_hit() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }

    supertonic_clear_capability_cache();
    const uint64_t cold_before = supertonic_capability_probe_call_count();
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);
    const uint64_t cold_after = supertonic_capability_probe_call_count();
    // Cold call must run the uncached probe set exactly once.
    CHECK(cold_after - cold_before == 1);

    const uint64_t warm_before = supertonic_capability_probe_call_count();
    // Five mixed calls on the same backend handle.  Order
    // intentionally varies the public-forwarder triple so the
    // test catches a regression where one forwarder skips the
    // cache.
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);
    (void) supertonic_backend_supports_f16_mul_mat(cpu);
    (void) supertonic_backend_supports_q8_0_kv_flash_attn(cpu);
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);
    (void) supertonic_backend_supports_f16_mul_mat(cpu);
    const uint64_t warm_after = supertonic_capability_probe_call_count();
    // All five calls hit the cache — counter must NOT advance.
    CHECK(warm_after == warm_before);

    ggml_backend_free(cpu);
}

// Test 3 — Cache clear forces a re-probe.
//
// After `supertonic_clear_capability_cache()` the next call on
// the same backend must run the uncached probe set again (the
// counter advances by exactly 1).  Verifies the test seam works
// — same plumbing the regression test relies on for repeatable
// cold-cache assertions.
void test_clear_cache_forces_reprobe() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }

    // First, populate the cache.
    supertonic_clear_capability_cache();
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);

    // Next call must hit the cache.
    const uint64_t before_clear = supertonic_capability_probe_call_count();
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);
    CHECK(supertonic_capability_probe_call_count() == before_clear);

    // Clear + re-call: counter advances by exactly 1.
    supertonic_clear_capability_cache();
    const uint64_t before_reprobe = supertonic_capability_probe_call_count();
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);
    CHECK(supertonic_capability_probe_call_count() == before_reprobe + 1);

    ggml_backend_free(cpu);
}

// Test 4 — Public forwarders are idempotent.
//
// Calling the same forwarder N times on the same backend must
// return the same boolean every time (no random / state-dependent
// answer).  Combined with the cache short-circuit test above this
// gives the engine + bench paths the contract they rely on:
// "the answer at construction matches the answer at first synth".
void test_forwarders_idempotent() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    supertonic_clear_capability_cache();

    bool a1 = supertonic_backend_supports_f16_kv_flash_attn(cpu);
    bool a2 = supertonic_backend_supports_f16_kv_flash_attn(cpu);
    bool a3 = supertonic_backend_supports_f16_kv_flash_attn(cpu);
    CHECK(a1 == a2);
    CHECK(a2 == a3);

    bool b1 = supertonic_backend_supports_f16_mul_mat(cpu);
    bool b2 = supertonic_backend_supports_f16_mul_mat(cpu);
    bool b3 = supertonic_backend_supports_f16_mul_mat(cpu);
    CHECK(b1 == b2);
    CHECK(b2 == b3);

    bool c1 = supertonic_backend_supports_q8_0_kv_flash_attn(cpu);
    bool c2 = supertonic_backend_supports_q8_0_kv_flash_attn(cpu);
    bool c3 = supertonic_backend_supports_q8_0_kv_flash_attn(cpu);
    CHECK(c1 == c2);
    CHECK(c2 == c3);

    ggml_backend_free(cpu);
}

// Test 5 — Two backends get independent cache entries.
//
// Construct two CPU backends (different handles) and verify that
// each gets its own cache entry: a cold call on the second
// backend must advance the probe-call counter even though the
// first backend's entry is already cached.
void test_per_backend_cache_independence() {
    ggml_backend_t cpu_a = ggml_backend_cpu_init();
    ggml_backend_t cpu_b = ggml_backend_cpu_init();
    if (!cpu_a || !cpu_b) {
        std::fprintf(stderr, "skip: dual CPU backend init failed\n");
        if (cpu_a) ggml_backend_free(cpu_a);
        if (cpu_b) ggml_backend_free(cpu_b);
        return;
    }

    supertonic_clear_capability_cache();
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu_a);

    const uint64_t before_b = supertonic_capability_probe_call_count();
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu_b);
    // Different backend handle → separate cache entry → counter
    // must advance.
    CHECK(supertonic_capability_probe_call_count() == before_b + 1);

    // Re-querying the first backend still hits its cache entry.
    const uint64_t before_a = supertonic_capability_probe_call_count();
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu_a);
    CHECK(supertonic_capability_probe_call_count() == before_a);

    ggml_backend_free(cpu_a);
    ggml_backend_free(cpu_b);
}

// Test 6 — F16 mul_mat probe returns true for the GGML CPU backend.
//
// CPU's matmul kernel handles the (F16 weight, F32 activation)
// combination via the existing dot-product fallback path.  This
// is the only backend-specific assertion in this TU; if a future
// CPU backend revision drops F16 support the test catches it.
//
// Probe shape mirrors the live vector-estimator attention W_query
// matmul: weight=[256, 256] F16, activation=[256, 16] F32.
void test_f16_mul_mat_probe_returns_true_on_cpu() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    supertonic_clear_capability_cache();
    bool ok = supertonic_backend_supports_f16_mul_mat(cpu);
    std::fprintf(stderr,
                 "probe(F16 mul_mat, CPU) = %s\n",
                 ok ? "true" : "false");
    CHECK(ok == true);
    ggml_backend_free(cpu);
}

// Test 7 — Q8_0 K/V flash-attn probe smoke test.
//
// We don't pin the boolean (the CPU backend's flash-attn kernel
// support for Q8_0 K/V depends on the build configuration), but
// the probe must run without crashing and return a stable answer
// across repeated calls.  Mostly a "the probe doesn't tickle a
// ggml_can_mul_mat assertion" check — Q8_0 has stricter
// stride / block-size constraints than F16 K/V so a probe-shape
// regression would surface here.
void test_q8_0_kv_flash_attn_probe_smoke() {
    CHECK(supertonic_backend_supports_q8_0_kv_flash_attn(nullptr) == false);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    supertonic_clear_capability_cache();
    bool a = supertonic_backend_supports_q8_0_kv_flash_attn(cpu);
    bool b = supertonic_backend_supports_q8_0_kv_flash_attn(cpu);
    CHECK(a == b);
    std::fprintf(stderr,
                 "probe(Q8_0-K/V flash-attn, CPU) = %s\n",
                 a ? "true" : "false");
    ggml_backend_free(cpu);
}

// Test 8 — BF16 K/V flash-attn probe smoke test (round 3, TDD).
//
// Vulkan's `GGML_OP_FLASH_ATTN_EXT` `supports_op` advertises BF16
// in the coopmat2 path only (`ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT`
// case branch around line 15257).  Like the Q8_0 probe, we don't
// pin the CPU answer (depends on whether ggml-cpu was compiled
// with BF16 dot-product) — we only verify the probe is callable,
// stable across repeated calls, and shares the cache slot with
// the other capability probes.
//
// Probe shape mirrors the live vector-estimator attention site,
// with K/V dtype set to GGML_TYPE_BF16.  Same `kv_len = 16` as
// the F16 probe (BF16 has the same per-element size as F16, so
// no stride / block-size adjustment is needed).
//
// This test is written FIRST (TDD).  It MUST fail before the
// `supertonic_backend_supports_bf16_kv_flash_attn` symbol is
// added.  After implementation, the test must pass without any
// behaviour change to the existing 7 tests above.
void test_bf16_kv_flash_attn_probe_smoke() {
    CHECK(supertonic_backend_supports_bf16_kv_flash_attn(nullptr) == false);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    supertonic_clear_capability_cache();
    bool a = supertonic_backend_supports_bf16_kv_flash_attn(cpu);
    bool b = supertonic_backend_supports_bf16_kv_flash_attn(cpu);
    CHECK(a == b);
    std::fprintf(stderr,
                 "probe(BF16-K/V flash-attn, CPU) = %s\n",
                 a ? "true" : "false");
    ggml_backend_free(cpu);
}

// Test 9 — BF16 K/V probe shares the cache slot (round 3, TDD).
//
// After the cold cache populates via any forwarder, calling the
// BF16-K/V probe must NOT advance the probe-call counter — the
// 5th flag must live in the same `backend_capabilities` struct
// the cache stores per backend handle.  Catches a regression
// where someone adds the new flag but forgets to populate it
// inside `cached_backend_capabilities`.
void test_bf16_kv_probe_shares_cache_slot() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    supertonic_clear_capability_cache();
    // Cold: any forwarder populates the cache.
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);

    // BF16 K/V probe must hit the cache (counter does not advance).
    const uint64_t before = supertonic_capability_probe_call_count();
    (void) supertonic_backend_supports_bf16_kv_flash_attn(cpu);
    CHECK(supertonic_capability_probe_call_count() == before);

    ggml_backend_free(cpu);
}

// Test 10 — pinned-host-buffer probe smoke (round 3, TDD).
//
// `ggml_backend_vk_host_buffer_type()` returns a host-visible,
// device-coherent buffer type that lets the CPU fill an input
// tensor without going through ggml-vulkan's internal staging
// buffer.  Wiring the actual upload path through that buffer is
// a follow-up (requires per-engine input-scratchpad refactor);
// this round only adds the probe so the capability cache is
// primed.
//
// Contract: returns `true` iff the backend is Vulkan AND
// `ggml_backend_vk_host_buffer_type()` returns non-null (the
// only failure mode is a Vulkan-disabled build, where the probe
// returns `false`).  CPU backend → always `false`.
//
// Like the BF16 / Q8_0 K/V probes, this test only verifies the
// probe is callable + idempotent + stable across calls.  The
// CPU answer is pinned to `false` (CPU backend isn't Vulkan).
void test_pinned_host_buffer_probe_smoke() {
    CHECK(supertonic_backend_supports_pinned_host_buffer(nullptr) == false);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    supertonic_clear_capability_cache();
    bool a = supertonic_backend_supports_pinned_host_buffer(cpu);
    bool b = supertonic_backend_supports_pinned_host_buffer(cpu);
    CHECK(a == b);
    // CPU is never Vulkan — pin the answer for CPU.
    CHECK(a == false);
    std::fprintf(stderr,
                 "probe(pinned-host-buffer, CPU) = %s\n",
                 a ? "true" : "false");
    ggml_backend_free(cpu);
}

// Test 11 — pinned-host-buffer probe shares the cache slot (TDD).
//
// 6th flag — must hit the cache after cold-populate.  Same
// regression-catch contract as test 9.
void test_pinned_host_buffer_probe_shares_cache_slot() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    supertonic_clear_capability_cache();
    // Cold: any forwarder populates the cache.
    (void) supertonic_backend_supports_f16_kv_flash_attn(cpu);

    const uint64_t before = supertonic_capability_probe_call_count();
    (void) supertonic_backend_supports_pinned_host_buffer(cpu);
    CHECK(supertonic_capability_probe_call_count() == before);

    ggml_backend_free(cpu);
}

} // namespace

int main() {
    test_null_backend_returns_false();
    test_cache_short_circuits_on_hit();
    test_clear_cache_forces_reprobe();
    test_forwarders_idempotent();
    test_per_backend_cache_independence();
    test_f16_mul_mat_probe_returns_true_on_cpu();
    test_q8_0_kv_flash_attn_probe_smoke();
    test_bf16_kv_flash_attn_probe_smoke();
    test_bf16_kv_probe_shares_cache_slot();
    test_pinned_host_buffer_probe_smoke();
    test_pinned_host_buffer_probe_shares_cache_slot();

    std::fprintf(stderr,
                 "test_supertonic_capability_cache: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
