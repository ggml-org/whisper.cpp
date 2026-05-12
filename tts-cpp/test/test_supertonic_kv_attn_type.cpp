// QVAC-18605 round 4 — CPU-only TDD test for the multi-dtype
// K/V flash-attention dispatch resolver.
//
// Round 4 generalises the round-1 `use_f16_attn` boolean (F16 vs
// F32 only) into a four-valued enum (auto, f32, f16, bf16, q8_0)
// so operators can opt into BF16 K/V (Vulkan coopmat2 — better
// quality than F16 at identical bandwidth) or Q8_0 K/V (Vulkan +
// half the K/V upload bandwidth) when their adapter advertises
// the corresponding capability.
//
// The dispatch policy lives in the pure-logic helper
// `resolve_kv_attn_type(requested, legacy_use_f16_attn,
// backend_supports_f16, backend_supports_bf16,
// backend_supports_q8_0)` so the policy is testable on CPU
// without a Vulkan device.  The actual Vulkan-side cast lives
// behind `#ifdef GGML_USE_VULKAN` in the vector estimator (round
// 4 implementation).
//
// API contract:
//
//   enum class kv_attn_dtype : int {
//       autoselect = -1,  // EngineOptions sentinel; resolver
//                          // never returns this (always concrete).
//       f32        = 0,
//       f16        = 1,
//       bf16       = 2,
//       q8_0       = 3,
//   };
//
//   kv_attn_dtype resolve_kv_attn_type(
//       int requested,                 // -1 / 0 / 1 / 2 / 3 from
//                                      //   EngineOptions::kv_attn_type
//       bool legacy_use_f16_attn,      // model.use_f16_attn (round 1
//                                      //   auto-policy outcome)
//       bool backend_supports_f16,     // probe result
//       bool backend_supports_bf16,    // probe result
//       bool backend_supports_q8_0);   // probe result
//
// Behaviour matrix:
//
//   requested == -1 (auto):
//     legacy_use_f16_attn == true  + backend_supports_f16 → f16
//     legacy_use_f16_attn == true  + !backend_supports_f16 → f32
//     legacy_use_f16_attn == false                          → f32
//
//   requested == 0 (f32 forced):
//     → f32  (regardless of any probe)
//
//   requested == 1 (f16 forced):
//     backend_supports_f16  → f16
//     !backend_supports_f16 → f32 (graceful fallback; loud
//                                  warning logged at the live
//                                  dispatch site, not here)
//
//   requested == 2 (bf16 forced):
//     backend_supports_bf16  → bf16
//     !backend_supports_bf16 → f32 (graceful fallback)
//
//   requested == 3 (q8_0 forced):
//     backend_supports_q8_0  → q8_0
//     !backend_supports_q8_0 → f32 (graceful fallback)
//
//   requested out of [-1..3] → throws std::runtime_error
//                              (caller surfaces the message
//                              verbatim; same pattern as
//                              `resolve_vulkan_device_index`'s
//                              reserved-negative throw).
//
// Why "graceful fallback to F32" instead of "throw" on
// unsupported dtypes?  The probes are advisory — operators
// should be able to set `--kv-attn-type bf16` once in their
// production config and have the engine fall back to F32 on
// Intel ARC (no coopmat2) without crashing.  Loud-failure only
// for actual config errors (out-of-range int).
//
// Written FIRST (TDD).  Whole TU MUST fail to compile before
// the symbol is added, then pass after.

#include "supertonic_internal.h"

#include <cstdio>
#include <stdexcept>

using tts_cpp::supertonic::detail::kv_attn_dtype;
using tts_cpp::supertonic::detail::resolve_kv_attn_type;

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

template <typename F>
bool throws_runtime_error(F && fn) {
    try { fn(); return false; }
    catch (const std::runtime_error &) { return true; }
    catch (...) { return false; }
}

// Test 1 — auto + legacy boolean back-compatibility matrix.
//
// `requested == -1` is the default for the new EngineOptions
// field; it MUST preserve the round-1 `use_f16_attn` semantics
// exactly so existing operator configs see zero behaviour change.
void test_auto_falls_back_to_legacy_boolean() {
    // legacy_use_f16_attn=true + backend supports F16 → f16
    CHECK(resolve_kv_attn_type(-1, /*legacy=*/true,  true,  true,  true)  == kv_attn_dtype::f16);
    CHECK(resolve_kv_attn_type(-1, /*legacy=*/true,  true,  false, false) == kv_attn_dtype::f16);

    // legacy_use_f16_attn=true + backend doesn't support F16 → f32
    // (the round-1 auto-policy probe-gates F16; this reproduces
    // the same fallback semantics for explicit auto + missing probe.)
    CHECK(resolve_kv_attn_type(-1, /*legacy=*/true,  false, true,  true)  == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(-1, /*legacy=*/true,  false, false, false) == kv_attn_dtype::f32);

    // legacy_use_f16_attn=false → f32 regardless of probes.
    // This is the CPU default — auto must NOT silently flip on
    // F16 just because the CPU's flash-attn supports it.
    CHECK(resolve_kv_attn_type(-1, /*legacy=*/false, true,  true,  true)  == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(-1, /*legacy=*/false, false, false, false) == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(-1, /*legacy=*/false, true,  true,  false) == kv_attn_dtype::f32);
}

// Test 2 — f32 forced overrides everything.
//
// `--kv-attn-type 0` (f32) means "I explicitly want F32 K/V even
// if the auto-policy / probes would have promoted me to F16/BF16/Q8_0".
// Useful for parity-harness runs and for triaging perf cliffs
// caused by F16 underflow on a specific model + adapter combo.
void test_f32_forced_overrides_legacy() {
    CHECK(resolve_kv_attn_type(0, /*legacy=*/true,  true,  true,  true) == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(0, /*legacy=*/false, true,  true,  true) == kv_attn_dtype::f32);
    // Probes don't matter for explicit F32.
    CHECK(resolve_kv_attn_type(0, /*legacy=*/true,  false, false, false) == kv_attn_dtype::f32);
}

// Test 3 — f16 forced + probe-gated graceful fallback.
//
// `--kv-attn-type 1` (f16) is the round-1 `--f16-attn 1` semantic
// generalised: enable F16 if the backend supports it, fall back
// to F32 otherwise (same fallback the round-1 auto-policy applies).
void test_f16_forced_probe_gated() {
    // Backend supports F16 → f16.
    CHECK(resolve_kv_attn_type(1, /*legacy=*/true,  true,  false, false) == kv_attn_dtype::f16);
    CHECK(resolve_kv_attn_type(1, /*legacy=*/false, true,  false, false) == kv_attn_dtype::f16);

    // Backend doesn't support F16 → graceful fallback to f32.
    CHECK(resolve_kv_attn_type(1, /*legacy=*/true,  false, true,  true)  == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(1, /*legacy=*/false, false, true,  true)  == kv_attn_dtype::f32);
}

// Test 4 — bf16 forced + probe-gated graceful fallback.
//
// `--kv-attn-type 2` (bf16) is the new dispatch added in round 4.
// Vulkan with coopmat2 supports BF16 K/V; Intel ARC (no coopmat2)
// doesn't.  Graceful fallback to F32 on missing-probe so an
// operator config that says `--kv-attn-type bf16` works on both
// platforms (with the win on coopmat2 hardware, parity F32 on
// the rest).
void test_bf16_forced_probe_gated() {
    // BF16 supported → bf16.
    CHECK(resolve_kv_attn_type(2, /*legacy=*/true,  true,  true,  false) == kv_attn_dtype::bf16);
    CHECK(resolve_kv_attn_type(2, /*legacy=*/false, false, true,  false) == kv_attn_dtype::bf16);

    // BF16 not supported → graceful fallback to f32.  Even when
    // F16 IS supported, we fall back to F32 (not F16) because the
    // operator asked for BF16 specifically; silently downgrading
    // to F16 would mask drift differences between BF16 and F16.
    CHECK(resolve_kv_attn_type(2, /*legacy=*/true,  true,  false, true)  == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(2, /*legacy=*/false, false, false, false) == kv_attn_dtype::f32);
}

// Test 5 — q8_0 forced + probe-gated graceful fallback.
//
// Same shape as the BF16 case; Q8_0 is the bandwidth-saving
// option (half the K/V upload size).  Vulkan supports Q8_0 K/V
// in both scalar and coopmat2 paths.  Forward-compat at this
// round — the probe is in the cache (round 2) but the live
// dispatch only wires when the operator opts in via
// `--kv-attn-type q8_0`.
void test_q8_0_forced_probe_gated() {
    // Q8_0 supported → q8_0.
    CHECK(resolve_kv_attn_type(3, /*legacy=*/true,  true,  true,  true)  == kv_attn_dtype::q8_0);
    CHECK(resolve_kv_attn_type(3, /*legacy=*/false, false, false, true)  == kv_attn_dtype::q8_0);

    // Q8_0 not supported → graceful fallback to f32.
    CHECK(resolve_kv_attn_type(3, /*legacy=*/true,  true,  true,  false) == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(3, /*legacy=*/false, false, false, false) == kv_attn_dtype::f32);
}

// Test 6 — out-of-range request throws.
//
// Loud-failure for actual config errors (CLI typo).  Same pattern
// as `resolve_vulkan_device_index`'s reserved-negative throw.
void test_out_of_range_throws() {
    CHECK(throws_runtime_error([] {
        (void) resolve_kv_attn_type(4, true, true, true, true);
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_kv_attn_type(99, true, true, true, true);
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_kv_attn_type(-2, true, true, true, true);
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_kv_attn_type(-100, true, true, true, true);
    }));
}

// Test 7 — resolver NEVER returns `autoselect`.
//
// `kv_attn_dtype::autoselect` is the EngineOptions sentinel;
// the resolver always returns a concrete dispatch dtype.  This
// test pins the contract so a future refactor can't accidentally
// leak the sentinel through to the dispatch site (which would
// crash on the switch's default branch).
void test_resolver_returns_concrete_only() {
    for (int requested : { -1, 0, 1, 2, 3 }) {
        for (int legacy_bit : { 0, 1 }) {
            const bool legacy = legacy_bit != 0;
            for (int probe_mask = 0; probe_mask < 8; ++probe_mask) {
                const bool sf16  = (probe_mask & 1) != 0;
                const bool sbf16 = (probe_mask & 2) != 0;
                const bool sq8   = (probe_mask & 4) != 0;
                const auto dt = resolve_kv_attn_type(requested, legacy, sf16, sbf16, sq8);
                CHECK(dt != kv_attn_dtype::autoselect);
                // Implicit: every other enum value is OK.
            }
        }
    }
}

} // namespace

int main() {
    test_auto_falls_back_to_legacy_boolean();
    test_f32_forced_overrides_legacy();
    test_f16_forced_probe_gated();
    test_bf16_forced_probe_gated();
    test_q8_0_forced_probe_gated();
    test_out_of_range_throws();
    test_resolver_returns_concrete_only();

    std::fprintf(stderr,
                 "test_supertonic_kv_attn_type: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
