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

// Test 7 — resolver NEVER returns `autoselect`, AND every
// happy-path branch maps to the EXACT expected concrete dtype.
//
// `kv_attn_dtype::autoselect` is the EngineOptions sentinel;
// the resolver always returns a concrete dispatch dtype.  This
// test pins the contract so a future refactor can't accidentally
// leak the sentinel through to the dispatch site (which would
// crash on the switch's default branch).
//
// PR #18 reviewer (Omar) follow-up: the original exhaustive
// 5 × 2 × 8 sweep only asserted `dt != autoselect`, so a typo
// in the resolver (e.g., returning `f16` when `bf16` was
// requested + supported) would pass silently.  This test now
// computes the expected concrete dtype as a pure function of
// the inputs (mirror of the resolver's behaviour matrix) and
// `CHECK`s the resolver's return value against that expected
// dtype on every one of the 80 grid points — a typo in any
// dispatch branch now fails LOUD with the exact mismatch.
void test_resolver_returns_concrete_only() {
    // Reference resolver — same behaviour matrix, separately
    // implemented so a typo on one side doesn't cancel out
    // a typo on the other.  Reads like the table in
    // `supertonic_internal.h`'s docstring on
    // `resolve_kv_attn_type`.
    auto expected = [](int requested, bool legacy,
                       bool sf16, bool sbf16, bool sq8) -> kv_attn_dtype {
        switch (requested) {
            case -1: return (legacy && sf16) ? kv_attn_dtype::f16 : kv_attn_dtype::f32;
            case 0:  return kv_attn_dtype::f32;
            case 1:  return sf16  ? kv_attn_dtype::f16  : kv_attn_dtype::f32;
            case 2:  return sbf16 ? kv_attn_dtype::bf16 : kv_attn_dtype::f32;
            case 3:  return sq8   ? kv_attn_dtype::q8_0 : kv_attn_dtype::f32;
        }
        // Unreachable for the request range we sweep below.
        return kv_attn_dtype::autoselect;
    };
    for (int requested : { -1, 0, 1, 2, 3 }) {
        for (int legacy_bit : { 0, 1 }) {
            const bool legacy = legacy_bit != 0;
            for (int probe_mask = 0; probe_mask < 8; ++probe_mask) {
                const bool sf16  = (probe_mask & 1) != 0;
                const bool sbf16 = (probe_mask & 2) != 0;
                const bool sq8   = (probe_mask & 4) != 0;
                const auto dt  = resolve_kv_attn_type(requested, legacy, sf16, sbf16, sq8);
                const auto exp = expected(requested, legacy, sf16, sbf16, sq8);
                CHECK(dt != kv_attn_dtype::autoselect);
                CHECK(dt == exp);
            }
        }
    }

    // Belt-and-suspenders happy-path spot checks (Omar's
    // example): the explicit-request paths get the dtype they
    // asked for when the probe says yes, AND don't accidentally
    // wander into a neighbouring enum value.
    CHECK(resolve_kv_attn_type(2, /*legacy=*/false, /*sf16=*/true,
                               /*sbf16=*/true, /*sq8=*/true) == kv_attn_dtype::bf16);
    CHECK(resolve_kv_attn_type(3, /*legacy=*/false, /*sf16=*/true,
                               /*sbf16=*/true, /*sq8=*/true) == kv_attn_dtype::q8_0);
    CHECK(resolve_kv_attn_type(1, /*legacy=*/false, /*sf16=*/true,
                               /*sbf16=*/false, /*sq8=*/false) == kv_attn_dtype::f16);
    // Cross-dtype non-contamination: requesting bf16 with f16 +
    // q8_0 supported but bf16 NOT supported MUST fall to f32,
    // not silently to f16 or q8_0.
    CHECK(resolve_kv_attn_type(2, /*legacy=*/true, /*sf16=*/true,
                               /*sbf16=*/false, /*sq8=*/true) == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(3, /*legacy=*/true, /*sf16=*/true,
                               /*sbf16=*/true, /*sq8=*/false) == kv_attn_dtype::f32);
}

// Test 8 — `out_was_downgraded` signal on explicit-request +
// missing-probe paths.
//
// PR #18 reviewer (Omar) follow-up: the resolver silently
// returns f32 when the operator explicitly requests f16/bf16/q8_0
// and the corresponding backend probe is false.  The operator-
// facing call sites need a programmatic signal so they can emit
// a `fprintf(stderr, "warning: ...")` (auto + missing probe is
// NOT a downgrade — the operator didn't ask for a specific
// dtype).  This test pins:
//   - Auto + missing probe → flag stays false.
//   - Auto + matching probe → flag stays false.
//   - f32 explicit → flag stays false (no concept of "downgrade
//     from f32").
//   - f16 / bf16 / q8_0 explicit + matching probe → flag stays
//     false (operator got what they asked for).
//   - f16 / bf16 / q8_0 explicit + missing probe → flag set.
//   - Optional out-pointer: nullptr (default) MUST be safe.
void test_downgrade_flag_signal() {
    bool downgraded = true;  // pre-set to true to detect "no write"

    // Auto + nothing supported.  Not a downgrade — auto policy.
    (void) resolve_kv_attn_type(-1, /*legacy=*/true,
                                false, false, false, &downgraded);
    CHECK(downgraded == false);

    // f32 explicit.  Never a downgrade.
    downgraded = true;
    (void) resolve_kv_attn_type(0, /*legacy=*/false,
                                true, true, true, &downgraded);
    CHECK(downgraded == false);

    // f16 explicit + supported.  Not a downgrade.
    downgraded = true;
    (void) resolve_kv_attn_type(1, /*legacy=*/false,
                                /*sf16=*/true, false, false, &downgraded);
    CHECK(downgraded == false);

    // bf16 explicit + supported.  Not a downgrade.
    downgraded = true;
    (void) resolve_kv_attn_type(2, /*legacy=*/false,
                                false, /*sbf16=*/true, false, &downgraded);
    CHECK(downgraded == false);

    // q8_0 explicit + supported.  Not a downgrade.
    downgraded = true;
    (void) resolve_kv_attn_type(3, /*legacy=*/false,
                                false, false, /*sq8=*/true, &downgraded);
    CHECK(downgraded == false);

    // f16 explicit + NOT supported.  Downgrade signal.
    downgraded = false;
    CHECK(resolve_kv_attn_type(1, /*legacy=*/false,
                               /*sf16=*/false, true, true, &downgraded)
          == kv_attn_dtype::f32);
    CHECK(downgraded == true);

    // bf16 explicit + NOT supported.  Downgrade signal.
    downgraded = false;
    CHECK(resolve_kv_attn_type(2, /*legacy=*/false,
                               true, /*sbf16=*/false, true, &downgraded)
          == kv_attn_dtype::f32);
    CHECK(downgraded == true);

    // q8_0 explicit + NOT supported.  Downgrade signal.
    downgraded = false;
    CHECK(resolve_kv_attn_type(3, /*legacy=*/false,
                               true, true, /*sq8=*/false, &downgraded)
          == kv_attn_dtype::f32);
    CHECK(downgraded == true);

    // Nullptr default argument must not crash on the same paths.
    CHECK(resolve_kv_attn_type(2, /*legacy=*/false, true, false, true)
          == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(3, /*legacy=*/false, true, true, false)
          == kv_attn_dtype::f32);
    CHECK(resolve_kv_attn_type(2, /*legacy=*/false, true, true, false)
          == kv_attn_dtype::bf16);
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
    test_downgrade_flag_signal();

    std::fprintf(stderr,
                 "test_supertonic_kv_attn_type: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
