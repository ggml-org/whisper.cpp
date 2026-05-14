// QVAC-18605 round 3 — CPU-only TDD test for the multi-device
// Vulkan auto-pick helper.
//
// `--vulkan-device -1` was reserved for "auto-pick best device"
// behaviour in the QVAC-18605 bring-up but treated as 0 (the
// historical hard-coded value).  Round 3 wires the auto-pick
// logic via a pure-logic helper that takes the per-device free-
// VRAM list as input — keeps the policy decoupled from the
// Vulkan-only `ggml_backend_vk_get_device_memory()` plumbing,
// which means the policy is testable on CPU with synthetic
// inputs.  The Vulkan-side wrapper that calls
// `ggml_backend_vk_get_device_memory()` for each device and
// dispatches into the helper lives behind `#ifdef GGML_USE_VULKAN`
// in `init_supertonic_backend`.
//
// QVAC-18605 round 12 — extend the policy to bias against UMA
// (unified-memory-architecture, i.e., integrated) GPUs when a
// discrete GPU is present.  Background: on the dev rig (RTX 5090
// discrete + AMD RADV iGPU), the iGPU reports system RAM (128+
// GB) as "free VRAM" via `ggml_backend_vk_get_device_memory()`
// because UMA shares the host RAM pool with the CPU.  The
// round-3 `argmax(free_vram)` policy therefore picked the iGPU,
// silently delivering ~7× realtime instead of the discrete's
// 273× realtime — a ~40× perf regression for any operator who
// followed the help text "auto-pick adapter with most free VRAM".
//
// New signature (round 12):
//
//   int resolve_vulkan_device_index(int requested,
//                                   const std::vector<size_t> & free_vram_per_device,
//                                   const std::vector<bool>   & is_uma_per_device = {});
//
// `is_uma_per_device` is OPTIONAL (default empty vector).  When
// empty, the round-3 `argmax(free_vram)` policy is preserved
// verbatim — backwards-compatible with every caller that hasn't
// been updated.  When non-empty, it MUST have the same length as
// `free_vram_per_device`; mismatch throws.
//
// New behaviour matrix (with `is_uma_per_device` populated):
//
//   | requested | discrete? | uma?  | result                                |
//   |-----------|-----------|-------|---------------------------------------|
//   | -1        | all       | none  | argmax(free_vram) over all            |
//   | -1        | none      | all   | argmax(free_vram) over all            |
//   | -1        | mixed     | mixed | argmax(free_vram) over DISCRETE only  |
//   | 0..N      | any       | any   | explicit passthrough (range-checked)  |
//
// Returns the device index to use, or throws `std::runtime_error`
// on invalid input (caller surfaces the message verbatim).
//
// Original round-3 behaviour matrix (when `is_uma_per_device` is empty):
//
//   | requested | dev_count | result                                  |
//   |-----------|-----------|-----------------------------------------|
//   | -1        | 0         | throws (no device to pick)              |
//   | 0         | 0         | throws (no device to pick)              |
//   | -1        | 1         | 0  (only choice)                        |
//   | 0         | 1         | 0                                       |
//   | -1        | 2         | argmax(free_vram); ties → first         |
//   | 0         | 2         | 0  (explicit override)                  |
//   | 1         | 2         | 1                                       |
//   | 2         | 2         | throws (out of range)                   |
//   | -2        | any       | throws (negative != -1 reserved)        |
//
// Tie-breaking on equal free VRAM picks the lower index — gives
// stable behaviour across runs on identical-spec multi-GPU
// machines.  Documented in `init_supertonic_backend` so operators
// who need a different policy can `--vulkan-device N` explicitly.
//
// This test is written FIRST (TDD).  Round 3 checks (tests 1-8)
// already pass; round 12 checks (tests 9-13) fail until the new
// `is_uma_per_device` parameter is implemented.

#include "supertonic_internal.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

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

// Helper: assert that `fn()` throws std::runtime_error.  Used to
// verify the no-device / out-of-range / negative-non-auto cases.
template <typename F>
bool throws_runtime_error(F && fn) {
    try {
        fn();
        return false;
    } catch (const std::runtime_error &) {
        return true;
    } catch (...) {
        return false;
    }
}

// Test 1 — Empty device list throws regardless of request.
//
// `init_supertonic_backend` falls through to OpenCL / CPU when
// `ggml_backend_vk_get_device_count()` returns 0; the helper
// throws here so the caller has a clear signal to skip the
// Vulkan branch instead of accidentally returning device index
// 0 against a zero-length list.
void test_empty_device_list_throws() {
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(-1, {});
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index( 0, {});
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index( 1, {});
    }));
}

// Test 2 — Single device, requested 0 or -1 returns 0.
//
// The auto-pick is a no-op when there's only one candidate.
// Explicit index 0 also returns 0 (the historical hard-coded
// path).  Any other index throws (out of range).
void test_single_device_returns_zero() {
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{100}) == 0);
    CHECK(resolve_vulkan_device_index( 0, std::vector<size_t>{100}) == 0);
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(1, std::vector<size_t>{100});
    }));
}

// Test 3 — Auto-pick (`-1`) picks the device with most free VRAM.
//
// Simulates a multi-GPU machine where one card has more head-
// room than the other (e.g. NVIDIA RTX 5090 with 32 GB free
// alongside an RTX 4090 with 16 GB free).  Auto-pick should
// land on the 5090.
void test_auto_pick_max_vram() {
    // dev0 = 100 free, dev1 = 500 free → pick dev1.
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{100, 500}) == 1);
    // dev0 = 500 free, dev1 = 100 free → pick dev0.
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{500, 100}) == 0);
    // 4 devices, dev2 has the most.
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{100, 200, 800, 400}) == 2);
}

// Test 4 — Tie-breaking picks the lower index.
//
// Identical-spec multi-GPU machines (lab racks of A100s, e.g.)
// produce identical free-VRAM readings; tie-breaking on the
// lower index gives stable per-run device assignment instead of
// depending on driver enumeration order.
void test_auto_pick_ties_pick_lower_index() {
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{300, 300}) == 0);
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{500, 500, 500}) == 0);
    // Tie at the back: dev1 + dev2 both have 500, pick dev1.
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{100, 500, 500}) == 1);
}

// Test 5 — Explicit valid index in range returns it.
//
// Auto-pick is opt-in via `-1`; an operator who knows their
// machine + workload can pin to a specific device with
// `--vulkan-device N`, and the helper must not second-guess the
// choice based on VRAM.  (Useful when the higher-VRAM card is
// reserved for another workload, e.g. a model-server alongside
// a TTS worker on the same box.)
void test_explicit_index_returns_unchanged() {
    CHECK(resolve_vulkan_device_index(0, std::vector<size_t>{100, 500}) == 0);
    CHECK(resolve_vulkan_device_index(1, std::vector<size_t>{100, 500}) == 1);
    CHECK(resolve_vulkan_device_index(2, std::vector<size_t>{100, 500, 200}) == 2);
    CHECK(resolve_vulkan_device_index(0, std::vector<size_t>{100, 500, 200}) == 0);
}

// Test 6 — Out-of-range explicit index throws.
//
// Same loud-failure contract as the existing
// `init_supertonic_backend` Vulkan branch: a CLI typo that asks
// for `--vulkan-device 7` on a 2-GPU machine surfaces here as a
// hard error, not a silent CPU fallback that hides the perf
// cliff.
void test_out_of_range_throws() {
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(2, std::vector<size_t>{100, 500});
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(7, std::vector<size_t>{100, 500});
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(99, std::vector<size_t>{100});
    }));
}

// Test 7 — Negative-but-not-(-1) throws.
//
// `-1` is the documented "auto-pick" sentinel; any other
// negative value (e.g. `-2`, `-100`) is reserved for future
// policies.  Treating those as 0 (the bring-up's behaviour)
// silently masks operator typos; throwing surfaces them.
void test_reserved_negative_throws() {
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(-2, std::vector<size_t>{100, 500});
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(-100, std::vector<size_t>{100, 500});
    }));
}

// Test 8 — Zero-VRAM device handling.
//
// A reserved-but-listed device (e.g. iGPU listed but not
// available for compute) shows 0 free VRAM.  Auto-pick should
// still work — picks any other device with non-zero VRAM.  When
// all devices have zero VRAM (degenerate), picks index 0
// (consistent with the tie-breaking rule).
void test_zero_vram_handling() {
    // dev0 has zero free, dev1 has 500.  Auto-pick → dev1.
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{0, 500}) == 1);
    // All zero — pick the first (consistent with the
    // tie-breaking rule).
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{0, 0, 0}) == 0);
}

// =============================================================
// Round 12 — bias against UMA on hybrid discrete+iGPU machines.
// =============================================================

// Test 9 — Empty `is_uma_per_device` preserves round-3 behaviour.
//
// Backwards-compatibility gate.  Every existing caller passes
// only two arguments; the new third-argument default of `{}`
// must produce identical results to the round-3 helper for
// EVERY input shape.  This is a "no surprise" guarantee for any
// caller that hasn't been updated to pass the UMA flags.
void test_empty_uma_preserves_round3_behaviour() {
    // Empty UMA list explicitly passed — identical to round-3
    // 2-arg call.  Covers the main argmax(free_vram) path.
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{100, 500},
                                       std::vector<bool>{}) == 1);
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{500, 100},
                                       std::vector<bool>{}) == 0);
    // Explicit index also unchanged with empty UMA list.
    CHECK(resolve_vulkan_device_index(1, std::vector<size_t>{100, 500},
                                       std::vector<bool>{}) == 1);
    // Tie-break still picks lower index with empty UMA list.
    CHECK(resolve_vulkan_device_index(-1, std::vector<size_t>{300, 300},
                                       std::vector<bool>{}) == 0);
}

// Test 10 — Hybrid discrete + UMA: auto-pick prefers discrete
// even when UMA reports more "free VRAM".
//
// THE BUG ROUND 12 FIXES.  On the dev rig (RTX 5090 discrete +
// AMD RADV iGPU), free_vram_per_device looks like
// `[32 GB, 120 GB]` because RADV reports the entire system RAM
// as available to the iGPU's UMA pool.  Pre-round-12 argmax
// picks index 1 (iGPU), losing ~40× realtime.  Round 12 biases
// against UMA when a discrete is present, picking index 0.
void test_hybrid_prefer_discrete_over_uma() {
    // RTX 5090 (discrete, 32 GB) + AMD RADV iGPU (UMA, ~120 GB
    // reported via system RAM).  Pre-round-12 returned 1 (iGPU);
    // round-12 returns 0 (discrete) regardless of the UMA's
    // larger reported free pool.
    CHECK(resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{32ull * 1024 * 1024 * 1024,
                                 120ull * 1024 * 1024 * 1024},
            std::vector<bool>{false, true}) == 0);
    // Swapped enumeration order (iGPU first, discrete second).
    // Same outcome — picks the discrete one regardless of index.
    CHECK(resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{120ull * 1024 * 1024 * 1024,
                                 32ull * 1024 * 1024 * 1024},
            std::vector<bool>{true, false}) == 1);
}

// Test 11 — Multi-discrete + multi-UMA mixed: argmax over the
// discrete subset.
//
// Lab rack with 2 discrete cards + a CPU-emulator (lavapipe,
// reports UMA=true) + an iGPU.  The auto-pick should ignore
// the UMA devices entirely and run argmax over the discrete
// subset.
void test_multi_discrete_argmax_over_discrete_subset() {
    // 4 devices: 2 discrete (16/32 GB), 2 UMA (120/120 GB).
    // Discrete-only argmax picks dev1 (32 GB > 16 GB).
    CHECK(resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{
                16ull * 1024 * 1024 * 1024,    // dev0: discrete, 16 GB
                32ull * 1024 * 1024 * 1024,    // dev1: discrete, 32 GB
                120ull * 1024 * 1024 * 1024,   // dev2: UMA, 120 GB
                120ull * 1024 * 1024 * 1024},  // dev3: UMA, 120 GB
            std::vector<bool>{false, false, true, true}) == 1);
    // Discrete subset tie-break: dev0 + dev2 both discrete with
    // 16 GB, dev1 is UMA.  Tie → lower index = 0.
    CHECK(resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{
                16ull * 1024 * 1024 * 1024,
                120ull * 1024 * 1024 * 1024,
                16ull * 1024 * 1024 * 1024},
            std::vector<bool>{false, true, false}) == 0);
}

// Test 12 — All-UMA falls back to argmax(free_vram).
//
// Mobile / laptop with only an iGPU available, or a CPU-only
// build using lavapipe.  No discrete present, so the bias
// degenerates to the round-3 policy.
void test_all_uma_falls_back_to_argmax() {
    // Two iGPUs (rare but possible on some multi-socket boards).
    // Falls back to argmax(free_vram).
    CHECK(resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{100, 500},
            std::vector<bool>{true, true}) == 1);
    // Single iGPU.
    CHECK(resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{500},
            std::vector<bool>{true}) == 0);
}

// Test 13 — Explicit index passthrough is UMA-agnostic.
//
// An operator who knows their machine + workload can still pin
// `--vulkan-device 1` even when device 1 is UMA.  The bias
// applies ONLY to the `-1` auto-pick path.  (Useful for testing
// the iGPU path or for low-thermal scenarios where the
// operator deliberately offloads to UMA.)
void test_explicit_index_ignores_uma_bias() {
    // Pinned to UMA index 1 — passthrough, no bias kicks in.
    CHECK(resolve_vulkan_device_index(
            1,
            std::vector<size_t>{32ull * 1024 * 1024 * 1024,
                                 120ull * 1024 * 1024 * 1024},
            std::vector<bool>{false, true}) == 1);
    // Pinned to discrete index 0 — passthrough.
    CHECK(resolve_vulkan_device_index(
            0,
            std::vector<size_t>{32ull * 1024 * 1024 * 1024,
                                 120ull * 1024 * 1024 * 1024},
            std::vector<bool>{false, true}) == 0);
}

// Test 14 — Mismatched UMA list length throws.
//
// Caller bug guard.  If the UMA list is non-empty AND its size
// doesn't match `free_vram_per_device`, throw rather than
// silently truncating or out-of-bounds-reading.  Either zero
// (use round-3 policy) or the full length (use round-12 policy)
// — anything else is a wiring bug in the caller.
void test_mismatched_uma_list_length_throws() {
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{100, 500},
            std::vector<bool>{false});  // 1 entry vs 2 devices
    }));
    CHECK(throws_runtime_error([] {
        (void) resolve_vulkan_device_index(
            -1,
            std::vector<size_t>{100, 500},
            std::vector<bool>{false, true, false});  // 3 vs 2
    }));
}

} // namespace

int main() {
    test_empty_device_list_throws();
    test_single_device_returns_zero();
    test_auto_pick_max_vram();
    test_auto_pick_ties_pick_lower_index();
    test_explicit_index_returns_unchanged();
    test_out_of_range_throws();
    test_reserved_negative_throws();
    test_zero_vram_handling();
    // Round 12 — UMA bias.
    test_empty_uma_preserves_round3_behaviour();
    test_hybrid_prefer_discrete_over_uma();
    test_multi_discrete_argmax_over_discrete_subset();
    test_all_uma_falls_back_to_argmax();
    test_explicit_index_ignores_uma_bias();
    test_mismatched_uma_list_length_throws();

    std::fprintf(stderr,
                 "test_supertonic_vulkan_device_select: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
