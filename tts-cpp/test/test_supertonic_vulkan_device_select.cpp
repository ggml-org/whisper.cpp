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
// Helper contract:
//
//   int resolve_vulkan_device_index(int requested,
//                                   const std::vector<size_t> & free_vram_per_device);
//
// Returns the device index to use, or throws `std::runtime_error`
// on invalid input (caller surfaces the message verbatim, same
// pattern as the existing `--vulkan-device N out of range` error
// in `init_supertonic_backend`).
//
// Behaviour matrix:
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
// This test is written FIRST (TDD).  Every CHECK below MUST fail
// before the helper is implemented, and MUST pass after.

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

    std::fprintf(stderr,
                 "test_supertonic_vulkan_device_select: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
