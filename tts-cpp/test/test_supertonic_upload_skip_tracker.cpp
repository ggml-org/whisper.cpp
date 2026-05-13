// QVAC-18605 round 10 — CPU-only TDD test for the pointer-compare
// upload-skip tracker.
//
// Background
// ----------
// Per-step uploads of `text_emb` to the front-block cache and to
// the 3 group-graph caches happen 5 times per synth (once per
// denoise step), but `text_emb` is a `std::vector<float>` allocated
// ONCE in `Engine::Impl::synthesize()` (and once per bench run)
// — so the SAME pointer flows through 4 caches × 5 steps = 20
// uploads / synth, of which 16 are redundant re-uploads of
// identical data.
//
// The F4 pattern (already in `vector_res_style_qkv_cache` for
// `style_v_in` / `kctx_in`) skips redundant uploads via pointer
// comparison: if the host vector pointer is the same as the last
// successful upload's pointer, skip.  Round 10 generalises that
// pattern into a `upload_skip_tracker` struct so the same logic
// applies to the front-block / g1 / g2 / g3 `text_in` uploads.
//
// CROSS-SYNTH HAZARD
// ------------------
// `text_emb` lives on `Engine::Impl::synthesize()`'s stack (or
// the bench loop's stack) — destructed at end of call.  Modern
// heap allocators (jemalloc / tcmalloc / glibc) return the SAME
// address for an immediately-following same-size allocation
// (size-class reuse, locality optimisation), so synth N+1 may
// have `text_emb.data() == synth_N.text_emb.data()` despite
// holding completely different data.  A naive pointer-compare
// upload-skip would silently send stale text-encoder embeddings
// to the next synth.
//
// MITIGATION
// ----------
// Caller resets the tracker at every synth boundary (i.e., when
// `current_step == 0`).  The first step of every synth always
// uploads (cold-miss), populating the tracker; steps 1..N-1 hit
// the pointer-compare and skip.  Across synths, the reset
// invalidates the cached pointer so the next synth's upload
// always fires regardless of pointer match.
//
// API contract:
//
//   struct upload_skip_tracker {
//       const void * last_uploaded = nullptr;
//
//       // True iff `current` differs from the last recorded
//       // pointer (i.e., we MUST upload).  False iff we can
//       // skip.  After the consumer's upload call returns,
//       // they MUST call `mark_uploaded(current)` to update
//       // the cached pointer (else the next call re-uploads).
//       bool needs_upload(const void * current) const;
//
//       // Records a successful upload.  Call AFTER the upload
//       // completes (so a failed upload doesn't pin the
//       // pointer — the next call would correctly re-attempt).
//       void mark_uploaded(const void * current);
//
//       // Drops the cached pointer.  Caller invokes at synth
//       // boundary (current_step == 0) AND on cache rebuild
//       // (the underlying GPU buffer is reallocated, so the
//       // pointer-compare optimisation is invalid even if the
//       // host pointer matches).
//       void reset();
//   };
//
// Whole TU MUST fail to compile before the symbol is added,
// then pass after.

#include "supertonic_internal.h"

#include <cstdio>
#include <type_traits>

using tts_cpp::supertonic::detail::upload_skip_tracker;

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

// SFINAE: assert the public field exists at the documented type.
template <typename T>
auto has_last_uploaded(int) -> decltype(
    std::declval<T &>().last_uploaded, std::true_type{});
template <typename T>
auto has_last_uploaded(...) -> std::false_type;

// Test 1 — default state.  A fresh tracker has no cached pointer
// → needs_upload(...) ALWAYS returns true.  Catches the bug
// where a default-constructed tracker accidentally caches a
// non-null pointer (would silently skip the cold-miss upload).
void test_default_state() {
    static_assert(decltype(has_last_uploaded<upload_skip_tracker>(0))::value,
                  "upload_skip_tracker must expose last_uploaded "
                  "(documented field used by tests + diagnostics)");
    upload_skip_tracker t;
    CHECK(t.last_uploaded == nullptr);

    // Any pointer (including nullptr) needs upload on a fresh
    // tracker.  nullptr-vs-nullptr is technically equal but the
    // semantic is "we have NEVER uploaded" — needs_upload should
    // still return true.  The cleanest check: ensure
    // needs_upload(actual_pointer) is true.
    int dummy = 42;
    const void * p = &dummy;
    CHECK(t.needs_upload(p));

    // Same call twice should NOT mutate state — needs_upload is const.
    CHECK(t.needs_upload(p));
    CHECK(t.last_uploaded == nullptr);
}

// Test 2 — upload + skip happy path.
//
// The canonical 5-step pattern: step 0 uploads, steps 1-4 skip.
void test_upload_then_skip() {
    upload_skip_tracker t;
    int payload_a = 0;
    const void * p_a = &payload_a;

    // Step 0 — cold miss, must upload.
    CHECK(t.needs_upload(p_a));
    t.mark_uploaded(p_a);
    CHECK(t.last_uploaded == p_a);

    // Steps 1..4 — same pointer, skip.
    for (int i = 1; i < 5; ++i) {
        CHECK(!t.needs_upload(p_a));
    }
}

// Test 3 — pointer change forces upload.
//
// If the consumer calls with a different pointer, the tracker
// must indicate upload-needed.  Catches the bug where the
// tracker only checks the FIRST byte or some hash collision
// silently misses a real data change.
void test_pointer_change_triggers_upload() {
    upload_skip_tracker t;
    int payload_a = 0;
    int payload_b = 1;
    const void * p_a = &payload_a;
    const void * p_b = &payload_b;

    CHECK(t.needs_upload(p_a));
    t.mark_uploaded(p_a);
    CHECK(!t.needs_upload(p_a));

    // Different pointer — must upload.
    CHECK(t.needs_upload(p_b));
    t.mark_uploaded(p_b);
    CHECK(!t.needs_upload(p_b));

    // Switching back to p_a — also must upload (the cache only
    // remembers the LAST pointer, not all previously-seen ones).
    CHECK(t.needs_upload(p_a));
}

// Test 4 — reset() clears the cached pointer.
//
// This is the SYNTH-BOUNDARY GUARD.  The caller invokes
// reset() at the start of each synth (current_step == 0) so
// even if the new synth's text_emb happens to share the same
// stack address as the previous synth's text_emb, the tracker
// forces a re-upload (because the data may differ — modern
// allocators re-issue addresses on size-class reuse).
void test_reset_invalidates_cache() {
    upload_skip_tracker t;
    int payload = 0;
    const void * p = &payload;

    // Upload + verify skip.
    CHECK(t.needs_upload(p));
    t.mark_uploaded(p);
    CHECK(!t.needs_upload(p));

    // Reset — same pointer must now trigger upload again.
    t.reset();
    CHECK(t.last_uploaded == nullptr);
    CHECK(t.needs_upload(p));
}

// Test 5 — interleaved sites.
//
// Multiple trackers (one per cache) are independent — no shared
// state.  Catches the bug where the tracker accidentally uses
// a static / thread_local member that all instances share.
void test_independent_instances() {
    upload_skip_tracker t1;
    upload_skip_tracker t2;
    upload_skip_tracker t3;
    int payload_a = 0;
    int payload_b = 1;
    const void * p_a = &payload_a;
    const void * p_b = &payload_b;

    t1.mark_uploaded(p_a);
    t2.mark_uploaded(p_b);
    // t3 left untouched.

    CHECK(!t1.needs_upload(p_a));
    CHECK(t1.needs_upload(p_b));

    CHECK(!t2.needs_upload(p_b));
    CHECK(t2.needs_upload(p_a));

    CHECK(t3.needs_upload(p_a));
    CHECK(t3.needs_upload(p_b));
    CHECK(t3.last_uploaded == nullptr);
}

// Test 6 — cross-synth pointer-reuse hazard simulation.
//
// Simulate the production pattern: synth A allocates text_emb at
// address P, runs 5 steps (upload at step 0, skip at steps 1-4).
// Synth A ends, vector destructs.  Synth B allocates text_emb at
// the SAME address P (allocator size-class reuse) but with
// DIFFERENT data.
//
// Without reset() at synth boundary: the tracker would skip
// synth B's step-0 upload because pointer matches → BUG.
//
// With reset() at synth boundary (the documented contract): the
// tracker correctly forces synth B's step-0 upload.
void test_cross_synth_pointer_reuse() {
    upload_skip_tracker t;

    // Synth A: address P_A.
    char buf_a[64] = {0};
    const void * p_a = buf_a;
    CHECK(t.needs_upload(p_a));  // step 0 (cold miss)
    t.mark_uploaded(p_a);
    for (int s = 1; s < 5; ++s) {
        CHECK(!t.needs_upload(p_a));
    }

    // Synth B: SAME address (synth-A's buffer "freed" + reused).
    // Without reset, naive pointer-compare would incorrectly
    // skip the upload → upload-skip would silently leak synth-A
    // data into synth-B's GPU buffer.
    //
    // The documented contract is: caller MUST reset() at
    // current_step == 0.  We simulate that here.
    t.reset();
    const void * p_b = buf_a;        // intentionally same address.
    CHECK(t.needs_upload(p_b));      // upload fires despite matching pointer.
    t.mark_uploaded(p_b);
    for (int s = 1; s < 5; ++s) {
        CHECK(!t.needs_upload(p_b));
    }
}

// Test 7 — reset on already-empty tracker is a no-op.
//
// Defensive: caller might call reset() unconditionally at synth
// start without checking whether the tracker has cached state.
// Must not crash / mutate other state weirdly.
void test_reset_on_empty_tracker() {
    upload_skip_tracker t;
    CHECK(t.last_uploaded == nullptr);
    t.reset();
    CHECK(t.last_uploaded == nullptr);
    t.reset();
    t.reset();
    CHECK(t.last_uploaded == nullptr);

    // After reset chain, normal usage still works.
    int payload = 0;
    const void * p = &payload;
    CHECK(t.needs_upload(p));
    t.mark_uploaded(p);
    CHECK(!t.needs_upload(p));
}

} // namespace

int main() {
    test_default_state();
    test_upload_then_skip();
    test_pointer_change_triggers_upload();
    test_reset_invalidates_cache();
    test_independent_instances();
    test_cross_synth_pointer_reuse();
    test_reset_on_empty_tracker();

    std::fprintf(stderr,
                 "test_supertonic_upload_skip_tracker: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
