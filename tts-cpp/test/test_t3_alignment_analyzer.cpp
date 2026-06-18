// Unit tests for the host-side T3 alignment analyzer (Phase 2).
// Pure logic, no model/ggml dependency:
//
//   g++ -std=c++17 -I src test/test_t3_alignment_analyzer.cpp src/t3_alignment_analyzer.cpp -o /tmp/t && /tmp/t
//
// Coverage:
//   - mid-text frames never force EOS (climb that never completes)
//   - a healthy generation that reaches the end then dwells forces EOS a few
//     frames after completion (long_tail)
//   - a ramble that backtracks after completion forces EOS (alignment_repetition)
//   - token repetition only forces EOS once complete (avoids #519/#587 early cut)
//   - short text / empty row / disabled => no force

#include "t3_alignment_analyzer.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tts_cpp::chatterbox::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, ...) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
        ++g_failures;                                                    \
        fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);            \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                          \
    }                                                                    \
} while (0)

// One-hot-ish alignment row of length S peaked at column `peak`.
std::vector<float> row_at(int S, int peak) {
    std::vector<float> r((size_t) S, 0.0f);
    if (peak >= 0 && peak < S) r[(size_t) peak] = 1.0f;
    return r;
}

t3_align_analyzer_params params(int S) {
    t3_align_analyzer_params p;
    p.text_len        = S;
    p.complete_margin = 3;
    p.tail_cols       = 3;
    p.long_tail_thresh = 5.0f;
    p.rep_back_cols   = 5;
    p.rep_thresh      = 5.0f;
    p.min_text_len    = 6;
    p.enabled         = true;
    return p;
}

void test_midtext_never_forces() {
    const int S = 13;
    t3_alignment_analyzer az;
    az.reset(params(S));
    bool forced = false;
    // Climb only to position 8 (never reaches S-3 == 10), 40 frames.
    for (int f = 0; f < 40; ++f) {
        int peak = f < 8 ? f : 8;
        if (az.step(row_at(S, peak), /*token=*/100 + f) == t3_align_action::force_eos)
            forced = true;
    }
    CHECK(!forced, "mid-text climb must never force EOS");
    CHECK(!az.complete(), "must not be marked complete");
}

void test_healthy_long_tail_forces_after_completion() {
    const int S = 13;
    t3_alignment_analyzer az;
    az.reset(params(S));
    int force_frame = -1;
    int complete_frame = -1;
    for (int f = 0; f < 40; ++f) {
        // climb to 12 by frame 12, then dwell at 12
        int peak = f < 12 ? f : 12;
        t3_align_action a = az.step(row_at(S, peak), /*token=*/200 + (f % 7));
        if (complete_frame < 0 && az.complete()) complete_frame = f;
        if (force_frame < 0 && a == t3_align_action::force_eos) force_frame = f;
    }
    CHECK(complete_frame >= 0, "should reach completion");
    CHECK(force_frame >= 0, "long-tail dwell should force EOS");
    CHECK(force_frame > complete_frame, "must not force before completion (got force=%d complete=%d)",
          force_frame, complete_frame);
    // ~long_tail_thresh frames of dwelling after completion.
    CHECK(force_frame - complete_frame <= 8, "force should fire within a few frames of completion (got %d)",
          force_frame - complete_frame);
}

void test_ramble_backtrack_forces() {
    const int S = 13;
    t3_alignment_analyzer az;
    az.reset(params(S));
    // Climb to 12, complete, then backtrack to early positions (repetition).
    for (int f = 0; f <= 12; ++f) az.step(row_at(S, f), 300 + f);
    CHECK(az.complete(), "should be complete after reaching the end");
    bool forced = false;
    // Backtrack: attend to early columns (0..3) after completion -> repetition.
    for (int f = 0; f < 12 && !forced; ++f) {
        if (az.step(row_at(S, f % 4), 400 + f) == t3_align_action::force_eos) forced = true;
    }
    CHECK(forced, "post-completion backtracking should force EOS (alignment_repetition)");
}

void test_token_repetition_gated_by_complete() {
    const int S = 13;
    // Mid-text repeated tokens should NOT force EOS.
    t3_alignment_analyzer az;
    az.reset(params(S));
    bool forced = false;
    for (int f = 0; f < 10; ++f) {
        // stay mid-text (peak 5) with the same token repeated
        if (az.step(row_at(S, 5), /*token=*/777) == t3_align_action::force_eos) forced = true;
    }
    CHECK(!forced, "repeated token mid-text must NOT force EOS (avoids #587 early cut)");
}

void test_small_input_not_clipped() {
    // A very small input (smallest text length the analyzer acts on, e.g. "Hi."
    // => SOT + graphemes + EOT) must never be cut before its alignment reaches
    // the end of the text — i.e. force-EOS must not fire before `complete`.
    // That is the structural no-clipping guarantee (verified end-to-end on the
    // model: align-on speech-end == natural speech-end for "Hi.", "Yes.", ...).
    for (int S = 6; S <= 8; ++S) {
        t3_alignment_analyzer az;
        az.reset(params(S));
        int force_frame = -1, complete_frame = -1;
        for (int f = 0; f < 30; ++f) {
            const int peak = f < S - 1 ? f : S - 1;   // climb to the last col, then dwell
            t3_align_action a = az.step(row_at(S, peak), /*token=*/500 + (f % 5));
            if (complete_frame < 0 && az.complete()) complete_frame = f;
            if (force_frame < 0 && a == t3_align_action::force_eos) force_frame = f;
        }
        CHECK(complete_frame >= 0, "S=%d: short input should reach completion", S);
        CHECK(force_frame >= 0, "S=%d: short input should eventually stop", S);
        CHECK(force_frame > complete_frame,
              "S=%d: must NOT force before completion (no clip): force=%d complete=%d",
              S, force_frame, complete_frame);
    }
}

void test_suppress_eos_midtext() {
    const int S = 13;
    // Default (suppress enabled): while still mid-text (alignment far from the
    // end) the analyzer asks the caller to hold off on stopping -> anti early
    // truncation.
    {
        t3_alignment_analyzer az;
        az.reset(params(S));
        t3_align_action a = t3_align_action::none;
        for (int f = 0; f < 5; ++f) a = az.step(row_at(S, f), 600 + f);  // cur = 0..4 < S-3
        CHECK(a == t3_align_action::suppress_eos,
              "mid-text should request EOS suppression (anti-truncation)");
        CHECK(!az.complete(), "must not be complete mid-text");
    }
    // suppress_eos_enabled = false: analyzer never suppresses (only forces).
    {
        t3_align_analyzer_params p = params(S);
        p.suppress_eos_enabled = false;
        t3_alignment_analyzer az;
        az.reset(p);
        t3_align_action a = t3_align_action::none;
        for (int f = 0; f < 5; ++f) a = az.step(row_at(S, f), 600 + f);
        CHECK(a == t3_align_action::none, "suppress disabled -> none mid-text");
    }
}

void test_short_text_and_empty_row_noop() {
    // Short text (< min_text_len) -> disabled.
    t3_alignment_analyzer az;
    az.reset(params(4));
    bool forced = false;
    for (int f = 0; f < 20; ++f)
        if (az.step(row_at(4, 1), 1) == t3_align_action::force_eos) forced = true;
    CHECK(!forced, "short text must not force EOS");

    // Empty row -> always none.
    t3_alignment_analyzer az2;
    az2.reset(params(13));
    CHECK(az2.step(std::vector<float>{}, 5) == t3_align_action::none,
          "empty alignment row must be a no-op");
}

void test_params_for_language() {
    // Defaults (English-validated) for a known and an unknown language.
    for (const char * lang : {"en", "xx"}) {
        t3_align_analyzer_params p = t3_align_params_for_language(lang, 20);
        CHECK(p.text_len == 20, "%s: text_len propagated", lang);
        CHECK(p.complete_margin == 3, "%s: default complete_margin", lang);
        CHECK(p.long_tail_thresh == 5.0f, "%s: default long_tail", lang);
        CHECK(p.rep_thresh == 5.0f, "%s: default rep_thresh", lang);
        CHECK(p.min_text_len == 6, "%s: default min_text_len", lang);
        CHECK(p.suppress_eos_enabled, "%s: suppress on by default", lang);
    }

    // Environment overrides (on-device tuning).
    setenv("CHATTERBOX_ALIGN_COMPLETE_MARGIN", "2", 1);
    setenv("CHATTERBOX_ALIGN_LONG_TAIL", "9.5", 1);
    setenv("CHATTERBOX_ALIGN_MIN_TEXT", "10", 1);
    setenv("CHATTERBOX_ALIGN_SUPPRESS", "0", 1);
    {
        t3_align_analyzer_params p = t3_align_params_for_language("en", 20);
        CHECK(p.complete_margin == 2, "env complete_margin override");
        CHECK(p.long_tail_thresh == 9.5f, "env long_tail override");
        CHECK(p.min_text_len == 10, "env min_text override");
        CHECK(!p.suppress_eos_enabled, "env suppress disable");
    }
    unsetenv("CHATTERBOX_ALIGN_COMPLETE_MARGIN");
    unsetenv("CHATTERBOX_ALIGN_LONG_TAIL");
    unsetenv("CHATTERBOX_ALIGN_MIN_TEXT");
    unsetenv("CHATTERBOX_ALIGN_SUPPRESS");
}

void test_valid_heads() {
    // Full MTL geometry: all 3 reference heads valid.
    CHECK(t3_align_valid_heads(30, 16).size() == 3, "full model -> 3 aligned heads");
    // 13 layers: the layer-13 head (13,11) drops out.
    CHECK(t3_align_valid_heads(13, 16).size() == 2, "13 layers -> 2 heads");
    // Too few layers for any reference head -> empty -> alignment disabled.
    CHECK(t3_align_valid_heads(9, 16).empty(), "9 layers -> no aligned heads");
    // Fewer heads: only (9,2) survives a head<11 model.
    CHECK(t3_align_valid_heads(30, 11).size() == 1, "11 heads -> only (9,2)");
    // Degenerate geometry.
    CHECK(t3_align_valid_heads(0, 0).empty(), "0x0 -> empty");
}

} // namespace

int main() {
    test_midtext_never_forces();
    test_healthy_long_tail_forces_after_completion();
    test_ramble_backtrack_forces();
    test_token_repetition_gated_by_complete();
    test_small_input_not_clipped();
    test_suppress_eos_midtext();
    test_params_for_language();
    test_valid_heads();
    test_short_text_and_empty_row_noop();

    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL",
            g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
