// Model-free unit tests for the long-form encoder windowing arithmetic
// (src/long_form.h): the window planner, the window-size resolver, the seam
// trim, and the committed-frame append. No GGUF or ggml is needed, so these run
// on every checkout and are the primary regression gate for the bounded-memory
// long-audio fix.
//
// They assert the properties the fix depends on:
//   1. every window fed to the encoder is length-bounded (<= center + 2*ctx),
//      independent of total input length -- so peak encoder memory is bounded;
//   2. the committed centres are gap-free, non-overlapping and cover exactly
//      [0, n_units) -- so no audio is dropped or transcribed twice;
//   3. the resolved per-window frame ceiling never exceeds the encoder's trained
//      positional range (pos_emb_max_len), which is a hard ceiling over the
//      minimum-window floor;
//   4. trim + append stitch the per-window encoder outputs back into one
//      contiguous frame sequence with no gaps or duplicates at the seams;
//   5. the append refuses (rather than silently truncating) an out-of-range
//      request, so a broken invariant fails loud instead of reading past a buffer;
//   6. every window of an input longer than one window has the same length, so
//      the encoder sees a single input shape (one compute graph);
//   7. the CLI's --window-sec / --overlap conversions reject windows below 256
//      encoder frames or above pos_emb_max_len, and 0 % overlap reaches the
//      engine as -1 (no context), never 0 (auto).
//
// Exit 0 on success; non-zero (with a FAIL line per broken invariant) otherwise.

#include "long_form.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using parakeet::LongFormWindow;
using parakeet::WindowTrim;
using parakeet::append_committed_frames;
using parakeet::compute_window_trim;
using parakeet::long_form_context_frames_from_overlap;
using parakeet::long_form_window_frames_from_seconds;
using parakeet::plan_long_form_windows;
using parakeet::plan_long_form_windows_asymmetric;
using parakeet::resolve_coreml_exact_shape_plan;
using parakeet::resolve_coreml_fixed_shape_plan;
using parakeet::resolve_long_form_window_frames;
using parakeet::should_use_nemotron_cache_aware_offline;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

// Every invariant the planner must uphold for a valid (n, center, ctx) triple.
void check_plan(int n, int center, int ctx) {
    const std::string tag = "plan n=" + std::to_string(n) +
                            " center=" + std::to_string(center) +
                            " ctx=" + std::to_string(ctx);
    const std::vector<LongFormWindow> ws = plan_long_form_windows(n, center, ctx);

    expect(!ws.empty(), tag + ": expected at least one window");
    if (ws.empty()) return;

    expect(ws.front().center_start == 0, tag + ": first centre must start at 0");
    expect(ws.back().center_end == n, tag + ": last centre must end at n_units");

    long long covered = 0;
    for (size_t i = 0; i < ws.size(); ++i) {
        const LongFormWindow & w = ws[i];
        const std::string wtag = tag + " window[" + std::to_string(i) + "]";

        expect(w.center_start < w.center_end, wtag + ": empty committed centre");
        expect(w.window_start >= 0, wtag + ": window_start < 0");
        expect(w.window_start + w.window_len <= n, wtag + ": window runs past n_units");
        expect(w.window_start <= w.center_start, wtag + ": window_start after centre");
        expect(w.window_start + w.window_len >= w.center_end,
               wtag + ": window ends before centre");
        expect(w.window_len <= center + 2 * ctx, wtag + ": window length exceeds bound");
        if (n >= center + 2 * ctx) {
            expect(w.window_len == center + 2 * ctx,
                   wtag + ": boundary window did not reuse its full context budget");
        }
        expect(w.is_final == (i + 1 == ws.size()), wtag + ": is_final mismatch");
        if (i + 1 < ws.size()) {
            expect(w.center_end == ws[i + 1].center_start,
                   wtag + ": centres not contiguous (gap or overlap at seam)");
        }
        covered += (long long) (w.center_end - w.center_start);
    }
    expect(covered == n, tag + ": committed centres do not cover [0, n) exactly");
}

// resolve_long_form_window_frames() maps a requested window (0 = auto) plus the
// model's pos_emb_max_len onto the effective per-window frame ceiling. The
// load-bearing property is that pos_emb_max_len is a HARD ceiling that wins over
// the minimum-window floor: no window may exceed the encoder's trained
// positional range, even when that range is smaller than the floor.
// The automatic window shares the input equally among the fewest windows
// that fit; a requested window keeps its length.
void check_balanced_automatic_plan() {
    using namespace parakeet;
    constexpr int sub = 8, ctx = 256, center = 3750 - 2 * ctx;

    expect(balanced_center_frames(5159, center) == 2580, "balanced: 5159 frames -> two centres of 2580");
    expect(balanced_center_frames(7739, center) == 2580, "balanced: 7739 frames -> three centres of 2580");
    expect(balanced_center_frames(3751, center) == 1876, "balanced: just past one centre -> two halves");
    expect(balanced_center_frames(3238, center) == center, "balanced: one centre stays as it is");
    expect(balanced_center_frames(323800, center) == center, "balanced: an exact multiple keeps the full centre");
    expect(balanced_center_frames(100, 0) == 0, "balanced: a degenerate centre is returned unchanged");

    const LongFormPlan automatic = resolve_long_form_plan_frames(0, 0, 5000, sub, 41272);
    expect(automatic.enabled && automatic.center_frames == 2580 &&
           automatic.window_frames == 2580 + 2 * ctx && automatic.context_frames == ctx,
           "automatic plan over 412.7 s: two balanced 247 s windows with full context");

    const LongFormPlan single = resolve_long_form_plan_frames(0, 0, 5000, sub, 3750 * sub);
    expect(!single.enabled, "an input that fits one window keeps the single pass");

    const LongFormPlan requested = resolve_long_form_plan_frames(750, 94, 5000, sub, 41272);
    expect(requested.enabled && requested.window_frames == 750 && requested.center_frames == 750 - 2 * 94,
           "a requested window is honoured as given");

    const LongFormPlan long_input = resolve_long_form_plan_frames(0, 0, 5000, sub, 90 * 60 * 100);
    expect(long_input.enabled && long_input.center_frames <= center &&
           long_input.center_frames > center - center / 100,
           "a long input keeps nearly full centres");
}

// A model's recommended window replaces the automatic one, and only that one.
void check_long_form_hint() {
    using namespace parakeet;
    int window = -1, context = -1;

    expect(long_form_hint_for_model(0, 0, "TranscribePsy-250M-EU-sub, FastConformer-TDT", window, context) &&
           window == kTranscribePsyWindowFrames && context == kTranscribePsyContextFrames,
           "hint: a TranscribePsy description gives its validated 750 / 94 frames");
    expect(long_form_hint_for_model(1000, 100, "TranscribePsy-250M-EU-sub", window, context) &&
           window == 1000 && context == 100,
           "hint: explicit GGUF keys win over the description");
    expect(long_form_hint_for_model(1000, 0, "", window, context) && window == 1000 && context == 0,
           "hint: a window key without a context key leaves the context automatic");
    expect(!long_form_hint_for_model(0, 0, "NVIDIA Parakeet-TDT FastConformer ASR", window, context) &&
           window == 0 && context == 0,
           "hint: other models recommend nothing");

    int w = 0, c = 0;
    apply_long_form_hint(750, 94, w, c);
    expect(w == 750 && c == 94, "apply: automatic window and context take the hint");
    w = 0; c = 50;
    apply_long_form_hint(750, 94, w, c);
    expect(w == 750 && c == 50, "apply: a caller's context is kept");
    w = 1500; c = 0;
    apply_long_form_hint(750, 94, w, c);
    expect(w == 1500 && c == 0, "apply: an explicit window is left alone");
    w = -1; c = 0;
    apply_long_form_hint(750, 94, w, c);
    expect(w == -1 && c == 0, "apply: a disabled window stays disabled");
    w = 0; c = 0;
    apply_long_form_hint(0, 0, w, c);
    expect(w == 0 && c == 0, "apply: no hint changes nothing");

    const LongFormPlan hinted = resolve_long_form_plan_frames(750, 94, 5000, 8, 41272);
    expect(hinted.enabled && hinted.window_frames == 750 && hinted.center_frames == 750 - 2 * 94,
           "a hinted window plans like a requested one");
}

void check_window_resolution() {
    constexpr int kAuto = 3750;
    constexpr int kMin  = 256;

    // Auto (requested == 0): the generous default, capped by pos_emb_max_len.
    expect(resolve_long_form_window_frames(0, 0, kAuto, kMin) == kAuto,
           "resolve: auto with unknown pos_emb keeps the default");
    expect(resolve_long_form_window_frames(0, 5000, kAuto, kMin) == kAuto,
           "resolve: auto under a large pos_emb keeps the default");
    expect(resolve_long_form_window_frames(0, 3000, kAuto, kMin) == 3000,
           "resolve: auto is capped down to pos_emb_max_len");

    // The fix: a pos_emb_max_len below the floor still wins, rather than being
    // bumped back up to kMin (which would exceed the trained positional range).
    expect(resolve_long_form_window_frames(0, 200, kAuto, kMin) == 200,
           "resolve: tiny pos_emb ceiling wins over the min-window floor");
    expect(resolve_long_form_window_frames(0, kMin, kAuto, kMin) == kMin,
           "resolve: pos_emb exactly at the floor is honoured");

    // Explicit request: floored to kMin, and also clamped to the ceiling.
    expect(resolve_long_form_window_frames(1024, 5000, kAuto, kMin) == 1024,
           "resolve: explicit request within range is untouched");
    expect(resolve_long_form_window_frames(100, 5000, kAuto, kMin) == kMin,
           "resolve: explicit request below the floor is raised to it");
    expect(resolve_long_form_window_frames(6000, 5000, kAuto, kMin) == 5000,
           "resolve: explicit request above pos_emb is capped to it");
    expect(resolve_long_form_window_frames(1000, 100, kAuto, kMin) == 100,
           "resolve: pos_emb ceiling wins even when it is below the floor");
}

void check_coreml_fixed_shape_resolution() {
    using parakeet::LongFormPlan;

    expect(!resolve_coreml_fixed_shape_plan(0, 0, 8, 5000).enabled,
           "coreml resolve: unknown/flexible capacity stays disabled");
    expect(!resolve_coreml_fixed_shape_plan(1501, 0, 8, 1501).enabled,
           "coreml resolve: exact-capacity input stays single-pass");
    expect(!resolve_coreml_fixed_shape_plan(1501, 0, 8, 1000).enabled,
           "coreml resolve: shorter input stays single-pass");
    expect(!resolve_coreml_fixed_shape_plan(7, 0, 8, 100).enabled,
           "coreml resolve: capacity below one encoder frame is unusable");

    const LongFormPlan automatic =
        resolve_coreml_fixed_shape_plan(1501, 0, 8, 1502);
    expect(automatic.enabled, "coreml resolve: oversized input enables windowing");
    expect(automatic.sub == 8, "coreml resolve: preserves subsampling factor");
    expect(automatic.window_frames == 187,
           "coreml resolve: floors capacity to whole encoder frames");
    expect(automatic.context_frames == 46,
           "coreml resolve: clamps automatic context to window/4");
    expect(automatic.center_frames == 95,
           "coreml resolve: centre excludes both context regions");
    expect((long long) automatic.window_frames * automatic.sub <= 1501,
           "coreml resolve: mel window never exceeds sidecar capacity");

    const LongFormPlan no_context =
        resolve_coreml_fixed_shape_plan(1501, -1, 8, 5000);
    expect(no_context.enabled && no_context.context_frames == 0 &&
               no_context.center_frames == no_context.window_frames,
           "coreml resolve: negative context disables overlap");

    const LongFormPlan clamped_context =
        resolve_coreml_fixed_shape_plan(1501, 1000, 8, 5000);
    expect(clamped_context.context_frames == 46,
           "coreml resolve: explicit context is clamped to window/4");

    const LongFormPlan fallback_sub =
        resolve_coreml_fixed_shape_plan(1501, 0, 0, 5000);
    expect(fallback_sub.sub == 8,
           "coreml resolve: invalid subsampling uses the default factor");

    const int center_mel = automatic.center_frames * automatic.sub;
    const int context_mel = automatic.context_frames * automatic.sub;
    const std::vector<LongFormWindow> windows =
        plan_long_form_windows(10000, center_mel, context_mel);
    for (const LongFormWindow & window : windows) {
        expect(window.window_len <= 1501,
               "coreml resolve: planned mel window exceeded fixed capacity");
    }
}

void check_coreml_eou_exact_shape_resolution() {
    expect(!resolve_coreml_exact_shape_plan(1101, 0, 8, 1100).enabled,
           "eou coreml resolve: shorter input stays single-pass");
    expect(!resolve_coreml_exact_shape_plan(1101, 0, 8, 1101).enabled,
           "eou coreml resolve: exact input remains single-pass");
    expect(!resolve_coreml_exact_shape_plan(1101, 0, 8, 1102).enabled,
           "eou coreml resolve: oversized input falls back to ggml");
    expect(!resolve_coreml_exact_shape_plan(1101, 0, 8, 5000).enabled,
           "eou coreml resolve: long input preserves attention via ggml fallback");
}

void check_nemotron_cache_aware_resolution() {
    expect(!should_use_nemotron_cache_aware_offline(1101, 0, false, 1101),
           "nemotron resolve: exact-shape input remains on offline Core ML");
    expect(should_use_nemotron_cache_aware_offline(1101, 0, false, 1102),
           "nemotron resolve: oversized Core ML input uses cache-aware execution");
    expect(should_use_nemotron_cache_aware_offline(1101, 0, false, 10000),
           "nemotron resolve: non-aligned long input avoids exact-shape windows");
    expect(!should_use_nemotron_cache_aware_offline(0, 0, false, 10000),
           "nemotron resolve: fitting ggml input remains unwindowed");
    expect(should_use_nemotron_cache_aware_offline(0, 0, true, 40000),
           "nemotron resolve: generic long-form input uses cache-aware execution");
    expect(!should_use_nemotron_cache_aware_offline(1101, -1, true, 10000),
           "nemotron resolve: negative window option permits an unwindowed reference");
}

// Drive the real trim + append over a synthetic encoder output and assert the
// stitched frames are exactly [0, 1, ... T_total-1] -- i.e. no frame is dropped
// or duplicated at any seam. Each window's synthetic encoder frame carries its
// own global frame index, so contiguity of the stitched values is a direct
// check of compute_window_trim + append_committed_frames.
//
// Units are mel frames; `sub` mel frames map to one encoder frame. center/ctx
// and n are chosen as multiples of sub so the subsampling divides exactly and
// the expected total is n / sub.
void check_stitch(int n_mel, int center_mel, int ctx_mel, int sub) {
    const std::string tag = "stitch n_mel=" + std::to_string(n_mel) +
                            " center_mel=" + std::to_string(center_mel) +
                            " ctx_mel=" + std::to_string(ctx_mel) +
                            " sub=" + std::to_string(sub);
    const std::vector<LongFormWindow> ws =
        plan_long_form_windows(n_mel, center_mel, ctx_mel);

    std::vector<float> stitched;
    for (const LongFormWindow & w : ws) {
        const int t_enc = w.window_len / sub;  // synthetic subsampling
        std::vector<float> win_enc((size_t) t_enc);
        const int base = w.window_start / sub;  // global encoder offset of the window
        for (int j = 0; j < t_enc; ++j) {
            win_enc[(size_t) j] = (float) (base + j);
        }
        const WindowTrim trim = compute_window_trim(w, t_enc, sub);
        const bool ok = append_committed_frames(stitched, win_enc,
                                                trim.left_drop, trim.center_cnt, 1);
        expect(ok, tag + ": append_committed_frames failed for a valid range");
    }

    expect((int) stitched.size() == n_mel / sub,
           tag + ": stitched frame count != n_mel / sub");
    for (size_t i = 0; i < stitched.size(); ++i) {
        if (stitched[i] != (float) i) {
            fail(tag + ": frame " + std::to_string(i) + " = " +
                 std::to_string((long long) stitched[i]) + " (gap/dup at seam)");
            break;
        }
    }
}

// append_committed_frames must reject an out-of-range request rather than
// silently appending nothing (which would let a caller over-count frames).
void check_append_bounds() {
    std::vector<float> src(10, 1.0f);  // 10 frames, stride 1
    std::vector<float> dst;
    expect(append_committed_frames(dst, src, 0, 10, 1), "append: full range should succeed");
    expect(dst.size() == 10, "append: full range appended wrong count");
    expect(!append_committed_frames(dst, src, 5, 10, 1),
           "append: out-of-range request must return false");
    expect(dst.size() == 10, "append: failed request must not modify dst");
    expect(append_committed_frames(dst, src, 0, 0, 1), "append: zero-count is a no-op success");
}

// The planner keeps every check_plan() coverage invariant and gives every
// window of an input longer than one window the same length, so the encoder
// sees a single input shape (one compute graph).
void check_plan_uniform(int n, int center, int ctx) {
    const std::string tag = "uniform n=" + std::to_string(n) +
                            " center=" + std::to_string(center) +
                            " ctx=" + std::to_string(ctx);
    const std::vector<LongFormWindow> ws =
        plan_long_form_windows_asymmetric(n, center, ctx, ctx);

    expect(!ws.empty(), tag + ": expected at least one window");
    if (ws.empty()) return;
    expect(ws.front().center_start == 0, tag + ": first centre must start at 0");
    expect(ws.back().center_end == n, tag + ": last centre must end at n_units");

    const int expected_len = std::min(center + 2 * std::max(ctx, 0), n);
    long long covered = 0;
    for (size_t i = 0; i < ws.size(); ++i) {
        const LongFormWindow & w = ws[i];
        const std::string wtag = tag + " window[" + std::to_string(i) + "]";
        expect(w.center_start < w.center_end, wtag + ": empty committed centre");
        expect(w.window_start >= 0, wtag + ": window_start < 0");
        expect(w.window_start + w.window_len <= n, wtag + ": window runs past n_units");
        expect(w.window_start <= w.center_start, wtag + ": window_start after centre");
        expect(w.window_start + w.window_len >= w.center_end,
               wtag + ": window ends before centre");
        expect(w.window_len == expected_len,
               wtag + ": window length " + std::to_string(w.window_len) +
               " != " + std::to_string(expected_len) + " (more than one encoder shape)");
        expect(w.is_final == (i + 1 == ws.size()), wtag + ": is_final mismatch");
        if (i + 1 < ws.size()) {
            expect(w.center_end == ws[i + 1].center_start,
                   wtag + ": centres not contiguous (gap or overlap at seam)");
        }
        covered += (long long) (w.center_end - w.center_start);
    }
    expect(covered == n, tag + ": committed centres do not cover [0, n) exactly");
}

// Stitching through the equal-length planner. When n_mel and the window are
// multiples of sub every window starts on the encoder-frame grid and the result
// is exact. Otherwise the final window, pulled left to full length, starts off
// the grid: the frames stay in order and the total stays within one frame of
// n_mel / sub, with at most one repeated frame at the final seam.
void check_stitch_uniform(int n_mel, int center_mel, int ctx_mel, int sub) {
    const std::string tag = "uniform stitch n_mel=" + std::to_string(n_mel) +
                            " center_mel=" + std::to_string(center_mel) +
                            " ctx_mel=" + std::to_string(ctx_mel);
    const std::vector<LongFormWindow> ws =
        plan_long_form_windows_asymmetric(n_mel, center_mel, ctx_mel, ctx_mel);

    std::vector<float> stitched;
    for (const LongFormWindow & w : ws) {
        const int t_enc = w.window_len / sub;
        std::vector<float> win_enc((size_t) t_enc);
        const int base = w.window_start / sub;
        for (int j = 0; j < t_enc; ++j) {
            win_enc[(size_t) j] = (float) (base + j);
        }
        const WindowTrim trim = compute_window_trim(w, t_enc, sub);
        expect(append_committed_frames(stitched, win_enc,
                                       trim.left_drop, trim.center_cnt, 1),
               tag + ": append_committed_frames failed for a valid range");
    }

    if (n_mel % sub == 0 && (center_mel + 2 * ctx_mel) % sub == 0) {
        expect((int) stitched.size() == n_mel / sub,
               tag + ": stitched frame count != n_mel / sub");
        for (size_t i = 0; i < stitched.size(); ++i) {
            if (stitched[i] != (float) i) {
                fail(tag + ": frame " + std::to_string(i) + " out of place (gap/dup at seam)");
                break;
            }
        }
        return;
    }

    const int n_enc = n_mel / sub;
    expect((int) stitched.size() >= n_enc && (int) stitched.size() <= n_enc + 1,
           tag + ": stitched frame count not within one frame of n_mel / sub");
    int repeats = 0;
    for (size_t i = 1; i < stitched.size(); ++i) {
        if (stitched[i] < stitched[i - 1]) {
            fail(tag + ": frames out of order at " + std::to_string(i));
            break;
        }
        if (stitched[i] == stitched[i - 1]) ++repeats;
        if (stitched[i] > stitched[i - 1] + 1.0f) {
            fail(tag + ": gap at " + std::to_string(i));
            break;
        }
    }
    expect(repeats <= 1, tag + ": more than one repeated frame");
}

// --window-sec -> encoder frames, at the shipped 80 ms frame and
// TranscribePsy's pos_emb_max_len of 5000.
void check_seconds_conversion() {
    constexpr double kFrame = 160.0 * 8.0 / 16000.0;
    constexpr int    kMin   = 256;
    constexpr int    kMax   = 5000;

    auto conv = [&](double sec, int max_frames, int & out) {
        out = 0;
        return long_form_window_frames_from_seconds(sec, kFrame, kMin, max_frames, out);
    };
    int f = 0;

    expect(conv(60.0, kMax, f) && f == 750, "seconds: 60 s -> 750 frames");
    expect(conv(30.0, kMax, f) && f == 375, "seconds: 30 s -> 375 frames");
    expect(conv(20.5, kMax, f) && f == 256, "seconds: 20.5 s is the shortest accepted window (256 frames)");
    expect(!conv(20.4, kMax, f), "seconds: 20.4 s (255 frames) is rejected");
    expect(!conv(10.0, kMax, f), "seconds: 10 s is rejected");
    expect(conv(400.0, kMax, f) && f == 5000, "seconds: 400 s (pos_emb_max_len) is accepted");
    expect(!conv(400.1, kMax, f), "seconds: past pos_emb_max_len is rejected");
    expect(conv(1000.0, 0, f) && f == 12500, "seconds: max_window_frames <= 0 means no ceiling");

    expect(conv(0.0, kMax, f) && f == -1, "seconds: 0 s disables windowing");
    f = 0;
    expect(long_form_window_frames_from_seconds(0.0, 0.0, kMin, kMax, f) && f == -1,
           "seconds: 0 s disables windowing even without a usable frame length");
    expect(!long_form_window_frames_from_seconds(60.0, 0.0, kMin, kMax, f),
           "seconds: a window needs a usable frame length");

    expect(!conv(-1.0, kMax, f), "seconds: negative window is rejected");
    expect(!conv(std::nan(""), kMax, f), "seconds: NaN window is rejected");
    expect(!conv(INFINITY, kMax, f), "seconds: infinite window is rejected");
    expect(!conv(1e12, 0, f), "seconds: a window past INT_MAX frames is rejected");
    expect(!conv(1e308, 0, f), "seconds: a window overflowing to inf frames is rejected");

    int untouched = 123;
    expect(!long_form_window_frames_from_seconds(20.4, kFrame, kMin, kMax, untouched) &&
           untouched == 123,
           "seconds: a rejected request leaves the output untouched");
}

// --overlap -> context frames: 0 % reaches the engine as -1 (no context),
// never 0 (auto); 50 % clamps to window / 4.
void check_overlap_conversion() {
    int c = 0;
    expect(long_form_context_frames_from_overlap(0.25, 750, c) && c == 94,
           "overlap: 25 % of 750 -> 94 frames each side");
    expect(long_form_context_frames_from_overlap(0.0, 750, c) && c == -1,
           "overlap: 0 % -> context -1 (no context), never 0 (auto)");
    expect(long_form_context_frames_from_overlap(0.001, 750, c) && c == -1,
           "overlap: rounding to 0 frames -> context -1");
    expect(long_form_context_frames_from_overlap(0.5, 750, c) && c == 187,
           "overlap: 50 % clamps to window / 4 = 187");
    expect(long_form_context_frames_from_overlap(0.25, 375, c) && c == 47,
           "overlap: 25 % of 375 -> 47 frames each side");

    expect(!long_form_context_frames_from_overlap(-0.1, 750, c), "overlap: negative is rejected");
    expect(!long_form_context_frames_from_overlap(0.6, 750, c), "overlap: above 0.5 is rejected");
    expect(!long_form_context_frames_from_overlap(std::nan(""), 750, c), "overlap: NaN is rejected");
    expect(!long_form_context_frames_from_overlap(0.25, 0, c), "overlap: needs a window");

    int untouched = 45;
    expect(!long_form_context_frames_from_overlap(0.9, 750, untouched) && untouched == 45,
           "overlap: a rejected request leaves the output untouched");
}

// 30 min at 60 s / 25 %: a final centre shorter than the context makes the
// last two windows share a span, which the windowed encoder encodes once; at
// 0 % overlap the last two windows differ and nothing is shared.
void check_shared_final_span() {
    constexpr int kSub = 8;
    const int n_mel = 1 + 30 * 60 * 16000 / 160;

    const std::vector<LongFormWindow> w25 =
        plan_long_form_windows_asymmetric(n_mel, 562 * kSub, 94 * kSub, 94 * kSub);
    expect(w25.size() == 41, "shared span: 30 min at 60 s / 25 % -> 41 windows");
    expect(w25.size() == 41 &&
           w25[39].window_start == w25[40].window_start &&
           w25[39].window_len == w25[40].window_len,
           "shared span: final centre under the context -> last two windows share a span");

    const std::vector<LongFormWindow> w0 =
        plan_long_form_windows_asymmetric(n_mel, 750 * kSub, 0, 0);
    expect(w0.size() == 31, "shared span: 30 min at 60 s / 0 % -> 31 windows");
    expect(w0.size() == 31 && w0[29].window_start != w0[30].window_start,
           "shared span: 0 % overlap -> the last two windows have different spans");
    for (size_t i = 0; i + 1 < w0.size(); ++i) {
        expect(w0[i].window_start == w0[i].center_start && w0[i].window_len == 6000,
               "shared span: 0 % overlap -> windows before the last one do not overlap");
    }
}

}  // namespace

int main() {
    check_long_form_hint();
    check_balanced_automatic_plan();

    // Planner: single-window, exact multiples, remainders, degenerate context.
    check_plan(100, 1000, 100);
    check_plan(1000, 1000, 128);
    check_plan(1000, 250, 50);
    check_plan(4096, 512, 64);
    check_plan(1000, 300, 50);
    check_plan(5000, 700, 90);
    check_plan(1000, 200, 0);
    check_plan(500, 100, 100000);
    check_plan(1, 4096, 64);
    check_plan(4097, 4096, 8);

    // Planner: realistic long-form case (90 min of 16 kHz audio -> ~540k mel
    // frames at hop 160). The load-bearing assertion is that every window stays
    // bounded instead of the full single-pass length.
    {
        const int sub        = 8;
        const int center_mel = 3238 * sub;
        const int ctx_mel    = 256 * sub;
        const int n_mel      = 90 * 60 * 100;  // ~540,000 mel frames
        check_plan(n_mel, center_mel, ctx_mel);
        const std::vector<LongFormWindow> ws =
            plan_long_form_windows(n_mel, center_mel, ctx_mel);
        expect(ws.size() > 1, "long-form: expected the 90-min input to split into windows");
        const int bound = center_mel + 2 * ctx_mel;
        for (const LongFormWindow & w : ws) {
            expect(w.window_len <= bound, "long-form: a window exceeded the bounded size");
        }
    }

    // Window-size resolution: pos_emb_max_len is a hard ceiling over the floor.
    check_window_resolution();
    check_coreml_fixed_shape_resolution();
    check_coreml_eou_exact_shape_resolution();
    check_nemotron_cache_aware_resolution();

    // Trim + append seam stitching (multiples of sub so subsampling is exact).
    check_stitch(2048, 256, 64, 8);   // several equal windows
    check_stitch(2048, 512, 128, 8);  // fewer, wider windows
    check_stitch(2560, 640, 128, 8);  // exact multiple, remainder-free
    check_stitch(4096, 256, 256, 8);  // context == center
    check_stitch(800, 3200, 64, 8);   // whole input fits one window

    check_append_bounds();

    // Equal-length planner: one window length per input, whatever the remainder.
    check_plan_uniform(100, 1000, 100);
    check_plan_uniform(1000, 250, 50);
    check_plan_uniform(1000, 300, 50);
    check_plan_uniform(5000, 700, 90);
    check_plan_uniform(1000, 200, 0);
    check_plan_uniform(4097, 4096, 8);
    check_plan_uniform(6001, 4496, 752);   // just over one 60 s / 25 % window
    check_plan_uniform(9000, 4496, 752);   // second-to-last window pulled left too
    check_plan_uniform(13457, 4496, 752);  // n not a multiple of 8

    check_stitch_uniform(2048, 256, 64, 8);
    check_stitch_uniform(2560, 640, 128, 8);
    check_stitch_uniform(48000, 4496, 752, 8);
    check_stitch_uniform(48001, 4496, 752, 8);  // final window off the frame grid
    check_stitch_uniform(13457, 4496, 752, 8);
    check_stitch_uniform(180001, 6000, 0, 8);   // 30 min, no overlap

    check_seconds_conversion();
    check_overlap_conversion();
    check_shared_final_span();

    if (g_failures == 0) {
        std::printf("test-long-form-windows: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-long-form-windows: %d failure(s)\n", g_failures);
    return 1;
}
