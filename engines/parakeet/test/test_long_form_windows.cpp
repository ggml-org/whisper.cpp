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
//      request, so a broken invariant fails loud instead of reading past a buffer.
//
// Exit 0 on success; non-zero (with a FAIL line per broken invariant) otherwise.

#include "long_form.h"

#include <cstdio>
#include <string>
#include <vector>

using parakeet::LongFormWindow;
using parakeet::WindowTrim;
using parakeet::append_committed_frames;
using parakeet::compute_window_trim;
using parakeet::plan_long_form_windows;
using parakeet::resolve_long_form_window_frames;

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

}  // namespace

int main() {
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

    // Trim + append seam stitching (multiples of sub so subsampling is exact).
    check_stitch(2048, 256, 64, 8);   // several equal windows
    check_stitch(2048, 512, 128, 8);  // fewer, wider windows
    check_stitch(2560, 640, 128, 8);  // exact multiple, remainder-free
    check_stitch(4096, 256, 256, 8);  // context == center
    check_stitch(800, 3200, 64, 8);   // whole input fits one window

    check_append_bounds();

    if (g_failures == 0) {
        std::printf("test-long-form-windows: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-long-form-windows: %d failure(s)\n", g_failures);
    return 1;
}
