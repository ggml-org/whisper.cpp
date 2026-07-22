#pragma once

// Bounded-memory long-form encoder windowing for the offline transcribe paths.
//
// The conformer encoder's self-attention is O(T_enc^2) in the number of encoder
// frames, so running it over a multi-hour input in a single graph allocates
// tens of GB of attention scores and OOMs the process (a ~90 min file needs
// ~100 GB). The offline paths compute the mel-spectrogram once (so per-feature
// CMVN statistics are global, matching the single-pass path) and then slide the
// encoder over the mel in overlapping windows: plan_long_form_windows() tiles
// the mel into gap-free, non-overlapping "committed" centre regions, each padded
// with left/right context; run_encoder_windowed() (see parakeet_engine.cpp) runs
// the encoder per window, trims the context back off at the seams
// (compute_window_trim), and concatenates the committed frames
// (append_committed_frames) into the same [T_enc, d_model] buffer a single
// full-length pass would produce, at bounded peak memory.
//
// This header holds only pure, dependency-free helpers (the window planner, the
// window-size resolver, and the seam trim + append) so they can be unit-tested
// without loading a model (see test/test_long_form_windows.cpp).

#include <vector>

namespace parakeet {

struct LongFormWindow {
    int  window_start;  // first unit fed to the encoder (incl. left context)
    int  window_len;    // number of units fed to the encoder for this window
    int  center_start;  // first committed (kept) unit, in [0, n_units)
    int  center_end;    // one-past-last committed unit
    bool is_final;      // last window: keeps every encoder frame from center_start on
};

// Tiles [0, n_units) into gap-free, non-overlapping committed centres of
// `center_units` each (the final centre is the remainder), padding every centre
// with up to `ctx_units` of left/right context clamped to the bounds. Units are
// mel frames in the offline windowed path; the planner itself is unit-agnostic.
// The context is what the encoder needs to produce committed frames that match
// the single-pass encoder; run_encoder_windowed() trims it off after the encoder
// runs, so no committed unit is ever emitted twice.
//
// Guarantees (exercised by test/test_long_form_windows.cpp):
//   - returns >= 1 window for any n_units > 0;
//   - the committed centres are contiguous and cover exactly [0, n_units)
//     (windows[0].center_start == 0; windows[i].center_end ==
//      windows[i+1].center_start; windows.back().center_end == n_units);
//   - each window fully contains its committed centre
//     (window_start <= center_start && window_start + window_len >= center_end);
//   - window_len <= center_units + 2 * ctx_units, so peak encoder memory is
//     bounded regardless of total input length.
inline std::vector<LongFormWindow>
plan_long_form_windows(int n_units, int center_units, int ctx_units) {
    std::vector<LongFormWindow> windows;
    if (n_units <= 0 || center_units <= 0) {
        return windows;
    }
    if (ctx_units < 0) {
        ctx_units = 0;
    }

    for (int center_start = 0; center_start < n_units; center_start += center_units) {
        int center_end = center_start + center_units;
        if (center_end > n_units) {
            center_end = n_units;
        }

        int window_start = center_start - ctx_units;
        if (window_start < 0) {
            window_start = 0;
        }
        int window_end = center_end + ctx_units;
        if (window_end > n_units) {
            window_end = n_units;
        }

        LongFormWindow w;
        w.window_start = window_start;
        w.window_len   = window_end - window_start;
        w.center_start = center_start;
        w.center_end   = center_end;
        w.is_final     = (center_end >= n_units);
        windows.push_back(w);
    }
    return windows;
}

// Resolve the effective per-window encoder-frame ceiling from the requested
// value (EngineOptions::long_form_window_frames, which must be >= 0 here -- the
// < 0 "disabled" case is handled by the caller) and the model's trained
// positional range:
//   - `requested` == 0 selects `auto_window_frames`; otherwise it is the
//     explicit request;
//   - the result is floored to `min_window_frames` so a tiny request can't
//     collapse the committed centre;
//   - `pos_emb_max_len` (when > 0) is a HARD ceiling that wins over that floor:
//     a window past it would index positional embeddings the model never saw,
//     so a pathologically small range yields a small window (which the caller's
//     center_frames <= 0 guard may then reject in favour of single-pass) rather
//     than one that exceeds the trained range.
//
// Exercised model-free by test/test_long_form_windows.cpp.
inline int resolve_long_form_window_frames(int requested, int pos_emb_max_len,
                                           int auto_window_frames,
                                           int min_window_frames) {
    int window_frames = (requested == 0) ? auto_window_frames : requested;
    if (window_frames < min_window_frames) {
        window_frames = min_window_frames;
    }
    if (pos_emb_max_len > 0 && window_frames > pos_emb_max_len) {
        window_frames = pos_emb_max_len;
    }
    return window_frames;
}

struct WindowTrim {
    int left_drop;   // leading encoder frames (left context) to discard
    int center_cnt;  // committed encoder frames to keep
};

// Maps a planned window (mel-frame units) onto the encoder frames the window
// actually produced (`t_enc_frames`): drop the left-context frames, keep the
// committed centre, drop the right-context frames -- except the final window,
// which keeps its whole tail after the left context. `sub` is the subsampling
// factor (mel frames per encoder frame). Results are clamped into
// [0, t_enc_frames] so per-window subsampling rounding at the boundaries can
// never index out of range.
inline WindowTrim compute_window_trim(const LongFormWindow & w,
                                      int t_enc_frames, int sub) {
    const int div = sub > 0 ? sub : 1;
    WindowTrim t;
    t.left_drop  = (w.center_start - w.window_start) / div;
    t.center_cnt = w.is_final ? (t_enc_frames - t.left_drop)
                              : (w.center_end - w.center_start) / div;
    if (t.left_drop < 0) t.left_drop = 0;
    if (t.left_drop > t_enc_frames) t.left_drop = t_enc_frames;
    if (t.center_cnt < 0) t.center_cnt = 0;
    if (t.center_cnt > t_enc_frames - t.left_drop) {
        t.center_cnt = t_enc_frames - t.left_drop;
    }
    return t;
}

// Appends `count` frames (each `stride` floats) starting at `start_frame` from
// `src` onto `dst`. Returns false without touching `dst` when the requested
// range would read past `src` -- callers must treat that as an error rather than
// silently advancing their frame count, otherwise a downstream decoder could
// read past the concatenated buffer.
inline bool append_committed_frames(std::vector<float> & dst,
                                    const std::vector<float> & src,
                                    int start_frame, int count, int stride) {
    if (count <= 0 || stride <= 0) {
        return true;
    }
    const size_t off = (size_t) start_frame * (size_t) stride;
    const size_t n   = (size_t) count * (size_t) stride;
    if (off + n > src.size()) {
        return false;
    }
    dst.insert(dst.end(), src.begin() + off, src.begin() + off + n);
    return true;
}

}  // namespace parakeet
