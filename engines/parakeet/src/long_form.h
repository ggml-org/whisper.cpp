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

#include <climits>
#include <cmath>
#include <cstddef>
#include <string>
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
// with context clamped to the bounds. At the start/end, context that cannot be
// placed on one side is moved to the other side when input remains, keeping the
// encoder input at its full bounded size. Units are mel frames in the offline
// windowed path; the planner itself is unit-agnostic.
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
//   - normally window_len <= center_units + 2 * ctx_units; callers supplying
//     `exact_window_units` instead get that fixed bound for sidecar routing.
inline std::vector<LongFormWindow>
plan_long_form_windows_asymmetric(int n_units, int center_units,
                                  int left_ctx_units, int right_ctx_units,
                                  int exact_window_units = 0) {
    std::vector<LongFormWindow> windows;
    if (n_units <= 0 || center_units <= 0) {
        return windows;
    }
    if (left_ctx_units < 0) left_ctx_units = 0;
    if (right_ctx_units < 0) right_ctx_units = 0;

    for (int center_start = 0; center_start < n_units; center_start += center_units) {
        int center_end = center_start + center_units;
        if (center_end > n_units) {
            center_end = n_units;
        }

        int window_start = center_start - left_ctx_units;
        if (window_start < 0) {
            window_start = 0;
        }
        int window_end = center_end + right_ctx_units;
        if (window_end > n_units) {
            window_end = n_units;
        }

        // Fill unused boundary context from the opposite side. Besides giving
        // boundary frames more useful context, this keeps fixed-shape Core ML
        // windows near their exported capacity instead of adding a large block
        // of synthetic zero padding to the first and final predictions.
        const long long requested_len_ll = exact_window_units > 0
            ? (long long) exact_window_units
            : (long long) center_units + left_ctx_units + right_ctx_units;
        const int target_len = requested_len_ll < n_units
                             ? (int) requested_len_ll : n_units;
        int missing = target_len - (window_end - window_start);
        if (missing > 0) {
            int grow_right = n_units - window_end;
            if (grow_right > missing) grow_right = missing;
            window_end += grow_right;
            missing -= grow_right;
            if (missing > 0) {
                window_start -= missing;
                if (window_start < 0) window_start = 0;
            }
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

inline std::vector<LongFormWindow>
plan_long_form_windows(int n_units, int center_units, int ctx_units,
                       int exact_window_units = 0) {
    return plan_long_form_windows_asymmetric(
        n_units, center_units, ctx_units, ctx_units, exact_window_units);
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
// The 60 s / 25 % windows the research team validated TranscribePsy with on
// phones: 750 encoder frames at the 80 ms stride, 12.5 % of context each side.
constexpr int kTranscribePsyWindowFrames  = 750;
constexpr int kTranscribePsyContextFrames = 94;
constexpr const char * kTranscribePsyDescriptionMark = "TranscribePsy";

// The window a model recommends for the automatic long-form plan: the GGUF's
// parakeet.long_form.window_frames / context_frames keys when it carries
// them, or TranscribePsy's validated setting for a GGUF whose description
// names the model (the published file predates the keys). Returns false,
// with both outputs 0, when the model recommends nothing.
inline bool long_form_hint_for_model(int window_key, int context_key,
                                     const std::string & description,
                                     int & window, int & context) {
    if (window_key > 0) {
        window  = window_key;
        context = context_key > 0 ? context_key : 0;
        return true;
    }
    if (description.find(kTranscribePsyDescriptionMark) != std::string::npos) {
        window  = kTranscribePsyWindowFrames;
        context = kTranscribePsyContextFrames;
        return true;
    }
    window  = 0;
    context = 0;
    return false;
}

// A caller who left the window automatic (0) gets the model's recommended
// window, and its recommended context unless the caller set one; explicit
// and disabled (< 0) requests are left alone.
inline void apply_long_form_hint(int hint_window, int hint_context,
                                 int & requested_window, int & requested_context) {
    if (requested_window != 0 || hint_window <= 0) {
        return;
    }
    requested_window = hint_window;
    if (requested_context == 0 && hint_context > 0) {
        requested_context = hint_context;
    }
}

// Centre frames of the automatic windows for an input of `n_enc_frames`: the
// fewest windows whose centres fit `center_frames`, then every window takes an
// equal share, so a recording just past one window is encoded as two half
// windows instead of two full ones (the attention cost is quadratic in the
// window, and every window still gets its full context on both sides).
inline int balanced_center_frames(long long n_enc_frames, int center_frames) {
    if (center_frames <= 0 || n_enc_frames <= center_frames) {
        return center_frames;
    }
    const long long windows = (n_enc_frames + center_frames - 1) / center_frames;
    return (int) ((n_enc_frames + windows - 1) / windows);
}

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

// Converts a window length in seconds into EngineOptions::long_form_window_frames:
//   - 0 disables windowing (-1);
//   - otherwise round(window_sec / frame_sec), which must lie within
//     [min_window_frames, max_window_frames] (max_window_frames <= 0: no ceiling
//     other than INT_MAX).
// Returns false, leaving `out` untouched, for non-finite or out-of-range input.
inline bool long_form_window_frames_from_seconds(double window_sec, double frame_sec,
                                                 int min_window_frames,
                                                 int max_window_frames,
                                                 int & out) {
    if (!std::isfinite(window_sec) || window_sec < 0.0) {
        return false;
    }
    if (window_sec == 0.0) {
        out = -1;
        return true;
    }
    if (!std::isfinite(frame_sec) || frame_sec <= 0.0) {
        return false;
    }
    const double frames = std::round(window_sec / frame_sec);
    if (frames < (double) min_window_frames || frames > (double) INT_MAX ||
        (max_window_frames > 0 && frames > (double) max_window_frames)) {
        return false;
    }
    out = (int) frames;
    return true;
}

// Converts the fraction of a window shared with its neighbours, in [0, 0.5],
// into EngineOptions::long_form_context_frames for a `window_frames` window:
// round(window_frames * overlap / 2), clamped to window_frames / 4 like the
// engine, and -1 when it rounds to 0 (0 would mean "auto" to the engine).
// Returns false, leaving `out` untouched, for non-finite or out-of-range input.
inline bool long_form_context_frames_from_overlap(double overlap, int window_frames, int & out) {
    if (!std::isfinite(overlap) || overlap < 0.0 || overlap > 0.5 || window_frames <= 0) {
        return false;
    }
    int context_frames = (int) std::lround((double) window_frames * overlap / 2.0);
    if (context_frames > window_frames / 4) {
        context_frames = window_frames / 4;
    }
    out = context_frames > 0 ? context_frames : -1;
    return true;
}

// ── Long-form window policy (shared by the engine and the fit projector) ──

// Auto per-window encoder-frame ceiling (~300 s at 80 ms/frame). Caps the
// O(T_enc^2) attention score tensor at a few hundred MB per window while still
// giving the decoder long committed spans, so a multi-hour input needs only a
// handful of windows. Overridden downward by the model's pos_emb_max_len.
constexpr int kLongFormAutoWindowFrames  = 3750;
// Shared context each side (~20 s at 80 ms/frame). Comfortably exceeds the
// deepest shipped encoder's convolutional receptive field (24 conformer layers
// x depthwise kernel 9 ~= 100 frames each side) plus the subsampling stack, so
// committed centre frames closely match the single-pass encoder.
constexpr int kLongFormAutoContextFrames = 256;
// Floor so a pathologically small pos_emb_max_len can't collapse the centre.
constexpr int kLongFormMinWindowFrames   = 256;

struct LongFormPlan {
    bool enabled        = false;
    int  window_frames  = 0;  // encoder-frame capacity of each window
    int  context_frames = 0;  // legacy symmetric context (left for asymmetric plans)
    int  center_frames  = 0;  // committed encoder frames per window
    int  sub            = 0;  // subsampling factor (mel frames per encoder frame)
    int  left_context_frames  = 0;
    int  right_context_frames = 0;
    int  exact_mel_frames     = 0;  // non-zero requires every mel window to match
};

// Resolve the window required by a fixed-shape Core ML encoder. The sidecar's
// capacity is expressed in mel frames while LongFormPlan uses post-subsampling
// encoder frames, so floor to a whole encoder frame. This guarantees every
// generated mel window is no larger than the sidecar input.
inline LongFormPlan resolve_coreml_fixed_shape_plan(int fixed_mel_frames,
                                                    int requested_context_frames,
                                                    int subsampling_factor,
                                                    long long n_mel_frames) {
    LongFormPlan plan;
    if (fixed_mel_frames <= 0 || n_mel_frames <= fixed_mel_frames) {
        return plan;
    }

    const int sub = subsampling_factor > 0 ? subsampling_factor : 8;
    const int window_frames = fixed_mel_frames / sub;
    if (window_frames <= 0) {
        return plan;
    }

    int context_frames = requested_context_frames;
    if (context_frames == 0) {
        context_frames = kLongFormAutoContextFrames;
    }
    if (context_frames < 0) {
        context_frames = 0;
    }
    if (context_frames > window_frames / 4) {
        context_frames = window_frames / 4;
    }

    const int center_frames = window_frames - 2 * context_frames;
    if (center_frames <= 0) {
        return plan;
    }

    plan.enabled        = true;
    plan.window_frames  = window_frames;
    plan.context_frames = context_frames;
    plan.center_frames  = center_frames;
    plan.left_context_frames  = context_frames;
    plan.right_context_frames = context_frames;
    plan.sub            = sub;
    return plan;
}

inline bool should_use_nemotron_cache_aware_offline(
        int exact_mel_frames,
        int requested_window_frames,
        bool generic_windowing_required,
        long long n_mel_frames) {
    if (requested_window_frames < 0) {
        return false;
    }

    return generic_windowing_required ||
           (exact_mel_frames > 0 && n_mel_frames > exact_mel_frames);
}

// EOU sidecars are fixed causal/chunked graphs. Restarting one on overlapping
// windows must preserve the complete attention history required by every kept
// frame, not just subsampling and chunk alignment. The current 1101-mel-frame
// sidecar cannot provide that history, so oversized EOU inputs deliberately use
// the normal ggml plan until long-form Core ML parity is validated.
inline LongFormPlan resolve_coreml_exact_shape_plan(int fixed_mel_frames,
                                                    int requested_context_frames,
                                                    int subsampling_factor,
                                                    long long n_mel_frames) {
    (void) fixed_mel_frames;
    (void) requested_context_frames;
    (void) subsampling_factor;
    (void) n_mel_frames;
    return LongFormPlan{};
}

// Pure core of the engine's long-form resolution: decide the effective window
// from the requested EngineOptions values (`requested_window_frames` /
// `requested_context_frames`, EngineOptions::long_form_* semantics), the
// model's trained positional range and subsampling factor, and whether
// `n_mel_frames` is long enough to need windowing at all. Inputs that fit
// within a single window keep the bit-identical single-pass path.
// The fit projector (parakeet_fit.cpp) shares this so its projected window
// matches what a real Engine run would use.
inline LongFormPlan resolve_long_form_plan_frames(int requested_window_frames,
                                                  int requested_context_frames,
                                                  int pos_emb_max_len,
                                                  int subsampling_factor,
                                                  long long n_mel_frames) {
    LongFormPlan plan;

    if (requested_window_frames < 0) {
        return plan;
    }

    const int sub = subsampling_factor > 0 ? subsampling_factor : 8;
    plan.sub = sub;

    const int window_frames = resolve_long_form_window_frames(
        requested_window_frames, pos_emb_max_len,
        kLongFormAutoWindowFrames, kLongFormMinWindowFrames);

    int context_frames = requested_context_frames;
    if (context_frames == 0) {
        context_frames = kLongFormAutoContextFrames;
    }
    if (context_frames < 0) {
        context_frames = 0;
    }
    if (context_frames > window_frames / 4) {
        context_frames = window_frames / 4;
    }

    // Defense-in-depth: the window/4 context clamp above keeps
    // center = window - 2*context >= window/2 > 0, so this cannot trigger under
    // the current policy; it guards a future window/context tuning that could.
    const int center_frames = window_frames - 2 * context_frames;
    if (center_frames <= 0) {
        return plan;
    }

    // Only window when a single-pass encoder run would exceed the ceiling; the
    // encoder emits roughly n_mel_frames / sub frames.
    if (n_mel_frames <= (long long) window_frames * sub) {
        return plan;
    }

    // A requested window is honoured as given; the automatic one is balanced
    // over the input so no window is larger than its share needs.
    const long long n_enc_frames = (n_mel_frames + sub - 1) / sub;
    const int planned_center = requested_window_frames == 0
        ? balanced_center_frames(n_enc_frames, center_frames)
        : center_frames;

    plan.enabled        = true;
    plan.window_frames  = planned_center + 2 * context_frames;
    plan.context_frames = context_frames;
    plan.center_frames  = planned_center;
    plan.left_context_frames  = context_frames;
    plan.right_context_frames = context_frames;
    return plan;
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
