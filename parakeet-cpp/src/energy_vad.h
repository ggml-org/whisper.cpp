#pragma once

// RMS energy voice-activity detector for streaming when the model has no built-in VAD (e.g. CTC/TDT).
//
// Design:
//   - Sliding RMS over a `window_ms` window of mono f32 PCM samples.
//   - Speaking/Silent state machine with hangover: enter Speaking immediately
//     when a window's RMS crosses the threshold; fall back to Silent only
//     after `hangover_ms` of below-threshold audio (avoids flapping on
//     consonant gaps + breath pauses).
//   - Threshold expressed in dBFS (0 dB = full-scale RMS = 1.0). Default
//     -35 dBFS works well for clean 16 kHz mono speech; bump down to
//     -45 / -50 dBFS for noisy inputs.
//   - `process()` accepts a contiguous PCM buffer + an absolute-sample
//     starting position so it can stamp transitions with a stable
//     timestamp across `feed_pcm_*()` calls.
//
// Not exposed via the public include/parakeet/ headers; this is an
// internal helper used by `StreamSession::Impl`.

#include <cstdint>

namespace parakeet {

class EnergyVad {
public:
    enum class State : int {
        Unknown  = 0,
        Speaking = 1,
        Silent   = 2,
    };

    struct Transition {
        State   to_state    = State::Unknown;
        int64_t at_sample   = 0;
        float   rms         = 0.0f;
    };

    // `sample_rate` Hz, RMS window length in ms, hangover length in ms.
    // `threshold_db` is the dBFS level above which the window is considered
    // speech (default -35 dBFS = 0.0178 RMS).
    EnergyVad(int sample_rate, int window_ms, int hangover_ms,
              float threshold_db);

    // Reset state (e.g. on `cancel()` / new session). Default state stays
    // `Unknown` until the first window-worth of audio arrives, so the
    // first transition is always Unknown -> Speaking or Unknown -> Silent.
    void reset();

    // Feed a contiguous block of mono f32 PCM samples starting at absolute
    // sample index `start_sample`. Returns the *latest* state transition
    // produced by this block, or `to_state == Unknown` if no transition
    // happened within the block. The caller is expected to fire a single
    // `StreamEventType::VadStateChanged` per non-Unknown transition.
    //
    // We deliberately collapse multiple within-block transitions to the
    // last one: in practice the chunked emission cadence (1-2 s per
    // streaming chunk) is far slower than the within-block rate, so a
    // chunk that flips Silent -> Speaking -> Silent within 1 second
    // probably indicates noise rather than a meaningful turn boundary.
    Transition process(const float * samples, int n_samples,
                       int64_t start_sample);

    State current_state() const { return state_; }

private:
    int     sample_rate_   = 16000;
    int     window_n_      = 480;     // sample_rate * window_ms / 1000
    int     hangover_n_    = 3200;    // sample_rate * hangover_ms / 1000
    float   thresh_rms_sq_ = 0.0f;    // pre-squared for cheap compare

    // Running window state. We keep a circular f32 buffer of squared
    // samples + a running sum, so each new sample is O(1).
    int64_t total_samples_seen_ = 0;
    double  window_sum_sq_ = 0.0;
    int     window_fill_   = 0;
    int     window_pos_    = 0;       // write index into window_sq_
    float   window_sq_[16000];        // big enough for window_ms <= 1 s @ 16 kHz

    // Hangover counter -- counts samples since the last above-threshold
    // window. When it crosses `hangover_n_`, we transition to Silent.
    int     hangover_count_ = 0;

    State   state_ = State::Unknown;
};

}
