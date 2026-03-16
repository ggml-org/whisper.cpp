#pragma once

#include "whisper.h"
#include "whisper-state.h"

#include <cstdint>
#include <utility>
#include <vector>

// Build silence regions from VAD probabilities.
// Returns vector of (start_cs, end_cs) pairs in centiseconds.
std::vector<std::pair<int64_t, int64_t>> whisper_stable_build_silence_map(
        const float * vad_probs,
        int           n_probs,
        int           n_window,
        int           sample_rate,
        float         threshold,
        int64_t       min_silence_dur_cs);

// Build silence regions from raw PCM energy — mirrors stable-ts wav2mask.
// Uses 320-sample (20ms) token resolution, avg-pool smoothing, and quantization.
// No VAD model required.
std::vector<std::pair<int64_t, int64_t>> whisper_stable_build_silence_map_from_pcm(
        const float * pcm,
        int           n_samples,
        int           sample_rate,
        int64_t       min_silence_dur_cs);

// Snap word timestamps away from silence regions (in-place).
// Implements the stable-ts boundary-moving algorithm:
//   - start in silence  → move start to silence_end
//   - end in silence    → move end to silence_start
//   - silence in word   → snap the boundary with less overshoot
// min_word_dur_cs: minimum word duration in centiseconds after snapping
// min_snap_silence_dur_cs: ignore silence regions shorter than this
void whisper_stable_snap_timestamps(
        int64_t     * words_t0,
        int64_t     * words_t1,
        int           n_words,
        const int   * seg_first_word,
        const int   * seg_word_count,
        int           n_segments,
        const std::vector<std::pair<int64_t, int64_t>> & silence,
        int64_t       min_word_dur_cs,
        int64_t       min_snap_silence_dur_cs);

// Apply word-level snapping to all segments and update segment boundaries.
// Token timestamps must already be in the original audio timeline (offset applied
// by the per-segment VAD decode loop before calling this).
void whisper_stable_snap_segments(
        struct whisper_context * ctx,
        std::vector<whisper_segment> & result_all,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        int64_t min_word_dur_cs,
        int64_t min_snap_silence_dur_cs);

// Tokenize the DTW gap prefix (" ...") used for gap padding.
std::vector<int32_t> whisper_stable_get_gap_tokens(struct whisper_context * ctx);

// Select top-k monotonic heads in-place on attention weight data, zero out the rest.
// data layout: [n_heads][n_audio][n_tokens] (token-fast contiguous)
void whisper_stable_select_heads(
        float * data,
        int     n_tokens,
        int     n_audio,
        int     n_heads,
        int     top_k);

// Compute overlap in centiseconds between [t0, t1) and silence regions.
int64_t whisper_stable_silence_overlap_len(
        int64_t t0,
        int64_t t1,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs);

// Context for constrained timestamp decoding filter.
struct whisper_stable_ts_filter_ctx {
    std::vector<uint8_t>           timestamp_silence_mask;
    int64_t                        seek_cs       = 0;
    int64_t                        token_step_cs = 2; // 20ms per timestamp token
    whisper_logits_filter_callback wrapped_callback  = nullptr;
    void *                         wrapped_user_data = nullptr;
};

// Install constrained decoding filter that suppresses timestamp tokens in silence.
// Only effective when processed timeline == original timeline (no VAD stripping).
// Returns true if filter was installed.
bool whisper_stable_setup_filter(
        struct whisper_full_params & params,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        int64_t total_audio_cs,
        struct whisper_stable_ts_filter_ctx * filter_ctx);

// Update the current decode window seek position (centiseconds) for the filter.
void whisper_stable_set_filter_seek(void * user_data, int64_t seek_cs);

// Logits filter callback that suppresses timestamp tokens mapped to silence bins.
void whisper_stable_logits_filter_callback(
        struct whisper_context * ctx,
          struct whisper_state * state,
      const whisper_token_data * tokens,
                           int   n_tokens,
                         float * logits,
                          void * user_data);
