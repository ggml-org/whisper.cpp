#pragma once

#include "whisper.h"
#include "whisper-state.h"

#include <cstdint>
#include <utility>
#include <vector>

// Build silence regions from VAD probabilities.
// Returns vector of (start_cs, end_cs) pairs in centiseconds.
// vad_probs: per-frame speech probabilities (higher = speech)
// n_probs: number of frames
// n_window: samples per VAD frame (e.g. 512)
// sample_rate: audio sample rate (e.g. 16000)
// threshold: below this = silent (e.g. 0.35)
// min_silence_dur_cs: minimum silence duration in centiseconds to keep (e.g. 10 = 0.1s)
std::vector<std::pair<int64_t, int64_t>> whisper_stable_build_silence_map(
        const float * vad_probs,
        int           n_probs,
        int           n_window,
        int           sample_rate,
        float         threshold,
        int64_t       min_silence_dur_cs);

// Snap word timestamps away from silence regions (in-place).
// words_t0/t1: arrays of word start/end timestamps in centiseconds (modified in-place)
// n_words: number of words
// seg_indices: for each word, which segment it belongs to (for first/last word heuristic)
// seg_word_counts: number of words per segment
// n_segments: number of segments
// silence: vector of (start_cs, end_cs) silence regions
// min_word_dur_cs: minimum word duration in centiseconds after snapping (e.g. 5 = 0.05s)
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

// Tokenize a stable-ts style DTW gap prefix (" ...").
// Returns token ids as int32 values (compatible with whisper_token).
std::vector<int32_t> whisper_stable_get_gap_tokens(struct whisper_context * ctx);

// Select top-k monotonic heads in-place on [n_heads][n_audio][n_tokens] data
// (token-fast contiguous layout), and zero out all non-selected heads.
void whisper_stable_select_heads(
        float * data,
        int     n_tokens,
        int     n_audio,
        int     n_heads,
        int     top_k);

// Context for constrained timestamp decoding filter.
struct whisper_stable_ts_filter_ctx {
    std::vector<uint8_t>          timestamp_silence_mask;
    int64_t                       seek_cs = 0;
    int64_t                       token_step_cs = 2; // timestamp token resolution: 20 ms
    whisper_logits_filter_callback wrapped_callback = nullptr;
    void *                        wrapped_user_data = nullptr;
};

// Runtime data prepared before decode and consumed after decode.
struct whisper_stable_runtime {
    bool active = false;
    bool filter_installed = false;
    bool has_vad_mapping = false;
    std::vector<std::pair<int64_t, int64_t>> silence_regions_cs;
    std::vector<std::pair<int64_t, int64_t>> mapping_processed_to_original;
    whisper_stable_ts_filter_ctx filter_ctx;
};

// Build a timestamp-bin silence mask over the processed audio timeline.
// silence_regions_cs: silence on original timeline in centiseconds
// mapping_processed_to_original: optional processed->original mapping points
// total_processed_cs: processed audio duration in centiseconds
// token_step_cs: timestamp bin resolution in centiseconds (2 for Whisper)
std::vector<uint8_t> whisper_stable_build_timestamp_silence_mask(
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        const std::vector<std::pair<int64_t, int64_t>> & mapping_processed_to_original,
        int64_t total_processed_cs,
        int64_t token_step_cs);

// Convert internal VAD mapping entries into (processed_cs, original_cs) pairs.
std::vector<std::pair<int64_t, int64_t>> whisper_stable_copy_vad_mapping(
        const std::vector<vad_time_mapping> & mapping_table);

// Update current decode window seek (centiseconds) for constrained decoding.
void whisper_stable_set_filter_seek(void * user_data, int64_t seek_cs);

// Configure stable timestamp constrained decoding callback and wrapped callback chain.
// Returns true if callback was installed.
bool whisper_stable_setup_filter(
        struct whisper_full_params & params,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        const std::vector<std::pair<int64_t, int64_t>> & mapping_processed_to_original,
        int64_t total_processed_cs,
        struct whisper_stable_ts_filter_ctx * filter_ctx);

// Internal accessor implemented in whisper.cpp for opaque VAD context details.
int whisper_stable_vad_n_window(struct whisper_vad_context * vctx);

// Prepare stable runtime data and optional constrained decoding filter.
// Returns true when stable post-processing should run.
bool whisper_stable_prepare(
        struct whisper_full_params & params,
        const float * vad_probs,
        int n_vad_probs,
        int vad_n_window,
        int sample_rate,
        int64_t total_processed_cs,
        float vad_threshold,
        int64_t min_silence_dur_cs,
        bool has_vad_mapping,
        const std::vector<vad_time_mapping> & mapping_table,
        struct whisper_stable_runtime * runtime);

// Prepare using VAD context directly (extracts probs + frame window internally).
bool whisper_stable_prepare_from_ctx(
        struct whisper_full_params & params,
        struct whisper_vad_context * vctx,
        int n_samples,
        bool has_vad_mapping,
        const std::vector<vad_time_mapping> & mapping_table,
        struct whisper_stable_runtime * runtime);

// Apply word-level stable timestamp snapping to result segments and update each
// segment boundary from its first/last snapped word.
void whisper_stable_snap_segments(
        struct whisper_context * ctx,
        std::vector<whisper_segment> & result_all,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        const std::vector<std::pair<int64_t, int64_t>> & mapping_processed_to_original,
        bool has_vad_mapping,
        int64_t min_word_dur_cs,
        int64_t min_snap_silence_dur_cs);

// Compute overlap length in centiseconds between [t0, t1) and silence regions.
int64_t whisper_stable_silence_overlap_len(
        int64_t t0,
        int64_t t1,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs);

// Final stable step: apply prepared word-level snapping and clear VAD remap state.
void whisper_stable_finalize(
        struct whisper_context * ctx,
        std::vector<whisper_segment> & result_all,
        std::vector<vad_time_mapping> & vad_mapping_table,
        bool & has_vad_segments,
        const struct whisper_stable_runtime & runtime,
        int64_t min_word_dur_cs,
        int64_t min_snap_silence_dur_cs);

// Logits filter callback that suppresses timestamp tokens mapped to silence.
void whisper_stable_logits_filter_callback(
        struct whisper_context * ctx,
          struct whisper_state * state,
      const whisper_token_data * tokens,
                           int   n_tokens,
                         float * logits,
                          void * user_data);
