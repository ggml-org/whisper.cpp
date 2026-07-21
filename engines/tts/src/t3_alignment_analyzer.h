#pragma once

// Phase 2: text<->speech alignment analyzer.
//
// Host-side port of resemble-ai/chatterbox's AlignmentStreamAnalyzer.step
// (src/chatterbox/models/t3/inference/alignment_stream_analyzer.py).  It
// consumes, one decode frame at a time, the averaged cross-attention row of
// the aligned heads over the text-token columns (produced by the in-graph
// alignment probe; see t3_mtl.h / t3_align_last_row) and decides when the
// model has finished speaking the input text so the caller can force an EOS.
//
// Unlike the attention-free Phase 1 controller (which guesses from token
// repetition / a length budget), this knows the actual alignment position, so
// it stops precisely a few frames after the alignment reaches the end of the
// text — catching both the catastrophic ramble and the mild over-runs while
// leaving healthy generations intact.
//
// Pure STL (no ggml / model dependency) so it is unit-testable in isolation
// (see test/test_t3_alignment_analyzer.cpp).

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tts_cpp::chatterbox::detail {

struct t3_align_analyzer_params {
    // S = number of text tokens (the alignment row length).
    int   text_len        = 0;

    // "complete" once the tracked text position reaches text_len - complete_margin.
    int   complete_margin = 3;

    // long_tail: after completion, if the attention summed over post-completion
    // frames on any of the last `tail_cols` columns exceeds long_tail_thresh
    // (≈ that many frames dwelling on the final token) => force EOS.
    int   tail_cols        = 3;
    float long_tail_thresh = 5.0f;

    // alignment_repetition: after completion, if the summed per-frame max
    // attention on columns before (text_len - rep_back_cols) exceeds
    // rep_thresh (the model is re-reading earlier text) => force EOS.
    int   rep_back_cols = 5;
    float rep_thresh    = 5.0f;

    // Position-update discontinuity guard: only advance text_position when the
    // jump from the previous position is within (-disc_back, disc_fwd).
    int   disc_back = 4;
    int   disc_fwd  = 7;

    // Minimum text length below which the analyzer stays disabled (very short
    // inputs do not produce a reliable alignment; the Phase 1 controller and
    // the natural stop token handle them).
    int   min_text_len = 6;

    bool  enabled = true;

    // When true, `step` may return `suppress_eos` while the text is still being
    // spoken so the caller can stop the model from terminating early (the
    // "dropped first/last word / near-empty output" failure mode).  When false
    // the analyzer only ever forces EOS; it never suppresses.
    bool  suppress_eos_enabled = true;
};

enum class t3_align_action {
    none = 0,
    force_eos,      // bad ending detected -> caller should emit the stop token
    suppress_eos,   // mid-text token repetition -> caller should NOT stop yet
};

class t3_alignment_analyzer {
public:
    void reset(const t3_align_analyzer_params & p);

    // Feed one frame: `row` is the averaged softmax alignment over the text
    // columns (length params.text_len) for the newest query, `sampled_token`
    // is the speech token just produced for this frame.  Returns the action
    // the caller should apply before producing the next token.  Mirrors
    // AlignmentStreamAnalyzer.step.
    //
    // An empty `row` (probe unavailable, e.g. on a backend where the probe is
    // not wired) is a no-op that returns `none`, so callers degrade gracefully
    // to the Phase 1 controller.
    t3_align_action step(const std::vector<float> & row, int32_t sampled_token);

    int  text_position() const { return text_position_; }
    bool complete()      const { return complete_; }
    int  frames()        const { return frame_; }

private:
    t3_align_analyzer_params p_;
    int   frame_         = 0;     // T: frames seen so far
    int   text_position_ = 0;
    bool  complete_      = false;
    int   completed_at_  = -1;    // frame count at which `complete_` first set

    // Running reductions over the post-completion window, matching the
    // reference's A[completed_at:, -3:].sum(dim=0) and A[completed_at:, :-5].max(dim=1).sum().
    std::vector<double> tail_sum_;       // per-column sum over the last `tail_cols`
    double              early_sum_ = 0.0; // sum of per-frame max on early columns

    std::vector<int32_t> recent_tokens_;
};

// Build calibrated analyzer params for a generation in the given language.
// Starts from the English-validated defaults, applies any per-language
// calibration entry (the table is structured so a language can be tuned with
// a one-line edit; all languages currently use the validated defaults), then
// applies CHATTERBOX_ALIGN_* environment overrides for on-device tuning
// without a recompile:
//   CHATTERBOX_ALIGN_COMPLETE_MARGIN (int)
//   CHATTERBOX_ALIGN_LONG_TAIL       (float)
//   CHATTERBOX_ALIGN_REP_THRESH      (float)
//   CHATTERBOX_ALIGN_MIN_TEXT        (int)
//   CHATTERBOX_ALIGN_SUPPRESS        (0 disables EOS suppression)
// `text_len` is the alignment row length (number of text tokens).
t3_align_analyzer_params t3_align_params_for_language(const std::string & lang, int text_len);

// Reference aligned (layer, head) pairs (LLAMA_ALIGNED_HEADS from
// resemble-ai/chatterbox), filtered to those valid for a model with `n_layer`
// layers and `n_head` attention heads.  Returns the in-range subset; an empty
// result (e.g. a model too small, or a future conversion that changed the
// layer/head geometry) signals the caller to disable the alignment probe and
// fall back to the Phase 1 controller rather than reading out-of-range tensors.
std::vector<std::pair<int, int>> t3_align_valid_heads(int n_layer, int n_head);

} // namespace tts_cpp::chatterbox::detail
