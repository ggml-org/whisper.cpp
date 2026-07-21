#pragma once

// Robust end-of-speech stop controller for the Chatterbox T3 autoregressive
// decode loop.
//
// Background
// ----------
// The T3 decoder emits speech tokens one at a time until it samples the
// `stop_speech_token` or hits the `n_predict` cap (default 1000 ≈ 40 s of
// audio at ~25 tok/s).  On the multilingual (MTL) variant the model
// frequently fails to emit a stop token after it has finished speaking the
// input text and instead "rambles" — either repeating a near-silent cadence
// (audible as gutural/empty sounds) or hallucinating fresh low-energy content
// for many seconds.  The Python reference (resemble-ai/chatterbox) suppresses
// this with an attention-based `AlignmentStreamAnalyzer` that force-emits EOS.
//
// This controller is the host-side, attention-free half of the fix.  It is
// intentionally free of any ggml / model dependency so it can be unit-tested
// in isolation (see test/test_t3_stop_controller.cpp).  It folds three
// complementary signals into one place so the engine and CLI decode loops
// behave identically (previously the engine used a 3×-identical-token break
// with no floor while the CLI used a 2×-identical-token break with a 60-token
// floor):
//
//   1. EOS confidence — if the model's own (CFG-combined) argmax is the stop
//      token for `eos_argmax_streak` consecutive steps, the model wants to
//      stop but temperature/top-p sampling keeps missing it.  We force the
//      stop.  This is a safe signal: it never fires unless the model itself
//      prefers EOS.
//   2. Repetition — a token cycle of period p∈[1, rep_max_period] repeated
//      `rep_repeats`+ times is the stuck near-silent cadence; we stop and trim
//      the duplicated tail.
//   3. Budget — a generous, text-length-derived hard cap so a total EOS
//      failure is bounded to a few seconds instead of ~40 s.
//
// All of these only fire after a `min_tokens` floor so the initial speech
// onset (and the reference's known early-cutoff failure mode) is protected.

#include <cstdint>
#include <vector>

namespace tts_cpp::chatterbox::detail {

struct t3_stop_params {
    // Master gate.  Left false by default so the Turbo (English GPT-2) path —
    // which does not exhibit this bug — keeps its existing behaviour untouched
    // when a default-constructed param block is used.
    bool    enabled            = false;

    // Vocabulary id of the end-of-speech token (hparams.stop_speech_token).
    int32_t stop_token         = 0;

    // CFG mixing weight used to reconstruct the logits the sampler actually
    // ranks: combined = cond + cfg_weight * (cond - uncond).  Pass an empty
    // uncond vector to treat it as a plain (non-CFG) logits vector.
    float   cfg_weight         = 0.0f;

    // No heuristic stop may fire before this many speech tokens have been
    // generated.  Protects the speech onset and avoids the reference's
    // early-cutoff regression.
    int     min_tokens         = 16;

    // Hard upper bound on generated speech tokens (the last-resort backstop).
    // 0 disables the budget check (the caller's own n_predict still applies).
    int     max_tokens         = 0;

    // Repetition: a cycle of period in [1, rep_max_period] repeated at least
    // rep_repeats times triggers a stop.  rep_repeats <= 1 or rep_max_period
    // < 1 disables the check.
    int     rep_max_period     = 8;
    int     rep_repeats        = 3;

    // EOS confidence: force a stop once the CFG-combined argmax has been the
    // stop token for this many consecutive steps.  <= 0 disables the check.
    int     eos_argmax_streak  = 2;

    // Optional extra gate on EOS confidence: also require the stop token's
    // softmax probability to be >= this value.  0 disables the probability
    // gate (argmax + streak alone decide).
    float   eos_prob_threshold = 0.0f;
};

enum class t3_stop_reason {
    none = 0,
    eos_confidence, // model's argmax is EOS but sampling kept missing it
    repetition,     // periodic token cycle (stuck cadence)
    budget,         // hit the max_tokens safety net
};

struct t3_post_result {
    t3_stop_reason reason    = t3_stop_reason::none;
    // Number of tokens to drop from the tail of `generated` before
    // finalisation (used to strip a repeated cadence down to a single
    // period).  Always 0 unless reason == repetition.
    int            trim_tail = 0;
};

// Stateless-per-construction controller; `reset` (re)initialises it for one
// decode (one segment / one synthesize call).
class t3_stop_controller {
public:
    void reset(const t3_stop_params & p);

    const t3_stop_params & params() const { return params_; }

    // Pre-sampling hook.  Call with the per-step logits the sampler is about
    // to rank (cond + optional uncond) and the number of tokens generated so
    // far.  Returns true when the model's own preference is to stop and the
    // caller should emit `stop_token` instead of sampling.
    //
    // Updates the internal EOS-argmax streak as a side effect, so it must be
    // called exactly once per decode step even when the result is ignored.
    bool force_eos(int                        n_generated,
                   const std::vector<float> & logits_cond,
                   const std::vector<float> & logits_uncond);

    // Post-sampling hook.  Call with the full generated token vector (after
    // pushing the freshly sampled token).  Detects a stuck repetition cadence
    // or a blown budget and reports how much of the tail to trim.
    t3_post_result post_check(const std::vector<int32_t> & generated) const;

private:
    t3_stop_params params_;
    int            eos_streak_ = 0;
};

// Build a calibrated parameter block for the multilingual (MTL) variant.
//
//   stop_token     : hparams.stop_speech_token
//   cfg_weight     : sampling cfg_weight (for logit reconstruction)
//   n_text_tokens  : number of input text tokens for this segment (drives the
//                    proportional budget so long inputs are never clipped)
//   n_predict_cap  : the caller's hard n_predict (the budget never exceeds it)
//
// Reads optional environment overrides so the thresholds can be tuned on a
// device without recompiling:
//   CHATTERBOX_STOP_DISABLE      (any non-empty/non-"0" => disable entirely)
//   CHATTERBOX_STOP_MIN_TOKENS   (int)
//   CHATTERBOX_STOP_MAX_TOKENS   (int; overrides the computed budget)
//   CHATTERBOX_STOP_MAX_RATIO    (float; tokens-per-text-token budget ratio)
//   CHATTERBOX_STOP_EOS_STREAK   (int)
//   CHATTERBOX_STOP_EOS_PROB     (float)
//   CHATTERBOX_STOP_REP_PERIOD   (int; max repetition period)
//   CHATTERBOX_STOP_REP_REPEATS  (int)
t3_stop_params make_mtl_stop_params(int32_t stop_token,
                                    float   cfg_weight,
                                    int     n_text_tokens,
                                    int     n_predict_cap);

} // namespace tts_cpp::chatterbox::detail
