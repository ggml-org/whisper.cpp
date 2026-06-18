// Unit tests for the attention-free T3 end-of-speech stop controller.
// Pure host logic — no model or ggml dependency — so it builds
// and runs standalone:
//
//   g++ -std=c++17 -I src test/test_t3_stop_controller.cpp src/t3_stop_controller.cpp -o /tmp/t && /tmp/t
//
// Coverage:
//   - EOS-confidence: argmax streak forces a stop, respects the min-token
//     floor, resets on a non-EOS argmax, honours CFG combination and the
//     optional probability gate.
//   - Repetition: period-1/2/3 cadences are detected with the correct
//     trim_tail, the min-token floor is respected, and non-repeating tails are
//     left alone.
//   - Budget: fires exactly at max_tokens.
//   - Disabled params (the Turbo path) preserve the old behaviour: no forced
//     EOS, no post-check stop.
//   - make_mtl_stop_params budget scaling + env overrides.

#include "t3_stop_controller.h"

#include <cstdint>
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
        fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);             \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
    }                                                                    \
} while (0)

constexpr int     V   = 16;
constexpr int32_t EOS = 9;

// Logits whose argmax is `argmax_idx` (that entry = 5.0, the rest = 0.0).
std::vector<float> logits_with_argmax(int argmax_idx) {
    std::vector<float> l(V, 0.0f);
    if (argmax_idx >= 0 && argmax_idx < V) l[argmax_idx] = 5.0f;
    return l;
}

t3_stop_params base_params() {
    t3_stop_params p;
    p.enabled            = true;
    p.stop_token         = EOS;
    p.cfg_weight         = 0.0f;
    p.min_tokens         = 4;
    p.max_tokens         = 0;      // disable budget unless a test sets it
    p.rep_max_period     = 8;
    p.rep_repeats        = 3;
    p.eos_argmax_streak  = 2;
    p.eos_prob_threshold = 0.0f;
    return p;
}

void test_eos_confidence_streak() {
    t3_stop_controller c;
    c.reset(base_params());

    const std::vector<float> empty;
    const auto eos_logits   = logits_with_argmax(EOS);
    const auto other_logits = logits_with_argmax(3);

    // Past the floor, a single EOS-argmax step is not enough (streak == 2).
    CHECK(!c.force_eos(10, eos_logits, empty), "streak 1 should not force");
    // Second consecutive EOS-argmax step forces the stop.
    CHECK(c.force_eos(10, eos_logits, empty), "streak 2 should force");

    // A non-EOS argmax resets the streak.
    CHECK(!c.force_eos(10, other_logits, empty), "non-EOS resets streak");
    CHECK(!c.force_eos(10, eos_logits, empty), "streak rebuilds from 1");
    CHECK(c.force_eos(10, eos_logits, empty), "streak 2 forces again");
}

void test_eos_confidence_min_tokens_floor() {
    t3_stop_controller c;
    c.reset(base_params());                 // min_tokens = 4

    const std::vector<float> empty;
    const auto eos_logits = logits_with_argmax(EOS);

    // Streak builds even below the floor, but must not force before it.
    CHECK(!c.force_eos(1, eos_logits, empty), "below floor: no force (1)");
    CHECK(!c.force_eos(2, eos_logits, empty), "below floor: no force (2)");
    CHECK(!c.force_eos(3, eos_logits, empty), "below floor: no force (3)");
    // n_generated now >= floor and streak >= 2 -> force.
    CHECK(c.force_eos(4, eos_logits, empty), "at floor: force");
}

void test_eos_confidence_cfg() {
    // With CFG, combined = cond + w*(cond - uncond).  Choose cond/uncond so the
    // raw cond argmax is NOT EOS but the CFG-combined argmax IS.
    t3_stop_params p = base_params();
    p.cfg_weight       = 2.0f;
    p.eos_argmax_streak = 1;                 // single step for a focused check

    t3_stop_controller c;
    c.reset(p);

    std::vector<float> cond(V, 0.0f);
    std::vector<float> uncond(V, 0.0f);
    // cond: token 3 highest (3.0), EOS at 2.0.
    cond[3] = 3.0f;
    cond[EOS] = 2.0f;
    // uncond: strongly favours token 3 so CFG suppresses it.
    uncond[3] = 5.0f;
    uncond[EOS] = 0.0f;
    // combined[3]  = 3 + 2*(3-5)  = -1
    // combined[EOS]= 2 + 2*(2-0)  =  6  -> argmax is EOS
    CHECK(c.force_eos(10, cond, uncond), "CFG-combined argmax should be EOS");

    // Without CFG (cfg_weight ignored via empty uncond path) the raw argmax is
    // token 3, so no force.
    t3_stop_controller c2;
    c2.reset(p);
    CHECK(!c2.force_eos(10, cond, std::vector<float>{}), "raw cond argmax is not EOS");
}

void test_eos_prob_gate() {
    t3_stop_params p = base_params();
    p.eos_argmax_streak  = 1;
    p.eos_prob_threshold = 0.99f;            // demanding

    const std::vector<float> empty;

    // Flat-ish logits: EOS is argmax but only marginally, prob well under 0.99.
    std::vector<float> weak(V, 1.0f);
    weak[EOS] = 1.2f;
    t3_stop_controller c;
    c.reset(p);
    CHECK(!c.force_eos(10, weak, empty), "weak EOS prob should not pass gate");

    // Dominant EOS: prob ~1.0, passes the gate.
    std::vector<float> strong(V, 0.0f);
    strong[EOS] = 50.0f;
    t3_stop_controller c2;
    c2.reset(p);
    CHECK(c2.force_eos(10, strong, empty), "dominant EOS prob passes gate");
}

void test_repetition_period1() {
    t3_stop_controller c;
    c.reset(base_params());                  // rep_repeats = 3, min_tokens = 4

    // 5 distinct tokens then the value 7 four times in a row.
    std::vector<int32_t> g = {1, 2, 3, 4, 7, 7, 7, 7};
    t3_post_result r = c.post_check(g);
    CHECK(r.reason == t3_stop_reason::repetition, "period-1 repetition detected");
    // 4 copies of a period-1 block -> trim 3, leaving one 7.
    CHECK(r.trim_tail == 3, "period-1 trim_tail == 3, got %d", r.trim_tail);
}

void test_repetition_period2() {
    t3_stop_controller c;
    c.reset(base_params());

    // "5 6" repeated 3 times at the tail.
    std::vector<int32_t> g = {1, 2, 3, 4, 5, 6, 5, 6, 5, 6};
    t3_post_result r = c.post_check(g);
    CHECK(r.reason == t3_stop_reason::repetition, "period-2 repetition detected");
    // 3 copies of a 2-token block -> trim (3-1)*2 = 4.
    CHECK(r.trim_tail == 4, "period-2 trim_tail == 4, got %d", r.trim_tail);
}

void test_repetition_below_floor() {
    t3_stop_params p = base_params();
    p.min_tokens = 100;                      // floor above the input length
    t3_stop_controller c;
    c.reset(p);

    std::vector<int32_t> g = {7, 7, 7, 7, 7, 7};
    t3_post_result r = c.post_check(g);
    CHECK(r.reason == t3_stop_reason::none, "no repetition stop below min_tokens");
}

void test_no_false_repetition() {
    t3_stop_controller c;
    c.reset(base_params());

    // Strictly increasing tail — no cycle.
    std::vector<int32_t> g = {1, 2, 3, 4, 5, 6, 7, 8};
    t3_post_result r = c.post_check(g);
    CHECK(r.reason == t3_stop_reason::none, "non-repeating tail not flagged");

    // A single duplicated pair is below rep_repeats=3.
    std::vector<int32_t> g2 = {1, 2, 3, 4, 5, 9, 9};
    t3_post_result r2 = c.post_check(g2);
    CHECK(r2.reason == t3_stop_reason::none, "two identical not enough (need 3)");
}

void test_budget() {
    t3_stop_params p = base_params();
    p.rep_repeats = 0;                       // isolate the budget check
    p.max_tokens  = 6;
    t3_stop_controller c;
    c.reset(p);

    std::vector<int32_t> five = {1, 2, 3, 4, 5};
    CHECK(c.post_check(five).reason == t3_stop_reason::none, "under budget: no stop");

    std::vector<int32_t> six = {1, 2, 3, 4, 5, 6};
    t3_post_result r = c.post_check(six);
    CHECK(r.reason == t3_stop_reason::budget, "at budget: stop");
    CHECK(r.trim_tail == 0, "budget never trims");
}

void test_small_input_no_premature_stop() {
    // Very small inputs (e.g. text below the alignment analyzer's min length)
    // are handled by this controller.  A tiny utterance must never be
    // force-stopped below the min-token floor, even if the model's argmax is
    // already the stop token and tokens repeat — otherwise the word is clipped.
    t3_stop_params p = base_params();
    p.min_tokens = 16;                          // production floor
    t3_stop_controller c;
    c.reset(p);
    const std::vector<float> empty;
    const auto eos = logits_with_argmax(EOS);    // model "wants" to stop every step
    std::vector<int32_t> g;
    bool forced = false, posted = false;
    for (int i = 0; i < 12; ++i) {               // 12 tokens, all below the 16 floor
        if (c.force_eos((int) g.size(), eos, empty)) forced = true;
        g.push_back(7);                          // identical tokens (would be repetition if not floored)
        if (c.post_check(g).reason != t3_stop_reason::none) posted = true;
    }
    CHECK(!forced, "tiny utterance: EOS-confidence must not fire below min_tokens (no clip)");
    CHECK(!posted, "tiny utterance: repetition/budget must not fire below min_tokens (no clip)");
}

void test_disabled_preserves_turbo() {
    t3_stop_params p;                        // default: enabled == false
    t3_stop_controller c;
    c.reset(p);

    const auto eos_logits = logits_with_argmax(0);   // argmax == stop_token (0)
    CHECK(!c.force_eos(1000, eos_logits, std::vector<float>{}),
          "disabled controller never forces EOS");

    std::vector<int32_t> g = {0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(c.post_check(g).reason == t3_stop_reason::none,
          "disabled controller never stops on repetition/budget");
}

void test_make_mtl_params_budget_scaling() {
    // Short input: budget hits the absolute floor (96).
    {
        t3_stop_params p = make_mtl_stop_params(EOS, 0.5f, /*n_text=*/1, /*cap=*/1000);
        CHECK(p.enabled, "mtl params enabled");
        CHECK(p.stop_token == EOS, "stop token propagated");
        CHECK(p.max_tokens == 96, "short input -> budget floor 96, got %d", p.max_tokens);
    }
    // Medium input: 20*40 + 64 = 864, under the cap.
    {
        t3_stop_params p = make_mtl_stop_params(EOS, 0.0f, /*n_text=*/40, /*cap=*/1000);
        CHECK(p.max_tokens == 864, "20*40+64 == 864, got %d", p.max_tokens);
    }
    // Long input: budget clamped to the n_predict cap.
    {
        t3_stop_params p = make_mtl_stop_params(EOS, 0.0f, /*n_text=*/200, /*cap=*/1000);
        CHECK(p.max_tokens == 1000, "long input clamped to cap 1000, got %d", p.max_tokens);
    }
}

void test_make_mtl_params_env_overrides() {
    setenv("CHATTERBOX_STOP_MIN_TOKENS", "32", 1);
    setenv("CHATTERBOX_STOP_MAX_TOKENS", "123", 1);
    setenv("CHATTERBOX_STOP_EOS_STREAK", "5", 1);
    setenv("CHATTERBOX_STOP_REP_REPEATS", "4", 1);
    t3_stop_params p = make_mtl_stop_params(EOS, 0.0f, 40, 1000);
    CHECK(p.min_tokens == 32, "env min_tokens override");
    CHECK(p.max_tokens == 123, "env max_tokens override");
    CHECK(p.eos_argmax_streak == 5, "env eos_streak override");
    CHECK(p.rep_repeats == 4, "env rep_repeats override");

    setenv("CHATTERBOX_STOP_DISABLE", "1", 1);
    t3_stop_params p2 = make_mtl_stop_params(EOS, 0.0f, 40, 1000);
    CHECK(!p2.enabled, "env disable override");

    unsetenv("CHATTERBOX_STOP_MIN_TOKENS");
    unsetenv("CHATTERBOX_STOP_MAX_TOKENS");
    unsetenv("CHATTERBOX_STOP_EOS_STREAK");
    unsetenv("CHATTERBOX_STOP_REP_REPEATS");
    unsetenv("CHATTERBOX_STOP_DISABLE");
}

} // namespace

int main() {
    test_eos_confidence_streak();
    test_eos_confidence_min_tokens_floor();
    test_eos_confidence_cfg();
    test_eos_prob_gate();
    test_repetition_period1();
    test_repetition_period2();
    test_repetition_below_floor();
    test_no_false_repetition();
    test_budget();
    test_small_input_no_premature_stop();
    test_disabled_preserves_turbo();
    test_make_mtl_params_budget_scaling();
    test_make_mtl_params_env_overrides();

    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL",
            g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
