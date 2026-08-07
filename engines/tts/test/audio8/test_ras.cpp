// Repetition-aware sampling, which the fixture tests cannot reach: they decode
// greedily, and pick() skips the retry under greedy. So what counts as a repeat,
// which draws enter the window, what the retry narrows to, and what a window of
// zero means are only pinned here.
//
// The distributions are built so the retry is observable. two_way() puts a
// runner-up just under the leader, far enough above the tail that an ordinary
// draw reaches either, while the narrower nucleus the retry falls back to keeps
// the leader alone -- so a suppressed candidate cannot appear in the output.
// peaked_at() is the opposite: it makes one token certain, which is how a known
// entry gets into the window. Each trial starts from a fresh sampler, because a
// window that has absorbed a few accepted tokens no longer holds the entry
// under test.

#include "audio8/sampling.h"

#include <cstdio>
#include <random>
#include <set>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

constexpr int SEMANTIC_BEGIN = 0;
constexpr int SEMANTIC_END = 3;
constexpr int EOS = 4;
constexpr int VOCAB = 5;
constexpr int TRIALS = 200;
constexpr int LEADER = 3;
// Codebook index zero, which an opening window of zeros used to make
// indistinguishable from a token the model had really drawn.
constexpr int ZERO_TOKEN = 0;
// A token the opening window could never have held, for the tests that are
// about which draws are recorded rather than about index zero.
constexpr int OPENER = 1;
constexpr int WINDOW = 4;
constexpr float LEADER_LOGIT = 10.0f;
constexpr float RUNNER_UP_LOGIT = 9.9f;
constexpr float TAIL_LOGIT = 1.0f;
constexpr float PEAK_LOGIT = 40.0f;
constexpr float NARROW_TEMPERATURE = 0.5f;
constexpr float NARROW_TOP_P = 0.5f;

int failures = 0;

void check(bool condition, const char * what, int line) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", what, __FILE__, line);
}

#define CHECK(cond, msg) check((cond), (msg), __LINE__)

sampling_params wide() {
    sampling_params params;
    params.greedy = false;
    params.temperature = 1.0f;
    params.top_k = 0;
    params.top_p = 1.0f;
    return params;
}

sampling_params narrow() {
    sampling_params params = wide();
    params.temperature = NARROW_TEMPERATURE;
    params.top_p = NARROW_TOP_P;
    return params;
}

sampling_params greedy() {
    sampling_params params;
    params.greedy = true;
    return params;
}

RepetitionAwareSampler with_window(int window) {
    return RepetitionAwareSampler(window, SEMANTIC_BEGIN, SEMANTIC_END, narrow());
}

// LEADER takes just over half the mass and `runner_up` just under it: the
// narrow nucleus drops everything past the first rank, the wide one keeps both.
std::vector<float> two_way(int runner_up) {
    std::vector<float> logits(VOCAB, TAIL_LOGIT);
    logits[LEADER] = LEADER_LOGIT;
    logits[runner_up] = RUNNER_UP_LOGIT;
    return logits;
}

std::vector<float> peaked_at(int index) {
    std::vector<float> logits(VOCAB, 0.0f);
    logits[index] = PEAK_LOGIT;
    return logits;
}

// Gets the opening draw out of the way. The token it chose is not recorded, so
// this leaves the window empty rather than holding LEADER.
void prime(RepetitionAwareSampler & sampler, std::mt19937 & rng) {
    sampler.pick(peaked_at(LEADER), wide(), rng);
}

// The same draw against the bare sampler, so a control run stays at the same
// point in the generator.
void prime_control(std::mt19937 & rng) {
    sample_token(peaked_at(LEADER), wide(), rng);
}

void hold(RepetitionAwareSampler & sampler, int token, std::mt19937 & rng) {
    sampler.pick(peaked_at(token), wide(), rng);
}

bool picks_stay(RepetitionAwareSampler & sampler, const std::vector<float> & logits,
                const sampling_params & params, std::mt19937 & rng, int expected) {
    for (int trial = 0; trial < TRIALS; ++trial) {
        if (sampler.pick(logits, params, rng) != expected) return false;
    }
    return true;
}

// Whether `token` ever comes out twice running, which is what a token the
// window is allowed to hold can do and a repeat cannot.
bool repeats_consecutively(RepetitionAwareSampler & sampler,
                           const std::vector<float> & logits, std::mt19937 & rng,
                           int token) {
    int previous = -1;
    for (int trial = 0; trial < TRIALS; ++trial) {
        const int chosen = sampler.pick(logits, wide(), rng);
        if (chosen == token && previous == token) return true;
        previous = chosen;
    }
    return false;
}

// A window of zero used to leave the history empty and then pop it anyway. The
// tokens that come back are the same either way, so this only fails where the
// standard library checks the precondition -- hence the _GLIBCXX_ASSERTIONS the
// target is built with.
void test_zero_window_survives_repeated_picks() {
    RepetitionAwareSampler sampler = with_window(0);
    std::mt19937 rng(1);
    CHECK(picks_stay(sampler, peaked_at(LEADER), wide(), rng, LEADER),
          "window 0: every pick is the dominant candidate");
}

// With no window there is nothing to compare against, so the draw has to come
// through untouched -- including how much of the generator it consumed, which a
// diverging sequence would expose.
void test_zero_window_leaves_the_draw_alone() {
    RepetitionAwareSampler sampler = with_window(0);
    std::mt19937 picked(2);
    std::mt19937 drawn(2);
    const std::vector<float> logits = two_way(ZERO_TOKEN);
    bool same = true;
    for (int trial = 0; trial < TRIALS; ++trial) {
        if (sampler.pick(logits, wide(), picked) != sample_token(logits, wide(), drawn)) {
            same = false;
        }
    }
    CHECK(same, "window 0: the draw is exactly what sample_token would return");
}

// A window that has recorded nothing must not change the draw. It used to open
// as `window` zeros and treat them as history, which made codebook index 0 a
// repeat -- and sent it to the narrow retry -- before the model had emitted it
// even once.
void test_an_unrecorded_token_is_not_a_repeat() {
    std::set<int> picked;
    bool same = true;
    for (int trial = 0; trial < TRIALS; ++trial) {
        RepetitionAwareSampler sampler = with_window(WINDOW);
        std::mt19937 picking(static_cast<uint32_t>(trial));
        std::mt19937 drawing(static_cast<uint32_t>(trial));
        prime(sampler, picking);
        prime_control(drawing);
        const int chosen = sampler.pick(two_way(ZERO_TOKEN), wide(), picking);
        if (chosen != sample_token(two_way(ZERO_TOKEN), wide(), drawing)) same = false;
        picked.insert(chosen);
    }
    CHECK(same, "an empty window returns exactly what the bare draw returns");
    CHECK(picked.count(ZERO_TOKEN) == 1, "index 0 is reachable before it is drawn");
}

// The counterpart: a token the window really holds is re-drawn from the narrow
// nucleus, so the suppression the test above rules out still happens when it
// should.
void test_a_recorded_token_is_suppressed() {
    std::set<int> picked;
    for (int trial = 0; trial < TRIALS; ++trial) {
        RepetitionAwareSampler sampler = with_window(WINDOW);
        std::mt19937 rng(static_cast<uint32_t>(trial));
        prime(sampler, rng);
        hold(sampler, ZERO_TOKEN, rng);
        picked.insert(sampler.pick(two_way(ZERO_TOKEN), wide(), rng));
    }
    CHECK(picked.count(ZERO_TOKEN) == 0, "a token in the window never comes out");
    CHECK(picked.count(LEADER) == 1, "the narrow retry lands on the leader");
}

// The opening draw is not history. The reference passes it no window and builds
// the next step's window empty, so the token it chose can come straight back --
// and this is the one place the upstream paths disagree, the checkpoint's own
// code and the SGLang adapter skipping that token where the ONNX runtime keeps
// it. Pinned so the choice cannot be changed by accident.
void test_the_opening_token_is_not_history() {
    std::set<int> picked;
    for (int trial = 0; trial < TRIALS; ++trial) {
        RepetitionAwareSampler sampler = with_window(WINDOW);
        std::mt19937 rng(static_cast<uint32_t>(trial));
        hold(sampler, OPENER, rng);
        picked.insert(sampler.pick(two_way(OPENER), wide(), rng));
    }
    CHECK(picked.count(OPENER) == 1, "the opening token is drawable again");
}

// EOS is outside [semantic_begin, semantic_end], so it is never a repeat however
// often it lands: an end of speech must not be second-guessed. Consecutive draws
// are what separates it from a semantic token, since being in the window is only
// visible on the pick after.
void test_eos_repeats_freely() {
    RepetitionAwareSampler sampler = with_window(WINDOW);
    std::mt19937 rng(4);
    prime(sampler, rng);
    CHECK(repeats_consecutively(sampler, two_way(EOS), rng, EOS),
          "EOS follows EOS");
}

void test_a_semantic_token_does_not_repeat_freely() {
    RepetitionAwareSampler sampler = with_window(WINDOW);
    std::mt19937 rng(6);
    prime(sampler, rng);
    CHECK(!repeats_consecutively(sampler, two_way(SEMANTIC_END - 2), rng,
                                 SEMANTIC_END - 2),
          "a semantic token in the same position never follows itself");
}

void test_greedy_ignores_the_window() {
    RepetitionAwareSampler sampler = with_window(WINDOW);
    std::mt19937 rng(5);
    prime(sampler, rng);
    hold(sampler, ZERO_TOKEN, rng);
    CHECK(picks_stay(sampler, peaked_at(ZERO_TOKEN), greedy(), rng, ZERO_TOKEN),
          "greedy: the argmax is taken even when it repeats");
}

// The window holds `window` entries and evicts oldest-first, so a one-entry
// window loses what it is holding as soon as the next pick records something.
void test_window_evicts_oldest_first() {
    std::set<int> seen;
    for (int trial = 0; trial < TRIALS; ++trial) {
        RepetitionAwareSampler sampler = with_window(1);
        std::mt19937 rng(static_cast<uint32_t>(trial));
        prime(sampler, rng);
        hold(sampler, ZERO_TOKEN, rng);
        hold(sampler, LEADER, rng);
        seen.insert(sampler.pick(two_way(ZERO_TOKEN), wide(), rng));
    }
    CHECK(seen.count(ZERO_TOKEN) == 1, "an evicted token is drawable again");
}

}  // namespace

int main() {
    test_zero_window_survives_repeated_picks();
    test_zero_window_leaves_the_draw_alone();
    test_an_unrecorded_token_is_not_a_repeat();
    test_a_recorded_token_is_suppressed();
    test_the_opening_token_is_not_history();
    test_eos_repeats_freely();
    test_a_semantic_token_does_not_repeat_freely();
    test_greedy_ignores_the_window();
    test_window_evicts_oldest_first();

    if (failures == 0) {
        std::fprintf(stderr, "audio8 ras: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "audio8 ras: %d failure(s)\n", failures);
    return 1;
}
