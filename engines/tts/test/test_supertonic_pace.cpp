// Pins Supertonic's speaking-rate resolution: the pace step scales the
// checkpoint's own default_speed rather than replacing it, "moderate" is a
// bit-exact no-op on any checkpoint, and the two rate controls are exclusive.

#include "supertonic_pace.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using tts_cpp::supertonic::detail::apply_pace_factor;
using tts_cpp::supertonic::detail::resolve_pace_factor;

static int g_failures = 0;

static void fail(const std::string & message) {
    fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++g_failures;
}

static void expect_true(bool condition, const std::string & what) {
    if (!condition) fail(what);
}

// Bit-compared, not approximately: "moderate is a no-op" is only worth
// asserting if it holds exactly.
static void expect_exact(float got, float want, const std::string & what) {
    if (got != want) {
        fail(what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
    }
}

static void expect_throw(const std::function<void()> & action,
                         const std::vector<std::string> & needles, const std::string & what) {
    try {
        action();
        fail(what + ": no throw");
    } catch (const std::invalid_argument & e) {
        const std::string message = e.what();
        for (const std::string & needle : needles) {
            if (message.find(needle) == std::string::npos) {
                fail(what + ": message \"" + message + "\" missing \"" + needle + "\"");
            }
        }
    }
}

// Nothing here may assume 1.05: default_speed is per-checkpoint metadata, and
// the whole point of a factor is that every checkpoint keeps its own rate.
static const std::vector<float> & checkpoint_defaults() {
    static const std::vector<float> defaults = { 0.9f, 1.0f, 1.05f, 1.3f };
    return defaults;
}

static float speed_for(const std::string & pace, float default_speed) {
    return apply_pace_factor(0.0f, resolve_pace_factor(0.0f, pace), default_speed);
}

static void check_no_pace_is_the_checkpoint_default() {
    expect_exact(resolve_pace_factor(0.0f, ""), 1.0f, "no pace is exactly 1.0f");
    for (float base : checkpoint_defaults()) {
        expect_exact(speed_for("", base), base, "no pace leaves the checkpoint default");
    }
}

static void check_moderate_is_a_provable_no_op() {
    expect_exact(resolve_pace_factor(0.0f, "moderate"), 1.0f, "moderate is exactly 1.0f");
    for (float base : checkpoint_defaults()) {
        expect_exact(speed_for("moderate", base), speed_for("", base),
                     "moderate is bit-identical to setting nothing");
    }
}

static void check_pace_scales_the_checkpoint_default() {
    for (float base : checkpoint_defaults()) {
        const float slow = speed_for("slow", base);
        const float fast = speed_for("fast", base);
        expect_true(slow < base, "slow is below the checkpoint default");
        expect_true(fast > base, "fast is above the checkpoint default");
        // A factor, not an absolute rate: the ratio holds whatever the
        // checkpoint ships, which a hardcoded speed would not.
        expect_exact(slow, base * resolve_pace_factor(0.0f, "slow"),
                     "slow is the default scaled by its factor");
        expect_exact(fast, base * resolve_pace_factor(0.0f, "fast"),
                     "fast is the default scaled by its factor");
    }
    expect_exact(speed_for("SLOW", 1.0f), speed_for("slow", 1.0f), "pace case-folds");
}

static void check_speed_wins_when_set_alone() {
    for (float base : checkpoint_defaults()) {
        expect_exact(apply_pace_factor(2.0f, 1.0f, base), 2.0f,
                     "speed overrides the checkpoint default");
    }
}

static void check_rejections() {
    expect_throw([] { resolve_pace_factor(1.5f, "slow"); },
                 { "speed", "pace", "not both" },
                 "speed and pace together are rejected naming both");
    expect_throw([] { resolve_pace_factor(1.5f, "moderate"); }, { "not both" },
                 "a disengaging pace step still conflicts with speed");
    expect_throw([] { resolve_pace_factor(0.0f, "quick"); },
                 { "supertonic", "pace", "valid:" },
                 "a non-canonical pace step is rejected");
}

int main() {
    check_no_pace_is_the_checkpoint_default();
    check_moderate_is_a_provable_no_op();
    check_pace_scales_the_checkpoint_default();
    check_speed_wins_when_set_alone();
    check_rejections();

    if (g_failures) {
        fprintf(stderr, "test-supertonic-pace: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-supertonic-pace: all cases passed\n");
    return 0;
}
