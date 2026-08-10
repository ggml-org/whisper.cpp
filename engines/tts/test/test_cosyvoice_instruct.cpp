// Pins CosyVoice3 conditioning: the exact trained instruction bytes, which
// values disengage a channel, the one-instruction conflict rule, and the two
// LM prompt templates (whose divergence is deliberate and upstream-faithful).

#include "cosyvoice_instruct.h"

#include "tts-cpp/voice_controls.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using tts_cpp::cosyvoice::VoiceControls;
using tts_cpp::cosyvoice::detail::build_lm_prompt_instruct;
using tts_cpp::cosyvoice::detail::build_lm_prompt_zero_shot;
using tts_cpp::cosyvoice::detail::resolve_instruct;

static int g_failures = 0;

static void fail(const std::string & message) {
    fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++g_failures;
}

static void expect_eq(const std::string & got, const std::string & want,
                      const std::string & what) {
    if (got != want) fail(what + ": got \"" + got + "\", want \"" + want + "\"");
}

// Byte length is compared alongside content so a source-encoding regression on
// the Chinese literals fails loudly instead of looking like a content change.
static void expect_instruct(const VoiceControls & controls, const std::string & want,
                            const std::string & what) {
    try {
        const std::string got = resolve_instruct(controls);
        expect_eq(got, want, what);
        if (got.size() != want.size()) {
            fail(what + ": byte length " + std::to_string(got.size()) + " != " +
                 std::to_string(want.size()));
        }
    } catch (const std::exception & e) {
        fail(what + ": unexpected throw \"" + e.what() + "\"");
    }
}

static void expect_throw(const VoiceControls & controls,
                         const std::vector<std::string> & needles, const std::string & what) {
    try {
        const std::string got = resolve_instruct(controls);
        fail(what + ": no throw, got \"" + got + "\"");
    } catch (const std::invalid_argument & e) {
        const std::string message = e.what();
        for (const std::string & needle : needles) {
            if (message.find(needle) == std::string::npos) {
                fail(what + ": message \"" + message + "\" missing \"" + needle + "\"");
            }
        }
    }
}

static VoiceControls make(const std::string & emotion, const std::string & pace = "",
                          const std::string & instruct_text = "") {
    VoiceControls controls;
    controls.emotion = emotion;
    controls.pace = pace;
    controls.instruct_text = instruct_text;
    return controls;
}

static const char * const k_happy   = "请非常开心地说一句话。";
static const char * const k_sad     = "请非常伤心地说一句话。";
static const char * const k_anger   = "请非常生气地说一句话。";
static const char * const k_neutral = "请用正常平淡的语气说一句话。";
static const char * const k_slow    = "请用尽可能慢地语速说一句话。";
static const char * const k_fast    = "请用尽可能快地语速说一句话。";

static void check_trained_instructions() {
    expect_instruct(VoiceControls{}, "", "no controls is zero-shot");
    expect_instruct(make("happy"), k_happy, "happy");
    expect_instruct(make("sad"), k_sad, "sad");
    expect_instruct(make("anger"), k_anger, "anger");
    expect_instruct(make("neutral"), k_neutral, "neutral");
    expect_instruct(make("", "slow"), k_slow, "slow");
    expect_instruct(make("", "fast"), k_fast, "fast");
    expect_instruct(make("Happy"), k_happy, "emotion case-folds");
    expect_instruct(make("NEUTRAL"), k_neutral, "neutral case-folds");
    expect_instruct(make("", "FAST"), k_fast, "pace case-folds");
    expect_instruct(make("", "", "请用广东话表达。"), "请用广东话表达。",
                    "raw instruct passes through");
}

// "moderate" is the only value that emits no instruction: every emotion,
// neutral included, has one, but CosyVoice3 has no middle pace step.
static void check_disengaging_value() {
    expect_instruct(make("", "moderate"), "", "moderate emits no instruction");
    expect_instruct(make("neutral", "moderate"), k_neutral,
                    "moderate does not conflict with neutral");
    expect_instruct(make("happy", "moderate"), k_happy, "moderate does not conflict with emotion");
}

static void check_unsupported_values() {
    expect_throw(make("fear"), { "cosyvoice", "emotion", "valid:", "anger", "happy", "neutral", "sad" },
                 "an untrained emotion is rejected listing the supported set");
    expect_throw(make("surprise"), { "cosyvoice", "emotion", "valid:" },
                 "surprise is rejected on cosyvoice");
    // The upstream instruct_list key is "angry"; the canonical spelling is "anger".
    expect_throw(make("angry"), { "cosyvoice", "emotion", "angry", "anger" },
                 "the upstream spelling angry is rejected");
    expect_throw(make("", "quick"), { "cosyvoice", "pace", "valid:" },
                 "an unknown pace step is rejected");
}

static void check_conflicts() {
    expect_throw(make("happy", "fast"),
                 { "emotion", "pace", "happy", "fast", "one instruction" },
                 "emotion + pace conflicts");
    expect_throw(make("happy", "", "请用广东话表达。"),
                 { "emotion", "instruct_text", "one instruction" },
                 "emotion + raw instruct conflicts");
    expect_throw(make("", "slow", "请用广东话表达。"),
                 { "pace", "instruct_text", "one instruction" },
                 "pace + raw instruct conflicts");
    expect_throw(make("neutral", "fast"),
                 { "emotion", "pace", "neutral", "fast", "one instruction" },
                 "neutral + pace conflicts");
    expect_throw(make("neutral", "", "请用广东话表达。"),
                 { "emotion", "instruct_text", "one instruction" },
                 "neutral + raw instruct conflicts");
    expect_throw(make("sad", "slow", "请用广东话表达。"),
                 { "emotion", "pace", "instruct_text", "one instruction" },
                 "all three conflict");

    // A bad value must always report as a bad value, never be masked by the
    // conflict check that runs after it.
    expect_throw(make("banana", "fast"), { "emotion", "valid:" },
                 "validation precedes the conflict check");
}

static void check_prompt_templates() {
    // The divergence below is intentional: instruct mode has a space after
    // "assistant." and puts the instruction BEFORE <|endofprompt|>; zero-shot
    // has no space and puts the transcript AFTER it. Both are what the model
    // was trained on, so aligning them would move the prompt off-distribution.
    expect_eq(build_lm_prompt_instruct("X"), "You are a helpful assistant. X<|endofprompt|>",
              "instruct template");
    expect_eq(build_lm_prompt_zero_shot("T"), "You are a helpful assistant.<|endofprompt|>T",
              "zero-shot template");
    if (build_lm_prompt_instruct("A") == build_lm_prompt_zero_shot("A")) {
        fail("the two prompt templates must stay distinct");
    }
}

static void check_empty() {
    if (!VoiceControls{}.empty()) fail("default VoiceControls is empty()");
    if (make("happy").empty()) fail("a set emotion is not empty()");
    if (make("", "", "x").empty()) fail("a set instruct_text is not empty()");
}

int main() {
    check_trained_instructions();
    check_disengaging_value();
    check_unsupported_values();
    check_conflicts();
    check_prompt_templates();
    check_empty();

    if (g_failures) {
        fprintf(stderr, "test-cosyvoice-instruct: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-cosyvoice-instruct: all cases passed\n");
    return 0;
}
