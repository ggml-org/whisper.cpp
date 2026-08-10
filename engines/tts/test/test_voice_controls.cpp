// Pins the canonical cross-engine conditioning vocabulary: exact contents and
// order, the per-engine subsets, and the shape of every rejection message.
// This is the list the JS addon mirrors, so drift here is drift everywhere.

#include "tts-cpp/voice_controls.h"

#include "voice_controls_cli.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using tts_cpp::controls::EngineId;
using tts_cpp::controls::all_emotions;
using tts_cpp::controls::all_engines;
using tts_cpp::controls::all_paces;
using tts_cpp::controls::canon_emotion;
using tts_cpp::controls::canon_pace;
using tts_cpp::controls::engine_from_name;
using tts_cpp::controls::engine_name;
using tts_cpp::controls::is_supported_emotion;
using tts_cpp::controls::pace_rate_factor;
using tts_cpp::controls::supported_emotions;
using tts_cpp::controls::supported_paces;

static int g_failures = 0;

static void fail(const std::string & message) {
    fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++g_failures;
}

static void expect_true(bool condition, const std::string & what) {
    if (!condition) fail(what);
}

static void expect_eq(const std::string & got, const std::string & want,
                      const std::string & what) {
    if (got != want) fail(what + ": got \"" + got + "\", want \"" + want + "\"");
}

static void expect_list(const std::vector<std::string> & got,
                        const std::vector<std::string> & want, const std::string & what) {
    if (got == want) return;
    std::string got_str;
    for (const std::string & v : got) got_str += (got_str.empty() ? "" : ", ") + v;
    std::string want_str;
    for (const std::string & v : want) want_str += (want_str.empty() ? "" : ", ") + v;
    fail(what + ": got [" + got_str + "], want [" + want_str + "]");
}

// Runs `action` and checks the thrown message contains every needle. `absent`
// lets a case assert the message does NOT degenerate (e.g. an empty valid list).
static void expect_throw(const std::function<void()> & action,
                         const std::vector<std::string> & needles,
                         const std::string & what,
                         const std::vector<std::string> & absent = {}) {
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
        for (const std::string & needle : absent) {
            if (message.find(needle) != std::string::npos) {
                fail(what + ": message \"" + message + "\" must not contain \"" + needle + "\"");
            }
        }
    }
}

// Every engine's subset must appear in canonical order, so a hand-edited
// subset cannot silently reorder relative to all_emotions().
static bool is_subsequence(const std::vector<std::string> & subset,
                           const std::vector<std::string> & canonical) {
    std::size_t at = 0;
    for (const std::string & value : subset) {
        while (at < canonical.size() && canonical[at] != value) ++at;
        if (at == canonical.size()) return false;
        ++at;
    }
    return true;
}

static void check_canonical_lists() {
    expect_list(all_emotions(),
                { "command", "anger", "narration", "conversation", "disgust", "fear", "happy",
                  "neutral", "proper noun", "news", "sad", "surprise" },
                "all_emotions()");
    expect_list(all_paces(), { "slow", "moderate", "fast" }, "all_paces()");

    // "angry" is CosyVoice's upstream key; the canonical spelling is "anger".
    for (const std::string & name : all_emotions()) {
        expect_true(name != "angry", "all_emotions() must not contain \"angry\"");
    }
}

static void check_engine_identity() {
    expect_true(all_engines().size() == 5, "all_engines() lists 5 engines");
    for (EngineId engine : all_engines()) {
        expect_true(engine_from_name(engine_name(engine)) == engine,
                    std::string("engine_from_name round-trips ") + engine_name(engine));
    }
    expect_true(engine_from_name("COSYVOICE") == EngineId::CosyVoice,
                "engine_from_name is case-insensitive");
    expect_throw([] { engine_from_name("bogus"); }, { "engine", "parler", "chatterbox" },
                 "engine_from_name rejects an unknown name");
}

static void check_subsets() {
    expect_list(supported_emotions(EngineId::Parler), all_emotions(),
                "parler supports every canonical emotion");
    expect_list(supported_emotions(EngineId::CosyVoice), { "anger", "happy", "neutral", "sad" },
                "cosyvoice emotion subset");
    expect_list(supported_emotions(EngineId::Supertonic), {}, "supertonic has no emotion");
    expect_list(supported_emotions(EngineId::Chatterbox), {}, "chatterbox has no emotion");
    expect_list(supported_emotions(EngineId::Audio8), {}, "audio8 has no emotion");

    expect_list(supported_paces(EngineId::Parler), all_paces(), "parler pace subset");
    expect_list(supported_paces(EngineId::CosyVoice), all_paces(), "cosyvoice pace subset");
    expect_list(supported_paces(EngineId::Supertonic), all_paces(), "supertonic pace subset");
    expect_list(supported_paces(EngineId::Chatterbox), {}, "chatterbox has no pace");
    expect_list(supported_paces(EngineId::Audio8), {}, "audio8 has no pace");

    for (EngineId engine : all_engines()) {
        expect_true(is_subsequence(supported_emotions(engine), all_emotions()),
                    std::string("emotion subset is canonically ordered for ") + engine_name(engine));
        expect_true(is_subsequence(supported_paces(engine), all_paces()),
                    std::string("pace subset is canonically ordered for ") + engine_name(engine));
    }
}

static void check_canonicalization() {
    expect_eq(canon_emotion(EngineId::Parler, "HAPPY"), "happy", "emotion case-folds");
    expect_eq(canon_emotion(EngineId::Parler, "Proper Noun"), "proper noun",
              "multi-word emotion case-folds");
    expect_eq(canon_pace(EngineId::Supertonic, "Moderate"), "moderate", "pace case-folds");

    expect_true(is_supported_emotion(EngineId::CosyVoice, "Sad"),
                "is_supported_emotion is case-insensitive");
    expect_true(!is_supported_emotion(EngineId::CosyVoice, "fear"),
                "is_supported_emotion rejects an out-of-subset value");
}

static void check_rejections() {
    expect_throw([] { canon_emotion(EngineId::Parler, "angry"); },
                 { "parler", "emotion", "angry", "valid:", "anger" },
                 "parler rejects \"angry\"");
    expect_throw([] { canon_emotion(EngineId::CosyVoice, "fear"); },
                 { "cosyvoice", "emotion", "valid:", "anger", "happy", "neutral", "sad" },
                 "cosyvoice rejects an untrained emotion listing its own set");

    // An engine with no such control must say so, not print "(valid: )".
    expect_throw([] { canon_emotion(EngineId::Supertonic, "happy"); },
                 { "supertonic", "emotion", "not supported" },
                 "supertonic reports no emotion control", { "valid:" });
    expect_throw([] { canon_pace(EngineId::Chatterbox, "fast"); },
                 { "chatterbox", "pace", "not supported" },
                 "chatterbox reports no pace control", { "valid:" });
    expect_throw([] { canon_emotion(EngineId::Audio8, "happy"); },
                 { "audio8", "emotion", "not supported" },
                 "audio8 reports no emotion control", { "valid:" });

    expect_throw([] { canon_emotion(EngineId::Parler, ""); }, { "parler", "emotion" },
                 "an empty emotion is rejected");
    expect_throw([] { canon_pace(EngineId::Parler, "quick"); }, { "pace", "valid:" },
                 "an unknown pace step is rejected");
}

// The CLIs print these strings verbatim, so an engine with no such control has
// to say so rather than render an empty list.
static void check_cli_descriptions() {
    namespace cli = tts_cpp::controls::cli;

    expect_eq(cli::describe_emotions(EngineId::CosyVoice),
              "cosyvoice emotions: anger, happy, neutral, sad",
              "cosyvoice emotion listing");
    expect_eq(cli::describe_paces(EngineId::Supertonic), "supertonic paces: slow, moderate, fast",
              "supertonic pace listing");
    expect_eq(cli::describe_emotions(EngineId::Supertonic), "supertonic emotions: (not supported)",
              "an engine with no emotion control says so");
    expect_eq(cli::describe_paces(EngineId::Chatterbox), "chatterbox paces: (not supported)",
              "an engine with no pace control says so");

    // Comma-separated so the two-word "proper noun" stays one value.
    expect_true(cli::describe_emotions(EngineId::Parler).find("proper noun, news") !=
                    std::string::npos,
                "a multi-word emotion is not split by the separator");

    const std::string matrix =
        cli::describe_emotions({ EngineId::Parler, EngineId::Supertonic });
    expect_eq(matrix,
              cli::describe_emotions(EngineId::Parler) + "\n" +
                  cli::describe_emotions(EngineId::Supertonic) + "\n",
              "a multi-family listing is one line per engine");
}

static void check_pace_factors() {
    // Bit-compared: `default * 1.0f == default` exactly, which is what makes
    // "moderate" a provable no-op against an engine's own default rate.
    expect_true(pace_rate_factor("moderate") == 1.0f, "moderate is exactly 1.0f");
    expect_true(pace_rate_factor("slow") < 1.0f, "slow is below 1.0f");
    expect_true(pace_rate_factor("fast") > 1.0f, "fast is above 1.0f");
    expect_true(pace_rate_factor("SLOW") == pace_rate_factor("slow"),
                "pace_rate_factor case-folds");
    expect_throw([] { pace_rate_factor("normal"); }, { "pace", "valid:" },
                 "pace_rate_factor rejects a non-canonical step");
}

int main() {
    check_canonical_lists();
    check_engine_identity();
    check_subsets();
    check_canonicalization();
    check_rejections();
    check_cli_descriptions();
    check_pace_factors();

    if (g_failures) {
        fprintf(stderr, "test-voice-controls: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-voice-controls: all cases passed\n");
    return 0;
}
