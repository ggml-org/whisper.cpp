// Table-driven checks of the description template renderer: exact output
// strings (pinned so by-ear-verified wording can't drift) and validation.

#include "tts-cpp/parler/description.h"

#include "tts-cpp/voice_controls.h"

#include <cstdio>
#include <stdexcept>
#include <string>

using tts_cpp::parler::DescriptionSpec;
using tts_cpp::parler::build_description;
using tts_cpp::parler::emotions;

namespace ctl = tts_cpp::controls;

static int g_failures = 0;

static void expect(const DescriptionSpec & spec, const std::string & want) {
    try {
        const std::string got = build_description(spec);
        if (got != want) {
            fprintf(stderr, "FAIL:\n  got:  \"%s\"\n  want: \"%s\"\n",
                    got.c_str(), want.c_str());
            ++g_failures;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: unexpected throw \"%s\" (want \"%s\")\n",
                e.what(), want.c_str());
        ++g_failures;
    }
}

static void expect_throw(const DescriptionSpec & spec, const std::string & needle) {
    try {
        const std::string got = build_description(spec);
        fprintf(stderr, "FAIL: no throw, got \"%s\" (want error mentioning \"%s\")\n",
                got.c_str(), needle.c_str());
        ++g_failures;
    } catch (const std::invalid_argument & e) {
        const std::string what = e.what();
        if (what.find(needle) == std::string::npos ||
            what.find("valid:") == std::string::npos) {
            fprintf(stderr, "FAIL: error \"%s\" missing \"%s\" or valid list\n",
                    what.c_str(), needle.c_str());
            ++g_failures;
        }
    }
}

// Structural checks used to sweep the whole vocabulary, complementing the
// pinned strings above: catch a missing caption path without pinning a string
// for every combination.
static void expect_renders(const DescriptionSpec & s) {
    try {
        const std::string got = build_description(s);
        if (got.empty() || got.back() != '.') {
            fprintf(stderr, "FAIL: malformed render \"%s\"\n", got.c_str());
            ++g_failures;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: unexpected throw \"%s\"\n", e.what());
        ++g_failures;
    }
}

static void expect_anchor(const std::string & emotion) {
    DescriptionSpec s;
    s.emotion = emotion;
    const std::string anchor = " The intended style is " + emotion + ".";
    try {
        const std::string got = build_description(s);
        if (got.size() < anchor.size() ||
            got.compare(got.size() - anchor.size(), anchor.size(), anchor) != 0) {
            fprintf(stderr, "FAIL: \"%s\" missing anchor \"%s\"\n", got.c_str(), anchor.c_str());
            ++g_failures;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: emotion \"%s\" unexpected throw \"%s\"\n", emotion.c_str(), e.what());
        ++g_failures;
    }
}

static DescriptionSpec spec(const std::string & voice, const std::string & emotion,
                            const std::string & pitch = "", const std::string & pace = "",
                            const std::string & expressivity = "",
                            const std::string & noise = "", const std::string & reverb = "",
                            const std::string & quality = "") {
    DescriptionSpec s;
    s.voice = voice;   s.emotion = emotion;  s.pitch = pitch;   s.pace = pace;
    s.expressivity = expressivity; s.noise = noise; s.reverb = reverb; s.quality = quality;
    return s;
}

int main() {
    const std::string rec = "The recording is very high quality with no background noise.";

    // all-defaults renders the models' recommended fallback caption verbatim
    expect(DescriptionSpec{}, "The speaker speaks naturally. " + rec);
    expect(spec("Rohit", ""), "Rohit speaks naturally. " + rec);

    // every emotion; tone clauses + the trailing intended-style anchor
    expect(spec("Rohit", "happy"),
           "Rohit speaks with a happy tone. " + rec + " The intended style is happy.");
    expect(spec("", "sad"),
           "The speaker speaks with a sad tone. " + rec + " The intended style is sad.");
    expect(spec("", "neutral"),
           "The speaker speaks with a neutral tone. " + rec + " The intended style is neutral.");
    expect(spec("Jaya", "anger"),
           "Jaya speaks with an angry tone. " + rec + " The intended style is anger.");
    expect(spec("", "fear"),
           "The speaker speaks with a fearful tone. " + rec + " The intended style is fear.");
    expect(spec("", "disgust"),
           "The speaker speaks with a disgusted tone. " + rec + " The intended style is disgust.");
    expect(spec("", "surprise"),
           "The speaker speaks with a surprised tone. " + rec + " The intended style is surprise.");
    expect(spec("", "command"),
           "The speaker speaks in a commanding style. " + rec + " The intended style is command.");
    expect(spec("", "conversation"),
           "The speaker speaks in a conversational style. " + rec +
           " The intended style is conversation.");
    expect(spec("Suresh", "narration"),
           "Suresh speaks, perfect for narration. " + rec + " The intended style is narration.");
    expect(spec("Aditi", "news"),
           "Aditi speaks, delivering the news. " + rec + " The intended style is news.");
    expect(spec("", "proper noun"),
           "The speaker speaks naturally. " + rec + " The intended style is proper noun.");

    // individual knobs
    expect(spec("", "", "", "slow"), "The speaker speaks slowly. " + rec);
    expect(spec("", "", "", "moderate"), "The speaker speaks at a moderate pace. " + rec);
    expect(spec("", "", "", "fast"), "The speaker speaks at a fast pace. " + rec);
    expect(spec("", "", "low"), "The speaker speaks with a low pitch. " + rec);
    expect(spec("", "", "high"), "The speaker speaks with a high pitch. " + rec);
    expect(spec("", "", "", "", "monotone"),
           "The speaker speaks in a monotone manner. " + rec);
    expect(spec("", "", "", "", "slightly expressive"),
           "The speaker speaks in a slightly expressive manner. " + rec);
    expect(spec("", "", "", "", "expressive"),
           "The speaker speaks in an expressive manner. " + rec);
    expect(spec("", "", "", "", "", "noisy"),
           "The speaker speaks naturally. "
           "The recording is very high quality with noticeable background noise.");
    expect(spec("", "", "", "", "", "", "distant"),
           "The speaker speaks naturally. "
           "The recording is very high quality and distant-sounding with no background noise.");
    expect(spec("", "", "", "", "", "", "close"), "The speaker speaks naturally. " + rec);
    expect(spec("", "", "", "", "", "", "", "basic"),
           "The speaker speaks naturally. The recording is basic quality with no background noise.");
    expect(spec("", "", "", "", "", "", "", "high"),
           "The speaker speaks naturally. The recording is high quality with no background noise.");

    // combined spec: modifiers in order, emotion clause comma-attached
    expect(spec("Jaya", "anger", "high", "fast", "expressive"),
           "Jaya speaks at a fast pace with a high pitch in an expressive manner, "
           "with an angry tone. " + rec + " The intended style is anger.");
    expect(spec("Divya", "news", "", "slow"),
           "Divya speaks slowly, delivering the news. " + rec + " The intended style is news.");

    // case-insensitive values canonicalize
    expect(spec("", "Happy"),
           "The speaker speaks with a happy tone. " + rec + " The intended style is happy.");
    expect(spec("", "PROPER NOUN"),
           "The speaker speaks naturally. " + rec + " The intended style is proper noun.");
    expect(spec("", "", "", "SLOW"), "The speaker speaks slowly. " + rec);
    expect(spec("", "", "", "", "", "", "", "Very High"),
           "The speaker speaks naturally. " + rec);

    // invalid values throw naming the field and listing valid values
    expect_throw(spec("", "angry"), "emotion");
    expect_throw(spec("", "", "medium"), "pitch");
    expect_throw(spec("", "", "", "quick"), "pace");
    expect_throw(spec("", "", "", "", "flat"), "expressivity");
    expect_throw(spec("", "", "", "", "", "loud"), "noise");
    expect_throw(spec("", "", "", "", "", "", "far"), "reverb");
    expect_throw(spec("", "", "", "", "", "", "", "great"), "quality");

    // emotions() lists the canonical 12
    const auto & emo = emotions();
    if (emo.size() != 12 || emo.front() != "command" || emo.back() != "surprise") {
        fprintf(stderr, "FAIL: emotions() size/order wrong (n=%zu)\n", emo.size());
        ++g_failures;
    }

    // Parler's public list IS the shared canonical vocabulary, so a reorder or
    // respelling of the shared table fails here rather than silently changing
    // the rendered caption (the canonical name is interpolated into it).
    if (emo != ctl::all_emotions() || emo != ctl::supported_emotions(ctl::EngineId::Parler)) {
        fprintf(stderr, "FAIL: emotions() diverged from the shared vocabulary\n");
        ++g_failures;
    }

    // Every canonical emotion must render and end with its anchor sentence: a
    // vocabulary entry with no caption clause would otherwise only surface at
    // runtime for whoever happened to request it.
    for (const std::string & name : ctl::supported_emotions(ctl::EngineId::Parler)) {
        expect_anchor(name);
    }

    // Pace steps come from the shared vocabulary too, and every step must
    // render across the emotion axis (the comma-attachment branch depends on
    // both), so a new step cannot land without a caption path.
    for (const std::string & pace : ctl::supported_paces(ctl::EngineId::Parler)) {
        for (const std::string & name : ctl::supported_emotions(ctl::EngineId::Parler)) {
            expect_renders(spec("", name, "", pace));
        }
    }

    // DescriptionSpec::empty()
    if (!DescriptionSpec{}.empty() || spec("Rohit", "").empty()) {
        fprintf(stderr, "FAIL: DescriptionSpec::empty()\n");
        ++g_failures;
    }

    if (g_failures) {
        fprintf(stderr, "test-parler-description: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-parler-description: all cases passed\n");
    return 0;
}
