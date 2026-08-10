#include "tts-cpp/parler/description.h"

#include "tts-cpp/voice_controls.h"
#include "voice_controls_internal.h"

#include <initializer_list>
#include <stdexcept>

namespace tts_cpp {
namespace parler {

namespace {

namespace ctl = ::tts_cpp::controls;

constexpr const char * k_context = "parler description";
constexpr ctl::EngineId k_engine = ctl::EngineId::Parler;

std::string canon(const char * field, const std::string & value,
                  std::initializer_list<const char *> valid) {
    return ctl::detail::canon_in(k_context, field, value, valid);
}

struct emotion_clause {
    const char * name;    // canonical lowercase
    const char * clause;  // sentence-1 clause; "" = intended-style sentence only
    bool comma;           // always comma-attached, even with no other modifiers
};

// Clause wording follows the official Parler examples ("with an angry tone",
// "delivering the news").  Membership and order live in voice_controls.h.
const emotion_clause k_emotion_clauses[] = {
    { "command",      "in a commanding style",    false },
    { "anger",        "with an angry tone",       false },
    { "narration",    "perfect for narration",    true  },
    { "conversation", "in a conversational style", false },
    { "disgust",      "with a disgusted tone",    false },
    { "fear",         "with a fearful tone",      false },
    { "happy",        "with a happy tone",        false },
    { "neutral",      "with a neutral tone",      false },
    { "proper noun",  "",                         false },
    { "news",         "delivering the news",      true  },
    { "sad",          "with a sad tone",          false },
    { "surprise",     "with a surprised tone",    false },
};

const emotion_clause * find_clause(const std::string & canonical) {
    for (const emotion_clause & e : k_emotion_clauses) {
        if (canonical == e.name) return &e;
    }
    return nullptr;
}

// A canonical emotion with no clause row is a table bug, not user error, hence
// the runtime_error; test-parler-description walks the vocabulary to catch it.
const emotion_clause * resolve_emotion(const std::string & value) {
    if (value.empty()) return nullptr;
    const std::string canonical =
        ctl::detail::canon_in(k_context, "emotion", value, ctl::supported_emotions(k_engine));
    const emotion_clause * emo = find_clause(canonical);
    if (!emo) {
        throw std::runtime_error(std::string(k_context) + ": no caption clause for emotion \"" +
                                 canonical + "\"");
    }
    return emo;
}

} // namespace

const std::vector<std::string> & emotions() {
    return ctl::supported_emotions(k_engine);
}

std::string build_description(const DescriptionSpec & spec) {
    const std::string subject = spec.voice.empty() ? "The speaker" : spec.voice;

    const emotion_clause * emo = resolve_emotion(spec.emotion);

    std::string mods;
    if (!spec.pace.empty()) {
        const std::string v =
            ctl::detail::canon_in(k_context, "pace", spec.pace, ctl::supported_paces(k_engine));
        mods += v == "slow" ? " slowly" : " at a " + v + " pace";
    }
    if (!spec.pitch.empty()) {
        mods += " with a " + canon("pitch", spec.pitch, { "low", "moderate", "high" }) + " pitch";
    }
    if (!spec.expressivity.empty()) {
        const std::string v = canon("expressivity", spec.expressivity,
                                    { "monotone", "slightly expressive", "expressive" });
        mods += v == "expressive" ? " in an expressive manner" : " in a " + v + " manner";
    }

    std::string s1 = subject + " speaks";
    if (emo && emo->clause[0] != '\0') {
        if (mods.empty() && !emo->comma) {
            s1 += std::string(" ") + emo->clause;
        } else {
            s1 += mods + ", " + emo->clause;
        }
    } else if (mods.empty()) {
        s1 += " naturally";
    } else {
        s1 += mods;
    }
    s1 += ".";

    const std::string quality = spec.quality.empty() ? "very high"
        : canon("quality", spec.quality, { "basic", "high", "very high" });
    const std::string noise = spec.noise.empty() ? "clear"
        : canon("noise", spec.noise, { "clear", "noisy" });
    const std::string reverb = spec.reverb.empty() ? "close"
        : canon("reverb", spec.reverb, { "close", "distant" });

    std::string s2 = "The recording is " + quality + " quality";
    if (reverb == "distant") s2 += " and distant-sounding";
    s2 += noise == "clear" ? " with no background noise."
                           : " with noticeable background noise.";

    std::string out = s1 + " " + s2;
    if (emo) {
        out += std::string(" The intended style is ") + emo->name + ".";
    }
    return out;
}

} // namespace parler
} // namespace tts_cpp
