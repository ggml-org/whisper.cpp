#include "tts-cpp/parler/description.h"

#include <initializer_list>
#include <stdexcept>

namespace tts_cpp {
namespace parler {

namespace {

// ASCII-only on purpose: locale-independent (std::tolower would misbehave
// under e.g. a Turkish LC_CTYPE), and all valid field values are ASCII.
std::string to_lower(const std::string & s) {
    std::string out = s;
    for (char & c : out) {
        if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
    }
    return out;
}

// Validates a field value against its closed set (case-insensitive);
// returns the canonical lowercase form, or throws listing the valid values.
std::string canon(const char * field, const std::string & value,
                  std::initializer_list<const char *> valid) {
    const std::string low = to_lower(value);
    for (const char * v : valid) {
        if (low == v) return low;
    }
    std::string msg = "parler description: invalid ";
    msg += field;
    msg += " \"" + value + "\" (valid:";
    bool first = true;
    for (const char * v : valid) {
        msg += first ? " " : ", ";
        msg += v;
        first = false;
    }
    msg += ")";
    throw std::invalid_argument(msg);
}

struct emotion_clause {
    const char * name;    // canonical lowercase
    const char * clause;  // sentence-1 clause; "" = intended-style sentence only
    bool comma;           // always comma-attached, even with no other modifiers
};

// Clause wording follows the official examples ("with an angry tone",
// "delivering the news", "perfect for narration"); see also emotions().
const emotion_clause k_emotions[] = {
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

} // namespace

const std::vector<std::string> & emotions() {
    static const std::vector<std::string> list = [] {
        std::vector<std::string> v;
        for (const auto & e : k_emotions) v.push_back(e.name);
        return v;
    }();
    return list;
}

std::string build_description(const DescriptionSpec & spec) {
    const std::string subject = spec.voice.empty() ? "The speaker" : spec.voice;

    const emotion_clause * emo = nullptr;
    if (!spec.emotion.empty()) {
        const std::string low = to_lower(spec.emotion);
        for (const auto & e : k_emotions) {
            if (low == e.name) { emo = &e; break; }
        }
        if (!emo) {
            std::string msg = "parler description: invalid emotion \"" + spec.emotion + "\" (valid:";
            bool first = true;
            for (const auto & e : k_emotions) {
                msg += first ? " " : ", ";
                msg += e.name;
                first = false;
            }
            msg += ")";
            throw std::invalid_argument(msg);
        }
    }

    std::string mods;
    if (!spec.pace.empty()) {
        const std::string v = canon("pace", spec.pace, { "slow", "moderate", "fast" });
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
