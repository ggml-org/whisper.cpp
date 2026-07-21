#pragma once

// Description templates for Parler-TTS: render a natural-language voice
// description from structured fields, phrased the way the models' training
// captions were (data-speech / RASMALAI house style), so the conditioning
// stays in-distribution. All fields are optional; an all-default spec
// renders the models' recommended fallback caption.

#include "tts-cpp/export.h"

#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {

struct DescriptionSpec {
    // Speaker name used verbatim as the sentence subject (e.g. "Rohit",
    // "Laura"); empty renders the neutral "The speaker".
    std::string voice;

    // One of the 12 speaking styles the indic checkpoint was trained on
    // (see emotions()); case-insensitive; empty adds no style conditioning.
    std::string emotion;

    std::string pitch;         // low | moderate | high
    std::string pace;          // slow | moderate | fast
    std::string expressivity;  // monotone | slightly expressive | expressive
    std::string noise;         // clear (default) | noisy
    std::string reverb;        // close (default) | distant
    std::string quality;       // basic | high | very high (default)

    // Inline on purpose: the library builds with hidden visibility, so an
    // out-of-line method would not resolve for shared-lib consumers.
    bool empty() const {
        return voice.empty() && emotion.empty() && pitch.empty() && pace.empty() &&
               expressivity.empty() && noise.empty() && reverb.empty() && quality.empty();
    }
};

// Renders the description text for a spec. Throws std::invalid_argument
// naming the field and listing valid values when a field is out of range.
TTS_CPP_API std::string build_description(const DescriptionSpec & spec);

// The valid emotion values (canonical lowercase), in model-card order.
TTS_CPP_API const std::vector<std::string> & emotions();

} // namespace parler
} // namespace tts_cpp
