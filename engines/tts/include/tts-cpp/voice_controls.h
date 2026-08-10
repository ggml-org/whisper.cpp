#pragma once

// Canonical cross-engine conditioning vocabulary. One vocabulary is defined
// here; each engine declares the subset it supports and rejects the rest.

#include "tts-cpp/export.h"

#include <string>
#include <vector>

namespace tts_cpp {
namespace controls {

// Identifies an engine for a capability query, so a host can enumerate what
// is settable before loading a model. Append new engines at the end.
enum class EngineId : int {
    Parler = 0,
    CosyVoice,
    Supertonic,
    Chatterbox,
    Audio8,
};

// Stable lowercase wire name: parler | cosyvoice | supertonic | chatterbox |
// audio8.
TTS_CPP_API const char * engine_name(EngineId engine);

// Every EngineId in declaration order, so a binding can walk the capability
// matrix without hard-coding the enum on its own side.
TTS_CPP_API const std::vector<EngineId> & all_engines();

// Case-insensitive inverse of engine_name(). Throws std::invalid_argument
// listing the known names.
TTS_CPP_API EngineId engine_from_name(const std::string & name);

// The canonical emotion vocabulary, in model-card order. "anger" is the
// canonical spelling; "angry" is not a member and is not accepted as an alias.
TTS_CPP_API const std::vector<std::string> & all_emotions();

// The subset `engine` supports, ordered as in all_emotions(). Empty means the
// engine has no emotion control at all.
TTS_CPP_API const std::vector<std::string> & supported_emotions(EngineId engine);

// Case-insensitive; returns the canonical spelling. Throws
// std::invalid_argument naming the engine and listing its supported set.
TTS_CPP_API std::string canon_emotion(EngineId engine, const std::string & value);

TTS_CPP_API bool is_supported_emotion(EngineId engine, const std::string & value);

// The canonical speaking-rate steps. Enum steps only; an engine that also has
// a continuous rate multiplier keeps that as a separate, exact knob.
TTS_CPP_API const std::vector<std::string> & all_paces();

TTS_CPP_API const std::vector<std::string> & supported_paces(EngineId engine);

TTS_CPP_API std::string canon_pace(EngineId engine, const std::string & value);

TTS_CPP_API bool is_supported_pace(EngineId engine, const std::string & value);

// Rate multiplier for a canonical pace step, relative to an engine's own
// default rate. "moderate" is exactly 1.0f, so it is a no-op by construction
// and a checkpoint keeps its own idea of a natural rate. Throws on a
// non-canonical step.
TTS_CPP_API float pace_rate_factor(const std::string & canonical_pace);

} // namespace controls
} // namespace tts_cpp
