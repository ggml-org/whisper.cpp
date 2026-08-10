#pragma once

// Capability listing shared by every tts-cpp CLI, so all of them word an
// engine's supported values -- and the absence of a control -- identically.

#include "tts-cpp/voice_controls.h"

#include <string>
#include <vector>

namespace tts_cpp::controls::cli {

namespace internal {

constexpr const char * k_emotions_label = "emotions";
constexpr const char * k_paces_label    = "paces";
constexpr const char * k_unsupported    = "(not supported)";
constexpr const char * k_separator      = ", ";

// Comma-separated, matching how the rejection messages render a valid set;
// a space separator would split the two-word "proper noun".
inline std::string join(const std::vector<std::string> & values) {
    std::string list;
    for (const std::string & value : values) {
        if (!list.empty()) list += k_separator;
        list += value;
    }
    return list;
}

inline std::string describe(EngineId engine, const char * label,
                            const std::vector<std::string> & values) {
    const std::string prefix = std::string(engine_name(engine)) + " " + label + ": ";
    return prefix + (values.empty() ? k_unsupported : join(values));
}

inline std::string describe_each(const std::vector<std::string> & lines) {
    std::string out;
    for (const std::string & line : lines) out += line + "\n";
    return out;
}

}  // namespace internal

inline std::string describe_emotions(EngineId engine) {
    return internal::describe(engine, internal::k_emotions_label, supported_emotions(engine));
}

inline std::string describe_paces(EngineId engine) {
    return internal::describe(engine, internal::k_paces_label, supported_paces(engine));
}

// A CLI that routes several model families cannot know the engine before the
// model is read, so it reports every family it can drive, one per line.
inline std::string describe_emotions(const std::vector<EngineId> & engines) {
    std::vector<std::string> lines;
    for (EngineId engine : engines) lines.push_back(describe_emotions(engine));
    return internal::describe_each(lines);
}

inline std::string describe_paces(const std::vector<EngineId> & engines) {
    std::vector<std::string> lines;
    for (EngineId engine : engines) lines.push_back(describe_paces(engine));
    return internal::describe_each(lines);
}

// Throws naming the engine when it has no emotion control at all, or when the
// value is outside its set. For CLIs that take --emotion for another family.
inline void validate_emotion(EngineId engine, const std::string & value) {
    canon_emotion(engine, value);
}

}  // namespace tts_cpp::controls::cli
