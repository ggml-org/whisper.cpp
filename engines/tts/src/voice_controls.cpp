#include "tts-cpp/voice_controls.h"

#include "voice_controls_internal.h"

#include <cstddef>
#include <stdexcept>

namespace tts_cpp {
namespace controls {

namespace {

enum engine_bit : unsigned {
    bit_parler     = 1u << 0,
    bit_cosyvoice  = 1u << 1,
    bit_supertonic = 1u << 2,
    bit_chatterbox = 1u << 3,
    bit_audio8     = 1u << 4,
};

struct control_entry {
    const char * name;
    unsigned     engines;  // bitwise OR of the engine_bit values that accept it
};

// Membership and order are load-bearing: parler::emotions() forwards to the
// Parler subset, and the canonical name is rendered into Parler's caption.
const control_entry k_emotions[] = {
    { "command",      bit_parler                 },
    { "anger",        bit_parler | bit_cosyvoice },
    { "narration",    bit_parler                 },
    { "conversation", bit_parler                 },
    { "disgust",      bit_parler                 },
    { "fear",         bit_parler                 },
    { "happy",        bit_parler | bit_cosyvoice },
    { "neutral",      bit_parler | bit_cosyvoice },
    { "proper noun",  bit_parler                 },
    { "news",         bit_parler                 },
    { "sad",          bit_parler | bit_cosyvoice },
    { "surprise",     bit_parler                 },
};

// Chatterbox's only rate control is its continuous time-stretch multiplier,
// which stays a separate exact knob; Audio8 has no rate control at all.
const control_entry k_paces[] = {
    { "slow",     bit_parler | bit_cosyvoice | bit_supertonic },
    { "moderate", bit_parler | bit_cosyvoice | bit_supertonic },
    { "fast",     bit_parler | bit_cosyvoice | bit_supertonic },
};

constexpr float k_pace_factor_slow     = 0.80f;
constexpr float k_pace_factor_moderate = 1.00f;
constexpr float k_pace_factor_fast     = 1.25f;

constexpr const char * k_control_context = "voice controls";

unsigned bit_of(EngineId engine) {
    switch (engine) {
        case EngineId::Parler:     return bit_parler;
        case EngineId::CosyVoice:  return bit_cosyvoice;
        case EngineId::Supertonic: return bit_supertonic;
        case EngineId::Chatterbox: return bit_chatterbox;
        case EngineId::Audio8:     return bit_audio8;
    }
    throw std::invalid_argument("voice controls: unknown engine id");
}

template <std::size_t N>
std::vector<std::string> collect_all(const control_entry (&table)[N]) {
    std::vector<std::string> names;
    names.reserve(N);
    for (const control_entry & entry : table) {
        names.push_back(entry.name);
    }
    return names;
}

template <std::size_t N>
std::vector<std::string> collect_supported(const control_entry (&table)[N], unsigned mask) {
    std::vector<std::string> names;
    for (const control_entry & entry : table) {
        if (entry.engines & mask) names.push_back(entry.name);
    }
    return names;
}

template <std::size_t N>
std::vector<std::vector<std::string>> collect_subsets(const control_entry (&table)[N]) {
    std::vector<std::vector<std::string>> subsets;
    for (EngineId engine : all_engines()) {
        subsets.push_back(collect_supported(table, bit_of(engine)));
    }
    return subsets;
}

bool contains(const std::vector<std::string> & values, const std::string & value) {
    for (const std::string & candidate : values) {
        if (candidate == value) return true;
    }
    return false;
}

// Valid because EngineId is contiguous from 0 and all_engines() lists it in
// declaration order, which is what collect_subsets() indexes by.
std::size_t index_of(EngineId engine) {
    return static_cast<std::size_t>(engine);
}

const std::vector<std::string> & engine_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out;
        for (EngineId engine : all_engines()) out.push_back(engine_name(engine));
        return out;
    }();
    return names;
}

const std::vector<std::string> & emotion_subset(EngineId engine) {
    static const std::vector<std::vector<std::string>> subsets = collect_subsets(k_emotions);
    return subsets.at(index_of(engine));
}

const std::vector<std::string> & pace_subset(EngineId engine) {
    static const std::vector<std::vector<std::string>> subsets = collect_subsets(k_paces);
    return subsets.at(index_of(engine));
}

// Keeps the "engine has no such control" case out of canon_in(), whose
// "(valid: )" rendering would be empty and unhelpful.
std::string canon_against(EngineId engine, const char * field,
                          const std::string & value,
                          const std::vector<std::string> & supported) {
    if (supported.empty()) {
        throw std::invalid_argument(std::string(engine_name(engine)) + ": " + field +
                                    " control is not supported by this engine");
    }
    return detail::canon_in(engine_name(engine), field, value, supported);
}

} // namespace

const char * engine_name(EngineId engine) {
    switch (engine) {
        case EngineId::Parler:     return "parler";
        case EngineId::CosyVoice:  return "cosyvoice";
        case EngineId::Supertonic: return "supertonic";
        case EngineId::Chatterbox: return "chatterbox";
        case EngineId::Audio8:     return "audio8";
    }
    throw std::invalid_argument("voice controls: unknown engine id");
}

const std::vector<EngineId> & all_engines() {
    static const std::vector<EngineId> engines = {
        EngineId::Parler,     EngineId::CosyVoice, EngineId::Supertonic,
        EngineId::Chatterbox, EngineId::Audio8,
    };
    return engines;
}

EngineId engine_from_name(const std::string & name) {
    // canon_in throws for an unknown name, so the lookup below always hits.
    const std::string canonical =
        detail::canon_in(k_control_context, "engine", name, engine_names());
    for (EngineId engine : all_engines()) {
        if (canonical == engine_name(engine)) return engine;
    }
    throw std::invalid_argument("voice controls: unknown engine id");
}

const std::vector<std::string> & all_emotions() {
    static const std::vector<std::string> names = collect_all(k_emotions);
    return names;
}

const std::vector<std::string> & supported_emotions(EngineId engine) {
    return emotion_subset(engine);
}

std::string canon_emotion(EngineId engine, const std::string & value) {
    return canon_against(engine, "emotion", value, supported_emotions(engine));
}

bool is_supported_emotion(EngineId engine, const std::string & value) {
    return contains(supported_emotions(engine), detail::to_lower(value));
}

const std::vector<std::string> & all_paces() {
    static const std::vector<std::string> names = collect_all(k_paces);
    return names;
}

const std::vector<std::string> & supported_paces(EngineId engine) {
    return pace_subset(engine);
}

std::string canon_pace(EngineId engine, const std::string & value) {
    return canon_against(engine, "pace", value, supported_paces(engine));
}

bool is_supported_pace(EngineId engine, const std::string & value) {
    return contains(supported_paces(engine), detail::to_lower(value));
}

float pace_rate_factor(const std::string & canonical_pace) {
    // canon_in throws for a non-canonical step, so the chain below always hits.
    const std::string canonical =
        detail::canon_in(k_control_context, "pace", canonical_pace, all_paces());
    if (canonical == "slow") return k_pace_factor_slow;
    if (canonical == "fast") return k_pace_factor_fast;
    return k_pace_factor_moderate;
}

namespace detail {

std::string to_lower(const std::string & value) {
    std::string out = value;
    for (char & c : out) {
        if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
    }
    return out;
}

namespace {

template <typename Range>
std::string format_valid(const Range & valid) {
    std::string list;
    bool first = true;
    for (const auto & candidate : valid) {
        list += first ? " " : ", ";
        list += candidate;
        first = false;
    }
    return list;
}

template <typename Range>
std::string canon_impl(const char * context, const char * field,
                       const std::string & value, const Range & valid) {
    const std::string low = to_lower(value);
    for (const auto & candidate : valid) {
        if (low == candidate) return low;
    }
    throw std::invalid_argument(std::string(context) + ": invalid " + field + " \"" + value +
                                "\" (valid:" + format_valid(valid) + ")");
}

} // namespace

std::string canon_in(const char * context, const char * field,
                     const std::string & value,
                     std::initializer_list<const char *> valid) {
    return canon_impl(context, field, value, valid);
}

std::string canon_in(const char * context, const char * field,
                     const std::string & value,
                     const std::vector<std::string> & valid) {
    return canon_impl(context, field, value, valid);
}

} // namespace detail

} // namespace controls
} // namespace tts_cpp
