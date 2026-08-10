#include "cosyvoice_instruct.h"

#include "tts-cpp/voice_controls.h"

#include <stdexcept>
#include <vector>

namespace tts_cpp::cosyvoice::detail {

namespace {

namespace ctl = ::tts_cpp::controls;

constexpr ctl::EngineId k_engine = ctl::EngineId::CosyVoice;
constexpr const char * k_context = "cosyvoice";

constexpr const char * k_assistant_prefix = "You are a helpful assistant.";
constexpr const char * k_endofprompt      = "<|endofprompt|>";

struct instruct_row {
    const char * canonical;
    const char * instruct;  // "" = engages nothing, i.e. the zero-shot path
};

// Bare sentences; build_lm_prompt_instruct() adds the wrapper.  The emotive
// rows are verbatim upstream instruct_list entries; "neutral" has no upstream
// row and carries the preset from scripts/dump-cosyvoice3-reference.py.
const instruct_row k_emotion_instruct[] = {
    { "anger",   "请非常生气地说一句话。" },  // upstream instruct_list spells this "angry"
    { "happy",   "请非常开心地说一句话。" },
    { "neutral", "请用正常平淡的语气说一句话。" },
    { "sad",     "请非常伤心地说一句话。" },
};

const instruct_row k_pace_instruct[] = {
    { "slow",     "请用尽可能慢地语速说一句话。" },
    { "moderate", ""                             },
    { "fast",     "请用尽可能快地语速说一句话。" },
};

struct engaged_channel {
    const char * name;
    std::string  value;
    std::string  instruct;
};

template <std::size_t N>
std::string instruct_for(const instruct_row (&table)[N], const std::string & canonical) {
    for (const instruct_row & row : table) {
        if (canonical == row.canonical) return row.instruct;
    }
    throw std::runtime_error(std::string(k_context) + ": no instruction row for \"" + canonical +
                             "\"");
}

void engage(std::vector<engaged_channel> & channels, const char * name,
            const std::string & value, const std::string & instruct) {
    if (instruct.empty()) return;
    channels.push_back({ name, value, instruct });
}

std::string describe(const std::vector<engaged_channel> & channels) {
    std::string list;
    for (const engaged_channel & channel : channels) {
        if (!list.empty()) list += ", ";
        list += std::string(channel.name) + "=\"" + channel.value + "\"";
    }
    return list;
}

// Every set channel is canonicalized before any is counted, so a typo reports
// as a bad value instead of being masked by the conflict error.
std::vector<engaged_channel> engaged_channels(const VoiceControls & controls) {
    std::string emotion;
    if (!controls.emotion.empty()) emotion = ctl::canon_emotion(k_engine, controls.emotion);
    std::string pace;
    if (!controls.pace.empty()) pace = ctl::canon_pace(k_engine, controls.pace);

    std::vector<engaged_channel> channels;
    if (!emotion.empty()) engage(channels, "emotion", emotion, instruct_for(k_emotion_instruct, emotion));
    if (!pace.empty())    engage(channels, "pace", pace, instruct_for(k_pace_instruct, pace));
    engage(channels, "instruct_text", controls.instruct_text, controls.instruct_text);
    return channels;
}

} // namespace

std::string resolve_instruct(const VoiceControls & controls) {
    const std::vector<engaged_channel> channels = engaged_channels(controls);
    if (channels.size() > 1) {
        throw std::invalid_argument(
            std::string(k_context) + ": conflicting conditioning controls (" + describe(channels) +
            "); CosyVoice3 is trained on one instruction per synthesis -- set exactly one "
            "(pace=\"moderate\" disengages a channel)");
    }
    return channels.empty() ? std::string{} : channels.front().instruct;
}

void validate_controls(const VoiceControls & controls) {
    resolve_instruct(controls);
}

std::string build_lm_prompt_instruct(const std::string & instruct) {
    return std::string(k_assistant_prefix) + " " + instruct + k_endofprompt;
}

std::string build_lm_prompt_zero_shot(const std::string & transcript) {
    return std::string(k_assistant_prefix) + k_endofprompt + transcript;
}

} // namespace tts_cpp::cosyvoice::detail
