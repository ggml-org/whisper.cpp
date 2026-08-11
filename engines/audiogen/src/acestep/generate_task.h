#pragma once

// Task-type helpers for Engine::generate. Weight-free so unit tests can cover
// validation without loading GGUFs.

#include "audiogen-cpp/acestep/engine.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace tts_cpp::acestep {

inline constexpr const char * TASK_TEXT2MUSIC  = "text2music";
inline constexpr const char * TASK_COVER       = "cover";
inline constexpr const char * TASK_COVER_NOFSQ = "cover-nofsq";

inline bool is_cover_task(const std::string & task) {
    return task == TASK_COVER || task == TASK_COVER_NOFSQ;
}

// Normalize task_type / cover strengths and return a human-readable error, or
// empty on success. Does not touch PCM buffers.
inline std::string normalize_generate_task(GenerateParams & params) {
    if (params.task_type.empty()) params.task_type = TASK_TEXT2MUSIC;

    if (params.task_type != TASK_TEXT2MUSIC && params.task_type != TASK_COVER &&
        params.task_type != TASK_COVER_NOFSQ) {
        return "acestep engine: unsupported task_type '" + params.task_type +
               "' (expected text2music|cover|cover-nofsq)";
    }

    params.audio_cover_strength =
        std::clamp(params.audio_cover_strength, 0.0f, 1.0f);
    params.cover_noise_strength =
        std::clamp(params.cover_noise_strength, 0.0f, 1.0f);

    if (!is_cover_task(params.task_type)) return {};

    if (params.source_audio.empty()) {
        return "acestep engine: task '" + params.task_type + "' requires source_audio";
    }
    if ((params.source_audio.size() & 1u) != 0) {
        return "acestep engine: source_audio must be interleaved stereo";
    }
    if (!params.reference_audio.empty() && (params.reference_audio.size() & 1u) != 0) {
        return "acestep engine: reference_audio must be interleaved stereo";
    }
    if (params.task_type == TASK_COVER) {
        return "acestep engine: task 'cover' is not implemented yet (needs FSQ tokenizer); use cover-nofsq";
    }
    if (params.audio_cover_strength < 1.0f) {
        return "acestep engine: audio_cover_strength < 1 is not implemented yet for cover-nofsq";
    }
    return {};
}

} // namespace tts_cpp::acestep
