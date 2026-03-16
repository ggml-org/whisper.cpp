#pragma once

#include "whisper.h"

#include <cstdint>
#include <string>
#include <vector>

// Internal segment representation used inside whisper.cpp implementation.
// Kept in a private header so helper units (e.g. whisper-stable.cpp) can work
// with result segments without exposing this type in the public API.
struct whisper_segment {
    int64_t t0;
    int64_t t1;

    std::string text;
    float no_speech_prob;

    std::vector<whisper_token_data> tokens;

    bool speaker_turn_next;
};

