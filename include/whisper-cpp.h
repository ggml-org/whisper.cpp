#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "whisper.h"

// whisper diarization

struct whisper_diar_context_deleter {
    void operator()(whisper_diar_context  * p) {
        whisper_diar_free(p);
    }
};
typedef std::unique_ptr<whisper_diar_context, whisper_diar_context_deleter> whisper_diar_context_ptr;

struct whisper_diar_segments_deleter {
    void operator()(whisper_diar_segments * p) {
        whisper_diar_free_segments(p);
    }
};
typedef std::unique_ptr<whisper_diar_segments, whisper_diar_segments_deleter> whisper_diar_segments_ptr;

