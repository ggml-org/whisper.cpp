#pragma once

#include "ggml.h"

//
// logging
//
// Internal logging interface shared between whisper.cpp and the optional
// backends compiled into the library. All WHISPER_LOG_* messages are routed
// through the callback installed with whisper_log_set().
//

#ifdef __cplusplus
extern "C" {
#endif

GGML_ATTRIBUTE_FORMAT(2, 3)
void whisper_log_internal(ggml_log_level level, const char * format, ...);

#ifdef __cplusplus
}
#endif

#define WHISPER_LOG_ERROR(...) whisper_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define WHISPER_LOG_WARN(...)  whisper_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define WHISPER_LOG_INFO(...)  whisper_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
//#define WHISPER_DEBUG

#if defined(WHISPER_DEBUG)
#define WHISPER_LOG_DEBUG(...) whisper_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define WHISPER_LOG_DEBUG(...)
#endif
