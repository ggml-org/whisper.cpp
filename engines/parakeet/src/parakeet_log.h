#pragma once

// Logging: optional ggml-style callback or stderr; macros PARAKEET_LOG_*.

#include "ggml.h"

#ifdef __GNUC__
#define PARAKEET_LOG_PRINTF_ATTR(fmt_idx, vargs_idx) \
    __attribute__((format(printf, fmt_idx, vargs_idx)))
#else
#define PARAKEET_LOG_PRINTF_ATTR(fmt_idx, vargs_idx)
#endif

namespace parakeet {

void log_impl(enum ggml_log_level level, const char * fmt, ...) PARAKEET_LOG_PRINTF_ATTR(2, 3);

void log_set_callback(ggml_log_callback cb, void * user_data);

}

#define PARAKEET_LOG(level, ...) ::parakeet::log_impl((level), __VA_ARGS__)
#define PARAKEET_LOG_DEBUG(...)  PARAKEET_LOG(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define PARAKEET_LOG_INFO(...)   PARAKEET_LOG(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define PARAKEET_LOG_WARN(...)   PARAKEET_LOG(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define PARAKEET_LOG_ERROR(...)  PARAKEET_LOG(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
