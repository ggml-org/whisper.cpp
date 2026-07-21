#pragma once

// Install a ggml_log_callback for libparakeet; nullptr restores stderr.
//
// Also forwards to ggml_log_set so ggml and parakeet share one sink. Uses ggml.h types.

#include "export.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

PARAKEET_API void parakeet_log_set(ggml_log_callback cb,
                                             void *            user_data);

#ifdef __cplusplus
}
#endif
