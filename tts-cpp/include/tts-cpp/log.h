#pragma once

// Install a ggml_log_callback for libtts-cpp; nullptr restores stderr.
//
// Forwards to ggml_log_set so ggml and tts-cpp share one sink.  Thread-
// safe: the (callback, user_data) pair is updated atomically together,
// so a concurrent log delivery cannot see a (new_cb, old_user_data)
// mismatch.
//
// Default behaviour, when this function is NEVER called: ggml uses its
// own default sink (writes to stderr unconditionally), and any
// chatterbox/supertonic-internal log sites that fan out through ggml
// inherit that default.  Hosts that want gated / structured logging
// (telemetry pipelines, Bare addons, ...) install their own callback
// here and tts-cpp / ggml then both route through it.

#include "tts-cpp/export.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

TTS_CPP_API void tts_cpp_log_set(ggml_log_callback cb, void * user_data);

#ifdef __cplusplus
}
#endif
