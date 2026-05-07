#pragma once

// Install a ggml_log_callback for libtts-cpp; nullptr restores stderr.
//
// Forwards to ggml_log_set so ggml and tts-cpp share one sink.  Today
// the call is a thin pass-through: tts-cpp doesn't yet have an internal
// log_impl that fans out from chatterbox / supertonic source code, so
// the only consumer of the (cb, user_data) pair is ggml itself.  When
// such an internal site lands we'll mirror parakeet-cpp's
// std::atomic<std::shared_ptr<sink>> pattern; until then, the
// thread-safety guarantee is whatever ggml_log_set provides on the
// host's ggml build (upstream stores the cb and user_data in two
// separate static globals, so concurrent installers can transiently
// observe a (new_cb, old_user_data) mismatch - host code that calls
// tts_cpp_log_set from multiple threads should serialise externally).
//
// Default behaviour, when this function is NEVER called: ggml uses its
// own default sink (writes to stderr unconditionally), and any
// chatterbox/supertonic-internal log sites that fan out through ggml
// inherit that default.  Hosts that want gated / structured logging
// (telemetry pipelines, Bare addons, ...) install their own callback
// here.

#include "tts-cpp/export.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

TTS_CPP_API void tts_cpp_log_set(ggml_log_callback cb, void * user_data);

#ifdef __cplusplus
}
#endif
