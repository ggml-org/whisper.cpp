#define TTS_CPP_BUILD
#include "tts-cpp/log.h"

extern "C" TTS_CPP_API void tts_cpp_log_set(ggml_log_callback cb, void * user_data) {
    // Today this is just a thin forward over ggml_log_set: tts-cpp does
    // not yet have an internal log_impl that fans out from chatterbox /
    // supertonic source code through a separate sink, so there's nothing
    // to keep in sync with ggml's own (cb, user_data) pair.  The forward
    // is here so consumers can install one callback that catches both
    // ggml-internal logs and any future tts-cpp-internal log_impl
    // emissions through the same entry point.  Add a parakeet-style
    // log_impl when the first such site lands.
    ggml_log_set(cb, user_data);
}
