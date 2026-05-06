#define TTS_CPP_BUILD
#include "tts-cpp/log.h"

#include <mutex>

namespace {

// Pack (callback, user_data) so a concurrent tts_cpp_log_set vs an
// in-flight ggml-side log delivery cannot ever see a (new_cb,
// old_user_data) mismatch.  A plain mutex around the pair is enough:
// the log path isn't on a hot loop and the lock is held for a single
// pointer-pair copy.  Future C++20 hosts can swap to
// std::atomic<std::shared_ptr<sink>> without touching the API.
std::mutex        g_sink_mu;
ggml_log_callback g_sink_cb        = nullptr;
void *            g_sink_user_data = nullptr;

} // namespace

extern "C" TTS_CPP_API void tts_cpp_log_set(ggml_log_callback cb, void * user_data) {
    {
        std::lock_guard<std::mutex> lk(g_sink_mu);
        g_sink_cb        = cb;
        g_sink_user_data = user_data;
    }
    // Forward to ggml so ggml-internal logs route through the same sink.
    // ggml_log_set keeps its own copy of the pair internally; we own
    // the (cb, user_data) lifetime on our side via the mutex above for
    // any future log_impl-style helpers that fan out via g_sink_cb
    // directly.
    ggml_log_set(cb, user_data);
}
