// log_impl and log_set_callback; installs hook and forwards to ggml_log_set.

#include "parakeet_log.h"

#include "parakeet/log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

namespace parakeet {

namespace {

std::atomic<ggml_log_callback> g_callback{nullptr};
void * g_user_data = nullptr;

}

void log_set_callback(ggml_log_callback cb, void * user_data) {
    g_user_data = user_data;
    g_callback.store(cb, std::memory_order_release);
    ggml_log_set(cb, user_data);
}

void log_impl(ggml_log_level level, const char * fmt, ...) {
    char stack_buf[1024];

    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    const int n = std::vsnprintf(stack_buf, sizeof(stack_buf), fmt, args);
    va_end(args);

    const char * text = stack_buf;
    std::vector<char> heap_buf;
    if (n >= (int) sizeof(stack_buf)) {
        heap_buf.resize((size_t) n + 1);
        std::vsnprintf(heap_buf.data(), heap_buf.size(), fmt, args_copy);
        text = heap_buf.data();
    }
    va_end(args_copy);

    ggml_log_callback cb = g_callback.load(std::memory_order_acquire);
    if (cb) {
        cb(level, text, g_user_data);
        return;
    }
    std::fputs(text, stderr);
}

}

extern "C" PARAKEET_API void parakeet_log_set(ggml_log_callback cb,
                                                        void *            user_data) {
    parakeet::log_set_callback(cb, user_data);
}
