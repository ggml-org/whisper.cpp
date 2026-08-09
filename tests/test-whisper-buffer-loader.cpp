// Regression test for issue #3879 (bug 1):
// The buffer-based model loader's read callback passed buf->buffer + offset to
// memcpy even when nothing was left to copy. For an empty buffer that source is
// NULL, and passing a NULL pointer to memcpy is undefined behavior even for a
// zero-length copy. Loading from a null/empty or truncated buffer must fail
// gracefully (return NULL) without invoking that UB.

#include "whisper.h"

#include <cstdint>
#include <cstdio>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    // Empty buffer: the read callback is driven with a NULL source pointer.
    struct whisper_context * ctx_empty = whisper_init_from_buffer_with_params(nullptr, 0, cparams);
    assert(ctx_empty == nullptr);

    // Truncated, non-model buffer: the loader runs out of bytes mid-read.
    uint8_t truncated[8] = { 0 };
    struct whisper_context * ctx_trunc = whisper_init_from_buffer_with_params(truncated, sizeof(truncated), cparams);
    assert(ctx_trunc == nullptr);

    printf("test-whisper-buffer-loader: OK\n");
    return 0;
}
