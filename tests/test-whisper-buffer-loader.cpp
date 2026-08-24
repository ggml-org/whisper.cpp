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

    struct whisper_context * ctx_empty = whisper_init_from_buffer_with_params(nullptr, 1, cparams);
    assert(ctx_empty == nullptr);

    uint8_t truncated[8] = { 0 };
    struct whisper_context * ctx_trunc = whisper_init_from_buffer_with_params(truncated, sizeof(truncated), cparams);
    assert(ctx_trunc == nullptr);

    printf("test-whisper-buffer-loader: OK\n");
    return 0;
}
