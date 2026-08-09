#include "whisper.h"

#include <cstdio>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context * ctx = whisper_init_from_file_with_params(WHISPER_MODEL_PATH, cparams);
    assert(ctx != nullptr);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.no_timestamps = true;
    params.print_progress = false;
    params.print_realtime = false;

    const int rc = whisper_full(ctx, params, nullptr, 0);
    assert(rc == 0);
    assert(whisper_full_n_segments(ctx) == 0);

    whisper_free(ctx);

    printf("test-whisper-zero-samples: OK\n");
    return 0;
}
