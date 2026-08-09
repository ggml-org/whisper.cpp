// Regression test for issue #3978:
// whisper_full() called with n_samples == 0 on a fresh state must not read the
// never-computed (previously uninitialized) whisper_mel fields. With the mel
// default-initialized to "0 frames", the call takes the too-short path and
// returns cleanly with zero segments instead of running the encoder on garbage
// dimensions (which could dereference a NULL mel buffer).

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

    // n_samples == 0 with a fresh state: the mel is never computed.
    const int rc = whisper_full(ctx, params, nullptr, 0);
    assert(rc == 0);
    assert(whisper_full_n_segments(ctx) == 0);

    whisper_free(ctx);

    printf("test-whisper-zero-samples: OK\n");
    return 0;
}
