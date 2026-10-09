#include "whisper.h"

#include <cstdio>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

static int n_encoder_begin = 0;

static bool encoder_begin_cb(struct whisper_context *, struct whisper_state *, void *) {
    n_encoder_begin++;
    return true; // don't block anything, just count
}

static bool abort_cb(void *) {
    return true; // abort immediately
}

int main() {
    ggml_backend_load_all();

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context * ctx = whisper_init_from_file_with_params(WHISPER_MODEL_PATH, cparams);
    assert(ctx != nullptr);

    std::vector<float> pcmf32(2*WHISPER_SAMPLE_RATE, 0.0f);   // 2 s of silence

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language       = "auto";
    params.print_progress = false;
    params.print_realtime = false;

    params.encoder_begin_callback = encoder_begin_cb;
    params.abort_callback         = abort_cb;

    const int rc = whisper_full(ctx, params, pcmf32.data(), pcmf32.size());

    assert(rc != 0);                // the call must fail because we aborted
    assert(n_encoder_begin == 1);   // aborted during auto-detect: main-loop encoder never started

    whisper_free(ctx);

    printf("test-whisper-lang-detect-abort: OK\n");
    return 0;
}
