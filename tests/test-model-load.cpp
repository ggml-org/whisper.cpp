#include "whisper.h"

#include <cstdio>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

// A crafted legacy-format .bin whose first tensor record declares n_dims > 4
// must be rejected by whisper_model_load rather than overflowing the fixed
// 4-element ne[] stack array (CWE-121, #3944). Loading it must fail cleanly
// (return NULL), not crash.
static int test_invalid_n_dims_model_load() {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context * ctx =
        whisper_init_from_file_with_params_no_state(WHISPER_BAD_NDIMS_MODEL_PATH, cparams);
    if (ctx != nullptr) {
        fprintf(stderr, "expected model with n_dims > 4 to fail loading\n");
        whisper_free(ctx);
        return 1;
    }
    return 0;
}

int main() {
    if (test_invalid_n_dims_model_load() != 0) {
        return 1;
    }
    printf("test-model-load: crafted n_dims > 4 model rejected as expected\n");
    return 0;
}
