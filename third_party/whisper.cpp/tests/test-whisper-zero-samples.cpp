#include "whisper.h"

#include <cstdio>
#include <cstring>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    ggml_backend_load_all();

    // QVAC (see PATCHES.md): backend metadata is a selection diagnostic, not
    // evidence of graph execution or transcript accuracy. Reuse this existing
    // fixture load instead of allocating another full model state.
    assert(whisper_get_primary_backend_name(nullptr) == nullptr);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context * ctx = whisper_init_from_file_with_params(WHISPER_MODEL_PATH, cparams);
    assert(ctx != nullptr);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    assert(cpu != nullptr);
    const char * primary = whisper_get_primary_backend_name(ctx);
    assert(primary != nullptr);
    bool has_accel = false;
    bool primary_is_cpu_or_accel = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        has_accel = has_accel || type == GGML_BACKEND_DEVICE_TYPE_ACCEL;
        if (std::strcmp(primary, ggml_backend_dev_name(dev)) == 0) {
            primary_is_cpu_or_accel = primary_is_cpu_or_accel ||
                type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_ACCEL;
        }
    }
    // Existing automatic ACCEL initialization remains enabled in CPU mode;
    // GPU/IGPU selection is disabled. CPU-only lanes pin the exact CPU name.
    assert(primary_is_cpu_or_accel);
    if (!has_accel) {
        assert(std::strcmp(primary, ggml_backend_dev_name(cpu)) == 0);
    }

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.no_timestamps = true;
    params.print_progress = false;
    params.print_realtime = false;

    const int rc = whisper_full(ctx, params, nullptr, 0);
    assert(rc == 0);
    assert(whisper_full_n_segments(ctx) == 0);

    whisper_free(ctx);

    // A weight-only context has no default state/backends to report. The
    // committed fixture has no trained weights and does not test accuracy.
    ctx = whisper_init_from_file_with_params_no_state(WHISPER_MODEL_PATH, cparams);
    assert(ctx != nullptr);
    assert(whisper_get_primary_backend_name(ctx) == nullptr);
    whisper_free(ctx);

    printf("test-whisper-zero-samples: OK\n");
    return 0;
}
