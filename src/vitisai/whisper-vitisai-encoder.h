#pragma once

#include <cstddef>
#include <cstdbool>
#include <cstdint>

#if __cplusplus
extern "C" {
#endif

struct whisper_vitisai_context;

struct whisper_vitisai_context * whisper_vitisai_init(const char * path_model);
void whisper_vitisai_free(struct whisper_vitisai_context * ctx);

struct ggml_tensor;

int whisper_vitisai_encode(
    struct whisper_vitisai_context * ctx,
    struct ggml_tensor * mel,
    struct ggml_tensor * out);

#if __cplusplus
}
#endif
