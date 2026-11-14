// Wrapper of the RKNN Whisper encoder model.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_rknn_context;
struct ggml_tensor;

// Initialize the base.en 30s encoder from a .rknn file; null on failure.
struct whisper_rknn_context * whisper_rknn_init(const char * path_model);

// Clean up a context previously returned by whisper_rknn_init().
void whisper_rknn_free(struct whisper_rknn_context * ctx);

// Encode float32 mel [3000,80] into float32 embeddings [512,1500].
// Returns 1 on success, 0 on failure.
int whisper_rknn_encode(
    struct whisper_rknn_context * ctx,
    struct ggml_tensor * mel,
    struct ggml_tensor * out);

#ifdef __cplusplus
}
#endif
