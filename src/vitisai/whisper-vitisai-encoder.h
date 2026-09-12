#pragma once

#include <cstdbool>

#if __cplusplus
extern "C" {
#endif

struct whisper_vitisai_context;

struct whisper_vitisai_context * whisper_vitisai_init(const char * path_model);
void whisper_vitisai_free(struct whisper_vitisai_context * ctx);
bool whisper_vitisai_has_cross_proj(const struct whisper_vitisai_context * ctx);

struct ggml_tensor;

int whisper_vitisai_encode(
    struct whisper_vitisai_context * ctx,
    struct ggml_tensor * mel,
    struct ggml_tensor * out);

int whisper_vitisai_encode_with_cross(
    struct whisper_vitisai_context * ctx,
    struct ggml_tensor * mel,
    struct ggml_tensor * embd_enc,
    struct ggml_tensor * kv_cross_k,
    struct ggml_tensor * kv_cross_v,
    int n_text_layer,
    int n_ctx,
    int n_text_state,
    int n_text_head,
    bool flash_attn);

#if __cplusplus
}
#endif
