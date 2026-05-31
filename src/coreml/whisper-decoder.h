// Wrapper of the Core ML Whisper decoder model

#pragma once

#include <stdbool.h>
#include <stdint.h>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_decoder_context;

struct whisper_coreml_decoder_context * whisper_coreml_decoder_init(const char * path_model);
void whisper_coreml_decoder_free(struct whisper_coreml_decoder_context * ctx);
void whisper_coreml_decoder_reset(struct whisper_coreml_decoder_context * ctx);

const char * whisper_coreml_decoder_compute_units_name(const struct whisper_coreml_decoder_context * ctx);

int64_t whisper_coreml_decoder_state_pos(const struct whisper_coreml_decoder_context * ctx);
bool whisper_coreml_decoder_set_state_f16(struct whisper_coreml_decoder_context * ctx, const char * name, const void * data, int64_t n_elems);

bool whisper_coreml_decoder_decode(
        struct whisper_coreml_decoder_context * ctx,
                                      int64_t   n_tokens,
                                      int64_t   n_vocab,
                                      int64_t   n_audio_ctx,
                                      int64_t   n_audio_state,
                               const int32_t * tokens,
                                 const float * audio,
                                       float * out_logits,
                                         bool   is_prompt);

#if __cplusplus
}
#endif
