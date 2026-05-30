// Wrapper of the experimental Core ML Whisper Decoder model

#include <stdbool.h>
#include <stdint.h>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_decoder_context;

struct whisper_coreml_decoder_trace {
    bool    enabled;
    int64_t state_pos;
    int64_t cross_kv_write_count;
    int64_t cross_kv_bytes_written;
    int64_t cross_kv_write_us;
    int64_t self_kv_write_count;
    int64_t self_kv_bytes_written;
    int64_t self_kv_write_us;
    int64_t mlstate_create_us;
    int64_t input_array_create_us;
    int64_t feature_provider_create_us;
    int64_t prediction_us;
    int64_t logits_copy_us;
    int64_t prompt_step_count;
    int64_t generation_step_count;
};

struct whisper_coreml_decoder_context * whisper_coreml_decoder_init(const char * path_model);
void whisper_coreml_decoder_free(struct whisper_coreml_decoder_context * ctx);
void whisper_coreml_decoder_reset(struct whisper_coreml_decoder_context * ctx);
bool whisper_coreml_decoder_is_stateful(const struct whisper_coreml_decoder_context * ctx);
bool whisper_coreml_decoder_uses_audio_input(const struct whisper_coreml_decoder_context * ctx);
int64_t whisper_coreml_decoder_state_pos(const struct whisper_coreml_decoder_context * ctx);
void whisper_coreml_decoder_set_state_pos(struct whisper_coreml_decoder_context * ctx, int64_t state_pos);
bool whisper_coreml_decoder_set_state_f16(struct whisper_coreml_decoder_context * ctx, const char * name, const void * data, int64_t n_elems);
bool whisper_coreml_decoder_trace_enabled(const struct whisper_coreml_decoder_context * ctx);
void whisper_coreml_decoder_trace_reset(struct whisper_coreml_decoder_context * ctx);
void whisper_coreml_decoder_trace_get(const struct whisper_coreml_decoder_context * ctx, struct whisper_coreml_decoder_trace * trace);

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
