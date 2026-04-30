#ifndef PARAKEET_H
#define PARAKEET_H

#include "ggml.h"
#include "ggml-cpu.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __GNUC__
#    define PARAKEET_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define PARAKEET_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define PARAKEET_DEPRECATED(func, hint) func
#endif

#ifdef PARAKEET_SHARED
#    ifdef _WIN32
#        ifdef PARAKEET_BUILD
#            define PARAKEET_API __declspec(dllexport)
#        else
#            define PARAKEET_API __declspec(dllimport)
#        endif
#    else
#        define PARAKEET_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define PARAKEET_API
#endif

#define PARAKEET_SAMPLE_RATE 16000
#define PARAKEET_HOP_LENGTH  160

#ifdef __cplusplus
extern "C" {
#endif

    struct parakeet_context;
    struct parakeet_state;
    struct parakeet_full_params;

    typedef int32_t parakeet_pos;
    typedef int32_t parakeet_token;
    typedef int32_t parakeet_seq_id;

    struct parakeet_context_params {
        bool  use_gpu;
        bool  flash_attn;
        int   gpu_device;  // CUDA device
    };

    typedef struct parakeet_token_data {
        parakeet_token id;  // the BPE subword ID (0-8191)

        int duration_idx;   // index into the models durations array
        int duration_value; // actual duration value
        int frame_index;

        float p;
        float plog;

        int64_t t0;
        int64_t t1;

        bool is_word_start;
    } parakeet_token_data;

    typedef struct parakeet_model_loader {
        void * context;

        size_t (*read)(void * ctx, void * output, size_t read_size);
        bool    (*eof)(void * ctx);
        void  (*close)(void * ctx);
    } parakeet_model_loader;

    PARAKEET_API const char * parakeet_version(void);

    // Various functions for loading a ggml parakeet model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    PARAKEET_API struct parakeet_context * parakeet_init_from_file_with_params  (const char * path_model,              struct parakeet_context_params params);
    PARAKEET_API struct parakeet_context * parakeet_init_from_buffer_with_params(void * buffer, size_t buffer_size,    struct parakeet_context_params params);
    PARAKEET_API struct parakeet_context * parakeet_init_with_params            (struct parakeet_model_loader * loader, struct parakeet_context_params params);

    // These are the same as the above, but the internal state of the context is not allocated automatically
    // It is the responsibility of the caller to allocate the state using parakeet_init_state() (#523)
    PARAKEET_API struct parakeet_context * parakeet_init_from_file_with_params_no_state  (const char * path_model,              struct parakeet_context_params params);
    PARAKEET_API struct parakeet_context * parakeet_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size,    struct parakeet_context_params params);
    PARAKEET_API struct parakeet_context * parakeet_init_with_params_no_state            (struct parakeet_model_loader * loader, struct parakeet_context_params params);

    PARAKEET_API struct parakeet_state * parakeet_init_state(struct parakeet_context * ctx);

    // Frees all allocated memory
    PARAKEET_API void parakeet_free      (struct parakeet_context * ctx);
    PARAKEET_API void parakeet_free_state(struct parakeet_state * state);
    PARAKEET_API void parakeet_free_params(struct parakeet_full_params * params);
    PARAKEET_API void parakeet_free_context_params(struct parakeet_context_params * params);

    // Convert RAW PCM audio to log mel spectrogram.
    // The resulting spectrogram is stored inside the default state of the provided parakeet context.
    // Returns 0 on success
    PARAKEET_API int parakeet_pcm_to_mel(
            struct parakeet_context * ctx,
                        const float * samples,
                                int   n_samples,
                                int   n_threads);

    PARAKEET_API int parakeet_pcm_to_mel_with_state(
            struct parakeet_context * ctx,
              struct parakeet_state * state,
                        const float * samples,
                                int   n_samples,
                                int   n_threads);

    // This can be used to set a custom log mel spectrogram inside the default state of the provided parakeet context.
    // Use this instead of parakeet_pcm_to_mel() if you want to provide your own log mel spectrogram.
    // n_mel must be 80
    // Returns 0 on success
    PARAKEET_API int parakeet_set_mel(
            struct parakeet_context * ctx,
                        const float * data,
                                int   n_len,
                                int   n_mel);

    PARAKEET_API int parakeet_set_mel_with_state(
            struct parakeet_context * ctx,
              struct parakeet_state * state,
                        const float * data,
                                int   n_len,
                                int   n_mel);

    // Run the Parakeet encoder on the log mel spectrogram stored inside the default state in the provided parakeet context.
    // Make sure to call parakeet_pcm_to_mel() or parakeet_set_mel() first.
    // offset can be used to specify the offset of the first frame in the spectrogram.
    // Returns 0 on success
    PARAKEET_API int parakeet_encode(
            struct parakeet_context * ctx,
                                int   offset,
                                int   n_threads);

    PARAKEET_API int parakeet_encode_with_state(
            struct parakeet_context * ctx,
              struct parakeet_state * state,
                                int   offset,
                                int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    PARAKEET_API int parakeet_tokenize(
            struct parakeet_context * ctx,
                        const char * text,
                     parakeet_token * tokens,
                               int   n_max_tokens);

    // Return the number of tokens in the provided text
    // Equivalent to: -parakeet_tokenize(ctx, text, NULL, 0)
    int parakeet_token_count(struct parakeet_context * ctx, const char * text);

    PARAKEET_API int parakeet_n_len           (struct parakeet_context * ctx); // mel length
    PARAKEET_API int parakeet_n_len_from_state(struct parakeet_state * state); // mel length
    PARAKEET_API int parakeet_n_vocab         (struct parakeet_context * ctx);
    PARAKEET_API int parakeet_n_audio_ctx     (struct parakeet_context * ctx);

    PARAKEET_API int parakeet_model_n_vocab      (struct parakeet_context * ctx);
    PARAKEET_API int parakeet_model_n_audio_ctx  (struct parakeet_context * ctx);
    PARAKEET_API int parakeet_model_n_audio_state(struct parakeet_context * ctx);
    PARAKEET_API int parakeet_model_n_audio_head (struct parakeet_context * ctx);
    PARAKEET_API int parakeet_model_n_audio_layer(struct parakeet_context * ctx);
    PARAKEET_API int parakeet_model_n_mels       (struct parakeet_context * ctx);
    PARAKEET_API int parakeet_model_ftype        (struct parakeet_context * ctx);

    // Token logits obtained from the last call to parakeet_full/parakeet_chunk
    // The logits for the last token are stored in the last row
    // Rows: n_tokens
    // Cols: n_vocab
    PARAKEET_API float * parakeet_get_logits           (struct parakeet_context * ctx);
    PARAKEET_API float * parakeet_get_logits_from_state(struct parakeet_state * state);

    // Token Id -> String. Uses the vocabulary in the provided context
    PARAKEET_API const char * parakeet_token_to_str(struct parakeet_context * ctx, parakeet_token token);

    PARAKEET_API int parakeet_token_to_text(const char * token_str, bool is_first, char * output, int max_len);

    // Special tokens
    PARAKEET_API parakeet_token parakeet_token_blank (struct parakeet_context * ctx);
    PARAKEET_API parakeet_token parakeet_token_unk (struct parakeet_context * ctx);
    PARAKEET_API parakeet_token parakeet_token_bos(struct parakeet_context * ctx);

    // Performance information from the default state.
    struct parakeet_timings {
        float sample_ms;
        float encode_ms;
        float decode_ms;
        float batchd_ms;
        float prompt_ms;
    };
    PARAKEET_API struct parakeet_timings * parakeet_get_timings(struct parakeet_context * ctx);
    PARAKEET_API void parakeet_print_timings(struct parakeet_context * ctx);
    PARAKEET_API void parakeet_reset_timings(struct parakeet_context * ctx);

    // Print system information
    PARAKEET_API const char * parakeet_print_system_info(void);

    // Available sampling strategies
    enum parakeet_sampling_strategy {
        PARAKEET_SAMPLING_GREEDY,
    };

    // Token callback.
    // Called for each new predicted token.
    // Use the parakeet_full_...() functions to obtain the text segments
    typedef void (*parakeet_new_token_callback)(
            struct parakeet_context * ctx,
              struct parakeet_state * state,
          const parakeet_token_data * token_data,
                              void * user_data);

    // Text segment callback
    // Called on every newly generated text segment
    // Use the parakeet_full_...() functions to obtain the text segments
    typedef void (*parakeet_new_segment_callback)(struct parakeet_context * ctx, struct parakeet_state * state, int n_new, void * user_data);

    // Progress callback
    typedef void (*parakeet_progress_callback)(struct parakeet_context * ctx, struct parakeet_state * state, int progress, void * user_data);

    // Encoder begin callback
    // If not NULL, called before the encoder starts
    // If it returns false, the computation is aborted
    typedef bool (*parakeet_encoder_begin_callback)(struct parakeet_context * ctx, struct parakeet_state * state, void * user_data);

    // Parameters for the parakeet_full() function
    // If you change the order or add new parameters, make sure to update the default values in parakeet.cpp:
    // parakeet_full_default_params()
    struct parakeet_full_params {
        enum parakeet_sampling_strategy strategy;

        int n_threads;
        int offset_ms;          // start offset in ms
        int duration_ms;        // audio duration to process in ms

        bool no_context;        // do not use past transcription (if any) as context

        // [EXPERIMENTAL] speed-up techniques
        int  audio_ctx;         // overwrite the audio context size (0 = use default)

        int  chunk_length_ms;  // length of each chunk in ms
        int  left_context_ms;  // left context in ms
        int  right_context_ms; // right context in ms

        // called for every newly generated text segment
        parakeet_new_segment_callback new_segment_callback;
        void * new_segment_callback_user_data;

        // called for every newly generated token
        parakeet_new_token_callback new_token_callback;
        void * new_token_callback_user_data;

        // called on each progress update
        parakeet_progress_callback progress_callback;
        void * progress_callback_user_data;

        // called each time before the encoder starts
        parakeet_encoder_begin_callback encoder_begin_callback;
        void * encoder_begin_callback_user_data;

        // called each time before ggml computation starts
        ggml_abort_callback abort_callback;
        void * abort_callback_user_data;
    };

    // NOTE: this function allocates memory, and it is the responsibility of the caller to free the pointer - see parakeet_free_context_params() & parakeet_free_params()
    PARAKEET_API struct parakeet_context_params * parakeet_context_default_params_by_ref(void);
    PARAKEET_API struct parakeet_context_params   parakeet_context_default_params       (void);

    PARAKEET_API struct parakeet_full_params * parakeet_full_default_params_by_ref(enum parakeet_sampling_strategy strategy);
    PARAKEET_API struct parakeet_full_params   parakeet_full_default_params       (enum parakeet_sampling_strategy strategy);

    // Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
    // Not thread safe for same context
    PARAKEET_API int parakeet_full(
                struct parakeet_context * ctx,
            struct parakeet_full_params   params,
                            const float * samples,
                                    int   n_samples);

    PARAKEET_API int parakeet_full_with_state(
                struct parakeet_context * ctx,
                  struct parakeet_state * state,
            struct parakeet_full_params   params,
                            const float * samples,
                                    int   n_samples);

    // Process a single chunk of audio data that fits within the model's audio context window.
    // This is more efficient than parakeet_full() for short audio clips.
    PARAKEET_API int parakeet_chunk(
                struct parakeet_context * ctx,
                  struct parakeet_state * state,
            struct parakeet_full_params   params,
                            const float * samples,
                                   int    n_samples);

    // Initialize streaming state for a new stream.
    PARAKEET_API int parakeet_stream_init(
                struct parakeet_context * ctx,
                  struct parakeet_state * state,
            struct parakeet_full_params   params);

    // Push audio samples in streaming mode. Internally this function will structure
    // the samples in a buffer where with a left context, a center chunk, and a
    // right context. The encoder will see the complete buffer which enables it
    // to get boundry context for the target/center audio chunk. This avoids hard
    // cut offs at the chunk boundaries. The joint network then only sees the
    // center chunk and this function internally handles the context windowing.
    PARAKEET_API int parakeet_stream_push(
                struct parakeet_context * ctx,
                  struct parakeet_state * state,
                            const float * samples,
                                    int   n_samples);

    // Flush the final partial chunk at end-of-stream.
    PARAKEET_API int parakeet_stream_flush(
                struct parakeet_context * ctx,
                  struct parakeet_state * state);

    // Number of generated text segments
    PARAKEET_API int parakeet_full_n_segments           (struct parakeet_context * ctx);
    PARAKEET_API int parakeet_full_n_segments_from_state(struct parakeet_state * state);

    // Get the start and end time of the specified segment
    PARAKEET_API int64_t parakeet_full_get_segment_t0           (struct parakeet_context * ctx, int i_segment);
    PARAKEET_API int64_t parakeet_full_get_segment_t0_from_state(struct parakeet_state * state, int i_segment);

    PARAKEET_API int64_t parakeet_full_get_segment_t1           (struct parakeet_context * ctx, int i_segment);
    PARAKEET_API int64_t parakeet_full_get_segment_t1_from_state(struct parakeet_state * state, int i_segment);

    // Get the text of the specified segment
    PARAKEET_API const char * parakeet_full_get_segment_text           (struct parakeet_context * ctx, int i_segment);
    PARAKEET_API const char * parakeet_full_get_segment_text_from_state(struct parakeet_state * state, int i_segment);

    // Get number of tokens in the specified segment
    PARAKEET_API int parakeet_full_n_tokens           (struct parakeet_context * ctx, int i_segment);
    PARAKEET_API int parakeet_full_n_tokens_from_state(struct parakeet_state * state, int i_segment);

    // Get the token text of the specified token in the specified segment
    PARAKEET_API const char * parakeet_full_get_token_text           (struct parakeet_context * ctx, int i_segment, int i_token);
    PARAKEET_API const char * parakeet_full_get_token_text_from_state(struct parakeet_context * ctx, struct parakeet_state * state, int i_segment, int i_token);

    // Get the token id of the specified token in the specified segment
    PARAKEET_API parakeet_token parakeet_full_get_token_id           (struct parakeet_context * ctx, int i_segment, int i_token);
    PARAKEET_API parakeet_token parakeet_full_get_token_id_from_state(struct parakeet_state * state, int i_segment, int i_token);

    // Get token data for the specified token in the specified segment
    PARAKEET_API parakeet_token_data parakeet_full_get_token_data           (struct parakeet_context * ctx, int i_segment, int i_token);
    PARAKEET_API parakeet_token_data parakeet_full_get_token_data_from_state(struct parakeet_state * state, int i_segment, int i_token);

    // Get the probability of the specified token in the specified segment
    PARAKEET_API float parakeet_full_get_token_p           (struct parakeet_context * ctx, int i_segment, int i_token);
    PARAKEET_API float parakeet_full_get_token_p_from_state(struct parakeet_state * state, int i_segment, int i_token);

    // Control logging output; default behavior is to print to stderr

    PARAKEET_API void parakeet_log_set(ggml_log_callback log_callback, void * user_data);

#ifdef __cplusplus
}
#endif

#endif
