package io.github.ggerganov.whispercpp;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.model.WhisperModelLoader;
import io.github.ggerganov.whispercpp.model.WhisperTokenData;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;

public interface WhisperCppJnaLibrary extends Library {

    WhisperCppJnaLibrary instance = Native.load("whisper", WhisperCppJnaLibrary.class);

    String whisper_print_system_info();

    /**
     * DEPRECATED. Allocate (almost) all memory needed for the model by loading from a file.
     *
     * @param path_model Path to the model file
     * @return Whisper context on success, null on failure
     */
    Pointer whisper_init_from_file(String path_model);

    /**
     * Provides default params which can be used with `whisper_init_from_file_with_params()` etc.
     * Because this function allocates memory for the params, the caller must call either:
     * - call `whisper_free_context_params()`
     * - `Native.free(Pointer.nativeValue(pointer));`
     */
    Pointer whisper_context_default_params_by_ref();

    void whisper_free_context_params(Pointer params);

    /**
     * Allocate (almost) all memory needed for the model by loading from a file.
     *
     * @param path_model Path to the model file
     * @param params     Pointer to whisper_context_params
     * @return Whisper context on success, null on failure
     */
    Pointer whisper_init_from_file_with_params(String path_model, WhisperContextParams.ByValue params);

    /**
     * Allocate (almost) all memory needed for the model by loading from a buffer.
     *
     * @param buffer       Model buffer
     * @param buffer_size  Size of the model buffer
     * @return Whisper context on success, null on failure
     */
    Pointer whisper_init_from_buffer(Pointer buffer, int buffer_size);

    /**
     * Allocate (almost) all memory needed for the model using a model loader.
     *
     * @param loader Model loader
     * @return Whisper context on success, null on failure
     */
    Pointer whisper_init(WhisperModelLoader loader);

    /**
     * Allocate (almost) all memory needed for the model by loading from a file without allocating the state.
     *
     * @param path_model Path to the model file
     * @return Whisper context on success, null on failure
     */
    Pointer whisper_init_from_file_no_state(String path_model);

    /**
     * Allocate (almost) all memory needed for the model by loading from a buffer without allocating the state.
     *
     * @param buffer       Model buffer
     * @param buffer_size  Size of the model buffer
     * @return Whisper context on success, null on failure
     */
    Pointer whisper_init_from_buffer_no_state(Pointer buffer, int buffer_size);

//    Pointer whisper_init_from_buffer_no_state(Pointer buffer, long buffer_size);

    /**
     * Allocate (almost) all memory needed for the model using a model loader without allocating the state.
     *
     * @param loader Model loader
     * @return Whisper context on success, null on failure
     */
    Pointer whisper_init_no_state(WhisperModelLoader loader);

    /**
     * Allocate memory for the Whisper state.
     *
     * @param ctx Whisper context
     * @return Whisper state on success, null on failure
     */
    Pointer whisper_init_state(Pointer ctx);

    /**
     * Free all allocated memory associated with the Whisper context.
     *
     * @param ctx Whisper context
     */
    void whisper_free(Pointer ctx);

    /**
     * Free all allocated memory associated with the Whisper state.
     *
     * @param state Whisper state
     */
    void whisper_free_state(Pointer state);


    /**
     * Convert RAW PCM audio to log mel spectrogram.
     * The resulting spectrogram is stored inside the default state of the provided whisper context.
     *
     * @param ctx - Pointer to a WhisperContext
     * @return 0 on success
     */
    int whisper_pcm_to_mel(Pointer ctx, final float[] samples, int n_samples, int n_threads);

    /**
     * @param ctx Pointer to a WhisperContext
     * @param state Pointer to WhisperState
     * @param n_samples
     * @param n_threads
     * @return 0 on success
     */
    int whisper_pcm_to_mel_with_state(Pointer ctx, Pointer state, final float[] samples, int n_samples, int n_threads);

    /**
     * This can be used to set a custom log mel spectrogram inside the default state of the provided whisper context.
     * Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
     * n_mel must be 80
     * @return 0 on success
     */
    int whisper_set_mel(Pointer ctx, final float[] data, int n_len, int n_mel);
    int whisper_set_mel_with_state(Pointer ctx, Pointer state, final float[] data, int n_len, int n_mel);

    /**
     * Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided whisper context.
     * Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
     * Offset can be used to specify the offset of the first frame in the spectrogram.
     * @return 0 on success
     */
    int whisper_encode(Pointer ctx, int offset, int n_threads);

    int whisper_encode_with_state(Pointer ctx, Pointer state, int offset, int n_threads);

    /**
     * Run the Whisper decoder to obtain the logits and probabilities for the next token.
     * Make sure to call whisper_encode() first.
     * tokens + n_tokens is the provided context for the decoder.
     * n_past is the number of tokens to use from previous decoder calls.
     * Returns 0 on success
     * TODO: add support for multiple decoders
     */
    int whisper_decode(Pointer ctx, Pointer tokens, int n_tokens, int n_past, int n_threads);

    /**
     * @param ctx
     * @param state
     * @param tokens Pointer to int tokens
     * @param n_tokens
     * @param n_past
     * @param n_threads
     * @return
     */
    int whisper_decode_with_state(Pointer ctx, Pointer state, Pointer tokens, int n_tokens, int n_past, int n_threads);

    /**
     * Convert the provided text into tokens.
     * The tokens pointer must be large enough to hold the resulting tokens.
     * Returns the number of tokens on success, no more than n_max_tokens
     * Returns -1 on failure
     * TODO: not sure if correct
     */
    int whisper_tokenize(Pointer ctx, String text, Pointer tokens, int n_max_tokens);

    /** Largest language id (i.e. number of available languages - 1) */
    int whisper_lang_max_id();

    /**
     * @return the id of the specified language, returns -1 if not found.
     * Examples:
     *   "de" -&gt; 2
     *   "german" -&gt; 2
     */
    int whisper_lang_id(String lang);

    /** @return the short string of the specified language id (e.g. 2 -&gt; "de"), returns nullptr if not found */
    String whisper_lang_str(int id);

    /**
     * Use mel data at offset_ms to try and auto-detect the spoken language.
     * Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first
     * Returns the top language id or negative on failure
     * If not null, fills the lang_probs array with the probabilities of all languages
     * The array must be whisper_lang_max_id() + 1 in size
     *
     * ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69
     */
    int whisper_lang_auto_detect(Pointer ctx, int offset_ms, int n_threads, float[] lang_probs);

    int whisper_lang_auto_detect_with_state(Pointer ctx, Pointer state, int offset_ms, int n_threads, float[] lang_probs);

    int whisper_n_len           (Pointer ctx); // mel length
    int whisper_n_len_from_state(Pointer state); // mel length
    int whisper_n_vocab         (Pointer ctx);
    int whisper_n_text_ctx      (Pointer ctx);
    int whisper_n_audio_ctx     (Pointer ctx);
    int whisper_is_multilingual (Pointer ctx);

    int whisper_model_n_vocab      (Pointer ctx);
    int whisper_model_n_audio_ctx  (Pointer ctx);
    int whisper_model_n_audio_state(Pointer ctx);
    int whisper_model_n_audio_head (Pointer ctx);
    int whisper_model_n_audio_layer(Pointer ctx);
    int whisper_model_n_text_ctx   (Pointer ctx);
    int whisper_model_n_text_state (Pointer ctx);
    int whisper_model_n_text_head  (Pointer ctx);
    int whisper_model_n_text_layer (Pointer ctx);
    int whisper_model_n_mels       (Pointer ctx);
    int whisper_model_ftype        (Pointer ctx);
    int whisper_model_type         (Pointer ctx);

    /**
     * Token logits obtained from the last call to whisper_decode().
     * The logits for the last token are stored in the last row
     * Rows: n_tokens
     * Cols: n_vocab
     */
    float[] whisper_get_logits           (Pointer ctx);
    float[] whisper_get_logits_from_state(Pointer state);

    // Token Id -> String. Uses the vocabulary in the provided context
    String whisper_token_to_str(Pointer ctx, int token);
    String whisper_model_type_readable(Pointer ctx);

    // Special tokens
    int whisper_token_eot (Pointer ctx);
    int whisper_token_sot (Pointer ctx);
    int whisper_token_prev(Pointer ctx);
    int whisper_token_solm(Pointer ctx);
    int whisper_token_not (Pointer ctx);
    int whisper_token_beg (Pointer ctx);
    int whisper_token_lang(Pointer ctx, int lang_id);

    // Task tokens
    int whisper_token_translate (Pointer ctx);
    int whisper_token_transcribe(Pointer ctx);

    // Performance information from the default state.
    void whisper_print_timings(Pointer ctx);
    void whisper_reset_timings(Pointer ctx);

    // Note: Even if `whisper_full_params is stripped back to just 4 ints, JNA throws "Invalid memory access"
    //       when `whisper_full_default_params()` tries to return a struct.
    // WhisperFullParams whisper_full_default_params(int strategy);

    /**
     * Provides default params which can be used with `whisper_full()` etc.
     * Because this function allocates memory for the params, the caller must call either:
     * - call `whisper_free_params()`
     * - `Native.free(Pointer.nativeValue(pointer));`
     *
     * @param strategy - WhisperSamplingStrategy.value
     */
    Pointer whisper_full_default_params_by_ref(int strategy);

    void whisper_free_params(Pointer params);

    /**
     * Run the entire model: PCM -&gt; log mel spectrogram -&gt; encoder -&gt; decoder -&gt; text
     * Not thread safe for same context
     * Uses the specified decoding strategy to obtain the text.
     */
    int whisper_full(Pointer ctx, WhisperFullParams.ByValue params, final float[] samples, int n_samples);

    public int whisper_full_with_state(Pointer ctx, Pointer state, WhisperFullParams.ByValue params, float[] samples, int n_samples);
    //int whisper_full_with_state(Pointer ctx, Pointer state, WhisperFullParams params, final float[] samples, int n_samples);

    // Split the input audio in chunks and process each chunk separately using whisper_full_with_state()
    // Result is stored in the default state of the context
    // Not thread safe if executed in parallel on the same context.
    // It seems this approach can offer some speedup in some cases.
    // However, the transcription accuracy can be worse at the beginning and end of each chunk.
    int whisper_full_parallel(Pointer ctx, WhisperFullParams.ByValue params, final float[] samples, int n_samples, int n_processors);

    /**
     * Number of generated text segments.
     * A segment can be a few words, a sentence, or even a paragraph.
     * @param ctx Pointer to WhisperContext
     */
    int whisper_full_n_segments (Pointer ctx);

    /**
     * @param state Pointer to WhisperState
     */
    int whisper_full_n_segments_from_state(Pointer state);

    /**
     * Language id associated with the context's default state.
     * @param ctx Pointer to WhisperContext
     */
    int whisper_full_lang_id(Pointer ctx);

    /** Language id associated with the provided state */
    int whisper_full_lang_id_from_state(Pointer state);


    /** Get the start time of the specified segment. */
    long whisper_full_get_segment_t0(Pointer ctx, int i_segment);

    /** Get the start time of the specified segment from the state. */
    long whisper_full_get_segment_t0_from_state(Pointer state, int i_segment);

    /** Get the end time of the specified segment. */
    long whisper_full_get_segment_t1(Pointer ctx, int i_segment);

    /** Get the end time of the specified segment from the state. */
    long whisper_full_get_segment_t1_from_state(Pointer state, int i_segment);

    /** Get the text of the specified segment. */
    String whisper_full_get_segment_text(Pointer ctx, int i_segment);

    /** Get the text of the specified segment from the state. */
    String whisper_full_get_segment_text_from_state(Pointer state, int i_segment);

    /** Get the number of tokens in the specified segment. */
    int whisper_full_n_tokens(Pointer ctx, int i_segment);

    /** Get the number of tokens in the specified segment from the state. */
    int whisper_full_n_tokens_from_state(Pointer state, int i_segment);

    /** Get the token text of the specified token in the specified segment. */
    String whisper_full_get_token_text(Pointer ctx, int i_segment, int i_token);


    /** Get the token text of the specified token in the specified segment from the state. */
    String whisper_full_get_token_text_from_state(Pointer ctx, Pointer state, int i_segment, int i_token);

    /** Get the token ID of the specified token in the specified segment. */
    int whisper_full_get_token_id(Pointer ctx, int i_segment, int i_token);

    /** Get the token ID of the specified token in the specified segment from the state. */
    int whisper_full_get_token_id_from_state(Pointer state, int i_segment, int i_token);

    /** Get token data for the specified token in the specified segment. */
    WhisperTokenData whisper_full_get_token_data(Pointer ctx, int i_segment, int i_token);

    /** Get token data for the specified token in the specified segment from the state. */
    WhisperTokenData whisper_full_get_token_data_from_state(Pointer state, int i_segment, int i_token);

    /** Get the probability of the specified token in the specified segment. */
    float whisper_full_get_token_p(Pointer ctx, int i_segment, int i_token);

    /** Get the probability of the specified token in the specified segment from the state. */
    float whisper_full_get_token_p_from_state(Pointer state, int i_segment, int i_token);

    /**
     * Benchmark function for memcpy.
     *
     * @param nThreads Number of threads to use for the benchmark.
     * @return The result of the benchmark.
     */
    int whisper_bench_memcpy(int nThreads);

    /**
     * Benchmark function for memcpy as a string.
     *
     * @param nThreads Number of threads to use for the benchmark.
     * @return The result of the benchmark as a string.
     */
    String whisper_bench_memcpy_str(int nThreads);

    /**
     * Benchmark function for ggml_mul_mat.
     *
     * @param nThreads Number of threads to use for the benchmark.
     * @return The result of the benchmark.
     */
    int whisper_bench_ggml_mul_mat(int nThreads);

    /**
     * Benchmark function for ggml_mul_mat as a string.
     *
     * @param nThreads Number of threads to use for the benchmark.
     * @return The result of the benchmark as a string.
     */
    String whisper_bench_ggml_mul_mat_str(int nThreads);
}
