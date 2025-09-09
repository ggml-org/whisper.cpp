package io.github.ggerganov.whispercpp.params;

import com.sun.jna.*;
import io.github.ggerganov.whispercpp.callbacks.WhisperEncoderBeginCallback;
import io.github.ggerganov.whispercpp.callbacks.WhisperLogitsFilterCallback;
import io.github.ggerganov.whispercpp.callbacks.WhisperNewSegmentCallback;
import io.github.ggerganov.whispercpp.callbacks.WhisperProgressCallback;
import io.github.ggerganov.whispercpp.callbacks.GgmlAbortCallback;

import java.util.Arrays;
import java.util.List;

/**
 * Parameters for the whisper_full() function.
 * If you change the order or add new parameters, make sure to update the default values in whisper.cpp:
 * whisper_full_default_params()
 */
public class WhisperFullParams extends Structure {

    public WhisperFullParams() {
        super();
    }

    public WhisperFullParams(Pointer p) {
        super(p);
    }

    /** Sampling strategy for whisper_full() function. */
    public int strategy;

    /** Number of threads. (default = 4) */
    public int n_threads;

    /** Maximum tokens to use from past text as a prompt for the decoder. (default = 16384) */
    public int n_max_text_ctx;

    /** Start offset in milliseconds. (default = 0) */
    public int offset_ms;

    /** Audio duration to process in milliseconds. (default = 0) */
    public int duration_ms;

    /** Translate flag. (default = false) */
    public CBool translate;

    /** The compliment of translateMode() */
    public void transcribeMode() {
        translate = CBool.FALSE;
    }

    /** The compliment of transcribeMode() */
    public void translateMode() {
        translate = CBool.TRUE;
    }

    /** Flag to indicate whether to use past transcription (if any) as an initial prompt for the decoder. (default = true) */
    public CBool no_context;

    /** Flag to indicate whether to use past transcription (if any) as an initial prompt for the decoder. (default = true) */
    public void enableContext(boolean enable) {
        no_context = enable ? CBool.FALSE : CBool.TRUE;
    }

    /** Generate timestamps or not? */
    public CBool no_timestamps;

    /** Flag to force single segment output (useful for streaming). (default = false) */
    public CBool single_segment;

    /** Flag to force single segment output (useful for streaming). (default = false) */
    public void singleSegment(boolean single) {
        single_segment = single ? CBool.TRUE : CBool.FALSE;
    }

    /** Flag to print special tokens (e.g., &lt;SOT&gt;, &lt;EOT&gt;, &lt;BEG&gt;, etc.). (default = false) */
    public CBool print_special;

    /** Flag to print special tokens (e.g., &lt;SOT&gt;, &lt;EOT&gt;, &lt;BEG&gt;, etc.). (default = false) */
    public void printSpecial(boolean enable) {
        print_special = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Flag to print progress information. (default = true) */
    public CBool print_progress;

    /** Flag to print progress information. (default = true) */
    public void printProgress(boolean enable) {
        print_progress = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Flag to print results from within whisper.cpp (avoid it, use callback instead). (default = true) */
    public CBool print_realtime;

    /** Flag to print results from within whisper.cpp (avoid it, use callback instead). (default = true) */
    public void printRealtime(boolean enable) {
        print_realtime = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Flag to print timestamps for each text segment when printing realtime. (default = true) */
    public CBool print_timestamps;

    /** Flag to print timestamps for each text segment when printing realtime. (default = true) */
    public void printTimestamps(boolean enable) {
        print_timestamps = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** [EXPERIMENTAL] Flag to enable token-level timestamps. (default = false) */
    public CBool token_timestamps;

    /** [EXPERIMENTAL] Flag to enable token-level timestamps. (default = false) */
    public void tokenTimestamps(boolean enable) {
        token_timestamps = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** [EXPERIMENTAL] Timestamp token probability threshold (~0.01). (default = 0.01) */
    public float thold_pt;

    /** [EXPERIMENTAL] Timestamp token sum probability threshold (~0.01). */
    public float thold_ptsum;

    /** Maximum segment length in characters. (default = 0) */
    public int max_len;

    /** Flag to split on word rather than on token (when used with max_len). (default = false) */
    public CBool split_on_word;

    /** Flag to split on word rather than on token (when used with max_len). (default = false) */
    public void splitOnWord(boolean enable) {
        split_on_word = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Maximum tokens per segment (0, default = no limit) */
    public int max_tokens;

    /** [EXPERIMENTAL] Enable debug mode for extra info */
    public CBool debug_mode;

    /** Enable debug mode */
    public void enableDebugMode(boolean enable) {
        debug_mode = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Overwrite the audio context size (0 = use default). */
    public int audio_ctx;

    /** Enable tinydiarize (default = false) */
    public CBool tdrz_enable;

    /** Enable tinydiarize (default = false) */
    public void tdrzEnable(boolean enable) {
        tdrz_enable = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Regular expression matching tokens to suppress. */
    public String suppress_regex;

    /** Tokens to provide to the whisper decoder as an initial prompt.
     * These are prepended to any existing text context from a previous call. */
    public String initial_prompt;
    /** Always prepend initial_prompt for every decode chunk. */
    public CBool carry_initial_prompt;

    /** Prompt tokens. (int*) */
    public Pointer prompt_tokens;

    public void setPromptTokens(int[] tokens) {
        Memory mem = new Memory(tokens.length * 4L);
        mem.write(0, tokens, 0, tokens.length);
        prompt_tokens = mem;
    }

    /** Number of prompt tokens. */
    public int prompt_n_tokens;

    /** Language for auto-detection.
     * For auto-detection, set to `null`, `""`, or "auto". */
    public String language;

    /** Flag to indicate whether to detect language automatically. */
    public CBool detect_language;

    /** Flag to indicate whether to detect language automatically. */
    public void detectLanguage(boolean enable) {
        detect_language = enable ? CBool.TRUE : CBool.FALSE;
    }

    // Common decoding parameters.

    /** Flag to suppress blank tokens. */
    public CBool suppress_blank;

    public void suppressBlanks(boolean enable) {
        suppress_blank = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Flag to suppress non-speech tokens. */
    public CBool suppress_nst;

    /** Flag to suppress non-speech tokens. */
    public void suppressNonSpeechTokens(boolean enable) {
        suppress_nst = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Initial decoding temperature. */
    public float temperature;

    /** Maximum initial timestamp. */
    public float max_initial_ts;

    /** Length penalty. */
    public float length_penalty;

    // Fallback parameters.

    /** Temperature increment. */
    public float temperature_inc;

    /** Entropy threshold (similar to OpenAI's "compression_ratio_threshold"). */
    public float entropy_thold;

    /** Log probability threshold. */
    public float logprob_thold;

    /** No speech threshold. */
    public float no_speech_thold;

    /** Greedy decoding parameters. */
    public GreedyParams greedy;

    /**
     * Beam search decoding parameters.
     */
    public BeamSearchParams beam_search;

    public void setBestOf(int bestOf) {
        if (greedy == null) {
            greedy = new GreedyParams();
        }
        greedy.best_of = bestOf;
    }

    public void setBeamSize(int beamSize) {
        if (beam_search == null) {
            beam_search = new BeamSearchParams();
        }
        beam_search.beam_size = beamSize;
    }

    public void setBeamSizeAndPatience(int beamSize, float patience) {
        if (beam_search == null) {
            beam_search = new BeamSearchParams();
        }
        beam_search.beam_size = beamSize;
        beam_search.patience = patience;
    }

    /**
     * Callback for every newly generated text segment.
     * WhisperNewSegmentCallback
     */
    public Pointer new_segment_callback;

    /**
     * User data for the new_segment_callback.
     */
    public Pointer new_segment_callback_user_data;

    /**
     * Callback on each progress update.
     * WhisperProgressCallback
     */
    public Pointer progress_callback;

    /**
     * User data for the progress_callback.
     */
    public Pointer progress_callback_user_data;

    /**
     * Callback each time before the encoder starts.
     * WhisperEncoderBeginCallback
     */
    public Pointer encoder_begin_callback;

    /**
     * User data for the encoder_begin_callback.
     */
    public Pointer encoder_begin_callback_user_data;

    /** Callback used to abort GGML computation */
    public Pointer abort_callback;

    /** User data for the abort_callback */
    public Pointer abort_callback_user_data;

    public void setAbortCallback(GgmlAbortCallback callback) {
        abort_callback = CallbackReference.getFunctionPointer(callback);
    }

    /**
     * Callback by each decoder to filter obtained logits.
     * WhisperLogitsFilterCallback
     */
    public Pointer logits_filter_callback;

    /**
     * User data for the logits_filter_callback.
     */
    public Pointer logits_filter_callback_user_data;


    public void setNewSegmentCallback(WhisperNewSegmentCallback callback) {
        new_segment_callback = CallbackReference.getFunctionPointer(callback);
    }

    public void setProgressCallback(WhisperProgressCallback callback) {
        progress_callback = CallbackReference.getFunctionPointer(callback);
    }

    public void setEncoderBeginCallbackeginCallbackCallback(WhisperEncoderBeginCallback callback) {
        encoder_begin_callback = CallbackReference.getFunctionPointer(callback);
    }

    public void setLogitsFilterCallback(WhisperLogitsFilterCallback callback) {
        logits_filter_callback = CallbackReference.getFunctionPointer(callback);
    }

    /** Grammar stuff */
    public Pointer grammar_rules;
    public long n_grammar_rules;
    public long i_start_rule;
    public float grammar_penalty;

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("strategy", "n_threads", "n_max_text_ctx",
                "offset_ms", "duration_ms", "translate", "no_context",
                "no_timestamps", "single_segment", "print_special",
                "print_progress", "print_realtime", "print_timestamps",
                "token_timestamps", "thold_pt", "thold_ptsum", "max_len",
                "split_on_word", "max_tokens", "debug_mode", "audio_ctx",
                "tdrz_enable", "suppress_regex", "initial_prompt", "carry_initial_prompt",
                "prompt_tokens", "prompt_n_tokens", "language", "detect_language",
                "suppress_blank", "suppress_nst", "temperature",
                "max_initial_ts", "length_penalty", "temperature_inc",
                "entropy_thold", "logprob_thold", "no_speech_thold", "greedy",
                "beam_search", "new_segment_callback", "new_segment_callback_user_data",
                "progress_callback", "progress_callback_user_data",
                "encoder_begin_callback", "encoder_begin_callback_user_data",
                "abort_callback", "abort_callback_user_data",
                "logits_filter_callback", "logits_filter_callback_user_data",
                "grammar_rules", "n_grammar_rules", "i_start_rule", "grammar_penalty");
    }

    public static class ByValue extends WhisperFullParams implements Structure.ByValue {
        public ByValue() { super(); }
        public ByValue(Pointer p) { super(p); }
    }

}
