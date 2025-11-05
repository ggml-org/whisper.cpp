package io.github.ggerganov.whispercpp.params;
import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

/**
 * Parameters for the whisper_init_from_file_with_params() function.
 * If you change the order or add new parameters, make sure to update the default values in whisper.cpp:
 * whisper_context_default_params()
 */
public class WhisperContextParams extends Structure {
    public WhisperContextParams(Pointer p) {
        super(p);
    }

    public WhisperContextParams() {
        super();
    }

    /** Use GPU for inference (default = true) */
    public CBool use_gpu;

    /** Use flash attention (default = false) */
    public CBool flash_attn;

    /** CUDA device to use (default = 0) */
    public int gpu_device;

    /** [EXPERIMENTAL] Enable token-level timestamps with DTW (default = false) */
    public CBool dtw_token_timestamps;

    /** [EXPERIMENTAL] Alignment heads preset for DTW */
    public int dtw_aheads_preset;

    /** Number of top layers to use for DTW when using WHISPER_AHEADS_N_TOP_MOST preset */
    public int dtw_n_top;

    public WhisperAheads.ByValue dtw_aheads;

    /** DTW memory size (internal use) */
    public NativeLong dtw_mem_size;

    /** Use GPU for inference */
    public void useGpu(boolean enable) {
        use_gpu = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Use flash attention */
    public void useFlashAttn(boolean enable) {
        flash_attn = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Enable DTW token-level timestamps */
    public void enableDtwTokenTimestamps(boolean enable) {
        dtw_token_timestamps = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Set DTW alignment heads preset */
    public void setDtwAheadsPreset(int preset) {
        dtw_aheads_preset = preset;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList(
            "use_gpu",
            "flash_attn",
            "gpu_device",
            "dtw_token_timestamps",
            "dtw_aheads_preset",
            "dtw_n_top",
            "dtw_aheads",
            "dtw_mem_size"
        );
    }

    public static class ByValue extends WhisperContextParams implements Structure.ByValue {
        public ByValue() { super(); }
        public ByValue(Pointer p) { super(p); }
    }
}
