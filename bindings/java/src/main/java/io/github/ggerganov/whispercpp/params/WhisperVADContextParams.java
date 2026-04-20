package io.github.ggerganov.whispercpp.params;

import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

/**
 * Parameters for initializing a VAD context.
 */
public class WhisperVADContextParams extends Structure {

    public WhisperVADContextParams() {
        super();
    }

    public WhisperVADContextParams(Pointer p) {
        super(p);
    }

    /** Number of threads to use for VAD processing (default = 4) */
    public int n_threads;

    /** Use GPU for VAD (default = true) */
    public CBool use_gpu;

    /** CUDA device to use (default = 0) */
    public int gpu_device;

    /**
     * Set number of threads for VAD processing.
     * @param threads Number of threads
     */
    public void setThreads(int threads) {
        this.n_threads = threads;
    }

    /**
     * Enable or disable GPU for VAD.
     * @param enable Whether to use GPU
     */
    public void useGpu(boolean enable) {
        use_gpu = enable ? CBool.TRUE : CBool.FALSE;
    }

    /**
     * Set CUDA device for VAD.
     * @param device CUDA device ID
     */
    public void setGpuDevice(int device) {
        this.gpu_device = device;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList(
            "n_threads",
            "use_gpu",
            "gpu_device"
        );
    }

    public static class ByValue extends WhisperVADContextParams implements Structure.ByValue {
        public ByValue() { super(); }
        public ByValue(Pointer p) { super(p); }
    }
}
