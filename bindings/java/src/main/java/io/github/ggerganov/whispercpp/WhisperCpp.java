package io.github.ggerganov.whispercpp;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.bean.WhisperSegment;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Before calling most methods, you must call `initContext(modelPath)` to initialise the `ctx` Pointer.
 */
public class WhisperCpp implements AutoCloseable {
    private WhisperCppJnaLibrary lib = WhisperCppJnaLibrary.instance;
    private Pointer ctx = null;
    private Pointer paramsPointer = null;
    private Pointer greedyParamsPointer = null;
    private Pointer beamParamsPointer = null;

    public File modelDir() {
        String modelDirPath = System.getenv("XDG_CACHE_HOME");
        if (modelDirPath == null) {
            modelDirPath = System.getProperty("user.home") + "/.cache";
        }

        return new File(modelDirPath, "whisper");
    }

    /**
     * @param modelPath - absolute path, or just the name (eg: "base", "base-en" or "base.en")
     */
    public void initContext(String modelPath) throws FileNotFoundException {
        initContextImpl(modelPath, getContextDefaultParams());
    }

    /**
     * @param modelPath - absolute path, or just the name (eg: "base", "base-en" or "base.en")
     * @param params - params to use when initialising the context
     */
    public void initContext(String modelPath, WhisperContextParams.ByValue params) throws FileNotFoundException {
        initContextImpl(modelPath, params);
    }

    private void initContextImpl(String modelPath, WhisperContextParams.ByValue params) throws FileNotFoundException {
        if (ctx != null) {
            lib.whisper_free(ctx);
        }

        if (!modelPath.contains("/") && !modelPath.contains("\\")) {
            if (!modelPath.endsWith(".bin")) {
                modelPath = "ggml-" + modelPath.replace("-", ".") + ".bin";
            }

            modelPath = new File(modelDir(), modelPath).getAbsolutePath();
        }

        ctx = lib.whisper_init_from_file_with_params(modelPath, params);

        if (ctx == null) {
            throw new FileNotFoundException(modelPath);
        }
    }

    /**
     * Provides default params which can be used with `whisper_init_from_file_with_params()` etc.
     * Returns a ByValue instance to ensure proper parameter passing to native code.
     */
    public WhisperContextParams.ByValue getContextDefaultParams() {
        WhisperContextParams.ByValue valueParams = new WhisperContextParams.ByValue(
            lib.whisper_context_default_params_by_ref());
        valueParams.read();
        return valueParams;
    }
    
    /**
     * Provides default params which can be used with `whisper_full()` etc.
     * Because this function allocates memory for the params, the caller must call either:
     * - call `whisper_free_params()`
     * - `Native.free(Pointer.nativeValue(pointer));`
     *
     * @param strategy - GREEDY
     */
    public WhisperFullParams.ByValue getFullDefaultParams(WhisperSamplingStrategy strategy) {
        Pointer pointer;

        // whisper_full_default_params_by_ref allocates memory which we need to delete, so only create max 1 pointer for each strategy.
        if (strategy == WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY) {
            if (greedyParamsPointer == null) {
                greedyParamsPointer = lib.whisper_full_default_params_by_ref(strategy.ordinal());
            }
            pointer = greedyParamsPointer;
        } else {
            if (beamParamsPointer == null) {
                beamParamsPointer = lib.whisper_full_default_params_by_ref(strategy.ordinal());
            }
            pointer = beamParamsPointer;
        }

        WhisperFullParams.ByValue params = new WhisperFullParams.ByValue(pointer);
        params.read();
        return params;
    }

    @Override
    public void close() {
        freeContext();
        freeParams();
        System.out.println("Whisper closed");
    }

    private void freeContext() {
        if (ctx != null) {
            lib.whisper_free(ctx);
        }
    }

    private void freeParams() {
        if (paramsPointer != null) {
            Native.free(Pointer.nativeValue(paramsPointer));
            paramsPointer = null;
        }
        if (greedyParamsPointer != null) {
            Native.free(Pointer.nativeValue(greedyParamsPointer));
            greedyParamsPointer = null;
        }
        if (beamParamsPointer != null) {
            Native.free(Pointer.nativeValue(beamParamsPointer));
            beamParamsPointer = null;
        }
    }

    /**
     * Run the entire model: PCM -&gt; log mel spectrogram -&gt; encoder -&gt; decoder -&gt; text.
     * Not thread safe for same context
     * Uses the specified decoding strategy to obtain the text.
     */
    public String fullTranscribe(WhisperFullParams.ByValue whisperParams, float[] audioData) throws IOException {
        if (ctx == null) {
            throw new IllegalStateException("Model not initialised");
        }

        /*
        WhisperFullParams.ByValue valueParams = new WhisperFullParams.ByValue(
            lib.whisper_full_default_params_by_ref(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH.ordinal()));
        valueParams.read();
        */

        if (lib.whisper_full(ctx, whisperParams, audioData, audioData.length) != 0) {
            throw new IOException("Failed to process audio");
        }

        int nSegments = lib.whisper_full_n_segments(ctx);

        StringBuilder str = new StringBuilder();

        for (int i = 0; i < nSegments; i++) {
            String text = lib.whisper_full_get_segment_text(ctx, i);
            System.out.println("Segment:" + text);
            str.append(text);
        }

        return str.toString().trim();
    }

    /**
     * Full transcribe with time list.
     *
     * @param whisperParams the whisper params
     * @param audioData     the audio data
     * @return the list
     * @throws IOException the io exception
     */
    public List<WhisperSegment> fullTranscribeWithTime(WhisperFullParams.ByValue whisperParams, float[] audioData) throws IOException {
        if (ctx == null) {
            throw new IllegalStateException("Model not initialised");
        }

        if (lib.whisper_full(ctx, whisperParams, audioData, audioData.length) != 0) {
            throw new IOException("Failed to process audio");
        }

        int nSegments = lib.whisper_full_n_segments(ctx);
        List<WhisperSegment> segments= new ArrayList<>(nSegments);

        for (int i = 0; i < nSegments; i++) {
            long t0 = lib.whisper_full_get_segment_t0(ctx, i);
            String text = lib.whisper_full_get_segment_text(ctx, i);
            long t1 = lib.whisper_full_get_segment_t1(ctx, i);
            segments.add(new WhisperSegment(t0,t1,text));
        }

        return segments;
    }

//    public int getTextSegmentCount(Pointer ctx) {
//        return lib.whisper_full_n_segments(ctx);
//    }
//    public String getTextSegment(Pointer ctx, int index) {
//        return lib.whisper_full_get_segment_text(ctx, index);
//    }

    public String getSystemInfo() {
        return lib.whisper_print_system_info();
    }

    public int benchMemcpy(int nthread) {
        return lib.whisper_bench_memcpy(nthread);
    }

    public int benchGgmlMulMat(int nthread) {
        return lib.whisper_bench_ggml_mul_mat(nthread);
    }
}
