package io.github.ggerganov.whispercpp.params;

import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

/**
 * Voice Activity Detection (VAD) parameters.
 * Used for detecting speech segments in audio.
 */
public class WhisperVADParams extends Structure {

    public WhisperVADParams() {
        super();
    }

    public WhisperVADParams(Pointer p) {
        super(p);
    }

    /** Probability threshold to consider as speech (default = 0.5) */
    public float threshold;

    /** Minimum duration for a valid speech segment in milliseconds (default = 250) */
    public int min_speech_duration_ms;

    /** Minimum silence duration to consider speech as ended in milliseconds (default = 2000) */
    public int min_silence_duration_ms;

    /** Maximum duration of a speech segment before forcing a new segment in seconds (default = Float.MAX_VALUE) */
    public float max_speech_duration_s;

    /** Padding added before and after speech segments in milliseconds (default = 400) */
    public int speech_pad_ms;

    /** Overlap in seconds when copying audio samples from speech segment (default = 1.0) */
    public float samples_overlap;

    /**
     * Set probability threshold for speech detection.
     * @param threshold Probability threshold (0.0 to 1.0)
     */
    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    /**
     * Set minimum speech duration.
     * @param durationMs Duration in milliseconds
     */
    public void setMinSpeechDuration(int durationMs) {
        this.min_speech_duration_ms = durationMs;
    }

    /**
     * Set minimum silence duration.
     * @param durationMs Duration in milliseconds
     */
    public void setMinSilenceDuration(int durationMs) {
        this.min_silence_duration_ms = durationMs;
    }

    /**
     * Set maximum speech duration.
     * @param durationS Duration in seconds
     */
    public void setMaxSpeechDuration(float durationS) {
        this.max_speech_duration_s = durationS;
    }

    /**
     * Set speech padding.
     * @param paddingMs Padding in milliseconds
     */
    public void setSpeechPadding(int paddingMs) {
        this.speech_pad_ms = paddingMs;
    }

    /**
     * Set samples overlap.
     * @param overlapS Overlap in seconds
     */
    public void setSamplesOverlap(float overlapS) {
        this.samples_overlap = overlapS;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList(
            "threshold",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "max_speech_duration_s",
            "speech_pad_ms",
            "samples_overlap"
        );
    }

    public static class ByValue extends WhisperVADParams implements Structure.ByValue {
        public ByValue() { super(); }
        public ByValue(Pointer p) { super(p); }
    }
}
