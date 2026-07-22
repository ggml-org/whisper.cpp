package io.github.ggerganov.whispercpp.params;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * Voice Activity Detection (VAD) parameters.
 * Mirrors {@code struct whisper_vad_params} in include/whisper.h.
 */
public class WhisperVadParams extends Structure {

    public WhisperVadParams() {
        super();
    }

    public WhisperVadParams(Pointer p) {
        super(p);
    }

    /** Probability threshold to consider as speech. */
    public float threshold;

    /** Min duration for a valid speech segment. */
    public int min_speech_duration_ms;

    /** Min silence duration to consider speech as ended. */
    public int min_silence_duration_ms;

    /** Max duration of a speech segment before forcing a new segment. */
    public float max_speech_duration_s;

    /** Padding added before and after speech segments. */
    public int speech_pad_ms;

    /** Overlap in seconds when copying audio samples from speech segment. */
    public float samples_overlap;

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList(
                "threshold",
                "min_speech_duration_ms",
                "min_silence_duration_ms",
                "max_speech_duration_s",
                "speech_pad_ms",
                "samples_overlap");
    }

    public static class ByValue extends WhisperVadParams implements Structure.ByValue {
        public ByValue() {
            super();
        }

        public ByValue(Pointer p) {
            super(p);
        }
    }
}
