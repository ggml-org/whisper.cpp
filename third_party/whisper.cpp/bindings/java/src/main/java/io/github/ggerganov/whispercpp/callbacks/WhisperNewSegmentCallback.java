package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.WhisperContext;
import io.github.ggerganov.whispercpp.model.WhisperState;

/**
 * Callback for the text segment.
 * Called on every newly generated text segment.
 * Use the whisper_full_...() functions to obtain the text segments.
 */
public interface WhisperNewSegmentCallback extends Callback {

    /**
     * Callback method for the text segment.
     *
     * @param ctx        The whisper context.
     * @param state      The whisper state.
     * @param n_new      The number of newly generated text segments.
     * @param user_data  User data.
     */
    void callback(Pointer ctx, Pointer state, int n_new, Pointer user_data);
}
