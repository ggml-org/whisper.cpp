package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.WhisperContext;
import io.github.ggerganov.whispercpp.model.WhisperState;

/**
 * Callback for progress updates.
 */
public interface WhisperProgressCallback extends Callback {

    /**
     * Callback method for progress updates.
     *
     * @param ctx        The whisper context.
     * @param state      The whisper state.
     * @param progress   The progress value.
     * @param user_data  User data.
     */
    void callback(Pointer ctx, Pointer state, int progress, Pointer user_data);
}
