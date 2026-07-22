package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.WhisperContext;
import io.github.ggerganov.whispercpp.model.WhisperState;

/**
 * Callback before the encoder starts.
 * If not null, called before the encoder starts.
 * If it returns false, the computation is aborted.
 */
public interface WhisperEncoderBeginCallback extends Callback {

    /**
     * Callback method before the encoder starts.
     *
     * @param ctx        The whisper context.
     * @param state      The whisper state.
     * @param user_data  User data.
     * @return True if the computation should proceed, false otherwise.
     */
    boolean callback(Pointer ctx, Pointer state, Pointer user_data);
}
