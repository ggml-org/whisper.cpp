package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.model.WhisperTokenData;

/**
 * Callback to filter logits.
 * Can be used to modify the logits before sampling.
 * If not null, called after applying temperature to logits.
 */
public interface WhisperLogitsFilterCallback extends Callback {

    /**
     * Callback method to filter logits.
     *
     * @param ctx        The whisper context.
     * @param state      The whisper state.
     * @param tokens     The array of whisper_token_data.
     * @param n_tokens   The number of tokens.
     * @param logits     The array of logits.
     * @param user_data  User data.
     */
    void callback(Pointer ctx, Pointer state, WhisperTokenData[] tokens, int n_tokens, float[] logits, Pointer user_data);
}
