package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Callback;

/**
 * Callback for aborting GGML computation
 * Maps to the C typedef: bool (*ggml_abort_callback)(void * data)
 */
public interface GgmlAbortCallback extends Callback {
    /**
     * Return true to abort the computation, false to continue
     *
     * @param data User data passed to the callback
     * @return true to abort, false to continue
     */
    boolean invoke(com.sun.jna.Pointer data);
}
