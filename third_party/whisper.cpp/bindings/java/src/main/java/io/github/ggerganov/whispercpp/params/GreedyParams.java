package io.github.ggerganov.whispercpp.params;

import com.sun.jna.Structure;

import java.util.Collections;
import java.util.List;

public class GreedyParams extends Structure {
    /** <a href="https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264">...</a> */
    public int best_of;

    @Override
    protected List<String> getFieldOrder() {
        return Collections.singletonList("best_of");
    }
}
