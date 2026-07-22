package io.github.ggerganov.whispercpp.params;

import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

public class BeamSearchParams extends Structure {
    /** ref: <a href="https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265">...</a> */
    public int beam_size;

    /** ref: <a href="https://arxiv.org/pdf/2204.05424.pdf">...</a> */
    public float patience;

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("beam_size", "patience");
    }
}
