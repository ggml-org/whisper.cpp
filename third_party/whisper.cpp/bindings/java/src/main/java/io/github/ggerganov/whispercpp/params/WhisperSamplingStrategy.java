package io.github.ggerganov.whispercpp.params;

/** Available sampling strategies */
public enum WhisperSamplingStrategy {
    /** similar to OpenAI's GreedyDecoder */
    WHISPER_SAMPLING_GREEDY,

    /** similar to OpenAI's BeamSearchDecoder */
    WHISPER_SAMPLING_BEAM_SEARCH
}
