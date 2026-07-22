package io.github.ggerganov.whispercpp.params;

public class WhisperHParams {
    int n_vocab       = 51864;
    int n_audio_ctx   = 1500;
    int n_audio_state = 384;
    int n_audio_head  = 6;
    int n_audio_layer = 4;
    int n_text_ctx    = 448;
    int n_text_state  = 384;
    int n_text_head   = 6;
    int n_text_layer  = 4;
    int n_mels        = 80;
    int ftype         = 1;
}
