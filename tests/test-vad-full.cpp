#include "whisper.h"
#include "common-whisper.h"

#include <cstdio>
#include <cfloat>
#include <string>
#include <cstring>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

int main() {
    std::string whisper_model_path = "../../models/ggml-base.en.bin";
    std::string vad_model_path     = "../../models/for-tests-silero-v5.1.2-ggml.bin";
    std::string sample_path        = "../../samples/jfk.wav";

    // Load the sample audio file
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * wctx = whisper_init_from_file_with_params(
            whisper_model_path.c_str(),
            cparams);

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.vad                         = true;
    wparams.vad_model_path              = vad_model_path.c_str();
    wparams.vad_threshold               = 0.5f;
    wparams.vad_min_speech_duration_ms  = 250;
    wparams.vad_min_silence_duration_ms = 100;
    wparams.vad_max_speech_duration_s   = FLT_MAX;
    wparams.vad_speech_pad_ms           = 30;
    wparams.vad_window_size_samples     = 512;

    assert(whisper_full_parallel(wctx, wparams, pcmf32.data(), pcmf32.size(), 1) == 0);

    const int n_segments = whisper_full_n_segments(wctx);
    assert(n_segments == 2);
    assert(strcmp("And so my fellow Americans ask not what you country can do for you.",
           whisper_full_get_segment_text(wctx, 0)));
    assert(strcmp("Ask what you can do for your country.",
           whisper_full_get_segment_text(wctx, 1)));

    whisper_free(wctx);

    return 0;
}
