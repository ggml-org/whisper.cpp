#include "whisper.h"
#include "common-whisper.h"

#include <cstdio>
#include <string>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

void assert_default_params(const struct whisper_vad_params & params) {
    assert(params.threshold == 0.5);
    assert(params.min_speech_duration_ms == 250);
    assert(params.min_silence_duration_ms == 100);
    assert(params.window_size_samples == 512);
    assert(params.samples_overlap == 0.1f);
}

void assert_default_context_params(const struct whisper_vad_context_params & params) {
    assert(params.n_threads == 4);
    assert(params.use_gpu == false);
    assert(params.gpu_device == 0);
}

struct whisper_vad_speech test_detect_speech(
        struct whisper_vad_context * vctx,
        struct whisper_vad_params params,
        const float * pcmf32,
        int n_samples) {
    struct whisper_vad_speech speech = whisper_vad_detect_speech(vctx, pcmf32, n_samples);
    assert(speech.n_probs == 344);
    assert(speech.probs != nullptr);

    return speech;
}

struct whisper_vad_timestamps test_detect_timestamps(
        struct whisper_vad_context * vctx,
        struct whisper_vad_params params,
        struct whisper_vad_speech * speech) {
    struct whisper_vad_timestamps timestamps = whisper_vad_timestamps_from_probs(params, speech);
    assert(timestamps.n_segments == 5);
    assert(timestamps.segments != nullptr);

    for (int i = 0; i < timestamps.n_segments; ++i) {
        printf("VAD segment %d: start = %.2f, end = %.2f\n",
               i, timestamps.segments[i].start, timestamps.segments[i].end);
    }

    return timestamps;
}

int main() {
    std::string vad_model_path = "../../models/for-tests-silero-v5.1.2-ggml.bin";
    std::string sample_path    = "../../samples/jfk.wav";

    // Load the sample audio file
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));
    assert(pcmf32.size() > 0);
    assert(pcmf32s.size() == 0); // no stereo vector

    // Load the VAD model
    struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    assert_default_context_params(ctx_params);

    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(
            vad_model_path.c_str(),
            ctx_params);
    assert(vctx != nullptr);

    struct whisper_vad_params params = whisper_vad_default_params();
    assert_default_params(params);

    // Test speech probabilites
    struct whisper_vad_speech speech = test_detect_speech(vctx, params, pcmf32.data(), pcmf32.size());

    // Test speech timestamps (uses speech probabilities from above)
    struct whisper_vad_timestamps timestamps = test_detect_timestamps(vctx, params, &speech);

    whisper_vad_free_timestamps(&timestamps);
    whisper_vad_free(vctx);

    return 0;
}
