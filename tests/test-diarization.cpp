#include "whisper-cpp.h"
#include "common-whisper.h"

#include <cstdio>
#include <string>
#include <memory>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

int main() {
    ggml_backend_load_all();

    const std::string model_path  = DIARIZATION_MODEL_PATH;
    const std::string sample_path = SAMPLE_PATH;

    std::vector<float> audio;
    std::vector<std::vector<float>> audio_channels;
    assert(read_audio_data(sample_path.c_str(), audio, audio_channels, false));

    whisper_diar_context_params params = whisper_diar_default_context_params();
    whisper_diar_context_ptr ctx(whisper_diar_init_from_file_with_params(model_path.c_str(), params));
    if (!ctx) {
        fprintf(stderr, "failed to load diarization model '%s'\n", model_path.c_str());
        return 1;
    }

    assert(whisper_diar_detect_speakers(ctx.get(), audio.data(), audio.size()));

    const int     n_speakers = whisper_diar_n_speakers(ctx.get());
    const int64_t n_frames   = whisper_diar_n_frames(ctx.get());
    const float * probs      = whisper_diar_probs(ctx.get());

    printf("%lld frames, %d speaker slots\n", (long long)n_frames, n_speakers);

    for (int speaker = 0; speaker < n_speakers; ++speaker) {
        double activity = 0;
        for (int64_t frame = 0; frame < n_frames; ++frame) {
            activity += probs[frame * n_speakers + speaker];
        }
        printf("speaker %d mean activity %.6f\n", speaker, activity / n_frames);
    }

    /*
    const float * data = (const float *) probs;
    for (int t = 0; t < n_frames; ++t) {
        float start_sec = (t * 10) / 1000.0f;
        float end_sec   = ((t + 1) * 10) / 1000.0f;

        printf("frame: %d:", t);
        for (int spk = 0; spk < n_speakers; ++spk) {
            float prob = data[t * 8 + spk];
            printf(" %f ", prob);
        }
        printf("\n");
    }
    */

    const whisper_diar_params diar_params = whisper_diar_default_params();
    whisper_diar_segments_ptr segments(whisper_diar_segments_from_probs(ctx.get(), diar_params));

    assert(segments);
    for (int i = 0; i < whisper_diar_segments_n_segments(segments.get()); ++i) {
        const int64_t t0 = whisper_diar_segments_get_segment_t0(segments.get(), i);
        const int64_t t1 = whisper_diar_segments_get_segment_t1(segments.get(), i);
        const int     s  = whisper_diar_segments_get_speaker(segments.get(), i);

        printf("speaker %d: %.2f -> %.2f\n", s, t0 * 0.01, t1 * 0.01);
    }

    return 0;
}
