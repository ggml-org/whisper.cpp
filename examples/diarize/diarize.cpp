#include "whisper.h"
#include "whisper-cpp.h"
#include "common-whisper.h"

#include <cstdio>
#include <string>
#include <vector>

struct diarize_params {
    std::string model;
    std::string audio;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "usage: %s -m MODEL AUDIO_FILE\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help        show this help message\n");
    fprintf(stderr, "  -m, --model FILE  diarization model path\n");
    fprintf(stderr, "\n");
}

static bool parse_params(int argc, char ** argv, diarize_params & params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model = argv[++i];
        } else if (arg[0] != '-') {
            params.audio = arg;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (params.model.empty()) {
        fprintf(stderr, "error: model path is required (-m)\n");
        print_usage(argv[0]);
        return false;
    }

    if (params.audio.empty()) {
        fprintf(stderr, "error: audio file is required\n");
        print_usage(argv[0]);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    diarize_params params;
    if (!parse_params(argc, argv, params)) {
        return 1;
    }

    std::vector<float> audio;
    std::vector<std::vector<float>> audio_channels;
    if (!read_audio_data(params.audio.c_str(), audio, audio_channels, false)) {
        fprintf(stderr, "error: failed to read audio file '%s'\n", params.audio.c_str());
        return 1;
    }

    const whisper_diar_context_params ctx_params = whisper_diar_default_context_params();
    whisper_diar_context_ptr ctx(whisper_diar_init_from_file_with_params(params.model.c_str(), ctx_params));
    if (!ctx) {
        fprintf(stderr, "error: failed to load diarization model '%s'\n", params.model.c_str());
        return 1;
    }

    if (!whisper_diar_detect_speakers(ctx.get(), audio.data(), audio.size())) {
        fprintf(stderr, "error: speaker detection failed\n");
        return 1;
    }

    const whisper_diar_params diar_params = whisper_diar_default_params();
    whisper_diar_segments_ptr segments(whisper_diar_segments_from_probs(ctx.get(), diar_params));
    if (!segments) {
        fprintf(stderr, "error: failed to compute segments\n");
        return 1;
    }

    const int n = whisper_diar_segments_n_segments(segments.get());
    for (int i = 0; i < n; ++i) {
        const int64_t t0 = whisper_diar_segments_get_segment_t0(segments.get(), i);
        const int64_t t1 = whisper_diar_segments_get_segment_t1(segments.get(), i);
        const int s  = whisper_diar_segments_get_speaker(segments.get(), i);
        printf("speaker %1d  [%6.2f -> %6.2f]\n", s, t0 * 0.01, t1 * 0.01);
    }

    return 0;
}
