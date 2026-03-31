#include "common.h"
#include "common-whisper.h"

#include "whisper.h"

#include <cstdio>
#include <cfloat>
#include <string>

// command-line parameters
struct cli_params {
    int32_t     n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    std::string vad_model = "";
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;
    bool        use_gpu = false;
    std::string fname_inp = {};
    bool        no_prints       = false;
};

static void vad_print_usage(int /*argc*/, char ** argv, const cli_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file\n", argv[0]);
    fprintf(stderr, "supported audio formats: flac, mp3, ogg, wav\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help                          [default] show this help message and exit\n");
    fprintf(stderr, "  -f FNAME,  --file FNAME                    [%-7s] input audio file path\n",                            "");
    fprintf(stderr, "  -t N,      --threads N                     [%-7d] number of threads to use during computation\n",      params.n_threads);
    fprintf(stderr, "  -ug,       --use-gpu                       [%-7s] use GPU\n",                                          params.use_gpu ? "true" : "false");
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME               [%-7s] VAD model path\n",                                   params.vad_model.c_str());
    fprintf(stderr, "  -vt N,     --vad-threshold N               [%-7.2f] VAD threshold for speech recognition\n",           params.vad_threshold);
    fprintf(stderr, "  -vspd N,   --vad-min-speech-duration-ms  N [%-7d] VAD min speech duration (0.0-1.0)\n",                params.vad_min_speech_duration_ms);
    fprintf(stderr, "  -vsd N,    --vad-min-silence-duration-ms N [%-7d] VAD min silence duration (to split segments)\n",     params.vad_min_silence_duration_ms);
    fprintf(stderr, "  -vmsd N,   --vad-max-speech-duration-s   N [%-7s] VAD max speech duration (auto-split longer)\n",      params.vad_max_speech_duration_s == FLT_MAX ?
                                                                                                                                  std::string("FLT_MAX").c_str() :
                                                                                                                                  std::to_string(params.vad_max_speech_duration_s).c_str());
    fprintf(stderr, "  -vp N,     --vad-speech-pad-ms           N [%-7d] VAD speech padding (extend segments)\n",             params.vad_speech_pad_ms);
    fprintf(stderr, "  -vo N,     --vad-samples-overlap         N [%-7.2f] VAD samples overlap (seconds between segments)\n", params.vad_samples_overlap);
    fprintf(stderr, "  -np,       --no-prints                     [%-7s] do not print anything other than the results\n",     params.no_prints ? "true" : "false");
    fprintf(stderr, "\n");
}

static char * requires_value_error(const std::string & arg) {
    fprintf(stderr, "error: argument %s requires value\n", arg.c_str());
    exit(0);
}

static bool vad_params_parse(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            vad_print_usage(argc, argv, params);
            exit(0);
        }
        #define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))
        else if (arg == "-f"    || arg == "--file")                        { params.fname_inp = ARGV_NEXT; }
        else if (arg == "-t"    || arg == "--threads")                     { params.n_threads                   = std::stoi(ARGV_NEXT); }
        else if (arg == "-ug"   || arg == "--use-gpu")                     { params.use_gpu                     = true; }
        else if (arg == "-vm"   || arg == "--vad-model")                   { params.vad_model                   = ARGV_NEXT; }
        else if (arg == "-vt"   || arg == "--vad-threshold")               { params.vad_threshold               = std::stof(ARGV_NEXT); }
        else if (arg == "-vsd"  || arg == "--vad-min-speech-duration-ms")  { params.vad_min_speech_duration_ms  = std::stoi(ARGV_NEXT); }
        else if (arg == "-vsd"  || arg == "--vad-min-silence-duration-ms") { params.vad_min_speech_duration_ms  = std::stoi(ARGV_NEXT); }
        else if (arg == "-vmsd" || arg == "--vad-max-speech-duration-s")   { params.vad_max_speech_duration_s   = std::stof(ARGV_NEXT); }
        else if (arg == "-vp"   || arg == "--vad-speech-pad-ms")           { params.vad_speech_pad_ms           = std::stoi(ARGV_NEXT); }
        else if (arg == "-vo"   || arg == "--vad-samples-overlap")         { params.vad_samples_overlap         = std::stof(ARGV_NEXT); }
        else if (arg == "-np"   || arg == "--no-prints")                   { params.no_prints       = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            vad_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

static void cb_log_disable(enum ggml_log_level , const char * , void * ) { }

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    cli_params cli_params;

    if (!vad_params_parse(argc, argv, cli_params)) {
        vad_print_usage(argc, argv, cli_params);
        return 1;
    }

    if (cli_params.no_prints) {
        whisper_log_set(cb_log_disable, NULL);
    }

    // Load the input sample audio file.
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(cli_params.fname_inp.c_str(), pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read audio data from %s\n", cli_params.fname_inp.c_str());
        return 2;
    }

    // Initialize the context which loads the VAD model.
    struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    ctx_params.n_threads  = cli_params.n_threads;
    ctx_params.use_gpu    = cli_params.use_gpu;
    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(
            cli_params.vad_model.c_str(),
            ctx_params);
    if (vctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    // Detect speech in the input audio file.
    if (!whisper_vad_detect_speech(vctx, pcmf32.data(), pcmf32.size())) {
        fprintf(stderr, "error: failed to detect speech\n");
        return 3;
    }

    // Get the the vad segements using the probabilities that have been computed
    // previously and stored in the whisper_vad_context.
    struct whisper_vad_params params = whisper_vad_default_params();
    params.threshold = cli_params.vad_threshold;
    params.min_speech_duration_ms = cli_params.vad_min_speech_duration_ms;
    params.min_silence_duration_ms = cli_params.vad_min_silence_duration_ms;
    params.max_speech_duration_s = cli_params.vad_max_speech_duration_s;
    params.speech_pad_ms = cli_params.vad_speech_pad_ms;
    params.samples_overlap = cli_params.vad_samples_overlap;
    struct whisper_vad_segments * segments = whisper_vad_segments_from_probs(vctx, params);

    printf("\n");
    printf("Detected %d speech segments:\n", whisper_vad_segments_n_segments(segments));
    for (int i = 0; i < whisper_vad_segments_n_segments(segments); ++i) {
        printf("Speech segment %d: start = %.2f, end = %.2f\n", i,
               whisper_vad_segments_get_segment_t0(segments, i),
               whisper_vad_segments_get_segment_t1(segments, i));
    }
    printf("\n");

    whisper_vad_free_segments(segments);
    whisper_vad_free(vctx);

    return 0;
}
