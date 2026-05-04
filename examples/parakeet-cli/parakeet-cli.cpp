#include "parakeet.h"
#include "common-whisper.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <cstring>

// command-line parameters
struct parakeet_params {
    int32_t n_threads         = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t chunk_length_ms   = 10000;
    int32_t left_context_ms   = 10000;
    int32_t right_context_ms  = 4960;

    bool use_gpu       = true;
    bool flash_attn    = true;
    int32_t gpu_device = 0;

    bool print_segments = false;

    std::string model    = "models/ggml-parakeet-tdt-0.6b-v3.bin";
    std::vector<std::string> fname_inp = {};
};

static void parakeet_print_usage(int argc, char ** argv, const parakeet_params & params);

static char * requires_value_error(const std::string & arg) {
    fprintf(stderr, "error: argument %s requires value\n", arg.c_str());
    exit(1);
}

static bool parakeet_params_parse(int argc, char ** argv, parakeet_params & params) {
    if (const char * env_device = std::getenv("PARAKEET_ARG_DEVICE")) {
        params.gpu_device = std::stoi(env_device);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-"){
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            parakeet_print_usage(argc, argv, params);
            exit(0);
        }
        #define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads         = std::stoi(ARGV_NEXT); }
        else if (arg == "-cl"   || arg == "--chunk-length")    { params.chunk_length_ms   = std::stoi(ARGV_NEXT); }
        else if (arg == "-lc"   || arg == "--left-context")    { params.left_context_ms   = std::stoi(ARGV_NEXT); }
        else if (arg == "-rc"   || arg == "--right-context")   { params.right_context_ms  = std::stoi(ARGV_NEXT); }
        else if (arg == "-m"    || arg == "--model")           { params.model             = ARGV_NEXT; }
        else if (arg == "-f"    || arg == "--file")            { params.fname_inp.emplace_back(ARGV_NEXT); }
        else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu           = false; }
        else if (arg == "-dev"  || arg == "--device")          { params.gpu_device        = std::stoi(ARGV_NEXT); }
        else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn        = false; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn")   { params.flash_attn        = false; }
        else if (arg == "-ps"   || arg == "--print-segments")  { params.print_segments    = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            parakeet_print_usage(argc, argv, params);
            exit(1);
        }
    }

    return true;
}

static void parakeet_print_usage(int /*argc*/, char ** argv, const parakeet_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0 file1 ...\n", argv[0]);
    fprintf(stderr, "supported audio formats: flac, mp3, ogg, wav\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,     --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,   --threads N         [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -cl N,  --chunk-length N    [%-7d] chunk length in milliseconds\n",                 params.chunk_length_ms);
    fprintf(stderr, "  -lc N,  --left-context N    [%-7d] left context in milliseconds\n",                params.left_context_ms);
    fprintf(stderr, "  -rc N,  --right-context N   [%-7d] right context in milliseconds\n",               params.right_context_ms);
    fprintf(stderr, "  -m,     --model FILE        [%-7s] model path\n",                                  params.model.c_str());
    fprintf(stderr, "  -f,     --file FILE         [%-7s] input audio file\n",                            "");
    fprintf(stderr, "  -ng,    --no-gpu            [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -dev N, --device N          [%-7d] GPU device to use\n",                           params.gpu_device);
    fprintf(stderr, "  -fa,    --flash-attn        [%-7s] enable flash attention\n",                      params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,   --no-flash-attn     [%-7s] disable flash attention\n",                     !params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -ps,    --print-segments    [%-7s] print segment information\n",                   params.print_segments ? "true" : "false");
    fprintf(stderr, "\n");
}

void token_callback(parakeet_context * ctx, parakeet_state * state, const parakeet_token_data * token_data, void * user_data) {
    static bool is_first = true;

    const char * token_str = parakeet_token_to_str(ctx, token_data->id);
    char text_buf[256];
    parakeet_token_to_text(token_str, is_first, text_buf, sizeof(text_buf));
    printf("%s", text_buf);
    fflush(stdout);

    is_first = false;
}


int main(int argc, char ** argv) {
    ggml_backend_load_all();

    parakeet_params params;

    if (parakeet_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: no input files specified\n");
        parakeet_print_usage(argc, argv, params);
        return 1;
    }

    // Process each input file
    for (const auto & fname : params.fname_inp) {
        fprintf(stderr, "\nProcessing file: %s\n", fname.c_str());

        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(fname.c_str(), pcmf32, pcmf32s, false)) {
            fprintf(stderr, "error: failed to read audio file '%s'\n", fname.c_str());
            continue;
        }

        if (pcmf32.empty()) {
            fprintf(stderr, "error: no audio data in file '%s'\n", fname.c_str());
            continue;
        }

        fprintf(stderr, "Loading Parakeet model from: %s\n", params.model.c_str());

        struct parakeet_context_params ctx_params = parakeet_context_default_params();
        ctx_params.use_gpu     = params.use_gpu;
        ctx_params.flash_attn  = params.flash_attn;
        ctx_params.gpu_device  = params.gpu_device;

        struct parakeet_context * pctx = parakeet_init_from_file_with_params(params.model.c_str(), ctx_params);
        if (pctx == nullptr) {
            fprintf(stderr, "error: failed to load Parakeet model from '%s'\n", params.model.c_str());
            return 1;
        }

        fprintf(stderr, "Successfully loaded Parakeet model\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, (int32_t) std::thread::hardware_concurrency(), parakeet_print_system_info());
        fprintf(stderr, "Processing audio (%zu samples, %.2f seconds)\n",
                pcmf32.size(), (float)pcmf32.size() / PARAKEET_SAMPLE_RATE);

        struct parakeet_full_params full_params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
        full_params.n_threads           = params.n_threads;
        full_params.chunk_length_ms     = params.chunk_length_ms;
        full_params.left_context_ms     = params.left_context_ms;
        full_params.right_context_ms    = params.right_context_ms;
        full_params.new_token_callback  = token_callback;
        full_params.new_token_callback_user_data = nullptr;

        const int mel_frames = (int)(pcmf32.size() / PARAKEET_HOP_LENGTH);
        if (mel_frames <= parakeet_n_audio_ctx(pctx)) {
            full_params.chunk_length_ms = 0;
        }

        int ret = parakeet_full(pctx, full_params, pcmf32.data(), pcmf32.size());

        if (ret != 0) {
            fprintf(stderr, "error: failed to process audio file '%s'\n", fname.c_str());
            parakeet_free(pctx);
            continue;
        }

        printf("\n");

        parakeet_print_timings(pctx);

        if (params.print_segments) {
            const int n_segments = parakeet_full_n_segments(pctx);
            fprintf(stderr, "\nSegments (%d):\n", n_segments);

            for (int i = 0; i < n_segments; i++) {
                const char * text = parakeet_full_get_segment_text(pctx, i);
                const int64_t t0 = parakeet_full_get_segment_t0(pctx, i);
                const int64_t t1 = parakeet_full_get_segment_t1(pctx, i);
                const int n_tokens = parakeet_full_n_tokens(pctx, i);

                fprintf(stderr, "Segment %d: [%lld -> %lld] \"%s\"\n", i, (long long)t0, (long long)t1, text);
                fprintf(stderr, "Tokens [%d]:\n", n_tokens);

                for (int j = 0; j < n_tokens; j++) {
                    parakeet_token_data token_data = parakeet_full_get_token_data(pctx, i, j);
                    const char * token_str = parakeet_token_to_str(pctx, token_data.id);

                    fprintf(stderr, "  [%2d] id=%5d frame=%3d dur_idx=%2d dur_val=%2d p=%.4f plog=%.4f t0=%4lld t1=%4lld word_start=%s \"%s\"\n",
                           j,
                           token_data.id,
                           token_data.frame_index,
                           token_data.duration_idx,
                           token_data.duration_value,
                           token_data.p,
                           token_data.plog,
                           (long long)token_data.t0,
                           (long long)token_data.t1,
                           token_data.is_word_start ? "true": "false",
                           token_str);
                }
            }
        }

        parakeet_free(pctx);
    }

    return 0;
}
