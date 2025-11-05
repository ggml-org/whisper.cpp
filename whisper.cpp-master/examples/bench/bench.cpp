#include "whisper.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

// command-line parameters
struct whisper_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t what = 0; // what to benchmark: 0 - whisper encoder, 1 - memcpy, 2 - ggml_mul_mat

    std::string model = "models/ggml-base.en.bin";

    bool use_gpu    = true;
    bool flash_attn = false;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"  || arg == "--threads")    { params.n_threads  = std::stoi(argv[++i]); }
        else if (arg == "-m"  || arg == "--model")      { params.model      = argv[++i]; }
        else if (arg == "-w"  || arg == "--what")       { params.what       = atoi(argv[++i]); }
        else if (arg == "-ng" || arg == "--no-gpu")     { params.use_gpu    = false; }
        else if (arg == "-fa" || arg == "--flash-attn") { params.flash_attn = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help        [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N   [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME [%-7s] model path\n",                                  params.model.c_str());
    fprintf(stderr, "  -w N,     --what N      [%-7d] what to benchmark:\n",                          params.what);
    fprintf(stderr, "                           %-7s  0 - whisper\n",                                 "");
    fprintf(stderr, "                           %-7s  1 - memcpy\n",                                  "");
    fprintf(stderr, "                           %-7s  2 - ggml_mul_mat\n",                            "");
    fprintf(stderr, "  -ng,      --no-gpu      [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn  [%-7s] enable flash attention\n",                      params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}

static int whisper_bench_full(const whisper_params & params) {
    // whisper init

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads, std::thread::hardware_concurrency(), whisper_print_system_info());
    }

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    const int n_mels = whisper_model_n_mels(ctx);

    if (int ret = whisper_set_mel(ctx, nullptr, 0, n_mels)) {
        fprintf(stderr, "error: failed to set mel: %d\n", ret);
        return 3;
    }
    // heat encoder
    if (int ret = whisper_encode(ctx, 0, params.n_threads) != 0) {
        fprintf(stderr, "error: failed to encode: %d\n", ret);
        return 4;
    }

    whisper_token tokens[512];
    memset(tokens, 0, sizeof(tokens));

    // prompt heat
    if (int ret = whisper_decode(ctx, tokens, 256, 0, params.n_threads) != 0) {
        fprintf(stderr, "error: failed to decode: %d\n", ret);
        return 4;
    }

    // text-generation heat
    if (int ret = whisper_decode(ctx, tokens, 1, 256, params.n_threads) != 0) {
        fprintf(stderr, "error: failed to decode: %d\n", ret);
        return 4;
    }

    whisper_reset_timings(ctx);

    // actual run
    if (int ret = whisper_encode(ctx, 0, params.n_threads) != 0) {
        fprintf(stderr, "error: failed to encode: %d\n", ret);
        return 4;
    }

    // text-generation
    for (int i = 0; i < 256; i++) {
        if (int ret = whisper_decode(ctx, tokens, 1, i, params.n_threads) != 0) {
            fprintf(stderr, "error: failed to decode: %d\n", ret);
            return 4;
        }
    }

    // batched decoding
    for (int i = 0; i < 64; i++) {
        if (int ret = whisper_decode(ctx, tokens, 5, 0, params.n_threads) != 0) {
            fprintf(stderr, "error: failed to decode: %d\n", ret);
            return 4;
        }
    }

    // prompt processing
    for (int i = 0; i < 16; i++) {
        if (int ret = whisper_decode(ctx, tokens, 256, 0, params.n_threads) != 0) {
            fprintf(stderr, "error: failed to decode: %d\n", ret);
            return 4;
        }
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);

    fprintf(stderr, "\n");
    fprintf(stderr, "If you wish, you can submit these results here:\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  https://github.com/ggerganov/whisper.cpp/issues/89\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Please include the following information:\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  - CPU model\n");
    fprintf(stderr, "  - Operating system\n");
    fprintf(stderr, "  - Compiler\n");
    fprintf(stderr, "\n");

    return 0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int ret = -1;

    switch (params.what) {
        case 0: ret = whisper_bench_full(params);                break;
        case 1: ret = whisper_bench_memcpy(params.n_threads);       break;
        case 2: ret = whisper_bench_ggml_mul_mat(params.n_threads); break;
        default: fprintf(stderr, "error: unknown benchmark: %d\n", params.what); break;
    }

    return ret;
}
