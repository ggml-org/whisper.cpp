#include <string>
#include "whisper.h"
#include "server-params.h"
#include "whisper-server.h"

#define CONVERT_FROM_PCM_16
std::string forward_url = "http://127.0.0.1:8080/completion";
size_t max_messages = 1000;

void print_usage(int argc, char** argv, const ServerParams& params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          show this help message and exit\n");
    fprintf(stderr, "  -H HOST,  --host HOST     [%-7s] hostname or ip\n", params.host.c_str());
    fprintf(stderr, "  -p PORT,  --port PORT     [%-7d] server port\n", params.port);
    fprintf(stderr, "  -f FORWARD_URL,  --forward-url FORWARD_URL     [%-7s] forward url\n", forward_url.c_str());
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads\n", params.n_threads);
    fprintf(stderr, "  -nm max_messages,    --max-messages max_messages     [%-7d] max messages before send to backend\n", max_messages);
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity threshold\n", params.vad_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate to english\n", params.translate ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n", params.beam_size);
    fprintf(stderr, "\n");
}

bool parse_params(int argc, char** argv, ServerParams& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-H" || arg == "--host")      { params.host = argv[++i]; }
        else if (arg == "-p" || arg == "--port")      { params.port = std::stoi(argv[++i]); }
        else if (arg == "-f" || arg == "--forward-url")      { forward_url = argv[++i]; }
        else if (arg == "-t" || arg == "--threads")   { params.n_threads = std::stoi(argv[++i]); }
        else if (arg == "-nm" || arg == "--max-messages")   { max_messages = std::stoi(argv[++i]); }
        else if (arg == "-m" || arg == "--model")     { params.model = argv[++i]; }
        else if (arg == "-l" || arg == "--language")  { params.language = argv[++i]; }
        else if (arg == "-vth" || arg == "--vad-thold") { params.vad_thold = std::stof(argv[++i]); }
        else if (arg == "-tr" || arg == "--translate")  { params.translate = true; }
        else if (arg == "-bs" || arg == "--beam-size")  { params.beam_size = std::stoi(argv[++i]); }
        else if (arg == "-ng" || arg == "--no-gpu")     { params.use_gpu = false; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    ServerParams params;
    if (!parse_params(argc, argv, params)) {
        return 1;
    }
    if (params.port < 1 || params.port > 65535) {
        throw std::invalid_argument("Invalid port number");
    }
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        return 1;
    }

    WhisperServer server(params);
    server.run();
    return 0;
}
