// Command line voice assisted chess
//
// Speak chess move commands to the microphone.
// The moves will translated to chessboard positions.
//
//

#include "WChess.h"
#include "common-sdl.h"
#include <iostream>

#include <memory>
#include <thread>

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t prompt_ms  = 5000;
    int32_t command_ms = 8000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold  = 0.6f;
    float freq_thold = 100.0f;

    float grammar_penalty = 100.0f;

    bool translate     = false;
    bool print_special = false;
    bool print_energy  = false;
    bool no_timestamps = true;
    bool use_gpu       = true;
    bool flash_attn    = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
    std::string commands;
    std::string prompt;
    std::string context;
    std::string grammar;
};

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,         --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,       --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -pms N,     --prompt-ms N    [%-7d] prompt duration in milliseconds\n",             params.prompt_ms);
    fprintf(stderr, "  -cms N,     --command-ms N   [%-7d] command duration in milliseconds\n",            params.command_ms);
    fprintf(stderr, "  -c ID,      --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,      --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,      --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -vth N,     --vad-thold N    [%-7.2f] voice activity detection threshold\n",        params.vad_thold);
    fprintf(stderr, "  -fth N,     --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -tr,        --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,        --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,        --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  -ng,        --no-gpu         [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,        --flash-attn     [%-7s] flash attention during decoding\n",             params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -l LANG,    --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -m FNAME,   --model FNAME    [%-7s] model path\n",                                  params.model.c_str());
    fprintf(stderr, "  -f FNAME,   --file FNAME     [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "  -cmd FNAME, --commands FNAME [%-7s] text file with allowed commands\n",             params.commands.c_str());
    fprintf(stderr, "  -p,         --prompt         [%-7s] the required activation prompt\n",              params.prompt.c_str());
    fprintf(stderr, "  -ctx,       --context        [%-7s] sample text to help the transcription\n",       params.context.c_str());
    fprintf(stderr, "  --grammar-penalty N          [%-7.1f] scales down logits of nongrammar tokens\n",   params.grammar_penalty);
    fprintf(stderr, "\n");
}

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (arg == "-pms" || arg == "--prompt-ms")     { params.prompt_ms     = std::stoi(argv[++i]); }
        else if (arg == "-cms" || arg == "--command-ms")    { params.command_ms    = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"  || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-pe"  || arg == "--print-energy")  { params.print_energy  = true; }
        else if (arg == "-ng"  || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"  || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-cmd" || arg == "--commands")      { params.commands      = argv[++i]; }
        else if (arg == "-p"   || arg == "--prompt")        { params.prompt        = argv[++i]; }
        else if (arg == "-ctx" || arg == "--context")       { params.context       = argv[++i]; }
        else if (                 arg == "--grammar-penalty") { params.grammar_penalty = std::stof(argv[++i]); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

std::unique_ptr<WChess> g_wchess;
int g_moveCount = 0;
void set_move(const std::string & move, float) {
    if (!move.empty()) {
        g_moveCount++;
        fprintf(stdout, "Move: %s\n\n", move.c_str());
    }
    else fprintf(stdout, "Move rejected\n\n");
    fprintf(stdout, "%s\n", g_wchess->stringify_board().c_str());
    fprintf(stdout, "%s\n", g_moveCount ? "White's turn" : "Black's turn");
}

audio_async g_audio(30*1000);
bool g_listening = false;
std::vector<float> g_pcmf32;

bool read_input() {
    std::string input;
    while (true) {
        fprintf(stdout, "[(l)isten/(p)ause/(q)uit]: ");
        std::cin >> input;
        fprintf(stdout, "\n");
        if (input[0] == 'q') {
            fprintf(stdout, "Quitting\n");
            return false;
        }
        if (input[0] == 'l') {
            if (!g_listening) {
                fprintf(stdout, "Listening\n");
                g_listening = true;
                g_pcmf32.clear();
                g_audio.resume();
                g_audio.clear();
            }
            else fprintf(stdout, "Still listening\n");
            return true;
        }
        else {
            if (g_listening) {
                g_listening = false;
                g_audio.get(0, g_pcmf32);
                g_audio.pause();
                fprintf(stdout, "Processing\n");
            }
            else fprintf(stdout, "Not listening\n");
            return true;
        }
    }
    return true;
}

bool get_audio(std::vector<float> & pcmf32_cur) {
    if (!read_input()) return false;
    if (!g_pcmf32.empty()) pcmf32_cur = std::move(g_pcmf32);
    else pcmf32_cur.clear();
    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // whisper init

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "%s: whisper_init_from_file_with_params() failed!\n", __func__);
        return 1;
    }

    // init audio

    if (!g_audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    struct whisper_full_params wparams = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
    wparams.offset_ms        = 0;
    wparams.translate        = false;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;
    wparams.no_timestamps    = true;

    wparams.max_tokens       = 32;
    wparams.audio_ctx        = 768; // partial encoder context for better performance

    wparams.temperature     = 0.0f;
    wparams.temperature_inc = 2.0f;
    wparams.greedy.best_of  = 1;

    wparams.beam_search.beam_size = 1;

    wparams.language         = "en";

    wparams.grammar_penalty = 100.0;

    wparams.initial_prompt = params.context.data();

    WChess::callbacks cb;
    cb.get_audio = get_audio;
    cb.set_move = set_move;

    WChess::settings s;
    s.vad_ms = 2000;
    s.prompt_ms = params.prompt_ms;
    s.command_ms = params.command_ms;
    s.vad_thold = params.vad_thold;
    s.freq_thold = params.freq_thold;
    s.print_energy = params.print_energy;

    g_wchess.reset(new WChess(ctx, wparams, cb, s));
    set_move("start", 0);
    g_wchess->run();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
