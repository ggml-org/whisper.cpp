// Talk with AI
//

#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <regex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

static std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t n_gpu_layers = 999;
    int32_t seed = 0;
    int32_t top_k = 5;
    int32_t min_keep = 1;
    float top_p = 0.80f;
    float min_p = 0.01f;
    float temp  = 0.30f;
    
    float vad_thold  = 0.6f;
    float freq_thold = 100.0f;

    bool translate      = false;
    bool print_special  = false;
    bool print_energy   = false;
    bool no_timestamps  = true;
    bool verbose_prompt = false;
    bool use_gpu        = true;
    bool flash_attn     = false;

    std::string person      = "Georgi";
    std::string bot_name    = "LLaMA";
    std::string wake_cmd    = "";
    std::string heard_ok    = "";
    std::string language    = "en";
    std::string model_wsp   = "models/ggml-base.en.bin";
    std::string model_llama = "models/ggml-llama-7B.bin";
    std::string speak       = "./examples/talk-llama/speak";
    std::string speak_file  = "./examples/talk-llama/to_speak.txt";
    std::string prompt      = "";
    std::string fname_out;
    std::string path_session = "";       // path to file for saving/loading model eval state
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")        { params.n_threads      = std::stoi(argv[++i]); }
        else if (arg == "-vms" || arg == "--voice-ms")       { params.voice_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")        { params.capture_id     = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")     { params.max_tokens     = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")      { params.audio_ctx      = std::stoi(argv[++i]); }
        else if (arg == "-ngl" || arg == "--n-gpu-layers")   { params.n_gpu_layers   = std::stoi(argv[++i]); }
        else if (arg == "--seed")                            { params.seed           = std::stoi(argv[++i]); }
        else if (arg == "--top-k")                           { params.top_k          = std::stoi(argv[++i]); }
        else if (arg == "--min-keep")                        { params.min_keep       = std::stoul(argv[++i]);}
        else if (arg == "--top-p")                           { params.top_p          = std::stof(argv[++i]); }
        else if (arg == "--min-p")                           { params.min_p          = std::stof(argv[++i]); }
        else if (arg == "--temp")                            { params.temp           = std::stof(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")      { params.vad_thold      = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")     { params.freq_thold     = std::stof(argv[++i]); }
        else if (arg == "-tr"  || arg == "--translate")      { params.translate      = true; }
        else if (arg == "-ps"  || arg == "--print-special")  { params.print_special  = true; }
        else if (arg == "-pe"  || arg == "--print-energy")   { params.print_energy   = true; }
        else if (arg == "-vp"  || arg == "--verbose-prompt") { params.verbose_prompt = true; }
        else if (arg == "-ng"  || arg == "--no-gpu")         { params.use_gpu        = false; }
        else if (arg == "-fa"  || arg == "--flash-attn")     { params.flash_attn     = true; }
        else if (arg == "-p"   || arg == "--person")         { params.person         = argv[++i]; }
        else if (arg == "-bn"   || arg == "--bot-name")      { params.bot_name       = argv[++i]; }
        else if (arg == "--session")                         { params.path_session   = argv[++i]; }
        else if (arg == "-w"   || arg == "--wake-command")   { params.wake_cmd       = argv[++i]; }
        else if (arg == "-ho"  || arg == "--heard-ok")       { params.heard_ok       = argv[++i]; }
        else if (arg == "-l"   || arg == "--language")       { params.language       = argv[++i]; }
        else if (arg == "-mw"  || arg == "--model-whisper")  { params.model_wsp      = argv[++i]; }
        else if (arg == "-ml"  || arg == "--model-llama")    { params.model_llama    = argv[++i]; }
        else if (arg == "-s"   || arg == "--speak")          { params.speak          = argv[++i]; }
        else if (arg == "-sf"  || arg == "--speak-file")     { params.speak_file     = argv[++i]; }
        else if (arg == "--prompt-file")                     {
            std::ifstream file(argv[++i]);
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
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
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n",           params.n_gpu_layers);
    fprintf(stderr, "  --seed N                   [%-7d] seed sampling\n",                               params.seed);
    fprintf(stderr, "  --top-k N                  [%-7d] top-k sampling (0 = disabled)\n",               params.top_k);
    fprintf(stderr, "  --min-keep N               [%-7d] minimum number of tokens to keep\n",            params.min_keep);
    fprintf(stderr, "  --top-p N                  [%-7.2f] top-p sampling\n",                            params.top_p);
    fprintf(stderr, "  --min-p N                  [%-7.2f] min-p sampling\n",                            params.min_p);
    fprintf(stderr, "  --temp N                   [%-7.2f] temperature\n",                               params.temp);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] voice activity detection threshold\n",        params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -tr,      --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  -vp,      --verbose-prompt [%-7s] print prompt at start\n",                       params.verbose_prompt ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu         [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn     [%-7s] flash attention\n",                             params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name (for prompt selection)\n",          params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name (to display)\n",                       params.bot_name.c_str());
    fprintf(stderr, "  -w TEXT,  --wake-command T [%-7s] wake-up command to listen for\n",               params.wake_cmd.c_str());
    fprintf(stderr, "  -ho TEXT, --heard-ok TEXT  [%-7s] said by TTS before generating reply\n",         params.heard_ok.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n",                          params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n",                            params.model_llama.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT     [%-7s] command for TTS\n",                             params.speak.c_str());
    fprintf(stderr, "  -sf FILE, --speak-file     [%-7s] file to pass to TTS\n",                         params.speak_file.c_str());
    fprintf(stderr, "  --prompt-file FNAME        [%-7s] file with custom prompt to start dialog\n",     "");
    fprintf(stderr, "  --session FNAME                   file to cache model state in (may be large!) (default: none)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME     [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "\n");
}

static std::string transcribe(
        whisper_context * ctx,
        const whisper_params & params,
        const std::vector<float> & pcmf32,
        const std::string prompt_text,
        float & prob,
        int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    prob = 0.0f;
    t_ms = 0;

    std::vector<whisper_token> prompt_tokens;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    prompt_tokens.resize(1024);
    prompt_tokens.resize(whisper_tokenize(ctx, prompt_text.c_str(), prompt_tokens.data(), prompt_tokens.size()));

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.prompt_tokens    = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
    wparams.prompt_n_tokens  = prompt_tokens.empty() ? 0       : prompt_tokens.size();

    wparams.audio_ctx        = params.audio_ctx;

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}

static std::vector<std::string> get_words(const std::string &txt) {
    std::vector<std::string> words;

    std::istringstream iss(txt);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

const std::string k_prompt_whisper = R"(A conversation with a person called {1}.)";

const std::string k_prompt_llama = R"(Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
The transcript only includes text, it does not include markup like HTML and Markdown.
{1} responds with short and concise answers.

{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It is {2} o'clock.
{0}{4} What year is it?
{1}{4} We are in {3}.
{0}{4} What is a cat?
{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{0}{4} Name a color.
{1}{4} Blue
{0}{4})";

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // whisper init

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx_wsp = whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);
    if (!ctx_wsp) {
        fprintf(stderr, "No whisper.cpp model specified. Please provide using -mw <modelfile>\n");
        return 1;
    }

    // llama init

    llama_backend_init();

    auto lmparams = llama_model_default_params();
    if (!params.use_gpu) {
        lmparams.n_gpu_layers = 0;
    } else {
        lmparams.n_gpu_layers = params.n_gpu_layers;
    }

    struct llama_model * model_llama = llama_model_load_from_file(params.model_llama.c_str(), lmparams);
    if (!model_llama) {
        fprintf(stderr, "No llama.cpp model specified. Please provide using -ml <modelfile>\n");
        return 1;
    }

    const llama_vocab * vocab_llama = llama_model_get_vocab(model_llama);

    llama_context_params lcparams = llama_context_default_params();

    // tune these to your liking
    lcparams.n_ctx      = 2048;
    lcparams.n_threads  = params.n_threads;
    lcparams.flash_attn = params.flash_attn;

    struct llama_context * ctx_llama = llama_init_from_model(model_llama, lcparams);

    // print some info about the processing
    {
        fprintf(stderr, "\n");

        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }

    // init audio

    audio_async audio(30*1000);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    bool is_running  = true;
    bool force_speak = false;

    float prob0 = 0.0f;

    const std::string chat_symb = ":";

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

    const std::string prompt_whisper = ::replace(k_prompt_whisper, "{1}", params.bot_name);

    // construct the initial prompt for LLaMA inference
    std::string prompt_llama = params.prompt.empty() ? k_prompt_llama : params.prompt;

    // need to have leading ' '
    prompt_llama.insert(0, 1, ' ');

    prompt_llama = ::replace(prompt_llama, "{0}", params.person);
    prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);

    {
        // get time string
        std::string time_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%H:%M", now);
            time_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
    }

    {
        // get year string
        std::string year_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%Y", now);
            year_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
    }

    prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);

    llama_batch batch = llama_batch_init(llama_n_ctx(ctx_llama), 0, 1);

    // init sampler
    auto sparams = llama_sampler_chain_default_params();

    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (params.temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, params.min_keep));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp (params.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist (params.seed));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p (params.min_p, params.min_keep));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    // init session
    std::string path_session = params.path_session;
    std::vector<llama_token> session_tokens;
    auto embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);

    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from %s\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(llama_n_ctx(ctx_llama));
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            for (size_t i = 0; i < session_tokens.size(); i++) {
                embd_inp[i] = session_tokens[i];
            }

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // evaluate the initial prompt

    printf("\n");
    printf("%s : initializing - please wait ...\n", __func__);

    // prepare batch
    {
        batch.n_tokens = embd_inp.size();

        for (int i = 0; i < batch.n_tokens; i++) {
            batch.token[i]     = embd_inp[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i == batch.n_tokens - 1;
        }
    }

    if (llama_decode(ctx_llama, batch)) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        return 1;
    }

    if (params.verbose_prompt) {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s", prompt_llama.c_str());
        fflush(stdout);
    }

     // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
    // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
    // initial prompt so it doesn't need to be an exact match.
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < (embd_inp.size() * 3 / 4);

    printf("%s : done! start speaking in the microphone\n", __func__);

    // show wake command if enabled
    const std::string wake_cmd = params.wake_cmd;
    const int wake_cmd_length = get_words(wake_cmd).size();
    const bool use_wake_cmd = wake_cmd_length > 0;

    if (use_wake_cmd) {
        printf("%s : the wake-up command is: '%s%s%s'\n", __func__, "\033[1m", wake_cmd.c_str(), "\033[0m");
    }

    printf("\n");
    printf("%s%s", params.person.c_str(), chat_symb.c_str());
    fflush(stdout);

    // clear audio buffer
    audio.clear();

    // text inference variables
    const int voice_id = 2;
    const int n_keep   = embd_inp.size();
    const int n_ctx    = llama_n_ctx(ctx_llama);

    int n_past = n_keep;
    int n_prev = 64; // TODO arg
    int n_session_consumed = !path_session.empty() && session_tokens.size() > 0 ? session_tokens.size() : 0;

    std::vector<llama_token> embd;

    // reverse prompts for detecting when it's time to stop speaking
    std::vector<std::string> antiprompts = {
        params.person + chat_symb,
    };

    // main loop
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        int64_t t_ms = 0;

        {
            audio.get(2000, pcmf32_cur);

            if (::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1250, params.vad_thold, params.freq_thold, params.print_energy) || force_speak) {
                //fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);

                audio.get(params.voice_ms, pcmf32_cur);

                std::string all_heard;

                if (!force_speak) {
                    all_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper, prob0, t_ms));
                }

                const auto words = get_words(all_heard);

                std::string wake_cmd_heard;
                std::string text_heard;

                for (int i = 0; i < (int) words.size(); ++i) {
                    if (i < wake_cmd_length) {
                        wake_cmd_heard += words[i] + " ";
                    } else {
                        text_heard += words[i] + " ";
                    }
                }

                // check if audio starts with the wake-up command if enabled
                if (use_wake_cmd) {
                    const float sim = similarity(wake_cmd_heard, wake_cmd);

                    if ((sim < 0.7f) || (text_heard.empty())) {
                        audio.clear();
                        continue;
                    }
                }

                // optionally give audio feedback that the current text is being processed
                if (!params.heard_ok.empty()) {
                    speak_with_file(params.speak, params.heard_ok, params.speak_file, voice_id);
                }

                // remove text between brackets using regex
                {
                    std::regex re("\\[.*?\\]");
                    text_heard = std::regex_replace(text_heard, re, "");
                }

                // remove text between brackets using regex
                {
                    std::regex re("\\(.*?\\)");
                    text_heard = std::regex_replace(text_heard, re, "");
                }

                // remove all characters, except for letters, numbers, punctuation and ':', '\'', '-', ' '
                text_heard = std::regex_replace(text_heard, std::regex("[^a-zA-Z0-9åäöÅÄÖ\\.,\\?!\\s\\:\\'\\-]"), "");

                // take first line
                text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));

                // remove leading and trailing whitespace
                text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
                text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");

                const std::vector<llama_token> tokens = llama_tokenize(ctx_llama, text_heard.c_str(), false);

                if (text_heard.empty() || tokens.empty() || force_speak) {
                    //fprintf(stdout, "%s: Heard nothing, skipping ...\n", __func__);
                    audio.clear();

                    continue;
                }

                force_speak = false;

                text_heard.insert(0, 1, ' ');
                text_heard += "\n" + params.bot_name + chat_symb;
                fprintf(stdout, "%s%s%s", "\033[1m", text_heard.c_str(), "\033[0m");
                fflush(stdout);

                embd = ::llama_tokenize(ctx_llama, text_heard, false);

                // Append the new input tokens to the session_tokens vector
                if (!path_session.empty()) {
                    session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
                }

                // text inference
                bool done = false;
                std::string text_to_speak;
                while (true) {
                    // predict
                    if (embd.size() > 0) {
                        if (n_past + (int) embd.size() > n_ctx) {
                            n_past = n_keep;

                            // insert n_left/2 tokens at the start of embd from last_n_tokens
                            embd.insert(embd.begin(), embd_inp.begin() + embd_inp.size() - n_prev, embd_inp.end());
                            // stop saving session if we run out of context
                            path_session = "";
                            //printf("\n---\n");
                            //printf("resetting: '");
                            //for (int i = 0; i < (int) embd.size(); i++) {
                            //    printf("%s", llama_token_to_piece(ctx_llama, embd[i]));
                            //}
                            //printf("'\n");
                            //printf("\n---\n");
                        }

                        // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
                        // REVIEW
                        if (n_session_consumed < (int) session_tokens.size()) {
                            size_t i = 0;
                            for ( ; i < embd.size(); i++) {
                                if (embd[i] != session_tokens[n_session_consumed]) {
                                    session_tokens.resize(n_session_consumed);
                                    break;
                                }

                                n_past++;
                                n_session_consumed++;

                                if (n_session_consumed >= (int) session_tokens.size()) {
                                    i++;
                                    break;
                                }
                            }
                            if (i > 0) {
                                embd.erase(embd.begin(), embd.begin() + i);
                            }
                        }

                        if (embd.size() > 0 && !path_session.empty()) {
                            session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                            n_session_consumed = session_tokens.size();
                        }

                        // prepare batch
                        {
                            batch.n_tokens = embd.size();

                            for (int i = 0; i < batch.n_tokens; i++) {
                                batch.token[i]     = embd[i];
                                batch.pos[i]       = n_past + i;
                                batch.n_seq_id[i]  = 1;
                                batch.seq_id[i][0] = 0;
                                batch.logits[i]    = i == batch.n_tokens - 1;
                            }
                        }

                        if (llama_decode(ctx_llama, batch)) {
                            fprintf(stderr, "%s : failed to decode\n", __func__);
                            return 1;
                        }
                    }


                    embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
                    n_past += embd.size();

                    embd.clear();

                    if (done) break;

                    {
                        // out of user input, sample next token

                        if (!path_session.empty() && need_to_save_session) {
                            need_to_save_session = false;
                            llama_state_save_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.size());
                        }

                        const llama_token id = llama_sampler_sample(smpl, ctx_llama, -1);

                        if (id != llama_vocab_eos(vocab_llama)) {
                            // add it to the context
                            embd.push_back(id);

                            text_to_speak += llama_token_to_piece(ctx_llama, id);

                            printf("%s", llama_token_to_piece(ctx_llama, id).c_str());
                            fflush(stdout);
                        }
                    }

                    {
                        std::string last_output;
                        for (int i = embd_inp.size() - 16; i < (int) embd_inp.size(); i++) {
                            last_output += llama_token_to_piece(ctx_llama, embd_inp[i]);
                        }
                        last_output += llama_token_to_piece(ctx_llama, embd[0]);

                        for (std::string & antiprompt : antiprompts) {
                            if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                                done = true;
                                text_to_speak = ::replace(text_to_speak, antiprompt, "");
                                fflush(stdout);
                                need_to_save_session = true;
                                break;
                            }
                        }
                    }

                    is_running = sdl_poll_events();

                    if (!is_running) {
                        break;
                    }
                }

                speak_with_file(params.speak, text_to_speak, params.speak_file, voice_id);

                audio.clear();
            }
        }
    }

    audio.pause();

    whisper_print_timings(ctx_wsp);
    whisper_free(ctx_wsp);

    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx_llama);

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx_llama);

    llama_backend_free();

    return 0;
}
