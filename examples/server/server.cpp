#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#define CPPHTTPLIB_LISTEN_BACKLOG_SIZE 128
#include "httplib.h"
#include "json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <atomic>
#include <map>
#include <random>
#include <iostream>
#include <sys/stat.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#else
#include <direct.h>  // For _mkdir
#include <io.h>      // For _access
#ifndef W_OK
#define W_OK 2
#endif
#endif
#include <semaphore>
#include <csignal>


// ---- PORTABLE COUNTING SEMAPHORE ----
class CountingSemaphore {
public:
    explicit CountingSemaphore(int initial) : count_(initial) {}
    void release() {
        std::unique_lock<std::mutex> lock(mtx_);
        ++count_;
        cv_.notify_one();
    }
    bool try_acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (count_ > 0) {
            --count_;
            return true;
        }
        return false;
    }
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    int count_;
};

using namespace httplib;
using json = nlohmann::ordered_json;

// Add debug logging macro for convenience
#define SERVER_DEBUG(msg) \
    do { \
        if (g_debug_mode) \
            std::cerr << "[SERVER_DEBUG] " << msg << std::endl; \
    } while (0)

// ---- GLOBALS FOR SIGNAL HANDLING ----
std::atomic<bool> shutdown_requested{false};
httplib::Server *g_svr_ptr = nullptr;
std::atomic<int> g_active_tasks{0};
std::unique_ptr<CountingSemaphore> task_queue_slots;
std::atomic<int> active_http_requests{0};
// --------------------------------------

bool g_debug_mode = false;

// RAII guard for HTTP request counting
struct HttpRequestCounter {
    explicit HttpRequestCounter(std::atomic<int>& counter) : ctr(counter) { ctr.fetch_add(1, std::memory_order_relaxed); }
    ~HttpRequestCounter() { ctr.fetch_sub(1, std::memory_order_relaxed); }
    std::atomic<int>& ctr;
};

// output formats
const std::string json_format   = "json";
const std::string text_format   = "text";
const std::string srt_format    = "srt";
const std::string vjson_format  = "verbose_json";
const std::string vtt_format    = "vtt";

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::string public_path = "examples/server/public";
    std::string request_path = "";
    std::string inference_path = "/inference";

    int32_t port          = 8080;
    int32_t read_timeout  = 600; ///< Read timeout in seconds
    int32_t write_timeout = 600; ///< Write timeout in seconds
    int32_t num_workers   = 1; // Default to 1 worker
    bool ffmpeg_converter = false;
    int32_t keep_alive_max_count = 5; ///< Max requests per connection
    int32_t keep_alive_timeout = 60;  ///< Idle keep-alive timeout in seconds
    int32_t listen_backlog = 128; ///< Maximum number of pending connections in listen backlog
    size_t max_upload_size = 100 * 1024 * 1024; ///< Maximum upload size in bytes (default 100MB)
    std::string temp_upload_dir = "/tmp/whisper_server_uploads/"; ///< Directory for temporary uploads
    int32_t inference_timeout_sec = 30; ///< Inference timeout in seconds (default 30)
    int32_t max_task_queue = 100; ///< Maximum number of tasks in the inference queue (default 100)
};

struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t best_of       = 2;
    int32_t beam_size     = -1;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float temperature     =  0.00f;
    float temperature_inc =  0.20f;
    float no_speech_thold = 0.6f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_realtime  = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool suppress_nst    = false;
    bool no_context      = false;

    std::string language        = "en";
    std::string prompt          = "";
    std::string font_path       = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model           = "models/ggml-base.en.bin";

    std::string response_format     = json_format;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";
};

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params, const server_params& sparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] \n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n",                    params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n",                           params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n",   params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ml N,     --max-len N         [%-7d] maximum segment length in characters\n",           params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word     [%-7s] split on word rather than on token\n",             params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n",              params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n",         params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n",           params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n",   params.logprob_thold);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n",           params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -tr,       --translate         [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n",                       params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback       [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,       --print-special     [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -pc,       --print-colors      [%-7s] print colors\n",                                   params.print_colors ? "true" : "false");
    fprintf(stderr, "  -pr,       --print-realtime    [%-7s] print output in realtime\n",                       params.print_realtime ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n",                                 params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",                        params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect)\n",       params.language.c_str());
    fprintf(stderr, "  -dl,       --detect-language   [%-7s] exit after automatically detecting language\n",    params.detect_language ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt\n",                                 params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
    // server params
    fprintf(stderr, "  -dtw MODEL --dtw MODEL         [%-7s] compute token-level timestamps\n", params.dtw.c_str());
    fprintf(stderr, "  --host HOST,                   [%-7s] Hostname/ip-adress for the server\n", sparams.hostname.c_str());
    fprintf(stderr, "  --port PORT,                   [%-7d] Port number for the server\n", sparams.port);
    fprintf(stderr, "  --public PATH,                 [%-7s] Path to the public folder\n", sparams.public_path.c_str());
    fprintf(stderr, "  --request-path PATH,           [%-7s] Request path for all requests\n", sparams.request_path.c_str());
    fprintf(stderr, "  --inference-path PATH,         [%-7s] Inference path for all requests\n", sparams.inference_path.c_str());
    fprintf(stderr, "  --convert,                     [%-7s] Convert audio to WAV, requires ffmpeg on the server\n", sparams.ffmpeg_converter ? "true" : "false");
    fprintf(stderr, "  -sns,      --suppress-nst      [%-7s] suppress non-speech tokens\n", params.suppress_nst ? "true" : "false");
    fprintf(stderr, "  -nth N,    --no-speech-thold N [%-7.2f] no speech threshold\n",   params.no_speech_thold);
    fprintf(stderr, "  -nc,       --no-context        [%-7s] do not use previous audio context\n", params.no_context ? "true" : "false");
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] do not use gpu\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "  --timeout N,                   [%-7d] Read/write timeout in seconds\n", sparams.read_timeout);
    fprintf(stderr, "  --keep-alive-max N,            [%-7d] Max requests per connection (keep-alive)\n", sparams.keep_alive_max_count);
    fprintf(stderr, "  --keep-alive-timeout N,        [%-7d] Idle keep-alive timeout in seconds\n", sparams.keep_alive_timeout);
    fprintf(stderr, "  --backlog N,                   [%-7d] Listen() backlog (max pending connections)\n", sparams.listen_backlog);
    fprintf(stderr, "  --max-upload-size N,           [%-7zu] Max upload size in bytes (default 104857600)\n", sparams.max_upload_size);
    fprintf(stderr, "  --temp-upload-dir DIR,          [%-7s] Directory for temporary uploads\n", sparams.temp_upload_dir.c_str());
    fprintf(stderr, "  --inference-timeout N,         [%-7d] Inference timeout in seconds (default 30)\n", sparams.inference_timeout_sec);
    fprintf(stderr, "  --max-queue N,                 [%-7d] Maximum number of tasks in the inference queue\n", sparams.max_task_queue);
    fprintf(stderr, "\n");
}

bool whisper_params_parse(int argc, char ** argv, whisper_params & params, server_params & sparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params, sparams);
            exit(0);
        }
        try {
            if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(argv[++i]); }
            else if (arg == "-p"    || arg == "--processors")      { params.n_processors    = std::stoi(argv[++i]); }
            else if (arg == "-ot"   || arg == "--offset-t")        { params.offset_t_ms     = std::stoi(argv[++i]); }
            else if (arg == "-on"   || arg == "--offset-n")        { params.offset_n        = std::stoi(argv[++i]); }
            else if (arg == "-d"    || arg == "--duration")        { params.duration_ms     = std::stoi(argv[++i]); }
            else if (arg == "-mc"   || arg == "--max-context")     { params.max_context     = std::stoi(argv[++i]); }
            else if (arg == "-ml"   || arg == "--max-len")         { params.max_len         = std::stoi(argv[++i]); }
            else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(argv[++i]); }
            else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(argv[++i]); }
            else if (arg == "-ac"   || arg == "--audio-ctx")       { params.audio_ctx       = std::stoi(argv[++i]); }
            else if (arg == "-wt"   || arg == "--word-thold")      { params.word_thold      = std::stof(argv[++i]); }
            else if (arg == "-et"   || arg == "--entropy-thold")   { params.entropy_thold   = std::stof(argv[++i]); }
            else if (arg == "-lpt"  || arg == "--logprob-thold")   { params.logprob_thold   = std::stof(argv[++i]); }
            else if (arg == "-debug"|| arg == "--debug-mode")      { params.debug_mode      = true; }
            else if (arg == "-tr"   || arg == "--translate")       { params.translate       = true; }
            else if (arg == "-di"   || arg == "--diarize")         { params.diarize         = true; }
            else if (arg == "-tdrz" || arg == "--tinydiarize")     { params.tinydiarize     = true; }
            else if (arg == "-sow"  || arg == "--split-on-word")   { params.split_on_word   = true; }
            else if (arg == "-nf"   || arg == "--no-fallback")     { params.no_fallback     = true; }
            else if (arg == "-fp"   || arg == "--font-path")       { params.font_path       = argv[++i]; }
            else if (arg == "-ps"   || arg == "--print-special")   { params.print_special   = true; }
            else if (arg == "-pc"   || arg == "--print-colors")    { params.print_colors    = true; }
            else if (arg == "-pr"   || arg == "--print-realtime")  { params.print_realtime  = true; }
            else if (arg == "-pp"   || arg == "--print-progress")  { params.print_progress  = true; }
            else if (arg == "-nt"   || arg == "--no-timestamps")   { params.no_timestamps   = true; }
            else if (arg == "-l"    || arg == "--language")        { params.language        = argv[++i]; }
            else if (arg == "-dl"   || arg == "--detect-language") { params.detect_language = true; }
            else if (                  arg == "--prompt")          { params.prompt          = argv[++i]; }
            else if (arg == "-m"    || arg == "--model")           { params.model           = argv[++i]; }
            else if (arg == "-oved" || arg == "--ov-e-device")     { params.openvino_encode_device = argv[++i]; }
            else if (arg == "-dtw"  || arg == "--dtw")             { params.dtw             = argv[++i]; }
            else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu         = false; }
            else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn      = true; }
            else if (arg == "-sns"  || arg == "--suppress-nst")    { params.suppress_nst    = true; }
            else if (arg == "-nth"  || arg == "--no-speech-thold") { params.no_speech_thold = std::stof(argv[++i]); }
            else if (arg == "-nc"   || arg == "--no-context")      { params.no_context      = true; }
            else if (                  arg == "--workers")         { sparams.num_workers   = std::stoi(argv[++i]); }
            else if (                  arg == "--port")            { sparams.port        = std::stoi(argv[++i]); }
            else if (                  arg == "--host")            { sparams.hostname    = argv[++i]; }
            else if (                  arg == "--public")          { sparams.public_path = argv[++i]; }
            else if (                  arg == "--request-path")    { sparams.request_path = argv[++i]; }
            else if (                  arg == "--inference-path")  { sparams.inference_path = argv[++i]; }
            else if (                  arg == "--convert")         { sparams.ffmpeg_converter     = true; }
            else if (                  arg == "--timeout")         { sparams.read_timeout = sparams.write_timeout = std::stoi(argv[++i]); }
            else if (                  arg == "--keep-alive-max")  { sparams.keep_alive_max_count = std::stoi(argv[++i]); }
            else if (                  arg == "--keep-alive-timeout") { sparams.keep_alive_timeout = std::stoi(argv[++i]); }
            else if (                  arg == "--backlog")         { sparams.listen_backlog = std::stoi(argv[++i]); }
            else if (                  arg == "--max-upload-size") { sparams.max_upload_size = std::stoull(argv[++i]); }
            else if (                  arg == "--temp-upload-dir") { sparams.temp_upload_dir = argv[++i]; }
            else if (                  arg == "--inference-timeout") { sparams.inference_timeout_sec = std::stoi(argv[++i]); }
            else if (                  arg == "--max-queue")        { sparams.max_task_queue = std::stoi(argv[++i]); }
            else {
                fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
                whisper_print_usage(argc, argv, params, sparams);
                exit(0);
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "error: invalid value for %s: %s. %s\n", arg.c_str(), argv[i], e.what());
            whisper_print_usage(argc, argv, params, sparams);
            exit(1);
        }
    }

    return true;
}

struct whisper_print_user_data {
    const whisper_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

void check_ffmpeg_availability() {
    int result = system("ffmpeg -version");

    if (result == 0) {
        std::cout << "ffmpeg is available." << std::endl;
    } else {
        // ffmpeg is not available
        std::cout << "ffmpeg is not found. Please ensure that ffmpeg is installed ";
        std::cout << "and that its executable is included in your system's PATH. ";
        exit(0);
    }
}

std::string generate_temp_filename(const std::string &prefix, const std::string &extension) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long long> dist(0, 1e9);

    std::stringstream ss;
    std::tm tm_buf; // For localtime_r
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm_buf, &now_time_t);
#else
    localtime_r(&now_time_t, &tm_buf);
#endif
    ss << prefix
       << "-"
       << std::put_time(&tm_buf, "%Y%m%d-%H%M%S") // Use tm_buf
       << "-"
       << dist(rng)
       << extension;

    return ss.str();
}

bool convert_to_wav(const std::string & temp_filename, std::string & error_resp) {
    std::ostringstream cmd_stream;
    std::string converted_filename_temp = temp_filename + "_temp.wav";
    cmd_stream << "ffmpeg -i \"" << temp_filename << "\" -y -ar 16000 -ac 1 -c:a pcm_s16le \"" << converted_filename_temp << "\" 2>&1";
    std::string cmd = cmd_stream.str();

    int status = std::system(cmd.c_str());
    if (status != 0) {
        error_resp = "{\"error\":\"FFmpeg conversion failed.\"}";
        return false;
    }

    // Remove the original file
    if (remove(temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to remove the original file.\"}";
        return false;
    }

    // Rename the temporary file to match the original filename
    if (rename(converted_filename_temp.c_str(), temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to rename the temporary file.\"}";
        return false;
    }
    return true;
}

std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1*energy1) {
        speaker = "0";
    } else if (energy1 > 1.1*energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

void whisper_print_progress_callback(struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
    int progress_step = ((whisper_print_user_data *) user_data)->params->progress_step;
    int * progress_prev  = &(((whisper_print_user_data *) user_data)->progress_prev);
    if (progress >= *progress_prev + progress_step) {
        *progress_prev += progress_step;
        fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
    }
}

void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        if (params.print_colors) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p   (ctx, i, j);

                const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size()))));

                printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
            }
        } else {
            const char * text = whisper_full_get_segment_text(ctx, i);

            printf("%s%s", speaker.c_str(), text);
        }

        if (params.tinydiarize) {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }
        fflush(stdout);
    }
}

std::string output_str(struct whisper_context * ctx, const whisper_params & params, std::vector<std::vector<float>> pcmf32s) {
    std::stringstream result;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        std::string speaker = "";

        if (params.diarize && pcmf32s.size() == 2)
        {
            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        result << speaker << text << "\n";
    }
    return result.str();
}

bool parse_str_to_bool(const std::string & s) {
    if (s == "true" || s == "1" || s == "yes" || s == "y") {
        return true;
    }
    return false;
}

// --- Structured Error Response Helpers ---
enum class ErrorType {
    INVALID_REQUEST,
    SERVER_ERROR,
    NOT_FOUND,
};

std::string error_type_to_string(ErrorType type) {
    switch (type) {
        case ErrorType::INVALID_REQUEST: return "invalid_request_error";
        case ErrorType::SERVER_ERROR:    return "server_error";
        case ErrorType::NOT_FOUND:       return "not_found_error";
        default:                               return "unknown_error";
    }
}

int error_type_to_http_status(ErrorType type) {
    switch (type) {
        case ErrorType::INVALID_REQUEST: return 400;
        case ErrorType::SERVER_ERROR:    return 500;
        case ErrorType::NOT_FOUND:       return 404;
        default:                               return 500;
    }
}

void set_error_response(httplib::Response &res, const std::string& message, ErrorType type) {
    json error_json = {
        {"error", {
            {"message", message},
            {"type", error_type_to_string(type)},
            {"code", error_type_to_http_status(type)}
        }}
    };
    res.set_content(error_json.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    res.status = error_type_to_http_status(type);
}

// --- Flexible Parameter Parsing (JSON or multipart) ---
bool get_request_parameters_flexible(const httplib::Request & req, whisper_params & params, bool &is_json_request, httplib::Response &res_for_error) {
    is_json_request = false;
    std::string content_type = req.get_header_value("Content-Type");
    json json_body;
    if (content_type.rfind("application/json", 0) == 0) {
        is_json_request = true;
        try {
            json_body = json::parse(req.body);
        } catch (json::parse_error& e) {
            SERVER_DEBUG("JSON parse error: " << e.what());
            set_error_response(res_for_error, "Invalid JSON body: " + std::string(e.what()), ErrorType::INVALID_REQUEST);
            return false; // Indicate failure
        }
    }

    auto get_param_value = [&](const std::string& key, const std::string& default_val_str = "") -> std::string {
        if (is_json_request) {
            if (json_body.contains(key) && !json_body[key].is_null()) {
                if (json_body[key].is_string()) return json_body[key].get<std::string>();
                if (json_body[key].is_number()) return std::to_string(json_body[key].get<double>());
                if (json_body[key].is_boolean()) return json_body[key].get<bool>() ? "true" : "false";
            }
        } else if (req.has_file(key)) {
            return req.get_file_value(key).content;
        }
        return default_val_str;
    };

    try {
        std::string offset_t_str = get_param_value("offset_t");
        if (!offset_t_str.empty()) params.offset_t_ms = std::stoi(offset_t_str);
        std::string offset_n_str = get_param_value("offset_n");
        if (!offset_n_str.empty()) params.offset_n = std::stoi(offset_n_str);
        std::string duration_str = get_param_value("duration");
        if (!duration_str.empty()) params.duration_ms = std::stoi(duration_str);
        std::string max_context_str = get_param_value("max_context");
        if (!max_context_str.empty()) params.max_context = std::stoi(max_context_str);
        std::string max_len_str = get_param_value("max_len");
        if (!max_len_str.empty()) params.max_len = std::stoi(max_len_str);
        std::string best_of_str = get_param_value("best_of");
        if (!best_of_str.empty()) params.best_of = std::stoi(best_of_str);
        std::string beam_size_str = get_param_value("beam_size");
        if (!beam_size_str.empty()) params.beam_size = std::stoi(beam_size_str);
        std::string audio_ctx_str = get_param_value("audio_ctx");
        if (!audio_ctx_str.empty()) params.audio_ctx = std::stof(audio_ctx_str);
        std::string word_thold_str = get_param_value("word_thold");
        if (!word_thold_str.empty()) params.word_thold = std::stof(word_thold_str);
        std::string entropy_thold_str = get_param_value("entropy_thold");
        if (!entropy_thold_str.empty()) params.entropy_thold = std::stof(entropy_thold_str);
        std::string logprob_thold_str = get_param_value("logprob_thold");
        if (!logprob_thold_str.empty()) params.logprob_thold = std::stof(logprob_thold_str);
        std::string temperature_str = get_param_value("temperature");
        if (!temperature_str.empty()) params.temperature = std::stof(temperature_str);
        std::string temperature_inc_str = get_param_value("temperature_inc");
        if (!temperature_inc_str.empty()) params.temperature_inc = std::stof(temperature_inc_str);
    } catch (const std::exception &e) {
        SERVER_DEBUG("Error parsing numeric parameter: " << e.what());
        set_error_response(res_for_error, std::string("Invalid numeric parameter: ") + e.what(), ErrorType::INVALID_REQUEST);
        return false; // Indicate failure
    }

    std::string debug_mode_str = get_param_value("debug_mode");
    if (!debug_mode_str.empty()) params.debug_mode = parse_str_to_bool(debug_mode_str);
    std::string translate_str = get_param_value("translate");
    if (!translate_str.empty()) params.translate = parse_str_to_bool(translate_str);
    std::string diarize_str = get_param_value("diarize");
    if (!diarize_str.empty()) params.diarize = parse_str_to_bool(diarize_str);
    std::string tinydiarize_str = get_param_value("tinydiarize");
    if (!tinydiarize_str.empty()) params.tinydiarize = parse_str_to_bool(tinydiarize_str);
    std::string split_on_word_str = get_param_value("split_on_word");
    if (!split_on_word_str.empty()) params.split_on_word = parse_str_to_bool(split_on_word_str);
    std::string no_timestamps_str = get_param_value("no_timestamps");
    if (!no_timestamps_str.empty()) params.no_timestamps = parse_str_to_bool(no_timestamps_str);
    std::string language_str = get_param_value("language");
    if (!language_str.empty()) params.language = language_str;
    std::string detect_language_str = get_param_value("detect_language");
    if (!detect_language_str.empty()) params.detect_language = parse_str_to_bool(detect_language_str);
    std::string prompt_str = get_param_value("prompt");
    if (!prompt_str.empty()) params.prompt = prompt_str;
    std::string response_format_str = get_param_value("response_format");
    if (!response_format_str.empty()) params.response_format = response_format_str;
    std::string suppress_non_speech_str = get_param_value("suppress_non_speech");
    if (!suppress_non_speech_str.empty()) params.suppress_nst = parse_str_to_bool(suppress_non_speech_str);
    std::string suppress_nst_str = get_param_value("suppress_nst");
    if (!suppress_nst_str.empty()) params.suppress_nst = parse_str_to_bool(suppress_nst_str);
    std::string no_context_str = get_param_value("no_context");
    if (!no_context_str.empty()) params.no_context = parse_str_to_bool(no_context_str);
    return true; // Indicate success
}

// --- Task Queue and Worker Thread for Concurrency ---
struct TranscriptionTask {
    int id;
    std::shared_ptr<std::vector<float>> audio_data_pcmf32_ptr;
    std::shared_ptr<std::vector<std::vector<float>>> audio_data_pcmf32s_ptr;
    whisper_params params;
    std::promise<json> result_promise;
    std::string original_filename;
    std::shared_ptr<std::atomic<bool>> connection_alive;
    std::shared_ptr<std::atomic<bool>> cancel_flag;
};

std::queue<TranscriptionTask> task_queue;
std::mutex task_queue_mutex;
std::condition_variable task_queue_cv;
std::atomic<int> next_task_id{0};

// RAII for whisper_context
using ContextPtr = std::unique_ptr<whisper_context, decltype(&whisper_free)>;

std::vector<ContextPtr> context_pool;
std::vector<std::thread> worker_threads;
std::mutex contextPoolMutex;
std::mutex contextManagementMutex;

// RAII thread guard for monitor thread
class ThreadGuard { // Renamed
    std::thread t;
public:
    explicit ThreadGuard(std::thread&& thr) : t(std::move(thr)) {}
    ~ThreadGuard() { if (t.joinable()) t.join(); }
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
    ThreadGuard(ThreadGuard&& other) noexcept : t(std::move(other.t)) {}
    ThreadGuard& operator=(ThreadGuard&& other) noexcept { if (t.joinable()) t.join(); t = std::move(other.t); return *this; }
};

/**
 * @brief Sets the Dynamic Time Warping (DTW) preset parameters for whisper.
 * 
 * @param dtw_str The string representing the desired DTW preset (e.g., "tiny", "base.en").
 * @param cparams The whisper_context_params struct to be modified.
 */
void set_dtw_preset(const std::string& dtw_str, whisper_context_params &cparams) {
    if (!dtw_str.empty()) {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;
        if (dtw_str == "tiny") cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        else if (dtw_str == "tiny.en") cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        else if (dtw_str == "base") cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        else if (dtw_str == "base.en") cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        else if (dtw_str == "small") cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        else if (dtw_str == "small.en") cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        else if (dtw_str == "medium") cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        else if (dtw_str == "medium.en") cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        else if (dtw_str == "large.v1") cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        else if (dtw_str == "large.v2") cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        else if (dtw_str == "large.v3") cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        
        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE && dtw_str != "none") { // Allow explicit "none"
            fprintf(stderr, "warning: unknown DTW preset '%s', disabling DTW timestamps.\n", dtw_str.c_str());
            cparams.dtw_token_timestamps = false; // Disable if preset is unknown and not explicitly none
        }
    }
}

/**
 * @brief Worker thread function that processes transcription tasks from a queue.
 * 
 * Each worker thread has its own whisper_context.
 * It waits for tasks on a condition variable, processes them, and sets the result in a promise.
 * 
 * @param ctx Pointer to the whisper_context for this worker thread.
 */
void process_transcription_tasks(whisper_context * ctx) {
    whisper_print_user_data user_data_worker_s{};
    while (true) {
        TranscriptionTask current_task;
        {
            std::unique_lock<std::mutex> lock(task_queue_mutex);
            task_queue_cv.wait_for(lock, std::chrono::milliseconds(100), []{
                return shutdown_requested.load(std::memory_order_acquire) || !task_queue.empty();
            });
            if (shutdown_requested.load(std::memory_order_acquire) && task_queue.empty()) {
                SERVER_DEBUG("Worker exiting: shutdown requested and task queue empty.");
                break;
            }
            if (task_queue.empty()) {
                continue;
            }
            current_task = std::move(task_queue.front());
            task_queue.pop();
            if (task_queue_slots) task_queue_slots->release();
            SERVER_DEBUG("Worker picked up task id: " << current_task.id << ", queue size now: " << task_queue.size());
        }
        g_active_tasks.fetch_add(1, std::memory_order_relaxed);
        try {
            SERVER_DEBUG("Worker starting inference for task id: " << current_task.id);
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.strategy = current_task.params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
            wparams.print_realtime   = false;
            wparams.print_progress   = current_task.params.print_progress;
            wparams.print_timestamps = !current_task.params.no_timestamps;
            wparams.print_special    = current_task.params.print_special;
            wparams.translate        = current_task.params.translate;
            wparams.language         = current_task.params.language.c_str();
            wparams.detect_language  = current_task.params.detect_language;
            wparams.n_threads        = current_task.params.n_threads;
            wparams.n_max_text_ctx   = current_task.params.max_context >= 0 ? current_task.params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms        = current_task.params.offset_t_ms;
            wparams.duration_ms      = current_task.params.duration_ms;
            wparams.thold_pt         = current_task.params.word_thold;
            wparams.max_len          = current_task.params.max_len == 0 ? 60 : current_task.params.max_len;
            wparams.split_on_word    = current_task.params.split_on_word;
            wparams.audio_ctx        = current_task.params.audio_ctx;
            wparams.debug_mode       = current_task.params.debug_mode;
            wparams.tdrz_enable      = current_task.params.tinydiarize;
            wparams.initial_prompt   = current_task.params.prompt.c_str();
            wparams.greedy.best_of        = current_task.params.best_of;
            wparams.beam_search.beam_size = current_task.params.beam_size;
            wparams.temperature      = current_task.params.temperature;
            wparams.no_speech_thold = current_task.params.no_speech_thold;
            wparams.temperature_inc  = current_task.params.temperature_inc;
            wparams.entropy_thold    = current_task.params.entropy_thold;
            wparams.logprob_thold    = current_task.params.logprob_thold;
            wparams.no_timestamps    = current_task.params.no_timestamps;
            wparams.token_timestamps = !current_task.params.no_timestamps && current_task.params.response_format == vjson_format;
            wparams.no_context       = current_task.params.no_context;
            wparams.suppress_nst     = current_task.params.suppress_nst;
            wparams.progress_callback           = whisper_print_progress_callback;
            user_data_worker_s.params = &current_task.params;
            user_data_worker_s.pcmf32s = current_task.audio_data_pcmf32s_ptr.get();
            user_data_worker_s.progress_prev = 0;
            wparams.progress_callback_user_data = &user_data_worker_s;
            wparams.new_segment_callback = nullptr;

            if (current_task.cancel_flag) {
                wparams.abort_callback = [](void* user_data) -> bool {
                    auto* task_cancel_flag = static_cast<std::atomic<bool>*>(user_data);
                    return (task_cancel_flag && *task_cancel_flag) || shutdown_requested.load(std::memory_order_acquire);
                };
                wparams.abort_callback_user_data = current_task.cancel_flag.get();
            } else {
                wparams.abort_callback = [](void*) -> bool {
                    return shutdown_requested.load(std::memory_order_acquire);
                };
                wparams.abort_callback_user_data = nullptr;
            }

            if (whisper_full_parallel(ctx, wparams, current_task.audio_data_pcmf32_ptr->data(), current_task.audio_data_pcmf32_ptr->size(), current_task.params.n_processors) != 0) {
                if (current_task.connection_alive && !(*current_task.connection_alive)) {
                    SERVER_DEBUG("Worker task id: " << current_task.id << " - client disconnected during whisper_full_parallel.");
                    current_task.result_promise.set_value({ {"error", "client disconnected"} });
                } else if (current_task.cancel_flag && *current_task.cancel_flag) {
                    SERVER_DEBUG("Worker task id: " << current_task.id << " - whisper_full_parallel aborted by cancel_flag.");
                    current_task.result_promise.set_value({ {"error", "Transcription cancelled by server"} });
                } else {
                    SERVER_DEBUG("Worker task id: " << current_task.id << " - whisper_full_parallel failed.");
                    current_task.result_promise.set_value({ {"error", "failed to process audio by worker"} });
                }
            } else {
                SERVER_DEBUG("Worker finished inference for task id: " << current_task.id);
                if (current_task.params.response_format == text_format) {
                    current_task.result_promise.set_value({ {"text", output_str(ctx, current_task.params, *current_task.audio_data_pcmf32s_ptr)} });
                } else if (current_task.params.response_format == srt_format) {
                    std::stringstream ss;
                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const char * text = whisper_full_get_segment_text(ctx, i);
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                        std::string speaker = "";
                        if (current_task.params.diarize && current_task.audio_data_pcmf32s_ptr->size() == 2) {
                            speaker = estimate_diarization_speaker(*current_task.audio_data_pcmf32s_ptr, t0, t1);
                        }
                        ss << i + 1 + current_task.params.offset_n << "\n";
                        ss << to_timestamp(t0, true) << " --> " << to_timestamp(t1, true) << "\n";
                        ss << speaker << text << "\n\n";
                    }
                    current_task.result_promise.set_value({ {"srt", ss.str()} });
                } else if (current_task.params.response_format == vtt_format) {
                    std::stringstream ss;
                    ss << "WEBVTT\n\n";
                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const char * text = whisper_full_get_segment_text(ctx, i);
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                        std::string speaker = "";
                        if (current_task.params.diarize && current_task.audio_data_pcmf32s_ptr->size() == 2) {
                            speaker = estimate_diarization_speaker(*current_task.audio_data_pcmf32s_ptr, t0, t1, true);
                            speaker.insert(0, "<v Speaker");
                            speaker.append(">");
                        }
                        ss << to_timestamp(t0) << " --> " << to_timestamp(t1) << "\n";
                        ss << speaker << text << "\n\n";
                    }
                    current_task.result_promise.set_value({ {"vtt", ss.str()} });
                } else if (current_task.params.response_format == vjson_format) {
                    std::string results_vjson = output_str(ctx, current_task.params, *current_task.audio_data_pcmf32s_ptr);
                    std::vector<float> lang_probs_vjson(whisper_lang_max_id() + 1, 0.0f);
                    const auto detected_lang_id_vjson = whisper_lang_auto_detect(ctx, 0, current_task.params.n_threads, lang_probs_vjson.data());
                    json jres_vjson = json{
                        {"task", current_task.params.translate ? "translate" : "transcribe"},
                        {"language", whisper_lang_str_full(whisper_full_lang_id(ctx))},
                        {"duration", float(current_task.audio_data_pcmf32_ptr->size())/WHISPER_SAMPLE_RATE},
                        {"text", results_vjson},
                        {"segments", json::array()},
                        {"detected_language", whisper_lang_str_full(detected_lang_id_vjson)},
                        {"detected_language_probability", lang_probs_vjson[detected_lang_id_vjson]},
                        {"language_probabilities", json::object()}
                    };
                    for (int i = 0; i <= whisper_lang_max_id(); ++i) {
                        if (lang_probs_vjson[i] > 0.001f) {
                            jres_vjson["language_probabilities"][whisper_lang_str(i)] = lang_probs_vjson[i];
                        }
                    }
                    const int n_segments_vjson = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments_vjson; ++i) {
                        json segment_vjson = json{
                            {"id", i},
                            {"text", whisper_full_get_segment_text(ctx, i)},
                        };
                        if (!current_task.params.no_timestamps) {
                            segment_vjson["start"] = whisper_full_get_segment_t0(ctx, i) * 0.01;
                            segment_vjson["end"] = whisper_full_get_segment_t1(ctx, i) * 0.01;
                        }
                        jres_vjson["segments"].push_back(segment_vjson);
                    }
                    current_task.result_promise.set_value(jres_vjson);
                } else {
                    std::string results_default = output_str(ctx, current_task.params, *current_task.audio_data_pcmf32s_ptr);
                    current_task.result_promise.set_value({ {"text", results_default} });
                }
            }
        } catch (const std::exception& e) {
            SERVER_DEBUG("Worker task id: " << current_task.id << " - EXCEPTION during processing: " << e.what());
            try {
                current_task.result_promise.set_value({ {"error", "Internal server error during transcription: " + std::string(e.what())} });
            } catch (const std::exception& promise_ex) {
                SERVER_DEBUG("Worker task id: " << current_task.id << " - EXCEPTION while setting promise after another exception: " << promise_ex.what());
            }
        } catch (...) {
            SERVER_DEBUG("Worker task id: " << current_task.id << " - UNKNOWN EXCEPTION during processing.");
            try {
                current_task.result_promise.set_value({ {"error", "Unknown internal server error during transcription"} });
            } catch (const std::exception& promise_ex) {
                SERVER_DEBUG("Worker task id: " << current_task.id << " - EXCEPTION while setting promise after unknown exception: " << promise_ex.what());
            }
        }
        g_active_tasks.fetch_sub(1, std::memory_order_relaxed);
        SERVER_DEBUG("Worker finished handling task id: " << current_task.id << ". Active tasks: " << g_active_tasks.load());
    }
}

// Forward declaration for parse_request_data, enqueue_task, and prepare_and_send_response
bool parse_request_data(const Request &req, const server_params &sparams, whisper_params &req_params, bool &is_json_req, std::string &filename, std::shared_ptr<std::vector<float>> &pcmf32_ptr, std::shared_ptr<std::vector<std::vector<float>>> &pcmf32s_ptr, Response &res);
bool enqueue_task(std::atomic<int> &task_id_counter, std::shared_ptr<std::vector<float>> pcmf32_ptr, std::shared_ptr<std::vector<std::vector<float>>> pcmf32s_ptr, const std::string& original_filename, const whisper_params& params, std::shared_ptr<std::atomic<bool>> connection_alive, std::shared_ptr<std::atomic<bool>> cancel_flag, const server_params& sparams, std::shared_future<json>& result_fut);
void prepare_and_send_response(Response &res, const Request &req, std::shared_future<json> fut, std::shared_ptr<std::atomic<bool>> connection_alive, std::shared_ptr<std::atomic<bool>> cancel_flag, int task_id, const server_params& sparams);

void handle_inference_request(
    const Request &req,
    Response &res,
    whisper_params &global_default_params,
    const server_params &sparams,
    std::atomic<int> &task_id_counter)
{
    SERVER_DEBUG("Task ID " << task_id_counter.load() << ": handle_inference_request START for path " << req.path);
    whisper_params req_params = global_default_params;
    bool is_json_req = false;
    std::string filename;
    auto pcmf32_ptr = std::make_shared<std::vector<float>>();
    auto pcmf32s_ptr = std::make_shared<std::vector<std::vector<float>>>();

    if (!parse_request_data(req, sparams, req_params, is_json_req, filename, pcmf32_ptr, pcmf32s_ptr, res)) {
        // parse_request_data already set error response
        SERVER_DEBUG("Task ID " << task_id_counter.load() << ": parse_request_data FAILED. Response status: " << res.status);
        return;
    }
    SERVER_DEBUG("Audio loaded and parsed for file: " << filename << " for Task ID " << task_id_counter.load());

    auto connection_alive = std::make_shared<std::atomic<bool>>(true); // For worker to know if client disconnected
    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);    // For handler to signal worker (e.g. on client disconnect)

    int current_task_id = task_id_counter.load(); // Get current id for logging before enqueue
    std::shared_future<json> fut;
    if (!enqueue_task(task_id_counter, pcmf32_ptr, pcmf32s_ptr, filename, req_params, connection_alive, cancel_flag, sparams, fut)) {
        SERVER_DEBUG("Task ID " << current_task_id << ": enqueue_task FAILED, queue full.");
        set_error_response(res, "Service unavailable: Task queue is full.", ErrorType::SERVER_ERROR);
        res.status = 503;
        return;
    }

    prepare_and_send_response(res, req, fut, connection_alive, cancel_flag, current_task_id, sparams);
    SERVER_DEBUG("Task ID " << current_task_id << ": handle_inference_request END. Response status: " << res.status);
}

bool parse_request_data(const Request &req, const server_params &sparams, whisper_params &req_params, bool &is_json_req, std::string &filename, std::shared_ptr<std::vector<float>> &pcmf32_ptr, std::shared_ptr<std::vector<std::vector<float>>> &pcmf32s_ptr, Response &res) {
    SERVER_DEBUG("parse_request_data: START");
    if (!get_request_parameters_flexible(req, req_params, is_json_req, res)) {
        SERVER_DEBUG("parse_request_data: get_request_parameters_flexible FAILED. Status: " << res.status);
        return false; // Propagate failure
    }
    if (!is_json_req) {
        if (!req.has_file("file")) {
            set_error_response(res, "No 'file' field in the request", ErrorType::INVALID_REQUEST);
            SERVER_DEBUG("parse_request_data: No 'file' field. Status: " << res.status);
            return false;
        }
        auto audio_file = req.get_file_value("file");
        filename = audio_file.filename;
        if (audio_file.content.length() > sparams.max_upload_size) {
            set_error_response(res, "Audio file too large.", ErrorType::INVALID_REQUEST);
            SERVER_DEBUG("parse_request_data: Audio file too large. Status: " << res.status);
            return false;
        }
        if (sparams.ffmpeg_converter) {
            const std::string temp_upload_name = generate_temp_filename("upload", ".dat");
            const std::string temp_filename = sparams.temp_upload_dir + temp_upload_name;
            std::ofstream temp_file{temp_filename, std::ios::binary};
            if (!temp_file.is_open()) {
                SERVER_DEBUG("Failed to open temporary file for writing: " << temp_filename);
                set_error_response(res, "Server error: Could not write temporary file.", ErrorType::SERVER_ERROR);
                return false;
            }
            temp_file.write(audio_file.content.data(), audio_file.content.length());
            temp_file.close();
            if (!temp_file) { // Check for errors after closing
                SERVER_DEBUG("Failed to write to temporary file: " << temp_filename);
                set_error_response(res, "Server error: Could not write temporary file content.", ErrorType::SERVER_ERROR);
                std::remove(temp_filename.c_str()); // Attempt to clean up
                return false;
            }
            std::string error_resp_ffmpeg_detail;
            const bool is_converted = convert_to_wav(temp_filename, error_resp_ffmpeg_detail);
            if (!is_converted) {
                SERVER_DEBUG("FFmpeg conversion failed: " << error_resp_ffmpeg_detail);
                set_error_response(res, "FFmpeg audio conversion failed.", ErrorType::SERVER_ERROR);
                std::remove(temp_filename.c_str());
                return false;
            }
            if (!::read_audio_data(temp_filename, *pcmf32_ptr, *pcmf32s_ptr, req_params.diarize)) {
                SERVER_DEBUG("Failed to read WAV file after conversion");
                set_error_response(res, "Failed to read WAV file after conversion", ErrorType::SERVER_ERROR);
                std::remove(temp_filename.c_str());
                return false;
            }
            std::remove(temp_filename.c_str());
        } else {
            if (!::read_audio_data(audio_file.content, *pcmf32_ptr, *pcmf32s_ptr, req_params.diarize)) {
                SERVER_DEBUG("Failed to read audio data");
                set_error_response(res, "Failed to read audio data", ErrorType::SERVER_ERROR);
                return false;
            }
        }
    } else {
        set_error_response(res, "JSON audio input not supported. Use multipart/form-data.", ErrorType::INVALID_REQUEST);
        SERVER_DEBUG("parse_request_data: JSON audio input not supported. Status: " << res.status);
        return false;
    }
    SERVER_DEBUG("parse_request_data: END SUCCESS");
    return true;
}

bool enqueue_task(std::atomic<int> &task_id_counter, std::shared_ptr<std::vector<float>> pcmf32_ptr, std::shared_ptr<std::vector<std::vector<float>>> pcmf32s_ptr, const std::string& original_filename, const whisper_params& params, std::shared_ptr<std::atomic<bool>> connection_alive, std::shared_ptr<std::atomic<bool>> cancel_flag, const server_params& sparams, std::shared_future<json>& result_fut) {
    if (!task_queue_slots || !task_queue_slots->try_acquire()) {
        SERVER_DEBUG("Task queue full (semaphore). Current slots: 0, Max size: " << sparams.max_task_queue);
        return false;
    }
    g_active_tasks.fetch_add(1, std::memory_order_relaxed);
    TranscriptionTask new_task;
    new_task.id = task_id_counter++;
    new_task.audio_data_pcmf32_ptr = pcmf32_ptr;
    new_task.audio_data_pcmf32s_ptr = pcmf32s_ptr;
    new_task.original_filename = original_filename;
    new_task.params = params;
    new_task.connection_alive = connection_alive;
    new_task.cancel_flag = cancel_flag;
    {
        std::lock_guard<std::mutex> lock(task_queue_mutex);
        result_fut = new_task.result_promise.get_future().share();
        task_queue.push(std::move(new_task));
        SERVER_DEBUG("Task enqueued (semaphore). Task ID: " << (task_queue.empty() ? -1 : task_queue.back().id) << ". Queue size: " << task_queue.size());
    }
    task_queue_cv.notify_one();
    return true;
}

void prepare_and_send_response(Response &res, const Request &req, std::shared_future<json> fut, std::shared_ptr<std::atomic<bool>> connection_alive, std::shared_ptr<std::atomic<bool>> cancel_flag, int task_id, const server_params& sparams) {
    SERVER_DEBUG("Waiting for worker to process task id: " << task_id);
    try {
        using namespace std::chrono_literals;
        std::future_status status = fut.wait_for(std::chrono::seconds(sparams.inference_timeout_sec));
        if (req.is_connection_closed()) {
            SERVER_DEBUG("Request cancelled (client disconnected before future get) for task id: " << task_id);
            if(cancel_flag) *cancel_flag = true;
            if(connection_alive) *connection_alive = false;
            set_error_response(res, "Client disconnected", ErrorType::INVALID_REQUEST);
            res.status = 499;
            res.set_header("Connection", "close");
            return;
        }
        if (status == std::future_status::ready) {
            json result_data = fut.get();
            SERVER_DEBUG("Worker finished processing task id: " << task_id);
            if (req.is_connection_closed() || (cancel_flag && *cancel_flag)) {
                SERVER_DEBUG("Request cancelled (client disconnect after future get) for task id: " << task_id);
                if (result_data.contains("error") && result_data["error"] == "client disconnected") {
                    set_error_response(res, "Client disconnected", ErrorType::INVALID_REQUEST);
                    res.status = 499;
                } else {
                    set_error_response(res, "Client disconnected during processing", ErrorType::SERVER_ERROR);
                    res.status = 499;
                }
                res.set_header("Connection", "close");
                return;
            }
            if (result_data.contains("error")) {
                std::string error_message = result_data["error"].get<std::string>();
                if (error_message == "client disconnected") {
                    set_error_response(res, error_message, ErrorType::INVALID_REQUEST);
                    res.status = 499;
                } else {
                    set_error_response(res, error_message, ErrorType::SERVER_ERROR);
                }
                SERVER_DEBUG("Error in result for task id: " << task_id << ": " << error_message);
            } else {
                if (result_data.contains("srt")) {
                    SERVER_DEBUG("SRT response content for task id " << task_id << ": " << result_data["srt"]);
                    res.set_content(result_data["srt"], "application/x-subrip");
                } else if (result_data.contains("vtt")) {
                    SERVER_DEBUG("VTT response content for task id " << task_id << ": " << result_data["vtt"]);
                    res.set_content(result_data["vtt"], "text/vtt");
                } else {
                    SERVER_DEBUG("JSON response content for task id " << task_id << ": " << result_data.dump(-1, ' ', false, json::error_handler_t::replace));
                    res.set_content(result_data.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
                }
                res.status = 200;
                SERVER_DEBUG("Response prepared for task id: " << task_id);
            }
        } else if (status == std::future_status::timeout) {
            SERVER_DEBUG("Timeout waiting for worker to process task id: " << task_id);
            if(cancel_flag) *cancel_flag = true;
            set_error_response(res, "Timeout waiting for transcription result.", ErrorType::SERVER_ERROR);
        } else {
            SERVER_DEBUG("Future was deferred for task id: " << task_id);
            if(cancel_flag) *cancel_flag = true;
            set_error_response(res, "Internal server error: task processing deferred.", ErrorType::SERVER_ERROR);
        }
    } catch (const std::future_error& e) {
        SERVER_DEBUG("Future error while waiting for transcription result for task " << task_id << ": " << e.what());
        set_error_response(res, "Internal server error processing the request (future error).", ErrorType::SERVER_ERROR);
    } catch (const std::exception& e) {
        SERVER_DEBUG("Exception while waiting for transcription result for task " << task_id << ": " << e.what());
        set_error_response(res, std::string("Internal server error processing the request: ") + e.what(), ErrorType::SERVER_ERROR);
    }
    res.set_header("Connection", "close");
    SERVER_DEBUG("Finalizing response for task id: " << task_id << " with status " << res.status);
}

void signal_handler_main(int signum) {
    if (shutdown_requested.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    SERVER_DEBUG("Interrupt signal (" << signum << ") received. Initiating shutdown.");
    if (g_svr_ptr) {
        g_svr_ptr->stop();
    }
    task_queue_cv.notify_all();
}

int main(int argc, char ** argv) {
    whisper_params params;
    server_params sparams;

    if (whisper_params_parse(argc, argv, params, sparams) == false) {
        whisper_print_usage(argc, argv, params, sparams);
        return 1;
    }

    g_debug_mode = params.debug_mode;

    // Initialize the task_queue_slots semaphore with the configured max_task_queue size
    task_queue_slots = std::make_unique<CountingSemaphore>(sparams.max_task_queue > 0 ? sparams.max_task_queue : 100);
    SERVER_DEBUG("Task queue semaphore initialized with max slots: " << sparams.max_task_queue);

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params, sparams);
        exit(0);
    }

    if (params.diarize && params.tinydiarize) {
        fprintf(stderr, "error: cannot use both --diarize and --tinydiarize\n");
        whisper_print_usage(argc, argv, params, sparams);
        exit(0);
    }

    if (sparams.ffmpeg_converter) {
        check_ffmpeg_availability();
        // Ensure temp_upload_dir exists and is writable
        struct stat st = {0};
        if (stat(sparams.temp_upload_dir.c_str(), &st) == -1) {
#if defined(_WIN32) || defined(_WIN64)
            if (_mkdir(sparams.temp_upload_dir.c_str()) == -1) {
#else
            if (mkdir(sparams.temp_upload_dir.c_str(), 0700) == -1) {
#endif
                fprintf(stderr, "error: could not create temp upload dir: %s\n", sparams.temp_upload_dir.c_str());
                exit(1);
            }
        }
#if defined(_WIN32) || defined(_WIN64)
        if (_access(sparams.temp_upload_dir.c_str(), W_OK) == -1) {
#else
        if (access(sparams.temp_upload_dir.c_str(), W_OK) == -1) {
#endif
            fprintf(stderr, "error: temp upload dir not writable: %s\n", sparams.temp_upload_dir.c_str());
            exit(1);
        }
    }
    // whisper init: create context pool
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    set_dtw_preset(params.dtw, cparams); // Use the helper function
    // Create context pool
    int num_workers = sparams.num_workers > 0 ? sparams.num_workers : 1;
    for (int i = 0; i < num_workers; ++i) {
        ContextPtr ctx{ whisper_init_from_file_with_params(params.model.c_str(), cparams), whisper_free };
        if (!ctx) {
            fprintf(stderr, "error: failed to initialize whisper context for worker %d\n", i);
            // unique_ptr will handle freeing any contexts already in context_pool when it goes out of scope or is cleared.
            context_pool.clear(); // Ensure any partially created contexts are freed.
            return 3; // Or handle error appropriately
        }
        whisper_ctx_init_openvino_encoder(ctx.get(), nullptr, params.openvino_encode_device.c_str(), nullptr);
        context_pool.push_back(std::move(ctx));
    }
    whisper_params default_params = params;
    // Start N worker threads
    for (int i = 0; i < num_workers; ++i) {
        // Ensure context_pool[i] is valid before starting a thread with it
        if (i < context_pool.size() && context_pool[i]) { 
            worker_threads.emplace_back(std::thread{process_transcription_tasks, context_pool[i].get()});
        }
    }

    Server svr;
    g_svr_ptr = &svr;
    std::signal(SIGINT, signal_handler_main);
    std::signal(SIGTERM, signal_handler_main);

    std::string const default_content = R"(
    <html>
    <head>
        <title>Whisper.cpp Server</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width">
        <style>
        body {
            font-family: sans-serif;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        label {
            margin-bottom: 0.5rem;
        }
        input, select {
            margin-bottom: 1rem;
        }
        button {
            margin-top: 1rem;
        }
        </style>
    </head>
    <body>
        <h1>Whisper.cpp Server</h1>

        <h2>/inference</h2>
        <pre>
    curl 127.0.0.1:)" + std::to_string(sparams.port) + R"(/inference \
    -H "Content-Type: multipart/form-data" \
    -F file="@&lt;file-path&gt;" \
    -F temperature="0.0" \
    -F temperature_inc="0.2" \
    -F response_format="json"
        </pre>

        <h2>/load</h2>
        <pre>
    curl 127.0.0.1:)" + std::to_string(sparams.port) + R"(/load \
    -H "Content-Type: multipart/form-data" \
    -F model="&lt;path-to-model-file&gt;"
        </pre>

        <div>
            <h2>Try it out</h2>
            <form action="/inference" method="POST" enctype="multipart/form-data">
                <label for="file">Choose an audio file:</label>
                <input type="file" id="file" name="file" accept="audio/*" required><br>

                <label for="temperature">Temperature:</label>
                <input type="number" id="temperature" name="temperature" value="0.0" step="0.01" placeholder="e.g., 0.0"><br>

                <label for="response_format">Response Format:</label>
                <select id="response_format" name="response_format">
                    <option value="verbose_json">Verbose JSON</option>
                    <option value="json">JSON</option>
                    <option value="text">Text</option>
                    <option value="srt">SRT</option>
                    <option value="vtt">VTT</option>
                </select><br>

                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    )";

    svr.Get(sparams.request_path + "/", [&default_content](const Request &req, Response &res){
        HttpRequestCounter guard(active_http_requests);
        res.set_content(default_content, "text/html");
        return false;
    });

    svr.Options(sparams.request_path + sparams.inference_path, [&](const Request &req, Response &res){
        HttpRequestCounter guard(active_http_requests);
        // No-op for OPTIONS
    });

    // Pass default_params and sparams by capture to the lambda, then to the handler function.
    // Pass next_task_id (as task_id_counter) by reference to the handler.
    svr.Post(sparams.request_path + sparams.inference_path, [&](const Request &req, Response &res){
        HttpRequestCounter guard(active_http_requests);
        handle_inference_request(req, res, default_params, sparams, next_task_id);
    });

    svr.Post(sparams.request_path + "/load", [&](const Request &req, Response &res){
        HttpRequestCounter guard(active_http_requests);
        if (!req.has_file("model"))
        {
            set_error_response(res, "no 'model' field in the request", ErrorType::INVALID_REQUEST);
            return;
        }
        std::string model_path = req.get_file_value("model").content;
        if (!is_file_exist(model_path.c_str()))
        {
            set_error_response(res, "model file not found: " + model_path, ErrorType::NOT_FOUND);
            return;
        }

        fprintf(stderr, "Loading new model: %s\n", model_path.c_str());

        {
            std::lock_guard<std::mutex> lock(contextManagementMutex); // Protect worker and context changes

            // 1. Signal existing workers to shut down
            shutdown_requested.store(true, std::memory_order_release);
            task_queue_cv.notify_all();

            // 2. Join all worker threads
            fprintf(stderr, "Stopping worker threads...\n");
            // ThreadGuard destructors will join threads. Explicitly clear to trigger destructors.
            for (std::thread &worker : worker_threads) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            worker_threads.clear();
            fprintf(stderr, "Worker threads stopped.\n");

            // 3. Clear the old context_pool
            fprintf(stderr, "Freeing old contexts...\n");
            // ContextPtr destructors will free contexts.
            context_pool.clear(); 
            fprintf(stderr, "Old contexts freed.\n");

            // Update the global model path if params is shared or make a copy
            params.model = model_path; // Update model path in the global/default params struct

            // 4. Initialize new contexts and populate context_pool
            struct whisper_context_params cparams = whisper_context_default_params();
            cparams.use_gpu    = params.use_gpu; // Use potentially updated params
            cparams.flash_attn = params.flash_attn;
            set_dtw_preset(params.dtw, cparams);

            bool all_contexts_loaded = true;
            for (int i = 0; i < num_workers; ++i) {
                ContextPtr new_ctx{ whisper_init_from_file_with_params(params.model.c_str(), cparams), whisper_free };
                if (!new_ctx) {
                    fprintf(stderr, "error: failed to initialize whisper context for worker %d with new model %s\n", i, params.model.c_str());
                    all_contexts_loaded = false;
                    break; 
                }
                whisper_ctx_init_openvino_encoder(new_ctx.get(), nullptr, params.openvino_encode_device.c_str(), nullptr);
                context_pool.push_back(std::move(new_ctx));
            }

            if (!all_contexts_loaded) {
                context_pool.clear(); // Free any partially loaded contexts
                set_error_response(res, "Failed to initialize all whisper contexts with the new model.", ErrorType::SERVER_ERROR);
                shutdown_requested.store(false, std::memory_order_release); 
                return;
            }
            
            fprintf(stderr, "%d new contexts initialized with model %s.\n", (int)context_pool.size(), params.model.c_str());

            // 5. Start new worker threads
            shutdown_requested.store(false, std::memory_order_release); 
            for (int i = 0; i < num_workers; ++i) {
                if (i < context_pool.size() && context_pool[i]) { 
                    worker_threads.emplace_back(std::thread{process_transcription_tasks, context_pool[i].get()});
                }
            }
            fprintf(stderr, "%d new worker threads started.\n", (int)worker_threads.size());
        }

        // Correct success response for /load
        json success_payload = {
            {"message", "Model reloaded successfully: " + model_path}
        };
        res.set_content(success_payload.dump(), "application/json");
        res.status = 200;
    });

    svr.Get(sparams.request_path + "/health", [&](const Request &req, Response &res){
        HttpRequestCounter guard(active_http_requests);
        json health_json = {
            {"status", "ok"},
            {"model_loaded", !context_pool.empty() && context_pool[0] && context_pool[0].get() != nullptr},
            {"model_path", context_pool.empty() ? "" : params.model},
            {"queue_size", (int)task_queue.size()},
            {"num_workers", (int)worker_threads.size()},
            {"shutdown_requested", shutdown_requested.load()},
        };
        res.set_content(health_json.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        res.status = 200;
    });

    svr.set_exception_handler([](const Request &req, Response &res, std::exception_ptr ep) {
        HttpRequestCounter guard(active_http_requests);
        const char fmt[] = "500 Internal Server Error\n%s";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception &e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) {
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/plain");
        res.status = 500;
    });

    svr.set_error_handler([](const Request &req, Response &res) {
        HttpRequestCounter guard(active_http_requests);
        if (res.status == 400) {
            res.set_content("Invalid request", "text/plain");
        } else if (res.status != 500) {
            res.set_content("File Not Found (" + req.path + ")", "text/plain");
            res.status = 404;
        }
    });

    // set timeouts and change hostname and port
    svr.set_read_timeout(sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);
    svr.set_keep_alive_max_count(sparams.keep_alive_max_count);
    svr.set_keep_alive_timeout(sparams.keep_alive_timeout);

    // Set the base directory for serving static files
    svr.set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    printf("\nwhisper server listening at http://%s:%d\n\n", sparams.hostname.c_str(), sparams.port);

    // Use listen() with backlog parameter
    if (!svr.listen(sparams.hostname.c_str(), sparams.port, sparams.listen_backlog)) {
        SERVER_DEBUG("Backlog may be too small: expected " << sparams.listen_backlog);
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d backlog=%d\n\n",
                sparams.hostname.c_str(), sparams.port, sparams.listen_backlog);
        return 1;
    }

    bool server_ran_ok = svr.listen_after_bind();

    // ---- Graceful Shutdown Sequence ----
    const auto graceful_shutdown_wait_seconds = std::chrono::seconds(10);
    const auto shutdown_initiated_time = std::chrono::steady_clock::now(); // Base deadline on when listen_after_bind returned or signal received.
    const auto graceful_shutdown_deadline = shutdown_initiated_time + graceful_shutdown_wait_seconds;

    if (shutdown_requested.load(std::memory_order_acquire)) {
        SERVER_DEBUG("Shutdown initiated by signal. Allowing up to " << graceful_shutdown_wait_seconds.count() << "s for active tasks to complete.");
    } else if (!server_ran_ok) {
        SERVER_DEBUG("Server listen_after_bind() returned false. Initiating shutdown procedure.");
        shutdown_requested.store(true, std::memory_order_release); // Ensure flag is set
    } else { // server_ran_ok is true, meaning listen_after_bind returned normally without a prior signal.
        SERVER_DEBUG("Server listen_after_bind() completed (returned true). Initiating shutdown procedure.");
        shutdown_requested.store(true, std::memory_order_release); // Ensure flag is set
    }
    
    // Notify workers that shutdown has begun, in case they are waiting on the CV
    task_queue_cv.notify_all(); 
    SERVER_DEBUG("Waiting for worker threads to process remaining queue and finish active tasks...");

    // --- NEW DYNAMIC DRAIN LOGIC ---
    bool all_tasks_cleared_gracefully = false;
    SERVER_DEBUG("Graceful drain loop started. Waiting for up to " << graceful_shutdown_wait_seconds.count() << "s for tasks to clear.");

    // Loop until deadline or all tasks are done
    while (std::chrono::steady_clock::now() < graceful_shutdown_deadline) {
        bool all_clear_now = false;
        {
            std::lock_guard<std::mutex> lock(task_queue_mutex); // Lock acquired here
            // All checks are now performed under this single lock
            if (task_queue.empty() &&
                g_active_tasks.load(std::memory_order_relaxed) == 0 &&
                active_http_requests.load(std::memory_order_relaxed) == 0) {
                SERVER_DEBUG("All tasks, queue, and HTTP requests cleared (under lock) before graceful shutdown deadline.");
                all_tasks_cleared_gracefully = true;
                all_clear_now = true; // Signal to break loop
            }
        } // Lock released here

        if (all_clear_now) {
            break; // Exit loop, all work done
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Poll every 100ms
    }

    if (all_tasks_cleared_gracefully) {
        SERVER_DEBUG("Graceful drain successful: All tasks and queue cleared before deadline.");
    } else { // Deadline reached or loop exited for other reasons (though break is only on success)
        SERVER_DEBUG("Graceful shutdown deadline (" << graceful_shutdown_wait_seconds.count() << "s) reached or not all tasks cleared. Forcing abort if any remain.");
    }

    // Ensure shutdown_requested is true (it should be, but this is a safeguard)
    // and notify workers again to ensure they pick up the abort signal if they missed the first one
    // or if they were in the middle of a long task.
    SERVER_DEBUG("Issuing final shutdown signal to ensure all workers stop.");
    shutdown_requested.store(true, std::memory_order_release);
    task_queue_cv.notify_all(); 
    // --- END OF NEW DYNAMIC DRAIN LOGIC ---

    bool any_threads_still_active = false;
    for (const auto& worker : worker_threads) {
        if (worker.joinable()) {
            any_threads_still_active = true;
            break;
        }
    }
    if (any_threads_still_active && std::chrono::steady_clock::now() >= graceful_shutdown_deadline) {
        SERVER_DEBUG("Graceful shutdown period (" << graceful_shutdown_wait_seconds.count() << "s) ended, some workers still active. Setting hard abort flag (shutdown_requested).");
        shutdown_requested.store(true, std::memory_order_release);
        task_queue_cv.notify_all();
    } else if (!any_threads_still_active) {
        SERVER_DEBUG("All worker threads appear to have completed their tasks gracefully or were already finished.");
    } else {
         SERVER_DEBUG("Graceful shutdown period not fully elapsed or all threads finished. Proceeding to final join.");
    }
    SERVER_DEBUG("Waiting for all worker threads to join...");
    for (size_t i = 0; i < worker_threads.size(); ++i) {
        if (worker_threads[i].joinable()) {
            worker_threads[i].join();
            SERVER_DEBUG("Worker thread " << i << " joined.");
        }
    }
    worker_threads.clear();
    SERVER_DEBUG("All worker threads joined and cleared.");
    SERVER_DEBUG("Freeing Whisper contexts...");
    {
        std::lock_guard<std::mutex> lock(contextManagementMutex);
        if (!context_pool.empty() && context_pool[0] && context_pool[0].get() != nullptr) {
            whisper_print_timings(context_pool[0].get());
        }
        context_pool.clear();
        SERVER_DEBUG("Whisper contexts freed.");
    }
    int exit_code = 0;
    if (shutdown_requested.load(std::memory_order_acquire)) {
        SERVER_DEBUG("Graceful shutdown complete.");
        exit_code = 0;
    } else {
        if (!server_ran_ok) {
            SERVER_DEBUG("Server loop terminated (listen_after_bind returned false) without a prior signal. This indicates an issue or non-signal stop.");
            exit_code = 1;
        } else {
            SERVER_DEBUG("Server loop completed (listen_after_bind returned true) without shutdown request. Unexpected for blocking server.");
            exit_code = 0;
        }
    }
    SERVER_DEBUG("Exiting main with code: " << exit_code);
    return exit_code;
}
