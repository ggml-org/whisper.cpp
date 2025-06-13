#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#include "httplib.h"
#include "json.hpp"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <csignal>
#include <atomic>
#include <functional>
#include <cstdlib>
#if defined (_WIN32)
#include <windows.h>
#endif

using namespace httplib;
using json = nlohmann::ordered_json;

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

namespace {

// output formats
const std::string json_format   = "json";
const std::string text_format   = "text";
const std::string srt_format    = "srt";
const std::string vjson_format  = "verbose_json";
const std::string vtt_format    = "vtt";

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::string public_path = "examples/server/public";
    std::string request_path = "";
    std::string inference_path = "/inference";

    int32_t port          = 8080;
    int32_t read_timeout  = 600;
    int32_t write_timeout = 600;

    bool ffmpeg_converter = false;
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

    // Voice Activity Detection (VAD) parameters
    bool        vad           = false;
    std::string vad_model     = "";
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;
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
    // Voice Activity Detection (VAD) parameters
    fprintf(stderr, "\nVoice Activity Detection (VAD) options:\n");
    fprintf(stderr, "             --vad                           [%-7s] enable Voice Activity Detection (VAD)\n",            params.vad ? "true" : "false");
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME               [%-7s] VAD model path\n",                                   params.vad_model.c_str());
    fprintf(stderr, "  -vt N,     --vad-threshold N               [%-7.2f] VAD threshold for speech recognition\n",           params.vad_threshold);
    fprintf(stderr, "  -vspd N,   --vad-min-speech-duration-ms  N [%-7d] VAD min speech duration (0.0-1.0)\n",                params.vad_min_speech_duration_ms);
    fprintf(stderr, "  -vsd N,    --vad-min-silence-duration-ms N [%-7d] VAD min silence duration (to split segments)\n",      params.vad_min_silence_duration_ms);
    fprintf(stderr, "  -vmsd N,   --vad-max-speech-duration-s   N [%-7s] VAD max speech duration (auto-split longer)\n",      params.vad_max_speech_duration_s == FLT_MAX ?
                                                                                                                                  std::string("FLT_MAX").c_str() :
                                                                                                                                  std::to_string(params.vad_max_speech_duration_s).c_str());
    fprintf(stderr, "  -vp N,     --vad-speech-pad-ms           N [%-7d] VAD speech padding (extend segments)\n",             params.vad_speech_pad_ms);
    fprintf(stderr, "  -vo N,     --vad-samples-overlap         N [%-7.2f] VAD samples overlap (seconds between segments)\n", params.vad_samples_overlap);
    fprintf(stderr, "\n");
}

bool whisper_params_parse(int argc, char ** argv, whisper_params & params, server_params & sparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params, sparams);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(argv[++i]); }
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

        // server params
        else if (                  arg == "--port")            { sparams.port        = std::stoi(argv[++i]); }
        else if (                  arg == "--host")            { sparams.hostname    = argv[++i]; }
        else if (                  arg == "--public")          { sparams.public_path = argv[++i]; }
        else if (                  arg == "--request-path")    { sparams.request_path = argv[++i]; }
        else if (                  arg == "--inference-path")  { sparams.inference_path = argv[++i]; }
        else if (                  arg == "--convert")         { sparams.ffmpeg_converter     = true; }

        // Voice Activity Detection (VAD)
        else if (                  arg == "--vad")                         { params.vad                         = true; }
        else if (arg == "-vm"   || arg == "--vad-model")                   { params.vad_model                   = argv[++i]; }
        else if (arg == "-vt"   || arg == "--vad-threshold")               { params.vad_threshold               = std::stof(argv[++i]); }
        else if (arg == "-vspd" || arg == "--vad-min-speech-duration-ms")  { params.vad_min_speech_duration_ms  = std::stoi(argv[++i]); }
        else if (arg == "-vsd"  || arg == "--vad-min-silence-duration-ms") { params.vad_min_speech_duration_ms  = std::stoi(argv[++i]); }
        else if (arg == "-vmsd" || arg == "--vad-max-speech-duration-s")   { params.vad_max_speech_duration_s   = std::stof(argv[++i]); }
        else if (arg == "-vp"   || arg == "--vad-speech-pad-ms")           { params.vad_speech_pad_ms           = std::stoi(argv[++i]); }
        else if (arg == "-vo"   || arg == "--vad-samples-overlap")         { params.vad_samples_overlap         = std::stof(argv[++i]); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params, sparams);
            exit(0);
        }
    }

    return true;
}

struct whisper_print_user_data {
    const whisper_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

void check_ffmpeg_availibility() {
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
    ss << prefix
       << "-"
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d-%H%M%S")
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

void get_req_parameters(const Request & req, whisper_params & params)
{
    if (req.has_file("offset_t"))
    {
        params.offset_t_ms = std::stoi(req.get_file_value("offset_t").content);
    }
    if (req.has_file("offset_n"))
    {
        params.offset_n = std::stoi(req.get_file_value("offset_n").content);
    }
    if (req.has_file("duration"))
    {
        params.duration_ms = std::stoi(req.get_file_value("duration").content);
    }
    if (req.has_file("max_context"))
    {
        params.max_context = std::stoi(req.get_file_value("max_context").content);
    }
    if (req.has_file("max_len"))
    {
        params.max_len = std::stoi(req.get_file_value("max_len").content);
    }
    if (req.has_file("best_of"))
    {
        params.best_of = std::stoi(req.get_file_value("best_of").content);
    }
    if (req.has_file("beam_size"))
    {
        params.beam_size = std::stoi(req.get_file_value("beam_size").content);
    }
    if (req.has_file("audio_ctx"))
    {
        params.audio_ctx = std::stof(req.get_file_value("audio_ctx").content);
    }
    if (req.has_file("word_thold"))
    {
        params.word_thold = std::stof(req.get_file_value("word_thold").content);
    }
    if (req.has_file("entropy_thold"))
    {
        params.entropy_thold = std::stof(req.get_file_value("entropy_thold").content);
    }
    if (req.has_file("logprob_thold"))
    {
        params.logprob_thold = std::stof(req.get_file_value("logprob_thold").content);
    }
    if (req.has_file("debug_mode"))
    {
        params.debug_mode = parse_str_to_bool(req.get_file_value("debug_mode").content);
    }
    if (req.has_file("translate"))
    {
        params.translate = parse_str_to_bool(req.get_file_value("translate").content);
    }
    if (req.has_file("diarize"))
    {
        params.diarize = parse_str_to_bool(req.get_file_value("diarize").content);
    }
    if (req.has_file("tinydiarize"))
    {
        params.tinydiarize = parse_str_to_bool(req.get_file_value("tinydiarize").content);
    }
    if (req.has_file("split_on_word"))
    {
        params.split_on_word = parse_str_to_bool(req.get_file_value("split_on_word").content);
    }
    if (req.has_file("no_timestamps"))
    {
        params.no_timestamps = parse_str_to_bool(req.get_file_value("no_timestamps").content);
    }
    if (req.has_file("language"))
    {
        params.language = req.get_file_value("language").content;
    }
    if (req.has_file("detect_language"))
    {
        params.detect_language = parse_str_to_bool(req.get_file_value("detect_language").content);
    }
    if (req.has_file("prompt"))
    {
        params.prompt = req.get_file_value("prompt").content;
    }
    if (req.has_file("response_format"))
    {
        params.response_format = req.get_file_value("response_format").content;
    }
    if (req.has_file("temperature"))
    {
        params.temperature = std::stof(req.get_file_value("temperature").content);
    }
    if (req.has_file("temperature_inc"))
    {
        params.temperature_inc = std::stof(req.get_file_value("temperature_inc").content);
    }
    if (req.has_file("suppress_non_speech"))
    {
        params.suppress_nst = parse_str_to_bool(req.get_file_value("suppress_non_speech").content);
    }
    if (req.has_file("suppress_nst"))
    {
        params.suppress_nst = parse_str_to_bool(req.get_file_value("suppress_nst").content);
    }
    if (req.has_file("no_context"))
    {
        params.no_context = parse_str_to_bool(req.get_file_value("no_context").content);
    }
    if (req.has_file("vad"))
    {
        params.vad = parse_str_to_bool(req.get_file_value("vad").content);
    }
    if (req.has_file("vad_threshold"))
    {
        params.vad_threshold = std::stof(req.get_file_value("vad_threshold").content);
    }
    if (req.has_file("vad_min_speech_duration_ms"))
    {
        params.vad_min_speech_duration_ms = std::stof(req.get_file_value("vad_min_speech_duration_ms").content);
    }
    if (req.has_file("vad_min_silence_duration_ms"))
    {
        params.vad_min_silence_duration_ms = std::stof(req.get_file_value("vad_min_silence_duration_ms").content);
    }
    if (req.has_file("vad_max_speech_duration_s"))
    {
        params.vad_max_speech_duration_s = std::stof(req.get_file_value("vad_max_speech_duration_s").content);
    }
    if (req.has_file("vad_speech_pad_ms"))
    {
        params.vad_speech_pad_ms = std::stoi(req.get_file_value("vad_speech_pad_ms").content);
    }
    if (req.has_file("vad_samples_overlap"))
    {
        params.vad_samples_overlap = std::stof(req.get_file_value("vad_samples_overlap").content);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;
    server_params sparams;

    std::mutex whisper_mutex;

    if (whisper_params_parse(argc, argv, params, sparams) == false) {
        whisper_print_usage(argc, argv, params, sparams);
        return 1;
    }

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
        check_ffmpeg_availibility();
    }
    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    if (!params.dtw.empty()) {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

        if (params.dtw == "tiny") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        }
        if (params.dtw == "tiny.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        }
        if (params.dtw == "base") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        }
        if (params.dtw == "base.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        }
        if (params.dtw == "small") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        }
        if (params.dtw == "small.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        }
        if (params.dtw == "medium") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        }
        if (params.dtw == "medium.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        }
        if (params.dtw == "large.v1") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        }
        if (params.dtw == "large.v2") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        }
        if (params.dtw == "large.v3") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        }

        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
            fprintf(stderr, "error: unknown DTW preset '%s'\n", params.dtw.c_str());
            return 3;
        }
    }

    std::unique_ptr<httplib::Server> svr = std::make_unique<httplib::Server>();
    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 3;
    }

    // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);
    state.store(SERVER_STATE_READY);


    svr->set_default_headers({{"Server", "whisper.cpp"},
                             {"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "content-type, authorization"}});

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

    // store default params so we can reset after each inference request
    whisper_params default_params = params;

    // this is only called if no index.html is found in the public --path
    svr->Get(sparams.request_path + "/", [&](const Request &, Response &res){
        res.set_content(default_content, "text/html");
        return false;
    });

    svr->Options(sparams.request_path + sparams.inference_path, [&](const Request &, Response &){
    });

    svr->Post(sparams.request_path + sparams.inference_path, [&](const Request &req, Response &res){
        // acquire whisper model mutex lock
        std::lock_guard<std::mutex> lock(whisper_mutex);

        // first check user requested fields of the request
        if (!req.has_file("file"))
        {
            fprintf(stderr, "error: no 'file' field in the request\n");
            const std::string error_resp = "{\"error\":\"no 'file' field in the request\"}";
            res.set_content(error_resp, "application/json");
            return;
        }
        auto audio_file = req.get_file_value("file");

        // check non-required fields
        get_req_parameters(req, params);

        std::string filename{audio_file.filename};
        printf("Received request: %s\n", filename.c_str());

        // audio arrays
        std::vector<float> pcmf32;               // mono-channel F32 PCM
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        if (sparams.ffmpeg_converter) {
            // if file is not wav, convert to wav
            // write to temporary file
            const std::string temp_filename = generate_temp_filename("whisper-server", ".wav");
            std::ofstream temp_file{temp_filename, std::ios::binary};
            temp_file << audio_file.content;
            temp_file.close();

            std::string error_resp = "{\"error\":\"Failed to execute ffmpeg command.\"}";
            const bool is_converted = convert_to_wav(temp_filename, error_resp);
            if (!is_converted) {
                res.set_content(error_resp, "application/json");
                return;
            }

            // read audio content into pcmf32
            if (!::read_audio_data(temp_filename, pcmf32, pcmf32s, params.diarize))
            {
                fprintf(stderr, "error: failed to read WAV file '%s'\n", temp_filename.c_str());
                const std::string error_resp = "{\"error\":\"failed to read WAV file\"}";
                res.set_content(error_resp, "application/json");
                std::remove(temp_filename.c_str());
                return;
            }
            // remove temp file
            std::remove(temp_filename.c_str());
        } else {
            if (!::read_audio_data(audio_file.content, pcmf32, pcmf32s, params.diarize))
            {
                fprintf(stderr, "error: failed to read audio data\n");
                const std::string error_resp = "{\"error\":\"failed to read audio data\"}";
                res.set_content(error_resp, "application/json");
                return;
            }
        }

        printf("Successfully loaded %s\n", filename.c_str());

        // print system information
        {
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                    params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
        }

        // print some info about the processing
        {
            fprintf(stderr, "\n");
            if (!whisper_is_multilingual(ctx)) {
                if (params.language != "en" || params.translate) {
                    params.language = "en";
                    params.translate = false;
                    fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
                }
            }
            if (params.detect_language) {
                params.language = "auto";
            }
            fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, %stimestamps = %d ...\n",
                    __func__, filename.c_str(), int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                    params.n_threads, params.n_processors,
                    params.language.c_str(),
                    params.translate ? "translate" : "transcribe",
                    params.tinydiarize ? "tdrz = 1, " : "",
                    params.no_timestamps ? 0 : 1);

            fprintf(stderr, "\n");
        }

        // run the inference
        {
            printf("Running whisper.cpp inference on %s\n", filename.c_str());
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

            wparams.print_realtime   = false;
            wparams.print_progress   = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.print_special    = params.print_special;
            wparams.translate        = params.translate;
            wparams.language         = params.language.c_str();
            wparams.detect_language  = params.detect_language;
            wparams.n_threads        = params.n_threads;
            wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms        = params.offset_t_ms;
            wparams.duration_ms      = params.duration_ms;

            wparams.thold_pt         = params.word_thold;
            wparams.max_len          = params.max_len == 0 ? 60 : params.max_len;
            wparams.split_on_word    = params.split_on_word;
            wparams.audio_ctx        = params.audio_ctx;

            wparams.debug_mode       = params.debug_mode;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            wparams.initial_prompt   = params.prompt.c_str();

            wparams.greedy.best_of        = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.temperature      = params.temperature;
            wparams.no_speech_thold = params.no_speech_thold;
            wparams.temperature_inc  = params.temperature_inc;
            wparams.entropy_thold    = params.entropy_thold;
            wparams.logprob_thold    = params.logprob_thold;

            wparams.no_timestamps    = params.no_timestamps;
            wparams.token_timestamps = !params.no_timestamps && params.response_format == vjson_format;
            wparams.no_context       = params.no_context;

            wparams.suppress_nst     = params.suppress_nst;

            wparams.vad              = params.vad;
            wparams.vad_model_path   = params.vad_model.c_str();

            wparams.vad_params.threshold               = params.vad_threshold;
            wparams.vad_params.min_speech_duration_ms  = params.vad_min_speech_duration_ms;
            wparams.vad_params.min_silence_duration_ms = params.vad_min_silence_duration_ms;
            wparams.vad_params.max_speech_duration_s   = params.vad_max_speech_duration_s;
            wparams.vad_params.speech_pad_ms           = params.vad_speech_pad_ms;
            wparams.vad_params.samples_overlap         = params.vad_samples_overlap;

            whisper_print_user_data user_data = { &params, &pcmf32s, 0 };

            // this callback is called on each new segment
            if (params.print_realtime) {
                wparams.new_segment_callback           = whisper_print_segment_callback;
                wparams.new_segment_callback_user_data = &user_data;
            }

            if (wparams.print_progress) {
                wparams.progress_callback           = whisper_print_progress_callback;
                wparams.progress_callback_user_data = &user_data;
            }

            // tell whisper to abort if the HTTP connection closed
            wparams.abort_callback = [](void *user_data) {
                // user_data is a pointer to our Request
                auto req_ptr = static_cast<const httplib::Request*>(user_data);
                return req_ptr->is_connection_closed();
            };
            wparams.abort_callback_user_data = (void*)&req;

            if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
                // handle failure or early abort
                if (req.is_connection_closed()) {
                    // log client disconnect
                    fprintf(stderr, "client disconnected, aborted processing\n");
                    res.status = 499; // Client Closed Request (nginx convention)
                    res.set_content("{\"error\":\"client disconnected\"}", "application/json");
                    return;
                }
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                res.status = 500; // Internal Server Error
                const std::string error_resp = "{\"error\":\"failed to process audio\"}";
                res.set_content(error_resp, "application/json");
                return;
            }
        }

        // return results to user
        if (params.response_format == text_format)
        {
            std::string results = output_str(ctx, params, pcmf32s);
            res.set_content(results.c_str(), "text/html; charset=utf-8");
        }
        else if (params.response_format == srt_format)
        {
            std::stringstream ss;
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                std::string speaker = "";

                if (params.diarize && pcmf32s.size() == 2)
                {
                    speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
                }

                ss << i + 1 + params.offset_n << "\n";
                ss << to_timestamp(t0, true) << " --> " << to_timestamp(t1, true) << "\n";
                ss << speaker << text << "\n\n";
            }
            res.set_content(ss.str(), "application/x-subrip");
        } else if (params.response_format == vtt_format) {
            std::stringstream ss;

            ss << "WEBVTT\n\n";

            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                std::string speaker = "";

                if (params.diarize && pcmf32s.size() == 2)
                {
                    speaker = estimate_diarization_speaker(pcmf32s, t0, t1, true);
                    speaker.insert(0, "<v Speaker");
                    speaker.append(">");
                }

                ss << to_timestamp(t0) << " --> " << to_timestamp(t1) << "\n";
                ss << speaker << text << "\n\n";
            }
            res.set_content(ss.str(), "text/vtt");
        } else if (params.response_format == vjson_format) {
            /* try to match openai/whisper's Python format */
            std::string results = output_str(ctx, params, pcmf32s); 
            // Get language probabilities
            std::vector<float> lang_probs(whisper_lang_max_id() + 1, 0.0f);
            const auto detected_lang_id = whisper_lang_auto_detect(ctx, 0, params.n_threads, lang_probs.data());
            json jres = json{
                {"task", params.translate ? "translate" : "transcribe"},
                {"language", whisper_lang_str_full(whisper_full_lang_id(ctx))},
                {"duration", float(pcmf32.size())/WHISPER_SAMPLE_RATE},
                {"text", results},
                {"segments", json::array()},
                {"detected_language", whisper_lang_str_full(detected_lang_id)},
                {"detected_language_probability", lang_probs[detected_lang_id]},
                {"language_probabilities", json::object()}
            };
            // Add all language probabilities
            for (int i = 0; i <= whisper_lang_max_id(); ++i) {
                if (lang_probs[i] > 0.001f) { // Only include non-negligible probabilities
                    jres["language_probabilities"][whisper_lang_str(i)] = lang_probs[i];
                }
            }
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i)
            {
                json segment = json{
                    {"id", i},
                    {"text", whisper_full_get_segment_text(ctx, i)},
                };

                if (!params.no_timestamps) {
                    segment["start"] = whisper_full_get_segment_t0(ctx, i) * 0.01;
                    segment["end"] = whisper_full_get_segment_t1(ctx, i) * 0.01;
                }

                float total_logprob = 0;
                const int n_tokens = whisper_full_n_tokens(ctx, i);
                for (int j = 0; j < n_tokens; ++j) {
                    whisper_token_data token = whisper_full_get_token_data(ctx, i, j);
                    if (token.id >= whisper_token_eot(ctx)) {
                        continue;
                    }

                    segment["tokens"].push_back(token.id);
                    json word = json{{"word", whisper_full_get_token_text(ctx, i, j)}};
                    if (!params.no_timestamps) {
                        word["start"] = token.t0 * 0.01;
                        word["end"] = token.t1 * 0.01;
                        word["t_dtw"] = token.t_dtw;
                    }
                    word["probability"] = token.p;
                    total_logprob += token.plog;
                    segment["words"].push_back(word);
                }

                segment["temperature"] = params.temperature;
                segment["avg_logprob"] = total_logprob / n_tokens;

                // TODO compression_ratio and no_speech_prob are not implemented yet
                // segment["compression_ratio"] = 0;
                segment["no_speech_prob"] = whisper_full_get_segment_no_speech_prob(ctx, i);

                jres["segments"].push_back(segment);
            }
            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        }
        // TODO add more output formats
        else
        {
            std::string results = output_str(ctx, params, pcmf32s);
            json jres = json{
                {"text", results}
            };
            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        }

        // reset params to their defaults
        params = default_params;
    });
    svr->Post(sparams.request_path + "/load", [&](const Request &req, Response &res){
        std::lock_guard<std::mutex> lock(whisper_mutex);
        state.store(SERVER_STATE_LOADING_MODEL);
        if (!req.has_file("model"))
        {
            fprintf(stderr, "error: no 'model' field in the request\n");
            const std::string error_resp = "{\"error\":\"no 'model' field in the request\"}";
            res.set_content(error_resp, "application/json");
            return;
        }
        std::string model = req.get_file_value("model").content;
        if (!is_file_exist(model.c_str()))
        {
            fprintf(stderr, "error: 'model': %s not found!\n", model.c_str());
            const std::string error_resp = "{\"error\":\"model not found!\"}";
            res.set_content(error_resp, "application/json");
            return;
        }

        // clean up
        whisper_free(ctx);

        // whisper init
        ctx = whisper_init_from_file_with_params(model.c_str(), cparams);

        // TODO perhaps load prior model here instead of exit
        if (ctx == nullptr) {
            fprintf(stderr, "error: model init  failed, no model loaded must exit\n");
            exit(1);
        }

        // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
        whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

        state.store(SERVER_STATE_READY);
        const std::string success = "Load was successful!";
        res.set_content(success, "application/text");

        // check if the model is in the file system
    });

    svr->Get(sparams.request_path + "/health", [&](const Request &, Response &res){
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_READY) {
            const std::string health_response = "{\"status\":\"ok\"}";
            res.set_content(health_response, "application/json");
        } else {
            res.set_content("{\"status\":\"loading model\"}", "application/json");
            res.status = 503;
        }
    });

    svr->set_exception_handler([](const Request &, Response &res, std::exception_ptr ep) {
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

    svr->set_error_handler([](const Request &req, Response &res) {
        if (res.status == 400) {
            res.set_content("Invalid request", "text/plain");
        } else if (res.status != 500) {
            res.set_content("File Not Found (" + req.path + ")", "text/plain");
            res.status = 404;
        }
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout(sparams.read_timeout);
    svr->set_write_timeout(sparams.write_timeout);

    if (!svr->bind_to_port(sparams.hostname, sparams.port))
    {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n",
                sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    // Set the base directory for serving static files
    svr->set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    printf("\nwhisper server listening at http://%s:%d\n\n", sparams.hostname.c_str(), sparams.port);

    shutdown_handler = [&](int signal) {
        printf("\nCaught signal %d, shutting down gracefully...\n", signal);
        if (svr) {
            svr->stop();
        }
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // clean up function, to be called before exit
    auto clean_up = [&]() {
        whisper_print_timings(ctx);
        whisper_free(ctx);
    };

    std::thread t([&] {
        if (!svr->listen_after_bind()) {
            fprintf(stderr, "error: server listen failed\n");
        }
    });

    svr->wait_until_ready();

    t.join();


    clean_up();

    return 0;
}
