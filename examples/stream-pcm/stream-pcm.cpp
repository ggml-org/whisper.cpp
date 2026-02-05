// Real-time speech recognition from a raw PCM stream (stdin or pipe)
//
// This mirrors the mic-based whisper-stream example but reads PCM audio
// from stdin or a file/pipe to avoid external audio device dependencies.
//
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = true;

    std::string language   = "en";
    std::string model      = "models/ggml-base.en.bin";
    std::string fname_out;

    std::string input      = "-"; // "-" = stdin
    std::string format     = "f32"; // f32 or s16
    int32_t     sample_rate = WHISPER_SAMPLE_RATE;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

enum class pcm_format {
    f32,
    s16,
};

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn") { params.flash_attn    = false; }
        else if (arg == "-i"    || arg == "--input")         { params.input         = argv[++i]; }
        else if (arg == "--format")                           { params.format        = argv[++i]; }
        else if (arg == "--sample-rate")                      { params.sample_rate   = std::stoi(argv[++i]); }

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
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] enable flash attention during inference\n",        params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,     --no-flash-attn [%-7s] disable flash attention during inference\n",       params.flash_attn ? "false" : "true");
    fprintf(stderr, "  -i PATH,  --input PATH    [%-7s] input path ('-' for stdin)\n",                     params.input.c_str());
    fprintf(stderr, "            --format FMT    [%-7s] input format: f32 or s16 (little-endian)\n",      params.format.c_str());
    fprintf(stderr, "            --sample-rate N [%-7d] input sample rate (must be 16000)\n",             params.sample_rate);
    fprintf(stderr, "\n");
}

class pcm_async {
public:
    pcm_async(int len_ms, int sample_rate, pcm_format format)
        : m_len_ms(len_ms), m_sample_rate(sample_rate), m_format(format) {
        m_running = false;
        m_stop = false;
        m_eof = false;
    }

    ~pcm_async() {
        pause();
    }

    bool init(const std::string & input_path) {
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
#endif

        if (input_path == "-") {
            m_in = stdin;
            m_owns_input = false;
        } else {
            m_in = fopen(input_path.c_str(), "rb");
            m_owns_input = true;
        }

        if (!m_in) {
            fprintf(stderr, "%s: failed to open input '%s'\n", __func__, input_path.c_str());
            return false;
        }

        m_audio.resize((m_sample_rate*m_len_ms)/1000);
        m_audio_pos = 0;
        m_audio_len = 0;

        return true;
    }

    bool resume() {
        if (!m_in) {
            fprintf(stderr, "%s: no input to resume!\n", __func__);
            return false;
        }

        if (m_running) {
            fprintf(stderr, "%s: already running!\n", __func__);
            return false;
        }

        m_stop = false;
        m_thread = std::thread(&pcm_async::reader_loop, this);
        m_running = true;

        return true;
    }

    bool pause() {
        if (!m_running) {
            return true;
        }

        m_stop = true;

        if (m_owns_input && m_in) {
            fclose(m_in);
            m_in = nullptr;
        }

        if (m_thread.joinable()) {
            if (m_owns_input) {
                m_thread.join();
            } else {
                m_thread.detach();
            }
        }

        m_running = false;

        return true;
    }

    bool clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_audio_pos = 0;
        m_audio_len = 0;
        return true;
    }

    void get(int ms, std::vector<float> & result) {
        result.clear();

        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_audio_len) {
            n_samples = m_audio_len;
        }

        result.resize(n_samples);

        int s0 = (int) m_audio_pos - (int) n_samples;
        if (s0 < 0) {
            s0 += (int) m_audio.size();
        }

        if (s0 + (int) n_samples > (int) m_audio.size()) {
            const size_t n0 = m_audio.size() - (size_t) s0;
            memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
            memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
        } else {
            memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
        }
    }

    size_t available_samples() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_audio_len;
    }

    bool is_eof() const {
        return m_eof.load();
    }

private:
    void reader_loop() {
        const size_t bytes_per_sample = (m_format == pcm_format::f32) ? 4 : 2;
        std::vector<uint8_t> buffer(4096);
        std::vector<uint8_t> carry;

        while (!m_stop) {
            size_t n_read = fread(buffer.data(), 1, buffer.size(), m_in);

            if (n_read == 0) {
                if (feof(m_in)) {
                    m_eof = true;
                }
                break;
            }

            std::vector<uint8_t> data;
            data.reserve(carry.size() + n_read);
            if (!carry.empty()) {
                data.insert(data.end(), carry.begin(), carry.end());
                carry.clear();
            }
            data.insert(data.end(), buffer.begin(), buffer.begin() + n_read);

            const size_t total_bytes = data.size();
            const size_t n_samples = total_bytes / bytes_per_sample;
            const size_t rem = total_bytes % bytes_per_sample;

            if (rem > 0) {
                carry.insert(carry.end(), data.end() - rem, data.end());
            }

            if (n_samples == 0) {
                continue;
            }

            std::vector<float> samples(n_samples);

            if (m_format == pcm_format::f32) {
                for (size_t i = 0; i < n_samples; ++i) {
                    float v = 0.0f;
                    memcpy(&v, &data[i * 4], sizeof(float));
                    samples[i] = v;
                }
            } else {
                for (size_t i = 0; i < n_samples; ++i) {
                    int16_t v = 0;
                    memcpy(&v, &data[i * 2], sizeof(int16_t));
                    samples[i] = v / 32768.0f;
                }
            }

            push_samples(samples.data(), samples.size());
        }
    }

    void push_samples(const float * data, size_t n_samples) {
        if (n_samples == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);

        if (n_samples > m_audio.size()) {
            data += (n_samples - m_audio.size());
            n_samples = m_audio.size();
        }

        if (m_audio_pos + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - m_audio_pos;
            memcpy(&m_audio[m_audio_pos], data, n0 * sizeof(float));
            memcpy(&m_audio[0], data + n0, (n_samples - n0) * sizeof(float));
        } else {
            memcpy(&m_audio[m_audio_pos], data, n_samples * sizeof(float));
        }

        m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
        m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
    }

    FILE * m_in = nullptr;
    bool m_owns_input = false;

    int m_len_ms = 0;
    int m_sample_rate = 0;
    pcm_format m_format = pcm_format::f32;

    std::atomic_bool m_running;
    std::atomic_bool m_stop;
    std::atomic_bool m_eof;

    mutable std::mutex m_mutex;

    std::vector<float> m_audio;
    size_t             m_audio_pos = 0;
    size_t             m_audio_len = 0;

    std::thread m_thread;
};

static std::atomic_bool g_running(true);

static void signal_handler(int) {
    g_running = false;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.sample_rate != WHISPER_SAMPLE_RATE) {
        fprintf(stderr, "error: only --sample-rate %d is supported (got %d). resample before streaming.\n",
                WHISPER_SAMPLE_RATE, params.sample_rate);
        return 1;
    }

    pcm_format input_format = pcm_format::f32;
    if (params.format == "f32") {
        input_format = pcm_format::f32;
    } else if (params.format == "s16") {
        input_format = pcm_format::s16;
    } else {
        fprintf(stderr, "error: unknown --format '%s' (expected f32 or s16)\n", params.format.c_str());
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    std::signal(SIGINT, signal_handler);

    // init audio
    pcm_async audio(params.length_ms, WHISPER_SAMPLE_RATE, input_format);
    if (!audio.init(params.input)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

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
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    wav_writer wavWriter;
    // save wav file
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    printf("[Start streaming]\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    // main audio loop
    while (is_running && g_running) {
        if (audio.is_eof() && audio.available_samples() == 0) {
            break;
        }

        if (params.save_audio) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }

        // process new audio

        if (!use_vad) {
            bool eof_after_this = false;
            while (true) {
                if (!g_running) {
                    is_running = false;
                    break;
                }

                audio.get(params.step_ms, pcmf32_new);

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                if (audio.is_eof() && audio.available_samples() > 0) {
                    // flush remaining samples on EOF
                    audio.clear();
                    eof_after_this = true;
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (!is_running) {
                break;
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;

            if (eof_after_this && pcmf32.empty()) {
                break;
            }
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            t_last = t_now;
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.audio_ctx        = params.audio_ctx;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            // print result;
            {
                if (!use_vad) {
                    printf("\33[2K\r");

                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());

                    printf("\33[2K\r");
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    printf("\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        printf("%s", text);
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        printf("%s", output.c_str());
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);
        }
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
