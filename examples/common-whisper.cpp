#define _USE_MATH_DEFINES // for M_PI

#include "common-whisper.h"

#include "common.h"

#include "whisper.h"

// third-party utilities
// use your favorite implementations
#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"    /* Enables Vorbis decoding. */

#ifdef _WIN32
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#endif

#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <cstring>
#include <fstream>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>

#ifndef _WIN32
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef WHISPER_FFMPEG
// as implemented in ffmpeg_trancode.cpp only embedded in common lib if whisper built with ffmpeg support
extern bool ffmpeg_decode_audio(const std::string & ifname, std::vector<uint8_t> & wav_data);
#endif

bool read_audio_data(const std::string & fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
    std::vector<uint8_t> audio_data; // used for pipe input from stdin or ffmpeg decoding output

    ma_result result;
    ma_decoder_config decoder_config;
    ma_decoder decoder;

    decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, WHISPER_SAMPLE_RATE);

    if (fname == "-") {
		#ifdef _WIN32
		_setmode(_fileno(stdin), _O_BINARY);
		#endif

		uint8_t buf[1024];
		while (true)
		{
			const size_t n = fread(buf, 1, sizeof(buf), stdin);
			if (n == 0) {
				break;
			}
			audio_data.insert(audio_data.end(), buf, buf + n);
		}

		if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {

			fprintf(stderr, "Error: failed to open audio data from stdin (%s)\n", ma_result_description(result));

			return false;
		}

		fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, audio_data.size());
    }
    else if (((result = ma_decoder_init_file(fname.c_str(), &decoder_config, &decoder)) != MA_SUCCESS)) {
#if defined(WHISPER_FFMPEG)
		if (ffmpeg_decode_audio(fname, audio_data) != 0) {
			fprintf(stderr, "error: failed to ffmpeg decode '%s'\n", fname.c_str());

			return false;
		}

		if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {
			fprintf(stderr, "error: failed to read audio data as wav (%s)\n", ma_result_description(result));

			return false;
		}
#else
		if ((result = ma_decoder_init_memory(fname.c_str(), fname.size(), &decoder_config, &decoder)) != MA_SUCCESS) {
			fprintf(stderr, "error: failed to read audio data as wav (%s)\n", ma_result_description(result));

			return false;
		}
#endif
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;

    if ((result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count)) != MA_SUCCESS) {
		fprintf(stderr, "error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));

		return false;
    }

    pcmf32.resize(stereo ? frame_count*2 : frame_count);

    if ((result = ma_decoder_read_pcm_frames(&decoder, pcmf32.data(), frame_count, &frames_read)) != MA_SUCCESS) {
		fprintf(stderr, "error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));

		return false;
    }

    if (stereo) {
        std::vector<float> stereo_data = pcmf32;
        pcmf32.resize(frame_count);

        for (uint64_t i = 0; i < frame_count; i++) {
            pcmf32[i] = (stereo_data[2*i] + stereo_data[2*i + 1]);
        }

        pcmf32s.resize(2);
        pcmf32s[0].resize(frame_count);
        pcmf32s[1].resize(frame_count);
        for (uint64_t i = 0; i < frame_count; i++) {
            pcmf32s[0][i] = stereo_data[2*i];
            pcmf32s[1][i] = stereo_data[2*i + 1];
        }
    }

    ma_decoder_uninit(&decoder);

    return true;
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*whisper_sample_rate)/100)));
}

bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id) {
    return speak_with_file(command, text, path, voice_id, nullptr);
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id, speak_metrics * metrics) {
    const int64_t t_start = now_ms();

    std::ofstream speak_file(path.c_str());
    if (speak_file.fail()) {
        fprintf(stderr, "%s: failed to open speak_file\n", __func__);
        if (metrics != nullptr) {
            metrics->ok = false;
        }
        return false;
    } else {
        speak_file.write(text.c_str(), text.size());
        speak_file.close();
        if (metrics != nullptr) {
            metrics->startup_ms = now_ms() - t_start;
        }
        int ret = system((command + " " + std::to_string(voice_id) + " " + path).c_str());
        if (ret != 0) {
            fprintf(stderr, "%s: failed to speak\n", __func__);
            if (metrics != nullptr) {
                metrics->ok = false;
                metrics->total_ms = now_ms() - t_start;
            }
            return false;
        }
    }
    if (metrics != nullptr) {
        metrics->total_ms = now_ms() - t_start;
    }
    return true;
}

struct tts_worker::impl {
    explicit impl(tts_worker_params p) : params(std::move(p)) {}

    struct job {
        uint64_t turn_id = 0;
        std::string text;
    };

    tts_worker_params params;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    std::deque<job> jobs;
    bool running = false;
    bool stop_requested = false;
    bool turn_open = false;
    bool turn_closed = false;
    bool in_flight = false;
    uint64_t active_turn_id = 0;
    uint64_t completed_turn_id = 0;
    int64_t turn_first_enqueue_ms = 0;
    tts_worker_turn_metrics turn_metrics;

#ifndef _WIN32
    pid_t piper_pid = -1;
    FILE * piper_stdin = nullptr;
#endif

    bool start() {
        std::lock_guard<std::mutex> lock(mutex);
        if (running) {
            return true;
        }
        stop_requested = false;
        worker = std::thread(&impl::run, this);
        running = true;
        return true;
    }

    void begin_turn() {
        std::lock_guard<std::mutex> lock(mutex);
        ++active_turn_id;
        turn_open = true;
        turn_closed = false;
        turn_first_enqueue_ms = 0;
        turn_metrics = {};
    }

    bool submit(const std::string & text) {
        if (text.empty()) {
            return true;
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (!running || !turn_open) {
            return false;
        }
        if (turn_first_enqueue_ms == 0) {
            turn_first_enqueue_ms = now_ms();
        }
        jobs.push_back(job{active_turn_id, text});
        cv.notify_all();
        return true;
    }

    tts_worker_turn_metrics end_turn() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!turn_open) {
            return {};
        }
        turn_closed = true;

        if (turn_first_enqueue_ms == 0) {
            completed_turn_id = active_turn_id;
            turn_open = false;
            turn_closed = false;
            return {};
        }

        cv.notify_all();
        const uint64_t turn_id = active_turn_id;
        cv.wait(lock, [&]() { return completed_turn_id >= turn_id || !running; });

        tts_worker_turn_metrics metrics = turn_metrics;
        turn_open = false;
        turn_closed = false;
        return metrics;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!running) {
                return;
            }
            stop_requested = true;
            cv.notify_all();
        }

        if (worker.joinable()) {
            worker.join();
        }

        std::lock_guard<std::mutex> lock(mutex);
        running = false;
    }

    void run() {
        while (true) {
            job current;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() { return stop_requested || !jobs.empty(); });
                if (stop_requested && jobs.empty()) {
                    break;
                }

                current = jobs.front();
                jobs.pop_front();
                in_flight = true;
            }

            process_job(current);

            {
                std::lock_guard<std::mutex> lock(mutex);
                in_flight = false;
                if (turn_closed && current.turn_id == active_turn_id && jobs.empty()) {
                    completed_turn_id = current.turn_id;
                    if (turn_first_enqueue_ms != 0) {
                        turn_metrics.total_ms = now_ms() - turn_first_enqueue_ms;
                    }
                }
                cv.notify_all();
            }
        }

        shutdown_piper();
    }

    void process_job(const job & current) {
        const int64_t t_job_start = now_ms();

        {
            std::lock_guard<std::mutex> lock(mutex);
            if (current.turn_id == active_turn_id && turn_metrics.chunks == 0 && turn_first_enqueue_ms != 0) {
                turn_metrics.startup_ms = t_job_start - turn_first_enqueue_ms;
            }
        }

        speak_metrics metrics;
        bool ok = false;
        if (params.mode == tts_mode::piper_persistent) {
            ok = speak_with_piper(current.text, &metrics);
        } else {
            ok = speak_with_file(params.command, current.text, params.file_path, params.voice_id, &metrics);
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (current.turn_id == active_turn_id) {
            turn_metrics.ok = turn_metrics.ok && ok && metrics.ok;
            turn_metrics.chunks++;
        }
    }

    bool speak_with_piper(const std::string & text, speak_metrics * metrics) {
#ifdef _WIN32
        (void) text;
        if (metrics != nullptr) {
            metrics->ok = false;
        }
        fprintf(stderr, "%s: piper-persistent is not supported on Windows, use script mode instead\n", __func__);
        return false;
#else
        const int64_t t_start = now_ms();

        if (!ensure_piper_process()) {
            if (metrics != nullptr) {
                metrics->ok = false;
            }
            return false;
        }

        if (metrics != nullptr) {
            metrics->startup_ms = now_ms() - t_start;
        }

        if (std::fwrite(text.data(), 1, text.size(), piper_stdin) != text.size()) {
            fprintf(stderr, "%s: failed to write text to piper\n", __func__);
            if (metrics != nullptr) {
                metrics->ok = false;
                metrics->total_ms = now_ms() - t_start;
            }
            shutdown_piper();
            return false;
        }

        std::fputc('\n', piper_stdin);
        std::fflush(piper_stdin);

        if (metrics != nullptr) {
            metrics->total_ms = now_ms() - t_start;
        }
        return true;
#endif
    }

#ifndef _WIN32
    bool ensure_piper_process() {
        if (piper_stdin != nullptr) {
            return true;
        }

        int pipefd[2];
        if (pipe(pipefd) != 0) {
            fprintf(stderr, "%s: failed to create piper stdin pipe\n", __func__);
            return false;
        }

        std::string shell_cmd = "piper --model " + params.piper_model + " --output-raw | " + params.piper_output_cmd;
        pid_t pid = fork();
        if (pid == -1) {
            fprintf(stderr, "%s: failed to fork piper worker\n", __func__);
            close(pipefd[0]);
            close(pipefd[1]);
            return false;
        }

        if (pid == 0) {
            dup2(pipefd[0], STDIN_FILENO);
            close(pipefd[0]);
            close(pipefd[1]);
            execl("/bin/sh", "sh", "-lc", shell_cmd.c_str(), (char *) nullptr);
            _exit(127);
        }

        close(pipefd[0]);
        piper_pid = pid;
        piper_stdin = fdopen(pipefd[1], "w");
        if (piper_stdin == nullptr) {
            fprintf(stderr, "%s: failed to open piper stdin stream\n", __func__);
            close(pipefd[1]);
            shutdown_piper();
            return false;
        }

        return true;
    }

    void shutdown_piper() {
        if (piper_stdin != nullptr) {
            fclose(piper_stdin);
            piper_stdin = nullptr;
        }

        if (piper_pid > 0) {
            int status = 0;
            waitpid(piper_pid, &status, 0);
            piper_pid = -1;
        }
    }
#else
    bool ensure_piper_process() {
        return false;
    }

    void shutdown_piper() {}
#endif
};

tts_worker::tts_worker(tts_worker_params params) : impl_(new impl(std::move(params))) {}

tts_worker::~tts_worker() {
    stop();
}

bool tts_worker::start() {
    return impl_->start();
}

void tts_worker::begin_turn() {
    impl_->begin_turn();
}

bool tts_worker::submit(const std::string & text) {
    return impl_->submit(text);
}

tts_worker_turn_metrics tts_worker::end_turn() {
    return impl_->end_turn();
}

void tts_worker::stop() {
    impl_->stop();
}

#undef STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"
