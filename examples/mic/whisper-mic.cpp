#include <string>

struct mic_params {
    std::string model = "models/ggml-base.bin";
    int timeout = 30;
    int capture_id = 5;
    std::string language = "zh";
    bool use_gpu = true;
};

void mic_print_usage(int argc, char ** argv, const mic_params & params) {
    printf("\n");
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h,   --help           show this help message and exit\n");
    printf("  -m F, --model F        [%-7s] model path\n", params.model.c_str());
    printf("  -t N, --timeout N      [%-7d] max recording time in seconds\n", params.timeout);
    printf("  -c N, --capture N      [%-7d] capture device ID\n", params.capture_id);
    printf("  -l S, --language S     [%-7s] language (e.g. zh, en)\n", params.language.c_str());
    printf("  -ng,  --no-gpu         [%-7s] disable GPU inference\n", params.use_gpu ? "false" : "true");
    printf("\n");
    printf("example: %s -m models/ggml-base.bin -t 30 -c 0 -l zh\n", argv[0]);
    printf("\n");
}

static bool mic_params_parse(int argc, char ** argv, mic_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            mic_print_usage(argc, argv, params);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-t" || arg == "--timeout") {
            params.timeout = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--capture") {
            params.capture_id = std::stoi(argv[++i]);
        } else if (arg == "-l" || arg == "--language") {
            params.language = argv[++i];
        } else if (arg == "-ng" || arg == "--no-gpu") {
            params.use_gpu = false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            mic_print_usage(argc, argv, params);
            exit(1);
        }
    }
    return true;
}
#include "whisper.h"
#include "common.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <vector>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <thread>
#include <csignal>
#include <mutex>
#include <unistd.h>
#include <sys/select.h>

// =============================================
// Shared constants
// =============================================
struct RecordingConfig {
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int PROGRESS_MS = 100;           
    static constexpr int UI_LOOP_MS = 10;             
    static constexpr int SELECT_TIMEOUT_MS = 20;      
    static constexpr int SMOOTH_FINISH_MS = 300;
    static constexpr int CLOCK_TOLERANCE_MS = 350;    
    // Used to clear leftover characters in terminal progress rendering.
    static const char* CLEAR_LINE; 
};
const char* RecordingConfig::CLEAR_LINE = "                                        ";

std::atomic<bool> is_recording(false);
std::atomic<bool> exit_program(false);
std::atomic<int> recorded_seconds(0);
std::vector<float> audio_buffer;
std::mutex buffer_mutex;
int g_timeout_limit = 30; 

void signal_handler(int sig) {
    if (sig == SIGINT) {
        exit_program.store(true);
        is_recording.store(false);
        exit(0);
    }
}

bool check_stdin_ready(int timeout_ms = RecordingConfig::SELECT_TIMEOUT_MS) {
    fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
    struct timeval tv = {0, timeout_ms * 1000};
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

void clear_stdin() {
    while (check_stdin_ready(0)) getchar();
}

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (!is_recording.load() || pInput == NULL) return;
    std::lock_guard<std::mutex> lock(buffer_mutex);
    audio_buffer.insert(audio_buffer.end(), (float*)pInput, (float*)pInput + frameCount);
    recorded_seconds.store(static_cast<int>(audio_buffer.size() / (float)RecordingConfig::SAMPLE_RATE));
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    mic_params params;
    mic_params_parse(argc, argv, params);
    g_timeout_limit = params.timeout;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    struct whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from '%s'\n", params.model.c_str());
        return 1;
    }

    ma_context context;
    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        fprintf(stderr, "Failed to initialize miniaudio context\n");
        whisper_free(ctx);
        return 1;
    }
    ma_device_info* pCapInfos; ma_uint32 capCount;
    if (ma_context_get_devices(&context, NULL, NULL, &pCapInfos, &capCount) != MA_SUCCESS || capCount == 0) {
        fprintf(stderr, "No audio capture devices found\n");
        ma_context_uninit(&context);
        whisper_free(ctx);
        return 1;
    }

    printf("\n📜 Available microphones:\n");
    for (ma_uint32 i = 0; i < capCount; ++i) printf("  [%u] %s\n", i, pCapInfos[i].name);
    printf("👉 Enter device ID (default %d): ", params.capture_id);
    ma_uint32 dev_id = params.capture_id;
    if(scanf("%u", &dev_id) != 1) dev_id = params.capture_id;
    clear_stdin();

    ma_device_config devCfg = ma_device_config_init(ma_device_type_capture);
    devCfg.capture.format = ma_format_f32; devCfg.capture.channels = 1;
    devCfg.sampleRate = RecordingConfig::SAMPLE_RATE; devCfg.dataCallback = data_callback;
    if (dev_id < capCount) devCfg.capture.pDeviceID = &pCapInfos[dev_id].id;

    ma_device device;
    if (ma_device_init(&context, &devCfg, &device) != MA_SUCCESS) {
        fprintf(stderr, "Failed to initialize audio capture device\n");
        ma_context_uninit(&context);
        whisper_free(ctx);
        return 1;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
        fprintf(stderr, "Failed to start audio capture device\n");
        ma_device_uninit(&device);
        ma_context_uninit(&context);
        whisper_free(ctx);
        return 1;
    }

    while (!exit_program.load()) {
        printf("\n=============================================\n");
        printf("🎙️  Controls (auto stop after %d seconds):\n", g_timeout_limit);
        printf("  ▶ [Enter] : start recording\n");
        printf("  ■ [Enter] : stop recording (with 1.5s tail capture)\n");
        printf("=============================================\n");
        printf("👉 Waiting for input...");
        fflush(stdout);

        while (!check_stdin_ready(100) && !exit_program.load());
        if (exit_program.load()) break;
        clear_stdin();

        { std::lock_guard<std::mutex> lock(buffer_mutex); audio_buffer.clear(); }
        recorded_seconds.store(0);
        is_recording.store(true);
        auto start_time = std::chrono::steady_clock::now();

        printf("\n🎙️  Recording...\n");

        std::thread progress_thread([&]() {
            while (is_recording.load()) {
                printf("\r%s\r📊 Elapsed: %d s", RecordingConfig::CLEAR_LINE, recorded_seconds.load());
                fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(RecordingConfig::PROGRESS_MS));
            }
        });

        bool trigger_stop = false;
        while (!trigger_stop && !exit_program.load()) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            if (check_stdin_ready(RecordingConfig::UI_LOOP_MS)) {
                if (getchar() == '\n') {
                    // Use carriage return to overwrite and clear the previous line.
                    printf("\r%s\r🛑 Stopping manually, finalizing capture...", RecordingConfig::CLEAR_LINE);
                    fflush(stdout);
                    trigger_stop = true;
                }
            } 
            else if (elapsed >= (g_timeout_limit * 1000 + RecordingConfig::CLOCK_TOLERANCE_MS)) {
                printf("\r%s\r📊 Elapsed: %d s", RecordingConfig::CLEAR_LINE, g_timeout_limit);
                printf("\n⏱️  Time limit reached (%d s), finalizing capture...", g_timeout_limit);
                fflush(stdout);
                trigger_stop = true;
            }
        }


        // Stop progress meter immediately
        is_recording.store(false);
        if (progress_thread.joinable()) progress_thread.join();
        // Now wait for smooth finish (buffer tail)
        std::this_thread::sleep_for(std::chrono::milliseconds(RecordingConfig::SMOOTH_FINISH_MS));

        std::vector<float> captured;
        { std::lock_guard<std::mutex> lock(buffer_mutex); captured = audio_buffer; }

        if (captured.empty()) {
            printf("\n⚠️  No audio captured, skipping recognition.\n");
            continue;
        }

        printf("\n🔍 Running recognition (audio length: %.2fs)...", (float)captured.size()/RecordingConfig::SAMPLE_RATE);
        auto start_recognition = std::chrono::steady_clock::now();

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = params.language.c_str();
        if (params.language == "zh") {
            wparams.initial_prompt = "The following speech is Mandarin Chinese. Output in Simplified Chinese.";
        }
        wparams.n_threads = 4;

        if (whisper_full(ctx, wparams, captured.data(), captured.size()) != 0) {
            printf("\n❌ Speech recognition failed.\n");
            continue;
        }

        int n_segments = whisper_full_n_segments(ctx);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_recognition).count();
        printf("\n📝 Recognition result (%.2f s):", elapsed/1000.0f);
        for (int i = 0; i < n_segments; ++i) {
            printf("\n   %s", whisper_full_get_segment_text(ctx, i));
        }
        printf("\n");
    }

    ma_device_uninit(&device);
    ma_context_uninit(&context);
    whisper_free(ctx);
    return 0;
}
