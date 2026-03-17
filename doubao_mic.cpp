#include "whisper.h"
#include "common.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <vector>
#include <cstdio>
#include <string>
#include <atomic>
#include <chrono>
#include <thread>
#include <csignal>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>

// 全局状态管理
std::atomic<bool> is_recording(false);
std::atomic<bool> exit_program(false);
std::atomic<int> recorded_seconds(0);

// 音频缓冲区与锁
std::vector<float> audio_buffer;
std::mutex buffer_mutex;

// 配置常量
const int RECORD_TIMEOUT = 24; // 目标 30 秒

void signal_handler(int sig) {
    if (sig == SIGINT) {
        printf("\n\n🛑 收到退出信号，正在清理资源...\n");
        exit_program.store(true);
        is_recording.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        exit(0);
    }
}

bool check_input_non_blocking(int timeout_ms = 20) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = timeout_ms * 1000;
    int ret;
    do {
        ret = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
    } while (ret == -1 && errno == EINTR);
    return ret > 0;
}

void clear_input_buffer() {
    while (check_input_non_blocking(5)) {
        char c;
        read(STDIN_FILENO, &c, 1);
    }
}

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (!is_recording.load() || pInput == NULL) return;
    const float* pInputFloat = (const float*)pInput;
    std::lock_guard<std::mutex> lock(buffer_mutex);
    audio_buffer.insert(audio_buffer.end(), pInputFloat, pInputFloat + frameCount);
    recorded_seconds.store(static_cast<int>(audio_buffer.size() / 16000.0));
}

void print_status_guide() {
    printf("\n=============================================\n");
    printf("🎙️  操作提示：\n");
    printf("  ▶ [回车键] : 开始录制\n");
    printf("  ■ [回车键] : 停止录制并识别\n");
    printf("  ⏳ [自动停止]: 达到 %d 秒自动截断\n", RECORD_TIMEOUT);
    printf("=============================================\n");
}

void recognize_audio(struct whisper_context* ctx, const std::vector<float>& audio_data) {
    if (audio_data.empty()) return;
    float total_sec = (float)audio_data.size() / 16000.0f;
    printf("\n🔍 正在识别（总长度：%.2fs）...\n", total_sec);
    
    auto t_start = std::chrono::steady_clock::now();
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "zh";
    wparams.n_threads = std::max(2, (int)std::thread::hardware_concurrency());
    wparams.print_progress = false;

    if (whisper_full(ctx, wparams, audio_data.data(), audio_data.size()) != 0) {
        fprintf(stderr, "❌ 识别失败\n");
        return;
    }

    auto t_end = std::chrono::steady_clock::now();
    float msec = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    printf("⏱️  识别耗时：%.2f 秒 | 速度：%.2fx\n", msec/1000.0f, total_sec/(msec/1000.0f));
    
    int n_segments = whisper_full_n_segments(ctx);
    printf("📝 结果：\n");
    for (int i = 0; i < n_segments; ++i) {
        printf("   %s\n", whisper_full_get_segment_text(ctx, i));
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    if (argc < 2) return 1;

    ma_context context;
    ma_context_init(NULL, 0, NULL, &context);
    ma_device_info* pCaptureInfos = NULL;
    ma_uint32 captureCount = 0;
    ma_context_get_devices(&context, NULL, NULL, &pCaptureInfos, &captureCount);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; 
    struct whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cparams);
    if (!ctx) return 1;

    ma_device_config devCfg = ma_device_config_init(ma_device_type_capture);
    devCfg.capture.format = ma_format_f32;
    devCfg.capture.channels = 1;
    devCfg.sampleRate = 16000;
    devCfg.dataCallback = data_callback;
    if (captureCount > 5) devCfg.capture.pDeviceID = &pCaptureInfos[5].id; // 锁定你的 AB13X

    ma_device device;
    ma_device_init(&context, &devCfg, &device);
    ma_device_start(&device);

    while (!exit_program.load()) {
        print_status_guide(); // 修复：增加每轮提示
        printf("👉 等待按回车开始...");
        fflush(stdout);

        while (!check_input_non_blocking(50) && !exit_program.load());
        if (exit_program.load()) break;
        clear_input_buffer();

        // 开始录制
        { std::lock_guard<std::mutex> lock(buffer_mutex); audio_buffer.clear(); }
        recorded_seconds.store(0);
        is_recording.store(true);
        auto start_time = std::chrono::steady_clock::now();

        printf("\n🎙️  录制中 (按回车停止)... \n");

        std::thread progress_thread([&]() {
            while (is_recording.load() && !exit_program.load()) {
                printf("\r📊 进度: %d 秒", recorded_seconds.load());
                fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });

        bool stopped = false;
        while (!exit_program.load() && !stopped) {
            auto now = std::chrono::steady_clock::now();
            // 修复：使用更精确的毫秒对比，并增加 500ms 冗余以确保达到 30s
            double elapsed = std::chrono::duration<double, std::milli>(now - start_time).count();

            if (check_input_non_blocking(10)) {
                char c;
                if (read(STDIN_FILENO, &c, 1) > 0 && c == '\n') {
                    printf("\n🛑 手动停止录制...");
                    stopped = true;
                }
            } else if (elapsed >= (RECORD_TIMEOUT * 1000 + 500)) { // 严格 30.5 秒逻辑
                printf("\n⏱️  达到 30 秒限制，自动切断...");
                stopped = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        // 停止回调并捕获尾音
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        is_recording.store(false);
        if (progress_thread.joinable()) progress_thread.join();

        std::vector<float> captured;
        { std::lock_guard<std::mutex> lock(buffer_mutex); captured = audio_buffer; }
        recognize_audio(ctx, captured);
    }

    ma_device_uninit(&device);
    whisper_free(ctx);
    return 0;
}