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

// =============================================
// 全局状态管理
// =============================================
std::atomic<bool> is_recording(false);
std::atomic<bool> exit_program(false);
std::atomic<int> recorded_seconds(0);

// 音频缓冲区与锁
std::vector<float> audio_buffer;
std::mutex buffer_mutex;

// 配置常量
int RECORD_TIMEOUT = 30; // 可变，支持动态测试

// =============================================
// 系统工具函数
// =============================================

void signal_handler(int sig) {
    if (sig == SIGINT) {
        printf("\n\n🛑 收到退出信号，正在清理资源...\n");
        exit_program.store(true);
        is_recording.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        exit(0);
    }
}

// 非阻塞检查标准输入（带毫秒超时）
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

// =============================================
// 音频回调 (硬件驱动层)
// =============================================
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    // 只要 is_recording 为 true，回调就会持续把数据写入 buffer
    if (!is_recording.load() || pInput == NULL) return;
    const float* pInputFloat = (const float*)pInput;
    std::lock_guard<std::mutex> lock(buffer_mutex);
    audio_buffer.insert(audio_buffer.end(), pInputFloat, pInputFloat + frameCount);
    recorded_seconds.store(static_cast<int>(audio_buffer.size() / 16000.0));
}

// =============================================
// 识别逻辑
// =============================================
void recognize_audio(struct whisper_context* ctx, const std::vector<float>& audio_data) {
    if (audio_data.empty()) return;
    float total_sec = (float)audio_data.size() / 16000.0f;
    
    printf("\n🔍 正在识别（总采样长度：%.2fs）...\n", total_sec);
    
    auto t_start = std::chrono::steady_clock::now();
    
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "zh";
    wparams.n_threads = std::max(2, (int)std::thread::hardware_concurrency());
    wparams.print_progress = false;

    if (whisper_full(ctx, wparams, audio_data.data(), audio_data.size()) != 0) {
        fprintf(stderr, "❌ Whisper 推理失败\n");
        return;
    }

    auto t_end = std::chrono::steady_clock::now();
    float msec = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    
    printf("⏱️  推理耗时：%.2f 秒 | 速度：%.2fx 实时\n", msec/1000.0f, total_sec/(msec/1000.0f));
    
    int n_segments = whisper_full_n_segments(ctx);
    printf("📝 结果：\n");
    for (int i = 0; i < n_segments; ++i) {
        printf("   %s\n", whisper_full_get_segment_text(ctx, i));
    }
}

// =============================================
// 主程序
// =============================================
int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    if (argc < 2) {
        printf("Usage: %s <model_path> [timeout_seconds]\n", argv[0]);
        return 1;
    }
    if (argc >= 3) {
        RECORD_TIMEOUT = atoi(argv[2]);
    }

    // 1. Whisper 初始化 (强制 GPU 后端)
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; 
    struct whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cparams);
    if (!ctx) return 1;

    // 2. 音频设备初始化 (AB13X USB)
    ma_context context;
    ma_context_init(NULL, 0, NULL, &context);
    ma_device_config devCfg = ma_device_config_init(ma_device_type_capture);
    devCfg.capture.format = ma_format_f32;
    devCfg.capture.channels = 1;
    devCfg.sampleRate = 16000;
    devCfg.dataCallback = data_callback;
    
    ma_device device;
    if (ma_device_init(&context, &devCfg, &device) != MA_SUCCESS) return 1;
    ma_device_start(&device);

    while (!exit_program.load()) {
        printf("\n=============================================\n");
        printf("🎙️  操作提示 (当前超时: %d秒):\n", RECORD_TIMEOUT);
        printf("  ▶ [回车键] : 开始录制\n");
        printf("  ■ [回车键] : 停止录制 (会有1.5秒平滑收尾)\n");
        printf("=============================================\n");
        printf("👉 等待指令...");
        fflush(stdout);

        while (!check_input_non_blocking(50) && !exit_program.load());
        if (exit_program.load()) break;
        clear_input_buffer();

        // 重置状态
        { std::lock_guard<std::mutex> lock(buffer_mutex); audio_buffer.clear(); }
        recorded_seconds.store(0);
        is_recording.store(true);
        auto start_time = std::chrono::steady_clock::now();

        printf("\n🎙️  正在录制 (进度将在下方实时更新)... \n");

        // 进度显示线程
        std::thread progress_thread([&]() {
            while (is_recording.load() && !exit_program.load()) {
                printf("\r📊 进度: %d 秒    ", recorded_seconds.load());
                fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });

        bool stop_triggered = false;
        while (!exit_program.load() && !stop_triggered) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();

            // 检查手动停止
            if (check_input_non_blocking(10)) {
                char c;
                if (read(STDIN_FILENO, &c, 1) > 0 && c == '\n') {
                    printf("\n🛑 检测到手动回车，准备收尾...");
                    stop_triggered = true;
                }
            } 
            // 检查超时停止
            else if (elapsed_ms >= (RECORD_TIMEOUT * 1000)) {
                printf("\n⏱️  达到 %d 秒阈值，准备收尾...", RECORD_TIMEOUT);
                stop_triggered = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // --- 核心改动：平滑收尾逻辑 ---
        // 即使触发了停止，我们也不立即关闭 is_recording 开关
        // 这样可以确保正在 ALSA 缓冲区或 DMA 队列里的数据被 data_callback 继续捞走
        if (stop_triggered) {
            printf("正在执行 1.5 秒平滑数据刷新...");
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(1500)); 
            is_recording.store(false); // 此时才真正切断回调写入
            printf("完成。\n");
        }

        if (progress_thread.joinable()) progress_thread.join();

        // 拷贝数据进行识别
        std::vector<float> captured;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            captured = audio_buffer;
        }
        recognize_audio(ctx, captured);
    }

    ma_device_uninit(&device);
    ma_context_uninit(&context);
    whisper_free(ctx);
    return 0;
}