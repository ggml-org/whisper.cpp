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
const int RECORD_TIMEOUT = 30; // 严格30秒超时

// 信号处理
void signal_handler(int sig) {
    if (sig == SIGINT) {
        printf("\n\n🛑 收到退出信号，正在清理资源...\n");
        exit_program.store(true);
        is_recording.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        exit(0);
    }
}

// 非阻塞检查标准输入
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
    while (check_input_non_blocking(10)) {
        char c;
        read(STDIN_FILENO, &c, 1);
    }
}

// 音频采集回调
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (!is_recording.load() || pInput == NULL) return;
    const float* pInputFloat = (const float*)pInput;
    std::lock_guard<std::mutex> lock(buffer_mutex);
    audio_buffer.insert(audio_buffer.end(), pInputFloat, pInputFloat + frameCount);
    recorded_seconds.store(static_cast<int>(audio_buffer.size() / 16000.0));
}

// 裁剪开头静音
int trim_silence(const float* audio_data, int audio_len, float threshold = 0.001f) {
    int start = 0;
    while (start < audio_len && std::abs(audio_data[start]) < threshold) {
        start++;
    }
    return std::max(audio_len - start, 16000); 
}

// 核心识别函数
void recognize_audio(struct whisper_context* ctx, const std::vector<float>& audio_data) {
    if (audio_data.empty()) return;

    int valid_len = trim_silence(audio_data.data(), audio_data.size());
    float total_sec = (float)audio_data.size() / 16000.0f;
    
    printf("🔍 正在识别（有效长度：%.2fs，总长：%.2fs）...\n", (float)valid_len/16000.0f, total_sec);
    
    auto t_start = std::chrono::steady_clock::now();
    
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "zh";
    wparams.n_threads = std::max(2, (int)std::thread::hardware_concurrency());
    wparams.print_progress = false;
    wparams.no_context = true;
    wparams.single_segment = false;

    if (whisper_full(ctx, wparams, audio_data.data(), valid_len) != 0) {
        fprintf(stderr, "❌ 识别失败\n");
        return;
    }

    auto t_end = std::chrono::steady_clock::now();
    float msec = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    
    printf("⏱️  识别耗时：%.2f 秒 | 识别速度：%.2fx\n", msec/1000.0f, total_sec/(msec/1000.0f));
    
    int n_segments = whisper_full_n_segments(ctx);
    printf("📝 识别结果：\n");
    for (int i = 0; i < n_segments; ++i) {
        printf("   %s\n", whisper_full_get_segment_text(ctx, i));
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    // 1. 初始化音频与设备列表
    ma_context context;
    ma_context_init(NULL, 0, NULL, &context);
    ma_device_info* pCaptureInfos = NULL;
    ma_uint32 captureCount = 0;
    ma_context_get_devices(&context, NULL, NULL, &pCaptureInfos, &captureCount);

    printf("\n📜 系统可用麦克风：\n");
    for (ma_uint32 i = 0; i < captureCount; ++i) {
        printf(" [%u] %s\n", i, pCaptureInfos[i].name);
    }

    ma_uint32 device_id = 0;
    printf("\n👉 选择麦克风ID: ");
    if(scanf("%u", &device_id) != 1) device_id = 0;
    clear_input_buffer();

    // 2. 初始化 Whisper (开启 GPU)
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // 确认开启 GPU
    cparams.gpu_device = 0;

    printf("\n🚀 正在加载模型: %s\n", argv[1]);
    struct whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cparams);
    if (!ctx) return 1;

    // 3. 配置录音设备
    ma_device_config devCfg = ma_device_config_init(ma_device_type_capture);
    devCfg.capture.format = ma_format_f32;
    devCfg.capture.channels = 1;
    devCfg.sampleRate = 16000;
    devCfg.dataCallback = data_callback;
    if (captureCount > 0) devCfg.capture.pDeviceID = &pCaptureInfos[device_id].id;

    ma_device device;
    if (ma_device_init(&context, &devCfg, &device) != MA_SUCCESS) return 1;
    ma_device_start(&device);

    // 统一收尾 Lambda
    auto stop_and_collect = [&](const char* reason) {
        printf("\n%s，捕获 800ms 余音...", reason);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(800)); // 保证 30s 结尾不丢包
        is_recording.store(false);
        printf("完成。\n");
    };

    while (!exit_program.load()) {
        printf("\n👉 按回车键开始录制...");
        while (!check_input_non_blocking(50) && !exit_program.load());
        if (exit_program.load()) break;
        clear_input_buffer();

        // 精准计时起点
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            audio_buffer.clear();
        }
        recorded_seconds.store(0);
        auto start_time = std::chrono::steady_clock::now(); // 严格对齐
        is_recording.store(true);

        printf("🎙️  正在录制 (最长 30s)... \n");

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
            double elapsed = std::chrono::duration<double>(now - start_time).count();

            // 1. 检查手动回车
            if (check_input_non_blocking(10)) {
                char c;
                if (read(STDIN_FILENO, &c, 1) > 0 && c == '\n') {
                    stop_and_collect("🛑 手动停止");
                    stopped = true;
                }
            }
            // 2. 检查 30s 超时
            else if (elapsed >= (double)RECORD_TIMEOUT) {
                stop_and_collect("⏱️ 超时停止 (30s)");
                stopped = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        if (progress_thread.joinable()) progress_thread.join();

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