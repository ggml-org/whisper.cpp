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
// 1. 工程常量定义 (拒绝 Magic Numbers)
// =============================================
struct RecordingConfig {
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int PROGRESS_REFRESH_MS = 100;    // 进度条刷新频率
    static constexpr int UI_WAIT_MS = 10;              // 循环等待步长
    static constexpr int INPUT_CHECK_MS = 20;          // 输入检测超时
    static constexpr int POST_STOP_BUFFER_MS = 1500;   // 停止后的平滑采样缓冲
    static constexpr int CLOCK_TOLERANCE_MS = 200;     // 系统时钟容差（确保跳到目标秒数）
};

// 全局状态管理
std::atomic<bool> is_recording(false);
std::atomic<bool> exit_program(false);
std::atomic<int> recorded_seconds(0);
std::vector<float> audio_buffer;
std::mutex buffer_mutex;
int g_timeout_setting = 30; // 用户设定的超时秒数

// =============================================
// 系统辅助函数
// =============================================
void signal_handler(int sig) {
    if (sig == SIGINT) {
        exit_program.store(true);
        is_recording.store(false);
        exit(0);
    }
}

bool check_input_non_blocking(int timeout_ms = RecordingConfig::INPUT_CHECK_MS) {
    fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
    struct timeval tv = {0, timeout_ms * 1000};
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

// =============================================
// 音频回调
// =============================================
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (!is_recording.load() || pInput == NULL) return;
    std::lock_guard<std::mutex> lock(buffer_mutex);
    audio_buffer.insert(audio_buffer.end(), (float*)pInput, (float*)pInput + frameCount);
    recorded_seconds.store(static_cast<int>(audio_buffer.size() / (float)RecordingConfig::SAMPLE_RATE));
}

// =============================================
// 主逻辑
// =============================================
int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    if (argc < 2) return 1;
    if (argc >= 3) g_timeout_setting = atoi(argv[2]);

    // 初始化 Whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    struct whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cparams);

    // 麦克风设备枚举与选择
    ma_context context; ma_context_init(NULL, 0, NULL, &context);
    ma_device_info* pCapInfos; ma_uint32 capCount;
    ma_context_get_devices(&context, NULL, NULL, &pCapInfos, &capCount);
    for (ma_uint32 i = 0; i < capCount; ++i) printf("[%u] %s\n", i, pCapInfos[i].name);
    printf("👉 请输入设备 ID: ");
    ma_uint32 dev_id; scanf("%u", &dev_id);
    while (getchar() != '\n'); 

    ma_device_config devCfg = ma_device_config_init(ma_device_type_capture);
    devCfg.capture.format = ma_format_f32; 
    devCfg.capture.channels = 1;
    devCfg.sampleRate = RecordingConfig::SAMPLE_RATE; 
    devCfg.dataCallback = data_callback;
    if (dev_id < capCount) devCfg.capture.pDeviceID = &pCapInfos[dev_id].id;

    ma_device device; ma_device_init(&context, &devCfg, &device);
    ma_device_start(&device);

    while (!exit_program.load()) {
        printf("\n[回车] 录制 | [回车] 停止\n👉 等待指令...");
        while (!check_input_non_blocking(50) && !exit_program.load());
        if (exit_program.load()) break;
        while (check_input_non_blocking(0)) getchar();

        { std::lock_guard<std::mutex> lock(buffer_mutex); audio_buffer.clear(); }
        is_recording.store(true);
        auto start_time = std::chrono::steady_clock::now();

        // 进度显示线程
        std::thread progress_thread([&]() {
            while (is_recording.load()) {
                printf("\r📊 进度: %d 秒    ", recorded_seconds.load());
                fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(RecordingConfig::PROGRESS_REFRESH_MS));
            }
        });

        bool stop_triggered = false;
        while (!stop_triggered && !exit_program.load()) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            if (check_input_non_blocking(RecordingConfig::UI_WAIT_MS)) {
                if (getchar() == '\n') { 
                    printf("\n🛑 手动停止 (进入平滑刷新模式)..."); 
                    stop_triggered = true; 
                }
            } 
            // 修正边界：加上 CLOCK_TOLERANCE_MS 确保进度条在视觉上能显示到设定的秒数
            else if (elapsed >= (g_timeout_setting * 1000 + RecordingConfig::CLOCK_TOLERANCE_MS)) {
                printf("\r📊 进度: %d 秒    ", g_timeout_setting); // 强制补完最后一秒显示
                printf("\n⏱️  超时停止 (%d秒，进入平滑刷新模式)...", g_timeout_setting);
                stop_triggered = true;
            }
        }

        // 平滑刷新：等待硬件缓冲区数据入库，防止丢字
        std::this_thread::sleep_for(std::chrono::milliseconds(RecordingConfig::POST_STOP_BUFFER_MS));
        is_recording.store(false); 
        
        if (progress_thread.joinable()) progress_thread.join();

        // 识别逻辑
        std::vector<float> captured;
        { std::lock_guard<std::mutex> lock(buffer_mutex); captured = audio_buffer; }
        
        printf("\n🔍 识别中 (音频长: %.2fs)...", (float)captured.size()/RecordingConfig::SAMPLE_RATE);
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = "zh";
        wparams.n_threads = 4;
        whisper_full(ctx, wparams, captured.data(), captured.size());
        
        int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) printf("\n📝 %s", whisper_full_get_segment_text(ctx, i));
        printf("\n");
    }

    return 0;
}