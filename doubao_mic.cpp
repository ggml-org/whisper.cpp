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
// 工程常量：全局统一管理，杜绝 Magic Numbers
// =============================================
struct RecordingConfig {
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int PROGRESS_MS = 100;           // 进度条刷新间隔
    static constexpr int UI_LOOP_MS = 10;             // UI循环步长
    static constexpr int SELECT_TIMEOUT_MS = 20;      // select 超时
    static constexpr int SMOOTH_FINISH_MS = 1500;     // 平滑收尾时长 (1.5秒)
    static constexpr int CLOCK_TOLERANCE_MS = 300;    // 边界容差，确保显示完整
};

std::atomic<bool> is_recording(false);
std::atomic<bool> exit_program(false);
std::atomic<int> recorded_seconds(0);
std::vector<float> audio_buffer;
std::mutex buffer_mutex;
int g_timeout_limit = 30; // 默认 30s

void signal_handler(int sig) {
    if (sig == SIGINT) {
        exit_program.store(true);
        is_recording.store(false);
        exit(0);
    }
}

// 非阻塞输入检测
bool check_stdin_ready(int timeout_ms = RecordingConfig::SELECT_TIMEOUT_MS) {
    fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
    struct timeval tv = {0, timeout_ms * 1000};
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

void clear_stdin() {
    while (check_stdin_ready(0)) getchar();
}

// 音频采集回调
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (!is_recording.load() || pInput == NULL) return;
    std::lock_guard<std::mutex> lock(buffer_mutex);
    audio_buffer.insert(audio_buffer.end(), (float*)pInput, (float*)pInput + frameCount);
    recorded_seconds.store(static_cast<int>(audio_buffer.size() / (float)RecordingConfig::SAMPLE_RATE));
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    if (argc < 2) {
        printf("用法: %s <模型路径> [超时秒数]\n", argv[0]);
        return 1;
    }
    if (argc >= 3) g_timeout_limit = atoi(argv[2]);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    struct whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cparams);

    // 设备枚举与选择
    ma_context context; ma_context_init(NULL, 0, NULL, &context);
    ma_device_info* pCapInfos; ma_uint32 capCount;
    ma_context_get_devices(&context, NULL, NULL, &pCapInfos, &capCount);
    printf("\n📜 可用麦克风列表:\n");
    for (ma_uint32 i = 0; i < capCount; ++i) printf("  [%u] %s\n", i, pCapInfos[i].name);
    printf("👉 请输入设备 ID (默认5): ");
    ma_uint32 dev_id = 5; 
    if(scanf("%u", &dev_id) != 1) dev_id = 5;
    clear_stdin();

    ma_device_config devCfg = ma_device_config_init(ma_device_type_capture);
    devCfg.capture.format = ma_format_f32; devCfg.capture.channels = 1;
    devCfg.sampleRate = RecordingConfig::SAMPLE_RATE; devCfg.dataCallback = data_callback;
    if (dev_id < capCount) devCfg.capture.pDeviceID = &pCapInfos[dev_id].id;

    ma_device device; ma_device_init(&context, &devCfg, &device);
    ma_device_start(&device);

    while (!exit_program.load()) {
        printf("\n=============================================\n");
        printf("🎙️  操作提示 (自动断开设置: %d 秒):\n", g_timeout_limit);
        printf("  ▶ [回车键] : 开始录制\n");
        printf("  ■ [回车键] : 停止录制 (含 %.1f 秒补录)\n", (float)RecordingConfig::SMOOTH_FINISH_MS/1000.0f);
        printf("=============================================\n");
        printf("👉 等待指令...");
        fflush(stdout);

        while (!check_stdin_ready(100) && !exit_program.load());
        if (exit_program.load()) break;
        clear_stdin();

        { std::lock_guard<std::mutex> lock(buffer_mutex); audio_buffer.clear(); }
        recorded_seconds.store(0);
        is_recording.store(true);
        auto start_time = std::chrono::steady_clock::now();

        printf("\n🎙️  录制中...\n");

        std::thread progress_thread([&]() {
            while (is_recording.load()) {
                printf("\r📊 进度: %d 秒    ", recorded_seconds.load());
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
                    printf("\n🛑 手动停止，正在收尾以确保不丢字...");
                    trigger_stop = true;
                }
            } 
            // 关键：n + 容差，确保进度条能显示出最后那一秒
            else if (elapsed >= (g_timeout_limit * 1000 + RecordingConfig::CLOCK_TOLERANCE_MS)) {
                printf("\r📊 进度: %d 秒    ", g_timeout_limit);
                printf("\n⏱️  时间已到 (%d秒)，正在自动收尾...", g_timeout_limit);
                trigger_stop = true;
            }
        }

        // 执行平滑刷新
        std::this_thread::sleep_for(std::chrono::milliseconds(RecordingConfig::SMOOTH_FINISH_MS));
        is_recording.store(false); 
        if (progress_thread.joinable()) progress_thread.join();

        std::vector<float> captured;
        { std::lock_guard<std::mutex> lock(buffer_mutex); captured = audio_buffer; }
        
        printf("\n🔍 正在识别 (音频长度: %.2fs)...", (float)captured.size()/RecordingConfig::SAMPLE_RATE);
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = "zh";
        // 2. 【核心修正】注入简体中文引导词，强制模型输出简体
        // "以下是普通话的句子。" 作为一个初始提示（Prompt）
        wparams.initial_prompt = "以下是普通话的句子，使用简体中文。";
        // 3. 翻译控制（确保不开启翻译模式）
        wparams.translate = false;
        wparams.n_threads = 4;
        whisper_full(ctx, wparams, captured.data(), captured.size());
        
        int n_segments = whisper_full_n_segments(ctx);
        printf("\n📝 识别结果：");
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