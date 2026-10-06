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
// 工程常量：全局统一管理
// =============================================
struct RecordingConfig {
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int PROGRESS_MS = 100;           
    static constexpr int UI_LOOP_MS = 10;             
    static constexpr int SELECT_TIMEOUT_MS = 20;      
    static constexpr int SMOOTH_FINISH_MS = 1500;     
    static constexpr int CLOCK_TOLERANCE_MS = 350;    
    // 用于清理控制台残余的空格
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
    if (argc < 2) return 1;
    if (argc >= 3) g_timeout_limit = atoi(argv[2]);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    struct whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cparams);

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
        printf("  ■ [回车键] : 停止录制 (含 1.5 秒补录)\n");
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
                printf("\r%s\r📊 进度: %d 秒", RecordingConfig::CLEAR_LINE, recorded_seconds.load());
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
                    // 使用 \r 覆盖并清理残余
                    printf("\r%s\r🛑 手动停止，正在收尾以确保不丢字...", RecordingConfig::CLEAR_LINE);
                    fflush(stdout);
                    trigger_stop = true;
                }
            } 
            else if (elapsed >= (g_timeout_limit * 1000 + RecordingConfig::CLOCK_TOLERANCE_MS)) {
                printf("\r%s\r📊 进度: %d 秒", RecordingConfig::CLEAR_LINE, g_timeout_limit);
                printf("\n⏱️  时间已到 (%d秒)，正在自动收尾...", g_timeout_limit);
                fflush(stdout);
                trigger_stop = true;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(RecordingConfig::SMOOTH_FINISH_MS));
        is_recording.store(false); 
        if (progress_thread.joinable()) progress_thread.join();

        std::vector<float> captured;
        { std::lock_guard<std::mutex> lock(buffer_mutex); captured = audio_buffer; }
        
        printf("\n🔍 正在识别 (音频长度: %.2fs)...", (float)captured.size()/RecordingConfig::SAMPLE_RATE);
        auto start_recognition = std::chrono::steady_clock::now();
        
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = "zh";
        // 【正式固化】简体中文引导词
        wparams.initial_prompt = "以下是普通话，使用简体中文输出。"; 
        wparams.n_threads = 4;
        
        whisper_full(ctx, wparams, captured.data(), captured.size());
        
        int n_segments = whisper_full_n_segments(ctx);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_recognition).count();
        printf("\n📝 识别结果(%.2f秒)：", elapsed/1000.0f);
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