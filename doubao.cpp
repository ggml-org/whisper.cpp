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

// 全局原子变量控制录制状态（线程安全）
std::atomic<bool> is_recording(false);
// 音频缓冲区
std::vector<float> audio_buffer;

// 音频回调：仅在录制状态时才采集数据
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (!is_recording.load()) return; // 非录制状态直接返回，不采集数据
    const float* pInputFloat = (const float*)pInput;
    if (pInputFloat == NULL) return;

    // 采集数据到缓冲区（限制最大录制时长为30秒，防止溢出）
    const size_t max_frames = 16000 * 30; // 30秒 @ 16kHz
    const size_t available = max_frames - audio_buffer.size();
    if (available == 0) return; // 缓冲区已满，停止采集

    const size_t copy_frames = (frameCount > available) ? available : frameCount;
    audio_buffer.insert(audio_buffer.end(), pInputFloat, pInputFloat + copy_frames);
}

// 提示信息函数
void print_usage() {
    printf("=============================================\n");
    printf("🎤 语音识别程序（精准录制版）\n");
    printf("操作说明：\n");
    printf("  1. 按下【回车键】开始录制\n");
    printf("  2. 说话完成后，再次按下【回车键】停止录制并识别\n");
    printf("  3. 录制超过30秒会自动停止\n");
    printf("  4. Ctrl+C 退出程序\n");
    printf("=============================================\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    
    // 1. 初始化 Whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // 4050 显卡
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "❌ 初始化Whisper模型失败\n");
        return 1;
    }

    // 2. 初始化 Miniaudio（仅初始化设备，不立即采集）
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32; // Whisper 需要 float32
    deviceConfig.capture.channels = 1;             // 单声道
    deviceConfig.sampleRate       = 16000;         // Whisper 硬指标 16kHz
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData        = nullptr; // 不再传buffer，用全局变量

    ma_device device;
    if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        fprintf(stderr, "❌ 打开录音设备失败\n");
        whisper_free(ctx);
        return -2;
    }

    // 启动设备（但此时is_recording=false，不会采集数据）
    if (ma_device_start(&device) != MA_SUCCESS) {
        fprintf(stderr, "❌ 启动录音设备失败\n");
        ma_device_uninit(&device);
        whisper_free(ctx);
        return -3;
    }

    print_usage();

    while (true) {
        // 第一步：等待用户按回车开始录制
        printf("\n👉 按下回车键开始录制...\n");
        getchar(); // 等待回车

        // 开始录制
        is_recording.store(true);
        audio_buffer.clear(); // 清空旧数据
        printf("🎙️  正在录制（说话完成后按回车键停止，最长录制30秒）...\n");

        // 等待用户停止录制（按回车）或超时30秒
        std::thread wait_thread([&]() {
            getchar(); // 等待用户按回车停止
            is_recording.store(false);
        });

        // 超时控制（30秒）
        auto start_time = std::chrono::steady_clock::now();
        while (is_recording.load()) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (duration >= 30) {
                printf("⏱️  录制超时（30秒），自动停止\n");
                is_recording.store(false);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 避免CPU空转
        }

        wait_thread.join(); // 等待停止线程结束
        is_recording.store(false); // 确保录制停止

        // 检查录制的数据量
        if (audio_buffer.empty()) {
            printf("⚠️  未采集到任何音频数据，请重新录制\n");
            continue;
        }

        // 第二步：开始识别
        printf("🔍 正在识别...\n");
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = "zh";
        wparams.n_threads = 12;
        wparams.print_progress = false;
        wparams.print_realtime = false;

        if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) != 0) {
            fprintf(stderr, "❌ 识别失败\n");
            continue;
        }

        // 输出识别结果
        const int n_segments = whisper_full_n_segments(ctx);
        if (n_segments == 0) {
            printf("📝: 未识别到有效内容\n");
        } else {
            printf("📝 识别结果：\n");
            for (int i = 0; i < n_segments; ++i) {
                const char* text = whisper_full_get_segment_text(ctx, i);
                printf("   %s\n", text);
            }
        }
    }

    // 清理资源（实际中Ctrl+C会中断，这里是兜底）
    ma_device_uninit(&device);
    whisper_free(ctx);
    return 0;
}

