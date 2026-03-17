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
#include <mutex>  // 关键：补充缺失的mutex头文件

// 全局原子变量（线程安全）
std::atomic<bool> is_recording(false);
std::atomic<bool> exit_program(false);
// 音频缓冲区（加锁保护，避免多线程冲突）
std::vector<float> audio_buffer;
std::mutex buffer_mutex;  // 现在有头文件支持，不会报错

// 信号处理：Ctrl+C 优雅退出
void signal_handler(int sig) {
    if (sig == SIGINT) {
        printf("\n\n🛑 收到退出信号，正在清理资源...\n");
        exit_program.store(true);
        is_recording.store(false);
        exit(0);
    }
}

// 音频回调（旧版 miniaudio 兼容）
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (!is_recording.load() || pInput == NULL) return;

    const float* pInputFloat = (const float*)pInput;
    if (pInputFloat == NULL) return;

    // 加锁操作缓冲区（避免主线程/回调线程冲突）
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    // 限制最大录制时长 30 秒（16000Hz）
    const size_t max_frames = 16000 * 30;
    const size_t available = max_frames - audio_buffer.size();
    if (available == 0) {
        is_recording.store(false);
        return;
    }

    const size_t copy_frames = (frameCount > available) ? available : frameCount;
    audio_buffer.insert(audio_buffer.end(), pInputFloat, pInputFloat + copy_frames);
}

// 列出系统音频设备（兼容旧版 API）
void list_audio_devices(ma_context& context, ma_device_info** pCaptureInfos, ma_uint32& captureCount) {
    printf("\n📜 系统可用麦克风设备列表：\n");
    printf("=============================================\n");

    ma_result result = ma_context_get_devices(&context, NULL, NULL, pCaptureInfos, &captureCount);
    if (result != MA_SUCCESS) {
        fprintf(stderr, "❌ 获取设备列表失败，使用默认设备\n");
        *pCaptureInfos = NULL;
        captureCount = 0;
        return;
    }

    for (ma_uint32 i = 0; i < captureCount; ++i) {
        printf("🔧 设备ID: %u | 名称: %s\n", i, (*pCaptureInfos)[i].name);
        printf("   声道数: 1 | 采样率: 16000 Hz\n"); // 固定 16000Hz 避免采样率冲突
        printf("---------------------------------------------\n");
    }
    printf("=============================================\n");
}

// 提示信息
void print_usage() {
    printf("=============================================\n");
    printf("🎤 语音识别程序（旧版兼容）\n");
    printf("操作说明：\n");
    printf("  1. 按下【回车键】开始录制\n");
    printf("  2. 说话完成后按回车停止录制并识别\n");
    printf("  3. 录制超过30秒自动停止\n");
    printf("  4. Ctrl+C 退出程序\n");
    printf("=============================================\n");
}

// GPU 状态提示（兼容旧版）
void check_gpu_status() {
    printf("🔍 GPU加速配置说明：\n");
    printf("   ❌ 若识别速度慢，说明使用CPU运行\n");
    printf("   ✅ 启用GPU：重新编译whisper.cpp时添加 -DWHISPER_CUDA=ON\n");
}

int main(int argc, char** argv) {
    // 注册信号处理
    signal(SIGINT, signal_handler);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];

    // 1. 初始化音频上下文（旧版兼容）
    ma_context context;
    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        fprintf(stderr, "❌ 初始化音频上下文失败\n");
        return 1;
    }

    // 2. 枚举麦克风设备
    ma_device_info* pCaptureInfos = NULL;
    ma_uint32 captureCount = 0;
    list_audio_devices(context, &pCaptureInfos, captureCount);

    // 3. 选择麦克风设备
    ma_uint32 device_id = 0;
    if (captureCount > 0) {
        printf("\n👉 请输入要使用的麦克风设备ID：");
        if (scanf("%u", &device_id) != 1 || device_id >= captureCount) {
            fprintf(stderr, "❌ 输入无效，使用默认设备ID 0\n");
            device_id = 0;
        }
        // 清空输入缓冲区
        while (getchar() != '\n');
    }

    // 4. 初始化 Whisper 模型
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;

    printf("\n🚀 正在加载模型：%s\n", model_path);
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "❌ 初始化Whisper模型失败\n");
        ma_context_uninit(&context);
        return 1;
    }

    // GPU 状态提示
    check_gpu_status();
    printf("✅ 模型加载成功！\n");

    // 5. 初始化录音设备（旧版 miniaudio 核心兼容）
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32;    // Whisper 要求 float32
    deviceConfig.capture.channels = 1;                // 单声道
    deviceConfig.sampleRate       = 16000;            // 固定 16000Hz 避免采样率错误
    deviceConfig.dataCallback     = data_callback;    // 回调函数
    deviceConfig.pUserData        = NULL;

    // 指定选中的麦克风设备（旧版用 pDeviceID）
    if (captureCount > 0 && pCaptureInfos != NULL) {
        deviceConfig.capture.pDeviceID = &pCaptureInfos[device_id].id;
        printf("\n✅ 已选择麦克风：%s\n", pCaptureInfos[device_id].name);
    } else {
        printf("\n✅ 使用默认麦克风设备\n");
    }

    ma_device device;
    if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
        fprintf(stderr, "❌ 打开录音设备失败\n");
        whisper_free(ctx);
        ma_context_uninit(&context);
        return 1;
    }

    // 启动录音设备（仅初始化，不采集数据）
    if (ma_device_start(&device) != MA_SUCCESS) {
        fprintf(stderr, "❌ 启动录音设备失败\n");
        ma_device_uninit(&device);
        whisper_free(ctx);
        ma_context_uninit(&context);
        return 1;
    }

    print_usage();

    // 主循环
    while (!exit_program.load()) {
        // 等待用户按回车开始录制
        printf("\n👉 按下回车键开始录制...\n");
        getchar();

        if (exit_program.load()) break;

        // 重置录制状态
        is_recording.store(true);
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            audio_buffer.clear();
        }
        printf("🎙️  正在录制（按回车停止，最长30秒）...\n");

        // 等待用户停止录制（子线程监听回车）
        std::thread wait_thread([&]() {
            getchar();
            is_recording.store(false);
        });

        // 超时控制（30秒）
        auto start_time = std::chrono::steady_clock::now();
        while (is_recording.load() && !exit_program.load()) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            
            if (duration >= 30) {
                printf("⏱️  录制超时，自动停止\n");
                is_recording.store(false);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        wait_thread.join();
        is_recording.store(false);

        if (exit_program.load()) break;

        // 检查录制数据
        std::vector<float> captured_audio;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            captured_audio = audio_buffer; // 拷贝数据避免锁冲突
        }

        if (captured_audio.empty()) {
            printf("⚠️  未采集到音频数据，请重新录制\n");
            continue;
        }

        // 开始识别
        printf("🔍 正在识别（音频长度：%.2f秒）...\n", (float)captured_audio.size() / 16000);
        auto recognize_start = std::chrono::steady_clock::now();
        
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = "zh";
        wparams.n_threads = std::max(1, (int)std::thread::hardware_concurrency());
        wparams.print_progress = false;
        wparams.print_realtime = false;
        wparams.temperature = 0.0;
        wparams.max_len = 0;
        wparams.translate = false;
        wparams.no_context = true;

        if (whisper_full(ctx, wparams, captured_audio.data(), captured_audio.size()) != 0) {
            fprintf(stderr, "❌ 识别失败\n");
            continue;
        }

        // 输出识别结果
        auto recognize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - recognize_start).count();
        printf("⏱️  识别耗时：%.2f 秒\n", recognize_duration / 1000.0);
        
        const int n_segments = whisper_full_n_segments(ctx);
        if (n_segments == 0) {
            printf("📝 未识别到有效内容\n");
        } else {
            printf("📝 识别结果：\n");
            for (int i = 0; i < n_segments; ++i) {
                const char* text = whisper_full_get_segment_text(ctx, i);
                printf("   %s\n", text);
            }
        }
    }

    // 清理资源
    ma_device_uninit(&device);
    ma_context_uninit(&context);
    whisper_free(ctx);
    printf("✅ 资源清理完成，程序退出\n");
    return 0;
}
