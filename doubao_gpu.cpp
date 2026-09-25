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

// 适配旧版本的GPU状态提示（不依赖新函数）
void check_gpu_status() {
    printf("🔍 GPU加速配置说明...\n");
    printf("   当前已启用GPU加速（use_gpu = true）\n");
    printf("   ✅ 如果编译时链接了CUDA库，模型会自动使用GPU\n");
    printf("   ❌ 如果识别速度很慢，说明实际使用CPU运行\n");
    printf("   验证方法：观察识别耗时，GPU版本比CPU快5-10倍\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    
    // GPU状态提示（适配旧版本）
    check_gpu_status();
    
    // 1. 初始化 Whisper（仅保留旧版本支持的参数）
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // 启用GPU（旧版本核心参数）
    // 移除use_gpu_fp16和gpu_device（旧版本没有这些字段）
    
    printf("\n🚀 正在加载模型：%s\n", model_path);
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "❌ 初始化Whisper模型失败\n");
        return 1;
    }
    
    // 旧版本没有whisper_is_using_gpu，改用间接提示
    printf("✅ 模型加载成功！\n");
    printf("   📌 若识别速度快（几秒内完成）= GPU运行\n");
    printf("   📌 若识别速度慢（十几秒/分钟）= CPU运行\n");

    // 2. 初始化 Miniaudio（仅初始化设备，不立即采集）
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32; // Whisper 需要 float32
    deviceConfig.capture.channels = 1;             // 单声道
    deviceConfig.sampleRate       = 16000;         // Whisper 硬指标 16kHz
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData        = nullptr;

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

        // 第二步：开始识别（优化识别参数提升精度）
        printf("🔍 正在识别...\n");
        // 记录识别开始时间（用于判断GPU/CPU）
        auto recognize_start = std::chrono::steady_clock::now();
        
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = "zh";
        wparams.n_threads = 12; // 根据CPU核心数调整
        wparams.print_progress = false;
        wparams.print_realtime = false;
        
        // 精度优化参数（旧版本也支持）
        wparams.temperature = 0.0; // 降低随机性，提升稳定性
        wparams.max_len = 0; // 不限制输出长度
        wparams.translate = false; // 不翻译，直接识别
        wparams.no_context = true; // 不使用上下文，避免干扰

        if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) != 0) {
            fprintf(stderr, "❌ 识别失败\n");
            continue;
        }

        // 计算识别耗时（判断GPU/CPU）
        auto recognize_end = std::chrono::steady_clock::now();
        auto recognize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(recognize_end - recognize_start).count();
        printf("⏱️  识别耗时：%.2f 秒\n", recognize_duration / 1000.0);
        if (recognize_duration < 5000) {
            printf("   🎯 识别速度快，应该是GPU在运行！\n");
        } else {
            printf("   ⚠️  识别速度慢，可能是CPU在运行！\n");
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

    // 清理资源
    ma_device_uninit(&device);
    whisper_free(ctx);
    return 0;
}
