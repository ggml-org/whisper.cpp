#include "whisper.h"
#include <portaudio.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include <string>

// ====================== 1. 枚举并选择麦克风设备（纯PortAudio原生实现） ======================
void enumerate_audio_devices() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "❌ PortAudio初始化失败: %s\n", Pa_GetErrorText(err));
        return;
    }

    int numDevices = Pa_GetDeviceCount();
    printf("\n📜 系统可用麦克风设备列表：\n");
    printf("=============================================\n");
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* pInfo = Pa_GetDeviceInfo(i);
        // 只显示输入设备（麦克风，至少1个输入声道）
        if (pInfo->maxInputChannels > 0) {
            printf("🔧 设备ID: %d | 名称: %s\n", i, pInfo->name);
            printf("   最大输入声道: %d | 默认采样率: %.1f Hz\n", 
                   pInfo->maxInputChannels, pInfo->defaultSampleRate);
            printf("---------------------------------------------\n");
        }
    }
    printf("=============================================\n\n");

    Pa_Terminate();
}

int select_mic_device() {
    int selected_id = -1;
    printf("👉 请输入你要使用的麦克风设备ID（比如苹果耳机对应的ID）：");
    std::cin >> selected_id;

    // 验证设备ID有效性
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "❌ PortAudio初始化失败: %s\n", Pa_GetErrorText(err));
        return -1;
    }

    int numDevices = Pa_GetDeviceCount();
    if (selected_id < 0 || selected_id >= numDevices) {
        fprintf(stderr, "❌ 设备ID无效！请输入列表中的有效ID\n");
        Pa_Terminate();
        return -1;
    }

    const PaDeviceInfo* pInfo = Pa_GetDeviceInfo(selected_id);
    if (pInfo->maxInputChannels == 0) {
        fprintf(stderr, "❌ 选择的设备不是麦克风（无输入声道）！\n");
        Pa_Terminate();
        return -1;
    }

    printf("\n✅ 已选择麦克风：\n");
    printf("   ID: %d | 名称: %s\n", selected_id, pInfo->name);
    printf("   采样率: %.1f Hz | 声道数: %d\n\n", 
           pInfo->defaultSampleRate, pInfo->maxInputChannels);
    
    Pa_Terminate();
    return selected_id;
}

// ====================== 2. 音频采集函数（纯PortAudio原生实现） ======================
int audio_record(short* buffer, int buffer_size, int sample_rate, int channels, int max_seconds, int device_id) {
    PaError err;
    PaStream* stream;
    PaStreamParameters input_params;

    // 初始化PortAudio
    err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "❌ PortAudio初始化失败: %s\n", Pa_GetErrorText(err));
        return -1;
    }

    // 配置输入参数（指定麦克风设备ID）
    input_params.device = device_id;
    input_params.channelCount = channels;
    input_params.sampleFormat = paInt16; // 16位深（Whisper要求）
    input_params.suggestedLatency = Pa_GetDeviceInfo(device_id)->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = NULL;

    // 打开音频流
    err = Pa_OpenStream(
        &stream,
        &input_params,
        NULL, // 无输出
        sample_rate,
        1024, // 缓冲区大小
        paClipOff, // 关闭裁剪
        NULL, // 无回调
        NULL
    );

    if (err != paNoError) {
        fprintf(stderr, "❌ 打开音频流失败: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        return -1;
    }

    // 开始录制
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "❌ 开始录制失败: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        return -1;
    }

    printf("🎙️  录制中（按回车键停止，最长%d秒）...\n", max_seconds);
    int total_samples = 0;
    time_t start_time = time(NULL);

    // 录制逻辑：要么按回车停止，要么超时停止
    while (1) {
        // 读取音频数据
        int samples_to_read = buffer_size - total_samples;
        if (samples_to_read <= 0) break;

        err = Pa_ReadStream(stream, buffer + total_samples, 1024);
        if (err != paNoError) {
            fprintf(stderr, "❌ 读取音频失败: %s\n", Pa_GetErrorText(err));
            break;
        }

        total_samples += 1024;

        // 超时检查（max_seconds秒）
        if (difftime(time(NULL), start_time) >= max_seconds) {
            printf("\n⏰ 录制超时（%d秒），自动停止\n", max_seconds);
            break;
        }

        // 检查是否按了回车
        if (std::cin.rdbuf()->in_avail() > 0) {
            getchar();
            printf("\n🛑 用户停止录制\n");
            break;
        }
    }

    // 停止录制
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    return total_samples;
}

// ====================== 3. 新增：short转float（Whisper要求） ======================
void convert_short_to_float(const short* src, float* dst, int count) {
    // 16位short的范围是[-32768, 32767]，归一化到float的[-1.0, 1.0]
    for (int i = 0; i < count; i++) {
        dst[i] = static_cast<float>(src[i]) / 32768.0f;
    }
}

// ====================== 4. 主函数（修正数据类型转换） ======================
int main(int argc, char **argv) {
    // 检查参数
    if (argc < 2) {
        fprintf(stderr, "用法: %s 模型文件路径（如 ./models/ggml-medium.bin）\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];

    // 步骤1：枚举并选择麦克风
    enumerate_audio_devices();
    int mic_device_id = select_mic_device();
    if (mic_device_id < 0) {
        fprintf(stderr, "❌ 麦克风选择失败，程序退出\n");
        return 1;
    }

    // 步骤2：GPU加速配置说明
    printf("\n🔍 GPU加速配置说明...\n");
    printf("   当前已启用GPU加速（use_gpu = true）\n");
    printf("   ✅ 如果编译时链接了CUDA库，模型会自动使用GPU\n");
    printf("   ❌ 如果识别速度很慢，说明实际使用CPU运行\n");
    printf("   验证方法：观察识别耗时，GPU版本比CPU快5-10倍\n\n");

    // 步骤3：加载Whisper模型（启用GPU）
    printf("🚀 正在加载模型：%s\n", model_path);
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    cparams.gpu_device = 0;

    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "❌ 加载模型失败: %s\n", model_path);
        return 1;
    }

    // 打印模型信息
    whisper_print_system_info();
    printf("✅ 模型加载成功！\n");
    printf("   📌 若识别速度快（几秒内完成）= GPU运行\n");
    printf("   📌 若识别速度慢（十几秒/分钟）= CPU运行\n");
    printf("=============================================\n");
    printf("🎤 语音识别程序（指定麦克风版）\n");
    printf("操作说明：\n");
    printf("  1. 按下【回车键】开始录制\n");
    printf("  2. 说话完成后，再次按下【回车键】停止录制并识别\n");
    printf("  3. 录制超过30秒会自动停止\n");
    printf("  4. Ctrl+C 退出程序\n");
    printf("=============================================\n\n");

    // 步骤4：准备音频缓冲区
    const int sample_rate = 16000; // Whisper标准采样率
    const int channels = 1;        // 单声道
    const int max_seconds = 30;    // 最长录制30秒
    const int buffer_size = sample_rate * channels * max_seconds;
    
    // 原始音频缓冲区（short类型）
    short* buffer_short = (short*)malloc(buffer_size * sizeof(short));
    // Whisper输入缓冲区（float类型）
    float* buffer_float = (float*)malloc(buffer_size * sizeof(float));
    
    if (!buffer_short || !buffer_float) {
        fprintf(stderr, "❌ 分配音频缓冲区失败\n");
        free(buffer_short);
        free(buffer_float);
        whisper_free(ctx);
        return 1;
    }

    // 步骤5：等待用户开始录制
    printf("👉 按下回车键开始录制...\n");
    getchar();

    // 步骤6：录制音频（指定选择的麦克风）
    int samples_read = audio_record(buffer_short, buffer_size, sample_rate, channels, max_seconds, mic_device_id);
    if (samples_read <= 0) {
        fprintf(stderr, "❌ 录制音频失败\n");
        free(buffer_short);
        free(buffer_float);
        whisper_free(ctx);
        return 1;
    }

    // 步骤7：关键修正：short转float（Whisper要求）
    convert_short_to_float(buffer_short, buffer_float, samples_read);

    // 步骤8：语音识别（传入float缓冲区）
    printf("\n🔍 正在识别...\n");
    clock_t start = clock();

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "zh";       // 中文识别
    wparams.translate = false;
    wparams.print_special = false;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;

    // 传入float类型的buffer_float，而非short类型的buffer_short
    if (whisper_full(ctx, wparams, buffer_float, samples_read) != 0) {
        fprintf(stderr, "❌ 识别音频失败\n");
        free(buffer_short);
        free(buffer_float);
        whisper_free(ctx);
        return 1;
    }

    // 步骤9：输出结果
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️  识别耗时：%.2f 秒\n", elapsed);
    
    if (elapsed < 5.0) {
        printf("   🎯 识别速度快，应该是GPU在运行！\n");
    } else {
        printf("   ⚠️  识别速度慢，当前使用CPU运行（需编译CUDA版本）\n");
    }

    printf("📝 识别结果：\n   ");
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        printf("%s\n   ", text);
    }
    printf("\n");

    // 步骤10：清理资源
    free(buffer_short);
    free(buffer_float);
    whisper_free(ctx);

    return 0;
}
