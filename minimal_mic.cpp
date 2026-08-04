#include "whisper.h"
#include "common.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <vector>
#include <cstdio>
#include <string>

// 音频回调：将采集到的数据存入 buffer
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    std::vector<float>* pBuffer = (std::vector<float>*)pDevice->pUserData;
    const float* pInputFloat = (const float*)pInput;
    if (pInputFloat == NULL) return;

    pBuffer->insert(pBuffer->end(), pInputFloat, pInputFloat + frameCount);
    // 保持 buffer 在最近 10 秒以内，防止内存溢出
    if (pBuffer->size() > 16000 * 10) {
        pBuffer->erase(pBuffer->begin(), pBuffer->begin() + (pBuffer->size() - 16000 * 10));
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    
    // 1. 初始化 Whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // 你的 4050 显卡
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) return 1;

    // 2. 初始化 Miniaudio
    std::vector<float> audio_buffer;
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32; // Whisper 需要 float32
    deviceConfig.capture.channels = 1;             // 单声道
    deviceConfig.sampleRate       = 16000;         // Whisper 硬指标 16kHz
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData        = &audio_buffer;

    ma_device device;
    if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        fprintf(stderr, "Failed to open capture device.\n");
        return -2;
    }

    ma_device_start(&device);
    printf("🎤 录音中... 请说话 (按回车键进行单次识别，Ctrl+C 退出)\n");

    while (true) {
        getchar(); // 等待用户敲回车触发识别
        
        printf("正在识别...\n");
        
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = "zh";
        wparams.n_threads = 12;
        wparams.print_progress = false;

        if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) != 0) {
            fprintf(stderr, "识别失败\n");
            continue;
        }

        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            printf("📝: %s\n", text);
        }
        audio_buffer.clear(); // 清空，准备下一轮
    }

    ma_device_uninit(&device);
    whisper_free(ctx);
    return 0;
}
