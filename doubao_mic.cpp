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

// 全局原子变量（线程安全）
std::atomic<bool> is_recording(false);
std::atomic<bool> exit_program(false);
std::atomic<bool> is_stopping(false); // 新增：标记是否正在停止
std::atomic<int> recorded_seconds(0);
// 音频缓冲区（加锁保护）
std::vector<float> audio_buffer;
std::mutex buffer_mutex;
// 配置常量
const int RECORD_TIMEOUT = 30; // 超时时间（秒）
const int STOP_WAIT_MS = 2000; // 停止前等待2秒

// 信号处理：Ctrl+C 优雅退出
void signal_handler(int sig) {
    if (sig == SIGINT) {
        printf("\n\n🛑 收到退出信号，正在清理资源...\n");
        exit_program.store(true);
        is_recording.store(false);
        is_stopping.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        exit(0);
    }
}

// 非阻塞检查输入（核心：永不阻塞）
bool check_input_non_blocking(int timeout_ms = 50) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = timeout_ms * 1000;
    
    // 处理EINTR错误（避免信号中断导致的异常）
    int ret;
    do {
        ret = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
    } while (ret == -1 && errno == EINTR);
    
    return ret > 0;
}

// 清空输入缓冲区（非阻塞）
void clear_input_buffer() {
    while (check_input_non_blocking(10)) {
        char c;
        ssize_t ret = read(STDIN_FILENO, &c, 1);
        (void)ret;
    }
}

// 音频回调（仅写数据，无额外逻辑）
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    // 只有在真正停止录制标记为 false 时才退出
    if (!is_recording.load() || pInput == NULL) return;

    const float* pInputFloat = (const float*)pInput;
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    // 持续写入，直到 is_recording 被主线程关掉
    audio_buffer.insert(audio_buffer.end(), pInputFloat, pInputFloat + frameCount);
    recorded_seconds.store(static_cast<int>(audio_buffer.size() / 16000.0));
}

// 静音检测（仅裁剪开头）
int trim_silence(const float* audio_data, int audio_len, float threshold = 0.001f) {
    int start = 0;
    while (start < audio_len && fabs(audio_data[start]) < threshold) {
        start++;
    }
    return std::max(audio_len - start, 16000);
}

// 列出系统音频设备
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
        printf("   声道数: 1 | 采样率: 16000 Hz\n");
        printf("---------------------------------------------\n");
    }
    printf("=============================================\n");
}

// 异步停止录制函数（无阻塞，避免死锁）
void async_stop_recording(const char* stop_type) {
    is_stopping.store(true); // 标记正在停止，阻止新数据写入
    printf("\n%s 正在等待最后音频数据写入（2秒）...", stop_type);
    fflush(stdout);
    
    // 独立线程等待，不阻塞主线程
    std::thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(STOP_WAIT_MS));
        is_recording.store(false);
        is_stopping.store(false);
        printf("完成\n");
        fflush(stdout);
    }).detach(); // 分离线程，自动回收资源
}

// 提示信息
void print_usage() {
    printf("=============================================\n");
    printf("🎤 语音识别程序（无死锁+统一逻辑版）\n");
    printf("操作说明：\n");
    printf("  1. 按下【回车键】开始录制\n");
    printf("  2. 说话完成后按回车停止（异步等2秒收尾）\n");
    printf("  3. 录制超过30秒自动停止（同样异步等2秒）\n");
    printf("  4. 录制中实时显示时长\n");
    printf("  5. Ctrl+C 退出程序\n");
    printf("=============================================\n");
}

// CPU优化提示
void print_cpu_optimize_tips() {
    printf("⚡ CPU优化配置说明：\n");
    printf("   ✅ 异步停止逻辑，彻底解决死锁\n");
    printf("   ✅ 手动/超时停止共用同一套逻辑\n");
    printf("   ✅ 非阻塞输入检查，永不卡死\n");
    printf("   📌 模型优化：推荐使用 ggml-medium-q4_0.bin\n");
    printf("=============================================\n");
}

// 核心识别函数
void recognize_audio(struct whisper_context* ctx, const std::vector<float>& audio_data) {
    if (audio_data.empty()) {
        printf("⚠️  未采集到音频数据，跳过识别\n");
        return;
    }

    int valid_len = trim_silence(audio_data.data(), audio_data.size());
    float valid_seconds = (float)valid_len / 16000;
    printf("🔍 正在识别（有效音频长度：%.2f秒，原始：%.2f秒）...\n", 
           valid_seconds, (float)audio_data.size() / 16000);
    
    auto recognize_start = std::chrono::steady_clock::now();
    
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "zh";
    wparams.n_threads = std::max(2, (int)std::thread::hardware_concurrency());
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.temperature = 0.0;
    wparams.max_len = 0;
    wparams.translate = false;
    wparams.no_context = true;
    wparams.single_segment = true;
    wparams.print_special = false;
    wparams.token_timestamps = false;

    if (whisper_full(ctx, wparams, audio_data.data(), valid_len) != 0) {
        fprintf(stderr, "❌ 识别失败\n");
        return;
    }

    auto recognize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - recognize_start).count();
    float speed = valid_seconds / (recognize_duration / 1000.0);
    printf("⏱️  识别耗时：%.2f 秒 | 识别速度：%.2fx实时速度\n", 
           recognize_duration / 1000.0, speed);
    
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

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];

    // 1. 初始化音频上下文
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
        clear_input_buffer();
    }

    // 4. 初始化 Whisper 模型
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    printf("\n🚀 正在加载模型：%s\n", model_path);
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "❌ 初始化Whisper模型失败\n");
        ma_context_uninit(&context);
        return 1;
    }

    print_cpu_optimize_tips();
    printf("✅ 模型加载成功！\n");

    // 5. 初始化录音设备
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate       = 16000;
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData        = NULL;

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
        printf("\n👉 按下回车键开始录制...\n");
        
        // 阻塞等待用户回车（非阻塞检查，避免死锁）
        char input_char = 0;
        while (!check_input_non_blocking() && !exit_program.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (exit_program.load()) break;
        
			ssize_t ret1 = read(STDIN_FILENO, &input_char, 1);
			(void)ret1;
			clear_input_buffer();

        if (exit_program.load()) break;
        if (input_char != '\n') {
            printf("⚠️  请按回车键触发录制！\n");
            continue;
        }

        // 重置录制状态
        is_recording.store(true);
        is_stopping.store(false);
        recorded_seconds.store(0);
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            audio_buffer.clear();
        }
        printf("🎙️  正在录制（按回车停止，最长30秒）...\n");

        // 录制时长实时显示线程
        std::thread progress_thread([&]() {
            while (is_recording.load() && !exit_program.load() && !is_stopping.load()) {
                printf("\r📊 录制中... %d秒", recorded_seconds.load());
                fflush(stdout);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });

        bool stopped = false;
        auto start_time = std::chrono::steady_clock::now();

        // 核心循环 - 无死锁逻辑
        while (!exit_program.load() && !stopped) {
            // 1. 检查手动停止（非阻塞）
           if (check_input_non_blocking(50)) {
				char c;
				read(STDIN_FILENO, &c, 1);
				if (c == '\n') {
					printf("\n🛑 手动停止，正在收尾音频数据...");
					// 关键：先睡 500ms 捕获回车瞬间的余音，但不开启新线程
					std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
					is_recording.store(false); // 此时回调停止写入
					stopped = true;
				}
			}


            // 2. 检查超时停止（非阻塞）
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
 			// 超时判断同理
			if (duration >= RECORD_TIMEOUT) {
				printf("\n⏱️  超时停止，正在收尾...");
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				is_recording.store(false);
				stopped = true;
			}

            // 3. 检查是否已停止
            if (!is_recording.load()) {
                stopped = true;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // 等待进度线程退出
        progress_thread.join();

        if (exit_program.load()) break;

        // 等待异步停止线程完成（确保数据写完）
        while (is_stopping.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 拷贝音频数据
        std::vector<float> captured_audio;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            captured_audio = audio_buffer;
        }

        // 执行识别
        recognize_audio(ctx, captured_audio);
    }

    // 清理资源
    ma_device_uninit(&device);
    ma_context_uninit(&context);
    whisper_free(ctx);
    printf("✅ 资源清理完成，程序退出\n");
    return 0;
}
