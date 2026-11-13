// whisper-stream.h 
#pragma once
#include <chrono>
#include <vector>
#include <string>
#include "whisper.h"
#include <thread>

struct StreamParams {
    int n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    int step_ms = 3000;
    int length_ms = 10000;
    int keep_ms = 200;
    int max_tokens = 32;
    int audio_ctx = 0;
    int beam_size = -1;
    float vad_thold = 0.6f;
    float freq_thold = 100.0f;
    bool translate = false;
    bool no_fallback = false;
    bool print_special = false;
    bool no_context = true;
    bool no_timestamps = false;
    bool tinydiarize = false;
    bool save_audio = false;
    bool use_gpu = true;
    bool flash_attn = false;
    std::string language = "en";
    std::string model;
};

struct TranscriptionResult {
    std::string text;
    bool final;
};

class WhisperStream {
public:
    WhisperStream(const StreamParams &stream_params);
    ~WhisperStream();

    bool init();
    TranscriptionResult process(const std::vector<float> &pcmf32_chunk);
    void free(); // optional explicit free

private:
    StreamParams params;

    // whisper context
    struct whisper_context *ctx = nullptr;

    // buffers (samples, not bytes)
    std::vector<float> pcmf32;       // assembled input for inference
    std::vector<float> pcmf32_new;   // appended incoming samples buffer
    std::vector<float> pcmf32_old;   // overlap kept for next chunk

    std::vector<whisper_token> prompt_tokens;

    // sample counts and flags 
    int n_samples_step = 0;
    int n_samples_len  = 0;
    int n_samples_keep = 0;
    int n_samples_30s  = 0;
    bool use_vad = false;
    int n_new_line = 1;
    int n_iter = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_last;
};
