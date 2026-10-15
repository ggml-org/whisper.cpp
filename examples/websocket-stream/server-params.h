#ifndef SERVER_PARAMS_H
#define SERVER_PARAMS_H
#include <thread>
struct ServerParams {
    int32_t port = 9002;
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t audio_ctx = 0;
    int32_t beam_size = -1;

    float vad_thold = 0.6f;

    bool translate = false;
    bool print_special = false;
    bool no_timestamps = true;
    bool tinydiarize = false;
    bool use_gpu = true;
    bool flash_attn = true;

    std::string language = "en";
    std::string model = "ggml-large-v3-turbo.bin";
    std::string host = "0.0.0.0";
};
#endif
