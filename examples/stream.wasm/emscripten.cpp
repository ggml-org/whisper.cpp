#include "ggml.h"
#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

constexpr int N_THREAD = 8;

std::vector<struct whisper_context *> g_contexts(4, nullptr);

std::mutex g_mutex;
std::thread g_worker;

std::atomic<bool> g_running(false);

std::string g_status        = "";
std::string g_status_forced = "";
std::string g_transcribed   = "";

std::vector<float> g_pcmf32;

void stream_set_status(const std::string & status) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_status = status;
}

void stream_main(size_t index, const std::string & lang) {
    stream_set_status("loading data ...");

    struct whisper_full_params wparams = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
    bool is_multilingual = whisper_is_multilingual(g_contexts[index]);

    wparams.n_threads        = std::min(N_THREAD, (int) std::thread::hardware_concurrency());
    wparams.offset_ms        = 0;
    wparams.translate        = false;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;

    wparams.max_tokens       = 32;
    wparams.audio_ctx        = 768; // partial encoder context for better performance

    // disable temperature fallback
    wparams.temperature_inc  = -1.0f;

    wparams.language         = is_multilingual ? lang.c_str() : "en";

    printf("stream: using %d threads\n", wparams.n_threads);

    std::vector<float> pcmf32;

    // whisper context
    auto & ctx = g_contexts[index];

    // 5 seconds interval
    const int64_t window_samples = 5*WHISPER_SAMPLE_RATE;

    while (g_running) {
        stream_set_status("waiting for audio ...");

        {
            std::unique_lock<std::mutex> lock(g_mutex);

            if (g_pcmf32.size() < 1024) {
                lock.unlock();

                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                continue;
            }

            pcmf32 = std::vector<float>(g_pcmf32.end() - std::min((int64_t) g_pcmf32.size(), window_samples), g_pcmf32.end());
            g_pcmf32.clear();
        }

        {
            const auto t_start = std::chrono::high_resolution_clock::now();

            stream_set_status("running whisper ...");

            int ret = whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size());
            if (ret != 0) {
                printf("whisper_full() failed: %d\n", ret);
                break;
            }

            const auto t_end = std::chrono::high_resolution_clock::now();

            printf("stream: whisper_full() returned %d in %f seconds\n", ret, std::chrono::duration<double>(t_end - t_start).count());
        }

        {
            std::string text_heard;

            {
                const int n_segments = whisper_full_n_segments(ctx);
                if (n_segments > 0) {
                    const char * text = whisper_full_get_segment_text(ctx, n_segments - 1);

                    const int64_t t0 = whisper_full_get_segment_t0(ctx, n_segments - 1);
                    const int64_t t1 = whisper_full_get_segment_t1(ctx, n_segments - 1);

                    printf("transcribed: %s\n", text);

                    text_heard += text;
                }
            }

            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_transcribed = text_heard;
            }
        }
    }

    if (index < g_contexts.size()) {
        whisper_free(g_contexts[index]);
        g_contexts[index] = nullptr;
    }
}
EMSCRIPTEN_BINDINGS(stream) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model, const std::string & lang) {
        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_contexts[i] != nullptr) {
                    g_running = true;
                    if (g_worker.joinable()) {
                        g_worker.join();
                    }
                    g_worker = std::thread([i, lang]() {
                        stream_main(i, lang);
                    });

                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        if (g_running) {
            g_running = false;
        }
    }));

    emscripten::function("set_audio", emscripten::optional_override([](size_t index, const emscripten::val & audio) {
        --index;

        if (index >= g_contexts.size()) {
            return -1;
        }

        if (g_contexts[index] == nullptr) {
            return -2;
        }

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            const int n = audio["length"].as<int>();

            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];

            g_pcmf32.resize(n);

            emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(g_pcmf32.data()), n);
            memoryView.call<void>("set", audio);
        }

        return 0;
    }));

    emscripten::function("get_transcribed", emscripten::optional_override([]() {
        std::string transcribed;

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            transcribed = std::move(g_transcribed);
        }

        return transcribed;
    }));

    emscripten::function("get_status", emscripten::optional_override([]() {
        std::string status;

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            status = g_status_forced.empty() ? g_status : g_status_forced;
        }

        return status;
    }));

    emscripten::function("set_status", emscripten::optional_override([](const std::string & status) {
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_status_forced = status;
        }
    }));
}
