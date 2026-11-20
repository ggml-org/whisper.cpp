#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

std::thread g_worker;

std::vector<struct whisper_context *> g_contexts(4, nullptr);

// the transcription runs on g_worker while JS reads it from the main thread
struct token {
    std::string text;
    float       p;
};

struct segment {
    int64_t t0;
    int64_t t1;

    std::string text;

    std::vector<token> tokens;
};

std::mutex g_mutex;
std::vector<std::vector<segment>> g_transcripts(4);

static inline int mpow2(int n) {
    int p = 1;
    while (p <= n) p *= 2;
    return p/2;
}

// called on the worker thread each time whisper decodes new segments
static void cb_new_segment(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const int index = (int) (intptr_t) user_data;

    const int n_segments = whisper_full_n_segments(ctx);

    for (int i = n_segments - n_new; i < n_segments; ++i) {
        segment s;

        s.t0   = whisper_full_get_segment_t0(ctx, i);
        s.t1   = whisper_full_get_segment_t1(ctx, i);
        s.text = whisper_full_get_segment_text(ctx, i);

        const int n_tokens = whisper_full_n_tokens(ctx, i);

        s.tokens.reserve(n_tokens);
        for (int j = 0; j < n_tokens; ++j) {
            // special tokens ([_BEG_], timestamps) are not part of the text, same test cli.cpp uses
            if (whisper_full_get_token_id(ctx, i, j) >= whisper_token_eot(ctx)) {
                continue;
            }

            s.tokens.push_back({ whisper_full_get_token_text(ctx, i, j), whisper_full_get_token_p(ctx, i, j) });
        }

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_transcripts[index].push_back(std::move(s));
        }

        // the page renders the segment by pulling it back through the getters below
        MAIN_THREAD_EM_ASM({
            if (typeof onNewSegment === 'function') {
                onNewSegment($0, $1);
            }
        }, index + 1, i);
    }
}

EMSCRIPTEN_BINDINGS(whisper) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_contexts[i] != nullptr) {
                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index < g_contexts.size()) {
            whisper_free(g_contexts[index]);
            g_contexts[index] = nullptr;

            std::lock_guard<std::mutex> lock(g_mutex);
            g_transcripts[index].clear();
        }
    }));

    emscripten::function("full_default", emscripten::optional_override([](size_t index, const emscripten::val & audio, const std::string & lang, int nthreads, bool translate) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index >= g_contexts.size()) {
            return -1;
        }

        if (g_contexts[index] == nullptr) {
            return -2;
        }

        struct whisper_full_params params = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        bool is_multilingual = whisper_is_multilingual(g_contexts[index]);

        // the transcript reaches the page through cb_new_segment, so printing it is redundant
        params.print_realtime   = false;
        params.print_progress   = false;
        params.print_timestamps = true;
        params.print_special    = false;
        params.translate        = translate;
        params.language         = is_multilingual ? strdup(lang.c_str()) : "en";
        params.n_threads        = std::min(nthreads, std::min(16, mpow2(std::thread::hardware_concurrency())));
        params.offset_ms        = 0;

        params.new_segment_callback           = cb_new_segment;
        params.new_segment_callback_user_data = (void *) (intptr_t) index;

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_transcripts[index].clear();
        }

        std::vector<float> pcmf32;
        const int n = audio["length"].as<int>();

        emscripten::val heap = emscripten::val::module_property("HEAPU8");
        emscripten::val memory = heap["buffer"];

        pcmf32.resize(n);

        emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
        memoryView.call<void>("set", audio);

        // print system information
        {
            printf("system_info: n_threads = %d / %d | %s\n",
                    params.n_threads, std::thread::hardware_concurrency(), whisper_print_system_info());

            printf("%s: processing %d samples, %.1f sec, %d threads, %d processors, lang = %s, task = %s ...\n",
                    __func__, int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                    params.n_threads, 1,
                    params.language,
                    params.translate ? "translate" : "transcribe");

            printf("\n");
        }

        // run the worker
        {
            g_worker = std::thread([index, params, pcmf32 = std::move(pcmf32), is_multilingual]() {
                whisper_reset_timings(g_contexts[index]);
                const int ret = whisper_full(g_contexts[index], params, pcmf32.data(), pcmf32.size());
                whisper_print_timings(g_contexts[index]);
                if (is_multilingual) {
                    free((void*)params.language);
                }

                // full_default returns before any of this has happened, so the page
                // cannot otherwise tell a finished transcription from a stalled one
                MAIN_THREAD_EM_ASM({
                    if (typeof onTranscriptionDone === 'function') {
                        onTranscriptionDone($0, $1);
                    }
                }, index + 1, ret);
            });
        }

        return 0;
    }));

    // the transcript, read back from the page. all of these are safe to call while the
    // worker is still decoding: they only expose segments cb_new_segment has finished with
    emscripten::function("get_n_segments", emscripten::optional_override([](size_t index) {
        --index;

        if (index >= g_transcripts.size()) {
            return -1;
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        return (int) g_transcripts[index].size();
    }));

    emscripten::function("get_segment_t0", emscripten::optional_override([](size_t index, size_t i) {
        --index;

        std::lock_guard<std::mutex> lock(g_mutex);
        if (index >= g_transcripts.size() || i >= g_transcripts[index].size()) {
            return -1;
        }

        // in 10 ms units, as whisper_full_get_segment_t0 returns them
        return (int) g_transcripts[index][i].t0;
    }));

    emscripten::function("get_segment_t1", emscripten::optional_override([](size_t index, size_t i) {
        --index;

        std::lock_guard<std::mutex> lock(g_mutex);
        if (index >= g_transcripts.size() || i >= g_transcripts[index].size()) {
            return -1;
        }

        return (int) g_transcripts[index][i].t1;
    }));

    emscripten::function("get_segment_text", emscripten::optional_override([](size_t index, size_t i) {
        --index;

        std::lock_guard<std::mutex> lock(g_mutex);
        if (index >= g_transcripts.size() || i >= g_transcripts[index].size()) {
            return std::string();
        }

        return g_transcripts[index][i].text;
    }));

    emscripten::function("get_n_tokens", emscripten::optional_override([](size_t index, size_t i) {
        --index;

        std::lock_guard<std::mutex> lock(g_mutex);
        if (index >= g_transcripts.size() || i >= g_transcripts[index].size()) {
            return -1;
        }

        return (int) g_transcripts[index][i].tokens.size();
    }));

    emscripten::function("get_token_text", emscripten::optional_override([](size_t index, size_t i, size_t j) {
        --index;

        std::lock_guard<std::mutex> lock(g_mutex);
        if (index >= g_transcripts.size() || i >= g_transcripts[index].size() || j >= g_transcripts[index][i].tokens.size()) {
            return std::string();
        }

        return g_transcripts[index][i].tokens[j].text;
    }));

    emscripten::function("get_token_p", emscripten::optional_override([](size_t index, size_t i, size_t j) {
        --index;

        std::lock_guard<std::mutex> lock(g_mutex);
        if (index >= g_transcripts.size() || i >= g_transcripts[index].size() || j >= g_transcripts[index][i].tokens.size()) {
            return -1.0f;
        }

        return g_transcripts[index][i].tokens[j].p;
    }));
}
