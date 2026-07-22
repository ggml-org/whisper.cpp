#include <WChess.h>
#include <emscripten.h>
#include <emscripten/bind.h>

#include <thread>

constexpr int N_THREAD = 8;

std::vector<struct whisper_context *> g_contexts(4, nullptr);

std::mutex  g_mutex;
std::thread g_worker;

std::condition_variable g_cv;

bool g_running(false);
std::vector<float> g_pcmf32;

void set_move(const std::string & move, float prob) {
    MAIN_THREAD_EM_ASM({
        setMove(UTF8ToString($0), $1)
    }, move.c_str(), prob);
}

void set_grammar(const std::string & grammar) {
    MAIN_THREAD_EM_ASM({
        setGrammar(UTF8ToString($0))
    }, grammar.c_str());
}

bool get_audio(std::vector<float> & audio) {
    std::unique_lock<std::mutex> lock(g_mutex);
    g_cv.wait(lock, [] { return !g_running || !g_pcmf32.empty(); });
    if (!g_running) return false;
    audio = std::move(g_pcmf32);
    return true;
}

void wchess_main(size_t i) {
    struct whisper_full_params wparams = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

    wparams.n_threads        = std::min(N_THREAD, (int) std::thread::hardware_concurrency());
    wparams.offset_ms        = 0;
    wparams.translate        = false;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;
    wparams.no_timestamps    = true;

    wparams.max_tokens       = 32;
    wparams.audio_ctx        = 1280; // partial encoder context for better performance

    wparams.temperature      = 0.0f;
    wparams.temperature_inc  = 2.0f;
    wparams.greedy.best_of   = 1;

    wparams.beam_search.beam_size = 1;

    wparams.language         = "en";

    wparams.grammar_penalty = 100.0;
    wparams.initial_prompt = "bishop to c3, rook to d4, knight to e5, d4 d5, knight to c3, c3, queen to d4, king b1, pawn to a1, bishop to b2, knight to c3,";

    printf("command: using %d threads\n", wparams.n_threads);

    WChess::callbacks cb;
    cb.get_audio = get_audio;
    cb.set_move = set_move;
    cb.set_grammar = set_grammar;

    WChess(g_contexts[i], wparams, cb, {}).run();

    if (i < g_contexts.size()) {
        whisper_free(g_contexts[i]);
        g_contexts[i] = nullptr;
    }
}

EMSCRIPTEN_BINDINGS(command) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_contexts[i] != nullptr) {
                    g_running = true;
                    if (g_worker.joinable()) {
                        g_worker.join();
                    }
                    g_worker = std::thread([i]() {
                        wchess_main(i);
                    });

                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t /* index */) {
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_running = false;
        }
        g_cv.notify_one();
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
        g_cv.notify_one();

        return 0;
    }));
}
