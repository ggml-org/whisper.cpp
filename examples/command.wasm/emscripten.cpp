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
#include <regex>

constexpr int N_THREAD = 8;

whisper_context * g_context;
std::vector<struct whisper_state *> g_states(4, nullptr);

std::mutex  g_mutex;
std::thread g_worker;

std::atomic<bool> g_running(false);

std::string g_status        = "";
std::string g_status_forced = "";
std::string g_transcribed   = "";

std::vector<float> g_pcmf32;

static std::string trim(const std::string & s) {
    std::regex e("^\\s+|\\s+$");
    return std::regex_replace(s, e, "");
}

static void high_pass_filter(std::vector<float> & data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

// compute similarity between two strings using Levenshtein distance
static float similarity(const std::string & s0, const std::string & s1) {
    const size_t len0 = s0.size() + 1;
    const size_t len1 = s1.size() + 1;

    std::vector<int> col(len1, 0);
    std::vector<int> prevCol(len1, 0);

    for (size_t i = 0; i < len1; i++) {
        prevCol[i] = i;
    }

    for (size_t i = 0; i < len0; i++) {
        col[0] = i;
        for (size_t j = 1; j < len1; j++) {
            col[j] = std::min(std::min(1 + col[j - 1], 1 + prevCol[j]), prevCol[j - 1] + (s0[i - 1] == s1[j - 1] ? 0 : 1));
        }
        col.swap(prevCol);
    }

    const float dist = prevCol[len1 - 1];

    return 1.0f - (dist / std::max(s0.size(), s1.size()));
}

void command_set_status(const std::string & status) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_status = status;
}

bool command_vad_simple(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples      = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all  = 0.0f;
    float energy_last = 0.0f;

    for (size_t i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);

        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all  /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        fprintf(stderr, "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold*energy_all) {
        return false;
    }

    return true;
}

std::string command_transcribe(whisper_context * ctx, whisper_state * state, const whisper_full_params & wparams, const std::vector<float> & pcmf32, float & prob, int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    prob = 0.0f;
    t_ms = 0;

    if (whisper_full_with_state(ctx, state, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(state);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(state, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(state, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(state, i, j);

            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}

void command_get_audio(int ms, int sample_rate, std::vector<float> & audio) {
    const int64_t n_samples = (ms * sample_rate) / 1000;

    int64_t n_take = 0;
    if (g_pcmf32.size() < n_samples) {
        n_take = g_pcmf32.size();
    } else {
        n_take = n_samples;
    }

    audio.resize(n_take);
    std::copy(g_pcmf32.end() - n_take, g_pcmf32.end(), audio.begin());
}

void command_main(size_t index) {
    command_set_status("loading data ...");

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

    wparams.max_tokens       = 32;
    wparams.audio_ctx        = 768; // partial encoder context for better performance

    wparams.language         = "en";

    printf("command: using %d threads\n", wparams.n_threads);

    bool is_running   = true;
    bool have_prompt  = false;
    bool ask_prompt   = true;
    bool print_energy = false;

    float prob0 = 0.0f;
    float prob  = 0.0f;

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

    const std::string k_prompt = "Ok Whisper, start listening for commands.";

    // whisper context
    auto & ctx = g_context;
    auto & state = g_states[index];

    const int32_t vad_ms     = 2000;
    const int32_t prompt_ms  = 5000;
    const int32_t command_ms = 4000;

    const float vad_thold  = 0.1f;
    const float freq_thold = -1.0f;

    while (g_running) {
        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (ask_prompt) {
            fprintf(stdout, "\n");
            fprintf(stdout, "%s: Say the following phrase: '%s%s%s'\n", __func__, "\033[1m", k_prompt.c_str(), "\033[0m");
            fprintf(stdout, "\n");

            {
                char txt[1024];
                snprintf(txt, sizeof(txt), "Say the following phrase: '%s'", k_prompt.c_str());
                command_set_status(txt);
            }

            ask_prompt = false;
        }

        int64_t t_ms = 0;

        {
            command_get_audio(vad_ms, WHISPER_SAMPLE_RATE, pcmf32_cur);

            if (command_vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1000, vad_thold, freq_thold, print_energy)) {
                fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);
                command_set_status("Speech detected! Processing ...");

                if (!have_prompt) {
                    command_get_audio(prompt_ms, WHISPER_SAMPLE_RATE, pcmf32_cur);

                    const auto txt = ::trim(::command_transcribe(ctx, state, wparams, pcmf32_cur, prob0, t_ms));

                    fprintf(stdout, "%s: Heard '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", txt.c_str(), "\033[0m", (int) t_ms);

                    const float sim = similarity(txt, k_prompt);

                    if (txt.length() < 0.8*k_prompt.length() || txt.length() > 1.2*k_prompt.length() || sim < 0.8f) {
                        fprintf(stdout, "%s: WARNING: prompt not recognized, try again\n", __func__);
                        ask_prompt = true;
                    } else {
                        fprintf(stdout, "\n");
                        fprintf(stdout, "%s: The prompt has been recognized!\n", __func__);
                        fprintf(stdout, "%s: Waiting for voice commands ...\n", __func__);
                        fprintf(stdout, "\n");

                        {
                            char txt[1024];
                            snprintf(txt, sizeof(txt), "Success! Waiting for voice commands ...");
                            command_set_status(txt);
                        }

                        // save the audio for the prompt
                        pcmf32_prompt = pcmf32_cur;
                        have_prompt = true;
                    }
                } else {
                    command_get_audio(command_ms, WHISPER_SAMPLE_RATE, pcmf32_cur);

                    // prepend the prompt audio
                    pcmf32_cur.insert(pcmf32_cur.begin(), pcmf32_prompt.begin(), pcmf32_prompt.end());

                    const auto txt = ::trim(::command_transcribe(ctx, state, wparams, pcmf32_cur, prob, t_ms));

                    prob = 100.0f*(prob - prob0);

                    fprintf(stdout, "%s: heard '%s'\n", __func__, txt.c_str());

                    // find the prompt in the text
                    float best_sim = 0.0f;
                    size_t best_len = 0;
                    for (int n = 0.8*k_prompt.size(); n <= 1.2*k_prompt.size(); ++n) {
                        const auto prompt = txt.substr(0, n);

                        const float sim = similarity(prompt, k_prompt);

                        //fprintf(stderr, "%s: prompt = '%s', sim = %f\n", __func__, prompt.c_str(), sim);

                        if (sim > best_sim) {
                            best_sim = sim;
                            best_len = n;
                        }
                    }

                    const std::string command = ::trim(txt.substr(best_len));

                    fprintf(stdout, "%s: Command '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", command.c_str(), "\033[0m", (int) t_ms);
                    fprintf(stdout, "\n");

                    {
                        char txt[1024];
                        snprintf(txt, sizeof(txt), "Command '%s', (t = %d ms)", command.c_str(), (int) t_ms);
                        command_set_status(txt);
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        g_transcribed = command;
                    }
                }

                g_pcmf32.clear();
            }
        }
    }

    if (index < g_states.size()) {
        whisper_free_state(g_states[index]);
        g_states[index] = nullptr;
    }

    whisper_free(g_context);
    g_context = nullptr;
}

EMSCRIPTEN_BINDINGS(command) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        if (g_context == nullptr) {
            g_context = whisper_init_from_file(path_model.c_str());
        }

        for (size_t i = 0; i < g_states.size(); ++i) {
            if (g_states[i] == nullptr) {
                g_states[i] = whisper_init_state(g_context);
                if (g_states[i] != nullptr) {
                    g_running = true;
                    if (g_worker.joinable()) {
                        g_worker.join();
                    }
                    g_worker = std::thread([i]() {
                        command_main(i);
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

        if (index >= g_states.size()) {
            return -1;
        }

        if (g_states[index] == nullptr) {
            return -2;
        }

        if (g_context == nullptr) {
            return -3;
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
