// A minimal cross-platform desktop GUI for whisper.cpp.
//
// Built on Dear ImGui (SDL2 + OpenGL3 backend). It loads a model, transcribes
// an audio file on a background thread (so the UI stays responsive), shows the
// timestamped result and can export it to txt / srt / json.
//
// This is intentionally small: it reuses the existing audio decoding
// (read_audio_data, which handles wav/mp3/flac/ogg) and the whisper C API.
// Video files (e.g. .mp4) must be converted to audio first - see the
// conversion guide in examples/cli/README.md.

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"

#include <SDL.h>
#include <SDL_opengl.h>

#include "whisper.h"
#include "common-whisper.h"

#include "diarization.h"
#include "neural_diarize.h"
#include "file_browser.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

struct segment {
    int64_t     t0;
    int64_t     t1;
    std::string text;
    int         speaker = -1; // -1 = not diarized
};

// state shared between the UI thread and the transcription worker
struct app_state {
    // inputs (owned by the UI thread)
    char  model_path[1024] = "models/ggml-base.en.bin";
    char  audio_path[1024] = "";
    int   n_threads        = std::min(4, (int) std::thread::hardware_concurrency());
    bool  translate        = false;
    int   lang_index       = 0; // index into k_languages
    int   diarize_mode     = 0; // 0 = off, 1 = fast (built-in MFCC), 2 = accurate (sherpa-onnx)
    int   num_speakers     = 2; // 0 = auto-detect
    char  diarize_script[1024] = "examples/whisper.gui/diarize.py";

    // model is kept loaded between runs and reloaded only when the path changes
    whisper_context * ctx          = nullptr;
    std::string       loaded_model;

    // worker <-> UI communication
    std::thread       worker;
    std::atomic<bool> running{false};
    std::atomic<bool> abort_flag{false};
    std::atomic<int>  progress{0};

    std::mutex           mtx;     // guards the fields below
    std::vector<segment> segments;
    std::string          status = "Idle. Load a model and choose an audio file.";
};

// a small, common subset; "auto" lets whisper detect the language
static const char * k_languages[] = {
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "pl",
    "tr", "uk", "zh", "ja", "ko", "ar", "hi",
};

static void set_status(app_state & s, const std::string & msg) {
    std::lock_guard<std::mutex> lock(s.mtx);
    s.status = msg;
}

// runs on the worker thread
static void transcribe_worker(app_state * s,
                              std::string model_path,
                              std::string audio_path,
                              std::string language,
                              bool translate,
                              int n_threads,
                              int diarize_mode,
                              int num_speakers,
                              std::string diarize_script) {
    s->running    = true;
    s->abort_flag = false;
    s->progress   = 0;

    // 1. decode the audio file (wav/mp3/flac/ogg)
    std::vector<float>              pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    {
        std::error_code ec;
        if (!std::filesystem::exists(audio_path, ec)) {
            set_status(*s, "File not found: '" + audio_path +
                           "'. On WSL use /mnt/<drive>/... with forward slashes and no quotes "
                           "(e.g. /mnt/f/folder/clip.wav), or use the Browse button.");
            s->running = false;
            return;
        }
    }
    if (!read_audio_data(audio_path, pcmf32, pcmf32s, false)) {
        set_status(*s, "Could not decode '" + audio_path +
                       "'. Supported audio formats: wav/mp3/flac/ogg. "
                       "If this is a video (.mp4), extract the audio to a .wav first.");
        s->running = false;
        return;
    }

    // 2. (re)load the model if needed
    if (s->ctx == nullptr || s->loaded_model != model_path) {
        if (s->ctx != nullptr) {
            whisper_free(s->ctx);
            s->ctx = nullptr;
        }
        set_status(*s, "Loading model '" + model_path + "'...");
        whisper_context_params cparams = whisper_context_default_params();
        s->ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
        if (s->ctx == nullptr) {
            set_status(*s, "Failed to load model '" + model_path + "'.");
            s->running = false;
            return;
        }
        s->loaded_model = model_path;
    }

    // 3. run the transcription
    {
        std::lock_guard<std::mutex> lock(s->mtx);
        s->segments.clear();
        s->status = "Transcribing...";
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.n_threads        = n_threads;
    wparams.translate        = translate;
    wparams.language         = language == "auto" ? nullptr : language.c_str();
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;

    wparams.progress_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/,
                                   int progress, void * user_data) {
        ((app_state *) user_data)->progress = progress;
    };
    wparams.progress_callback_user_data = s;

    wparams.abort_callback = [](void * user_data) {
        return ((app_state *) user_data)->abort_flag.load();
    };
    wparams.abort_callback_user_data = s;

    const int ret = whisper_full(s->ctx, wparams, pcmf32.data(), (int) pcmf32.size());

    if (s->abort_flag.load()) {
        set_status(*s, "Cancelled.");
        s->running = false;
        return;
    }
    if (ret != 0) {
        set_status(*s, "Transcription failed (whisper_full returned " + std::to_string(ret) + ").");
        s->running = false;
        return;
    }

    // 4. collect the result
    const int n = whisper_full_n_segments(s->ctx);
    std::vector<segment> segs;
    segs.reserve(n);
    for (int i = 0; i < n; ++i) {
        segment seg;
        seg.t0   = whisper_full_get_segment_t0(s->ctx, i);
        seg.t1   = whisper_full_get_segment_t1(s->ctx, i);
        const char * txt = whisper_full_get_segment_text(s->ctx, i);
        seg.text = txt ? txt : "";
        segs.push_back(std::move(seg));
    }

    // 5. optional speaker diarization
    std::string done_msg = "Done - " + std::to_string(n) + " segment(s).";
    if (diarize_mode == 1 && n > 0) {
        // fast, built-in: embed each segment's audio span (MFCC) and cluster
        set_status(*s, "Diarizing (fast)...");
        std::vector<std::vector<float>> embeddings(segs.size());
        const int total = (int) pcmf32.size();
        for (size_t i = 0; i < segs.size(); ++i) {
            int beg = (int) (segs[i].t0 * WHISPER_SAMPLE_RATE / 100);
            int end = (int) (segs[i].t1 * WHISPER_SAMPLE_RATE / 100);
            beg = std::max(0, std::min(beg, total));
            end = std::max(beg, std::min(end, total));
            embeddings[i] = diarize::compute_embedding(pcmf32.data() + beg, end - beg, WHISPER_SAMPLE_RATE);
        }
        const std::vector<int> labels = diarize::cluster(embeddings, num_speakers, 0.15f);
        int n_spk = 0;
        for (size_t i = 0; i < segs.size(); ++i) {
            segs[i].speaker = labels[i];
            n_spk = std::max(n_spk, labels[i] + 1);
        }
        done_msg += " " + std::to_string(n_spk) + " speaker(s).";
    } else if (diarize_mode == 2 && n > 0) {
        // accurate: run the sherpa-onnx helper (diarize.py) as a subprocess
        set_status(*s, "Diarizing (neural, this can take a moment)...");
        std::vector<std::pair<int64_t, int64_t>> spans(segs.size());
        for (size_t i = 0; i < segs.size(); ++i) spans[i] = {segs[i].t0, segs[i].t1};
        std::vector<int> labels;
        std::string err;
        if (neural_diarize("python3", diarize_script, pcmf32, spans, num_speakers, "", labels, err)) {
            int n_spk = 0;
            for (size_t i = 0; i < segs.size() && i < labels.size(); ++i) {
                segs[i].speaker = labels[i];
                n_spk = std::max(n_spk, labels[i] + 1);
            }
            done_msg += " " + std::to_string(n_spk) + " speaker(s) [neural].";
        } else {
            done_msg += "  (neural diarization failed - transcript only. " + err.substr(0, 200) + ")";
        }
    }

    {
        std::lock_guard<std::mutex> lock(s->mtx);
        s->segments = std::move(segs);
        s->status   = done_msg;
    }
    s->progress = 100;
    s->running  = false;
}

// "Speaker N" (1-based) for display/export; empty when not diarized
static std::string speaker_label(int speaker) {
    return speaker >= 0 ? "Speaker " + std::to_string(speaker + 1) : std::string();
}

// export helpers (called from the UI thread while no worker is running)
static bool export_txt(const std::string & path, const std::vector<segment> & segs) {
    std::ofstream f(path);
    if (!f) return false;
    for (const auto & s : segs) {
        const char * t = s.text.c_str();
        while (*t == ' ') ++t; // trim leading space whisper adds
        if (s.speaker >= 0) f << speaker_label(s.speaker) << ": ";
        f << t << "\n";
    }
    return true;
}

static bool export_srt(const std::string & path, const std::vector<segment> & segs) {
    std::ofstream f(path);
    if (!f) return false;
    for (size_t i = 0; i < segs.size(); ++i) {
        f << (i + 1) << "\n";
        f << to_timestamp(segs[i].t0, true) << " --> " << to_timestamp(segs[i].t1, true) << "\n";
        if (segs[i].speaker >= 0) f << speaker_label(segs[i].speaker) << ":";
        f << segs[i].text << "\n\n";
    }
    return true;
}

static std::string json_escape(const std::string & in) {
    std::string out;
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': break;
            default:   out += c;       break;
        }
    }
    return out;
}

static bool export_json(const std::string & path, const std::vector<segment> & segs) {
    std::ofstream f(path);
    if (!f) return false;
    f << "{\n  \"transcription\": [\n";
    for (size_t i = 0; i < segs.size(); ++i) {
        f << "    {\n";
        f << "      \"from\": " << segs[i].t0 << ",\n";
        f << "      \"to\": "   << segs[i].t1 << ",\n";
        if (segs[i].speaker >= 0) f << "      \"speaker\": " << (segs[i].speaker + 1) << ",\n";
        f << "      \"text\": \"" << json_escape(segs[i].text) << "\"\n";
        f << "    }" << (i + 1 < segs.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    return true;
}

// strip the extension from a path, so "a/b/clip.wav" -> "a/b/clip"
static std::string strip_ext(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot   = path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        return path.substr(0, dot);
    }
    return path;
}

int main(int, char **) {
    // disable whisper's own logging to stderr; status is shown in the UI
    whisper_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return 1;
    }

    const char * glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow(
        "whisper.cpp", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 900, 640,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (window == nullptr) {
        fprintf(stderr, "SDL_CreateWindow error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr; // don't litter an imgui.ini in the cwd
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    app_state state;
    fb::FileBrowser model_browser;
    fb::FileBrowser audio_browser;

    bool done = false;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
            // drag a file onto the window to set the audio path
            if (event.type == SDL_DROPFILE) {
                char * dropped = event.drop.file;
                if (dropped) {
                    if (!state.running.load()) {
                        snprintf(state.audio_path, sizeof(state.audio_path), "%s", dropped);
                    }
                    SDL_free(dropped); // must always free, even while a run is in progress
                }
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // single full-window panel
        const ImGuiViewport * vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("whisper.cpp", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        const bool running = state.running.load();

        const float browse_w = 90.0f;
        const float input_w  = -(browse_w + ImGui::GetStyle().ItemSpacing.x);

        ImGui::TextUnformatted("Model");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(input_w);
        ImGui::InputText("##model", state.model_path, sizeof(state.model_path));
        ImGui::SameLine();
        if (ImGui::Button("Browse##m", ImVec2(browse_w, 0))) model_browser.Open(state.model_path);

        ImGui::TextUnformatted("Audio");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(input_w);
        ImGui::InputText("##audio", state.audio_path, sizeof(state.audio_path));
        ImGui::SameLine();
        if (ImGui::Button("Browse##a", ImVec2(browse_w, 0))) audio_browser.Open(state.audio_path);

        // file-picker modals (must be drawn every frame); apply the picked path
        {
            std::string picked;
            if (model_browser.Draw("Select model", picked))
                std::snprintf(state.model_path, sizeof(state.model_path), "%s", picked.c_str());
            if (audio_browser.Draw("Select audio", picked))
                std::snprintf(state.audio_path, sizeof(state.audio_path), "%s", picked.c_str());
        }
        ImGui::TextDisabled("Tip: click Browse, or paste a path. On WSL use /mnt/<drive>/... (e.g. /mnt/f/...), not F:\\...");

        ImGui::Separator();

        ImGui::SetNextItemWidth(120);
        ImGui::Combo("Language", &state.lang_index, k_languages, IM_ARRAYSIZE(k_languages));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderInt("Threads", &state.n_threads, 1,
                         std::max(1, (int) std::thread::hardware_concurrency()));
        ImGui::SameLine();
        ImGui::Checkbox("Translate to English", &state.translate);

        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Diarize", &state.diarize_mode,
                     "Off\0Fast (built-in)\0Accurate (sherpa-onnx)\0");
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Label who is speaking.\n"
                "  Fast      - built-in MFCC, no setup, weak on similar voices.\n"
                "  Accurate  - neural (sherpa-onnx) via diarize.py; needs:\n"
                "                pip install sherpa-onnx numpy\n"
                "                ./examples/whisper.gui/download-diarization-models.sh\n"
                "Set the known speaker count for best results.");
        }
        if (state.diarize_mode != 0) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150);
            ImGui::InputInt("Speakers (0 = auto)", &state.num_speakers);
            if (state.num_speakers < 0) state.num_speakers = 0;
        }
        if (state.diarize_mode == 2) {
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("diarize.py", state.diarize_script, sizeof(state.diarize_script));
        }

        // begin / cancel
        if (!running) {
            const bool can_run = state.audio_path[0] != '\0' && state.model_path[0] != '\0';
            if (!can_run) ImGui::BeginDisabled();
            if (ImGui::Button("Transcribe", ImVec2(120, 0))) {
                if (state.worker.joinable()) state.worker.join();
                // set running on the UI thread *before* spawning so the button
                // flips to Cancel on the very next frame (avoids a join-on-click race)
                state.running    = true;
                state.abort_flag = false;
                state.progress   = 0;
                state.worker = std::thread(transcribe_worker, &state,
                                           std::string(state.model_path),
                                           std::string(state.audio_path),
                                           std::string(k_languages[state.lang_index]),
                                           state.translate, state.n_threads,
                                           state.diarize_mode, state.num_speakers,
                                           std::string(state.diarize_script));
            }
            if (!can_run) ImGui::EndDisabled();
        } else {
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                state.abort_flag = true;
            }
            ImGui::SameLine();
            ImGui::ProgressBar(state.progress.load() / 100.0f, ImVec2(-1, 0));
        }

        // status line
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            ImGui::TextWrapped("%s", state.status.c_str());
        }

        ImGui::Separator();

        // export buttons (only when there is a finished result)
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            const bool have_result = !running && !state.segments.empty();
            const std::string base  = strip_ext(state.audio_path);

            if (!have_result) ImGui::BeginDisabled();
            if (ImGui::Button("Export .txt"))  { if (export_txt(base + ".txt", state.segments))   state.status = "Saved " + base + ".txt"; }
            ImGui::SameLine();
            if (ImGui::Button("Export .srt"))  { if (export_srt(base + ".srt", state.segments))   state.status = "Saved " + base + ".srt"; }
            ImGui::SameLine();
            if (ImGui::Button("Export .json")) { if (export_json(base + ".json", state.segments)) state.status = "Saved " + base + ".json"; }
            if (!have_result) ImGui::EndDisabled();
        }

        // transcript
        ImGui::BeginChild("transcript", ImVec2(0, 0), true);
        {
            // a small palette so each speaker gets a distinct, stable color
            static const ImVec4 spk_colors[] = {
                ImVec4(0.40f, 0.75f, 1.00f, 1.0f), ImVec4(1.00f, 0.65f, 0.40f, 1.0f),
                ImVec4(0.55f, 0.90f, 0.55f, 1.0f), ImVec4(0.95f, 0.60f, 0.85f, 1.0f),
                ImVec4(0.90f, 0.85f, 0.45f, 1.0f), ImVec4(0.70f, 0.70f, 1.00f, 1.0f),
            };
            const int n_colors = IM_ARRAYSIZE(spk_colors);

            std::lock_guard<std::mutex> lock(state.mtx);
            for (const auto & seg : state.segments) {
                ImGui::TextDisabled("[%s -> %s]", to_timestamp(seg.t0).c_str(), to_timestamp(seg.t1).c_str());
                ImGui::SameLine();
                if (seg.speaker >= 0) {
                    ImGui::TextColored(spk_colors[seg.speaker % n_colors], "Speaker %d:", seg.speaker + 1);
                    ImGui::SameLine();
                }
                ImGui::TextWrapped("%s", seg.text.c_str());
            }
        }
        ImGui::EndChild();

        ImGui::End();

        ImGui::Render();
        ImGuiIO & io = ImGui::GetIO();
        glViewport(0, 0, (int) io.DisplaySize.x, (int) io.DisplaySize.y);
        glClearColor(0.10f, 0.10f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    // make sure the worker is done before tearing anything down
    state.abort_flag = true;
    if (state.worker.joinable()) state.worker.join();
    if (state.ctx) whisper_free(state.ctx);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
