#include "parakeet.h"
#include "common-whisper.h"
#include "server-common.h"

#include "httplib.h"
#include "json.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <csignal>
#include <atomic>

using namespace httplib;
using json = nlohmann::ordered_json;

struct parakeet_params {
    int32_t n_threads   = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t offset_ms   = 0;
    int32_t duration_ms = 0;

    bool no_context      = false;
    int32_t audio_ctx    = 0;  // 0 = use default

    bool use_gpu         = true;
    int32_t gpu_device   = 0;

    std::string model           = "models/ggml-parakeet-tdt-0.6b-v3.bin";
    std::string response_format = json_format;
};

static void parakeet_print_usage(int /*argc*/, char ** argv, const parakeet_params & params, const server_params & sparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help        [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N   [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -ot N,     --offset-t N  [%-7d] time offset in milliseconds\n", params.offset_ms);
    fprintf(stderr, "  -d N,      --duration N  [%-7d] duration of audio to process in milliseconds\n", params.duration_ms);
    fprintf(stderr, "  -ac N,     --audio-ctx N [%-7d] audio context size (0 = use default)\n", params.audio_ctx);
    fprintf(stderr, "  -nc,       --no-context  [%-7s] do not use past transcription as context\n", params.no_context ? "true" : "false");
    fprintf(stderr, "  -m FNAME,  --model FNAME [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -ng,       --no-gpu      [%-7s] do not use GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -dev N,    --device N    [%-7d] GPU device ID\n", params.gpu_device);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --host HOST,             [%-7s] Hostname/IP address for the server\n", sparams.hostname.c_str());
    fprintf(stderr, "  --port PORT,             [%-7d] Port number for the server\n", sparams.port);
    fprintf(stderr, "  --public PATH,           [%-7s] Path to the public folder\n", sparams.public_path.c_str());
    fprintf(stderr, "  --request-path PATH,     [%-7s] Request path prefix\n", sparams.request_path.c_str());
    fprintf(stderr, "  --inference-path PATH,   [%-7s] Inference endpoint path\n", sparams.inference_path.c_str());
    fprintf(stderr, "  --convert,               [%-7s] Convert audio to WAV via ffmpeg\n", sparams.ffmpeg_converter ? "true" : "false");
    fprintf(stderr, "  --tmp-dir PATH,          [%-7s] Temporary directory for converted files\n", sparams.tmp_dir.c_str());
    fprintf(stderr, "\n");
}

static bool parakeet_params_parse(int argc, char ** argv, parakeet_params & params, server_params & sparams) {
    if (const char * env_device = std::getenv("PARAKEET_ARG_DEVICE")) {
        params.gpu_device = std::stoi(env_device);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            parakeet_print_usage(argc, argv, params, sparams);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")     { params.n_threads       = std::stoi(argv[++i]); }
        else if (arg == "-ot"   || arg == "--offset-t")    { params.offset_ms       = std::stoi(argv[++i]); }
        else if (arg == "-d"    || arg == "--duration")    { params.duration_ms     = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")   { params.audio_ctx       = std::stoi(argv[++i]); }
        else if (arg == "-nc"   || arg == "--no-context")  { params.no_context      = true; }
        else if (arg == "-m"    || arg == "--model")       { params.model           = argv[++i]; }
        else if (arg == "-ng"   || arg == "--no-gpu")      { params.use_gpu         = false; }
        else if (arg == "-dev"  || arg == "--device")      { params.gpu_device      = std::stoi(argv[++i]); }
        else if (arg == "--host")                          { sparams.hostname       = argv[++i]; }
        else if (arg == "--port")                          { sparams.port           = std::stoi(argv[++i]); }
        else if (arg == "--public")                        { sparams.public_path    = argv[++i]; }
        else if (arg == "--request-path")                  { sparams.request_path   = argv[++i]; }
        else if (arg == "--inference-path")                { sparams.inference_path = argv[++i]; }
        else if (arg == "--convert")                       { sparams.ffmpeg_converter = true; }
        else if (arg == "--tmp-dir")                       { sparams.tmp_dir        = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            parakeet_print_usage(argc, argv, params, sparams);
            exit(1);
        }
    }

    return true;
}

static void get_req_parameters(const Request & req, parakeet_params & params) {
    if (req.has_file("offset_t")) {
        params.offset_ms = std::stoi(req.get_file_value("offset_t").content);
    }
    if (req.has_file("duration")) {
        params.duration_ms = std::stoi(req.get_file_value("duration").content);
    }
    if (req.has_file("audio_ctx")) {
        params.audio_ctx = std::stoi(req.get_file_value("audio_ctx").content);
    }
    if (req.has_file("no_context")) {
        params.no_context = parse_str_to_bool(req.get_file_value("no_context").content);
    }
    if (req.has_file("response_format")) {
        params.response_format = req.get_file_value("response_format").content;
    }
}

struct parakeet_result : transcription_result {
    parakeet_context * ctx;

    explicit parakeet_result(parakeet_context * c) : ctx(c) {}

    int n_segments() const override {
        return parakeet_full_n_segments(ctx);
    }

    segment_info get_segment(int i) const override {
        segment_info seg;
        seg.text           = parakeet_full_get_segment_text(ctx, i);
        seg.t0             = parakeet_full_get_segment_t0(ctx, i);
        seg.t1             = parakeet_full_get_segment_t1(ctx, i);
        seg.no_speech_prob = 0.0f;

        const int n_tokens = parakeet_full_n_tokens(ctx, i);
        seg.tokens.reserve(n_tokens);
        for (int j = 0; j < n_tokens; ++j) {
            parakeet_token_data tok = parakeet_full_get_token_data(ctx, i, j);
            seg.tokens.push_back({tok.id, parakeet_full_get_token_text(ctx, i, j),
                                  tok.t0, tok.t1, tok.p});
        }
        return seg;
    }

    std::string get_language() const override { return "N/A"; }
};

static std::string generate_index_page(const server_params & sparams) {
    std::ostringstream oss;
    oss << R"(
    <html>
    <head>
        <title>Parakeet.cpp Server</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width">
        <style>
        body {
            font-family: sans-serif;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        label {
            margin-bottom: 0.5rem;
        }
        input, select {
            margin-bottom: 1rem;
        }
        button {
            margin-top: 1rem;
        }
        </style>
    </head>
    <body>
        <h1>Parakeet.cpp Server</h1>

        <h2>)" << sparams.request_path << sparams.inference_path << R"(</h2>
        <pre>
    curl 127.0.0.1:)" << sparams.port << sparams.request_path << sparams.inference_path << R"( \
    -H "Content-Type: multipart/form-data" \
    -F file="@<file-path>" \
    -F response_format="json"
        </pre>

        <div>
            <h2>Try it out</h2>
            <form action=")" << sparams.request_path << sparams.inference_path << R"(" method="POST" enctype="multipart/form-data">
                <label for="file">Choose an audio file:</label>
                <input type="file" id="file" name="file" accept="audio/*" required><br>

                <label for="response_format">Response Format:</label>
                <select id="response_format" name="response_format">
                    <option value="json">JSON</option>
                    <option value="text">Text</option>
                    <option value="srt">SRT</option>
                    <option value="vtt">VTT</option>
                    <option value="verbose_json">Verbose JSON</option>
                </select><br>

                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    )";
    return oss.str();
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    parakeet_params params;
    server_params sparams;

    std::mutex model_mutex;

    if (!parakeet_params_parse(argc, argv, params, sparams)) {
        parakeet_print_usage(argc, argv, params, sparams);
        return 1;
    }

    if (sparams.ffmpeg_converter) {
        check_ffmpeg_availability();
    }

    parakeet_context_params cparams = parakeet_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.gpu_device = params.gpu_device;

    std::unique_ptr<httplib::Server> svr = std::make_unique<httplib::Server>();
    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    struct parakeet_context * ctx = parakeet_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize parakeet context from '%s'\n", params.model.c_str());
        return 1;
    }

    state.store(SERVER_STATE_READY);

    printf("Successfully loaded Parakeet model from: %s\n", params.model.c_str());
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            params.n_threads, (int32_t) std::thread::hardware_concurrency(), parakeet_print_system_info());

    const std::string default_content = generate_index_page(sparams);

    parakeet_params default_params = params;

    auto inference_handler = [&](const Request & req, Response & res) {
        std::lock_guard<std::mutex> lock(model_mutex);

        if (!req.has_file("file")) {
            fprintf(stderr, "error: no 'file' field in the request\n");
            res.status = 400;
            res.set_content("{\"error\":\"no 'file' field in the request\"}", "application/json");
            return;
        }

        auto audio_file = req.get_file_value("file");
        parakeet_params cur_params = default_params;
        get_req_parameters(req, cur_params);

        std::string filename{audio_file.filename};
        printf("Received request: %s\n", filename.c_str());

        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;

        if (sparams.ffmpeg_converter) {
            const std::string temp_filename = generate_temp_filename(sparams.tmp_dir, "parakeet-server", ".wav");
            std::ofstream temp_file{temp_filename, std::ios::binary};
            temp_file << audio_file.content;
            temp_file.close();

            std::string error_resp;
            if (!convert_to_wav(temp_filename, error_resp, false)) {
                res.status = 500;
                res.set_content(error_resp, "application/json");
                return;
            }

            if (!::read_audio_data(temp_filename, pcmf32, pcmf32s, false)) {
                fprintf(stderr, "error: failed to read WAV file '%s'\n", temp_filename.c_str());
                res.status = 400;
                res.set_content("{\"error\":\"failed to read WAV file\"}", "application/json");
                std::remove(temp_filename.c_str());
                return;
            }
            std::remove(temp_filename.c_str());
        } else {
            if (!::read_audio_data(audio_file.content.data(), audio_file.content.size(), pcmf32, pcmf32s, false)) {
                fprintf(stderr, "error: failed to read audio data\n");
                res.status = 400;
                res.set_content("{\"error\":\"failed to read audio data\"}", "application/json");
                return;
            }
        }

        printf("Successfully loaded %s\n", filename.c_str());

        fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads ...\n",
                __func__, filename.c_str(), (int)pcmf32.size(),
                float(pcmf32.size()) / PARAKEET_SAMPLE_RATE, cur_params.n_threads);

        {
            printf("Running parakeet.cpp inference on %s\n", filename.c_str());

            parakeet_full_params fparams = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
            fparams.n_threads    = cur_params.n_threads;
            fparams.offset_ms    = cur_params.offset_ms;
            fparams.duration_ms  = cur_params.duration_ms;
            fparams.no_context   = cur_params.no_context;
            fparams.audio_ctx    = cur_params.audio_ctx;

            // Abort callback for HTTP disconnect
            fparams.abort_callback = [](void * user_data) {
                auto req_ptr = static_cast<const httplib::Request *>(user_data);
                return req_ptr->is_connection_closed();
            };
            fparams.abort_callback_user_data = (void *) &req;

            int ret = parakeet_full(ctx, fparams, pcmf32.data(), (int)pcmf32.size());
            if (ret != 0) {
                if (req.is_connection_closed()) {
                    fprintf(stderr, "client disconnected, aborted processing\n");
                    res.status = 499;
                    res.set_content("{\"error\":\"client disconnected\"}", "application/json");
                    return;
                }
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                res.status = 500;
                res.set_content("{\"error\":\"failed to process audio\"}", "application/json");
                return;
            }
        }

        // Format response
        parakeet_result result{ctx};

        if (cur_params.response_format == text_format) {
            std::string text = format_text(result);
            res.set_content(text, "text/plain; charset=utf-8");
        } else if (cur_params.response_format == srt_format) {
            std::string srt = format_srt(result, 0);
            res.set_content(srt, "application/x-subrip");
        } else if (cur_params.response_format == vtt_format) {
            std::string vtt = format_vtt(result);
            res.set_content(vtt, "text/vtt");
        } else if (cur_params.response_format == vjson_format) {
            float duration = float(pcmf32.size()) / PARAKEET_SAMPLE_RATE;
            std::string vjson = format_verbose_json(result, 0.0f, duration, false, true);
            res.set_content(vjson, "application/json");
        } else {
            std::string j = format_json(result);
            res.set_content(j, "application/json");
        }
    };

    auto load_handler = [&](const Request & req, Response & res) {
        std::lock_guard<std::mutex> lock(model_mutex);
        state.store(SERVER_STATE_LOADING_MODEL);

        if (!req.has_file("model")) {
            fprintf(stderr, "error: no 'model' field in the request\n");
            res.status = 400;
            res.set_content("{\"error\":\"no 'model' field in the request\"}", "application/json");
            return;
        }

        std::string model = req.get_file_value("model").content;

        parakeet_free(ctx);

        ctx = parakeet_init_from_file_with_params(model.c_str(), cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "error: failed to load model '%s'\n", model.c_str());
            res.status = 500;
            res.set_content("{\"error\":\"failed to load model\"}", "application/json");
            return;
        }

        state.store(SERVER_STATE_READY);
        res.set_content("Load was successful!", "text/plain");
    };

    setup_server_common(*svr, sparams, state, load_handler, inference_handler, default_content, "parakeet.cpp");

    setup_signal_handler([&]() {
        printf("\nShutting down gracefully...\n");
        svr->stop();
    });

    if (!svr->bind_to_port(sparams.hostname, sparams.port)) {
        fprintf(stderr, "couldn't bind to server socket: hostname=%s port=%d\n",
                sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    svr->set_base_dir(sparams.public_path);

    printf("\nparakeet server listening at http://%s:%d\n\n", sparams.hostname.c_str(), sparams.port);

    auto clean_up = [&]() {
        parakeet_print_timings(ctx);
        parakeet_free(ctx);
    };

    std::thread t([&]() {
        if (!svr->listen_after_bind()) {
            fprintf(stderr, "error: server listen failed\n");
        }
    });

    svr->wait_until_ready();
    t.join();

    clean_up();

    return 0;
}
