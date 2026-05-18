// miniaudio implementation lives in examples/miniaudio_impl.cpp so
// MINIAUDIO_IMPLEMENTATION is not duplicated across example targets.
#include "miniaudio.h"

#include "parakeet/engine.h"
#include "ggml.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::mutex              g_mu;
std::condition_variable g_cv;
std::vector<float>      g_pending;
std::atomic<bool>       g_stop{false};

void on_sigint(int) {
    g_stop.store(true);
    g_cv.notify_all();
}

std::atomic<int> g_min_log_level{GGML_LOG_LEVEL_WARN};

void ggml_log_filter(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level < g_min_log_level.load()) return;
    std::fputs(text, stderr);
}

void data_callback(ma_device * /*device*/, void * /*output*/, const void * input, ma_uint32 frame_count) {
    const float * in = static_cast<const float *>(input);
    if (!in || frame_count == 0) return;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const size_t prev = g_pending.size();
        g_pending.resize(prev + frame_count);
        std::memcpy(g_pending.data() + prev, in, frame_count * sizeof(float));
    }
    g_cv.notify_one();
}

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <parakeet.gguf | sortformer.gguf> [options]\n"
        "\n"
        "Captures the default input device at 16 kHz mono. If --model is a\n"
        "transcription (CTC/TDT) GGUF, runs live transcription. If --model is\n"
        "a Sortformer GGUF, runs live speaker diarization (segments labeled\n"
        "speaker_0..speaker_3).\n"
        "Press Ctrl-C to stop; the final partial chunk is flushed before exit.\n"
        "\n"
        "options:\n"
        "  --model PATH                   path to the GGUF (required)\n"
        "  --n-gpu-layers N               GPU offload (build with -DGGML_METAL=ON etc.)\n"
        "  --threads N                    CPU threads (0 = hardware_concurrency)\n"
        "  --chunk-ms N                   transcription: segment stride in ms (default 1000)\n"
        "                                 diarization:   chunk stride in ms (default 2000)\n"
        "  --left-context-ms N            transcription: left context per chunk (default 5000)\n"
        "  --right-lookahead-ms N         transcription: right lookahead per chunk (default 1000)\n"
        "  --history-ms N                 diarization: sliding history window (default 30000)\n"
        "  --list-devices                 list available capture devices and exit\n"
        "  --device N                     use device with this index (default: system default)\n"
        "  --accumulate                   transcription only: accumulate on one line; emit a\n"
        "                                 newline after --silence-flush-ms of silence\n"
        "  --silence-flush-ms N           --accumulate flush threshold in ms (default 1000)\n"
        "  --verbose                      let ggml / Metal info logs through to stderr\n"
        "  --help                         print this help\n",
        argv0);
}

struct Args {
    std::string model_path;
    int  n_gpu_layers = 0;
    int  n_threads    = 0;
    int  chunk_ms     = -1;
    int  left_ms      = 5000;
    int  right_ms     = 1000;
    int  history_ms   = 30000;
    bool list_devices = false;
    int  device_index = -1;
    bool verbose      = false;
    bool accumulate   = false;
    int  silence_flush_ms = 1000;
};

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--help" || s == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (s == "--model"              && i + 1 < argc) a.model_path    = argv[++i];
        else if (s == "--n-gpu-layers"       && i + 1 < argc) a.n_gpu_layers  = std::atoi(argv[++i]);
        else if (s == "--threads"            && i + 1 < argc) a.n_threads     = std::atoi(argv[++i]);
        else if (s == "--chunk-ms"           && i + 1 < argc) a.chunk_ms      = std::atoi(argv[++i]);
        else if (s == "--left-context-ms"    && i + 1 < argc) a.left_ms       = std::atoi(argv[++i]);
        else if (s == "--right-lookahead-ms" && i + 1 < argc) a.right_ms      = std::atoi(argv[++i]);
        else if (s == "--history-ms"         && i + 1 < argc) a.history_ms    = std::atoi(argv[++i]);
        else if (s == "--list-devices")                       a.list_devices  = true;
        else if (s == "--device"             && i + 1 < argc) a.device_index  = std::atoi(argv[++i]);
        else if (s == "--accumulate")                          a.accumulate    = true;
        else if (s == "--silence-flush-ms"   && i + 1 < argc) a.silence_flush_ms = std::atoi(argv[++i]);
        else if (s == "--verbose" || s == "-v")               a.verbose       = true;
        else {
            std::fprintf(stderr, "unknown option: %s\n", s.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (!a.list_devices && a.model_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

int list_devices_and_exit() {
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_context_init failed\n");
        return 1;
    }
    ma_device_info * playback_infos = nullptr;
    ma_uint32 n_playback = 0;
    ma_device_info * capture_infos = nullptr;
    ma_uint32 n_capture = 0;
    if (ma_context_get_devices(&ctx, &playback_infos, &n_playback,
                               &capture_infos, &n_capture) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_context_get_devices failed\n");
        ma_context_uninit(&ctx);
        return 1;
    }
    std::fprintf(stderr, "capture devices:\n");
    for (ma_uint32 i = 0; i < n_capture; ++i) {
        std::fprintf(stderr, "  [%u] %s%s\n", i, capture_infos[i].name,
                     capture_infos[i].isDefault ? " (default)" : "");
    }
    ma_context_uninit(&ctx);
    return 0;
}

}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    g_min_log_level.store(args.verbose ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_WARN);
    ggml_log_set(ggml_log_filter, nullptr);

    if (args.list_devices) return list_devices_and_exit();

    std::fprintf(stderr, "[live-mic] loading %s\n", args.model_path.c_str());
    parakeet::EngineOptions eopts;
    eopts.model_gguf_path = args.model_path;
    eopts.n_gpu_layers    = args.n_gpu_layers;
    eopts.n_threads       = args.n_threads;

    parakeet::Engine engine(eopts);

    const bool diarization_mode = engine.is_diarization_model();
    if (args.chunk_ms < 0) args.chunk_ms = diarization_mode ? 2000 : 1000;

    std::unique_ptr<parakeet::StreamSession>           tx_sess;
    std::unique_ptr<parakeet::SortformerStreamSession> diar_sess;

    bool   line_open       = false;
    double last_voice_end_s = 0.0;

    if (diarization_mode) {
        parakeet::SortformerStreamingOptions sopts;
        sopts.sample_rate    = 16000;
        sopts.chunk_ms       = args.chunk_ms;
        sopts.history_ms     = std::max(args.history_ms, args.chunk_ms);
        sopts.threshold      = 0.5f;
        sopts.min_segment_ms = 200;
        diar_sess = engine.diarize_start(sopts,
            [&](const parakeet::StreamingDiarizationSegment & s) {
                if (s.speaker_id < 0) return;
                std::printf("[%.2f-%.2f] speaker_%d (chunk %d%s)\n",
                            s.start_s, s.end_s, s.speaker_id, s.chunk_index,
                            s.is_final ? ", final" : "");
                std::fflush(stdout);
            });
    } else {
        parakeet::StreamingOptions sopts;
        sopts.sample_rate        = 16000;
        sopts.chunk_ms           = args.chunk_ms;
        sopts.left_context_ms    = args.left_ms;
        sopts.right_lookahead_ms = args.right_ms;
        tx_sess = engine.stream_start(sopts,
            [&](const parakeet::StreamingSegment & seg) {
                // EOU models: `is_eou_boundary` when `<EOU>` fired (often absent from `seg.text`).
                if (!args.accumulate) {
                    if (!seg.text.empty()) {
                        std::printf("\033[2K\r[%.2f-%.2f]%s\n",
                                    seg.start_s, seg.end_s, seg.text.c_str());
                    }
                    if (seg.is_eou_boundary) {
                        std::printf("\033[2K\r[%.2f-%.2f]  <EOU>  end of user turn\n",
                                    seg.start_s, seg.end_s);
                    }
                    if (!seg.text.empty() || seg.is_eou_boundary) {
                        std::fflush(stdout);
                    }
                    return;
                }

                if (!seg.text.empty()) {
                    if (!line_open) {
                        std::fputs(seg.text.c_str() +
                                   (seg.text.front() == ' ' ? 1 : 0),
                                   stdout);
                        line_open = true;
                    } else {
                        std::fputs(seg.text.c_str(), stdout);
                    }
                    std::fflush(stdout);
                    last_voice_end_s = seg.end_s;
                }

                if (seg.is_eou_boundary && line_open) {
                    std::fputs("  <EOU>\n", stdout);
                    std::fflush(stdout);
                    line_open = false;
                    return;
                }

                if (line_open &&
                    (seg.end_s - last_voice_end_s) * 1000.0 >= args.silence_flush_ms) {
                    std::fputc('\n', stdout);
                    std::fflush(stdout);
                    line_open = false;
                }
            });
    }

    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_context_init failed\n");
        return 1;
    }

    ma_device_info capture_info;
    ma_device_id   chosen_id;
    ma_device_id * chosen_ptr = nullptr;
    if (args.device_index >= 0) {
        ma_device_info * playback_infos = nullptr;
        ma_uint32 n_playback = 0;
        ma_device_info * capture_infos = nullptr;
        ma_uint32 n_capture = 0;
        ma_context_get_devices(&ctx, &playback_infos, &n_playback,
                               &capture_infos, &n_capture);
        if (args.device_index >= (int) n_capture) {
            std::fprintf(stderr, "device index %d out of range (have %u capture devices)\n",
                         args.device_index, n_capture);
            ma_context_uninit(&ctx);
            return 1;
        }
        capture_info = capture_infos[args.device_index];
        chosen_id    = capture_info.id;
        chosen_ptr   = &chosen_id;
        std::fprintf(stderr, "[live-mic] capture device: [%d] %s\n",
                     args.device_index, capture_info.name);
    } else {
        std::fprintf(stderr, "[live-mic] capture device: <system default>\n");
    }

    ma_device_config dcfg      = ma_device_config_init(ma_device_type_capture);
    dcfg.capture.pDeviceID     = chosen_ptr;
    dcfg.capture.format        = ma_format_f32;
    dcfg.capture.channels      = 1;
    dcfg.sampleRate            = 16000;
    dcfg.dataCallback          = data_callback;

    ma_device device;
    if (ma_device_init(&ctx, &dcfg, &device) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_device_init failed\n");
        ma_context_uninit(&ctx);
        return 1;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_device_start failed\n");
        ma_device_uninit(&device);
        ma_context_uninit(&ctx);
        return 1;
    }

    std::signal(SIGINT,  on_sigint);
    std::signal(SIGTERM, on_sigint);

    if (diarization_mode) {
        std::fprintf(stderr,
            "[live-mic] listening at 16 kHz mono (diarization).  "
            "chunk=%d ms  history=%d ms. Speak, Ctrl-C to stop.\n\n",
            args.chunk_ms, args.history_ms);
    } else {
        std::fprintf(stderr,
            "[live-mic] listening at 16 kHz mono.  "
            "chunk=%d ms  left=%d ms  right=%d ms. Speak, Ctrl-C to stop.\n\n",
            args.chunk_ms, args.left_ms, args.right_ms);
    }

    auto feed = [&](const float * data, int n) {
        if (diarization_mode) diar_sess->feed_pcm_f32(data, n);
        else                  tx_sess->feed_pcm_f32(data, n);
    };

    while (!g_stop.load()) {
        std::vector<float> batch;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_cv.wait_for(lk, std::chrono::milliseconds(100),
                          [] { return !g_pending.empty() || g_stop.load(); });
            if (g_pending.empty()) continue;
            batch.swap(g_pending);
        }
        feed(batch.data(), (int) batch.size());
    }

    std::fprintf(stderr, "\n[live-mic] stopping...\n");
    ma_device_stop(&device);

    {
        std::vector<float> tail;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            tail.swap(g_pending);
        }
        if (!tail.empty()) feed(tail.data(), (int) tail.size());
    }

    if (diarization_mode) diar_sess->finalize();
    else                  tx_sess->finalize();

    if (args.accumulate && line_open) {
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }

    ma_device_uninit(&device);
    ma_context_uninit(&ctx);
    return 0;
}
