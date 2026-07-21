// miniaudio implementation lives in examples/miniaudio_impl.cpp so
// MINIAUDIO_IMPLEMENTATION is not duplicated across example targets.
#include "miniaudio.h"

#include "parakeet/engine.h"
#include "parakeet/attributed.h"
#include "ggml.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
        "usage: %s --asr-model <gguf> --diar-model <gguf> [options]\n"
        "\n"
        "Captures the default input device at 16 kHz mono and runs both a\n"
        "transcription engine (CTC/TDT) and a Sortformer diarization engine\n"
        "on the same audio stream. Each transcript segment is tagged with\n"
        "the speaker whose live diarization range overlaps it the most.\n"
        "Press Ctrl-C to stop; tail audio is flushed before exit.\n"
        "\n"
        "options:\n"
        "  --asr-model PATH               path to a CTC or TDT GGUF (required)\n"
        "  --diar-model PATH              path to a Sortformer GGUF (required)\n"
        "  --asr-n-gpu-layers N           ASR engine GPU offload (Metal/CUDA build)\n"
        "  --diar-n-gpu-layers N          Sortformer engine GPU offload\n"
        "  --threads N                    CPU threads per engine (0 = hardware_concurrency)\n"
        "  --asr-chunk-ms N               transcription stride in ms (default 1000)\n"
        "  --asr-left-context-ms N        left context per chunk in ms (default 5000)\n"
        "  --asr-right-lookahead-ms N     right lookahead per chunk in ms (default 1000)\n"
        "  --diar-chunk-ms N              diarization stride in ms (default 2000)\n"
        "  --diar-history-ms N            diarization sliding history (default 30000)\n"
        "  --list-devices                 list capture devices and exit\n"
        "  --device N                     capture device index (default: system default)\n"
        "  --speaker-history-ms N         retain this much diarization history for\n"
        "                                 attribution lookups (default 60000)\n"
        "  --accumulate                   accumulate transcript on one line per speaker;\n"
        "                                 emit a newline on speaker change or after\n"
        "                                 --silence-flush-ms of silence\n"
        "  --silence-flush-ms N           silence duration that triggers a newline in\n"
        "                                 --accumulate mode (default 1000)\n"
        "  --verbose                      let ggml / Metal info logs through to stderr\n"
        "  --help                         print this help\n",
        argv0);
}

struct Args {
    std::string asr_model_path;
    std::string diar_model_path;
    int  asr_n_gpu_layers   = 0;
    int  diar_n_gpu_layers  = 0;
    int  n_threads          = 0;
    int  asr_chunk_ms       = 1000;
    int  asr_left_ms        = 5000;
    int  asr_right_ms       = 1000;
    int  diar_chunk_ms      = 2000;
    int  diar_history_ms    = 30000;
    bool list_devices       = false;
    int  device_index       = -1;
    int  speaker_history_ms = 60000;
    bool accumulate         = false;
    int  silence_flush_ms   = 1000;
    bool verbose            = false;
};

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--help" || s == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (s == "--asr-model"            && i + 1 < argc) a.asr_model_path     = argv[++i];
        else if (s == "--diar-model"           && i + 1 < argc) a.diar_model_path    = argv[++i];
        else if (s == "--asr-n-gpu-layers"     && i + 1 < argc) a.asr_n_gpu_layers   = std::atoi(argv[++i]);
        else if (s == "--diar-n-gpu-layers"    && i + 1 < argc) a.diar_n_gpu_layers  = std::atoi(argv[++i]);
        else if (s == "--threads"              && i + 1 < argc) a.n_threads          = std::atoi(argv[++i]);
        else if (s == "--asr-chunk-ms"         && i + 1 < argc) a.asr_chunk_ms       = std::atoi(argv[++i]);
        else if (s == "--asr-left-context-ms"  && i + 1 < argc) a.asr_left_ms        = std::atoi(argv[++i]);
        else if (s == "--asr-right-lookahead-ms" && i + 1 < argc) a.asr_right_ms     = std::atoi(argv[++i]);
        else if (s == "--diar-chunk-ms"        && i + 1 < argc) a.diar_chunk_ms      = std::atoi(argv[++i]);
        else if (s == "--diar-history-ms"      && i + 1 < argc) a.diar_history_ms    = std::atoi(argv[++i]);
        else if (s == "--list-devices")                          a.list_devices       = true;
        else if (s == "--device"               && i + 1 < argc) a.device_index       = std::atoi(argv[++i]);
        else if (s == "--speaker-history-ms"   && i + 1 < argc) a.speaker_history_ms = std::atoi(argv[++i]);
        else if (s == "--accumulate")                            a.accumulate         = true;
        else if (s == "--silence-flush-ms"     && i + 1 < argc) a.silence_flush_ms   = std::atoi(argv[++i]);
        else if (s == "--verbose" || s == "-v")                  a.verbose            = true;
        else {
            std::fprintf(stderr, "unknown option: %s\n", s.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (!a.list_devices && (a.asr_model_path.empty() || a.diar_model_path.empty())) {
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

struct DiarSpan {
    int    speaker_id;
    double start_s;
    double end_s;
};

int speaker_for_range(const std::deque<DiarSpan> & spans, double s, double e) {
    if (e <= s) return -1;
    double best_overlap = 0.0;
    int    best_spk     = -1;
    for (const auto & d : spans) {
        const double lo = std::max(s, d.start_s);
        const double hi = std::min(e, d.end_s);
        const double ov = hi - lo;
        if (ov > best_overlap) {
            best_overlap = ov;
            best_spk     = d.speaker_id;
        }
    }
    return best_spk;
}

}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    g_min_log_level.store(args.verbose ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_WARN);
    ggml_log_set(ggml_log_filter, nullptr);

    if (args.list_devices) return list_devices_and_exit();

    using namespace parakeet;

    std::fprintf(stderr, "[live-mic-attributed] loading ASR  %s\n", args.asr_model_path.c_str());
    EngineOptions asr_eopts;
    asr_eopts.model_gguf_path = args.asr_model_path;
    asr_eopts.n_gpu_layers    = args.asr_n_gpu_layers;
    asr_eopts.n_threads       = args.n_threads;
    Engine asr_engine(asr_eopts);
    if (asr_engine.is_diarization_model()) {
        std::fprintf(stderr, "error: --asr-model %s is a Sortformer GGUF; expected CTC or TDT\n",
                     args.asr_model_path.c_str());
        return 1;
    }

    std::fprintf(stderr, "[live-mic-attributed] loading DIAR %s\n", args.diar_model_path.c_str());
    EngineOptions diar_eopts;
    diar_eopts.model_gguf_path = args.diar_model_path;
    diar_eopts.n_gpu_layers    = args.diar_n_gpu_layers;
    diar_eopts.n_threads       = args.n_threads;
    Engine diar_engine(diar_eopts);
    if (!diar_engine.is_diarization_model()) {
        std::fprintf(stderr, "error: --diar-model %s is not a Sortformer GGUF\n",
                     args.diar_model_path.c_str());
        return 1;
    }

    std::deque<DiarSpan> diar_history;
    const double speaker_history_s = args.speaker_history_ms / 1000.0;
    int last_spk_seen = -1;

    auto on_diar = [&](const StreamingDiarizationSegment & s) {
        if (s.speaker_id < 0) return;
        diar_history.push_back({s.speaker_id, s.start_s, s.end_s});
        const double cutoff = s.end_s - speaker_history_s;
        while (!diar_history.empty() && diar_history.front().end_s < cutoff) {
            diar_history.pop_front();
        }
        if (s.speaker_id != last_spk_seen) {
            last_spk_seen = s.speaker_id;
            std::fprintf(stderr, "[diar] active speaker_%d at %.2fs\n",
                         s.speaker_id, s.start_s);
        }
    };

    bool   line_open         = false;
    int    line_speaker_id   = -2;
    double last_voice_end_s  = 0.0;

    auto close_line_if_open = [&]() {
        if (line_open) {
            std::fputc('\n', stdout);
            std::fflush(stdout);
            line_open = false;
        }
    };

    auto on_tx = [&](const StreamingSegment & seg) {
        // EOU models: `is_eou_boundary` when `<EOU>` fired (same semantics as live-mic).
        if (!args.accumulate) {
            const int spk = speaker_for_range(diar_history, seg.start_s, seg.end_s);
            if (!seg.text.empty()) {
                if (spk >= 0) {
                    std::printf("[%.2f-%.2f] speaker_%d:%s\n",
                                seg.start_s, seg.end_s, spk, seg.text.c_str());
                } else {
                    std::printf("[%.2f-%.2f] speaker_?:%s\n",
                                seg.start_s, seg.end_s, seg.text.c_str());
                }
            }
            if (seg.is_eou_boundary) {
                if (spk >= 0) {
                    std::printf("[%.2f-%.2f] speaker_%d:  <EOU>  end of user turn\n",
                                seg.start_s, seg.end_s, spk);
                } else {
                    std::printf("[%.2f-%.2f] speaker_?:  <EOU>  end of user turn\n",
                                seg.start_s, seg.end_s);
                }
            }
            if (!seg.text.empty() || seg.is_eou_boundary) {
                std::fflush(stdout);
            }
            return;
        }

        const int spk = (seg.text.empty() && !seg.is_eou_boundary)
                          ? line_speaker_id
                          : speaker_for_range(diar_history, seg.start_s, seg.end_s);

        if (!seg.text.empty()) {
            if (line_open && spk != line_speaker_id) {
                close_line_if_open();
            }

            if (!line_open) {
                if (spk >= 0) {
                    std::printf("speaker_%d:%s",
                                spk,
                                seg.text.c_str() + (seg.text.front() == ' ' ? 1 : 0));
                } else {
                    std::printf("speaker_?:%s",
                                seg.text.c_str() + (seg.text.front() == ' ' ? 1 : 0));
                }
                line_speaker_id = spk;
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
            line_speaker_id = -1;
            return;
        }

        if (seg.text.empty() && line_open &&
            (seg.end_s - last_voice_end_s) * 1000.0 >= args.silence_flush_ms) {
            close_line_if_open();
        }
    };

    StreamingOptions tx_sopts;
    tx_sopts.sample_rate        = 16000;
    tx_sopts.chunk_ms           = args.asr_chunk_ms;
    tx_sopts.left_context_ms    = args.asr_left_ms;
    tx_sopts.right_lookahead_ms = args.asr_right_ms;
    auto tx_sess = asr_engine.stream_start(tx_sopts, on_tx);

    SortformerStreamingOptions diar_sopts;
    diar_sopts.sample_rate    = 16000;
    diar_sopts.chunk_ms       = args.diar_chunk_ms;
    diar_sopts.history_ms     = std::max(args.diar_history_ms, args.diar_chunk_ms);
    diar_sopts.threshold      = 0.5f;
    diar_sopts.min_segment_ms = 200;
    auto diar_sess = diar_engine.diarize_start(diar_sopts, on_diar);

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
        std::fprintf(stderr, "[live-mic-attributed] capture device: [%d] %s\n",
                     args.device_index, capture_info.name);
    } else {
        std::fprintf(stderr, "[live-mic-attributed] capture device: <system default>\n");
    }

    ma_device_config dcfg = ma_device_config_init(ma_device_type_capture);
    dcfg.capture.pDeviceID = chosen_ptr;
    dcfg.capture.format    = ma_format_f32;
    dcfg.capture.channels  = 1;
    dcfg.sampleRate        = 16000;
    dcfg.dataCallback      = data_callback;

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

    std::fprintf(stderr,
        "[live-mic-attributed] listening at 16 kHz mono.\n"
        "  asr  chunk=%d ms  left=%d ms  right=%d ms\n"
        "  diar chunk=%d ms  history=%d ms  speaker_history=%d ms\n"
        "Speak, Ctrl-C to stop.\n\n",
        args.asr_chunk_ms, args.asr_left_ms, args.asr_right_ms,
        args.diar_chunk_ms, args.diar_history_ms, args.speaker_history_ms);

    while (!g_stop.load()) {
        std::vector<float> batch;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_cv.wait_for(lk, std::chrono::milliseconds(100),
                          [] { return !g_pending.empty() || g_stop.load(); });
            if (g_pending.empty()) continue;
            batch.swap(g_pending);
        }
        diar_sess->feed_pcm_f32(batch.data(), (int) batch.size());
        tx_sess->feed_pcm_f32(batch.data(), (int) batch.size());
    }

    std::fprintf(stderr, "\n[live-mic-attributed] stopping...\n");
    ma_device_stop(&device);

    {
        std::vector<float> tail;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            tail.swap(g_pending);
        }
        if (!tail.empty()) {
            diar_sess->feed_pcm_f32(tail.data(), (int) tail.size());
            tx_sess->feed_pcm_f32(tail.data(), (int) tail.size());
        }
    }

    diar_sess->finalize();
    tx_sess->finalize();

    if (args.accumulate) close_line_if_open();

    ma_device_uninit(&device);
    ma_context_uninit(&ctx);
    return 0;
}
