// Streaming API parity: Mode 2 vs Mode 1 and Mode 3 error handling.
//
// Usage:
//   test-streaming --model <parakeet-ctc.gguf> --wav <input.wav> [--threads N] [--n-gpu-layers N] [--verbose]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet/engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace parakeet;

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    int  n_gpu_layers = 0;
    int  n_threads    = 0;
    bool verbose      = false;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <parakeet-ctc.gguf> --wav <input.wav> [options]\n"
        "\n"
        "validates Mode 2 (transcribe_stream) byte-equality vs Mode 1 (transcribe)\n"
        "across a range of chunk sizes, plus the Mode 3 (stream_start) error path.\n"
        "\n"
        "options:\n"
        "  --n-gpu-layers N     offload to GPU backend when > 0\n"
        "  --threads N          CPU threads (0 = hardware_concurrency)\n"
        "  --verbose            print per-test segment counts and timings\n",
        argv0);
}

int parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) o.model_path = argv[++i];
        else if (a == "--wav" && i + 1 < argc) o.wav_path = argv[++i];
        else if (a == "--n-gpu-layers" && i + 1 < argc) o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) o.n_threads = std::atoi(argv[++i]);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (o.model_path.empty() || o.wav_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    return 0;
}

bool approx_equal(double a, double b, double tol_ms) {
    return std::abs(a - b) * 1000.0 <= tol_ms;
}

}

int main(int argc, char ** argv) {
    Opts opts;
    if (int rc = parse_args(argc, argv, opts); rc != 0) return rc;

    EngineOptions eopts;
    eopts.model_gguf_path = opts.model_path;
    eopts.n_gpu_layers    = opts.n_gpu_layers;
    eopts.n_threads       = opts.n_threads;
    eopts.verbose         = opts.verbose;

    std::fprintf(stderr, "[test-streaming] loading %s\n", opts.model_path.c_str());
    Engine engine(eopts);

    std::fprintf(stderr, "[test-streaming] Mode 1 reference: transcribe(%s)\n",
                 opts.wav_path.c_str());
    EngineResult ref = engine.transcribe(opts.wav_path);
    std::fprintf(stderr,
                 "[test-streaming] Mode 1: %d encoder frames, %zu tokens, %.1f ms total\n"
                 "[test-streaming] Mode 1 text: \"%.120s%s\"\n",
                 ref.encoder_frames, ref.token_ids.size(), ref.total_ms,
                 ref.text.c_str(), ref.text.size() > 120 ? "..." : "");

    const int audio_ms = (int) (1000.0 * (double) ref.audio_samples / (double) ref.sample_rate);
    const std::vector<int> chunk_ms_list = {250, 500, 1000, 2000, 4000, audio_ms};

    int failures = 0;
    for (int chunk_ms : chunk_ms_list) {
        if (chunk_ms <= 0) continue;
        StreamingOptions sopts;
        sopts.sample_rate = ref.sample_rate;
        sopts.chunk_ms    = chunk_ms;

        int n_segments = 0;
        double last_end_s = 0.0;
        bool timestamps_ok = true;
        std::string concat_text;

        EngineResult stream_result = engine.transcribe_stream(
            opts.wav_path, sopts,
            [&](const StreamingSegment & seg) {
                if (seg.chunk_index != n_segments) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: chunk_index jump "
                        "(expected %d, got %d)\n",
                        chunk_ms, n_segments, seg.chunk_index);
                    timestamps_ok = false;
                }
                if (!approx_equal(seg.start_s, last_end_s, 1.0)) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: seg %d start_s=%.3f "
                        "does not match previous end_s=%.3f\n",
                        chunk_ms, seg.chunk_index, seg.start_s, last_end_s);
                    timestamps_ok = false;
                }
                if (seg.end_s <= seg.start_s) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: seg %d has end_s=%.3f <= start_s=%.3f\n",
                        chunk_ms, seg.chunk_index, seg.end_s, seg.start_s);
                    timestamps_ok = false;
                }
                if (!seg.is_final) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: seg %d is_final=false "
                        "(Phase 1 Mode 2 must emit final segments only)\n",
                        chunk_ms, seg.chunk_index);
                    timestamps_ok = false;
                }
                concat_text += seg.text;
                last_end_s = seg.end_s;
                ++n_segments;
            });

        const double audio_s = (double) ref.audio_samples / (double) ref.sample_rate;
        if (!approx_equal(last_end_s, audio_s, 100.0)) {
            std::fprintf(stderr,
                "[test-streaming] WARN chunk_ms=%d: last end_s=%.3f differs "
                "from audio duration=%.3f (tolerance 100ms for frame-stride rounding)\n",
                chunk_ms, last_end_s, audio_s);
        }

        if (concat_text != ref.text) {
            std::fprintf(stderr,
                "[test-streaming] FAIL chunk_ms=%d: concatenated segment text "
                "does not match Mode 1 reference\n", chunk_ms);
            std::fprintf(stderr, "  ref:    \"%.200s%s\"\n",
                         ref.text.c_str(), ref.text.size() > 200 ? "..." : "");
            std::fprintf(stderr, "  stream: \"%.200s%s\"\n",
                         concat_text.c_str(), concat_text.size() > 200 ? "..." : "");
            ++failures;
        } else if (!timestamps_ok) {
            ++failures;
        } else {
            std::fprintf(stderr,
                "[test-streaming] PASS chunk_ms=%4d: %3d segments, %.1f ms stream total, "
                "text byte-equal\n",
                chunk_ms, n_segments, stream_result.total_ms);
        }
    }

    do     {
        const bool is_tdt = (engine.model_type() == "tdt");
        std::vector<float> pcm;
        {
            FILE * f = std::fopen(opts.wav_path.c_str(), "rb");
            if (!f) { std::fprintf(stderr, "[test-streaming] FAIL Mode 3: cannot open wav\n"); ++failures; break; }
            std::fseek(f, 0, SEEK_END);
            long sz = std::ftell(f);
            std::fseek(f, 44, SEEK_SET);
            std::vector<int16_t> i16((sz - 44) / 2);
            std::fread(i16.data(), 2, i16.size(), f);
            std::fclose(f);
            pcm.resize(i16.size());
            constexpr float inv = 1.0f / 32768.0f;
            for (size_t i = 0; i < i16.size(); ++i) pcm[i] = i16[i] * inv;
        }

        const struct ModeCfg { int chunk_ms; int left_ms; int right_ms; int max_rel_wer_pct; int max_rel_wer_pct_tdt; } configs[] = {
            {1000, 2000,  500,  5, 40},
            {2000, 2000, 1000,  5,  5},
            {2000, 5000, 2000,  5,  5},
        };

        for (const auto & c : configs) {
            StreamingOptions sopts;
            sopts.sample_rate       = 16000;
            sopts.chunk_ms          = c.chunk_ms;
            sopts.left_context_ms   = c.left_ms;
            sopts.right_lookahead_ms= c.right_ms;

            int n_segments = 0;
            double last_end_s = 0.0;
            std::string concat_text;
            bool ordering_ok = true;

            auto sess = engine.stream_start(sopts,
                [&](const StreamingSegment & seg) {
                    if (seg.chunk_index != n_segments) ordering_ok = false;
                    if (seg.end_s < seg.start_s)       ordering_ok = false;
                    concat_text += seg.text;
                    last_end_s = seg.end_s;
                    ++n_segments;
                });

            unsigned rng = 0xDEADBEEFu;
            size_t i = 0;
            while (i < pcm.size()) {
                rng = rng * 1103515245u + 12345u;
                size_t burst = 512 + (rng % 3500);
                if (i + burst > pcm.size()) burst = pcm.size() - i;
                sess->feed_pcm_f32(pcm.data() + i, (int) burst);
                i += burst;
            }
            sess->finalize();

            const double audio_s = (double) ref.audio_samples / 16000.0;

            auto words = [](const std::string & s) {
                std::vector<std::string> out; std::string cur;
                for (char c : s) { if (c == ' ' || c == '\t' || c == '\n') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } } else cur.push_back(c); }
                if (!cur.empty()) out.push_back(cur);
                return out;
            };
            auto ref_words = words(ref.text);
            auto hyp_words = words(concat_text);
            std::vector<std::vector<int>> d(ref_words.size() + 1, std::vector<int>(hyp_words.size() + 1, 0));
            for (size_t r = 0; r <= ref_words.size(); ++r) d[r][0] = (int) r;
            for (size_t h = 0; h <= hyp_words.size(); ++h) d[0][h] = (int) h;
            for (size_t r = 1; r <= ref_words.size(); ++r) {
                for (size_t h = 1; h <= hyp_words.size(); ++h) {
                    int cost = ref_words[r-1] == hyp_words[h-1] ? 0 : 1;
                    d[r][h] = std::min({d[r-1][h] + 1, d[r][h-1] + 1, d[r-1][h-1] + cost});
                }
            }
            double wer = ref_words.empty() ? 0.0 :
                         100.0 * d[ref_words.size()][hyp_words.size()] / (double) ref_words.size();

            const int tol = is_tdt ? c.max_rel_wer_pct_tdt : c.max_rel_wer_pct;
            if (!ordering_ok) {
                std::fprintf(stderr, "[test-streaming] FAIL Mode 3 chunk=%d left=%d right=%d: "
                                     "callback ordering / timestamps broken\n",
                             c.chunk_ms, c.left_ms, c.right_ms);
                ++failures;
            } else if (wer > tol) {
                std::fprintf(stderr, "[test-streaming] FAIL Mode 3 chunk=%d left=%d right=%d: "
                                     "WER %.2f%% exceeds tolerance %d%% (%s)\n"
                                     "  ref:    \"%.200s%s\"\n"
                                     "  stream: \"%.200s%s\"\n",
                             c.chunk_ms, c.left_ms, c.right_ms, wer, tol,
                             is_tdt ? "tdt" : "ctc",
                             ref.text.c_str(), ref.text.size() > 200 ? "..." : "",
                             concat_text.c_str(), concat_text.size() > 200 ? "..." : "");
                ++failures;
            } else {
                std::fprintf(stderr,
                    "[test-streaming] PASS Mode 3 chunk=%4d left=%5d right=%5d: "
                    "%3d segments, end_s=%.2f audio=%.2fs, WER=%.2f%% (tol %d%% %s)\n",
                    c.chunk_ms, c.left_ms, c.right_ms,
                    n_segments, last_end_s, audio_s, wer, tol,
                    is_tdt ? "tdt" : "ctc");
            }
        }

        {
            StreamingOptions sopts;
            sopts.sample_rate       = 16000;
            sopts.chunk_ms          = 2000;
            sopts.left_context_ms   = 2000;
            sopts.right_lookahead_ms= 1000;
            int callbacks = 0;
            auto sess = engine.stream_start(sopts,
                [&](const StreamingSegment &) { ++callbacks; });
            if (!pcm.empty()) {
                sess->feed_pcm_f32(pcm.data(), (int) std::min<size_t>(pcm.size(), 8000));
            }
            sess->cancel();
            std::fprintf(stderr,
                "[test-streaming] PASS Mode 3 cancel: cancelled after %d callback(s)\n", callbacks);
        }

        // Phase 13 -- opt-in energy-VAD on CTC/TDT. Default is off; flip
        // it on, feed the wav, assert at least one Speaking transition.
        {
            StreamingOptions sopts;
            sopts.sample_rate        = 16000;
            sopts.chunk_ms           = 2000;
            sopts.left_context_ms    = 2000;
            sopts.right_lookahead_ms = 1000;
            sopts.enable_energy_vad  = true;
            int n_vad_events      = 0;
            int n_speaking_events = 0;
            sopts.on_event = [&](const StreamEvent & ev) {
                if (ev.type == StreamEventType::VadStateChanged) {
                    ++n_vad_events;
                    if (ev.vad_state == VadState::Speaking) ++n_speaking_events;
                }
            };
            auto sess = engine.stream_start(sopts,
                [&](const StreamingSegment &) { });
            unsigned rng = 0xCAFEBABEu;
            size_t i = 0;
            while (i < pcm.size()) {
                rng = rng * 1103515245u + 12345u;
                size_t burst = 512 + (rng % 3500);
                if (i + burst > pcm.size()) burst = pcm.size() - i;
                sess->feed_pcm_f32(pcm.data() + i, (int) burst);
                i += burst;
            }
            sess->finalize();
            if (n_speaking_events == 0) {
                std::fprintf(stderr,
                    "[test-streaming] FAIL energy-VAD: no Speaking events fired on a "
                    "wav with audible speech (got %d total VadStateChanged events)\n",
                    n_vad_events);
                ++failures;
            } else {
                std::fprintf(stderr,
                    "[test-streaming] PASS energy-VAD: %d VadStateChanged events "
                    "(%d Speaking transitions)\n",
                    n_vad_events, n_speaking_events);
            }
        }

    } while (0);

    if (failures == 0) {
        std::fprintf(stderr, "[test-streaming] all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "[test-streaming] %d check(s) failed\n", failures);
    return 1;
}
