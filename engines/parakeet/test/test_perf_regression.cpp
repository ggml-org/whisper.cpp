// Perf regression smoke for repeated Engine transcribe/diarize calls.
//
// Asserts stable transcript or Sortformer fingerprint, encoder cache reuse, and
// optional timing ceilings (see --help).
//
// Usage:
//   test-perf-regression --model <gguf> --wav <wav> [options]
//
// Exit 0 on success; non-zero on regression or invalid arguments.

#include "parakeet/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    std::string expected_text;
    int n_runs = 6;
    int n_warmup = 2;
    int n_threads = 0;
    int n_gpu_layers = 0;
    double max_enc_ms = 0.0;
    double cache_hit_ratio_max = 1.10; // warm enc_ms must be ≤ 1.10x median
    // Round-3 additions: catch FFT-specific regressions (mel jumps
    // from ~2.8 ms back toward ~5.5 ms if the real-FFT path breaks)
    // and Adreno cold-start regressions (warmup_1 enc_ms ≫ steady-
    // state median if the kernel binary cache patch regresses or
    // `clBuildProgram` hot path resurfaces).
    double max_mel_ms = 0.0;
    double max_cold_overhead_ratio = 0.0; // 0 = disabled
    // Pass-through to EngineOptions::prewarm. With
    // prewarm on, the constructor pays the cold-graph-build cost so
    // warmup_1 should be in the warm steady-state band; gate
    // `--max-cold-overhead-ratio` is the suggested validation.
    bool   prewarm                = false;
    float  prewarm_audio_seconds  = 1.0f;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav> [opts]\n"
        "\n"
        "  --model PATH         CTC / TDT / EOU / Sortformer GGUF\n"
        "  --wav PATH           16 kHz mono wav\n"
        "  --expect TEXT        expected transcript (asserted byte-equal). Optional;\n"
        "                       default: parity with the first run.\n"
        "                       Ignored on Sortformer GGUFs (segment fingerprint\n"
        "                       compared instead of text).\n"
        "  --runs N             number of timed runs (default 6)\n"
        "  --warmup N           warmup runs not counted in stats (default 2)\n"
        "  --threads N          CPU threads (0 = HW concurrency)\n"
        "  --n-gpu-layers N     pass-through to Engine\n"
        "  --max-enc-ms F       fail if any timed run's encoder_ms exceeds this\n"
        "  --cache-hit-ratio F  fail if any timed encoder_ms exceeds median*F (default 1.10)\n"
        "  --max-mel-ms F       fail if any timed run's mel_ms exceeds this (catches\n"
        "                       FFT regressions specifically; mel runs host-side on\n"
        "                       every backend, so the budget is the same on Adreno)\n"
        "  --max-cold-overhead-ratio F  fail if first warmup's enc_ms exceeds median*F.\n"
        "                       0 = disabled (default). Useful on Adreno + GGML_OPENCL_CACHE_DIR\n"
        "                       to assert the kernel binary cache cuts cold-start to within\n"
        "                       F\u00d7 of warm; suggested 1.5\u20132.0x when the cache is warm\n"
        "                       across processes; ~1.10x with --prewarm because the cold\n"
        "                       cost has been paid in the Engine constructor.\n"
        "  --prewarm            pass-through to EngineOptions::prewarm. Engine constructor\n"
        "                       runs one synthetic forward pass through the encoder so the\n"
        "                       graph build / GPU pipeline compile cost is paid up front\n"
        "                       instead of in warmup_1.\n"
        "  --prewarm-audio-seconds F  length of the synthetic mel input used by prewarm.\n"
        "                       Default 1.0.\n",
        argv0);
}

}

int main(int argc, char ** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model" && i + 1 < argc)              o.model_path = argv[++i];
        else if (a == "--wav"   && i + 1 < argc)              o.wav_path   = argv[++i];
        else if (a == "--expect" && i + 1 < argc)             o.expected_text = argv[++i];
        else if (a == "--runs" && i + 1 < argc)               o.n_runs = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc)             o.n_warmup = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)            o.n_threads = std::atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc)       o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--max-enc-ms" && i + 1 < argc)         o.max_enc_ms = std::atof(argv[++i]);
        else if (a == "--cache-hit-ratio" && i + 1 < argc)    o.cache_hit_ratio_max = std::atof(argv[++i]);
        else if (a == "--max-mel-ms" && i + 1 < argc)         o.max_mel_ms = std::atof(argv[++i]);
        else if (a == "--max-cold-overhead-ratio" && i + 1 < argc) o.max_cold_overhead_ratio = std::atof(argv[++i]);
        else if (a == "--prewarm")                            o.prewarm = true;
        else if (a == "--prewarm-audio-seconds" && i + 1 < argc) o.prewarm_audio_seconds = (float) std::atof(argv[++i]);
        else if (a == "--help" || a == "-h")                  { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (o.model_path.empty() || o.wav_path.empty()) { usage(argv[0]); return 2; }

    parakeet::EngineOptions eopts;
    eopts.model_gguf_path        = o.model_path;
    eopts.n_threads              = o.n_threads;
    eopts.n_gpu_layers           = o.n_gpu_layers;
    eopts.verbose                = false;
    eopts.prewarm                = o.prewarm;
    eopts.prewarm_audio_seconds  = o.prewarm_audio_seconds;

    const auto t_ctor = std::chrono::steady_clock::now();
    parakeet::Engine engine(eopts);
    const double ctor_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - t_ctor).count() / 1000.0;
    std::fprintf(stderr, "[test-perf-regression] engine ctor=%.2fms (prewarm=%s)\n",
                 ctor_ms, o.prewarm ? "on" : "off");

    const bool is_sortformer = engine.is_diarization_model();
    if (is_sortformer) {
        std::fprintf(stderr, "[test-perf-regression] dispatching to Engine::diarize for Sortformer GGUF\n");
    }

    std::string reference_text;             // transcribe path
    std::string reference_segments_fingerprint;  // diarize path: "<n_segs>:<n_floats>:<sum_speaker_ids>:<max_end_s_x100>"
    std::vector<double> enc_ms;
    std::vector<double> mel_ms;
    enc_ms.reserve(o.n_runs);
    mel_ms.reserve(o.n_runs);

    // Track the first warmup's enc_ms separately for the cold-start
    // overhead gate (--max-cold-overhead-ratio). On Adreno + warm
    // GGML_OPENCL_CACHE_DIR + --prewarm this should be close to
    // median; on a cold cache without --prewarm it'll be the multi-
    // second outlier the ggml-opencl-program-binary-cache.patch is
    // meant to eliminate.
    double first_warmup_enc_ms = -1.0;

    // For Sortformer, build a deterministic "transcript-equivalent"
    // fingerprint from segments + speaker_probs so the determinism
    // gate (every run identical) catches numerical drift the same
    // way the transcribe-path text comparison does.
    auto sortformer_fingerprint =
        [](const parakeet::DiarizationResult & r) -> std::string {
            // (n_segments, n_speaker_probs_floats, sum_of_speaker_ids,
            //  max_end_s × 100 rounded; the *100 keeps two decimals
            //  without making it locale-dependent.)
            int    n_segs        = (int) r.segments.size();
            size_t n_probs       = r.speaker_probs.size();
            int    sum_spk       = 0;
            int    max_end_x100  = 0;
            for (const auto & s : r.segments) {
                sum_spk     += s.speaker_id;
                const int e = (int) std::llround(s.end_s * 100.0);
                if (e > max_end_x100) max_end_x100 = e;
            }
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%d:%zu:%d:%d",
                          n_segs, n_probs, sum_spk, max_end_x100);
            return std::string(buf);
        };

    const auto run_once = [&](int idx, bool is_warmup) {
        const auto t0 = std::chrono::steady_clock::now();
        double enc, mel;
        std::string text_or_fingerprint;
        if (is_sortformer) {
            auto res = engine.diarize(o.wav_path);
            const double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - t0).count() / 1000.0;
            enc = res.encoder_ms > 0.0 ? res.encoder_ms : total_ms;
            mel = res.preprocess_ms;
            text_or_fingerprint = sortformer_fingerprint(res);
            std::fprintf(stderr,
                "[test-perf-regression] %s %d/%d  mel=%.2fms enc=%.2fms total=%.2fms diar_fp=%s\n",
                is_warmup ? "warmup" : "run", idx, is_warmup ? o.n_warmup : o.n_runs,
                mel, enc, total_ms, text_or_fingerprint.c_str());
            if (idx == 1 && is_warmup) {
                reference_segments_fingerprint = text_or_fingerprint;
                first_warmup_enc_ms = enc;
            } else if (text_or_fingerprint != reference_segments_fingerprint) {
                std::fprintf(stderr,
                    "[test-perf-regression] FAIL: non-deterministic diarize fingerprint on run %d\n"
                    "  reference: \"%s\"\n  got      : \"%s\"\n",
                    idx, reference_segments_fingerprint.c_str(), text_or_fingerprint.c_str());
                std::exit(1);
            }
        } else {
            auto res = engine.transcribe(o.wav_path);
            const double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - t0).count() / 1000.0;
            enc = res.encoder_ms > 0.0 ? res.encoder_ms : total_ms;
            mel = res.preprocess_ms;
            std::fprintf(stderr, "[test-perf-regression] %s %d/%d  mel=%.2fms enc=%.2fms total=%.2fms\n",
                         is_warmup ? "warmup" : "run", idx, is_warmup ? o.n_warmup : o.n_runs,
                         mel, enc, total_ms);
            if (idx == 1 && is_warmup) {
                reference_text = res.text;
                first_warmup_enc_ms = enc;
                if (!o.expected_text.empty() && res.text != o.expected_text) {
                    std::fprintf(stderr,
                        "[test-perf-regression] FAIL: transcript mismatch\n"
                        "  expected: \"%s\"\n  got     : \"%s\"\n",
                        o.expected_text.c_str(), res.text.c_str());
                    std::exit(1);
                }
            } else if (res.text != reference_text) {
                std::fprintf(stderr,
                    "[test-perf-regression] FAIL: non-deterministic transcript on run %d\n"
                    "  reference: \"%s\"\n  got      : \"%s\"\n",
                    idx, reference_text.c_str(), res.text.c_str());
                std::exit(1);
            }
        }
        if (!is_warmup) {
            enc_ms.push_back(enc);
            mel_ms.push_back(mel);
        }
    };

    for (int i = 1; i <= o.n_warmup; ++i) run_once(i, true);
    for (int i = 1; i <= o.n_runs;   ++i) run_once(i, false);

    if (enc_ms.empty()) {
        std::fprintf(stderr, "[test-perf-regression] FAIL: no timed runs\n");
        return 1;
    }

    auto stats = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return std::tuple<double, double, double, double>(
            std::accumulate(v.begin(), v.end(), 0.0) / v.size(),  // mean
            v[v.size() / 2],                                       // median
            v.front(),                                             // min
            v.back());                                             // max
    };
    auto [enc_mean, enc_median, enc_min, enc_max] = stats(enc_ms);
    auto [mel_mean, mel_median, mel_min, mel_max] = stats(mel_ms);

    std::fprintf(stderr,
        "[test-perf-regression] enc_ms summary  mean=%.2f  median=%.2f  min=%.2f  max=%.2f\n",
        enc_mean, enc_median, enc_min, enc_max);
    std::fprintf(stderr,
        "[test-perf-regression] mel_ms summary  mean=%.2f  median=%.2f  min=%.2f  max=%.2f\n",
        mel_mean, mel_median, mel_min, mel_max);

    if (o.max_enc_ms > 0.0 && enc_max > o.max_enc_ms) {
        std::fprintf(stderr,
            "[test-perf-regression] FAIL: max enc_ms %.2f > ceiling %.2f\n",
            enc_max, o.max_enc_ms);
        return 1;
    }
    if (o.max_mel_ms > 0.0 && mel_max > o.max_mel_ms) {
        std::fprintf(stderr,
            "[test-perf-regression] FAIL: max mel_ms %.2f > ceiling %.2f "
            "(FFT regression in mel preprocess?)\n",
            mel_max, o.max_mel_ms);
        return 1;
    }
    const double ratio = enc_max / enc_median;
    if (ratio > o.cache_hit_ratio_max) {
        std::fprintf(stderr,
            "[test-perf-regression] FAIL: max/median = %.2fx > %.2fx (cache miss in steady state?)\n",
            ratio, o.cache_hit_ratio_max);
        return 1;
    }
    if (o.max_cold_overhead_ratio > 0.0 && first_warmup_enc_ms > 0.0) {
        const double cold_ratio = first_warmup_enc_ms / enc_median;
        if (cold_ratio > o.max_cold_overhead_ratio) {
            std::fprintf(stderr,
                "[test-perf-regression] FAIL: warmup_1/median = %.2fx > %.2fx "
                "(kernel binary cache regressed? GGML_OPENCL_CACHE_DIR not set?)\n",
                cold_ratio, o.max_cold_overhead_ratio);
            return 1;
        }
        std::fprintf(stderr,
            "[test-perf-regression] cold-overhead: warmup_1=%.2fms median=%.2fms ratio=%.2fx (limit %.2fx)\n",
            first_warmup_enc_ms, enc_median, cold_ratio, o.max_cold_overhead_ratio);
    }

    if (is_sortformer) {
        std::fprintf(stderr, "[test-perf-regression] PASS  diarize fingerprint: \"%s\"\n",
                     reference_segments_fingerprint.c_str());
    } else {
        std::fprintf(stderr, "[test-perf-regression] PASS  transcript: \"%s\"\n",
                     reference_text.c_str());
    }
    return 0;
}
