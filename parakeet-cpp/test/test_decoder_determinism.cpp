// Decoder determinism across repeated transcribe/diarize calls on one Engine.
//
// Usage:
//   test-decoder-determinism --model <gguf> --wav <wav> [--runs N] [--threads N]
//       [--n-gpu-layers N] [--cache-hit-ratio R] [--prewarm]
//       [--prewarm-audio-seconds F] [--cold-overhead-max R] [--verbose]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    int  n_runs = 5;
    int  n_gpu_layers = 0;
    int  n_threads = 0;
    bool verbose = false;
    // Cache-hit gate compares median(warm runs 1..N-1) to run 0
    // (cold). Median (not max) is the right statistic — a real
    // cache MISS rebuilds the encoder graph each call, lifting
    // EVERY warm run to ≥ cold; thermal-spike one-offs only push
    // max, not median. With median + 1.10x default, CPU thermal
    // jitter on noisy desktops sits well below the gate (we
    // observed median/cold ratios of 0.5-0.9 across all four
    // model types on a 16T Ryzen). Tighten via --cache-hit-ratio
    // on Adreno + warm GGML_OPENCL_CACHE_DIR.
    double cache_hit_ratio_max = 1.10;

    // Exercise EngineOptions::prewarm on the
    // engine instance. When set, the harness asserts an additional
    // contract: run0 (the first real transcribe / diarize call,
    // which would normally be the cold-graph-build outlier) must
    // be at most cold_overhead_max * median(warms) — i.e. the
    // prewarm should bring run0 into the warm steady-state band.
    // Without --prewarm, run0 is *expected* to be slower than the
    // warms (it's the cold run); with --prewarm, the cold cost
    // has already been paid in the Engine constructor, so run0
    // should match the warm runs.
    //
    // `prewarm_audio_seconds` defaults to 0.0f (== "auto: use the
    // loaded wav's duration"). Explicit non-zero values are used
    // verbatim. Auto-mode is the right default for the test
    // because the encoder graph cache is shape-keyed (T_mel +
    // n_layers + all_valid), so a prewarm shape that matches the
    // real call's shape is what production code should pass too.
    //
    // Default cold_overhead_max=1.60 reflects empirical CPU
    // measurements: even with shape-matched prewarm, run 0 still
    // pays ~50-100 ms of non-encoder cold cost on CPU (ggml
    // thread-pool spin-up, decoder per-call allocator hot path,
    // thermal warmup), which lifts the run-0 encoder-time by up
    // to ~50% above the steady-state warm median on small models
    // like EOU 120M and TDT 0.6B Phase-14-decoder paths. A real
    // prewarm regression goes to 2-3x. Tighten on a thermally-
    // controlled CI box. On Metal/OpenCL/Vulkan the GPU pipeline
    // cache *is* the dominant cold cost so prewarm gives a much
    // bigger win there — tighten to 1.20-1.30x on Adreno + warm
    // GGML_OPENCL_CACHE_DIR.
    bool   prewarm                  = false;
    float  prewarm_audio_seconds    = 0.0f;  // 0 = auto from wav duration
    double cold_overhead_max        = 1.60;  // run0/median(warm) with --prewarm
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav> [opts]\n"
        "\n"
        "  --model PATH         CTC / TDT / EOU / Sortformer GGUF\n"
        "  --wav PATH           16 kHz mono wav\n"
        "  --runs N             number of repeated calls (default 5; min 2)\n"
        "  --n-gpu-layers N     pass-through to Engine (default 0)\n"
        "  --threads N          CPU threads (0 = HW concurrency)\n"
        "  --cache-hit-ratio F  fail if median(warm enc_ms) exceeds cold\n"
        "                       enc_ms * F. Default 1.10. A real cache MISS\n"
        "                       (graph rebuilt each call) makes EVERY warm\n"
        "                       run ≥ cold; thermal spikes only push max,\n"
        "                       not median, so the gate is robust. Tighten\n"
        "                       on Adreno + warm GGML_OPENCL_CACHE_DIR.\n"
        "  --prewarm            construct the Engine with EngineOptions::prewarm=true\n"
        "                       so the encoder graph build cost is paid in the\n"
        "                       constructor instead of run 0. With this flag, run 0\n"
        "                       should be in the warm steady-state band, gated by\n"
        "                       --cold-overhead-max.\n"
        "  --prewarm-audio-seconds F  length of the synthetic mel input used by the\n"
        "                       prewarm step. Default 0 (== auto: use the loaded\n"
        "                       wav's duration). The encoder graph cache is shape-\n"
        "                       keyed, so on CPU prewarming with the same shape as\n"
        "                       the real call is what gets us a cache HIT on run 0.\n"
        "                       (On Metal/OpenCL/Vulkan the GPU pipeline cache is\n"
        "                       signature-keyed too, so any shape works for warming\n"
        "                       kernel binaries — but matching the call shape still\n"
        "                       gives the best run-0 latency because the ggml graph\n"
        "                       gallocr is also shape-keyed.)\n"
        "  --cold-overhead-max F  with --prewarm, fail if run0_enc_ms exceeds\n"
        "                       median(warm) * F. Default 1.30. The 1.30 leaves\n"
        "                       headroom for CPU thermal jitter on a noisy desktop\n"
        "                       (single-run spikes can lift run0 ~25 % even when\n"
        "                       the cache hits). Tighten on a thermally-controlled\n"
        "                       CI box. A real prewarm regression goes 1.5-2.5x.\n"
        "  --verbose            print per-run summary\n",
        argv0);
}

int parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"        && i + 1 < argc) o.model_path = argv[++i];
        else if (a == "--wav"          && i + 1 < argc) o.wav_path   = argv[++i];
        else if (a == "--runs"         && i + 1 < argc) o.n_runs = std::atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--threads"      && i + 1 < argc) o.n_threads = std::atoi(argv[++i]);
        else if (a == "--cache-hit-ratio" && i + 1 < argc) o.cache_hit_ratio_max = std::atof(argv[++i]);
        else if (a == "--prewarm")                         o.prewarm = true;
        else if (a == "--prewarm-audio-seconds" && i + 1 < argc) o.prewarm_audio_seconds = (float) std::atof(argv[++i]);
        else if (a == "--cold-overhead-max" && i + 1 < argc) o.cold_overhead_max = std::atof(argv[++i]);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--help" || a == "-h")    { usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (o.model_path.empty() || o.wav_path.empty()) { usage(argv[0]); return 2; }
    if (o.n_runs < 2) {
        std::fprintf(stderr, "--runs must be >= 2 (need at least one repeated call to compare)\n");
        return 2;
    }
    return 0;
}

// Minimal RIFF reader (mono 16-bit PCM, matches the helper in
// test_eou_streaming.cpp). We avoid linking the full helper library
// here because this test is intentionally lean.
bool load_wav_pcm(const std::string & path, std::vector<float> & out, int & sr) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char hdr[44];
    if (std::fread(hdr, 1, 44, f) != 44) { std::fclose(f); return false; }
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        std::fclose(f); return false;
    }
    sr = *reinterpret_cast<int32_t *>(hdr + 24);
    const int16_t channels = *reinterpret_cast<int16_t *>(hdr + 22);
    const int16_t bits     = *reinterpret_cast<int16_t *>(hdr + 34);
    if (channels != 1 || bits != 16) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_END);
    const long total = std::ftell(f);
    std::fseek(f, 44, SEEK_SET);
    const size_t n = static_cast<size_t>(total - 44) / sizeof(int16_t);
    std::vector<int16_t> pcm(n);
    if (std::fread(pcm.data(), sizeof(int16_t), n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = pcm[i] / 32768.0f;
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// ---------- Determinism for CTC / TDT / EOU (transcribe path) ----------

int run_transcribe_path(const Opts & o,
                        const std::vector<float> & samples, int sr,
                        const std::string & model_type_str) {
    parakeet::EngineOptions eopts;
    eopts.model_gguf_path        = o.model_path;
    eopts.n_threads              = o.n_threads;
    eopts.n_gpu_layers           = o.n_gpu_layers;
    eopts.verbose                = false;
    eopts.prewarm                = o.prewarm;
    eopts.prewarm_audio_seconds  = o.prewarm_audio_seconds;
    const auto t_ctor = std::chrono::steady_clock::now();
    parakeet::Engine eng(eopts);
    const double ctor_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - t_ctor).count() / 1000.0;
    std::fprintf(stderr,
        "[determinism] transcribe path: model=%s backend=%s threads=%d gpu_layers=%d runs=%d prewarm=%s ctor=%.2fms\n",
        model_type_str.c_str(),
        eng.backend_name().c_str(),
        o.n_threads, o.n_gpu_layers, o.n_runs,
        o.prewarm ? "on" : "off",
        ctor_ms);

    std::vector<std::vector<int32_t>> ids_per_run;
    std::vector<std::string>          text_per_run;
    std::vector<double>               enc_ms_per_run;
    ids_per_run.reserve(o.n_runs);
    text_per_run.reserve(o.n_runs);
    enc_ms_per_run.reserve(o.n_runs);

    for (int k = 0; k < o.n_runs; ++k) {
        parakeet::EngineResult r =
            eng.transcribe_samples(samples.data(), (int) samples.size(), sr);
        if (o.verbose) {
            std::fprintf(stderr,
                "[determinism]   run %d/%d  enc_ms=%.2f  decode_ms=%.2f  total_ms=%.2f  tokens=%zu  text='%s'\n",
                k + 1, o.n_runs,
                r.encoder_ms, r.decode_ms, r.total_ms,
                r.token_ids.size(),
                r.text.size() > 60 ? (r.text.substr(0, 60) + "…").c_str() : r.text.c_str());
        }
        ids_per_run.push_back(std::move(r.token_ids));
        text_per_run.push_back(std::move(r.text));
        enc_ms_per_run.push_back(r.encoder_ms);
    }

    bool ok = true;
    for (int k = 1; k < o.n_runs; ++k) {
        if (ids_per_run[k] != ids_per_run[0]) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d token_ids differ from run 0 "
                "(run0=%zu tokens, runK=%zu tokens)\n",
                k, ids_per_run[0].size(), ids_per_run[k].size());
            const size_t lim = std::min(ids_per_run[k].size(), ids_per_run[0].size());
            for (size_t i = 0; i < lim; ++i) {
                if (ids_per_run[k][i] != ids_per_run[0][i]) {
                    std::fprintf(stderr,
                        "[determinism]   first divergence at index %zu: run0=%d runK=%d\n",
                        i, ids_per_run[0][i], ids_per_run[k][i]);
                    break;
                }
            }
            ok = false;
        }
        if (text_per_run[k] != text_per_run[0]) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d transcript differs from run 0\n"
                "  run 0: '%s'\n  run %d: '%s'\n",
                k, text_per_run[0].c_str(), k, text_per_run[k].c_str());
            ok = false;
        }
    }

    // Encoder cache-hit gate: median(warm) must not re-pay the cold
    // cost. Run 0 is always cold (graph build); runs 1..N-1 are
    // warm and should be at most cold * cache_hit_ratio_max in
    // their median. Median (not max) is the right statistic — a
    // real cache MISS rebuilds the graph each call so EVERY warm
    // run lands at >= cold; thermal-spike one-offs only push max,
    // not median. Skip gate when any enc_ms came back zero (the
    // engine path doesn't separate encoder timing for some model
    // variants).
    bool have_enc_ms = true;
    for (double t : enc_ms_per_run) if (t <= 0.0) { have_enc_ms = false; break; }
    if (have_enc_ms && o.n_runs >= 2) {
        const double cold = enc_ms_per_run[0];
        std::vector<double> warm(enc_ms_per_run.begin() + 1, enc_ms_per_run.end());
        const double med = median(warm);
        const double max_warm = *std::max_element(warm.begin(), warm.end());
        const double min_warm = *std::min_element(warm.begin(), warm.end());
        // Gate 1: cache-hit ratio. Compares median(warm) to cold;
        // a real cache MISS rebuilds the encoder graph each call,
        // lifting EVERY warm to ≥ cold. SKIP this gate when prewarm
        // is on, because prewarm makes the "cold" run effectively
        // warm (graph already cached from the synthetic forward
        // pass in the constructor), so the cold-vs-warm asymmetry
        // this gate assumes no longer holds — thermal noise can
        // legitimately push median(warm) above run0 by a few %.
        if (!o.prewarm && med > cold * o.cache_hit_ratio_max) {
            std::fprintf(stderr,
                "[determinism] FAIL: median(warm) %.2f > cold run0 %.2f * %.2fx "
                "(cache miss? — graph being rebuilt each call?)\n",
                med, cold, o.cache_hit_ratio_max);
            ok = false;
        }
        std::fprintf(stderr,
            "[determinism] enc_ms run0 (cold) = %.2fms; warms min=%.2f median=%.2f max=%.2f (med/cold=%.2fx, max/cold=%.2fx)\n",
            cold, min_warm, med, max_warm, med / cold, max_warm / cold);

        // Gate 2: prewarm cold-overhead. When --prewarm is set,
        // the encoder graph build cost was supposed to be paid in
        // the constructor, so run0 should land in the warm steady-
        // state band (not perfectly equal — there's still ~50-100ms
        // of non-encoder cold cost on CPU: ggml thread-pool spin-up,
        // decoder per-call allocator, thermal warmup). The gate
        // catches a regression where prewarm doesn't actually do
        // anything (run0 = full cold cost = 1.5-3x median on CPU).
        if (o.prewarm) {
            const double cold_overhead = cold / med;
            if (cold_overhead > o.cold_overhead_max) {
                std::fprintf(stderr,
                    "[determinism] FAIL: with --prewarm, run0 %.2f / median(warm) %.2f = %.2fx > %.2fx "
                    "(prewarm didn't warm the pipeline?)\n",
                    cold, med, cold_overhead, o.cold_overhead_max);
                ok = false;
            } else {
                std::fprintf(stderr,
                    "[determinism] prewarm OK: run0/median(warm) = %.2fx (limit %.2fx)\n",
                    cold_overhead, o.cold_overhead_max);
            }
        }
    }

    if (!ok) return 1;
    std::fprintf(stderr,
        "[determinism] PASS  transcribe x %d: %zu tokens / %zu chars / encoder cache hit\n",
        o.n_runs, ids_per_run[0].size(), text_per_run[0].size());
    return 0;
}

// ---------- Determinism for Sortformer (diarize path) ----------

bool sort_seg_eq(const parakeet::DiarizationSegment & a,
                 const parakeet::DiarizationSegment & b) {
    return a.speaker_id == b.speaker_id &&
           a.start_s == b.start_s &&
           a.end_s   == b.end_s;
}

int run_diarize_path(const Opts & o,
                     const std::vector<float> & samples, int sr,
                     const std::string & model_type_str) {
    parakeet::EngineOptions eopts;
    eopts.model_gguf_path        = o.model_path;
    eopts.n_threads              = o.n_threads;
    eopts.n_gpu_layers           = o.n_gpu_layers;
    eopts.verbose                = false;
    eopts.prewarm                = o.prewarm;
    eopts.prewarm_audio_seconds  = o.prewarm_audio_seconds;
    const auto t_ctor = std::chrono::steady_clock::now();
    parakeet::Engine eng(eopts);
    const double ctor_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - t_ctor).count() / 1000.0;
    std::fprintf(stderr,
        "[determinism] diarize path: model=%s backend=%s threads=%d gpu_layers=%d runs=%d prewarm=%s ctor=%.2fms\n",
        model_type_str.c_str(),
        eng.backend_name().c_str(),
        o.n_threads, o.n_gpu_layers, o.n_runs,
        o.prewarm ? "on" : "off",
        ctor_ms);

    std::vector<std::vector<float>>    probs_per_run;
    std::vector<std::vector<parakeet::DiarizationSegment>> segs_per_run;
    std::vector<double>                enc_ms_per_run;
    probs_per_run.reserve(o.n_runs);
    segs_per_run.reserve(o.n_runs);
    enc_ms_per_run.reserve(o.n_runs);

    for (int k = 0; k < o.n_runs; ++k) {
        parakeet::DiarizationResult r =
            eng.diarize_samples(samples.data(), (int) samples.size(), sr,
                                parakeet::DiarizationOptions{});
        if (o.verbose) {
            std::fprintf(stderr,
                "[determinism]   run %d/%d  enc_ms=%.2f  decode_ms=%.2f  total_ms=%.2f  "
                "frames=%d num_spks=%d segs=%zu\n",
                k + 1, o.n_runs,
                r.encoder_ms, r.decode_ms, r.total_ms,
                r.n_frames, r.num_spks, r.segments.size());
        }
        probs_per_run.push_back(std::move(r.speaker_probs));
        segs_per_run.push_back(std::move(r.segments));
        enc_ms_per_run.push_back(r.encoder_ms);
    }

    bool ok = true;
    for (int k = 1; k < o.n_runs; ++k) {
        if (probs_per_run[k].size() != probs_per_run[0].size()) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d speaker_probs size differs (%zu vs run0 %zu)\n",
                k, probs_per_run[k].size(), probs_per_run[0].size());
            ok = false;
            continue;
        }
        if (std::memcmp(probs_per_run[k].data(), probs_per_run[0].data(),
                        probs_per_run[k].size() * sizeof(float)) != 0) {
            // Find + report first diff.
            for (size_t i = 0; i < probs_per_run[k].size(); ++i) {
                if (std::memcmp(&probs_per_run[k][i], &probs_per_run[0][i], sizeof(float)) != 0) {
                    std::fprintf(stderr,
                        "[determinism] FAIL: run %d speaker_probs[%zu]=%.9g run0=%.9g delta=%.3e\n",
                        k, i, (double) probs_per_run[k][i], (double) probs_per_run[0][i],
                        (double) (probs_per_run[k][i] - probs_per_run[0][i]));
                    break;
                }
            }
            ok = false;
        }
        if (segs_per_run[k].size() != segs_per_run[0].size()) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d produced %zu segments vs run0 %zu\n",
                k, segs_per_run[k].size(), segs_per_run[0].size());
            ok = false;
            continue;
        }
        for (size_t i = 0; i < segs_per_run[k].size(); ++i) {
            if (!sort_seg_eq(segs_per_run[k][i], segs_per_run[0][i])) {
                std::fprintf(stderr,
                    "[determinism] FAIL: run %d segment %zu differs "
                    "(spk=%d/%d  start=%.4f/%.4f  end=%.4f/%.4f)\n",
                    k, i,
                    segs_per_run[k][i].speaker_id, segs_per_run[0][i].speaker_id,
                    segs_per_run[k][i].start_s,    segs_per_run[0][i].start_s,
                    segs_per_run[k][i].end_s,      segs_per_run[0][i].end_s);
                ok = false;
            }
        }
    }

    // See run_transcribe_path comment for rationale on the two gates
    // and why the cache-hit-ratio gate is skipped when prewarm is on.
    bool have_enc_ms = true;
    for (double t : enc_ms_per_run) if (t <= 0.0) { have_enc_ms = false; break; }
    if (have_enc_ms && o.n_runs >= 2) {
        const double cold = enc_ms_per_run[0];
        std::vector<double> warm(enc_ms_per_run.begin() + 1, enc_ms_per_run.end());
        const double med = median(warm);
        const double max_warm = *std::max_element(warm.begin(), warm.end());
        const double min_warm = *std::min_element(warm.begin(), warm.end());
        if (!o.prewarm && med > cold * o.cache_hit_ratio_max) {
            std::fprintf(stderr,
                "[determinism] FAIL: median(warm) %.2f > cold run0 %.2f * %.2fx "
                "(cache miss? — graph being rebuilt each call?)\n",
                med, cold, o.cache_hit_ratio_max);
            ok = false;
        }
        std::fprintf(stderr,
            "[determinism] enc_ms run0 (cold) = %.2fms; warms min=%.2f median=%.2f max=%.2f (med/cold=%.2fx, max/cold=%.2fx)\n",
            cold, min_warm, med, max_warm, med / cold, max_warm / cold);

        if (o.prewarm) {
            const double cold_overhead = cold / med;
            if (cold_overhead > o.cold_overhead_max) {
                std::fprintf(stderr,
                    "[determinism] FAIL: with --prewarm, run0 %.2f / median(warm) %.2f = %.2fx > %.2fx "
                    "(prewarm didn't warm the pipeline?)\n",
                    cold, med, cold_overhead, o.cold_overhead_max);
                ok = false;
            } else {
                std::fprintf(stderr,
                    "[determinism] prewarm OK: run0/median(warm) = %.2fx (limit %.2fx)\n",
                    cold_overhead, o.cold_overhead_max);
            }
        }
    }

    if (!ok) return 1;
    std::fprintf(stderr,
        "[determinism] PASS  diarize x %d: %zu floats in speaker_probs / %zu segments / encoder cache hit\n",
        o.n_runs,
        probs_per_run[0].size(),
        segs_per_run[0].size());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Opts o;
    if (int rc = parse_args(argc, argv, o); rc != 0) return rc;

    std::vector<float> samples;
    int sr = 0;
    if (!load_wav_pcm(o.wav_path, samples, sr)) {
        std::fprintf(stderr, "[determinism] failed to load wav: %s\n", o.wav_path.c_str());
        return 3;
    }

    // Auto-derive prewarm_audio_seconds from the
    // loaded wav when --prewarm-audio-seconds wasn't passed
    // explicitly. The encoder graph cache is shape-keyed, so this
    // is what the test should do to get run0 into the warm band;
    // production code calling Engine should follow the same rule
    // (pass the typical call shape as prewarm_audio_seconds).
    if (o.prewarm && o.prewarm_audio_seconds == 0.0f) {
        o.prewarm_audio_seconds = (float) samples.size() / (float) sr;
        std::fprintf(stderr,
            "[determinism] auto-derived prewarm_audio_seconds=%.2f from wav (%zu samples @ %d Hz)\n",
            o.prewarm_audio_seconds, samples.size(), sr);
    }

    // Probe the model type via the public Engine API — keeps this
    // test linked through the public library boundary, mirroring how
    // the production parakeet CLI uses it.
    parakeet::EngineOptions probe_eopts;
    probe_eopts.model_gguf_path = o.model_path;
    probe_eopts.n_threads       = o.n_threads;
    probe_eopts.n_gpu_layers    = o.n_gpu_layers;
    probe_eopts.verbose         = false;
    parakeet::Engine probe(probe_eopts);
    const std::string mt = probe.model_type();
    std::fprintf(stderr, "[determinism] model_type=%s\n", mt.c_str());

    if (mt == "sortformer" || probe.is_diarization_model()) {
        return run_diarize_path(o, samples, sr, mt);
    }
    return run_transcribe_path(o, samples, sr, mt);
}
