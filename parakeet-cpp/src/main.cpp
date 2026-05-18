// CLI executable: flags, WAV/model paths, transcribe/diarize/streaming modes.

#include "parakeet/cli.h"
#include "parakeet/engine.h"
#include "parakeet/streaming.h"
#include "parakeet/diarization.h"
#include "parakeet/attributed.h"

#include "parakeet_ctc.h"
#include "parakeet_log.h"
#include "parakeet_tdt.h"
#include "parakeet_eou.h"
#include "mel_preprocess.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
static int parakeet_setenv(const char * name, const char * value, int /*overwrite*/) {
    return _putenv_s(name, value);
}
#define setenv parakeet_setenv
#endif

namespace {

void print_usage(const char * argv0) {
    PARAKEET_LOG_INFO(
        "usage: %s --model <gguf> (--wav <input.wav> | --pcm-in <input.raw>) [options]\n"
        "\n"
        "Single CLI for all four engine families. The GGUF is auto-detected:\n"
        "  CTC        (parakeet-ctc-0.6b/1.1b)        -> transcription\n"
        "  TDT        (parakeet-tdt-0.6b-v3, 1.1b)    -> multilingual transcription\n"
        "  EOU        (parakeet_realtime_eou_120m-v1) -> low-latency streaming ASR with\n"
        "                                                native end-of-utterance token\n"
        "  Sortformer (diar_sortformer_4spk-v1, v2)   -> 4-speaker diarization\n"
        "Combined ASR + diarization (\"who said what\") via --diarization-model.\n"
        "\n"
        "options:\n"
        "  --model PATH         path to a CTC, TDT, EOU, or Sortformer GGUF (required)\n"
        "  --wav PATH           path to a 16 kHz mono wav file\n"
        "  --pcm-in PATH        path to a raw PCM file (mono, format selected by --pcm-format)\n"
        "  --pcm-format FMT     raw PCM sample format: s16le (default) or f32le\n"
        "  --pcm-rate HZ        sample rate of the raw PCM file. Required to match the\n"
        "                       model's sample rate exactly (resampling is not yet wired).\n"
        "                       If omitted, falls back to the model's rate with a warning.\n"
        "  --threads N          number of CPU threads (0 = hardware_concurrency)\n"
        "  --n-gpu-layers N     when > 0, run the encoder on the compiled-in GPU\n"
        "                       backend (build with -DGGML_METAL=ON / -DGGML_CUDA=ON\n"
        "                       / -DGGML_VULKAN=ON / -DGGML_OPENCL=ON; only one is\n"
        "                       active per binary -- CUDA wins over Metal wins over\n"
        "                       Vulkan wins over OpenCL if multiple are compiled in.\n"
        "                       N is only checked >0 today: the whole encoder moves;\n"
        "                       partial layer offload is not implemented.\n"
        "                       OpenCL note: ggml-opencl is tuned for Adreno (Android);\n"
        "                       on commodity desktop GPUs build with\n"
        "                       -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF (the parakeet\n"
        "                       patch under patches/ relaxes the upstream Adreno-only\n"
        "                       device whitelist for dev/CI parity testing). Production\n"
        "                       Adreno deployments leave both at their defaults.\n"
        "  --opencl-cache-dir DIR             persistent OpenCL kernel binary cache directory\n"
        "                                     (sets $GGML_OPENCL_CACHE_DIR; consumed by\n"
        "                                     patches/ggml-opencl-program-binary-cache.patch).\n"
        "                                     Empty string disables the cache; default\n"
        "                                     resolves to $XDG_CACHE_HOME/ggml/opencl\n"
        "                                     -> $HOME/.cache/ggml/opencl.\n"
        "  --opencl-platform NAME_OR_INDEX    select OpenCL platform (sets $GGML_OPENCL_PLATFORM).\n"
        "                                     Useful when several ICDs are loaded e.g.\n"
        "                                     'NVIDIA CUDA' alongside 'rusticl'/'PoCL'.\n"
        "  --opencl-device NAME_OR_INDEX      select OpenCL device (sets $GGML_OPENCL_DEVICE).\n"
        "  --opencl-disable-fusion            sets $GGML_OPENCL_DISABLE_FUSION=1; disables\n"
        "                                     the NORM/GROUP_NORM + MUL + ADD fusion that\n"
        "                                     ggml-opencl auto-detects. Useful for A/B-ing\n"
        "                                     fusion impact on first-Adreno bring-up.\n"
        "  --opencl-adreno-use-large-buffer   sets $GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1;\n"
        "                                     Adreno-only knob to allow >256 MB single\n"
        "                                     allocations when cl_qcom_large_buffer is\n"
        "                                     exposed by the driver. Required for 0.6B+\n"
        "                                     Q8_0 GGUFs on Adreno (the model weights\n"
        "                                     exceed the 256 MB single-allocation cap).\n"
        "  --verbose            print per-stage wall times and shapes to stderr\n"
        "\n"
        "  --stream             enable streaming. Without --stream-duplex this is Mode 2:\n"
        "                       runs the offline encoder once, then emits one segment per\n"
        "                       --stream-chunk-ms window via callback. Transcript is\n"
        "                       byte-equal to the non-streaming path.\n"
        "  --stream-duplex      enable Mode 3 (cache-aware duplex streaming): feeds the\n"
        "                       audio into a StreamSession in blocks, runs the encoder per\n"
        "                       chunk with left-context + right-lookahead, emits segments\n"
        "                       as soon as each chunk is processed. Incurs per-chunk\n"
        "                       encoder cost but first segment lands at ~chunk_ms +\n"
        "                       right_lookahead_ms. Typical WER: ~0 %% on short clean\n"
        "                       speech, ~4 %% on long sci-fi narration at the default\n"
        "                       context budget. Requires --stream.\n"
        "  --stream-chunk-ms N  segment window stride in ms (default 1000 Mode 2 /\n"
        "                       2000 Mode 3 recommended; snaps to the encoder frame\n"
        "                       stride, which is 80 ms on every shipped GGUF -- a\n"
        "                       different mel hop or subsampling factor would change\n"
        "                       this. Hard floor of 80 ms enforced at parse time.)\n"
        "  --stream-left-context-ms N    (Mode 3) left-context audio per chunk (default 10000)\n"
        "  --stream-right-lookahead-ms N (Mode 3) right-lookahead audio per chunk (default 2000)\n"
        "  --stream-feed-bytes N         (Mode 3) feed PCM into StreamSession in N-byte\n"
        "                                blocks (default 4096 bytes = 1024 samples); exercise\n"
        "                                with smaller values to stress the session state machine\n"
        "  --stream-history-ms N         (Sortformer streaming) sliding history window in ms\n"
        "                                (default 30000). Larger values stabilise speaker IDs\n"
        "                                across chunks at the cost of per-chunk encoder work.\n"
        "  --emit FMT           --stream output format: 'text' (default) prints segment text\n"
        "                       one per line; 'jsonl' prints {text,start,end,chunk,is_final,\n"
        "                       is_eou_boundary}\n"
        "                       JSON Lines, one per segment. For Sortformer streaming, prints\n"
        "                       speaker segments instead of text.\n"
        "\n"
        "  --diarization-model PATH         path to a Sortformer GGUF; combined with a CTC/TDT\n"
        "                                    --model, runs speaker-attributed transcription\n"
        "                                    (writes [start-end] speaker_N: text per segment).\n"
        "                                    Implies --emit text|jsonl per the same flag.\n"
        "  --diarization-min-segment-ms N    drop diarization segments shorter than N ms\n"
        "                                    before attributing transcripts (default 200).\n"
        "  --diarization-pad-segment-ms N    pad each diarization segment by N ms on each\n"
        "                                    side before slicing audio for ASR (default 0).\n"
        "\n"
        "  --bench              benchmark mode: run the inference path multiple times\n"
        "                       with warmup, print aggregated stats + RTF.\n"
        "                       (Transcript is printed once after the stats.)\n"
        "  --bench-runs N       timed runs for --bench (default 3)\n"
        "  --bench-warmup N     warmup runs NOT counted in stats (default 2)\n"
        "  --bench-json PATH    in --bench mode, also write the stats as JSON to PATH\n"
        "  --profile            per-sub-stage encoder profiling: runs the encoder\n"
        "                       with n_layers = {0, 1, N/2, N} (N from the GGUF) and\n"
        "                       attributes time to subsampling / CTC-head / per-block.\n"
        "  --profile-runs N     timed runs per configuration in --profile (default 5)\n"
        "  --profile-warmup N   warmup runs per configuration (default 2)\n"
        "\n"
        "  --dump-mel PATH      write the C++ log-mel tensor as raw float32\n"
        "                       (n_mels, T_mel) to PATH; handy for offline diffing\n"
        "                       against mel.npy. n_mels is read from the loaded\n"
        "                       GGUF (80 for CTC; 128 for TDT/EOU/Sortformer); the\n"
        "                       actual shape is logged at dump time.\n"
        "  --version            print version and exit\n"
        "  --help               this help text\n",
        argv0);
}

int load_raw_pcm(const std::string & path,
                 const std::string & format,
                 std::vector<float> & out_samples) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        PARAKEET_LOG_ERROR("error: could not open raw PCM file %s\n", path.c_str());
        return 1;
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size <= 0) {
        PARAKEET_LOG_ERROR("error: raw PCM file %s is empty\n", path.c_str());
        return 2;
    }
    if (format == "s16le") {
        if (size % 2 != 0) {
            PARAKEET_LOG_ERROR("error: s16le PCM size %lld not multiple of 2\n",
                         (long long) size);
            return 3;
        }
        const size_t n = static_cast<size_t>(size) / 2;
        std::vector<int16_t> buf(n);
        if (!f.read(reinterpret_cast<char *>(buf.data()), size)) {
            PARAKEET_LOG_ERROR("error: short read from %s\n", path.c_str());
            return 4;
        }
        out_samples.resize(n);
        constexpr float inv = 1.0f / 32768.0f;
        for (size_t i = 0; i < n; ++i) out_samples[i] = (float) buf[i] * inv;
        return 0;
    }
    if (format == "f32le") {
        if (size % 4 != 0) {
            PARAKEET_LOG_ERROR("error: f32le PCM size %lld not multiple of 4\n",
                         (long long) size);
            return 3;
        }
        const size_t n = static_cast<size_t>(size) / 4;
        out_samples.resize(n);
        if (!f.read(reinterpret_cast<char *>(out_samples.data()), size)) {
            PARAKEET_LOG_ERROR("error: short read from %s\n", path.c_str());
            return 4;
        }
        return 0;
    }
    PARAKEET_LOG_ERROR("error: unknown --pcm-format '%s' (expected s16le or f32le)\n",
                 format.c_str());
    return 5;
}

void emit_segment(const parakeet::StreamingSegment & seg,
                  const std::string & format) {
    if (format == "jsonl") {
        std::printf("{\"chunk\":%d,\"start\":%.3f,\"end\":%.3f,\"is_final\":%s,"
                    "\"is_eou_boundary\":%s,\"text\":\"",
                    seg.chunk_index, seg.start_s, seg.end_s,
                    seg.is_final ? "true" : "false",
                    seg.is_eou_boundary ? "true" : "false");
        for (char c : seg.text) {
            switch (c) {
                case '"':  std::fputs("\\\"", stdout); break;
                case '\\': std::fputs("\\\\", stdout); break;
                case '\n': std::fputs("\\n",  stdout); break;
                case '\r': std::fputs("\\r",  stdout); break;
                case '\t': std::fputs("\\t",  stdout); break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        std::printf("\\u%04x", c);
                    } else {
                        std::fputc(c, stdout);
                    }
            }
        }
        std::printf("\"}\n");
    } else {
        std::printf("[%.2f-%.2f] %s\n",
                    seg.start_s, seg.end_s, seg.text.c_str());
    }
    std::fflush(stdout);
}

struct ExtraCliOpts {
    std::string dump_mel_path;
    bool        bench         = false;
    int         bench_runs    = 3;
    int         bench_warmup  = 2;
    std::string bench_json_path;
    bool        profile       = false;
    int         profile_runs  = 5;
    int         profile_warmup = 2;

    std::string pcm_in_path;
    std::string pcm_format   = "s16le";
    int         pcm_rate     = 0;

    bool        stream            = false;
    int         stream_chunk_ms   = 1000;
    int         stream_left_ms    = -1;
    int         stream_right_ms   = -1;
    bool        stream_duplex     = false;
    int         stream_feed_bytes = 0;
    int         stream_history_ms = 30000;
    std::string emit_format       = "text";

    std::string diarization_model_path;
    int         attributed_min_segment_ms = 200;
    int         attributed_pad_segment_ms = 0;

    // OpenCL CLI surface: ggml-opencl's runtime knobs are exposed through
    // CLI flags so bench scripts can A/B them without `env VAR=… ./binary`.
    // All four are read by ggml-opencl via getenv() and (for the cache
    // dir) by `patches/ggml-opencl-program-binary-cache.patch`. Applied
    // via `setenv()` BEFORE any parakeet API call so the backend
    // init cascade picks them up. Empty string for any field => leave
    // the existing process-env value untouched (do not setenv).
    std::string opencl_cache_dir;
    std::string opencl_platform;     // GGML_OPENCL_PLATFORM
    std::string opencl_device;       // GGML_OPENCL_DEVICE
    bool        opencl_disable_fusion = false; // GGML_OPENCL_DISABLE_FUSION=1
    bool        opencl_adreno_use_large_buffer = false; // GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1
};

// Apply OpenCL runtime overrides from the CLI to the process env.
// Must run BEFORE the first ggml-opencl backend init (i.e. before
// load_from_gguf / Engine ctor) so the backend reads our settings.
// Empty / false fields are no-ops -- leave any pre-existing
// $GGML_OPENCL_* env value alone so a wrapper script's settings
// still take precedence over the CLI-default-empty fields.
void apply_opencl_cli_env(const ExtraCliOpts & e) {
    if (!e.opencl_cache_dir.empty()) {
        setenv("GGML_OPENCL_CACHE_DIR", e.opencl_cache_dir.c_str(), /*overwrite=*/1);
    }
    if (!e.opencl_platform.empty()) {
        setenv("GGML_OPENCL_PLATFORM", e.opencl_platform.c_str(), 1);
    }
    if (!e.opencl_device.empty()) {
        setenv("GGML_OPENCL_DEVICE", e.opencl_device.c_str(), 1);
    }
    if (e.opencl_disable_fusion) {
        setenv("GGML_OPENCL_DISABLE_FUSION", "1", 1);
    }
    if (e.opencl_adreno_use_large_buffer) {
        setenv("GGML_OPENCL_ADRENO_USE_LARGE_BUFFER", "1", 1);
    }
}

double ms_since(std::chrono::steady_clock::time_point a) {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now() - a).count() / 1000.0;
}

struct RunTimes {
    double mel_ms         = 0.0;
    double enc_ms         = 0.0;
    double dec_ms         = 0.0;
    double inference_ms   = 0.0;
    int    tokens         = 0;
    int    encoder_frames = 0;
};

struct AggStats {
    double mean   = 0.0;
    double stdev  = 0.0;
    double min    = 0.0;
    double max    = 0.0;
    double median = 0.0;
};

AggStats aggregate(std::vector<double> v) {
    AggStats s;
    if (v.empty()) return s;
    s.min = v.front();
    s.max = v.front();
    double sum = 0.0;
    for (double x : v) { sum += x; s.min = std::min(s.min, x); s.max = std::max(s.max, x); }
    s.mean = sum / (double) v.size();
    double ss = 0.0;
    for (double x : v) { const double d = x - s.mean; ss += d * d; }
    s.stdev = v.size() > 1 ? std::sqrt(ss / (double)(v.size() - 1)) : 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() % 2 == 1) s.median = v[v.size() / 2];
    else                   s.median = 0.5 * (v[v.size()/2 - 1] + v[v.size()/2]);
    return s;
}

// Private CLI options struct -- this binary's parsed flags only. The
// public C++ API (parakeet::EngineOptions) doesn't carry the wav
// path because it's a property of each transcribe() call, not the
// loaded engine; the CLI happens to want both in one bag during arg
// parsing so we keep a tiny local struct.
struct CliOpts {
    std::string model_gguf_path;
    std::string wav_path;
    int  n_threads    = 0;
    int  n_gpu_layers = 0;
    bool verbose      = false;
};

}

extern "C" int parakeet_cli_main(int argc, char ** argv) {
    CliOpts      opts;
    ExtraCliOpts extra;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--version") {
#ifndef PARAKEET_VERSION
#define PARAKEET_VERSION "unknown"
#endif
            std::printf("parakeet %s\n", PARAKEET_VERSION);
            return 0;
        } else if (a == "--model" && i + 1 < argc) {
            opts.model_gguf_path = argv[++i];
        } else if (a == "--wav" && i + 1 < argc) {
            opts.wav_path = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            opts.n_threads = std::atoi(argv[++i]);
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            opts.n_gpu_layers = std::atoi(argv[++i]);
        } else if (a == "--opencl-cache-dir" && i + 1 < argc) {
            extra.opencl_cache_dir = argv[++i];
        } else if (a == "--opencl-platform" && i + 1 < argc) {
            extra.opencl_platform = argv[++i];
        } else if (a == "--opencl-device" && i + 1 < argc) {
            extra.opencl_device = argv[++i];
        } else if (a == "--opencl-disable-fusion") {
            extra.opencl_disable_fusion = true;
        } else if (a == "--opencl-adreno-use-large-buffer") {
            extra.opencl_adreno_use_large_buffer = true;
        } else if (a == "--verbose" || a == "-v") {
            opts.verbose = true;
        } else if (a == "--dump-mel" && i + 1 < argc) {
            extra.dump_mel_path = argv[++i];
        } else if (a == "--bench") {
            extra.bench = true;
        } else if (a == "--bench-runs" && i + 1 < argc) {
            extra.bench_runs = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--bench-warmup" && i + 1 < argc) {
            extra.bench_warmup = std::max(0, std::atoi(argv[++i]));
        } else if (a == "--bench-json" && i + 1 < argc) {
            extra.bench_json_path = argv[++i];
        } else if (a == "--profile") {
            extra.profile = true;
        } else if (a == "--profile-runs" && i + 1 < argc) {
            extra.profile_runs = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--profile-warmup" && i + 1 < argc) {
            extra.profile_warmup = std::max(0, std::atoi(argv[++i]));
        } else if (a == "--pcm-in" && i + 1 < argc) {
            extra.pcm_in_path = argv[++i];
        } else if (a == "--pcm-format" && i + 1 < argc) {
            extra.pcm_format = argv[++i];
        } else if (a == "--pcm-rate" && i + 1 < argc) {
            extra.pcm_rate = std::atoi(argv[++i]);
        } else if (a == "--stream") {
            extra.stream = true;
        } else if (a == "--stream-chunk-ms" && i + 1 < argc) {
            // 80 ms is the encoder frame stride of every shipped GGUF
            // (16 kHz x hop=160 x sub=8). A future GGUF with a smaller
            // stride would silently round up here; revisit when we
            // actually ship one.
            extra.stream_chunk_ms = std::max(80, std::atoi(argv[++i]));
        } else if (a == "--stream-left-context-ms" && i + 1 < argc) {
            extra.stream_left_ms = std::max(0, std::atoi(argv[++i]));
        } else if (a == "--stream-right-lookahead-ms" && i + 1 < argc) {
            extra.stream_right_ms = std::max(0, std::atoi(argv[++i]));
        } else if (a == "--stream-duplex") {
            extra.stream_duplex = true;
        } else if (a == "--stream-feed-bytes" && i + 1 < argc) {
            extra.stream_feed_bytes = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--stream-history-ms" && i + 1 < argc) {
            extra.stream_history_ms = std::max(1000, std::atoi(argv[++i]));
        } else if (a == "--emit" && i + 1 < argc) {
            extra.emit_format = argv[++i];
        } else if (a == "--diarization-model" && i + 1 < argc) {
            extra.diarization_model_path = argv[++i];
        } else if (a == "--diarization-min-segment-ms" && i + 1 < argc) {
            extra.attributed_min_segment_ms = std::max(0, std::atoi(argv[++i]));
        } else if (a == "--diarization-pad-segment-ms" && i + 1 < argc) {
            extra.attributed_pad_segment_ms = std::max(0, std::atoi(argv[++i]));
        } else {
            PARAKEET_LOG_ERROR("unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (opts.model_gguf_path.empty() ||
        (opts.wav_path.empty() && extra.pcm_in_path.empty())) {
        print_usage(argv[0]);
        return 2;
    }
    if (!opts.wav_path.empty() && !extra.pcm_in_path.empty()) {
        PARAKEET_LOG_ERROR("error: --wav and --pcm-in are mutually exclusive\n");
        return 2;
    }
    if (extra.emit_format != "text" && extra.emit_format != "jsonl") {
        PARAKEET_LOG_ERROR("error: --emit must be 'text' or 'jsonl' (got '%s')\n",
                     extra.emit_format.c_str());
        return 2;
    }

    using namespace parakeet;
    using clock = std::chrono::steady_clock;

    // Apply CLI -> $GGML_OPENCL_* env overrides before any backend init
    // so the `init_gpu_backend()` cascade reads our settings. No-op
    // when the binary was built without -DGGML_OPENCL=ON (the env vars
    // just aren't read by anything).
    apply_opencl_cli_env(extra);

    const auto t_load = clock::now();
    ParakeetCtcModel model;
    if (int rc = load_from_gguf(opts.model_gguf_path, model, opts.n_threads, opts.n_gpu_layers, opts.verbose); rc != 0) {
        PARAKEET_LOG_ERROR("error: failed to load %s (rc=%d)\n", opts.model_gguf_path.c_str(), rc);
        return 3;
    }
    const double load_ms = ms_since(t_load);

    const auto t_wav = clock::now();
    std::vector<float> samples;
    int sr = model.mel_cfg.sample_rate;
    if (!opts.wav_path.empty()) {
        if (int rc = load_wav_mono_f32(opts.wav_path, samples, sr); rc != 0) {
            PARAKEET_LOG_ERROR("error: failed to load %s (rc=%d)\n", opts.wav_path.c_str(), rc);
            return 4;
        }
    } else {
        if (int rc = load_raw_pcm(extra.pcm_in_path, extra.pcm_format, samples); rc != 0) {
            return 4;
        }
        if (extra.pcm_rate > 0) {
            sr = extra.pcm_rate;
        } else {
            PARAKEET_LOG_INFO(
                "warning: --pcm-in without --pcm-rate; assuming %d Hz to match the model.\n"
                "         Pass --pcm-rate explicitly to silence this warning and to fail-fast\n"
                "         on a mismatched raw PCM rate (resampling is not yet wired).\n",
                model.mel_cfg.sample_rate);
            sr = model.mel_cfg.sample_rate;
        }
    }
    if (sr != model.mel_cfg.sample_rate) {
        PARAKEET_LOG_ERROR("error: input is %d Hz but model expects %d Hz (resampling not yet wired)\n",
                     sr, model.mel_cfg.sample_rate);
        return 5;
    }
    const double wav_ms = ms_since(t_wav);
    const double audio_ms = 1000.0 * (double) samples.size() / (double) sr;

    if (!extra.diarization_model_path.empty()) {
        if (model.model_type == ParakeetModelType::SORTFORMER) {
            PARAKEET_LOG_ERROR("error: --diarization-model expects --model to be a transcription\n"
                                 "       (CTC/TDT) GGUF; got Sortformer at --model. Swap them.\n");
            return 5;
        }

        EngineOptions sf_opts;
        sf_opts.model_gguf_path = extra.diarization_model_path;
        sf_opts.n_gpu_layers    = opts.n_gpu_layers;
        sf_opts.n_threads       = opts.n_threads;
        sf_opts.verbose         = opts.verbose;
        Engine sf_engine(sf_opts);

        if (!sf_engine.is_diarization_model()) {
            PARAKEET_LOG_ERROR("error: --diarization-model %s is not a Sortformer GGUF\n",
                         extra.diarization_model_path.c_str());
            return 5;
        }

        EngineOptions asr_opts;
        asr_opts.model_gguf_path = opts.model_gguf_path;
        asr_opts.n_gpu_layers    = opts.n_gpu_layers;
        asr_opts.n_threads       = opts.n_threads;
        asr_opts.verbose         = opts.verbose;
        Engine asr_engine(asr_opts);

        AttributedTranscriptionOptions topts;
        topts.min_segment_ms = extra.attributed_min_segment_ms;
        topts.pad_segment_ms = extra.attributed_pad_segment_ms;

        const auto t_attr = clock::now();
        AttributedTranscriptionResult attr = transcribe_samples_with_speakers(
            sf_engine, asr_engine, samples.data(), (int) samples.size(), sr, topts);
        const double attr_ms = ms_since(t_attr);

        const std::string emit_fmt = extra.emit_format;
        for (const auto & s : attr.segments) {
            if (emit_fmt == "jsonl") {
                std::printf("{\"speaker\":%d,\"start\":%.3f,\"end\":%.3f,\"text\":\"",
                            s.speaker_id, s.start_s, s.end_s);
                for (char c : s.text) {
                    switch (c) {
                        case '"':  std::fputs("\\\"", stdout); break;
                        case '\\': std::fputs("\\\\", stdout); break;
                        case '\n': std::fputs("\\n",  stdout); break;
                        case '\r': std::fputs("\\r",  stdout); break;
                        case '\t': std::fputs("\\t",  stdout); break;
                        default:
                            if ((unsigned char) c < 0x20) std::printf("\\u%04x", c);
                            else std::fputc(c, stdout);
                    }
                }
                std::printf("\"}\n");
            } else {
                std::printf("[%.2f-%.2f] speaker_%d: %s\n",
                            s.start_s, s.end_s, s.speaker_id, s.text.c_str());
            }
        }
        if (opts.verbose) {
            PARAKEET_LOG_INFO(
                "[attributed] audio=%.2fs samples=%zu@%dHz diar.segments=%zu asr_calls=%d\n"
                "[attributed] total=%.1fms RTF=%.3f merged.segments=%zu\n",
                audio_ms / 1000.0, samples.size(), sr,
                attr.diarization.segments.size(), attr.asr_calls,
                attr_ms, attr_ms / audio_ms, attr.segments.size());
        }
        return 0;
    }

    if (model.model_type == ParakeetModelType::SORTFORMER) {
        EngineOptions eopts;
        eopts.model_gguf_path = opts.model_gguf_path;
        eopts.n_gpu_layers    = opts.n_gpu_layers;
        eopts.n_threads       = opts.n_threads;
        eopts.verbose         = opts.verbose;
        Engine engine(eopts);

        const std::string emit_fmt = extra.emit_format;

        if (extra.stream) {
            SortformerStreamingOptions sopts;
            sopts.sample_rate    = sr;
            sopts.chunk_ms       = extra.stream_chunk_ms > 0 ? extra.stream_chunk_ms : 2000;
            sopts.history_ms     = extra.stream_history_ms;
            if (sopts.history_ms < sopts.chunk_ms) sopts.history_ms = sopts.chunk_ms;
            sopts.threshold      = 0.5f;
            sopts.min_segment_ms = extra.attributed_min_segment_ms > 0
                                 ? extra.attributed_min_segment_ms : 200;

            int seg_count = 0;
            const auto t_stream_start = std::chrono::steady_clock::now();
            auto on_seg = [&](const StreamingDiarizationSegment & s) {
                if (s.speaker_id < 0) return;
                ++seg_count;
                if (emit_fmt == "jsonl") {
                    std::printf("{\"speaker\":%d,\"start\":%.3f,\"end\":%.3f,"
                                "\"chunk\":%d,\"is_final\":%s}\n",
                                s.speaker_id, s.start_s, s.end_s, s.chunk_index,
                                s.is_final ? "true" : "false");
                } else {
                    std::printf("[%.2f-%.2f] speaker_%d (chunk %d%s)\n",
                                s.start_s, s.end_s, s.speaker_id, s.chunk_index,
                                s.is_final ? ", final" : "");
                }
                std::fflush(stdout);
            };

            auto session = engine.diarize_start(sopts, on_seg);
            const int chunk_samples = sr * sopts.chunk_ms / 1000;
            const int feed = extra.stream_feed_bytes > 0
                           ? std::max(1, extra.stream_feed_bytes / (int) sizeof(float))
                           : chunk_samples;
            for (size_t off = 0; off < samples.size(); off += feed) {
                const int n = (int) std::min<size_t>(feed, samples.size() - off);
                session->feed_pcm_f32(samples.data() + off, n);
            }
            session->finalize();
            const double stream_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_stream_start).count() / 1000.0;

            if (opts.verbose) {
                PARAKEET_LOG_INFO(
                    "[diarize-stream] load=%.1fms audio=%.2fs samples=%zu@%dHz\n"
                    "[diarize-stream] chunk_ms=%d history_ms=%d segments=%d total=%.1fms RTF=%.3f\n",
                    load_ms, audio_ms / 1000.0, samples.size(), sr,
                    sopts.chunk_ms, sopts.history_ms,
                    seg_count, stream_ms, stream_ms / audio_ms);
            }
            return 0;
        }

        DiarizationOptions dopts;
        DiarizationResult diar = engine.diarize_samples(
            samples.data(), (int) samples.size(), sr, dopts);

        for (const auto & s : diar.segments) {
            if (emit_fmt == "jsonl") {
                std::printf("{\"speaker\":%d,\"start\":%.3f,\"end\":%.3f}\n",
                            s.speaker_id, s.start_s, s.end_s);
            } else {
                std::printf("[%.2f-%.2f] speaker_%d\n",
                            s.start_s, s.end_s, s.speaker_id);
            }
        }
        if (opts.verbose) {
            PARAKEET_LOG_INFO(
                "[diarize] load=%.1fms audio=%.2fs samples=%zu@%dHz frames=%d num_spks=%d\n"
                "[diarize] mel=%.1fms enc=%.1fms dec=%.1fms total=%.1fms RTF=%.3f segments=%zu\n",
                load_ms, audio_ms / 1000.0, samples.size(), sr,
                diar.n_frames, diar.num_spks,
                diar.preprocess_ms, diar.encoder_ms, diar.decode_ms,
                diar.total_ms, diar.total_ms / audio_ms,
                diar.segments.size());
        }
        return 0;
    }

    auto run_once = [&](std::string & text_out, std::vector<int32_t> & ids_out,
                        int & n_frames_out, RunTimes & times) -> int {
        const auto t1 = clock::now();
        std::vector<float> mel;
        int n_frames = 0;
        if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                     model.mel_cfg, mel, n_frames); rc != 0) return rc;
        times.mel_ms = ms_since(t1);

        if (!extra.dump_mel_path.empty()) {
            std::vector<float> transposed((size_t) model.mel_cfg.n_mels * n_frames);
            for (int t = 0; t < n_frames; ++t)
                for (int m = 0; m < model.mel_cfg.n_mels; ++m)
                    transposed[m * n_frames + t] = mel[t * model.mel_cfg.n_mels + m];
            FILE * fp = std::fopen(extra.dump_mel_path.c_str(), "wb");
            if (fp) {
                std::fwrite(transposed.data(), sizeof(float), transposed.size(), fp);
                std::fclose(fp);
                PARAKEET_LOG_INFO("[dump] wrote mel (%d, %d) to %s\n",
                                  model.mel_cfg.n_mels, n_frames,
                                  extra.dump_mel_path.c_str());
            }
            extra.dump_mel_path.clear();
        }

        const auto t2 = clock::now();
        EncoderOutputs enc_out;
        if (int rc = run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, enc_out,
                                 /*max_layers=*/-1,
                                 /*capture_intermediates=*/false); rc != 0) return rc;
        times.enc_ms = ms_since(t2);
        times.encoder_frames = enc_out.n_enc_frames;

        if (const char * dump_path = std::getenv("PARAKEET_DUMP_OUR_MEL")) {
            FILE * fp = std::fopen(dump_path, "wb");
            if (fp) {
                std::vector<float> transposed((size_t) model.mel_cfg.n_mels * n_frames);
                for (int t = 0; t < n_frames; ++t)
                    for (int m = 0; m < model.mel_cfg.n_mels; ++m)
                        transposed[m * n_frames + t] = mel[t * model.mel_cfg.n_mels + m];
                std::fwrite(transposed.data(), sizeof(float), transposed.size(), fp);
                std::fclose(fp);
                PARAKEET_LOG_INFO("[dump] wrote our_mel (%d, %d) to %s\n",
                             model.mel_cfg.n_mels, n_frames, dump_path);
            }
        }
        if (const char * dump_path = std::getenv("PARAKEET_DUMP_ENCODER")) {
            FILE * fp = std::fopen(dump_path, "wb");
            if (fp) {
                std::fwrite(enc_out.encoder_out.data(), sizeof(float),
                            enc_out.encoder_out.size(), fp);
                std::fclose(fp);
                PARAKEET_LOG_INFO("[dump] wrote encoder_out (%d frames x %d) to %s\n",
                             enc_out.n_enc_frames, enc_out.d_model, dump_path);
            }
        }
        if (const char * dump_path = std::getenv("PARAKEET_DUMP_SUBSAMPLE")) {
            FILE * fp = std::fopen(dump_path, "wb");
            if (fp) {
                std::fwrite(enc_out.subsampling_out.data(), sizeof(float),
                            enc_out.subsampling_out.size(), fp);
                std::fclose(fp);
                PARAKEET_LOG_INFO("[dump] wrote subsampling_out (%zu floats) to %s\n",
                             enc_out.subsampling_out.size(), dump_path);
            }
        }
        if (const char * dump_path = std::getenv("PARAKEET_DUMP_BLOCK0")) {
            FILE * fp = std::fopen(dump_path, "wb");
            if (fp) {
                std::fwrite(enc_out.block_0_out.data(), sizeof(float),
                            enc_out.block_0_out.size(), fp);
                std::fclose(fp);
                PARAKEET_LOG_INFO("[dump] wrote block_0_out (%zu floats) to %s\n",
                             enc_out.block_0_out.size(), dump_path);
            }
        }

        const auto t3 = clock::now();
        if (model.model_type == ParakeetModelType::TDT) {
            static TdtRuntimeWeights rt;
            static bool rt_ready = false;
            if (!rt_ready) {
                if (tdt_prepare_runtime(model, rt) != 0) return 20;
                rt_ready = true;
            }
            TdtDecodeOptions dopts;
            TdtDecodeResult  dres;
            if (int rc = tdt_greedy_decode(model, rt,
                                           enc_out.encoder_out.data(),
                                           enc_out.n_enc_frames, enc_out.d_model,
                                           dopts, dres); rc != 0) return rc;
            ids_out  = std::move(dres.token_ids);
            text_out = std::move(dres.text);
        } else if (model.model_type == ParakeetModelType::EOU) {
            static EouRuntimeWeights rt;
            static bool rt_ready = false;
            if (!rt_ready) {
                if (eou_prepare_runtime(model, rt) != 0) return 20;
                rt_ready = true;
            }
            EouDecodeOptions dopts;
            dopts.max_symbols_per_step = model.encoder_cfg.eou_max_symbols_per_step;
            EouDecodeResult  dres;
            if (int rc = eou_greedy_decode(model, rt,
                                           enc_out.encoder_out.data(),
                                           enc_out.n_enc_frames, enc_out.d_model,
                                           dopts, dres); rc != 0) return rc;
            ids_out  = std::move(dres.token_ids);
            text_out = std::move(dres.text);
        } else {
            ids_out = ctc_greedy_decode(
                enc_out.logits.data(), enc_out.n_enc_frames, model.vocab_size, model.blank_id);
            text_out = detokenize(model.vocab, ids_out);
        }
        times.dec_ms = ms_since(t3);
        times.inference_ms = times.mel_ms + times.enc_ms + times.dec_ms;
        times.tokens = (int) ids_out.size();
        n_frames_out = n_frames;
        return 0;
    };

    std::string text;
    std::vector<int32_t> ids;
    int n_frames = 0;

    if (extra.profile) {
        PARAKEET_LOG_INFO("[profile] model=%s  wav=%s (%.2f s audio, %d samples @ %d Hz)\n",
                     opts.model_gguf_path.c_str(), opts.wav_path.c_str(),
                     audio_ms / 1000.0, (int) samples.size(), sr);
        PARAKEET_LOG_INFO("[profile] threads=%d  warmup=%d  runs=%d per config\n",
                     opts.n_threads, extra.profile_warmup, extra.profile_runs);

        RunTimes t_mel;
        std::string text_tmp;
        std::vector<int32_t> ids_tmp;
        int n_frames_tmp = 0;
        auto clk = std::chrono::steady_clock::now();
        std::vector<float> mel_buf;
        if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                     model.mel_cfg, mel_buf, n_frames_tmp); rc != 0) {
            PARAKEET_LOG_ERROR("profile: mel failed rc=%d\n", rc);
            return 20;
        }
        const double mel_ms = ms_since(clk);
        PARAKEET_LOG_INFO("[profile] mel preprocess: %.2f ms   (audio=%.2fs, mel_frames=%d)\n",
                     mel_ms, audio_ms / 1000.0, n_frames_tmp);

        const int nl_full = (int) model.encoder_cfg.n_layers;
        const int nl_mid  = std::max(2, nl_full / 2);
        const std::vector<int> layer_points = {0, 1, nl_mid, nl_full};
        std::vector<std::pair<int, AggStats>> results;

        for (int nl : layer_points) {
            std::vector<double> timings;
            timings.reserve(extra.profile_runs);

            EncoderOutputs tmp_out;
            for (int w = 0; w < extra.profile_warmup; ++w) {
                if (int rc = run_encoder(model, mel_buf.data(), n_frames_tmp,
                                         model.mel_cfg.n_mels, tmp_out, nl); rc != 0) {
                    PARAKEET_LOG_ERROR("profile: run_encoder failed rc=%d at nl=%d\n", rc, nl);
                    return 21;
                }
            }
            for (int r = 0; r < extra.profile_runs; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                if (int rc = run_encoder(model, mel_buf.data(), n_frames_tmp,
                                         model.mel_cfg.n_mels, tmp_out, nl); rc != 0) {
                    PARAKEET_LOG_ERROR("profile: run_encoder failed rc=%d at nl=%d\n", rc, nl);
                    return 22;
                }
                timings.push_back(ms_since(t0));
            }
            AggStats s = aggregate(timings);
            results.emplace_back(nl, s);
            PARAKEET_LOG_INFO("[profile] n_layers=%2d:  mean=%7.2f ms   median=%7.2f ms   min=%7.2f ms   max=%7.2f ms   std=%6.2f\n",
                         nl, s.mean, s.median, s.min, s.max, s.stdev);
        }

        double t0 = 0, t1 = 0, t_mid = 0, t_full = 0;
        for (auto & kv : results) {
            if (kv.first == 0)       t0     = kv.second.median;
            if (kv.first == 1)       t1     = kv.second.median;
            if (kv.first == nl_mid)  t_mid  = kv.second.median;
            if (kv.first == nl_full) t_full = kv.second.median;
        }

        const double per_block_from_1_to_full = (t_full - t1) / (double)(nl_full - 1);
        const double per_block_from_1_to_mid  = (t_mid  - t1) / (double)(nl_mid - 1);
        const double block_0_extra = t1 - t0 - per_block_from_1_to_full;
        const double sub_plus_ctc = t0;

        PARAKEET_LOG_INFO("\n[profile] ---------- encoder attribution (median ms) ----------\n");
        PARAKEET_LOG_INFO("[profile]   mel preprocess                    %7.2f   (%.1f%% of total)\n",
                     mel_ms, mel_ms / (mel_ms + t_full) * 100.0);
        PARAKEET_LOG_INFO("[profile]   subsampling + CTC head (nl=0)     %7.2f   (%.1f%% of total)\n",
                     sub_plus_ctc, sub_plus_ctc / (mel_ms + t_full) * 100.0);
        PARAKEET_LOG_INFO("[profile]   block-0 overhead above avg block  %+7.2f   (extra captures / first-block warmup)\n",
                     block_0_extra);
        PARAKEET_LOG_INFO("[profile]   per-block avg (nl=1..%d range)    %7.2f   (x %d blocks = %7.2f ms, %.1f%% of total)\n",
                     nl_full,
                     per_block_from_1_to_full, nl_full,
                     per_block_from_1_to_full * nl_full,
                     per_block_from_1_to_full * nl_full / (mel_ms + t_full) * 100.0);
        PARAKEET_LOG_INFO("[profile]   per-block avg (nl=1..%d range)    %7.2f   (sanity check)\n",
                     nl_mid, per_block_from_1_to_mid);
        PARAKEET_LOG_INFO("[profile]   full encoder (nl=%d)               %7.2f\n",
                     nl_full, t_full);
        PARAKEET_LOG_INFO("[profile]   total (mel + encoder)              %7.2f   RTF = %.4f\n",
                     mel_ms + t_full, (mel_ms + t_full) / audio_ms);
        PARAKEET_LOG_INFO("[profile] -------------------------------------------------------\n");

        const int T_enc = n_frames_tmp / 8;
        PARAKEET_LOG_INFO("\n[profile] sub-stage breakdown of a single conformer block (T_enc=%d)\n", T_enc);
        parakeet::BlockSubstageTimes sub;
        if (parakeet::profile_block_substages(model, T_enc,
                extra.profile_warmup, extra.profile_runs, sub) == 0) {
            const double sum = sub.ff1_ms + sub.attn_ms + sub.conv_ms + sub.ff2_ms + sub.norm_out_ms;
            auto row = [&](const char * label, double ms) {
                const double pct_block = sub.block_full_ms > 0 ? ms / sub.block_full_ms * 100.0 : 0.0;
                const double pct_sum   = sum > 0              ? ms / sum * 100.0               : 0.0;
                PARAKEET_LOG_INFO("[profile]   %-14s %7.2f ms   (%5.1f%% of sum-of-parts,  %5.1f%% of full-block)\n",
                             label, ms, pct_sum, pct_block);
            };
            row("FF1  (macaron)", sub.ff1_ms);
            row("Attention",      sub.attn_ms);
            row("Conv module",    sub.conv_ms);
            row("FF2  (macaron)", sub.ff2_ms);
            row("norm_out",       sub.norm_out_ms);
            PARAKEET_LOG_INFO("[profile]   %-14s %7.2f ms   (sum of parts, slight overhead vs full)\n",
                         "sum of parts", sum);
            PARAKEET_LOG_INFO("[profile]   %-14s %7.2f ms   (actual full-block forward)\n",
                         "full block", sub.block_full_ms);

            const double per_block_measured = per_block_from_1_to_full;
            const double n_layers_full = (double) model.encoder_cfg.n_layers;
            PARAKEET_LOG_INFO("\n[profile] extrapolated cost over all %d blocks (mean per-block = %.2f ms):\n",
                         (int) n_layers_full, per_block_measured);
            auto extrap = [&](const char * label, double ms) {
                const double frac = sum > 0 ? ms / sum : 0.0;
                const double total_ms = frac * per_block_measured * n_layers_full;
                PARAKEET_LOG_INFO("[profile]   %-14s ~%7.2f ms across encoder  (= %.1f%% of encoder time)\n",
                             label, total_ms, total_ms / t_full * 100.0);
            };
            extrap("FF1",       sub.ff1_ms);
            extrap("Attention", sub.attn_ms);
            extrap("Conv",      sub.conv_ms);
            extrap("FF2",       sub.ff2_ms);
            extrap("norm_out",  sub.norm_out_ms);
        } else {
            PARAKEET_LOG_ERROR("[profile] sub-stage profiling failed\n");
        }
        return 0;
    }

    if (extra.stream) {
        EngineOptions eopts;
        eopts.model_gguf_path = opts.model_gguf_path;
        eopts.n_gpu_layers    = opts.n_gpu_layers;
        eopts.n_threads       = opts.n_threads;
        eopts.verbose         = opts.verbose;

        Engine engine(eopts);

        StreamingOptions sopts;
        sopts.sample_rate       = sr;
        sopts.chunk_ms          = extra.stream_chunk_ms;
        if (extra.stream_left_ms  >= 0) sopts.left_context_ms    = extra.stream_left_ms;
        if (extra.stream_right_ms >= 0) sopts.right_lookahead_ms = extra.stream_right_ms;

        const std::string emit_fmt = extra.emit_format;
        const auto t_stream = clock::now();

        if (extra.stream_duplex) {
            int segment_count = 0;
            auto sess = engine.stream_start(sopts,
                [&](const StreamingSegment & seg) {
                    emit_segment(seg, emit_fmt);
                    ++segment_count;
                });

            const int feed_bytes = extra.stream_feed_bytes > 0
                                 ? extra.stream_feed_bytes
                                 : 4096;
            const int feed_samples = std::max(1, feed_bytes / (int) sizeof(float));

            for (int i = 0; i < (int) samples.size(); i += feed_samples) {
                const int n = std::min(feed_samples, (int) samples.size() - i);
                sess->feed_pcm_f32(samples.data() + i, n);
            }
            sess->finalize();

            const double stream_ms = ms_since(t_stream);
            if (opts.verbose) {
                PARAKEET_LOG_INFO(
                    "[stream-duplex] load=%.1fms audio=%.2fs samples=%zu@%dHz\n"
                    "[stream-duplex] chunk_ms=%d left=%dms right=%dms segments=%d total=%.1fms RTF=%.3f\n",
                    load_ms, audio_ms / 1000.0, samples.size(), sr,
                    sopts.chunk_ms, sopts.left_context_ms, sopts.right_lookahead_ms,
                    segment_count, stream_ms, stream_ms / audio_ms);
            }
            return 0;
        }

        auto result = engine.transcribe_samples_stream(
            samples.data(), (int) samples.size(), sr, sopts,
            [&](const StreamingSegment & seg) {
                emit_segment(seg, emit_fmt);
            });

        const double stream_ms = ms_since(t_stream);
        if (opts.verbose) {
            PARAKEET_LOG_INFO(
                "[stream] load=%.1fms audio=%.2fs samples=%zu@%dHz mel_frames=%d enc_frames=%d\n"
                "[stream] mel=%.1fms enc=%.1fms dec=%.1fms total=%.1fms RTF=%.3f tokens=%zu\n",
                load_ms, audio_ms / 1000.0, samples.size(), sr,
                result.mel_frames, result.encoder_frames,
                result.preprocess_ms, result.encoder_ms,
                result.decode_ms, stream_ms, stream_ms / audio_ms,
                result.token_ids.size());
        }
        return 0;
    }

    if (!extra.bench) {
        RunTimes times;
        if (int rc = run_once(text, ids, n_frames, times); rc != 0) return 6 + rc;
        std::printf("%s\n", text.c_str());
        if (opts.verbose) {
            const double inf_rtf   = times.inference_ms / audio_ms;
            const double total_ms  = ms_since(t_load);
            const double total_rtf = total_ms / audio_ms;
            PARAKEET_LOG_INFO(
                "[BENCH] load=%.1fms wav=%.1fs (%zu samples@%dHz) mel=%dx%d\n"
                "[BENCH] mel=%.1fms enc=%.1fms dec=%.1fms inference=%.1fms  RTF=%.3f\n"
                "[BENCH] total(load+wav+inf)=%.1fms  total_RTF=%.3f  tokens=%zu\n",
                load_ms, audio_ms / 1000.0, samples.size(), sr, n_frames, model.mel_cfg.n_mels,
                times.mel_ms, times.enc_ms, times.dec_ms, times.inference_ms, inf_rtf,
                total_ms, total_rtf, ids.size());
        }
        return 0;
    }

    PARAKEET_LOG_INFO("[bench] model=%s  wav=%s (%.2f s audio, %d samples @ %d Hz)\n",
                 opts.model_gguf_path.c_str(), opts.wav_path.c_str(),
                 audio_ms / 1000.0, (int) samples.size(), sr);
    PARAKEET_LOG_INFO("[bench] threads=%d  warmup=%d  runs=%d\n",
                 opts.n_threads, extra.bench_warmup, extra.bench_runs);
    PARAKEET_LOG_INFO("[bench] load=%.1fms wav_read=%.1fms\n", load_ms, wav_ms);

    for (int w = 0; w < extra.bench_warmup; ++w) {
        RunTimes t;
        if (int rc = run_once(text, ids, n_frames, t); rc != 0) return 10 + rc;
        PARAKEET_LOG_INFO("[bench] warmup %d/%d  mel=%.1fms enc=%.1fms dec=%.1fms  RTF=%.3f\n",
                     w + 1, extra.bench_warmup, t.mel_ms, t.enc_ms, t.dec_ms, t.inference_ms / audio_ms);
    }

    std::vector<double> mel_v, enc_v, dec_v, inf_v;
    mel_v.reserve(extra.bench_runs);
    enc_v.reserve(extra.bench_runs);
    dec_v.reserve(extra.bench_runs);
    inf_v.reserve(extra.bench_runs);

    int enc_frames_last = 0;
    for (int r = 0; r < extra.bench_runs; ++r) {
        RunTimes t;
        if (int rc = run_once(text, ids, n_frames, t); rc != 0) return 20 + rc;
        mel_v.push_back(t.mel_ms);
        enc_v.push_back(t.enc_ms);
        dec_v.push_back(t.dec_ms);
        inf_v.push_back(t.inference_ms);
        enc_frames_last = t.encoder_frames;
        PARAKEET_LOG_INFO("[bench] run %d/%d    mel=%.1fms enc=%.1fms dec=%.1fms inference=%.1fms  RTF=%.3f\n",
                     r + 1, extra.bench_runs, t.mel_ms, t.enc_ms, t.dec_ms,
                     t.inference_ms, t.inference_ms / audio_ms);
    }

    std::printf("%s\n", text.c_str());

    const AggStats s_mel = aggregate(mel_v);
    const AggStats s_enc = aggregate(enc_v);
    const AggStats s_dec = aggregate(dec_v);
    const AggStats s_inf = aggregate(inf_v);
    const double   rtf_median = s_inf.median / audio_ms;
    const double   rtf_best   = s_inf.min    / audio_ms;
    const double   rtf_mean   = s_inf.mean   / audio_ms;
    const bool     noisy      = s_inf.stdev > 0.2 * s_inf.mean;

    PARAKEET_LOG_INFO(
        "[bench] ----------- summary (%d timed runs, warmup excluded) -----------\n"
        "[bench]   audio              = %.3f s (%zu samples @ %d Hz)\n"
        "[bench]   model load         = %.1f ms\n"
        "[bench]   wav read           = %.1f ms\n"
        "[bench]                         mean      med       min       max       std\n"
        "[bench]   mel        ms    %8.2f  %7.2f  %7.2f  %7.2f  %7.2f\n"
        "[bench]   encoder    ms    %8.2f  %7.2f  %7.2f  %7.2f  %7.2f\n"
        "[bench]   decode     ms    %8.2f  %7.2f  %7.2f  %7.2f  %7.2f\n"
        "[bench]   inference  ms    %8.2f  %7.2f  %7.2f  %7.2f  %7.2f\n"
        "[bench]   RTF (median/best) = %.3f / %.3f    (realtime multiple = %.1fx / %.1fx)\n"
        "[bench]   tokens             = %d%s\n"
        "[bench] ---------------------------------------------------------------%s\n",
        extra.bench_runs, audio_ms / 1000.0, samples.size(), sr,
        load_ms, wav_ms,
        s_mel.mean, s_mel.median, s_mel.min, s_mel.max, s_mel.stdev,
        s_enc.mean, s_enc.median, s_enc.min, s_enc.max, s_enc.stdev,
        s_dec.mean, s_dec.median, s_dec.min, s_dec.max, s_dec.stdev,
        s_inf.mean, s_inf.median, s_inf.min, s_inf.max, s_inf.stdev,
        rtf_median, rtf_best,
        (rtf_median > 0 ? 1.0 / rtf_median : 0.0),
        (rtf_best   > 0 ? 1.0 / rtf_best   : 0.0),
        (int) ids.size(),
        noisy ? "  (WARNING: stdev > 20% of mean — consider more warmup / runs)" : "",
        noisy ? "\n[bench] prefer the median / best RTF — mean is skewed by outliers." : "");

    if (!extra.bench_json_path.empty()) {
        FILE * fp = std::fopen(extra.bench_json_path.c_str(), "w");
        if (!fp) {
            PARAKEET_LOG_ERROR("error: cannot open %s for writing\n", extra.bench_json_path.c_str());
            return 30;
        }
        auto fmt_stats = [&](const char * name, const AggStats & s, const std::vector<double> & v) {
            std::fprintf(fp, "    \"%s_ms\": {\"mean\": %.3f, \"median\": %.3f, \"stdev\": %.3f, \"min\": %.3f, \"max\": %.3f, \"samples\": [",
                         name, s.mean, s.median, s.stdev, s.min, s.max);
            for (size_t i = 0; i < v.size(); ++i) std::fprintf(fp, "%s%.3f", i == 0 ? "" : ", ", v[i]);
            std::fprintf(fp, "]}");
        };
        std::fprintf(fp, "{\n");
        std::fprintf(fp, "  \"model\": \"%s\",\n",  opts.model_gguf_path.c_str());
        std::fprintf(fp, "  \"wav\": \"%s\",\n",    opts.wav_path.c_str());
        // Runtime-accurate backend label. Reads the post-fallback active
        // backend off the loaded model rather than guessing from
        // compile-time #ifdefs. Covers OpenCL (which the previous
        // ifdef cascade missed entirely) and disambiguates correctly
        // when multiple GPU backends are compiled into the same
        // binary -- ggml_backend_name() prints what was actually
        // selected by `init_gpu_backend()`'s CUDA -> Metal -> Vulkan ->
        // OpenCL -> CPU cascade. Format mirrors the legacy strings
        // ("ggml-metal" / "ggml-cuda" / etc.) so downstream bench
        // sweeps that grep on these labels keep working; we just
        // lower-case ggml_backend_name's "Metal" / "CUDA0" / "OpenCL"
        // to "ggml-metal" / "ggml-cuda0" / "ggml-opencl".
        std::string backend_label;
        if (ggml_backend_t b = model.backend_active(); b != nullptr) {
            const char * raw = ggml_backend_name(b);
            backend_label = std::string("ggml-") + (raw ? raw : "cpu");
            std::transform(backend_label.begin(), backend_label.end(),
                           backend_label.begin(),
                           [](unsigned char c) { return std::tolower(c); });
        } else {
            backend_label = "ggml-cpu";
        }
        std::fprintf(fp, "  \"backend\": \"%s\",\n", backend_label.c_str());
        std::fprintf(fp, "  \"n_gpu_layers\": %d,\n", opts.n_gpu_layers);
        std::fprintf(fp, "  \"threads\": %d,\n",    opts.n_threads);
        std::fprintf(fp, "  \"warmup_runs\": %d,\n", extra.bench_warmup);
        std::fprintf(fp, "  \"timed_runs\":  %d,\n", extra.bench_runs);
        std::fprintf(fp, "  \"audio_seconds\": %.6f,\n", audio_ms / 1000.0);
        std::fprintf(fp, "  \"audio_samples\": %zu,\n", samples.size());
        std::fprintf(fp, "  \"sample_rate\": %d,\n", sr);
        std::fprintf(fp, "  \"mel_frames\": %d,\n",  n_frames);
        std::fprintf(fp, "  \"encoder_frames\": %d,\n", enc_frames_last);
        std::fprintf(fp, "  \"tokens\": %d,\n",      (int) ids.size());
        std::fprintf(fp, "  \"transcript\": \"");
        for (char c : text) { if (c == '"') std::fputs("\\\"", fp); else if (c == '\\') std::fputs("\\\\", fp); else std::fputc(c, fp); }
        std::fprintf(fp, "\",\n");
        std::fprintf(fp, "  \"load_ms\": %.3f,\n",       load_ms);
        std::fprintf(fp, "  \"wav_read_ms\": %.3f,\n",   wav_ms);
        fmt_stats("mel",       s_mel, mel_v);           std::fprintf(fp, ",\n");
        fmt_stats("encoder",   s_enc, enc_v);           std::fprintf(fp, ",\n");
        fmt_stats("decode",    s_dec, dec_v);           std::fprintf(fp, ",\n");
        fmt_stats("inference", s_inf, inf_v);           std::fprintf(fp, ",\n");
        std::fprintf(fp, "    \"rtf_mean\":   %.6f,\n", rtf_mean);
        std::fprintf(fp, "    \"rtf_median\": %.6f,\n", rtf_median);
        std::fprintf(fp, "    \"rtf_best\":   %.6f\n",  rtf_best);
        std::fprintf(fp, "}\n");
        std::fclose(fp);
        PARAKEET_LOG_INFO("[bench] wrote %s\n", extra.bench_json_path.c_str());
    }

    return 0;
}
