#include "tts-cpp/supertonic/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model supertonic2.gguf --text TEXT --out out.wav\n"
        "          [--language en] [--voice NAME] [--steps N] [--speed X]\n"
        "          (voice/steps/speed default to GGUF metadata when omitted)\n"
        "          [--seed 42] [--threads N] [--n-gpu-layers N]\n"
        "          [--vulkan-device N] (Vulkan adapter index; ignored unless\n"
        "                            built with -DGGML_VULKAN=ON; default 0,\n"
        "                            -1 = auto-pick adapter with most free VRAM)\n"
        "          [--f16-attn 0|1] (vector-estimator F16 K/V attention;\n"
        "                            defaults to auto: on for GPU, off for CPU)\n"
        "          [--f16-weights 0|1] (load-time F16 materialization for the\n"
        "                            audit-identified hot matmul / pwconv weights;\n"
        "                            defaults to auto: on for GPU, off for CPU)\n"
        "          [--precision f32|f16|q8_0]   (default: f32)\n"
        "          [--f16-weights-deny PATTERN1,PATTERN2,...] (substring patterns,\n"
        "                            comma-separated; matching tensors stay F32 even\n"
        "                            when --f16-weights is on.  Default empty.)\n"
        "          [--prewarm TEXT] (run one throwaway synth on TEXT at engine\n"
        "                            construction so first-real-call latency on\n"
        "                            Vulkan / OpenCL doesn't pay the shader-\n"
        "                            compile cost; no-op on CPU)\n"
        "          [--noise-npy /path/to/noise.npy]\n"
        "          [--stream-chunk-tokens N]    (0 = batch; >0 enables\n"
        "                            streaming with target ~N text-token chunks)\n"
        "          [--stream-first-chunk-tokens N]  (override 1st-chunk target;\n"
        "                            0 = same as --stream-chunk-tokens)\n"
        "          [--stream-chunk-tolerance-pct N] (boundary-snap window; default 20)\n"
        "          [--stream-min-chunk-tokens N]    (hard floor on chunk size;\n"
        "                            default 30 — below this the model glitches\n"
        "                            on stub input; chunks below the floor are\n"
        "                            merged with their neighbor)\n"
        "\n"
        "          When --out is '-', the CLI emits raw s16le PCM to stdout as\n"
        "          each chunk completes.  Pipe into a player, e.g.:\n"
        "            %s --model ... --text '...' --out - --stream-chunk-tokens 50 \\\n"
        "              | aplay -f S16_LE -r 44100 -c 1\n",
        argv0, argv0);
}

tts_cpp::supertonic::Precision parse_precision(const std::string & s) {
    if (s == "f32" || s == "F32") return tts_cpp::supertonic::Precision::F32;
    if (s == "f16" || s == "F16") return tts_cpp::supertonic::Precision::F16;
    if (s == "q8_0" || s == "Q8_0" || s == "q8") return tts_cpp::supertonic::Precision::Q8_0;
    throw std::runtime_error("unknown --precision value: " + s + " (expected f32|f16|q8_0)");
}

// Emit `pcm` as raw signed-16-bit little-endian samples on stdout.  Used
// by the streaming path so a consumer like `ffplay -f s16le -ar 44100 ...`
// can begin playback as soon as the first chunk arrives.  Builds the
// full chunk's worth of int16 into a contiguous buffer and writes it
// with a single fwrite — a per-sample fwrite loop would do ~44k-132k
// syscall-adjacent calls per chunk and noticeably tax streaming
// throughput on slower terminals / pipes.
void stream_emit_pcm_stdout(const float * pcm, std::size_t samples) {
    std::vector<int16_t> buf(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        float c = std::max(-1.0f, std::min(1.0f, pcm[i]));
        buf[i] = (int16_t) std::lrintf(c * 32767.0f);
    }
    std::fwrite(buf.data(), sizeof(int16_t), samples, stdout);
    std::fflush(stdout);
}

void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output wav: " + path);
    uint32_t n = (uint32_t) wav.size();
    uint32_t byte_rate = (uint32_t) sr * 2;
    uint32_t data_size = n * 2;
    uint32_t chunk_size = 36 + data_size;
    uint16_t fmt = 1, channels = 1, align = 2, bps = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f); std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, f); std::fwrite(&fmt, 2, 1, f);
    std::fwrite(&channels, 2, 1, f); std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f); std::fwrite(&align, 2, 1, f);
    std::fwrite(&bps, 2, 1, f); std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float c = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t) std::lrintf(c * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

} // namespace

int main(int argc, char ** argv) {
    tts_cpp::supertonic::EngineOptions opts;
    std::string text;
    std::string out;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
            return argv[++i];
        };
        if (arg == "--model") opts.model_gguf_path = next("--model");
        else if (arg == "--text") text = next("--text");
        else if (arg == "--out") out = next("--out");
        else if (arg == "--language") opts.language = next("--language");
        else if (arg == "--voice") opts.voice = next("--voice");
        else if (arg == "--steps") opts.steps = std::stoi(next("--steps"));
        else if (arg == "--speed") opts.speed = std::stof(next("--speed"));
        else if (arg == "--seed") opts.seed = std::stoi(next("--seed"));
        else if (arg == "--threads") opts.n_threads = std::stoi(next("--threads"));
        else if (arg == "--n-gpu-layers") opts.n_gpu_layers = std::stoi(next("--n-gpu-layers"));
        else if (arg == "--vulkan-device") opts.vulkan_device = std::stoi(next("--vulkan-device"));
        else if (arg == "--f16-attn") opts.f16_attn = std::stoi(next("--f16-attn"));
        else if (arg == "--f16-weights") opts.f16_weights = std::stoi(next("--f16-weights"));
        else if (arg == "--precision") opts.precision = parse_precision(next("--precision"));
        else if (arg == "--f16-weights-deny") {
            // Comma-split into a vector<string>.  Empty entries
            // are tolerated (predicate skips them defensively).
            opts.f16_weights_deny_list.clear();
            const std::string raw = next("--f16-weights-deny");
            size_t start = 0;
            for (size_t k = 0; k <= raw.size(); ++k) {
                if (k == raw.size() || raw[k] == ',') {
                    opts.f16_weights_deny_list.emplace_back(raw.substr(start, k - start));
                    start = k + 1;
                }
            }
        }
        else if (arg == "--prewarm") opts.prewarm_text = next("--prewarm");
        else if (arg == "--noise-npy") opts.noise_npy_path = next("--noise-npy");
        else if (arg == "--stream-chunk-tokens") {
            opts.stream_chunk_tokens = std::stoi(next("--stream-chunk-tokens"));
        }
        else if (arg == "--stream-first-chunk-tokens") {
            opts.stream_first_chunk_tokens = std::stoi(next("--stream-first-chunk-tokens"));
        }
        else if (arg == "--stream-chunk-tolerance-pct") {
            opts.stream_chunk_tolerance_pct = std::stoi(next("--stream-chunk-tolerance-pct"));
        }
        else if (arg == "--stream-min-chunk-tokens") {
            opts.stream_min_chunk_tokens = std::stoi(next("--stream-min-chunk-tokens"));
        }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", arg.c_str()); usage(argv[0]); return 2; }
    }
    if (opts.model_gguf_path.empty() || text.empty() || out.empty()) {
        usage(argv[0]);
        return 2;
    }
    try {
        const bool streaming = opts.stream_chunk_tokens > 0;
        const bool stdout_pcm = (out == "-");

        if (!streaming) {
            if (stdout_pcm) {
                fprintf(stderr,
                    "error: --out - requires --stream-chunk-tokens > 0 "
                    "(stdout streaming is the streaming-mode output)\n");
                return 2;
            }
            auto result = tts_cpp::supertonic::synthesize(opts, text);
            write_wav(out, result.pcm, result.sample_rate);
            fprintf(stderr, "wrote %s (%.2fs @ %d Hz, %zu samples)\n",
                    out.c_str(), result.duration_s, result.sample_rate, result.pcm.size());
            return 0;
        }

        // Streaming path.  Construct a persistent Engine so per-chunk
        // synth doesn't pay GGUF load each iteration.
        tts_cpp::supertonic::Engine engine(opts);
        if (stdout_pcm) {
            fprintf(stderr,
                "streaming: emitting raw s16le PCM on stdout "
                "(chunk target: %d text tokens; first chunk: %d; backend: %s)\n",
                opts.stream_chunk_tokens,
                opts.stream_first_chunk_tokens > 0
                    ? opts.stream_first_chunk_tokens
                    : opts.stream_chunk_tokens,
                engine.backend_name().c_str());
        }

        // Optional per-chunk WAV dump for debugging.  When the env var
        // SUPERTONIC_DUMP_CHUNK_WAVS_PREFIX is set, the callback writes
        // each chunk's PCM to "<prefix><idx>.wav" so you can play chunks
        // individually and see which one contains a glitch.
        const char * dump_prefix = std::getenv("SUPERTONIC_DUMP_CHUNK_WAVS_PREFIX");

        std::size_t total_samples = 0;
        int          n_chunks      = 0;
        auto on_chunk = [&](const float * pcm, std::size_t samples,
                            int chunk_index, bool is_last) {
            if (stdout_pcm) {
                stream_emit_pcm_stdout(pcm, samples);
            }
            if (dump_prefix) {
                std::string path = std::string(dump_prefix)
                    + std::to_string(chunk_index) + ".wav";
                std::vector<float> tmp(pcm, pcm + samples);
                // 44.1 kHz is the Supertonic model default; the real SR
                // comes back on the final SynthesisResult but isn't
                // visible here.  Hard-coding here is fine for a debug
                // dump — if a future model ships at a different SR this
                // will be wrong, but the callback signature doesn't
                // surface it.
                write_wav(path, tmp, 44100);
            }
            total_samples += samples;
            ++n_chunks;
            fprintf(stderr,
                    "chunk %d%s: %zu samples%s%s\n",
                    chunk_index, is_last ? " (last)" : "",
                    samples,
                    stdout_pcm ? " -> stdout" : "",
                    dump_prefix ? " (+ dumped)" : "");
        };

        auto result = engine.synthesize(text, on_chunk);

        if (!stdout_pcm) {
            // File mode: write the concatenated PCM as a WAV.
            write_wav(out, result.pcm, result.sample_rate);
            fprintf(stderr, "wrote %s (%.2fs @ %d Hz, %zu samples across %d chunks)\n",
                    out.c_str(), result.duration_s, result.sample_rate,
                    result.pcm.size(), n_chunks);
        } else {
            fprintf(stderr, "streamed %zu samples across %d chunks (%.2fs)\n",
                    total_samples, n_chunks, result.duration_s);
        }
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
