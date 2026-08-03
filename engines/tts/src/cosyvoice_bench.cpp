// cosyvoice-bench: per-stage CosyVoice3 benchmark over the public Engine.
//
// Mirrors parler-bench's JSON schema so the existing .diag summarisers read it
// unchanged.  Two things here exist specifically to keep an A/B honest:
//
//   --tokens-out / --tokens-in  pin the LM speech-token trajectory.  Sampling
//       is chaotic in the logits, so a CPU leg and a GPU leg will not agree on
//       tokens even at the same seed -- and then they synthesise different
//       amounts of audio and the comparison is meaningless.  Run the CPU leg
//       with --tokens-out, every other leg with --tokens-in.
//   the JSON carries n_speech_tokens / tm / mel_len / n_samples, so a
//       cross-leg check can assert equal work instead of trusting the operator.

#include "tts-cpp/cosyvoice/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace tts_cpp::cosyvoice;

namespace {

struct Stage {
    std::string name;
    std::vector<double> ms;
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t) idx;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - (double) lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}
double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }
double mean(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double s = 0; for (double x : v) s += x; return s / (double) v.size();
}
double minv(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double m = v[0]; for (double x : v) m = std::min(m, x); return m;
}
double maxv(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double m = v[0]; for (double x : v) m = std::max(m, x); return m;
}

void print_stage(const Stage & s) {
    if (s.ms.empty()) { printf("  %-16s n=0\n", s.name.c_str()); return; }
    printf("  %-16s n=%zu  min=%9.2f  med=%9.2f  mean=%9.2f  p95=%9.2f  max=%9.2f  ms\n",
           s.name.c_str(), s.ms.size(),
           minv(s.ms), median(s.ms), mean(s.ms), percentile(s.ms, 0.95), maxv(s.ms));
}

std::string json_escape(const std::string & s) {
    std::string out;
    for (char ch : s) {
        if (ch == '\\' || ch == '"') { out.push_back('\\'); out.push_back(ch); }
        else if (ch == '\n') out += "\\n";
        else out.push_back(ch);
    }
    return out;
}

void write_json_stage(std::ofstream & os, const Stage & s, bool comma) {
    os << "    \"" << json_escape(s.name) << "\": {"
       << "\"n\": " << s.ms.size()
       << ", \"min_ms\": " << minv(s.ms)
       << ", \"median_ms\": " << median(s.ms)
       << ", \"mean_ms\": " << mean(s.ms)
       << ", \"p95_ms\": " << percentile(s.ms, 0.95)
       << ", \"max_ms\": " << maxv(s.ms)
       << "}" << (comma ? "," : "") << "\n";
}

void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
    uint32_t num_samples = (uint32_t) wav.size();
    uint32_t byte_rate = sr * 2, data_size = num_samples * 2, chunk_size = 36 + data_size;
    uint32_t fcs = 16, sr32 = (uint32_t) sr; uint16_t af = 1, nc = 1, ba = 2, bps = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); std::fwrite(&fcs, 4, 1, f); std::fwrite(&af, 2, 1, f); std::fwrite(&nc, 2, 1, f);
    std::fwrite(&sr32, 4, 1, f); std::fwrite(&byte_rate, 4, 1, f); std::fwrite(&ba, 2, 1, f); std::fwrite(&bps, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float c = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t) std::lrintf(c * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

bool read_tokens(const std::string & path, std::vector<int> & out) {
    std::ifstream is(path);
    if (!is) return false;
    out.clear();
    int t;
    while (is >> t) out.push_back(t);
    return !out.empty();
}

bool write_tokens(const std::string & path, const std::vector<int> & toks) {
    std::ofstream os(path);
    if (!os) return false;
    for (size_t i = 0; i < toks.size(); ++i) os << toks[i] << (i + 1 == toks.size() ? '\n' : ' ');
    return true;
}

void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s --model-dir DIR [--text TEXT]\n"
        "          [--n-gpu-layers N] [--threads N] [--seed 42] [--greedy]\n"
        "          [--runs 3] [--warmup 1]\n"
        "          [--tokens-out FILE]  pin: write the LM trajectory this run used\n"
        "          [--tokens-in FILE]   pin: reuse a trajectory (skips the LM)\n"
        "          [--backends-dir DIR] [--opencl-cache-dir DIR]\n"
        "          [--wav-out FILE] [--json-out FILE]\n", a0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_dir, text = "The quick brown fox jumps over the lazy dog.";
    std::string tokens_out, tokens_in, wav_out, json_out, backends_dir, opencl_cache_dir;
    int seed = 42, n_gpu_layers = 0, n_threads = 0, runs = 3, warmup = 1;
    bool greedy = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--text" && i + 1 < argc) text = argv[++i];
        else if ((a == "--n-gpu-layers" || a == "-ngl") && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if ((a == "--threads" || a == "-t") && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--greedy") greedy = true;
        else if (a == "--tokens-out" && i + 1 < argc) tokens_out = argv[++i];
        else if (a == "--tokens-in" && i + 1 < argc) tokens_in = argv[++i];
        else if (a == "--wav-out" && i + 1 < argc) wav_out = argv[++i];
        else if (a == "--json-out" && i + 1 < argc) json_out = argv[++i];
        else if (a == "--backends-dir" && i + 1 < argc) backends_dir = argv[++i];
        else if (a == "--opencl-cache-dir" && i + 1 < argc) opencl_cache_dir = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if (model_dir.empty()) { usage(argv[0]); return 1; }

    EngineOptions opts;
    opts.model_dir        = model_dir;
    opts.seed             = seed;
    opts.greedy           = greedy;
    opts.n_gpu_layers     = n_gpu_layers;
    opts.n_threads        = n_threads;
    if (!backends_dir.empty())     opts.backends_dir     = backends_dir;
    if (!opencl_cache_dir.empty()) opts.opencl_cache_dir = opencl_cache_dir;

    if (!tokens_in.empty()) {
        if (!read_tokens(tokens_in, opts.force_speech_tokens)) {
            fprintf(stderr, "cosyvoice-bench: cannot read --tokens-in %s\n", tokens_in.c_str());
            return 1;
        }
        fprintf(stderr, "pinned %zu speech tokens from %s\n",
                opts.force_speech_tokens.size(), tokens_in.c_str());
    }

    fprintf(stderr, "loading %s ...\n", model_dir.c_str());
    const auto t_load0 = std::chrono::steady_clock::now();
    Engine engine(opts);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

    printf("backend: %s%s   n_gpu_layers=%d  threads=%d  greedy=%d\n",
           engine.backend_name().c_str(),
           engine.gpu_unsupported() ? " (GPU present but declined)" : "",
           n_gpu_layers, n_threads, (int) greedy);
    printf("load: %.1f ms\n", load_ms);

    Stage st_lm_pre{"lm_prefill", {}}, st_lm_dec{"lm_decode", {}};
    Stage st_fe{"flow_frontend", {}}, st_dit{"dit_euler", {}};
    Stage st_f0{"hift_f0", {}}, st_src{"hift_source", {}};
    Stage st_stft{"hift_stft", {}}, st_hdec{"hift_decode", {}};
    Stage st_tot{"total", {}};
    std::vector<double> rtfs;

    SynthesisResult last;
    for (int r = 0; r < warmup + runs; ++r) {
        SynthesisResult res = engine.synthesize(text);
        const bool measured = r >= warmup;
        if (measured) {
            const StageTimings & t = res.timings;
            st_lm_pre.ms.push_back(t.lm_prefill_ms);
            st_lm_dec.ms.push_back(t.lm_decode_ms);
            st_fe.ms.push_back(t.flow_frontend_ms);
            st_dit.ms.push_back(t.dit_euler_ms);
            st_f0.ms.push_back(t.hift_f0_ms);
            st_src.ms.push_back(t.hift_source_ms);
            st_stft.ms.push_back(t.hift_stft_ms);
            st_hdec.ms.push_back(t.hift_decode_ms);
            st_tot.ms.push_back(t.total_ms);
            if (res.duration_s > 0) rtfs.push_back((t.total_ms / 1000.0) / res.duration_s);
        }
        fprintf(stderr, "  run %d/%d%s  total=%.1f ms  audio=%.2f s  tokens=%d\n",
                r + 1, warmup + runs, measured ? "" : " (warmup)",
                res.timings.total_ms, res.duration_s, res.timings.n_speech_tokens);
        last = std::move(res);
    }

    printf("\nwork done (must match across legs, else the comparison is void):\n");
    printf("  speech_tokens=%d  decode_steps=%d  tm=%d  mel_len=%d  samples=%zu  audio=%.3f s\n",
           last.timings.n_speech_tokens, last.timings.n_decode_steps,
           last.timings.tm, last.timings.mel_len, last.pcm.size(), last.duration_s);

    printf("\nper-stage (over %d measured runs):\n", runs);
    for (const Stage * s : { &st_lm_pre, &st_lm_dec, &st_fe, &st_dit,
                             &st_f0, &st_src, &st_stft, &st_hdec, &st_tot }) {
        print_stage(*s);
    }
    if (!rtfs.empty()) {
        printf("  RTF med=%.3f  (real-time multiplier %.2fx)\n",
               median(rtfs), 1.0 / median(rtfs));
    }

    if (!tokens_out.empty()) {
        if (write_tokens(tokens_out, last.speech_tokens)) {
            fprintf(stderr, "wrote %zu speech tokens to %s\n",
                    last.speech_tokens.size(), tokens_out.c_str());
        } else {
            fprintf(stderr, "cosyvoice-bench: cannot write --tokens-out %s\n", tokens_out.c_str());
        }
    }
    if (!wav_out.empty()) write_wav(wav_out, last.pcm, last.sample_rate);

    if (!json_out.empty()) {
        std::ofstream os(json_out);
        if (os) {
            os << "{\n";
            os << "  \"runtime\": \"ggml-cpp\",\n";
            os << "  \"engine\": \"cosyvoice3\",\n";
            os << "  \"model_dir\": \"" << json_escape(model_dir) << "\",\n";
            os << "  \"backend\": \"" << json_escape(engine.backend_name()) << "\",\n";
            os << "  \"gpu_declined\": " << (engine.gpu_unsupported() ? "true" : "false") << ",\n";
            os << "  \"n_gpu_layers\": " << n_gpu_layers << ",\n";
            os << "  \"threads\": " << n_threads << ",\n";
            os << "  \"greedy\": " << (greedy ? "true" : "false") << ",\n";
            os << "  \"tokens_pinned\": " << (tokens_in.empty() ? "false" : "true") << ",\n";
            os << "  \"seed\": " << seed << ",\n";
            os << "  \"runs\": " << runs << ",\n";
            os << "  \"warmup\": " << warmup << ",\n";
            os << "  \"load_ms\": " << load_ms << ",\n";
            os << "  \"audio_s\": " << last.duration_s << ",\n";
            os << "  \"work\": {"
               << "\"speech_tokens\": " << last.timings.n_speech_tokens
               << ", \"decode_steps\": " << last.timings.n_decode_steps
               << ", \"tm\": " << last.timings.tm
               << ", \"mel_len\": " << last.timings.mel_len
               << ", \"n_samples\": " << last.pcm.size() << "},\n";
            os << "  \"rtf\": {"
               << "\"min\": " << minv(rtfs) << ", \"median\": " << median(rtfs)
               << ", \"mean\": " << mean(rtfs) << ", \"max\": " << maxv(rtfs) << "},\n";
            os << "  \"stages\": {\n";
            write_json_stage(os, st_lm_pre, true);
            write_json_stage(os, st_lm_dec, true);
            write_json_stage(os, st_fe,     true);
            write_json_stage(os, st_dit,    true);
            write_json_stage(os, st_f0,     true);
            write_json_stage(os, st_src,    true);
            write_json_stage(os, st_stft,   true);
            write_json_stage(os, st_hdec,   true);
            write_json_stage(os, st_tot,    false);
            os << "  }\n}\n";
            fprintf(stderr, "wrote %s\n", json_out.c_str());
        } else {
            fprintf(stderr, "cosyvoice-bench: cannot open --json-out %s\n", json_out.c_str());
        }
    }
    return 0;
}
