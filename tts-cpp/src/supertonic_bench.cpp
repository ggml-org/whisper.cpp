// Benchmark for the Supertonic 2 C++ GGML port.
//
// Measures wall-clock time for each stage of the synthesis pipeline:
//   1. text preprocessing -> token ids
//   2. duration predictor
//   3. text encoder
//   4. N denoising steps (vector estimator)
//   5. vocoder
//
// Reports min / median / mean / p95 across `--runs` iterations (after a
// configurable number of warmup runs that are dropped).  An optional
// --noise-npy switches to a fixed noise tensor for reproducible runs.
//
// Usage:
//   ./build/supertonic-bench --model models/supertonic2.gguf \
//       --text "..." [--voice M1] [--language en] [--steps 5] [--speed 1.05] \
//       [--seed 42] [--noise-npy noise.npy] [--runs 5] [--warmup 1] [--json-out result.json]

#include "supertonic_internal.h"
#include "npy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
using ms_t = std::chrono::duration<double, std::milli>;

namespace {

struct Stage {
    std::string name;
    std::vector<double> ms;
};

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model supertonic2.gguf --text TEXT\n"
        "          [--voice M1] [--language en] [--steps 5] [--speed 1.05]\n"
        "          [--seed 42] [--noise-npy /path/to/noise.npy]\n"
        "          [--runs 5] [--warmup 1] [--threads N] [--json-out FILE]\n",
        argv0);
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = (size_t) idx;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - (double) lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double median(std::vector<double> v) { return percentile(v, 0.5); }
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
    if (s.ms.empty()) { printf("  %-22s n=0\n", s.name.c_str()); return; }
    printf("  %-22s n=%zu  min=%7.2f  med=%7.2f  mean=%7.2f  p95=%7.2f  max=%7.2f  ms\n",
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

} // namespace

int main(int argc, char ** argv) {
    using namespace tts_cpp::supertonic::detail;

    std::string model_path, text;
    std::string voice = "M1", language = "en";
    std::string noise_npy;
    std::string json_out;
    int steps = 5;
    float speed = 1.05f;
    int seed = 42;
    int runs = 5;
    int warmup = 1;
    int n_threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * f) {
            if (i + 1 >= argc) throw std::runtime_error(std::string(f) + " requires value");
            return std::string(argv[++i]);
        };
        if (a == "--model") model_path = next("--model");
        else if (a == "--text") text = next("--text");
        else if (a == "--voice") voice = next("--voice");
        else if (a == "--language") language = next("--language");
        else if (a == "--steps") steps = std::stoi(next("--steps"));
        else if (a == "--speed") speed = std::stof(next("--speed"));
        else if (a == "--seed") seed = std::stoi(next("--seed"));
        else if (a == "--noise-npy") noise_npy = next("--noise-npy");
        else if (a == "--runs") runs = std::stoi(next("--runs"));
        else if (a == "--warmup") warmup = std::stoi(next("--warmup"));
        else if (a == "--threads") n_threads = std::stoi(next("--threads"));
        else if (a == "--json-out") json_out = next("--json-out");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || text.empty()) { usage(argv[0]); return 2; }

    supertonic_model model;
    if (!load_supertonic_gguf(model_path, model)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    supertonic_set_n_threads(model, n_threads);

    auto vit = model.voices.find(voice);
    if (vit == model.voices.end()) {
        fprintf(stderr, "unknown voice: %s\n", voice.c_str());
        free_supertonic_model(model);
        return 1;
    }
    std::vector<float> style_ttl((size_t) ggml_nelements(vit->second.ttl));
    std::vector<float> style_dp((size_t) ggml_nelements(vit->second.dp));
    ggml_backend_tensor_get(vit->second.ttl, style_ttl.data(), 0, ggml_nbytes(vit->second.ttl));
    ggml_backend_tensor_get(vit->second.dp,  style_dp.data(),  0, ggml_nbytes(vit->second.dp));

    std::vector<float> ref_noise;
    int ref_noise_len = -1;
    if (!noise_npy.empty()) {
        npy_array n = npy_load(noise_npy);
        if (n.dtype != "<f4" || n.shape.size() != 3 || n.shape[0] != 1 ||
            n.shape[1] != model.hparams.latent_channels) {
            fprintf(stderr, "noise npy must be float32 [1, latent_channels, L]\n");
            free_supertonic_model(model);
            return 1;
        }
        ref_noise_len = (int) n.shape[2];
        ref_noise.resize(n.n_elements());
        std::memcpy(ref_noise.data(), npy_as_f32(n), ref_noise.size() * sizeof(float));
    }

    Stage st_pre{"preprocess", {}};
    Stage st_dur{"duration", {}};
    Stage st_te {"text_encoder", {}};
    Stage st_ve {"vector_estimator (5 step)", {}};
    Stage st_voc{"vocoder", {}};
    Stage st_tot{"total", {}};
    std::vector<double> rtfs;
    double last_audio_s = 0;

    int total_runs = runs + warmup;
    for (int r = 0; r < total_runs; ++r) {
        bool record = r >= warmup;
        std::string error;

        auto t0 = clk::now();

        std::vector<int32_t> text_ids_i32;
        std::string normalized;
        if (!supertonic_text_to_ids(model, text, language, text_ids_i32, &normalized, &error)) {
            fprintf(stderr, "preprocess failed: %s\n", error.c_str());
            free_supertonic_model(model); return 1;
        }
        std::vector<int64_t> text_ids(text_ids_i32.begin(), text_ids_i32.end());
        auto t1 = clk::now();

        float duration_raw = 0;
        if (!supertonic_duration_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                              style_dp.data(), duration_raw, &error)) {
            fprintf(stderr, "duration failed: %s\n", error.c_str());
            free_supertonic_model(model); return 1;
        }
        auto t2 = clk::now();

        const int sample_rate = model.hparams.sample_rate;
        const int chunk = model.hparams.base_chunk_size * model.hparams.ttl_chunk_compress_factor;
        int latent_len;
        std::vector<float> latent;
        if (!ref_noise.empty()) {
            latent_len = ref_noise_len;
            latent = ref_noise;
        } else {
            float duration_s = duration_raw / speed;
            int wav_len = (int) (duration_s * sample_rate);
            latent_len = std::max(1, (wav_len + chunk - 1) / chunk);
            std::mt19937 rng((uint32_t) seed + (uint32_t) r); // unique noise per run
            std::normal_distribution<float> normal(0.0f, 1.0f);
            latent.assign((size_t) model.hparams.latent_channels * latent_len, 0.0f);
            for (float & v : latent) v = normal(rng);
        }

        std::vector<float> text_emb;
        if (!supertonic_text_encoder_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                                  style_ttl.data(), text_emb, &error)) {
            fprintf(stderr, "text encoder failed: %s\n", error.c_str());
            free_supertonic_model(model); return 1;
        }
        auto t3 = clk::now();

        std::vector<float> latent_mask((size_t) latent_len, 1.0f);
        std::vector<float> next;
        for (int s = 0; s < steps; ++s) {
            if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                             text_emb.data(), (int) text_ids.size(),
                                             style_ttl.data(), latent_mask.data(),
                                             s, steps, next, &error)) {
                fprintf(stderr, "vector step %d failed: %s\n", s, error.c_str());
                free_supertonic_model(model); return 1;
            }
            latent.swap(next);
        }
        auto t4 = clk::now();

        std::vector<float> wav;
        if (!supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, wav, &error)) {
            fprintf(stderr, "vocoder failed: %s\n", error.c_str());
            free_supertonic_model(model); return 1;
        }
        auto t5 = clk::now();

        double audio_s = (double) wav.size() / (double) sample_rate;
        double tot_ms = ms_t(t5 - t0).count();
        if (record) {
            st_pre.ms.push_back(ms_t(t1 - t0).count());
            st_dur.ms.push_back(ms_t(t2 - t1).count());
            st_te .ms.push_back(ms_t(t3 - t2).count());
            st_ve .ms.push_back(ms_t(t4 - t3).count());
            st_voc.ms.push_back(ms_t(t5 - t4).count());
            st_tot.ms.push_back(tot_ms);
            rtfs.push_back((tot_ms / 1000.0) / audio_s);
            last_audio_s = audio_s;
        }
        fprintf(stderr, "[run %d/%d] %s total=%.1fms audio=%.2fs RTF=%.3f%s\n",
                r + 1, total_runs, record ? "" : "(warmup) ",
                tot_ms, audio_s, (tot_ms / 1000.0) / audio_s,
                record ? "" : " [discarded]");
    }

    printf("\nSupertonic 2 C++ benchmark\n");
    printf("  text length: %zu chars\n", text.size());
    printf("  voice: %s, language: %s, steps: %d, speed: %.2f\n",
           voice.c_str(), language.c_str(), steps, speed);
    printf("  threads: %d\n", model.n_threads);
    printf("  audio per run: %.3fs @ %d Hz\n", last_audio_s, model.hparams.sample_rate);
    printf("  runs: %d (warmup discarded: %d)\n", runs, warmup);
    printf("\n");
    print_stage(st_pre);
    print_stage(st_dur);
    print_stage(st_te);
    print_stage(st_ve);
    print_stage(st_voc);
    print_stage(st_tot);
    if (!rtfs.empty()) {
        printf("\n  RTF (total / audio):    min=%.3f  med=%.3f  mean=%.3f  p95=%.3f  max=%.3f\n",
               minv(rtfs), median(rtfs), mean(rtfs), percentile(rtfs, 0.95), maxv(rtfs));
        printf("  Real-time multiplier:   med=%.2fx (1 second of audio per %.2f ms)\n",
               1.0 / median(rtfs), median(st_tot.ms) / last_audio_s);
    }
    if (!json_out.empty()) {
        std::ofstream os(json_out);
        if (!os) {
            fprintf(stderr, "failed to open json output: %s\n", json_out.c_str());
            free_supertonic_model(model);
            return 1;
        }
        os << "{\n";
        os << "  \"runtime\": \"ggml-cpp\",\n";
        os << "  \"model\": \"" << json_escape(model_path) << "\",\n";
        os << "  \"text_length\": " << text.size() << ",\n";
        os << "  \"voice\": \"" << json_escape(voice) << "\",\n";
        os << "  \"language\": \"" << json_escape(language) << "\",\n";
        os << "  \"steps\": " << steps << ",\n";
        os << "  \"speed\": " << speed << ",\n";
        os << "  \"threads\": " << model.n_threads << ",\n";
        os << "  \"audio_s\": " << last_audio_s << ",\n";
        os << "  \"runs\": " << runs << ",\n";
        os << "  \"warmup\": " << warmup << ",\n";
        os << "  \"rtf\": {"
           << "\"min\": " << minv(rtfs)
           << ", \"median\": " << median(rtfs)
           << ", \"mean\": " << mean(rtfs)
           << ", \"p95\": " << percentile(rtfs, 0.95)
           << ", \"max\": " << maxv(rtfs)
           << "},\n";
        os << "  \"stages\": {\n";
        write_json_stage(os, st_pre, true);
        write_json_stage(os, st_dur, true);
        write_json_stage(os, st_te, true);
        write_json_stage(os, st_ve, true);
        write_json_stage(os, st_voc, true);
        write_json_stage(os, st_tot, false);
        os << "  }\n";
        os << "}\n";
    }

    free_supertonic_model(model);
    return 0;
}
