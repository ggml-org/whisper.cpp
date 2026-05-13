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

#ifdef GGML_USE_VULKAN
// QVAC-18605 — needed for `ggml_backend_vk_get_device_description`
// in the bench's backend annotator (Vulkan-only).
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <map>
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
        "          [--runs 5] [--warmup 1] [--threads N] [--n-gpu-layers N]\n"
        "          [--vulkan-device N] (-1 = auto-pick adapter with most free VRAM)\n"
        "          [--f16-attn 0|1] [--f16-weights 0|1]\n"
        "          [--precision f32|f16|q8_0]   (default: f32)\n"
        "          [--kv-attn-type auto|f32|f16|bf16|q8_0]\n"
        "                            (multi-dtype K/V flash-attn dispatch; generalises\n"
        "                            --f16-attn.  default auto: falls back to --f16-attn.\n"
        "                            bf16/q8_0 require Vulkan adapter support; silent\n"
        "                            fallback to f32 on probe miss.)\n"
        "          [--f16-weights-deny PATTERN1,PATTERN2,...] (substring patterns,\n"
        "                            comma-separated; matching tensors stay F32 even\n"
        "                            when --f16-weights is on.  Layered on top of the\n"
        "                            curated allow-list.  Default empty.)\n"
        "          [--prewarm TEXT] (one cold-start synth before timed loop;\n"
        "                            independent of --warmup; CPU is no-op)\n"
        "          [--vulkan-prefer-host-memory]    (sets GGML_VK_PREFER_HOST_MEMORY=1)\n"
        "          [--vulkan-disable-coopmat2]      (sets GGML_VK_DISABLE_COOPMAT2=1)\n"
        "          [--vulkan-disable-bfloat16]      (sets GGML_VK_DISABLE_BFLOAT16=1)\n"
        "          [--vulkan-perf-logger]           (sets GGML_VK_PERF_LOGGER=1)\n"
        "          [--vulkan-async-transfer]        (sets GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1)\n"
        "          [--vulkan-env KEY=VALUE]         (set arbitrary GGML_VK_* env var; may repeat)\n"
        "          [--no-bench-sync]   (skip ggml_backend_synchronize at stage boundaries;\n"
        "                               default off for accurate per-stage attribution on Vulkan)\n"
        "          [--bench-per-step]  (time each denoise step individually so the first-step\n"
        "                               cold-pipeline cost is distinguished from steady-state)\n"
        "          [--json-out FILE]\n",
        argv0);
}

tts_cpp::supertonic::detail::supertonic_precision parse_bench_precision(const std::string & s) {
    using P = tts_cpp::supertonic::detail::supertonic_precision;
    if (s == "f32" || s == "F32") return P::F32;
    if (s == "f16" || s == "F16") return P::F16;
    if (s == "q8_0" || s == "Q8_0" || s == "q8") return P::Q8_0;
    throw std::runtime_error("unknown --precision value: " + s + " (expected f32|f16|q8_0)");
}

const char * precision_to_string(tts_cpp::supertonic::detail::supertonic_precision p) {
    using P = tts_cpp::supertonic::detail::supertonic_precision;
    switch (p) {
        case P::F32:  return "f32";
        case P::F16:  return "f16";
        case P::Q8_0: return "q8_0";
    }
    return "f32";
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
    int n_gpu_layers = 0;
    // -1 = auto (GPU on, CPU off); 0/1 to force.  See model.use_f16_attn.
    int f16_attn = -1;
    // Phase 2A — F16 load-time materialization of the hot matmul /
    // pwconv weights.  -1 auto / 0 / 1 force.
    int f16_weights = -1;
    supertonic_precision precision = supertonic_precision::F32;
    // QVAC-18605 — Vulkan adapter index.  Default 0 (the historical
    // hard-coded value in `init_supertonic_backend`).  Range-checked
    // at GGUF load against `ggml_backend_vk_get_device_count()`; an
    // out-of-range value is a hard error.
    int vulkan_device = 0;
    // QVAC-18605 follow-up — first-synth pre-warm.  When non-empty,
    // a throwaway synth on `prewarm_text` runs after model load + before
    // the timed runs, forcing every per-stage GPU graph cache + shader
    // pipeline to populate up-front.  No-op on CPU backends.  Note that
    // bench's existing `--warmup N` flag is independent: it discards
    // the first N timed runs from the median, but it doesn't avoid the
    // shader-compile hit on the first warmup run.  `--prewarm TEXT`
    // does, so the first warmup run reflects actual steady-state warm
    // time rather than the cold-start outlier.
    std::string prewarm_text;
    // QVAC-18605 round 6 — comma-separated list of substring patterns
    // that force matching tensors to stay F32 even when --f16-weights
    // is on.  Layered on top of the curated allow-list in
    // `should_materialise_f16_weight()`.  Default empty (zero
    // behaviour change for every existing bench invocation).
    std::vector<std::string> f16_weights_deny_list;
    // QVAC-18605 round 4 — multi-dtype K/V flash-attn dispatch.
    // -1 = auto (falls back to --f16-attn for back-compat); 0=f32,
    // 1=f16, 2=bf16, 3=q8_0.  Probe-gated graceful fallback to f32
    // on adapters that don't support the requested dtype.
    int kv_attn_type = -1;
    // QVAC-18605 round 7 — Vulkan env-var overrides applied via
    // `apply_vulkan_env_overrides` BEFORE `init_supertonic_backend`.
    std::map<std::string, std::string> vulkan_env_overrides;
    // QVAC-18605 round 7 — bench observability flags.
    //
    // `bench_sync` (default true) inserts an explicit
    // `ggml_backend_synchronize` at every per-stage boundary so
    // the wall-clock attributes to the right stage on async
    // backends (Vulkan / OpenCL).  Cheap on CPU (no-op).
    // `--no-bench-sync` opts out for the rare case the operator
    // wants to observe pipelined / overlapped behaviour.
    //
    // `bench_per_step` (default false) times each
    // `supertonic_vector_step_ggml` call individually so the
    // first-step (cold pipelines) cost can be distinguished from
    // steady-state.  Adds an extra stage column per step in the
    // human output and a `vector_step_ms` array in the JSON.
    bool bench_sync     = true;
    bool bench_per_step = false;

    auto split_csv = [](const std::string & s) {
        std::vector<std::string> out;
        size_t start = 0;
        for (size_t i = 0; i <= s.size(); ++i) {
            if (i == s.size() || s[i] == ',') {
                out.emplace_back(s.substr(start, i - start));
                start = i + 1;
            }
        }
        return out;
    };

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
        else if (a == "--n-gpu-layers") n_gpu_layers = std::stoi(next("--n-gpu-layers"));
        else if (a == "--vulkan-device") vulkan_device = std::stoi(next("--vulkan-device"));
        else if (a == "--prewarm") prewarm_text = next("--prewarm");
        else if (a == "--f16-attn") f16_attn = std::stoi(next("--f16-attn"));
        else if (a == "--f16-weights") f16_weights = std::stoi(next("--f16-weights"));
        else if (a == "--precision") precision = parse_bench_precision(next("--precision"));
        else if (a == "--f16-weights-deny") f16_weights_deny_list = split_csv(next("--f16-weights-deny"));
        else if (a == "--kv-attn-type") {
            const std::string v = next("--kv-attn-type");
            if      (v == "auto") kv_attn_type = -1;
            else if (v == "f32")  kv_attn_type = 0;
            else if (v == "f16")  kv_attn_type = 1;
            else if (v == "bf16") kv_attn_type = 2;
            else if (v == "q8_0") kv_attn_type = 3;
            else { fprintf(stderr,
                "--kv-attn-type expects auto|f32|f16|bf16|q8_0 (got: %s)\n", v.c_str());
                return 2; }
        }
        else if (a == "--vulkan-prefer-host-memory") vulkan_env_overrides["GGML_VK_PREFER_HOST_MEMORY"]      = "1";
        else if (a == "--vulkan-disable-coopmat2")   vulkan_env_overrides["GGML_VK_DISABLE_COOPMAT2"]        = "1";
        else if (a == "--vulkan-disable-bfloat16")   vulkan_env_overrides["GGML_VK_DISABLE_BFLOAT16"]        = "1";
        else if (a == "--vulkan-perf-logger")        vulkan_env_overrides["GGML_VK_PERF_LOGGER"]             = "1";
        else if (a == "--vulkan-async-transfer")     vulkan_env_overrides["GGML_VK_ASYNC_USE_TRANSFER_QUEUE"]= "1";
        else if (a == "--vulkan-env") {
            const std::string raw = next("--vulkan-env");
            const auto eq = raw.find('=');
            if (eq == std::string::npos || eq == 0) {
                fprintf(stderr, "--vulkan-env expects KEY=VALUE (got: %s)\n", raw.c_str());
                return 2;
            }
            vulkan_env_overrides[raw.substr(0, eq)] = raw.substr(eq + 1);
        }
        else if (a == "--no-bench-sync") bench_sync = false;
        else if (a == "--bench-sync")    bench_sync = true;  // explicit on; default
        else if (a == "--bench-per-step") bench_per_step = true;
        else if (a == "--json-out") json_out = next("--json-out");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || text.empty()) { usage(argv[0]); return 2; }

    // QVAC-18605 round 7 — apply Vulkan env-var overrides BEFORE
    // `load_supertonic_gguf` (which calls `init_supertonic_backend`,
    // which is when ggml-vulkan reads its GGML_VK_* env vars).
    // Throws on any non-`GGML_VK_` key (operator-config typo
    // guard); we let the throw propagate to surface as an
    // uncaught-exception backtrace, since bench is for operators
    // who can read it (matches the legacy behaviour for `--vulkan-device
    // abc` and similar).
    apply_vulkan_env_overrides(vulkan_env_overrides);

    supertonic_model model;
    if (!load_supertonic_gguf(model_path, model, n_gpu_layers,
                              /*verbose=*/false, f16_weights, precision,
                              vulkan_device, f16_weights_deny_list)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    supertonic_set_n_threads(model, n_threads);
    // F16 K/V flash-attention dispatch: same auto policy as Engine
    // (auto ⇒ on for GPU backends that pass the F16-K/V probe, off
    // for CPU; user can force).  See `supertonic_backend_supports_f16_kv_flash_attn`
    // in supertonic_gguf.cpp for the rationale (QVAC-18605).
    if (f16_attn < 0) {
        model.use_f16_attn = !model.backend_is_cpu &&
                             supertonic_backend_supports_f16_kv_flash_attn(model.backend);
    } else {
        model.use_f16_attn = f16_attn != 0;
    }
    // QVAC-18605 round 4 — multi-dtype K/V dispatch resolution.
    // Same plumbing as Engine::Impl ctor; out-of-range throws
    // (caller surface).  Probes are advisory + cached.
    model.kv_attn_type = resolve_kv_attn_type(
        kv_attn_type,
        model.use_f16_attn,
        supertonic_backend_supports_f16_kv_flash_attn(model.backend),
        supertonic_backend_supports_bf16_kv_flash_attn(model.backend),
        supertonic_backend_supports_q8_0_kv_flash_attn(model.backend));
    model.use_f16_attn = (model.kv_attn_type == kv_attn_dtype::f16);

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
    char st_ve_label[64];
    std::snprintf(st_ve_label, sizeof(st_ve_label), "vector_estimator (%d step)", steps);
    Stage st_ve {st_ve_label, {}};
    Stage st_voc{"vocoder", {}};
    Stage st_tot{"total", {}};
    // QVAC-18605 round 7 — per-denoise-step breakdown.  Populated
    // only when `--bench-per-step` is on; otherwise stays empty
    // and is omitted from human + JSON output.  One Stage per
    // step index (step 0 typically reflects cold-pipeline cost
    // on Vulkan/OpenCL; steps 1+ reflect steady-state).
    std::vector<Stage> st_ve_per_step;
    if (bench_per_step) {
        st_ve_per_step.reserve((size_t) steps);
        for (int s = 0; s < steps; ++s) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "  vector_step[%d]", s);
            st_ve_per_step.push_back(Stage{lbl, {}});
        }
    }
    std::vector<double> rtfs;
    double last_audio_s = 0;

    // QVAC-18605 round 7 — explicit backend sync at stage
    // boundaries.  Cheap on CPU (returns immediately when no GPU
    // work pending); on Vulkan / OpenCL ensures the next
    // `clk::now()` reflects work-completed-by-the-prior-stage.
    // No-op when `bench_sync` is false (operator opt-out).
    auto bench_sync_now = [&]() {
        if (bench_sync) ggml_backend_synchronize(model.backend);
    };

    // QVAC-18605 follow-up — first-synth pre-warm.
    //
    // Independent of the existing `--warmup N` flag.  `--warmup`
    // discards the first N timed runs from the median; `--prewarm
    // TEXT` runs ONE additional throwaway synth here, BEFORE the
    // timed loop even starts, so the first warmup run reflects the
    // post-shader-compile steady-state cost rather than the cold-
    // start outlier.  No-op on CPU (no shader-compile cost to amortise)
    // and on empty `--prewarm` (the operator didn't ask).
    double prewarm_ms = 0.0;
    if (!prewarm_text.empty() && !model.backend_is_cpu) {
        auto pw_t0 = clk::now();
        std::string pw_error;
        std::vector<int32_t> pw_ids_i32;
        std::string pw_norm;
        if (supertonic_text_to_ids(model, prewarm_text, language, pw_ids_i32, &pw_norm, &pw_error)) {
            std::vector<int64_t> pw_ids(pw_ids_i32.begin(), pw_ids_i32.end());
            float pw_dur = 0;
            std::vector<float> pw_text_emb;
            if (supertonic_duration_forward_ggml(model, pw_ids.data(), (int) pw_ids.size(),
                                                 style_dp.data(), pw_dur, &pw_error) &&
                supertonic_text_encoder_forward_ggml(model, pw_ids.data(), (int) pw_ids.size(),
                                                     style_ttl.data(), pw_text_emb, &pw_error)) {
                const int chunk = model.hparams.base_chunk_size * model.hparams.ttl_chunk_compress_factor;
                int pw_latent_len = std::max(1, (int) (pw_dur / speed * model.hparams.sample_rate + chunk - 1) / chunk);
                std::vector<float> pw_latent((size_t) model.hparams.latent_channels * pw_latent_len, 0.0f);
                std::vector<float> pw_mask((size_t) pw_latent_len, 1.0f);
                std::vector<float> pw_next;
                bool pw_ok = true;
                for (int s = 0; s < steps && pw_ok; ++s) {
                    pw_ok = supertonic_vector_step_ggml(model, pw_latent.data(), pw_latent_len,
                                                        pw_text_emb.data(), (int) pw_ids.size(),
                                                        style_ttl.data(), pw_mask.data(),
                                                        s, steps, pw_next, &pw_error);
                    pw_latent.swap(pw_next);
                }
                std::vector<float> pw_wav;
                if (pw_ok) {
                    supertonic_vocoder_forward_ggml(model, pw_latent.data(), pw_latent_len,
                                                    pw_wav, &pw_error);
                }
            }
        }
        prewarm_ms = ms_t(clk::now() - pw_t0).count();
        fprintf(stderr, "[prewarm] cold-start synth on '%s' took %.1fms\n",
                prewarm_text.c_str(), prewarm_ms);
    }

    int total_runs = runs + warmup;
    for (int r = 0; r < total_runs; ++r) {
        bool record = r >= warmup;
        std::string error;

        bench_sync_now();
        auto t0 = clk::now();

        std::vector<int32_t> text_ids_i32;
        std::string normalized;
        if (!supertonic_text_to_ids(model, text, language, text_ids_i32, &normalized, &error)) {
            fprintf(stderr, "preprocess failed: %s\n", error.c_str());
            free_supertonic_model(model); return 1;
        }
        std::vector<int64_t> text_ids(text_ids_i32.begin(), text_ids_i32.end());
        bench_sync_now();
        auto t1 = clk::now();

        float duration_raw = 0;
        if (!supertonic_duration_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                              style_dp.data(), duration_raw, &error)) {
            fprintf(stderr, "duration failed: %s\n", error.c_str());
            free_supertonic_model(model); return 1;
        }
        bench_sync_now();
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
        bench_sync_now();
        auto t3 = clk::now();

        std::vector<float> latent_mask((size_t) latent_len, 1.0f);
        std::vector<float> next;
        // QVAC-18605 round 7 — per-step timing.  When
        // `bench_per_step` is on, a sync + clock sample bracket
        // each `supertonic_vector_step_ggml` call.  When off, a
        // single sync at end-of-loop matches the legacy timing
        // semantics exactly (zero overhead added).
        for (int s = 0; s < steps; ++s) {
            auto step_t0 = bench_per_step ? clk::now() : clk::time_point{};
            if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                             text_emb.data(), (int) text_ids.size(),
                                             style_ttl.data(), latent_mask.data(),
                                             s, steps, next, &error)) {
                fprintf(stderr, "vector step %d failed: %s\n", s, error.c_str());
                free_supertonic_model(model); return 1;
            }
            latent.swap(next);
            if (bench_per_step) {
                bench_sync_now();
                auto step_t1 = clk::now();
                if (record) {
                    st_ve_per_step[(size_t) s].ms.push_back(ms_t(step_t1 - step_t0).count());
                }
            }
        }
        bench_sync_now();
        auto t4 = clk::now();

        std::vector<float> wav;
        if (!supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, wav, &error)) {
            fprintf(stderr, "vocoder failed: %s\n", error.c_str());
            free_supertonic_model(model); return 1;
        }
        bench_sync_now();
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
    printf("  threads: %d, n_gpu_layers: %d, precision: %s\n",
           model.n_threads, n_gpu_layers, precision_to_string(precision));
    {
        // QVAC-18605 — bench backend description.  On Vulkan the
        // adapter description is appended so multi-GPU machines
        // unambiguously identify which device ran the bench.
        std::string desc = ggml_backend_name(model.backend) ? ggml_backend_name(model.backend) : "(unknown)";
#ifdef GGML_USE_VULKAN
        if (model.backend_is_vk) {
            char vk_desc[256] = {0};
            ggml_backend_vk_get_device_description(vulkan_device < 0 ? 0 : vulkan_device,
                                                   vk_desc, sizeof(vk_desc) - 1);
            if (vk_desc[0]) {
                desc += " (device " + std::to_string(vulkan_device < 0 ? 0 : vulkan_device) +
                        ": " + vk_desc + ")";
            }
        }
#endif
        // QVAC-18605 follow-up — surface every backend-capability
        // dispatch flag plus the cold-start prewarm latency so log
        // grep'ing across multiple machines can attribute perf
        // differences to the right cause (e.g. "use_f16_weights=off
        // on this run because the F16 mul_mat probe rejected the
        // shape" is much faster to triage than "why is this synth
        // 30 % slower than the other one").
        // QVAC-18605 round 3 — also surface BF16 K/V availability and
        // the host-pinned-buffer-type availability.  Both are forward-
        // compat capabilities (no live dispatch yet); the bench tag
        // lets operators verify a future `--kv-attn-type bf16` /
        // `--vulkan-pinned-uploads` opt-in will actually take effect
        // on their machine before they flip the flag.
        // QVAC-18605 round 4 — surface the resolved K/V dispatch
        // dtype.  When the operator opts out of `--kv-attn-type`
        // the resolved value falls through to `f16` / `f32` per
        // `--f16-attn`, so the existing `f16_attn=on` tag still
        // matches the historical baseline; new tag fires when
        // bf16 / q8_0 actually take effect.
        const char * kv_dtype_str = "f32";
        switch (model.kv_attn_type) {
            case kv_attn_dtype::f32:        kv_dtype_str = "f32";  break;
            case kv_attn_dtype::f16:        kv_dtype_str = "f16";  break;
            case kv_attn_dtype::bf16:       kv_dtype_str = "bf16"; break;
            case kv_attn_dtype::q8_0:       kv_dtype_str = "q8_0"; break;
            case kv_attn_dtype::autoselect: kv_dtype_str = "auto-leaked!"; break;
        }
        printf("  backend: %s%s%s%s (kv_attn_type=%s)%s%s%s\n",
               desc.c_str(),
               model.use_f16_attn        ? " (f16_attn=on)"        : "",
               model.use_f16_weights     ? " (f16_weights=on)"     : "",
               model.use_native_leaky_relu ? " (native_leaky_relu=on)" : "",
               kv_dtype_str,
               supertonic_backend_supports_q8_0_kv_flash_attn(model.backend) ? " (q8_0_kv_attn=available)" : "",
               supertonic_backend_supports_bf16_kv_flash_attn(model.backend) ? " (bf16_kv_attn=available)" : "",
               supertonic_backend_supports_pinned_host_buffer(model.backend) ? " (pinned_host_buffer=available)" : "");
        // QVAC-18605 round 6 — confirm the F16-weights deny-list took
        // effect.  Silent when the operator didn't supply one (no
        // visual noise on the default path).
        if (!f16_weights_deny_list.empty()) {
            printf("  f16_weights_deny_list: %zu pattern%s; %d tensor%s excluded\n",
                   f16_weights_deny_list.size(),
                   f16_weights_deny_list.size() == 1 ? "" : "s",
                   model.f16_weights_excluded_count,
                   model.f16_weights_excluded_count == 1 ? "" : "s");
        }
        if (prewarm_ms > 0.0) {
            printf("  prewarm: %.1fms (cold-start, discarded)\n", prewarm_ms);
        }
    }
    printf("  audio per run: %.3fs @ %d Hz\n", last_audio_s, model.hparams.sample_rate);
    printf("  runs: %d (warmup discarded: %d)\n", runs, warmup);
    printf("\n");
    print_stage(st_pre);
    print_stage(st_dur);
    print_stage(st_te);
    print_stage(st_ve);
    // QVAC-18605 round 7 — per-step breakdown lines.  Indented
    // under the aggregate vector-estimator line for visual
    // grouping.  Only emitted when --bench-per-step is on.
    for (auto & st : st_ve_per_step) {
        if (!st.ms.empty()) print_stage(st);
    }
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
        os << "  \"n_gpu_layers\": " << n_gpu_layers << ",\n";
        os << "  \"precision\": \"" << precision_to_string(precision) << "\",\n";
        os << "  \"audio_s\": " << last_audio_s << ",\n";
        os << "  \"runs\": " << runs << ",\n";
        os << "  \"warmup\": " << warmup << ",\n";
        os << "  \"prewarm_ms\": " << prewarm_ms << ",\n";
        os << "  \"f16_attn\": " << (model.use_f16_attn ? "true" : "false") << ",\n";
        os << "  \"f16_weights\": " << (model.use_f16_weights ? "true" : "false") << ",\n";
        // QVAC-18605 round 4 — surface the resolved K/V dispatch
        // dtype.  Always emitted (string label), so JSON consumers
        // can attribute drift / perf differences to the right cause
        // even on the default `auto` path.
        {
            const char * kv = "f32";
            switch (model.kv_attn_type) {
                case kv_attn_dtype::f32:  kv = "f32";  break;
                case kv_attn_dtype::f16:  kv = "f16";  break;
                case kv_attn_dtype::bf16: kv = "bf16"; break;
                case kv_attn_dtype::q8_0: kv = "q8_0"; break;
                case kv_attn_dtype::autoselect: kv = "auto-leaked"; break;
            }
            os << "  \"kv_attn_type\": \"" << kv << "\",\n";
            os << "  \"kv_attn_type_requested\": " << kv_attn_type << ",\n";
        }
        // QVAC-18605 round 6 — surface the user-supplied deny-list +
        // the count of tensors it excluded.  Always emitted (even on
        // the default empty path) so JSON consumers can attribute
        // any quality regression observed in CI to a config change.
        os << "  \"f16_weights_deny_list\": [";
        for (size_t k = 0; k < f16_weights_deny_list.size(); ++k) {
            if (k) os << ", ";
            os << "\"" << json_escape(f16_weights_deny_list[k]) << "\"";
        }
        os << "],\n";
        os << "  \"f16_weights_excluded_count\": " << model.f16_weights_excluded_count << ",\n";
        os << "  \"native_leaky_relu\": " << (model.use_native_leaky_relu ? "true" : "false") << ",\n";
        os << "  \"q8_0_kv_attn_available\": "
           << (supertonic_backend_supports_q8_0_kv_flash_attn(model.backend) ? "true" : "false") << ",\n";
        // QVAC-18605 round 3 — extra capability flags surfaced for the
        // forward-compat probes (BF16 K/V flash-attn + pinned-host-
        // buffer-type).  Operators / CI scripts grep on these to
        // pre-flight whether a future `--kv-attn-type bf16` /
        // `--vulkan-pinned-uploads` opt-in will be effective on the
        // resolved backend.
        os << "  \"bf16_kv_attn_available\": "
           << (supertonic_backend_supports_bf16_kv_flash_attn(model.backend) ? "true" : "false") << ",\n";
        os << "  \"pinned_host_buffer_available\": "
           << (supertonic_backend_supports_pinned_host_buffer(model.backend) ? "true" : "false") << ",\n";
        // QVAC-18605 round 7 — bench observability surface.
        // `bench_sync` documents whether the per-stage times
        // include a `ggml_backend_synchronize` boundary; useful
        // when comparing JSON across machines / configs.
        os << "  \"bench_sync\": " << (bench_sync ? "true" : "false") << ",\n";
        // QVAC-18605 round 7 — Vulkan env-var overrides surfaced
        // verbatim so the JSON consumer can attribute drift to
        // a specific override (or its absence).  Always emitted
        // (object — empty on the default-config path).
        os << "  \"vulkan_env_overrides\": {";
        {
            bool first = true;
            for (const auto & kv : vulkan_env_overrides) {
                if (!first) os << ", ";
                first = false;
                os << "\"" << json_escape(kv.first) << "\": \""
                   << json_escape(kv.second) << "\"";
            }
        }
        os << "},\n";
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
        // QVAC-18605 round 7 — when --bench-per-step is on, emit
        // each step as its own stage entry.  When off, the
        // aggregate `vector_estimator` stage is the only entry
        // for the vector-estimator buckets (legacy JSON shape).
        if (!st_ve_per_step.empty()) {
            write_json_stage(os, st_ve, true);
            for (auto & st : st_ve_per_step) {
                if (!st.ms.empty()) write_json_stage(os, st, true);
            }
        } else {
            write_json_stage(os, st_ve, true);
        }
        write_json_stage(os, st_voc, true);
        write_json_stage(os, st_tot, false);
        os << "  }\n";
        os << "}\n";
    }

    free_supertonic_model(model);
    return 0;
}
