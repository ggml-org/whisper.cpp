// Benchmark for the Parler-TTS C++ GGML engine.
//
// Times each stage of the synthesis pipeline (mirrors engine.cpp run()):
//   1. T5 description encode (+ cross-K/V precompute)
//   2. decoder prefill (prompt + BOS start frame)
//   3. decoder loop (autoregressive delay-pattern generation)
//   4. DAC codec decode -> PCM
//
// Reports min / median / mean / p95 / max across --runs iterations (after
// --warmup dropped runs), plus RTF (total / audio). Greedy is deterministic
// per configuration but can emit EOS, so backends may do different work.
// For bounded throughput runs, --min-new-tokens N with --max-frames N
// suppresses EOS until the length bound; still compare reported step/audio
// lengths. Unlike the Engine, this keeps argmax: it measures throughput, and
// the audio it makes is degenerate, not a speech-quality measurement.
//
// Usage:
//   ./parler-bench --model parler-indic-q8_0.gguf --text "..." \
//       [--description DESC] [--n-gpu-layers N] [--backend NAME] [--threads N] \
//       [--max-frames N] [--min-new-tokens N] [--seed 42] [--runs 5] [--warmup 1] [--wav-out audio.wav] [--json-out f.json]

#include "internal.h"
#include "decoder_runtime.h"
#include "tokenizer.h"
#include "bpe_tokenizer.h"
#include "text_norm.h"
#include "delay.h"
#include "sampler.h"
#include "backend_selection.h"
#include "bench_wav.h"
#include "backend_util.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <set>
#include <thread>
#include <vector>

using namespace tts_cpp::parler;
using namespace tts_cpp::parler::detail;
using clk = std::chrono::steady_clock;
using ms_t = std::chrono::duration<double, std::milli>;

namespace {

struct Stage {
    std::string name;
    std::vector<double> ms;
};

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model parler.gguf --text TEXT\n"
        "          [--description DESC] (default: a neutral studio caption)\n"
        "          [--n-gpu-layers N] (offload to GPU: Metal/Vulkan/...; 0 = CPU)\n"
        "          [--backend auto|cpu|opencl|hexagon|DEVICE] (explicit device, no fallback)\n"
        "          [--threads N] [--max-frames N] (decoder steps; ~86/s audio)\n"
        "          [--min-new-tokens N] (-1 = model default; >=0 overrides EOS suppression)\n"
        "          For bounded throughput, use --min-new-tokens N with --max-frames N; not a quality test.\n"
        "          [--coreml-decoder] (experimental Core ML decoder steps)\n"
        "          [--profile-coreml-decoder] (requires --coreml-decoder; successful measured steps only)\n"
        "          [--seed 42] [--sampled] (default greedy/deterministic)\n"
        "          [--runs 5] [--warmup 1] [--wav-out FILE] [--json-out FILE]\n",
        argv0);
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double) (v.size() - 1);
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
    printf("  %-16s n=%zu  min=%8.2f  med=%8.2f  mean=%8.2f  p95=%8.2f  max=%8.2f  ms\n",
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
    std::string model_path, text, json_out, wav_out, backend;
    std::string description =
        "A female speaker with a calm, clear voice, close up, studio quality "
        "with no background noise.";
    int n_gpu_layers = 0;
    int n_threads = 0;   // 0 => all hardware cores
    int max_frames = 0;  // 0 => GGUF default (~2580)
    int min_new_tokens = -1;  // -1 => GGUF default
    int seed = 42;
    int runs = 5;
    int warmup = 1;
    bool greedy = true;
    bool coreml_decoder = false;
    bool profile_coreml_decoder = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * f) {
            if (i + 1 >= argc) throw std::runtime_error(std::string(f) + " requires a value");
            return std::string(argv[++i]);
        };
        if      (a == "--model")        model_path  = next("--model");
        else if (a == "--text")         text        = next("--text");
        else if (a == "--description")  description = next("--description");
        else if (a == "--n-gpu-layers") n_gpu_layers = std::stoi(next("--n-gpu-layers"));
        else if (a == "--backend")      backend     = next("--backend");
        else if (a == "--threads")      n_threads   = std::stoi(next("--threads"));
        else if (a == "--max-frames")   max_frames  = std::stoi(next("--max-frames"));
        else if (a == "--min-new-tokens") min_new_tokens = std::stoi(next("--min-new-tokens"));
        else if (a == "--seed")         seed        = std::stoi(next("--seed"));
        else if (a == "--runs")         runs        = std::stoi(next("--runs"));
        else if (a == "--warmup")       warmup      = std::stoi(next("--warmup"));
        else if (a == "--coreml-decoder") coreml_decoder = true;
        else if (a == "--profile-coreml-decoder") profile_coreml_decoder = true;
        else if (a == "--sampled")      greedy      = false;
        else if (a == "--wav-out")      wav_out     = next("--wav-out");
        else if (a == "--json-out")     json_out    = next("--json-out");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (profile_coreml_decoder && !coreml_decoder) {
        fprintf(stderr, "--profile-coreml-decoder requires --coreml-decoder\n");
        return 2;
    }
    if (min_new_tokens < -1) {
        fprintf(stderr, "--min-new-tokens must be -1 (model default) or nonnegative\n");
        return 2;
    }
    if (runs <= 0 || warmup < 0 || warmup > std::numeric_limits<int>::max() - runs) {
        fprintf(stderr, "--runs must be positive and --warmup nonnegative (without overflow)\n");
        return 2;
    }
    if (model_path.empty() || text.empty()) { usage(argv[0]); return 2; }

    const int n_thr = n_threads > 0 ? n_threads
                                    : (int) std::max(1u, std::thread::hardware_concurrency());

    parler_model model;
    std::string err;
    if (!parler_load_gguf(model_path, model, n_gpu_layers, &err, backend)) {
        fprintf(stderr, "parler-bench: load failed: %s\n", err.c_str());
        return 1;
    }
    const bool is_cpu = ::tts_cpp::detail::backend_is_cpu(model.backend);
    // the Core ML DAC sidecar next to the GGUF, as the Engine attaches it
    // (PARLER_COREML_DISABLE=1 benchmarks the ggml DAC instead)
    try {
        parler_attach_dac_sidecar(model, model_path);
    } catch (const std::exception & e) {
        fprintf(stderr, "parler-bench: %s\n", e.what());
        parler_free_model(model);
        return 1;
    }
    std::string dac_backend = PARLER_DAC_GGML_BACKEND;
    parler_decoder_runtime decoder(model);
    if (coreml_decoder && !parler_coreml_disabled()) {
        auto sidecar = parler_open_decoder_sidecar(model, model_path, err);
        if (!sidecar) {
            fprintf(stderr, "parler-bench: %s\n", err.c_str());
            if (parler_coreml_strict()) { parler_free_model(model); return 1; }
        } else {
            decoder.attach(std::move(sidecar), parler_coreml_strict());
        }
    }

    parler_tokenizer tokenizer;
    if (!tokenizer.load(model.tok_pieces, model.tok_scores, model.tok_charsmap,
                        model.tok_unk_id, model.tok_eos_id, model.tok_add_eos)) {
        fprintf(stderr, "parler-bench: tokenizer load failed\n");
        parler_free_model(model);
        return 1;
    }
    parler_bpe_tokenizer prompt_tokenizer;
    if (model.has_prompt_tok &&
        !prompt_tokenizer.load(model.ptok_pieces, model.ptok_merges,
                               model.ptok_unk_id, model.ptok_bos_id, model.ptok_add_bos)) {
        fprintf(stderr, "parler-bench: prompt tokenizer load failed\n");
        parler_free_model(model);
        return 1;
    }

    const parler_hparams & hp = model.hparams;
    const int effective_max_length = max_frames > 0 ? std::min(max_frames, hp.gen_max_length) : hp.gen_max_length;
    const int effective_min_new_tokens = min_new_tokens >= 0 ? min_new_tokens : hp.gen_min_new_tokens;
    const std::string spoken = model.has_prompt_tok ? normalize_numbers_indic(text)
                                                     : normalize_numbers_en(text);
    const std::vector<int32_t> prompt_ids = model.has_prompt_tok
        ? prompt_tokenizer.encode(spoken)
        : tokenizer.encode(spoken);
    const std::vector<int32_t> desc_ids = tokenizer.encode(description);
    if (prompt_ids.empty() || desc_ids.empty()) {
        fprintf(stderr, "parler-bench: prompt/description tokenized to zero tokens\n");
        parler_free_model(model);
        return 1;
    }

    ggml_gallocr_t allocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    parler_sampling_params sp;
    sp.greedy      = greedy || !hp.gen_do_sample;
    sp.temperature = hp.gen_temperature;
    sp.top_k       = hp.gen_top_k;
    sp.top_p       = 1.0f;

    auto sync = [&]() { if (!is_cpu) ggml_backend_synchronize(model.backend); };

    Stage st_t5   {"t5_encode",   {}};
    Stage st_pre  {"prefill",     {}};
    Stage st_dec  {"decode_loop", {}};
    Stage st_dac  {"dac_decode",  {}};
    Stage st_tot  {"total",       {}};
    std::vector<double> rtfs;
    std::set<std::string> measured_decoder_backends, measured_dac_backends;
    parler_decoder_profile_summary decoder_profile;
    double last_audio_s = 0.0;
    int    last_steps = 0;

    // Retain only the final measured synthesis; WAV conversion and I/O are untimed.
    std::vector<float> last_pcm;
    const int total_runs = runs + warmup;
    for (int r = 0; r < total_runs; ++r) {
        const bool record = r >= warmup;
        parler_decoder_profile_summary run_decoder_profile;

        delay_config dcfg;
        dcfg.n_codebooks = hp.n_codebooks;
        dcfg.bos_id = hp.bos_id;
        dcfg.eos_id = hp.eos_id;
        dcfg.pad_id = hp.pad_id;
        dcfg.max_length = effective_max_length;
        dcfg.min_new_tokens = effective_min_new_tokens;
        delay_state st(dcfg);
        std::mt19937 rng((uint32_t) seed);

        sync(); auto t0 = clk::now();
        if (!parler_encode_description(model, desc_ids, n_thr, nullptr)) {
            fprintf(stderr, "parler-bench: encode failed\n"); parler_free_model(model); return 1;
        }
        sync(); auto t1 = clk::now();

        parler_step_logits logits;
        parler_sampler_scratch scratch;
        std::vector<int32_t> frame;
        int n_past = 0;
        if (!decoder.prefill(prompt_ids, st.input_frame(), allocr, n_thr, logits, n_past, dcfg.max_length)) {
            fprintf(stderr, "parler-bench: prefill failed\n"); parler_free_model(model); return 1;
        }
        sync(); auto t2 = clk::now();

        int steps = 0;
        while (true) {
            st.process_logits(logits.view, hp.dec_vocab);
            parler_sample_frame(logits.view, hp.n_codebooks, hp.dec_vocab, sp, rng, scratch,
                                frame);
            st.append(frame);
            if (st.finished()) break;
            parler_decoder_step_profile step_profile;
            if (!decoder.step(st.input_frame(), n_past, allocr, n_thr, logits,
                              profile_coreml_decoder && record ? &step_profile : nullptr)) {
                fprintf(stderr, "parler-bench: dec step failed\n"); parler_free_model(model); return 1;
            }
            if (profile_coreml_decoder) run_decoder_profile.add(step_profile, record);
            n_past++;
            steps++;
        }
        sync(); auto t3 = clk::now();

        int n_frames = 0;
        const std::vector<int32_t> codes = st.undelay(hp.dac_codebook_size, &n_frames);
        std::vector<float> pcm;
        if (n_frames <= 0 || !parler_dac_decode(model, codes.data(), n_frames, n_thr, pcm,
                                                nullptr, 0, -1, nullptr, &dac_backend)) {
            fprintf(stderr, "parler-bench: dac decode failed\n"); parler_free_model(model); return 1;
        }
        sync(); auto t4 = clk::now();

        const double audio_s = pcm.empty() ? 0.0 : (double) pcm.size() / (double) hp.dac_sample_rate;
        const double tot_ms = ms_t(t4 - t0).count();
        if (record && audio_s > 0.0) {
            if (profile_coreml_decoder) decoder_profile.merge(std::move(run_decoder_profile));
            measured_decoder_backends.insert(decoder.backend());
            measured_dac_backends.insert(dac_backend);
            st_t5 .ms.push_back(ms_t(t1 - t0).count());
            st_pre.ms.push_back(ms_t(t2 - t1).count());
            st_dec.ms.push_back(ms_t(t3 - t2).count());
            st_dac.ms.push_back(ms_t(t4 - t3).count());
            st_tot.ms.push_back(tot_ms);
            rtfs.push_back((tot_ms / 1000.0) / audio_s);
            last_audio_s = audio_s;
            last_steps = steps;
        }
        if (!wav_out.empty() && record && r + 1 == total_runs) last_pcm.swap(pcm);
        fprintf(stderr, "[run %d/%d]%s total=%.1fms audio=%.2fs steps=%d RTF=%.3f\n",
                r + 1, total_runs, record ? "" : " (warmup)",
                tot_ms, audio_s, steps, audio_s > 0 ? (tot_ms / 1000.0) / audio_s : 0.0);
    }

    if (!wav_out.empty()) {
        std::string wav_error;
        if (!bench_write_wav(wav_out, last_pcm, hp.dac_sample_rate, wav_error)) {
            fprintf(stderr, "%s\n", wav_error.c_str());
            ggml_gallocr_free(allocr); parler_free_model(model);
            return 1;
        }
    }

    const std::string decoder_label = measured_decoder_backends.size() == 1
        ? *measured_decoder_backends.begin() : measured_decoder_backends.empty() ? decoder.backend() : "mixed";
    const std::string dac_label = measured_dac_backends.size() == 1
        ? *measured_dac_backends.begin() : measured_dac_backends.empty() ? dac_backend : "mixed";
    printf("\nParler-TTS C++ benchmark\n");
    printf("  model: %s\n", model_path.c_str());
    printf("  backend: %s%s\n", ggml_backend_name(model.backend),
           is_cpu ? " (CPU)" : " (GPU)");
    printf("  dac: %s\n", dac_label.c_str());
    printf("  decoder: %s (ggml prefill)\n", decoder_label.c_str());
    printf("  threads: %d, n_gpu_layers: %d, greedy: %s\n",
           n_thr, n_gpu_layers, greedy ? "yes" : "no");
    printf("  effective generation: min_new_tokens=%d, max_length=%d\n",
           effective_min_new_tokens, effective_max_length);
    printf("  prompt tokens: %zu, description tokens: %zu\n",
           prompt_ids.size(), desc_ids.size());
    printf("  audio per run: %.3fs @ %d Hz (%d decode steps)\n",
           last_audio_s, hp.dac_sample_rate, last_steps);
    printf("  runs: %d (warmup discarded: %d)\n\n", runs, warmup);
    print_stage(st_t5);
    print_stage(st_pre);
    print_stage(st_dec);
    print_stage(st_dac);
    print_stage(st_tot);
    if (profile_coreml_decoder) decoder_profile.write_text(std::cout);
    if (!rtfs.empty()) {
        printf("\n  RTF (total / audio):  min=%.3f  med=%.3f  mean=%.3f  p95=%.3f  max=%.3f\n",
               minv(rtfs), median(rtfs), mean(rtfs), percentile(rtfs, 0.95), maxv(rtfs));
        printf("  Real-time multiplier: med=%.2fx\n", 1.0 / median(rtfs));
    }

    if (!json_out.empty()) {
        std::ofstream os(json_out);
        if (os) {
            os << "{\n";
            os << "  \"runtime\": \"ggml-cpp\",\n";
            os << "  \"model\": \"" << json_escape(model_path) << "\",\n";
            os << "  \"backend\": \"" << json_escape(ggml_backend_name(model.backend)) << "\",\n";
            os << "  \"is_cpu\": " << (is_cpu ? "true" : "false") << ",\n";
            os << "  \"dac_backend\": \"" << json_escape(dac_label) << "\",\n";
            os << "  \"decoder_backend\": \"" << json_escape(decoder_label) << "\",\n";
            os << "  \"threads\": " << n_thr << ",\n";
            os << "  \"n_gpu_layers\": " << n_gpu_layers << ",\n";
            os << "  \"greedy\": " << (greedy ? "true" : "false") << ",\n";
            os << "  \"min_new_tokens\": " << effective_min_new_tokens << ",\n";
            os << "  \"max_length\": " << effective_max_length << ",\n";
            os << "  \"audio_s\": " << last_audio_s << ",\n";
            os << "  \"decode_steps\": " << last_steps << ",\n";
            os << "  \"runs\": " << runs << ",\n";
            os << "  \"warmup\": " << warmup << ",\n";
            if (profile_coreml_decoder) {
                os << "  \"coreml_decoder_profile\": ";
                decoder_profile.write_json(os);
                os << ",\n";
            }
            os << "  \"rtf\": {"
               << "\"min\": " << minv(rtfs) << ", \"median\": " << median(rtfs)
               << ", \"mean\": " << mean(rtfs) << ", \"p95\": " << percentile(rtfs, 0.95)
               << ", \"max\": " << maxv(rtfs) << "},\n";
            os << "  \"stages\": {\n";
            write_json_stage(os, st_t5, true);
            write_json_stage(os, st_pre, true);
            write_json_stage(os, st_dec, true);
            write_json_stage(os, st_dac, true);
            write_json_stage(os, st_tot, false);
            os << "  }\n}\n";
        } else {
            fprintf(stderr, "parler-bench: failed to open json output: %s\n", json_out.c_str());
        }
    }

    ggml_gallocr_free(allocr);
    parler_free_model(model);
    return 0;
}
