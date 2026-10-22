// S3Gen + HiFT backend-agreement harness: the same GGUF is synthesized on the
// CPU and on the selected GPU in one process. Comparing the two backends on
// identical weights makes quantization cancel exactly, so the check holds for
// any tier of the model -- including the q4_0 GGUFs the model registry
// serves -- and needs no reference dump.
// Identical weights remove the model-vs-reference gap, but the two backends
// still run different quantized-matmul paths (dequantization arithmetic,
// accumulation order), so the residual disagreement is larger at q4_0 than at
// f16 and the tolerances are per tier, passed by the registration. The strict
// gate is the post-CFM mel, which is deterministic feedforward compute. The waveform is deliberately held to a weaker bar: HiFT's SineGen
// integrates the predicted f0 into a phase, so a microscopic cross-backend f0
// difference drifts the phase and pointwise sample comparison shows order-1
// spikes on audio that is equivalent -- observed as tiny mean error with a
// large max on every backend pair. Length, finiteness and relative RMS still
// catch silence, truncation and garbage.
//
// The token count is sized so the audio crosses the 65535 CUDA grid-dimension
// limit (a ~30 s chunk aborted the launch before the ggml-cuda grid-striding
// fix), so a regression there fails this test instead of only failing users.

#include "test_env_portable.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int    SEED             = 42;
constexpr int    THREADS          = 4;
constexpr int    GPU_LAYERS       = 99;
constexpr int    LONG_TOKENS      = 900;  // ~36 s of audio at ~40 ms per token
constexpr int    SHORT_TOKENS     = 50;   // ~2 s, matches the sched-equivalence list
constexpr int    TOKEN_VOCAB_SPAN = 4200; // valid S3 tokenizer id range
// Defaults fit the f16 tier: worst observed on the RTX 3090 was mel rel
// 1.08e-2 (CUDA, long case) and waveform rms ratio 0.054, so both carry about
// 2x headroom. The q4_0 registrations pass looser bounds measured the same
// way (worst 2.40e-2 and 0.280).
constexpr double DEFAULT_MEL_TOLERANCE  = 2e-2;
constexpr double DEFAULT_MAX_RMS_RATIO  = 0.15;

std::vector<int32_t> make_tokens(int count) {
    std::vector<int32_t> tokens;
    tokens.reserve(count);
    for (int i = 0; i < count; ++i) {
        tokens.push_back((i * 73 + 11) % TOKEN_VOCAB_SPAN);
    }
    return tokens;
}

struct device_choice {
    int         n_gpu_layers = 0;
    std::string backend;
};

bool synthesize(const std::string & gguf, const device_choice & device, int token_count,
                std::vector<float> & pcm, const std::string & mel_path, const char * label) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.out_wav_path.clear();
    opts.pcm_out       = &pcm;
    opts.seed          = SEED;
    opts.n_threads     = THREADS;
    opts.n_gpu_layers  = device.n_gpu_layers;
    opts.backend       = device.backend;
    opts.dump_mel_path = mel_path;

    const int rc = s3gen_synthesize_to_wav(make_tokens(token_count), opts);
    if (rc != 0) {
        std::fprintf(stderr, "%s: s3gen_synthesize_to_wav rc=%d\n", label, rc);
        return false;
    }
    if (pcm.empty()) {
        std::fprintf(stderr, "%s: produced no audio\n", label);
        return false;
    }
    return true;
}

bool compare_mels(const std::string & cpu_path, const std::string & gpu_path,
                  double mel_tolerance) {
    npy_array cpu_mel = npy_load(cpu_path);
    npy_array gpu_mel = npy_load(gpu_path);
    if (cpu_mel.n_elements() != gpu_mel.n_elements() || cpu_mel.n_elements() == 0) {
        std::fprintf(stderr, "  FAIL mel shapes differ: cpu=%zu gpu=%zu\n",
                     cpu_mel.n_elements(), gpu_mel.n_elements());
        return false;
    }
    const compare_stats stats = compare_f32(npy_as_f32(gpu_mel), npy_as_f32(cpu_mel),
                                            cpu_mel.n_elements());
    print_compare("mel cpu-vs-gpu", stats);
    if (!compare_within(stats, mel_tolerance)) {
        std::fprintf(stderr, "  FAIL mel rel=%.3e tolerance=%.3e non_finite=%zu\n",
                     stats.rel_err, mel_tolerance, stats.non_finite);
        return false;
    }
    return true;
}

double rms(const std::vector<float> & x) {
    double sum = 0;
    for (float v : x) sum += (double) v * v;
    return std::sqrt(sum / (double) x.size());
}

bool compare_waveforms(const std::vector<float> & cpu_pcm, const std::vector<float> & gpu_pcm,
                       double max_rms_ratio) {
    if (cpu_pcm.size() != gpu_pcm.size()) {
        std::fprintf(stderr, "  FAIL sample counts differ: cpu=%zu gpu=%zu\n",
                     cpu_pcm.size(), gpu_pcm.size());
        return false;
    }
    const compare_stats stats = compare_f32(gpu_pcm.data(), cpu_pcm.data(), cpu_pcm.size());
    if (stats.non_finite != 0) {
        std::fprintf(stderr, "  FAIL waveform carries %zu non-finite sample(s)\n", stats.non_finite);
        return false;
    }
    const double signal    = rms(cpu_pcm);
    const double rms_ratio = signal > 0 ? stats.rms_err / signal : stats.rms_err;
    std::fprintf(stderr, "  [waveform cpu-vs-gpu] n=%zu rms_err=%.3e signal_rms=%.3e ratio=%.3f\n",
                 cpu_pcm.size(), stats.rms_err, signal, rms_ratio);
    if (rms_ratio > max_rms_ratio) {
        std::fprintf(stderr, "  FAIL waveform rms ratio %.3f exceeds %.3f\n",
                     rms_ratio, max_rms_ratio);
        return false;
    }
    return true;
}

// dump_mel_path also writes per-stage sidecars, named from the path with its
// .npy extension stripped.
void remove_mel_dumps(const std::string & mel_path) {
    std::remove(mel_path.c_str());
    std::string base = mel_path;
    if (base.size() > 4 && base.substr(base.size() - 4) == ".npy") {
        base.resize(base.size() - 4);
    }
    for (const char * sidecar : { "_cond.npy", "_mu.npy", "_spks.npy", "_step0_dxdt.npy" }) {
        std::remove((base + sidecar).c_str());
    }
}

bool compare_case(const std::string & gguf, const device_choice & device, int token_count,
                  const char * label, double mel_tolerance, double max_rms_ratio) {
    std::fprintf(stderr, "\n== %s: %d tokens ==\n", label, token_count);

    const std::string tmp     = test_tmpdir();
    const std::string tag     = test_process_tag();
    const std::string cpu_mel = tmp + "/s3gen_agree_" + tag + "_cpu_" + label + ".npy";
    const std::string gpu_mel = tmp + "/s3gen_agree_" + tag + "_gpu_" + label + ".npy";

    std::vector<float> cpu_pcm;
    if (!synthesize(gguf, device_choice{0, "cpu"}, token_count, cpu_pcm, cpu_mel, "cpu")) {
        return false;
    }
    std::vector<float> gpu_pcm;
    if (!synthesize(gguf, device, token_count, gpu_pcm, gpu_mel, "gpu")) {
        return false;
    }

    const bool mel_ok = compare_mels(cpu_mel, gpu_mel, mel_tolerance);
    const bool wav_ok = compare_waveforms(cpu_pcm, gpu_pcm, max_rms_ratio);
    remove_mel_dumps(cpu_mel);
    remove_mel_dumps(gpu_mel);
    return mel_ok && wav_ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s S3GEN.gguf [--mel-tolerance REL] [--max-rms-ratio R]"
                             " [--backend NAME]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    double mel_tolerance = DEFAULT_MEL_TOLERANCE;
    double max_rms_ratio = DEFAULT_MAX_RMS_RATIO;
    device_choice device{GPU_LAYERS, {}};
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--mel-tolerance" && i + 1 < argc)      mel_tolerance = atof(argv[++i]);
        else if (flag == "--max-rms-ratio" && i + 1 < argc) max_rms_ratio = atof(argv[++i]);
        else if (flag == "--backend" && i + 1 < argc)       device = device_choice{0, argv[++i]};
    }

    int failures = 0;
    if (!compare_case(gguf, device, SHORT_TOKENS, "short", mel_tolerance, max_rms_ratio)) ++failures;
    if (!compare_case(gguf, device, LONG_TOKENS, "long", mel_tolerance, max_rms_ratio)) ++failures;

    if (failures) {
        std::fprintf(stderr, "\n%d case(s) failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "\nPASS\n");
    return 0;
}
