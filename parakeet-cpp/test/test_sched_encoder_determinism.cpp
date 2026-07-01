// Cached encoder-graph reuse determinism: reusing the cached encoder graph across
// many run_encoder calls must produce byte-identical output on every backend.
//
// Regression guard for the cached-graph reuse corruption. Routing a *reused* cached
// graph through ggml_backend_sched produced correct output on first use but garbage
// on every reuse on Adreno OpenCL/Vulkan (CPU and Metal were unaffected). The encoder
// now allocates and computes its cached graph directly on the active backend through
// a persistent gallocr (EncoderGraph::alloc), which reuses byte-identically on every
// backend. This test runs run_encoder N times on the same cached graph and asserts
// every run's encoder_out and logits are byte-identical to run 0 (and finite /
// non-degenerate), on whatever backend --n-gpu-layers selects (CPU / Metal / OpenCL /
// Vulkan). Streaming (AOSC) reuses the cached graph across chunks, so this locks the
// exact production path that regressed.
//
// Usage:
//   test-sched-encoder-determinism --model <gguf> --wav <wav>
//                                  [--runs N] [--n-gpu-layers N] [--verbose]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav> [--runs N] [--n-gpu-layers N] [--verbose]\n"
        "\n"
        "Runs run_encoder --runs times (default 20) on the same cached encoder graph\n"
        "and asserts encoder_out / logits stay byte-identical across every reuse on\n"
        "the selected backend (--n-gpu-layers 0 = CPU, > 0 = GPU).\n",
        argv0);
}

bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

size_t first_diff(const std::vector<float> & a, const std::vector<float> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return i;
    }
    return n;  // sizes differ but common prefix equal
}

// A degenerate encoder output (all zero, or any NaN/Inf) would make a byte-equal
// assertion pass trivially; reject it so the determinism check has real content.
bool non_degenerate(const std::vector<float> & v) {
    if (v.empty()) return false;
    bool any_nonzero = false;
    for (float x : v) {
        if (!std::isfinite(x)) return false;
        if (x != 0.0f) any_nonzero = true;
    }
    return any_nonzero;
}

}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string wav_path;
    int  n_runs       = 20;
    int  n_gpu_layers = 0;
    bool verbose      = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"        && i + 1 < argc) model_path   = argv[++i];
        else if (a == "--wav"          && i + 1 < argc) wav_path     = argv[++i];
        else if (a == "--runs"         && i + 1 < argc) n_runs       = std::atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--verbose" || a == "-v")         verbose      = true;
        else if (a == "-h" || a == "--help")            { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || wav_path.empty()) { usage(argv[0]); return 2; }
    if (n_runs < 2) n_runs = 2;

    using namespace parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(model_path, model, /*n_threads=*/0,
                                n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[test-sched-encoder-determinism] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    const std::string backend_name = model_active_backend_name(model);
    std::fprintf(stderr,
        "[test-sched-encoder-determinism] backend=%s runs=%d\n", backend_name.c_str(), n_runs);

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "[test-sched-encoder-determinism] load_wav rc=%d\n", rc);
        return 1;
    }
    if (sr != model.mel_cfg.sample_rate) {
        std::fprintf(stderr,
            "[test-sched-encoder-determinism] FAIL: wav sr %d != model sr %d\n",
            sr, model.mel_cfg.sample_rate);
        return 1;
    }

    std::vector<float> mel;
    int                n_mel_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model.mel_cfg, mel, n_mel_frames); rc != 0) {
        std::fprintf(stderr, "[test-sched-encoder-determinism] compute_log_mel rc=%d\n", rc);
        return 1;
    }

    std::vector<float> ref_encoder_out;
    std::vector<float> ref_logits;

    for (int k = 0; k < n_runs; ++k) {
        EncoderOutputs out;
        if (int rc = run_encoder(model, mel.data(), n_mel_frames, model.mel_cfg.n_mels,
                                 out, /*max_layers=*/-1,
                                 /*capture_intermediates=*/false); rc != 0) {
            std::fprintf(stderr,
                "[test-sched-encoder-determinism] run_encoder rc=%d (run %d)\n", rc, k);
            return 1;
        }

        // Every run must stay finite / non-degenerate: a corrupted reuse typically
        // reads garbage or goes non-finite.
        if (!non_degenerate(out.encoder_out)) {
            std::fprintf(stderr,
                "[test-sched-encoder-determinism] FAIL: run %d encoder_out degenerate "
                "(empty / all-zero / non-finite)\n", k);
            return 1;
        }

        if (k == 0) {
            ref_encoder_out = out.encoder_out;
            ref_logits      = out.logits;
            if (verbose) {
                std::fprintf(stderr, "[verbose] run  0 (ref): %zu enc, %zu logits\n",
                             ref_encoder_out.size(), ref_logits.size());
            }
            continue;
        }

        if (!bit_equal(out.encoder_out, ref_encoder_out)) {
            const size_t i = first_diff(out.encoder_out, ref_encoder_out);
            std::fprintf(stderr,
                "[test-sched-encoder-determinism] FAIL: run %d encoder_out differs from run 0 "
                "at index %zu (%.9g vs %.9g) on backend %s -- cached-graph reuse is not "
                "deterministic\n",
                k, i, i < out.encoder_out.size() ? out.encoder_out[i] : 0.0f,
                i < ref_encoder_out.size() ? ref_encoder_out[i] : 0.0f, backend_name.c_str());
            return 1;
        }
        if (!bit_equal(out.logits, ref_logits)) {
            const size_t i = first_diff(out.logits, ref_logits);
            std::fprintf(stderr,
                "[test-sched-encoder-determinism] FAIL: run %d logits differ from run 0 "
                "at index %zu (%.9g vs %.9g) on backend %s\n",
                k, i, i < out.logits.size() ? out.logits[i] : 0.0f,
                i < ref_logits.size() ? ref_logits[i] : 0.0f, backend_name.c_str());
            return 1;
        }
        if (verbose) {
            std::fprintf(stderr, "[verbose] run %2d: byte-identical to run 0\n", k);
        }
    }

    std::fprintf(stderr,
        "[test-sched-encoder-determinism] PASS  %d cached-graph reuses byte-identical on %s "
        "(%zu enc, %zu logits)\n",
        n_runs, backend_name.c_str(), ref_encoder_out.size(), ref_logits.size());
    return 0;
}
