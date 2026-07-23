// Core ML encoder parity: compares the Apple Neural Engine encoder output against the
// ggml encoder on identical mel, reporting per-frame cosine similarity.
//
// Both encoder outputs come from a single model load: run_encoder routes
// capture_intermediates=true to the ggml encoder (baseline) and
// capture_intermediates=false to the Core ML sidecar when it is active, so this
// harness exercises exactly the production Core ML path against the ggml reference.
//
// Skips (exit 0) when the Core ML sidecar is not active -- i.e. on non-Apple builds,
// builds without PARAKEET_COREML, or when no `<model>-encoder.mlmodelc` is present --
// so it is a no-op on CI without Apple hardware and a real gate on Apple.
//
// Usage:
//   test-encoder-coreml-parity --model <gguf> --wav <wav> [--n-gpu-layers N] [--min-cos C]
//
// Exit 0 on success (or skip); non-zero on parity failure or invalid arguments.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav> [--n-gpu-layers N] [--min-cos C]\n"
        "\n"
        "Compares the Core ML (Apple Neural Engine) encoder against the ggml encoder\n"
        "via per-frame cosine similarity. Skips when the Core ML sidecar is inactive.\n",
        argv0);
}

double frame_cosine(const float * a, const float * b, int d) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < d; ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0 ? dot / denom : 1.0;
}

bool has_nan_or_inf(const std::vector<float> & v) {
    for (float x : v) {
        if (std::isnan(x) || std::isinf(x)) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string wav_path;
    int         n_gpu_layers = 1;
    double      min_cos      = 0.99;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"        && i + 1 < argc) model_path   = argv[++i];
        else if (a == "--wav"          && i + 1 < argc) wav_path     = argv[++i];
        else if (a == "--n-gpu-layers" && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--min-cos"      && i + 1 < argc) min_cos      = std::atof(argv[++i]);
        else if (a == "-h" || a == "--help")            { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || wav_path.empty()) { usage(argv[0]); return 2; }

    using namespace parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(model_path, model, /*n_threads=*/0, n_gpu_layers, /*verbose=*/true); rc != 0) {
        std::fprintf(stderr, "[coreml-parity] load_from_gguf rc=%d\n", rc);
        return 1;
    }

    if (!model_encoder_on_coreml(model)) {
        std::fprintf(stderr,
            "[coreml-parity] SKIP: Core ML encoder not active "
            "(non-Apple build, PARAKEET_COREML off, or no <model>-encoder.mlmodelc).\n");
        return 0;
    }

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "[coreml-parity] load_wav rc=%d\n", rc);
        return 1;
    }
    if (sr != model.mel_cfg.sample_rate) {
        std::fprintf(stderr, "[coreml-parity] FAIL: wav sr %d != model sr %d\n", sr, model.mel_cfg.sample_rate);
        return 1;
    }

    std::vector<float> mel;
    int                n_mel_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model.mel_cfg, mel, n_mel_frames); rc != 0) {
        std::fprintf(stderr, "[coreml-parity] compute_log_mel rc=%d\n", rc);
        return 1;
    }

    EncoderOutputs out_ggml;
    if (int rc = run_encoder(model, mel.data(), n_mel_frames, model.mel_cfg.n_mels,
                             out_ggml, /*max_layers=*/-1, /*capture_intermediates=*/true); rc != 0) {
        std::fprintf(stderr, "[coreml-parity] ggml run_encoder rc=%d\n", rc);
        return 1;
    }

    EncoderOutputs out_coreml;
    if (int rc = run_encoder(model, mel.data(), n_mel_frames, model.mel_cfg.n_mels,
                             out_coreml, /*max_layers=*/-1, /*capture_intermediates=*/false); rc != 0) {
        std::fprintf(stderr, "[coreml-parity] Core ML run_encoder rc=%d\n", rc);
        return 1;
    }

    if (out_ggml.n_enc_frames != out_coreml.n_enc_frames || out_ggml.d_model != out_coreml.d_model) {
        std::fprintf(stderr,
            "[coreml-parity] FAIL: shape mismatch ggml=(%d,%d) coreml=(%d,%d)\n",
            out_ggml.n_enc_frames, out_ggml.d_model,
            out_coreml.n_enc_frames, out_coreml.d_model);
        return 1;
    }
    if (has_nan_or_inf(out_coreml.encoder_out)) {
        std::fprintf(stderr, "[coreml-parity] FAIL: Core ML encoder_out has NaN/Inf\n");
        return 1;
    }

    const int T = out_ggml.n_enc_frames;
    const int D = out_ggml.d_model;
    double sum_cos = 0.0;
    double min_frame_cos = 1.0;
    for (int t = 0; t < T; ++t) {
        const double c = frame_cosine(out_ggml.encoder_out.data() + (size_t) t * D,
                                      out_coreml.encoder_out.data() + (size_t) t * D, D);
        sum_cos += c;
        if (c < min_frame_cos) min_frame_cos = c;
    }
    const double mean_cos = T > 0 ? sum_cos / T : 1.0;

    std::fprintf(stderr,
        "[coreml-parity] encoder=%s frames=%d d_model=%d  mean_cos=%.6f  min_cos=%.6f  (gate mean>=%.4f)\n",
        model_encoder_backend_name(model).c_str(), T, D, mean_cos, min_frame_cos, min_cos);

    if (mean_cos < min_cos) {
        std::fprintf(stderr, "[coreml-parity] FAIL: mean cosine %.6f below gate %.4f\n", mean_cos, min_cos);
        return 1;
    }
    std::fprintf(stderr, "[coreml-parity] PASS\n");
    return 0;
}
