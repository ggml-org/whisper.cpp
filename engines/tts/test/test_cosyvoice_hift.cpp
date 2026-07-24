// Parity test — CosyVoice3 CausalHiFT vocoder: mel -> 24 kHz waveform.
//
// Runs cosyvoice_hift_synth() on a reference mel and asserts the waveform
// correlates with the PyTorch reference (hift_wav.npy) above a threshold.
// HiFT's NSF source excitation is partly stochastic (unvoiced noise, different
// RNG than the reference), so an exact match is not expected — the gate is a
// Pearson correlation, which caught the earlier metallic phase bug
// (0.136 -> 0.996 after the fix). Exits non-zero on failure; registered as a
// ctest, auto-disabled when fixtures are absent.
//
// Fixtures (generate with scripts/dump-cosyvoice3-reference.py):
//   <in-dir>/{hift_mel_in,hift_wav}.npy
//
// Usage:
//   test-cosyvoice-hift --hift-gguf HIFT.gguf --in-dir DIR [--min-corr 0.90]

#include "npy.h"
#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double pearson(const float * a, const float * b, size_t n) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= (double)n; mb /= (double)n;
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    return cov / (std::sqrt(va) * std::sqrt(vb));
}

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double min_corr = 0.90;
    int seed = 42;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hift-gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--min-corr" && i + 1 < argc) min_corr = std::atof(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --hift-gguf HIFT.gguf --in-dir DIR [--min-corr 0.90]\n", argv[0]); return 2; }
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "missing --hift-gguf / --in-dir\n"); return 2; }

    const int MEL = 80;
    model_ctx m = cosyvoice_load_gguf(gguf);

    // mel npy shape (80, T) -> channel-major flat[ch*T + t].
    npy_array mel_a = npy_load(in_dir + "/hift_mel_in.npy");
    int T_mel = (int)mel_a.shape[1];
    std::vector<float> mel((size_t)T_mel * MEL);
    std::memcpy(mel.data(), npy_as_f32(mel_a), mel.size() * sizeof(float));

    npy_array ref_a = npy_load(in_dir + "/hift_wav.npy");
    size_t ref_n = 1;
    for (size_t d = 0; d < ref_a.shape.size(); ++d) ref_n *= (size_t)ref_a.shape[d];

    std::vector<float> wav = cosyvoice_hift_synth(m, mel, T_mel, seed);

    size_t n = std::min(wav.size(), ref_n);
    if (n == 0) { fprintf(stderr, "FAIL: empty waveform\n"); return 1; }
    double corr = pearson(wav.data(), npy_as_f32(ref_a), n);

    fprintf(stderr, "hift waveform corr = %.6f  (threshold %.4f, got=%zu ref=%zu)\n",
            corr, min_corr, wav.size(), ref_n);
    if (!(corr >= min_corr)) { fprintf(stderr, "FAIL: hift waveform correlation below threshold\n"); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
