// End-to-end test for the LavaSR enhancer GGUF path: loads the converted
// GGUF (f32 or f16) via load_enhancer_gguf(), runs the scalar forward, and
// compares real/imag against the onnxruntime golden. Also smoke-tests the
// public tts_cpp::lavasr::Enhancer API.
//
// Fixture test (links ggml for GGUF I/O):
//   argv[1] = enhancer GGUF path
//   argv[2] = fixtures dir (mel.npy / real.npy / imag.npy)
//   argv[3] = optional max-abs-error tolerance (default 3e-3; pass e.g. 8e-2
//             for an f16 GGUF). DISABLED via CMake REQUIRES when absent.

#include "lavasr/enhancer_core.h"
#include "lavasr/enhancer_gguf.h"
#include "tts-cpp/lavasr/enhancer.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> load_f32(const std::string & path, std::vector<int> * shape = nullptr) {
    npy_array a = npy_load(path);
    std::vector<float> v(a.n_elements());
    std::memcpy(v.data(), a.data.data(), v.size() * sizeof(float));
    if (shape) {
        shape->clear();
        for (auto d : a.shape) shape->push_back(static_cast<int>(d));
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <enhancer.gguf> <fixtures-dir> [tol]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string dir  = argv[2];
    const float tol = argc > 3 ? std::strtof(argv[3], nullptr) : 3e-3f;

    tts_cpp::lavasr::EnhancerWeights w;
    std::string err;
    if (!tts_cpp::lavasr::load_enhancer_gguf(gguf, w, &err)) {
        std::fprintf(stderr, "FAIL: %s\n", err.c_str());
        return 1;
    }
    std::printf("loaded GGUF: dim=%d blocks=%d n_mels=%d n_fft=%d spec_bins=%d clip_max=%.1f\n",
                w.dim, w.n_blocks, w.n_mels, w.n_fft, w.spec_bins, w.clip_max);

    std::vector<int> mel_shape;
    std::vector<float> mel = load_f32(dir + "/mel.npy", &mel_shape);
    const int T = mel_shape[1];

    std::vector<float> real, imag;
    tts_cpp::lavasr::enhancer_spec_forward(w, mel, T, real, imag);

    std::vector<float> real_gold = load_f32(dir + "/real.npy");
    std::vector<float> imag_gold = load_f32(dir + "/imag.npy");

    auto cmp = [](const char * n, const std::vector<float> & a, const std::vector<float> & b) {
        float m = 0.0f, g = 0.0f;
        for (size_t i = 0; i < a.size() && i < b.size(); i++) {
            m = std::max(m, std::fabs(a[i] - b[i]));
            g = std::max(g, std::fabs(b[i]));
        }
        std::printf("  %-5s max_abs_err=%.3e rel=%.3e\n", n, m, m / (g + 1e-9f));
        return m;
    };

    int failures = 0;
    std::printf("GGUF enhancer parity (T=%d, tol=%.0e):\n", T, tol);
    if (cmp("real", real, real_gold) >= tol) ++failures;
    if (cmp("imag", imag, imag_gold) >= tol) ++failures;

    // Public Enhancer API + end-to-end DSP golden: enhance(pcm_in) is compared
    // against the numpy reference pipeline output (resample + mel + ISTFT +
    // FastLR + neural), so the DSP front/back-end is checked, not just the
    // neural core. Falls back to a finite/length smoke test if the e2e fixtures
    // are absent (older fixture dumps). argv[4] overrides the e2e tolerance.
    try {
        auto enh = tts_cpp::lavasr::Enhancer::load(gguf);
        std::vector<float> pcm_in;
        bool have_e2e = true;
        try {
            pcm_in = load_f32(dir + "/pcm_in.npy");
        } catch (...) {
            have_e2e = false;
        }
        if (have_e2e) {
            const float e2e_tol = argc > 4 ? std::strtof(argv[4], nullptr) : 5e-2f;
            std::vector<float> golden = load_f32(dir + "/enhanced_48k.npy");
            auto out = enh->enhance(pcm_in, 24000);
            std::printf("Enhancer::enhance e2e (in=%zu @24k -> out=%zu @%d, tol=%.0e):\n",
                        pcm_in.size(), out.size(), enh->output_sample_rate(), e2e_tol);
            if (enh->output_sample_rate() != 48000) {
                std::fprintf(stderr, "FAIL: enhance() reports %d, expected 48000\n",
                             enh->output_sample_rate());
                ++failures;
            }
            if (out.size() != golden.size()) {
                std::fprintf(stderr, "FAIL: enhance() length %zu != golden %zu\n",
                             out.size(), golden.size());
                ++failures;
            } else {
                const float d = cmp("e2e", out, golden);
                if (d >= e2e_tol) {
                    std::fprintf(stderr, "FAIL: e2e exceeds tolerance %.0e\n", e2e_tol);
                    ++failures;
                }
            }
        } else {
            std::vector<float> pcm(12000, 0.0f);
            for (int i = 0; i < 12000; i++)
                pcm[i] = 0.2f * std::sin(2.0f * 3.14159265f * 200.0f * i / 24000.0f);
            auto out = enh->enhance(pcm, 24000);
            bool finite = true;
            for (float v : out) if (!std::isfinite(v)) { finite = false; break; }
            std::printf("  (e2e golden absent) smoke: out=%zu sr=%d finite=%d\n",
                        out.size(), enh->output_sample_rate(), finite ? 1 : 0);
            if (out.empty() || !finite || enh->output_sample_rate() != 48000) ++failures;
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: Enhancer API: %s\n", e.what());
        ++failures;
    }

    if (failures == 0) {
        std::printf("OK: enhancer GGUF path matches golden + public API works\n");
        return 0;
    }
    return 1;
}
