// LavaSR denoiser GGUF round-trip parity test (QVAC-16579 follow-up).
//
// Loads the converted denoiser GGUF, runs the UL-UNAS neural core
// (denoiser_net_forward) on the onnxruntime golden input and compares against
// the golden output, then exercises the full public Denoiser::denoise()
// pipeline (resample + STFT + chunked overlap-add + ISTFT) for length/finiteness.
//
// Fixture test: argv[1] = GGUF path, argv[2] = fixtures dir (spec_in.npy /
// spec_out.npy from scripts/dump-lavasr-denoiser-fixtures.py).  Registered
// DISABLED via the CMake REQUIRES gate when the model/fixtures are absent.

#include "tts-cpp/lavasr/denoiser.h" // public API
#include "lavasr/denoiser_core.h"    // internal: DenoiserWeights, denoiser_net_forward
#include "lavasr/denoiser_gguf.h"    // internal: load_denoiser_gguf
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using tts_cpp::lavasr::DenoiserWeights;

static std::vector<float> load_f32(const std::string & path, std::vector<int> * shape = nullptr) {
    npy_array a = npy_load(path);
    if (a.dtype != "<f4") {
        throw std::runtime_error("expected <f4 in " + path + " got " + a.dtype);
    }
    const size_t      n = a.n_elements();
    std::vector<float> v(n);
    std::memcpy(v.data(), a.data.data(), n * sizeof(float));
    if (shape) {
        shape->clear();
        for (auto d : a.shape) {
            shape->push_back(static_cast<int>(d));
        }
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <denoiser.gguf> <fixtures-dir>\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string dir  = argv[2];

    DenoiserWeights w;
    std::string     err;
    if (!tts_cpp::lavasr::load_denoiser_gguf(gguf, w, &err)) {
        std::fprintf(stderr, "FAIL: %s\n", err.c_str());
        return 1;
    }
    std::printf("loaded GGUF: work_sr=%d spec_bins=%d erb=%d/%d chunk=%d/%d tensors=%zu\n",
                w.work_sample_rate, w.spec_bins, w.erb_low, w.erb_high,
                w.chunk_frames, w.chunk_hop, w.t.size());

    // --- neural core parity vs onnxruntime golden ---
    std::vector<int> ishape;
    std::vector<float> spec_in  = load_f32(dir + "/spec_in.npy", &ishape);
    std::vector<float> spec_out = load_f32(dir + "/spec_out.npy");
    if (ishape.size() != 3 || ishape[0] != 2 || ishape[2] != w.spec_bins) {
        std::fprintf(stderr, "FAIL: spec_in shape unexpected\n");
        return 1;
    }
    const int    T  = ishape[1];
    const size_t TF = static_cast<size_t>(T) * w.spec_bins;
    std::vector<float> re(spec_in.begin(), spec_in.begin() + TF);
    std::vector<float> im(spec_in.begin() + TF, spec_in.begin() + 2 * TF);

    std::vector<float> orr, oii;
    tts_cpp::lavasr::denoiser_net_forward(w, re, im, T, orr, oii);

    float maxd = 0.0f, maxg = 0.0f;
    for (size_t i = 0; i < TF; i++) {
        maxd = std::max(maxd, std::fabs(orr[i] - spec_out[i]));
        maxd = std::max(maxd, std::fabs(oii[i] - spec_out[TF + i]));
        maxg = std::max(maxg, std::fabs(spec_out[i]));
        maxg = std::max(maxg, std::fabs(spec_out[TF + i]));
    }
    std::printf("denoiser core parity (T=%d): max_abs_err=%.3e rel=%.3e\n", T, maxd,
                maxd / (maxg + 1e-9f));

    int failures = 0;
    // exp/tanh/sigmoid/GRU in f32 vs onnxruntime: numpy ref matched to ~2e-6;
    // allow a small margin for std-lib transcendental differences.
    const float tol = 3e-3f;
    if (!(maxd < tol)) {
        std::fprintf(stderr, "FAIL: denoiser core exceeds tolerance %.0e\n", tol);
        ++failures;
    }

    auto dn = tts_cpp::lavasr::Denoiser::load(gguf);
    if (dn->native_sample_rate() != w.work_sample_rate) {
        std::fprintf(stderr, "FAIL: native_sample_rate mismatch\n");
        ++failures;
    }

    // --- FULL pipeline parity vs the reference golden (STFT + multi-chunk
    //     squared-Hann overlap-add + ISTFT).  pcm_in/pcm_out are 16 kHz, so the
    //     resampler is identity and this isolates the STFT/OLA/ISTFT math that
    //     the spec-level core parity above can't see.  The golden is produced by
    //     scripts/dump-lavasr-denoiser-pipeline-fixtures.py (T>63 -> exercises
    //     the overlap-add seams). ---
    {
        std::vector<int> pshape;
        std::vector<float> pcm_in  = load_f32(dir + "/pcm_in.npy", &pshape);
        std::vector<float> pcm_out = load_f32(dir + "/pcm_out.npy");
        std::vector<float> got     = dn->denoise(pcm_in, w.work_sample_rate);
        if (got.size() != pcm_in.size() || pcm_out.size() != pcm_in.size()) {
            std::fprintf(stderr, "FAIL: pipeline length %zu / golden %zu / in %zu\n",
                         got.size(), pcm_out.size(), pcm_in.size());
            ++failures;
        } else {
            float pmax = 0.0f, psig = 0.0f;
            double se = 0.0, sg = 0.0;
            bool   finite = true;
            for (size_t i = 0; i < got.size(); i++) {
                if (!std::isfinite(got[i])) { finite = false; break; }
                const float e = std::fabs(got[i] - pcm_out[i]);
                pmax = std::max(pmax, e);
                psig = std::max(psig, std::fabs(pcm_out[i]));
                se += static_cast<double>(e) * e;
                sg += static_cast<double>(pcm_out[i]) * pcm_out[i];
            }
            const float rrms = static_cast<float>(std::sqrt(se / std::max<double>(sg, 1e-12)));
            std::printf("  pipeline parity (N=%zu, T>63): max_abs_err=%.3e rel_rms=%.3e (sig<=%.3f)\n",
                        got.size(), pmax, rrms, psig);
            // f32 radix-2 STFT/ISTFT vs the numpy f64 golden: allow a small
            // absolute margin (the OLA + ISTFT reconstruction is well within it).
            const float tol_pipe = 2e-3f;
            if (!finite) {
                std::fprintf(stderr, "FAIL: pipeline produced non-finite samples\n");
                ++failures;
            }
            if (!(pmax < tol_pipe)) {
                std::fprintf(stderr, "FAIL: pipeline exceeds tolerance %.0e\n", tol_pipe);
                ++failures;
            }
        }
    }

    // --- pipeline smoke at a non-work rate (exercises the resampler in+out and
    //     the rate-preserving length/finiteness contract). ---
    {
        const int          sr_in = 24000;
        std::vector<float> pcm(sr_in, 0.0f);
        for (int i = 0; i < sr_in; i++) {
            pcm[i] = 0.3f * std::sin(2.0f * 3.14159265f * 220.0f * i / sr_in) +
                     0.05f * std::sin(2.0f * 3.14159265f * 5000.0f * i / sr_in);
        }
        std::vector<float> out = dn->denoise(pcm, sr_in);
        bool               finite = true;
        for (float v : out) {
            if (!std::isfinite(v)) { finite = false; break; }
        }
        std::printf("  pipeline smoke: in=%zu @24k -> out=%zu (rate-preserving)\n", pcm.size(), out.size());
        if (out.size() != pcm.size()) {
            std::fprintf(stderr, "FAIL: denoise() length %zu != %zu\n", out.size(), pcm.size());
            ++failures;
        }
        if (!finite) {
            std::fprintf(stderr, "FAIL: denoise() produced non-finite samples\n");
            ++failures;
        }
        if (!dn->denoise({}, sr_in).empty()) {
            std::fprintf(stderr, "FAIL: denoise({}) not empty\n");
            ++failures;
        }
    }

    if (failures == 0) {
        std::printf("OK: denoiser matches onnxruntime golden + pipeline runs (tol=%.0e)\n", tol);
        return 0;
    }
    return 1;
}
