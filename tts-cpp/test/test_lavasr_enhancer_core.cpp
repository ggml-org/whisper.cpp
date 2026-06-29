// Parity test for the LavaSR enhancer scalar CPU core against an onnxruntime
// golden (real / imag spectrogram from the original ONNX backbone + spec head).
//
// Fixture test: pass the fixtures directory (produced by
// scripts/dump-lavasr-enhancer-fixtures.py) as argv[1].  The directory holds
// the enhancer weights as <canonical-name>.npy plus mel.npy / real.npy /
// imag.npy.  When the directory is absent the test is registered DISABLED via
// the CMake REQUIRES gate, so CI without the model still passes.

#include "lavasr/enhancer.h"
#include "lavasr/enhancer_core.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using tts_cpp::lavasr::EnhancerWeights;
using tts_cpp::lavasr::EnhTensor;

static std::vector<float> load_f32(const std::string & path, std::vector<int> * shape = nullptr) {
    npy_array a = npy_load(path);
    if (a.dtype != "<f4") {
        throw std::runtime_error("expected <f4 in " + path + " got " + a.dtype);
    }
    const size_t n = a.n_elements();
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

static void load_weight(EnhancerWeights & w, const std::string & dir, const std::string & name) {
    EnhTensor t;
    std::vector<int> shape;
    t.data  = load_f32(dir + "/" + name + ".npy", &shape);
    t.shape = shape;
    w.t[name] = std::move(t);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixtures-dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];

    EnhancerWeights w;
    try {
        load_weight(w, dir, "enhancer.embed.weight");
        load_weight(w, dir, "enhancer.embed.bias");
        load_weight(w, dir, "enhancer.norm.weight");
        load_weight(w, dir, "enhancer.norm.bias");
        for (int i = 0; i < w.n_blocks; i++) {
            const std::string p = "enhancer.block." + std::to_string(i) + ".";
            for (const char * s : {"dwconv.weight", "dwconv.bias", "norm.weight",
                                   "norm.bias", "pwconv1.weight", "pwconv1.bias",
                                   "pwconv2.weight", "pwconv2.bias", "gamma"}) {
                load_weight(w, dir, p + s);
            }
        }
        load_weight(w, dir, "enhancer.final_norm.weight");
        load_weight(w, dir, "enhancer.final_norm.bias");
        load_weight(w, dir, "spec_head.out.weight");
        load_weight(w, dir, "spec_head.out.bias");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: loading weights: %s\n", e.what());
        return 1;
    }

    std::vector<int> mel_shape;
    std::vector<float> mel = load_f32(dir + "/mel.npy", &mel_shape);
    if (mel_shape.size() != 2 || mel_shape[0] != w.n_mels) {
        std::fprintf(stderr, "FAIL: mel shape unexpected\n");
        return 1;
    }
    const int T = mel_shape[1];

    std::vector<float> real_out, imag_out;
    tts_cpp::lavasr::enhancer_spec_forward(w, mel, T, real_out, imag_out);

    std::vector<int> rshape;
    std::vector<float> real_gold = load_f32(dir + "/real.npy", &rshape);
    std::vector<float> imag_gold = load_f32(dir + "/imag.npy");

    auto compare = [](const char * name, const std::vector<float> & a,
                      const std::vector<float> & b) -> float {
        if (a.size() != b.size()) {
            std::fprintf(stderr, "FAIL: %s size %zu != golden %zu\n", name, a.size(),
                         b.size());
            return 1e9f;
        }
        float maxd = 0.0f, maxg = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            maxd = std::max(maxd, std::fabs(a[i] - b[i]));
            maxg = std::max(maxg, std::fabs(b[i]));
        }
        std::printf("  %-6s max_abs_err=%.3e  rel=%.3e  (n=%zu)\n", name, maxd,
                    maxd / (maxg + 1e-9f), a.size());
        return maxd;
    };

    std::printf("LavaSR enhancer core parity (T=%d):\n", T);
    const float dr = compare("real", real_out, real_gold);
    const float di = compare("imag", imag_out, imag_gold);

    // erf / exp / cos / sin in f32 vs onnxruntime: numpy reference matched to
    // ~9e-4; allow a small margin for std-lib transcendental differences.
    const float tol = 3e-3f;
    int failures = 0;
    if (!(dr < tol && di < tol)) {
        std::fprintf(stderr, "FAIL: enhancer core exceeds tolerance %.0e\n", tol);
        ++failures;
    }

    // End-to-end pipeline smoke test: a 1 s 24 kHz tone enhances to a finite
    // 48 kHz signal of the expected length (exercises resampler + mel + core +
    // ISTFT + FastLR merge together).
    {
        const int sr_in = 24000;
        std::vector<float> pcm(sr_in, 0.0f);
        for (int i = 0; i < sr_in; i++) {
            pcm[i] = 0.3f * std::sin(2.0f * 3.14159265f * 220.0f * i / sr_in);
        }
        std::vector<float> out48 = tts_cpp::lavasr::enhance(w, pcm, sr_in);
        const size_t expect = static_cast<size_t>(std::lround(pcm.size() * 48000.0 / sr_in));
        bool finite = true;
        for (float v : out48) {
            if (!std::isfinite(v)) { finite = false; break; }
        }
        std::printf("  pipeline: in=%zu @24k -> out=%zu @48k (expect=%zu)\n",
                    pcm.size(), out48.size(), expect);
        if (out48.size() != expect) {
            std::fprintf(stderr, "FAIL: enhance() output length %zu != %zu\n",
                         out48.size(), expect);
            ++failures;
        }
        if (!finite) {
            std::fprintf(stderr, "FAIL: enhance() produced non-finite samples\n");
            ++failures;
        }
    }

    if (failures == 0) {
        std::printf("OK: enhancer core matches onnxruntime golden + pipeline runs (tol=%.0e)\n", tol);
        return 0;
    }
    return 1;
}
