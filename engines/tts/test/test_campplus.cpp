// Validation harness for the C++ CAMPPlus port.
//
// Usage:
//   ./build/test-campplus S3GEN.gguf FBANK.npy EMBEDDING.npy

#include "campplus.h"
#include "npy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s S3GEN.gguf FBANK.npy EMBEDDING.npy\n", argv[0]);
        return 1;
    }
    const std::string gguf  = argv[1];
    const std::string fbank = argv[2];
    const std::string embn  = argv[3];

    fprintf(stderr, "[1/3] loading CAMPPlus weights from %s\n", gguf.c_str());
    campplus_weights w;
    if (!campplus_load(gguf, w)) return 1;
    fprintf(stderr, "      feat_dim=%d  embedding_size=%d  seg_pool_len=%d\n",
            w.feat_dim, w.embedding_size, w.seg_pool_len);

    fprintf(stderr, "[2/3] loading fbank %s\n", fbank.c_str());
    npy_array fb = npy_load(fbank);
    if (fb.shape.size() != 2 || (int)fb.shape[1] != w.feat_dim) {
        fprintf(stderr, "bad fbank shape: got [");
        for (auto s : fb.shape) fprintf(stderr, "%lld ", (long long)s);
        fprintf(stderr, "]; expected (T, %d)\n", w.feat_dim);
        return 1;
    }
    const int T = (int)fb.shape[0];
    fprintf(stderr, "      fbank (T=%d, C=%d)\n", T, (int)fb.shape[1]);
    std::vector<float> fbank_flat((const float *)fb.data.data(),
                                  (const float *)fb.data.data() + fb.n_elements());

    fprintf(stderr, "[3/3] running CAMPPlus forward\n");
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> emb;
    if (!campplus_embed(fbank_flat, T, w, /*backend=*/nullptr, emb)) return 1;
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "      forward pass: %.1f ms  out=%zu\n", ms, emb.size());

    npy_array ref = npy_load(embn);
    size_t n = std::min(emb.size(), ref.n_elements());
    const float * r = (const float *)ref.data.data();
    float ma = 0.0f, rsum = 0.0f, mref = 0.0f;
    double dot = 0.0, n_sq_c = 0.0, n_sq_r = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float d = emb[i] - r[i];
        ma = std::max(ma, std::fabs(d));
        rsum += d * d;
        mref = std::max(mref, std::fabs(r[i]));
        dot   += (double)emb[i] * (double)r[i];
        n_sq_c += (double)emb[i] * (double)emb[i];
        n_sq_r += (double)r[i]   * (double)r[i];
    }
    double rms = std::sqrt(rsum / n);
    double rel = ma / std::max(mref, 1e-12f);
    double cos = dot / (std::sqrt(n_sq_c) * std::sqrt(n_sq_r) + 1e-30);
    fprintf(stderr,
        "\n[result] C++ vs Python embedding:\n"
        "    n=%zu  max_abs=%.4e  rms=%.4e  max|ref|=%.4e  rel=%.4e\n"
        "    cosine similarity = %.6f\n",
        n, (double)ma, rms, (double)mref, rel, cos);
    return 0;
}
