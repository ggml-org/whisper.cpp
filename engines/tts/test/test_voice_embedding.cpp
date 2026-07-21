// End-to-end voice-embedding validation harness.
//
// Pipeline:
//   wav_16k (float32) → fbank_kaldi_80 → mean-subtract over T
//   → campplus_embed → 192-d speaker embedding
//
// Usage:
//   ./build/test-voice-embedding S3GEN.gguf WAV_16K.npy EMBEDDING.npy

#include "campplus.h"
#include "voice_features.h"
#include "npy.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s S3GEN.gguf WAV_16K.npy EMBEDDING.npy\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_path  = argv[3];

    // Load the Kaldi mel filterbank from the GGUF.
    fprintf(stderr, "[1/5] loading mel filterbank from %s\n", gguf_path.c_str());
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) { fprintf(stderr, "cannot open gguf\n"); return 1; }
    ggml_tensor * fb_t = ggml_get_tensor(tmp_ctx, "campplus/mel_fb_kaldi_80");
    if (!fb_t) { fprintf(stderr, "mel_fb_kaldi_80 missing\n"); gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx); return 1; }
    std::vector<float> mel_fb(ggml_nelements(fb_t));
    std::memcpy(mel_fb.data(), ggml_get_data(fb_t), ggml_nbytes(fb_t));
    gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);

    fprintf(stderr, "[2/5] loading CAMPPlus weights\n");
    campplus_weights w;
    if (!campplus_load(gguf_path, w)) return 1;

    fprintf(stderr, "[3/5] loading wav %s\n", wav_path.c_str());
    npy_array wav_npy = npy_load(wav_path);
    std::vector<float> wav((const float*)wav_npy.data.data(),
                           (const float*)wav_npy.data.data() + wav_npy.n_elements());
    fprintf(stderr, "      %zu samples (%.2f s)\n", wav.size(), (double)wav.size() / 16000.0);

    fprintf(stderr, "[4/5] fbank → mean-subtract → CAMPPlus\n");
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> fb_cpp = fbank_kaldi_80(wav, mel_fb);
    auto t1 = std::chrono::steady_clock::now();
    const int T = (int)(fb_cpp.size() / 80);
    fprintf(stderr, "      fbank (T=%d, 80)  %.1f ms\n", T,
            std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Mean-subtract over T (per channel) — matches extract_feature().
    std::vector<float> col_mean(80, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) col_mean[c] += fb_cpp[(size_t)t * 80 + c];
    for (int c = 0; c < 80; ++c) col_mean[c] /= (float)T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) fb_cpp[(size_t)t * 80 + c] -= col_mean[c];

    std::vector<float> emb;
    auto t2 = std::chrono::steady_clock::now();
    if (!campplus_embed(fb_cpp, T, w, /*backend=*/nullptr, emb)) return 1;
    auto t3 = std::chrono::steady_clock::now();
    fprintf(stderr, "      CAMPPlus  %.1f ms  → %zu dims\n",
            std::chrono::duration<double, std::milli>(t3 - t2).count(), emb.size());
    fprintf(stderr, "      total wav→embedding  %.1f ms\n",
            std::chrono::duration<double, std::milli>(t3 - t0).count());

    fprintf(stderr, "[5/5] comparing against %s\n", ref_path.c_str());
    npy_array ref = npy_load(ref_path);
    const size_t n = std::min(emb.size(), ref.n_elements());
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
        "\n[result] C++ wav→embedding vs Python:\n"
        "    n=%zu  max_abs=%.4e  rms=%.4e  max|ref|=%.4e  rel=%.4e\n"
        "    cosine similarity = %.6f\n",
        n, (double)ma, rms, (double)mref, rel, cos);
    return 0;
}
