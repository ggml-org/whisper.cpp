// Validation harness for the C++ Kaldi fbank port.
//
// Usage:
//   ./build/test-fbank S3GEN.gguf WAV_16K.npy FBANK_RAW.npy

#include "voice_features.h"
#include "npy.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s S3GEN.gguf WAV_16K.npy FBANK_RAW.npy\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_path  = argv[3];

    // Load the Kaldi mel filterbank from the GGUF metadata.
    fprintf(stderr, "[1/3] loading mel filterbank from %s\n", gguf_path.c_str());
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) { fprintf(stderr, "cannot open gguf\n"); return 1; }
    ggml_tensor * fb = ggml_get_tensor(tmp_ctx, "campplus/mel_fb_kaldi_80");
    if (!fb) {
        fprintf(stderr, "campplus/mel_fb_kaldi_80 missing; rerun convert-s3gen-to-gguf.py\n");
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return 1;
    }
    std::vector<float> mel_fb(ggml_nelements(fb));
    std::memcpy(mel_fb.data(), ggml_get_data(fb), ggml_nbytes(fb));
    gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);

    fprintf(stderr, "[2/3] loading 16 kHz wav %s\n", wav_path.c_str());
    npy_array wav_npy = npy_load(wav_path);
    std::vector<float> wav((const float*)wav_npy.data.data(),
                           (const float*)wav_npy.data.data() + wav_npy.n_elements());
    fprintf(stderr, "      %zu samples (%.2f s)\n", wav.size(), (double)wav.size() / 16000.0);

    fprintf(stderr, "[3/3] running fbank_kaldi_80\n");
    std::vector<float> fb_cpp = fbank_kaldi_80(wav, mel_fb);
    if (fb_cpp.empty()) { fprintf(stderr, "fbank failed\n"); return 1; }
    const int T = (int)(fb_cpp.size() / 80);
    fprintf(stderr, "      output (T=%d, 80)\n", T);

    npy_array ref = npy_load(ref_path);
    const float * r_dbg = (const float *)ref.data.data();
    fprintf(stderr, "  C++ fb[0, :8]:");
    for (int j = 0; j < 8; ++j) fprintf(stderr, " %.4f", fb_cpp[j]);
    fprintf(stderr, "\n  Py  fb[0, :8]:");
    for (int j = 0; j < 8; ++j) fprintf(stderr, " %.4f", r_dbg[j]);
    fprintf(stderr, "\n  C++ fb[100, :8]:");
    for (int j = 0; j < 8; ++j) fprintf(stderr, " %.4f", fb_cpp[100*80 + j]);
    fprintf(stderr, "\n  Py  fb[100, :8]:");
    for (int j = 0; j < 8; ++j) fprintf(stderr, " %.4f", r_dbg[100*80 + j]);
    fprintf(stderr, "\n");
    const size_t n = std::min(fb_cpp.size(), ref.n_elements());
    const float * r = (const float *)ref.data.data();
    float ma = 0.0f, rsum = 0.0f, mref = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fb_cpp[i] - r[i];
        ma = std::max(ma, std::fabs(d));
        rsum += d * d;
        mref = std::max(mref, std::fabs(r[i]));
    }
    double rms = std::sqrt(rsum / n);
    double rel = ma / std::max(mref, 1e-12f);
    fprintf(stderr,
        "\n[result] C++ vs Python fbank:\n"
        "    n=%zu  max_abs=%.4e  rms=%.4e  max|ref|=%.4e  rel=%.4e\n",
        n, (double)ma, rms, (double)mref, rel);
    return 0;
}
