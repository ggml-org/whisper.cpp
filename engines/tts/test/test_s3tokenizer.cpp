// Validation harness for the C++ S3TokenizerV2 port.
//
// Usage:
//   ./build/test-s3tokenizer S3GEN.gguf WAV_16K.npy LOG_MEL.npy TOKENS.npy

#include "s3tokenizer.h"
#include "npy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s S3GEN.gguf WAV_16K.npy LOG_MEL.npy TOKENS.npy\n", argv[0]);
        return 1;
    }
    const std::string gguf = argv[1];
    const std::string wavp = argv[2];
    const std::string melp = argv[3];
    const std::string tokp = argv[4];

    fprintf(stderr, "[1/4] loading S3TokenizerV2 weights from %s\n", gguf.c_str());
    s3tokv2_weights w;
    if (!s3tokv2_load(gguf, w)) return 1;
    fprintf(stderr, "      n_mels=%d n_state=%d n_head=%d n_layer=%d codebook=%d\n",
            w.n_mels, w.n_state, w.n_head, w.n_layer, w.codebook_size);

    fprintf(stderr, "[2/4] loading wav %s\n", wavp.c_str());
    npy_array wav_npy = npy_load(wavp);
    std::vector<float> wav((const float*)wav_npy.data.data(),
                           (const float*)wav_npy.data.data() + wav_npy.n_elements());
    fprintf(stderr, "      %zu samples (%.2f s)\n", wav.size(), (double)wav.size() / 16000.0);

    // ---- log-mel parity ----
    fprintf(stderr, "[3/4] comparing log-mel\n");
    int T_mel = 0;
    std::vector<float> mel_cpp = s3tokv2_log_mel(wav, w, T_mel);
    npy_array mel_ref = npy_load(melp);
    fprintf(stderr, "      C++ log_mel shape=(%d, %d)  Python=(%lld, %lld)\n",
            w.n_mels, T_mel, (long long)mel_ref.shape[0], (long long)mel_ref.shape[1]);
    size_t n_mel = std::min(mel_cpp.size(), mel_ref.n_elements());
    const float * r = (const float *)mel_ref.data.data();
    float ma = 0.0f, mref = 0.0f;
    for (size_t i = 0; i < n_mel; ++i) {
        float d = mel_cpp[i] - r[i];
        if (std::fabs(d) > ma) ma = std::fabs(d);
        if (std::fabs(r[i]) > mref) mref = std::fabs(r[i]);
    }
    fprintf(stderr, "      log_mel: max_abs=%.4e  max|ref|=%.4e  rel=%.4e\n",
            (double)ma, (double)mref, (double)(ma / std::max(mref, 1e-12f)));

    // ---- tokens ----
    fprintf(stderr, "[4/4] running tokenizer\n");
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> tokens;
    if (!s3tokv2_tokenize(wav, w, -1, tokens)) return 1;
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "      C++ tokens: n=%zu  (%.1f ms)\n", tokens.size(), ms);
    fprintf(stderr, "      first10 : ");
    for (int i = 0; i < std::min((int)tokens.size(), 10); ++i) fprintf(stderr, "%d ", tokens[i]);
    fprintf(stderr, "\n");

    npy_array ref_tok = npy_load(tokp);
    const int32_t * rt = (const int32_t *)ref_tok.data.data();
    fprintf(stderr, "      Py tokens : n=%zu  first10: ", ref_tok.n_elements());
    for (int i = 0; i < std::min((int)ref_tok.n_elements(), 10); ++i) fprintf(stderr, "%d ", rt[i]);
    fprintf(stderr, "\n");

    size_t n_cmp = std::min(tokens.size(), ref_tok.n_elements());
    int n_match = 0;
    for (size_t i = 0; i < n_cmp; ++i) if (tokens[i] == rt[i]) ++n_match;
    double acc = (double)n_match / std::max<size_t>(n_cmp, 1);
    fprintf(stderr, "\n[result] token accuracy: %d / %zu (%.2f%%)\n",
            n_match, n_cmp, 100.0 * acc);
    return 0;
}
