// Validation harness for the C++ voice-cloning preprocessing primitives:
//   wav_load + resample_sinc + mel_extract_24k_80
// Compares the C++ 80-channel log-mel at 24 kHz against a Python-dumped
// prompt_feat.npy (produced by scripts/dump-s3gen-reference.py or any
// other path that calls chatterbox.models.s3gen.utils.mel.mel_spectrogram).
//
// Usage:
//   ./build/test-voice-features S3GEN.gguf REF.wav PROMPT_FEAT.npy

#include "voice_features.h"

#include "ggml.h"
#include "gguf.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s S3GEN.gguf REF.wav PROMPT_FEAT.npy\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_path  = argv[3];

    // Load the mel filterbank (80, 961) from the GGUF.
    fprintf(stderr, "[1/4] loading mel filterbank from %s\n", gguf_path.c_str());
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) { fprintf(stderr, "gguf_init_from_file failed\n"); return 1; }

    ggml_tensor * fb_t = ggml_get_tensor(tmp_ctx, "s3gen/mel_fb/24k_80");
    if (!fb_t) {
        fprintf(stderr, "error: s3gen/mel_fb/24k_80 not found in GGUF (re-run the converter?)\n");
        return 1;
    }
    std::vector<float> mel_fb(ggml_nelements(fb_t));
    std::memcpy(mel_fb.data(), ggml_get_data(fb_t), ggml_nbytes(fb_t));
    fprintf(stderr, "      filterbank shape=[%lld, %lld]\n",
            (long long)fb_t->ne[0], (long long)fb_t->ne[1]);

    // Load the wav and resample to 24 kHz if needed.
    fprintf(stderr, "[2/4] loading %s\n", wav_path.c_str());
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return 1;
    fprintf(stderr, "      sr=%d samples=%zu (%.2f s)\n", sr, wav.size(), (double)wav.size() / sr);

    if (sr != 24000) {
        fprintf(stderr, "[2b] resampling %d -> 24000\n", sr);
        wav = resample_sinc(wav, sr, 24000);
        fprintf(stderr, "      resampled samples=%zu (%.2f s)\n", wav.size(), (double)wav.size() / 24000.0);
    }

    // Compute the C++ prompt_feat.
    fprintf(stderr, "[3/4] computing 80-channel log-mel at 24 kHz\n");
    std::vector<float> feat = mel_extract_24k_80(wav, mel_fb);
    const int n_mels = 80;
    const int T = (int)(feat.size() / n_mels);
    fprintf(stderr, "      C++ prompt_feat shape=(%d, %d)  n=%zu\n", T, n_mels, feat.size());

    // Load the Python reference.
    fprintf(stderr, "[4/4] loading reference %s\n", ref_path.c_str());
    npy_array ref = npy_load(ref_path);
    fprintf(stderr, "      Python prompt_feat shape=(%lld, %lld)  n=%zu\n",
            (long long)ref.shape[0], (long long)ref.shape[1], ref.n_elements());

    size_t n = std::min(feat.size(), ref.n_elements());
    if (n == 0) { fprintf(stderr, "empty comparison\n"); return 1; }

    const float * r = (const float *)ref.data.data();
    float ma = 0.0f, rsum = 0.0f, max_ref = 0.0f, sum_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = feat[i] - r[i];
        float ad = std::fabs(d);
        ma = std::max(ma, ad);
        rsum += d * d;
        sum_abs += ad;
        max_ref = std::max(max_ref, std::fabs(r[i]));
    }
    const double rms = std::sqrt(rsum / (double)n);
    const double mean_abs = sum_abs / (double)n;
    const double rel = ma / std::max(max_ref, 1e-12f);
    fprintf(stderr,
        "\n[result] C++ vs Python prompt_feat:\n"
        "    n=%zu  max_abs=%.4e  mean_abs=%.4e  rms=%.4e  max|ref|=%.4e  rel=%.4e\n",
        n, (double)ma, mean_abs, rms, (double)max_ref, rel);

    gguf_free(g);
    if (tmp_ctx) ggml_free(tmp_ctx);
    return 0;
}
