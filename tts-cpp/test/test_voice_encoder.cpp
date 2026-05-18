// Validation harness for the C++ VoiceEncoder port.
//
// Usage:
//   ./build/test-voice-encoder T3.gguf REF.wav SPEAKER_EMB.npy

#include "voice_encoder.h"
#include "voice_features.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s T3.gguf REF.wav SPEAKER_EMB.npy\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_path  = argv[3];

    fprintf(stderr, "[1/3] loading VoiceEncoder from %s\n", gguf_path.c_str());
    voice_encoder_weights w;
    if (!voice_encoder_load(gguf_path, w)) return 1;
    fprintf(stderr, "      n_layers=%d hidden=%d embedding=%d partial=%d rate=%.2f\n",
            w.n_layers, w.hidden, w.embedding, w.partial_frames, w.rate);

    fprintf(stderr, "[2/3] loading %s\n", wav_path.c_str());
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return 1;
    fprintf(stderr, "      sr=%d samples=%zu (%.2f s)\n", sr, wav.size(), (double)wav.size() / sr);
    {
        double L0 = measure_lufs(wav, sr);
        normalise_lufs(wav, sr, -27.0);
        fprintf(stderr, "      loudness %.2f LUFS → -27 LUFS\n", L0);
    }
    if (sr != 16000) {
        fprintf(stderr, "      resampling %d -> 16000\n", sr);
        wav = resample_sinc(wav, sr, 16000);
    }

    fprintf(stderr, "[3/3] running VoiceEncoder\n");
    std::vector<float> emb;
    if (!voice_encoder_embed(wav, w, /*backend=*/ nullptr, emb)) return 1;
    fprintf(stderr, "      C++ speaker_emb size=%zu\n", emb.size());

    npy_array ref = npy_load(ref_path);
    fprintf(stderr, "      Python speaker_emb size=%zu\n", ref.n_elements());

    size_t n = std::min(emb.size(), ref.n_elements());
    const float * r = (const float *)ref.data.data();
    float ma = 0.0f, rsum = 0.0f, max_ref = 0.0f;
    double dot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float d = emb[i] - r[i];
        ma = std::max(ma, std::fabs(d));
        rsum += d * d;
        max_ref = std::max(max_ref, std::fabs(r[i]));
        dot += (double)emb[i] * (double)r[i];
    }
    const double rms = std::sqrt(rsum / (double)n);
    const double rel = ma / std::max(max_ref, 1e-12f);
    fprintf(stderr,
        "\n[result] C++ vs Python speaker_emb:\n"
        "    n=%zu  max_abs=%.4e  rms=%.4e  max|ref|=%.4e  rel=%.4e\n"
        "    cosine similarity = %.6f\n",
        n, (double)ma, rms, (double)max_ref, rel, dot);
    return 0;
}
