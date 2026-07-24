// cosyvoice-hift: CosyVoice3 CausalHiFTGenerator vocoder CLI (mel -> 24 kHz wav).
//
// The vocoder graph (f0_predictor + SineGen2 excitation + STFT + decode) now
// lives in cosyvoice_pipeline.cpp (shared with the Engine); this CLI just loads
// a mel .npy, runs cosyvoice_hift_synth(), and writes a WAV.
//
// Validated against the PyTorch reference: fed
// hift_mel_in.npy, the output matches hift_wav.npy at 0.989 log-mel corr /
// 0.999 energy-envelope corr.
//
// Consumes the GGUF from scripts/convert-cosyvoice3-hift-to-gguf.py (hift/*).
//
// Usage:
//   cosyvoice-hift --hift-gguf MODEL.gguf --mel-npy MEL.npy --out OUT.wav [--seed N]

#include "npy.h"

#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Write 24 kHz 16-bit PCM WAV
static void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t num_samples = (uint32_t)wav.size();
    uint32_t byte_rate = sr * 2;
    uint32_t data_size = num_samples * 2;
    uint32_t chunk_size = 36 + data_size;
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_fmt = 1;
    uint16_t n_channels = 1;
    uint32_t sample_rate = (uint32_t)sr;
    uint32_t br = byte_rate;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;
    std::fwrite(&fmt_chunk_size, 4, 1, f);
    std::fwrite(&audio_fmt, 2, 1, f);
    std::fwrite(&n_channels, 2, 1, f);
    std::fwrite(&sample_rate, 4, 1, f);
    std::fwrite(&br, 4, 1, f);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bits_per_sample, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float cl = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t)std::lrintf(cl * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

int main(int argc, char ** argv) {
    std::string gguf_path, mel_path, out_path;
    int seed = 42;
    int sampling_rate = 24000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--hift-gguf" || a == "--s3gen-gguf") && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--mel-npy" && i + 1 < argc) mel_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--sr" && i + 1 < argc) sampling_rate = std::atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s --hift-gguf MODEL.gguf --mel-npy MEL.npy --out OUT.wav [--seed N] [--sr 24000]\n", argv[0]);
            return 1;
        }
    }
    if (gguf_path.empty() || mel_path.empty() || out_path.empty()) {
        fprintf(stderr, "missing required arguments\n");
        return 1;
    }

    fprintf(stderr, "Loading %s\n", gguf_path.c_str());
    model_ctx m = cosyvoice_load_gguf(gguf_path);
    fprintf(stderr, "  %zu tensors loaded\n", m.tensors.size());

    // Load mel: npy shape (80, T) -> channel-major flat[ch*T + t].
    npy_array mel_npy = npy_load(mel_path);
    int T_mel = (int)mel_npy.shape[1];
    std::vector<float> mel((size_t)T_mel * 80);
    std::memcpy(mel.data(), npy_as_f32(mel_npy), mel.size() * sizeof(float));
    fprintf(stderr, "Mel shape: (%lld, %lld)\n", (long long)mel_npy.shape[0], (long long)mel_npy.shape[1]);

    fprintf(stderr, "Running HiFT synth (seed=%d)...\n", seed);
    auto wav = cosyvoice_hift_synth(m, mel, T_mel, seed);
    fprintf(stderr, "  wav shape: (%zu,)\n", wav.size());

    write_wav(out_path, wav, sampling_rate);
    fprintf(stderr, "Wrote %s\n", out_path.c_str());
    return 0;
}
