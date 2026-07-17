// acestep-cli: standalone harness for the ACE-Step Oobleck VAE stage (QVAC-21921).
//
// Two modes:
//   --decode     : feed a structured synthetic latent through the decoder and
//                  write the resulting 48 kHz stereo WAV. Proves real weights
//                  load + the decode graph (col2im_1d + snake) runs on CPU.
//   --roundtrip  : read a 48 kHz WAV, encode -> 64-ch latent -> decode, write the
//                  reconstruction and report correlation vs the input. The audible
//                  end-to-end VAE check.
//
// Usage:
//   acestep-cli --model vae.gguf [--decode] [--t-latent 32] [--out out.wav]
//   acestep-cli --model vae.gguf --roundtrip --in in.wav [--seconds 2.56] [--out out.wav]

#include "tts-cpp/acestep/vae.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ------------------------------------------------------------- minimal WAV I/O
static std::vector<float> wav_read(const char * path, int * frames, int * rate) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return {}; }
    char riff[4]; if (fread(riff, 1, 4, f) != 4) { fclose(f); return {}; }
    uint32_t rsz; fread(&rsz, 4, 1, f);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) || memcmp(wave, "WAVE", 4)) { fprintf(stderr, "not a WAV\n"); fclose(f); return {}; }
    uint16_t channels = 0, bits = 0; uint32_t srate = 0;
    while (!feof(f)) {
        char id[4]; if (fread(id, 1, 4, f) != 4) break;
        uint32_t sz; if (fread(&sz, 4, 1, f) != 1) break;
        if (!memcmp(id, "fmt ", 4)) {
            uint16_t fmt; fread(&fmt, 2, 1, f); fread(&channels, 2, 1, f); fread(&srate, 4, 1, f);
            uint32_t br; fread(&br, 4, 1, f); uint16_t ba; fread(&ba, 2, 1, f); fread(&bits, 2, 1, f);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            if (bits != 16) { fprintf(stderr, "only 16-bit PCM supported\n"); fclose(f); return {}; }
            int n = (int) (sz / 2);
            std::vector<int16_t> pcm(n);
            fread(pcm.data(), 2, n, f);
            int fr = n / channels;
            std::vector<float> out((size_t) fr * 2);
            for (int t = 0; t < fr; t++) {
                float l = pcm[(size_t) t * channels] / 32768.0f;
                float r = (channels >= 2) ? pcm[(size_t) t * channels + 1] / 32768.0f : l;
                out[(size_t) t * 2] = l; out[(size_t) t * 2 + 1] = r;
            }
            *frames = fr; *rate = (int) srate; fclose(f); return out;
        } else fseek(f, sz, SEEK_CUR);
    }
    fclose(f); fprintf(stderr, "no data chunk\n"); return {};
}

// interleaved stereo in, peak-normalized 16-bit stereo out
static void wav_write(const char * path, const std::vector<float> & pcm, int frames, int rate) {
    float peak = 1e-9f;
    for (int i = 0; i < frames * 2; i++) peak = std::max(peak, std::fabs(pcm[i]));
    float gain = 0.9f / peak;
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    const int ch = 2, bits = 16; uint32_t db = (uint32_t) frames * ch * (bits / 8);
    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); }; auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + db); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(ch); w32((uint32_t) rate);
    w32((uint32_t) rate * ch * (bits / 8)); w16(ch * (bits / 8)); w16(bits);
    fwrite("data", 1, 4, f); w32(db);
    for (int i = 0; i < frames * 2; i++) {
        float v = pcm[i] * gain * 32767.0f;
        if (v > 32767.0f) v = 32767.0f; if (v < -32768.0f) v = -32768.0f;
        w16((uint16_t) (int16_t) lrintf(v));
    }
    fclose(f);
    fprintf(stderr, "wrote %s: %d frames, %.2fs @ %d Hz stereo\n", path, frames, (float) frames / rate, rate);
}

static double correlation(const float * a, const float * b, int n) {
    double sa = 0, sb = 0; for (int i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
    double ma = sa / n, mb = sb / n, cov = 0, va = 0, vb = 0;
    for (int i = 0; i < n; i++) { double da = a[i] - ma, db = b[i] - mb; cov += da * db; va += da * da; vb += db * db; }
    return (va <= 0 || vb <= 0) ? 0.0 : cov / std::sqrt(va * vb);
}

static const char * arg_val(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
}
static bool arg_flag(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], key)) return true;
    return false;
}

int main(int argc, char ** argv) {
    const char * model = arg_val(argc, argv, "--model");
    if (!model) {
        fprintf(stderr,
            "usage: acestep-cli --model vae.gguf [--decode] [--t-latent 32] [--out out.wav]\n"
            "       acestep-cli --model vae.gguf --roundtrip --in in.wav [--seconds 2.56] [--out out.wav]\n");
        return 1;
    }
    const bool   roundtrip = arg_flag(argc, argv, "--roundtrip");
    const char * out_path  = arg_val(argc, argv, "--out");

    tts_cpp::acestep::VaeOptions opts;
    opts.verbose      = true;
    opts.with_encoder = roundtrip;  // decode mode does not need the encoder

    std::unique_ptr<tts_cpp::acestep::Vae> vae;
    try {
        vae = tts_cpp::acestep::Vae::load(model, opts);
    } catch (const std::exception & e) {
        fprintf(stderr, "load failed: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "[acestep-cli] backend=%s sr=%d\n", vae->backend_name().c_str(), vae->sample_rate());

    if (roundtrip) {
        const char * in_path = arg_val(argc, argv, "--in");
        if (!in_path) { fprintf(stderr, "--roundtrip needs --in in.wav\n"); return 1; }
        const float seconds = arg_val(argc, argv, "--seconds") ? (float) atof(arg_val(argc, argv, "--seconds")) : 2.56f;
        const char * outp   = out_path ? out_path : "acestep_roundtrip.wav";

        int in_fr = 0, in_rate = 0;
        std::vector<float> in_pcm = wav_read(in_path, &in_fr, &in_rate);
        if (in_pcm.empty()) return 1;
        if (in_rate != 48000) fprintf(stderr, "[acestep-cli] WARNING: VAE expects 48 kHz, input is %d Hz\n", in_rate);

        int up  = vae->upsample_factor();
        int cap = (int) (seconds * 48000.0f);
        int fr  = std::min(in_fr, cap);
        fr      = (fr / up) * up;  // multiple of 1920
        if (fr < up) { fprintf(stderr, "input too short\n"); return 1; }
        in_pcm.resize((size_t) fr * 2);
        fprintf(stderr, "[acestep-cli] roundtrip on %d frames (%.2fs)\n", fr, (float) fr / 48000.0f);

        int T_lat = 0;
        std::vector<float> latent = vae->encode(in_pcm, fr, &T_lat);
        if (latent.empty()) { fprintf(stderr, "encode failed\n"); return 1; }
        fprintf(stderr, "[acestep-cli] encoded -> latent T=%d (64ch)\n", T_lat);

        std::vector<float> rec = vae->decode(latent, T_lat);
        if (rec.empty()) { fprintf(stderr, "decode failed\n"); return 1; }
        int T_out = (int) (rec.size() / 2);

        int N = std::min(fr, T_out);
        std::vector<float> il(N), rl(N), ir(N), rr(N);
        for (int t = 0; t < N; t++) {
            il[t] = in_pcm[(size_t) t * 2];     ir[t] = in_pcm[(size_t) t * 2 + 1];
            rl[t] = rec[(size_t) t * 2];         rr[t] = rec[(size_t) t * 2 + 1];
        }
        fprintf(stderr, "[acestep-cli] reconstruction correlation  L=%.4f  R=%.4f\n",
                correlation(il.data(), rl.data(), N), correlation(ir.data(), rr.data(), N));
        wav_write(outp, rec, T_out, 48000);
    } else {
        const int    T_latent = arg_val(argc, argv, "--t-latent") ? atoi(arg_val(argc, argv, "--t-latent")) : 32;
        const char * outp     = out_path ? out_path : "acestep_decode.wav";

        // structured synthetic latent (no DiT): per-channel low-frequency sinusoids
        std::vector<float> latent((size_t) T_latent * 64);
        for (int c = 0; c < 64; ++c) {
            float freq = 0.5f + 0.15f * c, phase = 0.37f * c, amp = 3.0f * expf(-c / 48.0f);
            for (int t = 0; t < T_latent; ++t)
                latent[(size_t) t * 64 + c] = amp * sinf(2.0f * (float) M_PI * freq * ((float) t / T_latent) + phase);
        }
        std::vector<float> pcm = vae->decode(latent, T_latent);
        if (pcm.empty()) { fprintf(stderr, "decode failed\n"); return 1; }
        int T_out = (int) (pcm.size() / 2);
        fprintf(stderr, "[acestep-cli] decoded T_latent=%d -> %d frames (%.2fs)\n",
                T_latent, T_out, (float) T_out / 48000.0f);
        wav_write(outp, pcm, T_out, 48000);
    }
    return 0;
}
