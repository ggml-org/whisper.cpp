// cond-smoke: load + single-forward harness for the ACE-Step condition encoder
// Loads the lyric/timbre encoders + text projector from the DiT GGUF and packs
// enc_hidden. Proves weights load + the graph runs producing finite conditioning
// states.
//
// Inputs are random by default. Random N(0,1) exercises the graph but not the
// activation range a real caption+lyrics produces, so --lyric-bin/--text-bin
// replay recorded --dump-stages tensors instead: that is the input class where
// the encoder has actually been seen to break.
//
// Usage:
//   cond-smoke --model dit.gguf [--s-text 8] [--s-lyric 24] [--seed 1234]
//              [--lyric-bin 05_lyric_embed.bin] [--text-bin 04_text_hidden.bin]
//              [--gpu] [--backends-dir dir] [--threads N] [--dump enc.bin]

#include "acestep/backend_registry.h"
#include "acestep/cond_ggml.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace tts_cpp::acestep;

static const char * arg_val(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
}
static bool arg_flag(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], key)) return true;
    return false;
}

// --dump-stages format: int32 hdr[3] = {ndim, d0, d1} then d0*d1 float32.
static bool read_stage_dump(const char * path, std::vector<float> & out, int * d0, int * d1) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[cond-smoke] cannot open %s\n", path); return false; }
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, f) != 3 || hdr[0] != 2 || hdr[1] <= 0 || hdr[2] <= 0) {
        fprintf(stderr, "[cond-smoke] %s is not a rank-2 stage dump\n", path);
        fclose(f);
        return false;
    }
    const size_t n = (size_t) hdr[1] * hdr[2];
    out.resize(n);
    const bool ok = fread(out.data(), sizeof(float), n, f) == n;
    fclose(f);
    if (!ok) { fprintf(stderr, "[cond-smoke] %s is truncated\n", path); return false; }
    *d0 = hdr[1];
    *d1 = hdr[2];
    return true;
}

static bool write_stage_dump(const char * path, const std::vector<float> & v, int d0, int d1) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[cond-smoke] cannot write %s\n", path); return false; }
    const int32_t hdr[3] = { 2, d0, d1 };
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    return true;
}

// Exponent all ones. std::isfinite can be compiled away under fast-math, so the
// bit pattern is the only reading that survives every build of every backend.
static bool bits_non_finite(float x) {
    uint32_t b;
    std::memcpy(&b, &x, sizeof(b));
    return (b & 0x7F800000u) == 0x7F800000u;
}

// A row wiped by a normalisation that saw an infinity: every element exactly
// zero except the one that held it. Distinct from a merely quiet row, and the
// signature the Adreno OpenCL encoder produced on a real 180-token prompt.
static int count_wiped_rows(const std::vector<float> & v, int rows, int cols) {
    int wiped = 0;
    for (int r = 0; r < rows; r++) {
        int zeros = 0, non_finite = 0;
        for (int c = 0; c < cols; c++) {
            const float x = v[(size_t) r * cols + c];
            if (bits_non_finite(x)) non_finite++;
            else if (x == 0.0f) zeros++;
        }
        if (non_finite > 0 && zeros == cols - non_finite) wiped++;
    }
    return wiped;
}

static void fill_random(std::vector<float> & v, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & x : v) x = nd(rng);
}

int main(int argc, char ** argv) {
    const char * model = arg_val(argc, argv, "--model");
    if (!model) {
        fprintf(stderr,
            "usage: cond-smoke --model dit.gguf [--s-text 8] [--s-lyric 24] [--seed 1234]\n"
            "       [--lyric-bin 05_lyric_embed.bin] [--text-bin 04_text_hidden.bin]\n"
            "       [--gpu] [--threads N] [--dump enc.bin]\n"
            "       [--backends-dir <dir>]  (required on builds with dlopen'd ggml backends)\n");
        return 1;
    }
    const unsigned seed = arg_val(argc, argv, "--seed") ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1234u;
    const int      nth  = arg_val(argc, argv, "--threads") ? atoi(arg_val(argc, argv, "--threads")) : 4;
    const bool     gpu  = arg_flag(argc, argv, "--gpu");

    if (const char * bd = arg_val(argc, argv, "--backends-dir")) load_backends(bd);

    ggml_backend_t backend = nullptr;
    if (gpu) {
        backend = backend_gpu_init();
        if (!backend) { fprintf(stderr, "[cond-smoke] no GPU backend available\n"); return 1; }
    } else {
        backend = backend_cpu_init();
        if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
        backend_set_n_threads(backend, nth);
    }

    std::mt19937 rng(seed);

    std::vector<float> text_hidden, lyric_embed;
    int                S_text = 0, S_lyric = 0, width = 0;

    if (const char * p = arg_val(argc, argv, "--lyric-bin")) {
        if (!read_stage_dump(p, lyric_embed, &S_lyric, &width)) { ggml_backend_free(backend); return 1; }
        if (width != 1024) {
            fprintf(stderr, "[cond-smoke] %s has width %d, expected 1024\n", p, width);
            ggml_backend_free(backend);
            return 1;
        }
    } else {
        S_lyric = arg_val(argc, argv, "--s-lyric") ? atoi(arg_val(argc, argv, "--s-lyric")) : 24;
        lyric_embed.resize((size_t) 1024 * S_lyric);
        fill_random(lyric_embed, rng);
    }

    if (const char * p = arg_val(argc, argv, "--text-bin")) {
        if (!read_stage_dump(p, text_hidden, &S_text, &width)) { ggml_backend_free(backend); return 1; }
        if (width != 1024) {
            fprintf(stderr, "[cond-smoke] %s has width %d, expected 1024\n", p, width);
            ggml_backend_free(backend);
            return 1;
        }
    } else {
        S_text = arg_val(argc, argv, "--s-text") ? atoi(arg_val(argc, argv, "--s-text")) : 8;
        text_hidden.resize((size_t) 1024 * S_text);
        fill_random(text_hidden, rng);
    }

    CondModel * m = cond_model_load(model, backend, /*verbose=*/true);
    if (!m) { fprintf(stderr, "cond_model_load failed\n"); ggml_backend_free(backend); return 1; }

    // text2music feeds one frame of the silence latent to the timbre encoder, so
    // taking it here keeps the packed layout identical to what the engine builds.
    const std::vector<float> & silence = cond_model_silence_frame(m);
    const float *              timbre  = silence.empty() ? nullptr : silence.data();
    const int                  S_ref   = silence.empty() ? 0 : 1;

    fprintf(stderr, "[cond-smoke] backend=%s S_lyric=%d S_text=%d timbre=%d\n",
            ggml_backend_name(backend), S_lyric, S_text, S_ref);

    std::vector<float> enc_hidden;
    int                enc_S = 0;
    if (!cond_model_forward(m, text_hidden.data(), S_text, lyric_embed.data(), S_lyric,
                            timbre, S_ref, enc_hidden, &enc_S)) {
        fprintf(stderr, "cond_model_forward failed\n");
        cond_model_free(m); ggml_backend_free(backend); return 1;
    }

    const size_t expect = (size_t) 2048 * (S_lyric + S_ref + S_text);
    double sum = 0, sq = 0, amax = 0; size_t nan = 0;
    for (float v : enc_hidden) {
        if (bits_non_finite(v)) { nan++; continue; }
        sum += v; sq += (double) v * v; amax = std::max(amax, (double) std::fabs(v));
    }
    const int wiped = count_wiped_rows(enc_hidden, enc_S, 2048);

    fprintf(stderr,
        "[cond-smoke] forward ok: enc_hidden=%zu (expect %zu) enc_S=%d mean=%.4f rms=%.4f max=%.4f "
        "nan/inf=%zu wiped_rows=%d\n",
        enc_hidden.size(), expect, enc_S, sum / enc_hidden.size(), std::sqrt(sq / enc_hidden.size()),
        amax, nan, wiped);

    if (const char * p = arg_val(argc, argv, "--dump")) {
        if (write_stage_dump(p, enc_hidden, enc_S, 2048))
            fprintf(stderr, "[cond-smoke] wrote %s [%d, 2048]\n", p, enc_S);
    }

    int rc = (enc_hidden.size() == expect && enc_S == S_lyric + S_ref + S_text && nan == 0 && wiped == 0) ? 0 : 1;
    fprintf(stderr, "[cond-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    cond_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
