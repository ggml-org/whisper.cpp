// dit-smoke: minimal load + single-forward harness for the ACE-Step DiT stage
// (QVAC-21921). Proves real DiT weights load onto the CPU backend and one
// flow-matching forward runs to completion producing finite velocities. This is
// a "does it run" check, not numerical parity (that lives in the reference-parity
// scripts against PyTorch / acestep.cpp).
//
// Usage:
//   dit-smoke --model dit.gguf [--s 16] [--enc-s 8] [--n 1] [--seed 1234]

#include "acestep/dit_ggml.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
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

int main(int argc, char ** argv) {
    const char * model = arg_val(argc, argv, "--model");
    if (!model) { fprintf(stderr, "usage: dit-smoke --model dit.gguf [--s 16] [--enc-s 8] [--n 1]\n"); return 1; }
    const int S     = arg_val(argc, argv, "--s")     ? atoi(arg_val(argc, argv, "--s"))     : 16;
    const int enc_S = arg_val(argc, argv, "--enc-s") ? atoi(arg_val(argc, argv, "--enc-s")) : 8;
    const int N     = arg_val(argc, argv, "--n")     ? atoi(arg_val(argc, argv, "--n"))     : 1;
    const unsigned seed = arg_val(argc, argv, "--seed") ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1234u;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }

    DitModel * m = dit_model_load(model, backend, /*verbose=*/true);
    if (!m) { fprintf(stderr, "dit_model_load failed\n"); ggml_backend_free(backend); return 1; }

    const DitConfig & c = dit_model_config(m);
    const int T     = S * c.patch_size;
    const int H_enc = c.enc_hidden_size;
    fprintf(stderr,
        "[dit-smoke] loaded: H=%d inter=%d heads=%d kv=%d hd=%d layers=%d in=%d out=%d patch=%d "
        "sw=%d theta=%.1f eps=%.1e H_enc=%d  (%.1f MiB weights)\n",
        c.hidden_size, c.intermediate_size, c.n_heads, c.n_kv_heads, c.head_dim, c.n_layers,
        c.in_channels, c.out_channels, c.patch_size, c.sliding_window, c.rope_theta, c.rms_norm_eps,
        H_enc, dit_model_weight_bytes(m) / 1048576.0);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> latents((size_t) c.in_channels * T * N);
    for (auto & v : latents) v = nd(rng);
    std::vector<float> enc((size_t) H_enc * enc_S * N);
    for (auto & v : enc) v = 0.1f * nd(rng);

    DitForwardInputs in;
    in.input_latents = latents.data();
    in.T             = T;
    in.N             = N;
    in.enc_hidden    = enc.data();
    in.enc_S         = enc_S;
    in.H_enc         = H_enc;
    in.t             = 1.0f;
    in.t_r           = 1.0f;

    // --sample: run the full turbo flow-matching loop (8 Euler steps, no CFG).
    if (arg_val(argc, argv, "--sample") || (argc > 1 && !strcmp(argv[argc - 1], "--sample"))) {
        const int   steps = arg_val(argc, argv, "--steps") ? atoi(arg_val(argc, argv, "--steps")) : 8;
        const float shift = arg_val(argc, argv, "--shift") ? (float) atof(arg_val(argc, argv, "--shift")) : 3.0f;
        const int   ctx_ch = c.in_channels - c.out_channels;

        std::vector<float> noise((size_t) c.out_channels * T * N);
        for (auto & v : noise) v = nd(rng);
        std::vector<float> context((size_t) ctx_ch * T * N, 0.0f);  // silence conditioning
        std::vector<float> schedule;
        dit_build_schedule(shift, steps, schedule);
        fprintf(stderr, "[dit-smoke] schedule (shift=%.1f, %d steps):", shift, steps);
        for (float s : schedule) fprintf(stderr, " %.3f", s);
        fprintf(stderr, "\n");

        DitSampleParams sp;
        sp.noise = noise.data(); sp.context_latents = context.data();
        sp.enc_hidden = enc.data(); sp.enc_S = enc_S; sp.H_enc = H_enc;
        sp.T = T; sp.N = N; sp.schedule = schedule.data(); sp.num_steps = steps;

        std::vector<float> latent;
        if (!dit_sample(m, sp, latent)) {
            fprintf(stderr, "dit_sample failed\n");
            dit_model_free(m); ggml_backend_free(backend); return 1;
        }
        double s2 = 0, mx = 0; size_t nn = 0;
        for (float v : latent) { if (!std::isfinite(v)) { nn++; continue; } s2 += (double) v * v; mx = std::max(mx, (double) std::fabs(v)); }
        const size_t exp2 = (size_t) c.out_channels * T * N;
        fprintf(stderr, "[dit-smoke] sample ok: latent=%zu (expect %zu) rms=%.4f max=%.4f nan/inf=%zu\n",
                latent.size(), exp2, std::sqrt(s2 / latent.size()), mx, nn);
        int rc = (latent.size() == exp2 && nn == 0) ? 0 : 1;
        fprintf(stderr, "[dit-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
        dit_model_free(m); ggml_backend_free(backend);
        return rc;
    }

    std::vector<float> vel;
    if (!dit_model_forward(m, in, vel)) {
        fprintf(stderr, "dit_model_forward failed\n");
        dit_model_free(m); ggml_backend_free(backend); return 1;
    }

    const size_t expect = (size_t) c.out_channels * T * N;
    double sum = 0, sq = 0, amax = 0; size_t nan = 0;
    for (float v : vel) {
        if (!std::isfinite(v)) { nan++; continue; }
        sum += v; sq += (double) v * v; amax = std::max(amax, (double) std::fabs(v));
    }
    const double mean = sum / vel.size();
    const double rms  = std::sqrt(sq / vel.size());
    fprintf(stderr,
        "[dit-smoke] forward ok: velocity=%zu (expect %zu)  mean=%.4f rms=%.4f max=%.4f  nan/inf=%zu\n",
        vel.size(), expect, mean, rms, amax, nan);

    int rc = (vel.size() == expect && nan == 0) ? 0 : 1;
    fprintf(stderr, "[dit-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    dit_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
