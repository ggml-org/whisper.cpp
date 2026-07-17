// cond-smoke: load + single-forward harness for the ACE-Step condition encoder
// (QVAC-21921). Loads the lyric/timbre encoders + text projector from the DiT
// GGUF and packs enc_hidden from random text/lyric inputs (no timbre). Proves
// weights load + the graph runs producing finite conditioning states.
//
// Usage:
//   cond-smoke --model dit.gguf [--s-text 8] [--s-lyric 24] [--seed 1234]

#include "acestep/cond_ggml.h"

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
    if (!model) { fprintf(stderr, "usage: cond-smoke --model dit.gguf [--s-text 8] [--s-lyric 24]\n"); return 1; }
    const int      S_text  = arg_val(argc, argv, "--s-text")  ? atoi(arg_val(argc, argv, "--s-text"))  : 8;
    const int      S_lyric = arg_val(argc, argv, "--s-lyric") ? atoi(arg_val(argc, argv, "--s-lyric")) : 24;
    const unsigned seed    = arg_val(argc, argv, "--seed")    ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1234u;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }

    CondModel * m = cond_model_load(model, backend, /*verbose=*/true);
    if (!m) { fprintf(stderr, "cond_model_load failed\n"); ggml_backend_free(backend); return 1; }

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> text_hidden((size_t) 1024 * S_text);
    for (auto & v : text_hidden) v = nd(rng);
    std::vector<float> lyric_embed((size_t) 1024 * S_lyric);
    for (auto & v : lyric_embed) v = nd(rng);

    std::vector<float> enc_hidden;
    int enc_S = 0;
    if (!cond_model_forward(m, text_hidden.data(), S_text, lyric_embed.data(), S_lyric,
                            nullptr, 0, enc_hidden, &enc_S)) {
        fprintf(stderr, "cond_model_forward failed\n");
        cond_model_free(m); ggml_backend_free(backend); return 1;
    }

    const size_t expect = (size_t) 2048 * (S_lyric + S_text);
    double sum = 0, sq = 0, amax = 0; size_t nan = 0;
    for (float v : enc_hidden) {
        if (!std::isfinite(v)) { nan++; continue; }
        sum += v; sq += (double) v * v; amax = std::max(amax, (double) std::fabs(v));
    }
    fprintf(stderr,
        "[cond-smoke] forward ok: enc_hidden=%zu (expect %zu) enc_S=%d mean=%.4f rms=%.4f max=%.4f nan/inf=%zu\n",
        enc_hidden.size(), expect, enc_S, sum / enc_hidden.size(), std::sqrt(sq / enc_hidden.size()), amax, nan);

    int rc = (enc_hidden.size() == expect && enc_S == S_lyric + S_text && nan == 0) ? 0 : 1;
    fprintf(stderr, "[cond-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    cond_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
