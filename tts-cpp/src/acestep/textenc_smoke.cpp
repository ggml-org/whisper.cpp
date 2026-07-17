// textenc-smoke: load + single-forward harness for the ACE-Step text encoder
// (Qwen3-Embedding-0.6B, QVAC-21921). Proves real weights load onto the CPU
// backend and one forward runs to completion producing finite hidden states.
// Not a tokenizer test (BPE lives with the LM port) — feeds dummy token IDs.
//
// Usage:
//   textenc-smoke --model qwen3-embedding.gguf [--s 16] [--seed 1234]

#include "acestep/textenc_ggml.h"

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
    if (!model) { fprintf(stderr, "usage: textenc-smoke --model qwen3-embedding.gguf [--s 16]\n"); return 1; }
    const int      S    = arg_val(argc, argv, "--s") ? atoi(arg_val(argc, argv, "--s")) : 16;
    const unsigned seed = arg_val(argc, argv, "--seed") ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1234u;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }

    TextEncModel * m = textenc_model_load(model, backend, /*verbose=*/true);
    if (!m) { fprintf(stderr, "textenc_model_load failed\n"); ggml_backend_free(backend); return 1; }

    const TextEncConfig & c = textenc_model_config(m);

    // Dummy token IDs in a safe range (small positive ints).
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(1, 10000);
    std::vector<int32_t> ids(S);
    for (auto & v : ids) v = tok(rng);

    std::vector<float> hidden;
    if (!textenc_model_forward(m, ids.data(), S, hidden)) {
        fprintf(stderr, "textenc_model_forward failed\n");
        textenc_model_free(m); ggml_backend_free(backend); return 1;
    }

    const size_t expect = (size_t) c.hidden_size * S;
    double sum = 0, sq = 0, amax = 0; size_t nan = 0;
    for (float v : hidden) {
        if (!std::isfinite(v)) { nan++; continue; }
        sum += v; sq += (double) v * v; amax = std::max(amax, (double) std::fabs(v));
    }
    fprintf(stderr,
        "[textenc-smoke] forward ok: hidden=%zu (expect %zu) mean=%.4f rms=%.4f max=%.4f nan/inf=%zu\n",
        hidden.size(), expect, sum / hidden.size(), std::sqrt(sq / hidden.size()), amax, nan);

    int rc = (hidden.size() == expect && nan == 0) ? 0 : 1;
    fprintf(stderr, "[textenc-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    textenc_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
