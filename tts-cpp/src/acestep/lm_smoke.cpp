// lm-smoke: load + prefill + greedy-decode harness for the ACE-Step LM core
// (ace-lm Qwen3 0.6B, QVAC-21921). Proves real weights load, the KV cache
// works across prefill + N decode steps, and logits stay finite. Not a
// tokenizer/generation-quality test (BPE + FSM + sampling live in the pipeline).
//
// Usage:
//   lm-smoke --model ace-lm.gguf [--prefill 8] [--decode 8] [--seed 1234]

#include "acestep/lm_ggml.h"

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

static int argmax(const std::vector<float> & v, double * maxval, size_t * nan) {
    int    best = 0;
    double bv   = -1e300;
    *nan        = 0;
    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isfinite(v[i])) { (*nan)++; continue; }
        if (v[i] > bv) { bv = v[i]; best = (int) i; }
    }
    *maxval = bv;
    return best;
}

int main(int argc, char ** argv) {
    const char * model = arg_val(argc, argv, "--model");
    if (!model) { fprintf(stderr, "usage: lm-smoke --model ace-lm.gguf [--prefill 8] [--decode 8]\n"); return 1; }
    const int      P    = arg_val(argc, argv, "--prefill") ? atoi(arg_val(argc, argv, "--prefill")) : 8;
    const int      Dn   = arg_val(argc, argv, "--decode")  ? atoi(arg_val(argc, argv, "--decode"))  : 8;
    const unsigned seed = arg_val(argc, argv, "--seed")    ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1234u;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }

    LMModel * m = lm_model_load(model, backend, /*max_seq_len=*/2048, /*verbose=*/true);
    if (!m) { fprintf(stderr, "lm_model_load failed\n"); ggml_backend_free(backend); return 1; }

    const LMConfig & c = lm_model_config(m);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(1, c.vocab_size > 32000 ? 32000 : c.vocab_size - 1);

    // Prefill with P dummy tokens.
    std::vector<int32_t> prompt(P);
    for (auto & v : prompt) v = tok(rng);
    std::vector<float> logits;
    if (!lm_model_forward(m, prompt.data(), P, logits)) {
        fprintf(stderr, "prefill failed\n"); lm_model_free(m); ggml_backend_free(backend); return 1;
    }
    double mv; size_t nan;
    int    next = argmax(logits, &mv, &nan);
    fprintf(stderr, "[lm-smoke] prefill %d tok -> logits=%zu argmax=%d max=%.3f nan/inf=%zu kv_pos=%d\n",
            P, logits.size(), next, mv, nan, lm_kv_pos(m));
    if (nan) { fprintf(stderr, "[lm-smoke] FAIL (nan in prefill logits)\n"); lm_model_free(m); ggml_backend_free(backend); return 1; }

    // Greedy decode Dn steps, feeding argmax back in.
    size_t total_nan = 0;
    for (int s = 0; s < Dn; s++) {
        int32_t t = (int32_t) next;
        if (!lm_model_forward(m, &t, 1, logits)) {
            fprintf(stderr, "decode step %d failed\n", s); lm_model_free(m); ggml_backend_free(backend); return 1;
        }
        next = argmax(logits, &mv, &nan);
        total_nan += nan;
        fprintf(stderr, "[lm-smoke]   step %d: argmax=%d max=%.3f nan=%zu kv_pos=%d\n", s, next, mv, nan, lm_kv_pos(m));
    }

    int rc = (total_nan == 0 && (int) logits.size() == c.vocab_size) ? 0 : 1;
    fprintf(stderr, "[lm-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    lm_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
