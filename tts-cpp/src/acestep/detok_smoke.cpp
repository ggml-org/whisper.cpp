// detok-smoke: load + decode harness for the ACE-Step FSQ detokenizer
// (QVAC-21921). Loads the tokenizer/detokenizer weights from the DiT GGUF,
// feeds a run of audio codes (random FSQ indices, or a fixed pattern) and
// verifies the context latents [64, T_25Hz] are finite with the expected shape
// (T_25Hz = T_5Hz * 5). This is the LM-codes -> DiT-context bridge.
//
// Usage:
//   detok-smoke --model acestep-v15-turbo.gguf [--codes 20] [--seed 1]

#include "acestep/detok_ggml.h"

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
    if (!model) { fprintf(stderr, "usage: detok-smoke --model acestep-v15-turbo.gguf [--codes 20]\n"); return 1; }
    const int      T5   = arg_val(argc, argv, "--codes") ? atoi(arg_val(argc, argv, "--codes")) : 20;
    const unsigned seed = arg_val(argc, argv, "--seed")  ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1u;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }

    DetokModel * m = detok_model_load(model, backend, /*verbose=*/true);
    if (!m) { fprintf(stderr, "detok_model_load failed\n"); ggml_backend_free(backend); return 1; }

    // Random FSQ indices in [0, 8*8*8*5*5*5).
    const int    FSQ_MAX = 8 * 8 * 8 * 5 * 5 * 5;  // 64000
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, FSQ_MAX - 1);
    std::vector<int> codes(T5);
    for (auto & c : codes) c = dist(rng);

    std::vector<float> ctx((size_t) 64 * T5 * 5);
    int                T25 = detok_model_decode(m, codes.data(), T5, ctx.data());

    int rc = 1;
    if (T25 == T5 * 5) {
        size_t nan = 0;
        double mn = 1e300, mx = -1e300, sum = 0.0;
        for (float v : ctx) {
            if (!std::isfinite(v)) { nan++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stderr, "[detok-smoke] %d codes -> [64, %d] context (%.1fs @25Hz) | min=%.3f max=%.3f mean=%.4f nan=%zu\n",
                T5, T25, T25 / 25.0f, mn, mx, sum / ctx.size(), nan);
        rc = (nan == 0) ? 0 : 1;
    } else {
        fprintf(stderr, "[detok-smoke] decode returned %d (expected %d)\n", T25, T5 * 5);
    }

    fprintf(stderr, "[detok-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    detok_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
