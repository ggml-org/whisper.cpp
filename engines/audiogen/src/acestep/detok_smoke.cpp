// detok-smoke: load + decode harness for the ACE-Step FSQ detokenizer
// Loads the tokenizer/detokenizer weights from the DiT GGUF,
// feeds a run of audio codes (random FSQ indices, or a fixed pattern) and
// verifies the context latents [64, T_25Hz] are finite with the expected shape
// (T_25Hz = T_5Hz * 5). This is the LM-codes -> DiT-context bridge.
//
// With --gpu it runs the same deterministic codes on a GPU backend instead of the
// CPU, which makes it a backend-parity harness: the codes come from --seed alone,
// so two runs differing only in --gpu isolate the backend as the single variable.
// --dump writes the context latents so the two can be compared numerically.
//
// Usage:
//   detok-smoke --model acestep-v15-turbo.gguf [--codes 20] [--seed 1]
//               [--gpu] [--threads N] [--dump context.bin] [--repeat N]

#include "acestep/backend_registry.h"
#include "acestep/detok_ggml.h"

#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
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

// Same .bin layout the engine parity dumps use: 3x int32 [ndim, d0, d1] + f32.
static bool write_dump(const char * path, const std::vector<float> & v, int d0, int d1) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[detok-smoke] cannot write %s\n", path); return false; }
    const int32_t hdr[3] = { 2, d0, d1 };
    bool ok = fwrite(hdr, sizeof(int32_t), 3, f) == 3;
    ok = ok && fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    fclose(f);
    if (ok) fprintf(stderr, "[detok-smoke] wrote %s ([%d, %d], %zu floats)\n", path, d0, d1, v.size());
    return ok;
}

int main(int argc, char ** argv) {
    const char * model = arg_val(argc, argv, "--model");
    if (!model) {
        fprintf(stderr, "usage: detok-smoke --model acestep-v15-turbo.gguf [--codes 20] [--seed 1]"
                        " [--gpu] [--threads N] [--dump context.bin] [--repeat N]\n"
                        "       [--backends-dir <dir>]  (required on builds with dlopen'd ggml backends)\n");
        return 1;
    }
    const int      T5     = arg_val(argc, argv, "--codes") ? atoi(arg_val(argc, argv, "--codes")) : 20;
    const unsigned seed   = arg_val(argc, argv, "--seed")  ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1u;
    const bool     gpu    = arg_flag(argc, argv, "--gpu");
    const char *   dump   = arg_val(argc, argv, "--dump");
    // Repeating the decode separates one-time cost (Vulkan pipeline compilation on
    // first use) from steady-state throughput; without it a single run conflates them.
    const int      repeat = arg_val(argc, argv, "--repeat") ? atoi(arg_val(argc, argv, "--repeat")) : 1;

    int nth = arg_val(argc, argv, "--threads") ? atoi(arg_val(argc, argv, "--threads"))
                                               : (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 4;

    // Go through the registry like the engine does, so this tool resolves a backend
    // on arm64 dlopen builds too -- `ggml_backend_cpu_init` is not even linked there.
    if (const char * bd = arg_val(argc, argv, "--backends-dir")) load_backends(bd);

    ggml_backend_t backend = nullptr;
    if (gpu) {
        backend = backend_gpu_init();
        if (!backend) { fprintf(stderr, "[detok-smoke] no GPU backend available\n"); return 1; }
    } else {
        backend = backend_cpu_init();
        if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
        // The engine sets this; leaving the default (4) would understate the CPU path.
        backend_set_n_threads(backend, nth);
    }
    fprintf(stderr, "[detok-smoke] backend=%s codes=%d seed=%u threads=%d\n",
            ggml_backend_name(backend), T5, seed, gpu ? 0 : nth);

    DetokModel * m = detok_model_load(model, backend, /*verbose=*/true);
    if (!m) { fprintf(stderr, "detok_model_load failed\n"); ggml_backend_free(backend); return 1; }

    // Random FSQ indices in [0, 8*8*8*5*5*5).
    const int    FSQ_MAX = 8 * 8 * 8 * 5 * 5 * 5;  // 64000
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, FSQ_MAX - 1);
    std::vector<int> codes(T5);
    for (auto & c : codes) c = dist(rng);

    std::vector<float> ctx((size_t) 64 * T5 * 5);

    int    T25 = 0;
    double ms  = 0.0;
    for (int r = 0; r < repeat; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        T25           = detok_model_decode(m, codes.data(), T5, ctx.data());
        const auto t1 = std::chrono::steady_clock::now();
        ms            = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (repeat > 1) fprintf(stderr, "[detok-smoke] pass %d/%d: %.1f ms (%.3f ms/code)\n",
                                r + 1, repeat, ms, ms / T5);
        if (T25 != T5 * 5) break;
    }

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
        fprintf(stderr, "[detok-smoke] decode %.1f ms (%.3f ms/code)\n", ms, ms / T5);
        rc = (nan == 0) ? 0 : 1;
        // Rows are 25Hz frames, each 64 channels — matches the engine's
        // 02_detok_latent dump so the same comparison scripts apply.
        if (dump && !write_dump(dump, ctx, T25, 64)) rc = 1;
    } else {
        fprintf(stderr, "[detok-smoke] decode returned %d (expected %d)\n", T25, T5 * 5);
    }

    fprintf(stderr, "[detok-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    detok_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
