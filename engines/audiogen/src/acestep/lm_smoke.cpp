// lm-smoke: load + prefill + greedy-decode harness for the ACE-Step LM core
// (ace-lm Qwen3 0.6B). Proves real weights load, the KV cache
// works across prefill + N decode steps, and logits stay finite. Not a
// tokenizer/generation-quality test (BPE + FSM + sampling live in the pipeline).
//
// With --gpu it runs the same deterministic prompt on a GPU backend instead of
// the CPU, which makes it a backend-parity harness: the prompt is derived from
// --seed alone, so two runs that differ only in --gpu isolate the backend as the
// single variable. --dump writes every step's logits using the same header layout as
// the engine's stage dumps, so dropping each run's file into its own directory lets
// scripts/stage_cos.py report the cosine between the two.
//
// Usage:
//   lm-smoke --model ace-lm.gguf [--prefill 8] [--decode 8] [--seed 1234]
//            [--gpu] [--threads N] [--kv-sets 1]
//            [--dump logits.bin] [--dump-layers layers.bin]
//            [--quantized-batch-cfg-regression]

#include "acestep/backend_registry.h"
#include "acestep/bpe_tokenizer.h"
#include "acestep/lm_ggml.h"

#include "ggml-backend.h"

#include <algorithm>
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
    if (!f) { fprintf(stderr, "[lm-smoke] cannot write %s\n", path); return false; }
    const int32_t hdr[3] = { 2, d0, d1 };
    bool ok = fwrite(hdr, sizeof(int32_t), 3, f) == 3;
    ok = ok && fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    fclose(f);
    if (ok) fprintf(stderr, "[lm-smoke] wrote %s ([%d, %d], %zu floats)\n", path, d0, d1, v.size());
    return ok;
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
    if (!model) {
        fprintf(stderr, "usage: lm-smoke --model ace-lm.gguf [--prefill 8] [--decode 8]\n"
                        "       [--backends-dir <dir>] [--quantized-batch-cfg-regression]\n"
                        "       (--backends-dir is required on builds with dlopen'd ggml backends)\n");
        return 1;
    }
    const int      P    = arg_val(argc, argv, "--prefill") ? atoi(arg_val(argc, argv, "--prefill")) : 8;
    const int      Dn   = arg_val(argc, argv, "--decode")  ? atoi(arg_val(argc, argv, "--decode"))  : 8;
    const unsigned seed = arg_val(argc, argv, "--seed")    ? (unsigned) atoi(arg_val(argc, argv, "--seed")) : 1234u;
    const bool     gpu  = arg_flag(argc, argv, "--gpu");
    const bool     batch_regression = arg_flag(argc, argv, "--quantized-batch-cfg-regression");
    const char *   dump = arg_val(argc, argv, "--dump");
    const char *   dumpl = arg_val(argc, argv, "--dump-layers");
    const int requested_sets = arg_val(argc, argv, "--kv-sets") ? atoi(arg_val(argc, argv, "--kv-sets")) : 1;
    const int sets = batch_regression ? std::max(requested_sets, 4) : requested_sets;

    int nth = arg_val(argc, argv, "--threads") ? atoi(arg_val(argc, argv, "--threads"))
                                               : (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 4;

    // Go through the registry like the engine does, so this tool resolves a backend
    // on arm64 dlopen builds too -- `ggml_backend_cpu_init` is not even linked there.
    if (const char * bd = arg_val(argc, argv, "--backends-dir")) load_backends(bd);

    ggml_backend_t backend = nullptr;
    if (gpu) {
        backend = backend_gpu_init();
        if (!backend) { fprintf(stderr, "[lm-smoke] no GPU backend available\n"); return 1; }
    } else {
        backend = backend_cpu_init();
        if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
        // The engine sets this; leaving the default (4) would understate the CPU path
        // and make any CPU-vs-GPU timing meaningless.
        backend_set_n_threads(backend, nth);
    }
    fprintf(stderr, "[lm-smoke] backend=%s prefill=%d decode=%d seed=%u threads=%d\n",
            ggml_backend_name(backend), P, Dn, seed, gpu ? 0 : nth);

    LMModel * m = lm_model_load(model, backend, /*max_seq_len=*/2048, /*verbose=*/true, /*n_kv_sets=*/sets);
    if (!m) { fprintf(stderr, "lm_model_load failed\n"); ggml_backend_free(backend); return 1; }

    const LMConfig & c = lm_model_config(m);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(1, c.vocab_size > 32000 ? 32000 : c.vocab_size - 1);

    // Prefill with P dummy tokens.
    std::vector<int32_t> prompt(P);
    for (auto & v : prompt) v = tok(rng);

    if (batch_regression) {
        if (!gpu || P < 1 || !lm_model_supports_batched_decode(m) || !lm_model_embeddings_quantized(m) ||
            TOKEN_IM_END >= c.vocab_size) {
            fprintf(stderr,
                    "[lm-smoke] quantized batched-CFG regression requires a quantized LM on a GPU with batched decode\n");
            lm_model_free(m);
            ggml_backend_free(backend);
            return 1;
        }

        std::vector<float> prefill_logits;
        for (int set = 0; set < 4; ++set) {
            if (!lm_model_forward(m, prompt.data(), P, prefill_logits, set)) {
                fprintf(stderr, "[lm-smoke] regression prefill failed for set %d\n", set);
                lm_model_free(m);
                ggml_backend_free(backend);
                return 1;
            }
        }

        const int32_t next_token = prompt.back();
        const int32_t batch_ids[2] = { next_token, next_token };
        const int batch_sets[2] = { 0, 1 };
        std::vector<float> compact;
        std::vector<float> reference[2];
        bool ok = lm_model_forward_batch(m, batch_ids, batch_sets, 2, compact, TOKEN_IM_END);
        ok = ok && lm_model_forward(m, &next_token, 1, reference[0], 2);
        ok = ok && lm_model_forward(m, &next_token, 1, reference[1], 3);

        const int out_v = c.vocab_size - TOKEN_IM_END;
        if (!ok || compact.size() != (size_t) 2 * out_v) {
            fprintf(stderr, "[lm-smoke] quantized batched-CFG forward failed\n");
            lm_model_free(m);
            ggml_backend_free(backend);
            return 1;
        }

        bool pass = true;
        for (int seq = 0; seq < 2; ++seq) {
            const float * got = compact.data() + (size_t) seq * out_v;
            const float * ref = reference[seq].data() + TOKEN_IM_END;
            double dot = 0.0, got_norm = 0.0, ref_norm = 0.0, max_abs = 0.0;
            int got_argmax = 0, ref_argmax = 0;
            for (int i = 0; i < out_v; ++i) {
                if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
                    pass = false;
                    continue;
                }
                dot += (double) got[i] * ref[i];
                got_norm += (double) got[i] * got[i];
                ref_norm += (double) ref[i] * ref[i];
                max_abs = std::max(max_abs, std::fabs((double) got[i] - ref[i]));
                if (got[i] > got[got_argmax]) got_argmax = i;
                if (ref[i] > ref[ref_argmax]) ref_argmax = i;
            }
            const double cosine = dot / std::sqrt(got_norm * ref_norm);
            fprintf(stderr,
                    "[lm-smoke] quantized batch seq=%d cosine=%.9f max_abs=%.6g argmax=%d/%d\n",
                    seq, cosine, max_abs, got_argmax + TOKEN_IM_END, ref_argmax + TOKEN_IM_END);
            pass = pass && cosine >= 0.99999 && got_argmax == ref_argmax;
        }

        fprintf(stderr, "[lm-smoke] quantized batched-CFG regression %s\n", pass ? "PASS" : "FAIL");
        lm_model_free(m);
        ggml_backend_free(backend);
        return pass ? 0 : 1;
    }

    std::vector<float> logits;
    std::vector<float> all_logits;   // [n_steps, vocab], only filled when --dump is set
    std::vector<int>   traj;         // argmax per step, cheap parity signal
    std::vector<float> layers;
    if (!lm_model_forward(m, prompt.data(), P, logits, 0, dumpl ? &layers : nullptr)) {
        fprintf(stderr, "prefill failed\n"); lm_model_free(m); ggml_backend_free(backend); return 1;
    }
    if (dumpl) {
        const int n_rows = (int) (layers.size() / ((size_t) c.hidden_size * P));
        write_dump(dumpl, layers, n_rows, c.hidden_size * P);
    }
    if (dump) all_logits.insert(all_logits.end(), logits.begin(), logits.end());
    double mv; size_t nan;
    int    next = argmax(logits, &mv, &nan);
    traj.push_back(next);
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
        if (dump) all_logits.insert(all_logits.end(), logits.begin(), logits.end());
        next = argmax(logits, &mv, &nan);
        traj.push_back(next);
        total_nan += nan;
        fprintf(stderr, "[lm-smoke]   step %d: argmax=%d max=%.3f nan=%zu kv_pos=%d\n", s, next, mv, nan, lm_kv_pos(m));
    }

    fprintf(stderr, "[lm-smoke] argmax trajectory:");
    for (int t : traj) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    int rc = (total_nan == 0 && (int) logits.size() == c.vocab_size) ? 0 : 1;
    if (dump && !write_dump(dump, all_logits, (int) traj.size(), c.vocab_size)) rc = 1;
    fprintf(stderr, "[lm-smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    lm_model_free(m);
    ggml_backend_free(backend);
    return rc;
}
