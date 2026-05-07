// Parity test: C++ T3-MTL (Llama-520M) vs PyTorch reference dumps.
//
// Loads models/chatterbox-t3-mtl.gguf, runs one prompt-path forward pass
// (both cond and uncond to exercise CFG), and compares:
//   - C++ cond logits at last position vs speech_logits_call0.npy[0, -1, :]
//   - C++ uncond logits at last position vs speech_logits_call0.npy[1, -1, :]
//
// The reference is produced by `python scripts/dump-t3-mtl-reference.py`.

#include "chatterbox_t3_internal.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "npy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Pull the implementation-detail symbols (`load_model_gguf`,
// `chatterbox_model`, `eval_*_mtl`, sampling-params struct, etc.) into the
// global scope so the test code can stay compact.  Fine for a test
// executable: any rename inside the detail namespace would only break this
// test's build, not the public Engine API.
using namespace tts_cpp::chatterbox::detail;

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s MODEL.gguf REF_DIR [--threads N] [--n-gpu-layers N]\n"
            "  MODEL.gguf   produced by scripts/convert-t3-mtl-to-gguf.py\n"
            "  REF_DIR      produced by scripts/dump-t3-mtl-reference.py (contains\n"
            "               text_tokens.npy, speech_logits_call0.npy, ...)\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string ref_dir    = argv[2];

    int n_threads = (int) std::thread::hardware_concurrency();
    int n_gpu_layers = 0;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc)      n_threads    = atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) n_gpu_layers = atoi(argv[++i]);
    }
    if (n_threads <= 0) n_threads = 4;

    fprintf(stderr, "test-t3-mtl: model=%s  ref=%s  threads=%d  gpu_layers=%d\n",
            model_path.c_str(), ref_dir.c_str(), n_threads, n_gpu_layers);

    chatterbox_model model;
    if (!load_model_gguf(model_path, model, /*requested_ctx=*/0, n_gpu_layers)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    if (model.hparams.variant != CHBX_VARIANT_MTL) {
        fprintf(stderr, "model is not t3_mtl variant\n");
        return 1;
    }

    npy_array text_tokens_npy = npy_load(ref_dir + "/text_tokens.npy");
    if (text_tokens_npy.dtype != "<i4") {
        fprintf(stderr, "expected text_tokens int32, got %s\n", text_tokens_npy.dtype.c_str());
        return 1;
    }
    std::vector<int32_t> text_tokens(text_tokens_npy.n_elements());
    std::memcpy(text_tokens.data(), text_tokens_npy.data.data(),
                text_tokens.size() * sizeof(int32_t));
    fprintf(stderr, "text_tokens (n=%zu):", text_tokens.size());
    for (size_t i = 0; i < text_tokens.size(); ++i) fprintf(stderr, " %d", text_tokens[i]);
    fprintf(stderr, "\n");

    npy_array logits_ref = npy_load(ref_dir + "/speech_logits_call0.npy");
    if (logits_ref.dtype != "<f4") {
        fprintf(stderr, "expected speech_logits_call0 float32, got %s\n", logits_ref.dtype.c_str());
        return 1;
    }
    if (logits_ref.shape.size() != 3 || logits_ref.shape[0] != 2) {
        fprintf(stderr, "expected (2, T, V), got %zu dims\n", logits_ref.shape.size());
        return 1;
    }
    const int64_t T_ref = logits_ref.shape[1];
    const int64_t V_ref = logits_ref.shape[2];
    const float * ref_ptr = reinterpret_cast<const float *>(logits_ref.data.data());
    const float * ref_cond_last   = ref_ptr + 0 * T_ref * V_ref + (T_ref - 1) * V_ref;
    const float * ref_uncond_last = ref_ptr + 1 * T_ref * V_ref + (T_ref - 1) * V_ref;

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    std::vector<float> logits_cond, logits_uncond;
    int prompt_len = 0;
    fprintf(stderr, "running eval_prompt_mtl (cond + uncond)...\n");
    const float exaggeration = 0.5f;
    if (!eval_prompt_mtl(model, allocr, n_threads, text_tokens, exaggeration,
                         logits_cond, logits_uncond, prompt_len)) {
        fprintf(stderr, "eval_prompt_mtl failed\n");
        return 1;
    }
    fprintf(stderr, "prompt_len(C++)=%d  seq_len(ref)=%lld  vocab=%lld\n",
            prompt_len, (long long) T_ref, (long long) V_ref);

    if (prompt_len != T_ref) {
        fprintf(stderr, "WARNING: prompt length mismatch: C++=%d vs Python=%lld\n",
                prompt_len, (long long) T_ref);
    }
    if ((int64_t) logits_cond.size() != V_ref) {
        fprintf(stderr, "ERROR: logits size mismatch: C++=%zu vs Python=%lld\n",
                logits_cond.size(), (long long) V_ref);
        return 1;
    }

    auto stats_cond   = compare_f32(logits_cond.data(),   ref_cond_last,   (size_t) V_ref);
    print_compare("cond   logits (last pos)", stats_cond);
    auto stats_uncond = compare_f32(logits_uncond.data(), ref_uncond_last, (size_t) V_ref);
    print_compare("uncond logits (last pos)", stats_uncond);

    const bool pass = stats_cond.rel_err < 1e-3 && stats_uncond.rel_err < 1e-3;
    fprintf(stderr, "%s\n", pass ? "RESULT: PASS (rel < 1e-3)" : "RESULT: FAIL");
    return pass ? 0 : 1;
}
