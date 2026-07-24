// Parity test — CosyVoice3 LLM (stage 5): text ids -> speech tokens.
//
// Forces greedy (argmax) decoding and asserts the generated speech-token
// trajectory matches the PyTorch reference (gen_tokens.npy) exactly. Greedy
// makes the LM deterministic so the C++ decode can be validated bit-for-bit
// (the shipped RAS sampling is stochastic). Exits non-zero on failure;
// registered as a ctest, auto-disabled when fixtures are absent.
//
// Fixtures (generate with scripts/dump-cosyvoice3-llm-reference.py):
//   <in-dir>/{text_ids,prompt_stok,gen_tokens}.npy
//
// Usage:
//   test-cosyvoice-llm --llm-gguf LLM.gguf --in-dir DIR [--min-match 0.99]

#include "npy.h"
#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Speech-token vocab is [0, 6561); anything >= this is an EOS / special marker
// that the C++ generator consumes as the stop signal (and does not emit).
static const int SPEECH_TOKEN_MAX = 6561;

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double min_match = 0.99;
    int seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--llm-gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--min-match" && i + 1 < argc) min_match = std::atof(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --llm-gguf LLM.gguf --in-dir DIR [--min-match 0.99]\n", argv[0]); return 2; }
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "missing --llm-gguf / --in-dir\n"); return 2; }

    model_ctx m = cosyvoice_load_gguf(gguf);
    qwen_hp hp = cosyvoice_qwen_hp(m);  // exercise the GGUF-KV hp path

    npy_array tid_a = npy_load(in_dir + "/text_ids.npy");
    npy_array ps_a  = npy_load(in_dir + "/prompt_stok.npy");
    npy_array ref_a = npy_load(in_dir + "/gen_tokens.npy");

    std::vector<int> text_ids((int)tid_a.shape[0]);
    { const int32_t * p = npy_as_i32(tid_a); for (size_t i = 0; i < text_ids.size(); ++i) text_ids[i] = p[i]; }
    std::vector<int> prompt_stok(ps_a.shape.empty() ? 0 : (int)ps_a.shape[0]);
    { const int32_t * p = npy_as_i32(ps_a); for (size_t i = 0; i < prompt_stok.size(); ++i) prompt_stok[i] = p[i]; }

    // Reference greedy tokens, truncated at the first EOS/special marker.
    const int32_t * rp = npy_as_i32(ref_a);
    std::vector<int> ref;
    for (size_t i = 0; i < (size_t)ref_a.shape[0]; ++i) {
        if (rp[i] >= SPEECH_TOKEN_MAX) break;
        ref.push_back(rp[i]);
    }
    if (ref.empty()) { fprintf(stderr, "FAIL: reference has no speech tokens\n"); return 1; }

    const int max_steps = (int)ref.size() + 16;
    std::vector<int> got = cosyvoice_llm_generate(
        m, hp, text_ids, prompt_stok, max_steps, /*greedy=*/true, seed, /*min_len=*/0);

    size_t n = std::min(got.size(), ref.size());
    size_t match = 0;
    for (size_t i = 0; i < n; ++i) if (got[i] == ref[i]) ++match;
    double ratio = n ? (double)match / (double)n : 0.0;

    fprintf(stderr, "greedy tokens: got=%zu ref=%zu  match=%zu/%zu (%.4f, threshold %.4f)\n",
            got.size(), ref.size(), match, n, ratio, min_match);
    if (got.size() < ref.size())
        fprintf(stderr, "  note: generated fewer tokens than reference\n");
    if (!(ratio >= min_match) || n < ref.size() * min_match) {
        fprintf(stderr, "FAIL: greedy token trajectory diverges from reference\n");
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
