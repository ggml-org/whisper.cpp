// Parity test — CosyVoice3 flow (stage 4): speech tokens -> mel.
//
// Runs the shared cosyvoice_flow_run() on reference prompt/token inputs and
// asserts the produced mel matches the PyTorch reference (flow_mel.npy) at
// cosine >= threshold. Exits non-zero on failure (registered as a ctest,
// auto-disabled when the fixtures below are absent).
//
// Fixtures (generate with scripts/dump-cosyvoice3-reference.py):
//   <in-dir>/{prompt_token,speech_tokens,prompt_feat,embedding,flow_mel}.npy
//
// Usage:
//   test-cosyvoice-flow --flow-gguf FLOW.gguf --in-dir DIR [--min-cosine 0.99]

#include "npy.h"
#include "cosyvoice_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double min_cosine = 0.99;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--flow-gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--min-cosine" && i + 1 < argc) min_cosine = std::atof(argv[++i]);
        else { fprintf(stderr, "usage: %s --flow-gguf FLOW.gguf --in-dir DIR [--min-cosine 0.99]\n", argv[0]); return 2; }
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "missing --flow-gguf / --in-dir\n"); return 2; }

    const int MEL = 80;
    model_ctx m = cosyvoice_load_gguf(gguf);

    npy_array ptok_a  = npy_load(in_dir + "/prompt_token.npy");
    npy_array stok_a  = npy_load(in_dir + "/speech_tokens.npy");
    npy_array pfeat_a = npy_load(in_dir + "/prompt_feat.npy");
    npy_array emb_a   = npy_load(in_dir + "/embedding.npy");
    npy_array ref_a   = npy_load(in_dir + "/flow_mel.npy");

    int T_ptok = (int)ptok_a.shape[0], T_stok = (int)stok_a.shape[0];
    int mel_len1 = (int)pfeat_a.shape[0];
    std::vector<int> ptok(T_ptok), stok(T_stok);
    { const int32_t * p = npy_as_i32(ptok_a); for (int i = 0; i < T_ptok; ++i) ptok[i] = p[i]; }
    { const int32_t * p = npy_as_i32(stok_a); for (int i = 0; i < T_stok; ++i) stok[i] = p[i]; }
    std::vector<float> pfeat(npy_as_f32(pfeat_a), npy_as_f32(pfeat_a) + (size_t)mel_len1 * MEL);
    std::vector<float> emb(npy_as_f32(emb_a), npy_as_f32(emb_a) + (size_t)emb_a.shape[0]);

    int mel_len2 = 0;
    std::vector<float> mel = cosyvoice_flow_run(m, ptok, stok, pfeat, mel_len1, emb, mel_len2);

    if (mel.size() != (size_t)MEL * mel_len2 || (int)ref_a.shape[0] != MEL) {
        fprintf(stderr, "FAIL: shape mismatch (got %zu, ref [%lld,%lld])\n",
                mel.size(), (long long)ref_a.shape[0], (long long)ref_a.shape[1]);
        return 1;
    }
    size_t n = std::min(mel.size(), (size_t)MEL * (size_t)ref_a.shape[1]);
    const float * e = npy_as_f32(ref_a);
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) { const double a = mel[i], b = e[i]; dot += a * b; na += a * a; nb += b * b; }
    double cosine = dot / (std::sqrt(na) * std::sqrt(nb));

    fprintf(stderr, "flow_mel cosine = %.6f  (threshold %.4f, %d frames)\n", cosine, min_cosine, mel_len2);
    if (!(cosine >= min_cosine)) { fprintf(stderr, "FAIL: flow_mel cosine below threshold\n"); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
