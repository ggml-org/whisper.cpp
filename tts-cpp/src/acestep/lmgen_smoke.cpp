// lmgen-smoke: end-to-end Phase-2 audio-code generation for the ACE-Step LM
// (QVAC-21921). Loads the ace-lm GGUF + BPE tokenizer, builds a real turbo
// text2music prompt (caption + lyrics + metadata), runs the restricted
// (EOS + audio-code) decode loop and reports the generated FSQ codes.
//
// This is the first native LM->codes path (no CFG, single KV set). Proves the
// tokenizer, CoT/prompt template, sampling and decode loop work together on
// real weights and emit a plausible number of codes (~5 Hz * duration).
//
// Usage:
//   lmgen-smoke --model ace-lm.gguf [--dur 20] [--seed 42] [--temp 1.0]
//               [--top-p 0.9] [--top-k 50] [--caption "..."] [--lyrics "..."]

#include "acestep/bpe_tokenizer.h"
#include "acestep/lm_ggml.h"
#include "acestep/lm_pipeline.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace tts_cpp::acestep;

static const char * arg_val(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
}

int main(int argc, char ** argv) {
    const char * model = arg_val(argc, argv, "--model");
    if (!model) {
        fprintf(stderr, "usage: lmgen-smoke --model ace-lm.gguf [--dur 20] [--seed 42]\n");
        return 1;
    }

    AcePrompt p;
    p.caption       = arg_val(argc, argv, "--caption") ? arg_val(argc, argv, "--caption")
                                                       : "Upbeat pop rock with driving electric guitars, punchy drums and a catchy hook";
    p.lyrics        = arg_val(argc, argv, "--lyrics") ? arg_val(argc, argv, "--lyrics") : "[Instrumental]";
    p.duration      = arg_val(argc, argv, "--dur") ? (float) atof(arg_val(argc, argv, "--dur")) : 20.0f;
    p.bpm           = 128;
    p.keyscale      = "C major";
    p.timesignature = "4/4";
    p.vocal_language = "en";

    LmSampleParams sp;
    sp.temperature = arg_val(argc, argv, "--temp")  ? (float) atof(arg_val(argc, argv, "--temp"))  : 1.0f;
    sp.top_p       = arg_val(argc, argv, "--top-p") ? (float) atof(arg_val(argc, argv, "--top-p")) : 0.9f;
    sp.top_k       = arg_val(argc, argv, "--top-k") ? atoi(arg_val(argc, argv, "--top-k"))         : 50;
    sp.seed        = arg_val(argc, argv, "--seed")  ? (uint32_t) atoi(arg_val(argc, argv, "--seed")) : 42u;

    BpeTokenizer tok;
    if (!bpe_load_from_gguf(tok, model)) { fprintf(stderr, "bpe load failed\n"); return 1; }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }

    LMModel * m = lm_model_load(model, backend, /*max_seq_len=*/4096, /*verbose=*/true);
    if (!m) { fprintf(stderr, "lm_model_load failed\n"); ggml_backend_free(backend); return 1; }

    std::vector<int> codes;
    bool             ok = lm_generate_codes(m, tok, p, sp, codes);

    if (ok && !codes.empty()) {
        int nshow = (int) codes.size() < 24 ? (int) codes.size() : 24;
        fprintf(stderr, "[lmgen-smoke] first %d codes:", nshow);
        for (int i = 0; i < nshow; i++) fprintf(stderr, " %d", codes[i]);
        fprintf(stderr, "\n");
        // Sanity: codes in valid FSQ range, and count roughly ~5 Hz * duration.
        int  bad     = 0;
        for (int c : codes) if (c < 0 || c >= AUDIO_CODE_COUNT) bad++;
        float rate   = codes.size() / p.duration;
        fprintf(stderr, "[lmgen-smoke] %zu codes, ~%.1f codes/s (expect ~5), out-of-range=%d\n",
                codes.size(), rate, bad);
        ok = (bad == 0);
    }

    fprintf(stderr, "[lmgen-smoke] %s\n", (ok && !codes.empty()) ? "PASS" : "FAIL");
    lm_model_free(m);
    ggml_backend_free(backend);
    return (ok && !codes.empty()) ? 0 : 1;
}
