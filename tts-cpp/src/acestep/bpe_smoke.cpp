// bpe-smoke: load + encode/decode roundtrip harness for the ACE-Step BPE
// tokenizer (Qwen3/GPT-2 byte-level, QVAC-21921). Proves the vocab + merges
// load from the LM GGUF and that encode->decode reproduces the input text
// (including unicode, digits, punctuation, newlines).
//
// Usage:
//   bpe-smoke --model ace-lm.gguf

#include "acestep/bpe_tokenizer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tts_cpp::acestep;

static const char * arg_val(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
}

static bool check_roundtrip(const BpeTokenizer & tok, const std::string & text) {
    std::vector<int> ids = bpe_encode(tok, text, /*add_eos=*/false);
    std::string      back = bpe_decode(tok, ids);
    bool             ok   = (back == text);
    fprintf(stderr, "[bpe-smoke] %s  %2zu tok  %-40.40s %s\n", ok ? "ok  " : "FAIL",
            ids.size(), text.c_str(), ok ? "" : ("-> '" + back + "'").c_str());
    return ok;
}

int main(int argc, char ** argv) {
    const char * model = arg_val(argc, argv, "--model");
    if (!model) { fprintf(stderr, "usage: bpe-smoke --model ace-lm.gguf\n"); return 1; }

    BpeTokenizer tok;
    if (!bpe_load_from_gguf(tok, model)) { fprintf(stderr, "bpe load failed\n"); return 1; }

    const char * cases[] = {
        "hello world",
        "A pop song with catchy melody.",
        "bpm: 120\nduration: 30\nkeyscale: C major\n",
        "[Verse]\nWalking down the street tonight",
        "café, naïve, 123 test!",
        "中文测试 mixed with English",
        "punctuation... yes! (really?) 'quotes' \"double\"",
    };

    int fails = 0;
    for (const char * s : cases) if (!check_roundtrip(tok, s)) fails++;

    // Special-token sanity: im_start/end round-trip is intentionally dropped,
    // <think> tags expand back to text.
    std::vector<int> special = { TOKEN_THINK, TOKEN_THINK_END };
    std::string      dec     = bpe_decode(tok, special);
    bool             sp_ok   = (dec == "<think></think>");
    fprintf(stderr, "[bpe-smoke] %s  think-tags -> '%s'\n", sp_ok ? "ok  " : "FAIL", dec.c_str());
    if (!sp_ok) fails++;

    fprintf(stderr, "[bpe-smoke] vocab=%d eos=%d  %s\n", tok.n_vocab, tok.eos_id, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
