// Parity test — CosyVoice3 Qwen2 byte-level BPE tokenizer (text frontend).
//
// The hand-rolled qwen_tokenizer is on the critical path (its ids feed the LM),
// and the pipeline claims bit-identical text_ids vs HF AutoTokenizer. This test
// asserts encode() reproduces the HF reference ids exactly for a handful of
// strings — ASCII, the <|endofprompt|> special token, and CJK — so a regression
// in pretokenize / merge ranking / byte mapping / special splitting is caught
// directly, not only end-to-end.
//
// Fixtures:
//   --vocab / --merges : the real Qwen2 vocab.json + merges.txt (model bundle)
//   --ref              : tokenizer_ref.txt from
//                        scripts/dump-cosyvoice3-tokenizer-reference.py, one
//                        case per line as "<id,id,...>\t<exact text>".
//
// Usage:
//   test-cosyvoice-tokenizer --vocab vocab.json --merges merges.txt --ref REF

#include "qwen_tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<int> parse_ids(const std::string & csv) {
    std::vector<int> ids;
    size_t i = 0;
    while (i < csv.size()) {
        size_t j = csv.find(',', i);
        std::string tok = csv.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (!tok.empty()) ids.push_back(std::atoi(tok.c_str()));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return ids;
}

int main(int argc, char ** argv) {
    std::string vocab, merges, ref;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--vocab"  && i + 1 < argc) vocab  = argv[++i];
        else if (a == "--merges" && i + 1 < argc) merges = argv[++i];
        else if (a == "--ref"    && i + 1 < argc) ref    = argv[++i];
        else { fprintf(stderr, "usage: %s --vocab V --merges M --ref REF\n", argv[0]); return 2; }
    }
    if (vocab.empty() || merges.empty() || ref.empty()) {
        fprintf(stderr, "missing --vocab / --merges / --ref\n"); return 2;
    }

    QwenTokenizer tok;
    if (!tok.load(vocab, merges)) { fprintf(stderr, "FAIL: could not load vocab/merges\n"); return 1; }

    std::ifstream rf(ref);
    if (!rf) { fprintf(stderr, "FAIL: could not open ref %s\n", ref.c_str()); return 1; }

    int cases = 0, failures = 0;
    std::string line;
    while (std::getline(rf, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) { fprintf(stderr, "FAIL: malformed ref line: %s\n", line.c_str()); return 1; }
        std::vector<int> want = parse_ids(line.substr(0, tab));
        std::string text = line.substr(tab + 1);

        std::vector<int> got = tok.encode(text);
        ++cases;
        bool ok = (got == want);
        if (!ok) {
            ++failures;
            fprintf(stderr, "MISMATCH for \"%s\"\n  want:", text.c_str());
            for (int v : want) fprintf(stderr, " %d", v);
            fprintf(stderr, "\n  got :");
            for (int v : got) fprintf(stderr, " %d", v);
            fprintf(stderr, "\n");
        }
    }

    if (cases == 0) { fprintf(stderr, "FAIL: no cases in ref\n"); return 1; }
    fprintf(stderr, "tokenizer parity: %d/%d cases match\n", cases - failures, cases);
    if (failures) { fprintf(stderr, "FAIL: %d tokenizer mismatch(es)\n", failures); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
