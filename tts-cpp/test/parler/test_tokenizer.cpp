// Verifies parler_tokenizer against the HF fast-tokenizer corpus fixture:
//   test-parler-tokenizer MODEL.gguf REF_DIR
// REF_DIR must contain tokenizer_corpus.json ({"texts": [...], "ids": [[...]]},
// ids include the trailing </s>).

#include "gguf.h"
#include "json.hpp"
#include "parler/tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using tts_cpp::parler::detail::parler_tokenizer;
using nlohmann::json;

static bool load_tokenizer_from_gguf(const char * path, parler_tokenizer & tok) {
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "failed to open gguf: %s\n", path);
        return false;
    }

    bool ok = false;
    do {
        const int64_t k_tok = gguf_find_key(g, "tokenizer.ggml.tokens");
        const int64_t k_sc  = gguf_find_key(g, "tokenizer.ggml.scores");
        const int64_t k_unk = gguf_find_key(g, "tokenizer.ggml.unknown_token_id");
        const int64_t k_eos = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
        const int64_t k_add = gguf_find_key(g, "parler.tokenizer.add_eos");
        if (k_tok < 0 || k_sc < 0 || k_unk < 0 || k_eos < 0 || k_add < 0) {
            fprintf(stderr, "missing tokenizer metadata keys\n");
            break;
        }

        const size_t n = gguf_get_arr_n(g, k_tok);
        if (gguf_get_arr_type(g, k_sc) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(g, k_sc) != n) {
            fprintf(stderr, "bad tokenizer.ggml.scores array\n");
            break;
        }
        std::vector<std::string> pieces(n);
        for (size_t i = 0; i < n; ++i) {
            pieces[i] = gguf_get_arr_str(g, k_tok, i);
        }
        const float * sc = (const float *) gguf_get_arr_data(g, k_sc);
        std::vector<float> scores(sc, sc + n);

        std::vector<uint8_t> charsmap;
        const int64_t k_cm = gguf_find_key(g, "tokenizer.ggml.precompiled_charsmap");
        if (k_cm >= 0) {
            if (gguf_get_arr_type(g, k_cm) != GGUF_TYPE_UINT8) {
                fprintf(stderr, "bad tokenizer.ggml.precompiled_charsmap array type\n");
                break;
            }
            const uint8_t * cm = (const uint8_t *) gguf_get_arr_data(g, k_cm);
            charsmap.assign(cm, cm + gguf_get_arr_n(g, k_cm));
        }

        const int  unk_id  = (int) gguf_get_val_u32(g, k_unk);
        const int  eos_id  = (int) gguf_get_val_u32(g, k_eos);
        const bool add_eos = gguf_get_val_bool(g, k_add);

        ok = tok.load(std::move(pieces), std::move(scores), std::move(charsmap),
                      unk_id, eos_id, add_eos);
    } while (false);

    gguf_free(g);
    return ok;
}

static void print_ids(const char * label, const std::vector<int32_t> & ids) {
    fprintf(stderr, "  %s:", label);
    for (int32_t v : ids) {
        fprintf(stderr, " %d", v);
    }
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf REF_DIR\n", argv[0]);
        return 2;
    }

    parler_tokenizer tok;
    if (!load_tokenizer_from_gguf(argv[1], tok)) {
        return 1;
    }

    const std::string corpus_path = std::string(argv[2]) + "/tokenizer_corpus.json";
    std::ifstream ifs(corpus_path);
    if (!ifs) {
        fprintf(stderr, "cannot open %s\n", corpus_path.c_str());
        return 1;
    }
    json j = json::parse(ifs, nullptr, /*allow_exceptions*/ false);
    if (j.is_discarded() || !j.contains("texts") || !j.contains("ids") ||
        !j["texts"].is_array() || !j["ids"].is_array() ||
        j["texts"].size() != j["ids"].size()) {
        fprintf(stderr, "malformed corpus: %s\n", corpus_path.c_str());
        return 1;
    }

    int rc = 0;
    size_t n_cases = 0;
    try {
        const auto & texts = j["texts"];
        const auto & ids   = j["ids"];
        n_cases = texts.size();
        for (size_t i = 0; i < n_cases; ++i) {
            const std::string text = texts[i].get<std::string>();
            const std::vector<int32_t> expected = ids[i].get<std::vector<int32_t>>();
            const std::vector<int32_t> got = tok.encode(text);
            if (got != expected) {
                fprintf(stderr, "case %zu FAILED: \"%s\"\n", i, text.c_str());
                print_ids("expected", expected);
                print_ids("got     ", got);
                rc = 1;
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    if (rc == 0) {
        fprintf(stderr, "parler tokenizer: PASS (%zu cases)\n", n_cases);
    }
    return rc;
}
