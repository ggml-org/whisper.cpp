// CTC greedy decode parity vs dumped NeMo logits.
//
// Replays reference logits through the C++ decoder and checks the transcript.
//
// Usage:
//   test-ctc <parakeet-ctc.gguf> <reference-logits.npy> [<decoded.txt>]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "sentencepiece_bpe.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int load_npy_f32(const std::string & path,
                 std::vector<float>  & out_data,
                 std::vector<int64_t> & out_shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 1;

    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return 2;

    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl = 0;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    const size_t shp_s = header.find("'shape':");
    const size_t lp = header.find('(', shp_s);
    const size_t rp = header.find(')', lp);
    const std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    out_shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) ++pos;
        if (pos >= shape_str.size()) break;
        size_t end = pos;
        while (end < shape_str.size() && std::isdigit(static_cast<unsigned char>(shape_str[end]))) ++end;
        if (end > pos) {
            out_shape.push_back(std::stoll(shape_str.substr(pos, end - pos)));
            pos = end;
        } else {
            break;
        }
    }

    size_t n = 1;
    for (int64_t d : out_shape) n *= static_cast<size_t>(d);
    out_data.resize(n);
    f.read(reinterpret_cast<char*>(out_data.data()), n * sizeof(float));
    return 0;
}

std::string read_text_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <parakeet-ctc.gguf> <reference-logits.npy> [<expected-text-file>]\n"
            "\n"
            "Replays NeMo's logits.npy (T_enc, vocab+blank) through the C++ CTC greedy\n"
            "decoder + SentencePiece detokenizer and asserts the decoded text matches\n"
            "the reference. The expected text path defaults to <dirname-of-logits>/decoded.txt\n"
            "(produced by scripts/dump-ctc-reference.py).\n",
            argv[0]);
        return 2;
    }

    const std::string gguf_path   = argv[1];
    const std::string logits_path = argv[2];

    std::string expected_path;
    if (argc >= 4) {
        expected_path = argv[3];
    } else {
        const size_t slash = logits_path.find_last_of('/');
        const std::string dir = (slash == std::string::npos) ? std::string(".")
                                                              : logits_path.substr(0, slash);
        expected_path = dir + "/decoded.txt";
    }

    parakeet::ParakeetCtcModel model;
    if (int rc = parakeet::load_from_gguf(gguf_path, model,
                                               /*n_threads=*/0,
                                               /*n_gpu_layers=*/0,
                                               /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[test-ctc] load_from_gguf %s failed (rc=%d)\n",
                     gguf_path.c_str(), rc);
        return 3;
    }
    if (model.model_type != parakeet::ParakeetModelType::CTC) {
        std::fprintf(stderr,
            "[test-ctc] %s is not a CTC GGUF (model_type=%d). The CTC parity harness\n"
            "          only applies to parakeet-ctc-* checkpoints; use test-tdt-encoder-parity\n"
            "          for TDT and test-sortformer-parity for Sortformer.\n",
            gguf_path.c_str(), (int) model.model_type);
        return 4;
    }

    std::vector<float>   logits;
    std::vector<int64_t> shape;
    if (int rc = load_npy_f32(logits_path, logits, shape); rc != 0) {
        std::fprintf(stderr, "[test-ctc] load_npy_f32 %s failed (rc=%d)\n",
                     logits_path.c_str(), rc);
        return 5;
    }
    if (shape.size() != 2) {
        std::fprintf(stderr, "[test-ctc] expected logits rank=2 (T_enc, vocab+blank), got rank=%zu\n",
                     shape.size());
        return 6;
    }
    const int n_frames = (int) shape[0];
    const int vocab    = (int) shape[1];
    if (vocab != (int) model.vocab_size) {
        std::fprintf(stderr,
            "[test-ctc] vocab mismatch: logits vocab=%d, GGUF vocab_size=%d\n",
            vocab, (int) model.vocab_size);
        return 7;
    }

    std::vector<int32_t> ids = parakeet::ctc_greedy_decode(
        logits.data(), n_frames, model.vocab_size, model.blank_id);
    const std::string text = parakeet::detokenize(model.vocab, ids);

    std::string expected = read_text_file(expected_path);
    std::fprintf(stderr, "[test-ctc] gguf=%s\n", gguf_path.c_str());
    std::fprintf(stderr, "[test-ctc] logits=%s shape=(%d, %d) blank=%d\n",
                 logits_path.c_str(), n_frames, vocab, (int) model.blank_id);
    std::fprintf(stderr, "[test-ctc] decoded: \"%s\"\n", text.c_str());

    if (expected.empty()) {
        std::fprintf(stderr,
            "[test-ctc] WARN: expected text not found at %s; printed decoded text only.\n",
            expected_path.c_str());
        std::printf("%s\n", text.c_str());
        return 0;
    }

    std::fprintf(stderr, "[test-ctc] expected: \"%s\"\n", expected.c_str());
    if (text != expected) {
        std::fprintf(stderr, "[test-ctc] FAIL: transcript mismatch\n");
        return 1;
    }
    std::fprintf(stderr, "[test-ctc] PASS (transcript byte-equal to NeMo reference)\n");
    return 0;
}
