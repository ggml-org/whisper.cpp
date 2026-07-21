// Encoder stage parity vs NeMo reference tensors under reference-dir/.
//
// Usage:
//   test-encoder <parakeet-ctc.gguf> <reference-dir> [n_gpu_layers]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int load_npy_f32(const std::string & path,
                 std::vector<float> & out_data,
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

void compute_parity(const std::vector<float> & a, const std::vector<float> & b,
                    double & out_max_abs, double & out_rel) {
    double max_abs = 0.0, ref_sq = 0.0, diff_sq = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        max_abs = std::max(max_abs, std::abs(d));
        diff_sq += d * d;
        ref_sq  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    out_max_abs = max_abs;
    out_rel = (ref_sq > 0) ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
}

}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <parakeet-ctc.gguf> <reference-dir> [n_gpu_layers]\n"
            "\n"
            "walks through per-stage reference tensors produced by\n"
            "scripts/dump-ctc-reference.py and asserts parity for each stage.\n"
            "\n"
            "stages checked:\n"
            "  1. subsampling_out.npy   (n_enc_frames, d_model) f32\n"
            "\n"
            "n_gpu_layers: 0 (default) = CPU, >0 = GPU backend if compiled in.\n",
            argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    const std::string ref_dir   = argv[2];
    const int n_gpu_layers = (argc >= 4) ? std::atoi(argv[3]) : 0;

    using namespace parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf_path, model, 0, n_gpu_layers, true); rc != 0) return rc;

    std::vector<float> ref_mel;
    std::vector<int64_t> mel_shape;
    if (int rc = load_npy_f32(ref_dir + "/mel.npy", ref_mel, mel_shape); rc != 0) {
        std::fprintf(stderr, "error: cannot load %s/mel.npy (rc=%d)\n", ref_dir.c_str(), rc);
        return rc;
    }
    if (mel_shape.size() != 2 || mel_shape[0] != model.mel_cfg.n_mels) {
        std::fprintf(stderr, "error: mel.npy has unexpected shape\n");
        return 3;
    }
    const int n_mels      = static_cast<int>(mel_shape[0]);
    const int n_mel_frames = static_cast<int>(mel_shape[1]);

    std::vector<float> mel_in((size_t) n_mels * n_mel_frames);
    for (int t = 0; t < n_mel_frames; ++t) {
        for (int m = 0; m < n_mels; ++m) {
            mel_in[(size_t) t * n_mels + m] = ref_mel[(size_t) m * n_mel_frames + t];
        }
    }

    EncoderOutputs enc_out;
    if (int rc = run_encoder(model, mel_in.data(), n_mel_frames, n_mels, enc_out); rc != 0) {
        std::fprintf(stderr, "run_encoder failed (rc=%d)\n", rc);
        return rc;
    }

    struct StageGate {
        const char *        label;
        std::string         file;
        const std::vector<float> * cpp;
        double              threshold;
    };

    const std::vector<StageGate> gates = {
        { "B  subsampling_out",  ref_dir + "/subsampling_out.npy",    &enc_out.subsampling_out,  2e-3 },
        { "C0 post_ff1  (b0)",   ref_dir + "/block_0_post_ff1.npy",   &enc_out.block_0_post_ff1, 3e-3 },
        { "C1 post_attn (b0)",   ref_dir + "/block_0_post_attn.npy",  &enc_out.block_0_post_attn,3e-3 },
        { "C2 post_conv (b0)",   ref_dir + "/block_0_post_conv.npy",  &enc_out.block_0_post_conv,3e-3 },
        { "C3 post_ff2  (b0)",   ref_dir + "/block_0_post_ff2.npy",   &enc_out.block_0_post_ff2, 3e-3 },
        { "C  block_0_out",      ref_dir + "/block_0_out.npy",        &enc_out.block_0_out,      5e-3 },
        { "D  block_last_out",   ref_dir + "/block_last_out.npy",     &enc_out.block_last_out,   5e-3 },
        { "E  encoder_out",      ref_dir + "/encoder_out.npy",        &enc_out.encoder_out,      5e-3 },
    };

    bool all_pass = true;
    for (const auto & g : gates) {
        std::vector<float> ref;
        std::vector<int64_t> shape;
        if (int rc = load_npy_f32(g.file, ref, shape); rc != 0) {
            std::fprintf(stderr, "[test-encoder] stage %s  cannot load %s (rc=%d)\n",
                         g.label, g.file.c_str(), rc);
            all_pass = false;
            continue;
        }
        double max_abs = 0, rel = 0;
        compute_parity(*g.cpp, ref, max_abs, rel);
        const bool pass = rel < g.threshold;
        std::fprintf(stderr, "[test-encoder] stage %s  shape=%lldx%lld  rel=%.3e  max_abs=%.3e  %s\n",
                     g.label, (long long) shape[0], (long long) shape[1],
                     rel, max_abs, pass ? "ok" : "FAIL");
        if (!pass) all_pass = false;
    }

    {
        std::vector<float> ref_logits;
        std::vector<int64_t> shape;
        if (load_npy_f32(ref_dir + "/logits.npy", ref_logits, shape) == 0 &&
            shape.size() == 2 &&
            enc_out.logits.size() == (size_t)(shape[0] * shape[1])) {

            const int T = (int) shape[0];
            const int V = (int) shape[1];
            std::vector<float> log_sm(enc_out.logits.size());
            for (int t = 0; t < T; ++t) {
                const float * row = enc_out.logits.data() + (size_t) t * V;
                float m = row[0];
                for (int i = 1; i < V; ++i) if (row[i] > m) m = row[i];
                double s = 0.0;
                for (int i = 0; i < V; ++i) s += std::exp((double) (row[i] - m));
                const float log_s = m + (float) std::log(s);
                for (int i = 0; i < V; ++i) log_sm[(size_t) t * V + i] = row[i] - log_s;
            }

            double max_abs = 0, rel = 0;
            compute_parity(log_sm, ref_logits, max_abs, rel);
            const bool pass = rel < 5e-3;
            std::fprintf(stderr, "[test-encoder] stage F  logits (host log_softmax)  rel=%.3e  max_abs=%.3e  %s\n",
                         rel, max_abs, pass ? "ok" : "FAIL");
            if (!pass) all_pass = false;
        }
    }

    return all_pass ? 0 : 1;
}
