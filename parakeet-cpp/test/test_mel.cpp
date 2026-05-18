// Log-mel parity vs NeMo reference dump for a wav file.
//
// Usage:
//   test-mel <parakeet-ctc.gguf> <input.wav> <ref-mel.npy>
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "mel_preprocess.h"
#include "parakeet_ctc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int load_npy_f32(const std::string & path, std::vector<float> & out_data, std::vector<int64_t> & out_shape) {
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
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <parakeet-ctc.gguf> <input.wav> <ref-mel.npy>\n"
            "\n"
            "asserts C++ log-mel matches the NeMo reference dump.  The reference mel\n"
            "is shape (n_mels, T_mel) (f32) saved by scripts/dump-ctc-reference.py.\n",
            argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_path  = argv[3];

    using namespace parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf_path, model, 0, 0, true); rc != 0) return rc;

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) return rc;

    std::vector<float> mel;
    int n_frames = 0;
    if (int rc = compute_log_mel(samples.data(), static_cast<int>(samples.size()),
                                 model.mel_cfg, mel, n_frames); rc != 0) return rc;

    std::vector<float> mel_transposed(static_cast<size_t>(model.mel_cfg.n_mels) * n_frames);
    for (int t = 0; t < n_frames; ++t) {
        for (int m = 0; m < model.mel_cfg.n_mels; ++m) {
            mel_transposed[m * n_frames + t] = mel[t * model.mel_cfg.n_mels + m];
        }
    }

    std::vector<float> ref;
    std::vector<int64_t> ref_shape;
    if (int rc = load_npy_f32(ref_path, ref, ref_shape); rc != 0) {
        std::fprintf(stderr, "error: failed to load %s (rc=%d)\n", ref_path.c_str(), rc);
        return rc;
    }

    std::fprintf(stderr, "[test-mel] c++ mel: (%d, %d)   ref mel:", model.mel_cfg.n_mels, n_frames);
    for (int64_t d : ref_shape) std::fprintf(stderr, " %lld", (long long) d);
    std::fprintf(stderr, "\n");

    if (ref_shape.size() == 2 && ref_shape[0] == model.mel_cfg.n_mels) {
        const int ref_frames = static_cast<int>(ref_shape[1]);
        if (ref_frames != n_frames) {
            std::fprintf(stderr, "[test-mel] frame-count mismatch: c++=%d ref=%d — truncating to min\n",
                         n_frames, ref_frames);
        }
        const int cmp_frames = std::min(n_frames, ref_frames);
        std::vector<float> a(cmp_frames * model.mel_cfg.n_mels);
        std::vector<float> b(cmp_frames * model.mel_cfg.n_mels);
        for (int m = 0; m < model.mel_cfg.n_mels; ++m) {
            for (int t = 0; t < cmp_frames; ++t) {
                a[m * cmp_frames + t] = mel_transposed[m * n_frames + t];
                b[m * cmp_frames + t] = ref[m * ref_frames + t];
            }
        }
        double max_abs = 0, rel = 0;
        compute_parity(a, b, max_abs, rel);
        std::fprintf(stderr, "[test-mel] rel = %.3e   max_abs = %.3e   (diagnostic only; full-window includes trailing-frame padding diff vs NeMo)\n", rel, max_abs);

        const int inner_end = std::max(0, cmp_frames - 2);
        std::vector<float> a_inner(inner_end * model.mel_cfg.n_mels);
        std::vector<float> b_inner(inner_end * model.mel_cfg.n_mels);
        for (int m = 0; m < model.mel_cfg.n_mels; ++m) {
            for (int t = 0; t < inner_end; ++t) {
                a_inner[m * inner_end + t] = mel_transposed[m * n_frames + t];
                b_inner[m * inner_end + t] = ref[m * ref_frames + t];
            }
        }
        double inner_max_abs = 0, inner_rel = 0;
        compute_parity(a_inner, b_inner, inner_max_abs, inner_rel);
        std::fprintf(stderr, "[test-mel]   inner (excluding last 2 frames; gated; target rel < 5e-3):  rel = %.3e   max_abs = %.3e\n",
                     inner_rel, inner_max_abs);

        return (inner_rel < 5e-3) ? 0 : 1;
    }

    std::fprintf(stderr, "[test-mel] unexpected ref shape; skipping parity\n");
    return 1;
}
