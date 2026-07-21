// TDT encoder parity vs NeMo reference tensors under ref-dir/.
//
// Usage:
//   test-tdt-encoder-parity <parakeet-tdt.gguf> <wav> <ref-dir>
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <parakeet-tdt.gguf> <wav> <ref-dir>\n"
            "\n"
            "Validates the C++ TDT-GGUF encoder output vs NeMo reference tensors:\n"
            "  <ref-dir>/mel.npy         (128, T_mel)\n"
            "  <ref-dir>/encoder_out.npy (T_enc, 1024)\n"
            "\n"
            "Produce refs with: python scripts/dump-tdt-reference.py --wav <wav>\n",
            argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_dir   = argv[3];

    std::fprintf(stderr, "[tdt-parity] loading %s\n", gguf_path.c_str());
    using namespace parakeet;
    ParakeetCtcModel model;
    // Force CPU encoder: this harness gates operator parity vs NeMo's FP32
    // reference; backend-induced drift (CPU<->GPU) is gated separately by
    // test-vk-vs-cpu and would otherwise mask real encoder regressions here.
    if (int rc = load_from_gguf(gguf_path, model, 0, 0, false); rc != 0) {
        std::fprintf(stderr, "  load_from_gguf failed rc=%d\n", rc);
        return 3;
    }
    if (model.model_type != ParakeetModelType::TDT) {
        std::fprintf(stderr, "  error: expected TDT model, got CTC\n");
        return 3;
    }

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "  load_wav failed rc=%d\n", rc);
        return 4;
    }
    std::fprintf(stderr, "[tdt-parity] wav: %zu samples @ %d Hz\n", samples.size(), sr);

    std::vector<float> mel_cpp;
    int n_frames_cpp = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model.mel_cfg, mel_cpp, n_frames_cpp); rc != 0) {
        std::fprintf(stderr, "  compute_log_mel failed rc=%d\n", rc);
        return 5;
    }

    std::vector<float> mel_ref;
    std::vector<int64_t> mel_shape;
    if (load_npy_f32(ref_dir + "/mel.npy", mel_ref, mel_shape) != 0 || mel_shape.size() != 2) {
        std::fprintf(stderr, "  failed to load %s/mel.npy\n", ref_dir.c_str());
        return 5;
    }
    const int n_mels_ref      = static_cast<int>(mel_shape[0]);
    const int n_frames_ref    = static_cast<int>(mel_shape[1]);
    if (n_mels_ref != model.mel_cfg.n_mels) {
        std::fprintf(stderr, "  mel.npy n_mels=%d != model.mel_cfg.n_mels=%d\n",
                     n_mels_ref, model.mel_cfg.n_mels);
        return 5;
    }
    std::vector<float> mel_ref_tf(static_cast<size_t>(n_frames_ref) * n_mels_ref);
    for (int t = 0; t < n_frames_ref; ++t) {
        for (int m = 0; m < n_mels_ref; ++m) {
            mel_ref_tf[static_cast<size_t>(t) * n_mels_ref + m] =
                mel_ref[static_cast<size_t>(m) * n_frames_ref + t];
        }
    }

    {
        std::fprintf(stderr, "[tdt-parity] mel: cpp=(%d, %d) ref=(%d, %d)\n",
                     n_frames_cpp, n_mels_ref, n_frames_ref, n_mels_ref);
        const size_t common = static_cast<size_t>(std::min(n_frames_cpp, n_frames_ref)) * n_mels_ref;
        std::vector<float> a(mel_cpp.begin(),    mel_cpp.begin()    + common);
        std::vector<float> b(mel_ref_tf.begin(), mel_ref_tf.begin() + common);
        double max_abs = 0, rel = 0;
        compute_parity(a, b, max_abs, rel);
        std::fprintf(stderr, "[tdt-parity] mel  : max_abs=%.4e rel=%.4e  (diagnostic; trailing-frame padding diff vs NeMo)\n",
                     max_abs, rel);
    }

    // Encoder parity is gated against ref-mel input -- matches test-encoder's
    // (CTC) shape so both tests measure the encoder operator alone, decoupled
    // from the trailing-frame padding diff in the C++ mel pipeline.
    EncoderOutputs enc_out;
    if (int rc = run_encoder(model, mel_ref_tf.data(), n_frames_ref, n_mels_ref, enc_out); rc != 0) {
        std::fprintf(stderr, "  run_encoder failed rc=%d\n", rc);
        return 6;
    }
    std::fprintf(stderr, "[tdt-parity] encoder: cpp frames=%d d_model=%d\n",
                 enc_out.n_enc_frames, enc_out.d_model);

    std::vector<float> enc_ref;
    std::vector<int64_t> shape;
    if (load_npy_f32(ref_dir + "/encoder_out.npy", enc_ref, shape) != 0) {
        std::fprintf(stderr, "  failed to load %s/encoder_out.npy\n", ref_dir.c_str());
        return 7;
    }
    const int T_ref = static_cast<int>(shape[0]);
    const int D_ref = static_cast<int>(shape[1]);
    std::fprintf(stderr, "[tdt-parity] encoder ref: (%d, %d)\n", T_ref, D_ref);

    const size_t common = static_cast<size_t>(std::min(enc_out.n_enc_frames, T_ref)) * enc_out.d_model;
    std::vector<float> a(enc_out.encoder_out.begin(), enc_out.encoder_out.begin() + common);
    std::vector<float> b(enc_ref.begin(), enc_ref.begin() + common);
    double max_abs = 0, rel = 0;
    compute_parity(a, b, max_abs, rel);
    std::fprintf(stderr, "[tdt-parity] enc  : max_abs=%.4e rel=%.4e  (%s)\n",
                 max_abs, rel, rel < 5e-3 ? "PASS" : "FAIL");

    if (rel < 5e-3) {
        std::fprintf(stderr, "[tdt-parity] PASS\n");
        return 0;
    }
    std::fprintf(stderr, "[tdt-parity] FAIL: encoder parity rel=%.4e > 5e-3\n", rel);
    return 1;
}
