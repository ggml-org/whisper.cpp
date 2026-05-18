// Sortformer encoder and head parity vs NeMo reference tensors.
//
// Usage:
//   test-sortformer-parity <sortformer.gguf> <wav> <ref-dir> [--enc-rel-tol R] [--probs-abs-tol R]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "parakeet_sortformer.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
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
    char magic[6]; f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return 2;
    uint8_t major = 0, minor = 0;
    f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint32_t header_len = 0;
    if (major == 1) { uint16_t h = 0; f.read((char*)&h, 2); header_len = h; }
    else            { f.read((char*)&header_len, 4); }
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shp_s = header.find("'shape':");
    const size_t lp = header.find('(', shp_s);
    const size_t rp = header.find(')', lp);
    out_shape.clear();
    size_t pos = 0;
    const std::string s = header.substr(lp + 1, rp - lp - 1);
    while (pos < s.size()) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',')) ++pos;
        if (pos >= s.size()) break;
        size_t end = pos;
        while (end < s.size() && std::isdigit((unsigned char)s[end])) ++end;
        if (end > pos) { out_shape.push_back(std::stoll(s.substr(pos, end - pos))); pos = end; }
        else break;
    }
    size_t n = 1;
    for (auto d : out_shape) n *= (size_t) d;
    out_data.resize(n);
    f.read((char*)out_data.data(), n * sizeof(float));
    return 0;
}

void parity(const std::vector<float> & a, const std::vector<float> & b,
            double & out_max_abs, double & out_rel) {
    double max_abs = 0.0, ref_sq = 0.0, diff_sq = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        if (std::abs(d) > max_abs) max_abs = std::abs(d);
        diff_sq += d * d;
        ref_sq += (double) b[i] * b[i];
    }
    out_max_abs = max_abs;
    out_rel = ref_sq > 0 ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
}

}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <sortformer.gguf> <wav> <ref-dir> [--enc-rel-tol R] [--probs-abs-tol R]\n"
            "  ref-dir produced by scripts/dump-sortformer-reference.py\n"
            "  --enc-rel-tol     gates encoder rel-err vs NeMo (default 5e-3 for f16;\n"
            "                    bump to ~3e-2 for q8_0, ~5e-1 for q4_0 -- the head\n"
            "                    absorbs the inflated intermediates)\n"
            "  --probs-abs-tol   gates speaker_probs max-abs vs NeMo (default 5e-2;\n"
            "                    raise to 1.5e-1 for q4_0)\n", argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_dir   = argv[3];
    double enc_rel_tol   = 5e-3;
    double probs_abs_tol = 5e-2;
    for (int i = 4; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--enc-rel-tol")   enc_rel_tol   = std::atof(argv[++i]);
        else if (a == "--probs-abs-tol") probs_abs_tol = std::atof(argv[++i]);
    }

    using namespace parakeet;
    std::fprintf(stderr, "[sf-parity] loading %s\n", gguf_path.c_str());
    ParakeetCtcModel model;
    // Force CPU encoder: this harness gates operator parity vs NeMo's FP32
    // reference; backend-induced drift (CPU<->GPU) is gated separately by
    // test-vk-vs-cpu and would otherwise mask real encoder regressions here.
    if (int rc = load_from_gguf(gguf_path, model, 0, 0, false); rc != 0) return 3;
    if (model.model_type != ParakeetModelType::SORTFORMER) {
        std::fprintf(stderr, "  error: expected Sortformer model\n");
        return 3;
    }

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) return 4;
    std::fprintf(stderr, "[sf-parity] wav: %zu samples @ %d Hz\n", samples.size(), sr);

    {
        float peak = 0.0f;
        for (float v : samples) if (v > peak) peak = v;
        const float inv = 1.0f / (peak + 1e-8f);
        for (float & v : samples) v *= inv;
        std::fprintf(stderr, "[sf-parity] peak-normalised input: peak was %.4f\n", peak);
    }

    std::vector<float> mel_cpp;
    int n_mel_frames_cpp = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model.mel_cfg, mel_cpp, n_mel_frames_cpp); rc != 0) return 5;

    {
        std::vector<float> mel_ref;
        std::vector<int64_t> shape;
        if (load_npy_f32(ref_dir + "/mel.npy", mel_ref, shape) != 0) {
            std::fprintf(stderr, "[sf-parity] WARN: %s/mel.npy missing, skipping\n", ref_dir.c_str());
        } else {
            const int n_mels = (int) shape[0];
            const int T_ref  = (int) shape[1];
            std::vector<float> mel_ref_tf((size_t) T_ref * n_mels);
            for (int t = 0; t < T_ref; ++t)
                for (int m = 0; m < n_mels; ++m)
                    mel_ref_tf[(size_t) t * n_mels + m] = mel_ref[(size_t) m * T_ref + t];
            const size_t common = (size_t) std::min(n_mel_frames_cpp, T_ref) * n_mels;
            std::vector<float> a(mel_cpp.begin(), mel_cpp.begin() + common);
            std::vector<float> b(mel_ref_tf.begin(), mel_ref_tf.begin() + common);
            double max_abs = 0, rel = 0;
            parity(a, b, max_abs, rel);
            std::fprintf(stderr, "[sf-parity] mel  : max_abs=%.4e rel=%.4e  (%s)\n",
                         max_abs, rel, rel < 5e-3 ? "PASS" : "FAIL");
        }
    }

    EncoderOutputs enc_out;
    if (int rc = run_encoder(model, mel_cpp.data(), n_mel_frames_cpp, model.mel_cfg.n_mels, enc_out); rc != 0) return 6;
    std::fprintf(stderr, "[sf-parity] encoder: cpp frames=%d d_model=%d\n",
                 enc_out.n_enc_frames, enc_out.d_model);

    std::vector<float> enc_ref;
    std::vector<int64_t> shape;
    if (load_npy_f32(ref_dir + "/encoder_out.npy", enc_ref, shape) != 0) {
        std::fprintf(stderr, "[sf-parity] missing %s/encoder_out.npy\n", ref_dir.c_str());
        return 7;
    }
    const int T_ref = (int) shape[0];
    const int D_ref = (int) shape[1];
    std::fprintf(stderr, "[sf-parity] encoder ref: (%d, %d)\n", T_ref, D_ref);

    const size_t common = (size_t) std::min(enc_out.n_enc_frames, T_ref) * enc_out.d_model;
    std::vector<float> a(enc_out.encoder_out.begin(), enc_out.encoder_out.begin() + common);
    std::vector<float> b(enc_ref.begin(), enc_ref.begin() + common);
    double max_abs_enc = 0, rel_enc = 0;
    parity(a, b, max_abs_enc, rel_enc);
    const bool enc_pass = rel_enc < enc_rel_tol;
    std::fprintf(stderr, "[sf-parity] enc  : max_abs=%.4e rel=%.4e  (%s, rel tol=%.1e)\n",
                 max_abs_enc, rel_enc, enc_pass ? "PASS" : "FAIL", enc_rel_tol);

    ggml_backend_t backend = model_active_backend(model);
    if (!backend) { std::fprintf(stderr, "  error: no active ggml backend\n"); return 8; }
    SortformerDiarizationOptions dopts;
    SortformerDiarizationResult dres;
    if (sortformer_diarize_ggml(model,
                                enc_out.encoder_out.data(),
                                enc_out.n_enc_frames, enc_out.d_model,
                                backend, dopts, dres) != 0) return 9;

    int worst = 0;

    {
        std::vector<float> probs_ref;
        std::vector<int64_t> sshape;
        if (load_npy_f32(ref_dir + "/speaker_probs.npy", probs_ref, sshape) == 0) {
            const size_t n = std::min(dres.speaker_probs.size(), probs_ref.size());
            std::vector<float> aa(dres.speaker_probs.begin(), dres.speaker_probs.begin() + n);
            std::vector<float> bb(probs_ref.begin(), probs_ref.begin() + n);
            double max_abs = 0, rel = 0;
            parity(aa, bb, max_abs, rel);
            const bool ok = max_abs < probs_abs_tol;
            std::fprintf(stderr, "[sf-parity] probs: max_abs=%.4e rel=%.4e  (%s, max_abs tol=%.1e)\n",
                         max_abs, rel, ok ? "PASS" : "FAIL", probs_abs_tol);
            if (!ok) worst = 1;
        }
    }

    int spk_active[32] = {0};
    for (int t = 0; t < dres.n_frames; ++t) {
        for (int s = 0; s < dres.num_spks && s < 32; ++s) {
            if (dres.speaker_probs[(size_t) t * dres.num_spks + s] > dopts.threshold) {
                ++spk_active[s];
            }
        }
    }
    std::fprintf(stderr, "[sf-parity] frames=%d num_spks=%d frame_stride=%.4fs\n",
                 dres.n_frames, dres.num_spks, dres.frame_stride_s);
    for (int s = 0; s < dres.num_spks; ++s) {
        std::fprintf(stderr, "[sf-parity]   speaker %d active in %d frames (%.1f%%)\n",
                     s, spk_active[s], 100.0 * spk_active[s] / std::max(1, dres.n_frames));
    }
    std::fprintf(stderr, "[sf-parity] segments=%zu\n", dres.segments.size());
    for (size_t i = 0; i < dres.segments.size() && i < 8; ++i) {
        std::fprintf(stderr, "[sf-parity]   seg[%zu]: spk=%d  %.2fs - %.2fs\n",
                     i, dres.segments[i].speaker_id,
                     dres.segments[i].start_s, dres.segments[i].end_s);
    }

    return (enc_pass && worst == 0) ? 0 : 1;
}
