// EOU decoder parity: the scalar host decode (n_gpu_layers=0 runtime) and
// the ggml graph decode (n_gpu_layers=1 runtime) must produce identical
// token IDs AND identical <EOU> segment boundaries when fed the SAME
// encoder output. Greedy decoding is deterministic, so this is an exact
// integer comparison.
//
// Unlike test_tdt_decoder_parity.cpp (short clips, full pipeline per leg),
// the encoder here runs ONCE on the CPU model and its output is fed to
// both decoders. This isolates the decoder — the thing the dual-path port
// changed — from cross-backend encoder numerics, which on minutes-long
// audio can legitimately flip a token upstream of the decoder (and Vulkan
// has a known pre-existing long-audio encoder divergence).
//
// On a GPU-enabled build (Metal / Vulkan / CUDA) the graph leg exercises
// the span-batched joint, the fused LSTM+joint graph, the on-device argmax
// and the <EOU> on-device state reset. On a CPU-only build both legs take
// the host path and the test degenerates to a self-consistency check.
//
// Usage:
//   test-eou-decoder-parity <parakeet-eou.gguf> <wav>
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "parakeet_eou.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Decode the given encoder output with a runtime prepared from a model
// loaded at the requested n_gpu_layers (0 -> scalar host path, 1 -> ggml
// graph path on the compiled GPU backend). Loads its own model so that
// backend selection — and therefore the decode-path split in
// eou_prepare_runtime — is honoured per-call.
int decode_eou(const std::string & gguf_path,
               const std::vector<float> & encoder_out,
               int T_enc, int D_enc,
               int n_gpu_layers,
               std::vector<int32_t> & out_tokens,
               std::vector<int>     & out_boundaries,
               std::string          & out_text) {
    using namespace parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf_path, model, /*n_threads=*/0,
                                 n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "  load_from_gguf failed rc=%d\n", rc);
        return 100 + rc;
    }
    if (model.model_type != ParakeetModelType::EOU) {
        std::fprintf(stderr, "  error: expected EOU model in %s\n",
                     gguf_path.c_str());
        return 110;
    }

    EouRuntimeWeights rt;
    if (int rc = eou_prepare_runtime(model, rt); rc != 0) {
        std::fprintf(stderr, "  eou_prepare_runtime failed rc=%d\n", rc);
        return 150 + rc;
    }
    std::fprintf(stderr, "  n_gpu_layers=%d -> %s decode path\n",
                 n_gpu_layers, rt.use_graphs ? "ggml-graph" : "scalar host");

    EouDecodeOptions dopts;
    dopts.max_symbols_per_step = model.encoder_cfg.eou_max_symbols_per_step;
    EouDecodeResult dres;
    if (int rc = eou_greedy_decode(model, rt,
                                   encoder_out.data(),
                                   T_enc, D_enc,
                                   dopts, dres); rc != 0) {
        std::fprintf(stderr, "  eou_greedy_decode failed rc=%d\n", rc);
        return 160 + rc;
    }

    out_tokens = std::move(dres.token_ids);
    out_boundaries.clear();
    out_boundaries.reserve(dres.segments.size());
    for (const auto & s : dres.segments) out_boundaries.push_back(s.token_index);
    out_text = std::move(dres.text);
    return 0;
}

// Mel + encoder on the CPU-loaded model: the shared input both decode
// paths consume.
int encode_reference(const std::string & gguf_path,
                     const std::string & wav_path,
                     std::vector<float> & out_encoder,
                     int & out_T, int & out_D) {
    using namespace parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf_path, model, /*n_threads=*/0,
                                 /*n_gpu_layers=*/0, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "  load_from_gguf failed rc=%d\n", rc);
        return 100 + rc;
    }

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "  load_wav failed rc=%d\n", rc);
        return 120 + rc;
    }

    std::vector<float> mel;
    int n_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model.mel_cfg, mel, n_frames); rc != 0) {
        std::fprintf(stderr, "  compute_log_mel failed rc=%d\n", rc);
        return 130 + rc;
    }

    EncoderOutputs enc_out;
    if (int rc = run_encoder(model, mel.data(), n_frames,
                             model.mel_cfg.n_mels, enc_out); rc != 0) {
        std::fprintf(stderr, "  run_encoder failed rc=%d\n", rc);
        return 140 + rc;
    }

    out_encoder = std::move(enc_out.encoder_out);
    out_T = enc_out.n_enc_frames;
    out_D = enc_out.d_model;
    return 0;
}

template <typename T>
void print_first_diff(const std::vector<T> & a, const std::vector<T> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            std::fprintf(stderr, "  first diff at index %zu: a=%d  b=%d\n",
                         i, (int) a[i], (int) b[i]);
            return;
        }
    }
    std::fprintf(stderr, "  no per-index diff in shared prefix; lengths %zu vs %zu\n",
                 a.size(), b.size());
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <parakeet-eou.gguf> <wav>\n"
            "\n"
            "Cross-checks the EOU greedy decoder: n_gpu_layers=0 (scalar host\n"
            "path) vs n_gpu_layers=1 (ggml graph path) must produce identical\n"
            "token IDs and identical <EOU> segment boundaries.\n",
            argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];

    std::fprintf(stderr, "[eou-decode-parity] running CPU mel + encoder (shared input)...\n");
    std::vector<float> encoder_out;
    int T_enc = 0, D_enc = 0;
    if (int rc = encode_reference(gguf_path, wav_path, encoder_out, T_enc, D_enc); rc != 0) {
        return rc;
    }
    std::fprintf(stderr, "[eou-decode-parity] encoder: %d frames x %d\n", T_enc, D_enc);

    std::fprintf(stderr, "[eou-decode-parity] running host decode (n_gpu_layers=0)...\n");
    std::vector<int32_t> ids_cpu;
    std::vector<int>     seg_cpu;
    std::string          text_cpu;
    if (int rc = decode_eou(gguf_path, encoder_out, T_enc, D_enc, 0,
                            ids_cpu, seg_cpu, text_cpu); rc != 0) {
        return rc;
    }
    std::fprintf(stderr, "[eou-decode-parity] CPU: tokens=%zu segments=%zu text=%.80s%s\n",
                 ids_cpu.size(), seg_cpu.size(), text_cpu.c_str(),
                 text_cpu.size() > 80 ? "..." : "");

    std::fprintf(stderr, "[eou-decode-parity] running graph decode (n_gpu_layers=1)...\n");
    std::vector<int32_t> ids_gpu;
    std::vector<int>     seg_gpu;
    std::string          text_gpu;
    if (int rc = decode_eou(gguf_path, encoder_out, T_enc, D_enc, 1,
                            ids_gpu, seg_gpu, text_gpu); rc != 0) {
        return rc;
    }
    std::fprintf(stderr, "[eou-decode-parity] GPU: tokens=%zu segments=%zu text=%.80s%s\n",
                 ids_gpu.size(), seg_gpu.size(), text_gpu.c_str(),
                 text_gpu.size() > 80 ? "..." : "");

    bool ok = true;
    if (ids_cpu.size() != ids_gpu.size() ||
        !std::equal(ids_cpu.begin(), ids_cpu.end(), ids_gpu.begin())) {
        std::fprintf(stderr,
            "[eou-decode-parity] FAIL: CPU vs graph token IDs differ\n");
        print_first_diff(ids_cpu, ids_gpu);
        ok = false;
    } else {
        std::fprintf(stderr,
            "[eou-decode-parity] PASS: CPU vs graph token IDs match (%zu tokens)\n",
            ids_cpu.size());
    }

    if (seg_cpu != seg_gpu) {
        std::fprintf(stderr,
            "[eou-decode-parity] FAIL: CPU vs graph <EOU> segment boundaries differ\n");
        print_first_diff(seg_cpu, seg_gpu);
        ok = false;
    } else {
        std::fprintf(stderr,
            "[eou-decode-parity] PASS: CPU vs graph <EOU> segment boundaries match (%zu)\n",
            seg_cpu.size());
    }

    if (ok) {
        std::fprintf(stderr, "[eou-decode-parity] all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "[eou-decode-parity] one or more checks failed\n");
    return 1;
}
