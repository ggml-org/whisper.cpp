// TDT decoder parity vs reference token IDs (NeMo dump or cross-backend).
//
// Greedy decoding is deterministic; this compares integer token IDs only.
//
// Usage:
//   test-tdt-decoder-parity <gguf> <wav> [<ref-dir>]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int load_npy_i32(const std::string & path,
                 std::vector<int32_t> & out_data) {
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

    // Crude shape parser: looks for "'shape': (N,)" or "(N, M)".
    const size_t shp_s = header.find("'shape':");
    if (shp_s == std::string::npos) return 3;
    const size_t lp = header.find('(', shp_s);
    const size_t rp = header.find(')', lp);
    const std::string shape_str = header.substr(lp + 1, rp - lp - 1);

    size_t total = 1;
    size_t pos = 0;
    bool any = false;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() &&
               (shape_str[pos] == ' ' || shape_str[pos] == ',')) ++pos;
        if (pos >= shape_str.size()) break;
        size_t end = pos;
        while (end < shape_str.size() &&
               std::isdigit(static_cast<unsigned char>(shape_str[end]))) ++end;
        if (end > pos) {
            total *= static_cast<size_t>(std::stoll(shape_str.substr(pos, end - pos)));
            any = true;
            pos = end;
        } else {
            break;
        }
    }
    if (!any) return 4;

    out_data.resize(total);
    f.read(reinterpret_cast<char*>(out_data.data()),
           total * sizeof(int32_t));
    return f ? 0 : 5;
}

// Run the full pipeline (mel -> encoder -> TDT greedy) on the given GGUF
// using the requested n_gpu_layers, returning the produced token IDs and
// the decoded transcript. Loads its own model so that backend selection is
// honoured per-call.
int transcribe_tdt(const std::string & gguf_path,
                   const std::string & wav_path,
                   int                 n_gpu_layers,
                   std::vector<int32_t> & out_tokens,
                   std::string         & out_text) {
    using namespace parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf_path, model, /*n_threads=*/0,
                                 n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "  load_from_gguf failed rc=%d\n", rc);
        return 100 + rc;
    }
    if (model.model_type != ParakeetModelType::TDT) {
        std::fprintf(stderr, "  error: expected TDT model in %s\n",
                     gguf_path.c_str());
        return 110;
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

    TdtRuntimeWeights rt;
    if (int rc = tdt_prepare_runtime(model, rt); rc != 0) {
        std::fprintf(stderr, "  tdt_prepare_runtime failed rc=%d\n", rc);
        return 150 + rc;
    }

    TdtDecodeOptions dopts;
    TdtDecodeResult  dres;
    if (int rc = tdt_greedy_decode(model, rt,
                                   enc_out.encoder_out.data(),
                                   enc_out.n_enc_frames, enc_out.d_model,
                                   dopts, dres); rc != 0) {
        std::fprintf(stderr, "  tdt_greedy_decode failed rc=%d\n", rc);
        return 160 + rc;
    }

    out_tokens = std::move(dres.token_ids);
    out_text   = std::move(dres.text);
    return 0;
}

void print_first_diff(const std::vector<int32_t> & a,
                      const std::vector<int32_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            std::fprintf(stderr, "  first diff at index %zu: a=%d  b=%d\n",
                         i, a[i], b[i]);
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
            "usage: %s <parakeet-tdt.gguf> <wav> [<ref-dir>]\n"
            "\n"
            "Validates the C++ TDT greedy decoder.\n"
            "  Pass <ref-dir> containing token_ids.npy from\n"
            "    `scripts/dump-tdt-reference.py --wav <wav>`\n"
            "  to compare against the NeMo reference.\n"
            "\n"
            "Always cross-checks the n_gpu_layers=0 (scalar CPU fallback) path\n"
            "against n_gpu_layers=1 (ggml graph path) so the new graph code\n"
            "must reproduce the proven baseline bit-for-bit.\n",
            argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string ref_dir   = (argc >= 4) ? argv[3] : "";

    // ---- Run the CPU-fallback scalar path (n_gpu_layers=0). ----
    std::fprintf(stderr, "[tdt-decode-parity] running CPU fallback (n_gpu_layers=0)...\n");
    std::vector<int32_t> ids_cpu;
    std::string text_cpu;
    if (int rc = transcribe_tdt(gguf_path, wav_path, 0, ids_cpu, text_cpu); rc != 0) {
        return rc;
    }
    std::fprintf(stderr, "[tdt-decode-parity] CPU: tokens=%zu text=%.80s%s\n",
                 ids_cpu.size(), text_cpu.c_str(),
                 text_cpu.size() > 80 ? "..." : "");

    // ---- Run the ggml-graph path (n_gpu_layers=1). On a Metal-enabled
    //      build this exercises the graph code on the GPU; on a CPU-only
    //      build n_gpu_layers=1 falls back to CPU and the call still
    //      validates that no path regressed. ----
    std::fprintf(stderr, "[tdt-decode-parity] running graph path (n_gpu_layers=1)...\n");
    std::vector<int32_t> ids_gpu;
    std::string text_gpu;
    if (int rc = transcribe_tdt(gguf_path, wav_path, 1, ids_gpu, text_gpu); rc != 0) {
        return rc;
    }
    std::fprintf(stderr, "[tdt-decode-parity] GPU: tokens=%zu text=%.80s%s\n",
                 ids_gpu.size(), text_gpu.c_str(),
                 text_gpu.size() > 80 ? "..." : "");

    bool ok = true;
    if (ids_cpu.size() != ids_gpu.size() ||
        !std::equal(ids_cpu.begin(), ids_cpu.end(), ids_gpu.begin())) {
        std::fprintf(stderr,
            "[tdt-decode-parity] FAIL: CPU vs graph token IDs differ\n");
        print_first_diff(ids_cpu, ids_gpu);
        ok = false;
    } else {
        std::fprintf(stderr,
            "[tdt-decode-parity] PASS: CPU vs graph token IDs match (%zu tokens)\n",
            ids_cpu.size());
    }

    // ---- Optional: external NeMo reference. ----
    if (!ref_dir.empty()) {
        std::vector<int32_t> ids_ref;
        const std::string p = ref_dir + "/token_ids.npy";
        if (load_npy_i32(p, ids_ref) != 0) {
            std::fprintf(stderr,
                "[tdt-decode-parity] WARN: could not load %s; skipping NeMo reference check\n",
                p.c_str());
        } else {
            if (ids_ref.size() != ids_gpu.size() ||
                !std::equal(ids_ref.begin(), ids_ref.end(), ids_gpu.begin())) {
                std::fprintf(stderr,
                    "[tdt-decode-parity] FAIL: NeMo reference (%zu tokens) vs C++ (%zu) mismatch\n",
                    ids_ref.size(), ids_gpu.size());
                print_first_diff(ids_ref, ids_gpu);
                ok = false;
            } else {
                std::fprintf(stderr,
                    "[tdt-decode-parity] PASS: NeMo reference matches (%zu tokens)\n",
                    ids_ref.size());
            }
        }
    }

    if (ok) {
        std::fprintf(stderr, "[tdt-decode-parity] all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "[tdt-decode-parity] one or more checks failed\n");
    return 1;
}
