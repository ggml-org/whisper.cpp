// CPU vs Vulkan encoder bisect.
//
// Loads the model on CPU and Vulkan, runs both encoders on identical mel, and
// reports per-stage parity (max abs diff + relative L2).
//
// Usage:
//   test-vk-vs-cpu <gguf> <wav>
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

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

bool any_nan_or_inf(const std::vector<float> & v) {
    for (float x : v) {
        if (std::isnan(x) || std::isinf(x)) return true;
    }
    return false;
}

// rel_tol: stages accumulate fp16 rounding across 24 blocks, so later
// stages need a looser gate. 5e-2 is well above observed noise (~1.4e-2
// on block_last_out) but catches real divergence (broken GLU was >0.8).
bool summarize(const char * name,
               const std::vector<float> & cpu,
               const std::vector<float> & vk,
               double rel_tol = 5e-2) {
    double max_abs = 0.0, rel = 0.0;
    compute_parity(vk, cpu, max_abs, rel);
    const bool nan_cpu = any_nan_or_inf(cpu);
    const bool nan_vk  = any_nan_or_inf(vk);
    const bool pass = !nan_cpu && !nan_vk && rel < rel_tol;
    std::fprintf(stderr,
                 "%s stage %-25s n=%zu  max_abs=%9.3e  rel=%9.3e (tol=%7.1e)  %s%s\n",
                 pass ? "PASS" : "FAIL",
                 name, cpu.size(), max_abs, rel, rel_tol,
                 nan_cpu ? "[CPU has NaN/Inf!] " : "",
                 nan_vk  ? "[VK  has NaN/Inf!] " : "");
    return pass;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <parakeet-ctc.gguf> <wav>\n"
                     "\n"
                     "Loads the model twice (CPU + Vulkan), runs the same mel\n"
                     "through both encoders, and reports per-stage parity so\n"
                     "we can localise where Vulkan diverges.\n",
                     argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];

    using namespace parakeet;
    using namespace parakeet;

    // ---- load CPU model ----
    std::fprintf(stderr, "[*] loading CPU model...\n");
    ParakeetCtcModel m_cpu;
    if (int rc = load_from_gguf(gguf_path, m_cpu, 0, /*n_gpu_layers=*/0, false); rc != 0) {
        std::fprintf(stderr, "error: CPU load_from_gguf rc=%d\n", rc);
        return 10;
    }

    // ---- load Vulkan model ----
    std::fprintf(stderr, "[*] loading Vulkan model...\n");
    ParakeetCtcModel m_vk;
    if (int rc = load_from_gguf(gguf_path, m_vk, 0, /*n_gpu_layers=*/1, false); rc != 0) {
        std::fprintf(stderr, "error: Vulkan load_from_gguf rc=%d\n", rc);
        return 11;
    }

    // ---- read wav and compute mel on CPU (once) ----
    std::fprintf(stderr, "[*] reading wav %s\n", wav_path.c_str());
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "error: load_wav_mono_f32 rc=%d\n", rc);
        return 20;
    }
    if (sr != m_cpu.mel_cfg.sample_rate) {
        std::fprintf(stderr, "error: wav sr=%d != model sr=%d\n", sr, m_cpu.mel_cfg.sample_rate);
        return 21;
    }

    std::vector<float> mel;
    int n_mel_frames = 0;
    compute_log_mel(samples.data(), (int) samples.size(), m_cpu.mel_cfg, mel, n_mel_frames);
    std::fprintf(stderr, "[*] mel: %d frames x %d mels\n", n_mel_frames, m_cpu.mel_cfg.n_mels);

    // ---- run encoder on both backends with the same mel ----
    std::fprintf(stderr, "[*] running CPU encoder...\n");
    EncoderOutputs out_cpu;
    if (int rc = run_encoder(m_cpu, mel.data(), n_mel_frames, m_cpu.mel_cfg.n_mels, out_cpu); rc != 0) {
        std::fprintf(stderr, "error: CPU run_encoder rc=%d\n", rc);
        return 30;
    }

    std::fprintf(stderr, "[*] running Vulkan encoder...\n");
    EncoderOutputs out_vk;
    if (int rc = run_encoder(m_vk, mel.data(), n_mel_frames, m_vk.mel_cfg.n_mels, out_vk); rc != 0) {
        std::fprintf(stderr, "error: Vulkan run_encoder rc=%d\n", rc);
        return 31;
    }

    // ---- per-stage parity ----
    std::fprintf(stderr, "\n=== per-stage parity (Vulkan vs CPU baseline) ===\n");
    int n_fail = 0;
    if (!summarize("subsampling_out",   out_cpu.subsampling_out,   out_vk.subsampling_out))   ++n_fail;
    if (!summarize("block0_post_ff1",   out_cpu.block_0_post_ff1,  out_vk.block_0_post_ff1))  ++n_fail;
    if (!summarize("block0_post_attn",  out_cpu.block_0_post_attn, out_vk.block_0_post_attn)) ++n_fail;
    if (!summarize("block0_post_conv",  out_cpu.block_0_post_conv, out_vk.block_0_post_conv)) ++n_fail;
    if (!summarize("block0_post_ff2",   out_cpu.block_0_post_ff2,  out_vk.block_0_post_ff2))  ++n_fail;
    if (!summarize("block0_out",        out_cpu.block_0_out,       out_vk.block_0_out))       ++n_fail;
    if (!summarize("block_last_out",    out_cpu.block_last_out,    out_vk.block_last_out))    ++n_fail;
    if (!summarize("encoder_out",       out_cpu.encoder_out,       out_vk.encoder_out))       ++n_fail;
    if (!summarize("logits",            out_cpu.logits,            out_vk.logits))             ++n_fail;

    if (n_fail > 0) {
        std::fprintf(stderr, "\nFAILED: %d stage(s) exceeded tolerance\n", n_fail);
        return 1;
    }
    std::fprintf(stderr, "\nall stages passed\n");
    return 0;
}
