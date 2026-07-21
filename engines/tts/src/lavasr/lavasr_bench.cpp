// LavaSR two-stage bench + by-ear harness: denoise() -> enhance() on a wav or
// synthetic tone; reports per-stage wall-time + RTF over N runs (warmup dropped).

#include "tts-cpp/lavasr/denoiser.h"
#include "tts-cpp/lavasr/enhancer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

using clk  = std::chrono::steady_clock;
using ms_t = std::chrono::duration<double, std::milli>;

namespace {

constexpr double kPi = 3.14159265358979323846;

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx  = p * (v.size() - 1);
    size_t lo   = (size_t) idx;
    size_t hi   = std::min(lo + 1, v.size() - 1);
    double frac = idx - (double) lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}
double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }
double mean(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double s = 0; for (double x : v) s += x; return s / (double) v.size();
}
double minv(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double m = v[0]; for (double x : v) m = std::min(m, x); return m;
}

// Manual 16-bit PCM RIFF writer (same shape as supertonic_cli's write_wav).
void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output wav: " + path);
    uint32_t n = (uint32_t) wav.size();
    uint32_t byte_rate = (uint32_t) sr * 2;
    uint32_t data_size = n * 2;
    uint32_t chunk_size = 36 + data_size;
    uint16_t fmt = 1, channels = 1, align = 2, bps = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f); std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, f); std::fwrite(&fmt, 2, 1, f);
    std::fwrite(&channels, 2, 1, f); std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f); std::fwrite(&align, 2, 1, f);
    std::fwrite(&bps, 2, 1, f); std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float   c = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t) std::lrintf(c * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

std::vector<float> load_wav_mono(const std::string & path, int & sr) {
    unsigned int  ch = 0, rate = 0;
    drwav_uint64  frames = 0;
    float * d = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &rate, &frames, nullptr);
    if (!d) throw std::runtime_error("cannot read wav: " + path);
    std::vector<float> mono((size_t) frames);
    const unsigned int nch = std::max(1u, ch);
    for (drwav_uint64 i = 0; i < frames; i++) {
        double s = 0.0;
        for (unsigned int c = 0; c < nch; c++) s += d[i * nch + c];
        mono[(size_t) i] = (float) (s / nch);
    }
    drwav_free(d, nullptr);
    sr = (int) rate;
    return mono;
}

// Synthetic speech-band signal.  RTF is content-independent (no early-exit in
// the pipeline), so a fixed tone of N seconds is a valid, reproducible bench.
std::vector<float> synth_signal(double seconds, int rate) {
    size_t n = (size_t) (seconds * rate);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        double t = (double) i / rate;
        x[i] = (float) (0.15 * std::sin(2 * kPi * 180.0 * t) +
                        0.10 * std::sin(2 * kPi * 440.0 * t) +
                        0.05 * std::sin(2 * kPi * 900.0 * t));
    }
    return x;
}

void report(const char * name, const std::vector<double> & ms, double audio_s) {
    if (ms.empty()) { std::printf("  %-10s n=0\n", name); return; }
    double md = median(ms);
    std::printf("  %-10s min=%8.1f  med=%8.1f  mean=%8.1f  max=%8.1f ms   RTF(med)=%.3f\n",
                name, minv(ms), md, mean(ms), *std::max_element(ms.begin(), ms.end()),
                (md / 1000.0) / audio_s);
}

void usage(const char * a0) {
    std::fprintf(stderr,
        "usage: %s --denoiser DN.gguf --enhancer ENH.gguf\n"
        "          [--in speech.wav] [--seconds 6] [--in-rate 24000]\n"
        "          [--runs 5] [--warmup 1] [--n-gpu-layers 0]\n"
        "          [--out-denoised dn.wav] [--out-enhanced enh.wav]\n", a0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string denoiser_path, enhancer_path, in_wav, out_dn, out_enh;
    double seconds = 6.0;
    int    in_rate = 24000, runs = 5, warmup = 1, n_gpu_layers = 0, dn_gpu_layers = 0;
    bool   verify = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * f) {
            if (i + 1 >= argc) throw std::runtime_error(std::string(f) + " requires a value");
            return std::string(argv[++i]);
        };
        if      (a == "--denoiser")      denoiser_path = next("--denoiser");
        else if (a == "--enhancer")      enhancer_path = next("--enhancer");
        else if (a == "--in")            in_wav        = next("--in");
        else if (a == "--seconds")       seconds       = std::stod(next("--seconds"));
        else if (a == "--in-rate")       in_rate       = std::stoi(next("--in-rate"));
        else if (a == "--runs")          runs          = std::stoi(next("--runs"));
        else if (a == "--warmup")        warmup        = std::stoi(next("--warmup"));
        else if (a == "--n-gpu-layers")  n_gpu_layers  = std::stoi(next("--n-gpu-layers"));
        else if (a == "--dn-gpu-layers") dn_gpu_layers = std::stoi(next("--dn-gpu-layers"));
        else if (a == "--verify")        verify        = true;
        else if (a == "--out-denoised")  out_dn        = next("--out-denoised");
        else if (a == "--out-enhanced")  out_enh       = next("--out-enhanced");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (denoiser_path.empty() || enhancer_path.empty()) { usage(argv[0]); return 2; }

    std::unique_ptr<tts_cpp::lavasr::Denoiser> dn;
    std::unique_ptr<tts_cpp::lavasr::Enhancer> enh;
    try {
        dn  = tts_cpp::lavasr::Denoiser::load(denoiser_path, dn_gpu_layers);
        tts_cpp::lavasr::EnhancerOptions eopts;
        eopts.use_gpu = n_gpu_layers > 0;
        eopts.verbose = true;
        enh = tts_cpp::lavasr::Enhancer::load(enhancer_path, eopts);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "load failed: %s\n", e.what());
        return 1;
    }

    int                sr = in_rate;
    std::vector<float> pcm;
    if (!in_wav.empty()) {
        try { pcm = load_wav_mono(in_wav, sr); }
        catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    } else {
        pcm = synth_signal(seconds, in_rate);
    }
    if (pcm.empty()) { std::fprintf(stderr, "empty input\n"); return 1; }
    const double audio_s = (double) pcm.size() / sr;

    std::vector<double> dn_ms, enh_ms, tot_ms;
    std::vector<float>  cleaned, enhanced;
    const int total_runs = runs + warmup;
    for (int r = 0; r < total_runs; ++r) {
        const bool rec = r >= warmup;
        auto t0 = clk::now();
        cleaned = dn->denoise(pcm, sr);
        auto t1 = clk::now();
        enhanced = enh->enhance(cleaned, sr);
        auto t2 = clk::now();
        double d = ms_t(t1 - t0).count(), e = ms_t(t2 - t1).count(), tt = ms_t(t2 - t0).count();
        if (rec) { dn_ms.push_back(d); enh_ms.push_back(e); tot_ms.push_back(tt); }
        std::fprintf(stderr, "[run %d/%d]%s denoise=%.1f enhance=%.1f total=%.1f ms\n",
                     r + 1, total_runs, rec ? "" : " (warmup)", d, e, tt);
    }

    std::printf("\nLavaSR bench\n");
    std::printf("  denoiser backend: %s\n", dn->backend_name().c_str());
    std::printf("  enhancer backend: %s\n", enh->backend_name().c_str());
    std::printf("  input: %s (%.3fs @ %d Hz, %zu samples)\n",
                in_wav.empty() ? "synthetic" : in_wav.c_str(), audio_s, sr, pcm.size());
    std::printf("  enhancer out rate: %d Hz\n", enh->output_sample_rate());
    std::printf("  denoiser dn_gpu_layers: %d (0=scalar, >0=ggml-GPU, <0=ggml-CPU)\n", dn_gpu_layers);
    std::printf("  enhancer n_gpu_layers: %d (0=scalar, >0=ggml-GPU, <0=ggml-CPU)\n", n_gpu_layers);
    std::printf("  runs: %d (warmup dropped: %d)\n\n", runs, warmup);
    report("denoise", dn_ms, audio_s);
    report("enhance", enh_ms, audio_s);
    report("total",   tot_ms, audio_s);

    // On-device parity gate: re-run the denoiser on the scalar CPU path and
    // compare against the selected-backend cleaned output.
    if (verify && dn_gpu_layers != 0) {
        try {
            auto               dn_ref = tts_cpp::lavasr::Denoiser::load(denoiser_path, 0);
            std::vector<float> ref    = dn_ref->denoise(pcm, sr);
            double             se = 0.0, sr2 = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
            float              m = 0.0f, g = 0.0f;
            const size_t       n = std::min(ref.size(), cleaned.size());
            for (size_t i = 0; i < n; i++) {
                float d = cleaned[i] - ref[i];
                m = std::max(m, std::fabs(d)); g = std::max(g, std::fabs(ref[i]));
                se += (double) d * d; sr2 += (double) ref[i] * ref[i];
                dot += (double) cleaned[i] * ref[i];
                na += (double) cleaned[i] * cleaned[i]; nb += (double) ref[i] * ref[i];
            }
            std::printf("  DENOISE VERIFY vs scalar: n=%zu cos_sim=%.6f nrmse=%.3e max_abs=%.3e rel=%.3e\n",
                        n, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12), std::sqrt(se / (sr2 + 1e-12)),
                        m, m / (g + 1e-9f));
        } catch (const std::exception & e) {
            std::printf("  DENOISE VERIFY failed: %s\n", e.what());
        }
    }

    // On-device parity gate: re-run the enhancer on the ggml-CPU path and
    // compare against the selected-backend output (relative, scale-aware).
    if (verify && n_gpu_layers != 0) {
        try {
            auto enh_ref = tts_cpp::lavasr::Enhancer::load(enhancer_path, {});
            std::vector<float> ref = enh_ref->enhance(cleaned, sr);
            float m = 0.0f, g = 0.0f;
            double se = 0.0, sr2 = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
            const size_t n = std::min(ref.size(), enhanced.size());
            for (size_t i = 0; i < n; i++) {
                float d = enhanced[i] - ref[i];
                m = std::max(m, std::fabs(d));
                g = std::max(g, std::fabs(ref[i]));
                se += (double) d * d; sr2 += (double) ref[i] * ref[i];
                dot += (double) enhanced[i] * ref[i];
                na += (double) enhanced[i] * enhanced[i]; nb += (double) ref[i] * ref[i];
            }
            const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
            const double nrmse = std::sqrt(se / (sr2 + 1e-12));  // rms error / rms signal
            std::printf("  ENHANCE VERIFY vs ggml-cpu: n=%zu cos_sim=%.6f nrmse=%.3e max_abs=%.3e rel=%.3e\n",
                        n, cos, nrmse, m, m / (g + 1e-9f));
        } catch (const std::exception & e) {
            std::printf("  VERIFY failed: %s\n", e.what());
        }
    }

    if (!out_dn.empty())  { write_wav(out_dn,  cleaned,  sr); std::printf("  wrote %s\n", out_dn.c_str()); }
    if (!out_enh.empty()) { write_wav(out_enh, enhanced, enh->output_sample_rate());
                            std::printf("  wrote %s\n", out_enh.c_str()); }
    return 0;
}
