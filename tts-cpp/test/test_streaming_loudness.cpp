// QVAC-21118: engine-level regression test for chunk-streaming loudness.
//
// Synthesizes the same text twice through one tts_cpp::chatterbox::Engine --
// the exact API the qvac tts-ggml addon uses:
//   * batch     : engine.synthesize(text)            (uses cfm_steps default)
//   * streaming : engine.synthesize(text, on_chunk)  (uses stream_cfm_steps)
// with a low stream_cfm_steps (the realistic low-latency config the addon's
// `cfmSteps` produces) -- and asserts that the streaming output lands at the
// same loudness as batch and does not clip.
//
// PRE-FIX this FAILS: a low stream_cfm_steps (1-2) under-integrates the
// Multilingual model's standard 10-step CFM, and the per-chunk streaming path
// collapses (peak ~0.99, RMS 4-9x batch, dying tail).  The fix floors the
// streaming CFM step count to the model's n_timesteps for standard CFM, so the
// streaming output matches batch.  POST-FIX both paths land at the same level.
//
// Pure library link (no whisper / no subprocess).  Needs the Chatterbox
// Multilingual GGUFs + a baked voice dir, so CMake registers it DISABLED
// unless those fixtures are present (mirrors test-eos-roundtrip).

#include "tts-cpp/chatterbox/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Stats {
    double peak = 0.0;
    double rms  = 0.0;
    int    clipped = 0;
    double win_cv = 0.0;   // coefficient of variation of windowed RMS (pumping proxy)
};

Stats measure(const std::vector<float> & x) {
    Stats s;
    if (x.empty()) return s;
    double sq = 0.0;
    for (float v : x) {
        const double a = std::fabs((double) v);
        if (a > s.peak) s.peak = a;
        sq += (double) v * (double) v;
        if (a > 0.999) s.clipped++;
    }
    s.rms = std::sqrt(sq / (double) x.size());

    const size_t w = 2400;  // 100 ms @ 24 kHz
    std::vector<double> wr;
    for (size_t i = 0; i + w <= x.size(); i += w) {
        double s2 = 0.0;
        for (size_t j = 0; j < w; ++j) { const double v = x[i + j]; s2 += v * v; }
        const double r = std::sqrt(s2 / (double) w);
        if (r > 1e-4) wr.push_back(r);   // ignore silent gaps
    }
    if (wr.size() > 1) {
        double m = 0.0; for (double r : wr) m += r; m /= (double) wr.size();
        double v = 0.0; for (double r : wr) v += (r - m) * (r - m); v /= (double) wr.size();
        if (m > 1e-9) s.win_cv = std::sqrt(v) / m;
    }
    return s;
}

// Minimal 16-bit mono PCM WAV writer (for manual A/B inspection via --out-*).
void write_wav_s16(const std::string & path, const std::vector<float> & pcm, int sr) {
    if (path.empty()) return;
    std::vector<int16_t> s16(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        float v = pcm[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        s16[i] = (int16_t) std::lrint(v * 32767.0f);
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "warning: cannot write %s\n", path.c_str()); return; }
    const uint32_t data_bytes = (uint32_t)(s16.size() * sizeof(int16_t));
    auto u32 = [&](uint32_t x){ f.write(reinterpret_cast<const char*>(&x), 4); };
    auto u16 = [&](uint16_t x){ f.write(reinterpret_cast<const char*>(&x), 2); };
    f.write("RIFF", 4); u32(36 + data_bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16); u16(1); u16(1);
    u32((uint32_t) sr); u32((uint32_t)(sr * 2)); u16(2); u16(16);
    f.write("data", 4); u32(data_bytes);
    f.write(reinterpret_cast<const char*>(s16.data()), data_bytes);
}

struct Args {
    std::string t3, s3gen, voice_dir, lang = "en";
    std::string text =
        "The quick brown fox jumps over the lazy dog, and then it ran away "
        "quickly into the dark forest.";
    int seed = 12, gpu = 0, chunk = 25, cfm = 2;
    std::string out_batch, out_stream; // optional wav dumps for manual A/B
    // Tolerances (generous: this is a clipping / gross-loudness gate).
    double max_peak  = 0.90;   // streaming must not approach full scale
    double min_ratio = 0.50;   // streaming RMS / batch RMS lower bound (-6 dB)
    double max_ratio = 2.00;   // upper bound (+6 dB)
};

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char * n) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", n); return nullptr; }
            return argv[++i];
        };
        if      (s == "--t3")          { auto v = next("--t3");          if (!v) return false; a.t3 = v; }
        else if (s == "--s3gen")       { auto v = next("--s3gen");       if (!v) return false; a.s3gen = v; }
        else if (s == "--voice-dir")   { auto v = next("--voice-dir");   if (!v) return false; a.voice_dir = v; }
        else if (s == "--lang")        { auto v = next("--lang");        if (!v) return false; a.lang = v; }
        else if (s == "--text")        { auto v = next("--text");        if (!v) return false; a.text = v; }
        else if (s == "--seed")        { auto v = next("--seed");        if (!v) return false; a.seed = std::atoi(v); }
        else if (s == "--n-gpu-layers"){ auto v = next("--n-gpu-layers");if (!v) return false; a.gpu = std::atoi(v); }
        else if (s == "--chunk-tokens"){ auto v = next("--chunk-tokens");if (!v) return false; a.chunk = std::atoi(v); }
        else if (s == "--cfm-steps")   { auto v = next("--cfm-steps");   if (!v) return false; a.cfm = std::atoi(v); }
        else if (s == "--max-peak")    { auto v = next("--max-peak");    if (!v) return false; a.max_peak = std::atof(v); }
        else if (s == "--max-ratio")   { auto v = next("--max-ratio");   if (!v) return false; a.max_ratio = std::atof(v); }
        else if (s == "--min-ratio")   { auto v = next("--min-ratio");   if (!v) return false; a.min_ratio = std::atof(v); }
        else if (s == "--out-batch")   { auto v = next("--out-batch");   if (!v) return false; a.out_batch = v; }
        else if (s == "--out-stream")  { auto v = next("--out-stream");  if (!v) return false; a.out_stream = v; }
        else { fprintf(stderr, "unknown arg: %s\n", s.c_str()); return false; }
    }
    if (a.t3.empty() || a.s3gen.empty()) {
        fprintf(stderr, "usage: test-streaming-loudness --t3 T3.gguf --s3gen S3.gguf "
                        "[--voice-dir DIR] [--lang en] [--text ...] [--cfm-steps 2] "
                        "[--chunk-tokens 25] [--seed 12] [--n-gpu-layers 0]\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 64;

    using namespace tts_cpp::chatterbox;
    EngineOptions opts;
    opts.t3_gguf_path    = a.t3;
    opts.s3gen_gguf_path = a.s3gen;
    opts.voice_dir       = a.voice_dir;
    opts.language        = a.lang;
    opts.seed            = a.seed;
    opts.n_gpu_layers    = a.gpu;
    opts.stream_chunk_tokens = a.chunk;
    opts.stream_cfm_steps    = a.cfm;   // low -> exercises the streaming CFM floor

    Stats sb, ss;
    std::vector<float> stream_pcm, result_pcm;
    int n_chunks = 0;
    try {
        Engine engine(opts);

        // Batch (no callback -> batch path, uses cfm_steps default == golden).
        SynthesisResult rb = engine.synthesize(a.text);
        sb = measure(rb.pcm);
        write_wav_s16(a.out_batch, rb.pcm, 24000);

        // Streaming (callback -> chunked path, uses stream_cfm_steps).
        SynthesisResult rs = engine.synthesize(
            a.text,
            [&](const float * pcm, std::size_t n, int idx, bool last) {
                double s = 0.0, pk = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double v = pcm[i];
                    s += v * v;
                    pk = std::max(pk, std::fabs(v));
                }
                const double r = n ? std::sqrt(s / (double) n) : 0.0;
                fprintf(stderr, "  chunk %2d: n=%6zu rms=%.4f peak=%.3f%s\n",
                        idx, n, r, pk, last ? "  (last)" : "");
                stream_pcm.insert(stream_pcm.end(), pcm, pcm + n);
                ++n_chunks;
            });
        result_pcm = rs.pcm;
        ss = measure(stream_pcm);
        write_wav_s16(a.out_stream, stream_pcm, 24000);
    } catch (const std::exception & e) {
        fprintf(stderr, "synthesis failed: %s\n", e.what());
        return 2;
    }

    const double ratio = sb.rms > 1e-9 ? ss.rms / sb.rms : 0.0;
    fprintf(stderr,
        "batch : peak=%.3f rms=%.4f clipped=%d\n"
        "stream: peak=%.3f rms=%.4f clipped=%d win_cv=%.2f chunks=%d (cfm=%d chunk=%d)\n"
        "rms_ratio stream/batch = %.2f (target ~1.0)\n",
        sb.peak, sb.rms, sb.clipped,
        ss.peak, ss.rms, ss.clipped, ss.win_cv, n_chunks, a.cfm, a.chunk, ratio);

    int fails = 0;
    if (result_pcm.size() != stream_pcm.size()) {
        fprintf(stderr, "FAIL: result.pcm (%zu) != concat(callback chunks) (%zu)\n",
                result_pcm.size(), stream_pcm.size());
        ++fails;
    }
    if (ss.peak > a.max_peak) {
        fprintf(stderr, "FAIL: streaming peak %.3f > %.3f (near full-scale / clipping)\n",
                ss.peak, a.max_peak);
        ++fails;
    }
    if (ss.clipped > 0) {
        fprintf(stderr, "FAIL: streaming clipped %d samples\n", ss.clipped);
        ++fails;
    }
    if (ratio < a.min_ratio || ratio > a.max_ratio) {
        fprintf(stderr, "FAIL: rms_ratio %.2f outside [%.2f, %.2f] "
                        "(streaming loudness != batch)\n", ratio, a.min_ratio, a.max_ratio);
        ++fails;
    }

    if (fails) { fprintf(stderr, "RESULT: FAIL (%d check(s))\n", fails); return 1; }
    fprintf(stderr, "RESULT: PASS\n");
    return 0;
}
