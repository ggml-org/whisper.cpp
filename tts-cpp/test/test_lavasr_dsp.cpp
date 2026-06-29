// Unit tests for the LavaSR DSP primitives (resampler, STFT/ISTFT, Slaney mel
// filterbank, FastLR spectral crossover merge).
//
// Pure host math: no model, no ggml, no fixture, so it runs everywhere (CI
// included).  These mirror the @qvac/tts-onnx DSP unit tests so the GGML
// enhancer's pre/post-processing stays bit-comparable with the ONNX addon.

#include "lavasr/dsp/fastlr_merge.h"
#include "lavasr/dsp/mel_filterbank.h"
#include "lavasr/dsp/resampler.h"
#include "lavasr/dsp/stft_processor.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace tts_cpp::lavasr::dsp;

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__,         \
                         __LINE__);                                            \
        }                                                                      \
    } while (0)

static constexpr double kPi = 3.14159265358979323846;

static std::vector<float> generate_sine(int n, float freq, int sr,
                                         float amp = 0.5f) {
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = amp * std::sin(2.0f * static_cast<float>(kPi) * freq * i / sr);
    }
    return x;
}

static float rms(const std::vector<float> & x) {
    if (x.empty()) {
        return 0.0f;
    }
    double s = 0.0;
    for (float v : x) {
        s += static_cast<double>(v) * v;
    }
    return static_cast<float>(std::sqrt(s / x.size()));
}

static float max_abs_diff(const std::vector<float> & a,
                          const std::vector<float> & b, int from, int to) {
    float m = 0.0f;
    for (int i = from; i < to && i < (int) a.size() && i < (int) b.size(); i++) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static void test_resampler() {
    // Identity passthrough.
    {
        auto x = generate_sine(1000, 220.0f, 16000);
        auto y = Resampler::resample(x, 16000, 16000);
        CHECK(y.size() == x.size(), "resampler identity keeps length");
        CHECK(max_abs_diff(x, y, 0, (int) x.size()) == 0.0f,
              "resampler identity is exact");
    }
    // Upsample 24k -> 48k: length doubles (rounded).
    {
        auto x = generate_sine(2400, 300.0f, 24000);
        auto y = Resampler::resample(x, 24000, 48000);
        CHECK(y.size() == 4800, "resampler 24k->48k doubles length");
    }
    // Downsample 48k -> 16k: length thirds.
    {
        auto x = generate_sine(4800, 300.0f, 48000);
        auto y = Resampler::resample(x, 48000, 16000);
        CHECK(y.size() == 1600, "resampler 48k->16k thirds length");
    }
    // Energy preservation on a low-frequency tone (well below Nyquist).
    {
        auto x  = generate_sine(4800, 300.0f, 24000);
        auto y  = Resampler::resample(x, 24000, 48000);
        float r = rms(y) / std::max(rms(x), 1e-9f);
        CHECK(r > 0.85f && r < 1.15f, "resampler preserves low-freq energy");
    }
    // Round-trip up then down recovers the interior.
    {
        auto x  = generate_sine(4800, 250.0f, 24000);
        auto up = Resampler::resample(x, 24000, 48000);
        auto rt = Resampler::resample(up, 48000, 24000);
        CHECK(rt.size() == x.size(), "resampler round-trip keeps length");
        CHECK(max_abs_diff(x, rt, 200, 4600) < 0.05f,
              "resampler round-trip recovers interior");
    }
}

static void test_fft() {
    ComplexVec x(64);
    for (int i = 0; i < 64; i++) {
        x[i] = {std::sin(2.0f * (float) kPi * 5 * i / 64), 0.0f};
    }
    ComplexVec orig = x;
    StftProcessor::fft(x, false);
    StftProcessor::fft(x, true);
    float m = 0.0f;
    for (int i = 0; i < 64; i++) {
        m = std::max(m, std::fabs(x[i].real() - orig[i].real()));
        m = std::max(m, std::fabs(x[i].imag() - orig[i].imag()));
    }
    CHECK(m < 1e-3f, "fft forward+inverse round-trips");
}

static void test_stft_istft() {
    // Enhancer STFT config: n_fft=2048, hop=512, win=2048, center_pad=false.
    StftProcessor stft(2048, 512, 2048, false);
    auto          x    = generate_sine(24000, 440.0f, 48000);
    auto          spec = stft.stft(x);
    CHECK(!spec.empty(), "stft produces frames");
    CHECK(spec[0].size() == 1025, "stft has n_fft/2+1 = 1025 bins");

    auto recon = stft.istft(spec, (int) x.size());
    CHECK(recon.size() == x.size(), "istft recovers requested length");
    // Interior reconstruction (Hann + hop=win/4 satisfies COLA).
    CHECK(max_abs_diff(x, recon, 2048, 22000) < 0.02f,
          "stft->istft reconstructs the interior");
    float r = rms(recon) / std::max(rms(x), 1e-9f);
    CHECK(r > 0.9f && r < 1.1f, "stft->istft preserves energy");
}

static void test_mel() {
    MelFilterbank mel(44100, 2048, 80, 0.0f, 8000.0f);
    auto          tone    = generate_sine(24000, 1000.0f, 48000);
    auto          mel_out = mel.mel_spectrogram(tone, 512);
    CHECK(mel_out.size() == 80, "mel has 80 bands");
    CHECK(!mel_out[0].empty(), "mel has time frames");

    bool finite = true;
    for (auto & row : mel_out) {
        for (float v : row) {
            if (!std::isfinite(v)) {
                finite = false;
            }
        }
    }
    CHECK(finite, "mel output is finite");

    std::vector<float> silence(24000, 0.0f);
    auto               mel_sil = mel.mel_spectrogram(silence, 512);
    float              e_tone = 0.0f, e_sil = 0.0f;
    for (int m = 0; m < 80; m++) {
        for (size_t t = 0; t < mel_out[m].size(); t++) {
            e_tone += mel_out[m][t];
        }
        for (size_t t = 0; t < mel_sil[m].size(); t++) {
            e_sil += mel_sil[m][t];
        }
    }
    CHECK(e_tone > e_sil, "tone mel energy exceeds silence");
}

static void test_fastlr_merge() {
    const int sr = 48000;
    // Identical signals -> output ~= input.
    {
        auto x = generate_sine(8192, 1000.0f, sr);
        auto y = FastLRMerge::merge(x, x, sr, 4000, 256);
        CHECK(y.size() == x.size(), "fastlr keeps length");
        CHECK(max_abs_diff(x, y, 100, 8000) < 0.01f,
              "fastlr of identical signals is ~identity");
    }
    // Low-frequency content should come mostly from `original`.
    {
        auto enhanced = generate_sine(8192, 100.0f, sr, 1.0f);
        auto original = std::vector<float>(8192, 0.0f);
        auto y        = FastLRMerge::merge(enhanced, original, sr, 4000, 256);
        // Below cutoff, enhanced high-amplitude tone should be suppressed.
        CHECK(rms(y) < 0.3f * rms(enhanced),
              "fastlr suppresses enhanced low-freq (uses original)");
    }
    // High-frequency content should come mostly from `enhanced`.
    {
        auto enhanced = generate_sine(8192, 12000.0f, sr, 1.0f);
        auto original = std::vector<float>(8192, 0.0f);
        auto y        = FastLRMerge::merge(enhanced, original, sr, 4000, 256);
        CHECK(rms(y) > 0.7f * rms(enhanced),
              "fastlr passes enhanced high-freq");
    }
}

int main() {
    test_resampler();
    test_fft();
    test_stft_istft();
    test_mel();
    test_fastlr_merge();

    if (g_failures == 0) {
        std::printf("OK: all LavaSR DSP tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d LavaSR DSP test(s) failed\n", g_failures);
    return 1;
}
