// Smoke-test for resample_sinc. Generates a broadband test signal (multi-tone)
// in memory, round-trips it through 24 kHz -> 48 kHz -> 24 kHz, and reports
// SNR in the middle of the buffer (well past the filter transient).
//
// Also covers the output-frequency helpers layered on top of
// resample_sinc: validate_output_sample_rate (bounds) and resample_for_output
// (the "0 / native == passthrough, otherwise resample" engine policy).

#include "voice_features.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

// --- output-frequency helper coverage --------------------------

static int g_failures = 0;
#define CHECK(cond, msg) do {                                            \
        if (!(cond)) { std::printf("  FAIL: %s\n", (msg)); ++g_failures; } \
        else         { std::printf("  ok:   %s\n", (msg)); }              \
    } while (0)

// Returns true iff calling validate_output_sample_rate(sr) throws.
static bool validate_throws(int sr) {
    try { validate_output_sample_rate(sr, "test"); return false; }
    catch (const std::exception &) { return true; }
}

static void test_output_helpers() {
    std::printf("\noutput-frequency helpers:\n");

    // validate_output_sample_rate: 0 (native) + the documented window pass;
    // sub-floor / super-ceiling / negative throw.
    CHECK(!validate_throws(0),      "validate accepts 0 (keep native)");
    CHECK(!validate_throws(8000),   "validate accepts 8000 (floor)");
    CHECK(!validate_throws(16000),  "validate accepts 16000");
    CHECK(!validate_throws(44100),  "validate accepts 44100");
    CHECK(!validate_throws(192000), "validate accepts 192000 (ceiling)");
    CHECK(validate_throws(7999),    "validate rejects 7999 (below floor)");
    CHECK(validate_throws(192001),  "validate rejects 192001 (above ceiling)");
    CHECK(validate_throws(-16000),  "validate rejects negative");

    // resample_for_output passthrough: target 0 or == native returns the
    // input bit-for-bit (no resampling artefacts, no length change).
    std::vector<float> in(2000);
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = std::sin(2.0 * M_PI * 1000.0 * (double)i / 24000.0);
    }
    CHECK(resample_for_output(in, 24000, 0)     == in, "passthrough when target 0");
    CHECK(resample_for_output(in, 24000, 24000) == in, "passthrough when target == native");

    // Downsample 24k -> 16k: length tracks the 2/3 ratio and a tone well
    // below the new Nyquist (1 kHz << 8 kHz) survives at ~unit amplitude.
    auto down = resample_for_output(in, 24000, 16000);
    const size_t expect_down = (size_t) std::floor(in.size() * 16000.0 / 24000.0);
    CHECK(down.size() == expect_down, "24k->16k output length matches ratio");
    float peak = 0.0f;
    for (size_t i = 64; i + 64 < down.size(); ++i) peak = std::max(peak, std::fabs(down[i]));
    CHECK(peak > 0.8f && peak < 1.2f, "24k->16k preserves a 1 kHz tone amplitude");

    // Upsample 44.1k -> 48k: length tracks the ratio.
    std::vector<float> in441(4410);
    for (size_t i = 0; i < in441.size(); ++i) {
        in441[i] = std::sin(2.0 * M_PI * 440.0 * (double)i / 44100.0);
    }
    auto up = resample_for_output(in441, 44100, 48000);
    const size_t expect_up = (size_t) std::floor(in441.size() * 48000.0 / 44100.0);
    CHECK(up.size() == expect_up, "44.1k->48k output length matches ratio");
}

// --- stateful streaming resampler ------------------------------

// Stream `in` through an OutputResampler, cutting it into chunks whose sizes
// cycle through `sizes`; `trim` samples are dropped from the very first chunk to
// mimic the pipeline's trim-faded first chunk (a non-ratio-aligned boundary).
// Returns the concatenation of every process() result followed by finish().
static std::vector<float> stream_all(const std::vector<float> & in,
                                     int native_sr, int target_sr,
                                     const std::vector<int> & sizes, int trim) {
    OutputResampler rs(native_sr, target_sr);
    std::vector<float> out;
    size_t pos = 0, si = 0;
    bool first = true;
    while (pos < in.size()) {
        int want = sizes[si % sizes.size()] - (first ? trim : 0);
        if (want < 1) want = 1;
        const size_t end = std::min(in.size(), pos + (size_t) want);
        std::vector<float> chunk(in.begin() + pos, in.begin() + end);
        std::vector<float> e = rs.process(chunk);
        out.insert(out.end(), e.begin(), e.end());
        pos = end;
        ++si;
        first = false;
    }
    std::vector<float> tail = rs.finish();
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

static void test_streaming_resampler() {
    std::printf("\nstreaming resampler (batch-exact, seam-free):\n");

    // Broadband signal with energy up near the 24 kHz Nyquist, so any per-chunk
    // seam (window truncation / fractional-phase reset / length drift) would
    // surface as a mismatch against the whole-buffer resample.
    std::vector<float> sig(20000);
    for (size_t i = 0; i < sig.size(); ++i) {
        const double t = (double) i / 24000.0;
        sig[i] = 0.4f * (float) std::sin(2.0 * M_PI *  600.0 * t)
               + 0.3f * (float) std::sin(2.0 * M_PI * 4000.0 * t)
               + 0.3f * (float) std::sin(2.0 * M_PI * 9000.0 * t);
    }

    // Passthrough (0 / == native): process() returns input verbatim, finish() empty.
    CHECK(OutputResampler(24000, 0).passthrough(),     "target 0 is passthrough");
    CHECK(OutputResampler(24000, 24000).passthrough(), "target == native is passthrough");
    CHECK(stream_all(sig, 24000, 0,     {1000}, 0) == sig, "passthrough(0) streams input verbatim");
    CHECK(stream_all(sig, 24000, 24000, {333},  0) == sig, "passthrough(native) streams input verbatim");

    // For every rate / chunking the streamed result must be BIT-IDENTICAL to
    // resampling the whole signal once.  These cover the cases per-chunk
    // resampling corrupts: misaligned chunk lengths, a trim-faded first chunk,
    // and a "hostile" rate (11025, whose denominator does not divide the chunk
    // grid).  Bit-equality also proves there is no length drift.
    struct Case { int tgt; std::vector<int> sizes; int trim; const char * name; };
    const Case cases[] = {
        {16000, {4800},        0,  "24k->16k  uniform aligned chunks    == batch"},
        {16000, {4799},        0,  "24k->16k  misaligned chunks         == batch"},
        {16000, {4800},        33, "24k->16k  trim-faded first chunk    == batch"},
        { 8000, {1920, 2400},  7,  "24k->8k   varied/misaligned chunks  == batch"},
        {48000, {4800},        0,  "24k->48k  upsample                  == batch"},
        {44100, {4801},        0,  "24k->44.1k upsample, ugly ratio     == batch"},
        {11025, {4800},        13, "24k->11025 hostile rate + trim      == batch"},
    };
    for (const Case & c : cases) {
        const std::vector<float> streamed = stream_all(sig, 24000, c.tgt, c.sizes, c.trim);
        const std::vector<float> batch    = resample_for_output(sig, 24000, c.tgt);
        const bool same = (streamed == batch);
        CHECK(same, c.name);
        if (!same) {
            std::printf("    (streamed=%zu batch=%zu)\n", streamed.size(), batch.size());
        }
    }
}

int main(int argc, char ** argv) {
    (void)argc; (void)argv;

    test_output_helpers();
    test_streaming_resampler();

    // 4 seconds of a multi-tone signal at 24 kHz.
    const int sr = 24000;
    const int N = 4 * sr;
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i) {
        double t = (double)i / sr;
        // Frequencies well below Nyquist so the resampler shouldn't have to
        // attenuate them.
        double s = 0.25 * std::sin(2 * M_PI *  220.0 * t)
                 + 0.25 * std::sin(2 * M_PI *  880.0 * t)
                 + 0.25 * std::sin(2 * M_PI * 2200.0 * t)
                 + 0.25 * std::sin(2 * M_PI * 4400.0 * t);
        in[i] = (float)s;
    }

    auto up   = resample_sinc(in, 24000, 48000);
    auto back = resample_sinc(up,  48000, 24000);
    printf("in:   samples=%zu sr=24000\n", in.size());
    printf("up:   samples=%zu sr=48000\n", up.size());
    printf("back: samples=%zu sr=24000 (expected ~%zu)\n", back.size(), in.size());

    // Compare middle region (skip the half-filter-length boundary).
    const size_t N_ = std::min(in.size(), back.size());
    const size_t skip = 64;
    float in_rms = 0, diff_rms = 0, diff_max = 0;
    for (size_t i = skip; i < N_ - skip; ++i) {
        in_rms   += in[i] * in[i];
        float d   = in[i] - back[i];
        diff_rms += d * d;
        diff_max  = std::max(diff_max, std::fabs(d));
    }
    size_t M = N_ - 2 * skip;
    in_rms   = std::sqrt(in_rms   / (float)M);
    diff_rms = std::sqrt(diff_rms / (float)M);
    double snr = 20.0 * std::log10(std::max((double)in_rms, 1e-12) /
                                    std::max((double)diff_rms, 1e-12));
    printf("\nround-trip 24k -> 48k -> 24k on a 4-tone test signal:\n"
           "  input RMS  = %.4e\n"
           "  diff RMS   = %.4e\n"
           "  diff max   = %.4e\n"
           "  SNR        = %.2f dB\n",
           in_rms, diff_rms, diff_max, snr);

    const bool snr_ok = snr >= 60.0;
    if (!snr_ok)      std::printf("\nFAIL: round-trip SNR below 60 dB\n");
    if (g_failures)   std::printf("\nFAIL: %d output-frequency helper check(s) failed\n", g_failures);
    return (snr_ok && g_failures == 0) ? 0 : 1;
}
