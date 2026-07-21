// Mel preprocess regression tests (FFT spectrum, MelState, repeated-call invariance).
//
// Pure unit test; no GGUF or weights.
//
// Usage:
//   test-mel-fft-parity
//
// Exit 0 on success; non-zero on failure.

#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr float kPI = 3.14159265358979323846f;

// Deterministic synthetic test signal: 1 second of a 440 Hz sine plus
// a 1320 Hz harmonic and a small dither, at 16 kHz mono. Picks
// frequencies that land squarely on bin centers for n_fft=512 (since
// 16000 / 512 = 31.25 Hz/bin; 440 isn't on a bin center, which is a
// feature -- it stresses the windowed FFT path), while the dither
// breaks any pathological all-zeros / all-ones invariants.
std::vector<float> make_signal(int n_samples) {
    std::vector<float> x(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        const float t = (float) i / 16000.0f;
        const float carrier = std::sin(2.0f * kPI * 440.0f * t);
        const float second  = 0.3f * std::sin(2.0f * kPI * 1320.0f * t);
        // tiny dither, deterministic
        const float dither  = 1e-3f * std::sin(2.0f * kPI * 7777.0f * t + 0.123f);
        x[i] = carrier + second + dither;
    }
    return x;
}

// Hann window matching NeMo's win_length convention (centered, symmetric).
std::vector<float> make_hann_window(int win_length) {
    std::vector<float> w(win_length);
    for (int i = 0; i < win_length; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * kPI * (float) i / (float) (win_length - 1)));
    }
    return w;
}

// Deterministic synthetic filterbank. Not NeMo-correct -- only valid
// for testing internal consistency of the mel preprocess pipeline.
std::vector<float> make_synthetic_filterbank(int n_mels, int n_bins) {
    std::vector<float> fb((size_t) n_mels * n_bins);
    for (int m = 0; m < n_mels; ++m) {
        for (int k = 0; k < n_bins; ++k) {
            // Triangular-ish: peak at bin k_center = m * n_bins / n_mels,
            // falloff linear to zero. Always non-negative, never all-zero.
            const float center = ((float) m + 0.5f) * (float) n_bins / (float) n_mels;
            const float width  = (float) n_bins / (2.0f * (float) n_mels) + 2.0f;
            const float dist   = std::abs((float) k - center);
            const float val    = std::max(0.0f, 1.0f - dist / width) + 1e-3f;
            fb[(size_t) m * n_bins + k] = val;
        }
    }
    return fb;
}

parakeet::MelConfig make_test_cfg() {
    parakeet::MelConfig cfg;
    cfg.sample_rate = 16000;
    cfg.n_fft       = 512;
    cfg.win_length  = 400;
    cfg.hop_length  = 160;
    cfg.n_mels      = 80;
    cfg.preemph     = 0.97f;
    cfg.log_zero_guard_value = parakeet::kDefaultLogZeroGuard;
    cfg.normalize   = parakeet::MelNormalize::PerFeature;

    cfg.window     = make_hann_window(cfg.win_length);
    cfg.filterbank = make_synthetic_filterbank(cfg.n_mels, cfg.n_fft / 2 + 1);
    return cfg;
}

bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Float-bit-aware diff that surfaces the first divergence so a future
// regression failure has a concrete sample index to debug from.
void report_first_diff(const std::vector<float> & a, const std::vector<float> & b,
                       const char * label) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            std::fprintf(stderr,
                "[%s] FAIL at index %zu: a=%.9g  b=%.9g  delta=%.3e\n",
                label, i, (double) a[i], (double) b[i],
                (double) (a[i] - b[i]));
            return;
        }
    }
    if (a.size() != b.size()) {
        std::fprintf(stderr, "[%s] FAIL: size mismatch (%zu vs %zu)\n",
                     label, a.size(), b.size());
    }
}

int section_repeated_call_invariance(const parakeet::MelConfig & cfg,
                                     const std::vector<float> & signal) {
    using namespace parakeet;

    MelState                state;
    std::vector<float>      first_mel;
    int                     first_n_frames = 0;

    const int N = 7; // arbitrary; large enough to surface state bleed
    for (int i = 0; i < N; ++i) {
        std::vector<float> mel;
        int n_frames = 0;
        if (compute_log_mel(signal.data(), (int) signal.size(),
                             cfg, state, mel, n_frames) != 0) {
            std::fprintf(stderr, "[repeat] compute_log_mel failed on call %d\n", i + 1);
            return 1;
        }
        if (i == 0) {
            first_mel = std::move(mel);
            first_n_frames = n_frames;
            continue;
        }
        if (n_frames != first_n_frames) {
            std::fprintf(stderr,
                "[repeat] FAIL: n_frames drift on call %d (%d vs %d)\n",
                i + 1, n_frames, first_n_frames);
            return 1;
        }
        if (!bit_equal(mel, first_mel)) {
            report_first_diff(first_mel, mel, "repeat");
            return 1;
        }
    }
    std::fprintf(stderr, "[repeat] PASS  %d sequential stateful calls bit-equal\n", N);
    return 0;
}

int section_stateful_vs_stateless_parity(const parakeet::MelConfig & cfg,
                                         const std::vector<float> & signal) {
    using namespace parakeet;

    std::vector<float> mel_stateless;
    int                n_frames_stateless = 0;
    if (compute_log_mel(signal.data(), (int) signal.size(),
                         cfg, mel_stateless, n_frames_stateless) != 0) {
        std::fprintf(stderr, "[parity] stateless compute_log_mel failed\n");
        return 1;
    }

    MelState           state;
    std::vector<float> mel_stateful;
    int                n_frames_stateful = 0;
    if (compute_log_mel(signal.data(), (int) signal.size(),
                         cfg, state, mel_stateful, n_frames_stateful) != 0) {
        std::fprintf(stderr, "[parity] stateful compute_log_mel failed\n");
        return 1;
    }

    if (n_frames_stateful != n_frames_stateless) {
        std::fprintf(stderr,
            "[parity] FAIL: n_frames mismatch stateful=%d vs stateless=%d\n",
            n_frames_stateful, n_frames_stateless);
        return 1;
    }
    if (!bit_equal(mel_stateless, mel_stateful)) {
        report_first_diff(mel_stateless, mel_stateful, "parity");
        return 1;
    }
    std::fprintf(stderr,
        "[parity] PASS  stateful overload bit-equal stateless on %d frames * %d mels\n",
        n_frames_stateless, cfg.n_mels);
    return 0;
}

// Reference radix-2 complex FFT (textbook, no twiddle cache, no
// trig precomputation). Used to derive a known-good power spectrum
// for the FFT correctness gate. Decoupled from `mel_preprocess.cpp`
// internals so a future regression in the production FFT can't also
// silently regress the reference.
void reference_complex_fft_power(const float * __restrict x_real, int n_fft,
                                 std::vector<float> & power) {
    const int n_bins = n_fft / 2 + 1;
    power.assign((size_t) n_bins, 0.0f);

    // Pack real input into complex. No bit-reversal optimisation, no
    // twiddle cache -- correctness over speed for the reference.
    std::vector<std::pair<double, double>> y((size_t) n_fft);
    for (int i = 0; i < n_fft; ++i) y[i] = {(double) x_real[i], 0.0};

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n_fft; ++i) {
        int bit = n_fft >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(y[i], y[j]);
    }

    // Cooley-Tukey
    for (int len = 2; len <= n_fft; len <<= 1) {
        const double ang = -2.0 * (double) kPI / (double) len;
        const double wlen_re = std::cos(ang);
        const double wlen_im = std::sin(ang);
        for (int i = 0; i < n_fft; i += len) {
            double w_re = 1.0, w_im = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                const auto u = y[i + k];
                const auto v_re = y[i + k + len / 2].first  * w_re - y[i + k + len / 2].second * w_im;
                const auto v_im = y[i + k + len / 2].first  * w_im + y[i + k + len / 2].second * w_re;
                y[i + k]               = {u.first + v_re, u.second + v_im};
                y[i + k + len / 2]     = {u.first - v_re, u.second - v_im};
                const double nw_re = w_re * wlen_re - w_im * wlen_im;
                const double nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re; w_im = nw_im;
            }
        }
    }

    for (int k = 0; k < n_bins; ++k) {
        const double re = y[k].first;
        const double im = y[k].second;
        power[k] = (float) (re * re + im * im);
    }
}

// FFT correctness gate. The production rfft_power_radix2 is internal
// to mel_preprocess.cpp (anonymous namespace) so we can't call it
// directly. Instead we exercise it indirectly: build a single-frame
// MelConfig (n_fft = win_length = full length, n_mels = 1, identity
// filterbank summing all bins to mel[0]), call compute_log_mel on a
// single windowed frame, and reverse-engineer what the production
// FFT must have produced. This catches any rfft_power_radix2 bug
// that would shift the per-bin power.
//
// For a stricter direct test we also synthesise the same windowed
// frame here, run the reference complex FFT, and compute the same
// "sum of all bins under unit filterbank" scalar. These two scalars
// must agree to ULP scale.
int section_real_fft_parity(const parakeet::MelConfig & cfg,
                            const std::vector<float> & signal) {
    using namespace parakeet;

    // Single-frame config: filterbank that's "row 0 = ones, rest = 0";
    // log-mel of that single mel bin per frame is log(sum_k power[t, k]).
    // Skip CMVN so the inverse log -> raw bin-sum is unambiguous.
    MelConfig sf_cfg = cfg;
    sf_cfg.n_mels    = 1;
    sf_cfg.normalize = MelNormalize::None;
    const int n_bins = sf_cfg.n_fft / 2 + 1;
    sf_cfg.filterbank.assign((size_t) n_bins, 1.0f); // all ones, single mel row

    std::vector<float> mel;
    int                n_frames = 0;
    if (compute_log_mel(signal.data(), (int) signal.size(),
                         sf_cfg, mel, n_frames) != 0) {
        std::fprintf(stderr, "[fft] single-row compute_log_mel failed\n");
        return 1;
    }
    if (n_frames < 1 || (int) mel.size() != n_frames) {
        std::fprintf(stderr, "[fft] FAIL: unexpected mel shape (%zu vs %d frames)\n",
                     mel.size(), n_frames);
        return 1;
    }

    // Reproduce the production reflect-pad + window for frame 0
    // (start = 0). compute_log_mel applies preemph in-place over the
    // entire signal first, so do that here too.
    std::vector<float> work(signal);
    {
        const float p = sf_cfg.preemph;
        if (p != 0.0f && work.size() >= 2) {
            for (size_t t = work.size() - 1; t >= 1; --t) {
                work[t] -= p * work[t - 1];
            }
        }
    }
    const int pad = sf_cfg.n_fft / 2;
    std::vector<float> padded((size_t) work.size() + 2 * pad);
    {
        const int n = (int) work.size();
        for (int i = 0; i < pad; ++i) {
            const int src = std::min(pad - i, n - 1);
            padded[i] = work[src];
        }
        std::memcpy(padded.data() + pad, work.data(), (size_t) n * sizeof(float));
        for (int i = 0; i < pad; ++i) {
            const int src = std::max(n - 2 - i, 0);
            padded[pad + n + i] = work[src];
        }
    }

    // Frame 0: hop_length offset = 0; window applied with the
    // n_fft-length zero-padded analysis window matching the
    // production make_padded_window.
    const int win_length = sf_cfg.win_length;
    const int win_pad_left = (sf_cfg.n_fft - win_length) / 2;
    std::vector<float> windowed_frame((size_t) sf_cfg.n_fft, 0.0f);
    for (int i = 0; i < sf_cfg.n_fft; ++i) {
        const float wcoef = (i >= win_pad_left && i < win_pad_left + win_length)
                          ? sf_cfg.window[i - win_pad_left] : 0.0f;
        windowed_frame[i] = padded[i] * wcoef;
    }

    std::vector<float> ref_power;
    reference_complex_fft_power(windowed_frame.data(), sf_cfg.n_fft, ref_power);

    double ref_sum = 0.0;
    for (float p : ref_power) ref_sum += p;
    const float ref_log = std::log((float) ref_sum + sf_cfg.log_zero_guard_value);

    const float prod_log = mel[0]; // first frame, single mel bin

    const float diff = std::abs(ref_log - prod_log);
    const float denom = std::max(std::abs(ref_log), std::abs(prod_log));
    const float rel  = denom > 0.0f ? diff / denom : diff;

    // Threshold: 1e-5 relative. Real-FFT rearranges sums vs the
    // reference so ULP-level drift is normal; double-precision
    // reference -> single-precision production should clear 1e-5
    // comfortably on this signal magnitude.
    constexpr float kRelTol = 1e-5f;
    if (!(rel < kRelTol)) {
        std::fprintf(stderr,
            "[fft] FAIL: frame 0 sum-bin parity ref=%.9g prod=%.9g rel=%.3e (tol %.1e)\n",
            (double) ref_log, (double) prod_log, (double) rel, (double) kRelTol);
        return 1;
    }
    std::fprintf(stderr,
        "[fft] PASS  rfft_power_radix2 vs textbook complex FFT  rel=%.3e (tol %.1e)\n",
        (double) rel, (double) kRelTol);
    return 0;
}

}

int main(int /*argc*/, char ** /*argv*/) {
    using namespace parakeet;

    const MelConfig cfg = make_test_cfg();
    const std::vector<float> signal = make_signal(cfg.sample_rate); // 1 sec

    int rc = 0;
    rc |= section_real_fft_parity(cfg, signal);
    rc |= section_stateful_vs_stateless_parity(cfg, signal);
    rc |= section_repeated_call_invariance(cfg, signal);

    if (rc != 0) {
        std::fprintf(stderr, "[test-mel-fft-parity] FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "[test-mel-fft-parity] all checks passed\n");
    return 0;
}
