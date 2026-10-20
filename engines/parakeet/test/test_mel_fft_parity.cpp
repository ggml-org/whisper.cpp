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

// Engine::compute_mel keeps a thread-local MelState that outlives the engine,
// and a later engine can place another model's tables at the same address;
// the state's caches must follow the table contents, never the address.
int section_reused_state_after_table_change(const parakeet::MelConfig & cfg,
                                            const std::vector<float> & signal) {
    using namespace parakeet;

    MelState  state;
    MelConfig same_address = cfg;
    std::vector<float> before, reused, fresh;
    int n_before = 0, n_reused = 0, n_fresh = 0;
    if (compute_log_mel(signal.data(), (int) signal.size(), same_address, state, before, n_before) != 0) {
        std::fprintf(stderr, "[reload] compute_log_mel failed before the table change\n");
        return 1;
    }

    for (float & w : same_address.window)     w *= 0.75f;
    for (float & f : same_address.filterbank) f *= 0.5f;

    if (compute_log_mel(signal.data(), (int) signal.size(), same_address, state, reused, n_reused) != 0 ||
        compute_log_mel(signal.data(), (int) signal.size(), same_address, fresh, n_fresh) != 0) {
        std::fprintf(stderr, "[reload] compute_log_mel failed after the table change\n");
        return 1;
    }
    if (bit_equal(fresh, before)) {
        std::fprintf(stderr, "[reload] FAIL: the changed tables did not change the mel\n");
        return 1;
    }
    if (n_reused != n_fresh || !bit_equal(reused, fresh)) {
        report_first_diff(fresh, reused, "reload");
        std::fprintf(stderr, "[reload] FAIL: a reused state kept the previous tables' caches\n");
        return 1;
    }
    std::fprintf(stderr, "[reload] PASS  reused state follows new tables at the same address\n");
    return 0;
}

int section_incremental_state_after_table_change(const parakeet::MelConfig & cfg,
                                                 const std::vector<float> & signal) {
    using namespace parakeet;

    IncrementalMelState state;
    MelConfig same_address = cfg;
    same_address.normalize = MelNormalize::None;
    std::vector<float> before, reused, fresh;
    int n_before = 0, n_reused = 0, n_fresh = 0;
    if (append_log_mel(signal.data(), (int) signal.size(), true, same_address, state, before, n_before) != 0) {
        std::fprintf(stderr, "[reload-incremental] append_log_mel failed before the table change\n");
        return 1;
    }

    for (float & w : same_address.window)     w *= 0.75f;
    for (float & f : same_address.filterbank) f *= 0.5f;

    reset_incremental_mel(state);
    IncrementalMelState fresh_state;
    if (append_log_mel(signal.data(), (int) signal.size(), true, same_address, state, reused, n_reused) != 0 ||
        append_log_mel(signal.data(), (int) signal.size(), true, same_address, fresh_state, fresh, n_fresh) != 0) {
        std::fprintf(stderr, "[reload-incremental] append_log_mel failed after the table change\n");
        return 1;
    }
    if (bit_equal(fresh, before)) {
        std::fprintf(stderr, "[reload-incremental] FAIL: the changed tables did not change the mel\n");
        return 1;
    }
    if (n_reused != n_fresh || !bit_equal(reused, fresh)) {
        report_first_diff(fresh, reused, "reload-incremental");
        std::fprintf(stderr, "[reload-incremental] FAIL: a reused incremental state kept the previous tables' caches\n");
        return 1;
    }
    std::fprintf(stderr, "[reload-incremental] PASS  reused incremental state follows new tables\n");
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

int section_incremental_short_input_parity(const parakeet::MelConfig & cfg) {
    using namespace parakeet;

    MelConfig incremental_cfg = cfg;
    incremental_cfg.normalize = MelNormalize::None;
    const int sample_counts[] = {
        1,
        incremental_cfg.hop_length - 1,
        incremental_cfg.hop_length,
        incremental_cfg.n_fft / 2 - 1,
        incremental_cfg.n_fft / 2,
        incremental_cfg.n_fft / 2 + 1,
    };

    for (const int sample_count : sample_counts) {
        const std::vector<float> signal = make_signal(sample_count);
        std::vector<float> offline_mel;
        int offline_frames = 0;
        if (compute_log_mel(
                signal.data(), sample_count, incremental_cfg,
                offline_mel, offline_frames) != 0) {
            std::fprintf(stderr,
                "[incremental-short] offline mel failed for %d samples\n",
                sample_count);
            return 1;
        }

        IncrementalMelState state;
        std::vector<float> incremental_mel;
        int incremental_frames = 0;
        if (append_log_mel(
                signal.data(), sample_count, true, incremental_cfg,
                state, incremental_mel, incremental_frames) != 0) {
            std::fprintf(stderr,
                "[incremental-short] incremental mel failed for %d samples\n",
                sample_count);
            return 1;
        }
        if (incremental_frames != offline_frames) {
            std::fprintf(stderr,
                "[incremental-short] FAIL: frame count for %d samples "
                "(%d vs %d)\n",
                sample_count, incremental_frames, offline_frames);
            return 1;
        }
        if (!bit_equal(incremental_mel, offline_mel)) {
            report_first_diff(
                offline_mel, incremental_mel, "incremental-short");
            return 1;
        }
    }

    std::fprintf(stderr,
        "[incremental-short] PASS  finalized short inputs match offline mel\n");
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

int section_per_feature_cmvn(void) {
    using namespace parakeet;

    const int n_frames = 4;
    const int n_mels   = 2;
    const double eps   = 1e-5;

    std::vector<float> mel = {
        1.0f, 5.0f,
        2.0f, 5.0f,
        3.0f, 5.0f,
        4.0f, 5.0f,
    };
    std::vector<float> expected(mel.size());
    for (int idx = 0; idx < n_mels; ++idx) {
        double sum = 0.0;
        for (int t = 0; t < n_frames; ++t) sum += mel[t * n_mels + idx];
        const double mean = sum / n_frames;
        double ss = 0.0;
        for (int t = 0; t < n_frames; ++t) {
            const double d = mel[t * n_mels + idx] - mean;
            ss += d * d;
        }
        const double std_ = std::sqrt(ss / (n_frames - 1)) + eps;
        for (int t = 0; t < n_frames; ++t) {
            expected[t * n_mels + idx] = (float)
                ((mel[t * n_mels + idx] - (float) mean) *
                 (1.0f / (float) std_));
        }
    }

    apply_per_feature_cmvn(mel, n_frames, n_mels);
    if (!bit_equal(mel, expected)) {
        report_first_diff(expected, mel, "cmvn");
        return 1;
    }
    for (int t = 0; t < n_frames; ++t) {
        if (mel[t * n_mels + 1] != 0.0f) {
            std::fprintf(stderr,
                "[cmvn] FAIL: constant feature not zeroed at frame %d\n", t);
            return 1;
        }
    }

    std::vector<float> single = {3.0f, -2.0f};
    apply_per_feature_cmvn(single, 1, 2);
    if (single[0] != 0.0f || single[1] != 0.0f) {
        std::fprintf(stderr, "[cmvn] FAIL: single frame must normalize to 0\n");
        return 1;
    }

    std::vector<float> untouched = {1.0f, 2.0f};
    std::vector<float> before = untouched;
    apply_per_feature_cmvn(untouched, 0, 2);
    apply_per_feature_cmvn(untouched, 1, 0);
    if (!bit_equal(untouched, before)) {
        std::fprintf(stderr, "[cmvn] FAIL: degenerate dims must be a no-op\n");
        return 1;
    }

    std::vector<float> partial = {
        10.0f, 10.0f,
        20.0f, 20.0f,
        777.0f, 888.0f,
    };
    apply_per_feature_cmvn(partial, 2, 2);
    if (partial[4] != 777.0f || partial[5] != 888.0f) {
        std::fprintf(stderr,
            "[cmvn] FAIL: frames past n_valid_frames were modified\n");
        return 1;
    }

    std::fprintf(stderr, "[cmvn] PASS  per-feature CMVN matches the NeMo "
                         "two-pass f64 reference\n");
    return 0;
}

int section_per_feature_valid_frame_zeroing() {
    using namespace parakeet;

    MelConfig cfg = make_test_cfg();
    cfg.normalize = MelNormalize::PerFeature;
    const int n_samples = cfg.hop_length + 1;
    const std::vector<float> signal = make_signal(n_samples);
    std::vector<float> mel;
    int n_frames = 0;
    if (compute_log_mel(
            signal.data(), n_samples, cfg, mel, n_frames) != 0) {
        std::fprintf(stderr, "[valid-frames] compute_log_mel failed\n");
        return 1;
    }

    const int expected_frames = 1 + n_samples / cfg.hop_length;
    if (n_frames != expected_frames) {
        std::fprintf(
            stderr,
            "[valid-frames] FAIL: n_frames=%d expected=%d\n",
            n_frames,
            expected_frames);
        return 1;
    }

    const int last = n_frames - 1;
    bool last_all_zero = true;
    for (int m = 0; m < cfg.n_mels; ++m) {
        if (mel[last * cfg.n_mels + m] != 0.0f) {
            last_all_zero = false;
            break;
        }
    }
    if (last_all_zero) {
        std::fprintf(
            stderr,
            "[valid-frames] FAIL: PerFeature hop+1 input used floor "
            "seq_len and zeroed a valid frame\n");
        return 1;
    }

    const int floor_n_samples = cfg.hop_length;
    const std::vector<float> floor_signal = make_signal(floor_n_samples);
    std::vector<float> floor_mel;
    int floor_frames = 0;
    if (compute_log_mel(
            floor_signal.data(),
            floor_n_samples,
            cfg,
            floor_mel,
            floor_frames) != 0) {
        std::fprintf(stderr, "[valid-frames] floor compute_log_mel failed\n");
        return 1;
    }
    const int floor_seq_len = floor_n_samples / cfg.hop_length;
    if (floor_frames <= floor_seq_len) {
        std::fprintf(
            stderr,
            "[valid-frames] FAIL: expected a pad frame to zero\n");
        return 1;
    }
    for (int t = floor_seq_len; t < floor_frames; ++t) {
        for (int m = 0; m < cfg.n_mels; ++m) {
            if (floor_mel[t * cfg.n_mels + m] != 0.0f) {
                std::fprintf(
                    stderr,
                    "[valid-frames] FAIL: pad frame %d feature %d "
                    "was not zeroed\n",
                    t,
                    m);
                return 1;
            }
        }
    }

    std::fprintf(
        stderr,
        "[valid-frames] PASS  PerFeature ceil seq_len and pad zeroing\n");
    return 0;
}

// The STFT runs in blocks of `MelConfig::stft_block_frames` frames. Each block
// rebuilds the samples it needs straight from the input, which is only sound
// because pre-emphasis and reflect padding are pointwise; a mis-derived seam
// would show up as a different frame. A block of n_frames is a single pass
// over the whole input, so comparing it against smaller blocks tests the
// chunking, and equality has to be exact: the per-frame arithmetic does not
// depend on the block.
int compare_blocked_against_single_pass(parakeet::MelConfig cfg,
                                        const std::vector<float> & signal,
                                        const char * label) {
    using namespace parakeet;

    auto mel_with_block = [&](int block, std::vector<float> & mel, int & n) {
        cfg.stft_block_frames = block;
        MelState state;
        return compute_log_mel(signal.data(), (int) signal.size(), cfg, state, mel, n);
    };

    std::vector<float> ref;
    int ref_frames = 0;
    if (mel_with_block(1 << 20, ref, ref_frames) != 0) {
        std::fprintf(stderr, "[block/%s] reference compute_log_mel failed\n", label);
        return 1;
    }

    // 1 and 2 are degenerate (every block is a seam); 2048 is the default;
    // 2047/2049 straddle it so an off-by-one in the span or the base offset
    // cannot hide behind an exact multiple.
    for (int block : {1, 2, 7, 64, 2047, 2048, 2049, 3000}) {
        std::vector<float> mel;
        int n_frames = 0;
        if (mel_with_block(block, mel, n_frames) != 0) {
            std::fprintf(stderr, "[block/%s] compute_log_mel failed at block=%d\n", label, block);
            return 1;
        }
        if (n_frames != ref_frames || !bit_equal(mel, ref)) {
            std::fprintf(stderr, "[block/%s] block=%d differs from the single pass\n", label, block);
            report_first_diff(mel, ref, "block");
            return 1;
        }
    }
    std::fprintf(stderr, "[block/%s] %d frames bit-identical across 8 block sizes\n",
                 label, ref_frames);
    return 0;
}

// 30 s crosses the default block and ends mid-block, so the interior seams
// and the final short block are both exercised; the other sections run one
// second, entirely inside one block. The second configuration takes the
// right-zero-pad branch of the padding, which the first never reaches.
int section_stft_block_invariance(const parakeet::MelConfig & cfg) {
    const std::vector<float> signal = make_signal(30 * cfg.sample_rate);

    parakeet::MelConfig zero_pad = cfg;
    zero_pad.right_zero_pad = true;
    zero_pad.normalize      = parakeet::MelNormalize::None;

    int rc = 0;
    rc |= compare_blocked_against_single_pass(cfg, signal, "reflect");
    rc |= compare_blocked_against_single_pass(zero_pad, signal, "zero-pad");
    return rc;
}

// The block buffers are sized by the block, never by the input, and
// releasing them between calls changes nothing about the next result.
int section_scratch_bounded_and_releasable(const parakeet::MelConfig & cfg) {
    using namespace parakeet;

    const std::vector<float> signal = make_signal(30 * cfg.sample_rate);
    const int block  = 256;
    const int n_bins = cfg.n_fft / 2 + 1;

    MelConfig blocked = cfg;
    blocked.stft_block_frames = block;

    MelState state;
    std::vector<float> first;
    int n_first = 0;
    if (compute_log_mel(signal.data(), (int) signal.size(), blocked, state, first, n_first) != 0) {
        std::fprintf(stderr, "[scratch] compute_log_mel failed\n");
        return 1;
    }
    const size_t span_bound  = (size_t) (block - 1) * cfg.hop_length + cfg.n_fft;
    const size_t power_bound = (size_t) block * n_bins;
    if (state.x_padded.size() > span_bound || state.power.size() > power_bound) {
        std::fprintf(stderr, "[scratch] FAIL: scratch grew past the block (%zu > %zu or %zu > %zu)\n",
                     state.x_padded.size(), span_bound, state.power.size(), power_bound);
        return 1;
    }

    state.release_scratch();
    if (!state.x_padded.empty() || !state.power.empty() || state.window_padded.empty()) {
        std::fprintf(stderr, "[scratch] FAIL: release_scratch did not free the block buffers "
                             "or dropped the cached window\n");
        return 1;
    }

    std::vector<float> second;
    int n_second = 0;
    if (compute_log_mel(signal.data(), (int) signal.size(), blocked, state, second, n_second) != 0 ||
        n_second != n_first || !bit_equal(first, second)) {
        std::fprintf(stderr, "[scratch] FAIL: result changed after release_scratch\n");
        report_first_diff(first, second, "scratch");
        return 1;
    }
    std::fprintf(stderr, "[scratch] PASS  block-bounded scratch, identical after release\n");
    return 0;
}

int main(int /*argc*/, char ** /*argv*/) {
    using namespace parakeet;

    const MelConfig cfg = make_test_cfg();
    const std::vector<float> signal = make_signal(cfg.sample_rate); // 1 sec

    int rc = 0;
    rc |= section_real_fft_parity(cfg, signal);
    rc |= section_stateful_vs_stateless_parity(cfg, signal);
    rc |= section_repeated_call_invariance(cfg, signal);
    rc |= section_reused_state_after_table_change(cfg, signal);
    rc |= section_incremental_state_after_table_change(cfg, signal);
    rc |= section_incremental_short_input_parity(cfg);
    rc |= section_per_feature_cmvn();
    rc |= section_per_feature_valid_frame_zeroing();
    rc |= section_stft_block_invariance(cfg);
    rc |= section_scratch_bounded_and_releasable(cfg);

    if (rc != 0) {
        std::fprintf(stderr, "[test-mel-fft-parity] FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "[test-mel-fft-parity] all checks passed\n");
    return 0;
}
