// WAV load, Hann window, RFFT mel pipeline, MelState buffering for streaming.

#include "mel_preprocess.h"
#include "parakeet_log.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace parakeet {

int load_wav_mono_f32(const std::string & wav_path,
                      std::vector<float>   & out_samples,
                      int                  & out_sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, wav_path.c_str(), nullptr)) {
        PARAKEET_LOG_ERROR("error: could not open wav file %s\n", wav_path.c_str());
        return 1;
    }

    const int channels       = static_cast<int>(wav.channels);
    const int sample_rate    = static_cast<int>(wav.sampleRate);
    const drwav_uint64 total = wav.totalPCMFrameCount;

    std::vector<float> interleaved(total * channels);
    const drwav_uint64 read = drwav_read_pcm_frames_f32(&wav, total, interleaved.data());
    drwav_uninit(&wav);

    if (read != total) {
        PARAKEET_LOG_ERROR("error: short read from %s\n", wav_path.c_str());
        return 2;
    }

    out_samples.resize(total);
    if (channels == 1) {
        std::memcpy(out_samples.data(), interleaved.data(), total * sizeof(float));
    } else {
        const float inv_c = 1.0f / static_cast<float>(channels);
        for (drwav_uint64 i = 0; i < total; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
            out_samples[i] = acc * inv_c;
        }
    }

    out_sample_rate = sample_rate;
    return 0;
}

namespace {

// Precomputed cooley-tukey twiddle table keyed on FFT size. The
// reference implementation accumulated `w *= wlen` inside the inner
// butterfly which costs one complex multiply per butterfly (4 muls +
// 2 adds + a fused cos/sin during table seeding). For our use case
// (n_fft ∈ {256, 512, 1024} called once per frame for ~T_mel frames
// per inference) the cost is dominated by trig + per-butterfly mul,
// so caching cos/sin per (len, k) is a clean ~1.5-2x win on the FFT
// alone. Each FFT length costs sum_{len=2..n step ×2} (len/2) =
// (n - 1) twiddles, i.e. 511 complex twiddles for n_fft=512: a
// 4 KiB table that's reused across every frame for the rest of the
// process lifetime.
struct FftTwiddleTable {
    int n_fft = 0;
    std::vector<std::complex<float>> w; // size = n - 1
};

// Process-wide twiddle cache keyed on n_fft. `compute_log_mel` is
// allowed to be called from multiple threads in principle (the
// engine doesn't today, but we don't want to assume), so the cache
// uses a thread-local store -- per-thread cache is cheap (bytes per
// FFT length) and avoids any locking on the hot path.
std::complex<float> * get_fft_twiddles(int n) {
    thread_local std::vector<FftTwiddleTable> cache;
    for (auto & e : cache) {
        if (e.n_fft == n) return e.w.data();
    }
    FftTwiddleTable tab;
    tab.n_fft = n;
    tab.w.reserve(n - 1);
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * 3.14159265358979323846f / (float) len;
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        std::complex<float> wk(1.0f, 0.0f);
        for (int k = 0; k < len / 2; ++k) {
            tab.w.push_back(wk);
            wk *= wlen;
        }
    }
    cache.push_back(std::move(tab));
    return cache.back().w.data();
}

void fft_radix2_inplace(std::complex<float> * data, int n) {
    int log_n = 0;
    while ((1 << log_n) < n) ++log_n;
    if ((1 << log_n) != n) throw std::runtime_error("fft_radix2_inplace: n must be a power of two");

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    const std::complex<float> * twiddles = get_fft_twiddles(n);
    int twiddle_off = 0;
    for (int len = 2; len <= n; len <<= 1) {
        const std::complex<float> * w_table = twiddles + twiddle_off;
        const int half = len / 2;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; ++k) {
                const std::complex<float> u = data[i + k];
                const std::complex<float> v = data[i + k + half] * w_table[k];
                data[i + k]        = u + v;
                data[i + k + half] = u - v;
            }
        }
        twiddle_off += half;
    }
}

// Real-input FFT via the standard "pack-as-half-N-complex" trick:
// for a real input x[0..n-1], compute the spectrum X[0..n/2] (n/2+1
// non-redundant bins) by running an n/2-point complex FFT on the
// packed sequence y[k] = x[2k] + i*x[2k+1] and then unpacking.
// Cuts the butterfly count in half (256 vs 512 at n_fft=512) at the
// cost of an O(n/2) post-processing pass that reuses the same
// thread_local twiddle cache. We compute the *power* spectrum
// directly so the caller never sees the unpacked complex bins.
//
// Reference: any FFT textbook (Numerical Recipes §12.3, "Fast
// Fourier Transform of Real Functions"). The trick relies on
// X[k] for real input being conjugate-symmetric: X[n-k] = conj(X[k]),
// so n/2+1 bins fully describe the spectrum.
//
// Bit-equivalence: floats are not associative; the post-processing
// reorders sums vs the equivalent complex-FFT path. The resulting
// bin powers differ from the complex-FFT version by ~1e-7 relative
// (ULP-level), well below the f16 quantization floor of the
// downstream encoder. Encoder transcripts on jfk.wav and
// sample-16k.wav stay bit-equal to the NeMo PyTorch reference at
// f16 / Q8_0 -- gated by `test-perf-regression` + `test-streaming`
// in the optimization audit.
void rfft_power_radix2(const float * __restrict x_real,
                       float       * __restrict power,
                       int                       n_fft,
                       std::complex<float>     * scratch /* size >= n_fft / 2 */) {
    const int half = n_fft / 2;

    // Pack: y[k] = x[2k] + i*x[2k+1], complex sequence of length n/2.
    for (int k = 0; k < half; ++k) {
        scratch[k] = std::complex<float>(x_real[2 * k], x_real[2 * k + 1]);
    }

    // n/2-point complex FFT. Twiddles for size `half` cached in the
    // shared thread_local table.
    fft_radix2_inplace(scratch, half);

    // Unpack to recover power[0..half], the real-input spectrum's
    // n/2+1 non-redundant bins. Two real-valued endpoints (DC and
    // Nyquist) and (half-1) interior pairs.
    //
    //   X[0]      = Y[0].re + Y[0].im       (real)
    //   X[n/2]    = Y[0].re - Y[0].im       (real)
    //   X[k]      = Y_e[k] + W[k] * Y_o[k]  (1 <= k < n/2)
    //
    // where Y_e[k] = (Y[k] + conj(Y[n/2-k])) / 2  (even-indexed FFT),
    //       Y_o[k] = -i * (Y[k] - conj(Y[n/2-k])) / 2 (odd-indexed FFT),
    //       W[k]   = exp(-2πi*k/n).
    {
        const float r0   = scratch[0].real() + scratch[0].imag();
        const float rNy  = scratch[0].real() - scratch[0].imag();
        power[0]    = r0  * r0;
        power[half] = rNy * rNy;
    }

    // Reuse the shared twiddle cache for size `n_fft` to grab
    // W[k] = twiddles[half - 1 + k] for k in [1, half - 1]. The
    // cache layout per `get_fft_twiddles` is:
    //   for len=2..n step *=2: twiddles[off..off+len/2)] = exp(-2πi k / len)
    // so the segment for `len = n_fft` starts at `n_fft/2 - 1` and
    // contains exactly the n_fft/2 values exp(-2πi*k/n_fft) for
    // k = 0..n_fft/2 - 1. We need k = 1..half-1 from that segment.
    const std::complex<float> * twiddles = get_fft_twiddles(n_fft);
    const std::complex<float> * w_n      = twiddles + (n_fft / 2 - 1);
    for (int k = 1; k < half; ++k) {
        const std::complex<float> yk = scratch[k];
        const std::complex<float> ym = std::conj(scratch[half - k]);

        // Y_e[k] = (yk + ym) * 0.5, Y_o[k] = -i * (yk - ym) * 0.5
        const std::complex<float> ye  = (yk + ym) * 0.5f;
        const std::complex<float> dif = (yk - ym) * 0.5f;
        const std::complex<float> yo(dif.imag(), -dif.real()); // -i * dif

        // X[k] = Y_e + W[k] * Y_o
        const std::complex<float> xk = ye + w_n[k] * yo;
        power[k] = xk.real() * xk.real() + xk.imag() * xk.imag();
    }
}

void apply_preemph(std::vector<float> & x, float preemph) {
    if (preemph == 0.0f || x.size() < 2) return;
    for (size_t t = x.size() - 1; t >= 1; --t) {
        x[t] = x[t] - preemph * x[t - 1];
    }
}

std::vector<float> reflect_pad(const std::vector<float> & x, int pad) {
    const int n = static_cast<int>(x.size());
    std::vector<float> out(n + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        const int src = std::min(pad - i, n - 1);
        out[i] = x[src];
    }
    std::memcpy(out.data() + pad, x.data(), n * sizeof(float));
    for (int i = 0; i < pad; ++i) {
        const int src = std::max(n - 2 - i, 0);
        out[pad + n + i] = x[src];
    }
    return out;
}

std::vector<float> make_padded_window(const std::vector<float> & hann400, int n_fft) {
    const int win_length = static_cast<int>(hann400.size());
    const int pad_total  = n_fft - win_length;
    const int pad_left   = pad_total / 2;
    std::vector<float> out(n_fft, 0.0f);
    std::memcpy(out.data() + pad_left, hann400.data(), win_length * sizeof(float));
    return out;
}

}

namespace {

// Stateful inner. Both public `compute_log_mel` overloads call into
// this one; the stateless overload uses a scratch `MelState` allocated
// on the stack (so its semantics are unchanged for callers that aren't
// stream-shaped).
int compute_log_mel_impl(const float        * samples,
                         int                  n_samples,
                         const MelConfig    & cfg,
                         MelState           & state,
                         std::vector<float> & out_mel,
                         int                & out_n_frames) {
    if (n_samples <= 0) return 1;
    if (cfg.filterbank.size() != static_cast<size_t>(cfg.n_mels * (cfg.n_fft / 2 + 1))) {
        PARAKEET_LOG_ERROR("mel: unexpected filterbank size (%zu != %d)\n",
                           cfg.filterbank.size(), cfg.n_mels * (cfg.n_fft / 2 + 1));
        return 2;
    }
    if (static_cast<int>(cfg.window.size()) != cfg.win_length) {
        PARAKEET_LOG_ERROR("mel: unexpected window size (%zu != %d)\n",
                           cfg.window.size(), cfg.win_length);
        return 3;
    }

    state.x.resize((size_t) n_samples);
    std::memcpy(state.x.data(), samples, (size_t) n_samples * sizeof(float));
    apply_preemph(state.x, cfg.preemph);

    const int pad = cfg.n_fft / 2;
    const int n_padded = n_samples + 2 * pad;
    state.x_padded.resize((size_t) n_padded);
    {
        // Inline reflect-pad into the cached buffer instead of
        // returning a fresh std::vector from `reflect_pad`.
        const int n = n_samples;
        for (int i = 0; i < pad; ++i) {
            const int src = std::min(pad - i, n - 1);
            state.x_padded[i] = state.x[src];
        }
        std::memcpy(state.x_padded.data() + pad, state.x.data(),
                    (size_t) n * sizeof(float));
        for (int i = 0; i < pad; ++i) {
            const int src = std::max(n - 2 - i, 0);
            state.x_padded[pad + n + i] = state.x[src];
        }
    }

    const int n_frames = 1 + n_samples / cfg.hop_length;
    out_n_frames = n_frames;

    const int n_bins = cfg.n_fft / 2 + 1;

    // Window padding only depends on cfg.n_fft + the (immutable) cfg.window
    // contents. Cache the result on `state` so we rebuild it at most once
    // per engine lifetime.
    if (state.window_padded_n_fft != cfg.n_fft || state.window_padded_src != &cfg.window) {
        state.window_padded     = make_padded_window(cfg.window, cfg.n_fft);
        state.window_padded_n_fft = cfg.n_fft;
        state.window_padded_src = &cfg.window;
    }
    const float * __restrict window_padded = state.window_padded.data();

    state.power.resize((size_t) n_frames * n_bins);

    float                  * __restrict power_data = state.power.data();
    const float            * __restrict x_padded   = state.x_padded.data();

    // NOTE on threading: this loop was experimentally parallelised with
    // `#pragma omp parallel { local tbuf; #pragma omp for ... }` during
    // the optimization audit. On a 16-thread Ryzen the
    // result was a +120 % regression with stdev of 18 ms because the
    // ggml-cpu encoder also uses an OpenMP thread pool and the two
    // pools oversubscribe the cores during the encoder warmup window.
    // Kept serial; the real-FFT change below + the precomputed twiddle
    // table from a prior patch already get mel under ~3 ms median.
    //
    // Per-frame work: pre-multiply by the analysis window into a
    // n_fft-long real buffer, then run the real-input radix-2 FFT
    // which packs into n_fft/2 complex points, FFTs, and unpacks
    // straight into power[0..n_fft/2]. Roughly 2x fewer butterflies
    // than the previous complex-on-real path. `tbuf` only needs
    // n_fft/2 complex slots now.
    std::vector<float>               windowed((size_t) cfg.n_fft);
    std::vector<std::complex<float>> tbuf((size_t) cfg.n_fft / 2);
    float               * __restrict windowed_data = windowed.data();
    std::complex<float> * __restrict tbuf_data     = tbuf.data();

    for (int t = 0; t < n_frames; ++t) {
        const int start = t * cfg.hop_length;
        // Pre-multiply by the window in a single pass.
        for (int i = 0; i < cfg.n_fft; ++i) {
            windowed_data[i] = x_padded[start + i] * window_padded[i];
        }
        rfft_power_radix2(windowed_data,
                          power_data + (size_t) t * n_bins,
                          cfg.n_fft,
                          tbuf_data);
    }

    const int n_mels = cfg.n_mels;
    out_mel.resize((size_t) n_frames * n_mels);
    const float * __restrict fb = cfg.filterbank.data();
    // Mel filterbank projection: out[t,m] = sum_k fb[m,k] * power[t,k].
    // Per-frame outer loop is independent; SIMD inner is handled by
    // `__restrict` + `#pragma GCC ivdep` so gcc-13 emits AVX2 FMA at
    // 8 lanes wide. The same threading caveat as the FFT loop above
    // applies (parallelising this regressed the bench during the
    // optimization audit because of OpenMP oversubscription with
    // ggml-cpu's encoder thread pool).
    for (int t = 0; t < n_frames; ++t) {
        const float * __restrict frame_power = power_data + t * n_bins;
        float * __restrict mel_t = out_mel.data() + t * n_mels;
        for (int m = 0; m < n_mels; ++m) {
            const float * __restrict row = fb + m * n_bins;
            float acc = 0.0f;
            #pragma GCC ivdep
            for (int k = 0; k < n_bins; ++k) acc += row[k] * frame_power[k];
            mel_t[m] = acc;
        }
    }

    const float guard = cfg.log_zero_guard_value;
    #pragma GCC ivdep
    for (size_t i = 0; i < out_mel.size(); ++i) {
        out_mel[i] = std::log(out_mel[i] + guard);
    }

    const int seq_len = (n_samples + cfg.hop_length - 1) / cfg.hop_length;
    const int valid_frames = std::min(seq_len, n_frames);

    if (cfg.normalize == MelNormalize::PerFeature) {
        apply_per_feature_cmvn(out_mel, valid_frames, n_mels);

        // Per-feature CMVN sets the trailing padded frames to mean=0 implicitly,
        // but we still want them to contribute zero energy to the encoder mask
        // path. For NeMo `normalize=NA` we leave them as raw log-mel values
        // (typically near -16, the log_zero_guard floor) -- the encoder's
        // pad_mask zeros them out at the conv module anyway, and crucially the
        // CMVN-free branch must not introduce a bin-wise mean shift the model
        // wasn't trained against.
        for (int t = valid_frames; t < n_frames; ++t) {
            for (int m = 0; m < n_mels; ++m) out_mel[t * n_mels + m] = 0.0f;
        }
    }

    return 0;
}

}

int compute_log_mel(const float        * samples,
                    int                  n_samples,
                    const MelConfig    & cfg,
                    std::vector<float> & out_mel,
                    int                & out_n_frames) {
    MelState scratch;
    return compute_log_mel_impl(samples, n_samples, cfg, scratch, out_mel, out_n_frames);
}

int compute_log_mel(const float        * samples,
                    int                  n_samples,
                    const MelConfig    & cfg,
                    MelState           & state,
                    std::vector<float> & out_mel,
                    int                & out_n_frames) {
    return compute_log_mel_impl(samples, n_samples, cfg, state, out_mel, out_n_frames);
}

void apply_per_feature_cmvn(std::vector<float> & mel, int n_valid_frames, int n_mels) {
    if (n_valid_frames <= 0 || n_mels <= 0) return;

    // Two-pass per-feature normalize. The two reductions (sum, ss)
    // are cache-unfriendly column-major reads on a row-major buffer
    // and small enough (n_valid_frames * n_mels = ~1100 * 80 = 88k
    // floats on jfk.wav) that we accept the column-strided access
    // pattern. We keep the mean/variance in `double` because the
    // reference NeMo `normalize_batch('per_feature')` accumulator
    // is f64 -- match exact for transcript byte-equality.
    float * __restrict m = mel.data();
    for (int idx = 0; idx < n_mels; ++idx) {
        double sum = 0.0;
        for (int t = 0; t < n_valid_frames; ++t) sum += m[t * n_mels + idx];
        const double mean = sum / n_valid_frames;

        double ss = 0.0;
        for (int t = 0; t < n_valid_frames; ++t) {
            const double d = m[t * n_mels + idx] - mean;
            ss += d * d;
        }
        const double denom = std::max(1, n_valid_frames - 1);
        const double std_ = std::sqrt(ss / denom) + 1e-5;
        const float inv_std = 1.0f / static_cast<float>(std_);
        const float fmean   = static_cast<float>(mean);

        // Final scaling pass; trivially vectorisable per-row but the
        // column-strided access defeats the auto-vectoriser unless
        // we hint with `ivdep` here too. ~0.2 ms on jfk.wav.
        #pragma GCC ivdep
        for (int t = 0; t < n_valid_frames; ++t) {
            m[t * n_mels + idx] = (m[t * n_mels + idx] - fmean) * inv_std;
        }
    }
}

}
