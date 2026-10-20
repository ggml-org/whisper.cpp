// WAV load, Hann window, RFFT mel pipeline, MelState buffering for streaming.

#include "mel_preprocess.h"
#include "parakeet_log.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <thread>
#include <utility>
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

// Frames are independent, so the FFT and filterbank loops fan out over a few
// short-lived threads that join before the encoder starts. Capped so a many-core
// host does not burn cores on ~5 us frames; small inputs stay single-threaded.
constexpr int k_mel_max_threads      = 8;
constexpr int k_mel_frames_per_thread = 256;

int mel_thread_count(int n_frames, int requested) {
    if (requested > 0) return requested;
    const int by_work = n_frames / k_mel_frames_per_thread;
    const int by_hw   = (int) std::thread::hardware_concurrency();
    return std::max(1, std::min({k_mel_max_threads, by_work, by_hw}));
}

// Joins every started worker on scope exit, so a throw anywhere never destroys
// a joinable thread.
struct JoinedThreads {
    explicit JoinedThreads(int n) { threads.reserve((size_t) n); }
    ~JoinedThreads() { for (std::thread & t : threads) t.join(); }
    template <typename F> void start(F && f) { threads.emplace_back(std::forward<F>(f)); }
    std::vector<std::thread> threads;
};

// fn(t0, t1) handles frames [t0, t1); the calling thread takes the last range.
// A worker's exception is carried back and rethrown on the calling thread.
template <typename Fn>
void parallel_over_frames(int n_frames, int requested_threads, Fn && fn) {
    const int n_threads = mel_thread_count(n_frames, requested_threads);
    if (n_threads == 1) { fn(0, n_frames); return; }
    const int per = (n_frames + n_threads - 1) / n_threads;
    std::vector<std::exception_ptr> errors((size_t) n_threads);
    {
        JoinedThreads workers(n_threads - 1);
        for (int i = 0; i < n_threads - 1; ++i) {
            workers.start([&fn, &errors, i, t0 = i * per, t1 = std::min(n_frames, (i + 1) * per)] {
                try { fn(t0, t1); } catch (...) { errors[(size_t) i] = std::current_exception(); }
            });
        }
        try {
            fn(std::min(n_frames, (n_threads - 1) * per), n_frames);
        } catch (...) {
            errors[(size_t) n_threads - 1] = std::current_exception();
        }
    }
    for (const std::exception_ptr & e : errors) {
        if (e) std::rethrow_exception(e);
    }
}

// Window + real FFT for frames [t0, t1) with this thread's own scratch.
void power_spectrum_rows(int t0, int t1, int n_fft, int hop, int n_bins,
                         const float * __restrict x_padded, const float * __restrict window_padded,
                         float * __restrict power_data) {
    std::vector<float>               windowed((size_t) n_fft);
    std::vector<std::complex<float>> tbuf((size_t) n_fft / 2);
    float               * __restrict windowed_data = windowed.data();
    std::complex<float> * __restrict tbuf_data     = tbuf.data();
    for (int t = t0; t < t1; ++t) {
        const int start = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            windowed_data[i] = x_padded[start + i] * window_padded[i];
        }
        rfft_power_radix2(windowed_data, power_data + (size_t) t * n_bins, n_fft, tbuf_data);
    }
}

// Mel filterbank projection for frames [t0, t1): out[t,m] = sum_k fb[m,k] * power[t,k].
// The inner loop is left exactly as the serial version so the reduction order is unchanged.
void filterbank_rows(int t0, int t1, int n_bins, int n_mels,
                     const float * __restrict fb, const MelFilterBand * __restrict bands,
                     float log_zero_guard, const float * __restrict power_data,
                     float * __restrict mel_data) {
    for (int t = t0; t < t1; ++t) {
        log_mel_frame(power_data + t * n_bins, n_bins, n_mels, fb, bands, log_zero_guard,
                      mel_data + t * n_mels);
    }
}

// Frames per STFT block. The block's padded span and power spectrogram are the
// only scratch buffers, so memory is bounded by the block rather than by the
// input, and a full block still feeds every worker thread.
constexpr int k_stft_block_frames_default = k_mel_max_threads * k_mel_frames_per_thread;

int stft_block_frames(const MelConfig & cfg, int n_frames) {
    const int block = cfg.stft_block_frames > 0 ? cfg.stft_block_frames
                                                : k_stft_block_frames_default;
    return std::min(block, n_frames);
}

// Pre-emphasis as a pointwise function of the input: y[m] = x[m] - preemph * x[m-1]
// with x[0] unchanged, so a block can rebuild any sample from the raw input.
inline float preemphasized_sample(const float * __restrict samples, int m, float preemph) {
    if (m <= 0) return samples[0];
    return samples[m] - preemph * samples[m - 1];
}

// Source sample index of padded position `j` under reflect padding, matching
// torch.stft(center=True, pad_mode="reflect"); -1 marks a zero in the right pad.
inline int padded_source_index(int j, int pad, int n, bool right_zero_pad) {
    if (j < pad)        return std::min(pad - j, n - 1);
    if (j < pad + n)    return j - pad;
    if (right_zero_pad) return -1;
    return std::max(n - 2 - (j - pad - n), 0);
}

// Padded positions [j0, j1) that fall in a reflect (or zero) tail, sample by sample.
void fill_padded_tail(const float * __restrict samples, int n_samples, const MelConfig & cfg,
                      int pad, int j0, int j1, float * __restrict dst) {
    for (int j = j0; j < j1; ++j) {
        const int src = padded_source_index(j, pad, n_samples, cfg.right_zero_pad);
        dst[j - j0] = src < 0 ? 0.0f : preemphasized_sample(samples, src, cfg.preemph);
    }
}

// Input samples [m0, m1) pre-emphasized in one straight pass, so the bulk of
// every block vectorises; sample 0, which has no predecessor, is copied.
void fill_preemphasized_interior(const float * __restrict samples, float preemph,
                                 int m0, int m1, float * __restrict dst) {
    int m = m0;
    if (m < m1 && m == 0) {
        dst[0] = samples[0];
        ++m;
    }
    for (; m < m1; ++m) {
        dst[m - m0] = samples[m] - preemph * samples[m - 1];
    }
}

// Rebuilds the `count` padded, pre-emphasized samples starting at padded
// position `base` straight from the raw input: the two tails position by
// position, the interior as a plain pass over the input.
void fill_padded_span(const float * __restrict samples, int n_samples, const MelConfig & cfg,
                      int pad, int base, size_t count, float * __restrict dst) {
    const int end            = base + (int) count;
    const int interior_begin = std::min(end, std::max(base, pad));
    const int interior_end   = std::max(interior_begin, std::min(end, pad + n_samples));

    fill_padded_tail(samples, n_samples, cfg, pad, base, interior_begin, dst);
    fill_preemphasized_interior(samples, cfg.preemph, interior_begin - pad, interior_end - pad,
                                dst + (interior_begin - base));
    fill_padded_tail(samples, n_samples, cfg, pad, interior_end, end, dst + (interior_end - base));
}

std::vector<float> make_padded_window(const std::vector<float> & hann400, int n_fft) {
    const int win_length = static_cast<int>(hann400.size());
    const int pad_total  = n_fft - win_length;
    const int pad_left   = pad_total / 2;
    std::vector<float> out(n_fft, 0.0f);
    std::memcpy(out.data() + pad_left, hann400.data(), win_length * sizeof(float));
    return out;
}

// A cached table is valid only while its source still holds the same values;
// the address of the source proves nothing once configs come and go.
bool same_table(const std::vector<float> & cached, const std::vector<float> & source) {
    return cached.size() == source.size() &&
           (cached.empty() ||
            std::memcmp(cached.data(), source.data(), cached.size() * sizeof(float)) == 0);
}

}

std::vector<MelFilterBand> make_filter_bands(const std::vector<float> & filterbank, int n_mels, int n_bins) {
    std::vector<MelFilterBand> bands((size_t) n_mels);
    for (int m = 0; m < n_mels; ++m) {
        const float * row = filterbank.data() + (size_t) m * n_bins;
        int lo = 0;
        while (lo < n_bins && row[lo] == 0.0f) ++lo;
        int hi = n_bins;
        while (hi > lo && row[hi - 1] == 0.0f) --hi;
        bands[(size_t) m] = { lo, hi };
    }
    return bands;
}

// Skipped weights are exact zeros, so the banded sum is bit-identical to the dense loop.
void log_mel_frame(const float * __restrict frame_power, int n_bins, int n_mels,
                   const float * __restrict filterbank, const MelFilterBand * __restrict bands,
                   float log_zero_guard, float * __restrict mel_out) {
    for (int m = 0; m < n_mels; ++m) {
        const float * __restrict row = filterbank + m * n_bins;
        const int lo = bands[m].lo;
        const int hi = bands[m].hi;
        float acc = 0.0f;
        #pragma GCC ivdep
        for (int k = lo; k < hi; ++k) acc += row[k] * frame_power[k];
        mel_out[m] = std::log(acc + log_zero_guard);
    }
}

namespace {

// One block's frames [t0, t1): window + real FFT into the block's power rows,
// then the banded filterbank and log into the output. Both steps are per
// frame, so a thread's range needs nothing from any other thread.
void mel_block_rows(int t0, int t1, const MelConfig & cfg, int n_bins,
                    const float * __restrict x_padded, const float * __restrict window_padded,
                    const MelFilterBand * __restrict bands,
                    float * __restrict power_data, float * __restrict mel_data) {
    power_spectrum_rows(t0, t1, cfg.n_fft, cfg.hop_length, n_bins, x_padded, window_padded,
                        power_data);
    filterbank_rows(t0, t1, n_bins, cfg.n_mels, cfg.filterbank.data(), bands,
                    cfg.log_zero_guard_value, power_data, mel_data);
}

// Blocked STFT. Every block rebuilds the padded samples it needs straight
// from the raw input (pre-emphasis and reflect padding are pointwise), so the
// scratch is bounded by the block and the output is bit-identical for any
// block size. Each worker thread owns its window/FFT scratch, and the workers
// join before ggml-cpu's pool wakes for the encoder.
void compute_log_mel_blocks(const float * samples, int n_samples, const MelConfig & cfg,
                            MelState & state, int pad, int n_frames, int n_bins, int block,
                            float * mel_data) {
    const float         * window_padded = state.window_padded.data();
    const MelFilterBand * bands         = state.filter_bands.data();
    for (int t0 = 0; t0 < n_frames; t0 += block) {
        const int    nb   = std::min(block, n_frames - t0);
        const size_t need = (size_t) (nb - 1) * cfg.hop_length + cfg.n_fft;
        fill_padded_span(samples, n_samples, cfg, pad, t0 * cfg.hop_length, need,
                         state.x_padded.data());
        float * block_mel = mel_data + (size_t) t0 * cfg.n_mels;
        parallel_over_frames(nb, cfg.n_threads, [&](int a, int b) {
            mel_block_rows(a, b, cfg, n_bins, state.x_padded.data(), window_padded, bands,
                           state.power.data(), block_mel);
        });
    }
}

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

    const int pad      = cfg.n_fft / 2;
    const int n_frames = 1 + n_samples / cfg.hop_length;
    out_n_frames = n_frames;

    const int n_bins = cfg.n_fft / 2 + 1;
    const int n_mels = cfg.n_mels;

    if (state.window_padded_n_fft != cfg.n_fft || !same_table(state.window_src, cfg.window)) {
        state.window_padded       = make_padded_window(cfg.window, cfg.n_fft);
        state.window_padded_n_fft = cfg.n_fft;
        state.window_src          = cfg.window;
    }
    if (state.filter_bands.size() != (size_t) n_mels || !same_table(state.filterbank_src, cfg.filterbank)) {
        state.filter_bands   = make_filter_bands(cfg.filterbank, n_mels, n_bins);
        state.filterbank_src = cfg.filterbank;
    }

    const int block = stft_block_frames(cfg, n_frames);
    state.x_padded.resize((size_t) (block - 1) * cfg.hop_length + cfg.n_fft);
    state.power.resize((size_t) block * n_bins);
    out_mel.resize((size_t) n_frames * n_mels);

    compute_log_mel_blocks(samples, n_samples, cfg, state, pad, n_frames, n_bins, block,
                           out_mel.data());

    const int seq_len = cfg.normalize == MelNormalize::None
        ? std::max(1, n_samples / cfg.hop_length)
        : std::max(1, (n_samples + cfg.hop_length - 1) / cfg.hop_length);
    const int valid_frames = std::min(seq_len, n_frames);

    if (cfg.normalize == MelNormalize::PerFeature) {
        apply_per_feature_cmvn(out_mel, valid_frames, n_mels);
    }

    // NeMo masks frames past seq_len after optional CMVN. PerFeature
    // models keep the historical ceil(n_samples / hop) length. None
    // (Nemotron `normalize=NA`) uses floor division to match NeMo.
    for (int t = valid_frames; t < n_frames; ++t) {
        for (int m = 0; m < n_mels; ++m) {
            out_mel[t * n_mels + m] = 0.0f;
        }
    }

    return 0;
}

}

struct IncrementalMelState::Impl {
    std::vector<float> samples;
    std::vector<float> prefix;
    std::vector<float> window;
    std::vector<float> windowed;
    std::vector<float> power;
    std::vector<std::complex<float>> fft;
    std::vector<MelFilterBand> filter_bands;
    std::vector<float> filterbank_src;

    int64_t sample_base = 0;
    int64_t total_samples = 0;
    int64_t next_frame = 0;
    int window_n_fft = 0;
    std::vector<float> window_src;
    float last_raw_sample = 0.0f;
    bool has_last_raw_sample = false;
    bool finalized = false;
};

IncrementalMelState::IncrementalMelState()
    : impl(std::make_unique<Impl>()) {}

IncrementalMelState::~IncrementalMelState() = default;
IncrementalMelState::IncrementalMelState(
    IncrementalMelState &&) noexcept = default;
IncrementalMelState & IncrementalMelState::operator=(
    IncrementalMelState &&) noexcept = default;

void reset_incremental_mel(IncrementalMelState & state) {
    state.impl = std::make_unique<IncrementalMelState::Impl>();
}

namespace {

void append_preemphasized_samples(
    IncrementalMelState::Impl & state,
    const float * samples,
    int n_samples,
    float preemphasis,
    int prefix_size) {
    state.samples.reserve(
        state.samples.size() + static_cast<size_t>(n_samples));
    for (int index = 0; index < n_samples; ++index) {
        const float raw = samples[index];
        const float value = state.has_last_raw_sample
            ? raw - preemphasis * state.last_raw_sample
            : raw;
        state.samples.push_back(value);
        if (static_cast<int>(state.prefix.size()) < prefix_size) {
            state.prefix.push_back(value);
        }
        state.last_raw_sample = raw;
        state.has_last_raw_sample = true;
        ++state.total_samples;
    }
}

float incremental_sample_at(
    const IncrementalMelState::Impl & state,
    int64_t sample_index,
    bool finalized) {
    if (sample_index < 0) {
        sample_index = std::min<int64_t>(
            -sample_index,
            state.total_samples - 1);
    } else if (sample_index >= state.total_samples && finalized) {
        sample_index = std::max<int64_t>(
            0, 2 * state.total_samples - 2 - sample_index);
    }
    if (sample_index < static_cast<int64_t>(state.prefix.size())) {
        return state.prefix[static_cast<size_t>(sample_index)];
    }
    return state.samples[
        static_cast<size_t>(sample_index - state.sample_base)];
}

bool incremental_frame_ready(
    const IncrementalMelState::Impl & state,
    const MelConfig & config,
    bool finalized) {
    if (finalized) {
        return state.next_frame <=
            state.total_samples / config.hop_length;
    }
    const int64_t center =
        state.next_frame * config.hop_length;
    const int64_t padding = config.n_fft / 2;
    int64_t maximum_sample = center + padding - 1;
    if (center < padding) {
        maximum_sample = std::max(
            maximum_sample, padding - center);
    }
    return maximum_sample < state.total_samples;
}

void prepare_incremental_window(
    IncrementalMelState::Impl & state,
    const MelConfig & config) {
    if (state.window_n_fft == config.n_fft &&
        same_table(state.window_src, config.window)) {
        return;
    }
    state.window = make_padded_window(
        config.window, config.n_fft);
    state.window_n_fft = config.n_fft;
    state.window_src = config.window;
    state.windowed.resize(static_cast<size_t>(config.n_fft));
    state.power.resize(
        static_cast<size_t>(config.n_fft / 2 + 1));
    state.fft.resize(static_cast<size_t>(config.n_fft / 2));
}

void compute_incremental_frame(
    IncrementalMelState::Impl & state,
    const MelConfig & config,
    bool finalized,
    float * destination) {
    const int64_t center =
        state.next_frame * config.hop_length;
    const int64_t padding = config.n_fft / 2;
    for (int index = 0; index < config.n_fft; ++index) {
        const int64_t sample_index =
            center + index - padding;
        state.windowed[static_cast<size_t>(index)] =
            incremental_sample_at(state, sample_index, finalized) *
            state.window[static_cast<size_t>(index)];
    }
    rfft_power_radix2(
        state.windowed.data(),
        state.power.data(),
        config.n_fft,
        state.fft.data());

    const int bins = config.n_fft / 2 + 1;
    if (state.filter_bands.size() != static_cast<size_t>(config.n_mels) ||
        !same_table(state.filterbank_src, config.filterbank)) {
        state.filter_bands   = make_filter_bands(config.filterbank, config.n_mels, bins);
        state.filterbank_src = config.filterbank;
    }
    log_mel_frame(state.power.data(), bins, config.n_mels, config.filterbank.data(),
                  state.filter_bands.data(), config.log_zero_guard_value, destination);
}

void compact_incremental_samples(
    IncrementalMelState::Impl & state,
    const MelConfig & config) {
    const int64_t next_left =
        state.next_frame * config.hop_length -
        config.n_fft / 2;
    const int64_t keep_from = std::max<int64_t>(
        state.sample_base, std::max<int64_t>(0, next_left));
    const int64_t remove_count =
        std::min<int64_t>(
            keep_from - state.sample_base,
            static_cast<int64_t>(state.samples.size()));
    if (remove_count <= 0) {
        return;
    }
    state.samples.erase(
        state.samples.begin(),
        state.samples.begin() +
            static_cast<std::ptrdiff_t>(remove_count));
    state.sample_base += remove_count;
}

}

int append_log_mel(
    const float * samples,
    int n_samples,
    bool finalize,
    const MelConfig & config,
    IncrementalMelState & state,
    std::vector<float> & out_mel,
    int & out_n_frames) {
    out_mel.clear();
    out_n_frames = 0;
    if (!state.impl || state.impl->finalized) {
        return finalize ? 0 : 1;
    }
    if (config.normalize != MelNormalize::None) {
        return 2;
    }
    if (n_samples < 0 || (n_samples > 0 && !samples)) {
        return 3;
    }
    if (config.filterbank.size() !=
            static_cast<size_t>(
                config.n_mels * (config.n_fft / 2 + 1)) ||
        static_cast<int>(config.window.size()) !=
            config.win_length) {
        return 4;
    }
    if (n_samples > 0) {
        append_preemphasized_samples(
            *state.impl,
            samples,
            n_samples,
            config.preemph,
            config.n_fft / 2 + 1);
    }
    if (finalize && state.impl->total_samples == 0) {
        state.impl->finalized = true;
        return 0;
    }

    prepare_incremental_window(*state.impl, config);
    const int64_t first_frame = state.impl->next_frame;
    while (incremental_frame_ready(
        *state.impl, config, finalize)) {
        out_mel.resize(
            out_mel.size() +
            static_cast<size_t>(config.n_mels));
        float * destination =
            out_mel.data() + out_mel.size() - config.n_mels;
        const int64_t valid_frames =
            std::max<int64_t>(
                1,
                state.impl->total_samples / config.hop_length);
        if (finalize &&
            state.impl->next_frame >= valid_frames) {
            std::fill(
                destination,
                destination + config.n_mels,
                0.0f);
        } else {
            compute_incremental_frame(
                *state.impl, config, finalize, destination);
        }
        ++state.impl->next_frame;
    }
    out_n_frames = static_cast<int>(
        state.impl->next_frame - first_frame);
    compact_incremental_samples(*state.impl, config);
    if (finalize) {
        state.impl->finalized = true;
    }
    return 0;
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

void MelState::release_scratch() {
    std::vector<float>().swap(x_padded);
    std::vector<float>().swap(power);
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
