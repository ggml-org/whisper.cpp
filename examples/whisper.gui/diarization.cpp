#include "diarization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace diarize {

namespace {

// ---- small radix-2 FFT (in-place, complex) ----------------------------------
// re/im must have length n, a power of two.
void fft(std::vector<float> & re, std::vector<float> & im) {
    const int n = (int) re.size();
    // bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / len;
        const float wlen_re = (float) std::cos(ang);
        const float wlen_im = (float) std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                const float u_re = re[i + k];
                const float u_im = im[i + k];
                const float v_re = re[i + k + len / 2] * w_re - im[i + k + len / 2] * w_im;
                const float v_im = re[i + k + len / 2] * w_im + im[i + k + len / 2] * w_re;
                re[i + k]           = u_re + v_re;
                im[i + k]           = u_im + v_im;
                re[i + k + len / 2] = u_re - v_re;
                im[i + k + len / 2] = u_im - v_im;
                const float nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
    }
}

float hz_to_mel(float hz)  { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

// MFCC configuration (typical speaker-feature settings)
constexpr int   kFrame  = 400;   // 25 ms @ 16 kHz
constexpr int   kHop    = 160;   // 10 ms
constexpr int   kFFT    = 512;
constexpr int   kNMel   = 26;
constexpr int   kNMFCC  = 13;

// Build the triangular mel filterbank as [kNMel][kFFT/2+1] weights (cached).
const std::vector<std::vector<float>> & mel_filterbank(int sample_rate) {
    static std::vector<std::vector<float>> fb;
    static int cached_sr = -1;
    if (cached_sr == sample_rate && !fb.empty()) {
        return fb;
    }
    cached_sr = sample_rate;

    const int   n_bins = kFFT / 2 + 1;
    const float mel_lo = hz_to_mel(0.0f);
    const float mel_hi = hz_to_mel(sample_rate / 2.0f);

    std::vector<float> bin(kNMel + 2);
    for (int i = 0; i < kNMel + 2; ++i) {
        const float mel = mel_lo + (mel_hi - mel_lo) * i / (kNMel + 1);
        const float hz  = mel_to_hz(mel);
        bin[i] = std::floor((kFFT + 1) * hz / sample_rate);
    }

    fb.assign(kNMel, std::vector<float>(n_bins, 0.0f));
    for (int m = 1; m <= kNMel; ++m) {
        const int f0 = (int) bin[m - 1];
        const int f1 = (int) bin[m];
        const int f2 = (int) bin[m + 1];
        for (int k = f0; k < f1; ++k) {
            if (k >= 0 && k < n_bins && f1 != f0) fb[m - 1][k] = (float) (k - f0) / (f1 - f0);
        }
        for (int k = f1; k < f2; ++k) {
            if (k >= 0 && k < n_bins && f2 != f1) fb[m - 1][k] = (float) (f2 - k) / (f2 - f1);
        }
    }
    return fb;
}

} // namespace

std::vector<float> compute_embedding(const float * samples, int n_samples, int sample_rate) {
    if (samples == nullptr || n_samples <= 0) {
        return {};
    }

    const auto & fb = mel_filterbank(sample_rate);

    // precompute Hamming window and DCT matrix
    static std::vector<float> hamming;
    if ((int) hamming.size() != kFrame) {
        hamming.resize(kFrame);
        for (int i = 0; i < kFrame; ++i) {
            hamming[i] = 0.54f - 0.46f * std::cos(2.0f * (float) M_PI * i / (kFrame - 1));
        }
    }

    // accumulate per-coefficient mean and variance across frames (Welford-free,
    // two-moment) so the embedding is a fixed size regardless of segment length
    std::vector<double> sum(kNMFCC, 0.0), sum2(kNMFCC, 0.0);
    int n_frames = 0;

    std::vector<float> re(kFFT), im(kFFT);
    std::vector<float> mfcc(kNMFCC);

    // at least one frame even for very short spans
    const int last = std::max(0, n_samples - kFrame);
    for (int start = 0; start <= last; start += kHop) {
        // windowed frame -> zero-padded FFT buffer
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        const int len = std::min(kFrame, n_samples - start);
        for (int i = 0; i < len; ++i) {
            re[i] = samples[start + i] * hamming[i];
        }

        fft(re, im);

        // power spectrum -> mel energies -> log
        std::vector<float> mel(kNMel, 0.0f);
        const int n_bins = kFFT / 2 + 1;
        for (int m = 0; m < kNMel; ++m) {
            float e = 0.0f;
            for (int k = 0; k < n_bins; ++k) {
                const float power = re[k] * re[k] + im[k] * im[k];
                e += power * fb[m][k];
            }
            mel[m] = std::log(e + 1e-10f);
        }

        // DCT-II -> MFCCs
        for (int c = 0; c < kNMFCC; ++c) {
            float acc = 0.0f;
            for (int m = 0; m < kNMel; ++m) {
                acc += mel[m] * std::cos((float) M_PI * c * (m + 0.5f) / kNMel);
            }
            mfcc[c] = acc;
            sum[c]  += mfcc[c];
            sum2[c] += (double) mfcc[c] * mfcc[c];
        }
        ++n_frames;
    }

    if (n_frames == 0) {
        return {};
    }

    // embedding = [mean(0..12), std(0..12)]  -> 26-dim
    std::vector<float> emb(2 * kNMFCC);
    for (int c = 0; c < kNMFCC; ++c) {
        const double mean = sum[c] / n_frames;
        const double var  = std::max(0.0, sum2[c] / n_frames - mean * mean);
        emb[c]          = (float) mean;
        emb[kNMFCC + c] = (float) std::sqrt(var);
    }

    // L2-normalize so clustering can use cosine distance
    double norm = 0.0;
    for (float v : emb) norm += (double) v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-9) {
        for (float & v : emb) v = (float) (v / norm);
    }
    return emb;
}

namespace {

float cosine_distance(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 2.0f; // maximally distant
    }
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return 1.0f - dot; // vectors are L2-normalized
}

} // namespace

std::vector<int> cluster(const std::vector<std::vector<float>> & embeddings,
                         int num_speakers, float threshold) {
    const int n = (int) embeddings.size();
    std::vector<int> labels(n, 0);
    if (n == 0) {
        return labels;
    }

    // pairwise point distance matrix
    std::vector<std::vector<float>> dist(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const float d = cosine_distance(embeddings[i], embeddings[j]);
            dist[i][j] = dist[j][i] = d;
        }
    }

    // each point starts in its own cluster
    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i] = {i};

    auto avg_linkage = [&](const std::vector<int> & a, const std::vector<int> & b) {
        float s = 0.0f;
        for (int x : a) for (int y : b) s += dist[x][y];
        return s / (a.size() * b.size());
    };

    const int target = num_speakers > 0 ? std::min(num_speakers, n) : 1;

    while ((int) clusters.size() > target) {
        // find the closest pair of clusters
        int    bi = -1, bj = -1;
        float  best = std::numeric_limits<float>::max();
        for (int i = 0; i < (int) clusters.size(); ++i) {
            for (int j = i + 1; j < (int) clusters.size(); ++j) {
                const float d = avg_linkage(clusters[i], clusters[j]);
                if (d < best) { best = d; bi = i; bj = j; }
            }
        }
        if (bi < 0) break;

        // auto mode: stop once the closest clusters exceed the threshold
        if (num_speakers <= 0 && best > threshold) {
            break;
        }

        // merge bj into bi
        clusters[bi].insert(clusters[bi].end(), clusters[bj].begin(), clusters[bj].end());
        clusters.erase(clusters.begin() + bj);
    }

    // assign final labels in order of first appearance so Speaker 0 is whoever
    // speaks first, Speaker 1 next, etc. (stable, intuitive numbering)
    std::vector<int> first_index(clusters.size());
    for (int c = 0; c < (int) clusters.size(); ++c) {
        first_index[c] = *std::min_element(clusters[c].begin(), clusters[c].end());
    }
    std::vector<int> order(clusters.size());
    for (int c = 0; c < (int) clusters.size(); ++c) order[c] = c;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return first_index[a] < first_index[b]; });

    for (int rank = 0; rank < (int) order.size(); ++rank) {
        for (int idx : clusters[order[rank]]) {
            labels[idx] = rank;
        }
    }
    return labels;
}

} // namespace diarize
