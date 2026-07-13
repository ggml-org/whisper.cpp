#include "overlap_add.h"

#include <algorithm>
#include <cmath>

namespace tts_cpp::lavasr::dsp {

namespace {

// Squared symmetric-Hann chunk-blend weights for the overlap-add stitch of the
// fixed-L chunks.  This is the SYMMETRIC Hann (denominator L-1, endpoints -> 0),
// squared, with a small floor — a byte-for-byte match of the reference
// @qvac/tts-onnx LavaSRDenoiser::buildChunkWeights().  Note this is deliberately
// a different window from the STFT analysis window (StftProcessor uses the
// PERIODIC Hann, denominator L); do not "unify" them.
std::vector<float> chunk_weights(int L) {
    std::vector<float> w(L);
    for (int i = 0; i < L; i++) {
        const float h = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / (L - 1)));
        w[i]          = std::max(h * h, 1e-4f);
    }
    return w;
}

} // namespace

void overlap_add_normalize(const std::vector<int> & starts, int T, int F, int L,
                           const ChunkPlanes & chunk_out,
                           std::vector<float> & outRe, std::vector<float> & outIm) {
    outRe.assign(static_cast<size_t>(T) * F, 0.0f);
    outIm.assign(static_cast<size_t>(T) * F, 0.0f);
    if (T <= L) {
        const std::pair<const float *, const float *> p = chunk_out(0);
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                outRe[static_cast<size_t>(t) * F + f] = p.first[static_cast<size_t>(t) * F + f];
                outIm[static_cast<size_t>(t) * F + f] = p.second[static_cast<size_t>(t) * F + f];
            }
        return;
    }
    const std::vector<float> cw = chunk_weights(L);
    const int                Nc = static_cast<int>(starts.size());
    std::vector<std::pair<const float *, const float *>> planes(Nc);
    for (int c = 0; c < Nc; c++) planes[c] = chunk_out(c);
    std::vector<float>       accR(static_cast<size_t>(T) * F, 0.0f);
    std::vector<float>       accI(static_cast<size_t>(T) * F, 0.0f);
    std::vector<float>       wacc(T, 0.0f);
    // Gather per output row over chunks in ascending c — same accumulation order as the
    // serial overlap-add, so bit-identical, and each row is written by one thread.
    #pragma omp parallel for schedule(static)
    for (int to = 0; to < T; to++) {
        float wsum = 0.0f;
        for (int c = 0; c < Nc; c++) {
            const int t = to - starts[c];
            if (t < 0 || t >= L) continue;
            const float ww = cw[t];
            for (int f = 0; f < F; f++) {
                accR[static_cast<size_t>(to) * F + f] += planes[c].first[static_cast<size_t>(t) * F + f] * ww;
                accI[static_cast<size_t>(to) * F + f] += planes[c].second[static_cast<size_t>(t) * F + f] * ww;
            }
            wsum += ww;
        }
        wacc[to] = wsum;
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float ww = std::max(wacc[t], 1e-6f);
        for (int f = 0; f < F; f++) {
            outRe[static_cast<size_t>(t) * F + f] = accR[static_cast<size_t>(t) * F + f] / ww;
            outIm[static_cast<size_t>(t) * F + f] = accI[static_cast<size_t>(t) * F + f] / ww;
        }
    }
}

} // namespace tts_cpp::lavasr::dsp
