#include "denoiser.h"

#include "dsp/resampler.h"
#include "dsp/stft_processor.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace tts_cpp::lavasr {

int denoiser_work_sample_rate(const DenoiserWeights & w) {
    // Carried in the GGUF metadata (lavasr.denoiser.work_sample_rate; 16 kHz for
    // the shipped UL-UNAS model).
    return w.work_sample_rate;
}

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

std::vector<float> denoise(const DenoiserWeights & w,
                           const std::vector<float> & pcm_in, int sr_in) {
    if (pcm_in.empty()) {
        return {};
    }
    const int work_sr = denoiser_work_sample_rate(w);

    // 1) Resample to the network's 16 kHz working rate.
    std::vector<float> wav = dsp::Resampler::resample(pcm_in, sr_in, work_sr);
    if (wav.empty()) {
        return pcm_in;
    }

    // 2) STFT (center-padded periodic Hann, 512/256) -> [T][spec_bins].
    dsp::StftProcessor stft(w.n_fft, w.hop, w.win, /*center_pad=*/true);
    dsp::Spectrogram   spec = stft.stft(wav);
    const int          T    = static_cast<int>(spec.size());
    const int          F    = w.spec_bins;
    if (T == 0) {
        return dsp::Resampler::resample(wav, work_sr, sr_in);
    }

    std::vector<float> re(static_cast<size_t>(T) * F), im(static_cast<size_t>(T) * F);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < F; f++) {
            re[static_cast<size_t>(t) * F + f] = spec[t][f].real();
            im[static_cast<size_t>(t) * F + f] = spec[t][f].imag();
        }
    }

    // 3) UL-UNAS forward over fixed-L chunks with squared-Hann overlap-add
    //    (each chunk runs zero-state — matches the fixed-length ONNX export).
    const int          L = w.chunk_frames, H = w.chunk_hop;
    std::vector<float> outRe(static_cast<size_t>(T) * F, 0.0f);
    std::vector<float> outIm(static_cast<size_t>(T) * F, 0.0f);

    auto run_chunk = [&](int start, int len, std::vector<float> & orr, std::vector<float> & oii) {
        std::vector<float> cr(static_cast<size_t>(L) * F, 0.0f), ci(static_cast<size_t>(L) * F, 0.0f);
        for (int t = 0; t < len; t++)
            for (int f = 0; f < F; f++) {
                cr[static_cast<size_t>(t) * F + f] = re[static_cast<size_t>(start + t) * F + f];
                ci[static_cast<size_t>(t) * F + f] = im[static_cast<size_t>(start + t) * F + f];
            }
        denoiser_net_forward(w, cr, ci, L, orr, oii);
    };

    if (T <= L) {
        std::vector<float> orr, oii;
        run_chunk(0, T, orr, oii);
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                outRe[static_cast<size_t>(t) * F + f] = orr[static_cast<size_t>(t) * F + f];
                outIm[static_cast<size_t>(t) * F + f] = oii[static_cast<size_t>(t) * F + f];
            }
    } else {
        std::vector<int> starts;
        for (int s = 0; s <= T - L; s += H) {
            starts.push_back(s);
        }
        if (starts.back() != T - L) {
            starts.push_back(T - L);
        }
        const std::vector<float> cw = chunk_weights(L);
        std::vector<float>       accR(static_cast<size_t>(T) * F, 0.0f);
        std::vector<float>       accI(static_cast<size_t>(T) * F, 0.0f);
        std::vector<float>       wacc(T, 0.0f);
        for (int start : starts) {
            std::vector<float> orr, oii;
            run_chunk(start, L, orr, oii);
            for (int t = 0; t < L; t++) {
                const float ww = cw[t];
                for (int f = 0; f < F; f++) {
                    accR[static_cast<size_t>(start + t) * F + f] += orr[static_cast<size_t>(t) * F + f] * ww;
                    accI[static_cast<size_t>(start + t) * F + f] += oii[static_cast<size_t>(t) * F + f] * ww;
                }
                wacc[start + t] += ww;
            }
        }
        for (int t = 0; t < T; t++) {
            const float ww = std::max(wacc[t], 1e-6f);
            for (int f = 0; f < F; f++) {
                outRe[static_cast<size_t>(t) * F + f] = accR[static_cast<size_t>(t) * F + f] / ww;
                outIm[static_cast<size_t>(t) * F + f] = accI[static_cast<size_t>(t) * F + f] / ww;
            }
        }
    }

    // 4) ISTFT back to a 16 kHz waveform.
    dsp::Spectrogram out(T, dsp::ComplexVec(F));
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            out[t][f] = {outRe[static_cast<size_t>(t) * F + f], outIm[static_cast<size_t>(t) * F + f]};
    std::vector<float> wav_dn = stft.istft(out, static_cast<int>(wav.size()));

    // 5) Resample back to the caller's rate; keep the input length exactly
    //    (rate-preserving contract — callers splice this in place of the PCM).
    std::vector<float> cleaned = dsp::Resampler::resample(wav_dn, work_sr, sr_in);
    cleaned.resize(pcm_in.size(), 0.0f);
    return cleaned;
}

} // namespace tts_cpp::lavasr
