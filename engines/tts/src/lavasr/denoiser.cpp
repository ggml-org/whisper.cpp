#include "denoiser.h"

#include "dsp/overlap_add.h"
#include "dsp/resampler.h"
#include "dsp/stft_processor.h"

#include <algorithm>
#include <complex>
#include <utility>
#include <vector>

namespace tts_cpp::lavasr {

int denoiser_work_sample_rate(const DenoiserWeights & w) {
    // Carried in the GGUF metadata (lavasr.denoiser.work_sample_rate; 16 kHz for
    // the shipped UL-UNAS model).
    return w.work_sample_rate;
}

namespace {

struct PreppedSpec {
    std::vector<float> wav;    // input resampled to the work rate
    std::vector<float> re, im; // spectrogram planes, [T * spec_bins] row-major
    int                T = 0;
};

// Resample to the work rate and STFT into planes.  Returns false when the wrapper
// must return `early` unchanged: empty pcm -> {}; empty resampled wav -> pcm_in;
// T==0 -> resample(wav) back (deliberately NOT resized to the pcm length).
bool denoise_prep(const DenoiserWeights & w, const std::vector<float> & pcm_in, int sr_in,
                  PreppedSpec & p, std::vector<float> & early) {
    if (pcm_in.empty()) {
        early = {};
        return false;
    }
    const int work_sr = denoiser_work_sample_rate(w);
    p.wav             = dsp::Resampler::resample(pcm_in, sr_in, work_sr);
    if (p.wav.empty()) {
        early = pcm_in;
        return false;
    }
    dsp::StftProcessor stft(w.n_fft, w.hop, w.win, /*center_pad=*/true);
    dsp::Spectrogram   spec = stft.stft(p.wav);
    p.T                     = static_cast<int>(spec.size());
    const int F             = w.spec_bins;
    if (p.T == 0) {
        early = dsp::Resampler::resample(p.wav, work_sr, sr_in);
        return false;
    }
    p.re.resize(static_cast<size_t>(p.T) * F);
    p.im.resize(static_cast<size_t>(p.T) * F);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < p.T; t++) {
        for (int f = 0; f < F; f++) {
            p.re[static_cast<size_t>(t) * F + f] = spec[t][f].real();
            p.im[static_cast<size_t>(t) * F + f] = spec[t][f].imag();
        }
    }
    return true;
}

// Chunk starts: {0} when T<=L; else 0,H,2H,... plus a forced tail start at T-L.
std::vector<int> compute_starts(int T, int L, int H) {
    std::vector<int> starts;
    if (T <= L) {
        starts.push_back(0);
        return starts;
    }
    for (int s = 0; s <= T - L; s += H) {
        starts.push_back(s);
    }
    if (starts.back() != T - L) {
        starts.push_back(T - L);
    }
    return starts;
}

// Copy frames [start, start + min(L, T-start)) into an [L][F] chunk, zero-padding the tail.
void extract_chunk(const std::vector<float> & re, const std::vector<float> & im,
                   int start, int T, int F, int L, float * cr, float * ci) {
    std::fill(cr, cr + static_cast<size_t>(L) * F, 0.0f);
    std::fill(ci, ci + static_cast<size_t>(L) * F, 0.0f);
    const int len = std::min(L, T - start);
    for (int t = 0; t < len; t++)
        for (int f = 0; f < F; f++) {
            cr[static_cast<size_t>(t) * F + f] = re[static_cast<size_t>(start + t) * F + f];
            ci[static_cast<size_t>(t) * F + f] = im[static_cast<size_t>(start + t) * F + f];
        }
}

// ISTFT the masked planes back to the work-rate waveform, resample to sr_in, and
// pad/trim to the input length (rate/length-preserving contract).
std::vector<float> denoise_finish(const DenoiserWeights & w, int sr_in, size_t pcm_len,
                                  const PreppedSpec & p,
                                  const std::vector<float> & outRe, const std::vector<float> & outIm) {
    const int          T = p.T, F = w.spec_bins;
    dsp::StftProcessor stft(w.n_fft, w.hop, w.win, /*center_pad=*/true);
    dsp::Spectrogram   out(T, dsp::ComplexVec(F));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            out[t][f] = {outRe[static_cast<size_t>(t) * F + f], outIm[static_cast<size_t>(t) * F + f]};
    std::vector<float> wav_dn  = stft.istft(out, static_cast<int>(p.wav.size()));
    std::vector<float> cleaned = dsp::Resampler::resample(wav_dn, denoiser_work_sample_rate(w), sr_in);
    cleaned.resize(pcm_len, 0.0f);
    return cleaned;
}

} // namespace

std::vector<float> denoise_with_core(const DenoiserWeights & w, const std::vector<float> & pcm_in,
                                     int sr_in, const DenoiseChunkCore & core) {
    PreppedSpec        p;
    std::vector<float> early;
    if (!denoise_prep(w, pcm_in, sr_in, p, early)) {
        return early;
    }
    const int              F      = w.spec_bins, L = w.chunk_frames;
    const std::vector<int> starts = compute_starts(p.T, L, w.chunk_hop);
    const int              Nc     = static_cast<int>(starts.size());

    // Run the core once per chunk, stacking the outputs [Nc][L*F] for overlap-add.
    std::vector<float> orr(static_cast<size_t>(Nc) * L * F), oii(static_cast<size_t>(Nc) * L * F);
    std::vector<float> cr(static_cast<size_t>(L) * F), ci(static_cast<size_t>(L) * F);
    for (int c = 0; c < Nc; c++) {
        extract_chunk(p.re, p.im, starts[c], p.T, F, L, cr.data(), ci.data());
        std::vector<float> orc, oic;
        core(cr, ci, L, orc, oic);
        std::copy(orc.begin(), orc.end(), orr.begin() + static_cast<size_t>(c) * L * F);
        std::copy(oic.begin(), oic.end(), oii.begin() + static_cast<size_t>(c) * L * F);
    }

    std::vector<float> outRe, outIm;
    dsp::overlap_add_normalize(starts, p.T, F, L,
                          [&](int c) {
                              return std::make_pair(orr.data() + static_cast<size_t>(c) * L * F,
                                                    oii.data() + static_cast<size_t>(c) * L * F);
                          },
                          outRe, outIm);
    return denoise_finish(w, sr_in, pcm_in.size(), p, outRe, outIm);
}

std::vector<float> denoise_with_batch_core(const DenoiserWeights & w, const std::vector<float> & pcm_in,
                                           int sr_in, const DenoiseBatchCore & core) {
    PreppedSpec        p;
    std::vector<float> early;
    if (!denoise_prep(w, pcm_in, sr_in, p, early)) {
        return early;
    }
    const int              F      = w.spec_bins, L = w.chunk_frames;
    const std::vector<int> starts = compute_starts(p.T, L, w.chunk_hop);
    const int              Nc     = static_cast<int>(starts.size());

    // Extract all chunks stacked [Nc][L*F] and run the core once for the whole clip.
    std::vector<float> cre(static_cast<size_t>(Nc) * L * F), cim(static_cast<size_t>(Nc) * L * F);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < Nc; c++) {
        extract_chunk(p.re, p.im, starts[c], p.T, F, L,
                      cre.data() + static_cast<size_t>(c) * L * F,
                      cim.data() + static_cast<size_t>(c) * L * F);
    }
    std::vector<float> orr, oii;
    core(cre, cim, L, Nc, orr, oii);

    std::vector<float> outRe, outIm;
    dsp::overlap_add_normalize(starts, p.T, F, L,
                          [&](int c) {
                              return std::make_pair(orr.data() + static_cast<size_t>(c) * L * F,
                                                    oii.data() + static_cast<size_t>(c) * L * F);
                          },
                          outRe, outIm);
    return denoise_finish(w, sr_in, pcm_in.size(), p, outRe, outIm);
}

std::vector<float> denoise(const DenoiserWeights & w, const std::vector<float> & pcm_in, int sr_in) {
    return denoise_with_core(
        w, pcm_in, sr_in,
        [&w](const std::vector<float> & re, const std::vector<float> & im, int L,
             std::vector<float> & orr, std::vector<float> & oii) {
            denoiser_net_forward(w, re, im, L, orr, oii);
        });
}

} // namespace tts_cpp::lavasr
