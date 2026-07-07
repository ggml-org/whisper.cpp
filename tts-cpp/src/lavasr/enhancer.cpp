#include "enhancer.h"

#include "dsp/fastlr_merge.h"
#include "dsp/mel_filterbank.h"
#include "dsp/resampler.h"
#include "dsp/stft_processor.h"

#include <complex>

namespace tts_cpp::lavasr {

int enhancer_work_sample_rate(const EnhancerWeights & w) {
    // Carried in the GGUF metadata (lavasr.enhancer.work_sample_rate; default
    // 48000 for the current LavaSR enhancer).
    return w.work_sample_rate;
}

std::vector<float> enhance_with(const EnhancerWeights & w,
                                const std::vector<float> & pcm_in, int sr_in,
                                const SpecForwardFn & spec_fwd) {
    if (pcm_in.empty()) {
        return {};
    }

    const int work_sr = enhancer_work_sample_rate(w);

    // 1) Upsample to the enhancer working rate (48 kHz).
    std::vector<float> wav = dsp::Resampler::resample(pcm_in, sr_in, work_sr);

    // 2) Log-mel (Slaney mel computed at the reference rate from GGUF metadata
    //    — 44.1k on 48k audio for the current model, matching the upstream
    //    LavaSR / Vocos training configuration).
    dsp::MelFilterbank mel_fb(/*sample_rate=*/w.mel_ref_sample_rate, w.n_fft,
                              w.n_mels, /*f_min=*/0.0f, /*f_max=*/8000.0f);
    const auto mel = mel_fb.mel_spectrogram(wav, w.hop); // [n_mels][T]
    const int  T   = mel.empty() ? 0 : static_cast<int>(mel[0].size());
    if (T == 0) {
        return wav; // too short to enhance — return the upsampled signal
    }

    std::vector<float> mel_flat(static_cast<size_t>(w.n_mels) * T);
    for (int c = 0; c < w.n_mels; c++) {
        for (int t = 0; t < T; t++) {
            mel_flat[static_cast<size_t>(c) * T + t] = mel[c][t];
        }
    }

    // 3) Backbone + spec head -> real/imag [spec_bins][T] (scalar or GPU core).
    std::vector<float> real, imag;
    spec_fwd(mel_flat, T, real, imag);

    // 4) ISTFT back to a 48 kHz waveform.
    dsp::Spectrogram spec(T, std::vector<std::complex<float>>(w.spec_bins));
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < w.spec_bins; f++) {
            spec[t][f] = {real[static_cast<size_t>(f) * T + t],
                          imag[static_cast<size_t>(f) * T + t]};
        }
    }
    dsp::StftProcessor stft(w.n_fft, w.hop, w.win, /*center_pad=*/false);
    std::vector<float> enhanced =
        stft.istft(spec, static_cast<int>(wav.size()));

    // 5) FastLR crossover: keep the original low band (below the engine's
    //    Nyquist), take the synthesised high band from the network.
    const int cutoff_hz = sr_in / 2;
    return dsp::FastLRMerge::merge(enhanced, wav, work_sr, cutoff_hz);
}

std::vector<float> enhance(const EnhancerWeights & w,
                           const std::vector<float> & pcm_in, int sr_in) {
    return enhance_with(w, pcm_in, sr_in,
                        [&w](const std::vector<float> & mel, int T,
                             std::vector<float> & real, std::vector<float> & imag) {
                            enhancer_spec_forward(w, mel, T, real, imag);
                        });
}

} // namespace tts_cpp::lavasr
