#include "denoiser.h"

namespace tts_cpp::lavasr {

int denoiser_work_sample_rate(const DenoiserWeights & w) {
    // Carried in the GGUF metadata (lavasr.denoiser.work_sample_rate).
    return w.work_sample_rate;
}

std::vector<float> denoise(const DenoiserWeights & w,
                           const std::vector<float> & pcm_in, int sr_in) {
    if (pcm_in.empty()) {
        return {};
    }
    // TODO(QVAC-16579 follow-up: LavaSR denoiser stage): resample sr_in->work_sr,
    // STFT, UL-UNAS forward (denoiser_core), ISTFT, resample back to sr_in.  The
    // scaffold delegates straight to the (unimplemented) core so the wiring and
    // signatures are already in place; denoiser_forward() throws for now.
    (void) sr_in;
    return denoiser_forward(w, pcm_in);
}

} // namespace tts_cpp::lavasr
