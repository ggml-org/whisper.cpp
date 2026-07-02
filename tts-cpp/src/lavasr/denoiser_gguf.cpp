#include "denoiser_gguf.h"

namespace tts_cpp::lavasr {

bool load_denoiser_gguf(const std::string & /*path*/, DenoiserWeights & /*out*/,
                        std::string * err) {
    // TODO(QVAC-16579 follow-up: LavaSR denoiser stage): read the
    // "lavasr-denoiser" GGUF via the ggml gguf reader (see enhancer_gguf.cpp for
    // the working reference) into DenoiserWeights — metadata (n_fft / hop /
    // win / spec_bins / work_sample_rate / freq_fold_r / ...) plus the
    // orientation-matched weight tensors.
    if (err) {
        *err = "lavasr denoiser: GGUF loader not yet implemented (scaffold — "
               "QVAC-16579 follow-up: LavaSR denoiser stage)";
    }
    return false;
}

} // namespace tts_cpp::lavasr
