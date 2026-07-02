#include "denoiser_core.h"

#include <stdexcept>

namespace tts_cpp::lavasr {

const DnTensor & DenoiserWeights::get(const std::string & name) const {
    auto it = t.find(name);
    if (it == t.end()) {
        throw std::runtime_error("lavasr denoiser: missing tensor '" + name + "'");
    }
    return it->second;
}

std::vector<float> denoiser_forward(const DenoiserWeights & /*w*/,
                                    const std::vector<float> & /*pcm*/) {
    // TODO(QVAC-16579 follow-up: LavaSR denoiser stage): port the UL-UNAS U-Net
    // forward — encoder/decoder efficient conv blocks + affine PReLU + cTFA
    // (GRU/BiGRU) + skip connections + mask -> ISTFT.  Kept as a scaffold so the
    // file/CMake structure and the public API land first; see denoiser_core.h
    // for the architecture and convert-lavasr-denoiser-to-gguf.py for weights.
    throw std::runtime_error(
        "lavasr denoiser: forward not yet implemented (scaffold — QVAC-16579 "
        "follow-up: LavaSR denoiser stage)");
}

} // namespace tts_cpp::lavasr
