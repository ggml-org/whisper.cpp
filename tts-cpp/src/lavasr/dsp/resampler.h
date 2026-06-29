#pragma once

#include <vector>

namespace tts_cpp::lavasr::dsp {

// Windowed-sinc (Lanczos) resampler used to move PCM between the TTS engine
// rate, the denoiser's 16 kHz working rate, the enhancer's 48 kHz working
// rate, and the caller-requested output rate.  Stateless: a single static
// entry point, identical in behaviour to the @qvac/tts-onnx Resampler.
class Resampler {
public:
    // Resample `input` from `sr_in` Hz to `sr_out` Hz.  Returns `input`
    // unchanged when the rates match or the input is empty.  Output length
    // is round(input.size() * sr_out / sr_in).
    static std::vector<float> resample(const std::vector<float> & input,
                                       int sr_in, int sr_out);
};

} // namespace tts_cpp::lavasr::dsp
