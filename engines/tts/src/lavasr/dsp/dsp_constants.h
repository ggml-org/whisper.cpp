#pragma once

namespace tts_cpp::lavasr::dsp {

// Shared math constant for the LavaSR DSP primitives.  Kept in one place so
// the resampler / STFT / mel / crossover code all use the exact same value
// as the @qvac/tts-onnx reference implementation this module is ported from.
constexpr double PI = 3.14159265358979323846;

} // namespace tts_cpp::lavasr::dsp
