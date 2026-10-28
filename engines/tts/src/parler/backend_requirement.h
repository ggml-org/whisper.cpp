#pragma once

#include "../backend_selection.h"

namespace tts_cpp::parler {

// Backends the complete Parler pipeline (T5, autoregressive decode, DAC
// synthesis) has been validated against end to end: Metal, Vulkan, OpenCL
// (Adreno) and CUDA against the reference fixtures, WebGPU (Dawn on an RTX
// 5090) against the CPU render. Anything else falls back to the CPU.
inline ::tts_cpp::detail::GpuBackendRequirement gpu_requirement() {
    using ::tts_cpp::detail::GpuBackendRequirement;
    return GpuBackendRequirement::Metal | GpuBackendRequirement::Vulkan | GpuBackendRequirement::OpenCL |
           GpuBackendRequirement::CUDA | GpuBackendRequirement::WebGPU;
}

} // namespace tts_cpp::parler
