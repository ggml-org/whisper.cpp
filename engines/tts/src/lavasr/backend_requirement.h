#pragma once

#include "../backend_selection.h"

namespace tts_cpp::lavasr {

// LavaSR runs its graphs directly on the selected backend, with no per-op CPU
// fallback, so only backends that implement its whole op set may be picked:
// WebGPU lacks MEAN and would abort the denoiser.
inline ::tts_cpp::detail::GpuBackendRequirement gpu_requirement() {
    using ::tts_cpp::detail::GpuBackendRequirement;
    return GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal |
           GpuBackendRequirement::OpenCL | GpuBackendRequirement::CUDA;
}

} // namespace tts_cpp::lavasr
