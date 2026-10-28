#pragma once

#include "../backend_selection.h"

namespace tts_cpp::audio8 {

// Backends the automatic GPU selection may pick for the Audio8 LM and codec:
// the ones its output has been validated on. WebGPU joined once ggml-webgpu
// honoured the GGML_PREC_F32 the LM and codec products request.
inline ::tts_cpp::detail::GpuBackendRequirement gpu_requirement() {
    using ::tts_cpp::detail::GpuBackendRequirement;
    return GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal | GpuBackendRequirement::OpenCL |
           GpuBackendRequirement::CUDA | GpuBackendRequirement::WebGPU;
}

} // namespace tts_cpp::audio8
