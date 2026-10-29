#pragma once

#include <stdexcept>
#include <string>

// The accelerator backend a MOSS arm covers, named on its command line. A
// configured arm must fail when that backend is unavailable, never pass on the
// CPU or on another accelerator registered alongside it.
struct MossGpuArm {
    std::string request;  // TTS_CPP_GPU_BACKEND / TranscribeOptions::backend
    std::string prefix;   // ggml instance name: "CUDA0", "Vulkan0", "HTP0", "OpenCL", "WebGPU: WebGPU", "MTL0"
    std::string device;   // ggml device name, as fit results report it: "GPUOpenCL" for OpenCL, else as `prefix`

    bool selected(const char * name) const {
        if (!name) return false;
        const std::string value(name);
        return value.rfind(prefix, 0) == 0 || value.rfind(device, 0) == 0;
    }

    void require(const char * name) const {
        if (!selected(name)) {
            throw std::runtime_error("expected " + prefix + ", selected " + (name ? name : "null"));
        }
    }
};

inline MossGpuArm moss_gpu_arm(const std::string & request) {
    if (request == "cuda") return {request, "CUDA", "CUDA"};
    if (request == "vulkan") return {request, "Vulkan", "Vulkan"};
    if (request == "hexagon") return {request, "HTP", "HTP"};
    if (request == "opencl") return {request, "OpenCL", "GPUOpenCL"};
    if (request == "webgpu") return {request, "WebGPU", "WebGPU"};
    if (request == "metal") return {request, "MTL", "MTL"};
    throw std::runtime_error("unknown accelerator arm '" + request +
                             "' (expected cuda, vulkan, hexagon, opencl, webgpu or metal)");
}
