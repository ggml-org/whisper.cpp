#pragma once

#include <stdexcept>
#include <string>

// The accelerator backend a MOSS arm covers, named on its command line. A
// configured arm must fail when that backend is unavailable, never pass on the
// CPU or on another accelerator registered alongside it.
struct MossGpuArm {
    std::string request;  // TTS_CPP_GPU_BACKEND / TranscribeOptions::backend
    std::string prefix;   // ggml instance name: "CUDA0", "Vulkan0", "HTP0"

    bool selected(const char * name) const {
        return name && std::string(name).rfind(prefix, 0) == 0;
    }

    void require(const char * name) const {
        if (!selected(name)) {
            throw std::runtime_error("expected " + prefix + ", selected " + (name ? name : "null"));
        }
    }
};

inline MossGpuArm moss_gpu_arm(const std::string & request) {
    if (request == "cuda") return {request, "CUDA"};
    if (request == "vulkan") return {request, "Vulkan"};
    if (request == "hexagon") return {request, "HTP"};
    throw std::runtime_error("unknown accelerator arm '" + request + "' (expected cuda, vulkan or hexagon)");
}
