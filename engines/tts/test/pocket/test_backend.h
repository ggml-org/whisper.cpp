#pragma once

#include "pocket/backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

inline std::string pocket_requested_test_backend() {
    const char * value = std::getenv("TTS_CPP_GPU_BACKEND");
    std::string backend = value ? value : "";
    std::transform(backend.begin(), backend.end(), backend.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    return backend;
}

inline tts_cpp::pocket::detail::Backend pocket_test_backend(const std::string & option) {
    return tts_cpp::pocket::detail::select_backend(option == "--gpu", option == "--gpu" ? "" : option);
}

inline void check_pocket_backend(const char * actual, bool gpu) {
    const std::string name = actual ? actual : "";
    if (!gpu) {
        if (name != "CPU") throw std::runtime_error("default backend changed: " + name);
        return;
    }
    const auto requested = pocket_requested_test_backend();
    const bool metal = name.find("MTL") == 0 || name == "Metal";
    const bool cuda = name.find("CUDA") == 0;
    const bool vulkan = name.find("Vulkan") == 0;
    const bool opencl = name.find("OpenCL") == 0 || name.find("GPUOpenCL") == 0;
    const bool matches = requested.empty() ? metal || cuda || vulkan || opencl :
        (requested == "metal" && metal) || (requested == "cuda" && cuda) ||
        (requested == "vulkan" && vulkan) || (requested == "opencl" && opencl);
    if (!matches) throw std::runtime_error("requested GPU backend " + requested + " but selected " + name);
}
