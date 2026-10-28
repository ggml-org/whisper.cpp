#pragma once

// Which backend a GPU arm asked for. ctest names it in AUDIO8_TEST_GPU
// ("cuda", "metal", "vulkan", "opencl", "webgpu"); a CPU arm leaves the variable unset.
//
// Both GPU arms run the same binaries and backend selection picks on its own
// preference order, so an arm that does not check its backend passes on
// whatever it was given -- the other arm's GPU when both are compiled in, or
// the CPU when the GPU failed to come up. Every numeric bar below still holds
// in those cases, which is what makes them worth asserting first.

#include "backend_util.h"

#include "ggml-backend.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace audio8_test {

constexpr const char * METAL_ARM = "metal";
constexpr const char * VULKAN_ARM = "vulkan";
constexpr const char * OPENCL_ARM = "opencl";
constexpr const char * CUDA_ARM = "cuda";
constexpr const char * WEBGPU_ARM = "webgpu";

inline std::string requested_gpu() {
    const char * name = std::getenv("AUDIO8_TEST_GPU");
    return name ? name : "";
}

inline bool is_gpu_test() {
    return !requested_gpu().empty();
}

// ggml names an instance <registry name><device index>: "MTL0", "Vulkan0".
inline std::string registry_of(const std::string & instance) {
    size_t end = instance.size();
    while (end > 0 && std::isdigit(static_cast<unsigned char>(instance[end - 1]))) --end;
    return instance.substr(0, end);
}

// backend_util owns every accepted spelling, ggml-metal's "MTL" among them.
inline bool registry_is_requested(const char * registry) {
    const std::string want = requested_gpu();
    if (want == METAL_ARM) return tts_cpp::detail::reg_name_is_metal(registry);
    if (want == VULKAN_ARM) return tts_cpp::detail::reg_name_is_vulkan(registry);
    if (want == OPENCL_ARM) return tts_cpp::detail::reg_name_is_opencl(registry);
    if (want == CUDA_ARM) return tts_cpp::detail::reg_name_is_cuda(registry);
    if (want == WEBGPU_ARM) return tts_cpp::detail::reg_name_is_webgpu(registry);
    return false;
}

// For callers holding only the public Engine::backend_name().
inline bool instance_is_requested(const std::string & instance) {
    return registry_is_requested(registry_of(instance).c_str());
}

inline bool report_wrong_gpu(const char * tag, const std::string & got) {
    std::fprintf(stderr, "%s: FAIL expected the %s backend, got %s\n", tag,
                 requested_gpu().c_str(), got.empty() ? "(none)" : got.c_str());
    return false;
}

inline bool is_discrete_vulkan(ggml_backend_dev_t device) {
    return ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
           tts_cpp::detail::reg_name_is_vulkan(
               ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)));
}

inline bool discrete_vulkan_visible() {
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        if (is_discrete_vulkan(ggml_backend_dev_get(index))) return true;
    }
    return false;
}

// A Vulkan arm on a host with a discrete adapter must not land on an integrated
// one: it is slower, and the RADV iGPU of a Ryzen 9950X3D misses the GPU bars.
inline bool on_preferred_vulkan_device(const char * tag, ggml_backend_t backend) {
    if (requested_gpu() != VULKAN_ARM || !discrete_vulkan_visible()) return true;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) return true;
    std::fprintf(stderr, "%s: FAIL ran on the integrated Vulkan adapter %s while a discrete "
                         "one is visible\n",
                 tag, device ? ggml_backend_dev_description(device) : "(none)");
    return false;
}

inline bool check_requested_gpu(const char * tag, ggml_backend_t backend) {
    if (!is_gpu_test()) return true;
    if (backend && registry_is_requested(tts_cpp::detail::backend_reg_name(backend))) {
        return on_preferred_vulkan_device(tag, backend);
    }
    return report_wrong_gpu(tag, backend ? ggml_backend_name(backend) : "");
}

}  // namespace audio8_test
