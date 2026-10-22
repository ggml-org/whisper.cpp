#include "backend_selection.h"

#include "test_env_portable.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char * VULKAN_BACKEND = "Vulkan";
constexpr const char * CUDA_BACKEND = "CUDA";
constexpr const char * CPU_BACKEND = "CPU";
constexpr const char * OPENCL_BACKEND = "OpenCL";
constexpr const char * OPENCL_DEVICE = "GPUOpenCL";
constexpr const char * HEXAGON_BACKEND = "HTP";
constexpr const char * HEXAGON_DEVICE = "HTP0";
constexpr const char * SECOND_HEXAGON_DEVICE = "HTP1";
constexpr const char * WEBGPU_BACKEND = "WebGPU";
constexpr const char * WEBGPU_DEVICE = "WebGPU";

int failures = 0;

void check(bool condition, const char * message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

// A GPU arm names its backend here, so an unusable value has to be rejected
// rather than skipping every device and leaving the arm to pass on the CPU.
// Only the unset and unrecognised cases are asserted: whether a recognised
// backend resolves is a property of the host, and it now throws when it does
// not, which is the behaviour a comparison run depends on.
bool selection_rejects(const char * forced) {
    if (forced) {
        setenv("TTS_CPP_GPU_BACKEND", forced, 1);
    } else {
        unsetenv("TTS_CPP_GPU_BACKEND");
    }
    try {
        tts_cpp::detail::init_gpu_backend(/*n_gpu_layers=*/1, /*verbose=*/false, "test");
    } catch (const std::exception &) {
        unsetenv("TTS_CPP_GPU_BACKEND");
        return true;
    }
    unsetenv("TTS_CPP_GPU_BACKEND");
    return false;
}

void check_backend_requests() {
    using tts_cpp::detail::backend_request_is_auto;
    using tts_cpp::detail::backend_request_matches;

    check(backend_request_is_auto(""), "an empty backend request must keep the policy walk");
    check(backend_request_is_auto("auto"), "auto must keep the policy walk");
    check(!backend_request_is_auto("cpu"), "cpu is an explicit request");
    check(backend_request_matches("cpu", CPU_BACKEND, CPU_BACKEND), "cpu must match the CPU registry");
    check(!backend_request_matches("cpu", OPENCL_BACKEND, OPENCL_DEVICE), "cpu must not match OpenCL");
    check(backend_request_matches("opencl", OPENCL_BACKEND, OPENCL_DEVICE), "opencl must match the OpenCL registry");
    check(backend_request_matches("webgpu", WEBGPU_BACKEND, WEBGPU_DEVICE), "webgpu must match the WebGPU registry");
    check(!backend_request_matches("webgpu", VULKAN_BACKEND, "Vulkan0"), "webgpu must not match Vulkan");
    check(!backend_request_matches("opencl", WEBGPU_BACKEND, WEBGPU_DEVICE), "opencl must not match WebGPU");
    check(backend_request_matches("hexagon", HEXAGON_BACKEND, HEXAGON_DEVICE), "hexagon must match HTP0");
    check(!backend_request_matches("hexagon", HEXAGON_BACKEND, SECOND_HEXAGON_DEVICE),
          "hexagon must select only the primary NPU");
    check(!backend_request_matches("hexagon", OPENCL_BACKEND, OPENCL_DEVICE), "hexagon must not match OpenCL");
    check(backend_request_matches(SECOND_HEXAGON_DEVICE, HEXAGON_BACKEND, SECOND_HEXAGON_DEVICE),
          "an exact device name must match that device");
    check(!backend_request_matches("auto", CPU_BACKEND, CPU_BACKEND), "auto must not match a device");
    check(!backend_request_matches("hexagon", nullptr, nullptr), "a request must not match an unnamed device");
}

void check_dsp_library_path() {
    using tts_cpp::detail::dsp_library_path_with;

    check(dsp_library_path_with("/apk/lib", nullptr) == "/apk/lib", "an unset path must become the directory");
    check(dsp_library_path_with("/apk/lib", "") == "/apk/lib", "an empty path must become the directory");
    check(dsp_library_path_with("/apk/lib", "/vendor/dsp") == "/apk/lib;/vendor/dsp",
          "the directory must be searched before the existing entries");
    check(dsp_library_path_with("/apk/lib", "/vendor/dsp;/apk/lib") == "/vendor/dsp;/apk/lib",
          "a directory already on the path must not be added twice");
}

void check_requested_init() {
    using tts_cpp::detail::init_requested_backend;
    using tts_cpp::detail::weight_buffer_type;

    ggml_backend_t cpu = init_requested_backend("cpu", /*verbose=*/false, "test");
    check(cpu != nullptr, "an explicit cpu request must initialise the CPU backend");
    if (cpu) {
        check(weight_buffer_type(cpu) == ggml_backend_get_default_buffer_type(cpu),
              "CPU weights must use the default buffer type");
        check(tts_cpp::detail::backend_shares_host_memory(cpu), "the CPU backend allocates from host memory");
        size_t free_bytes = 0, total_bytes = 0;
        tts_cpp::detail::backend_memory(cpu, free_bytes, total_bytes);
        check(total_bytes > 0, "the CPU backend must report host memory");
        ggml_backend_free(cpu);
    }
    check(init_requested_backend("no-such-device", /*verbose=*/false, "test") == nullptr,
          "a request for a missing device must return no backend");
}

int64_t first_unrunnable_of(const std::vector<ggml_type> & types) {
    ggml_init_params params = { ggml_tensor_overhead() * types.size(), nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    gguf_context * g = gguf_init_empty();
    for (size_t i = 0; i < types.size(); ++i) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, types[i], 32);
        ggml_set_name(t, ("t" + std::to_string(i)).c_str());
        gguf_add_tensor(g, t);
    }
    const int64_t bad = tts_cpp::detail::first_tensor_hexagon_cannot_run(g);
    gguf_free(g);
    ggml_free(ctx);
    return bad;
}

void check_hexagon_weight_types() {
    using tts_cpp::detail::hexagon_runs_weight_type;
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_I32 }) {
        check(hexagon_runs_weight_type(type), "Hexagon must accept its validated weight types");
    }
    for (ggml_type type : { GGML_TYPE_BF16, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1 }) {
        check(!hexagon_runs_weight_type(type), "Hexagon must refuse weight types it was not validated on");
    }
    check(first_unrunnable_of({ GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_I32 }) == -1,
          "a GGUF of runnable types must pass the Hexagon scan");
    check(first_unrunnable_of({ GGML_TYPE_F16, GGML_TYPE_Q5_0, GGML_TYPE_BF16 }) == 1,
          "the Hexagon scan must name the first unrunnable tensor");
}

}

int main() {
    check_backend_requests();
    check_dsp_library_path();
    check_requested_init();
    check_hexagon_weight_types();

    using tts_cpp::detail::GpuBackendRequirement;
    using tts_cpp::detail::gpu_backend_satisfies_requirement;

    check(gpu_backend_satisfies_requirement(nullptr, GpuBackendRequirement::Any),
          "unrestricted selection must accept an unnamed backend");
    check(gpu_backend_satisfies_requirement(CUDA_BACKEND, GpuBackendRequirement::Any),
          "unrestricted selection must accept CUDA");
    check(gpu_backend_satisfies_requirement(VULKAN_BACKEND,
                                            GpuBackendRequirement::Vulkan),
          "Vulkan selection must accept Vulkan");
    check(!gpu_backend_satisfies_requirement(CUDA_BACKEND,
                                             GpuBackendRequirement::Vulkan),
          "Vulkan selection must reject CUDA");
    check(!gpu_backend_satisfies_requirement(nullptr, GpuBackendRequirement::Vulkan),
          "Vulkan selection must reject an unnamed backend");
    const auto vkmtlcl = GpuBackendRequirement::Metal | GpuBackendRequirement::OpenCL |
                         GpuBackendRequirement::Vulkan;
    check(gpu_backend_satisfies_requirement(VULKAN_BACKEND, vkmtlcl),
          "Metal|OpenCL|Vulkan selection must accept Vulkan");
    check(!gpu_backend_satisfies_requirement(CUDA_BACKEND, vkmtlcl),
          "Metal|OpenCL|Vulkan selection must reject CUDA");

    check(gpu_backend_satisfies_requirement(WEBGPU_BACKEND, GpuBackendRequirement::WebGPU),
          "WebGPU selection must accept WebGPU");
    check(!gpu_backend_satisfies_requirement(WEBGPU_BACKEND, vkmtlcl),
          "Metal|OpenCL|Vulkan selection must reject WebGPU");

    check(selection_rejects("bogus"),
          "an unknown TTS_CPP_GPU_BACKEND must be rejected, not silently ignored");
    check(!selection_rejects(nullptr),
          "an unset TTS_CPP_GPU_BACKEND must leave selection alone");
    return failures == 0 ? 0 : 1;
}
