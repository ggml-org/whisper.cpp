#pragma once

#include "backend_selection.h"
#include "backend_util.h"
#include "pocket/memory.h"
#if __has_include("ggml-hexagon.h")
#include "ggml-hexagon.h"
#endif
#include <algorithm>
#include <cctype>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tts_cpp::pocket::detail {
inline std::recursive_mutex & accelerator_mutex() {
    // OpenCL kernel state and Hexagon queue/session state are shared across
    // backend handles. Recursive locking also covers nested Pocket graphs.
    static std::recursive_mutex mutex;
    return mutex;
}

inline std::unique_lock<std::recursive_mutex> lock_backend(ggml_backend_t backend) {
    std::unique_lock<std::recursive_mutex> lock(accelerator_mutex(), std::defer_lock);
    if (::tts_cpp::detail::backend_is_opencl(backend) || ::tts_cpp::detail::backend_is_hexagon(backend)) lock.lock();
    return lock;
}

inline void free_backend(ggml_backend_t backend) {
    auto lock = lock_backend(backend);
    ggml_backend_free(backend);
}
struct BackendDeleter { void operator()(ggml_backend_t backend) const { free_backend(backend); } };
using Backend = std::unique_ptr<ggml_backend, BackendDeleter>;

inline std::string backend_request(std::string requested) {
    std::transform(requested.begin(), requested.end(), requested.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (requested == "htp0") return "hexagon";
    if (requested.empty() || requested == "auto" || requested == "cpu" || requested == "opencl" ||
        requested == "webgpu" || requested == "hexagon") return requested;
    throw std::invalid_argument("Pocket TTS: backend must be cpu, opencl, webgpu, hexagon/HTP0 or auto");
}

struct BackendUnavailable : std::runtime_error { using std::runtime_error::runtime_error; };

inline void validate_hexagon_packing_limit(uint64_t packing_limit, uint64_t payload_limit) {
    // Both context allocation and graph arenas pack up to the buffer type's
    // advertised limit. Validating each tensor cannot bound a packed buffer.
    if (packing_limit > payload_limit)
        throw BackendUnavailable("Pocket TTS: Hexagon packed buffer limit (" + std::to_string(packing_limit) +
            ") exceeds the effective RPC payload cap (" + std::to_string(payload_limit) +
            "); lower GGML_HEXAGON_MBUF before backend initialization");
}

inline HexagonRuntime hexagon_runtime(ggml_backend_t backend) {
    auto lock = lock_backend(backend);
#if defined(GGML_HEXAGON_RUNTIME_INFO_VERSION) && GGML_HEXAGON_RUNTIME_INFO_VERSION >= 1
    const auto device = ggml_backend_get_device(backend);
    const auto registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    const auto query = registry ? reinterpret_cast<ggml_backend_hexagon_get_runtime_info_v1_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_hexagon_get_runtime_info_v1")) : nullptr;
    ggml_backend_hexagon_runtime_info_v1 info{};
    if (query && query(backend, 32768, &info, sizeof(info)) && info.version == 1) {
        if ((info.op_stage & 3) != 3)
            throw BackendUnavailable("Pocket TTS: Hexagon queue/compute disabled by backend configuration");
        if (info.host_bytes_bound && info.shared_bytes && info.max_buffer_bytes > info.tensor_guard_bytes &&
            info.max_batch_vmem_bytes && info.vtcm_bytes && info.session_count &&
            info.buffer_alignment && info.allocation_page_bytes &&
            !(info.allocation_page_bytes & (info.allocation_page_bytes - 1))) {
            validate_hexagon_packing_limit(
                ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend)), info.max_buffer_bytes);
            return {info.host_bytes_bound, info.shared_bytes, info.max_buffer_bytes, info.max_batch_vmem_bytes,
                info.vtcm_bytes, info.tensor_guard_bytes, info.buffer_alignment, info.allocation_page_bytes,
                info.session_count};
        }
    }
#endif
    throw BackendUnavailable("Pocket TTS: Hexagon runtime-info v1 unavailable or invalid; update ggml-speech");
}

inline bool supported_backend(ggml_backend_t backend) {
    namespace det = ::tts_cpp::detail;
    return det::backend_is_cpu(backend) || det::backend_is_opencl(backend) ||
           det::backend_is_metal(backend) || det::backend_is_cuda(backend) || det::backend_is_vulkan(backend) ||
           det::backend_is_webgpu(backend) || det::backend_is_hexagon(backend);
}

inline Backend select_backend(bool use_gpu, const std::string & requested) {
    const auto choice = backend_request(requested);
    std::lock_guard<std::recursive_mutex> lock(accelerator_mutex());
    ggml_backend_t backend = nullptr;
    const bool explicit_accelerator = choice == "opencl" || choice == "webgpu" || choice == "hexagon";
    if (explicit_accelerator) backend = ::tts_cpp::detail::init_requested_backend(choice, false, "pocket");
    else if (choice == "cpu" || !use_gpu) backend = ::tts_cpp::detail::init_cpu_backend();
    else backend = ::tts_cpp::detail::init_gpu_backend(1, false, "pocket", 0, false, nullptr,
                         ::tts_cpp::detail::GpuBackendRequirement::OpenCL |
                         ::tts_cpp::detail::GpuBackendRequirement::Metal |
                         ::tts_cpp::detail::GpuBackendRequirement::CUDA |
                         ::tts_cpp::detail::GpuBackendRequirement::Vulkan);
    const bool wants_opencl = choice == "opencl";
    const bool wants_webgpu = choice == "webgpu";
    if (backend && ((wants_opencl && !::tts_cpp::detail::backend_is_opencl(backend)) ||
                    (wants_webgpu && !::tts_cpp::detail::backend_is_webgpu(backend)) ||
                    (choice == "hexagon" && !::tts_cpp::detail::backend_is_hexagon(backend)))) {
        ggml_backend_free(backend); backend = nullptr;
    }
    if (!backend) {
        if (choice == "hexagon") throw BackendUnavailable("Pocket TTS: requested Hexagon HTP0 backend unavailable; install matching ggml-hexagon and DSP runtime libraries");
        if (wants_webgpu) throw BackendUnavailable("Pocket TTS: requested WebGPU backend unavailable; install a ggml-speech build with the WebGPU (Dawn) backend");
        throw BackendUnavailable(choice == "cpu" || (!wants_opencl && !use_gpu)
            ? "Pocket TTS: CPU backend unavailable"
            : wants_opencl ? "Pocket TTS: requested OpenCL backend unavailable; install a supported ggml-opencl backend"
                           : "Pocket TTS: Metal, CUDA, Vulkan or OpenCL backend unavailable");
    }
    Backend selected(backend);
    if (::tts_cpp::detail::backend_is_hexagon(backend)) (void)hexagon_runtime(backend);
    return selected;
}

inline ggml_backend_t init_stage_backend(ggml_backend_dev_t device) {
    std::lock_guard<std::recursive_mutex> lock(accelerator_mutex());
    auto * backend = device ? ggml_backend_dev_init(device, nullptr) : ::tts_cpp::detail::init_cpu_backend();
    if (!backend) throw BackendUnavailable("Pocket TTS: selected stage backend could not initialise");
    if (!supported_backend(backend)) {
        ggml_backend_free(backend);
        throw std::invalid_argument("Pocket TTS: unsupported stage backend");
    }
    Backend selected(backend);
    if (::tts_cpp::detail::backend_is_hexagon(backend)) (void)hexagon_runtime(backend);
    return selected.release();
}

inline bool shares_host_memory(ggml_backend_t backend) {
    if (::tts_cpp::detail::backend_shares_host_memory(backend)) return true;
    // ggml-opencl reports Adreno as GPU, including on shared-memory phones.
    const auto device = ggml_backend_get_device(backend);
    return ::tts_cpp::detail::backend_is_opencl(backend) && device &&
        ::tts_cpp::detail::is_qualcomm_adreno(ggml_backend_dev_name(device), ggml_backend_dev_description(device));
}

} // namespace tts_cpp::pocket::detail
