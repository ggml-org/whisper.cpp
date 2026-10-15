#pragma once

#include "backend_selection.h"
#include "backend_util.h"
#include <algorithm>
#include <cctype>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tts_cpp::pocket::detail {
inline std::mutex & opencl_graph_mutex() {
    // ggml-opencl handles for a device share mutable kernel arguments,
    // scratch and lifecycle state. This gate covers Pocket workers/instances.
    static std::mutex mutex;
    return mutex;
}

inline std::unique_lock<std::mutex> lock_backend(ggml_backend_t backend) {
    std::unique_lock<std::mutex> lock(opencl_graph_mutex(), std::defer_lock);
    if (::tts_cpp::detail::backend_is_opencl(backend)) lock.lock();
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
    if (requested.empty() || requested == "auto" || requested == "cpu" || requested == "opencl") return requested;
    throw std::invalid_argument("Pocket TTS: backend must be cpu, opencl or auto");
}

struct BackendUnavailable : std::runtime_error { using std::runtime_error::runtime_error; };

inline Backend select_backend(bool use_gpu, const std::string & requested) {
    const auto choice = backend_request(requested);
    std::lock_guard<std::mutex> lock(opencl_graph_mutex());
    ggml_backend_t backend = nullptr;
    if (choice == "opencl") backend = ::tts_cpp::detail::init_requested_backend("opencl", false, "pocket");
    else if (choice == "cpu" || !use_gpu) backend = ::tts_cpp::detail::init_cpu_backend();
    else backend = ::tts_cpp::detail::init_gpu_backend(1, false, "pocket", 0, false, nullptr,
                         ::tts_cpp::detail::GpuBackendRequirement::OpenCL);
    const bool wants_opencl = choice == "opencl" || (choice != "cpu" && use_gpu);
    if (backend && wants_opencl && !::tts_cpp::detail::backend_is_opencl(backend)) {
        ggml_backend_free(backend); backend = nullptr;
    }
    if (!backend) throw BackendUnavailable(choice == "cpu" || (choice != "opencl" && !use_gpu)
        ? "Pocket TTS: CPU backend unavailable"
        : "Pocket TTS: requested OpenCL backend unavailable; install a supported ggml-opencl backend");
    return Backend(backend);
}

inline ggml_backend_t init_stage_backend(ggml_backend_dev_t device) {
    std::lock_guard<std::mutex> lock(opencl_graph_mutex());
    auto * backend = device ? ggml_backend_dev_init(device, nullptr) : ::tts_cpp::detail::init_cpu_backend();
    if (!backend) throw BackendUnavailable("Pocket TTS: selected stage backend could not initialise");
    if (!::tts_cpp::detail::backend_is_cpu(backend) && !::tts_cpp::detail::backend_is_opencl(backend)) {
        ggml_backend_free(backend);
        throw std::invalid_argument("Pocket TTS: only CPU and OpenCL stage backends are supported");
    }
    return backend;
}

inline bool shares_host_memory(ggml_backend_t backend) {
    if (::tts_cpp::detail::backend_shares_host_memory(backend)) return true;
    // ggml-opencl reports Adreno as GPU, including on shared-memory phones.
    const auto device = ggml_backend_get_device(backend);
    return ::tts_cpp::detail::backend_is_opencl(backend) && device &&
        ::tts_cpp::detail::is_qualcomm_adreno(ggml_backend_dev_name(device), ggml_backend_dev_description(device));
}

} // namespace tts_cpp::pocket::detail
