#pragma once

#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <algorithm>
#include <vector>

// A rejected graph has no usable projection. In particular, scheduler errors
// for unsupported operations in preallocated buffers must not become a small
// partial compute budget and a false FITS verdict.
inline bool whisper_fit_measure_graph(ggml_backend_sched_t sched, ggml_cgraph * graph,
        const std::vector<ggml_backend_t> & backends, std::vector<size_t> & sizes) {
    sizes.assign(backends.size(), 0);
    if (!ggml_backend_sched_reserve_size(sched, graph, sizes.data())) {
        std::fill(sizes.begin(), sizes.end(), 0);
        return false;
    }
    // Repeated buffer types share one pool. Keep compatibility with producers
    // that report this pool per backend rather than only at its first index.
    for (size_t i = 1; i < backends.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (ggml_backend_get_default_buffer_type(backends[i]) ==
                ggml_backend_get_default_buffer_type(backends[j])) {
                sizes[i] = 0;
                break;
            }
        }
    }
    return true;
}

// A device may expose shared system RAM while having no device-specific
// capacity query (e.g. FastRPC). Keep its tensor accounting on that device,
// but compare the combined host/device requirement with the host RAM pool.
struct whisper_fit_memory_info {
    bool shares_host = false;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
};

// Android's OpenCL GPUs (Adreno, Mali) draw their buffers from host RAM but
// do not report unified memory.
inline bool whisper_fit_is_android_opencl(ggml_backend_dev_t dev) {
#if defined(__ANDROID__)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name != nullptr && std::strcmp(name, "OpenCL") == 0;
#else
    (void) dev;
    return false;
#endif
}

inline whisper_fit_memory_info whisper_fit_query_memory(ggml_backend_dev_t dev) {
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    whisper_fit_memory_info info;
    info.shares_host = props.type == GGML_BACKEND_DEVICE_TYPE_CPU ||
                       props.type == GGML_BACKEND_DEVICE_TYPE_IGPU || props.memory_unified ||
                       whisper_fit_is_android_opencl(dev);
    info.free_bytes = props.memory_free;
    info.total_bytes = props.memory_total;
    if (info.shares_host && info.total_bytes == 0) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr && cpu != dev) {
            ggml_backend_dev_memory(cpu, &info.free_bytes, &info.total_bytes);
        }
    }
    return info;
}

// Price only prospective registry metadata growth: the OS's available-RAM
// snapshot already reflects retained metadata and shared queues. Tensor
// buffers are priced separately by the graph/context allocators. Opaque
// future driver/DSP heaps are still covered only by the margin.
// Call once for the primary backend; the Hexagon query already covers all
// sessions, including sessions used by a VAD context.
inline bool whisper_fit_query_backend_overhead(ggml_backend_t backend, uint64_t max_graph_nodes, uint64_t & bytes) {
    bytes = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto query = reg ? reinterpret_cast<ggml_backend_hexagon_get_runtime_info_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_runtime_info_v1")) : nullptr;
    if (query == nullptr) {
        return true;
    }
    ggml_backend_hexagon_runtime_info_v1 retained = {}, future = {};
    if (!query(backend, 0, &retained, sizeof(retained)) ||
        !query(backend, max_graph_nodes, &future, sizeof(future)) ||
        retained.version != GGML_HEXAGON_RUNTIME_INFO_VERSION ||
        future.version != GGML_HEXAGON_RUNTIME_INFO_VERSION ||
        retained.session_count == 0 || retained.session_count != future.session_count ||
        future.host_bytes_bound < retained.host_bytes_bound || future.shared_bytes < retained.shared_bytes) {
        return false;
    }
    const uint64_t host = future.host_bytes_bound - retained.host_bytes_bound;
    const uint64_t shared = future.shared_bytes - retained.shared_bytes;
    bytes = shared > std::numeric_limits<uint64_t>::max() - host ?
        std::numeric_limits<uint64_t>::max() : host + shared;
    return true;
}
