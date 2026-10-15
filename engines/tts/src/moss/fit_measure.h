#pragma once

#include "tts-cpp/fit.h"
#include "backend_selection.h"
#include "backend_util.h"
#include "fit_price.h"
#include "fit_util.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <stdexcept>

namespace tts_cpp::moss::detail {

inline void set_fit_tensor_data(ggml_context * ctx, void * data) {
    for (auto * tensor = ggml_get_first_tensor(ctx); tensor;
         tensor = ggml_get_next_tensor(ctx, tensor)) {
        tensor->data = data;
    }
}

inline void mark_fit_tensors(ggml_context * ctx) {
    set_fit_tensor_data(ctx, reinterpret_cast<void *>(uintptr_t(1)));
}

inline uint64_t measure_fit_tensors(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    set_fit_tensor_data(ctx, nullptr);
    const uint64_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
    mark_fit_tensors(ctx);
    return bytes;
}

inline uint64_t measure_fit_tensors(ggml_context * ctx, ggml_backend_t backend) {
    return measure_fit_tensors(ctx, ggml_backend_get_default_buffer_type(backend));
}

inline FitResult fit_device(ggml_backend_t backend) {
    FitResult result;
    const auto device = ggml_backend_get_device(backend);
    if (!device) throw std::runtime_error("no compute backend available");
    result.device_name = ggml_backend_name(backend);
    result.device_is_cpu = ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
    result.device_shares_host_memory = ::tts_cpp::detail::backend_shares_host_memory(backend);
    size_t free = 0, total = 0;
    ::tts_cpp::detail::backend_memory(backend, free, total);
    result.device_free_bytes = free;
    result.device_total_bytes = total;
    return result;
}

inline ::tts_cpp::detail::fit_graph_price measure_fit_graph(
        ggml_backend_t backend, ggml_cgraph * graph, size_t nodes) {
    ::tts_cpp::detail::fit_graph_price price;
    if (!::tts_cpp::detail::fit_price_graph(backend, graph, nodes, price)) {
        throw std::runtime_error("graph measurement failed");
    }
    return price;
}

}
