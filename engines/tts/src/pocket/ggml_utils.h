#pragma once

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "pocket/memory.h"
#include "pocket/backend.h"
#include "gguf_stream.h"
#include "fit_util.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::pocket::detail {

inline void require_finite(const std::vector<float> & x, const std::string & what) {
    for (float value : x) if (!std::isfinite(value)) throw std::runtime_error("pocket: non-finite " + what);
}

// One source tensor at a time, never a second copy of the complete checkpoint.
// Validate payloads before exposing them to graph or host-side arithmetic.
inline void load_float_weight(::tts_cpp::detail::gguf_stream_reader & reader,
                              ggml_tensor * source, ggml_tensor * destination) {
    const auto n = static_cast<size_t>(ggml_nelements(source));
    std::vector<float> values(n);
    if (source->type == GGML_TYPE_F32) {
        if (!reader.to_host(ggml_get_name(source), values.data(), n * sizeof(float)))
            throw std::runtime_error("pocket: weight read failed");
    } else if (source->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(n);
        if (!reader.to_host(ggml_get_name(source), half.data(), n * sizeof(ggml_fp16_t)))
            throw std::runtime_error("pocket: half weight read failed");
        ggml_fp16_to_fp32_row(half.data(), values.data(), static_cast<int64_t>(n));
    } else {
        throw std::runtime_error("pocket: unsupported weight type");
    }
    require_finite(values, ggml_get_name(source));
    ggml_backend_tensor_set(destination, values.data(), 0, n * sizeof(float));
}

// A non-null data marker makes ggml's size-only allocator treat persistent
// tensors as externally allocated. It must never be dereferenced or executed.
inline void mark_external(ggml_context * ctx) {
    for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
        t->data = reinterpret_cast<void *>(uintptr_t(0x1000));
}
inline uint64_t price_cpu_graph(ggml_backend_t backend, ggml_gallocr_t allocator, ggml_cgraph * graph, int threads) {
    size_t arena = 0;
    ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &arena);
    // CPU variants can be MODULE libraries: resolve the planner on the same
    // backend that will execute the graph, without a direct CPU-library import.
    const auto device = ggml_backend_get_device(backend);
    const auto reg = ggml_backend_dev_backend_reg(device);
    const auto plan_graph = reinterpret_cast<decltype(&ggml_graph_plan)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_graph_plan"));
    if (!plan_graph) throw std::runtime_error("pocket: CPU backend lacks ggml_graph_plan; update ggml-speech");
    const auto plan = plan_graph(graph, threads, nullptr);
    return uint64_t(arena) + plan.work_size;
}

inline uint64_t hexagon_buffer_overhead(const HexagonRuntime & runtime) {
    return ::tts_cpp::fitutil::sat_add(runtime.tensor_guard_bytes, runtime.allocation_page_bytes - 1);
}

inline void validate_hexagon_tensor(const ggml_tensor * tensor, const HexagonRuntime & runtime) {
    if (!tensor || tensor->view_src) return;
    if (uint64_t(ggml_nbytes(tensor)) > runtime.max_buffer_bytes)
        throw std::runtime_error(std::string("pocket: Hexagon tensor exceeds the selected session's RPC buffer limit: ") +
            ggml_get_name(tensor));
}

inline uint64_t price_hexagon_context(ggml_backend_t backend, ggml_context * context) {
    if (!::tts_cpp::detail::backend_is_hexagon(backend)) return 0;
    const auto runtime = hexagon_runtime(backend);
    uint64_t allocations = 0;
    for (auto * tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor)) {
        validate_hexagon_tensor(tensor, runtime);
        if (!tensor->view_src) ++allocations;
    }
    // A context allocator can split at tensor boundaries. One RPC allocation
    // per tensor bounds every split, guard page and page-rounding fragment.
    return ::tts_cpp::fitutil::sat_mul(allocations, hexagon_buffer_overhead(runtime));
}

inline void validate_graph(ggml_backend_t backend, ggml_cgraph * graph, const HexagonRuntime * caps = nullptr) {
    if (::tts_cpp::detail::backend_is_cpu(backend)) return;
    const bool is_hexagon = ::tts_cpp::detail::backend_is_hexagon(backend);
    const auto runtime = is_hexagon ? (caps ? *caps : hexagon_runtime(backend)) : HexagonRuntime{};
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (is_hexagon) {
            validate_hexagon_tensor(node, runtime);
            for (const auto * source : node->src) validate_hexagon_tensor(source, runtime);
        }
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
        if (!ggml_backend_supports_op(backend, node))
            throw std::runtime_error(std::string("pocket: ") + ggml_backend_name(backend) +
                " does not support graph operation " + ggml_op_desc(node) + "; update ggml-speech");
    }
}

inline uint64_t price_graph(ggml_backend_t backend, ggml_gallocr_t allocator, ggml_cgraph * graph, int threads, const HexagonRuntime * caps = nullptr) {
    if (::tts_cpp::detail::backend_is_cpu(backend)) return price_cpu_graph(backend, allocator, graph, threads);
    size_t arena = 0;
    ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &arena);
    if (::tts_cpp::detail::backend_is_hexagon(backend)) {
        const auto runtime = caps ? *caps : hexagon_runtime(backend);
        return ::tts_cpp::fitutil::sat_add(arena, ::tts_cpp::fitutil::sat_mul(
            hexagon_graph_buffer_chunks, hexagon_buffer_overhead(runtime)));
    }
    return arena;
}

inline void measure_opencl_scratch(ggml_cgraph * graph, OpenCLScratch & retained) {
    using ::tts_cpp::fitutil::sat_add;
    using ::tts_cpp::fitutil::sat_mul;
    OpenCLScratch current;
    // Baseline activation pool in the speech OpenCL backend with Adreno kernels.
    current.activations = opencl_init_activation_bytes;
    auto grow = [&](uint64_t & capacity, uint64_t bytes) {
        if (bytes > capacity) { current.growth = sat_add(current.growth, capacity); capacity = bytes; }
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_MUL_MAT) continue;
        const auto * a = node->src[0], * b = node->src[1];
        if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32)
            throw std::runtime_error("pocket: OpenCL scratch pricing requires F32 matmul");
        if (!ggml_is_contiguous(a)) grow(current.src0, ggml_nbytes(a));
        if (!ggml_is_contiguous(b)) grow(current.src1, ggml_nbytes(b));
        // Bound the Adreno image GEMM independently of device image limits
        // and build flags (falling back to generic GEMM needs no larger pools).
        const uint64_t columns = sat_mul(uint64_t(b->ne[1]), sat_mul(b->ne[2], b->ne[3]));
        if (ggml_is_contiguous(a) && ggml_is_contiguous(b) && a->ne[0] % 32 == 0 &&
            a->ne[1] % 4 == 0 && columns > 1) {
            grow(current.weights, sat_mul(sat_mul(a->ne[0], a->ne[1]), sizeof(float)));
            grow(current.activations, sat_mul(sat_mul(a->ne[0], sat_add(columns, 7) / 8 * 8), sizeof(float)));
        }
    }
    retained.weights = std::max(retained.weights, current.weights);
    retained.activations = std::max(retained.activations, current.activations);
    retained.src0 = std::max(retained.src0, current.src0);
    retained.src1 = std::max(retained.src1, current.src1);
    retained.growth = std::max(retained.growth, current.growth);
}

inline uint64_t opencl_scratch_bound(const OpenCLScratch & a, const OpenCLScratch & b) {
    using ::tts_cpp::fitutil::sat_add;
    using ::tts_cpp::fitutil::sat_mul;
    // Known speech backend init buffers: quant, scales, activation transpose.
    // Each is capped by CL_DEVICE_MAX_MEM_ALLOC_SIZE; the uncapped sum is safe
    // for builds without Adreno kernels too. These are shared by both stages.
    uint64_t bytes = opencl_init_bytes;
    for (const auto pool : {std::max(a.weights, b.weights), std::max(a.src0, b.src0), std::max(a.src1, b.src1)})
        bytes = sat_add(bytes, sat_mul(pool, 2));
    const auto activation = std::max(a.activations, b.activations);
    if (activation > opencl_init_activation_bytes) bytes = sat_add(bytes, sat_mul(activation, 2));
    // One initial resident generation plus a graph's increasing generations
    // may coexist until its queued work finishes. The extra maximum above
    // bounds the initial generation regardless of which worker ran previously.
    return sat_add(bytes, std::max(a.growth, b.growth));
}
inline void price_persistent(ggml_backend_t backend, ggml_context * metadata,
                             ggml_context * weights, ggml_context * state,
                             gguf_context * file, MemoryMeasure & out) {
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    out.backend_name = ggml_backend_name(backend);
    out.shares_host_memory = ::tts_cpp::detail::backend_shares_host_memory(backend);
    size_t available = 0, total = 0;
    ::tts_cpp::detail::backend_memory(backend, available, total);
    out.device_free = available; out.device_total = total;
    out.weights = ::tts_cpp::fitutil::sat_add(ggml_backend_alloc_ctx_tensors_from_buft_size(weights, buft),
        price_hexagon_context(backend, weights));
    out.state = ::tts_cpp::fitutil::sat_add(ggml_backend_alloc_ctx_tensors_from_buft_size(state, buft),
        price_hexagon_context(backend, state));
    out.metadata = ggml_get_mem_size(metadata) + ggml_get_mem_size(weights) +
                   ggml_get_mem_size(state) + gguf_get_meta_size(file);
    for (auto * t = ggml_get_first_tensor(metadata); t; t = ggml_get_next_tensor(metadata, t)) {
        const auto n = uint64_t(ggml_nelements(t));
        out.load_staging = std::max(out.load_staging, n * (t->type == GGML_TYPE_F16 ? 6 : 4));
    }
    mark_external(weights); mark_external(state);
}

template<int Nodes>
struct SizedPocketGraph {
    ggml_backend_t backend_;
    std::unique_lock<std::recursive_mutex> session_lock_;
    HexagonRuntime hexagon_caps_;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::vector<ggml_tensor *> updates;
    explicit SizedPocketGraph(ggml_backend_t backend, const HexagonRuntime * caps = nullptr) : backend_(backend),
        session_lock_(accelerator_mutex(), std::defer_lock) {
        // HTP uploads, RPC buffer mapping/unmapping, and cached graph metadata
        // all share the session. Hold the gate through graph destruction.
        if (::tts_cpp::detail::backend_is_hexagon(backend)) {
            session_lock_.lock();
            hexagon_caps_ = caps ? *caps : hexagon_runtime(backend);
        }
        ctx = ggml_init({metadata_bytes(), nullptr, true});
        if (!ctx) throw std::runtime_error("pocket: graph context allocation failed");
        graph = ggml_new_graph_custom(ctx, Nodes, false);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!allocator) { ggml_free(ctx); ctx = nullptr; throw std::runtime_error("pocket: graph allocator failed"); }
    }
    ~SizedPocketGraph() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
    SizedPocketGraph(const SizedPocketGraph &) = delete;
    SizedPocketGraph & operator=(const SizedPocketGraph &) = delete;
    static size_t metadata_bytes() {
        return Nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(Nodes, false);
    }
    uint64_t prepare(ggml_tensor * output, bool size_only = false, int threads = 1,
                     OpenCLScratch * scratch = nullptr) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        // The old convolution / transformer histories must remain intact until
        // every consumer has run. Delay cache writes until after the output.
        for (auto * update : updates) ggml_build_forward_expand(graph, update);
        validate_graph(backend_, graph, &hexagon_caps_);
        if (scratch && ::tts_cpp::detail::backend_is_opencl(backend_)) measure_opencl_scratch(graph, *scratch);
        if (size_only) return price_graph(backend_, allocator, graph, threads, &hexagon_caps_);
        if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("pocket: graph allocation failed");
        return 0;
    }
    std::vector<float> compute(ggml_backend_t backend, ggml_tensor * output) {
        auto lock = lock_backend(backend);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("pocket: graph execution failed");
        std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        require_finite(values, "graph output");
        return values;
    }
};

// Keep each model's existing arena budget, sharing allocation and execution.
using PocketGraph = SizedPocketGraph<32768>;
using PocketFlowGraph = SizedPocketGraph<16384>;

inline ggml_tensor * pocket_mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x) {
    auto * result = ggml_mul_mat(c, w, x);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

inline ggml_tensor * pocket_mean(ggml_context * c, ggml_tensor * x, ggml_backend_t backend) {
    // HTP implements SUM_ROWS but not MEAN. Preserve F32 reduction arithmetic.
    return ::tts_cpp::detail::backend_is_hexagon(backend) ?
        ggml_scale(c, ggml_sum_rows(c, x), 1.0f / float(x->ne[0])) : ggml_mean(c, x);
}

inline ggml_tensor * pocket_gelu(ggml_context * c, ggml_tensor * x) {
    // Avoid ggml_gelu's CPU FP16 lookup: preserve PyTorch approximate="tanh"
    // arithmetic in F32 for both FlowLM and Mimi.
    auto * cubic = ggml_mul(c, ggml_sqr(c, x), x);
    auto * inner = ggml_scale(c, ggml_add(c, x, ggml_scale(c, cubic, 0.044715f)), 0.7978845608028654f);
    return ggml_mul(c, ggml_scale(c, x, 0.5f), ggml_scale_bias(c, ggml_tanh(c, inner), 1, 1));
}

inline std::vector<float> pocket_read(ggml_tensor * t, size_t offset = 0, size_t count = 0) {
    const auto elements = static_cast<size_t>(ggml_nelements(t));
    if (offset > elements) throw std::runtime_error("pocket: tensor read offset exceeds size");
    if (!count) count = elements - offset;
    if (count > elements - offset) throw std::runtime_error("pocket: tensor read exceeds size");
    std::vector<float> result(count);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, result.data(), offset * sizeof(float), count * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(count);
        ggml_backend_tensor_get(t, half.data(), offset * sizeof(ggml_fp16_t), count * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(half.data(), result.data(), static_cast<int64_t>(count));
    } else {
        throw std::runtime_error("pocket: unsupported tensor read type");
    }
    return result;
}
} // namespace tts_cpp::pocket::detail
