// Real scheduler failure regression: model-free and CPU-only. This exercises
// the same preallocated/unsupported CPY failure class as non-host HTP KV copies.
#include "whisper-fit-memory.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    ggml_backend_load_all();
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu);
    auto * destination_ctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
    auto * graph_ctx = ggml_init({16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(16, false), nullptr, true});
    CHECK(destination_ctx && graph_ctx);

    // IQ1_S has no CPU from-float copy operation. Attaching its destination
    // buffer prevents the scheduler from moving this CPY to a fallback pool.
    auto * destination = ggml_new_tensor_1d(destination_ctx, GGML_TYPE_IQ1_S, 256);
    auto * destination_buffer = ggml_backend_alloc_ctx_tensors(destination_ctx, cpu);
    CHECK(destination_buffer);
    auto * input = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F32, 256);
    auto * copy = ggml_cpy(graph_ctx, input, destination);
    CHECK(!ggml_backend_supports_op(cpu, copy));
    auto * rejected = ggml_new_graph_custom(graph_ctx, 16, false);
    ggml_build_forward_expand(rejected, copy);
    auto * scheduler = ggml_backend_sched_new(&cpu, nullptr, 1, 16, false, false);
    CHECK(scheduler);
    const std::vector<ggml_backend_t> backends = {cpu};
    std::vector<size_t> sizes(1, 12345);
    CHECK(!whisper_fit_measure_graph(scheduler, rejected, backends, sizes));
    CHECK(sizes.size() == 1 && sizes[0] == 0);
    CHECK(ggml_backend_sched_get_buffer_size(scheduler, cpu) == 0);

    // A later supported graph must measure normally; the failed reservation
    // does not poison the scheduler or allocate payloads during sizing.
    auto * supported = ggml_new_graph_custom(graph_ctx, 16, false);
    ggml_build_forward_expand(supported, ggml_scale(graph_ctx, input, 2.0f));
    CHECK(whisper_fit_measure_graph(scheduler, supported, backends, sizes));
    CHECK(sizes[0] > 0 && ggml_backend_sched_get_buffer_size(scheduler, cpu) == 0);
    CHECK(ggml_backend_sched_reserve(scheduler, supported));
    CHECK(ggml_backend_sched_get_buffer_size(scheduler, cpu) == sizes[0]);

    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(destination_buffer);
    ggml_free(destination_ctx);
    ggml_free(graph_ctx);
    ggml_backend_free(cpu);
    std::puts("test-whisper-fit-graph-error: unsupported preallocated graph rejected; supported graph measures exactly");
    return 0;
}
