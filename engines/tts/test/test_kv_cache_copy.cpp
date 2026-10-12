#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "kv_cache_copy.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

constexpr int width = 8;
constexpr int batches = 2;
constexpr int capacity = 6;
constexpr int past = 1;
constexpr int projection_slots = 3;
constexpr float untouched = -1.0f;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool supports_contiguous_copy(ggml_backend_t backend, const ggml_tensor *op) {
    return (op->op != GGML_OP_CPY || ggml_is_contiguous(op->src[0])) && ggml_backend_supports_op(backend, op);
}

struct copy_fixture {
    ggml_context *cache_context;
    ggml_context *graph_context;
    ggml_backend_buffer_t cache_buffer;
    ggml_gallocr_t allocator;

    explicit copy_fixture(ggml_backend_t cpu) {
        cache_context = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
        graph_context = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true});
        cache_buffer = nullptr;
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    }

    ~copy_fixture() {
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(cache_buffer);
        ggml_free(graph_context);
        ggml_free(cache_context);
    }
};

std::vector<float> make_projection_values(size_t size) {
    std::vector<float> values(size);
    for (size_t i = 0; i < size; ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    return values;
}

void build_batch_copies(copy_fixture &fixture, ggml_backend_t cpu, ggml_cgraph *graph, ggml_tensor *projection,
                        ggml_tensor *cache, int tokens, int slot, bool reject_strided_copy) {
    const size_t cache_row = ggml_row_size(cache->type, width);
    for (int batch = 0; batch < batches; ++batch) {
        ggml_tensor *source = ggml_view_2d(fixture.graph_context, projection, width, tokens, projection->nb[1],
                                           batch * projection->nb[2] + slot * width * sizeof(float));
        ggml_tensor *destination =
            ggml_view_2d(fixture.graph_context, cache, width, tokens, cache_row, (batch * capacity + past) * cache_row);
        auto supports_op = reject_strided_copy ? supports_contiguous_copy : ggml_backend_supports_op;
        ggml_tensor *copy =
            tts_cpp::detail::build_kv_cache_copy(fixture.graph_context, cpu, source, destination, supports_op);
        const bool needs_contiguous = reject_strided_copy && !ggml_is_contiguous(source);
        require((copy->src[0]->op == GGML_OP_CONT) == needs_contiguous, "Unexpected source materialization");
        require(needs_contiguous || copy->src[0] == source, "Supported source was replaced");
        require(copy->view_src == cache, "Copy lost the persistent cache destination");
        require(supports_op(cpu, copy), "Cache copy is still unsupported");
        ggml_build_forward_expand(graph, copy);
    }
}

void verify_cache(const std::vector<float> &output, const std::vector<float> &input, int tokens, int slot) {
    for (int batch = 0; batch < batches; ++batch) {
        for (int token = 0; token < capacity; ++token) {
            for (int column = 0; column < width; ++column) {
                const bool written = token >= past && token < past + tokens;
                const size_t source_index =
                    ((batch * tokens + token - past) * projection_slots + slot) * width + column;
                const float expected = written ? input[source_index] : untouched;
                require(output[(batch * capacity + token) * width + column] == expected,
                        "Cache data or neighboring positions changed");
            }
        }
    }
}

void test_cache_copy(ggml_backend_t cpu, ggml_type cache_type, int tokens, int slot, bool reject_strided_copy) {
    copy_fixture fixture(cpu);
    ggml_tensor *cache = ggml_new_tensor_3d(fixture.cache_context, cache_type, width, capacity, batches);
    fixture.cache_buffer = ggml_backend_alloc_ctx_tensors(fixture.cache_context, cpu);
    require(fixture.cache_buffer != nullptr, "Cache allocation failed");
    ggml_tensor *projection =
        ggml_new_tensor_3d(fixture.graph_context, GGML_TYPE_F32, width * projection_slots, tokens, batches);
    ggml_set_input(projection);
    ggml_cgraph *graph = ggml_new_graph(fixture.graph_context);
    build_batch_copies(fixture, cpu, graph, projection, cache, tokens, slot, reject_strided_copy);
    require(ggml_gallocr_alloc_graph(fixture.allocator, graph), "Graph allocation failed");
    const size_t cache_elements = width * capacity * batches;
    std::vector<float> output(cache_elements, untouched);
    std::vector<ggml_fp16_t> half(cache_elements, ggml_fp32_to_fp16(untouched));
    if (cache_type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(cache, half.data(), 0, ggml_nbytes(cache));
    } else {
        ggml_backend_tensor_set(cache, output.data(), 0, ggml_nbytes(cache));
    }
    const auto input = make_projection_values(ggml_nelements(projection));
    ggml_backend_tensor_set(projection, input.data(), 0, ggml_nbytes(projection));
    require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "Cache graph compute failed");
    if (cache_type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(cache, half.data(), 0, ggml_nbytes(cache));
        ggml_fp16_to_fp32_row(half.data(), output.data(), cache_elements);
    } else {
        ggml_backend_tensor_get(cache, output.data(), 0, ggml_nbytes(cache));
    }
    verify_cache(output, input, tokens, slot);
}

void run_copy_cases(ggml_backend_t cpu) {
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int tokens : {1, 3}) {
            for (int slot : {1, 2}) {
                for (bool reject_strided_copy : {false, true}) {
                    test_cache_copy(cpu, type, tokens, slot, reject_strided_copy);
                }
            }
        }
    }
}

}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    try {
        require(cpu != nullptr, "CPU backend initialization failed");
        run_copy_cases(cpu);
        ggml_backend_free(cpu);
        std::puts("PASS: KV cache copy values, batch offsets, dtypes and backend "
                  "support");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ggml_backend_free(cpu);
        return 1;
    }
}
