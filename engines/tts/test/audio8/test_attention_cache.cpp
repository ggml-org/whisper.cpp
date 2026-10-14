// Exercises production cache appends, masked prefill, single-token decoding,
// layer offsets, grouped query heads, and reuse after resetting the prefix.
#include "audio8/graph.h"
#include "backend_util.h"
#include "backend_selection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {
constexpr int HEAD = 64, HEADS = 14, KV_HEADS = 2, HIDDEN = HEAD * HEADS;
constexpr int CAPACITY = 160, LAYERS = 2;
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
float value_at(size_t index, int seed) {
    uint32_t value = uint32_t(index) ^ (uint32_t(seed + 1) * 0x9e3779b9u);
    value = (value ^ (value >> 16)) * 0x7feb352du;
    value = (value ^ (value >> 15)) * 0x846ca68bu;
    value ^= value >> 16;
    return float(int(value & 0xffffu) - 32768) / 32768.0f;
}
struct tensors {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~tensors() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
};
struct backend_owner {
    ggml_backend_t backend = nullptr;
    ~backend_owner() { if (backend) ggml_backend_free(backend); }
};
struct model {
    tensors weights, state;
    kv_cache cache;
    attention_weights blocks[LAYERS];
    ggml_tensor * cosine = nullptr, * sine = nullptr;
    ggml_backend_t backend;
    model(ggml_backend_t b, bool flash) : backend(b) {
        ggml_init_params p = {128 * ggml_tensor_overhead(), nullptr, true};
        weights.ctx = ggml_init(p); state.ctx = ggml_init(p);
        require(weights.ctx && state.ctx, "contexts");
        for (int l = 0; l < LAYERS; ++l) {
            blocks[l].wqkv = ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, HIDDEN, HEAD * (HEADS + 2 * KV_HEADS));
            blocks[l].wo = ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, HIDDEN, HIDDEN);
        }
        cosine = ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, HEAD / 2, CAPACITY);
        sine = ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, HEAD / 2, CAPACITY);
        weights.buffer = ggml_backend_alloc_ctx_tensors(weights.ctx, b);
        require(weights.buffer, "weight allocation");
        int ordinal = 0;
        for (auto t = ggml_get_first_tensor(weights.ctx); t; t = ggml_get_next_tensor(weights.ctx, t)) {
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); ++i) {
                if (t == cosine || t == sine) {
                    const double angle = (i / (HEAD / 2)) * std::pow(10000., -double(i % (HEAD / 2)) / (HEAD / 2));
                    data[i] = t == cosine ? std::cos(angle) : std::sin(angle);
                } else {
                    data[i] = 0.012f * value_at(i, ordinal);
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float)); ++ordinal;
        }
        cache.capacity = CAPACITY; cache.stride = HEAD * KV_HEADS; cache.flash_attention = flash;
        cache.k = ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, cache.stride, CAPACITY * LAYERS);
        cache.v = flash ? ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, cache.stride, CAPACITY * LAYERS)
                        : ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, CAPACITY, cache.stride * LAYERS);
        state.buffer = ggml_backend_alloc_ctx_tensors(state.ctx, b);
        require(state.buffer, "cache allocation");
        ggml_backend_buffer_clear(state.buffer, 0x7f);
    }
    std::vector<float> run(int width, int past, int seed) {
        scratch graph(512);
        require(graph.ok(), "graph context");
        auto input = input_f32(graph.ctx, "input", HIDDEN, width);
        auto x = input;
        auto rope = rope_window(graph.ctx, cosine, sine, past, width);
        ggml_tensor * mask = width > 1 ? input_f32(graph.ctx, "mask", past + width, width) : nullptr;
        for (int l = 0; l < LAYERS; ++l) {
            attention_shape s = {HEADS, KV_HEADS, HEAD, width, past, l, true};
            x = attention(graph.ctx, graph.graph, blocks[l], x, rope, cache, s, mask);
        }
        mark_output(graph.graph, x);
        int fa = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
            auto node = ggml_graph_node(graph.graph, i);
            require(ggml_backend_supports_op(backend, node), ggml_op_name(node->op));
            if (node->op == GGML_OP_FLASH_ATTN_EXT) ++fa;
        }
        require(fa == (cache.flash_attention && width == 1 ? LAYERS : 0), "attention routing");
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator, graph.graph), "graph allocation");
        std::vector<float> values(HIDDEN * width);
        for (size_t i = 0; i < values.size(); ++i) values[i] = value_at(i, seed);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        if (mask) {
            std::vector<float> m((past + width) * width);
            fill_causal_mask(m.data(), past + width, width, past, 0);
            ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
        }
        require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS, "graph compute");
        std::vector<float> output;
        read_output(x, output);
        ggml_gallocr_free(allocator);
        return output;
    }
};
void compare(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "output size");
    double error = 0, power = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite output");
        error += double(a[i] - b[i]) * (a[i] - b[i]); power += double(a[i]) * a[i];
    }
    const double nmse = error / std::max(power, 1e-20);
    require(nmse < 2e-4, "CPU attention NMSE");
    std::printf(" NMSE=%.9g", nmse);
}
}
int main(int argc, char ** argv) {
    try {
        ::tts_cpp::detail::ensure_backends_loaded();
        backend_owner cpu{ggml_backend_init_by_name("CPU", nullptr)};
        require(cpu.backend, "CPU backend");
        const char * name = argc > 1 ? argv[1] : "CPU";
        backend_owner candidate{ggml_backend_init_by_name(name, nullptr)};
        require(candidate.backend, "candidate backend");
        model reference(cpu.backend, false), changed(candidate.backend, true);
        int comparisons = 0;
        for (int pass = 0; pass < 2; ++pass) {
            int past = 0, seed = pass * 100;
            for (int width : {30, 1, 1, 2, 1, 60, 1, 32, 1, 1}) {
                std::printf("pass=%d past=%d width=%d", pass, past, width);
                compare(reference.run(width, past, seed), changed.run(width, past, seed));
                std::puts(" PASS"); ++comparisons; past += width; ++seed;
            }
        }
        std::printf("PASS: %d production attention cache comparisons versus CPU (%s)\n", comparisons, name);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
