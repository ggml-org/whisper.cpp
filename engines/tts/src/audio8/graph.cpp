#include "audio8/graph.h"

#include "backend_util.h"

#include <cmath>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

const size_t FLOAT = sizeof(float);

size_t graph_arena(int nodes) {
    return static_cast<size_t>(nodes) * ggml_tensor_overhead() +
           ggml_graph_overhead_custom(nodes, /*grads=*/false);
}

ggml_tensor * table_window(ggml_context * ctx, ggml_tensor * table, int first, int count) {
    ggml_tensor * rows = ggml_view_2d(ctx, table, table->ne[0], count, table->nb[1],
                                      static_cast<size_t>(first) * table->nb[1]);
    return ggml_reshape_3d(ctx, rows, table->ne[0], 1, count);
}

ggml_tensor * lower_half(ggml_context * ctx, ggml_tensor * x) {
    return ggml_view_3d(ctx, x, x->ne[0] / 2, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
}

ggml_tensor * upper_half(ggml_context * ctx, ggml_tensor * x) {
    const size_t half = static_cast<size_t>(x->ne[0] / 2) * x->nb[0];
    return ggml_view_3d(ctx, x, x->ne[0] / 2, x->ne[1], x->ne[2], x->nb[1], x->nb[2], half);
}

ggml_tensor * project_heads(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
                            ggml_tensor * x, int head_dim, int heads) {
    return ggml_reshape_3d(ctx, linear(ctx, weight, x, bias), head_dim, heads,
                           static_cast<int>(x->ne[1]));
}

// K is stored position-major so a step appends one contiguous row per position.
ggml_tensor * cache_keys(ggml_context * ctx, const kv_cache & cache,
                         const attention_shape & shape, int total) {
    const size_t layer_bytes = static_cast<size_t>(cache.capacity) * cache.stride * FLOAT;
    return ggml_view_3d(ctx, cache.k, shape.head_dim, shape.n_kv, total,
                        static_cast<size_t>(shape.head_dim) * FLOAT,
                        static_cast<size_t>(cache.stride) * FLOAT,
                        static_cast<size_t>(shape.layer) * layer_bytes);
}

// V is stored channel-major so the value matmul can read rows of positions
// without transposing the whole prefix on every step.
ggml_tensor * cache_values(ggml_context * ctx, const kv_cache & cache,
                           const attention_shape & shape, int total) {
    const size_t column = static_cast<size_t>(cache.capacity) * FLOAT;
    const size_t layer_bytes = column * cache.stride;
    return ggml_view_3d(ctx, cache.v, total, shape.head_dim, shape.n_kv, column,
                        column * shape.head_dim,
                        static_cast<size_t>(shape.layer) * layer_bytes);
}

void append_keys(ggml_context * ctx, ggml_cgraph * graph, const kv_cache & cache,
                 const attention_shape & shape, ggml_tensor * keys) {
    const size_t layer_bytes = static_cast<size_t>(cache.capacity) * cache.stride * FLOAT;
    const size_t offset = static_cast<size_t>(shape.layer) * layer_bytes +
                          static_cast<size_t>(shape.n_past) * cache.stride * FLOAT;
    ggml_tensor * slot = ggml_view_2d(ctx, cache.k, cache.stride, shape.width,
                                      static_cast<size_t>(cache.stride) * FLOAT, offset);
    ggml_tensor * flat = ggml_reshape_2d(ctx, keys, cache.stride, shape.width);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, flat, slot));
}

void append_values(ggml_context * ctx, ggml_cgraph * graph, const kv_cache & cache,
                   const attention_shape & shape, ggml_tensor * values) {
    const size_t column = static_cast<size_t>(cache.capacity) * FLOAT;
    const size_t offset = static_cast<size_t>(shape.layer) * column * cache.stride +
                          static_cast<size_t>(shape.n_past) * FLOAT;
    ggml_tensor * slot = ggml_view_2d(ctx, cache.v, shape.width, cache.stride, column, offset);
    ggml_tensor * flat = ggml_reshape_2d(ctx, values, cache.stride, shape.width);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, ggml_transpose(ctx, flat), slot));
}

ggml_tensor * attend(ggml_context * ctx, ggml_tensor * query, ggml_tensor * keys,
                     ggml_tensor * values, ggml_tensor * mask, int head_dim, int n_head) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * scores = ggml_mul_mat(ctx, keys, query);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * blended = ggml_mul_mat(ctx, values, weights);
    ggml_tensor * merged = ggml_cont(ctx, ggml_permute(ctx, blended, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx, merged, head_dim * n_head, blended->ne[1]);
}

ggml_tensor * rotated_query(ggml_context * ctx, const attention_weights & weights,
                            ggml_tensor * x, const rope_planes & rope,
                            const attention_shape & shape) {
    ggml_tensor * query = project_heads(ctx, weights.wq, weights.wq_b, x, shape.head_dim,
                                        shape.n_head);
    return ggml_permute(ctx, apply_rope(ctx, query, rope), 0, 2, 1, 3);
}

}  // namespace

scratch::scratch(int nodes) {
    ggml_init_params params = {graph_arena(nodes), nullptr, /*no_alloc=*/true};
    ctx = ggml_init(params);
    if (ctx) graph = ggml_new_graph_custom(ctx, nodes, /*grads=*/false);
}

scratch::~scratch() {
    if (ctx) ggml_free(ctx);
}

ggml_tensor * input_i32(ggml_context * ctx, const char * name, int count) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_tensor * input_f32(ggml_context * ctx, const char * name, int ne0, int ne1) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_tensor * mark_output(ggml_cgraph * graph, ggml_tensor * t) {
    ggml_set_output(t);
    ggml_build_forward_expand(graph, t);
    return t;
}

void write_input(ggml_cgraph * graph, const char * name, const void * data, size_t bytes) {
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, name), data, 0, bytes);
}

void read_output(ggml_tensor * t, std::vector<float> & out) {
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

void read_ids(ggml_tensor * t, std::vector<int32_t> & out) {
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

bool allocate_graph(ggml_gallocr_t allocr, ggml_cgraph * graph, const char * stage,
                    std::string * error) {
    if (ggml_gallocr_alloc_graph(allocr, graph)) return true;
    if (error) *error = std::string("audio8: ") + stage + " graph allocation failed";
    return false;
}

bool compute_graph(ggml_backend_t backend, ggml_cgraph * graph, int n_threads,
                   const char * stage, std::string * error) {
    ::tts_cpp::detail::backend_set_n_threads(backend, n_threads);
    if (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS) return true;
    if (error) *error = std::string("audio8: ") + stage + " graph compute failed";
    return false;
}

rope_planes rope_window(ggml_context * ctx, ggml_tensor * cos_table,
                        ggml_tensor * sin_table, int first, int count) {
    return {table_window(ctx, cos_table, first, count),
            table_window(ctx, sin_table, first, count)};
}

ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, const rope_planes & rope) {
    ggml_tensor * lower = lower_half(ctx, x);
    ggml_tensor * upper = upper_half(ctx, x);
    ggml_tensor * rotated_lower = ggml_sub(ctx, ggml_mul(ctx, lower, rope.cos),
                                           ggml_mul(ctx, upper, rope.sin));
    ggml_tensor * rotated_upper = ggml_add(ctx, ggml_mul(ctx, upper, rope.cos),
                                           ggml_mul(ctx, lower, rope.sin));
    return ggml_concat(ctx, rotated_lower, rotated_upper, 0);
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x,
                     ggml_tensor * bias) {
    ggml_tensor * out = ggml_mul_mat(ctx, weight, x);
    return bias ? ggml_add(ctx, out, bias) : out;
}

ggml_tensor * swiglu(ggml_context * ctx, ggml_tensor * w1, ggml_tensor * w2,
                     ggml_tensor * w3, ggml_tensor * x) {
    ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w1, x));
    return ggml_mul_mat(ctx, w2, ggml_mul(ctx, gate, ggml_mul_mat(ctx, w3, x)));
}

ggml_tensor * attention(ggml_context * ctx, ggml_cgraph * graph,
                        const attention_weights & weights, ggml_tensor * x,
                        const rope_planes & rope, const kv_cache & cache,
                        const attention_shape & shape, ggml_tensor * mask) {
    ggml_tensor * key = project_heads(ctx, weights.wk, weights.wk_b, x, shape.head_dim,
                                      shape.n_kv);
    ggml_tensor * value = project_heads(ctx, weights.wv, weights.wv_b, x, shape.head_dim,
                                        shape.n_kv);
    append_keys(ctx, graph, cache, shape, apply_rope(ctx, key, rope));
    append_values(ctx, graph, cache, shape, value);

    const int total = shape.n_past + shape.width;
    ggml_tensor * query = rotated_query(ctx, weights, x, rope, shape);
    ggml_tensor * keys = ggml_permute(ctx, cache_keys(ctx, cache, shape, total), 0, 2, 1, 3);
    ggml_tensor * values = cache_values(ctx, cache, shape, total);
    ggml_tensor * merged = attend(ctx, query, keys, values, mask, shape.head_dim,
                                  shape.n_head);
    return linear(ctx, weights.wo, merged, nullptr);
}

ggml_tensor * windowed_attention(ggml_context * ctx, const attention_weights & weights,
                                 ggml_tensor * x, const rope_planes & rope,
                                 const attention_shape & shape, ggml_tensor * mask) {
    ggml_tensor * key = project_heads(ctx, weights.wk, weights.wk_b, x, shape.head_dim,
                                      shape.n_kv);
    ggml_tensor * value = project_heads(ctx, weights.wv, weights.wv_b, x, shape.head_dim,
                                        shape.n_kv);
    ggml_tensor * query = rotated_query(ctx, weights, x, rope, shape);
    ggml_tensor * keys = ggml_permute(ctx, apply_rope(ctx, key, rope), 0, 2, 1, 3);
    ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, value, 1, 2, 0, 3));
    ggml_tensor * merged = attend(ctx, query, keys, values, mask, shape.head_dim,
                                  shape.n_head);
    return linear(ctx, weights.wo, merged, nullptr);
}

void fill_causal_mask(float * mask, int keys, int queries, int first_query, int window) {
    for (int query = 0; query < queries; ++query) {
        const int position = first_query + query;
        const int oldest = window > 0 ? position - window + 1 : 0;
        for (int key = 0; key < keys; ++key) {
            const bool visible = key <= position && key >= oldest;
            mask[static_cast<size_t>(query) * keys + key] = visible ? 0.0f : -INFINITY;
        }
    }
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
