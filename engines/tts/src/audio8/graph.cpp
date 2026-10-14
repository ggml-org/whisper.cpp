#include "audio8/graph.h"

#include "backend_util.h"

#include <cmath>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

const size_t FLOAT = sizeof(float);

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

ggml_tensor * swap_halves(ggml_context * ctx, ggml_tensor * x) {
    return ggml_concat(ctx, upper_half(ctx, x), lower_half(ctx, x), 0);
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

// rows: [stride, width], one position per row. Stacked QKV leaves a gap
// between key rows; OpenCL CPY requires that source to be contiguous even
// though its destination is a view into the persistent cache.
void append_keys(ggml_context * ctx, ggml_cgraph * graph, const kv_cache & cache,
                 const attention_shape & shape, ggml_tensor * rows) {
    const size_t layer_bytes = static_cast<size_t>(cache.capacity) * cache.stride * FLOAT;
    const size_t offset = static_cast<size_t>(shape.layer) * layer_bytes +
                          static_cast<size_t>(shape.n_past) * cache.stride * FLOAT;
    ggml_tensor * slot = ggml_view_2d(ctx, cache.k, cache.stride, shape.width,
                                      static_cast<size_t>(cache.stride) * FLOAT, offset);
    if (!ggml_is_contiguous(rows)) rows = ggml_cont(ctx, rows);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, rows, slot));
}

void append_values(ggml_context * ctx, ggml_cgraph * graph, const kv_cache & cache,
                   const attention_shape & shape, ggml_tensor * rows) {
    if (cache.flash_attention) {
        const size_t layer_bytes = static_cast<size_t>(cache.capacity) * cache.stride * FLOAT;
        const size_t offset = static_cast<size_t>(shape.layer) * layer_bytes +
                              static_cast<size_t>(shape.n_past) * cache.stride * FLOAT;
        ggml_tensor * slot = ggml_view_2d(ctx, cache.v, cache.stride, shape.width,
                                          static_cast<size_t>(cache.stride) * FLOAT, offset);
        if (!ggml_is_contiguous(rows)) rows = ggml_cont(ctx, rows);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, rows, slot));
        return;
    }
    const size_t column = static_cast<size_t>(cache.capacity) * FLOAT;
    const size_t offset = static_cast<size_t>(shape.layer) * column * cache.stride +
                          static_cast<size_t>(shape.n_past) * FLOAT;
    ggml_tensor * slot = ggml_view_2d(ctx, cache.v, shape.width, cache.stride, column, offset);
    // OpenCL CPY aborts on strided src on Adreno 830; materialise contiguously first.
    ggml_build_forward_expand(graph, ggml_cpy(ctx, ggml_cont(ctx, ggml_transpose(ctx, rows)), slot));
}

// The rotated query [head_dim, n_head, width] and the rotated keys and the
// values as [head_dim * n_kv, width] rows.
struct projected_heads {
    ggml_tensor * query = nullptr;
    ggml_tensor * key_rows = nullptr;
    ggml_tensor * value_rows = nullptr;
};

ggml_tensor * as_rows(ggml_context * ctx, ggml_tensor * heads) {
    return ggml_reshape_2d(ctx, heads, heads->ne[0] * heads->ne[1], heads->ne[2]);
}

projected_heads project_separately(ggml_context * ctx, const attention_weights & weights,
                                   ggml_tensor * x, const rope_planes & rope,
                                   const attention_shape & shape) {
    projected_heads out;
    out.query = apply_rope(ctx, project_heads(ctx, weights.wq, weights.wq_b, x,
                                              shape.head_dim, shape.n_head), rope);
    out.key_rows = as_rows(ctx, apply_rope(ctx, project_heads(ctx, weights.wk, weights.wk_b,
                                                              x, shape.head_dim, shape.n_kv),
                                           rope));
    out.value_rows = as_rows(ctx, project_heads(ctx, weights.wv, weights.wv_b, x,
                                                shape.head_dim, shape.n_kv));
    return out;
}

ggml_tensor * head_slice(ggml_context * ctx, ggml_tensor * heads, int first, int count) {
    return ggml_view_3d(ctx, heads, heads->ne[0], count, heads->ne[2], heads->nb[1],
                        heads->nb[2], static_cast<size_t>(first) * heads->nb[1]);
}

ggml_tensor * row_slice(ggml_context * ctx, ggml_tensor * heads, int first, int count) {
    return ggml_view_2d(ctx, heads, heads->ne[0] * count, heads->ne[2], heads->nb[2],
                        static_cast<size_t>(first) * heads->nb[1]);
}

// One matmul yields q, k and v as consecutive heads, and q and k rotate in one
// pass; every value is computed exactly as the separate path computes it.
projected_heads project_stacked(ggml_context * ctx, const attention_weights & weights,
                                ggml_tensor * x, const rope_planes & rope,
                                const attention_shape & shape) {
    const int rotated_heads = shape.n_head + shape.n_kv;
    ggml_tensor * heads = project_heads(ctx, weights.wqkv, weights.wqkv_b, x, shape.head_dim,
                                        rotated_heads + shape.n_kv);
    ggml_tensor * rotated = apply_rope(ctx, head_slice(ctx, heads, 0, rotated_heads), rope);
    projected_heads out;
    out.query = head_slice(ctx, rotated, 0, shape.n_head);
    out.key_rows = row_slice(ctx, rotated, shape.n_head, shape.n_kv);
    out.value_rows = row_slice(ctx, heads, rotated_heads, shape.n_kv);
    return out;
}

projected_heads project_attention(ggml_context * ctx, const attention_weights & weights,
                                  ggml_tensor * x, const rope_planes & rope,
                                  const attention_shape & shape) {
    return weights.wqkv ? project_stacked(ctx, weights, x, rope, shape)
                        : project_separately(ctx, weights, x, rope, shape);
}

ggml_tensor * attend(ggml_context * ctx, ggml_tensor * query, ggml_tensor * keys,
                     ggml_tensor * values, ggml_tensor * mask, int head_dim, int n_head,
                     bool precise_values) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * scores = precise_mul_mat(ctx, keys, query);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * blended = multiply_mat(ctx, values, weights, precise_values);
    if (blended->ne[1] == 1) return ggml_reshape_2d(ctx, blended, head_dim * n_head, 1);
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

// Per-backend route and its accuracy: docs/audio8.md, "Quantised weights take
// a different route per backend". Pinned by test-audio8-quantised-precision.
bool quantised_matmul_is_already_f32(const ggml_tensor * weight) {
    return ggml_is_quantized(weight->type) && weight->buffer &&
           ::tts_cpp::detail::reg_name_is_cuda(
               ::tts_cpp::detail::buft_reg_name(ggml_backend_buffer_get_type(weight->buffer)));
}

}  // namespace

ggml_tensor * precise_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    if (!quantised_matmul_is_already_f32(a)) {
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    }
    return out;
}

ggml_tensor * multiply_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                           bool precise) {
    return precise ? precise_mul_mat(ctx, a, b) : ggml_mul_mat(ctx, a, b);
}

size_t scratch_arena_bytes(int nodes) {
    return static_cast<size_t>(nodes) * ggml_tensor_overhead() +
           ggml_graph_overhead_custom(nodes, /*grads=*/false);
}

scratch::scratch(int nodes) {
    ggml_init_params params = {scratch_arena_bytes(nodes), nullptr, /*no_alloc=*/true};
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

bool prepare_graph(ggml_backend_t backend, ::tts_cpp::detail::sched_fallback & sched,
                   ggml_backend_buffer_t weight_buffer, ggml_gallocr_t allocr,
                   ggml_cgraph * graph, const char * stage, bool & use_sched,
                   std::string * error) {
    use_sched = ::tts_cpp::detail::sched_force_enabled() ||
                !::tts_cpp::detail::graph_fully_supported(backend, graph);
    if (!use_sched) {
        if (ggml_gallocr_reserve(allocr, graph) && ggml_gallocr_alloc_graph(allocr, graph)) {
            return true;
        }
        if (error) *error = std::string("audio8: ") + stage + " graph allocation failed";
        return false;
    }
    if (::tts_cpp::detail::graph_has_unsupported_preallocated_op(backend, graph)) {
        if (error) {
            *error = std::string("audio8: ") + stage +
                     " graph has an unsupported persistent-buffer operation";
        }
        return false;
    }
    if (!::tts_cpp::detail::sched_fallback_ensure(
            sched, backend, 2 * AUDIO8_MAX_NODES, {weight_buffer}) ||
        !::tts_cpp::detail::sched_fallback_alloc(sched, graph)) {
        if (error) *error = std::string("audio8: ") + stage + " graph allocation failed";
        return false;
    }
    return true;
}

bool compute_graph(ggml_backend_t backend, ::tts_cpp::detail::sched_fallback & sched,
                   ggml_cgraph * graph, bool use_sched, int n_threads,
                   const char * stage, std::string * error) {
    const ggml_status status =
        use_sched ? ::tts_cpp::detail::sched_fallback_compute(sched, backend, graph, n_threads)
                  : ::tts_cpp::detail::direct_compute(backend, graph, n_threads);
    if (status == GGML_STATUS_SUCCESS) return true;
    if (error) *error = std::string("audio8: ") + stage + " graph compute failed";
    return false;
}

rope_planes rope_window(ggml_context * ctx, ggml_tensor * cos_table,
                        ggml_tensor * sin_table, int first, int count) {
    ggml_tensor * cosines = table_window(ctx, cos_table, first, count);
    ggml_tensor * sines = table_window(ctx, sin_table, first, count);
    return {ggml_concat(ctx, cosines, cosines, 0),
            ggml_concat(ctx, ggml_neg(ctx, sines), sines, 0)};
}

ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, const rope_planes & rope) {
    return ggml_add(ctx, ggml_mul(ctx, x, rope.cos),
                    ggml_mul(ctx, swap_halves(ctx, x), rope.signed_sin));
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x,
                     ggml_tensor * bias) {
    ggml_tensor * out = precise_mul_mat(ctx, weight, x);
    return bias ? ggml_add(ctx, out, bias) : out;
}

ggml_tensor * swiglu(ggml_context * ctx, ggml_tensor * w1, ggml_tensor * w2,
                     ggml_tensor * w3, ggml_tensor * x) {
    ggml_tensor * gated = ggml_swiglu_split(ctx, precise_mul_mat(ctx, w1, x),
                                            precise_mul_mat(ctx, w3, x));
    return precise_mul_mat(ctx, w2, gated);
}

ggml_tensor * swiglu_stacked(ggml_context * ctx, ggml_tensor * w13, ggml_tensor * w2,
                             ggml_tensor * x) {
    return precise_mul_mat(ctx, w2, ggml_swiglu(ctx, precise_mul_mat(ctx, w13, x)));
}

ggml_tensor * attention(ggml_context * ctx, ggml_cgraph * graph,
                        const attention_weights & weights, ggml_tensor * x,
                        const rope_planes & rope, const kv_cache & cache,
                        const attention_shape & shape, ggml_tensor * mask) {
    const projected_heads heads = project_attention(ctx, weights, x, rope, shape);
    append_keys(ctx, graph, cache, shape, heads.key_rows);
    append_values(ctx, graph, cache, shape, heads.value_rows);

    const int total = shape.n_past + shape.width;
    ggml_tensor * query = ggml_permute(ctx, heads.query, 0, 2, 1, 3);
    ggml_tensor * keys = ggml_permute(ctx, cache_keys(ctx, cache, shape, total), 0, 2, 1, 3);
    ggml_tensor * values;
    if (cache.flash_attention) {
        const size_t layer_bytes = static_cast<size_t>(cache.capacity) * cache.stride * FLOAT;
        ggml_tensor * stored = ggml_view_3d(ctx, cache.v, shape.head_dim, shape.n_kv, total,
                                            static_cast<size_t>(shape.head_dim) * FLOAT,
                                            static_cast<size_t>(cache.stride) * FLOAT,
                                            static_cast<size_t>(shape.layer) * layer_bytes);
        if (!mask && shape.width == 1) {
            ggml_tensor * value_heads = ggml_permute(ctx, stored, 0, 2, 1, 3);
            const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
            ggml_tensor * blended = ggml_flash_attn_ext(ctx, query, keys, value_heads,
                                                        nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(blended, GGML_PREC_F32);
            ggml_tensor * merged = ggml_reshape_2d(ctx, blended, shape.head_dim * shape.n_head, 1);
            return linear(ctx, weights.wo, merged, nullptr);
        }
        // Prefill remains causal. Materialize its channel-major operand once
        // rather than changing the persistent cache layout between graph calls.
        values = ggml_cont(ctx, ggml_permute(ctx, stored, 1, 2, 0, 3));
    } else {
        values = cache_values(ctx, cache, shape, total);
    }
    ggml_tensor * merged = attend(ctx, query, keys, values, mask, shape.head_dim,
                                  shape.n_head, shape.precise_values);
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
                                  shape.n_head, shape.precise_values);
    return linear(ctx, weights.wo, merged, nullptr);
}

windowed_step windowed_attention_after(ggml_context * ctx, const attention_weights & weights,
                                       ggml_tensor * x, const rope_planes & rope,
                                       const attention_shape & shape, ggml_tensor * held_keys,
                                       ggml_tensor * held_values, ggml_tensor * mask) {
    ggml_tensor * key = apply_rope(ctx, project_heads(ctx, weights.wk, weights.wk_b, x,
                                                      shape.head_dim, shape.n_kv),
                                   rope);
    ggml_tensor * value = project_heads(ctx, weights.wv, weights.wv_b, x, shape.head_dim,
                                        shape.n_kv);
    windowed_step step;
    step.keys = held_keys ? ggml_concat(ctx, held_keys, key, 2) : key;
    step.values = held_values ? ggml_concat(ctx, held_values, value, 2) : value;
    ggml_tensor * query = rotated_query(ctx, weights, x, rope, shape);
    ggml_tensor * keys = ggml_permute(ctx, step.keys, 0, 2, 1, 3);
    ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, step.values, 1, 2, 0, 3));
    ggml_tensor * merged = attend(ctx, query, keys, values, mask, shape.head_dim,
                                  shape.n_head, shape.precise_values);
    step.out = linear(ctx, weights.wo, merged, nullptr);
    return step;
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
