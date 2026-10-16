#include "moss/sfx_networks.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tts_cpp::moss::detail {
namespace {

constexpr int TEXT_GRAPH_NODES = 8192;
constexpr float FFN_DOWN_ACCUMULATION_SCALE = 64.0f;

struct TextGraph {
    SfxModel & model;
    const SfxTextConfig & config;
    SfxGraph & graph;

    ggml_context * ctx() const { return graph.ctx(); }

    ggml_tensor * weight(int layer, const char * suffix) const {
        return model.tensor("text.blk." + std::to_string(layer) + "." + suffix);
    }

    ggml_tensor * rms_norm(ggml_tensor * cur, ggml_tensor * scale) const {
        return ggml_mul(ctx(), ggml_rms_norm(ctx(), cur, config.rms_eps), scale);
    }

    ggml_tensor * rope(ggml_tensor * cur, ggml_tensor * positions) const {
        return ggml_rope_ext(ctx(), cur, positions, nullptr, config.head_dim, GGML_ROPE_TYPE_NEOX,
                0, config.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * heads(ggml_tensor * cur, int n_heads, int64_t tokens) const {
        return ggml_reshape_3d(ctx(), cur, config.head_dim, n_heads, tokens);
    }

    ggml_tensor * attention(int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                            ggml_tensor * mask) const {
        ggml_tensor * q = heads(ggml_mul_mat(ctx(), weight(layer, "attn_q.weight"), cur), config.n_heads, tokens);
        ggml_tensor * k = heads(ggml_mul_mat(ctx(), weight(layer, "attn_k.weight"), cur), config.n_kv_heads, tokens);
        ggml_tensor * v = heads(ggml_mul_mat(ctx(), weight(layer, "attn_v.weight"), cur), config.n_kv_heads, tokens);
        q = rope(rms_norm(q, weight(layer, "attn_q_norm.weight")), positions);
        k = rope(rms_norm(k, weight(layer, "attn_k_norm.weight")), positions);
        ggml_tensor * attended = model.fused_attention() ? attend_fused(q, k, v, mask)
                                                         : attend_unfused(q, k, v, mask);
        return ggml_mul_mat(ctx(), weight(layer, "attn_output.weight"), attended);
    }

    float attention_scale() const {
        return 1.0f / std::sqrt((float) config.head_dim);
    }

    int64_t query_width() const {
        return (int64_t) config.head_dim * config.n_heads;
    }

    ggml_tensor * attend_unfused(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) const {
        const int64_t tokens = q->ne[2];
        q = ggml_permute(ctx(), q, 0, 2, 1, 3);
        k = ggml_permute(ctx(), k, 0, 2, 1, 3);
        ggml_tensor * values = ggml_cont(ctx(), ggml_permute(ctx(), v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_soft_max_ext(ctx(), ggml_mul_mat(ctx(), k, q), mask, attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_permute(ctx(), ggml_mul_mat(ctx(), values, scores), 0, 2, 1, 3);
        return ggml_cont_2d(ctx(), attended, query_width(), tokens);
    }

    ggml_tensor * attend_fused(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) const {
        const int64_t tokens = q->ne[2];
        ggml_tensor * attended = ggml_flash_attn_ext(ctx(), ggml_permute(ctx(), q, 0, 2, 1, 3),
                ggml_permute(ctx(), k, 0, 2, 1, 3), ggml_permute(ctx(), v, 0, 2, 1, 3), mask,
                attention_scale(), 0.0f, 0.0f);
        return ggml_reshape_2d(ctx(), attended, query_width(), tokens);
    }

    ggml_tensor * scaled_down_projection(ggml_tensor * down, ggml_tensor * hidden) const {
        ggml_tensor * shrunk = ggml_scale(ctx(), hidden, 1.0f / FFN_DOWN_ACCUMULATION_SCALE);
        return ggml_scale(ctx(), ggml_mul_mat(ctx(), down, shrunk), FFN_DOWN_ACCUMULATION_SCALE);
    }

    ggml_tensor * feed_forward(int layer, ggml_tensor * cur) const {
        ggml_tensor * gate = ggml_silu(ctx(), ggml_mul_mat(ctx(), weight(layer, "ffn_gate.weight"), cur));
        ggml_tensor * up = ggml_mul_mat(ctx(), weight(layer, "ffn_up.weight"), cur);
        return scaled_down_projection(weight(layer, "ffn_down.weight"), ggml_mul(ctx(), gate, up));
    }

    ggml_tensor * block(int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                        ggml_tensor * mask) const {
        ggml_tensor * attended = attention(layer, rms_norm(cur, weight(layer, "attn_norm.weight")),
                tokens, positions, mask);
        cur = ggml_add(ctx(), cur, attended);
        ggml_tensor * fed = feed_forward(layer, rms_norm(cur, weight(layer, "ffn_norm.weight")));
        return ggml_add(ctx(), cur, fed);
    }

    ggml_tensor * blocks(ggml_tensor * cur, int64_t tokens, ggml_tensor * positions, ggml_tensor * mask) const {
        for (int layer = 0; layer < config.n_layers; ++layer) {
            cur = block(layer, cur, tokens, positions, mask);
        }
        return cur;
    }
};

std::vector<int32_t> token_positions(size_t count) {
    std::vector<int32_t> positions(count);
    for (size_t i = 0; i < count; ++i) {
        positions[i] = (int32_t) i;
    }
    return positions;
}

std::vector<float> causal_mask(size_t count) {
    std::vector<float> mask(count * count, -std::numeric_limits<float>::infinity());
    for (size_t query = 0; query < count; ++query) {
        std::fill(mask.begin() + (std::ptrdiff_t) (query * count),
                  mask.begin() + (std::ptrdiff_t) (query * count + query + 1), 0.0f);
    }
    return mask;
}

void validate_text_layer(const SfxModel & model, const SfxTextConfig & config, int layer) {
    const std::string prefix = "text.blk." + std::to_string(layer) + ".";
    const int64_t q_dim = (int64_t) config.n_heads * config.head_dim;
    const int64_t kv_dim = (int64_t) config.n_kv_heads * config.head_dim;
    require_vector(model.tensor(prefix + "attn_norm.weight"), config.n_embd, 1);
    require_matrix(model.tensor(prefix + "attn_q.weight"), config.n_embd, q_dim);
    require_matrix(model.tensor(prefix + "attn_k.weight"), config.n_embd, kv_dim);
    require_matrix(model.tensor(prefix + "attn_v.weight"), config.n_embd, kv_dim);
    require_matrix(model.tensor(prefix + "attn_output.weight"), q_dim, config.n_embd);
    require_vector(model.tensor(prefix + "attn_q_norm.weight"), config.head_dim, 1);
    require_vector(model.tensor(prefix + "attn_k_norm.weight"), config.head_dim, 1);
    require_vector(model.tensor(prefix + "ffn_norm.weight"), config.n_embd, 1);
    require_matrix(model.tensor(prefix + "ffn_gate.weight"), config.n_embd, config.n_ff);
    require_matrix(model.tensor(prefix + "ffn_up.weight"), config.n_embd, config.n_ff);
    require_matrix(model.tensor(prefix + "ffn_down.weight"), config.n_ff, config.n_embd);
}

void validate_text_layers(const SfxModel & model, const SfxTextConfig & config) {
    for (int layer = 0; layer < config.n_layers; ++layer) {
        validate_text_layer(model, config, layer);
    }
}

void require_ids_in_vocabulary(const std::vector<int32_t> & ids, int64_t vocab) {
    for (int32_t id : ids) {
        if (id < 0 || id >= vocab) {
            throw std::runtime_error("moss sfx: token id outside the text vocabulary");
        }
    }
}

void validate_ids(const SfxModel & model, const std::vector<int32_t> & ids) {
    const SfxTextConfig & config = model.config().text;
    if ((int) ids.size() > config.max_tokens) {
        throw std::runtime_error("moss sfx: prompt exceeds the text encoder context");
    }
    require_ids_in_vocabulary(ids, model.tensor(SFX_TOKEN_TABLE)->ne[1]);
}

struct TextInputs {
    ggml_tensor * ids;
    ggml_tensor * positions;
    ggml_tensor * mask;
    ggml_tensor * output;
};

TextInputs build_text_graph(SfxModel & model, SfxGraph & graph, int64_t tokens) {
    const auto & config = model.config().text;
    TextGraph builder{model, config, graph};
    ggml_tensor * input_ids = graph.input_i32(tokens);
    ggml_tensor * positions = graph.input_i32(tokens);
    ggml_tensor * mask = graph.input_f32(tokens, tokens);
    ggml_tensor * attention_mask = model.fused_attention() ? ggml_cast(graph.ctx(), mask, GGML_TYPE_F16) : mask;
    ggml_tensor * cur = ggml_get_rows(graph.ctx(), model.tensor(SFX_TOKEN_TABLE), input_ids);
    cur = builder.blocks(cur, tokens, positions, attention_mask);
    cur = builder.rms_norm(cur, model.tensor("text.output_norm.weight"));
    ggml_set_output(cur);
    ggml_build_forward_expand(graph.graph(), cur);

    return {input_ids, positions, mask, cur};
}

} // namespace

void validate_text_encoder(const SfxModel & model) {
    const SfxTextConfig & config = model.config().text;
    const ggml_tensor * embedding = model.tensor(SFX_TOKEN_TABLE);
    require_matrix(embedding, config.n_embd, embedding->ne[1]);
    require_vector(model.tensor("text.output_norm.weight"), config.n_embd, 1);
    validate_text_layers(model, config);
}

std::vector<float> encode_text(SfxModel & model, const std::vector<int32_t> & ids) {
    validate_ids(model, ids);
    const SfxTextConfig & config = model.config().text;
    std::vector<float> context((size_t) config.max_tokens * config.n_embd, 0.0f);
    if (ids.empty()) {
        return context;
    }
    const int64_t tokens = (int64_t) ids.size();
    SfxGraph graph(TEXT_GRAPH_NODES);
    const auto inputs = build_text_graph(model, graph, tokens);
    model.allocate(graph);
    const std::vector<int32_t> position_data = token_positions(ids.size());
    const std::vector<float> mask_data = causal_mask(ids.size());
    ggml_backend_tensor_set(inputs.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(inputs.positions, position_data.data(), 0, position_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(inputs.mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    model.compute(graph);
    ggml_backend_tensor_get(inputs.output, context.data(), 0, ggml_nbytes(inputs.output));
    return context;
}

SfxMemory measure_text(SfxModel & model, const std::vector<int32_t> & ids) {
    validate_ids(model, ids);
    if (ids.empty()) return {};
    SfxGraph graph(TEXT_GRAPH_NODES);
    build_text_graph(model, graph, (int64_t) ids.size());
    return model.measure(graph);
}

} // namespace tts_cpp::moss::detail
