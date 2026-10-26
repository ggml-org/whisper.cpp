#include "moss/transcribe_networks.h"
#include "moss/transcribe_tensors.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "fit_util.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace parakeet::moss::detail {
namespace {

constexpr int DECODER_GRAPH_NODES = 8192;
constexpr float FFN_DOWN_ACCUMULATION_SCALE = 64.0f;
constexpr size_t CACHE_TENSOR_SLACK = 8;
constexpr int CACHE_ALIGNMENT = 256;
constexpr int STEP_WINDOW = 64;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

int aligned_context(int requested) {
    return (requested + CACHE_ALIGNMENT - 1) / CACHE_ALIGNMENT * CACHE_ALIGNMENT;
}

std::string text_name(int layer, const char * suffix) {
    return "text.blk." + std::to_string(layer) + "." + suffix;
}

void validate_text_layer(const TranscribeModel & model, const TranscribeTextConfig & config, int layer) {
    const int64_t q_dim = (int64_t) config.n_heads * config.head_dim;
    const int64_t kv_dim = (int64_t) config.n_kv_heads * config.head_dim;
    require_transcribe_f32(model.tensor(text_name(layer, "attn_norm.weight")), config.n_embd);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_q.weight")), config.n_embd, q_dim);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_k.weight")), config.n_embd, kv_dim);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_v.weight")), config.n_embd, kv_dim);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_output.weight")), q_dim, config.n_embd);
    require_transcribe_f32(model.tensor(text_name(layer, "attn_q_norm.weight")), config.head_dim);
    require_transcribe_f32(model.tensor(text_name(layer, "attn_k_norm.weight")), config.head_dim);
    require_transcribe_f32(model.tensor(text_name(layer, "ffn_norm.weight")), config.n_embd);
    require_transcribe_weight(model.tensor(text_name(layer, "ffn_gate.weight")), config.n_embd, config.n_ff);
    require_transcribe_weight(model.tensor(text_name(layer, "ffn_up.weight")), config.n_embd, config.n_ff);
    require_transcribe_weight(model.tensor(text_name(layer, "ffn_down.weight")), config.n_ff, config.n_embd);
}

void validate_text_layers(const TranscribeModel & model, const TranscribeTextConfig & config) {
    for (int layer = 0; layer < config.n_layers; ++layer) {
        validate_text_layer(model, config, layer);
    }
}

void require_token_in_vocabulary(int32_t id, int vocab, const char * name) {
    if (id < 0 || id >= vocab) {
        fail(std::string(name) + " is outside the text vocabulary");
    }
}

void validate_prompt_tokens(const TranscribeConfig & config) {
    const int vocab = config.text.vocab;
    require_token_in_vocabulary(config.tokens.audio_start, vocab, "audio_start");
    require_token_in_vocabulary(config.tokens.audio_end, vocab, "audio_end");
    require_token_in_vocabulary(config.tokens.audio_pad, vocab, "audio_pad");
    require_token_in_vocabulary(config.tokens.im_start, vocab, "im_start");
    require_token_in_vocabulary(config.tokens.im_end, vocab, "im_end");
    for (int32_t id : config.default_prompt_ids) {
        require_token_in_vocabulary(id, vocab, "default prompt token");
    }
}

ggml_tensor * as_2d(ggml_context * ctx, ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    return ggml_is_contiguous(tensor) ? ggml_reshape_2d(ctx, tensor, ne0, ne1) : ggml_cont_2d(ctx, tensor, ne0, ne1);
}

template <typename T>
void set_input(ggml_tensor * tensor, const std::vector<T> & values) {
    if (tensor != nullptr && tensor->buffer != nullptr) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(T));
    }
}

template <typename T>
void fill_causal_mask(T * mask, int64_t position, int64_t tokens, int64_t columns, T open, T blocked) {
    std::fill(mask, mask + columns * tokens, blocked);
    for (int64_t query = 0; query < tokens; ++query) {
        std::fill(mask + query * columns, mask + query * columns + position + query + 1, open);
    }
}

std::vector<int64_t> consecutive_slots(int64_t first, int64_t count) {
    std::vector<int64_t> slots((size_t) count);
    for (int64_t i = 0; i < count; ++i) {
        slots[(size_t) i] = first + i;
    }
    return slots;
}

void place_channel_slots(std::vector<int64_t> & slots, int64_t token, int64_t position, int64_t channels,
                         int64_t stride) {
    for (int64_t channel = 0; channel < channels; ++channel) {
        slots[(size_t) (token * channels + channel)] = channel * stride + position;
    }
}

std::vector<int64_t> transposed_slots(int64_t first, int64_t tokens, int64_t channels, int64_t stride) {
    std::vector<int64_t> slots((size_t) (tokens * channels));
    for (int64_t token = 0; token < tokens; ++token) {
        place_channel_slots(slots, token, first + token, channels, stride);
    }
    return slots;
}

struct BatchPlan {
    std::vector<int32_t> ids;
    std::vector<float> audio;
    std::vector<float> audio_mask;
    std::vector<float> text_mask;
    bool has_audio = false;
};

} // namespace

ggml_type transcribe_mask_type(const TranscribeModel & model) {
    return model.fused_attention() ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

uint64_t transcribe_mask_bytes(const TranscribeModel & model, int64_t positions, int64_t tokens) {
    return (uint64_t) positions * (uint64_t) tokens * ggml_type_size(transcribe_mask_type(model));
}

std::vector<uint8_t> build_window_mask(const TranscribeModel & model, int64_t position, int64_t tokens,
                                       int64_t columns) {
    std::vector<uint8_t> bytes(transcribe_mask_bytes(model, columns, tokens));
    const float blocked = -std::numeric_limits<float>::infinity();
    if (transcribe_mask_type(model) == GGML_TYPE_F16) {
        fill_causal_mask(reinterpret_cast<ggml_fp16_t *>(bytes.data()), position, tokens, columns,
                         ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(blocked));
    } else {
        fill_causal_mask(reinterpret_cast<float *>(bytes.data()), position, tokens, columns, 0.0f, blocked);
    }
    return bytes;
}

std::vector<uint8_t> build_causal_mask(const TranscribeModel & model, int64_t position, int64_t tokens) {
    return build_window_mask(model, position, tokens, position + tokens);
}

void validate_transcribe_decoder(const TranscribeModel & model) {
    const TranscribeConfig & config = model.config();
    require_transcribe_weight(model.token_table(), config.text.n_embd, config.text.vocab);
    require_transcribe_f32(model.tensor("text.output_norm.weight"), config.text.n_embd);
    validate_text_layers(model, config.text);
    validate_prompt_tokens(config);
}

struct TranscribeDecoder::Impl {
    TranscribeModel & model;
    const TranscribeTextConfig & config;
    int32_t audio_pad = 0;
    int n_ctx = 0;
    int pos = 0;
    int audio_cursor = 0;
    ggml_context * state = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    std::vector<ggml_tensor *> cache_k;
    std::vector<ggml_tensor *> cache_v;
    int64_t window = 0;
    ggml_tensor * key_slots = nullptr;
    ggml_tensor * value_slots = nullptr;

    Impl(TranscribeModel & owner, int context)
        : model(owner), config(owner.config().text), audio_pad(owner.config().tokens.audio_pad) {
        if (context < 1 || context > config.n_ctx_train) {
            fail("decoder context must be 1.." + std::to_string(config.n_ctx_train));
        }
        n_ctx = aligned_context(context);
        allocate_cache();
    }

    ~Impl() {
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (state) ggml_free(state);
    }

    int64_t kv_dim() const {
        return (int64_t) config.head_dim * config.n_kv_heads;
    }

    // Flash attention reads values in the key layout, one row per position;
    // the unfused graph multiplies by a transposed, one-row-per-channel cache.
    ggml_tensor * new_value_cache() const {
        return model.fused_attention() ? ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim(), n_ctx)
                                       : ggml_new_tensor_2d(state, GGML_TYPE_F16, n_ctx, kv_dim());
    }

    void create_cache_tensors() {
        cache_k.resize((size_t) config.n_layers);
        cache_v.resize((size_t) config.n_layers);
        for (int layer = 0; layer < config.n_layers; ++layer) {
            cache_k[(size_t) layer] = ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim(), n_ctx);
            cache_v[(size_t) layer] = new_value_cache();
        }
    }

    void allocate_cache() {
        state = ggml_init({(2 * (size_t) config.n_layers + CACHE_TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!state) {
            fail("KV cache context allocation failed");
        }
        create_cache_tensors();
        if (model.measure_only()) {
            prepare_cache_measurement();
            return;
        }
        state_buffer = ggml_backend_alloc_ctx_tensors(state, model.backend());
        if (!state_buffer) {
            fail("KV cache allocation failed; the audio is too long for this device");
        }
        ggml_backend_buffer_clear(state_buffer, 0);
    }

    void bind_cache_measurement() {
        for (auto * tensor = ggml_get_first_tensor(state); tensor; tensor = ggml_get_next_tensor(state, tensor)) {
            tensor->buffer = state_buffer;
            tensor->data = reinterpret_cast<void *>(uintptr_t(1));
        }
    }

    void prepare_cache_measurement() {
        const auto buft = ggml_backend_get_default_buffer_type(model.backend());
        state_buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        if (!state_buffer) fail("KV cache measurement buffer failed");
        bind_cache_measurement();
    }

    void clear_cache_measurement_data() {
        for (auto * tensor = ggml_get_first_tensor(state); tensor; tensor = ggml_get_next_tensor(state, tensor)) {
            tensor->data = nullptr;
        }
    }

    uint64_t measure_cache() {
        if (!model.measure_only()) fail("measurement requires a metadata-only model");
        clear_cache_measurement_data();
        const auto bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            state, ggml_backend_get_default_buffer_type(model.backend()));
        bind_cache_measurement();
        return bytes;
    }

    ggml_tensor * weight(int layer, const char * suffix) const {
        return model.tensor(text_name(layer, suffix));
    }

    ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * scale) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, cur, config.rms_eps), scale);
    }

    ggml_tensor * rope(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * positions) const {
        return ggml_rope_ext(ctx, cur, positions, nullptr, config.head_dim, GGML_ROPE_TYPE_NEOX,
                0, config.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * cache_rows(ggml_context * ctx, ggml_tensor * cache, int64_t first, int64_t count) const {
        return ggml_view_2d(ctx, cache, kv_dim(), count, cache->nb[1], first * cache->nb[1]);
    }

    ggml_tensor * key_rows(ggml_context * ctx, int layer, int64_t first, int64_t count) const {
        return cache_rows(ctx, cache_k[(size_t) layer], first, count);
    }

    ggml_tensor * cached_heads(ggml_context * ctx, ggml_tensor * cache, int64_t total) const {
        ggml_tensor * rows = ggml_reshape_3d(ctx, cache_rows(ctx, cache, 0, total), config.head_dim,
                config.n_kv_heads, total);
        return ggml_permute(ctx, rows, 0, 2, 1, 3);
    }

    ggml_tensor * value_columns(ggml_context * ctx, int layer, int64_t first, int64_t count) const {
        ggml_tensor * cache = cache_v[(size_t) layer];
        return ggml_view_2d(ctx, cache, count, kv_dim(), cache->nb[1], first * ggml_element_size(cache));
    }

    ggml_tensor * cached_values(ggml_context * ctx, int layer, int64_t total) const {
        ggml_tensor * cache = cache_v[(size_t) layer];
        return ggml_view_3d(ctx, cache, total, config.head_dim, config.n_kv_heads, cache->nb[1],
                cache->nb[1] * config.head_dim, 0);
    }

    void store_cache(TranscribeGraph & graph, int layer, ggml_tensor * k, ggml_tensor * v, int64_t tokens) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * k_rows = as_2d(ctx, k, kv_dim(), tokens);
        ggml_tensor * v_rows = as_2d(ctx, v, kv_dim(), tokens);
        ggml_build_forward_expand(graph.graph(), store_keys(ctx, layer, k_rows, tokens));
        ggml_build_forward_expand(graph.graph(), store_values(ctx, layer, v_rows, tokens));
    }

    ggml_tensor * store_keys(ggml_context * ctx, int layer, ggml_tensor * k_rows, int64_t tokens) const {
        if (key_slots) {
            return ggml_set_rows(ctx, cache_k[(size_t) layer], k_rows, key_slots);
        }
        return ggml_cpy(ctx, k_rows, key_rows(ctx, layer, pos, tokens));
    }

    ggml_tensor * store_slotted_values(ggml_context * ctx, int layer, ggml_tensor * v_rows, int64_t tokens) const {
        ggml_tensor * cache = cache_v[(size_t) layer];
        if (model.fused_attention()) {
            return ggml_set_rows(ctx, cache, v_rows, value_slots);
        }
        ggml_tensor * elements = ggml_view_2d(ctx, cache, 1, ggml_nelements(cache), ggml_element_size(cache), 0);
        return ggml_set_rows(ctx, elements, ggml_reshape_2d(ctx, v_rows, 1, kv_dim() * tokens), value_slots);
    }

    ggml_tensor * store_values(ggml_context * ctx, int layer, ggml_tensor * v_rows, int64_t tokens) const {
        if (value_slots) {
            return store_slotted_values(ctx, layer, v_rows, tokens);
        }
        if (model.fused_attention()) {
            return ggml_cpy(ctx, v_rows, cache_rows(ctx, cache_v[(size_t) layer], pos, tokens));
        }
        return copy_into_cache(ctx, model.backend(), ggml_transpose(ctx, v_rows),
                value_columns(ctx, layer, pos, tokens));
    }

    float attention_scale() const {
        return 1.0f / std::sqrt((float) config.head_dim);
    }

    ggml_tensor * attend_unfused(ggml_context * ctx, int layer, ggml_tensor * q, int64_t tokens,
                                 ggml_tensor * mask) const {
        ggml_tensor * keys = cached_heads(ctx, cache_k[(size_t) layer], window);
        ggml_tensor * scores = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, keys, q), mask, attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(ctx, cached_values(ctx, layer, window), scores);
        return as_2d(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3), (int64_t) config.head_dim * config.n_heads, tokens);
    }

    ggml_tensor * attend_grouped(ggml_context * ctx, int layer, ggml_tensor * q, ggml_tensor * mask) const {
        const int64_t group = config.n_heads / config.n_kv_heads;
        ggml_tensor * keys = cached_heads(ctx, cache_k[(size_t) layer], window);
        ggml_tensor * queries = ggml_reshape_3d(ctx, q, config.head_dim, group, config.n_kv_heads);
        ggml_tensor * raw = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, keys, queries), window, 1, config.n_heads);
        ggml_tensor * scores = ggml_soft_max_ext(ctx, raw, mask, attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(ctx, cached_values(ctx, layer, window), scores);
        return ggml_reshape_2d(ctx, attended, (int64_t) config.head_dim * config.n_heads, 1);
    }

    ggml_tensor * attend_fused(ggml_context * ctx, int layer, ggml_tensor * q, int64_t tokens,
                               ggml_tensor * mask) const {
        ggml_tensor * attended = ggml_flash_attn_ext(ctx, q, cached_heads(ctx, cache_k[(size_t) layer], window),
                cached_heads(ctx, cache_v[(size_t) layer], window), mask, attention_scale(), 0.0f, 0.0f);
        return ggml_reshape_2d(ctx, attended, (int64_t) config.head_dim * config.n_heads, tokens);
    }

    ggml_tensor * attention(TranscribeGraph & graph, int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                            ggml_tensor * mask) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(layer, "attn_q.weight"), cur),
                config.head_dim, config.n_heads, tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(layer, "attn_k.weight"), cur),
                config.head_dim, config.n_kv_heads, tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(layer, "attn_v.weight"), cur),
                config.head_dim, config.n_kv_heads, tokens);
        q = rope(ctx, rms_norm(ctx, q, weight(layer, "attn_q_norm.weight")), positions);
        k = rope(ctx, rms_norm(ctx, k, weight(layer, "attn_k_norm.weight")), positions);
        store_cache(graph, layer, k, v, tokens);
        return ggml_mul_mat(ctx, weight(layer, "attn_output.weight"), attend(ctx, layer, q, tokens, mask));
    }

    ggml_tensor * attend(ggml_context * ctx, int layer, ggml_tensor * q, int64_t tokens, ggml_tensor * mask) const {
        if (model.fused_attention()) {
            return attend_fused(ctx, layer, ggml_permute(ctx, q, 0, 2, 1, 3), tokens, mask);
        }
        if (vector_step(tokens)) {
            return attend_grouped(ctx, layer, q, mask);
        }
        return attend_unfused(ctx, layer, ggml_permute(ctx, q, 0, 2, 1, 3), tokens, mask);
    }

    ggml_tensor * guarded_feed_forward(ggml_context * ctx, int layer, ggml_tensor * cur) const {
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, weight(layer, "ffn_gate.weight"), cur));
        ggml_tensor * up = ggml_mul_mat(ctx, weight(layer, "ffn_up.weight"), cur);
        ggml_tensor * shrunk = ggml_scale(ctx, ggml_mul(ctx, gate, up), 1.0f / FFN_DOWN_ACCUMULATION_SCALE);
        return ggml_scale(ctx, ggml_mul_mat(ctx, weight(layer, "ffn_down.weight"), shrunk),
                FFN_DOWN_ACCUMULATION_SCALE);
    }

    ggml_tensor * vector_feed_forward(ggml_context * ctx, int layer, ggml_tensor * cur) const {
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, weight(layer, "ffn_gate.weight"), cur));
        ggml_tensor * up = ggml_mul_mat(ctx, weight(layer, "ffn_up.weight"), cur);
        return ggml_mul_mat(ctx, weight(layer, "ffn_down.weight"), ggml_mul(ctx, gate, up));
    }

    bool vector_step(int64_t tokens) const {
        return model.reuses_step_graph() && tokens == 1;
    }

    ggml_tensor * feed_forward(ggml_context * ctx, int layer, ggml_tensor * cur, int64_t tokens) const {
        return vector_step(tokens) ? vector_feed_forward(ctx, layer, cur) : guarded_feed_forward(ctx, layer, cur);
    }

    ggml_tensor * residual(ggml_context * ctx, ggml_tensor * stream, ggml_tensor * update, int64_t tokens) const {
        return vector_step(tokens) ? ggml_add(ctx, update, stream) : ggml_add(ctx, stream, update);
    }

    ggml_tensor * block(TranscribeGraph & graph, int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                        ggml_tensor * mask) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * attended = attention(graph, layer, rms_norm(ctx, cur, weight(layer, "attn_norm.weight")),
                tokens, positions, mask);
        cur = residual(ctx, cur, attended, tokens);
        ggml_tensor * fed = feed_forward(ctx, layer, rms_norm(ctx, cur, weight(layer, "ffn_norm.weight")), tokens);
        return residual(ctx, cur, fed, tokens);
    }

    ggml_tensor * blocks(TranscribeGraph & graph, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                         ggml_tensor * mask) const {
        for (int layer = 0; layer < config.n_layers; ++layer) {
            cur = block(graph, layer, cur, tokens, positions, mask);
        }
        return cur;
    }

    std::vector<int32_t> positions(int64_t tokens) const {
        std::vector<int32_t> values((size_t) tokens);
        for (int64_t i = 0; i < tokens; ++i) {
            values[(size_t) i] = (int32_t) (pos + i);
        }
        return values;
    }

    std::vector<ggml_tensor *> project_head(ggml_context * ctx, ggml_tensor * last) const {
        std::vector<ggml_tensor *> outputs;
        for (const TranscribeHeadChunk & chunk : model.head()) {
            outputs.push_back(ggml_mul_mat(ctx, chunk.weight, last));
        }
        return outputs;
    }

    std::vector<ggml_tensor *> logits(ggml_context * ctx, ggml_tensor * cur, int64_t tokens) const {
        ggml_tensor * last = ggml_view_2d(ctx, cur, config.n_embd, 1, cur->nb[1], (size_t) (tokens - 1) * cur->nb[1]);
        return project_head(ctx, rms_norm(ctx, last, model.tensor("text.output_norm.weight")));
    }

    ggml_tensor * text_embeddings(TranscribeGraph & graph, ggml_tensor * ids, ggml_tensor * host_rows) const {
        return host_rows ? host_rows : ggml_get_rows(graph.ctx(), model.tensor("text.token_embd.weight"), ids);
    }

    ggml_tensor * embed(TranscribeGraph & graph, bool has_audio, ggml_tensor * text, ggml_tensor * audio,
                        ggml_tensor * audio_mask, ggml_tensor * text_mask) const {
        ggml_context * ctx = graph.ctx();
        if (!has_audio) {
            return text;
        }
        return ggml_add(ctx, ggml_mul(ctx, text, text_mask), ggml_mul(ctx, audio, audio_mask));
    }

    void copy_audio_row(BatchPlan & plan, size_t slot, const std::vector<float> & audio_embeddings) {
        const size_t width = (size_t) config.n_embd;
        const size_t offset = (size_t) audio_cursor * width;
        if (offset + width > audio_embeddings.size()) {
            fail("prompt has more audio placeholders than audio embeddings");
        }
        std::copy(audio_embeddings.begin() + (std::ptrdiff_t) offset,
                  audio_embeddings.begin() + (std::ptrdiff_t) (offset + width),
                  plan.audio.begin() + (std::ptrdiff_t) (slot * width));
        plan.audio_mask[slot] = 1.0f;
        plan.text_mask[slot] = 0.0f;
        plan.has_audio = true;
        audio_cursor++;
    }

    void place_token(BatchPlan & plan, size_t slot, int32_t id, const std::vector<float> & audio_embeddings,
                     bool inject_audio) {
        if (id < 0 || id >= config.vocab) {
            fail("token id outside the text vocabulary");
        }
        plan.ids[slot] = id;
        if (inject_audio && id == audio_pad) {
            copy_audio_row(plan, slot, audio_embeddings);
        }
    }

    void place_tokens(BatchPlan & plan, const int32_t * ids, const std::vector<float> & audio_embeddings,
                      bool inject_audio) {
        for (size_t slot = 0; slot < plan.ids.size(); ++slot) {
            place_token(plan, slot, ids[slot], audio_embeddings, inject_audio);
        }
    }

    BatchPlan plan_batch(const int32_t * ids, size_t count, const std::vector<float> & audio_embeddings,
                         bool inject_audio) {
        BatchPlan plan;
        plan.ids.resize(count);
        plan.audio.assign(count * (size_t) config.n_embd, 0.0f);
        plan.audio_mask.assign(count, 0.0f);
        plan.text_mask.assign(count, 1.0f);
        place_tokens(plan, ids, audio_embeddings, inject_audio);
        return plan;
    }

    ggml_tensor * causal_mask_input(TranscribeGraph & graph, int64_t tokens) const {
        return transcribe_mask_type(model) == GGML_TYPE_F16 ? graph.input_f16(window, tokens)
                                                             : graph.input_f32(window, tokens);
    }

    struct DecoderInputs {
        ggml_tensor * ids = nullptr;
        ggml_tensor * text = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * mask = nullptr;
        ggml_tensor * key_slots = nullptr;
        ggml_tensor * value_slots = nullptr;
        ggml_tensor * audio = nullptr;
        ggml_tensor * audio_mask = nullptr;
        ggml_tensor * text_mask = nullptr;
        std::vector<ggml_tensor *> outputs;
    };

    struct GraphShape {
        int64_t tokens = 0;
        int64_t window = 0;
        bool has_audio = false;
        bool want_logits = false;
        bool slotted = false;
    };

    std::unique_ptr<TranscribeGraph> step_graph;
    DecoderInputs step_inputs;
    int64_t step_window = 0;

    static void expand_outputs(TranscribeGraph & graph, const std::vector<ggml_tensor *> & outputs) {
        for (ggml_tensor * output : outputs) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph.graph(), output);
        }
    }

    int64_t value_slot_count(int64_t tokens) const {
        return model.fused_attention() ? tokens : tokens * kv_dim();
    }

    void open_slots(TranscribeGraph & graph, const GraphShape & shape) {
        key_slots = shape.slotted ? graph.input_i64(shape.tokens) : nullptr;
        value_slots = shape.slotted ? graph.input_i64(value_slot_count(shape.tokens)) : nullptr;
    }

    DecoderInputs create_inputs(TranscribeGraph & graph, const GraphShape & shape) {
        const bool host_rows = model.embeds_on_host();
        const int64_t tokens = shape.tokens;
        window = shape.window;
        open_slots(graph, shape);
        DecoderInputs inputs;
        inputs.ids = host_rows ? nullptr : graph.input_i32(tokens);
        inputs.text = host_rows ? graph.input_f32(config.n_embd, tokens) : nullptr;
        inputs.positions = graph.input_i32(tokens);
        inputs.mask = tokens > 1 || shape.slotted ? causal_mask_input(graph, tokens) : nullptr;
        inputs.key_slots = key_slots;
        inputs.value_slots = value_slots;
        inputs.audio = shape.has_audio ? graph.input_f32(config.n_embd, tokens) : nullptr;
        inputs.audio_mask = shape.has_audio ? graph.input_f32(1, tokens) : nullptr;
        inputs.text_mask = shape.has_audio ? graph.input_f32(1, tokens) : nullptr;
        return inputs;
    }

    DecoderInputs build_graph(TranscribeGraph & graph, const GraphShape & shape) {
        DecoderInputs inputs = create_inputs(graph, shape);
        ggml_tensor * cur = embed(graph, shape.has_audio, text_embeddings(graph, inputs.ids, inputs.text), inputs.audio,
                inputs.audio_mask, inputs.text_mask);
        cur = blocks(graph, cur, shape.tokens, inputs.positions, inputs.mask);
        if (shape.want_logits) inputs.outputs = logits(graph.ctx(), cur, shape.tokens);
        expand_outputs(graph, inputs.outputs);
        return inputs;
    }

    GraphShape batch_shape(int64_t tokens, bool has_audio, bool want_logits) const {
        return {tokens, pos + tokens, has_audio, want_logits, false};
    }

    int64_t step_window_at(int64_t position) const {
        const int64_t needed = position + 1;
        return std::min<int64_t>(n_ctx, (needed + STEP_WINDOW - 1) / STEP_WINDOW * STEP_WINDOW);
    }

    GraphShape step_shape() const {
        return {1, step_window_at(pos), false, true, true};
    }

    bool uses_step_graph(int64_t tokens, bool has_audio, bool want_logits) const {
        return model.reuses_step_graph() && tokens == 1 && !has_audio && want_logits;
    }

    std::vector<float> read_logits(const std::vector<ggml_tensor *> & outputs) const {
        std::vector<float> values;
        if (outputs.empty()) {
            return values;
        }
        values.resize((size_t) config.vocab);
        const std::vector<TranscribeHeadChunk> & head = model.head();
        for (size_t i = 0; i < outputs.size(); ++i) {
            ggml_backend_tensor_get(outputs[i], values.data() + head[i].row0, 0, (size_t) head[i].rows * sizeof(float));
        }
        return values;
    }

    TranscribeMemory measure_batch(int position, int tokens, bool has_audio, bool want_logits) {
        if (!model.measure_only()) fail("measurement requires a metadata-only model");
        pos = position;
        TranscribeGraph graph(DECODER_GRAPH_NODES);
        const bool step = uses_step_graph(tokens, has_audio, want_logits);
        build_graph(graph, step ? step_shape() : batch_shape(tokens, has_audio, want_logits));
        return model.measure(graph);
    }

    std::vector<int64_t> value_slot_values(int64_t tokens) const {
        return model.fused_attention() ? consecutive_slots(pos, tokens)
                                       : transposed_slots(pos, tokens, kv_dim(), n_ctx);
    }

    void upload_inputs(const DecoderInputs & inputs, const BatchPlan & plan) {
        const int64_t tokens = (int64_t) plan.ids.size();
        set_input(inputs.ids, plan.ids);
        if (inputs.text) set_input(inputs.text, model.embed_tokens(plan.ids));
        set_input(inputs.positions, positions(tokens));
        if (inputs.mask) set_input(inputs.mask, build_window_mask(model, pos, tokens, inputs.mask->ne[0]));
        if (inputs.key_slots) set_input(inputs.key_slots, consecutive_slots(pos, tokens));
        if (inputs.value_slots) set_input(inputs.value_slots, value_slot_values(tokens));
        set_input(inputs.audio, plan.audio);
        set_input(inputs.audio_mask, plan.audio_mask);
        set_input(inputs.text_mask, plan.text_mask);
    }

    std::vector<float> run(TranscribeGraph & graph, const DecoderInputs & inputs, const BatchPlan & plan) {
        upload_inputs(inputs, plan);
        model.compute(graph);
        pos += (int) plan.ids.size();
        return read_logits(inputs.outputs);
    }

    std::vector<float> forward(const BatchPlan & plan, bool want_logits) {
        const int64_t tokens = (int64_t) plan.ids.size();
        if (pos + tokens > n_ctx) fail("decoder context overflow");
        TranscribeGraph graph(DECODER_GRAPH_NODES);
        const auto inputs = build_graph(graph, batch_shape(tokens, plan.has_audio, want_logits));
        model.allocate(graph);
        return run(graph, inputs, plan);
    }

    void build_step_graph() {
        const GraphShape shape = step_shape();
        auto graph = std::make_unique<TranscribeGraph>(DECODER_GRAPH_NODES);
        DecoderInputs inputs = build_graph(*graph, shape);
        model.allocate(*graph);
        step_graph = std::move(graph);
        step_inputs = std::move(inputs);
        step_window = shape.window;
    }

    bool step_graph_ready() const {
        return step_graph && pos + 1 <= step_window && model.allocated(*step_graph);
    }

    TranscribeGraph & ready_step_graph() {
        if (!step_graph_ready()) build_step_graph();
        return *step_graph;
    }

    std::vector<float> step_in_place(const BatchPlan & plan) {
        if (pos + 1 > n_ctx) fail("decoder context overflow");
        return run(ready_step_graph(), step_inputs, plan);
    }

    std::vector<float> prefill_batches(const std::vector<int32_t> & ids, const std::vector<float> & audio_embeddings,
                                       size_t batch) {
        std::vector<float> values;
        for (size_t first = 0; first < ids.size(); first += batch) {
            const size_t count = std::min(batch, ids.size() - first);
            const bool last = first + count == ids.size();
            values = forward(plan_batch(ids.data() + first, count, audio_embeddings, true), last);
        }
        return values;
    }

    std::vector<float> prefill(const std::vector<int32_t> & ids, const std::vector<float> & audio_embeddings,
                               int batch_tokens) {
        if (ids.empty() || batch_tokens < 1) {
            fail("prefill needs at least one token and a positive batch size");
        }
        if (audio_embeddings.size() % (size_t) config.n_embd != 0) {
            fail("audio embeddings do not match the decoder width");
        }
        audio_cursor = 0;
        std::vector<float> values = prefill_batches(ids, audio_embeddings, (size_t) batch_tokens);
        if ((size_t) audio_cursor * config.n_embd != audio_embeddings.size()) {
            fail("prompt audio placeholders do not match the audio embeddings");
        }
        return values;
    }

    std::vector<float> step(int32_t id) {
        const BatchPlan plan = plan_batch(&id, 1, {}, false);
        return uses_step_graph(1, false, true) ? step_in_place(plan) : forward(plan, true);
    }
};

TranscribeDecoder::TranscribeDecoder(TranscribeModel & model, int n_ctx) : impl_(new Impl(model, n_ctx)) {}

TranscribeDecoder::~TranscribeDecoder() = default;

std::vector<float> TranscribeDecoder::prefill(const std::vector<int32_t> & ids,
                                              const std::vector<float> & audio_embeddings, int batch_tokens) {
    return impl_->prefill(ids, audio_embeddings, batch_tokens);
}

std::vector<float> TranscribeDecoder::step(int32_t id) {
    return impl_->step(id);
}

uint64_t TranscribeDecoder::measure_cache() const { return impl_->measure_cache(); }
uint64_t TranscribeDecoder::host_state_bytes() const {
    return ggml_get_mem_size(impl_->state) +
        (impl_->cache_k.capacity() + impl_->cache_v.capacity()) * sizeof(ggml_tensor *);
}
TranscribeMemory TranscribeDecoder::measure_batch(int position, int tokens, bool has_audio, bool want_logits) {
    return impl_->measure_batch(position, tokens, has_audio, want_logits);
}

int TranscribeDecoder::position() const { return impl_->pos; }
int TranscribeDecoder::context() const { return impl_->n_ctx; }

} // namespace parakeet::moss::detail
