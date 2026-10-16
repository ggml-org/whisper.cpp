#include "moss/speech_lm.h"

#include "moss/gguf_metadata.h"
#include "moss/host_tables.h"
#include "moss/sfx_model.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "sched_dispatch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * ARCH = "moss-speech";
constexpr const char * OWNER = "moss speech";
constexpr int GRAPH_NODES = 16384;
constexpr int MAX_LAYERS = 256;
constexpr int MAX_WIDTH = 1 << 16;
constexpr int MAX_HEADS = 1024;
constexpr int MAX_THREADS = 1024;
constexpr int MAX_CONTEXT = 1 << 18;
constexpr int MAX_VOCAB = 1 << 24;
constexpr int CACHE_ALIGNMENT = 256;
constexpr size_t TENSOR_SLACK = 8;
constexpr float FFN_DOWN_ACCUMULATION_SCALE = 64.0f;
constexpr uint64_t MIB = 1024 * 1024;
constexpr int SHARED_BLOCK = 0;
constexpr int TEXT_BLOCK = 1;
constexpr int AUDIO_BLOCK = 2;
constexpr int BLOCK_COUNT = 3;
const char * const BLOCK_PREFIXES[BLOCK_COUNT] = {"blk.", "text.blk.", "audio.blk."};
const char * const HOST_TABLES[] = {"text.token_embd.weight", "audio.token_embd.weight", "text.output.weight"};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(std::string(OWNER) + ": " + message);
}

bool within(int value, int low, int high) {
    return value >= low && value <= high;
}

bool is_positive_finite(float value) {
    return std::isfinite(value) && value > 0.0f;
}

std::string key(const char * name) {
    return std::string(ARCH) + "." + name;
}

int aligned_context(int requested) {
    return (requested + CACHE_ALIGNMENT - 1) / CACHE_ALIGNMENT * CACHE_ALIGNMENT;
}

SpeechTokens read_tokens(const GgufMetadata & meta) {
    SpeechTokens tokens;
    tokens.text_placeholder = (int32_t) meta.u32(key("token.text_placeholder"));
    tokens.audio_pad        = (int32_t) meta.u32(key("token.audio_pad"));
    tokens.speech_start     = (int32_t) meta.u32(key("token.speech_start"));
    tokens.speech_end       = (int32_t) meta.u32(key("token.speech_end"));
    tokens.im_start         = (int32_t) meta.u32(key("token.im_start"));
    tokens.im_end           = (int32_t) meta.u32(key("token.im_end"));
    tokens.pad              = (int32_t) meta.u32(key("token.pad"));
    return tokens;
}

SpeechLmConfig read_config(const GgufMetadata & meta) {
    SpeechLmConfig config;
    config.n_shared_layers   = (int) meta.u32(key("block_count"));
    config.n_modality_layers = (int) meta.u32(key("modality_block_count"));
    config.n_embd            = (int) meta.u32(key("embedding_length"));
    config.n_ff              = (int) meta.u32(key("feed_forward_length"));
    config.n_heads           = (int) meta.u32(key("attention.head_count"));
    config.n_kv_heads        = (int) meta.u32(key("attention.head_count_kv"));
    config.head_dim          = (int) meta.u32(key("attention.key_length"));
    config.n_ctx_train       = (int) meta.u32(key("context_length"));
    config.rope_base         = meta.f32(key("rope.freq_base"));
    config.rms_eps           = meta.f32(key("attention.layer_norm_rms_epsilon"));
    config.text_vocab        = (int) meta.u32(key("text_vocab_size"));
    config.audio_vocab       = (int) meta.u32(key("audio_vocab_size"));
    config.tokens            = read_tokens(meta);
    config.audio_system_prompt = meta.str(key("audio_output_system_prompt"));
    config.text_system_prompt  = meta.str(key("text_output_system_prompt"));
    return config;
}

void validate_geometry(const SpeechLmConfig & c) {
    if (!within(c.n_shared_layers, 1, MAX_LAYERS) || !within(c.n_modality_layers, 1, MAX_LAYERS) ||
        !within(c.n_embd, 1, MAX_WIDTH) || !within(c.n_ff, 1, MAX_WIDTH * 4) ||
        !within(c.n_heads, 1, MAX_HEADS) || !within(c.n_kv_heads, 1, MAX_HEADS) || c.n_heads % c.n_kv_heads != 0 ||
        !within(c.head_dim, 2, MAX_WIDTH) || c.head_dim % 2 != 0 || !within(c.n_ctx_train, 1, MAX_CONTEXT) ||
        !is_positive_finite(c.rope_base) || !is_positive_finite(c.rms_eps)) {
        fail("invalid model geometry");
    }
}

bool is_weight_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q4_0;
}

void require_weight(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    require_shape(tensor, ne0, ne1);
    if (!is_weight_type(tensor->type)) {
        fail(std::string(ggml_get_name(tensor)) + " has an unsupported type");
    }
}

bool host_table(const char * name) {
    for (const char * table : HOST_TABLES) {
        if (std::strcmp(name, table) == 0) {
            return true;
        }
    }
    return false;
}

bool in_vocab(int32_t id, int vocab) {
    return id >= 0 && id < vocab;
}

void validate_tokens(const SpeechLmConfig & c) {
    const SpeechTokens & t = c.tokens;
    if (!within(c.text_vocab, 1, MAX_VOCAB) || !within(c.audio_vocab, 1, MAX_VOCAB) ||
        !in_vocab(t.text_placeholder, c.text_vocab) || !in_vocab(t.speech_start, c.text_vocab) ||
        !in_vocab(t.im_start, c.text_vocab) || !in_vocab(t.im_end, c.text_vocab) || !in_vocab(t.pad, c.text_vocab) ||
        !in_vocab(t.audio_pad, c.audio_vocab) || !in_vocab(t.speech_end, c.audio_vocab)) {
        fail("special tokens fall outside their vocabularies");
    }
}

std::string layer_name(int block, int layer, const char * suffix) {
    return std::string(BLOCK_PREFIXES[block]) + std::to_string(layer) + "." + suffix;
}

int block_layers(const SpeechLmConfig & c, int block) {
    return block == SHARED_BLOCK ? c.n_shared_layers : c.n_modality_layers;
}

struct LayerCache {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct BatchInputs {
    std::vector<int32_t> text_ids;
    std::vector<int32_t> audio_ids;
    std::vector<float> text_select;
    std::vector<float> audio_select;
};

template <typename T>
void set_input(ggml_tensor * tensor, const std::vector<T> & values) {
    if (tensor != nullptr && tensor->buffer != nullptr) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(T));
    }
}

} // namespace

void check_cache_memory(uint64_t cache_bytes, uint64_t free_bytes) {
    if (free_bytes > 0 && cache_bytes > free_bytes) {
        fail("the KV cache needs " + std::to_string(cache_bytes / MIB) + " MiB but " +
             std::to_string(free_bytes / MIB) + " MiB of memory is free; lower max_new_tokens");
    }
}

struct SpeechLM::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_context * host_weights = nullptr;
    ggml_context * state = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    HostTables host_tables{OWNER};
    ::tts_cpp::detail::sched_fallback sched;
    SpeechLmConfig config;
    SpeechLayout layout;
    bool tables_on_host = false;
    bool fused = false;
    std::string model_path;
    int n_threads = 1;
    int n_ctx = 0;
    int pos = 0;
    std::vector<LayerCache> caches[BLOCK_COUNT];

    ~Impl() {
        release_weights();
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    ggml_context * context_for(const char * name) const {
        return host_weights && host_table(name) ? host_weights : weights;
    }

    ggml_tensor * find(const std::string & name) const {
        ggml_tensor * tensor = ggml_get_tensor(context_for(name.c_str()), name.c_str());
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    ggml_tensor * weight(int block, int layer, const char * suffix) const {
        return find(layer_name(block, layer, suffix));
    }

    size_t count_metadata_tensors() const {
        size_t count = 0;
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            count++;
        }
        return count;
    }

    void mirror_metadata_tensors() {
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            ggml_tensor * dst = ggml_new_tensor(context_for(ggml_get_name(src)), src->type, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
    }

    ggml_context * new_weight_context() const {
        ggml_context * ctx = ggml_init({(count_metadata_tensors() + TENSOR_SLACK) * ggml_tensor_overhead(),
                nullptr, true});
        if (!ctx) {
            fail("weight context allocation failed");
        }
        return ctx;
    }

    void duplicate_metadata_tensors() {
        weights = new_weight_context();
        if (tables_on_host) {
            host_weights = new_weight_context();
        }
        mirror_metadata_tensors();
    }

    void require_device_weight_types() const {
        if (!::tts_cpp::detail::backend_is_hexagon(backend)) {
            return;
        }
        if (const ggml_tensor * refused = first_tensor_hexagon_cannot_run(weights)) {
            fail(hexagon_weight_refusal(refused, HEXAGON_SPEECH_WEIGHT_TIERS));
        }
    }

    void stream_tensors(::tts_cpp::detail::gguf_stream_reader & reader) {
        for (auto * dst = ggml_get_first_tensor(weights); dst; dst = ggml_get_next_tensor(weights, dst)) {
            if (!reader.to_backend(ggml_get_name(dst), dst)) {
                fail(std::string("failed to load tensor: ") + ggml_get_name(dst));
            }
        }
    }

    void upload_weights(const std::string & path) {
        ::tts_cpp::detail::gguf_stream_reader reader(file, path);
        if (!reader.ok()) {
            fail("cannot reopen GGUF for streaming: " + path);
        }
        if (host_weights) {
            host_tables.place(host_weights, file, path, reader);
        }
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights,
                ::tts_cpp::detail::weight_buffer_type(backend));
        if (!weight_buffer) {
            fail("weight allocation failed");
        }
        stream_tensors(reader);
    }

    void select_backend(const BackendRequest & request) {
        backend = init_backend(request, OWNER);
        tables_on_host = places_tables_on_host(layout.tables, backend);
        fused = fuses_attention(layout.attention, backend);
    }

    void validate_layer(int block, int layer) const {
        const int64_t q_dim = (int64_t) config.n_heads * config.head_dim;
        const int64_t kv_dim = (int64_t) config.n_kv_heads * config.head_dim;
        require_vector(weight(block, layer, "attn_norm.weight"), config.n_embd, 1);
        require_weight(weight(block, layer, "attn_q.weight"), config.n_embd, q_dim);
        require_weight(weight(block, layer, "attn_k.weight"), config.n_embd, kv_dim);
        require_weight(weight(block, layer, "attn_v.weight"), config.n_embd, kv_dim);
        require_weight(weight(block, layer, "attn_output.weight"), q_dim, config.n_embd);
        require_vector(weight(block, layer, "attn_q_norm.weight"), config.head_dim, 1);
        require_vector(weight(block, layer, "attn_k_norm.weight"), config.head_dim, 1);
        require_vector(weight(block, layer, "ffn_norm.weight"), config.n_embd, 1);
        require_weight(weight(block, layer, "ffn_gate.weight"), config.n_embd, config.n_ff);
        require_weight(weight(block, layer, "ffn_up.weight"), config.n_embd, config.n_ff);
        require_weight(weight(block, layer, "ffn_down.weight"), config.n_ff, config.n_embd);
    }

    void validate_block(int block) const {
        for (int layer = 0; layer < block_layers(config, block); ++layer) {
            validate_layer(block, layer);
        }
    }

    void validate_heads() const {
        require_weight(find("text.token_embd.weight"), config.n_embd, config.text_vocab);
        require_weight(find("audio.token_embd.weight"), config.n_embd, config.audio_vocab);
        require_vector(find("text.output_norm.weight"), config.n_embd, 1);
        require_vector(find("audio.output_norm.weight"), config.n_embd, 1);
        require_weight(find("text.output.weight"), config.n_embd, config.text_vocab);
        require_weight(find("audio.output.weight"), config.n_embd, config.audio_vocab);
    }

    void validate_tensors() const {
        validate_heads();
        for (int block = 0; block < BLOCK_COUNT; ++block) {
            validate_block(block);
        }
    }

    void load(const std::string & path, const BackendRequest & request, int threads) {
        if (!within(threads, 1, MAX_THREADS)) {
            fail("threads must be 1.." + std::to_string(MAX_THREADS));
        }
        n_threads = threads;
        model_path = path;
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) {
            fail("cannot read GGUF: " + path);
        }
        const GgufMetadata meta(file, OWNER);
        if (meta.str("general.architecture") != ARCH) {
            fail("unsupported architecture");
        }
        config = read_config(meta);
        validate_geometry(config);
        validate_tokens(config);
        select_backend(request);
        duplicate_metadata_tensors();
        validate_tensors();
        require_device_weight_types();
        upload_weights(path);
    }

    int64_t kv_dim() const {
        return (int64_t) config.head_dim * config.n_kv_heads;
    }

    void release_cache() {
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (state) ggml_free(state);
        state_buffer = nullptr;
        state = nullptr;
    }

    void release_generation() {
        // The scheduler references KV tensors, so destroy it before the cache.
        ::tts_cpp::detail::sched_fallback_free(sched);
        release_cache();
        for (auto & block : caches) block.clear();
        n_ctx = 0;
        pos = 0;
    }

    void release_weights() {
        release_generation();
        host_tables.release();
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (host_weights) ggml_free(host_weights);
        if (weights) ggml_free(weights);
        weight_buffer = nullptr;
        host_weights = nullptr;
        weights = nullptr;
    }

    void ensure_weights() {
        if (weight_buffer) return;
        // Clear any metadata left behind by a failed upload before retrying.
        release_weights();
        try {
            duplicate_metadata_tensors();
            upload_weights(model_path);
        } catch (...) {
            release_weights();
            throw;
        }
    }

    void create_block_cache(int block) {
        caches[block].assign((size_t) block_layers(config, block), {});
        for (LayerCache & cache : caches[block]) {
            cache.k = ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim(), n_ctx);
            cache.v = fused ? ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim(), n_ctx)
                            : ggml_new_tensor_2d(state, GGML_TYPE_F16, n_ctx, kv_dim());
        }
    }

    void create_caches() {
        for (int block = 0; block < BLOCK_COUNT; ++block) {
            create_block_cache(block);
        }
    }

    void require_cache_memory() const {
        if (!::tts_cpp::detail::backend_shares_host_memory(backend)) {
            return;
        }
        size_t free_bytes = 0, total_bytes = 0;
        ::tts_cpp::detail::backend_memory(backend, free_bytes, total_bytes);
        check_cache_memory(ggml_backend_alloc_ctx_tensors_from_buft_size(state,
                ggml_backend_get_default_buffer_type(backend)), free_bytes);
    }

    void begin(int context) {
        if (!within(context, 1, config.n_ctx_train)) {
            fail("context must be 1.." + std::to_string(config.n_ctx_train));
        }
        release_generation();
        ensure_weights();
        try {
            allocate_cache(aligned_context(context));
        } catch (...) {
            release_generation();
            throw;
        }
    }

    void allocate_cache(int context) {
        n_ctx = context;
        const size_t layers = (size_t) config.n_shared_layers + 2 * (size_t) config.n_modality_layers;
        state = ggml_init({(2 * layers + TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!state) {
            fail("KV cache context allocation failed");
        }
        create_caches();
        require_cache_memory();
        state_buffer = ggml_backend_alloc_ctx_tensors(state, backend);
        if (!state_buffer) {
            fail("KV cache allocation failed; the conversation is too long for this device");
        }
        ggml_backend_buffer_clear(state_buffer, 0);
        pos = 0;
    }

    ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * scale) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, cur, config.rms_eps), scale);
    }

    ggml_tensor * rope(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * positions) const {
        return ggml_rope_ext(ctx, cur, positions, nullptr, config.head_dim, GGML_ROPE_TYPE_NEOX,
                0, config.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * key_rows(ggml_context * ctx, const LayerCache & cache, int64_t first, int64_t count) const {
        return ggml_view_2d(ctx, cache.k, kv_dim(), count, cache.k->nb[1], first * cache.k->nb[1]);
    }

    ggml_tensor * value_columns(ggml_context * ctx, const LayerCache & cache, int64_t first, int64_t count) const {
        return ggml_view_2d(ctx, cache.v, count, kv_dim(), cache.v->nb[1], first * ggml_element_size(cache.v));
    }

    ggml_tensor * cached_values(ggml_context * ctx, const LayerCache & cache, int64_t total) const {
        return ggml_view_3d(ctx, cache.v, total, config.head_dim, config.n_kv_heads, cache.v->nb[1],
                cache.v->nb[1] * config.head_dim, 0);
    }

    ggml_tensor * value_rows(ggml_context * ctx, const LayerCache & cache, int64_t first, int64_t count) const {
        return ggml_view_2d(ctx, cache.v, kv_dim(), count, cache.v->nb[1], first * cache.v->nb[1]);
    }

    ggml_tensor * value_write(ggml_context * ctx, const LayerCache & cache, ggml_tensor * v_rows,
                              int64_t tokens) const {
        if (fused) {
            return ggml_cpy(ctx, v_rows, value_rows(ctx, cache, pos, tokens));
        }
        return ggml_cpy(ctx, ggml_transpose(ctx, v_rows), value_columns(ctx, cache, pos, tokens));
    }

    void store_cache(SfxGraph & graph, const LayerCache & cache, ggml_tensor * k, ggml_tensor * v,
                     int64_t tokens) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * k_rows = ggml_reshape_2d(ctx, ggml_cont(ctx, k), kv_dim(), tokens);
        ggml_tensor * v_rows = ggml_reshape_2d(ctx, ggml_cont(ctx, v), kv_dim(), tokens);
        ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx, k_rows, key_rows(ctx, cache, pos, tokens)));
        ggml_build_forward_expand(graph.graph(), value_write(ctx, cache, v_rows, tokens));
    }

    ggml_tensor * cache_heads(ggml_context * ctx, ggml_tensor * rows, int64_t total) const {
        return ggml_reshape_3d(ctx, rows, config.head_dim, config.n_kv_heads, total);
    }

    float attention_scale() const {
        return 1.0f / std::sqrt((float) config.head_dim);
    }

    ggml_tensor * attend_unfused(ggml_context * ctx, const LayerCache & cache, ggml_tensor * q, int64_t tokens,
                                 int64_t total, ggml_tensor * mask) const {
        ggml_tensor * keys = ggml_permute(ctx, cache_heads(ctx, key_rows(ctx, cache, 0, total), total), 0, 2, 1, 3);
        ggml_tensor * scores = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, keys, ggml_permute(ctx, q, 0, 2, 1, 3)),
                mask, attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(ctx, cached_values(ctx, cache, total), scores);
        return ggml_cont_2d(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3),
                (int64_t) config.head_dim * config.n_heads, tokens);
    }

    ggml_tensor * attend_fused(ggml_context * ctx, const LayerCache & cache, ggml_tensor * q, int64_t tokens,
                               int64_t total, ggml_tensor * mask) const {
        ggml_tensor * attended = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3),
                ggml_permute(ctx, cache_heads(ctx, key_rows(ctx, cache, 0, total), total), 0, 2, 1, 3),
                ggml_permute(ctx, cache_heads(ctx, value_rows(ctx, cache, 0, total), total), 0, 2, 1, 3),
                mask, attention_scale(), 0.0f, 0.0f);
        return ggml_reshape_2d(ctx, attended, (int64_t) config.head_dim * config.n_heads, tokens);
    }

    ggml_tensor * attention(SfxGraph & graph, int block, int layer, ggml_tensor * cur, int64_t tokens,
                            ggml_tensor * positions, ggml_tensor * mask) const {
        ggml_context * ctx = graph.ctx();
        const LayerCache & cache = caches[block][(size_t) layer];
        const int64_t total = pos + tokens;
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(block, layer, "attn_q.weight"), cur),
                config.head_dim, config.n_heads, tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(block, layer, "attn_k.weight"), cur),
                config.head_dim, config.n_kv_heads, tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(block, layer, "attn_v.weight"), cur),
                config.head_dim, config.n_kv_heads, tokens);
        q = rope(ctx, rms_norm(ctx, q, weight(block, layer, "attn_q_norm.weight")), positions);
        k = rope(ctx, rms_norm(ctx, k, weight(block, layer, "attn_k_norm.weight")), positions);
        store_cache(graph, cache, k, v, tokens);
        ggml_tensor * attended = fused ? attend_fused(ctx, cache, q, tokens, total, mask)
                                       : attend_unfused(ctx, cache, q, tokens, total, mask);
        return ggml_mul_mat(ctx, weight(block, layer, "attn_output.weight"), attended);
    }

    ggml_tensor * feed_forward(ggml_context * ctx, int block, int layer, ggml_tensor * cur) const {
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, weight(block, layer, "ffn_gate.weight"), cur));
        ggml_tensor * up = ggml_mul_mat(ctx, weight(block, layer, "ffn_up.weight"), cur);
        ggml_tensor * shrunk = ggml_scale(ctx, ggml_mul(ctx, gate, up), 1.0f / FFN_DOWN_ACCUMULATION_SCALE);
        return ggml_scale(ctx, ggml_mul_mat(ctx, weight(block, layer, "ffn_down.weight"), shrunk),
                FFN_DOWN_ACCUMULATION_SCALE);
    }

    ggml_tensor * layer_forward(SfxGraph & graph, int block, int layer, ggml_tensor * cur, int64_t tokens,
                                ggml_tensor * positions, ggml_tensor * mask) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * attended = attention(graph, block, layer,
                rms_norm(ctx, cur, weight(block, layer, "attn_norm.weight")), tokens, positions, mask);
        cur = ggml_add(ctx, cur, attended);
        ggml_tensor * fed = feed_forward(ctx, block, layer, rms_norm(ctx, cur, weight(block, layer, "ffn_norm.weight")));
        return ggml_add(ctx, cur, fed);
    }

    ggml_tensor * block_forward(SfxGraph & graph, int block, ggml_tensor * cur, int64_t tokens,
                                ggml_tensor * positions, ggml_tensor * mask) const {
        for (int layer = 0; layer < block_layers(config, block); ++layer) {
            cur = layer_forward(graph, block, layer, cur, tokens, positions, mask);
        }
        return cur;
    }

    ggml_tensor * branch_logits(SfxGraph & graph, int block, const char * head, ggml_tensor * shared,
                                int64_t tokens, ggml_tensor * positions, ggml_tensor * mask) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * cur = block_forward(graph, block, shared, tokens, positions, mask);
        ggml_tensor * last = ggml_view_2d(ctx, cur, config.n_embd, 1, cur->nb[1], (size_t) (tokens - 1) * cur->nb[1]);
        last = rms_norm(ctx, last, find(std::string(head) + ".output_norm.weight"));
        return ggml_mul_mat(ctx, find(std::string(head) + ".output.weight"), last);
    }

    ggml_tensor * embed(SfxGraph & graph, ggml_tensor * text_ids, ggml_tensor * audio_ids, ggml_tensor * text_select,
                        ggml_tensor * audio_select) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * text = ggml_get_rows(ctx, find("text.token_embd.weight"), text_ids);
        ggml_tensor * audio = ggml_get_rows(ctx, find("audio.token_embd.weight"), audio_ids);
        return ggml_add(ctx, ggml_mul(ctx, text, text_select), ggml_mul(ctx, audio, audio_select));
    }

    void place_row(BatchInputs & inputs, size_t slot, const SpeechRow & row) const {
        if (!in_vocab(row.text, config.text_vocab) || !in_vocab(row.audio, config.audio_vocab)) {
            fail("token outside its vocabulary");
        }
        const bool audio_row = row.text == config.tokens.text_placeholder;
        inputs.text_ids[slot] = audio_row ? 0 : row.text;
        inputs.audio_ids[slot] = row.audio;
        inputs.text_select[slot] = audio_row ? 0.0f : 1.0f;
        inputs.audio_select[slot] = audio_row ? 1.0f : 0.0f;
    }

    BatchInputs plan_batch(const SpeechRow * rows, size_t count) const {
        BatchInputs inputs{std::vector<int32_t>(count), std::vector<int32_t>(count),
                           std::vector<float>(count), std::vector<float>(count)};
        for (size_t slot = 0; slot < count; ++slot) {
            place_row(inputs, slot, rows[slot]);
        }
        return inputs;
    }

    std::vector<int32_t> positions(int64_t tokens) const {
        std::vector<int32_t> values((size_t) tokens);
        for (int64_t i = 0; i < tokens; ++i) {
            values[(size_t) i] = (int32_t) (pos + i);
        }
        return values;
    }

    std::vector<float> causal_mask(int64_t tokens) const {
        const int64_t total = pos + tokens;
        std::vector<float> mask((size_t) (tokens * total), -std::numeric_limits<float>::infinity());
        for (int64_t query = 0; query < tokens; ++query) {
            const auto row = mask.begin() + (std::ptrdiff_t) (query * total);
            std::fill(row, row + (std::ptrdiff_t) (pos + query + 1), 0.0f);
        }
        return mask;
    }

    void allocate(SfxGraph & graph) {
        if (!::tts_cpp::detail::sched_fallback_ensure(sched, backend, GRAPH_NODES,
                {weight_buffer, host_tables.buffer()})) {
            fail("scheduler initialization failed");
        }
        if (!::tts_cpp::detail::sched_fallback_alloc(sched, graph.graph())) {
            fail("graph allocation failed");
        }
    }

    void compute(SfxGraph & graph) {
        if (::tts_cpp::detail::sched_fallback_compute(sched, backend, graph.graph(), n_threads) != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }
    }

    static std::vector<float> read_logits(ggml_tensor * tensor) {
        std::vector<float> values;
        if (tensor != nullptr) {
            values.resize((size_t) ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
        return values;
    }

    static ggml_tensor * request_output(SfxGraph & graph, ggml_tensor * logits, bool wanted) {
        if (!wanted) {
            return nullptr;
        }
        ggml_set_output(logits);
        ggml_build_forward_expand(graph.graph(), logits);
        return logits;
    }

    SpeechLogits forward(const BatchInputs & inputs, SpeechHeads heads) {
        const int64_t tokens = (int64_t) inputs.text_ids.size();
        if (state == nullptr) {
            fail("begin() must be called before running the model");
        }
        if (pos + tokens > n_ctx) {
            fail("context overflow");
        }
        SfxGraph graph(GRAPH_NODES);
        ggml_tensor * text_ids = graph.input_i32(tokens);
        ggml_tensor * audio_ids = graph.input_i32(tokens);
        ggml_tensor * text_select = graph.input_f32(1, tokens);
        ggml_tensor * audio_select = graph.input_f32(1, tokens);
        ggml_tensor * positions_input = graph.input_i32(tokens);
        ggml_tensor * mask = tokens > 1 ? graph.input_f32(pos + tokens, tokens) : nullptr;
        ggml_tensor * attention_mask = mask && fused ? ggml_cast(graph.ctx(), mask, GGML_TYPE_F16) : mask;
        ggml_tensor * shared = block_forward(graph, SHARED_BLOCK, embed(graph, text_ids, audio_ids, text_select,
                audio_select), tokens, positions_input, attention_mask);
        ggml_tensor * text_logits = request_output(graph, branch_logits(graph, TEXT_BLOCK, "text", shared, tokens,
                positions_input, attention_mask), heads.text);
        ggml_tensor * audio_logits = request_output(graph, branch_logits(graph, AUDIO_BLOCK, "audio", shared, tokens,
                positions_input, attention_mask), heads.audio);

        allocate(graph);
        set_input(text_ids, inputs.text_ids);
        set_input(audio_ids, inputs.audio_ids);
        set_input(text_select, inputs.text_select);
        set_input(audio_select, inputs.audio_select);
        set_input(positions_input, positions(tokens));
        if (mask) {
            set_input(mask, causal_mask(tokens));
        }
        compute(graph);
        pos += (int) tokens;
        return SpeechLogits{read_logits(text_logits), read_logits(audio_logits)};
    }

    SpeechLogits prefill(const std::vector<SpeechRow> & rows, int batch_tokens, const SpeechStop & stop) {
        if (rows.empty() || batch_tokens < 1) {
            fail("prefill needs at least one row and a positive batch size");
        }
        SpeechLogits logits;
        const size_t batch = (size_t) batch_tokens;
        for (size_t first = 0; first < rows.size() && !(stop && stop()); first += batch) {
            const size_t count = std::min(batch, rows.size() - first);
            const bool last = first + count == rows.size();
            logits = forward(plan_batch(rows.data() + first, count), {last, last});
        }
        return logits;
    }
};

SpeechLM::SpeechLM(const std::string & path, const BackendRequest & backend, int n_threads,
                   const SpeechLayout & layout) : impl_(new Impl) {
    impl_->layout = layout;
    impl_->load(path, backend, n_threads);
}

SpeechLM::~SpeechLM() = default;

const SpeechLmConfig & SpeechLM::config() const { return impl_->config; }
const char * SpeechLM::backend_name() const { return ggml_backend_name(impl_->backend); }
bool SpeechLM::runs_on_cpu() const { return ::tts_cpp::detail::backend_is_cpu(impl_->backend); }
bool SpeechLM::tables_on_host() const { return impl_->tables_on_host; }
bool SpeechLM::fused_attention() const { return impl_->fused; }
int SpeechLM::position() const { return impl_->pos; }
int SpeechLM::context() const { return impl_->n_ctx; }

std::vector<std::string> SpeechLM::tokenizer_tokens() const {
    return GgufMetadata(impl_->file, OWNER).str_array("tokenizer.ggml.tokens");
}

std::vector<std::string> SpeechLM::tokenizer_merges() const {
    return GgufMetadata(impl_->file, OWNER).str_array("tokenizer.ggml.merges");
}

std::vector<int32_t> SpeechLM::tokenizer_types() const {
    return GgufMetadata(impl_->file, OWNER).int_array("tokenizer.ggml.token_type", (size_t) impl_->config.text_vocab);
}

void SpeechLM::begin(int n_ctx) { impl_->begin(n_ctx); }
void SpeechLM::release_generation() { impl_->release_generation(); }
void SpeechLM::release_weights() { impl_->release_weights(); }

SpeechLogits SpeechLM::prefill(const std::vector<SpeechRow> & rows, int batch_tokens, const SpeechStop & stop) {
    return impl_->prefill(rows, batch_tokens, stop);
}

SpeechLogits SpeechLM::step(const SpeechRow & row, SpeechHeads heads) {
    return impl_->forward(impl_->plan_batch(&row, 1), heads);
}

} // namespace tts_cpp::moss::detail
