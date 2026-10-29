#include "moss/delay_lm.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "sched_dispatch.h"
#include "moss/fit_measure.h"
#include "moss/host_tables.h"

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

constexpr const char * ARCH = "moss-tts-delay";
constexpr int GRAPH_NODES = 8192;
constexpr int MIN_CONTEXT = 32;
constexpr int MAX_CONTEXT = 32768;
constexpr int MAX_LAYERS = 512;
constexpr int MAX_CHANNELS = 64;
constexpr int MAX_EMBEDDING = 1 << 16;
constexpr int MAX_FEED_FORWARD = 1 << 18;
constexpr int MAX_HEADS = 1024;
constexpr int MAX_HEAD_DIM = 1024;
constexpr float FFN_DOWN_ACCUMULATION_SCALE = 64.0f;
constexpr const char * TEXT_TABLE = "token_embd.weight";
constexpr const char * AUDIO_TABLE_PREFIX = "token_embd_audio.";
constexpr const char * TEXT_HEAD = "output.weight";
constexpr const char * SLOT_HEAD = "output_slots.weight";
constexpr const char * OWNER = "moss-delay";
constexpr size_t SPARE_TENSORS = 8;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss delay LM: " + message);
}

bool within(int value, int low, int high) {
    return value >= low && value <= high;
}

bool starts_with(const char * text, const char * prefix) {
    return std::strncmp(text, prefix, std::strlen(prefix)) == 0;
}

bool host_table(const char * name) {
    return std::strcmp(name, TEXT_TABLE) == 0 || std::strcmp(name, TEXT_HEAD) == 0 ||
           starts_with(name, AUDIO_TABLE_PREFIX);
}

struct Layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * wq          = nullptr;
    ggml_tensor * wk          = nullptr;
    ggml_tensor * wv          = nullptr;
    ggml_tensor * wo          = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate    = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
};

struct GraphInputI32 {
    ggml_tensor * tensor;
    std::vector<int32_t> data;
};

struct GraphInputF32 {
    ggml_tensor * tensor;
    std::vector<float> data;
};

} // namespace

struct DelayLM::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_context * state = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    ggml_context * host_weights = nullptr;
    HostTables host_tables{OWNER};
    ::tts_cpp::detail::sched_fallback sched;

    DelayConfig config;
    int n_threads = 1;
    int n_ctx = 0;
    int pos = 0;
    bool measure_only = false;
    DelayLayout layout;
    bool tables_on_host = false;
    bool fused = false;
    uint64_t graph_host_bytes = 0;
    ::tts_cpp::detail::fit_graph_price graph_price;

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output_head = nullptr;
    ggml_tensor * slot_head = nullptr;
    std::vector<int32_t> slot_ids;
    std::vector<ggml_tensor *> tok_embd_audio;
    std::vector<ggml_tensor *> output_audio;
    std::vector<Layer> layers;
    std::vector<ggml_tensor *> cache_k;
    std::vector<ggml_tensor *> cache_v;

    ggml_context * graph_ctx = nullptr;
    std::vector<GraphInputI32> inputs_i32;
    std::vector<GraphInputF32> inputs_f32;

    ~Impl() {
        ::tts_cpp::detail::sched_fallback_free(sched);
        if (graph_ctx) ggml_free(graph_ctx);
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (state) ggml_free(state);
        if (host_weights) ggml_free(host_weights);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    int64_t find_key(const std::string & name) const {
        return gguf_find_key(file, name.c_str());
    }

    uint32_t meta_u32(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            fail("missing metadata: " + name);
        }
        switch (gguf_get_kv_type(file, key)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(file, key);
            case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(file, key);
            default:               fail("unsupported metadata type: " + name);
        }
    }

    uint32_t meta_u32_or(const std::string & name, uint32_t fallback) const {
        return find_key(name) >= 0 ? meta_u32(name) : fallback;
    }

    float meta_f32_or(const std::string & name, float fallback) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            return fallback;
        }
        if (gguf_get_kv_type(file, key) != GGUF_TYPE_FLOAT32) {
            fail("unsupported metadata type: " + name);
        }
        return gguf_get_val_f32(file, key);
    }

    std::string meta_str(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_STRING) {
            fail("missing string metadata: " + name);
        }
        return gguf_get_val_str(file, key);
    }

    std::vector<std::string> meta_str_array(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(file, key) != GGUF_TYPE_STRING) {
            fail("missing string array metadata: " + name);
        }
        const size_t count = gguf_get_arr_n(file, key);
        std::vector<std::string> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = gguf_get_arr_str(file, key, i);
        }
        return values;
    }

    ggml_tensor * find_tensor(const std::string & name) const {
        if (!weights) {
            return ggml_get_tensor(metadata, name.c_str());
        }
        ggml_tensor * tensor = ggml_get_tensor(weights, name.c_str());
        return tensor || !host_weights ? tensor : ggml_get_tensor(host_weights, name.c_str());
    }

    ggml_tensor * require_tensor(const std::string & name) const {
        ggml_tensor * tensor = find_tensor(name);
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    void read_config() {
        const std::string arch(ARCH);
        config.n_layers    = (int) meta_u32(arch + ".block_count");
        config.n_embd      = (int) meta_u32(arch + ".embedding_length");
        config.n_ff        = (int) meta_u32(arch + ".feed_forward_length");
        config.n_heads     = (int) meta_u32(arch + ".attention.head_count");
        config.n_kv_heads  = (int) meta_u32(arch + ".attention.head_count_kv");
        if (!within(config.n_embd, 1, MAX_EMBEDDING) || !within(config.n_heads, 1, MAX_HEADS)) {
            fail("invalid model geometry");
        }
        config.head_dim    = (int) meta_u32_or(arch + ".attention.key_length",
                (uint32_t) (config.n_embd / config.n_heads));
        config.n_ctx_train = (int) meta_u32(arch + ".context_length");
        config.rope_base   = meta_f32_or(arch + ".rope.freq_base", 10000.0f);
        config.rms_eps     = meta_f32_or(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        config.n_vq        = (int) meta_u32(arch + ".n_vq");
        config.audio_vocab = (int) meta_u32(arch + ".audio_vocab_size");
        config.audio_pad_code = (int) meta_u32(arch + ".audio_pad_code");
        config.audio_start_token_id = (int) meta_u32(arch + ".audio_start_token_id");
        config.audio_end_token_id   = (int) meta_u32(arch + ".audio_end_token_id");
        config.audio_user_slot_token_id = (int) meta_u32(arch + ".audio_user_slot_token_id");
        config.audio_assistant_gen_slot_token_id = (int) meta_u32(arch + ".audio_assistant_gen_slot_token_id");
        config.audio_assistant_delay_slot_token_id = (int) meta_u32(arch + ".audio_assistant_delay_slot_token_id");
        config.sampling_rate = (int) meta_u32_or(arch + ".sampling_rate", 24000);
        validate_geometry();
    }

    void validate_geometry() const {
        if (!within(config.n_layers, 1, MAX_LAYERS) || !within(config.n_kv_heads, 1, MAX_HEADS) ||
            !within(config.n_vq, 1, MAX_CHANNELS) || !within(config.head_dim, 1, MAX_HEAD_DIM) ||
            !within(config.n_ff, 1, MAX_FEED_FORWARD) || config.n_heads % config.n_kv_heads != 0) {
            fail("invalid model geometry");
        }
    }

    std::string layer_name(int il, const char * suffix) const {
        return "blk." + std::to_string(il) + "." + suffix;
    }

    void map_tensors() {
        tok_embd    = require_tensor(TEXT_TABLE);
        output_norm = require_tensor("output_norm.weight");
        output_head = require_tensor(TEXT_HEAD);
        config.text_vocab = (int) output_head->ne[1];
        tok_embd_audio.resize(config.n_vq);
        output_audio.resize(config.n_vq);
        for (int i = 0; i < config.n_vq; ++i) {
            tok_embd_audio[i] = require_tensor(AUDIO_TABLE_PREFIX + std::to_string(i) + ".weight");
            output_audio[i]   = require_tensor("output_audio." + std::to_string(i) + ".weight");
        }
        config.audio_vocab = (int) output_audio[0]->ne[1];
        layers.resize(config.n_layers);
        for (int il = 0; il < config.n_layers; ++il) {
            Layer & layer = layers[il];
            layer.attn_norm   = require_tensor(layer_name(il, "attn_norm.weight"));
            layer.wq          = require_tensor(layer_name(il, "attn_q.weight"));
            layer.wk          = require_tensor(layer_name(il, "attn_k.weight"));
            layer.wv          = require_tensor(layer_name(il, "attn_v.weight"));
            layer.wo          = require_tensor(layer_name(il, "attn_output.weight"));
            layer.attn_q_norm = require_tensor(layer_name(il, "attn_q_norm.weight"));
            layer.attn_k_norm = require_tensor(layer_name(il, "attn_k_norm.weight"));
            layer.ffn_norm    = require_tensor(layer_name(il, "ffn_norm.weight"));
            layer.ffn_gate    = require_tensor(layer_name(il, "ffn_gate.weight"));
            layer.ffn_up      = require_tensor(layer_name(il, "ffn_up.weight"));
            layer.ffn_down    = require_tensor(layer_name(il, "ffn_down.weight"));
        }
    }

    void require_shape(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) const {
        const int64_t rows = ggml_n_dims(tensor) > 1 ? tensor->ne[1] : 1;
        if (ggml_n_dims(tensor) > 2 || tensor->ne[0] != ne0 || rows != ne1) {
            fail(std::string(ggml_get_name(tensor)) + " has unexpected dimensions");
        }
    }

    void validate_vocabularies() const {
        if (tok_embd->ne[1] != output_head->ne[1]) {
            fail("token_embd and output disagree on the text vocabulary");
        }
        for (int i = 0; i < config.n_vq; ++i) {
            if (tok_embd_audio[i]->ne[1] != config.audio_vocab ||
                output_audio[i]->ne[1] != config.audio_vocab) {
                fail("audio channel " + std::to_string(i) + " disagrees on the audio vocabulary");
            }
        }
    }

    void validate_head_shapes() const {
        require_shape(tok_embd, config.n_embd, config.text_vocab);
        require_shape(output_norm, config.n_embd, 1);
        require_shape(output_head, config.n_embd, config.text_vocab);
        for (int i = 0; i < config.n_vq; ++i) {
            require_shape(tok_embd_audio[i], config.n_embd, config.audio_vocab);
            require_shape(output_audio[i], config.n_embd, config.audio_vocab);
        }
    }

    void validate_layer_shapes(const Layer & layer) const {
        const int64_t q_dim = (int64_t) config.n_heads * config.head_dim;
        const int64_t kv_dim = (int64_t) config.n_kv_heads * config.head_dim;
        require_shape(layer.attn_norm, config.n_embd, 1);
        require_shape(layer.wq, config.n_embd, q_dim);
        require_shape(layer.wk, config.n_embd, kv_dim);
        require_shape(layer.wv, config.n_embd, kv_dim);
        require_shape(layer.wo, q_dim, config.n_embd);
        require_shape(layer.attn_q_norm, config.head_dim, 1);
        require_shape(layer.attn_k_norm, config.head_dim, 1);
        require_shape(layer.ffn_norm, config.n_embd, 1);
        require_shape(layer.ffn_gate, config.n_embd, config.n_ff);
        require_shape(layer.ffn_up, config.n_embd, config.n_ff);
        require_shape(layer.ffn_down, config.n_ff, config.n_embd);
    }

    void validate_tensor_shapes() const {
        validate_vocabularies();
        validate_head_shapes();
        for (const Layer & layer : layers) {
            validate_layer_shapes(layer);
        }
    }

    void require_text_token_id(int id, const char * name) const {
        if (id < 0 || id >= config.text_vocab || (int64_t) id >= tok_embd->ne[1]) {
            fail(std::string(name) + " is outside the text vocabulary");
        }
    }

    void validate_token_ids() const {
        require_text_token_id(config.audio_start_token_id, "audio_start_token_id");
        require_text_token_id(config.audio_end_token_id, "audio_end_token_id");
        require_text_token_id(config.audio_user_slot_token_id, "audio_user_slot_token_id");
        require_text_token_id(config.audio_assistant_gen_slot_token_id,
                "audio_assistant_gen_slot_token_id");
        require_text_token_id(config.audio_assistant_delay_slot_token_id,
                "audio_assistant_delay_slot_token_id");
        if (config.audio_pad_code < 0 || config.audio_pad_code >= config.audio_vocab) {
            fail("audio_pad_code is outside the audio vocabulary");
        }
    }

    void allocate_cache() {
        state = ggml_init({(size_t) (2 * config.n_layers + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!state) {
            fail("state context allocation failed");
        }
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        cache_k.resize(config.n_layers);
        cache_v.resize(config.n_layers);
        for (int il = 0; il < config.n_layers; ++il) {
            cache_k[il] = ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim, n_ctx);
            cache_v[il] = fused ? ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim, n_ctx)
                                : ggml_new_tensor_2d(state, GGML_TYPE_F16, n_ctx, kv_dim);
        }
        if (measure_only) {
            mark_fit_tensors(state);
            return;
        }
        state_buffer = ggml_backend_alloc_ctx_tensors(state, backend);
        if (!state_buffer) {
            fail("KV cache allocation failed");
        }
        ggml_backend_buffer_clear(state_buffer, 0);
    }

    static size_t count_tensors(ggml_context * ctx) {
        size_t count = 0;
        for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            count++;
        }
        return count;
    }

    static ggml_context * new_weight_context(size_t count) {
        ggml_context * ctx = ggml_init({(count + SPARE_TENSORS) * ggml_tensor_overhead(), nullptr, true});
        if (!ctx) {
            fail("weight context allocation failed");
        }
        return ctx;
    }

    ggml_context * context_for(const char * name) const {
        return host_weights && host_table(name) ? host_weights : weights;
    }

    void duplicate_metadata_tensors() {
        const size_t count = count_tensors(metadata);
        weights = new_weight_context(count);
        if (tables_on_host) {
            host_weights = new_weight_context(count);
        }
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            auto * dst = ggml_new_tensor(context_for(ggml_get_name(src)), src->type, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
    }

    void add_slot_head() {
        if (!tables_on_host && !::tts_cpp::detail::backend_is_metal(backend)) {
            return;
        }
        slot_ids = {config.audio_assistant_gen_slot_token_id, config.audio_assistant_delay_slot_token_id};
        if (!tables_on_host && slot_ids[1] > slot_ids[0]) {
            return;
        }
        const ggml_type type = tables_on_host ? GGML_TYPE_F32 : output_head->type;
        slot_head = ggml_new_tensor_2d(weights, type, config.n_embd, (int64_t) slot_ids.size());
        ggml_set_name(slot_head, SLOT_HEAD);
    }

    void require_device_weight_types() const {
        if (!::tts_cpp::detail::backend_is_hexagon(backend)) {
            return;
        }
        if (const ggml_tensor * refused = first_tensor_hexagon_cannot_run(weights)) {
            fail(hexagon_weight_refusal(refused, HEXAGON_TTS_WEIGHT_TIERS));
        }
    }

    void stream_tensors(::tts_cpp::detail::gguf_stream_reader & reader, ggml_context * ctx) {
        for (auto * dst = ggml_get_first_tensor(ctx); dst; dst = ggml_get_next_tensor(ctx, dst)) {
            if (dst != slot_head && !reader.to_backend(ggml_get_name(dst), dst)) {
                fail(std::string("failed to load tensor: ") + ggml_get_name(dst));
            }
        }
    }

    void read_head_row(int32_t id, float * out) const {
        const char * row = (const char *) output_head->data + (size_t) id * output_head->nb[1];
        if (output_head->type == GGML_TYPE_F32) {
            std::memcpy(out, row, (size_t) config.n_embd * sizeof(float));
            return;
        }
        const auto * traits = ggml_get_type_traits(output_head->type);
        if (!traits->to_float) {
            fail(std::string("cannot read text head type ") + ggml_type_name(output_head->type));
        }
        traits->to_float(row, out, config.n_embd);
    }

    std::vector<float> slot_rows() const {
        std::vector<float> rows((size_t) config.n_embd * slot_ids.size());
        for (size_t i = 0; i < slot_ids.size(); ++i) {
            read_head_row(slot_ids[i], rows.data() + i * (size_t) config.n_embd);
        }
        return rows;
    }

    std::vector<uint8_t> native_slot_rows() const {
        const size_t row_bytes = ggml_row_size(slot_head->type, config.n_embd);
        std::vector<uint8_t> rows(row_bytes * slot_ids.size());
        for (size_t i = 0; i < slot_ids.size(); ++i) {
            ggml_backend_tensor_get(output_head, rows.data() + i * row_bytes,
                    (size_t) slot_ids[i] * output_head->nb[1], row_bytes);
        }
        return rows;
    }

    void upload_slot_head() {
        if (!tables_on_host) {
            const auto rows = native_slot_rows();
            ggml_backend_tensor_set(slot_head, rows.data(), 0, rows.size());
            return;
        }
        const std::vector<float> rows = slot_rows();
        ggml_backend_tensor_set(slot_head, rows.data(), 0, rows.size() * sizeof(float));
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
        stream_tensors(reader, weights);
        if (slot_head) {
            upload_slot_head();
        }
    }

    void select_backend(const BackendRequest & request) {
        backend = ::tts_cpp::moss::detail::init_backend(request, OWNER);
        tables_on_host = places_tables_on_host(layout.tables, backend);
        fused = fuses_attention(layout.attention, backend);
    }

    void load(const std::string & path, const BackendRequest & request, int threads, int context) {
        n_threads = threads;
        n_ctx = context;
        if (threads < 1 || threads > 1024) {
            fail("threads must be 1..1024");
        }
        if (n_ctx < MIN_CONTEXT || n_ctx > MAX_CONTEXT) {
            fail("context must be " + std::to_string(MIN_CONTEXT) + ".." + std::to_string(MAX_CONTEXT));
        }
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) {
            fail("cannot read GGUF: " + path);
        }
        if (meta_str("general.architecture") != ARCH) {
            fail("unsupported architecture");
        }
        read_config();
        select_backend(request);
        duplicate_metadata_tensors();
        map_tensors();
        validate_tensor_shapes();
        validate_token_ids();
        add_slot_head();
        require_device_weight_types();
        allocate_cache();
        if (measure_only) {
            mark_fit_tensors(weights);
            if (host_weights) mark_fit_tensors(host_weights);
            return;
        }
        upload_weights(path);
    }

    void begin_graph() {
        if (graph_ctx) {
            ggml_free(graph_ctx);
            graph_ctx = nullptr;
        }
        graph_host_bytes = 0;
        inputs_i32.clear();
        inputs_f32.clear();
        const size_t buffer = (size_t) GRAPH_NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(GRAPH_NODES, false);
        graph_ctx = ggml_init({buffer, nullptr, true});
        if (!graph_ctx) {
            fail("graph context allocation failed");
        }
    }

    ggml_tensor * input_i32(std::vector<int32_t> data) {
        ggml_tensor * tensor = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, (int64_t) data.size());
        ggml_set_input(tensor);
        graph_host_bytes = fitutil::sat_add(graph_host_bytes, ggml_nbytes(tensor));
        inputs_i32.push_back({tensor, std::move(data)});
        return tensor;
    }

    ggml_tensor * input_f32(std::vector<float> data, int64_t ne0, int64_t ne1) {
        ggml_tensor * tensor = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_input(tensor);
        graph_host_bytes = fitutil::sat_add(graph_host_bytes, ggml_nbytes(tensor));
        inputs_f32.push_back({tensor, std::move(data)});
        return tensor;
    }

    void set_graph_inputs() {
        for (auto & input : inputs_i32) {
            ggml_backend_tensor_set(input.tensor, input.data.data(), 0, input.data.size() * sizeof(int32_t));
        }
        for (auto & input : inputs_f32) {
            ggml_backend_tensor_set(input.tensor, input.data.data(), 0, input.data.size() * sizeof(float));
        }
    }

    ggml_tensor * rms_norm(ggml_tensor * cur, ggml_tensor * weight) {
        return ggml_mul(graph_ctx, ggml_rms_norm(graph_ctx, cur, config.rms_eps), weight);
    }

    ggml_tensor * rope(ggml_tensor * cur, ggml_tensor * positions) {
        return ggml_rope_ext(graph_ctx, cur, positions, nullptr, config.head_dim,
                GGML_ROPE_TYPE_NEOX, config.n_ctx_train, config.rope_base,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * cache_view(ggml_tensor * cache, int64_t count) {
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        return ggml_view_2d(graph_ctx, cache, kv_dim, count, cache->nb[1], 0);
    }

    ggml_tensor * cache_write_view(ggml_tensor * cache, int64_t first, int64_t count) {
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        return ggml_view_2d(graph_ctx, cache, kv_dim, count, cache->nb[1], first * cache->nb[1]);
    }

    ggml_tensor * cache_write_view_transposed(ggml_tensor * cache, int64_t first, int64_t count) {
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        return ggml_view_2d(graph_ctx, cache, count, kv_dim, cache->nb[1],
                first * ggml_element_size(cache));
    }

    ggml_tensor * value_prefix_view(ggml_tensor * cache, int64_t count) {
        const size_t row = cache->nb[1];
        return ggml_view_3d(graph_ctx, cache, count, config.head_dim, config.n_kv_heads,
                row, row * config.head_dim, 0);
    }

    float attention_scale() const {
        return 1.0f / std::sqrt((float) config.head_dim);
    }

    ggml_tensor * cache_heads(ggml_tensor * cache, int64_t total) {
        return ggml_reshape_3d(graph_ctx, cache_view(cache, total), config.head_dim, config.n_kv_heads, total);
    }

    void write_cache(ggml_cgraph * graph, int il, ggml_tensor * k, ggml_tensor * v, int64_t n_tokens) {
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        ggml_tensor * k_rows = ggml_reshape_2d(graph_ctx, ggml_cont(graph_ctx, k), kv_dim, n_tokens);
        ggml_tensor * v_rows = ggml_reshape_2d(graph_ctx, ggml_cont(graph_ctx, v), kv_dim, n_tokens);
        ggml_build_forward_expand(graph, ggml_cpy(graph_ctx, k_rows, cache_write_view(cache_k[il], pos, n_tokens)));
        ggml_tensor * value_write = fused
                ? ggml_cpy(graph_ctx, v_rows, cache_write_view(cache_v[il], pos, n_tokens))
                : ggml_cpy(graph_ctx, ggml_transpose(graph_ctx, v_rows),
                        cache_write_view_transposed(cache_v[il], pos, n_tokens));
        ggml_build_forward_expand(graph, value_write);
    }

    ggml_tensor * attend_unfused(ggml_tensor * q, int il, int64_t n_tokens, int64_t total, ggml_tensor * mask) {
        ggml_tensor * keys = ggml_permute(graph_ctx, cache_heads(cache_k[il], total), 0, 2, 1, 3);
        ggml_tensor * scores = ggml_mul_mat(graph_ctx, keys, ggml_permute(graph_ctx, q, 0, 2, 1, 3));
        scores = ggml_soft_max_ext(graph_ctx, scores, mask, attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(graph_ctx, value_prefix_view(cache_v[il], total), scores);
        attended = ggml_permute(graph_ctx, attended, 0, 2, 1, 3);
        return ggml_cont_2d(graph_ctx, attended, (int64_t) config.head_dim * config.n_heads, n_tokens);
    }

    ggml_tensor * attend_fused(ggml_tensor * q, int il, int64_t n_tokens, int64_t total, ggml_tensor * mask) {
        ggml_tensor * attended = ggml_flash_attn_ext(graph_ctx, ggml_permute(graph_ctx, q, 0, 2, 1, 3),
                ggml_permute(graph_ctx, cache_heads(cache_k[il], total), 0, 2, 1, 3),
                ggml_permute(graph_ctx, cache_heads(cache_v[il], total), 0, 2, 1, 3),
                mask, attention_scale(), 0.0f, 0.0f);
        return ggml_reshape_2d(graph_ctx, attended, (int64_t) config.head_dim * config.n_heads, n_tokens);
    }

    ggml_tensor * attention(ggml_cgraph * graph, const Layer & layer, int il, ggml_tensor * cur,
                            int64_t n_tokens, int64_t total, ggml_tensor * positions, ggml_tensor * mask) {
        ggml_tensor * q = ggml_mul_mat(graph_ctx, layer.wq, cur);
        ggml_tensor * k = ggml_mul_mat(graph_ctx, layer.wk, cur);
        ggml_tensor * v = ggml_mul_mat(graph_ctx, layer.wv, cur);
        q = ggml_reshape_3d(graph_ctx, q, config.head_dim, config.n_heads, n_tokens);
        k = ggml_reshape_3d(graph_ctx, k, config.head_dim, config.n_kv_heads, n_tokens);
        v = ggml_reshape_3d(graph_ctx, v, config.head_dim, config.n_kv_heads, n_tokens);
        q = rope(rms_norm(q, layer.attn_q_norm), positions);
        k = rope(rms_norm(k, layer.attn_k_norm), positions);
        write_cache(graph, il, k, v, n_tokens);
        ggml_tensor * attended = fused ? attend_fused(q, il, n_tokens, total, mask)
                                       : attend_unfused(q, il, n_tokens, total, mask);
        return ggml_mul_mat(graph_ctx, layer.wo, attended);
    }

    ggml_tensor * scaled_down_projection(ggml_tensor * weight, ggml_tensor * hidden) {
        ggml_tensor * shrunk = ggml_scale(graph_ctx, hidden, 1.0f / FFN_DOWN_ACCUMULATION_SCALE);
        return ggml_scale(graph_ctx, ggml_mul_mat(graph_ctx, weight, shrunk),
                FFN_DOWN_ACCUMULATION_SCALE);
    }

    ggml_tensor * feed_forward(const Layer & layer, ggml_tensor * cur) {
        ggml_tensor * gate = ggml_silu(graph_ctx, ggml_mul_mat(graph_ctx, layer.ffn_gate, cur));
        ggml_tensor * up = ggml_mul_mat(graph_ctx, layer.ffn_up, cur);
        return scaled_down_projection(layer.ffn_down, ggml_mul(graph_ctx, gate, up));
    }

    std::vector<float> attention_mask(int64_t n_tokens, int64_t total) const {
        if (measure_only) return {};
        std::vector<float> mask(n_tokens * total, -std::numeric_limits<float>::infinity());
        for (int64_t iq = 0; iq < n_tokens; ++iq) {
            for (int64_t ik = 0; ik <= pos + iq; ++ik) {
                mask[iq * total + ik] = 0.0f;
            }
        }
        return mask;
    }

    ggml_tensor * embed_rows(const std::vector<DelayRow> & rows) {
        const int64_t n_tokens = (int64_t) rows.size();
        std::vector<int32_t> text_ids(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            text_ids[i] = rows[i].text;
        }
        ggml_tensor * cur = ggml_get_rows(graph_ctx, tok_embd, input_i32(std::move(text_ids)));
        for (int channel = 0; channel < config.n_vq; ++channel) {
            std::vector<int32_t> channel_ids(n_tokens);
            for (int64_t i = 0; i < n_tokens; ++i) {
                channel_ids[i] = rows[i].audio[channel];
            }
            ggml_tensor * embedded = ggml_get_rows(graph_ctx, tok_embd_audio[channel],
                    input_i32(std::move(channel_ids)));
            cur = ggml_add(graph_ctx, cur, embedded);
        }
        return cur;
    }

    void validate_rows(const std::vector<DelayRow> & rows) const {
        if (rows.empty()) {
            fail("empty batch");
        }
        if (pos + (int64_t) rows.size() > n_ctx) {
            fail("context overflow; reset the session or raise the context");
        }
        for (const DelayRow & row : rows) {
            if (row.text < 0 || row.text >= config.text_vocab) {
                fail("text token out of range");
            }
            if ((int) row.audio.size() != config.n_vq) {
                fail("each row needs one code per audio channel");
            }
            for (int32_t code : row.audio) {
                if (code < 0 || code >= config.audio_vocab) {
                    fail("audio code out of range");
                }
            }
        }
    }

    TextLogits projected_text(TextLogits requested) const {
        return slot_ids.empty() ? TextLogits::Full : requested;
    }

    ggml_tensor * control_head() {
        return slot_head ? slot_head : ggml_view_2d(graph_ctx, output_head, config.n_embd,
                (int64_t) slot_ids.size(), (size_t) (slot_ids[1] - slot_ids[0]) * output_head->nb[1],
                (size_t) slot_ids[0] * output_head->nb[1]);
    }

    ggml_tensor * text_projection(ggml_tensor * last, TextLogits text) {
        switch (text) {
            case TextLogits::Full:  return ggml_mul_mat(graph_ctx, output_head, last);
            case TextLogits::Slots: return ggml_mul_mat(graph_ctx, control_head(), last);
            case TextLogits::None:  return nullptr;
        }
        return nullptr;
    }

    static std::vector<float> read_f32(ggml_tensor * tensor) {
        std::vector<float> values((size_t) ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    }

    void scatter_slot_logits(const std::vector<float> & slots, std::vector<float> & logits) const {
        for (size_t i = 0; i < slot_ids.size(); ++i) {
            logits[(size_t) slot_ids[i]] = slots[i];
        }
    }

    std::vector<float> read_text_logits(ggml_tensor * projected, TextLogits text) const {
        if (text == TextLogits::Full) {
            return read_f32(projected);
        }
        std::vector<float> logits((size_t) config.text_vocab, -std::numeric_limits<float>::infinity());
        if (text == TextLogits::Slots) {
            scatter_slot_logits(read_f32(projected), logits);
        }
        return logits;
    }

    ggml_tensor * pack_audio_logits(const std::vector<ggml_tensor *> & heads) {
        constexpr int HEAD_AXIS = 1;
        ggml_tensor * packed = heads.front();
        for (size_t i = 1; i < heads.size(); ++i) {
            packed = ggml_concat(graph_ctx, packed, heads[i], HEAD_AXIS);
        }
        return packed;
    }

    std::vector<std::vector<float>> read_audio_logits(
            const std::vector<ggml_tensor *> & heads, ggml_tensor * packed) const {
        const std::vector<float> values = packed ? read_f32(packed) : std::vector<float>{};
        std::vector<std::vector<float>> logits(heads.size());
        size_t offset = 0;
        for (size_t i = 0; i < heads.size(); ++i) {
            const size_t count = (size_t) ggml_nelements(heads[i]);
            if (packed) {
                logits[i].assign(values.begin() + offset, values.begin() + offset + count);
            } else {
                logits[i] = read_f32(heads[i]);
            }
            offset += count;
        }
        return logits;
    }

    DelayLogits forward(const std::vector<DelayRow> & rows, TextLogits requested_text = TextLogits::Full) {
        validate_rows(rows);
        const TextLogits text = projected_text(requested_text);
        begin_graph();
        const int64_t n_tokens = (int64_t) rows.size();
        const int64_t total = pos + n_tokens;
        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, GRAPH_NODES, false);

        std::vector<int32_t> positions((size_t) n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            positions[i] = (int32_t) (pos + i);
        }
        ggml_tensor * inp_pos = input_i32(std::move(positions));
        ggml_tensor * mask = input_f32(attention_mask(n_tokens, total), total, n_tokens);
        if (fused) {
            mask = ggml_cast(graph_ctx, mask, GGML_TYPE_F16);
        }

        ggml_tensor * cur = embed_rows(rows);
        for (int il = 0; il < config.n_layers; ++il) {
            const Layer & layer = layers[il];
            ggml_tensor * residual = cur;
            cur = attention(graph, layer, il, rms_norm(cur, layer.attn_norm), n_tokens, total, inp_pos, mask);
            cur = ggml_add(graph_ctx, cur, residual);
            residual = cur;
            cur = feed_forward(layer, rms_norm(cur, layer.ffn_norm));
            cur = ggml_add(graph_ctx, cur, residual);
        }
        cur = rms_norm(cur, output_norm);
        ggml_tensor * last = ggml_get_rows(graph_ctx, cur, input_i32({(int32_t) (n_tokens - 1)}));

        ggml_tensor * text_logits = text_projection(last, text);
        if (text_logits) {
            ggml_set_output(text_logits);
            ggml_build_forward_expand(graph, text_logits);
        }
        const bool pack_heads = ::tts_cpp::detail::backend_is_vulkan(backend);
        std::vector<ggml_tensor *> audio_logits(config.n_vq);
        for (int i = 0; i < config.n_vq; ++i) {
            audio_logits[i] = ggml_mul_mat(graph_ctx, output_audio[i], last);
            if (!pack_heads) {
                ggml_set_output(audio_logits[i]);
                ggml_build_forward_expand(graph, audio_logits[i]);
            }
        }
        ggml_tensor * packed_audio = pack_heads ? pack_audio_logits(audio_logits) : nullptr;
        if (packed_audio) {
            ggml_set_output(packed_audio);
            ggml_build_forward_expand(graph, packed_audio);
        }

        if (measure_only) {
            graph_price = measure_fit_graph(backend, graph, GRAPH_NODES);
            return {};
        }
        if (!::tts_cpp::detail::sched_fallback_ensure(sched, backend, GRAPH_NODES,
                {weight_buffer, host_tables.buffer()})) {
            fail("scheduler initialization failed");
        }
        if (!::tts_cpp::detail::sched_fallback_alloc(sched, graph)) {
            fail("graph allocation failed");
        }
        set_graph_inputs();
        const ggml_status status = ::tts_cpp::detail::sched_fallback_compute(sched, backend, graph, n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }

        DelayLogits logits;
        logits.text = read_text_logits(text_logits, text);
        logits.audio = read_audio_logits(audio_logits, packed_audio);
        pos += (int) n_tokens;
        return logits;
    }
};

DelayLM::DelayLM(const std::string & path, const BackendRequest & backend, int n_threads, int n_ctx,
                 bool measure_only, const DelayLayout & layout) : impl_(new Impl) {
    impl_->measure_only = measure_only;
    impl_->layout = layout;
    impl_->load(path, backend, n_threads, n_ctx);
}

FitResult DelayLM::measure(int prompt_rows, int generated_rows) {
    if (!impl_->measure_only) fail("measurement requires a metadata-only model");
    FitResult result = fit_device(impl_->backend);
    result.device.weights_bytes = measure_fit_tensors(impl_->weights,
            ::tts_cpp::detail::weight_buffer_type(impl_->backend));
    result.device.state_bytes = measure_fit_tensors(impl_->state, impl_->backend);
    DelayRow row;
    row.audio.assign(impl_->config.n_vq, 0);
    impl_->pos = 0;
    impl_->forward(std::vector<DelayRow>(prompt_rows, row));
    const auto prefill = impl_->graph_price;
    const uint64_t prefill_host = impl_->graph_host_bytes;
    impl_->pos = prompt_rows + generated_rows - 1;
    impl_->forward({row});
    result.device.lm_compute_bytes = std::max(prefill.device_bytes, impl_->graph_price.device_bytes);
    result.host_bytes = fitutil::sat_add(
        std::max(prefill_host, impl_->graph_host_bytes),
        std::max(prefill.host_bytes, impl_->graph_price.host_bytes));
    result.host_bytes = fitutil::sat_add(result.host_bytes, gguf_get_meta_size(impl_->file));
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->metadata));
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->weights));
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->graph_ctx));
    if (impl_->state) result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->state));
    if (impl_->host_weights) {
        result.host_bytes = fitutil::sat_add(result.host_bytes,
                measure_fit_tensors(impl_->host_weights, ggml_backend_cpu_buffer_type()));
    }
    return result;
}

DelayLM::~DelayLM() = default;

const DelayConfig & DelayLM::config() const { return impl_->config; }
const char * DelayLM::backend_name() const { return ggml_backend_name(impl_->backend); }
bool DelayLM::tables_on_host() const { return impl_->tables_on_host; }
bool DelayLM::fused_attention() const { return impl_->fused; }
int DelayLM::position() const { return impl_->pos; }
int DelayLM::context() const { return impl_->n_ctx; }

void DelayLM::reset() {
    impl_->pos = 0;
}

std::vector<std::string> DelayLM::tokenizer_tokens() const {
    return impl_->meta_str_array("tokenizer.ggml.tokens");
}

std::vector<std::string> DelayLM::tokenizer_merges() const {
    return impl_->meta_str_array("tokenizer.ggml.merges");
}

int32_t DelayLM::token_id(const std::string & token) const {
    const std::vector<std::string> tokens = tokenizer_tokens();
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == token) {
            return (int32_t) i;
        }
    }
    fail("unknown token: " + token);
}

DelayLogits DelayLM::prefill(const std::vector<DelayRow> & rows, TextLogits text) {
    return impl_->forward(rows, text);
}

DelayLogits DelayLM::step(const DelayRow & row, TextLogits text) {
    return impl_->forward({row}, text);
}

} // namespace tts_cpp::moss::detail
