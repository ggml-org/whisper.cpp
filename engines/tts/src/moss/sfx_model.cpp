#include "moss/sfx_model.h"
#include "moss/fit_measure.h"
#include "moss/gguf_rows.h"

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
#include <memory>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * ARCH = "moss-sfx";
constexpr const char * OWNER = "moss-sfx";
constexpr int SCHED_NODES = 32768;
constexpr int SCHED_SPLIT_INPUTS = 30;
constexpr int SCHED_BACKEND_ID_ARRAYS = 4;
constexpr int SCHED_COPY_DIRECTIONS = 2;
constexpr int MAX_LAYERS = 128;
constexpr int MAX_WIDTH = 1 << 16;
constexpr int MAX_HEADS = 1024;
constexpr int MAX_THREADS = 1024;
constexpr size_t TENSOR_SLACK = 8;
constexpr int MAX_TEXT_TOKENS = 4096;
constexpr int MAX_RATES = 16;
constexpr int MAX_RATE = 64;
constexpr int MIN_SAMPLE_RATE = 8000;
constexpr int MAX_SAMPLE_RATE = 192000;
constexpr float MAX_SECONDS_LIMIT = 600.0f;
constexpr int64_t MAX_HOP = 1 << 16;
constexpr int64_t MAX_LATENT_FRAMES = 1 << 14;
constexpr int ATTENTION_PROBE_NODES = 16;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss sfx: " + message);
}

bool within(int value, int low, int high) {
    return value >= low && value <= high;
}

bool is_matrix_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_Q8_0;
}

void require_f32(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32) {
        fail(std::string(ggml_get_name(tensor)) + " must be f32");
    }
}

int64_t rate_product(const std::vector<int> & rates) {
    int64_t product = 1;
    for (int rate : rates) {
        product = std::min<int64_t>(product * rate, MAX_HOP + 1);
    }
    return product;
}

ggml_tensor * attention_probe_input(SfxGraph & graph, const SfxDitConfig & config, int tokens) {
    auto * input = graph.input_f32(config.n_embd, tokens);
    auto * heads = ggml_reshape_3d(graph.ctx(), input, config.n_embd / config.n_heads, config.n_heads, tokens);
    return ggml_permute(graph.ctx(), heads, 0, 2, 1, 3);
}

bool supports_dit_flash(ggml_backend_t backend, const SfxConfig & config, int keys) {
    SfxGraph graph(ATTENTION_PROBE_NODES);
    auto * q = attention_probe_input(graph, config.dit, config.latent_frames());
    auto * k = attention_probe_input(graph, config.dit, keys);
    auto * v = attention_probe_input(graph, config.dit, keys);
    const float scale = 1.0f / std::sqrt(float(config.dit.n_embd / config.dit.n_heads));
    auto * attention = ggml_flash_attn_ext(graph.ctx(), q, k, v, nullptr, scale, 0.0f, 0.0f);
    return ggml_backend_supports_op(backend, attention);
}

bool fuses_dit_attention(const SfxLayout & layout, ggml_backend_t backend, const SfxConfig & config) {
    if (fuses_attention(layout.attention, backend)) return true;
    if (layout.attention != AttentionLayout::Automatic || !::tts_cpp::detail::backend_is_metal(backend)) return false;
    return supports_dit_flash(backend, config, config.latent_frames()) &&
           supports_dit_flash(backend, config, config.text.max_tokens);
}

} // namespace

void require_matrix(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    require_shape(tensor, ne0, ne1);
    if (!is_matrix_type(tensor->type)) {
        fail(std::string(ggml_get_name(tensor)) + " has an unsupported type");
    }
}

void require_vector(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    require_shape(tensor, ne0, ne1);
    require_f32(tensor);
}

void require_shape(const ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 || tensor->ne[2] != ne2 || tensor->ne[3] != 1) {
        fail(std::string(ggml_get_name(tensor)) + " has unexpected dimensions");
    }
}

int SfxVaeConfig::hop_length() const {
    return (int) rate_product(rates);
}

int SfxConfig::latent_frames() const {
    const int64_t samples = std::llround((double) vae.sample_rate * max_seconds);
    return (int) std::min<int64_t>(samples / rate_product(vae.rates), MAX_LATENT_FRAMES + 1);
}

SfxGraph::SfxGraph(int max_nodes) {
    const size_t bytes = (size_t) max_nodes * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(max_nodes, false);
    ctx_ = ggml_init({bytes, nullptr, true});
    if (!ctx_) {
        fail("graph context allocation failed");
    }
    graph_ = ggml_new_graph_custom(ctx_, max_nodes, false);
}

SfxGraph::~SfxGraph() {
    ggml_free(ctx_);
}

ggml_context * SfxGraph::ctx() const { return ctx_; }
ggml_cgraph * SfxGraph::graph() const { return graph_; }

ggml_tensor * SfxGraph::input_f32(int64_t ne0, int64_t ne1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ne0, ne1);
    ggml_set_input(tensor);
    return tensor;
}

ggml_tensor * SfxGraph::input_i32(int64_t ne0) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, ne0);
    ggml_set_input(tensor);
    return tensor;
}

struct SfxModel::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_context * host_weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    std::unique_ptr<GgufRows> token_rows;
    ::tts_cpp::detail::sched_fallback sched;
    SfxConfig config;
    SfxLayout layout;
    int n_threads = 1;
    const SfxGraph * allocated_graph = nullptr;
    bool measure_only = false;
    bool table_on_host = false;
    bool fused = false;
    bool fused_dit = false;
    uint64_t measured_weights = 0;

    ~Impl() {
        ::tts_cpp::detail::sched_fallback_free(sched);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (host_weights) ggml_free(host_weights);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    int64_t key(const std::string & name) const {
        return gguf_find_key(file, name.c_str());
    }

    uint32_t meta_u32(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0) {
            fail("missing metadata: " + name);
        }
        switch (gguf_get_kv_type(file, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(file, id);
            case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(file, id);
            default:               fail("unsupported metadata type: " + name);
        }
    }

    float meta_f32(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_FLOAT32) {
            fail("missing float metadata: " + name);
        }
        return gguf_get_val_f32(file, id);
    }

    std::string meta_str(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_STRING) {
            fail("missing string metadata: " + name);
        }
        return gguf_get_val_str(file, id);
    }

    std::vector<std::string> meta_str_array(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(file, id) != GGUF_TYPE_STRING) {
            fail("missing string array metadata: " + name);
        }
        return read_strings(id);
    }

    std::vector<std::string> read_strings(int64_t id) const {
        std::vector<std::string> values(gguf_get_arr_n(file, id));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = gguf_get_arr_str(file, id, i);
        }
        return values;
    }

    std::vector<int> meta_int_array(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_ARRAY) {
            fail("missing array metadata: " + name);
        }
        const gguf_type type = gguf_get_arr_type(file, id);
        if (type != GGUF_TYPE_INT32 && type != GGUF_TYPE_UINT32) {
            fail("unsupported array type: " + name);
        }
        const size_t count = gguf_get_arr_n(file, id);
        if (count > (size_t) MAX_RATES) {
            fail("array is too long: " + name);
        }
        const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(file, id));
        return std::vector<int>(data, data + count);
    }

    void read_text_config() {
        const std::string prefix = std::string(ARCH) + ".text";
        SfxTextConfig & text = config.text;
        text.n_layers   = (int) meta_u32(prefix + ".block_count");
        text.n_embd     = (int) meta_u32(prefix + ".embedding_length");
        text.n_ff       = (int) meta_u32(prefix + ".feed_forward_length");
        text.n_heads    = (int) meta_u32(prefix + ".attention.head_count");
        text.n_kv_heads = (int) meta_u32(prefix + ".attention.head_count_kv");
        text.head_dim   = (int) meta_u32(prefix + ".attention.key_length");
        text.max_tokens = (int) meta_u32(prefix + ".context_length");
        text.rope_base  = meta_f32(prefix + ".rope.freq_base");
        text.rms_eps    = meta_f32(prefix + ".attention.layer_norm_rms_epsilon");
    }

    void read_dit_config() {
        const std::string prefix = std::string(ARCH) + ".dit";
        SfxDitConfig & dit = config.dit;
        dit.n_layers     = (int) meta_u32(prefix + ".block_count");
        dit.n_embd       = (int) meta_u32(prefix + ".embedding_length");
        dit.n_ff         = (int) meta_u32(prefix + ".feed_forward_length");
        dit.n_heads      = (int) meta_u32(prefix + ".attention.head_count");
        dit.in_channels  = (int) meta_u32(prefix + ".in_channels");
        dit.out_channels = (int) meta_u32(prefix + ".out_channels");
        dit.text_dim     = (int) meta_u32(prefix + ".text_dim");
        dit.freq_dim     = (int) meta_u32(prefix + ".freq_dim");
        dit.eps          = meta_f32(prefix + ".epsilon");
    }

    void read_vae_config() {
        const std::string prefix = std::string(ARCH) + ".vae";
        SfxVaeConfig & vae = config.vae;
        vae.latent_dim  = (int) meta_u32(prefix + ".latent_dim");
        vae.decoder_dim = (int) meta_u32(prefix + ".decoder_dim");
        vae.sample_rate = (int) meta_u32(prefix + ".sample_rate");
        vae.rates       = meta_int_array(prefix + ".decoder_rates");
    }

    void read_generation_config() {
        const std::string prefix(ARCH);
        config.max_seconds      = meta_f32(prefix + ".max_seconds");
        config.sigma_shift      = meta_f32(prefix + ".sigma_shift");
        config.train_timesteps  = (int) meta_u32(prefix + ".num_train_timesteps");
        config.default_steps    = (int) meta_u32(prefix + ".default_steps");
        config.default_guidance = meta_f32(prefix + ".default_guidance");
    }

    void validate_text_config() const {
        const SfxTextConfig & text = config.text;
        if (!within(text.n_layers, 1, MAX_LAYERS) || !within(text.n_embd, 1, MAX_WIDTH) ||
            !within(text.n_ff, 1, MAX_WIDTH * 4) || !within(text.n_heads, 1, MAX_HEADS) ||
            !within(text.n_kv_heads, 1, MAX_HEADS) || !within(text.head_dim, 1, MAX_WIDTH) ||
            text.n_heads % text.n_kv_heads != 0 || text.head_dim % 2 != 0 ||
            !within(text.max_tokens, 1, MAX_TEXT_TOKENS)) {
            fail("invalid text encoder geometry");
        }
    }

    void validate_dit_config() const {
        const SfxDitConfig & dit = config.dit;
        if (!within(dit.n_layers, 1, MAX_LAYERS) || !within(dit.n_embd, 1, MAX_WIDTH) ||
            !within(dit.n_ff, 1, MAX_WIDTH * 4) || !within(dit.n_heads, 1, MAX_HEADS) ||
            dit.n_embd % dit.n_heads != 0 || ((dit.n_embd / dit.n_heads) % 2) != 0 ||
            !within(dit.in_channels, 1, MAX_WIDTH) || dit.out_channels != dit.in_channels ||
            dit.text_dim != config.text.n_embd || !within(dit.freq_dim, 2, MAX_WIDTH) ||
            dit.freq_dim % 2 != 0) {
            fail("invalid DiT geometry");
        }
    }

    void validate_vae_rates() const {
        const SfxVaeConfig & vae = config.vae;
        for (size_t i = 0; i < vae.rates.size(); ++i) {
            if (!within(vae.rates[i], 1, MAX_RATE) || (vae.decoder_dim >> (i + 1)) < 1) {
                fail("invalid VAE decoder rates");
            }
        }
    }

    void validate_vae_config() const {
        const SfxVaeConfig & vae = config.vae;
        if (vae.latent_dim != config.dit.out_channels || vae.rates.empty() ||
            !within(vae.decoder_dim, 1, MAX_WIDTH) ||
            !within(vae.sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)) {
            fail("invalid VAE geometry");
        }
        validate_vae_rates();
        if (rate_product(vae.rates) > MAX_HOP) {
            fail("VAE decoder rates multiply past the supported hop length");
        }
    }

    void validate_generation_config() const {
        if (!(config.max_seconds > 0.0f && config.max_seconds <= MAX_SECONDS_LIMIT) ||
            config.latent_frames() < 1 || config.latent_frames() > MAX_LATENT_FRAMES ||
            config.train_timesteps < 1 ||
            config.default_steps < 1 || !(config.sigma_shift > 0.0f)) {
            fail("invalid generation metadata");
        }
    }

    void read_config() {
        read_text_config();
        read_dit_config();
        read_vae_config();
        read_generation_config();
        validate_text_config();
        validate_dit_config();
        validate_vae_config();
        validate_generation_config();
    }

    size_t count_metadata_tensors() const {
        size_t count = 0;
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            count++;
        }
        return count;
    }

    ggml_context * context_for(const char * name) const {
        return host_weights && std::strcmp(name, SFX_TOKEN_TABLE) == 0 ? host_weights : weights;
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
        if (table_on_host) {
            host_weights = new_weight_context();
        }
        mirror_metadata_tensors();
    }

    void require_device_weight_types() const {
        if (!::tts_cpp::detail::backend_is_hexagon(backend)) {
            return;
        }
        if (const ggml_tensor * refused = first_tensor_hexagon_cannot_run(weights)) {
            fail(hexagon_weight_refusal(refused, HEXAGON_SFX_WEIGHT_TIERS));
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
            token_rows = std::make_unique<GgufRows>(path, file, find(SFX_TOKEN_TABLE), OWNER);
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
        table_on_host = places_tables_on_host(layout.table, backend);
        fused = fuses_attention(layout.attention, backend);
        fused_dit = fuses_dit_attention(layout, backend, config);
    }

    void load(const std::string & path, const BackendRequest & request, int threads) {
        if (threads < 1 || threads > MAX_THREADS) {
            fail("threads must be 1.." + std::to_string(MAX_THREADS));
        }
        n_threads = threads;
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
        require_device_weight_types();
        if (measure_only) {
            prepare_measurement();
        } else {
            upload_weights(path);
        }
    }

    static ggml_backend_buffer_t measurement_buffer(ggml_backend_buffer_type_t buft) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        if (!buffer) fail("weight measurement buffer failed");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        return buffer;
    }

    void prepare_measurement() {
        ggml_backend_buffer_type_t buft = ::tts_cpp::detail::weight_buffer_type(backend);
        measured_weights = measure_fit_tensors(weights, buft);
        weight_buffer = measurement_buffer(buft);
        bind_measurement_weights(weights, weight_buffer);
    }

    static void bind_measurement_weights(ggml_context * ctx, ggml_backend_buffer_t buffer) {
        for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            tensor->buffer = buffer;
        }
    }

    std::vector<ggml_backend_buffer_t> weight_buffers() const {
        return {weight_buffer};
    }

    bool ensure_scheduler() {
        return ::tts_cpp::detail::sched_fallback_ensure(sched, backend, SCHED_NODES, weight_buffers());
    }

    ggml_tensor * find(const std::string & name) const {
        ggml_tensor * tensor = ggml_get_tensor(context_for(name.c_str()), name.c_str());
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    void allocate(SfxGraph & graph) {
        if (measure_only) fail("cannot allocate a metadata-only model");
        if (!ensure_scheduler()) {
            fail("scheduler initialization failed");
        }
        allocated_graph = nullptr;
        if (!::tts_cpp::detail::sched_fallback_alloc(sched, graph.graph())) {
            fail("graph allocation failed");
        }
        allocated_graph = &graph;
    }

    void compute(SfxGraph & graph) {
        if (measure_only) fail("cannot compute a metadata-only model");
        if (allocated_graph != &graph) {
            fail("graph is no longer allocated; another graph ran on this model since");
        }
        const ggml_status status = ::tts_cpp::detail::sched_fallback_compute(sched, backend,
                graph.graph(), n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }
    }
};

SfxModel::SfxModel(const std::string & path, const BackendRequest & backend, int n_threads, bool measure_only,
                   const SfxLayout & layout) : impl_(new Impl) {
    impl_->measure_only = measure_only;
    impl_->layout = layout;
    impl_->load(path, backend, n_threads);
}

SfxModel::~SfxModel() = default;

const SfxConfig & SfxModel::config() const { return impl_->config; }
const char * SfxModel::backend_name() const { return ggml_backend_name(impl_->backend); }
bool SfxModel::table_on_host() const { return impl_->table_on_host; }
bool SfxModel::fused_attention() const { return impl_->fused; }
bool SfxModel::fused_dit_attention() const { return impl_->fused_dit; }
ggml_backend_buffer_type * SfxModel::buffer_type() const {
    return ::tts_cpp::detail::weight_buffer_type(impl_->backend);
}

std::vector<float> SfxModel::token_rows(const std::vector<int32_t> & ids) {
    if (!impl_->token_rows) fail("the token table is not on the host");
    return impl_->token_rows->gather(ids);
}
ggml_tensor * SfxModel::tensor(const std::string & name) const { return impl_->find(name); }

std::vector<std::string> SfxModel::tokenizer_tokens() const {
    return impl_->meta_str_array("tokenizer.ggml.tokens");
}

std::vector<std::string> SfxModel::tokenizer_merges() const {
    return impl_->meta_str_array("tokenizer.ggml.merges");
}

void SfxModel::allocate(SfxGraph & graph) { impl_->allocate(graph); }
void SfxModel::compute(SfxGraph & graph) { impl_->compute(graph); }

FitResult SfxModel::measure_weights() {
    if (!impl_->measure_only) fail("measurement requires a metadata-only model");
    auto result = fit_device(impl_->backend);
    result.device.weights_bytes = impl_->measured_weights;
    result.host_bytes = fitutil::sat_add(gguf_get_meta_size(impl_->file),
        fitutil::sat_add(ggml_get_mem_size(impl_->metadata), ggml_get_mem_size(impl_->weights)));
    if (impl_->host_weights) {
        const ggml_tensor * table = impl_->find(SFX_TOKEN_TABLE);
        result.host_bytes = fitutil::sat_add(result.host_bytes,
            fitutil::sat_add(ggml_get_mem_size(impl_->host_weights), ggml_row_size(table->type, table->ne[0])));
    }
    result.host_bytes = fitutil::sat_add(result.host_bytes, impl_->config.vae.rates.capacity() * sizeof(int));
    const uint64_t sched_nodes = uint64_t(SCHED_NODES) * (1 + SCHED_SPLIT_INPUTS * SCHED_COPY_DIRECTIONS);
    const uint64_t descriptors = uint64_t(SCHED_NODES) * SCHED_SPLIT_INPUTS * SCHED_COPY_DIRECTIONS * sizeof(ggml_tensor);
    result.host_bytes = fitutil::sat_add(result.host_bytes,
        fitutil::sat_add(descriptors, sched_nodes * sizeof(int) * SCHED_BACKEND_ID_ARRAYS));
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_graph_overhead_custom(SCHED_NODES, false));
    return result;
}

SfxMemory SfxModel::measure(SfxGraph & graph) {
    if (!impl_->measure_only) fail("measurement requires a metadata-only model");
    SfxMemory result;
    if (!impl_->ensure_scheduler()) {
        fail("scheduler measurement initialization failed");
    }
    size_t sizes[2] = {0, 0};
    ggml_backend_sched_reserve_size(impl_->sched.sched, graph.graph(), sizes);
    result.device_bytes = sizes[0];
    result.host_compute_bytes = sizes[1];
    result.host_bytes = ggml_get_mem_size(graph.ctx());
    return result;
}

} // namespace tts_cpp::moss::detail
