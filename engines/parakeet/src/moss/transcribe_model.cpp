#include "moss/transcribe_model.h"

#include "moss/gguf_metadata.h"

#include "moss/transcribe_runtime.h"
#include "moss/transcribe_tensors.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "backend_util.h"
#include "fit_util.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace parakeet::moss::detail {
namespace {

constexpr const char * ARCH = "moss-transcribe";
constexpr const char * OWNER = "moss transcribe";
constexpr int SCHED_NODES = 32768;
constexpr int MAX_LAYERS = 128;
constexpr int MAX_WIDTH = 1 << 16;
constexpr int MAX_HEADS = 1024;
constexpr int MAX_THREADS = 1024;
constexpr int MIN_SAMPLE_RATE = 8000;
constexpr int MAX_SAMPLE_RATE = 192000;
constexpr int MAX_FFT = 1 << 14;
constexpr int MAX_MELS = 512;
constexpr int MAX_CHUNK_SECONDS = 120;
constexpr int MAX_MERGE = 64;
constexpr int MAX_MARKER_SECONDS = 3600;
constexpr size_t MAX_PROMPT_IDS = 4096;
constexpr int MAX_CONTEXT = 1 << 18;
constexpr float MIN_TOKENS_PER_SECOND = 0.01f;
constexpr float MAX_TOKENS_PER_SECOND = (float) MAX_SAMPLE_RATE;
constexpr int ENCODER_STRIDE = 2;
constexpr size_t TENSOR_SLACK = 8;
constexpr int MIN_HEAD_ROWS = 32;
constexpr const char * TOKEN_TABLE = "text.token_embd.weight";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(std::string(OWNER) + ": " + message);
}

bool is_positive_finite(float value) {
    return std::isfinite(value) && value > 0.0f;
}

bool within(int value, int low, int high) {
    return value >= low && value <= high;
}

std::string key(const char * section, const char * name) {
    return std::string(ARCH) + "." + section + "." + name;
}

std::string key(const char * name) {
    return std::string(ARCH) + "." + name;
}

TranscribeAudioConfig read_audio(const GgufMetadata & meta) {
    TranscribeAudioConfig audio;
    audio.sample_rate   = (int) meta.u32(key("audio", "sample_rate"));
    audio.n_fft         = (int) meta.u32(key("audio", "n_fft"));
    audio.hop_length    = (int) meta.u32(key("audio", "hop_length"));
    audio.n_mels        = (int) meta.u32(key("audio", "n_mels"));
    audio.chunk_samples = (int) meta.u32(key("audio", "chunk_samples"));
    audio.chunk_frames  = (int) meta.u32(key("audio", "chunk_frames"));
    return audio;
}

TranscribeEncoderConfig read_encoder(const GgufMetadata & meta) {
    TranscribeEncoderConfig encoder;
    encoder.n_layers = (int) meta.u32(key("encoder", "block_count"));
    encoder.n_embd   = (int) meta.u32(key("encoder", "embedding_length"));
    encoder.n_ff     = (int) meta.u32(key("encoder", "feed_forward_length"));
    encoder.n_heads  = (int) meta.u32(key("encoder", "attention.head_count"));
    encoder.n_ctx    = (int) meta.u32(key("encoder", "context_length"));
    encoder.eps      = meta.f32(key("encoder", "attention.layer_norm_epsilon"));
    return encoder;
}

TranscribeTextConfig read_text(const GgufMetadata & meta) {
    TranscribeTextConfig text;
    text.n_layers    = (int) meta.u32(key("text", "block_count"));
    text.n_embd      = (int) meta.u32(key("text", "embedding_length"));
    text.n_ff        = (int) meta.u32(key("text", "feed_forward_length"));
    text.n_heads     = (int) meta.u32(key("text", "attention.head_count"));
    text.n_kv_heads  = (int) meta.u32(key("text", "attention.head_count_kv"));
    text.head_dim    = (int) meta.u32(key("text", "attention.key_length"));
    text.n_ctx_train = (int) meta.u32(key("text", "context_length"));
    text.rope_base   = meta.f32(key("text", "rope.freq_base"));
    text.rms_eps     = meta.f32(key("text", "attention.layer_norm_rms_epsilon"));
    return text;
}

TranscribeTokens read_tokens(const GgufMetadata & meta) {
    TranscribeTokens tokens;
    tokens.audio_start = (int32_t) meta.u32(key("token", "audio_start"));
    tokens.audio_end   = (int32_t) meta.u32(key("token", "audio_end"));
    tokens.audio_pad   = (int32_t) meta.u32(key("token", "audio_pad"));
    tokens.im_start    = (int32_t) meta.u32(key("token", "im_start"));
    tokens.im_end      = (int32_t) meta.u32(key("token", "im_end"));
    tokens.pad         = (int32_t) meta.u32(key("token", "pad"));
    return tokens;
}

TranscribeConfig read_config(const GgufMetadata & meta) {
    TranscribeConfig config;
    config.audio = read_audio(meta);
    config.encoder = read_encoder(meta);
    config.text = read_text(meta);
    config.tokens = read_tokens(meta);
    config.merge_size = (int) meta.u32(key("adaptor", "merge_size"));
    config.adaptor_eps = meta.f32(key("adaptor", "layer_norm_epsilon"));
    config.audio_tokens_per_second = meta.f32(key("audio_tokens_per_second"));
    config.time_marker_every_seconds = (int) meta.u32(key("time_marker_every_seconds"));
    config.time_markers = meta.boolean(key("time_markers"));
    config.system_prompt = meta.str(key("system_prompt"));
    config.default_prompt = meta.str(key("default_prompt"));
    config.hotword_prefix = meta.str(key("hotword_prefix"));
    config.hotword_separator = meta.str(key("hotword_separator"));
    config.default_prompt_ids = meta.int_array(key("default_prompt_ids"), MAX_PROMPT_IDS);
    config.default_max_new_tokens = (int) meta.u32(key("default_max_new_tokens"));
    return config;
}

void validate_audio(const TranscribeAudioConfig & audio) {
    if (!within(audio.sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE) || !within(audio.n_fft, 2, MAX_FFT) ||
        !within(audio.hop_length, 1, audio.n_fft) || !within(audio.n_mels, 1, MAX_MELS) ||
        !within(audio.chunk_samples, audio.n_fft, audio.sample_rate * MAX_CHUNK_SECONDS) ||
        audio.chunk_frames != audio.chunk_samples / audio.hop_length) {
        fail("invalid audio front-end metadata");
    }
}

void validate_encoder(const TranscribeEncoderConfig & encoder, const TranscribeAudioConfig & audio) {
    if (!within(encoder.n_layers, 1, MAX_LAYERS) || !within(encoder.n_embd, 1, MAX_WIDTH) ||
        !within(encoder.n_ff, 1, MAX_WIDTH * 4) || !within(encoder.n_heads, 1, MAX_HEADS) ||
        encoder.n_embd % encoder.n_heads != 0 || encoder.n_ctx * ENCODER_STRIDE != audio.chunk_frames || !is_positive_finite(encoder.eps)) {
        fail("invalid encoder geometry");
    }
}

void validate_text(const TranscribeTextConfig & text) {
    if (!within(text.n_layers, 1, MAX_LAYERS) || !within(text.n_embd, 1, MAX_WIDTH) ||
        !within(text.n_ff, 1, MAX_WIDTH * 4) || !within(text.n_heads, 1, MAX_HEADS) ||
        !within(text.n_kv_heads, 1, MAX_HEADS) || text.n_heads % text.n_kv_heads != 0 ||
        !within(text.head_dim, 2, MAX_WIDTH) || text.head_dim % 2 != 0 || !within(text.n_ctx_train, 1, MAX_CONTEXT) ||
        !is_positive_finite(text.rope_base) || !is_positive_finite(text.rms_eps)) {
        fail("invalid decoder geometry");
    }
}

void validate_prompt_config(const TranscribeConfig & config) {
    if (!within(config.merge_size, 1, MAX_MERGE) || config.encoder.n_ctx % config.merge_size != 0 ||
        !is_positive_finite(config.adaptor_eps) ||
        !(config.audio_tokens_per_second >= MIN_TOKENS_PER_SECOND &&
          config.audio_tokens_per_second <= MAX_TOKENS_PER_SECOND) ||
        !within(config.time_marker_every_seconds, 0, MAX_MARKER_SECONDS) ||
        config.default_prompt_ids.empty() || !within(config.default_max_new_tokens, 1, config.text.n_ctx_train)) {
        fail("invalid prompt metadata");
    }
}

void validate_config(const TranscribeConfig & config) {
    validate_audio(config.audio);
    validate_encoder(config.encoder, config.audio);
    validate_text(config.text);
    validate_prompt_config(config);
}

int largest_power_of_two_below(int n) {
    int power = 1;
    while (power * 2 < n) power *= 2;
    return power;
}

bool backend_projects(ggml_backend_t backend, const ggml_tensor * table, int rows) {
    ggml_context * ctx = ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
    if (!ctx) return false;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, table->type, table->ne[0], rows);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, table->ne[0], 1);
    const bool supported = ggml_backend_supports_op(backend, ggml_mul_mat(ctx, weight, hidden));
    ggml_free(ctx);
    return supported;
}

int largest_head_rows(ggml_backend_t backend, const ggml_tensor * table) {
    const int vocab = (int) table->ne[1];
    if (backend_projects(backend, table, vocab)) return vocab;
    for (int rows = largest_power_of_two_below(vocab); rows >= MIN_HEAD_ROWS; rows /= 2) {
        if (backend_projects(backend, table, rows)) return rows;
    }
    return 0;
}

} // namespace

std::vector<TranscribeHeadChunk> plan_transcribe_head(int vocab, int max_rows) {
    if (vocab < 1 || max_rows < 1) {
        fail("head chunks need a vocabulary and a positive row count");
    }
    std::vector<TranscribeHeadChunk> chunks;
    for (int row0 = 0; row0 < vocab; row0 += max_rows) {
        chunks.push_back({row0, std::min(max_rows, vocab - row0), nullptr});
    }
    return chunks;
}

int TranscribeConfig::samples_per_token() const {
    return audio.hop_length * ENCODER_STRIDE * merge_size;
}

struct TranscribeModel::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_context * head_weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_backend_buffer_t head_buffer = nullptr;
    std::unique_ptr<TranscribeScheduler> scheduler;
    std::unique_ptr<GgufRows> host_rows;
    std::vector<TranscribeHeadChunk> head;
    TranscribeConfig config;
    int n_threads = 1;
    bool measure_only = false;
    bool split_table = false;
    bool fused_attention = false;
    bool fused_encoder_attention = false;
    uint64_t measured_weights = 0;
    const TranscribeGraph * allocated_graph = nullptr;

    ~Impl() {
        scheduler.reset();
        if (head_buffer) ggml_backend_buffer_free(head_buffer);
        if (head_weights) ggml_free(head_weights);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    size_t count_metadata_tensors() const {
        size_t count = 0;
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            count++;
        }
        return count;
    }

    bool mirrors(const ggml_tensor * src) const {
        return !split_table || std::string(ggml_get_name(src)) != TOKEN_TABLE;
    }

    void mirror_metadata_tensors() {
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            if (!mirrors(src)) continue;
            ggml_tensor * dst = ggml_new_tensor(weights, src->type, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
    }

    ggml_tensor * table() const {
        ggml_tensor * tensor = ggml_get_tensor(metadata, TOKEN_TABLE);
        if (tensor == nullptr) {
            fail(std::string("missing tensor: ") + TOKEN_TABLE);
        }
        return tensor;
    }

    void plan_token_table(int forced_rows) {
        ggml_tensor * embedding = table();
        require_transcribe_weight(embedding, config.text.n_embd, embedding->ne[1]);
        const int vocab = (int) embedding->ne[1];
        const int rows = forced_rows > 0 ? forced_rows : largest_head_rows(backend, embedding);
        if (rows <= 0) {
            fail(std::string(ggml_backend_name(backend)) + " cannot project the " +
                 ggml_type_name(embedding->type) + " token table; convert the model to q8_0 or f16");
        }
        split_table = forced_rows > 0 || rows < vocab || !backend_runs_embedding_lookup(backend, embedding);
        head = plan_transcribe_head(vocab, split_table ? rows : vocab);
    }

    void create_head_tensors() {
        head_weights = ggml_init({(head.size() + TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!head_weights) {
            fail("head context allocation failed");
        }
        const ggml_tensor * embedding = table();
        for (TranscribeHeadChunk & chunk : head) {
            chunk.weight = ggml_new_tensor_2d(head_weights, embedding->type, embedding->ne[0], chunk.rows);
            ggml_set_name(chunk.weight, (std::string(TOKEN_TABLE) + ".rows." + std::to_string(chunk.row0)).c_str());
        }
    }

    void bind_head_tensors() {
        if (split_table) create_head_tensors();
        else head.front().weight = find(TOKEN_TABLE);
    }

    size_t table_offset() const {
        return gguf_tensor_data_offset(file, TOKEN_TABLE, ggml_nbytes(table()));
    }

    std::vector<TensorSource> head_sources() const {
        const size_t row_bytes = ggml_row_size(table()->type, table()->ne[0]);
        const size_t offset = table_offset();
        std::vector<TensorSource> sources;
        for (const TranscribeHeadChunk & chunk : head) {
            sources.push_back({chunk.weight, offset + (size_t) chunk.row0 * row_bytes});
        }
        return sources;
    }

    void upload_head(const std::string & path) {
        head_buffer = ggml_backend_alloc_ctx_tensors_from_buft(head_weights, weight_buffer_type(backend));
        if (!head_buffer) {
            fail("head weight allocation failed");
        }
        ggml_backend_buffer_set_usage(head_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        upload_file_ranges(path, head_sources());
        host_rows = std::make_unique<GgufRows>(path, table_offset(), table()->type, table()->ne[0], table()->ne[1]);
    }

    void duplicate_metadata_tensors() {
        weights = ggml_init({(count_metadata_tensors() + TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) {
            fail("weight context allocation failed");
        }
        mirror_metadata_tensors();
    }

    void upload_weights(const std::string & path) {
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, weight_buffer_type(backend));
        if (!weight_buffer) {
            fail("weight allocation failed");
        }
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        upload_gguf_tensors(file, path, weights);
        if (split_table) upload_head(path);
    }

    void read_vocabulary_size() {
        config.text.vocab = (int) table()->ne[1];
    }

    // Hexagon runs each attention as one flash-attention op: its unfused
    // batched F32 matmuls and score tensors fall to slow HVX kernels.
    void plan_attention(TranscribeAttention requested) {
        fused_attention = requested == TranscribeAttention::Fused ||
            (requested == TranscribeAttention::Auto && backend_is_hexagon(backend));
        fused_encoder_attention = fused_attention ||
            (requested == TranscribeAttention::Auto && backend_is_cpu(backend));
    }

    void load(const std::string & path, bool use_gpu, int threads, const std::string & requested,
              const TranscribeLayout & layout) {
        if (threads < 1 || threads > MAX_THREADS) {
            fail("threads must be 1.." + std::to_string(MAX_THREADS));
        }
        n_threads = threads;
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) {
            fail("cannot read GGUF: " + path);
        }
        const GgufMetadata meta(file, OWNER);
        if (meta.str("general.architecture") != ARCH) {
            fail("unsupported architecture");
        }
        config = read_config(meta);
        validate_config(config);
        backend = open_transcribe_backend(use_gpu, requested);
        read_vocabulary_size();
        plan_token_table(layout.head_rows);
        plan_attention(layout.attention);
        duplicate_metadata_tensors();
        bind_head_tensors();
        if (measure_only) prepare_measurement();
        else upload_weights(path);
        scheduler = std::make_unique<TranscribeScheduler>(backend, SCHED_NODES);
    }

    void bind_measurement_weights(ggml_context * ctx) {
        for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            tensor->buffer = weight_buffer;
            tensor->data = reinterpret_cast<void *>(uintptr_t(1));
        }
    }

    void prepare_measurement() {
        const auto buft = weight_buffer_type(backend);
        measured_weights = ggml_backend_alloc_ctx_tensors_from_buft_size(weights, buft);
        if (split_table) measured_weights += ggml_backend_alloc_ctx_tensors_from_buft_size(head_weights, buft);
        weight_buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        if (!weight_buffer) fail("weight measurement buffer failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bind_measurement_weights(weights);
        if (split_table) bind_measurement_weights(head_weights);
    }

    std::vector<float> embed_tokens(const std::vector<int32_t> & ids) const {
        if (!host_rows) fail("token embeddings are looked up in the graph on this backend");
        const size_t width = (size_t) config.text.n_embd;
        std::vector<float> rows(ids.size() * width);
        for (size_t i = 0; i < ids.size(); ++i) {
            host_rows->dequantize(ids[i], rows.data() + i * width);
        }
        return rows;
    }

    ggml_tensor * find(const std::string & name) const {
        ggml_tensor * tensor = ggml_get_tensor(weights, name.c_str());
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    void allocate(TranscribeGraph & graph) {
        if (measure_only) fail("cannot allocate a metadata-only model");
        allocated_graph = nullptr;
        if (!scheduler->allocate(graph.graph())) {
            fail("graph allocation failed");
        }
        allocated_graph = &graph;
    }

    void compute(TranscribeGraph & graph) {
        if (measure_only) fail("cannot compute a metadata-only model");
        if (allocated_graph != &graph) {
            fail("graph is no longer allocated; another graph ran on this model since");
        }
        if (!scheduler->compute(graph.graph(), n_threads)) {
            fail("graph compute failed");
        }
    }
};

TranscribeModel::TranscribeModel(const std::string & path, bool use_gpu, int n_threads,
                                 const std::string & backend, bool measure_only, const TranscribeLayout & layout)
    : impl_(new Impl) {
    impl_->measure_only = measure_only;
    impl_->load(path, use_gpu, n_threads, backend, layout);
}

TranscribeModel::~TranscribeModel() = default;

const TranscribeConfig & TranscribeModel::config() const { return impl_->config; }
const char * TranscribeModel::backend_name() const { return ggml_backend_name(impl_->backend); }
ggml_backend * TranscribeModel::backend() const { return impl_->backend; }
ggml_tensor * TranscribeModel::tensor(const std::string & name) const { return impl_->find(name); }
const ggml_tensor * TranscribeModel::token_table() const { return impl_->table(); }
bool TranscribeModel::embeds_on_host() const { return impl_->split_table; }
bool TranscribeModel::fused_attention() const { return impl_->fused_attention; }
bool TranscribeModel::fused_encoder_attention() const { return impl_->fused_encoder_attention; }
const std::vector<TranscribeHeadChunk> & TranscribeModel::head() const { return impl_->head; }

std::vector<float> TranscribeModel::embed_tokens(const std::vector<int32_t> & ids) const {
    return impl_->embed_tokens(ids);
}

std::vector<std::string> TranscribeModel::tokenizer_tokens() const {
    return GgufMetadata(impl_->file, OWNER).str_array("tokenizer.ggml.tokens");
}

std::vector<std::string> TranscribeModel::tokenizer_merges() const {
    return GgufMetadata(impl_->file, OWNER).str_array("tokenizer.ggml.merges");
}

std::vector<int32_t> TranscribeModel::tokenizer_types() const {
    return GgufMetadata(impl_->file, OWNER).int_array("tokenizer.ggml.token_type", (size_t) impl_->config.text.vocab);
}

uint64_t TranscribeModel::weight_bytes() const {
    const uint64_t head = impl_->head_buffer ? ggml_backend_buffer_get_size(impl_->head_buffer) : 0;
    return ggml_backend_buffer_get_size(impl_->weight_buffer) + head;
}
TranscribeMemory TranscribeModel::allocated_memory() const { return impl_->scheduler->allocated_memory(); }

bool TranscribeModel::measure_only() const { return impl_->measure_only; }

FitResult TranscribeModel::measure_weights() {
    if (!impl_->measure_only) fail("measurement requires a metadata-only model");
    FitResult result;
    const auto device = ggml_backend_get_device(impl_->backend);
    if (!device) fail("no compute backend available");
    result.device_name = ggml_backend_name(impl_->backend);
    result.device_is_cpu = backend_is_cpu(impl_->backend);
    result.device_shares_host_memory = backend_shares_host_memory(impl_->backend);
    size_t free = 0, total = 0;
    backend_device_memory(impl_->backend, free, total);
    result.device_free_bytes = free;
    result.device_total_bytes = total;
    result.device.weights_bytes = impl_->measured_weights;
    result.host_bytes = fitutil::sat_add(gguf_get_meta_size(impl_->file),
        fitutil::sat_add(ggml_get_mem_size(impl_->metadata), ggml_get_mem_size(impl_->weights)));
    if (impl_->head_weights) {
        result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->head_weights));
    }
    result.host_bytes =
        fitutil::sat_add(result.host_bytes, ggml_context_overhead() * 2);
    return result;
}

TranscribeMemory TranscribeModel::measure(TranscribeGraph & graph) {
    if (!impl_->measure_only) fail("measurement requires a metadata-only model");
    auto memory = impl_->scheduler->measure(graph.graph(), impl_->n_threads);
    memory.host_bytes = fitutil::sat_add(
        memory.host_bytes, fitutil::sat_add(ggml_get_mem_size(graph.ctx()),
                                            ggml_context_overhead()));
    return memory;
}

void TranscribeModel::allocate(TranscribeGraph & graph) { impl_->allocate(graph); }
void TranscribeModel::compute(TranscribeGraph & graph) { impl_->compute(graph); }

} // namespace parakeet::moss::detail
