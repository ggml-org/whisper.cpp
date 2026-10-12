#include "moss/transcribe_model.h"

#include "moss/gguf_metadata.h"

#include "moss/transcribe_runtime.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "backend_util.h"
#include "fit_util.h"
#include "ggml-backend.h"
#include "gguf.h"

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

} // namespace

int TranscribeConfig::samples_per_token() const {
    return audio.hop_length * ENCODER_STRIDE * merge_size;
}

struct TranscribeModel::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    std::unique_ptr<TranscribeScheduler> scheduler;
    TranscribeConfig config;
    int n_threads = 1;
    bool measure_only = false;
    uint64_t measured_weights = 0;
    const TranscribeGraph * allocated_graph = nullptr;

    ~Impl() {
        scheduler.reset();
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

    void mirror_metadata_tensors() {
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            ggml_tensor * dst = ggml_new_tensor(weights, src->type, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
    }

    void duplicate_metadata_tensors() {
        weights = ggml_init({(count_metadata_tensors() + TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) {
            fail("weight context allocation failed");
        }
        mirror_metadata_tensors();
    }

    void upload_weights(const std::string & path) {
        weight_buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        if (!weight_buffer) {
            fail("weight allocation failed");
        }
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        upload_gguf_tensors(file, path, weights);
    }

    void read_vocabulary_size() {
        const ggml_tensor * embedding = find("text.token_embd.weight");
        config.text.vocab = (int) embedding->ne[1];
    }

    void load(const std::string & path, bool use_gpu, int threads, const std::string & requested) {
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
        duplicate_metadata_tensors();
        read_vocabulary_size();
        if (measure_only) prepare_measurement();
        else upload_weights(path);
        scheduler = std::make_unique<TranscribeScheduler>(backend, SCHED_NODES);
    }

    void bind_measurement_weights() {
        for (auto * tensor = ggml_get_first_tensor(weights); tensor; tensor = ggml_get_next_tensor(weights, tensor)) {
            tensor->buffer = weight_buffer;
            tensor->data = reinterpret_cast<void *>(uintptr_t(1));
        }
    }

    void prepare_measurement() {
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        measured_weights = ggml_backend_alloc_ctx_tensors_from_buft_size(weights, buft);
        weight_buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        if (!weight_buffer) fail("weight measurement buffer failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bind_measurement_weights();
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
                                 const std::string & backend, bool measure_only) : impl_(new Impl) {
    impl_->measure_only = measure_only;
    impl_->load(path, use_gpu, n_threads, backend);
}

TranscribeModel::~TranscribeModel() = default;

const TranscribeConfig & TranscribeModel::config() const { return impl_->config; }
const char * TranscribeModel::backend_name() const { return ggml_backend_name(impl_->backend); }
ggml_backend * TranscribeModel::backend() const { return impl_->backend; }
ggml_tensor * TranscribeModel::tensor(const std::string & name) const { return impl_->find(name); }

std::vector<std::string> TranscribeModel::tokenizer_tokens() const {
    return GgufMetadata(impl_->file, OWNER).str_array("tokenizer.ggml.tokens");
}

std::vector<std::string> TranscribeModel::tokenizer_merges() const {
    return GgufMetadata(impl_->file, OWNER).str_array("tokenizer.ggml.merges");
}

std::vector<int32_t> TranscribeModel::tokenizer_types() const {
    return GgufMetadata(impl_->file, OWNER).int_array("tokenizer.ggml.token_type", (size_t) impl_->config.text.vocab);
}

uint64_t TranscribeModel::weight_bytes() const { return ggml_backend_buffer_get_size(impl_->weight_buffer); }
TranscribeMemory TranscribeModel::allocated_memory() const { return impl_->scheduler->allocated_memory(); }

bool TranscribeModel::measure_only() const { return impl_->measure_only; }

FitResult TranscribeModel::measure_weights() {
    if (!impl_->measure_only) fail("measurement requires a metadata-only model");
    FitResult result;
    const auto device = ggml_backend_get_device(impl_->backend);
    if (!device) fail("no compute backend available");
    result.device_name = ggml_backend_name(impl_->backend);
    result.device_is_cpu = backend_is_cpu(impl_->backend);
    result.device_shares_host_memory = result.device_is_cpu || backend_is_metal(impl_->backend) ||
        ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU;
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    result.device_free_bytes = free;
    result.device_total_bytes = total;
    result.device.weights_bytes = impl_->measured_weights;
    result.host_bytes = fitutil::sat_add(gguf_get_meta_size(impl_->file),
        fitutil::sat_add(ggml_get_mem_size(impl_->metadata), ggml_get_mem_size(impl_->weights)));
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
