#include "moss/speech_tokenizer.h"

#include "moss/gguf_metadata.h"
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
#include <limits>
#include <stdexcept>

std::vector<float> mel_extract_stft_hann_ggml(const std::vector<float> & wav, const std::vector<float> & mel_fb,
                                              int n_fft, int hop, int win, int n_mels, int center_mode,
                                              float power_exp, float log_floor, float mag_eps);

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * ARCH = "moss-speech-codec";
constexpr const char * OWNER = "moss speech tokenizer";
constexpr const char * PREFIX = "whispervq.";
constexpr int GRAPH_NODES = 8192;
constexpr int CONV_KERNEL = 3;
constexpr int CONV_DOWNSAMPLE = 2;
constexpr int MAX_LAYERS = 128;
constexpr int MAX_WIDTH = 1 << 16;
constexpr int MAX_HEADS = 1024;
constexpr int MAX_THREADS = 1024;
constexpr float LAYER_NORM_EPS = 1e-5f;
constexpr float MEL_FLOOR = 1e-10f;
constexpr float DYNAMIC_RANGE = 8.0f;
constexpr float LOG_OFFSET = 4.0f;
constexpr float LOG_SCALE = 4.0f;
constexpr int CENTERED = 1;
constexpr float POWER_SPECTRUM = 2.0f;
constexpr float NO_LOG = -1.0f;
constexpr size_t TENSOR_SLACK = 8;
constexpr int MIN_SAMPLE_RATE = 8000;
constexpr int MAX_SAMPLE_RATE = 192000;
constexpr int MAX_FFT = 1 << 14;
constexpr int MAX_MELS = 512;
constexpr int MAX_CHUNK_SECONDS = 120;
constexpr int MAX_CONTEXT = 1 << 16;
constexpr int MAX_POOLING = 64;
constexpr int MAX_CODEBOOK = 1 << 20;
constexpr int FEED_FORWARD_WIDTH_FACTOR = 4;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(std::string(OWNER) + ": " + message);
}

bool within(int value, int low, int high) {
    return value >= low && value <= high;
}

std::string key(const char * name) {
    return std::string(ARCH) + ".vq." + name;
}

size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

SpeechVqConfig read_config(const GgufMetadata & meta) {
    SpeechVqConfig c;
    c.sample_rate    = (int) meta.u32(key("sample_rate"));
    c.n_fft          = (int) meta.u32(key("n_fft"));
    c.hop_length     = (int) meta.u32(key("hop_length"));
    c.n_mels         = (int) meta.u32(key("n_mels"));
    c.chunk_samples  = (int) meta.u32(key("chunk_samples"));
    c.n_layers       = (int) meta.u32(key("block_count"));
    c.n_embd         = (int) meta.u32(key("embedding_length"));
    c.n_ff           = (int) meta.u32(key("feed_forward_length"));
    c.n_heads        = (int) meta.u32(key("attention.head_count"));
    c.n_ctx          = (int) meta.u32(key("context_length"));
    c.pooling_kernel = (int) meta.u32(key("pooling_kernel"));
    c.codebook_size  = (int) meta.u32(key("codebook_size"));
    return c;
}

bool valid_bounds(const SpeechVqConfig & c) {
    return within(c.sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE) && within(c.n_fft, 2, MAX_FFT) &&
           within(c.hop_length, 1, c.n_fft) && within(c.n_mels, 1, MAX_MELS) &&
           within(c.chunk_samples, c.n_fft, c.sample_rate * MAX_CHUNK_SECONDS) && within(c.n_layers, 1, MAX_LAYERS) &&
           within(c.n_embd, 1, MAX_WIDTH) && within(c.n_ff, 1, MAX_WIDTH * FEED_FORWARD_WIDTH_FACTOR) &&
           within(c.n_heads, 1, MAX_HEADS) && c.n_embd % c.n_heads == 0 && within(c.n_ctx, 1, MAX_CONTEXT) &&
           within(c.pooling_kernel, 1, MAX_POOLING) && within(c.codebook_size, 1, MAX_CODEBOOK);
}

size_t padded_segment_frames(const SpeechVqConfig & c) {
    return ceil_div((size_t) c.chunk_samples, (size_t) c.samples_per_token()) * (size_t) c.pooling_kernel;
}

void validate_config(const SpeechVqConfig & c) {
    if (!valid_bounds(c) || padded_segment_frames(c) > (size_t) c.n_ctx) {
        fail("invalid tokenizer geometry");
    }
}

std::string layer_name(int layer, const char * suffix) {
    return std::string(PREFIX) + "blk." + std::to_string(layer) + "." + suffix;
}

float whisper_log(float energy) {
    return std::log10(std::max(energy, MEL_FLOOR));
}

void scatter_frame(std::vector<float> & out, const float * frame, size_t t, size_t frames, size_t mels) {
    for (size_t m = 0; m < mels; ++m) {
        out[m * frames + t] = frame[m];
    }
}

std::vector<float> transposed(const std::vector<float> & frames_by_mels, size_t frames, size_t mels) {
    std::vector<float> out(frames * mels);
    for (size_t t = 0; t < frames; ++t) {
        scatter_frame(out, frames_by_mels.data() + t * mels, t, frames, mels);
    }
    return out;
}

double squared_norm(const float * values, size_t width) {
    double sum = 0.0;
    for (size_t d = 0; d < width; ++d) {
        sum += (double) values[d] * values[d];
    }
    return sum;
}

std::vector<float> half_squared_norms(const std::vector<float> & codebook, size_t entries, size_t width) {
    std::vector<float> norms(entries);
    for (size_t e = 0; e < entries; ++e) {
        norms[e] = (float) (-0.5 * squared_norm(codebook.data() + e * width, width));
    }
    return norms;
}

class CoremlSpeechTokenizer final : public SpeechTokenizerSidecar {
public:
    CoremlSpeechTokenizer(std::unique_ptr<CoremlModel> model, CoremlTensor mel, CoremlTensor states)
        : model_(std::move(model)), mel_(std::move(mel)), states_(std::move(states)) {}

    bool pooled_states(const std::vector<float> & mel, std::vector<float> & states) override {
        if ((int64_t) mel.size() != tensor_elements(mel_)) {
            return false;
        }
        states.assign((size_t) tensor_elements(states_), 0.0f);
        return model_->predict({mel_}, {mel.data()}, {states_}, {states.data()});
    }

    const char * label() const override {
        return model_->label();
    }

private:
    std::unique_ptr<CoremlModel> model_;
    CoremlTensor mel_;
    CoremlTensor states_;
};

std::vector<float> causal_mask(int64_t tokens) {
    std::vector<float> mask((size_t) (tokens * tokens), -std::numeric_limits<float>::infinity());
    for (int64_t query = 0; query < tokens; ++query) {
        const auto row = mask.begin() + (std::ptrdiff_t) (query * tokens);
        std::fill(row, row + (std::ptrdiff_t) (query + 1), 0.0f);
    }
    return mask;
}

} // namespace

int SpeechVqConfig::samples_per_token() const {
    return hop_length * CONV_DOWNSAMPLE * pooling_kernel;
}

size_t speech_segment_tokens(const SpeechVqConfig & config, size_t samples) {
    const size_t frames = ceil_div(samples, (size_t) config.hop_length);
    return ceil_div(ceil_div(frames, CONV_DOWNSAMPLE), (size_t) config.pooling_kernel);
}

size_t speech_padded_samples(const SpeechVqConfig & config, size_t samples) {
    return ceil_div(samples, (size_t) config.samples_per_token()) * (size_t) config.samples_per_token();
}

SpeechSegmentCapacity speech_segment_capacity(const SpeechVqConfig & config) {
    SpeechSegmentCapacity capacity;
    capacity.tokens = (int) speech_segment_tokens(config, speech_padded_samples(config, (size_t) config.chunk_samples));
    capacity.mel_frames = capacity.tokens * config.pooling_kernel * CONV_DOWNSAMPLE;
    return capacity;
}

std::unique_ptr<SpeechTokenizerSidecar> open_speech_tokenizer_sidecar(const std::string & codec_path,
                                                                      const SpeechVqConfig & config,
                                                                      const CoremlPolicy & policy) {
    std::unique_ptr<CoremlModel> model = CoremlModel::open(speech_tokenizer_sidecar_path(codec_path), policy);
    if (!model) {
        return nullptr;
    }
    const SpeechSegmentCapacity capacity = speech_segment_capacity(config);
    CoremlTensor mel{"mel", {1, config.n_mels, capacity.mel_frames}};
    CoremlTensor states{"states", {1, capacity.tokens, config.n_embd}};
    if (!model->declares(mel) || !model->declares(states)) {
        return nullptr;
    }
    return std::make_unique<CoremlSpeechTokenizer>(std::move(model), std::move(mel), std::move(states));
}

void normalize_whisper_log_mel(std::vector<float> & mel) {
    if (mel.empty()) {
        return;
    }
    const float floor = *std::max_element(mel.begin(), mel.end()) - DYNAMIC_RANGE;
    std::transform(mel.begin(), mel.end(), mel.begin(),
            [floor](float value) { return (std::max(value, floor) + LOG_OFFSET) / LOG_SCALE; });
}

struct SpeechTokenizer::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ::tts_cpp::detail::sched_fallback sched;
    SpeechVqConfig config;
    std::vector<float> mel_filters;
    std::vector<float> code_bias;
    int n_threads = 1;
    std::unique_ptr<SpeechTokenizerSidecar> sidecar;
    bool strict = false;
    AttentionLayout attention_layout = AttentionLayout::Automatic;
    bool fused = false;
    StageBackends backends;

    ~Impl() {
        ::tts_cpp::detail::sched_fallback_free(sched);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    ggml_tensor * find(const std::string & name) const {
        ggml_tensor * tensor = ggml_get_tensor(weights, name.c_str());
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    static bool owned(const ggml_tensor * tensor) {
        return std::string(ggml_get_name(tensor)).rfind(PREFIX, 0) == 0;
    }

    size_t count_owned_tensors() const {
        size_t count = 0;
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            count += owned(src) ? 1 : 0;
        }
        return count;
    }

    void mirror_owned_tensors() {
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            if (owned(src)) {
                ggml_tensor * dst = ggml_new_tensor(weights, src->type, GGML_MAX_DIMS, src->ne);
                ggml_set_name(dst, ggml_get_name(src));
            }
        }
    }

    void stream_tensors(::tts_cpp::detail::gguf_stream_reader & reader) {
        for (auto * dst = ggml_get_first_tensor(weights); dst; dst = ggml_get_next_tensor(weights, dst)) {
            if (!reader.to_backend(ggml_get_name(dst), dst)) {
                fail(std::string("failed to load tensor: ") + ggml_get_name(dst));
            }
        }
    }

    void require_device_weight_types() const {
        if (!::tts_cpp::detail::backend_is_hexagon(backend)) {
            return;
        }
        if (const ggml_tensor * refused = first_tensor_hexagon_cannot_run(weights)) {
            fail(hexagon_weight_refusal(refused, HEXAGON_SPEECH_TOKENIZER_WEIGHT_TIERS));
        }
    }

    void create_weight_context() {
        weights = ggml_init({(count_owned_tensors() + TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) {
            fail("weight context allocation failed");
        }
        mirror_owned_tensors();
    }

    void upload_weights(const std::string & path) {
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights,
                ::tts_cpp::detail::weight_buffer_type(backend));
        if (!weight_buffer) {
            fail("weight allocation failed");
        }
        ::tts_cpp::detail::gguf_stream_reader reader(file, path);
        if (!reader.ok()) {
            fail("cannot reopen GGUF for streaming: " + path);
        }
        stream_tensors(reader);
    }

    void select_backend(const BackendRequest & request) {
        backend = init_backend(request, OWNER);
        fused = fuses_attention(attention_layout, backend);
    }

    void validate_layer(int layer) const {
        const int64_t d = config.n_embd;
        require_vector(find(layer_name(layer, "attn_norm.weight")), d, 1);
        require_vector(find(layer_name(layer, "attn_norm.bias")), d, 1);
        require_matrix(find(layer_name(layer, "attn_q.weight")), d, d);
        require_vector(find(layer_name(layer, "attn_q.bias")), d, 1);
        require_matrix(find(layer_name(layer, "attn_k.weight")), d, d);
        require_matrix(find(layer_name(layer, "attn_v.weight")), d, d);
        require_vector(find(layer_name(layer, "attn_v.bias")), d, 1);
        require_matrix(find(layer_name(layer, "attn_output.weight")), d, d);
        require_vector(find(layer_name(layer, "attn_output.bias")), d, 1);
        require_vector(find(layer_name(layer, "ffn_norm.weight")), d, 1);
        require_vector(find(layer_name(layer, "ffn_norm.bias")), d, 1);
        require_matrix(find(layer_name(layer, "ffn_up.weight")), d, config.n_ff);
        require_vector(find(layer_name(layer, "ffn_up.bias")), config.n_ff, 1);
        require_matrix(find(layer_name(layer, "ffn_down.weight")), config.n_ff, d);
        require_vector(find(layer_name(layer, "ffn_down.bias")), d, 1);
    }

    void validate_layers() const {
        for (int layer = 0; layer < config.n_layers; ++layer) {
            validate_layer(layer);
        }
    }

    void validate_tensors() const {
        const int64_t d = config.n_embd;
        require_shape(find("whispervq.conv1.weight"), CONV_KERNEL, config.n_mels, d);
        require_shape(find("whispervq.conv2.weight"), CONV_KERNEL, d, d);
        if (find("whispervq.conv1.weight")->type != GGML_TYPE_F16 || find("whispervq.conv2.weight")->type != GGML_TYPE_F16) {
            fail("convolution kernels must be f16");
        }
        require_vector(find("whispervq.conv1.bias"), 1, d);
        require_vector(find("whispervq.conv2.bias"), 1, d);
        require_vector(find("whispervq.pos_embd"), d, config.n_ctx);
        require_vector(find("whispervq.codebook"), d, config.codebook_size);
        require_vector(find("whispervq.mel_filters"), config.n_fft / 2 + 1, config.n_mels);
        validate_layers();
    }

    static std::vector<float> read_tensor(ggml_tensor * tensor) {
        std::vector<float> values((size_t) ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        return values;
    }

    void load(const std::string & path, const BackendRequest & request, int threads) {
        if (!within(threads, 1, MAX_THREADS)) {
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
        select_backend(request);
        create_weight_context();
        require_device_weight_types();
        upload_weights(path);
        validate_tensors();
        mel_filters = read_tensor(find("whispervq.mel_filters"));
        code_bias = half_squared_norms(read_tensor(find("whispervq.codebook")), (size_t) config.codebook_size,
                (size_t) config.n_embd);
        const CoremlPolicy policy = read_coreml_policy();
        sidecar = open_speech_tokenizer_sidecar(path, config, policy);
        strict = policy.strict;
    }

    std::vector<float> log_mel(const float * samples, size_t count) const {
        std::vector<float> padded(speech_padded_samples(config, count), 0.0f);
        std::copy(samples, samples + count, padded.begin());
        std::vector<float> power = mel_extract_stft_hann_ggml(padded, mel_filters, config.n_fft, config.hop_length,
                config.n_fft, config.n_mels, CENTERED, POWER_SPECTRUM, NO_LOG, 0.0f);
        const size_t frames = padded.size() / (size_t) config.hop_length;
        if (power.size() < (frames + 1) * (size_t) config.n_mels) {
            fail("mel extraction failed");
        }
        power.resize(frames * (size_t) config.n_mels);
        std::transform(power.begin(), power.end(), power.begin(), whisper_log);
        normalize_whisper_log_mel(power);
        return transposed(power, frames, (size_t) config.n_mels);
    }

    ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * cur, const std::string & base) const {
        cur = ggml_norm(ctx, cur, LAYER_NORM_EPS);
        return ggml_add(ctx, ggml_mul(ctx, cur, find(base + ".weight")), find(base + ".bias"));
    }

    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * cur, const std::string & base, bool bias) const {
        cur = ggml_mul_mat(ctx, find(base + ".weight"), cur);
        return bias ? ggml_add(ctx, cur, find(base + ".bias")) : cur;
    }

    ggml_tensor * causal_conv(ggml_context * ctx, ggml_tensor * cur, const char * name, int stride) const {
        const std::string base = std::string(PREFIX) + name;
        cur = ggml_pad_ext(ctx, cur, CONV_KERNEL - 1, 0, 0, 0, 0, 0, 0, 0);
        cur = ggml_conv_1d(ctx, find(base + ".weight"), cur, stride, 0, 1);
        return ggml_gelu_erf(ctx, ggml_add(ctx, cur, find(base + ".bias")));
    }

    ggml_tensor * heads(ggml_context * ctx, ggml_tensor * cur, int64_t tokens) const {
        return ggml_reshape_3d(ctx, cur, config.n_embd / config.n_heads, config.n_heads, tokens);
    }

    float attention_scale() const {
        return 1.0f / std::sqrt((float) (config.n_embd / config.n_heads));
    }

    ggml_tensor * attend_unfused(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                 int64_t tokens, ggml_tensor * mask) const {
        ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, ggml_permute(ctx, k, 0, 2, 1, 3),
                ggml_permute(ctx, q, 0, 2, 1, 3)), mask, attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_permute(ctx, ggml_mul_mat(ctx, values, scores), 0, 2, 1, 3);
        return ggml_cont_2d(ctx, attended, config.n_embd, tokens);
    }

    ggml_tensor * attend_fused(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                               int64_t tokens, ggml_tensor * mask) const {
        ggml_tensor * attended = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3),
                ggml_permute(ctx, k, 0, 2, 1, 3), ggml_permute(ctx, v, 0, 2, 1, 3), mask,
                attention_scale(), 0.0f, 0.0f);
        return ggml_reshape_2d(ctx, attended, config.n_embd, tokens);
    }

    ggml_tensor * attention(ggml_context * ctx, int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * mask) const {
        const std::string base = std::string(PREFIX) + "blk." + std::to_string(layer) + ".";
        ggml_tensor * q = heads(ctx, linear(ctx, cur, base + "attn_q", true), tokens);
        ggml_tensor * k = heads(ctx, linear(ctx, cur, base + "attn_k", false), tokens);
        ggml_tensor * v = heads(ctx, linear(ctx, cur, base + "attn_v", true), tokens);
        ggml_tensor * attended = fused ? attend_fused(ctx, q, k, v, tokens, mask)
                                       : attend_unfused(ctx, q, k, v, tokens, mask);
        return linear(ctx, attended, base + "attn_output", true);
    }

    ggml_tensor * block(ggml_context * ctx, int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * mask) const {
        const std::string base = std::string(PREFIX) + "blk." + std::to_string(layer) + ".";
        cur = ggml_add(ctx, cur, attention(ctx, layer, layer_norm(ctx, cur, base + "attn_norm"), tokens, mask));
        ggml_tensor * fed = ggml_gelu_erf(ctx, linear(ctx, layer_norm(ctx, cur, base + "ffn_norm"), base + "ffn_up", true));
        return ggml_add(ctx, cur, linear(ctx, fed, base + "ffn_down", true));
    }

    ggml_tensor * blocks(ggml_context * ctx, ggml_tensor * cur, int64_t tokens, ggml_tensor * mask) const {
        for (int layer = 0; layer < config.n_layers; ++layer) {
            cur = block(ctx, layer, cur, tokens, mask);
        }
        return cur;
    }

    ggml_tensor * average_pool(ggml_context * ctx, ggml_tensor * states, int64_t frames) const {
        const int64_t groups = frames / config.pooling_kernel;
        ggml_tensor * grouped = ggml_reshape_3d(ctx, states, config.n_embd, config.pooling_kernel, groups);
        ggml_tensor * by_group = ggml_cont(ctx, ggml_permute(ctx, grouped, 1, 0, 2, 3));
        return ggml_reshape_2d(ctx, ggml_mean(ctx, by_group), config.n_embd, groups);
    }

    ggml_tensor * code_indices(ggml_context * ctx, ggml_tensor * pooled, ggml_tensor * bias) const {
        ggml_tensor * scores = ggml_add(ctx, ggml_mul_mat(ctx, find("whispervq.codebook"), pooled), bias);
        return ggml_argmax(ctx, scores);
    }

    std::vector<int32_t> read_codes(ggml_tensor * codes, int64_t tokens) const {
        std::vector<int32_t> values((size_t) tokens);
        ggml_backend_tensor_get(codes, values.data(), 0, values.size() * sizeof(int32_t));
        return values;
    }

    std::vector<int32_t> encode_on_ggml(const std::vector<float> & mel, int64_t mel_frames) {
        const int64_t frames = mel_frames / CONV_DOWNSAMPLE;
        const int64_t tokens = frames / config.pooling_kernel;
        SfxGraph graph(GRAPH_NODES);
        ggml_context * ctx = graph.ctx();
        ggml_tensor * input = graph.input_f32(mel_frames, config.n_mels);
        ggml_tensor * mask = graph.input_f32(frames, frames);
        ggml_tensor * bias = graph.input_f32(config.codebook_size, 1);
        ggml_tensor * cur = causal_conv(ctx, input, "conv1", 1);
        cur = causal_conv(ctx, cur, "conv2", CONV_DOWNSAMPLE);
        ggml_tensor * positions = ggml_view_2d(ctx, find("whispervq.pos_embd"), config.n_embd, frames,
                find("whispervq.pos_embd")->nb[1], 0);
        cur = ggml_add(ctx, ggml_cont(ctx, ggml_transpose(ctx, cur)), positions);
        ggml_tensor * attention_mask = fused ? ggml_cast(ctx, mask, GGML_TYPE_F16) : mask;
        cur = average_pool(ctx, blocks(ctx, cur, frames, attention_mask), frames);
        ggml_tensor * codes = code_indices(ctx, cur, bias);
        ggml_set_output(codes);
        ggml_build_forward_expand(graph.graph(), codes);
        const std::vector<float> mask_data = causal_mask(frames);
        run(graph, {{input, &mel}, {mask, &mask_data}, {bias, &code_bias}});
        return read_codes(codes, tokens);
    }

    std::vector<int32_t> quantize_pooled(const std::vector<float> & pooled, int64_t tokens) {
        SfxGraph graph(GRAPH_NODES);
        ggml_tensor * input = graph.input_f32(config.n_embd, tokens);
        ggml_tensor * bias = graph.input_f32(config.codebook_size, 1);
        ggml_tensor * codes = code_indices(graph.ctx(), input, bias);
        ggml_set_output(codes);
        ggml_build_forward_expand(graph.graph(), codes);
        run(graph, {{input, &pooled}, {bias, &code_bias}});
        return read_codes(codes, tokens);
    }

    bool sidecar_takes(int64_t mel_frames) const {
        return sidecar && mel_frames == speech_segment_capacity(config).mel_frames;
    }

    bool pooled_on_sidecar(const std::vector<float> & mel, std::vector<float> & pooled) {
        if (!sidecar->pooled_states(mel, pooled)) {
            sidecar.reset();
            return false;
        }
        backends.note(sidecar->label());
        return true;
    }

    std::vector<int32_t> encode_frames(const std::vector<float> & mel, int64_t mel_frames) {
        const int64_t tokens = mel_frames / CONV_DOWNSAMPLE / config.pooling_kernel;
        std::vector<float> pooled;
        if (sidecar_takes(mel_frames) && pooled_on_sidecar(mel, pooled)) {
            return quantize_pooled(pooled, tokens);
        }
        if (strict && !sidecar) {
            fail_strict(OWNER, "tokenizer");
        }
        backends.note(GGML_STAGE_BACKEND);
        return encode_on_ggml(mel, mel_frames);
    }

    struct GraphInput {
        ggml_tensor * tensor;
        const std::vector<float> * data;
    };

    void set_inputs(const std::vector<GraphInput> & inputs) const {
        for (const GraphInput & input : inputs) {
            ggml_backend_tensor_set(input.tensor, input.data->data(), 0, input.data->size() * sizeof(float));
        }
    }

    void run(SfxGraph & graph, const std::vector<GraphInput> & inputs) {
        if (!::tts_cpp::detail::sched_fallback_ensure(sched, backend, GRAPH_NODES, {weight_buffer}) ||
            !::tts_cpp::detail::sched_fallback_alloc(sched, graph.graph())) {
            fail("graph allocation failed");
        }
        set_inputs(inputs);
        if (::tts_cpp::detail::sched_fallback_compute(sched, backend, graph.graph(), n_threads) != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }
    }

    std::vector<int32_t> encode_segment(const float * samples, size_t count) {
        if (count == 0 || count > (size_t) config.chunk_samples) {
            fail("segment must hold 1.." + std::to_string(config.chunk_samples) + " samples");
        }
        const std::vector<float> mel = log_mel(samples, count);
        std::vector<int32_t> codes = encode_frames(mel, (int64_t) (mel.size() / (size_t) config.n_mels));
        codes.resize(std::min(codes.size(), speech_segment_tokens(config, count)));
        return codes;
    }

    std::vector<int32_t> encode(const std::vector<float> & pcm, const std::function<bool()> & stop) {
        std::vector<int32_t> codes;
        const size_t chunk = (size_t) config.chunk_samples;
        for (size_t first = 0; first < pcm.size() && !(stop && stop()); first += chunk) {
            const std::vector<int32_t> part = encode_segment(pcm.data() + first, std::min(chunk, pcm.size() - first));
            codes.insert(codes.end(), part.begin(), part.end());
        }
        return codes;
    }
};

SpeechTokenizer::SpeechTokenizer(const std::string & codec_path, const BackendRequest & backend, int n_threads,
                                 AttentionLayout attention) : impl_(new Impl) {
    impl_->attention_layout = attention;
    impl_->load(codec_path, backend, n_threads);
}

SpeechTokenizer::~SpeechTokenizer() = default;

const SpeechVqConfig & SpeechTokenizer::config() const { return impl_->config; }
const char * SpeechTokenizer::backend_name() const { return ggml_backend_name(impl_->backend); }
bool SpeechTokenizer::fused_attention() const { return impl_->fused; }

std::vector<float> SpeechTokenizer::log_mel(const float * samples, size_t count) const {
    return impl_->log_mel(samples, count);
}

std::vector<int32_t> SpeechTokenizer::encode_segment(const float * samples, size_t count) {
    return impl_->encode_segment(samples, count);
}

std::vector<int32_t> SpeechTokenizer::encode(const std::vector<float> & pcm_16k, const std::function<bool()> & stop) {
    return impl_->encode(pcm_16k, stop);
}

void SpeechTokenizer::attach_sidecar(std::unique_ptr<SpeechTokenizerSidecar> sidecar, bool strict) {
    impl_->sidecar = std::move(sidecar);
    impl_->strict = strict;
}

bool SpeechTokenizer::on_coreml() const {
    return impl_->sidecar != nullptr;
}

void SpeechTokenizer::begin_run() {
    impl_->backends.begin();
}

std::string SpeechTokenizer::run_backend() const {
    return impl_->backends.summary();
}

} // namespace tts_cpp::moss::detail
