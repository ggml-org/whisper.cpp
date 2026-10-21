#include "parakeet_diarization_v3.h"

#include "backend_util.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef PARAKEET_USE_COREML
#include "coreml/coreml_diarizers.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace parakeet {
namespace {

constexpr float kLayerNormEpsilon = 1.0e-5f;
constexpr size_t kGraphSlots = 4096;
constexpr float kDefaultPrewarmSeconds = 1.0f;
constexpr size_t kProbeTensorSlots = 32;
constexpr int kQkvComponents = 3;

ggml_tensor * layer_norm(
    ggml_context * context, ggml_tensor * input,
    ggml_tensor * weight, ggml_tensor * bias) {
    return ggml_add(context,
        ggml_mul(context, ggml_norm(context, input, kLayerNormEpsilon), weight), bias);
}

struct GraphOptions {
    bool fused_attention = true;
    bool f32_matmuls = false;
};

ggml_tensor * linear(
    ggml_context * context, ggml_tensor * input,
    ggml_tensor * weight, ggml_tensor * bias, const GraphOptions & options) {
    ggml_tensor * output = ggml_mul_mat(context, weight, input);
    if (options.f32_matmuls) ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return bias ? ggml_add(context, output, bias) : output;
}

ggml_tensor * qkv_view(
    ggml_context * context, ggml_tensor * qkv,
    const NemotronDiarizationConfig & config, int component, int frames) {
    const int head_width = config.encoder_width / config.attention_heads;
    return ggml_view_4d(context, qkv, head_width, config.attention_heads, frames, 1,
        static_cast<size_t>(head_width) * qkv->nb[0], qkv->nb[1], qkv->nb[2],
        static_cast<size_t>(component * config.encoder_width) * qkv->nb[0]);
}

struct AttentionHeads {
    ggml_tensor * query = nullptr;
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
};

int head_width_of(const NemotronDiarizationConfig & config) {
    return config.encoder_width / config.attention_heads;
}

float attention_scale(const NemotronDiarizationConfig & config) {
    return 1.0f / std::sqrt(static_cast<float>(head_width_of(config)));
}

ggml_tensor * rotate(
    ggml_context * context, ggml_tensor * heads, ggml_tensor * positions,
    const NemotronDiarizationConfig & config) {
    const int rotary_width = static_cast<int>(head_width_of(config) * config.rotary_fraction);
    return ggml_rope_ext(context, heads, positions, nullptr, rotary_width,
        GGML_ROPE_TYPE_NEOX, config.position_limit, config.rope_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

AttentionHeads attention_heads(
    ggml_context * context, ggml_tensor * qkv, ggml_tensor * positions,
    const NemotronDiarizationConfig & config, int frames) {
    AttentionHeads heads;
    heads.query = ggml_permute(context,
        rotate(context, qkv_view(context, qkv, config, 0, frames), positions, config), 0, 2, 1, 3);
    heads.key = ggml_permute(context,
        rotate(context, qkv_view(context, qkv, config, 1, frames), positions, config), 0, 2, 1, 3);
    heads.value = ggml_permute(context, qkv_view(context, qkv, config, 2, frames), 0, 2, 1, 3);
    return heads;
}

ggml_tensor * fused_attention(
    ggml_context * context, const AttentionHeads & heads,
    const NemotronDiarizationConfig & config) {
    return ggml_flash_attn_ext(context, heads.query, heads.key, heads.value,
        nullptr, attention_scale(config), 0.0f, 0.0f);
}

ggml_tensor * unfused_attention(
    ggml_context * context, const AttentionHeads & heads,
    const NemotronDiarizationConfig & config) {
    ggml_tensor * scores = ggml_mul_mat(context,
        ggml_cont(context, heads.key), ggml_cont(context, heads.query));
    scores = ggml_soft_max_ext(context, scores, nullptr, attention_scale(config), 0.0f);
    ggml_tensor * values = ggml_cont(context, ggml_permute(context, heads.value, 1, 0, 2, 3));
    return ggml_permute(context, ggml_mul_mat(context, values, scores), 0, 2, 1, 3);
}

ggml_tensor * rotary_attention(
    ggml_context * context, ggml_tensor * input, ggml_tensor * positions,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationLayer & layer, int frames, const GraphOptions & options) {
    ggml_tensor * qkv = linear(context, input, layer.qkv_w, nullptr, options);
    const AttentionHeads heads = attention_heads(context, qkv, positions, config, frames);
    ggml_tensor * attended = options.fused_attention
        ? fused_attention(context, heads, config)
        : unfused_attention(context, heads, config);
    ggml_tensor * merged = ggml_reshape_2d(context,
        ggml_cont(context, attended), config.encoder_width, frames);
    return linear(context, merged, layer.attention_w, layer.attention_b, options);
}

bool backend_runs_fused_attention(
    ggml_backend_t backend, const NemotronDiarizationConfig & config, int frames) {
    ggml_init_params parameters = {ggml_tensor_overhead() * kProbeTensorSlots, nullptr, true};
    ggml_context * context = ggml_init(parameters);
    if (!context) return false;
    ggml_tensor * qkv = ggml_new_tensor_2d(context, GGML_TYPE_F32,
        static_cast<int64_t>(kQkvComponents) * config.encoder_width, frames);
    ggml_tensor * positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, frames);
    const AttentionHeads heads = attention_heads(context, qkv, positions, config, frames);
    const bool supported = ggml_backend_supports_op(backend, fused_attention(context, heads, config));
    ggml_free(context);
    return supported;
}

ggml_tensor * encoder_layer(
    ggml_context * context, ggml_tensor * input, ggml_tensor * positions,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationLayer & layer, int frames, const GraphOptions & options) {
    ggml_tensor * normalized = layer_norm(
        context, input, layer.norm1_w, layer.norm1_b);
    ggml_tensor * attended = rotary_attention(
        context, normalized, positions, config, layer, frames, options);
    ggml_tensor * residual = ggml_add(context, input, attended);
    normalized = layer_norm(context, residual, layer.norm2_w, layer.norm2_b);
    ggml_tensor * hidden = linear(context, normalized,
        layer.feed_forward_in_w, layer.feed_forward_in_b, options);
    hidden = ggml_gelu_erf(context, hidden);
    ggml_tensor * output = linear(context, hidden,
        layer.feed_forward_out_w, layer.feed_forward_out_b, options);
    return ggml_add(context, residual, output);
}

ggml_tensor * encode(
    ggml_context * context, ggml_tensor * stacked, ggml_tensor * state,
    ggml_tensor * positions, ggml_tensor ** chunk_embeddings,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationWeights & weights, int frames, const GraphOptions & options) {
    *chunk_embeddings = linear(context, stacked, weights.feature_projection, nullptr, options);
    ggml_tensor * encoded = state
        ? ggml_concat(context, state, *chunk_embeddings, 1)
        : *chunk_embeddings;
    encoded = layer_norm(context, encoded,
        weights.embedding_norm_w, weights.embedding_norm_b);
    for (const auto & layer : weights.layers) {
        encoded = encoder_layer(context, encoded, positions, config, layer, frames, options);
    }
    return layer_norm(context, encoded,
        weights.final_norm_w, weights.final_norm_b);
}

ggml_tensor * speaker_head(
    ggml_context * context, ggml_tensor * encoded,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationWeights & weights, int frames, const GraphOptions & options) {
    ggml_tensor * projected = linear(context, encoded,
        weights.encoder_projection_w, weights.encoder_projection_b, options);
    ggml_tensor * convolution_input = ggml_cont(context,
        ggml_permute(context, projected, 1, 0, 2, 3));
    ggml_tensor * convolved = ggml_conv_1d(context,
        weights.upsample_w, convolution_input, 1, 1, 1);
    ggml_tensor * channels_first = ggml_cont(context,
        ggml_permute(context, convolved, 1, 0, 2, 3));
    channels_first = ggml_add(context, channels_first, weights.upsample_b);
    ggml_tensor * upsampled = ggml_reshape_2d(context,
        channels_first, config.output_width,
        static_cast<int64_t>(frames) * config.subsampling_factor);
    ggml_tensor * hidden = linear(context, ggml_relu(context, upsampled),
        weights.hidden_w, weights.hidden_b, options);
    ggml_tensor * logits = linear(context, ggml_relu(context, hidden),
        weights.speakers_w, weights.speakers_b, options);
    return ggml_sigmoid(context, logits);
}

void stack_mel(
    const float * mel, int mel_frames, int mel_bins, int factor,
    std::vector<float> & stacked) {
    const int frames = (mel_frames + factor - 1) / factor;
    stacked.assign(static_cast<size_t>(frames) * factor * mel_bins, 0.0f);
    std::memcpy(stacked.data(), mel,
        static_cast<size_t>(mel_frames) * mel_bins * sizeof(float));
}

void fill_positions(int frames, std::vector<int32_t> & positions) {
    positions.resize(frames);
    for (int frame = 0; frame < frames; ++frame) positions[frame] = frame;
}

struct DiarizationGraph {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * state_input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * chunk_embeddings = nullptr;
    ggml_tensor * output = nullptr;
};

long long ceil_divide(long long value, long long divisor) {
    return value / divisor + (value % divisor != 0);
}

int chunk_frames_for(const NemotronDiarizationConfig & config, int mel_frames) {
    return static_cast<int>(ceil_divide(mel_frames, config.subsampling_factor));
}

int validate_chunk_shape(
    const NemotronDiarizationConfig & config, int mel_frames, int state_frames) {
    if (mel_frames <= 0 || state_frames < 0 || config.subsampling_factor <= 0) return 1;
    if (state_frames + chunk_frames_for(config, mel_frames) > config.position_limit) return 2;
    return 0;
}

ggml_context * new_graph_context() {
    const size_t overhead = ggml_tensor_overhead() * kGraphSlots +
        ggml_graph_overhead_custom(kGraphSlots, false);
    ggml_init_params parameters = {overhead, nullptr, true};
    return ggml_init(parameters);
}

void add_graph_inputs(
    const ParakeetCtcModel & model, int chunk_frames, int state_frames,
    DiarizationGraph & graph) {
    const auto & config = model.nemotron_diarization_cfg;
    graph.input = ggml_new_tensor_2d(graph.context, GGML_TYPE_F32,
        static_cast<int64_t>(config.subsampling_factor) * model.mel_cfg.n_mels,
        chunk_frames);
    graph.state_input = state_frames > 0
        ? ggml_new_tensor_2d(graph.context, GGML_TYPE_F32, config.encoder_width, state_frames)
        : nullptr;
    graph.positions = ggml_new_tensor_1d(graph.context, GGML_TYPE_I32,
        state_frames + chunk_frames);
    ggml_set_input(graph.input);
    if (graph.state_input) ggml_set_input(graph.state_input);
    ggml_set_input(graph.positions);
}

GraphOptions graph_options(
    const ParakeetCtcModel & model, int frames, NemotronAttention attention) {
    ggml_backend_sched_t scheduler = model_sched(model);
    ggml_backend_t backend = scheduler ? ggml_backend_sched_get_backend(scheduler, 0) : nullptr;
    GraphOptions options;
    options.fused_attention = backend && attention == NemotronAttention::Automatic &&
        backend_runs_fused_attention(backend, model.nemotron_diarization_cfg, frames);
    options.f32_matmuls = backend && backend_is_adreno(backend);
    return options;
}

bool build_graph(
    const ParakeetCtcModel & model, int chunk_frames, int state_frames,
    NemotronAttention attention, DiarizationGraph & graph) {
    graph.context = new_graph_context();
    if (!graph.context) return false;
    add_graph_inputs(model, chunk_frames, state_frames, graph);
    const auto & config = model.nemotron_diarization_cfg;
    const int frames = state_frames + chunk_frames;
    const GraphOptions options = graph_options(model, frames, attention);
    ggml_tensor * encoded = encode(graph.context, graph.input, graph.state_input,
        graph.positions, &graph.chunk_embeddings,
        config, model.nemotron_diarization, frames, options);
    graph.output = speaker_head(graph.context, encoded,
        config, model.nemotron_diarization, frames, options);
    ggml_set_output(graph.output);
    ggml_set_output(graph.chunk_embeddings);
    graph.graph = ggml_new_graph_custom(graph.context, kGraphSlots, false);
    ggml_build_forward_expand(graph.graph, graph.output);
    return true;
}

int compute_graph(
    const ParakeetCtcModel & model, const DiarizationGraph & graph,
    const std::vector<float> & stacked,
    const float * state_values, int state_frames,
    const std::vector<int32_t> & position_values,
    NemotronDiarizationChunk & result) {
    ggml_backend_sched_t scheduler = model_sched(model);
    if (!scheduler) return 1;
    ggml_backend_sched_reset(scheduler);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph)) return 2;
    ggml_backend_tensor_set(graph.input, stacked.data(), 0,
        stacked.size() * sizeof(float));
    if (graph.state_input) {
        ggml_backend_tensor_set(graph.state_input, state_values, 0,
            static_cast<size_t>(state_frames) * model.nemotron_diarization_cfg.encoder_width * sizeof(float));
    }
    ggml_backend_tensor_set(graph.positions, position_values.data(), 0,
        position_values.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(scheduler, graph.graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    result.state_frames = state_frames;
    result.chunk_frames = static_cast<int>(graph.chunk_embeddings->ne[1]);
    result.probabilities.resize(static_cast<size_t>(graph.output->ne[1]) *
        model.nemotron_diarization_cfg.speakers);
    result.embeddings.resize(static_cast<size_t>(result.chunk_frames) *
        model.nemotron_diarization_cfg.encoder_width);
    ggml_backend_tensor_get(graph.output, result.probabilities.data(), 0,
        result.probabilities.size() * sizeof(float));
    ggml_backend_tensor_get(graph.chunk_embeddings, result.embeddings.data(), 0,
        result.embeddings.size() * sizeof(float));
    return 0;
}

size_t padded_host_bytes(
    ggml_backend_buffer_type_t buffer_type, const ggml_tensor * tensor) {
    if (!tensor) return 0;
    return GGML_PAD(ggml_backend_buft_get_alloc_size(buffer_type, tensor),
        ggml_backend_buft_get_alignment(buffer_type));
}

size_t host_input_bytes(ggml_backend_t host_backend, const DiarizationGraph & graph) {
    ggml_backend_buffer_type_t buffer_type = ggml_backend_get_default_buffer_type(host_backend);
    return padded_host_bytes(buffer_type, graph.input) +
        padded_host_bytes(buffer_type, graph.state_input) +
        padded_host_bytes(buffer_type, graph.positions);
}

int measure_graph(
    ggml_backend_sched_t scheduler, const DiarizationGraph & graph,
    NemotronDiarizationFitMeasure & output) {
    ggml_backend_t compute_backend = ggml_backend_sched_get_backend(scheduler, 0);
    ggml_gallocr_t pricer = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(compute_backend));
    if (!pricer) return 4;
    ggml_gallocr_reserve_n_size(pricer, graph.graph, nullptr, nullptr,
        &output.device_compute_bytes);
    ggml_gallocr_free(pricer);
    const int backends = ggml_backend_sched_get_n_backends(scheduler);
    if (backends > 1) {
        output.host_input_bytes = host_input_bytes(
            ggml_backend_sched_get_backend(scheduler, backends - 1), graph);
    }
    return 0;
}

GraphOptions stream_graph_options(const ParakeetCtcModel & model) {
    const int frames = nemotron_diarization_stream_state_frames() +
        nemotron_diarization_encoder_frames(model, nemotron_diarization_stream_mel_frames(model));
    return graph_options(model, frames, NemotronAttention::Automatic);
}

#ifdef PARAKEET_USE_COREML
constexpr const char * kCoremlFeatures = "features";
constexpr const char * kCoremlCache = "cache";
constexpr const char * kCoremlProbabilities = "probabilities";
constexpr const char * kCoremlEmbeddings = "embeddings";
constexpr int kCoremlInputs = 2;
constexpr int kCoremlOutputs = 2;

bool run_coreml_chunk(
    const ParakeetCtcModel & model, const std::vector<float> & stacked,
    const float * state, int state_frames, int chunk_frames,
    NemotronDiarizationChunk & result) {
    CoremlDiarizers * diarizers = model_coreml_diarizers(model);
    const int frames = state_frames + chunk_frames;
    if (!diarizers || !diarizers->covers(frames)) return false;
    const auto & config = model.nemotron_diarization_cfg;
    const int stacked_width = config.subsampling_factor * model.mel_cfg.n_mels;
    const int output_rows = frames * config.subsampling_factor;
    result.probabilities.resize(static_cast<size_t>(output_rows) * config.speakers);
    result.embeddings.resize(static_cast<size_t>(chunk_frames) * config.encoder_width);
    const parakeet_coreml_rows inputs[kCoremlInputs] = {
        {kCoremlFeatures, state_frames, chunk_frames, stacked_width, stacked.data()},
        {kCoremlCache, 0, state_frames, config.encoder_width, state},
    };
    const parakeet_coreml_rows_out outputs[kCoremlOutputs] = {
        {kCoremlProbabilities, 0, output_rows, config.speakers, result.probabilities.data()},
        {kCoremlEmbeddings, state_frames, chunk_frames, config.encoder_width,
            result.embeddings.data()},
    };
    if (diarizers->run(frames, inputs, kCoremlInputs, outputs, kCoremlOutputs) != 0) return false;
    result.state_frames = state_frames;
    result.chunk_frames = chunk_frames;
    result.used_coreml = true;
    return true;
}
#endif

int run_zero_chunk(const ParakeetCtcModel & model, int mel_frames, int state_frames) {
    const std::vector<float> mel(
        static_cast<size_t>(mel_frames) * model.mel_cfg.n_mels, 0.0f);
    const std::vector<float> state(static_cast<size_t>(state_frames) *
        model.nemotron_diarization_cfg.encoder_width, 0.0f);
    NemotronDiarizationChunk chunk;
    return run_nemotron_diarization_chunk(model, mel.data(), mel_frames,
        state_frames > 0 ? state.data() : nullptr, state_frames, chunk);
}

}

int run_nemotron_diarization_chunk(
    const ParakeetCtcModel & model, const float * mel, int mel_frames,
    const float * state, int state_frames, NemotronDiarizationChunk & output,
    NemotronAttention attention) {
    const auto & config = model.nemotron_diarization_cfg;
    if (!mel || (state_frames > 0 && !state)) return 1;
    const int shape = validate_chunk_shape(config, mel_frames, state_frames);
    if (shape != 0) return shape;
    const int chunk_frames = chunk_frames_for(config, mel_frames);
    std::vector<float> stacked;
    stack_mel(mel, mel_frames, model.mel_cfg.n_mels,
        config.subsampling_factor, stacked);
    output.used_coreml = false;
#ifdef PARAKEET_USE_COREML
    if (attention == NemotronAttention::Automatic &&
        run_coreml_chunk(model, stacked, state, state_frames, chunk_frames, output)) return 0;
#endif
    std::vector<int32_t> position_values;
    fill_positions(state_frames + chunk_frames, position_values);
    DiarizationGraph graph;
    if (!build_graph(model, chunk_frames, state_frames, attention, graph)) return 3;
    const int result = compute_graph(model, graph, stacked, state,
        state_frames, position_values, output);
    ggml_free(graph.context);
    return result;
}

int run_nemotron_diarization(
    const ParakeetCtcModel & model, const float * mel, int mel_frames,
    std::vector<float> & probabilities, NemotronAttention attention) {
    NemotronDiarizationChunk chunk;
    const int result = run_nemotron_diarization_chunk(
        model, mel, mel_frames, nullptr, 0, chunk, attention);
    if (result != 0) return result;
    probabilities = std::move(chunk.probabilities);
    probabilities.resize(static_cast<size_t>(mel_frames) *
        model.nemotron_diarization_cfg.speakers);
    return 0;
}

int prewarm_nemotron_diarization(const ParakeetCtcModel & model, float audio_seconds) {
    const float seconds = audio_seconds > 0.0f ? audio_seconds : kDefaultPrewarmSeconds;
    const int offline_frames = std::max(model.nemotron_diarization_cfg.subsampling_factor,
        static_cast<int>(std::lround(
            seconds * model.mel_cfg.sample_rate / model.mel_cfg.hop_length)));
    const int offline = run_zero_chunk(model, offline_frames, 0);
    if (offline != 0) return offline;
    return run_zero_chunk(model, nemotron_diarization_stream_mel_frames(model),
        nemotron_diarization_stream_state_frames());
}

long long nemotron_diarization_mel_frames(
    const ParakeetCtcModel & model, long long samples) {
    return 1 + samples / model.mel_cfg.hop_length;
}

int nemotron_diarization_encoder_frames(
    const ParakeetCtcModel & model, long long mel_frames) {
    const long long factor = std::max(1, model.nemotron_diarization_cfg.subsampling_factor);
    return static_cast<int>(std::min<long long>(
        ceil_divide(mel_frames, factor), std::numeric_limits<int>::max()));
}

int nemotron_diarization_stream_mel_frames(const ParakeetCtcModel & model) {
    const long long window_ms = kNemotronChunkMs + kNemotronRightContextMs;
    return static_cast<int>(window_ms * model.mel_cfg.sample_rate /
        (static_cast<long long>(kMillisecondsPerSecond) * model.mel_cfg.hop_length));
}

int nemotron_diarization_long_form_mel_frames(const ParakeetCtcModel & model) {
    const long long window_ms = kNemotronLongFormChunkMs + 2LL * kNemotronLongFormContextMs;
    return static_cast<int>(window_ms * model.mel_cfg.sample_rate /
        (static_cast<long long>(kMillisecondsPerSecond) * model.mel_cfg.hop_length));
}

int nemotron_diarization_stream_state_frames() {
    return kNemotronSpeakerCacheFrames + kNemotronFifoFrames;
}

int nemotron_diarization_long_form_frames(
    const ParakeetCtcModel & model, int requested_frames) {
    const int limit = model.nemotron_diarization_cfg.position_limit;
    if (requested_frames < 0) return 0;
    if (requested_frames == 0) return std::min(kNemotronLongFormAutoFrames, limit);
    return std::min(requested_frames, limit);
}

bool nemotron_diarization_uses_long_form(
    const ParakeetCtcModel & model, int requested_frames, long long mel_frames) {
    const int window = nemotron_diarization_long_form_frames(model, requested_frames);
    return window > 0 && nemotron_diarization_encoder_frames(model, mel_frames) > window;
}

bool nemotron_diarization_uses_fused_attention(const ParakeetCtcModel & model) {
    return stream_graph_options(model).fused_attention;
}

bool nemotron_diarization_uses_f32_matmuls(const ParakeetCtcModel & model) {
    return stream_graph_options(model).f32_matmuls;
}

int measure_nemotron_diarization(
    const ParakeetCtcModel & model, int mel_frames, int state_frames,
    NemotronDiarizationFitMeasure & output) {
    output = {};
    const auto & config = model.nemotron_diarization_cfg;
    const int shape = validate_chunk_shape(config, mel_frames, state_frames);
    if (shape != 0) return shape;
    ggml_backend_sched_t scheduler = model_sched(model);
    if (!scheduler) return 3;
    DiarizationGraph graph;
    if (!build_graph(model, chunk_frames_for(config, mel_frames), state_frames,
            NemotronAttention::Automatic, graph)) return 3;
    const int result = measure_graph(scheduler, graph, output);
    ggml_free(graph.context);
    return result;
}

}
