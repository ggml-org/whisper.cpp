#include "moss/transcribe_networks.h"
#include "moss/transcribe_tensors.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace parakeet::moss::detail {
namespace {

constexpr int ENCODER_GRAPH_NODES = 8192;
constexpr int CONV_KERNEL = 3;
constexpr int CONV_DOWNSAMPLE = 2;
constexpr float FFN_DOWN_ACCUMULATION_SCALE = 64.0f;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

std::string encoder_name(int layer, const char * suffix) {
    return "enc.blk." + std::to_string(layer) + "." + suffix;
}

void validate_encoder_layer(const TranscribeModel & model, const TranscribeEncoderConfig & config, int layer) {
    const int64_t width = config.n_embd;
    require_transcribe_f32(model.tensor(encoder_name(layer, "attn_norm.weight")), width);
    require_transcribe_f32(model.tensor(encoder_name(layer, "attn_norm.bias")), width);
    require_transcribe_weight(model.tensor(encoder_name(layer, "attn_q.weight")), width, width);
    require_transcribe_f32(model.tensor(encoder_name(layer, "attn_q.bias")), width);
    require_transcribe_weight(model.tensor(encoder_name(layer, "attn_k.weight")), width, width);
    require_transcribe_weight(model.tensor(encoder_name(layer, "attn_v.weight")), width, width);
    require_transcribe_f32(model.tensor(encoder_name(layer, "attn_v.bias")), width);
    require_transcribe_weight(model.tensor(encoder_name(layer, "attn_output.weight")), width, width);
    require_transcribe_f32(model.tensor(encoder_name(layer, "attn_output.bias")), width);
    require_transcribe_f32(model.tensor(encoder_name(layer, "ffn_norm.weight")), width);
    require_transcribe_f32(model.tensor(encoder_name(layer, "ffn_norm.bias")), width);
    require_transcribe_weight(model.tensor(encoder_name(layer, "ffn_up.weight")), width, config.n_ff);
    require_transcribe_f32(model.tensor(encoder_name(layer, "ffn_up.bias")), config.n_ff);
    require_transcribe_weight(model.tensor(encoder_name(layer, "ffn_down.weight")), config.n_ff, width);
    require_transcribe_f32(model.tensor(encoder_name(layer, "ffn_down.bias")), width);
}

void validate_encoder_layers(const TranscribeModel & model, const TranscribeEncoderConfig & config) {
    for (int layer = 0; layer < config.n_layers; ++layer) {
        validate_encoder_layer(model, config, layer);
    }
}

void validate_adaptor(const TranscribeModel & model) {
    const TranscribeConfig & config = model.config();
    const int64_t merged = (int64_t) config.encoder.n_embd * config.merge_size;
    const int64_t width = config.text.n_embd;
    require_transcribe_weight(model.tensor("adaptor.fc1.weight"), merged, width);
    require_transcribe_f32(model.tensor("adaptor.fc1.bias"), width);
    require_transcribe_weight(model.tensor("adaptor.fc2.weight"), width, width);
    require_transcribe_f32(model.tensor("adaptor.fc2.bias"), width);
    require_transcribe_f32(model.tensor("adaptor.norm.weight"), width);
    require_transcribe_f32(model.tensor("adaptor.norm.bias"), width);
}

void validate_front_end(const TranscribeModel & model) {
    const TranscribeConfig & config = model.config();
    const int64_t width = config.encoder.n_embd;
    require_transcribe_f32(model.tensor("audio.mel_filters"), config.audio.n_fft / 2 + 1, config.audio.n_mels);
    require_transcribe_kernel(model.tensor("enc.conv1.weight"), CONV_KERNEL, config.audio.n_mels, width);
    require_transcribe_f32(model.tensor("enc.conv1.bias"), 1, width);
    require_transcribe_kernel(model.tensor("enc.conv2.weight"), CONV_KERNEL, width, width);
    require_transcribe_f32(model.tensor("enc.conv2.bias"), 1, width);
    require_transcribe_f32(model.tensor("enc.pos_embd"), width, config.encoder.n_ctx);
    require_transcribe_f32(model.tensor("enc.output_norm.weight"), width);
    require_transcribe_f32(model.tensor("enc.output_norm.bias"), width);
}

struct EncoderGraph {
    TranscribeModel & model;
    const TranscribeEncoderConfig & config;
    TranscribeGraph & graph;

    ggml_context * ctx() const { return graph.ctx(); }

    ggml_tensor * weight(int layer, const char * suffix) const {
        return model.tensor(encoder_name(layer, suffix));
    }

    ggml_tensor * layer_norm(ggml_tensor * cur, ggml_tensor * scale, ggml_tensor * shift, float eps) const {
        return ggml_add(ctx(), ggml_mul(ctx(), ggml_norm(ctx(), cur, eps), scale), shift);
    }

    ggml_tensor * linear(ggml_tensor * cur, ggml_tensor * matrix, ggml_tensor * bias) const {
        return ggml_add(ctx(), ggml_mul_mat(ctx(), matrix, cur), bias);
    }

    ggml_tensor * convolution(ggml_tensor * cur, const char * name, int stride) const {
        const std::string prefix = std::string("enc.") + name;
        cur = ggml_conv_1d_ph(ctx(), model.tensor(prefix + ".weight"), cur, stride, 1);
        return ggml_gelu_erf(ctx(), ggml_add(ctx(), cur, model.tensor(prefix + ".bias")));
    }

    ggml_tensor * heads(ggml_tensor * cur, int64_t tokens) const {
        return ggml_reshape_3d(ctx(), cur, config.n_embd / config.n_heads, config.n_heads, tokens);
    }

    float attention_scale() const {
        return 1.0f / std::sqrt((float) (config.n_embd / config.n_heads));
    }

    ggml_tensor * attend_unfused(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int64_t tokens) const {
        ggml_tensor * values = ggml_cont(ctx(), ggml_permute(ctx(), v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_soft_max_ext(ctx(), ggml_mul_mat(ctx(), ggml_permute(ctx(), k, 0, 2, 1, 3),
                ggml_permute(ctx(), q, 0, 2, 1, 3)), nullptr, attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_permute(ctx(), ggml_mul_mat(ctx(), values, scores), 0, 2, 1, 3);
        return ggml_cont_2d(ctx(), attended, config.n_embd, tokens);
    }

    ggml_tensor * attend_fused(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int64_t tokens) const {
        ggml_tensor * attended = ggml_flash_attn_ext(ctx(), ggml_permute(ctx(), q, 0, 2, 1, 3),
                ggml_permute(ctx(), k, 0, 2, 1, 3), ggml_permute(ctx(), v, 0, 2, 1, 3), nullptr,
                attention_scale(), 0.0f, 0.0f);
        return ggml_reshape_2d(ctx(), attended, config.n_embd, tokens);
    }

    ggml_tensor * attention(int layer, ggml_tensor * cur, int64_t tokens) const {
        ggml_tensor * q = heads(linear(cur, weight(layer, "attn_q.weight"), weight(layer, "attn_q.bias")), tokens);
        ggml_tensor * k = heads(ggml_mul_mat(ctx(), weight(layer, "attn_k.weight"), cur), tokens);
        ggml_tensor * v = heads(linear(cur, weight(layer, "attn_v.weight"), weight(layer, "attn_v.bias")), tokens);
        ggml_tensor * attended = model.fused_attention() ? attend_fused(q, k, v, tokens)
                                                         : attend_unfused(q, k, v, tokens);
        return linear(attended, weight(layer, "attn_output.weight"), weight(layer, "attn_output.bias"));
    }

    ggml_tensor * scaled_down_projection(ggml_tensor * hidden, int layer) const {
        ggml_tensor * shrunk = ggml_scale(ctx(), hidden, 1.0f / FFN_DOWN_ACCUMULATION_SCALE);
        ggml_tensor * projected = ggml_mul_mat(ctx(), weight(layer, "ffn_down.weight"), shrunk);
        return ggml_add(ctx(), ggml_scale(ctx(), projected, FFN_DOWN_ACCUMULATION_SCALE),
                weight(layer, "ffn_down.bias"));
    }

    ggml_tensor * feed_forward(int layer, ggml_tensor * cur) const {
        cur = ggml_gelu_erf(ctx(), linear(cur, weight(layer, "ffn_up.weight"), weight(layer, "ffn_up.bias")));
        return scaled_down_projection(cur, layer);
    }

    ggml_tensor * block(int layer, ggml_tensor * cur, int64_t tokens) const {
        ggml_tensor * normed = layer_norm(cur, weight(layer, "attn_norm.weight"), weight(layer, "attn_norm.bias"),
                config.eps);
        cur = ggml_add(ctx(), cur, attention(layer, normed, tokens));
        normed = layer_norm(cur, weight(layer, "ffn_norm.weight"), weight(layer, "ffn_norm.bias"), config.eps);
        return ggml_add(ctx(), cur, feed_forward(layer, normed));
    }

    ggml_tensor * blocks(ggml_tensor * cur, int64_t tokens) const {
        for (int layer = 0; layer < config.n_layers; ++layer) {
            cur = block(layer, cur, tokens);
        }
        return cur;
    }

    ggml_tensor * encode(ggml_tensor * mel) const {
        ggml_tensor * cur = convolution(mel, "conv1", 1);
        cur = convolution(cur, "conv2", CONV_DOWNSAMPLE);
        cur = ggml_add(ctx(), ggml_cont(ctx(), ggml_transpose(ctx(), cur)), model.tensor("enc.pos_embd"));
        cur = blocks(cur, config.n_ctx);
        return layer_norm(cur, model.tensor("enc.output_norm.weight"), model.tensor("enc.output_norm.bias"),
                config.eps);
    }

    ggml_tensor * adapt(ggml_tensor * states, int tokens) const {
        const TranscribeConfig & full = model.config();
        const int64_t frames = (int64_t) tokens * full.merge_size;
        ggml_tensor * kept = ggml_cont(ctx(), ggml_view_2d(ctx(), states, config.n_embd, frames, states->nb[1], 0));
        ggml_tensor * merged = ggml_reshape_2d(ctx(), kept, (int64_t) config.n_embd * full.merge_size, tokens);
        ggml_tensor * cur = ggml_silu(ctx(), linear(merged, model.tensor("adaptor.fc1.weight"),
                model.tensor("adaptor.fc1.bias")));
        cur = linear(cur, model.tensor("adaptor.fc2.weight"), model.tensor("adaptor.fc2.bias"));
        return layer_norm(cur, model.tensor("adaptor.norm.weight"), model.tensor("adaptor.norm.bias"),
                full.adaptor_eps);
    }
};

void validate_chunk(const TranscribeModel & model, const std::vector<float> & mel, int tokens) {
    const TranscribeConfig & config = model.config();
    if (mel.size() != (size_t) config.audio.n_mels * config.audio.chunk_frames) {
        fail("mel chunk has the wrong size");
    }
    if (tokens < 1 || tokens * config.merge_size > config.encoder.n_ctx) {
        fail("chunk token count is outside the encoder window");
    }
}

struct EncoderOutputs {
    ggml_tensor * input;
    ggml_tensor * states;
    ggml_tensor * embeddings;
};

EncoderOutputs build_encoder_graph(TranscribeModel & model, TranscribeGraph & graph, int tokens) {
    const auto & config = model.config();
    EncoderGraph builder{model, config.encoder, graph};
    auto * input = graph.input_f32(config.audio.chunk_frames, config.audio.n_mels);
    auto * states = builder.encode(input);
    auto * embeddings = builder.adapt(states, tokens);
    ggml_set_output(states);
    ggml_set_output(embeddings);
    ggml_build_forward_expand(graph.graph(), states);
    ggml_build_forward_expand(graph.graph(), embeddings);
    return {input, states, embeddings};
}

std::vector<float> read_tensor(ggml_tensor * tensor) {
    std::vector<float> values((size_t) ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return values;
}

} // namespace

void validate_transcribe_encoder(const TranscribeModel & model) {
    validate_front_end(model);
    validate_encoder_layers(model, model.config().encoder);
    validate_adaptor(model);
}

std::vector<float> read_mel_filters(const TranscribeModel & model) {
    return read_tensor(model.tensor("audio.mel_filters"));
}

TranscribeMemory measure_audio_chunk(TranscribeModel & model, int tokens) {
    TranscribeGraph graph(ENCODER_GRAPH_NODES);
    build_encoder_graph(model, graph, tokens);
    return model.measure(graph);
}

TranscribeChunkEncoding encode_audio_chunk(TranscribeModel & model, const std::vector<float> & mel, int tokens,
                                           bool keep_encoder_states) {
    validate_chunk(model, mel, tokens);
    TranscribeGraph graph(ENCODER_GRAPH_NODES);
    const auto outputs = build_encoder_graph(model, graph, tokens);

    model.allocate(graph);
    ggml_backend_tensor_set(outputs.input, mel.data(), 0, mel.size() * sizeof(float));
    model.compute(graph);
    TranscribeChunkEncoding encoding;
    encoding.embeddings = read_tensor(outputs.embeddings);
    if (keep_encoder_states) {
        encoding.encoder_states = read_tensor(outputs.states);
    }
    return encoding;
}

} // namespace parakeet::moss::detail
