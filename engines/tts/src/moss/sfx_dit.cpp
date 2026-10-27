#include "moss/sfx_networks.h"
#include "moss/sfx_dit_cache.h"
#include "fit_util.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tts_cpp::moss::detail {
namespace {

constexpr int DIT_GRAPH_NODES = 16384;
constexpr int MODULATION_ROWS = 6;
constexpr int HEAD_MODULATION_ROWS = 2;
constexpr float ROPE_BASE = 10000.0f;
constexpr int TIMESTEP_MAX_PERIOD = 10000;

enum Modulation { SHIFT_MSA, SCALE_MSA, GATE_MSA, SHIFT_MLP, SCALE_MLP, GATE_MLP };
enum HeadModulation { HEAD_SHIFT, HEAD_SCALE };

struct DitGraph {
    SfxModel & model;
    const SfxDitConfig & config;
    SfxGraph & graph;
    SfxDitCache * cache = nullptr;
    ggml_tensor * context_index = nullptr;

    ggml_context * ctx() const { return graph.ctx(); }
    int head_dim() const { return config.n_embd / config.n_heads; }

    ggml_tensor * global(const std::string & name) const {
        return model.tensor("dit." + name);
    }

    ggml_tensor * weight(int layer, const std::string & name) const {
        return model.tensor("dit.blk." + std::to_string(layer) + "." + name);
    }

    ggml_tensor * linear(ggml_tensor * cur, const std::string & prefix) const {
        return ggml_add(ctx(), ggml_mul_mat(ctx(), model.tensor(prefix + ".weight"), cur),
                model.tensor(prefix + ".bias"));
    }

    ggml_tensor * layer_linear(int layer, const std::string & name, ggml_tensor * cur) const {
        return linear(cur, "dit.blk." + std::to_string(layer) + "." + name);
    }

    ggml_tensor * rms_norm(ggml_tensor * cur, ggml_tensor * scale) const {
        return ggml_mul(ctx(), ggml_rms_norm(ctx(), cur, config.eps), scale);
    }

    ggml_tensor * modulate(ggml_tensor * cur, ggml_tensor * shift, ggml_tensor * scale) const {
        ggml_tensor * normed = ggml_norm(ctx(), cur, config.eps);
        return ggml_add(ctx(), ggml_add(ctx(), normed, ggml_mul(ctx(), normed, scale)), shift);
    }

    ggml_tensor * row(ggml_tensor * table, int index) const {
        return ggml_view_1d(ctx(), table, table->ne[0], (size_t) index * table->nb[1]);
    }

    ggml_tensor * split_heads(ggml_tensor * cur) const {
        return ggml_reshape_3d(ctx(), cur, head_dim(), config.n_heads, cur->ne[1]);
    }

    ggml_tensor * rope(ggml_tensor * cur, ggml_tensor * positions) const {
        return ggml_rope_ext(ctx(), cur, positions, nullptr, head_dim(), GGML_ROPE_TYPE_NORMAL,
                0, ROPE_BASE, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    float attention_scale() const {
        return 1.0f / std::sqrt((float) head_dim());
    }

    ggml_tensor * attend_unfused(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) const {
        const int64_t queries = q->ne[2];
        q = ggml_permute(ctx(), q, 0, 2, 1, 3);
        k = ggml_permute(ctx(), k, 0, 2, 1, 3);
        ggml_tensor * values = ggml_cont(ctx(), ggml_permute(ctx(), v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_soft_max_ext(ctx(), ggml_mul_mat(ctx(), k, q), nullptr,
                attention_scale(), 0.0f);
        ggml_tensor * attended = ggml_permute(ctx(), ggml_mul_mat(ctx(), values, scores), 0, 2, 1, 3);
        return ggml_cont_2d(ctx(), attended, config.n_embd, queries);
    }

    ggml_tensor * attend_fused(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) const {
        const int64_t queries = q->ne[2];
        ggml_tensor * attended = ggml_flash_attn_ext(ctx(), ggml_permute(ctx(), q, 0, 2, 1, 3),
                ggml_permute(ctx(), k, 0, 2, 1, 3), ggml_permute(ctx(), v, 0, 2, 1, 3), nullptr,
                attention_scale(), 0.0f, 0.0f);
        return ggml_reshape_2d(ctx(), attended, config.n_embd, queries);
    }

    ggml_tensor * attend(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) const {
        return model.fused_dit_attention() ? attend_fused(q, k, v) : attend_unfused(q, k, v);
    }

    ggml_tensor * self_attention(int layer, ggml_tensor * cur, ggml_tensor * positions) const {
        ggml_tensor * q = rms_norm(layer_linear(layer, "self_q", cur), weight(layer, "self_norm_q.weight"));
        ggml_tensor * k = rms_norm(layer_linear(layer, "self_k", cur), weight(layer, "self_norm_k.weight"));
        ggml_tensor * v = layer_linear(layer, "self_v", cur);
        q = rope(split_heads(q), positions);
        k = rope(split_heads(k), positions);
        return layer_linear(layer, "self_o", attend(q, k, split_heads(v)));
    }

    ggml_tensor * cross_key(int layer, ggml_tensor * context) const {
        return rms_norm(layer_linear(layer, "cross_k", context), weight(layer, "cross_norm_k.weight"));
    }

    ggml_tensor * cross_attention(int layer, ggml_tensor * cur, ggml_tensor * context) const {
        ggml_tensor * q = rms_norm(layer_linear(layer, "cross_q", cur), weight(layer, "cross_norm_q.weight"));
        ggml_tensor * k = cache ? cache->key(graph, layer, context_index) : cross_key(layer, context);
        ggml_tensor * v = cache ? cache->value(graph, layer, context_index) : layer_linear(layer, "cross_v", context);
        return layer_linear(layer, "cross_o", attend(split_heads(q), split_heads(k), split_heads(v)));
    }

    ggml_tensor * affine_norm(int layer, ggml_tensor * cur) const {
        ggml_tensor * normed = ggml_norm(ctx(), cur, config.eps);
        return ggml_add(ctx(), ggml_mul(ctx(), normed, weight(layer, "cross_norm.weight")),
                weight(layer, "cross_norm.bias"));
    }

    ggml_tensor * feed_forward(int layer, ggml_tensor * cur) const {
        return layer_linear(layer, "ffn_down", ggml_gelu(ctx(), layer_linear(layer, "ffn_up", cur)));
    }

    ggml_tensor * block(int layer, ggml_tensor * x, ggml_tensor * context, ggml_tensor * t_mod,
                        ggml_tensor * positions) const {
        ggml_tensor * mod = ggml_add(ctx(), weight(layer, "modulation"), t_mod);
        ggml_tensor * attn_in = modulate(x, row(mod, SHIFT_MSA), row(mod, SCALE_MSA));
        x = ggml_add(ctx(), x, ggml_mul(ctx(), self_attention(layer, attn_in, positions), row(mod, GATE_MSA)));
        x = ggml_add(ctx(), x, cross_attention(layer, affine_norm(layer, x), context));
        ggml_tensor * ffn_in = modulate(x, row(mod, SHIFT_MLP), row(mod, SCALE_MLP));
        return ggml_add(ctx(), x, ggml_mul(ctx(), feed_forward(layer, ffn_in), row(mod, GATE_MLP)));
    }

    ggml_tensor * blocks(ggml_tensor * x, ggml_tensor * context, ggml_tensor * t_mod,
                         ggml_tensor * positions) const {
        for (int layer = 0; layer < config.n_layers; ++layer) {
            x = block(layer, x, context, t_mod, positions);
        }
        return x;
    }

    ggml_tensor * time_embedding(ggml_tensor * timestep) const {
        ggml_tensor * sinusoid = ggml_timestep_embedding(ctx(), timestep, config.freq_dim, TIMESTEP_MAX_PERIOD);
        return linear(ggml_silu(ctx(), linear(sinusoid, "dit.time_embd_1")), "dit.time_embd_2");
    }

    ggml_tensor * text_embedding(ggml_tensor * context) const {
        return linear(ggml_gelu(ctx(), linear(context, "dit.text_embd_1")), "dit.text_embd_2");
    }

    ggml_tensor * head(ggml_tensor * x, ggml_tensor * t) const {
        ggml_tensor * mod = ggml_add(ctx(), global("head.modulation"), t);
        ggml_tensor * out = modulate(x, row(mod, HEAD_SHIFT), row(mod, HEAD_SCALE));
        return linear(out, "dit.head");
    }

    ggml_tensor * forward(ggml_tensor * latents, ggml_tensor * timestep, ggml_tensor * context,
                          ggml_tensor * positions) const {
        ggml_tensor * x = ggml_cont(ctx(), ggml_transpose(ctx(), latents));
        x = linear(x, "dit.patch_embd");
        ggml_tensor * t = time_embedding(timestep);
        ggml_tensor * t_mod = ggml_reshape_2d(ctx(), linear(ggml_silu(ctx(), t), "dit.time_proj"),
                config.n_embd, MODULATION_ROWS);
        x = blocks(x, cache ? nullptr : text_embedding(context), t_mod, positions);
        return ggml_cont(ctx(), ggml_transpose(ctx(), head(x, t)));
    }
};

void require_linear(const SfxModel & model, const std::string & name, int64_t in, int64_t out) {
    require_matrix(model.tensor(name + ".weight"), in, out);
    require_vector(model.tensor(name + ".bias"), out, 1);
}

void require_norm_scales(const SfxModel & model, const std::string & prefix, int64_t width) {
    for (const char * name : {"self_norm_q", "self_norm_k", "cross_norm_q", "cross_norm_k", "cross_norm"}) {
        require_vector(model.tensor(prefix + name + ".weight"), width, 1);
    }
}

void require_norms(const SfxModel & model, const std::string & prefix, int64_t width) {
    require_norm_scales(model, prefix, width);
    require_vector(model.tensor(prefix + "cross_norm.bias"), width, 1);
}

void require_attention(const SfxModel & model, const std::string & prefix, int64_t width) {
    for (const char * name : {"self_q", "self_k", "self_v", "self_o", "cross_q", "cross_k", "cross_v", "cross_o"}) {
        require_linear(model, prefix + name, width, width);
    }
}

void validate_dit_layer(const SfxModel & model, const SfxDitConfig & config, int layer) {
    const std::string prefix = "dit.blk." + std::to_string(layer) + ".";
    require_attention(model, prefix, config.n_embd);
    require_norms(model, prefix, config.n_embd);
    require_linear(model, prefix + "ffn_up", config.n_embd, config.n_ff);
    require_linear(model, prefix + "ffn_down", config.n_ff, config.n_embd);
    require_vector(model.tensor(prefix + "modulation"), config.n_embd, MODULATION_ROWS);
}

void validate_dit_layers(const SfxModel & model, const SfxDitConfig & config) {
    for (int layer = 0; layer < config.n_layers; ++layer) {
        validate_dit_layer(model, config, layer);
    }
}

std::vector<int32_t> frame_positions(int frames) {
    std::vector<int32_t> positions((size_t) frames);
    for (int i = 0; i < frames; ++i) {
        positions[(size_t) i] = i;
    }
    return positions;
}

} // namespace

void validate_dit(const SfxModel & model) {
    const SfxDitConfig & config = model.config().dit;
    require_linear(model, "dit.patch_embd", config.in_channels, config.n_embd);
    require_linear(model, "dit.text_embd_1", config.text_dim, config.n_embd);
    require_linear(model, "dit.text_embd_2", config.n_embd, config.n_embd);
    require_linear(model, "dit.time_embd_1", config.freq_dim, config.n_embd);
    require_linear(model, "dit.time_embd_2", config.n_embd, config.n_embd);
    require_linear(model, "dit.time_proj", config.n_embd, (int64_t) config.n_embd * MODULATION_ROWS);
    require_linear(model, "dit.head", config.n_embd, config.out_channels);
    require_vector(model.tensor("dit.head.modulation"), config.n_embd, HEAD_MODULATION_ROWS);
    validate_dit_layers(model, config);
}

struct SfxDitSession::Impl {
    SfxModel & model;
    int frames;
    std::unique_ptr<SfxGraph> graph;
    std::unique_ptr<SfxDitCache> cache;
    std::unique_ptr<SfxGraph> preparation;
    std::vector<ggml_tensor *> context_inputs;
    std::vector<std::vector<float>> cached_contexts;
    bool prepared = false;
    bool measure_only;
    ggml_tensor * latents = nullptr;
    ggml_tensor * timestep = nullptr;
    ggml_tensor * context = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * context_index = nullptr;

    Impl(SfxModel & owner, int frame_count, bool measuring, int context_count)
        : model(owner), frames(frame_count), measure_only(measuring) {
        validate_dit(model);
        if (context_count < 0) throw std::runtime_error("moss sfx: invalid cache context count");
        if (context_count) build_cache(context_count);
        build_graph();
        if (!measure_only && !cache) model.allocate(*graph);
    }

    void build_graph() {
        graph = std::make_unique<SfxGraph>(DIT_GRAPH_NODES);
        const SfxConfig & config = model.config();
        DitGraph builder{model, config.dit, *graph};
        latents = graph->input_f32(frames, config.dit.in_channels);
        timestep = graph->input_f32(1, 1);
        if (cache) {
            context_index = graph->input_i32(1);
            builder.cache = cache.get();
            builder.context_index = context_index;
        } else {
            context = graph->input_f32(config.dit.text_dim, config.text.max_tokens);
        }
        positions = graph->input_i32(frames);
        output = builder.forward(latents, ggml_reshape_1d(graph->ctx(), timestep, 1), context, positions);
        ggml_set_output(output);
        ggml_build_forward_expand(graph->graph(), output);
    }

    void build_cache(int count) {
        cache = std::make_unique<SfxDitCache>(model, count, measure_only);
        build_preparation();
    }

    void build_preparation() {
        preparation.reset();
        preparation = std::make_unique<SfxGraph>(DIT_GRAPH_NODES);
        context_inputs.clear();
        const auto & config = model.config();
        DitGraph builder{model, config.dit, *preparation};
        for (int index = 0; index < cache->contexts(); ++index) {
            auto * input = preparation->input_f32(config.dit.text_dim, config.text.max_tokens);
            context_inputs.push_back(input);
            build_context(builder, input, index);
        }
    }

    void build_context(const DitGraph & builder, ggml_tensor * input, int index) {
        auto * embedding = builder.text_embedding(input);
        for (int layer = 0; layer < model.config().dit.n_layers; ++layer) {
            cache->store(*preparation, layer, index, builder.cross_key(layer, embedding),
                         builder.layer_linear(layer, "cross_v", embedding));
        }
    }

    void prepare_contexts(std::vector<std::vector<float>> contexts) {
        if (!cache || measure_only) throw std::runtime_error("moss sfx: session cannot prepare a DiT cache");
        if (contexts.size() != context_inputs.size()) throw std::runtime_error("moss sfx: wrong cache context count");
        for (const auto & data : contexts) {
            if (data.size() != context_elements()) throw std::runtime_error("moss sfx: DiT input has the wrong size");
        }
        prepared = false;
        graph.reset();
        build_preparation();
        model.allocate(*preparation);
        for (size_t i = 0; i < contexts.size(); ++i) {
            ggml_backend_tensor_set(context_inputs[i], contexts[i].data(), 0, contexts[i].size() * sizeof(float));
        }
        model.compute(*preparation);
        build_graph();
        model.allocate(*graph);
        cached_contexts = std::move(contexts);
        prepared = true;
    }

    size_t context_elements() const {
        return size_t(model.config().dit.text_dim) * model.config().text.max_tokens;
    }

    void require_sizes(const std::vector<float> & latent_data, const std::vector<float> & context_data) const {
        if (latent_data.size() != size_t(frames) * model.config().dit.in_channels ||
            context_data.size() != context_elements()) {
            throw std::runtime_error("moss sfx: DiT input has the wrong size");
        }
    }

    void set_context(const std::vector<float> & data) {
        if (!cache) {
            ggml_backend_tensor_set(context, data.data(), 0, data.size() * sizeof(float));
            return;
        }
        if (!prepared) throw std::runtime_error("moss sfx: DiT cache is not prepared");
        const auto found = std::find(cached_contexts.begin(), cached_contexts.end(), data);
        if (found == cached_contexts.end()) throw std::runtime_error("moss sfx: context is not in the DiT cache");
        const int32_t index = int32_t(found - cached_contexts.begin());
        ggml_backend_tensor_set(context_index, &index, 0, sizeof(index));
    }

    std::vector<float> velocity(const std::vector<float> & latent_data, float step,
                                const std::vector<float> & context_data) {
        require_sizes(latent_data, context_data);
        set_context(context_data);
        const std::vector<int32_t> position_data = frame_positions(frames);
        ggml_backend_tensor_set(positions, position_data.data(), 0, position_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(latents, latent_data.data(), 0, latent_data.size() * sizeof(float));
        ggml_backend_tensor_set(timestep, &step, 0, sizeof(float));
        model.compute(*graph);
        std::vector<float> result((size_t) ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
        return result;
    }

    SfxMemory measure() {
        auto result = model.measure(*graph);
        if (!cache) return result;
        const auto prep = model.measure(*preparation);
        result.persistent_device_bytes = cache->bytes();
        result.device_bytes =
            fitutil::sat_add(result.persistent_device_bytes, std::max(result.device_bytes, prep.device_bytes));
        result.host_compute_bytes = std::max(result.host_compute_bytes, prep.host_compute_bytes);
        result.host_bytes += prep.host_bytes + cache->host_bytes();
        result.host_bytes += context_elements() * cache->contexts() * sizeof(float);
        result.host_bytes += context_inputs.capacity() * sizeof(ggml_tensor *);
        result.host_bytes += size_t(cache->contexts()) * sizeof(std::vector<float>);
        return result;
    }
};

SfxDitSession::SfxDitSession(SfxModel & model, int frames, bool measure_only, int cached_contexts)
    : impl_(new Impl(model, frames, measure_only, cached_contexts)) {}

SfxDitSession::~SfxDitSession() = default;

SfxMemory SfxDitSession::measure() { return impl_->measure(); }

void SfxDitSession::prepare_contexts(std::vector<std::vector<float>> contexts) {
    impl_->prepare_contexts(std::move(contexts));
}

std::vector<float> SfxDitSession::velocity(const std::vector<float> & latents, float timestep,
                                           const std::vector<float> & context) {
    return impl_->velocity(latents, timestep, context);
}

} // namespace tts_cpp::moss::detail
