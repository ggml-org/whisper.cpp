#include "pocket/flow_lm.h"
#include "pocket/ggml_utils.h"
#include "pocket/backend.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "pocket/backend.h"
#include "gguf_stream.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>

namespace tts_cpp::pocket::detail {
namespace {
[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("pocket FlowLM: " + message);
}

void finite(const std::vector<float> & values) {
    if (!std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
        fail("input contains non-finite values");
}

void upload(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

} // namespace

struct FlowLM::Impl {
    FlowConfig cfg;
    int capacity = 0;
    int past = 0;
    int n_threads = 1;
    MemoryMeasure * measure = nullptr;
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_context * state = nullptr;
    ggml_backend_t backend = nullptr;
    // Session capabilities are immutable; avoid a metadata snapshot per frame.
    HexagonRuntime hexagon_caps;
    ggml_backend_buffer_t weights_buffer = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    std::vector<ggml_tensor *> keys, values;
    std::unordered_set<std::string> checked;

    ~Impl() {
        auto backend_lock = lock_backend(backend);
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (state) ggml_free(state);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) free_backend(backend);
    }

    int64_t key(const std::string & name, gguf_type type) const {
        const auto id = gguf_find_key(file, name.c_str());
        if (id < 0 || gguf_get_kv_type(file, id) != type) fail("missing or wrong-typed metadata: " + name);
        return id;
    }
    int integer(const char * name, int limit) const {
        const std::string full = std::string("pocket.") + name;
        const auto value = gguf_get_val_u32(file, key(full, GGUF_TYPE_UINT32));
        if (!value || value > static_cast<uint32_t>(limit)) fail("invalid metadata: " + full);
        return static_cast<int>(value);
    }
    void text_is(const char * name, const char * expected) const {
        if (std::string(gguf_get_val_str(file, key(name, GGUF_TYPE_STRING))) != expected)
            fail(std::string("unsupported ") + name);
    }
    ggml_tensor * tensor(const std::string & name) const {
        auto * t = ggml_get_tensor(weights ? weights : metadata, name.c_str());
        if (!t) fail("missing tensor: " + name);
        return t;
    }
    void shape(const std::string & name, int64_t n0, int64_t n1 = 1, bool matrix = false) {
        auto * t = tensor(name);
        if (t->ne[0] != n0 || t->ne[1] != n1 || t->ne[2] != 1 || t->ne[3] != 1 ||
            !(t->type == GGML_TYPE_F32 || (matrix && t->type == GGML_TYPE_F16)))
            fail("invalid tensor shape/type: " + name);
        checked.insert(name);
    }
    void linear_shape(const std::string & name, int in, int out, bool bias = true) {
        shape(name + ".weight", in, out, true);
        if (bias) shape(name + ".bias", out);
    }
    void norm_shape(const std::string & name, int width) {
        shape(name + ".weight", width);
        shape(name + ".bias", width);
    }
    void validate_tensors() {
        const int d = cfg.dim, f = cfg.flow_dim, l = cfg.latent_dim;
        shape("conditioner.embed.weight", d, cfg.vocab_size + 1, true);
        shape("bos_emb", l); shape("emb_mean", l); shape("emb_std", l);
        if (cfg.bos_before_voice) shape("bos_before_voice", d);
        if (ggml_get_tensor(metadata, "speaker_proj_weight")) shape("speaker_proj_weight", l, d, true);
        linear_shape("input_linear", l, d, false);
        norm_shape("out_norm", d);
        linear_shape("out_eos", d, 1);
        for (int i = 0; i < cfg.layers; ++i) {
            const auto p = "transformer.layers." + std::to_string(i);
            norm_shape(p + ".norm1", d); norm_shape(p + ".norm2", d);
            linear_shape(p + ".self_attn.in_proj", d, 3 * d, false);
            linear_shape(p + ".self_attn.out_proj", d, d, false);
            linear_shape(p + ".linear1", d, cfg.ff_dim, false);
            linear_shape(p + ".linear2", cfg.ff_dim, d, false);
        }
        linear_shape("flow_net.input_proj", l, f);
        linear_shape("flow_net.cond_embed", d, f);
        for (int i = 0; i < 2; ++i) {
            const auto p = "flow_net.time_embed." + std::to_string(i);
            shape(p + ".freqs", 128);
            linear_shape(p + ".mlp.0", 256, f);
            linear_shape(p + ".mlp.2", f, f);
            shape(p + ".mlp.3.alpha", f);
        }
        for (int i = 0; i < cfg.flow_depth; ++i) {
            const auto p = "flow_net.res_blocks." + std::to_string(i);
            norm_shape(p + ".in_ln", f);
            linear_shape(p + ".mlp.0", f, f);
            linear_shape(p + ".mlp.2", f, f);
            linear_shape(p + ".adaLN_modulation.1", f, 3 * f);
        }
        linear_shape("flow_net.final_layer.linear", f, l);
        linear_shape("flow_net.final_layer.adaLN_modulation.1", f, 2 * f);
        if (checked.size() != static_cast<size_t>(gguf_get_n_tensors(file))) fail("unrecognized tensors in FlowLM artifact");
    }

    void load(const std::string & path, int context, int threads, MemoryMeasure * price = nullptr,
              ggml_backend_dev_t device = nullptr) {
        measure = price; n_threads = threads;
        if (context < 1 || context > 8192 || threads < 1 || threads > 1024)
            fail("context must be 1..8192 and threads 1..1024");
        capacity = context;
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) fail("cannot read GGUF: " + path);
        text_is("general.architecture", "pocket-tts-flow-lm");
        if (integer("schema_version", 1) != 1) fail("unsupported schema version");
        text_is("pocket.flow_type", "lsd");
        cfg.dim = integer("dim", 4096); cfg.heads = integer("heads", 128);
        cfg.layers = integer("layers", 48); cfg.ff_dim = integer("ff_dim", 16384);
        cfg.latent_dim = integer("latent_dim", 512); cfg.flow_dim = integer("flow_dim", 4096);
        cfg.flow_depth = integer("flow_depth", 32); cfg.vocab_size = integer("vocab_size", 65536);
        cfg.rope_base = gguf_get_val_f32(file, key("pocket.rope_base", GGUF_TYPE_FLOAT32));
        cfg.bos_before_voice = gguf_get_val_bool(file, key("pocket.bos_before_voice", GGUF_TYPE_BOOL));
        if (cfg.dim % cfg.heads || (cfg.dim / cfg.heads) % 2 || cfg.flow_dim < 2 ||
            !std::isfinite(cfg.rope_base) || cfg.rope_base <= 0) fail("invalid architecture dimensions");
        validate_tensors(); // All validation precedes device allocations and graph construction.
        backend = init_stage_backend(device);
        auto backend_lock = lock_backend(backend);
        if (::tts_cpp::detail::backend_is_hexagon(backend)) hexagon_caps = hexagon_runtime(backend);
        ::tts_cpp::detail::backend_set_n_threads(backend, threads);
        // This baseline always computes in F32. Expanding half storage once
        // avoids ggml CPU's F16 matmul rounding *activations* to F16, which
        // noticeably changes the EOS head on the released checkpoint.
        weights = ggml_init({static_cast<size_t>(gguf_get_n_tensors(file) + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) fail("weight context allocation failed");
        for (auto * t = ggml_get_first_tensor(metadata); t; t = ggml_get_next_tensor(metadata, t)) {
            auto * destination = ggml_new_tensor(weights, GGML_TYPE_F32, GGML_MAX_DIMS, t->ne);
            ggml_set_name(destination, ggml_get_name(t));
        }
        if (!measure) {
        (void)price_hexagon_context(backend, weights);
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        if (!weights_buffer) fail("weight allocation failed");
        ::tts_cpp::detail::gguf_stream_reader reader(file, path);
        for (auto * t = ggml_get_first_tensor(weights); t; t = ggml_get_next_tensor(weights, t)) {
            auto * source = ggml_get_tensor(metadata, ggml_get_name(t));
            load_float_weight(reader, source, t);
        }
        const auto scales = pocket_read(tensor("emb_std"));
        for (float x : scales) if (!std::isfinite(x) || x <= 0) fail("invalid latent normalization");
        }
        state = ggml_init({static_cast<size_t>(2 * cfg.layers + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!state) fail("cache context allocation failed");
        for (int i = 0; i < cfg.layers; ++i) {
            keys.push_back(ggml_new_tensor_3d(state, GGML_TYPE_F32, cfg.dim / cfg.heads, cfg.heads, capacity));
            values.push_back(ggml_new_tensor_3d(state, GGML_TYPE_F32, cfg.dim / cfg.heads, cfg.heads, capacity));
        }
        if (measure) {
            price_persistent(backend, metadata, weights, state, file, *measure);
            return;
        }
        (void)price_hexagon_context(backend, state);
        state_buffer = ggml_backend_alloc_ctx_tensors(state, backend);
        if (!state_buffer) fail("KV cache allocation failed");
    }

    ggml_tensor * linear(ggml_context * c, const std::string & p, ggml_tensor * x, bool bias = true) const {
        x = pocket_mm(c, tensor(p + ".weight"), x);
        return bias ? ggml_add(c, x, tensor(p + ".bias")) : x;
    }
    ggml_tensor * voice_projection(PocketFlowGraph & g, ggml_tensor * input) const {
        return pocket_mm(g.ctx, tensor("speaker_proj_weight"), input);
    }
    ggml_tensor * audio_projection(PocketFlowGraph & g, ggml_tensor * input) const {
        return linear(g.ctx, "input_linear", input, false);
    }
    ggml_tensor * norm(ggml_context * c, const std::string & p, ggml_tensor * x, float eps) const {
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x, eps), tensor(p + ".weight")), tensor(p + ".bias"));
    }
    ggml_tensor * cache_view(ggml_context * c, ggml_tensor * cache, int start, int count) const {
        return ggml_view_3d(c, cache, cfg.dim / cfg.heads, cfg.heads, count,
                            cache->nb[1], cache->nb[2], static_cast<size_t>(start) * cache->nb[2]);
    }

    std::vector<float> backbone(const std::vector<float> & input, int price_frames = 0) {
        finite(input);
        if (!measure && (input.empty() || input.size() % cfg.dim)) fail("expected non-empty [frames, dim] embeddings");
        const size_t count = measure ? size_t(price_frames) : input.size() / cfg.dim;
        if (!count) fail("empty graph workload");
        if (count > static_cast<size_t>(capacity - past)) fail("KV context capacity exceeded");
        const int n = static_cast<int>(count), hd = cfg.dim / cfg.heads, total = past + n;
        PocketFlowGraph g(backend, &hexagon_caps); auto * c = g.ctx;
        auto * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, cfg.dim, n); ggml_set_input(in);
        auto * positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(positions);
        ggml_tensor * causal_mask = nullptr;
        if (::tts_cpp::detail::backend_is_opencl(backend) || ::tts_cpp::detail::backend_is_hexagon(backend)) {
            // Hexagon does not implement DIAG_MASK_INF; the reviewed OpenCL
            // DIAG_MASK_INF vector kernel overwrites
            // neighboring float4 vectors when total is divisible by eight.
            // Use the explicit-mask softmax path, also used by Mimi.
            causal_mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, total, n);
            ggml_set_input(causal_mask);
        }
        auto * x = in;
        for (int i = 0; i < cfg.layers; ++i) {
            const auto p = "transformer.layers." + std::to_string(i);
            auto * qkv = linear(c, p + ".self_attn.in_proj", norm(c, p + ".norm1", x, 1e-5f), false);
            auto part = [&](int index) {
                return ggml_reshape_3d(c, ggml_cont(c, ggml_view_2d(c, qkv, cfg.dim, n, qkv->nb[1],
                                           static_cast<size_t>(index * cfg.dim) * sizeof(float))), hd, cfg.heads, n);
            };
            auto rotate = [&](ggml_tensor * t) {
                return ggml_rope_ext(c, t, positions, nullptr, hd, 0, 0, cfg.rope_base, 1, 0, 1, 0, 0);
            };
            auto * q = rotate(part(0)); auto * k = rotate(part(1)); auto * v = part(2);
            ggml_build_forward_expand(g.graph, ggml_cpy(c, k, cache_view(c, keys[i], past, n)));
            ggml_build_forward_expand(g.graph, ggml_cpy(c, v, cache_view(c, values[i], past, n)));
            auto * all_k = ggml_permute(c, cache_view(c, keys[i], 0, total), 0, 2, 1, 3);
            auto * all_v = ggml_cont(c, ggml_permute(c, cache_view(c, values[i], 0, total), 1, 2, 0, 3));
            auto * scores = pocket_mm(c, all_k, ggml_permute(c, q, 0, 2, 1, 3));
            const float attention_scale = 1.0f / std::sqrt(static_cast<float>(hd));
            auto * probabilities = causal_mask ? ggml_soft_max_ext(c, scores, causal_mask, attention_scale, 0) :
                ggml_soft_max(c, ggml_diag_mask_inf(c, ggml_scale(c, scores, attention_scale), past));
            auto * merged = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, pocket_mm(c, all_v, probabilities), 0, 2, 1, 3)), cfg.dim, n);
            x = ggml_add(c, x, linear(c, p + ".self_attn.out_proj", merged, false));
            auto * ff = linear(c, p + ".linear1", norm(c, p + ".norm2", x, 1e-5f), false);
            x = ggml_add(c, x, linear(c, p + ".linear2", pocket_gelu(c, ff), false));
        }
        auto * output = norm(c, "out_norm", x, 1e-5f);
        if (measure) {
            measure->compute = std::max(measure->compute, g.prepare(output, true, n_threads, &measure->opencl_scratch));
            if (causal_mask) measure->host_compute = std::max(measure->host_compute, uint64_t(total)*n*sizeof(float));
            return {};
        }
        g.prepare(output);
        upload(in, input);
        std::vector<int32_t> pos(n);
        for (int i = 0; i < n; ++i) pos[i] = past + i;
        ggml_backend_tensor_set(positions, pos.data(), 0, n * sizeof(int32_t));
        if (causal_mask) {
            std::vector<float> mask(static_cast<size_t>(total)*n);
            for (int i = 0; i < n; ++i) for (int j = 0; j < total; ++j)
                mask[static_cast<size_t>(i)*total+j] = j <= past+i ? 0.0f : -INFINITY;
            ggml_backend_tensor_set(causal_mask, mask.data(), 0, mask.size()*sizeof(float));
        }
        // Failed computations may have written cache rows. Discard the prefix
        // rather than treating a partially written cache as valid.
        try {
            auto result = g.compute(backend, output);
            past = total;
            return result;
        } catch (...) { past = 0; throw; }
    }

    std::vector<float> flow(const std::vector<float> & condition, const std::vector<float> & noise, int steps) {
        if (condition.size() != static_cast<size_t>(cfg.dim) || noise.size() != static_cast<size_t>(cfg.latent_dim) ||
            steps < 1 || steps > 64) fail("invalid flow inputs or steps (expected 1..64)");
        finite(condition); finite(noise);
        PocketFlowGraph g(backend, &hexagon_caps); auto * c = g.ctx;
        auto * hidden = ggml_new_tensor_2d(c, GGML_TYPE_F32, cfg.dim, 1); ggml_set_input(hidden);
        auto * latent = ggml_new_tensor_2d(c, GGML_TYPE_F32, cfg.latent_dim, 1); ggml_set_input(latent);
        ggml_tensor * times[2]; ggml_tensor * embedded[2];
        for (int i = 0; i < 2; ++i) {
            const auto p = "flow_net.time_embed." + std::to_string(i);
            times[i] = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 1); ggml_set_input(times[i]);
            auto * h = linear(c, p + ".mlp.2", ggml_silu(c, linear(c, p + ".mlp.0", times[i])));
            // Kyutai calls this RMSNorm but divides by the sample VARIANCE
            // (unbiased, mean-subtracted), while multiplying the original h.
            auto * centered = ggml_sub(c, h, pocket_mean(c, h, backend));
            auto * variance = ggml_scale_bias(c, pocket_mean(c, ggml_sqr(c, centered), backend),
                                               float(cfg.flow_dim) / float(cfg.flow_dim - 1), 1e-5f);
            embedded[i] = ggml_mul(c, ggml_div(c, h, ggml_sqrt(c, variance)), tensor(p + ".mlp.3.alpha"));
        }
        auto * y = ggml_add(c, linear(c, "flow_net.cond_embed", hidden), ggml_scale(c, ggml_add(c, embedded[0], embedded[1]), 0.5f));
        auto * activated_y = ggml_silu(c, y);
        auto * x = linear(c, "flow_net.input_proj", latent);
        auto slice = [&](ggml_tensor * t, int index) {
            return ggml_view_1d(c, t, cfg.flow_dim, static_cast<size_t>(index * cfg.flow_dim) * sizeof(float));
        };
        auto modulate = [&](ggml_tensor * value, ggml_tensor * parameters) {
            return ggml_add(c, ggml_mul(c, value, ggml_scale_bias(c, slice(parameters, 1), 1, 1)), slice(parameters, 0));
        };
        for (int i = 0; i < cfg.flow_depth; ++i) {
            const auto p = "flow_net.res_blocks." + std::to_string(i);
            auto * modulation = linear(c, p + ".adaLN_modulation.1", activated_y);
            auto * h = modulate(norm(c, p + ".in_ln", x, 1e-6f), modulation);
            h = linear(c, p + ".mlp.2", ggml_silu(c, linear(c, p + ".mlp.0", h)));
            x = ggml_add(c, x, ggml_mul(c, slice(modulation, 2), h));
        }
        auto * modulation = linear(c, "flow_net.final_layer.adaLN_modulation.1", activated_y);
        auto * output = linear(c, "flow_net.final_layer.linear", modulate(ggml_norm(c, x, 1e-6f), modulation));
        if (measure) {
            measure->compute = std::max(measure->compute, g.prepare(output, true, n_threads, &measure->opencl_scratch));
            return {};
        }
        g.prepare(output);
        std::vector<float> current = noise;
        const auto freq0 = pocket_read(tensor("flow_net.time_embed.0.freqs"));
        const auto freq1 = pocket_read(tensor("flow_net.time_embed.1.freqs"));
        for (int step = 0; step < steps; ++step) {
            // Graph arenas may reuse input buffers after their last consumer.
            // Restore every input, even the unchanged conditioning vector.
            upload(hidden, condition);
            upload(latent, current);
            for (int i = 0; i < 2; ++i) {
                const auto & frequencies = i ? freq1 : freq0;
                const float time = float(step + i) / float(steps);
                std::vector<float> embedding(256);
                for (int j = 0; j < 128; ++j) {
                    embedding[j] = std::cos(time * frequencies[j]);
                    embedding[j + 128] = std::sin(time * frequencies[j]);
                }
                upload(times[i], embedding);
            }
            const auto velocity = g.compute(backend, output);
            for (size_t i = 0; i < current.size(); ++i) current[i] += velocity[i] / float(steps);
        }
        finite(current);
        return current;
    }
};

FlowLM::FlowLM(const std::string & path, int context, int threads, ggml_backend_dev_t device) : impl_(std::make_unique<Impl>()) {
    impl_->load(path, context, threads, nullptr, device);
}
FlowMemory FlowLM::measure(const std::string & path, int context, int prefill_frames, int threads, ggml_backend_dev_t device) {
    if (prefill_frames < 1 || prefill_frames > context) fail("invalid prefill shape");
    FlowMemory out;
    Impl model; model.load(path, context, threads, &out, device);
    out.config = model.cfg;
    out.source_hash = gguf_get_val_str(model.file, model.key("pocket.source_sha256", GGUF_TYPE_STRING));
    // Bound every permitted cache occupancy, including decoding at capacity.
    model.past = context - prefill_frames;
    model.backbone({}, prefill_frames);
    model.past = context - 1;
    model.backbone({}, 1);
    // advance() keeps its projection arena alive while backbone() runs.
    PocketFlowGraph projection(model.backend, &model.hexagon_caps);
    auto * input = ggml_new_tensor_2d(projection.ctx, GGML_TYPE_F32, model.cfg.latent_dim, 1);
    ggml_set_input(input);
    out.compute += projection.prepare(model.audio_projection(projection, input), true, threads, &out.opencl_scratch);
    model.flow(std::vector<float>(model.cfg.dim), std::vector<float>(model.cfg.latent_dim), 1);
    PocketFlowGraph voice_projection(model.backend, &model.hexagon_caps);
    auto * voice_input = ggml_new_tensor_2d(voice_projection.ctx, GGML_TYPE_F32, model.cfg.latent_dim, 375);
    ggml_set_input(voice_input);
    out.compute = std::max(out.compute, voice_projection.prepare(model.voice_projection(voice_projection, voice_input), true, threads, &out.opencl_scratch));
    out.graph_metadata = 2 * PocketFlowGraph::metadata_bytes();
    return out;
}
FlowLM::~FlowLM() = default;
const char * FlowLM::backend_name() const noexcept { return ggml_backend_name(impl_->backend); }
const FlowConfig & FlowLM::config() const { return impl_->cfg; }
int FlowLM::position() const { return impl_->past; }
void FlowLM::reset() { impl_->past = 0; }
std::vector<float> FlowLM::prefill(const std::vector<float> & embeddings) { return impl_->backbone(embeddings); }

std::vector<float> FlowLM::text_embeddings(const std::vector<int> & ids) const {
    auto backend_lock = lock_backend(impl_->backend);
    std::vector<float> result;
    if (ids.size() > static_cast<size_t>(impl_->capacity)) fail("text exceeds context capacity");
    for (int id : ids) {
        if (id < 0 || id >= config().vocab_size) fail("invalid text token id");
        const auto row = pocket_read(impl_->tensor("conditioner.embed.weight"),
                                      static_cast<size_t>(id) * config().dim, config().dim);
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

FrameCondition FlowLM::advance(const std::vector<float> & latent) {
    if (position() >= impl_->capacity) fail("KV context capacity exceeded");
    if (!latent.empty() && latent.size() != static_cast<size_t>(config().latent_dim)) fail("invalid latent dimensions");
    finite(latent);
    PocketFlowGraph g(impl_->backend, &impl_->hexagon_caps);
    auto * input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, config().latent_dim, 1); ggml_set_input(input);
    auto * output = impl_->audio_projection(g, input);
    g.prepare(output);
    upload(input, latent.empty() ? pocket_read(impl_->tensor("bos_emb")) : latent);
    FrameCondition result;
    result.hidden = impl_->backbone(g.compute(impl_->backend, output));
    // A small dot product avoids constructing a second graph for one logit.
    const auto w = pocket_read(impl_->tensor("out_eos.weight"));
    result.eos_logit = pocket_read(impl_->tensor("out_eos.bias"))[0];
    for (size_t i = 0; i < w.size(); ++i) result.eos_logit += w[i] * result.hidden[i];
    if (!std::isfinite(result.eos_logit)) { reset(); fail("non-finite EOS logit"); }
    return result;
}

std::vector<float> FlowLM::sample(const std::vector<float> & hidden, const std::vector<float> & noise, int steps) {
    return impl_->flow(hidden, noise, steps);
}
std::vector<float> FlowLM::denormalize(const std::vector<float> & latent) const {
    auto backend_lock = lock_backend(impl_->backend);
    if (latent.size() != static_cast<size_t>(config().latent_dim)) fail("invalid latent dimensions");
    finite(latent);
    auto result = latent;
    const auto mean = pocket_read(impl_->tensor("emb_mean")), std = pocket_read(impl_->tensor("emb_std"));
    for (size_t i = 0; i < result.size(); ++i) result[i] = result[i] * std[i] + mean[i];
    finite(result);
    return result;
}

std::string FlowLM::source_hash() const {
    return gguf_get_val_str(impl_->file, impl_->key("pocket.source_sha256", GGUF_TYPE_STRING));
}
namespace {
VoiceState read_voice_file(const std::string & path, const FlowConfig & cfg, int capacity,
                           const std::string & source_hash, bool load_payload) {
    ggml_context * metadata = nullptr;
    auto * raw = gguf_init_from_file(path.c_str(), {true, &metadata});
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(raw, gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(metadata, ggml_free);
    if (!raw || !metadata) fail("cannot read prepared voice");
    auto key = [&](const char * name, gguf_type type) {
        const auto k = gguf_find_key(raw, name);
        if (k < 0 || gguf_get_kv_type(raw, k) != type) fail(std::string("invalid voice metadata: ")+name);
        return k;
    };
    if (std::string(gguf_get_val_str(raw, key("general.architecture", GGUF_TYPE_STRING))) != "pocket-tts-voice" ||
        gguf_get_val_u32(raw, key("pocket.schema_version", GGUF_TYPE_UINT32)) != 1) fail("unsupported prepared voice");
    VoiceState voice;
    voice.source_hash = gguf_get_val_str(raw, key("pocket.source_sha256", GGUF_TYPE_STRING));
    const auto frames = gguf_get_val_u32(raw, key("pocket.voice_frames", GGUF_TYPE_UINT32));
    if (voice.source_hash != source_hash) fail("voice and FlowLM checkpoint hashes differ");
    if (!frames || frames >= static_cast<unsigned>(capacity)) fail("prepared voice exceeds context capacity");
    voice.frames = frames;
    if (gguf_get_n_tensors(raw) != 2*cfg.layers) fail("unexpected prepared voice tensors");
    ::tts_cpp::detail::gguf_stream_reader reader(raw, path);
    for (int i = 0; i < cfg.layers; ++i) for (int kv = 0; kv < 2; ++kv) {
        const auto name = "layer."+std::to_string(i)+(kv ? ".value" : ".key");
        auto * t = ggml_get_tensor(metadata, name.c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != cfg.dim/cfg.heads ||
            t->ne[1] != cfg.heads || t->ne[2] != voice.frames || t->ne[3] != 1) fail("invalid prepared voice cache");
        if (!load_payload) continue;
        std::vector<float> data(ggml_nelements(t));
        if (!reader.to_host(name.c_str(), data.data(), data.size()*sizeof(float))) fail("cannot read voice cache");
        finite(data); (kv ? voice.values : voice.keys).push_back(std::move(data));
    }
    return voice;
}
}
VoiceState FlowLM::read_voice(const std::string & path) const {
    return read_voice_file(path, config(), impl_->capacity, source_hash(), true);
}
int FlowLM::measure_voice(const std::string & path, const FlowMemory & model, int context) {
    return read_voice_file(path, model.config, context, model.source_hash, false).frames;
}
VoiceState FlowLM::capture_voice() const {
    auto backend_lock = lock_backend(impl_->backend);
    if (position() < 1) fail("cannot capture an empty voice prefix");
    VoiceState result; result.source_hash = source_hash(); result.frames = position();
    for (int i = 0; i < config().layers; ++i) {
        result.keys.push_back(pocket_read(impl_->keys[i], 0, result.frames*config().dim));
        result.values.push_back(pocket_read(impl_->values[i], 0, result.frames*config().dim));
    }
    return result;
}
void FlowLM::restore_voice(const VoiceState & voice) {
    auto backend_lock = lock_backend(impl_->backend);
    if (voice.source_hash != source_hash() || voice.frames < 1 || voice.frames >= impl_->capacity ||
        voice.keys.size() != size_t(config().layers) || voice.values.size() != voice.keys.size()) fail("invalid voice state");
    for (int i = 0; i < config().layers; ++i) {
        if (voice.keys[i].size() != size_t(voice.frames*config().dim) || voice.values[i].size() != voice.keys[i].size()) fail("invalid voice state dimensions");
        finite(voice.keys[i]); finite(voice.values[i]);
    }
    reset();
    for (int i = 0; i < config().layers; ++i) { upload(impl_->keys[i], voice.keys[i]); upload(impl_->values[i], voice.values[i]); }
    impl_->past = voice.frames;
}
std::vector<float> FlowLM::voice_embeddings(const std::vector<float> & latents) const {
    if (latents.empty() || latents.size()%config().latent_dim || latents.size()/config().latent_dim > 375) fail("invalid voice latents");
    finite(latents);
    PocketFlowGraph g(impl_->backend, &impl_->hexagon_caps);
    auto * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, config().latent_dim, latents.size()/config().latent_dim);
    ggml_set_input(x); auto * output = impl_->voice_projection(g, x);
    g.prepare(output); upload(x, latents); auto result = g.compute(impl_->backend, output);
    if (config().bos_before_voice) {
        auto bos = pocket_read(impl_->tensor("bos_before_voice")); result.insert(result.begin(), bos.begin(), bos.end());
    }
    return result;
}
} // namespace tts_cpp::pocket::detail
