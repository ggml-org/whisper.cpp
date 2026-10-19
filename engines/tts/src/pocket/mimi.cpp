#include "pocket/mimi.h"
#include "pocket/backend.h"
#include "pocket/ggml_utils.h"
#include "backend_selection.h"
#include "backend_util.h"
#include "pocket/backend.h"
#include "json.hpp"

#include <climits>
#include <map>
#include <set>

namespace tts_cpp::pocket::detail {
namespace {
[[noreturn]] void fail(const std::string & s) { throw std::runtime_error("pocket Mimi: " + s); }
constexpr int dim = 512, latent = Mimi::latent_dim(), heads = 8, hd = 64, layers = 2, window = 250;
constexpr int rate = Mimi::sample_rate(), hop = Mimi::frame_samples(), up = 16;
constexpr int ratios[] = {6, 5, 4};
}

struct Mimi::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr, * weights = nullptr, * state = nullptr;
    ggml_backend_t backend = nullptr;
    // Session capabilities are immutable; avoid a metadata snapshot per graph.
    HexagonRuntime hexagon_caps;
    ggml_backend_buffer_t wb = nullptr, sb = nullptr;
    std::string hash;
    std::set<std::string> checked;
    std::map<std::string, ggml_tensor *> histories;
    ggml_tensor * keys[2][layers] = {}, * values[2][layers] = {};
    int positions[2] = {};
    bool encoder_available = false;
    int n_threads = 1;
    MemoryMeasure * measure = nullptr;

    ~Impl() {
        auto backend_lock = lock_backend(backend);
        if (sb) ggml_backend_buffer_free(sb);
        if (wb) ggml_backend_buffer_free(wb);
        if (state) ggml_free(state);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) free_backend(backend);
    }
    int64_t key(const char * name, gguf_type type) {
        const auto k = gguf_find_key(file, name);
        if (k < 0 || gguf_get_kv_type(file, k) != type) fail(std::string("invalid metadata: ") + name);
        return k;
    }
    std::string text(const char * name) { return gguf_get_val_str(file, key(name, GGUF_TYPE_STRING)); }
    ggml_tensor * tensor(const std::string & name) {
        auto * t = ggml_get_tensor(weights ? weights : metadata, name.c_str());
        if (!t) fail("missing tensor: " + name);
        return t;
    }
    void shape(const std::string & name, int64_t a, int64_t b = 1, int64_t c = 1) {
        auto * t = tensor(name);
        if (t->type != GGML_TYPE_F32 || t->ne[0] != a || t->ne[1] != b || t->ne[2] != c || t->ne[3] != 1)
            fail("invalid tensor shape/type: " + name);
        checked.insert(name);
    }
    void conv_shape(const std::string & p, int in, int out, int kernel, int stride = 1,
                    bool transposed = false, bool bias = true) {
        shape(p + ".weight", in, out, kernel);
        if (bias) shape(p + ".bias", out);
        const int history = transposed ? kernel / stride - 1 : kernel - stride;
        if (history) histories[p] = ggml_new_tensor_2d(state, GGML_TYPE_F32, in, history);
    }
    void residual_shapes(const std::string & p, int width) {
        conv_shape(p + ".block.1.conv", width, width / 2, 3);
        conv_shape(p + ".block.3.conv", width / 2, width, 1);
    }
    void load(const std::string & path, int threads, MemoryMeasure * price = nullptr,
              ggml_backend_dev_t device = nullptr) {
        measure = price; n_threads = threads;
        if (threads < 1 || threads > 1024) fail("threads must be 1..1024");
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) fail("cannot read GGUF: " + path);
        if (text("general.architecture") != "pocket-tts-mimi" ||
            gguf_get_val_u32(file, key("pocket.schema_version", GGUF_TYPE_UINT32)) != 1) fail("unsupported artifact");
        hash = text("pocket.source_sha256");
        if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos) fail("invalid source hash");
        const auto cfg = nlohmann::json::parse(text("pocket.mimi_config"));
        const auto & s = cfg.at("seanet"), & t = cfg.at("transformer"), & q = cfg.at("quantizer");
        // Version 1 describes the released English codec. Explicit validation
        // prevents silently interpreting a future codec with different padding.
        if (cfg.at("sample_rate") != rate || cfg.at("frame_rate") != double(rate)/hop || cfg.at("channels") != 1 ||
            cfg.at("inner_dim") != latent || cfg.at("outer_dim") != dim ||
            s.at("dimension") != dim || s.at("channels") != 1 || s.at("n_filters") != 64 ||
            s.at("n_residual_layers") != 1 || s.at("ratios") != std::vector<int>({6,5,4}) ||
            s.at("kernel_size") != 7 || s.at("residual_kernel_size") != 3 || s.at("last_kernel_size") != 3 ||
            s.at("dilation_base") != 2 || s.at("compress") != 2 || s.at("pad_mode") != "constant" ||
            t.at("d_model") != dim || t.at("num_heads") != heads || t.at("num_layers") != layers ||
            t.at("context") != window || t.at("dim_feedforward") != 2048 || t.at("input_dimension") != dim ||
            t.at("output_dimensions") != std::vector<int>({dim}) || t.at("max_period") != 10000 ||
            !t.at("layer_scale").is_number() || !std::isfinite(t.at("layer_scale").get<double>()) ||
            q.at("dimension") != latent || q.at("output_dimension") != dim)
            fail("unsupported codec configuration");
        state = ggml_init({128 * ggml_tensor_overhead(), nullptr, true});
        if (!state) fail("state allocation failed");
        shape("quantizer.output_proj.weight", latent, dim);
        shape("upsample.convtr.convtr.weight", dim, 2 * up);
        histories["upsample"] = ggml_new_tensor_2d(state, GGML_TYPE_F32, dim, 1);
        conv_shape("decoder.model.0.conv", dim, dim, 7);
        int width = dim;
        for (int i = 0; i < 3; ++i) {
            conv_shape("decoder.model." + std::to_string(2 + 3*i) + ".convtr", width, width/2, 2*ratios[i], ratios[i], true);
            width /= 2;
            residual_shapes("decoder.model." + std::to_string(3 + 3*i), width);
        }
        conv_shape("decoder.model.11.conv", 64, 1, 3);
        conv_shape("encoder.model.0.conv", 1, 64, 7);
        width = 64;
        for (int i = 0; i < 3; ++i) {
            residual_shapes("encoder.model." + std::to_string(1 + 3*i), width);
            conv_shape("encoder.model." + std::to_string(3 + 3*i) + ".conv", width, width*2, 2*ratios[2-i], ratios[2-i]);
            width *= 2;
        }
        conv_shape("encoder.model.11.conv", dim, dim, 3);
        conv_shape("downsample.conv.conv", dim, latent, 2*up, up, false, false);
        for (int enc = 0; enc < 2; ++enc) for (int i = 0; i < layers; ++i) {
            const std::string p = std::string(enc ? "enc" : "dec") + ".layers." + std::to_string(i);
            shape(p + ".self_attn.in_proj.weight", dim, 3*dim);
            shape(p + ".self_attn.out_proj.weight", dim, dim);
            shape(p + ".linear1.weight", dim, 2048); shape(p + ".linear2.weight", 2048, dim);
            for (int n : {1,2}) {
                shape(p + ".norm" + std::to_string(n) + ".weight", dim);
                shape(p + ".norm" + std::to_string(n) + ".bias", dim);
                shape(p + ".layer_scale_" + std::to_string(n) + ".scale", dim);
            }
            keys[enc][i] = ggml_new_tensor_3d(state, GGML_TYPE_F32, hd, heads, window-1);
            values[enc][i] = ggml_new_tensor_3d(state, GGML_TYPE_F32, hd, heads, window-1);
        }
        if (checked.size() != static_cast<size_t>(gguf_get_n_tensors(file))) fail("unexpected codec tensors");
        backend = init_stage_backend(device);
        auto backend_lock = lock_backend(backend);
        if (::tts_cpp::detail::backend_is_hexagon(backend)) hexagon_caps = hexagon_runtime(backend);
        ::tts_cpp::detail::backend_set_n_threads(backend, threads);
        weights = ggml_init({(checked.size() + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) fail("weight context allocation failed");
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            auto * dst = ggml_new_tensor(weights, GGML_TYPE_F32, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
        if (measure) {
            price_persistent(backend, metadata, weights, state, file, *measure);
            measure->source_hash = hash;
            return;
        }
        (void)price_hexagon_context(backend, weights);
        (void)price_hexagon_context(backend, state);
        wb = ggml_backend_alloc_ctx_tensors(weights, backend);
        sb = ggml_backend_alloc_ctx_tensors(state, backend);
        if (!wb || !sb) fail("backend allocation failed");
        ::tts_cpp::detail::gguf_stream_reader reader(file, path);
        for (const auto & name : checked) load_float_weight(reader, ggml_get_tensor(metadata, name.c_str()), tensor(name));
        const auto first_encoder_weight = pocket_read(tensor("encoder.model.0.conv.weight"));
        encoder_available = std::any_of(first_encoder_weight.begin(), first_encoder_weight.end(), [](float x) { return x != 0; });
        ggml_backend_buffer_clear(sb, 0);
    }
    void reset(bool enc) {
        auto backend_lock = lock_backend(backend);
        positions[enc] = 0;
        for (const auto & item : histories) {
            const bool is_enc = item.first.find("encoder.") == 0 || item.first.find("downsample.") == 0;
            if (is_enc == enc) {
                std::vector<float> zero(ggml_nelements(item.second), 0);
                ggml_backend_tensor_set(item.second, zero.data(), 0, zero.size()*sizeof(float));
            }
        }
    }
    ggml_tensor * slice(ggml_context * c, ggml_tensor * x, int begin, int count) {
        return ggml_view_2d(c, x, x->ne[0], count, x->nb[1], begin*x->nb[1]);
    }
    ggml_tensor * history_input(PocketGraph & g, const std::string & p, ggml_tensor * x, bool replicate = false) {
        auto it = histories.find(p);
        if (it == histories.end()) return x;
        auto * h = it->second;
        auto * prefix = replicate ? ggml_repeat(g.ctx, slice(g.ctx, x, 0, 1), h) : h;
        auto * combined = ggml_concat(g.ctx, prefix, x, 1);
        g.updates.push_back(ggml_cpy(g.ctx, slice(g.ctx, combined, combined->ne[1]-h->ne[1], h->ne[1]), h));
        return combined;
    }
    ggml_tensor * conv(PocketGraph & g, const std::string & p, ggml_tensor * x, int stride = 1, bool bias = true, bool replicate = false) {
        auto * c = g.ctx; auto * w = tensor(p + ".weight");
        const int count = x->ne[1] / stride, ic = w->ne[0], oc = w->ne[1], kernel = w->ne[2];
        if (x->ne[1] % stride || !count || x->ne[0] != ic) fail("invalid convolution input");
        x = history_input(g, p, x, replicate);
        ggml_tensor * y = nullptr;
        for (int k = 0; k < kernel; ++k) {
            auto * tap = ggml_view_2d(c, w, ic, oc, ic*sizeof(float), k*ic*oc*sizeof(float));
            auto * signal = ggml_view_2d(c, x, ic, count, stride*x->nb[1], k*x->nb[1]);
            auto * update = pocket_mm(c, tap, signal);
            y = y ? ggml_add(c, y, update) : update;
        }
        return bias ? ggml_add(c, y, tensor(p + ".bias")) : y;
    }
    ggml_tensor * transpose_conv(PocketGraph & g, const std::string & p, ggml_tensor * x, int stride) {
        auto * c = g.ctx; auto * w = tensor(p + ".weight");
        const int count = x->ne[1], ic = w->ne[0], oc = w->ne[1];
        x = history_input(g, p, x);
        ggml_tensor * y = nullptr;
        for (int group = 0; group < 2; ++group) {
            auto * tap = ggml_view_2d(c, w, ic, oc*stride, ic*sizeof(float), group*stride*ic*oc*sizeof(float));
            auto * update = pocket_mm(c, tap, slice(c, x, 1-group, count));
            y = y ? ggml_add(c, y, update) : update;
        }
        return ggml_add(c, ggml_reshape_2d(c, y, oc, count*stride), tensor(p + ".bias"));
    }
    ggml_tensor * upsample(PocketGraph & g, ggml_tensor * x) {
        auto * c = g.ctx; auto * w = tensor("upsample.convtr.convtr.weight");
        const int count = x->ne[1];
        x = history_input(g, "upsample", x);
        ggml_tensor * y = nullptr;
        auto * target = ggml_new_tensor_3d(c, GGML_TYPE_F32, dim, up, count);
        for (int group = 0; group < 2; ++group) {
            auto * tap = ggml_view_2d(c, w, dim, up, dim*sizeof(float), group*up*dim*sizeof(float));
            auto * signal = ggml_reshape_3d(c, ggml_cont(c, slice(c, x, 1-group, count)), dim, 1, count);
            auto * update = ggml_mul(c, ggml_repeat(c, signal, target), tap);
            y = y ? ggml_add(c, y, update) : update;
        }
        return ggml_reshape_2d(c, y, dim, count*up);
    }
    ggml_tensor * norm(ggml_context * c, const std::string & p, ggml_tensor * x) {
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), tensor(p+".weight")), tensor(p+".bias"));
    }
    ggml_tensor * transformer(PocketGraph & g, ggml_tensor * x, bool enc, ggml_tensor * pos, ggml_tensor * mask) {
        auto * c = g.ctx;
        const int n = x->ne[1], retained = std::min(positions[enc], window-1), total = retained+n;
        auto kv_view = [&](ggml_tensor * t, int first, int count) {
            return ggml_view_3d(c, t, hd, heads, count, t->nb[1], t->nb[2], first*t->nb[2]);
        };
        for (int i = 0; i < layers; ++i) {
            const std::string p = std::string(enc ? "enc" : "dec")+".layers."+std::to_string(i);
            auto * qkv = pocket_mm(c, tensor(p+".self_attn.in_proj.weight"), norm(c, p+".norm1", x));
            auto part = [&](int index) {
                return ggml_reshape_3d(c, ggml_cont(c, ggml_view_2d(c, qkv, dim, n, qkv->nb[1], index*dim*sizeof(float))), hd, heads, n);
            };
            auto rotate = [&](ggml_tensor * v) { return ggml_rope_ext(c, v, pos, nullptr, hd, 0, 0, 10000, 1, 0, 1, 0, 0); };
            auto * q = rotate(part(0)), * k = rotate(part(1)), * v = part(2);
            if (retained) {
                k = ggml_concat(c, kv_view(keys[enc][i], 0, retained), k, 2);
                v = ggml_concat(c, kv_view(values[enc][i], 0, retained), v, 2);
            }
            const int keep = std::min(total, window-1);
            g.updates.push_back(ggml_cpy(c, kv_view(k, total-keep, keep), kv_view(keys[enc][i], 0, keep)));
            g.updates.push_back(ggml_cpy(c, kv_view(v, total-keep, keep), kv_view(values[enc][i], 0, keep)));
            auto * scores = pocket_mm(c, ggml_permute(c, k, 0, 2, 1, 3), ggml_permute(c, q, 0, 2, 1, 3));
            auto * probs = ggml_soft_max_ext(c, scores, mask, 1/std::sqrt(float(hd)), 0);
            auto * attended = pocket_mm(c, ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3)), probs);
            auto * merged = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, attended, 0, 2, 1, 3)), dim, n);
            auto * update = pocket_mm(c, tensor(p+".self_attn.out_proj.weight"), merged);
            x = ggml_add(c, x, ggml_mul(c, update, tensor(p+".layer_scale_1.scale")));
            auto * ff = pocket_gelu(c, pocket_mm(c, tensor(p+".linear1.weight"), norm(c, p+".norm2", x)));
            update = pocket_mm(c, tensor(p+".linear2.weight"), ff);
            x = ggml_add(c, x, ggml_mul(c, update, tensor(p+".layer_scale_2.scale")));
        }
        return x;
    }
    ggml_tensor * residual(PocketGraph & g, const std::string & p, ggml_tensor * x) {
        auto * y = conv(g, p+".block.1.conv", ggml_elu(g.ctx, x));
        y = conv(g, p+".block.3.conv", ggml_elu(g.ctx, y));
        return ggml_add(g.ctx, x, y);
    }
    std::vector<float> run(const std::vector<float> & input, bool enc) {
        PocketGraph g(backend, &hexagon_caps); auto * c = g.ctx;
        const int count = input.size() / (enc ? hop : latent), n = count*up;
        if (positions[enc] > INT_MAX-n) fail("position overflow; reset codec");
        auto * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, enc ? 1 : latent, enc ? count*hop : count);
        auto * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        const int retained = std::min(positions[enc], window-1), total = retained+n;
        auto * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, total, n);
        ggml_set_input(in); ggml_set_input(pos); ggml_set_input(mask);
        ggml_tensor * x = in;
        if (enc) {
            x = conv(g, "encoder.model.0.conv", x);
            for (int i = 0; i < 3; ++i) {
                x = residual(g, "encoder.model."+std::to_string(1+3*i), x);
                x = conv(g, "encoder.model."+std::to_string(3+3*i)+".conv", ggml_elu(c, x), ratios[2-i]);
            }
            x = conv(g, "encoder.model.11.conv", ggml_elu(c, x));
            x = transformer(g, x, true, pos, mask);
            x = conv(g, "downsample.conv.conv", x, up, false, positions[1] == 0);
        } else {
            x = upsample(g, pocket_mm(c, tensor("quantizer.output_proj.weight"), x));
            x = transformer(g, x, false, pos, mask);
            x = conv(g, "decoder.model.0.conv", x);
            for (int i = 0; i < 3; ++i) {
                x = transpose_conv(g, "decoder.model."+std::to_string(2+3*i)+".convtr", ggml_elu(c, x), ratios[i]);
                x = residual(g, "decoder.model."+std::to_string(3+3*i), x);
            }
            x = conv(g, "decoder.model.11.conv", ggml_elu(c, x));
        }
        if (measure) {
            measure->compute = std::max(measure->compute, g.prepare(x, true, n_threads, &measure->opencl_scratch));
            return {};
        }
        g.prepare(x);
        ggml_backend_tensor_set(in, input.data(), 0, input.size()*sizeof(float));
        std::vector<int32_t> indices(n);
        std::vector<float> mask_data(total*n);
        for (int i = 0; i < n; ++i) {
            indices[i] = positions[enc]+i;
            for (int j = 0; j < total; ++j) {
                const int delta = retained+i-j;
                mask_data[i*total+j] = delta >= 0 && delta < window ? 0 : -INFINITY;
            }
        }
        ggml_backend_tensor_set(pos, indices.data(), 0, indices.size()*sizeof(int32_t));
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size()*sizeof(float));
        try {
            auto result = g.compute(backend, x);
            positions[enc] += n;
            return result;
        } catch (...) { reset(enc); throw; }
    }
};

Mimi::Mimi(const std::string & path, int threads, ggml_backend_dev_t device) : impl_(new Impl) { impl_->load(path, threads, nullptr, device); }
MemoryMeasure Mimi::measure(const std::string & path, int threads, bool include_encoder, ggml_backend_dev_t device) {
    MemoryMeasure out;
    Impl model; model.load(path, threads, &out, device);
    // Price the maximum supported decode batch; include a full
    // retained attention window, plus the distinct empty-history shape.
    for (int past : {0, window-1}) {
        model.positions[0] = past;
        model.run(std::vector<float>(Mimi::max_decode_frames() * latent), false);
        if (include_encoder) {
            model.positions[1] = past;
            model.run(std::vector<float>(hop), true);
        }
    }
    out.graph_metadata = PocketGraph::metadata_bytes();
    return out;
}
Mimi::~Mimi() = default;
const char * Mimi::backend_name() const noexcept { return ggml_backend_name(impl_->backend); }
void Mimi::reset_decoder() { impl_->reset(false); }
const std::string & Mimi::source_hash() const { return impl_->hash; }
std::vector<float> Mimi::decode(const std::vector<float> & input) {
    if (input.empty() || input.size()%latent || input.size() > latent*Mimi::max_decode_frames()) fail("decode expects 1.." + std::to_string(Mimi::max_decode_frames()) + " latent frames");
    require_finite(input, "codec latents");
    return impl_->run(input, false);
}
std::vector<float> Mimi::encode(const std::vector<float> & input) {
    if (!impl_->encoder_available) fail("checkpoint has a disabled voice encoder; use a prepared voice");
    if (input.empty() || input.size() > rate*30) fail("voice reference must contain 0..30 seconds of mono audio");
    require_finite(input, "voice reference");
    impl_->reset(true);
    std::vector<float> output;
    for (size_t start = 0; start < input.size(); start += hop) {
        std::vector<float> chunk(hop, 0);
        std::copy_n(input.data()+start, std::min<size_t>(hop, input.size()-start), chunk.data());
        auto result = impl_->run(chunk, true);
        output.insert(output.end(), result.begin(), result.end());
    }
    return output;
}
} // namespace tts_cpp::pocket::detail
