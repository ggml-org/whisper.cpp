#include "moss/codec.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "sched_dispatch.h"
#include "moss/fit_measure.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tts_cpp::moss::detail {
namespace {

constexpr float LAYER_NORM_EPS = 1e-5f;
constexpr float L2_NORM_EPS    = 3.4526698e-4f;
constexpr float ROPE_FREQ_SCALE_NEUTRAL = 1.0f;
constexpr int   GRAPH_NODES    = 16384;
constexpr int   DEFAULT_SAMPLE_RATE = 24000;
constexpr int   MIN_SAMPLE_RATE     = 8000;
constexpr int   MAX_SAMPLE_RATE     = 192000;
constexpr int   MAX_CLIP_SECONDS    = 60;
constexpr int     MAX_PATCH_SIZE  = 8192;
constexpr int64_t MAX_DOWNSAMPLE  = 1 << 20;
constexpr uint32_t MAX_MODULES          = 64;
constexpr int      MAX_TRANSFORMER_LAYERS = 256;
constexpr int      MAX_QUANTIZERS       = 64;
constexpr int      MAX_CODEBOOK_SIZE    = 1 << 20;
constexpr int      MAX_DIMENSION        = 1 << 16;
constexpr int      MAX_HEADS            = 1024;
constexpr int      MAX_CONTEXT          = 1 << 16;

constexpr const char * ENCODER_ARCH = "moss-tts-audio-encoder";
constexpr const char * DECODER_ARCH = "moss-tts-audio-decoder";
constexpr const char * OWNER = "moss codec";
constexpr const char * SIDECAR_STAGE = "codec decoder";
constexpr int64_t REPLAY_WINDOW_FRAMES = 750;
constexpr int UNFUSED_QUERY_AXIS = 1;
constexpr int FUSED_QUERY_AXIS = 2;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss codec: " + message);
}

struct TransformerLayer {
    ggml_tensor * attn_qkv  = nullptr;
    ggml_tensor * attn_out  = nullptr;
    ggml_tensor * ffn_up    = nullptr;
    ggml_tensor * ffn_down  = nullptr;
    ggml_tensor * norm1_w   = nullptr;
    ggml_tensor * norm1_b   = nullptr;
    ggml_tensor * norm2_w   = nullptr;
    ggml_tensor * norm2_b   = nullptr;
    ggml_tensor * scale1    = nullptr;
    ggml_tensor * scale2    = nullptr;
};

struct TransformerBlock {
    int input_dimension  = 0;
    int output_dimension = 0;
    int d_model          = 0;
    int num_heads        = 0;
    int num_layers       = 0;
    int context          = 0;
    float max_period     = 10000.0f;
    ggml_tensor * input_proj  = nullptr;
    ggml_tensor * output_proj = nullptr;
    std::vector<TransformerLayer> layers;
};

struct Module {
    bool is_transformer = false;
    int patch_size = 1;
    TransformerBlock transformer;
};

struct LayerCache {
    ggml_tensor * keys = nullptr;
    ggml_tensor * values = nullptr;
};

struct BlockStream {
    std::vector<LayerCache> layers;
    int64_t position = 0;
    int64_t cached = 0;
    int64_t capacity = 0;
};

struct QuantizerMeta {
    int input_dim      = 0;
    int rvq_dim        = 0;
    int output_dim     = 0;
    int num_quantizers = 0;
    int codebook_size  = 0;
    int codebook_dim   = 0;
};

struct GraphInputI32 {
    ggml_tensor * tensor;
    std::vector<int32_t> data;
};

struct GraphInputF32 {
    ggml_tensor * tensor;
    std::vector<float> data;
};

std::vector<int32_t> sequential_positions(int64_t first, int64_t n) {
    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; ++i) {
        positions[i] = (int32_t) (first + i);
    }
    return positions;
}

bool within(int value, int low, int high) {
    return value >= low && value <= high;
}

std::vector<float> banded_causal_mask(int64_t query_start, int64_t query_count,
                                      int64_t key_start, int64_t key_count, int context) {
    std::vector<float> mask(query_count * key_count, -std::numeric_limits<float>::infinity());
    for (int64_t r = 0; r < query_count; ++r) {
        const int64_t iq = query_start + r;
        for (int64_t c = 0; c < key_count; ++c) {
            const int64_t ik = key_start + c;
            if (ik <= iq && (context <= 0 || iq - ik < context)) {
                mask[r * key_count + c] = 0.0f;
            }
        }
    }
    return mask;
}

struct AttentionBand {
    int64_t query_start = 0;
    int64_t query_count = 0;
    int64_t key_start = 0;
    int64_t key_count = 0;
    ggml_tensor * mask = nullptr;
    ggml_tensor * mask_f16 = nullptr;
};

int64_t align_up(int64_t value, int64_t multiple) {
    if (multiple <= 1) {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

struct Codec::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ::tts_cpp::detail::sched_fallback sched;

    std::string arch;
    bool encoder = false;
    int n_threads = 1;
    int rate = DEFAULT_SAMPLE_RATE;
    int64_t downsample = 0;
    QuantizerMeta quantizer;
    std::vector<Module> modules;

    ggml_context * graph_ctx = nullptr;
    std::vector<GraphInputI32> inputs_i32;
    std::vector<GraphInputF32> inputs_f32;
    std::vector<ggml_tensor *> cache_writes;

    ggml_context * stream_ctx = nullptr;
    ggml_backend_buffer_t stream_buffer = nullptr;
    std::vector<BlockStream> streams;
    int stream_channels = 0;
    int64_t frames_decoded = 0;
    bool measure_only = false;
    AttentionLayout attention_layout = AttentionLayout::Automatic;
    bool fused = false;
    uint64_t graph_host_bytes = 0;

    std::unique_ptr<CodecSidecarModel> sidecar;
    std::unique_ptr<CodecSidecarStream> sidecar_stream;
    bool strict = false;
    StageBackends backends;
    std::vector<int32_t> stream_codes;
    bool ggml_stream_open = false;

    ~Impl() {
        ::tts_cpp::detail::sched_fallback_free(sched);
        if (graph_ctx) ggml_free(graph_ctx);
        if (stream_buffer) ggml_backend_buffer_free(stream_buffer);
        if (stream_ctx) ggml_free(stream_ctx);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    int64_t find_key(const std::string & name) const {
        return gguf_find_key(file, name.c_str());
    }

    bool has_key(const std::string & name) const {
        return find_key(name) >= 0;
    }

    std::string meta_str(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_STRING) {
            fail("missing string metadata: " + name);
        }
        return gguf_get_val_str(file, key);
    }

    uint32_t meta_u32(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            fail("missing metadata: " + name);
        }
        switch (gguf_get_kv_type(file, key)) {
            case GGUF_TYPE_UINT32:  return gguf_get_val_u32(file, key);
            case GGUF_TYPE_INT32:   return (uint32_t) gguf_get_val_i32(file, key);
            case GGUF_TYPE_STRING:  return (uint32_t) std::stoul(gguf_get_val_str(file, key));
            default:                fail("unsupported metadata type: " + name);
        }
    }

    float meta_f32(const std::string & name, float fallback) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            return fallback;
        }
        switch (gguf_get_kv_type(file, key)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(file, key);
            case GGUF_TYPE_UINT32:  return (float) gguf_get_val_u32(file, key);
            case GGUF_TYPE_STRING:  return std::stof(gguf_get_val_str(file, key));
            default:                fail("unsupported metadata type: " + name);
        }
    }

    ggml_tensor * find_tensor(const std::string & name) const {
        return ggml_get_tensor(weights ? weights : metadata, name.c_str());
    }

    ggml_tensor * require_tensor(const std::string & name) const {
        ggml_tensor * tensor = find_tensor(name);
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    void read_quantizer_meta() {
        quantizer.input_dim      = (int) meta_u32(arch + ".quantizer.input_dim");
        quantizer.rvq_dim        = (int) meta_u32(arch + ".quantizer.rvq_dim");
        quantizer.output_dim     = (int) meta_u32(arch + ".quantizer.output_dim");
        quantizer.num_quantizers = (int) meta_u32(arch + ".quantizer.num_quantizers");
        quantizer.codebook_size  = (int) meta_u32(arch + ".quantizer.codebook_size");
        quantizer.codebook_dim   = (int) meta_u32(arch + ".quantizer.codebook_dim");
        if (!within(quantizer.num_quantizers, 1, MAX_QUANTIZERS) ||
            !within(quantizer.codebook_size, 1, MAX_CODEBOOK_SIZE) ||
            !within(quantizer.codebook_dim, 1, MAX_DIMENSION) ||
            !within(quantizer.rvq_dim, 1, MAX_DIMENSION) ||
            !within(quantizer.input_dim, 1, MAX_DIMENSION) ||
            !within(quantizer.output_dim, 1, MAX_DIMENSION)) {
            fail("invalid quantizer geometry");
        }
    }

    void validate_codebooks() {
        if (encoder) {
            require_matrix(require_tensor("quantizer.input_proj.weight"),
                    quantizer.input_dim, quantizer.rvq_dim);
        } else {
            require_matrix(require_tensor("quantizer.output_proj.weight"),
                    quantizer.rvq_dim, quantizer.output_dim);
        }
        for (int iq = 0; iq < quantizer.num_quantizers; ++iq) {
            const ggml_tensor * codebook =
                    require_tensor(quantizer_tensor_name(iq, "codebook.weight"));
            if (codebook->ne[0] != (int64_t) quantizer.codebook_dim ||
                codebook->ne[1] < (int64_t) quantizer.codebook_size) {
                fail("codebook " + std::to_string(iq) +
                     " does not match the declared quantizer geometry");
            }
            if (encoder) {
                require_matrix(require_tensor(quantizer_tensor_name(iq, "in_proj.weight")),
                        quantizer.rvq_dim, quantizer.codebook_dim);
            }
            require_matrix(require_tensor(quantizer_tensor_name(iq, "out_proj.weight")),
                    quantizer.codebook_dim, quantizer.rvq_dim);
        }
    }

    void validate_module_chain() {
        int64_t channels = encoder ? 1 : (int64_t) quantizer.output_dim;
        for (size_t im = 0; im < modules.size(); ++im) {
            const Module & module = modules[im];
            if (!module.is_transformer) {
                if (encoder) {
                    channels *= module.patch_size;
                } else {
                    if (channels % module.patch_size != 0) {
                        fail("module " + std::to_string(im) +
                             " patch size does not divide its input channels");
                    }
                    channels /= module.patch_size;
                }
                continue;
            }
            if (module.transformer.input_dimension != channels) {
                fail("module " + std::to_string(im) +
                     " input dimension does not match the chain");
            }
            channels = module.transformer.output_dimension;
        }
        const int64_t expected = encoder ? (int64_t) quantizer.input_dim : 1;
        if (channels != expected) {
            fail("module chain does not end at the expected channel count");
        }
    }

    std::string block_tensor_name(int block, int layer, const char * suffix) const {
        return "blk." + std::to_string(block) + ".layer." + std::to_string(layer) + "." + suffix;
    }

    void read_transformer_meta(const std::string & prefix, int tensor_block, TransformerBlock & block) {
        block.input_dimension  = (int) meta_u32(prefix + ".input_dimension");
        block.output_dimension = (int) meta_u32(prefix + ".output_dimension");
        block.d_model          = (int) meta_u32(prefix + ".d_model");
        block.num_heads        = (int) meta_u32(prefix + ".num_heads");
        block.num_layers       = (int) meta_u32(prefix + ".num_layers");
        block.context          = (int) meta_u32(prefix + ".context");
        block.max_period       = meta_f32(prefix + ".max_period", 10000.0f);
        if (!within(block.d_model, 1, MAX_DIMENSION) || !within(block.num_heads, 1, MAX_HEADS) ||
            block.d_model % block.num_heads != 0 ||
            !within(block.input_dimension, 1, MAX_DIMENSION) ||
            !within(block.output_dimension, 1, MAX_DIMENSION) ||
            !within(block.num_layers, 1, MAX_TRANSFORMER_LAYERS) ||
            !within(block.context, encoder ? 0 : 1, MAX_CONTEXT)) {
            fail("invalid transformer geometry in " + prefix);
        }
        block.input_proj  = find_tensor("blk." + std::to_string(tensor_block) + ".input_proj.weight");
        block.output_proj = find_tensor("blk." + std::to_string(tensor_block) + ".output_proj.weight");
        block.layers.resize(block.num_layers);
    }

    void read_transformer_layers(int tensor_block, TransformerBlock & block) {
        for (int il = 0; il < block.num_layers; ++il) {
            TransformerLayer & layer = block.layers[il];
            layer.attn_qkv = require_tensor(block_tensor_name(tensor_block, il, "attn_qkv.weight"));
            layer.attn_out = require_tensor(block_tensor_name(tensor_block, il, "attn_output.weight"));
            layer.ffn_up   = require_tensor(block_tensor_name(tensor_block, il, "ffn_up.weight"));
            layer.ffn_down = require_tensor(block_tensor_name(tensor_block, il, "ffn_down.weight"));
            layer.norm1_w  = require_tensor(block_tensor_name(tensor_block, il, "attn_norm.weight"));
            layer.norm1_b  = require_tensor(block_tensor_name(tensor_block, il, "attn_norm.bias"));
            layer.norm2_w  = require_tensor(block_tensor_name(tensor_block, il, "ffn_norm.weight"));
            layer.norm2_b  = require_tensor(block_tensor_name(tensor_block, il, "ffn_norm.bias"));
            layer.scale1   = find_tensor(block_tensor_name(tensor_block, il, "attn_scale.scale"));
            layer.scale2   = find_tensor(block_tensor_name(tensor_block, il, "ffn_scale.scale"));
        }
    }

    void read_modules() {
        const std::string section = encoder ? "encoder" : "decoder";
        const uint32_t block_count = meta_u32(arch + "." + section + ".block_count");
        if (block_count < 1 || block_count > MAX_MODULES) {
            fail("invalid module count");
        }
        modules.resize(block_count);
        int tensor_block = 0;
        for (uint32_t ib = 0; ib < block_count; ++ib) {
            const std::string prefix = arch + "." + section + "." + std::to_string(ib);
            const std::string type = meta_str(prefix + ".module_type");
            Module & module = modules[ib];
            if (type == "PatchedPretransform") {
                module.is_transformer = false;
                module.patch_size = (int) meta_u32(prefix + ".patch_size");
                if (module.patch_size < 1 || module.patch_size > MAX_PATCH_SIZE) {
                    fail("invalid patch size in " + prefix);
                }
                continue;
            }
            if (type != "Transformer") {
                fail("unsupported module type: " + type);
            }
            module.is_transformer = true;
            read_transformer_meta(prefix, tensor_block, module.transformer);
            read_transformer_layers(tensor_block, module.transformer);
            validate_transformer_shapes(prefix, module.transformer);
            tensor_block++;
        }
        validate_patch_product();
    }

    void validate_patch_product() {
        int64_t product = 1;
        for (const Module & module : modules) {
            if (!module.is_transformer) {
                product *= module.patch_size;
                if (product > MAX_DOWNSAMPLE) {
                    fail("patch sizes multiply past the supported downsample factor");
                }
            }
        }
        if (encoder && product != downsample) {
            fail("patch sizes do not multiply to the downsample rate");
        }
    }

    void matrix_dims(const ggml_tensor * tensor, int64_t & in, int64_t & out) const {
        const int n_dims = ggml_n_dims(tensor);
        if (n_dims <= 2) {
            in  = tensor->ne[0];
            out = tensor->ne[1];
            return;
        }
        if (n_dims == 3 && tensor->ne[0] == 1) {
            in  = tensor->ne[1];
            out = tensor->ne[2];
            return;
        }
        if (n_dims == 4 && tensor->ne[0] == 1 && tensor->ne[1] == 1) {
            in  = tensor->ne[2];
            out = tensor->ne[3];
            return;
        }
        fail(std::string("unsupported projection rank: ") + ggml_get_name(tensor));
    }

    void require_matrix(const ggml_tensor * tensor, int64_t in, int64_t out) const {
        if (tensor == nullptr) {
            return;
        }
        int64_t got_in = 0;
        int64_t got_out = 0;
        matrix_dims(tensor, got_in, got_out);
        if (got_in != in || (out > 0 && got_out != out)) {
            fail(std::string(ggml_get_name(tensor)) + " has unexpected dimensions");
        }
    }

    void require_vector(const ggml_tensor * tensor, int64_t n) const {
        if (tensor != nullptr && (ggml_n_dims(tensor) > 1 || tensor->ne[0] != n)) {
            fail(std::string(ggml_get_name(tensor)) + " has unexpected dimensions");
        }
    }

    void validate_transformer_shapes(const std::string & prefix, const TransformerBlock & block) const {
        require_matrix(block.input_proj, block.input_dimension, block.d_model);
        require_matrix(block.output_proj, block.d_model, block.output_dimension);
        for (const TransformerLayer & layer : block.layers) {
            require_matrix(layer.attn_qkv, block.d_model, 3 * (int64_t) block.d_model);
            require_matrix(layer.attn_out, block.d_model, block.d_model);
            int64_t ffn_in = 0;
            int64_t ffn_hidden = 0;
            matrix_dims(layer.ffn_up, ffn_in, ffn_hidden);
            if (ffn_in != block.d_model) {
                fail("ffn_up input does not match d_model in " + prefix);
            }
            require_matrix(layer.ffn_down, ffn_hidden, block.d_model);
            require_vector(layer.norm1_w, block.d_model);
            require_vector(layer.norm1_b, block.d_model);
            require_vector(layer.norm2_w, block.d_model);
            require_vector(layer.norm2_b, block.d_model);
            require_vector(layer.scale1, block.d_model);
            require_vector(layer.scale2, block.d_model);
        }
    }

    void duplicate_metadata_tensors() {
        size_t count = 0;
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            count++;
        }
        weights = ggml_init({(count + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) {
            fail("weight context allocation failed");
        }
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            auto * dst = ggml_new_tensor(weights, src->type, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
    }

    void require_device_weight_types() const {
        if (!::tts_cpp::detail::backend_is_hexagon(backend)) {
            return;
        }
        if (const ggml_tensor * refused = first_tensor_hexagon_cannot_run(weights)) {
            fail(hexagon_weight_refusal(refused, HEXAGON_TTS_WEIGHT_TIERS));
        }
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
        for (auto * dst = ggml_get_first_tensor(weights); dst; dst = ggml_get_next_tensor(weights, dst)) {
            if (!reader.to_backend(ggml_get_name(dst), dst)) {
                fail(std::string("failed to load tensor: ") + ggml_get_name(dst));
            }
        }
    }

    void load(const std::string & path, const BackendRequest & request, int threads) {
        n_threads = threads;
        if (threads < 1 || threads > 1024) {
            fail("threads must be 1..1024");
        }
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) {
            fail("cannot read GGUF: " + path);
        }
        arch = meta_str("general.architecture");
        if (arch == ENCODER_ARCH) {
            encoder = true;
        } else if (arch == DECODER_ARCH) {
            encoder = false;
        } else {
            fail("unsupported architecture: " + arch);
        }
        read_sample_rate();
        if (encoder) {
            downsample = meta_u32(arch + ".downsample_rate");
        }
        read_quantizer_meta();
        backend = init_backend(request, "moss-codec");
        fused = fuses_attention(attention_layout, backend);
        duplicate_metadata_tensors();
        require_device_weight_types();
        read_modules();
        validate_module_chain();
        validate_codebooks();
        if (!encoder) {
            downsample = decoder_upsample_factor();
        }
        if (downsample <= 0) {
            fail("invalid frame geometry");
        }
        if (measure_only) {
            mark_fit_tensors(weights);
            return;
        }
        upload_weights(path);
    }

    void read_sample_rate() {
        const uint32_t declared = has_key(arch + ".sampling_rate")
                ? meta_u32(arch + ".sampling_rate") : (uint32_t) DEFAULT_SAMPLE_RATE;
        if (declared < (uint32_t) MIN_SAMPLE_RATE || declared > (uint32_t) MAX_SAMPLE_RATE) {
            fail("sampling rate must be " + std::to_string(MIN_SAMPLE_RATE) + ".." +
                 std::to_string(MAX_SAMPLE_RATE) + " Hz");
        }
        rate = (int) declared;
    }

    int64_t decoder_upsample_factor() const {
        int64_t factor = 1;
        for (const Module & module : modules) {
            if (!module.is_transformer) {
                factor *= module.patch_size;
            }
        }
        return factor;
    }

    void begin_graph() {
        if (graph_ctx) {
            ggml_free(graph_ctx);
            graph_ctx = nullptr;
        }
        graph_host_bytes = 0;
        inputs_i32.clear();
        inputs_f32.clear();
        cache_writes.clear();
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

    ggml_tensor * as_matrix(ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        const int n_dims = ggml_n_dims(tensor);
        if (n_dims == 2) {
            return tensor;
        }
        if (n_dims == 3 && tensor->ne[0] == 1) {
            return ggml_reshape_2d(graph_ctx, tensor, tensor->ne[1], tensor->ne[2]);
        }
        if (n_dims == 4 && tensor->ne[0] == 1 && tensor->ne[1] == 1) {
            return ggml_reshape_2d(graph_ctx, tensor, tensor->ne[2], tensor->ne[3]);
        }
        fail(std::string("unsupported projection rank: ") + ggml_get_name(tensor));
    }

    ggml_tensor * as_f32(ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(graph_ctx, tensor, GGML_TYPE_F32);
    }

    ggml_tensor * linear(ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) {
        ggml_tensor * cur = as_f32(input);
        if (weight != nullptr) {
            cur = ggml_mul_mat(graph_ctx, as_matrix(weight), cur);
        }
        if (bias != nullptr) {
            cur = ggml_add(graph_ctx, cur, as_f32(bias));
        }
        return cur;
    }

    ggml_tensor * layer_norm(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias) {
        cur = ggml_norm(graph_ctx, cur, LAYER_NORM_EPS);
        cur = ggml_mul(graph_ctx, cur, as_f32(weight));
        return ggml_add(graph_ctx, cur, as_f32(bias));
    }

    ggml_tensor * l2_normalize(ggml_tensor * cur) {
        ggml_tensor * norm = ggml_sqrt(graph_ctx, ggml_sum_rows(graph_ctx, ggml_sqr(graph_ctx, cur)));
        norm = ggml_clamp(graph_ctx, norm, L2_NORM_EPS, INFINITY);
        return ggml_div(graph_ctx, cur, ggml_repeat(graph_ctx, norm, cur));
    }

    ggml_tensor * time_slice(ggml_tensor * heads_by_time, int64_t first, int64_t count) {
        return ggml_view_3d(graph_ctx, heads_by_time, heads_by_time->ne[0], heads_by_time->ne[1],
                count, heads_by_time->nb[1], heads_by_time->nb[2], first * heads_by_time->nb[2]);
    }

    ggml_tensor * band_attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                 const AttentionBand & band, int d_head) {
        ggml_tensor * q_band = ggml_permute(graph_ctx,
                time_slice(q, band.query_start, band.query_count), 0, 2, 1, 3);
        ggml_tensor * k_band = ggml_permute(graph_ctx,
                time_slice(k, band.key_start, band.key_count), 0, 2, 1, 3);
        ggml_tensor * v_band = ggml_cont(graph_ctx, ggml_permute(graph_ctx,
                time_slice(v, band.key_start, band.key_count), 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(graph_ctx, k_band, q_band);
        scores = ggml_soft_max_ext(graph_ctx, scores, band.mask,
                1.0f / std::sqrt((float) d_head), 0.0f);
        return ggml_mul_mat(graph_ctx, v_band, scores);
    }

    ggml_tensor * fused_band_attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                       const AttentionBand & band, int d_head) {
        return ggml_flash_attn_ext(graph_ctx,
                ggml_permute(graph_ctx, time_slice(q, band.query_start, band.query_count), 0, 2, 1, 3),
                ggml_permute(graph_ctx, time_slice(k, band.key_start, band.key_count), 0, 2, 1, 3),
                ggml_permute(graph_ctx, time_slice(v, band.key_start, band.key_count), 0, 2, 1, 3),
                band.mask_f16, 1.0f / std::sqrt((float) d_head), 0.0f, 0.0f);
    }

    ggml_tensor * banded_attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                   const std::vector<AttentionBand> & bands, int d_head) {
        const int query_dim = fused ? FUSED_QUERY_AXIS : UNFUSED_QUERY_AXIS;
        ggml_tensor * attended = nullptr;
        for (const AttentionBand & band : bands) {
            ggml_tensor * piece = fused ? fused_band_attention(q, k, v, band, d_head)
                                        : band_attention(q, k, v, band, d_head);
            attended = attended ? ggml_concat(graph_ctx, attended, piece, query_dim) : piece;
        }
        return attended;
    }

    ggml_tensor * merge_heads(ggml_tensor * attended) {
        if (fused) {
            return ggml_reshape_2d(graph_ctx, attended, attended->ne[0] * attended->ne[1], attended->ne[2]);
        }
        attended = ggml_permute(graph_ctx, attended, 0, 2, 1, 3);
        return ggml_cont_2d(graph_ctx, attended,
                attended->ne[0] * attended->ne[1], attended->ne[2] * attended->ne[3]);
    }

    ggml_tensor * with_history(ggml_tensor * fresh, ggml_tensor * cache, int64_t cached) {
        if (cache == nullptr || cached == 0) {
            return fresh;
        }
        ggml_tensor * history = ggml_view_3d(graph_ctx, cache, cache->ne[0], cache->ne[1], cached,
                cache->nb[1], cache->nb[2], 0);
        return ggml_concat(graph_ctx, history, fresh, 2);
    }

    void retain_tail(ggml_tensor * sequence, ggml_tensor * cache, int64_t keep) {
        if (cache == nullptr || keep == 0) {
            return;
        }
        const int64_t first = sequence->ne[2] - keep;
        ggml_tensor * tail = ggml_view_3d(graph_ctx, sequence, sequence->ne[0], sequence->ne[1], keep,
                sequence->nb[1], sequence->nb[2], first * sequence->nb[2]);
        ggml_tensor * slot = ggml_view_3d(graph_ctx, cache, cache->ne[0], cache->ne[1], keep,
                cache->nb[1], cache->nb[2], 0);
        cache_writes.push_back(ggml_cpy(graph_ctx, tail, slot));
    }

    ggml_tensor * attention(const TransformerLayer & layer, ggml_tensor * cur,
                            const TransformerBlock & block, int64_t frames,
                            ggml_tensor * positions, const std::vector<AttentionBand> & bands,
                            const LayerCache * cache, int64_t cached, int64_t keep) {
        const int d_head = block.d_model / block.num_heads;
        ggml_tensor * qkv = ggml_mul_mat(graph_ctx, as_matrix(layer.attn_qkv), cur);
        const size_t head_stride = ggml_row_size(qkv->type, d_head);
        ggml_tensor * q = ggml_view_3d(graph_ctx, qkv, d_head, block.num_heads, frames,
                head_stride, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_3d(graph_ctx, qkv, d_head, block.num_heads, frames,
                head_stride, qkv->nb[1], ggml_row_size(qkv->type, block.d_model));
        ggml_tensor * v = ggml_view_3d(graph_ctx, qkv, d_head, block.num_heads, frames,
                head_stride, qkv->nb[1], ggml_row_size(qkv->type, 2 * block.d_model));
        q = ggml_rope_ext(graph_ctx, q, positions, nullptr, d_head, 0, 0,
                block.max_period, ROPE_FREQ_SCALE_NEUTRAL, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(graph_ctx, k, positions, nullptr, d_head, 0, 0,
                block.max_period, ROPE_FREQ_SCALE_NEUTRAL, 0.0f, 1.0f, 0.0f, 0.0f);
        if (cache != nullptr) {
            k = with_history(ggml_cont(graph_ctx, k), cache->keys, cached);
            v = with_history(ggml_cont(graph_ctx, v), cache->values, cached);
            retain_tail(k, cache->keys, keep);
            retain_tail(v, cache->values, keep);
        }
        ggml_tensor * attended = merge_heads(banded_attention(q, k, v, bands, d_head));
        return ggml_mul_mat(graph_ctx, as_matrix(layer.attn_out), attended);
    }

    ggml_tensor * transformer_layer(const TransformerLayer & layer, ggml_tensor * cur,
                                    const TransformerBlock & block, int64_t frames,
                                    ggml_tensor * positions, const std::vector<AttentionBand> & bands,
                                    const LayerCache * cache, int64_t cached, int64_t keep) {
        ggml_tensor * attended = attention(layer, layer_norm(cur, layer.norm1_w, layer.norm1_b),
                block, frames, positions, bands, cache, cached, keep);
        if (layer.scale1 != nullptr) {
            attended = ggml_mul(graph_ctx, attended, as_f32(layer.scale1));
        }
        cur = ggml_add(graph_ctx, cur, attended);
        ggml_tensor * ff = layer_norm(cur, layer.norm2_w, layer.norm2_b);
        ff = ggml_mul_mat(graph_ctx, as_matrix(layer.ffn_up), ff);
        ff = ggml_gelu(graph_ctx, ff);
        ff = ggml_mul_mat(graph_ctx, as_matrix(layer.ffn_down), ff);
        if (layer.scale2 != nullptr) {
            ff = ggml_mul(graph_ctx, ff, as_f32(layer.scale2));
        }
        return ggml_add(graph_ctx, cur, ff);
    }

    AttentionBand plan_band(const TransformerBlock & block, int64_t history, int64_t query_start,
                            int64_t query_count) {
        AttentionBand band;
        band.query_start = query_start;
        band.query_count = query_count;
        const int64_t first_query = history + query_start;
        band.key_start = block.context > 0
                ? std::max<int64_t>(0, first_query - (block.context - 1)) : 0;
        band.key_count = first_query + query_count - band.key_start;
        band.mask = input_f32(measure_only ? std::vector<float>() :
                banded_causal_mask(first_query, query_count,
                band.key_start, band.key_count, block.context),
                band.key_count, query_count);
        if (fused) {
            band.mask_f16 = ggml_cast(graph_ctx, band.mask, GGML_TYPE_F16);
        }
        return band;
    }

    std::vector<AttentionBand> plan_attention_bands(const TransformerBlock & block, int64_t frames,
                                                    int64_t history) {
        std::vector<AttentionBand> bands;
        const int64_t band_rows = block.context > 0 && frames > 2 * (int64_t) block.context
                ? (int64_t) block.context : frames;
        for (int64_t query_start = 0; query_start < frames; query_start += band_rows) {
            bands.push_back(plan_band(block, history, query_start,
                    std::min(band_rows, frames - query_start)));
        }
        return bands;
    }

    ggml_tensor * transformer_block(const TransformerBlock & block, ggml_tensor * cur, int64_t frames,
                                    BlockStream * stream) {
        const int64_t first = stream ? stream->position : 0;
        const int64_t cached = stream ? stream->cached : 0;
        const int64_t keep = stream ? std::min(cached + frames, stream->capacity) : 0;
        ggml_tensor * positions = input_i32(sequential_positions(first, frames));
        const std::vector<AttentionBand> bands = plan_attention_bands(block, frames, cached);
        if (block.input_proj != nullptr) {
            cur = ggml_mul_mat(graph_ctx, as_matrix(block.input_proj), cur);
        }
        for (size_t il = 0; il < block.layers.size(); ++il) {
            cur = transformer_layer(block.layers[il], cur, block, frames, positions, bands,
                    stream ? &stream->layers[il] : nullptr, cached, keep);
        }
        if (stream != nullptr) {
            stream->position += frames;
            stream->cached = keep;
        }
        if (block.output_proj != nullptr) {
            cur = ggml_mul_mat(graph_ctx, as_matrix(block.output_proj), cur);
        }
        return cur;
    }

    ggml_tensor * patch_encode(ggml_tensor * cur, int channels, int64_t frames, int patch) {
        cur = ggml_reshape_3d(graph_ctx, cur, channels, patch, frames / patch);
        cur = ggml_cont(graph_ctx, ggml_permute(graph_ctx, cur, 1, 0, 2, 3));
        return ggml_reshape_2d(graph_ctx, cur, (int64_t) channels * patch, frames / patch);
    }

    ggml_tensor * patch_decode(ggml_tensor * cur, int channels, int64_t frames, int patch) {
        const int out_channels = channels / patch;
        cur = ggml_reshape_3d(graph_ctx, cur, patch, out_channels, frames);
        cur = ggml_cont(graph_ctx, ggml_permute(graph_ctx, cur, 1, 0, 2, 3));
        return ggml_reshape_2d(graph_ctx, cur, out_channels, frames * patch);
    }

    std::string quantizer_tensor_name(int index, const char * suffix) const {
        return "quantizer.quantizers." + std::to_string(index) + "." + suffix;
    }

    std::vector<int32_t> channel_codes(const std::vector<int32_t> & codes, int64_t frames,
                                       int n_channels, int channel) const {
        std::vector<int32_t> picked(frames);
        for (int64_t i = 0; i < frames; ++i) {
            picked[i] = codes[i * n_channels + channel];
        }
        return picked;
    }

    ggml_tensor * decoder_quantizer(const std::vector<int32_t> & codes, int64_t frames,
                                    int n_channels) {
        ggml_tensor * cur = nullptr;
        for (int iq = 0; iq < n_channels; ++iq) {
            std::vector<int32_t> channel = channel_codes(codes, frames, n_channels, iq);
            ggml_tensor * indices = input_i32(std::move(channel));
            ggml_tensor * codebook = as_f32(require_tensor(quantizer_tensor_name(iq, "codebook.weight")));
            ggml_tensor * embedded = ggml_get_rows(graph_ctx, codebook, indices);
            embedded = linear(embedded,
                    find_tensor(quantizer_tensor_name(iq, "out_proj.weight")),
                    find_tensor(quantizer_tensor_name(iq, "out_proj.bias")));
            cur = cur ? ggml_add(graph_ctx, cur, embedded) : embedded;
        }
        return linear(cur,
                find_tensor("quantizer.output_proj.weight"),
                find_tensor("quantizer.output_proj.bias"));
    }

    ggml_tensor * encoder_quantizer(ggml_tensor * cur) {
        cur = linear(cur,
                find_tensor("quantizer.input_proj.weight"),
                find_tensor("quantizer.input_proj.bias"));
        ggml_tensor * codes = nullptr;
        ggml_tensor * residual = cur;
        for (int iq = 0; iq < quantizer.num_quantizers; ++iq) {
            ggml_tensor * latent = linear(residual,
                    find_tensor(quantizer_tensor_name(iq, "in_proj.weight")),
                    find_tensor(quantizer_tensor_name(iq, "in_proj.bias")));
            ggml_tensor * codebook = as_f32(require_tensor(quantizer_tensor_name(iq, "codebook.weight")));
            ggml_tensor * scores = ggml_mul_mat(graph_ctx, l2_normalize(codebook), l2_normalize(latent));
            ggml_tensor * code = ggml_argmax(graph_ctx, scores);
            ggml_tensor * row = ggml_reshape_2d(graph_ctx, code, 1, code->ne[0]);
            codes = codes ? ggml_concat(graph_ctx, codes, row, 0) : row;
            ggml_tensor * decoded = linear(ggml_get_rows(graph_ctx, codebook, code),
                    find_tensor(quantizer_tensor_name(iq, "out_proj.weight")),
                    find_tensor(quantizer_tensor_name(iq, "out_proj.bias")));
            residual = ggml_sub(graph_ctx, residual, decoded);
        }
        return codes;
    }

    ggml_tensor * build_encode(const std::vector<float> & pcm, int64_t padded_samples) {
        std::vector<float> padded(padded_samples, 0.0f);
        std::copy(pcm.begin(), pcm.end(), padded.begin());
        ggml_tensor * cur = input_f32(std::move(padded), 1, padded_samples);
        int channels = 1;
        int64_t frames = padded_samples;
        int tensor_block = 0;
        for (const Module & module : modules) {
            if (!module.is_transformer) {
                cur = patch_encode(cur, channels, frames, module.patch_size);
                channels *= module.patch_size;
                frames /= module.patch_size;
                continue;
            }
            cur = transformer_block(module.transformer, cur, frames, nullptr);
            channels = module.transformer.output_dimension;
            tensor_block++;
        }
        if (channels != quantizer.input_dim) {
            fail("encoder output does not match quantizer input");
        }
        return encoder_quantizer(cur);
    }

    ggml_tensor * build_decode(const std::vector<int32_t> & codes, int64_t frames, int n_channels,
                               bool streaming) {
        ggml_tensor * cur = decoder_quantizer(codes, frames, n_channels);
        int channels = quantizer.output_dim;
        for (size_t im = 0; im < modules.size(); ++im) {
            const Module & module = modules[im];
            if (!module.is_transformer) {
                cur = patch_decode(cur, channels, frames, module.patch_size);
                channels /= module.patch_size;
                frames *= module.patch_size;
                continue;
            }
            cur = transformer_block(module.transformer, cur, frames,
                    streaming ? &streams[im] : nullptr);
            channels = module.transformer.output_dimension;
        }
        if (channels != 1) {
            fail("decoder did not reduce to mono output");
        }
        return cur;
    }

    std::vector<float> read_f32(ggml_tensor * tensor) {
        std::vector<float> out(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    std::vector<int32_t> read_i32(ggml_tensor * tensor) {
        std::vector<int32_t> out(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(int32_t));
        return out;
    }

    ggml_cgraph * finish_graph(ggml_tensor * output) {
        ggml_set_output(output);
        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, GRAPH_NODES, false);
        ggml_build_forward_expand(graph, output);
        for (ggml_tensor * write : cache_writes) {
            ggml_build_forward_expand(graph, write);
        }
        if (measure_only) return graph;
        if (!::tts_cpp::detail::sched_fallback_ensure(sched, backend, GRAPH_NODES, {weight_buffer})) {
            fail("scheduler initialization failed");
        }
        if (!::tts_cpp::detail::sched_fallback_alloc(sched, graph)) {
            fail("graph allocation failed");
        }
        return graph;
    }

    void compute(ggml_cgraph * graph) {
        set_graph_inputs();
        const ggml_status status = ::tts_cpp::detail::sched_fallback_compute(sched, backend, graph, n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }
    }

    size_t count_cache_tensors() const {
        size_t count = 0;
        for (const Module & module : modules) {
            if (module.is_transformer) {
                count += 2 * module.transformer.layers.size();
            }
        }
        return count;
    }

    void create_block_stream(const TransformerBlock & block, BlockStream & stream) {
        const int d_head = block.d_model / block.num_heads;
        stream.capacity = std::max(1, block.context - 1);
        stream.layers.resize(block.layers.size());
        for (LayerCache & cache : stream.layers) {
            cache.keys = ggml_new_tensor_3d(stream_ctx, GGML_TYPE_F32, d_head, block.num_heads,
                    stream.capacity);
            cache.values = ggml_new_tensor_3d(stream_ctx, GGML_TYPE_F32, d_head, block.num_heads,
                    stream.capacity);
        }
    }

    void allocate_streams() {
        stream_ctx = ggml_init({(count_cache_tensors() + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!stream_ctx) {
            fail("stream state context allocation failed");
        }
        streams.resize(modules.size());
        for (size_t im = 0; im < modules.size(); ++im) {
            if (modules[im].is_transformer) {
                create_block_stream(modules[im].transformer, streams[im]);
            }
        }
        if (measure_only) {
            mark_fit_tensors(stream_ctx);
            return;
        }
        stream_buffer = ggml_backend_alloc_ctx_tensors(stream_ctx, backend);
        if (!stream_buffer) {
            fail("stream state allocation failed");
        }
    }

    void reset_streams() {
        if (stream_ctx == nullptr) {
            allocate_streams();
        }
        for (BlockStream & stream : streams) {
            stream.position = 0;
            stream.cached = 0;
        }
    }

    std::vector<float> run_decode(const std::vector<int32_t> & codes, int n_channels, bool streaming) {
        begin_graph();
        const int64_t frames = (int64_t) codes.size() / n_channels;
        ggml_tensor * output = build_decode(codes, frames, n_channels, streaming);
        ggml_cgraph * graph = finish_graph(output);
        compute(graph);
        frames_decoded += frames;
        return read_f32(output);
    }

    CodecGeometry sidecar_geometry() const {
        CodecGeometry geometry;
        geometry.code_dim = quantizer.output_dim;
        geometry.hop = (int) downsample;
        int rate = 1;
        for (const Module & module : modules) {
            if (module.is_transformer) {
                const TransformerBlock & block = module.transformer;
                geometry.stages.push_back({rate, block.context, block.d_model / block.num_heads, block.max_period});
            } else {
                rate *= module.patch_size;
            }
        }
        return geometry;
    }

    void attach_sidecar(std::unique_ptr<CodecSidecarModel> model, bool strict_mode) {
        sidecar = std::move(model);
        sidecar_stream = sidecar ? std::make_unique<CodecSidecarStream>(*sidecar, sidecar_geometry()) : nullptr;
        strict = strict_mode;
    }

    void open_sidecar(const std::string & path) {
        const CoremlPolicy policy = read_coreml_policy();
        attach_sidecar(open_codec_sidecar(path, sidecar_geometry(), policy), policy.strict);
    }

    void retire_sidecar() {
        sidecar_stream.reset();
        sidecar.reset();
    }

    std::vector<float> quantizer_latents(const std::vector<int32_t> & codes, int n_channels) {
        begin_graph();
        ggml_tensor * latents = decoder_quantizer(codes, (int64_t) codes.size() / n_channels, n_channels);
        ggml_cgraph * graph = finish_graph(latents);
        compute(graph);
        return read_f32(latents);
    }

    bool decode_on_sidecar(const std::vector<int32_t> & codes, int n_channels, std::vector<float> & pcm) {
        if (!sidecar || !sidecar_stream->decode(quantizer_latents(codes, n_channels), pcm)) {
            retire_sidecar();
            return false;
        }
        frames_decoded += (int64_t) codes.size() / n_channels;
        backends.note(sidecar->label());
        return true;
    }

    std::vector<float> decode_batch(const std::vector<int32_t> & codes, int n_channels) {
        require_sidecar_when_strict();
        backends.note(GGML_STAGE_BACKEND);
        return run_decode(codes, n_channels, false);
    }

    void require_sidecar_when_strict() const {
        if (strict && !sidecar) {
            fail_strict(OWNER, SIDECAR_STAGE);
        }
    }

    bool sidecar_takes_pieces(int piece_frames) const {
        return sidecar && piece_frames > 0 && piece_frames <= sidecar->chunk_frames();
    }

    bool begin_sidecar_stream(int piece_frames) {
        if (!sidecar_takes_pieces(piece_frames)) {
            return false;
        }
        if (sidecar_stream->begin()) {
            return true;
        }
        retire_sidecar();
        return false;
    }

    void begin_stream(int n_channels, int piece_frames) {
        stream_channels = resolve_channels(n_channels);
        stream_codes.clear();
        ggml_stream_open = false;
        if (!begin_sidecar_stream(piece_frames)) {
            open_ggml_stream();
        }
    }

    void open_ggml_stream() {
        reset_streams();
        ggml_stream_open = true;
    }

    void replay_window(const std::vector<int32_t> & history, size_t begin, size_t end) {
        run_decode(std::vector<int32_t>(history.begin() + (std::ptrdiff_t) begin, history.begin() + (std::ptrdiff_t) end),
                stream_channels, true);
    }

    void replay_on_ggml(const std::vector<int32_t> & history, size_t count) {
        open_ggml_stream();
        const int64_t counted = frames_decoded;
        const size_t window = (size_t) REPLAY_WINDOW_FRAMES * (size_t) stream_channels;
        for (size_t begin = 0; begin < count; begin += window) {
            replay_window(history, begin, std::min(count, begin + window));
        }
        frames_decoded = counted;
    }

    std::vector<float> decode_stream_codes(const std::vector<int32_t> & codes) {
        std::vector<float> pcm;
        if (!ggml_stream_open && sidecar) {
            const size_t decoded_before = stream_codes.size();
            stream_codes.insert(stream_codes.end(), codes.begin(), codes.end());
            if (decode_on_sidecar(codes, stream_channels, pcm)) {
                return pcm;
            }
            require_sidecar_when_strict();
            replay_on_ggml(stream_codes, decoded_before);
        }
        require_sidecar_when_strict();
        if (!ggml_stream_open) {
            open_ggml_stream();
        }
        backends.note(GGML_STAGE_BACKEND);
        return run_decode(codes, stream_channels, true);
    }

    int resolve_channels(int requested) const {
        const int n_channels = requested > 0 ? requested : quantizer.num_quantizers;
        if (n_channels > quantizer.num_quantizers) {
            fail("decode channel count exceeds the quantizer count");
        }
        return n_channels;
    }

    void validate_codes(const std::vector<int32_t> & codes, int n_channels) const {
        if (encoder) {
            fail("decode called on an encoder checkpoint");
        }
        if (codes.empty() || codes.size() % (size_t) n_channels != 0) {
            fail("decode expects row-major [frames, channels] codes");
        }
        for (int32_t code : codes) {
            if (code < 0 || code >= quantizer.codebook_size) {
                fail("audio code out of range");
            }
        }
    }

};

Codec::Codec(const std::string & path, const BackendRequest & backend, int n_threads, bool measure_only,
             AttentionLayout attention) : impl_(new Impl) {
    impl_->measure_only = measure_only;
    impl_->attention_layout = attention;
    impl_->load(path, backend, n_threads);
    if (!impl_->encoder && !measure_only) {
        impl_->open_sidecar(path);
    }
}

FitResult Codec::measure(int64_t length, int n_channels, bool streaming) {
    if (!impl_->measure_only) fail("measurement requires a metadata-only model");
    FitResult result = fit_device(impl_->backend);
    result.device.weights_bytes = measure_fit_tensors(impl_->weights,
            ::tts_cpp::detail::weight_buffer_type(impl_->backend));
    if (streaming) {
        impl_->reset_streams();
        result.device.state_bytes = measure_fit_tensors(impl_->stream_ctx, impl_->backend);
        for (auto & stream : impl_->streams) {
            stream.position = stream.capacity;
            stream.cached = stream.capacity;
        }
    }
    impl_->begin_graph();
    ggml_tensor * output = impl_->encoder
        ? impl_->build_encode({}, align_up(length, impl_->downsample))
        : impl_->build_decode(std::vector<int32_t>(length * n_channels), length, n_channels, streaming);
    const auto price = measure_fit_graph(impl_->backend, impl_->finish_graph(output), GRAPH_NODES);
    result.device.codec_compute_bytes = price.device_bytes;
    result.host_bytes = fitutil::sat_add(price.host_bytes, impl_->graph_host_bytes);
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_nbytes(output));
    result.host_bytes = fitutil::sat_add(result.host_bytes, gguf_get_meta_size(impl_->file));
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->metadata));
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->weights));
    result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->graph_ctx));
    if (impl_->stream_ctx) result.host_bytes = fitutil::sat_add(result.host_bytes, ggml_get_mem_size(impl_->stream_ctx));
    return result;
}

Codec::~Codec() = default;

bool Codec::is_encoder() const { return impl_->encoder; }
int Codec::sample_rate() const { return impl_->rate; }
int Codec::num_quantizers() const { return impl_->quantizer.num_quantizers; }
int Codec::samples_per_frame() const { return (int) impl_->downsample; }
const char * Codec::backend_name() const { return ggml_backend_name(impl_->backend); }
bool Codec::fused_attention() const { return impl_->fused; }

std::vector<int32_t> Codec::encode(const std::vector<float> & pcm) {
    if (!impl_->encoder) {
        fail("encode called on a decoder checkpoint");
    }
    if (pcm.empty() || pcm.size() > (size_t) impl_->rate * MAX_CLIP_SECONDS) {
        fail("reference audio must contain up to " + std::to_string(MAX_CLIP_SECONDS) + " seconds of mono samples");
    }
    for (float sample : pcm) {
        if (!std::isfinite(sample)) {
            fail("reference audio contains non-finite samples");
        }
    }
    impl_->begin_graph();
    const int64_t padded = align_up((int64_t) pcm.size(), impl_->downsample);
    ggml_tensor * output = impl_->build_encode(pcm, padded);
    ggml_cgraph * graph = impl_->finish_graph(output);
    impl_->compute(graph);
    std::vector<int32_t> codes = impl_->read_i32(output);
    const size_t valid_frames = pcm.size() / (size_t) impl_->downsample;
    codes.resize(valid_frames * (size_t) impl_->quantizer.num_quantizers);
    return codes;
}

std::vector<float> Codec::decode(const std::vector<int32_t> & codes, int n_channels) {
    const int channels = impl_->resolve_channels(n_channels);
    impl_->validate_codes(codes, channels);
    return impl_->decode_batch(codes, channels);
}

void Codec::begin_decode_stream(int n_channels, int piece_frames) {
    if (impl_->encoder) {
        fail("decode streams need a decoder checkpoint");
    }
    impl_->begin_stream(n_channels, piece_frames);
}

std::vector<float> Codec::decode_stream(const std::vector<int32_t> & codes) {
    if (impl_->stream_channels == 0) {
        fail("decode_stream called before begin_decode_stream");
    }
    impl_->validate_codes(codes, impl_->stream_channels);
    return impl_->decode_stream_codes(codes);
}

int64_t Codec::frames_decoded() const {
    return impl_->frames_decoded;
}

CodecGeometry Codec::sidecar_geometry() const {
    return impl_->sidecar_geometry();
}

void Codec::attach_sidecar(std::unique_ptr<CodecSidecarModel> sidecar, bool strict) {
    impl_->attach_sidecar(std::move(sidecar), strict);
}

bool Codec::on_coreml() const {
    return impl_->sidecar != nullptr;
}

void Codec::begin_run() {
    impl_->backends.begin();
}

std::string Codec::run_backend() const {
    return impl_->backends.summary();
}

std::vector<float> decode_segments(Codec & codec, const std::vector<std::vector<int32_t>> & segments,
                                   int64_t max_single_frames, int n_channels) {
    std::vector<float> pcm;
    const size_t stride = (size_t) (n_channels > 0 ? n_channels : codec.num_quantizers());
    for (const std::vector<int32_t> & segment : segments) {
        const std::vector<float> rendered = segment.size() / stride <= (size_t) max_single_frames
                ? codec.decode(segment, n_channels)
                : decode_in_windows(codec, segment, max_single_frames, n_channels);
        pcm.insert(pcm.end(), rendered.begin(), rendered.end());
    }
    return pcm;
}

std::vector<float> decode_after_context(Codec & codec, const std::vector<std::vector<int32_t>> & segments,
                                        int64_t context_frames, int64_t max_single_frames,
                                        int n_channels) {
    std::vector<float> pcm = decode_segments(codec, segments, max_single_frames, n_channels);
    const size_t context = std::min(pcm.size(),
            (size_t) context_frames * (size_t) codec.samples_per_frame());
    pcm.erase(pcm.begin(), pcm.begin() + (std::ptrdiff_t) context);
    return pcm;
}

std::vector<float> decode_in_windows(Codec & codec, const std::vector<int32_t> & codes,
                                     int64_t window_frames, int n_channels) {
    const size_t channels = (size_t) (n_channels > 0 ? n_channels : codec.num_quantizers());
    const size_t stride = channels * (size_t) window_frames;
    std::vector<float> pcm;
    codec.begin_decode_stream(n_channels, (int) window_frames);
    for (size_t begin = 0; begin < codes.size(); begin += stride) {
        const size_t end = std::min(codes.size(), begin + stride);
        const std::vector<float> rendered = codec.decode_stream(
                std::vector<int32_t>(codes.begin() + (std::ptrdiff_t) begin,
                        codes.begin() + (std::ptrdiff_t) end));
        pcm.insert(pcm.end(), rendered.begin(), rendered.end());
    }
    return pcm;
}

} // namespace tts_cpp::moss::detail
