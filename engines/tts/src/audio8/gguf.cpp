#include "audio8/internal.h"

#include "backend_selection.h"
#include "gguf.h"
#include "gguf_stream.h"

#include <cstring>
#include <memory>
#include <string>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

// Reads the GGUF's metadata and tensor headers; the payload stays on disk
// until a gguf_stream_reader pulls it into the backend buffer.
class gguf_file {
public:
    explicit gguf_file(const std::string & path) : path_(path) {
        gguf_init_params params = {/*no_alloc=*/true, &meta_};
        ctx_ = gguf_init_from_file(path.c_str(), params);
    }
    ~gguf_file() {
        if (ctx_) gguf_free(ctx_);
        if (meta_) ggml_free(meta_);
    }
    gguf_file(const gguf_file &) = delete;
    gguf_file & operator=(const gguf_file &) = delete;

    bool ok() const { return ctx_ != nullptr; }
    const gguf_context * ctx() const { return ctx_; }
    const ggml_context * meta() const { return meta_; }
    const std::string & path() const { return path_; }

private:
    std::string path_;
    gguf_context * ctx_ = nullptr;
    ggml_context * meta_ = nullptr;
};

// Collects the first failure so a whole block of reads can be written as a
// straight list and checked once.
class metadata {
public:
    metadata(const gguf_context * ctx, const std::string & prefix)
        : ctx_(ctx), prefix_(prefix) {}

    void u32(const char * key, int & out) {
        const int64_t id = find(key);
        if (id >= 0) out = static_cast<int>(gguf_get_val_u32(ctx_, id));
    }
    void f32(const char * key, float & out) {
        const int64_t id = find(key);
        if (id >= 0) out = gguf_get_val_f32(ctx_, id);
    }
    void boolean(const char * key, bool & out) {
        const int64_t id = find(key);
        if (id >= 0) out = gguf_get_val_bool(ctx_, id);
    }
    void ints(const char * key, std::vector<int> & out) {
        const int64_t id = find(key);
        if (id < 0) return;
        const int count = static_cast<int>(gguf_get_arr_n(ctx_, id));
        const int32_t * values = static_cast<const int32_t *>(gguf_get_arr_data(ctx_, id));
        out.assign(values, values + count);
    }
    void strings(const char * key, std::vector<std::string> & out) {
        const int64_t id = find(key);
        if (id < 0) return;
        const int count = static_cast<int>(gguf_get_arr_n(ctx_, id));
        out.resize(count);
        for (int index = 0; index < count; ++index) {
            out[index] = gguf_get_arr_str(ctx_, id, index);
        }
    }
    std::string text(const char * key) {
        const int64_t id = find(key);
        return id >= 0 ? gguf_get_val_str(ctx_, id) : std::string();
    }

    bool ok() const { return error_.empty(); }
    const std::string & error() const { return error_; }

private:
    int64_t find(const char * key) {
        const std::string full = prefix_ + key;
        const int64_t id = gguf_find_key(ctx_, full.c_str());
        if (id < 0 && error_.empty()) error_ = "missing GGUF key: " + full;
        return id;
    }

    const gguf_context * ctx_;
    std::string prefix_;
    std::string error_;
};

// Absolute keys (tokenizer, architecture) sit outside the model's namespace.
class raw_metadata : public metadata {
public:
    explicit raw_metadata(const gguf_context * ctx) : metadata(ctx, "") {}
};

bool architecture_is(const gguf_file & file, const char * expected) {
    const int64_t id = gguf_find_key(file.ctx(), "general.architecture");
    return id >= 0 && std::string(gguf_get_val_str(file.ctx(), id)) == expected;
}

ggml_context * new_weight_context(int64_t n_tensors) {
    const size_t size = static_cast<size_t>(n_tensors + 8) * ggml_tensor_overhead();
    ggml_init_params params = {size, nullptr, /*no_alloc=*/true};
    return ggml_init(params);
}

void clone_tensor_headers(const ggml_context * meta, ggml_context * dst) {
    ggml_context * source = const_cast<ggml_context *>(meta);
    for (ggml_tensor * t = ggml_get_first_tensor(source); t;
         t = ggml_get_next_tensor(source, t)) {
        ggml_set_name(ggml_dup_tensor(dst, t), ggml_get_name(t));
    }
}

bool stream_weights(const gguf_file & file, ggml_context * ctx) {
    ::tts_cpp::detail::gguf_stream_reader reader(file.ctx(), file.path());
    if (!reader.ok()) return false;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!reader.to_backend(ggml_get_name(t), t)) return false;
    }
    return true;
}

bool load_weights(const gguf_file & file, ggml_backend_t backend, ggml_context ** ctx,
                  ggml_backend_buffer_t * buffer, std::string * error) {
    *ctx = new_weight_context(gguf_get_n_tensors(file.ctx()));
    if (!*ctx) {
        if (error) *error = "audio8: failed to create the weight context";
        return false;
    }
    clone_tensor_headers(file.meta(), *ctx);
    *buffer = ggml_backend_alloc_ctx_tensors(*ctx, backend);
    if (!*buffer) {
        if (error) *error = "audio8: failed to allocate the weight buffer";
        return false;
    }
    if (!stream_weights(file, *ctx)) {
        if (error) *error = "audio8: failed to read tensor data from " + file.path();
        return false;
    }
    return true;
}

// Looks tensors up by name and remembers the first one that was missing, so a
// wiring block reads as a plain list of assignments.
class tensor_map {
public:
    explicit tensor_map(ggml_context * ctx) : ctx_(ctx) {}

    ggml_tensor * get(const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(ctx_, name.c_str());
        if (!t && error_.empty()) error_ = "audio8: missing tensor " + name;
        return t;
    }
    ggml_tensor * maybe(const std::string & name) const {
        return ggml_get_tensor(ctx_, name.c_str());
    }
    bool ok() const { return error_.empty(); }
    const std::string & error() const { return error_; }

private:
    ggml_context * ctx_;
    std::string error_;
};

void read_attention(tensor_map & map, const std::string & prefix, bool has_bias,
                    attention_weights & attn) {
    attn.wq = map.get(prefix + "wq");
    attn.wk = map.get(prefix + "wk");
    attn.wv = map.get(prefix + "wv");
    attn.wo = map.get(prefix + "wo");
    attn.attn_norm = map.get(prefix + "attn_norm");
    if (!has_bias) return;
    attn.wq_b = map.get(prefix + "wq_b");
    attn.wk_b = map.get(prefix + "wk_b");
    attn.wv_b = map.get(prefix + "wv_b");
}

void read_block(tensor_map & map, const std::string & prefix, bool has_bias,
                block_weights & block) {
    read_attention(map, prefix, has_bias, block.attn);
    block.w1 = map.get(prefix + "w1");
    block.w2 = map.get(prefix + "w2");
    block.w3 = map.get(prefix + "w3");
    block.ffn_norm = map.get(prefix + "ffn_norm");
}

void read_branch(tensor_map & map, const std::string & branch, int depth, bool has_bias,
                 std::vector<block_weights> & blocks) {
    blocks.resize(depth);
    for (int index = 0; index < depth; ++index) {
        read_block(map, branch + "/blk/" + std::to_string(index) + "/", has_bias, blocks[index]);
    }
}

void read_lm_hparams(metadata & meta, lm_hparams & hp) {
    meta.u32("depth", hp.depth);
    meta.u32("hidden", hp.hidden);
    meta.u32("n_head", hp.n_head);
    meta.u32("n_kv", hp.n_kv);
    meta.u32("head_dim", hp.head_dim);
    meta.u32("inter", hp.inter);
    meta.u32("vocab", hp.vocab);
    meta.u32("fast_depth", hp.fast_depth);
    meta.u32("fast_hidden", hp.fast_hidden);
    meta.u32("fast_n_head", hp.fast_n_head);
    meta.u32("fast_n_kv", hp.fast_n_kv);
    meta.u32("fast_head_dim", hp.fast_head_dim);
    meta.u32("fast_inter", hp.fast_inter);
    meta.u32("num_codebooks", hp.num_codebooks);
    meta.u32("codebook_size", hp.codebook_size);
    meta.u32("semantic_begin", hp.semantic_begin);
    meta.u32("semantic_end", hp.semantic_end);
    meta.u32("eos", hp.eos);
    meta.u32("pad", hp.pad);
    meta.u32("max_seq_len", hp.max_seq_len);
    meta.u32("ras_window", hp.ras_window);
    meta.f32("rope_theta", hp.rope_theta);
    meta.f32("rms_eps", hp.rms_eps);
    meta.f32("ras_top_p", hp.ras_top_p);
    meta.f32("ras_temperature", hp.ras_temperature);
    meta.boolean("norm_fast_input", hp.norm_fast_input);
    meta.boolean("qkv_bias", hp.qkv_bias);
    meta.boolean("fast_qkv_bias", hp.fast_qkv_bias);
}

void read_tokenizer(raw_metadata & meta, TokenizerData & data) {
    meta.strings("tokenizer.ggml.tokens", data.tokens);
    meta.strings("tokenizer.ggml.merges", data.merges);
    meta.ints("tokenizer.ggml.added_token_ids", data.added_token_ids);
}

// K is position-major so a step appends whole rows; V is channel-major so the
// value matmul reads runs of positions without transposing the prefix.
bool alloc_kv(ggml_backend_t backend, int layers, int width, int capacity, kv_cache & cache) {
    ggml_init_params params = {4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true};
    cache.ctx = ggml_init(params);
    if (!cache.ctx) return false;
    cache.capacity = capacity;
    cache.stride = width;
    cache.k = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, width,
                                 static_cast<int64_t>(capacity) * layers);
    cache.v = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, capacity,
                                 static_cast<int64_t>(width) * layers);
    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    return cache.buffer != nullptr;
}

void free_kv(kv_cache & cache) {
    if (cache.buffer) ggml_backend_buffer_free(cache.buffer);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = kv_cache{};
}

void read_transformer_spec(metadata & meta, const std::string & name, transformer_spec & spec) {
    const std::string prefix = name + ".";
    meta.u32((prefix + "n_layer").c_str(), spec.n_layer);
    meta.u32((prefix + "n_head").c_str(), spec.n_head);
    meta.u32((prefix + "n_kv").c_str(), spec.n_kv);
    meta.u32((prefix + "dim").c_str(), spec.dim);
    meta.u32((prefix + "inter").c_str(), spec.inter);
    meta.u32((prefix + "window").c_str(), spec.window);
    meta.u32((prefix + "head_dim").c_str(), spec.head_dim);
    meta.f32((prefix + "rope_theta").c_str(), spec.rope_theta);
    meta.f32((prefix + "norm_eps").c_str(), spec.norm_eps);
}

void read_scaled_block(tensor_map & map, const std::string & prefix,
                       scaled_block_weights & block) {
    read_attention(map, prefix, /*has_bias=*/false, block.attn);
    block.attn_scale = map.get(prefix + "attn_scale");
    block.w1 = map.get(prefix + "w1");
    block.w2 = map.get(prefix + "w2");
    block.w3 = map.get(prefix + "w3");
    block.ffn_norm = map.get(prefix + "ffn_norm");
    block.ffn_scale = map.get(prefix + "ffn_scale");
}

void read_window_transformer(tensor_map & map, const std::string & prefix,
                             const std::string & rope, window_transformer & tf) {
    tf.blocks.resize(tf.spec.n_layer);
    for (int index = 0; index < tf.spec.n_layer; ++index) {
        read_scaled_block(map, prefix + "/blk/" + std::to_string(index) + "/", tf.blocks[index]);
    }
    tf.norm = map.get(prefix + "/norm/weight");
    tf.rope_cos = map.get("rope/" + rope + "_cos");
    tf.rope_sin = map.get("rope/" + rope + "_sin");
}

void read_conv(tensor_map & map, const std::string & prefix, conv_weights & conv) {
    conv.w = map.get(prefix + "/weight");
    conv.b = map.get(prefix + "/bias");
}

void read_convnext(tensor_map & map, const std::string & prefix, convnext_weights & block) {
    read_conv(map, prefix + "/dwconv/conv", block.dwconv);
    block.norm_w = map.get(prefix + "/norm/weight");
    block.norm_b = map.get(prefix + "/norm/bias");
    read_conv(map, prefix + "/pwconv1", block.pwconv1);
    read_conv(map, prefix + "/pwconv2", block.pwconv2);
    block.gamma = map.get(prefix + "/gamma");
}

// The quantizer's resamplers are a fixed pair of stride-2 stages, each a
// conv followed by a ConvNeXt block.
constexpr int RESAMPLE_STAGES = 2;
constexpr int RESAMPLE_STRIDE = 2;

void read_resamplers(tensor_map & map, const std::string & prefix,
                     std::vector<resample_stage> & stages) {
    stages.resize(RESAMPLE_STAGES);
    for (int index = 0; index < RESAMPLE_STAGES; ++index) {
        const std::string base = prefix + "/" + std::to_string(index);
        read_conv(map, base + "/0/conv", stages[index].conv);
        read_convnext(map, base + "/1", stages[index].convnext);
        stages[index].stride = RESAMPLE_STRIDE;
    }
}

void read_quantizers(tensor_map & map, const std::string & prefix, int count,
                     bool encoding, std::vector<quantizer_weights> & out) {
    out.resize(count);
    for (int index = 0; index < count; ++index) {
        const std::string base = prefix + "/" + std::to_string(index);
        out[index].codebook = map.get(base + "/codebook/weight");
        read_conv(map, base + "/out_proj", out[index].out_proj);
        if (encoding) read_conv(map, base + "/in_proj", out[index].in_proj);
    }
}

void read_residual_units(tensor_map & map, const std::string & prefix, int first, int count,
                         std::vector<residual_unit> & units) {
    units.resize(count);
    for (int index = 0; index < count; ++index) {
        const std::string base = prefix + "/" + std::to_string(index + first) + "/block";
        units[index].alpha1 = map.get(base + "/0/alpha");
        read_conv(map, base + "/1/conv", units[index].conv1);
        units[index].alpha2 = map.get(base + "/2/alpha");
        read_conv(map, base + "/3/conv", units[index].conv2);
    }
}

// The GGUF keeps the reference's Sequential indices, so stage n of a part sits
// at <part>/(n+1) and the entries below are that Sequential's own positions.
std::string stage_prefix(const std::string & part, size_t index) {
    return part + "/" + std::to_string(index + 1) + "/block";
}

void read_decoder_stages(tensor_map & map, const std::vector<int> & strides, int dilations,
                         std::vector<dac_stage> & stages) {
    stages.resize(strides.size());
    for (size_t index = 0; index < strides.size(); ++index) {
        const std::string base = stage_prefix("dec", index);
        stages[index].alpha = map.get(base + "/0/alpha");
        read_conv(map, base + "/1/conv", stages[index].conv);
        stages[index].stride = strides[index];
        read_residual_units(map, base, /*first=*/2, dilations, stages[index].units);
    }
}

void read_encoder_stages(tensor_map & map, const std::vector<int> & strides, int dilations,
                         const transformer_spec & spec, std::vector<dac_stage> & stages) {
    stages.resize(strides.size());
    for (size_t index = 0; index < strides.size(); ++index) {
        const std::string base = stage_prefix("enc", index);
        read_residual_units(map, base, /*first=*/0, dilations, stages[index].units);
        stages[index].alpha = map.get(base + "/" + std::to_string(dilations) + "/alpha");
        read_conv(map, base + "/" + std::to_string(dilations + 1) + "/conv", stages[index].conv);
        stages[index].stride = strides[index];
        const std::string nested = base + "/" + std::to_string(dilations + 2);
        stages[index].has_transformer = map.maybe(nested + "/norm/weight") != nullptr;
        if (!stages[index].has_transformer) continue;
        stages[index].transformer.spec = spec;
        read_window_transformer(map, nested, "enc_tf", stages[index].transformer);
    }
}

void read_codec_hparams(metadata & meta, codec_hparams & hp) {
    meta.u32("sample_rate", hp.sample_rate);
    meta.u32("frame_size", hp.frame_size);
    meta.u32("num_codebooks", hp.num_codebooks);
    meta.u32("latent_dim", hp.latent_dim);
    meta.u32("codebook_dim", hp.codebook_dim);
    meta.u32("max_frames", hp.max_frames);
    meta.u32("semantic_codebook_size", hp.semantic_codebook_size);
    meta.u32("residual_codebook_size", hp.residual_codebook_size);
    meta.u32("residual_codebooks", hp.residual_codebooks);
    meta.f32("convnext_norm_eps", hp.convnext_norm_eps);
    meta.f32("snake_epsilon", hp.snake_epsilon);
    meta.ints("encoder_strides", hp.encoder_strides);
    meta.ints("decoder_strides", hp.decoder_strides);
    meta.ints("residual_dilations", hp.residual_dilations);
}

bool read_codec_header(const gguf_file & file, codec_header & header,
                       std::string * error) {
    if (!architecture_is(file, "audio8-codec")) {
        if (error) *error = "audio8: " + file.path() + " is not an audio8-codec GGUF";
        return false;
    }
    metadata meta(file.ctx(), "audio8.codec.");
    read_codec_hparams(meta, header.hp);
    header.part = meta.text("part");
    if (!meta.ok()) {
        if (error) *error = meta.error();
        return false;
    }
    if (header.part != "encoder" && header.part != "decoder") {
        if (error) {
            *error = "audio8: unexpected codec part '" + header.part + "' in " +
                     file.path();
        }
        return false;
    }
    return true;
}

void read_quantizer_bank(tensor_map & map, codec_model & model, bool encoding) {
    read_quantizers(map, "q/sem", 1, encoding, model.semantic_quantizers);
    read_quantizers(map, "q/res", model.hp.residual_codebooks, encoding,
                    model.residual_quantizers);
}

void read_decoder(tensor_map & map, codec_model & model) {
    const codec_hparams & hp = model.hp;
    const int dilations = static_cast<int>(hp.residual_dilations.size());
    read_quantizer_bank(map, model, /*encoding=*/false);
    read_window_transformer(map, "q/post", "post_tf", model.post);
    read_resamplers(map, "q/up", model.upsample);
    read_conv(map, "dec/0/conv", model.dec_in);
    read_decoder_stages(map, hp.decoder_strides, dilations, model.dec_stages);
    const size_t tail = hp.decoder_strides.size() + 1;
    model.dec_out_alpha = map.get("dec/" + std::to_string(tail) + "/alpha");
    read_conv(map, "dec/" + std::to_string(tail + 1) + "/conv", model.dec_out);
}

void read_encoder(tensor_map & map, codec_model & model, const transformer_spec & nested) {
    const codec_hparams & hp = model.hp;
    const int dilations = static_cast<int>(hp.residual_dilations.size());
    read_conv(map, "enc/0/conv", model.enc_in);
    read_encoder_stages(map, hp.encoder_strides, dilations, nested, model.enc_stages);
    const size_t tail = hp.encoder_strides.size() + 1;
    model.enc_out_alpha = map.get("enc/" + std::to_string(tail) + "/alpha");
    read_conv(map, "enc/" + std::to_string(tail + 1) + "/conv", model.enc_out);
    read_window_transformer(map, "q/pre", "pre_tf", model.pre);
    read_resamplers(map, "q/down", model.downsample);
    read_quantizer_bank(map, model, /*encoding=*/true);
}

}  // namespace

bool load_lm(const std::string & path, lm_model & model, std::string * error) {
    gguf_file file(path);
    if (!file.ok()) {
        if (error) *error = "audio8: failed to open " + path;
        return false;
    }
    if (!architecture_is(file, "audio8-lm")) {
        if (error) *error = "audio8: " + path + " is not an audio8-lm GGUF";
        return false;
    }

    metadata meta(file.ctx(), "audio8.lm.");
    read_lm_hparams(meta, model.hp);
    raw_metadata shared(file.ctx());
    read_tokenizer(shared, model.tokenizer);
    if (!meta.ok() || !shared.ok()) {
        if (error) *error = meta.ok() ? shared.error() : meta.error();
        return false;
    }

    ::tts_cpp::detail::ensure_backends_loaded();
    model.backend = ::tts_cpp::detail::init_cpu_backend();
    if (!model.backend) {
        if (error) *error = "audio8: failed to init the CPU backend";
        return false;
    }
    if (!load_weights(file, model.backend, &model.ctx_w, &model.buffer_w, error)) {
        return false;
    }

    tensor_map map(model.ctx_w);
    model.tok_emb = map.get("lm/tok_emb");
    model.codebook_emb = map.get("lm/codebook_emb");
    model.norm = map.get("lm/norm");
    model.sem_head = map.get("lm/sem_head");
    model.rope_cos = map.get("lm/rope_cos");
    model.rope_sin = map.get("lm/rope_sin");
    read_branch(map, "lm", model.hp.depth, model.hp.qkv_bias, model.blocks);
    model.fast_emb = map.get("fast/emb");
    model.fast_norm = map.get("fast/norm");
    model.fast_out = map.get("fast/out");
    model.fast_rope_cos = map.get("fast/rope_cos");
    model.fast_rope_sin = map.get("fast/rope_sin");
    read_branch(map, "fast", model.hp.fast_depth, model.hp.fast_qkv_bias, model.fast_blocks);
    if (!map.ok()) {
        if (error) *error = map.error();
        return false;
    }

    const lm_hparams & hp = model.hp;
    if (!alloc_kv(model.backend, hp.depth, hp.n_kv * hp.head_dim, hp.max_seq_len,
                  model.slow_kv) ||
        !alloc_kv(model.backend, hp.fast_depth, hp.fast_n_kv * hp.fast_head_dim,
                  hp.num_codebooks, model.fast_kv)) {
        if (error) *error = "audio8: failed to allocate the KV cache";
        return false;
    }
    ggml_backend_buffer_type_t buffer_type = ggml_backend_get_default_buffer_type(model.backend);
    model.slow_allocr = ggml_gallocr_new(buffer_type);
    model.fast_allocr = ggml_gallocr_new(buffer_type);
    if (!model.slow_allocr || !model.fast_allocr) {
        if (error) *error = "audio8: failed to create the graph allocators";
        return false;
    }
    return true;
}

void free_lm(lm_model & model) {
    if (model.slow_allocr) ggml_gallocr_free(model.slow_allocr);
    if (model.fast_allocr) ggml_gallocr_free(model.fast_allocr);
    free_kv(model.slow_kv);
    free_kv(model.fast_kv);
    if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
    if (model.ctx_w) ggml_free(model.ctx_w);
    if (model.backend) ggml_backend_free(model.backend);
    model.slow_allocr = nullptr;
    model.fast_allocr = nullptr;
    model.buffer_w = nullptr;
    model.ctx_w = nullptr;
    model.backend = nullptr;
}

bool peek_codec_header(const std::string & path, codec_header & header,
                       std::string * error) {
    gguf_file file(path);
    if (!file.ok()) {
        if (error) *error = "audio8: failed to open " + path;
        return false;
    }
    return read_codec_header(file, header, error);
}

bool load_codec(const std::string & path, codec_model & model, std::string * error) {
    gguf_file file(path);
    if (!file.ok()) {
        if (error) *error = "audio8: failed to open " + path;
        return false;
    }
    codec_header header;
    if (!read_codec_header(file, header, error)) return false;
    model.hp = header.hp;
    model.has_decoder = header.part == "decoder";
    model.has_encoder = header.part == "encoder";

    metadata meta(file.ctx(), "audio8.codec.");
    transformer_spec nested;
    if (model.has_decoder) {
        read_transformer_spec(meta, "post_tf", model.post.spec);
    } else {
        read_transformer_spec(meta, "pre_tf", model.pre.spec);
        read_transformer_spec(meta, "enc_tf", nested);
    }
    if (!meta.ok()) {
        if (error) *error = meta.error();
        return false;
    }

    ::tts_cpp::detail::ensure_backends_loaded();
    model.backend = ::tts_cpp::detail::init_cpu_backend();
    if (!model.backend) {
        if (error) *error = "audio8: failed to init the CPU backend";
        return false;
    }
    if (!load_weights(file, model.backend, &model.ctx_w, &model.buffer_w, error)) {
        return false;
    }

    tensor_map map(model.ctx_w);
    if (model.has_decoder) {
        read_decoder(map, model);
    } else {
        read_encoder(map, model, nested);
    }
    if (!map.ok()) {
        if (error) *error = map.error();
        return false;
    }

    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_get_default_buffer_type(model.backend);
    model.allocr = ggml_gallocr_new(buffer_type);
    model.block_allocr = ggml_gallocr_new(buffer_type);
    if (!model.allocr || !model.block_allocr) {
        if (error) *error = "audio8: failed to create the graph allocators";
        return false;
    }
    return true;
}

void free_codec(codec_model & model) {
    if (model.allocr) ggml_gallocr_free(model.allocr);
    if (model.block_allocr) ggml_gallocr_free(model.block_allocr);
    if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
    if (model.ctx_w) ggml_free(model.ctx_w);
    if (model.backend) ggml_backend_free(model.backend);
    model.allocr = nullptr;
    model.block_allocr = nullptr;
    model.buffer_w = nullptr;
    model.ctx_w = nullptr;
    model.backend = nullptr;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
