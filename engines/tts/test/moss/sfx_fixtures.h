#pragma once

#include "gguf_fixtures.h"

#include <array>

namespace moss_sfx_fixtures {

using moss_fixtures::GgufBuilder;

constexpr int TEXT_EMBD = 16;
constexpr int TEXT_FF = 32;
constexpr int TEXT_HEADS = 2;
constexpr int TEXT_KV_HEADS = 1;
constexpr int TEXT_HEAD_DIM = 8;
constexpr int TEXT_TOKENS = 8;
constexpr int DIT_EMBD = 16;
constexpr int DIT_FF = 32;
constexpr int DIT_HEADS = 2;
constexpr int LATENT = 4;
constexpr int FREQ_DIM = 8;
constexpr int DECODER_DIM = 8;
constexpr std::array<int, 2> RATES = {2, 3};
constexpr int HOP = 6;
constexpr int SAMPLE_RATE = 8000;
constexpr float MAX_SECONDS = 0.2f;
constexpr int FRAMES = 266;
constexpr int STEPS = 2;
constexpr int RESIDUAL_UNITS = 3;
constexpr int KERNEL = 7;

inline void add_text_meta(gguf_context * f) {
    gguf_set_val_u32(f, "moss-sfx.text.block_count", 1);
    gguf_set_val_u32(f, "moss-sfx.text.context_length", TEXT_TOKENS);
    gguf_set_val_u32(f, "moss-sfx.text.embedding_length", TEXT_EMBD);
    gguf_set_val_u32(f, "moss-sfx.text.feed_forward_length", TEXT_FF);
    gguf_set_val_u32(f, "moss-sfx.text.attention.head_count", TEXT_HEADS);
    gguf_set_val_u32(f, "moss-sfx.text.attention.head_count_kv", TEXT_KV_HEADS);
    gguf_set_val_u32(f, "moss-sfx.text.attention.key_length", TEXT_HEAD_DIM);
    gguf_set_val_f32(f, "moss-sfx.text.rope.freq_base", 1000000.0f);
    gguf_set_val_f32(f, "moss-sfx.text.attention.layer_norm_rms_epsilon", 1e-6f);
}

inline void add_dit_meta(gguf_context * f) {
    gguf_set_val_u32(f, "moss-sfx.dit.block_count", 1);
    gguf_set_val_u32(f, "moss-sfx.dit.embedding_length", DIT_EMBD);
    gguf_set_val_u32(f, "moss-sfx.dit.feed_forward_length", DIT_FF);
    gguf_set_val_u32(f, "moss-sfx.dit.attention.head_count", DIT_HEADS);
    gguf_set_val_u32(f, "moss-sfx.dit.in_channels", LATENT);
    gguf_set_val_u32(f, "moss-sfx.dit.out_channels", LATENT);
    gguf_set_val_u32(f, "moss-sfx.dit.text_dim", TEXT_EMBD);
    gguf_set_val_u32(f, "moss-sfx.dit.freq_dim", FREQ_DIM);
    gguf_set_val_f32(f, "moss-sfx.dit.epsilon", 1e-6f);
}

inline void add_vae_meta(gguf_context * f) {
    gguf_set_val_u32(f, "moss-sfx.vae.latent_dim", LATENT);
    gguf_set_val_u32(f, "moss-sfx.vae.decoder_dim", DECODER_DIM);
    gguf_set_val_u32(f, "moss-sfx.vae.sample_rate", SAMPLE_RATE);
    gguf_set_arr_data(f, "moss-sfx.vae.decoder_rates", GGUF_TYPE_INT32, RATES.data(), RATES.size());
}

inline void add_generation_meta(gguf_context * f) {
    gguf_set_val_f32(f, "moss-sfx.max_seconds", MAX_SECONDS);
    gguf_set_val_f32(f, "moss-sfx.sigma_shift", 5.0f);
    gguf_set_val_u32(f, "moss-sfx.num_train_timesteps", 1000);
    gguf_set_val_u32(f, "moss-sfx.default_steps", STEPS);
    gguf_set_val_f32(f, "moss-sfx.default_guidance", 4.0f);
}

inline void add_tokenizer_meta(gguf_context * f) {
    const std::vector<std::string> tokens = moss_fixtures::byte_complete_vocab();
    std::vector<const char *> token_ptrs;
    for (const std::string & token : tokens) {
        token_ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(f, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    const char * merges[] = {"a b"};
    gguf_set_arr_str(f, "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_u32(f, "tokenizer.ggml.padding_token_id", 0);
}

inline void add_meta(gguf_context * f) {
    gguf_set_val_str(f, "general.architecture", "moss-sfx");
    add_text_meta(f);
    add_dit_meta(f);
    add_vae_meta(f);
    add_generation_meta(f);
    add_tokenizer_meta(f);
}

inline void fill_random_f16(ggml_tensor * tensor, std::mt19937 & rng) {
    std::uniform_real_distribution<float> draw(-moss_fixtures::RANDOM_WEIGHT_RANGE, moss_fixtures::RANDOM_WEIGHT_RANGE);
    auto * data = (ggml_fp16_t *) tensor->data;
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = ggml_fp32_to_fp16(draw(rng));
    }
}

inline bool & f32_kernels() {
    static bool enabled = false;
    return enabled;
}

inline void add_f16_kernel(GgufBuilder & b, const std::string & name, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (f32_kernels()) {
        b.register_tensor(ggml_new_tensor_3d(b.tensors, GGML_TYPE_F32, ne0, ne1, ne2), name);
        return;
    }
    ggml_tensor * tensor = ggml_new_tensor_3d(b.tensors, GGML_TYPE_F16, ne0, ne1, ne2);
    fill_random_f16(tensor, *b.random);
    ggml_set_name(tensor, name.c_str());
    gguf_add_tensor(b.file, tensor);
}

inline void add_text_tensors(GgufBuilder & b) {
    const int64_t q_dim = (int64_t) TEXT_HEADS * TEXT_HEAD_DIM;
    const int64_t kv_dim = (int64_t) TEXT_KV_HEADS * TEXT_HEAD_DIM;
    b.add2("text.token_embd.weight", TEXT_EMBD, moss_fixtures::TEXT_VOCAB);
    b.add1("text.output_norm.weight", TEXT_EMBD);
    b.add1("text.blk.0.attn_norm.weight", TEXT_EMBD);
    b.add2("text.blk.0.attn_q.weight", TEXT_EMBD, q_dim);
    b.add2("text.blk.0.attn_k.weight", TEXT_EMBD, kv_dim);
    b.add2("text.blk.0.attn_v.weight", TEXT_EMBD, kv_dim);
    b.add2("text.blk.0.attn_output.weight", q_dim, TEXT_EMBD);
    b.add1("text.blk.0.attn_q_norm.weight", TEXT_HEAD_DIM);
    b.add1("text.blk.0.attn_k_norm.weight", TEXT_HEAD_DIM);
    b.add1("text.blk.0.ffn_norm.weight", TEXT_EMBD);
    b.add2("text.blk.0.ffn_gate.weight", TEXT_EMBD, TEXT_FF);
    b.add2("text.blk.0.ffn_up.weight", TEXT_EMBD, TEXT_FF);
    b.add2("text.blk.0.ffn_down.weight", TEXT_FF, TEXT_EMBD);
}

inline void add_linear(GgufBuilder & b, const std::string & name, int64_t in, int64_t out) {
    b.add2(name + ".weight", in, out);
    b.add1(name + ".bias", out);
}

inline void add_dit_block(GgufBuilder & b, int layer = 0, int width = DIT_EMBD) {
    const std::string prefix = "dit.blk." + std::to_string(layer) + ".";
    for (const char * name : {"self_q", "self_k", "self_v", "self_o", "cross_q", "cross_k", "cross_v", "cross_o"}) {
        add_linear(b, prefix + name, width, width);
    }
    for (const char * name : {"self_norm_q", "self_norm_k", "cross_norm_q", "cross_norm_k"}) {
        b.add1(prefix + name + ".weight", width);
    }
    b.add1(prefix + "cross_norm.weight", width);
    b.add1(prefix + "cross_norm.bias", width);
    add_linear(b, prefix + "ffn_up", width, DIT_FF);
    add_linear(b, prefix + "ffn_down", DIT_FF, width);
    b.add2(prefix + "modulation", width, 6);
}

inline void add_dit_tensors(GgufBuilder & b, int layers = 1, int width = DIT_EMBD) {
    add_linear(b, "dit.patch_embd", LATENT, width);
    add_linear(b, "dit.text_embd_1", TEXT_EMBD, width);
    add_linear(b, "dit.text_embd_2", width, width);
    add_linear(b, "dit.time_embd_1", FREQ_DIM, width);
    add_linear(b, "dit.time_embd_2", width, width);
    add_linear(b, "dit.time_proj", width, width * 6);
    add_linear(b, "dit.head", width, LATENT);
    b.add2("dit.head.modulation", width, 2);
    for (int layer = 0; layer < layers; ++layer) add_dit_block(b, layer, width);
}

inline void add_snake(GgufBuilder & b, const std::string & name, int channels) {
    b.add2(name + ".alpha", 1, channels);
    b.add2(name + ".inv", 1, channels);
}

inline void add_conv(GgufBuilder & b, const std::string & name, int kernel, int in, int out) {
    add_f16_kernel(b, name + ".weight", kernel, in, out);
    b.add2(name + ".bias", 1, out);
}

inline void add_vae_block(GgufBuilder & b, int block, int in, int out) {
    const std::string name = "vae.blk." + std::to_string(block);
    add_snake(b, name + ".snake", in);
    b.add2(name + ".up.weight", in, (int64_t) 2 * RATES[(size_t) block] * out);
    b.add2(name + ".up.bias", 1, out);
    for (int unit = 0; unit < RESIDUAL_UNITS; ++unit) {
        const std::string res = name + ".res." + std::to_string(unit);
        add_snake(b, res + ".snake1", out);
        add_conv(b, res + ".conv1", KERNEL, out, out);
        add_snake(b, res + ".snake2", out);
        add_conv(b, res + ".conv2", 1, out, out);
    }
}

inline void add_vae_tensors(GgufBuilder & b, int decoder_dim = DECODER_DIM) {
    b.add2("vae.post_quant.weight", LATENT, LATENT);
    b.add2("vae.post_quant.bias", 1, LATENT);
    add_conv(b, "vae.conv_in", KERNEL, LATENT, decoder_dim);
    int channels = decoder_dim;
    for (int block = 0; block < (int) RATES.size(); ++block) {
        add_vae_block(b, block, channels, channels / 2);
        channels /= 2;
    }
    add_snake(b, "vae.snake_out", channels);
    add_conv(b, "vae.conv_out", KERNEL, channels, 1);
}

inline void scale_tensor(ggml_tensor * tensor, float scale) {
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        if (tensor->type == GGML_TYPE_F16) {
            auto * data = static_cast<ggml_fp16_t *>(tensor->data);
            data[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(data[i]) * scale);
        } else {
            static_cast<float *>(tensor->data)[i] *= scale;
        }
    }
}

inline void scale_vae_tensors(GgufBuilder & b, float scale) {
    for (auto * tensor = ggml_get_first_tensor(b.tensors); tensor; tensor = ggml_get_next_tensor(b.tensors, tensor)) {
        if (std::string(ggml_get_name(tensor)).find("vae.") == 0) scale_tensor(tensor, scale);
    }
}

inline std::filesystem::path write_sfx_model(
    const char * tag, uint32_t seed, const std::function<void(gguf_context *)> & mutate_meta = [](gguf_context *) {},
    const moss_fixtures::ConstantTensors & constants = {}, const moss_fixtures::TensorShapes & shapes = {},
    int dit_layers = 1, int dit_width = DIT_EMBD, int dit_heads = DIT_HEADS, int decoder_dim = DECODER_DIM,
    float vae_scale = 1.0f) {
    std::mt19937 rng(seed);
    GgufBuilder b;
    b.random = &rng;
    b.constants = constants;
    b.shapes = shapes;
    add_meta(b.file);
    gguf_set_val_u32(b.file, "moss-sfx.dit.block_count", dit_layers);
    gguf_set_val_u32(b.file, "moss-sfx.dit.embedding_length", dit_width);
    gguf_set_val_u32(b.file, "moss-sfx.dit.attention.head_count", dit_heads);
    gguf_set_val_u32(b.file, "moss-sfx.vae.decoder_dim", decoder_dim);
    mutate_meta(b.file);
    add_text_tensors(b);
    add_dit_tensors(b, dit_layers, dit_width);
    add_vae_tensors(b, decoder_dim);
    if (vae_scale != 1.0f) scale_vae_tensors(b, vae_scale);
    const auto path = moss_fixtures::temp_gguf(tag);
    b.write(path);
    return path;
}

} // namespace moss_sfx_fixtures
