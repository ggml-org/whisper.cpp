#pragma once

#include "qwen_tokenizer.h"

#include "gguf.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moss_fixtures {

constexpr int N_EMBD     = 16;
constexpr int N_FF       = 32;
constexpr int HEAD_DIM   = 8;
constexpr int N_HEADS    = 2;
constexpr int N_KV_HEADS = 1;
constexpr int N_VQ       = 2;
constexpr int SPECIALS   = 8;
constexpr int TEXT_VOCAB = SPECIALS + 256;
constexpr int D_MODEL    = 8;
constexpr int RVQ_DIM    = 8;
constexpr int CODE_DIM   = 4;
constexpr int CODE_SIZE  = 8;
constexpr int AUDIO_HEAD = CODE_SIZE + 1;
constexpr int OUT_DIM    = 4;
constexpr int GEN_SLOT   = 6;
constexpr int DELAY_SLOT = 7;

inline std::filesystem::path temp_gguf(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
        ("moss-" + std::string(tag) + "-" + std::to_string(stamp) + ".gguf");
}

struct ConstantTensor {
    ggml_type type;
    float value;
};

using ConstantTensors = std::map<std::string, ConstantTensor>;
using TensorShapes = std::map<std::string, std::pair<int64_t, int64_t>>;

constexpr float RANDOM_WEIGHT_RANGE = 0.5f;

struct GgufBuilder {
    gguf_context * file;
    ggml_context * tensors;
    ConstantTensors constants;
    TensorShapes shapes;
    std::mt19937 * random = nullptr;

    GgufBuilder() : file(gguf_init_empty()),
                    tensors(ggml_init({16 * 1024 * 1024, nullptr, false})) {}

    ~GgufBuilder() {
        ggml_free(tensors);
        gguf_free(file);
    }

    ggml_type type_of(const std::string & name) const {
        const auto found = constants.find(name);
        return found == constants.end() ? GGML_TYPE_F32 : found->second.type;
    }

    void fill_constant(ggml_tensor * t, float value) {
        const int64_t n = ggml_nelements(t);
        for (int64_t i = 0; i < n; ++i) {
            if (t->type == GGML_TYPE_F16) {
                ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(value);
            } else {
                ((float *) t->data)[i] = value;
            }
        }
    }

    void fill_random(ggml_tensor * t) {
        std::uniform_real_distribution<float> draw(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE);
        const int64_t n = ggml_nelements(t);
        for (int64_t i = 0; i < n; ++i) {
            ((float *) t->data)[i] = draw(*random);
        }
    }

    void fill(ggml_tensor * t, const std::string & name) {
        const auto found = constants.find(name);
        if (found != constants.end()) {
            fill_constant(t, found->second.value);
        } else if (random != nullptr) {
            fill_random(t);
        } else {
            std::memset(t->data, 0, ggml_nbytes(t));
        }
    }

    void register_tensor(ggml_tensor * t, const std::string & name) {
        fill(t, name);
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(file, t);
    }

    void add1(const std::string & name, int64_t n) {
        register_tensor(ggml_new_tensor_1d(tensors, type_of(name), n), name);
    }

    void add2(const std::string & name, int64_t ne0, int64_t ne1) {
        const auto shaped = shapes.find(name);
        if (shaped != shapes.end()) {
            ne0 = shaped->second.first;
            ne1 = shaped->second.second;
        }
        register_tensor(ggml_new_tensor_2d(tensors, type_of(name), ne0, ne1), name);
    }

    void write(const std::filesystem::path & path) {
        if (!gguf_write_to_file(file, path.string().c_str(), false)) {
            throw std::runtime_error("cannot write test GGUF");
        }
    }
};

inline std::vector<std::string> byte_complete_vocab() {
    std::vector<std::string> tokens(TEXT_VOCAB);
    tokens[0] = "<|endoftext|>";
    tokens[1] = "<|im_start|>";
    tokens[2] = "<|im_end|>";
    for (int i = 3; i < SPECIALS; ++i) {
        tokens[(size_t) i] = "tok" + std::to_string(i);
    }
    QwenTokenizer byte_map;
    byte_map.build_byte_map();
    for (int b = 0; b < 256; ++b) {
        tokens[(size_t) (SPECIALS + b)] = byte_map.byte2u[b];
    }
    return tokens;
}

inline void add_backbone_meta(gguf_context * f, int n_embd = N_EMBD) {
    gguf_set_val_str(f, "general.architecture", "moss-tts-delay");
    gguf_set_val_u32(f, "moss-tts-delay.block_count", 1);
    gguf_set_val_u32(f, "moss-tts-delay.embedding_length", (uint32_t) n_embd);
    gguf_set_val_u32(f, "moss-tts-delay.feed_forward_length", N_FF);
    gguf_set_val_u32(f, "moss-tts-delay.attention.head_count", N_HEADS);
    gguf_set_val_u32(f, "moss-tts-delay.attention.head_count_kv", N_KV_HEADS);
    gguf_set_val_u32(f, "moss-tts-delay.attention.key_length", HEAD_DIM);
    gguf_set_val_u32(f, "moss-tts-delay.context_length", 512);
    gguf_set_val_u32(f, "moss-tts-delay.n_vq", N_VQ);
    gguf_set_val_u32(f, "moss-tts-delay.audio_vocab_size", AUDIO_HEAD - 1);
    gguf_set_val_u32(f, "moss-tts-delay.audio_pad_code", AUDIO_HEAD - 1);
    gguf_set_val_u32(f, "moss-tts-delay.audio_start_token_id", 3);
    gguf_set_val_u32(f, "moss-tts-delay.audio_end_token_id", 4);
    gguf_set_val_u32(f, "moss-tts-delay.audio_user_slot_token_id", 5);
    gguf_set_val_u32(f, "moss-tts-delay.audio_assistant_gen_slot_token_id", GEN_SLOT);
    gguf_set_val_u32(f, "moss-tts-delay.audio_assistant_delay_slot_token_id", DELAY_SLOT);
    gguf_set_val_u32(f, "moss-tts-delay.sampling_rate", 24000);

    const std::vector<std::string> tokens = byte_complete_vocab();
    std::vector<const char *> token_ptrs;
    for (const std::string & token : tokens) {
        token_ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(f, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    const char * merges[] = {"a b"};
    gguf_set_arr_str(f, "tokenizer.ggml.merges", merges, 1);
}

inline void add_backbone_tensors(GgufBuilder & b, int n_embd = N_EMBD) {
    b.add2("token_embd.weight", n_embd, TEXT_VOCAB);
    b.add1("output_norm.weight", n_embd);
    b.add2("output.weight", n_embd, TEXT_VOCAB);
    for (int i = 0; i < N_VQ; ++i) {
        b.add2("token_embd_audio." + std::to_string(i) + ".weight", n_embd, AUDIO_HEAD);
        b.add2("output_audio." + std::to_string(i) + ".weight", n_embd, AUDIO_HEAD);
    }
    b.add1("blk.0.attn_norm.weight", n_embd);
    b.add2("blk.0.attn_q.weight", n_embd, (int64_t) N_HEADS * HEAD_DIM);
    b.add2("blk.0.attn_k.weight", n_embd, (int64_t) N_KV_HEADS * HEAD_DIM);
    b.add2("blk.0.attn_v.weight", n_embd, (int64_t) N_KV_HEADS * HEAD_DIM);
    b.add2("blk.0.attn_output.weight", (int64_t) N_HEADS * HEAD_DIM, n_embd);
    b.add1("blk.0.attn_q_norm.weight", HEAD_DIM);
    b.add1("blk.0.attn_k_norm.weight", HEAD_DIM);
    b.add1("blk.0.ffn_norm.weight", n_embd);
    b.add2("blk.0.ffn_gate.weight", n_embd, N_FF);
    b.add2("blk.0.ffn_up.weight", n_embd, N_FF);
    b.add2("blk.0.ffn_down.weight", N_FF, n_embd);
}

inline std::filesystem::path write_backbone(const char * tag,
        const std::function<void(gguf_context *)> & mutate_meta,
        const ConstantTensors & constants = {}, const TensorShapes & shapes = {}) {
    GgufBuilder b;
    b.constants = constants;
    b.shapes = shapes;
    add_backbone_meta(b.file);
    add_backbone_tensors(b);
    mutate_meta(b.file);
    const auto path = temp_gguf(tag);
    b.write(path);
    return path;
}

inline void add_decoder_meta(gguf_context * f, int num_quantizers) {
    gguf_set_val_str(f, "general.architecture", "moss-tts-audio-decoder");
    gguf_set_val_u32(f, "moss-tts-audio-decoder.sampling_rate", 24000);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.input_dim", RVQ_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.rvq_dim", RVQ_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.output_dim", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.num_quantizers", (uint32_t) num_quantizers);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.codebook_size", CODE_SIZE);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.codebook_dim", CODE_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.block_count", 2);
    gguf_set_val_str(f, "moss-tts-audio-decoder.decoder.0.module_type", "Transformer");
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.input_dimension", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.output_dimension", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.d_model", D_MODEL);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.num_heads", 2);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.num_layers", 1);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.context", 8);
    gguf_set_val_str(f, "moss-tts-audio-decoder.decoder.1.module_type", "PatchedPretransform");
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.1.patch_size", OUT_DIM);
}

inline void add_decoder_tensors(GgufBuilder & b, int num_quantizers, int64_t qkv_rows) {
    b.add2("quantizer.output_proj.weight", RVQ_DIM, OUT_DIM);
    for (int iq = 0; iq < num_quantizers; ++iq) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(iq) + ".";
        b.add2(prefix + "codebook.weight", CODE_DIM, CODE_SIZE);
        b.add2(prefix + "out_proj.weight", CODE_DIM, RVQ_DIM);
    }
    b.add2("blk.0.input_proj.weight", OUT_DIM, D_MODEL);
    b.add2("blk.0.output_proj.weight", D_MODEL, OUT_DIM);
    b.add2("blk.0.layer.0.attn_qkv.weight", D_MODEL, qkv_rows);
    b.add2("blk.0.layer.0.attn_output.weight", D_MODEL, D_MODEL);
    b.add2("blk.0.layer.0.ffn_up.weight", D_MODEL, 2 * D_MODEL);
    b.add2("blk.0.layer.0.ffn_down.weight", 2 * D_MODEL, D_MODEL);
    b.add1("blk.0.layer.0.attn_norm.weight", D_MODEL);
    b.add1("blk.0.layer.0.attn_norm.bias", D_MODEL);
    b.add1("blk.0.layer.0.ffn_norm.weight", D_MODEL);
    b.add1("blk.0.layer.0.ffn_norm.bias", D_MODEL);
}

inline std::filesystem::path write_decoder(const char * tag, int num_quantizers, int64_t qkv_rows,
        const std::function<void(gguf_context *)> & mutate_meta, std::mt19937 * random = nullptr) {
    GgufBuilder b;
    b.random = random;
    add_decoder_meta(b.file, num_quantizers);
    mutate_meta(b.file);
    add_decoder_tensors(b, num_quantizers, qkv_rows);
    const auto path = temp_gguf(tag);
    b.write(path);
    return path;
}

inline void add_encoder_meta(gguf_context * f) {
    gguf_set_val_str(f, "general.architecture", "moss-tts-audio-encoder");
    gguf_set_val_u32(f, "moss-tts-audio-encoder.sampling_rate", 24000);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.downsample_rate", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.quantizer.input_dim", RVQ_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.quantizer.rvq_dim", RVQ_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.quantizer.output_dim", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.quantizer.num_quantizers", N_VQ);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.quantizer.codebook_size", CODE_SIZE);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.quantizer.codebook_dim", CODE_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.block_count", 2);
    gguf_set_val_str(f, "moss-tts-audio-encoder.encoder.0.module_type", "PatchedPretransform");
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.0.patch_size", OUT_DIM);
    gguf_set_val_str(f, "moss-tts-audio-encoder.encoder.1.module_type", "Transformer");
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.1.input_dimension", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.1.output_dimension", RVQ_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.1.d_model", D_MODEL);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.1.num_heads", 2);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.1.num_layers", 1);
    gguf_set_val_u32(f, "moss-tts-audio-encoder.encoder.1.context", 8);
}

inline void add_encoder_tensors(GgufBuilder & b) {
    b.add2("quantizer.input_proj.weight", RVQ_DIM, RVQ_DIM);
    for (int iq = 0; iq < N_VQ; ++iq) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(iq) + ".";
        b.add2(prefix + "codebook.weight", CODE_DIM, CODE_SIZE);
        b.add2(prefix + "in_proj.weight", RVQ_DIM, CODE_DIM);
        b.add2(prefix + "out_proj.weight", CODE_DIM, RVQ_DIM);
    }
    b.add2("blk.0.input_proj.weight", OUT_DIM, D_MODEL);
    b.add2("blk.0.output_proj.weight", D_MODEL, RVQ_DIM);
    b.add2("blk.0.layer.0.attn_qkv.weight", D_MODEL, 3 * D_MODEL);
    b.add2("blk.0.layer.0.attn_output.weight", D_MODEL, D_MODEL);
    b.add2("blk.0.layer.0.ffn_up.weight", D_MODEL, 2 * D_MODEL);
    b.add2("blk.0.layer.0.ffn_down.weight", 2 * D_MODEL, D_MODEL);
    b.add1("blk.0.layer.0.attn_norm.weight", D_MODEL);
    b.add1("blk.0.layer.0.attn_norm.bias", D_MODEL);
    b.add1("blk.0.layer.0.ffn_norm.weight", D_MODEL);
    b.add1("blk.0.layer.0.ffn_norm.bias", D_MODEL);
}

inline std::filesystem::path write_encoder(const char * tag,
        const std::function<void(gguf_context *)> & mutate_meta = [](gguf_context *) {}) {
    GgufBuilder b;
    add_encoder_meta(b.file);
    mutate_meta(b.file);
    add_encoder_tensors(b);
    const auto path = temp_gguf(tag);
    b.write(path);
    return path;
}

inline std::filesystem::path write_tiny_wav(const char * tag, int samples) {
    const auto path = temp_gguf(tag).replace_extension(".wav");
    const uint32_t rate = 24000;
    const uint32_t data_bytes = (uint32_t) samples * 2;
    std::vector<unsigned char> header = {
        'R','I','F','F', 0,0,0,0, 'W','A','V','E',
        'f','m','t',' ', 16,0,0,0, 1,0, 1,0,
        0,0,0,0, 0,0,0,0, 2,0, 16,0,
        'd','a','t','a', 0,0,0,0,
    };
    auto put32 = [&header](size_t at, uint32_t v) {
        header[at] = (unsigned char) (v & 0xff);
        header[at + 1] = (unsigned char) ((v >> 8) & 0xff);
        header[at + 2] = (unsigned char) ((v >> 16) & 0xff);
        header[at + 3] = (unsigned char) ((v >> 24) & 0xff);
    };
    put32(4, 36 + data_bytes);
    put32(24, rate);
    put32(28, rate * 2);
    put32(40, data_bytes);
    std::ofstream out(path, std::ios::binary);
    out.write((const char *) header.data(), (std::streamsize) header.size());
    for (int i = 0; i < samples; ++i) {
        const int16_t sample = (int16_t) ((i % 32) * 512 - 8192);
        out.write((const char *) &sample, 2);
    }
    return path;
}

} // namespace moss_fixtures
