#pragma once

#include "backend.h"
#include "ane-validation.h"
#include "coreml-conditioner.h"
#include "gguf-weights.h"
#include "logic.h"
#include "weight-ctx.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

struct MM3LmConfig {

    uint32_t block_count          = 0;
    uint32_t context_length       = 0;
    uint32_t embedding_length     = 0;
    uint32_t feed_forward_length  = 0;
    uint32_t head_count           = 0;
    uint32_t head_count_kv        = 0;
    uint32_t key_length           = 0;
    uint32_t value_length         = 0;
    float    rms_eps              = 0.0f;
    float    rope_freq_base       = 0.0f;
    uint32_t vocab_size           = 0;

    uint32_t semantic_vocab_offset = 0;
    uint32_t semantic_vocab_size   = 0;
    uint32_t acoustic_vocab_size   = 0;
    uint32_t num_codebooks         = 0;
    uint32_t eos_audio             = 0;
    uint32_t frame_rate            = 0;
    uint32_t max_audio_frames      = 0;
    uint32_t max_prompt_tokens     = 0;
    float    ar_cfg_scale          = 0.0f;
    uint32_t ar_top_k              = 0;
    float    ar_embedding_scale    = 0.0f;

    uint32_t tok_im_start      = 0;
    uint32_t tok_im_end        = 0;
    uint32_t tok_audio_cfg     = 0;
    uint32_t tok_audio_start   = 0;
    uint32_t tok_audio_end     = 0;
    uint32_t tok_caption_start = 0;
    uint32_t tok_caption_end   = 0;
    uint32_t tok_lyrics_start  = 0;
    uint32_t tok_lyrics_end    = 0;
};

static bool mm3_lm_positions_fit(int64_t context_length, int64_t prompt_positions, int64_t generated_positions) {
    if (context_length <= 0 || prompt_positions < 0 || generated_positions < 0) {
        return false;
    }
    if (prompt_positions > context_length) {
        return false;
    }
    return generated_positions <= context_length - prompt_positions;
}

struct MM3DepthConfig {
    uint32_t block_count         = 0;
    uint32_t embedding_length    = 0;
    uint32_t feed_forward_length = 0;
    uint32_t head_count          = 0;
    uint32_t head_dim            = 0;
    uint32_t max_position        = 0;
    float    rms_eps             = 0.0f;
    uint32_t num_codebooks       = 0;
    uint32_t audio_vocab_size    = 0;
    uint32_t audio_embd_rows     = 0;
    bool     causal              = false;
    bool     rope                = false;
};

struct MM3CondConfig {
    uint32_t    num_layers           = 0;
    uint32_t    hidden_dim           = 0;
    uint32_t    out_dim              = 0;
    uint32_t    kernel_size          = 0;
    uint32_t    padding              = 0;
    uint32_t    input_sampling_rate  = 0;
    uint32_t    input_hop_length     = 0;
    uint32_t    output_sampling_rate = 0;
    uint32_t    output_hop_length    = 0;
    std::string interpolation;
    std::string layer_mix;
};

struct MM3DitConfig {
    uint32_t block_count      = 0;
    uint32_t embedding_length = 0;
    uint32_t head_count       = 0;
    uint32_t head_dim         = 0;
    uint32_t ff_inner         = 0;
    uint32_t in_channels      = 0;
    uint32_t condition_dim    = 0;
    uint32_t concat_channels  = 0;
    float    layer_norm_eps   = 0.0f;
    uint32_t rope_dim         = 0;
    float    rope_theta       = 0.0f;
    std::string rope_type;
    uint32_t fourier_dim      = 0;
    std::string glu_order;
    bool     timestep_token_prepended = false;
    bool     pre_post_conv_residual   = false;
    bool     attn_bias                = false;
    uint32_t window_frames  = 0;
    uint32_t hop_frames     = 0;
    uint32_t window_latents = 0;
    uint32_t hop_latents    = 0;
};

struct MM3FlowConfig {
    std::string scheduler;
    uint32_t    steps               = 0;
    float       cfg_scale           = 0.0f;
    bool        invert_sigmas       = false;
    float       shift               = 0.0f;
    uint32_t    num_train_timesteps = 0;
};

struct MM3VocConfig {
    uint32_t              latent_channels   = 0;
    uint32_t              fold_channels     = 0;
    uint32_t              dec_in_dim        = 0;
    uint32_t              hidden_dim        = 0;
    std::vector<int32_t>  upsample_rates;
    std::vector<int32_t>  res_dilations;
    uint32_t              total_upsample    = 0;
    uint32_t              sampling_rate     = 0;
    uint32_t              channels          = 0;
    float                 snake_eps         = 0.0f;
    bool                  final_tanh        = false;
    bool                  weight_norm_folded = false;
    std::string           snake;
};

struct MM3SynthConfig {
    std::vector<std::string> components;
    MM3DepthConfig depth;
    MM3CondConfig  cond;
    MM3DitConfig   dit;
    MM3FlowConfig  flow;
    MM3VocConfig   voc;
};

// attn_qkv and ffn_gate_up hold the [q|k|v] and [gate|up] rows stacked into
// one matrix when the parts share a type; the separate tensors stay null then.
struct MM3LmLayer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_qkv    = nullptr;
    ggml_tensor * attn_q      = nullptr;
    ggml_tensor * attn_k      = nullptr;
    ggml_tensor * attn_v      = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate_up = nullptr;
    ggml_tensor * ffn_gate    = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
};

struct MM3LmWeights {
    ggml_tensor *           token_embd     = nullptr;
    ggml_tensor *           output_norm    = nullptr;
    ggml_tensor *           output_compact = nullptr;
    std::vector<MM3LmLayer> blk;
};

// Stacked the same way as MM3LmLayer.
struct MM3DepthLayer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_qkv    = nullptr;
    ggml_tensor * attn_q      = nullptr;
    ggml_tensor * attn_k      = nullptr;
    ggml_tensor * attn_v      = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate_up = nullptr;
    ggml_tensor * ffn_gate    = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
};

struct MM3DepthWeights {
    ggml_tensor *              proj        = nullptr;
    ggml_tensor *              pos_embd    = nullptr;
    ggml_tensor *              audio_embd  = nullptr;
    ggml_tensor *              output_norm = nullptr;
    std::vector<ggml_tensor *> head;
    std::vector<MM3DepthLayer> blk;
};

struct MM3CondWeights {
    ggml_tensor * layer_logits = nullptr;
    ggml_tensor * layer_scale  = nullptr;
    ggml_tensor * proj_w       = nullptr;
    ggml_tensor * proj_b       = nullptr;
};

struct MM3DitBlock {
    ggml_tensor * attn_norm_w   = nullptr;
    ggml_tensor * attn_norm_b   = nullptr;
    ggml_tensor * attn_qkv      = nullptr;
    ggml_tensor * attn_output   = nullptr;
    ggml_tensor * ffn_norm_w    = nullptr;
    ggml_tensor * ffn_norm_b    = nullptr;
    ggml_tensor * ffn_in_w      = nullptr;
    ggml_tensor * ffn_in_b      = nullptr;
    ggml_tensor * ffn_out_w     = nullptr;
    ggml_tensor * ffn_out_b     = nullptr;
};

struct MM3DitWeights {
    ggml_tensor *             preprocess_conv  = nullptr;
    ggml_tensor *             postprocess_conv = nullptr;
    ggml_tensor *             time_fourier     = nullptr;
    ggml_tensor *             time_embd_w[2]   = { nullptr, nullptr };
    ggml_tensor *             time_embd_b[2]   = { nullptr, nullptr };
    ggml_tensor *             proj_in          = nullptr;
    ggml_tensor *             proj_out         = nullptr;
    ggml_tensor *             rope_inv_freq    = nullptr;
    std::vector<MM3DitBlock>  blk;
};

struct MM3VocResUnit {
    ggml_tensor * snake1_alpha = nullptr;
    ggml_tensor * conv1_w      = nullptr;
    ggml_tensor * conv1_b      = nullptr;
    ggml_tensor * snake2_alpha = nullptr;
    ggml_tensor * conv2_w      = nullptr;
    ggml_tensor * conv2_b      = nullptr;
};

struct MM3VocBlock {
    ggml_tensor *              snake_alpha = nullptr;
    ggml_tensor *              convt_w     = nullptr;
    ggml_tensor *              convt_b     = nullptr;
    std::vector<MM3VocResUnit> res;
};

struct MM3VocWeights {
    ggml_tensor *             dec_in_w    = nullptr;
    ggml_tensor *             dec_in_b    = nullptr;
    ggml_tensor *             conv_in_w   = nullptr;
    ggml_tensor *             conv_in_b   = nullptr;
    std::vector<MM3VocBlock>  blk;
    ggml_tensor *             snake_out_alpha = nullptr;
    ggml_tensor *             conv_out_w      = nullptr;
    ggml_tensor *             conv_out_b      = nullptr;
};

struct MM3SynthWeights {
    MM3DepthWeights depth;
    MM3CondWeights  cond;
    MM3DitWeights   dit;
    MM3VocWeights   voc;
};

struct MM3FileInfo {
    bool        found = false;
    std::string path;
    std::string name;
    std::string arch;
    std::string general_name;
    std::string license;
    std::string source_layout;
    uint32_t    converter_version = 0;
    uint32_t    file_type         = 0;
    uint64_t    file_bytes        = 0;
    uint64_t    tensor_bytes      = 0;
    int         n_tensors         = 0;
    bool        probe_ok          = false;
    std::string probe_error;
};

struct MM3Model {

    std::string              models_dir;
    std::vector<std::string> search_dirs;
    MM3FileInfo              lm_file;
    MM3FileInfo              synth_file;
    MM3LmConfig              lm_cfg;
    MM3SynthConfig           synth_cfg;
    std::vector<std::string> meta_errors;

    std::string coreml_conditioner_path;
    MM3CoreMLConditioner * coreml_conditioner = nullptr;
#ifdef MM3_ANE_VALIDATION
    std::string ane_validation_directory;
    std::string ane_validation_compute_units = "CPU_AND_NE";
    std::shared_ptr<mm3_ane_validation::Runtime> ane_validation;
#endif

    bool           loaded      = false;
    bool           backend_ref = false;
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    WeightCtx      wctx_lm     = {};
    WeightCtx      wctx_synth  = {};
    size_t         vram_lm     = 0;
    size_t         vram_synth  = 0;
    double         load_ms     = 0.0;

    MM3LmWeights    lm;
    MM3SynthWeights synth;

    std::map<std::string, ggml_tensor *> tmap_lm;
    std::map<std::string, ggml_tensor *> tmap_synth;
};

#ifdef _WIN32
#    define MM3_SEP "\\"
#else
#    define MM3_SEP "/"
#endif

static bool mm3_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

static std::string mm3_basename(const std::string & path) {
    size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? path : path.substr(p + 1);
}

static std::vector<int32_t> mm3_get_i32_arr(const GGUFModel & gf, const char * key) {
    std::vector<int32_t> out;
    int64_t              idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_ARRAY) {
        return out;
    }
    enum gguf_type et = gguf_get_arr_type(gf.gguf, idx);
    if (et != GGUF_TYPE_INT32 && et != GGUF_TYPE_UINT32) {
        return out;
    }
    size_t          n = gguf_get_arr_n(gf.gguf, idx);
    const int32_t * d = (const int32_t *) gguf_get_arr_data(gf.gguf, idx);
    out.assign(d, d + n);
    return out;
}

static std::vector<std::string> mm3_get_str_arr(const GGUFModel & gf, const char * key) {
    std::vector<std::string> out;
    int64_t                  idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_ARRAY) {
        return out;
    }
    if (gguf_get_arr_type(gf.gguf, idx) != GGUF_TYPE_STRING) {
        return out;
    }
    size_t n = gguf_get_arr_n(gf.gguf, idx);
    for (size_t i = 0; i < n; i++) {
        out.push_back(gguf_get_arr_str(gf.gguf, idx, i));
    }
    return out;
}

static int64_t mm3_total_rows(const std::vector<int64_t> & rows) {
    int64_t total = 0;
    for (int64_t count : rows) {
        total += count;
    }
    return total;
}

struct MM3Loader {
    WeightCtx *                            wctx   = nullptr;
    const GGUFModel *                      gf     = nullptr;
    std::map<std::string, ggml_tensor *> * tmap   = nullptr;
    std::vector<std::string> *             errors = nullptr;

    void fail(const std::string & msg) const {
        if (errors && errors->size() < 24) {
            errors->push_back(msg);
        }
    }

    ggml_tensor * req(const std::string & name, int64_t e0, int64_t e1 = 1, int64_t e2 = 1, int64_t e3 = 1) {
        if (gguf_find_tensor(gf->gguf, name.c_str()) < 0) {
            fail("missing tensor '" + name + "'");
            return nullptr;
        }
        ggml_tensor * src = ggml_get_tensor(gf->meta, name.c_str());
        if (!src) {
            fail("tensor '" + name + "' not in meta context");
            return nullptr;
        }
        const int64_t want[4] = { e0, e1, e2, e3 };
        for (int i = 0; i < 4; i++) {
            if (src->ne[i] != want[i]) {
                char buf[320];
                snprintf(buf, sizeof(buf),
                         "tensor '%s' shape mismatch: file [%lld,%lld,%lld,%lld] expected [%lld,%lld,%lld,%lld]",
                         name.c_str(), (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2],
                         (long long) src->ne[3], (long long) want[0], (long long) want[1], (long long) want[2],
                         (long long) want[3]);
                fail(buf);
                return nullptr;
            }
        }
        ggml_tensor * t = gf_load_tensor(wctx, *gf, name);
        if (t && tmap) {
            (*tmap)[name] = t;
        }
        return t;
    }

    // The file's [e0, e1] matrix of that name, or nullptr when it is absent or
    // has another shape.
    const ggml_tensor * file_matrix(const std::string & name, int64_t e0, int64_t e1) const {
        if (gguf_find_tensor(gf->gguf, name.c_str()) < 0) {
            return nullptr;
        }
        const ggml_tensor * src = ggml_get_tensor(gf->meta, name.c_str());
        const bool shape_ok = src && src->ne[0] == e0 && src->ne[1] == e1 && src->ne[2] == 1 && src->ne[3] == 1;
        return shape_ok ? src : nullptr;
    }

    // The type every part shares when each is an [e0, rows[i]] matrix in the
    // file, else GGML_TYPE_COUNT.
    ggml_type stacked_type(const std::vector<std::string> & parts, int64_t e0,
                           const std::vector<int64_t> & rows) const {
        ggml_type type = GGML_TYPE_COUNT;
        for (size_t i = 0; i < parts.size(); i++) {
            const ggml_tensor * src = file_matrix(parts[i], e0, rows[i]);
            if (!src || (i > 0 && src->type != type)) {
                return GGML_TYPE_COUNT;
            }
            type = src->type;
        }
        return type;
    }

    void queue_stacked_copies(ggml_tensor * dst, const std::vector<std::string> & parts) {
        size_t offset = 0;
        for (const std::string & part : parts) {
            const size_t nbytes = ggml_nbytes(ggml_get_tensor(gf->meta, part.c_str()));
            wctx_queue_copy(wctx, dst, gf_get_data(*gf, part.c_str()), nbytes, offset);
            offset += nbytes;
        }
    }

    // Loads same-width matrices as one tensor stacked along ne[1] in the given
    // order, so projections of one input run as a single matrix product.
    // Returns nullptr without an error when a part is missing, misshapen or of
    // another type (a mixed k-quant layer); the caller then loads the parts one
    // by one, which reports any real problem.
    ggml_tensor * try_stacked(const std::string & name, const std::vector<std::string> & parts, int64_t e0,
                              const std::vector<int64_t> & rows) {
        if (parts.empty() || parts.size() != rows.size()) {
            return nullptr;
        }
        const ggml_type type = stacked_type(parts, e0, rows);
        if (type == GGML_TYPE_COUNT) {
            return nullptr;
        }
        ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, type, e0, mm3_total_rows(rows));
        ggml_set_name(dst, name.c_str());
        queue_stacked_copies(dst, parts);
        if (tmap) {
            (*tmap)[name] = dst;
        }
        return dst;
    }

    // Loads only the semantic-block + EOS rows of an output-head tensor, skipping the
    // rest of the vocabulary (decode only ever samples those rows; see collect_ar_candidates).
    ggml_tensor * req_compact_head(const std::string & name, int64_t h, int64_t vocab, int64_t semantic_offset,
                                   int64_t semantic_vocab, int64_t eos) {
        if (gguf_find_tensor(gf->gguf, name.c_str()) < 0) {
            fail("missing tensor '" + name + "'");
            return nullptr;
        }
        ggml_tensor * src = ggml_get_tensor(gf->meta, name.c_str());
        if (!src) {
            fail("tensor '" + name + "' not in meta context");
            return nullptr;
        }
        if (src->ne[0] != h || src->ne[1] != vocab || src->ne[2] != 1 || src->ne[3] != 1) {
            char buf[320];
            snprintf(buf, sizeof(buf),
                     "tensor '%s' shape mismatch: file [%lld,%lld,%lld,%lld] expected [%lld,%lld,1,1]", name.c_str(),
                     (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3],
                     (long long) h, (long long) vocab);
            fail(buf);
            return nullptr;
        }

        const size_t                                             row_size = ggml_row_size(src->type, h);
        std::array<tts_cpp::minimax::detail::CompactHeadCopy, 2> plan{};
        if (!tts_cpp::minimax::detail::compact_head_copy_plan(vocab, semantic_offset, semantic_vocab, eos, row_size,
                                                              plan)) {
            fail("compact head bounds for '" + name + "' do not fit the vocabulary");
            return nullptr;
        }

        const int64_t compact_rows = tts_cpp::minimax::detail::compact_head_row_count(semantic_vocab);
        ggml_tensor * dst          = ggml_new_tensor_2d(wctx->ctx, src->type, h, compact_rows);
        ggml_set_name(dst, "output_compact.weight");

        queue_compact_head_copies(dst, (const uint8_t *) gf_get_data(*gf, name.c_str()), plan);
        if (tmap) {
            (*tmap)["output_compact.weight"] = dst;
        }
        return dst;
    }

    void queue_compact_head_copies(ggml_tensor * dst, const uint8_t * base,
                                   const std::array<tts_cpp::minimax::detail::CompactHeadCopy, 2> & plan) {
        if (!base) {
            return;
        }
        for (const tts_cpp::minimax::detail::CompactHeadCopy & copy : plan) {
            wctx_queue_copy(wctx, dst, base + copy.src_offset, copy.nbytes, copy.dst_offset);
        }
    }
};

static std::string mm3_fmt(const char * fmt, int a) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, a);
    return buf;
}

static std::string mm3_fmt2(const char * fmt, int a, int b) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, a, b);
    return buf;
}

static void mm3_parse_lm_config(const GGUFModel & gf, MM3LmConfig * c) {
    c->block_count         = gf_get_u32(gf, "qwen3.block_count");
    c->context_length      = gf_get_u32(gf, "qwen3.context_length");
    c->embedding_length    = gf_get_u32(gf, "qwen3.embedding_length");
    c->feed_forward_length = gf_get_u32(gf, "qwen3.feed_forward_length");
    c->head_count          = gf_get_u32(gf, "qwen3.attention.head_count");
    c->head_count_kv       = gf_get_u32(gf, "qwen3.attention.head_count_kv");
    c->key_length          = gf_get_u32(gf, "qwen3.attention.key_length");
    c->value_length        = gf_get_u32(gf, "qwen3.attention.value_length");
    c->rms_eps             = gf_get_f32(gf, "qwen3.attention.layer_norm_rms_epsilon");
    c->rope_freq_base      = gf_get_f32(gf, "qwen3.rope.freq_base");
    c->vocab_size          = gf_get_u32(gf, "qwen3.vocab_size");

    c->semantic_vocab_offset = gf_get_u32(gf, "mm3.semantic_vocab_offset");
    c->semantic_vocab_size   = gf_get_u32(gf, "mm3.semantic_vocab_size");
    c->acoustic_vocab_size   = gf_get_u32(gf, "mm3.acoustic_vocab_size");
    c->num_codebooks         = gf_get_u32(gf, "mm3.num_codebooks");
    c->eos_audio             = gf_get_u32(gf, "mm3.eos_audio");
    c->frame_rate            = gf_get_u32(gf, "mm3.frame_rate");
    c->max_audio_frames      = gf_get_u32(gf, "mm3.max_audio_frames");
    c->max_prompt_tokens     = gf_get_u32(gf, "mm3.max_prompt_tokens");
    c->ar_cfg_scale          = gf_get_f32(gf, "mm3.ar.cfg_scale");
    c->ar_top_k              = gf_get_u32(gf, "mm3.ar.top_k");
    c->ar_embedding_scale    = gf_get_f32(gf, "mm3.ar.embedding_scale");

    c->tok_im_start      = gf_get_u32(gf, "mm3.token.im_start");
    c->tok_im_end        = gf_get_u32(gf, "mm3.token.im_end");
    c->tok_audio_cfg     = gf_get_u32(gf, "mm3.token.audio_cfg");
    c->tok_audio_start   = gf_get_u32(gf, "mm3.token.audio_start");
    c->tok_audio_end     = gf_get_u32(gf, "mm3.token.audio_end");
    c->tok_caption_start = gf_get_u32(gf, "mm3.token.caption_start");
    c->tok_caption_end   = gf_get_u32(gf, "mm3.token.caption_end");
    c->tok_lyrics_start  = gf_get_u32(gf, "mm3.token.lyrics_start");
    c->tok_lyrics_end    = gf_get_u32(gf, "mm3.token.lyrics_end");
}

static void mm3_parse_synth_config(const GGUFModel & gf, MM3SynthConfig * c) {
    c->components = mm3_get_str_arr(gf, "mm3.synth.components");

    MM3DepthConfig & d     = c->depth;
    d.block_count          = gf_get_u32(gf, "mm3.depth.block_count");
    d.embedding_length     = gf_get_u32(gf, "mm3.depth.embedding_length");
    d.feed_forward_length  = gf_get_u32(gf, "mm3.depth.feed_forward_length");
    d.head_count           = gf_get_u32(gf, "mm3.depth.head_count");
    d.head_dim             = gf_get_u32(gf, "mm3.depth.head_dim");
    d.max_position         = gf_get_u32(gf, "mm3.depth.max_position");
    d.rms_eps              = gf_get_f32(gf, "mm3.depth.rms_eps");
    d.num_codebooks        = gf_get_u32(gf, "mm3.depth.num_codebooks");
    d.audio_vocab_size     = gf_get_u32(gf, "mm3.depth.audio_vocab_size");
    d.audio_embd_rows      = gf_get_u32(gf, "mm3.depth.audio_embd_rows");
    d.causal               = gf_get_bool(gf, "mm3.depth.causal");
    d.rope                 = gf_get_bool(gf, "mm3.depth.rope");

    MM3CondConfig & e        = c->cond;
    e.num_layers             = gf_get_u32(gf, "mm3.cond.num_layers");
    e.hidden_dim             = gf_get_u32(gf, "mm3.cond.hidden_dim");
    e.out_dim                = gf_get_u32(gf, "mm3.cond.out_dim");
    e.kernel_size            = gf_get_u32(gf, "mm3.cond.kernel_size");
    e.padding                = gf_get_u32(gf, "mm3.cond.padding");
    e.input_sampling_rate    = gf_get_u32(gf, "mm3.cond.input_sampling_rate");
    e.input_hop_length       = gf_get_u32(gf, "mm3.cond.input_hop_length");
    e.output_sampling_rate   = gf_get_u32(gf, "mm3.cond.output_sampling_rate");
    e.output_hop_length      = gf_get_u32(gf, "mm3.cond.output_hop_length");
    e.interpolation          = gf_get_str(gf, "mm3.cond.interpolation");
    e.layer_mix              = gf_get_str(gf, "mm3.cond.layer_mix");

    MM3DitConfig & t             = c->dit;
    t.block_count                = gf_get_u32(gf, "mm3.dit.block_count");
    t.embedding_length           = gf_get_u32(gf, "mm3.dit.embedding_length");
    t.head_count                 = gf_get_u32(gf, "mm3.dit.head_count");
    t.head_dim                   = gf_get_u32(gf, "mm3.dit.head_dim");
    t.ff_inner                   = gf_get_u32(gf, "mm3.dit.ff_inner");
    t.in_channels                = gf_get_u32(gf, "mm3.dit.in_channels");
    t.condition_dim              = gf_get_u32(gf, "mm3.dit.condition_dim");
    t.concat_channels            = gf_get_u32(gf, "mm3.dit.concat_channels");
    t.layer_norm_eps             = gf_get_f32(gf, "mm3.dit.layer_norm_eps");
    t.rope_dim                   = gf_get_u32(gf, "mm3.dit.rope_dim");
    t.rope_theta                 = gf_get_f32(gf, "mm3.dit.rope_theta");
    t.rope_type                  = gf_get_str(gf, "mm3.dit.rope_type");
    t.fourier_dim                = gf_get_u32(gf, "mm3.dit.fourier_dim");
    t.glu_order                  = gf_get_str(gf, "mm3.dit.glu_order");
    t.timestep_token_prepended   = gf_get_bool(gf, "mm3.dit.timestep_token_prepended");
    t.pre_post_conv_residual     = gf_get_bool(gf, "mm3.dit.pre_post_conv_residual");
    t.attn_bias                  = gf_get_bool(gf, "mm3.dit.attn_bias");
    t.window_frames              = gf_get_u32(gf, "mm3.dit.window_frames");
    t.hop_frames                 = gf_get_u32(gf, "mm3.dit.hop_frames");
    t.window_latents             = gf_get_u32(gf, "mm3.dit.window_latents");
    t.hop_latents                = gf_get_u32(gf, "mm3.dit.hop_latents");

    MM3FlowConfig & f      = c->flow;
    f.scheduler            = gf_get_str(gf, "mm3.flow.scheduler");
    f.steps                = gf_get_u32(gf, "mm3.flow.steps");
    f.cfg_scale            = gf_get_f32(gf, "mm3.flow.cfg_scale");
    f.invert_sigmas        = gf_get_bool(gf, "mm3.flow.invert_sigmas");
    f.shift                = gf_get_f32(gf, "mm3.flow.shift");
    f.num_train_timesteps  = gf_get_u32(gf, "mm3.flow.num_train_timesteps");

    MM3VocConfig & v      = c->voc;
    v.latent_channels     = gf_get_u32(gf, "mm3.voc.latent_channels");
    v.fold_channels       = gf_get_u32(gf, "mm3.voc.fold_channels");
    v.dec_in_dim          = gf_get_u32(gf, "mm3.voc.dec_in_dim");
    v.hidden_dim          = gf_get_u32(gf, "mm3.voc.hidden_dim");
    v.upsample_rates      = mm3_get_i32_arr(gf, "mm3.voc.upsample_rates");
    v.res_dilations       = mm3_get_i32_arr(gf, "mm3.voc.res_dilations");
    v.total_upsample      = gf_get_u32(gf, "mm3.voc.total_upsample");
    v.sampling_rate       = gf_get_u32(gf, "mm3.voc.sampling_rate");
    v.channels            = gf_get_u32(gf, "mm3.voc.channels");
    v.snake_eps           = gf_get_f32(gf, "mm3.voc.snake_eps");
    v.final_tanh          = gf_get_bool(gf, "mm3.voc.final_tanh");
    v.weight_norm_folded  = gf_get_bool(gf, "mm3.voc.weight_norm_folded");
    v.snake               = gf_get_str(gf, "mm3.voc.snake");
}

static void mm3_validate_lm_config(const MM3LmConfig & c, std::vector<std::string> * errs) {
    auto need = [&](bool ok, const char * what) {
        if (!ok && errs->size() < 24) {
            errs->push_back(std::string("LM config: ") + what);
        }
    };
    need(c.block_count > 0, "qwen3.block_count is 0 or missing");
    need(c.context_length > 0, "qwen3.context_length is 0 or missing");
    need(c.embedding_length > 0, "qwen3.embedding_length is 0 or missing");
    need(c.vocab_size > 0, "qwen3.vocab_size is 0 or missing");
    need(c.head_count > 0 && c.head_count_kv > 0, "head counts are 0 or missing");
    need(c.key_length > 0 && c.value_length > 0, "key/value length is 0 or missing");
    need(c.num_codebooks > 0, "mm3.num_codebooks is 0 or missing (not an MM3 LM?)");
    need(c.semantic_vocab_offset > 0, "mm3.semantic_vocab_offset is 0 or missing");
    need(c.vocab_size >= c.semantic_vocab_offset + c.semantic_vocab_size,
         "vocab_size does not cover semantic_vocab_offset + semantic_vocab_size");
    need(c.eos_audio > 0, "mm3.eos_audio is 0 or missing");
    need(c.frame_rate > 0, "mm3.frame_rate is 0 or missing");
    need(c.max_audio_frames > 0, "mm3.max_audio_frames is 0 or missing");
    need(c.max_prompt_tokens > 0, "mm3.max_prompt_tokens is 0 or missing");
    need(c.max_prompt_tokens <= c.context_length,
         "mm3.max_prompt_tokens exceeds qwen3.context_length");
    need(c.max_audio_frames <= c.context_length,
         "mm3.max_audio_frames exceeds qwen3.context_length");
    need(c.head_count * c.key_length == c.embedding_length,
         "head_count * key_length != embedding_length");
}

static void mm3_append_synth_contract_errors(const std::vector<std::string> & source,
                                             std::vector<std::string> * errors) {
    for (const std::string & error : source) {
        if (errors->size() >= 24) {
            return;
        }
        errors->push_back("synth config: " + error);
    }
}

static void mm3_validate_synth_config(const MM3SynthConfig & c, std::vector<std::string> * errs) {
    auto need = [&](bool ok, const char * what) {
        if (!ok && errs->size() < 24) {
            errs->push_back(std::string("synth config: ") + what);
        }
    };
    need(c.depth.block_count > 0, "mm3.depth.block_count is 0 or missing");
    need(c.depth.embedding_length > 0, "mm3.depth.embedding_length is 0 or missing");
    need(c.depth.num_codebooks > 1, "mm3.depth.num_codebooks must be > 1");
    need(c.depth.head_count * c.depth.head_dim == c.depth.embedding_length,
         "depth head_count * head_dim != embedding_length");
    need(c.depth.audio_embd_rows == (c.depth.num_codebooks - 1) * c.depth.audio_vocab_size,
         "depth audio_embd_rows != (num_codebooks-1) * audio_vocab_size");
    need(c.cond.num_layers > 0 && c.cond.hidden_dim > 0, "cond dims are 0 or missing");
    need(c.cond.input_sampling_rate > 0 && c.cond.input_hop_length > 0,
         "cond input rate metadata is 0 or missing");
    need(c.cond.output_sampling_rate > 0 && c.cond.output_hop_length > 0,
         "cond output rate metadata is 0 or missing");
    need(c.dit.block_count > 0 && c.dit.embedding_length > 0, "dit dims are 0 or missing");
    need(c.dit.head_count * c.dit.head_dim == c.dit.embedding_length,
         "dit head_count * head_dim != embedding_length");
    need(c.dit.concat_channels == c.dit.in_channels * 2 + c.dit.condition_dim,
         "dit concat_channels != 2*in_channels + condition_dim");
    need(c.dit.window_frames > 0 && c.dit.hop_frames > 0, "dit window metadata is 0 or missing");
    need(c.dit.window_latents > 0 && c.dit.hop_latents > 0,
         "dit latent window metadata is 0 or missing");
    need(c.flow.steps > 0 && c.flow.cfg_scale > 0.0f, "flow defaults are 0 or missing");
    need(c.voc.upsample_rates.size() == 4, "mm3.voc.upsample_rates is not 4 entries");
    const std::string upsample_error =
        tts_cpp::minimax::detail::vocoder_upsample_error(c.voc.upsample_rates, c.voc.total_upsample);
    need(upsample_error.empty(), upsample_error.c_str());
    need(c.voc.res_dilations.size() == 3, "mm3.voc.res_dilations is not 3 entries");
    need(c.voc.hidden_dim > 0 && c.voc.dec_in_dim > 0, "voc dims are 0 or missing");
    need(c.voc.sampling_rate > 0, "vocoder sampling rate is 0 or missing");
    need(c.voc.channels == 2, "vocoder channels must be 2");
    need(c.voc.latent_channels == c.voc.fold_channels * 2,
         "voc latent_channels != 2 * fold_channels (stereo fold)");
    const tts_cpp::minimax::detail::SynthesisContract contract{
        c.flow.scheduler,
        c.flow.invert_sigmas,
        c.flow.shift,
        c.flow.num_train_timesteps,
        c.dit.rope_type,
        c.dit.glu_order,
        c.dit.timestep_token_prepended,
        c.dit.pre_post_conv_residual,
        c.dit.attn_bias,
    };
    mm3_append_synth_contract_errors(
        tts_cpp::minimax::detail::validate_synthesis_contract(contract), errs);
}

using MM3GgufOpen = bool (*)(GGUFModel *, const char *);

static void mm3_probe_file(const std::string & path, MM3FileInfo * fi, MM3LmConfig * lm_cfg,
                           MM3SynthConfig * synth_cfg, std::vector<std::string> * errs,
                           MM3GgufOpen open_gguf = gf_load) {
    fi->found = true;
    fi->path  = path;
    fi->name  = mm3_basename(path);

    GGUFModel gf = {};
    if (!open_gguf(&gf, path.c_str())) {
        fi->probe_error = "gguf header parse failed";
        if (errs && errs->size() < 24) {
            errs->push_back(fi->name + ": " + fi->probe_error);
        }
        return;
    }

    fi->file_bytes        = (uint64_t) gf.file_size;
    fi->arch              = gf_get_str(gf, "general.architecture");
    fi->general_name      = gf_get_str(gf, "general.name");
    fi->license           = gf_get_str(gf, "general.license");
    fi->source_layout     = gf_get_str(gf, "mm3.source_layout");
    fi->converter_version = gf_get_u32(gf, "mm3.converter_version");
    fi->file_type         = gf_get_u32(gf, "general.file_type");
    fi->n_tensors         = (int) gguf_get_n_tensors(gf.gguf);

    uint64_t bytes = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gf.gguf); i++) {
        ggml_tensor * t = ggml_get_tensor(gf.meta, gguf_get_tensor_name(gf.gguf, i));
        if (t) {
            bytes += (uint64_t) ggml_nbytes(t);
        }
    }
    fi->tensor_bytes = bytes;

    const std::string model_kv = gf_get_str(gf, "mm3.model");
    if (model_kv != "MiniMax-Music3") {
        fi->probe_error = "mm3.model KV is '" + model_kv + "', expected 'MiniMax-Music3'";
        if (errs && errs->size() < 24) {
            errs->push_back(fi->name + ": " + fi->probe_error);
        }
        gf_close(&gf);
        return;
    }

    if (lm_cfg) {
        mm3_parse_lm_config(gf, lm_cfg);
        mm3_validate_lm_config(*lm_cfg, errs);
    }
    if (synth_cfg) {
        mm3_parse_synth_config(gf, synth_cfg);
        mm3_validate_synth_config(*synth_cfg, errs);
    }

    fi->probe_ok = true;
    gf_close(&gf);
}

static void mm3_discover(MM3Model * m, const char * models_dir) {
    m->models_dir = models_dir ? models_dir : "";
    m->search_dirs.clear();
    m->meta_errors.clear();

    if (m->models_dir.empty()) {
        return;
    }

    const std::string sub = m->models_dir + MM3_SEP "mm3";
    m->search_dirs.push_back(sub);
    m->search_dirs.push_back(m->models_dir);

    const char * quants[] = { "f16", "q8_0", "BF16", "Q8_0" };

    auto find_one = [&](const char * role, MM3FileInfo * fi) {
        for (const auto & dir : m->search_dirs) {
            for (const char * q : quants) {
                std::string cand = dir + MM3_SEP "mm3-" + role + "-" + q + ".gguf";
                if (mm3_file_exists(cand)) {
                    fi->found = true;
                    fi->path  = cand;
                    fi->name  = mm3_basename(cand);
                    return;
                }
            }
        }
    };

    find_one("lm", &m->lm_file);
    find_one("synth", &m->synth_file);

    if (m->lm_file.found) {
        mm3_probe_file(m->lm_file.path, &m->lm_file, &m->lm_cfg, nullptr, &m->meta_errors);
        if (m->lm_file.probe_ok && m->lm_file.arch != "qwen3") {
            m->meta_errors.push_back(m->lm_file.name + ": general.architecture is '" + m->lm_file.arch +
                                     "', expected 'qwen3'");
        }
    }
    if (m->synth_file.found) {
        mm3_probe_file(m->synth_file.path, &m->synth_file, nullptr, &m->synth_cfg, &m->meta_errors);
        if (m->synth_file.probe_ok && m->synth_file.arch != "mm3") {
            m->meta_errors.push_back(m->synth_file.name + ": general.architecture is '" + m->synth_file.arch +
                                     "', expected 'mm3'");
        }
    }

    if (!m->lm_file.found && !m->synth_file.found) {
        fprintf(stderr, "[MM3] No MiniMax-Music3 GGUFs found (looked in %s and %s)\n", sub.c_str(),
                m->models_dir.c_str());
        return;
    }
    fprintf(stderr, "[MM3] LM:    %s\n", m->lm_file.found ? m->lm_file.path.c_str() : "(not found)");
    fprintf(stderr, "[MM3] Synth: %s\n", m->synth_file.found ? m->synth_file.path.c_str() : "(not found)");
    if (m->lm_file.probe_ok) {
        fprintf(stderr, "[MM3] LM cfg: %uL H=%u V=%u heads=%u/%u codebooks=%u sem@%u+%u fps=%u\n",
                m->lm_cfg.block_count, m->lm_cfg.embedding_length, m->lm_cfg.vocab_size, m->lm_cfg.head_count,
                m->lm_cfg.head_count_kv, m->lm_cfg.num_codebooks, m->lm_cfg.semantic_vocab_offset,
                m->lm_cfg.semantic_vocab_size, m->lm_cfg.frame_rate);
    }
    if (m->synth_file.probe_ok) {
        fprintf(stderr, "[MM3] Synth cfg: depth %uL/%u  cond %u->%u  dit %uL/%u  voc %u->%u Hz x%u\n",
                m->synth_cfg.depth.block_count, m->synth_cfg.depth.embedding_length, m->synth_cfg.cond.hidden_dim,
                m->synth_cfg.cond.out_dim, m->synth_cfg.dit.block_count, m->synth_cfg.dit.embedding_length,
                m->synth_cfg.voc.latent_channels, m->synth_cfg.voc.sampling_rate, m->synth_cfg.voc.total_upsample);
    }
    for (const auto & e : m->meta_errors) {
        fprintf(stderr, "[MM3] WARNING: %s\n", e.c_str());
    }
}

static bool mm3_available(const MM3Model & m) {
    return m.lm_file.probe_ok && m.synth_file.probe_ok && m.meta_errors.empty();
}

// Loads a layer's q/k/v (or gate/up) projections stacked into one matrix, or
// one by one when they cannot be stacked. `prefix` is the layer's name prefix,
// e.g. "blk.3." or "depth.blk.3.".
static void mm3_load_attention_inputs(MM3Loader & ld, const std::string & prefix, int64_t H, int64_t Q, int64_t K,
                                      ggml_tensor ** qkv, ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v) {
    const std::string q_name = prefix + "attn_q.weight";
    const std::string k_name = prefix + "attn_k.weight";
    const std::string v_name = prefix + "attn_v.weight";
    *qkv = ld.try_stacked(prefix + "attn_qkv.weight", { q_name, k_name, v_name }, H, { Q, K, K });
    if (*qkv) {
        return;
    }
    *q = ld.req(q_name, H, Q);
    *k = ld.req(k_name, H, K);
    *v = ld.req(v_name, H, K);
}

static void mm3_load_ffn_inputs(MM3Loader & ld, const std::string & prefix, int64_t H, int64_t F,
                                ggml_tensor ** gate_up, ggml_tensor ** gate, ggml_tensor ** up) {
    const std::string gate_name = prefix + "ffn_gate.weight";
    const std::string up_name   = prefix + "ffn_up.weight";
    *gate_up = ld.try_stacked(prefix + "ffn_gate_up.weight", { gate_name, up_name }, H, { F, F });
    if (*gate_up) {
        return;
    }
    *gate = ld.req(gate_name, H, F);
    *up   = ld.req(up_name, H, F);
}

static bool mm3_load_lm_tensors(MM3Model * m, const GGUFModel & gf, std::vector<std::string> * errs) {
    const MM3LmConfig & c = m->lm_cfg;
    const int64_t       H = c.embedding_length;
    const int64_t       V = c.vocab_size;
    const int64_t       D = c.key_length;
    const int64_t       Q = (int64_t) c.head_count * D;
    const int64_t       K = (int64_t) c.head_count_kv * D;
    const int64_t       F = c.feed_forward_length;
    const int           L = (int) c.block_count;

    wctx_init(&m->wctx_lm, 3 + L * 11);
    MM3Loader ld{ &m->wctx_lm, &gf, &m->tmap_lm, errs };

    m->lm.token_embd     = ld.req("token_embd.weight", H, V);
    m->lm.output_norm    = ld.req("output_norm.weight", H);
    m->lm.output_compact = ld.req_compact_head("output.weight", H, V, (int64_t) c.semantic_vocab_offset,
                                               (int64_t) c.semantic_vocab_size, (int64_t) c.eos_audio);

    m->lm.blk.assign((size_t) L, MM3LmLayer{});
    for (int i = 0; i < L; i++) {
        MM3LmLayer &      b      = m->lm.blk[(size_t) i];
        const std::string prefix = mm3_fmt("blk.%d.", i);
        b.attn_norm    = ld.req(prefix + "attn_norm.weight", H);
        mm3_load_attention_inputs(ld, prefix, H, Q, K, &b.attn_qkv, &b.attn_q, &b.attn_k, &b.attn_v);
        b.attn_output  = ld.req(prefix + "attn_output.weight", Q, H);
        b.attn_q_norm  = ld.req(prefix + "attn_q_norm.weight", D);
        b.attn_k_norm  = ld.req(prefix + "attn_k_norm.weight", D);
        b.ffn_norm     = ld.req(prefix + "ffn_norm.weight", H);
        mm3_load_ffn_inputs(ld, prefix, H, F, &b.ffn_gate_up, &b.ffn_gate, &b.ffn_up);
        b.ffn_down     = ld.req(prefix + "ffn_down.weight", F, H);
        if (!errs->empty()) {
            break;
        }
    }
    return errs->empty();
}

static bool mm3_load_synth_tensors(MM3Model * m, const GGUFModel & gf, std::vector<std::string> * errs) {
    const MM3SynthConfig & c = m->synth_cfg;

    const int n_depth = 4 + (int) (c.depth.num_codebooks - 1) + (int) c.depth.block_count * 9;
    const int n_cond  = 4;
    const int n_dit   = 10 + (int) c.dit.block_count * 10;
    const int n_voc   = 7 + (int) c.voc.upsample_rates.size() * (3 + (int) c.voc.res_dilations.size() * 6);
    wctx_init(&m->wctx_synth, n_depth + n_cond + n_dit + n_voc);

    MM3Loader ld{ &m->wctx_synth, &gf, &m->tmap_synth, errs };

    {
        const MM3DepthConfig & d  = c.depth;
        const int64_t          H  = d.embedding_length;
        const int64_t          F  = d.feed_forward_length;
        const int              L  = (int) d.block_count;
        const int              NC = (int) d.num_codebooks - 1;

        m->synth.depth.proj        = ld.req("depth.proj.weight", H, H);
        m->synth.depth.pos_embd    = ld.req("depth.pos_embd.weight", H, d.max_position);
        m->synth.depth.audio_embd  = ld.req("depth.audio_embd.weight", H, d.audio_embd_rows);
        m->synth.depth.output_norm = ld.req("depth.output_norm.weight", H);

        m->synth.depth.head.assign((size_t) NC, nullptr);
        for (int i = 0; i < NC; i++) {
            m->synth.depth.head[(size_t) i] = ld.req(mm3_fmt("depth.head.%d.weight", i), H, d.audio_vocab_size);
        }

        m->synth.depth.blk.assign((size_t) L, MM3DepthLayer{});
        for (int i = 0; i < L && errs->empty(); i++) {
            MM3DepthLayer &   b      = m->synth.depth.blk[(size_t) i];
            const std::string prefix = mm3_fmt("depth.blk.%d.", i);
            b.attn_norm       = ld.req(prefix + "attn_norm.weight", H);
            mm3_load_attention_inputs(ld, prefix, H, H, H, &b.attn_qkv, &b.attn_q, &b.attn_k, &b.attn_v);
            b.attn_output     = ld.req(prefix + "attn_output.weight", H, H);
            b.ffn_norm        = ld.req(prefix + "ffn_norm.weight", H);
            mm3_load_ffn_inputs(ld, prefix, H, F, &b.ffn_gate_up, &b.ffn_gate, &b.ffn_up);
            b.ffn_down        = ld.req(prefix + "ffn_down.weight", F, H);
        }
    }

    {
        const MM3CondConfig & e = c.cond;
        m->synth.cond.layer_logits = ld.req("cond.layer_logits", e.num_layers);
        m->synth.cond.layer_scale  = ld.req("cond.layer_scale", 1);
        if (!m->coreml_conditioner) {
            m->synth.cond.proj_w = ld.req("cond.proj.weight", e.kernel_size, e.hidden_dim, e.out_dim);
            m->synth.cond.proj_b = ld.req("cond.proj.bias", e.out_dim);
        }
    }

    {
        const MM3DitConfig & t  = c.dit;
        const int64_t        E  = t.embedding_length;
        const int64_t        CC = t.concat_channels;
        const int64_t        IC = t.in_channels;
        const int64_t        FI = t.ff_inner;
        const int            L  = (int) t.block_count;

        m->synth.dit.preprocess_conv  = ld.req("dit.preprocess_conv.weight", 1, CC, CC);
        m->synth.dit.postprocess_conv = ld.req("dit.postprocess_conv.weight", 1, IC, IC);
        m->synth.dit.time_fourier     = ld.req("dit.time_fourier.weight", 1, t.fourier_dim / 2);
        m->synth.dit.time_embd_w[0]   = ld.req("dit.time_embd.0.weight", t.fourier_dim, E);
        m->synth.dit.time_embd_b[0]   = ld.req("dit.time_embd.0.bias", E);
        m->synth.dit.time_embd_w[1]   = ld.req("dit.time_embd.1.weight", E, E);
        m->synth.dit.time_embd_b[1]   = ld.req("dit.time_embd.1.bias", E);
        m->synth.dit.proj_in          = ld.req("dit.proj_in.weight", CC, E);
        m->synth.dit.proj_out         = ld.req("dit.proj_out.weight", E, IC);
        m->synth.dit.rope_inv_freq    = ld.req("dit.rope_inv_freq", t.rope_dim / 2);

        m->synth.dit.blk.assign((size_t) L, MM3DitBlock{});
        for (int i = 0; i < L && errs->empty(); i++) {
            MM3DitBlock & b = m->synth.dit.blk[(size_t) i];
            b.attn_norm_w   = ld.req(mm3_fmt("dit.blk.%d.attn_norm.weight", i), E);
            b.attn_norm_b   = ld.req(mm3_fmt("dit.blk.%d.attn_norm.bias", i), E);
            b.attn_qkv      = ld.req(mm3_fmt("dit.blk.%d.attn_qkv.weight", i), E, E * 3);
            b.attn_output   = ld.req(mm3_fmt("dit.blk.%d.attn_output.weight", i), E, E);
            b.ffn_norm_w    = ld.req(mm3_fmt("dit.blk.%d.ffn_norm.weight", i), E);
            b.ffn_norm_b    = ld.req(mm3_fmt("dit.blk.%d.ffn_norm.bias", i), E);
            b.ffn_in_w      = ld.req(mm3_fmt("dit.blk.%d.ffn_in.weight", i), E, FI * 2);
            b.ffn_in_b      = ld.req(mm3_fmt("dit.blk.%d.ffn_in.bias", i), FI * 2);
            b.ffn_out_w     = ld.req(mm3_fmt("dit.blk.%d.ffn_out.weight", i), FI, E);
            b.ffn_out_b     = ld.req(mm3_fmt("dit.blk.%d.ffn_out.bias", i), E);
        }
    }

    {
        const MM3VocConfig & v  = c.voc;
        const int            NB = (int) v.upsample_rates.size();
        const int            NR = (int) v.res_dilations.size();

        m->synth.voc.dec_in_w  = ld.req("voc.dec_in.weight", 1, v.fold_channels, v.dec_in_dim);
        m->synth.voc.dec_in_b  = ld.req("voc.dec_in.bias", v.dec_in_dim);
        m->synth.voc.conv_in_w = ld.req("voc.conv_in.weight", 7, v.dec_in_dim, v.hidden_dim);
        m->synth.voc.conv_in_b = ld.req("voc.conv_in.bias", v.hidden_dim);

        m->synth.voc.blk.assign((size_t) NB, MM3VocBlock{});
        int64_t cin = (int64_t) v.hidden_dim;
        for (int b = 0; b < NB && errs->empty(); b++) {
            const int64_t cout   = cin / 2;
            const int64_t stride = v.upsample_rates[(size_t) b];
            MM3VocBlock & vb     = m->synth.voc.blk[(size_t) b];

            vb.snake_alpha = ld.req(mm3_fmt("voc.blk.%d.snake.alpha", b), 1, cin, 1);
            vb.convt_w     = ld.req(mm3_fmt("voc.blk.%d.convt.weight", b), stride * 2, cout, cin);
            vb.convt_b     = ld.req(mm3_fmt("voc.blk.%d.convt.bias", b), cout);

            vb.res.assign((size_t) NR, MM3VocResUnit{});
            for (int r = 0; r < NR && errs->empty(); r++) {
                MM3VocResUnit & ru = vb.res[(size_t) r];
                ru.snake1_alpha    = ld.req(mm3_fmt2("voc.blk.%d.res.%d.snake1.alpha", b, r), 1, cout, 1);
                ru.conv1_w         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv1.weight", b, r), 7, cout, cout);
                ru.conv1_b         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv1.bias", b, r), cout);
                ru.snake2_alpha    = ld.req(mm3_fmt2("voc.blk.%d.res.%d.snake2.alpha", b, r), 1, cout, 1);
                ru.conv2_w         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv2.weight", b, r), 1, cout, cout);
                ru.conv2_b         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv2.bias", b, r), cout);
            }
            cin = cout;
        }

        m->synth.voc.snake_out_alpha = ld.req("voc.snake_out.alpha", 1, cin, 1);
        m->synth.voc.conv_out_w      = ld.req("voc.conv_out.weight", 7, cin, 1);
        m->synth.voc.conv_out_b      = ld.req("voc.conv_out.bias", 1);
    }

    return errs->empty();
}

static void mm3_unload(MM3Model * m) {
#ifdef MM3_ANE_VALIDATION
    m->ane_validation.reset();
#endif
    if (!m->loaded && !m->wctx_lm.ctx && !m->wctx_synth.ctx && !m->backend_ref && !m->coreml_conditioner) {
        return;
    }
    mm3_coreml_conditioner_free(m->coreml_conditioner);
    m->coreml_conditioner = nullptr;
    wctx_free(&m->wctx_lm);
    wctx_free(&m->wctx_synth);
    m->lm    = MM3LmWeights{};
    m->synth = MM3SynthWeights{};
    m->tmap_lm.clear();
    m->tmap_synth.clear();
    m->vram_lm    = 0;
    m->vram_synth = 0;
    m->load_ms    = 0.0;
    m->loaded     = false;
    if (m->backend_ref) {
        backend_release(m->backend, m->cpu_backend);
        m->backend     = nullptr;
        m->cpu_backend = nullptr;
        m->backend_ref = false;
    }
    fprintf(stderr, "[MM3] Unloaded\n");
}

static bool mm3_load(MM3Model * m, std::string * err_out) {
    if (m->loaded) {
        return true;
    }
    if (!m->lm_file.found || !m->synth_file.found) {
        if (err_out) {
            *err_out = "MiniMax-Music3 GGUFs not found (need mm3-lm-*.gguf and mm3-synth-*.gguf)";
        }
        return false;
    }
    if (!m->lm_file.probe_ok || !m->synth_file.probe_ok || !m->meta_errors.empty()) {
        if (err_out) {
            *err_out = m->meta_errors.empty() ? "MiniMax-Music3 GGUF metadata probe failed; see /mm3/props errors"
                                              : m->meta_errors[0];
        }
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();

#ifdef MM3_ANE_VALIDATION
    if (!m->ane_validation_directory.empty()) {
        m->ane_validation = mm3_ane_validation::load(*m, m->ane_validation_directory);
        m->loaded = true;
        m->load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count();
        return true;
    }
#endif

    if (!m->coreml_conditioner_path.empty()) {
        const auto & c = m->synth_cfg.cond;
        if (c.kernel_size != 3 || c.padding != 1) {
            if (err_out) *err_out = "Core ML conditioner requires kernel=3 and padding=1";
            return false;
        }
        m->coreml_conditioner = mm3_coreml_conditioner_load(
            m->coreml_conditioner_path, c.hidden_dim, c.out_dim, err_out);
        if (!m->coreml_conditioner) return false;
        fprintf(stderr, "[MM3] Conditioner: Core ML CPU + ANE, %s\n", m->coreml_conditioner_path.c_str());
    }

    BackendPair bp = backend_init("MM3");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->backend_ref = true;

    std::vector<std::string> errs;

    GGUFModel gf_lm = {};
    if (!gf_load(&gf_lm, m->lm_file.path.c_str())) {
        if (err_out) {
            *err_out = "cannot open " + m->lm_file.path;
        }
        mm3_unload(m);
        return false;
    }
    bool ok = mm3_load_lm_tensors(m, gf_lm, &errs);
    if (ok) {
        ok = wctx_alloc(&m->wctx_lm, m->backend);
        if (!ok) {
            errs.push_back("backend buffer allocation failed for the LM (out of VRAM?)");
        }
    }
    gf_close(&gf_lm);

    if (ok) {
        GGUFModel gf_sy = {};
        if (!gf_load(&gf_sy, m->synth_file.path.c_str())) {
            errs.push_back("cannot open " + m->synth_file.path);
            ok = false;
        } else {
            ok = mm3_load_synth_tensors(m, gf_sy, &errs);
            if (ok) {
                ok = wctx_alloc(&m->wctx_synth, m->backend);
                if (!ok) {
                    errs.push_back("backend buffer allocation failed for the synth stack (out of VRAM?)");
                }
            }
            gf_close(&gf_sy);
        }
    }

    if (!ok) {
        std::string msg = errs.empty() ? "MM3 load failed" : errs[0];
        for (size_t i = 1; i < errs.size() && i < 6; i++) {
            msg += "; " + errs[i];
        }
        fprintf(stderr, "[MM3] LOAD FAILED: %s\n", msg.c_str());
        if (err_out) {
            *err_out = msg;
        }
        mm3_unload(m);
        return false;
    }

    m->vram_lm    = m->wctx_lm.buffer ? ggml_backend_buffer_get_size(m->wctx_lm.buffer) : 0;
    m->vram_synth = m->wctx_synth.buffer ? ggml_backend_buffer_get_size(m->wctx_synth.buffer) : 0;
    m->loaded     = true;
    m->load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    fprintf(stderr, "[MM3] Loaded: LM %.2f GB (%zu tensors) + synth %.2f GB (%zu tensors) = %.2f GB in %.0f ms\n",
            (double) m->vram_lm / (1024.0 * 1024.0 * 1024.0), m->tmap_lm.size(),
            (double) m->vram_synth / (1024.0 * 1024.0 * 1024.0), m->tmap_synth.size(),
            (double) (m->vram_lm + m->vram_synth) / (1024.0 * 1024.0 * 1024.0), m->load_ms);
    return true;
}

static size_t mm3_vram_bytes(const MM3Model & m) {
    return m.vram_lm + m.vram_synth;
}
