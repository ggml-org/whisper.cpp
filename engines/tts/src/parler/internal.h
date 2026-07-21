#pragma once
// Parler-TTS engine internals: hparams, model weights, stage entry points.
// Everything hparam-driven from GGUF metadata — no mini/large branching.

#include "ggml.h"
#include "ggml-backend.h"
#include "sched_dispatch.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

// graph-node budget for every parler graph builder (t5 / decoder / dac)
constexpr int PARLER_MAX_NODES = 4096;

struct parler_hparams {
    // t5 encoder
    int   t5_n_layer      = 0;
    int   t5_d_model      = 0;
    int   t5_d_ff         = 0;
    int   t5_n_head       = 0;
    int   t5_d_kv         = 0;
    int   t5_rel_buckets  = 0;
    int   t5_rel_max_dist = 0;
    float t5_rms_eps      = 1e-6f;
    int   t5_vocab        = 0;
    // decoder LM
    int   dec_n_layer   = 0;
    int   dec_d_model   = 0;
    int   dec_n_head    = 0;
    int   dec_d_ff      = 0;
    int   n_codebooks   = 0;
    int   dec_vocab     = 0;   // 1088 (codes 0..1023, eos/pad 1024, bos 1025)
    float dec_ln_eps    = 1e-5f;
    int   bos_id        = 0;
    int   eos_id        = 0;
    int   pad_id        = 0;
    int   dec_start_id  = 0;
    int   max_position  = 0;   // sinusoidal table rows
    bool  enc_to_dec    = false;
    // generation defaults (from HF generation_config)
    int   gen_max_length     = 0;
    int   gen_min_new_tokens = 0;
    bool  gen_do_sample      = false;
    float gen_temperature    = 1.0f;
    int   gen_top_k          = 0;
    // dac
    int   dac_sample_rate   = 0;
    int   dac_n_q           = 0;
    int   dac_codebook_size = 0;
    int   dac_latent        = 0;
    int   dac_decoder_dim   = 0;
    int   dac_hop           = 0;
    std::vector<int> dac_rates;
    // runtime geometry
    int   n_ctx = 0;           // decoder KV capacity (prompt + delayed steps)
};

struct parler_t5_layer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
};

struct parler_dec_layer {
    ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr;
    ggml_tensor * cross_norm_w = nullptr, * cross_norm_b = nullptr;
    ggml_tensor * cq = nullptr, * ck = nullptr, * cv = nullptr, * co = nullptr;
    ggml_tensor * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
    ggml_tensor * up = nullptr, * down = nullptr;
};

struct parler_dac_residual {
    ggml_tensor * snake1_alpha = nullptr;
    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;  // k7, dilated
    ggml_tensor * snake2_alpha = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;  // k1
};

struct parler_dac_block {
    ggml_tensor * snake_alpha = nullptr;
    ggml_tensor * convt_w = nullptr, * convt_b = nullptr;  // k=2*stride
    int stride = 0;
    parler_dac_residual res[3];                            // dilations 1, 3, 9
};

struct parler_dac_quant {
    ggml_tensor * codebook = nullptr;                      // [codebook_dim, 1024]
    ggml_tensor * out_proj_w = nullptr, * out_proj_b = nullptr;  // 1x1 conv
};

struct parler_model {
    parler_hparams hparams;

    ggml_backend_t        backend  = nullptr;
    ggml_context        * ctx_w    = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    mutable ::tts_cpp::detail::sched_fallback sched_fb;

    // t5
    ggml_tensor * t5_embed = nullptr;
    ggml_tensor * t5_rel_b = nullptr;          // [n_head, rel_buckets]
    ggml_tensor * t5_output_norm = nullptr;
    std::vector<parler_t5_layer> t5_layers;
    // glue
    ggml_tensor * enc_to_dec_w = nullptr, * enc_to_dec_b = nullptr;  // optional
    ggml_tensor * embed_prompts = nullptr;
    // decoder
    ggml_tensor * embed_positions = nullptr;   // [d_model, max_position]
    std::vector<ggml_tensor *> dec_embed;      // n_codebooks tables
    std::vector<ggml_tensor *> lm_heads;       // n_codebooks heads
    ggml_tensor * dec_output_norm_w = nullptr, * dec_output_norm_b = nullptr;
    std::vector<parler_dec_layer> dec_layers;
    // dac
    std::vector<parler_dac_quant> dac_quant;
    ggml_tensor * dac_conv_in_w = nullptr, * dac_conv_in_b = nullptr;
    std::vector<parler_dac_block> dac_blocks;
    ggml_tensor * dac_snake_out_alpha = nullptr;
    ggml_tensor * dac_conv_out_w = nullptr, * dac_conv_out_b = nullptr;

    // decoder self-attention KV cache (token-major slab, one per K/V):
    // [d_model, n_ctx] rows per layer, stacked layer-major.
    ggml_context        * ctx_kv    = nullptr;
    ggml_backend_buffer_t buffer_kv = nullptr;
    ggml_tensor * memory_k = nullptr;
    ggml_tensor * memory_v = nullptr;

    // per-description cross-attention K/V (rebuilt when the description
    // changes): cross_k[l] = [d_model, T], cross_v_t[l] = [T, d_model].
    ggml_context        * ctx_cross    = nullptr;
    ggml_backend_buffer_t buffer_cross = nullptr;
    std::vector<ggml_tensor *> cross_k;
    std::vector<ggml_tensor *> cross_v_t;
    int cross_len = 0;

    // tokenizer payload (host-side; consumed by parler_tokenizer)
    std::vector<std::string> tok_pieces;
    std::vector<float>       tok_scores;
    std::vector<uint8_t>     tok_charsmap;
    int  tok_unk_id  = 2;
    int  tok_eos_id  = 1;
    bool tok_add_eos = true;

    // optional separate BPE prompt tokenizer (indic-class checkpoints);
    // absent => the unigram tokenizer above serves prompts too
    bool has_prompt_tok = false;
    std::vector<std::string> ptok_pieces;
    std::vector<std::string> ptok_merges;
    int  ptok_unk_id  = 0;
    int  ptok_bos_id  = 1;
    bool ptok_add_bos = true;
};

// ---- parler_gguf.cpp ----
bool parler_load_gguf(const std::string & path, parler_model & model,
                      std::string * error = nullptr);
void parler_free_model(parler_model & model);

// Dual-path graph dispatch honoring the sched_dispatch contract (gf must be
// freshly built per call; set inputs AFTER this returns true, via
// ggml_backend_tensor_set on named graph tensors).
bool parler_graph_prepare(const parler_model & model, ggml_cgraph * gf,
                          ggml_gallocr_t allocr, bool & use_sched, const char * caller);
bool parler_graph_compute(const parler_model & model, ggml_cgraph * gf,
                          bool use_sched, int n_threads, const char * caller);

// ---- parler_t5.cpp ----
// Run the T5 encoder on the description ids and produce the cross-attention
// source states [T, dec_d_model] (enc_to_dec projection applied when present),
// then precompute per-layer cross K/V into model.ctx_cross.
bool parler_encode_description(parler_model & model,
                               const std::vector<int32_t> & desc_ids,
                               int n_threads,
                               std::vector<float> * states_out = nullptr);

// ---- parler_decoder.cpp ----
// Prefill: prompt embeds prepended to the BOS start frame; returns logits for
// the LAST position, [n_codebooks * dec_vocab] row-major by codebook.
// n_past_out = prompt_len + 1.
bool parler_dec_prefill(const parler_model & model,
                        const std::vector<int32_t> & prompt_ids,
                        const std::vector<int32_t> & start_frame,
                        ggml_gallocr_t allocr, int n_threads,
                        std::vector<float> & logits_out, int & n_past_out);
// One decode step: frame = n_codebooks token ids (delay-mask applied by the
// caller), position = n_past.
bool parler_dec_step(const parler_model & model,
                     const std::vector<int32_t> & frame,
                     int n_past,
                     ggml_gallocr_t allocr, int n_threads,
                     std::vector<float> & logits_out);

// ---- parler_dac.cpp ----
// codes: [n_codebooks, n_frames] row-major, values in [0, codebook_size).
// latent_out (optional, tests): summed RVQ latent, [latent_dim, T] ggml
// layout (element (c, t) at t*latent_dim + c).
bool parler_dac_decode(const parler_model & model,
                       const int32_t * codes, int n_frames,
                       int n_threads,
                       std::vector<float> & pcm_out,
                       std::vector<float> * latent_out = nullptr);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
