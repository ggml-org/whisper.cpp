#include "internal.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

bool kv_u32(const gguf_context * g, const char * key, int & out, std::string * err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        if (err) *err = std::string("missing GGUF key: ") + key;
        return false;
    }
    out = (int) gguf_get_val_u32(g, id);
    return true;
}

bool kv_f32(const gguf_context * g, const char * key, float & out, std::string * err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        if (err) *err = std::string("missing GGUF key: ") + key;
        return false;
    }
    out = gguf_get_val_f32(g, id);
    return true;
}

bool kv_bool(const gguf_context * g, const char * key, bool & out, std::string * err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        if (err) *err = std::string("missing GGUF key: ") + key;
        return false;
    }
    out = gguf_get_val_bool(g, id);
    return true;
}

// FA probe: build a representative flash_attn_ext node and ask the backend if
// it supports it. CPU returns false so it keeps the validated manual F32 path.
bool parler_probe_fa_f16(ggml_backend_t backend, int head_dim, int n_heads) {
    if (::tts_cpp::detail::backend_is_cpu(backend)) return false;
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
    ggml_context * c = ggml_init(ip);
    if (!c) return false;
    ggml_tensor * q = ggml_new_tensor_3d(c, GGML_TYPE_F32, head_dim, 16, n_heads);
    ggml_tensor * k = ggml_new_tensor_3d(c, GGML_TYPE_F16, head_dim, 16, n_heads);
    ggml_tensor * v = ggml_new_tensor_3d(c, GGML_TYPE_F16, head_dim, 16, n_heads);
    ggml_tensor * op = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / (float) head_dim, 0.0f, 0.0f);
    const bool ok = op && ggml_backend_supports_op(backend, op);
    ggml_free(c);
    return ok;
}

} // namespace

bool parler_load_gguf(const std::string & path, parler_model & model,
                      int n_gpu_layers, std::string * error) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &ctx_meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        if (error) *error = "failed to open GGUF: " + path;
        return false;
    }

    auto fail = [&](const std::string & msg) {
        if (error) *error = msg;
        gguf_free(g);
        if (ctx_meta) ggml_free(ctx_meta);
        parler_free_model(model);
        return false;
    };

    if (gguf_find_key(g, "parler.arch") < 0) {
        return fail("not a parler GGUF (missing parler.arch)");
    }

    parler_hparams & hp = model.hparams;
    std::string err;
    if (!kv_u32(g, "parler.t5.n_layer", hp.t5_n_layer, &err) ||
        !kv_u32(g, "parler.t5.d_model", hp.t5_d_model, &err) ||
        !kv_u32(g, "parler.t5.d_ff", hp.t5_d_ff, &err) ||
        !kv_u32(g, "parler.t5.n_head", hp.t5_n_head, &err) ||
        !kv_u32(g, "parler.t5.d_kv", hp.t5_d_kv, &err) ||
        !kv_u32(g, "parler.t5.rel_buckets", hp.t5_rel_buckets, &err) ||
        !kv_u32(g, "parler.t5.rel_max_dist", hp.t5_rel_max_dist, &err) ||
        !kv_f32(g, "parler.t5.rms_eps", hp.t5_rms_eps, &err) ||
        !kv_u32(g, "parler.t5.vocab_size", hp.t5_vocab, &err) ||
        !kv_u32(g, "parler.dec.n_layer", hp.dec_n_layer, &err) ||
        !kv_u32(g, "parler.dec.d_model", hp.dec_d_model, &err) ||
        !kv_u32(g, "parler.dec.n_head", hp.dec_n_head, &err) ||
        !kv_u32(g, "parler.dec.d_ff", hp.dec_d_ff, &err) ||
        !kv_u32(g, "parler.dec.n_codebooks", hp.n_codebooks, &err) ||
        !kv_u32(g, "parler.dec.vocab_size", hp.dec_vocab, &err) ||
        !kv_f32(g, "parler.dec.ln_eps", hp.dec_ln_eps, &err) ||
        !kv_u32(g, "parler.dec.bos_token_id", hp.bos_id, &err) ||
        !kv_u32(g, "parler.dec.eos_token_id", hp.eos_id, &err) ||
        !kv_u32(g, "parler.dec.pad_token_id", hp.pad_id, &err) ||
        !kv_u32(g, "parler.dec.decoder_start_token_id", hp.dec_start_id, &err) ||
        !kv_u32(g, "parler.dec.max_position", hp.max_position, &err) ||
        !kv_bool(g, "parler.enc_to_dec", hp.enc_to_dec, &err) ||
        !kv_u32(g, "parler.gen.max_length", hp.gen_max_length, &err) ||
        !kv_u32(g, "parler.gen.min_new_tokens", hp.gen_min_new_tokens, &err) ||
        !kv_bool(g, "parler.gen.do_sample", hp.gen_do_sample, &err) ||
        !kv_f32(g, "parler.gen.temperature", hp.gen_temperature, &err) ||
        !kv_u32(g, "parler.gen.top_k", hp.gen_top_k, &err) ||
        !kv_u32(g, "parler.dac.sample_rate", hp.dac_sample_rate, &err) ||
        !kv_u32(g, "parler.dac.n_quantizers", hp.dac_n_q, &err) ||
        !kv_u32(g, "parler.dac.codebook_size", hp.dac_codebook_size, &err) ||
        !kv_u32(g, "parler.dac.latent_dim", hp.dac_latent, &err) ||
        !kv_u32(g, "parler.dac.decoder_dim", hp.dac_decoder_dim, &err) ||
        !kv_u32(g, "parler.dac.hop", hp.dac_hop, &err)) {
        return fail(err);
    }
    {
        const int64_t id = gguf_find_key(g, "parler.dac.rates");
        if (id < 0) return fail("missing GGUF key: parler.dac.rates");
        const int n = (int) gguf_get_arr_n(g, id);
        hp.dac_rates.resize(n);
        for (int i = 0; i < n; ++i) {
            hp.dac_rates[i] = (int) ((const int32_t *) gguf_get_arr_data(g, id))[i];
        }
    }

    // decoder KV capacity: full generation budget plus prompt headroom,
    // capped at the sinusoidal table size.
    hp.n_ctx = std::min(hp.max_position, hp.gen_max_length + 640);

    // ---- tokenizer payload ----
    {
        int64_t id = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (id < 0) return fail("missing tokenizer.ggml.tokens");
        const int n = (int) gguf_get_arr_n(g, id);
        model.tok_pieces.resize(n);
        for (int i = 0; i < n; ++i) {
            model.tok_pieces[i] = gguf_get_arr_str(g, id, i);
        }
        id = gguf_find_key(g, "tokenizer.ggml.scores");
        if (id < 0) return fail("missing tokenizer.ggml.scores");
        if ((int) gguf_get_arr_n(g, id) != n) return fail("tokenizer scores/tokens length mismatch");
        model.tok_scores.resize(n);
        std::memcpy(model.tok_scores.data(), gguf_get_arr_data(g, id), n * sizeof(float));
        id = gguf_find_key(g, "tokenizer.ggml.precompiled_charsmap");
        if (id >= 0) {
            const size_t nb = gguf_get_arr_n(g, id);
            model.tok_charsmap.resize(nb);
            std::memcpy(model.tok_charsmap.data(), gguf_get_arr_data(g, id), nb);
        }
        int v = 0;
        if (kv_u32(g, "tokenizer.ggml.unknown_token_id", v, nullptr)) model.tok_unk_id = v;
        if (kv_u32(g, "tokenizer.ggml.eos_token_id", v, nullptr)) model.tok_eos_id = v;
        bool b = true;
        if (kv_bool(g, "parler.tokenizer.add_eos", b, nullptr)) model.tok_add_eos = b;
    }

    // ---- optional BPE prompt tokenizer (indic-class checkpoints) ----
    if (gguf_find_key(g, "parler.prompt_tokenizer.model") >= 0) {
        int64_t id = gguf_find_key(g, "parler.prompt_tokenizer.tokens");
        if (id < 0) return fail("missing parler.prompt_tokenizer.tokens");
        const int n = (int) gguf_get_arr_n(g, id);
        model.ptok_pieces.resize(n);
        for (int i = 0; i < n; ++i) {
            model.ptok_pieces[i] = gguf_get_arr_str(g, id, i);
        }
        id = gguf_find_key(g, "parler.prompt_tokenizer.merges");
        if (id < 0) return fail("missing parler.prompt_tokenizer.merges");
        const int nm = (int) gguf_get_arr_n(g, id);
        model.ptok_merges.resize(nm);
        for (int i = 0; i < nm; ++i) {
            model.ptok_merges[i] = gguf_get_arr_str(g, id, i);
        }
        int v = 0;
        if (kv_u32(g, "parler.prompt_tokenizer.unknown_token_id", v, nullptr)) model.ptok_unk_id = v;
        if (kv_u32(g, "parler.prompt_tokenizer.bos_token_id", v, nullptr)) model.ptok_bos_id = v;
        bool b = true;
        if (kv_bool(g, "parler.prompt_tokenizer.add_bos", b, nullptr)) model.ptok_add_bos = b;
        model.has_prompt_tok = true;
    }

    // ---- weights: dup metadata tensors into ctx_w, alloc, stream ----
    const int64_t n_tensors = gguf_get_n_tensors(g);
    {
        const size_t ctx_size = (size_t)(n_tensors + 8) * ggml_tensor_overhead();
        ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/ true };
        model.ctx_w = ggml_init(ip);
        if (!model.ctx_w) return fail("ggml_init(ctx_w) failed");
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
        ggml_tensor * d = ggml_dup_tensor(model.ctx_w, t);
        ggml_set_name(d, ggml_get_name(t));
    }

    ::tts_cpp::detail::ensure_backends_loaded();
    model.backend = ::tts_cpp::detail::init_gpu_backend(n_gpu_layers, /*verbose=*/false, "parler");
    // Parler's GPU path (FA + fused weights + DAC phase-GEMM) is enabled only on
    // backends it has been validated against end-to-end (reference-fixture parity
    // per stage, plus greedy-token identity); anything else falls back to CPU.
    // Metal was validated in PR #103, Vulkan here. Feature-level probes such as
    // parler_probe_fa_f16() still apply on top, so a validated backend that lacks
    // an individual capability degrades rather than breaking.
    const bool backend_validated = model.backend &&
        (::tts_cpp::detail::backend_is_metal(model.backend) ||
         ::tts_cpp::detail::backend_is_vulkan(model.backend));
    if (model.backend && !backend_validated) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
    if (!model.backend) model.backend = ::tts_cpp::detail::init_cpu_backend();
    if (!model.backend) return fail("failed to init backend");

    model.on_gpu  = !::tts_cpp::detail::backend_is_cpu(model.backend);
    model.use_fa  = parler_probe_fa_f16(model.backend, hp.dec_d_model / hp.dec_n_head, hp.dec_n_head)
                    && std::getenv("PARLER_NO_FA") == nullptr;
    model.kv_type = model.use_fa ? GGML_TYPE_F16 : GGML_TYPE_F32;

    model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buffer_w) return fail("failed to allocate weight buffer");

    {
        ::tts_cpp::detail::gguf_stream_reader rd(g, path);
        if (!rd.ok()) return fail("failed to reopen GGUF for streaming");
        for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_w); t;
             t = ggml_get_next_tensor(model.ctx_w, t)) {
            if (!rd.to_backend(ggml_get_name(t), t)) {
                return fail(std::string("failed to stream tensor ") + ggml_get_name(t));
            }
        }
    }

    // ---- wire named tensors into the typed structs ----
    auto get = [&](const std::string & name, bool required = true) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(model.ctx_w, name.c_str());
        if (!t && required) {
            fprintf(stderr, "parler: missing tensor %s\n", name.c_str());
        }
        return t;
    };
    bool ok = true;
    auto req = [&](const std::string & name) {
        ggml_tensor * t = get(name);
        if (!t) ok = false;
        return t;
    };

    model.t5_embed       = req("t5.embed_tokens.weight");
    model.t5_rel_b       = req("t5.blk.0.attn_rel_b.weight");
    model.t5_output_norm = req("t5.output_norm.weight");
    model.t5_layers.resize(hp.t5_n_layer);
    for (int i = 0; i < hp.t5_n_layer; ++i) {
        auto & l = model.t5_layers[i];
        const std::string p = "t5.blk." + std::to_string(i) + ".";
        l.attn_norm = req(p + "attn_norm.weight");
        l.q = req(p + "attn_q.weight"); l.k = req(p + "attn_k.weight");
        l.v = req(p + "attn_v.weight"); l.o = req(p + "attn_o.weight");
        l.ffn_norm = req(p + "ffn_norm.weight");
        l.gate = req(p + "ffn_gate.weight");
        l.up   = req(p + "ffn_up.weight");
        l.down = req(p + "ffn_down.weight");
    }

    if (hp.enc_to_dec) {
        model.enc_to_dec_w = req("enc_to_dec.weight");
        model.enc_to_dec_b = req("enc_to_dec.bias");
    }
    model.embed_prompts   = req("dec.embed_prompts.weight");
    model.embed_positions = req("dec.embed_positions.weight");

    model.dec_embed.resize(hp.n_codebooks);
    model.lm_heads.resize(hp.n_codebooks);
    for (int k = 0; k < hp.n_codebooks; ++k) {
        model.dec_embed[k] = req("dec.embed_tokens." + std::to_string(k) + ".weight");
        model.lm_heads[k]  = req("dec.lm_heads." + std::to_string(k) + ".weight");
    }
    model.dec_output_norm_w = req("dec.output_norm.weight");
    model.dec_output_norm_b = req("dec.output_norm.bias");
    model.dec_layers.resize(hp.dec_n_layer);
    for (int i = 0; i < hp.dec_n_layer; ++i) {
        auto & l = model.dec_layers[i];
        const std::string p = "dec.blk." + std::to_string(i) + ".";
        l.attn_norm_w = req(p + "attn_norm.weight");
        l.attn_norm_b = req(p + "attn_norm.bias");
        l.q = req(p + "attn_q.weight"); l.k = req(p + "attn_k.weight");
        l.v = req(p + "attn_v.weight"); l.o = req(p + "attn_o.weight");
        l.cross_norm_w = req(p + "cross_norm.weight");
        l.cross_norm_b = req(p + "cross_norm.bias");
        l.cq = req(p + "cross_q.weight"); l.ck = req(p + "cross_k.weight");
        l.cv = req(p + "cross_v.weight"); l.co = req(p + "cross_o.weight");
        l.ffn_norm_w = req(p + "ffn_norm.weight");
        l.ffn_norm_b = req(p + "ffn_norm.bias");
        l.up   = req(p + "ffn_up.weight");
        l.down = req(p + "ffn_down.weight");
    }

    model.dac_quant.resize(hp.dac_n_q);
    for (int k = 0; k < hp.dac_n_q; ++k) {
        auto & q = model.dac_quant[k];
        const std::string p = "dac.quant." + std::to_string(k) + ".";
        q.codebook   = req(p + "codebook.weight");
        q.out_proj_w = req(p + "out_proj.weight");
        q.out_proj_b = req(p + "out_proj.bias");
    }
    model.dac_conv_in_w = req("dac.dec.conv_in.weight");
    model.dac_conv_in_b = req("dac.dec.conv_in.bias");
    model.dac_blocks.resize(hp.dac_rates.size());
    for (size_t i = 0; i < hp.dac_rates.size(); ++i) {
        auto & b = model.dac_blocks[i];
        const std::string p = "dac.dec.blk." + std::to_string(i) + ".";
        b.stride      = hp.dac_rates[i];
        b.snake_alpha = req(p + "snake.alpha");
        b.convt_w     = req(p + "convt.weight");
        b.convt_b     = req(p + "convt.bias");
        for (int j = 0; j < 3; ++j) {
            auto & r = b.res[j];
            const std::string rp = p + "res." + std::to_string(j) + ".";
            r.snake1_alpha = req(rp + "snake1.alpha");
            r.conv1_w = req(rp + "conv1.weight"); r.conv1_b = req(rp + "conv1.bias");
            r.snake2_alpha = req(rp + "snake2.alpha");
            r.conv2_w = req(rp + "conv2.weight"); r.conv2_b = req(rp + "conv2.bias");
        }
    }
    model.dac_snake_out_alpha = req("dac.dec.snake_out.alpha");
    model.dac_conv_out_w = req("dac.dec.conv_out.weight");
    model.dac_conv_out_b = req("dac.dec.conv_out.bias");

    if (!ok) return fail("one or more expected tensors missing (see log)");

    // shape spot-checks against hparams (catches converter/loader drift)
    if (model.t5_embed->ne[0] != hp.t5_d_model || model.t5_embed->ne[1] != hp.t5_vocab) {
        return fail("t5.embed_tokens shape mismatch");
    }
    if (model.lm_heads[0]->ne[0] != hp.dec_d_model || model.lm_heads[0]->ne[1] != hp.dec_vocab) {
        return fail("dec.lm_heads shape mismatch");
    }
    if (model.dec_embed[0]->ne[0] != hp.dec_d_model || model.dec_embed[0]->ne[1] < hp.dec_vocab) {
        return fail("dec.embed_tokens shape mismatch");
    }
    if (model.embed_positions->ne[0] != hp.dec_d_model ||
        model.embed_positions->ne[1] != hp.max_position) {
        return fail("dec.embed_positions shape mismatch");
    }
    if (model.embed_prompts->ne[0] != hp.dec_d_model ||
        model.embed_prompts->ne[1] < (int64_t) (model.has_prompt_tok ? model.ptok_pieces.size()
                                                                     : model.tok_pieces.size())) {
        return fail("dec.embed_prompts smaller than the prompt tokenizer vocab");
    }

    // ---- decoder self-KV cache slabs ----
    {
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
        model.ctx_kv = ggml_init(ip);
        if (!model.ctx_kv) return fail("ggml_init(ctx_kv) failed");
        const int64_t rows = (int64_t) hp.n_ctx * hp.dec_n_layer;
        model.memory_k = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, hp.dec_d_model, rows);
        model.memory_v = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, hp.dec_d_model, rows);
        ggml_set_name(model.memory_k, "parler_kv_k");
        ggml_set_name(model.memory_v, "parler_kv_v");
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
        if (!model.buffer_kv) return fail("failed to allocate KV buffer");
    }

    // ---- GPU: fused weights (fewer N=1 decode dispatches; byte-exact row concat) ----
    // qkv[l] = q|k|v stacked on the output dim -> one mul_mat/layer; lm_head_stacked
    // = the n_codebooks heads stacked -> one mul_mat/step. Gated on GPU; the CPU path
    // keeps the separate weights so the reference parity tests stay byte-identical.
    if (model.on_gpu) {
        const int nl = hp.dec_n_layer, nq = hp.n_codebooks;
        const int64_t d = hp.dec_d_model, vocab = model.lm_heads[0]->ne[1];
        bool fuse_qkv = true;
        for (int i = 0; i < nl; ++i) {
            const auto & l = model.dec_layers[i];
            if (l.q->type != l.k->type || l.q->type != l.v->type ||
                l.q->ne[0] != d || l.q->ne[1] != d ||
                l.k->ne[0] != d || l.k->ne[1] != d ||
                l.v->ne[0] != d || l.v->ne[1] != d) { fuse_qkv = false; break; }
        }
        bool fuse_heads = true;
        for (int k = 0; k < nq; ++k) {
            if (model.lm_heads[k]->type != model.lm_heads[0]->type ||
                model.lm_heads[k]->ne[0] != d || model.lm_heads[k]->ne[1] != vocab) { fuse_heads = false; break; }
        }
        // Skip the fused context entirely when nothing fuses: an empty ctx makes
        // ggml_backend_alloc_ctx_tensors return NULL (0 buffers) and abort the load.
        if (fuse_qkv || fuse_heads) {
            ggml_init_params ip = { (size_t)(nl + 2) * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
            model.ctx_fused = ggml_init(ip);
            if (!model.ctx_fused) return fail("ggml_init(ctx_fused) failed");
            std::vector<ggml_tensor *> qkv(nl, nullptr);
            if (fuse_qkv) {
                for (int i = 0; i < nl; ++i)
                    qkv[i] = ggml_new_tensor_2d(model.ctx_fused, model.dec_layers[i].q->type, d, 3 * d);
            }
            ggml_tensor * heads = fuse_heads
                ? ggml_new_tensor_2d(model.ctx_fused, model.lm_heads[0]->type, d, vocab * nq) : nullptr;
            model.buffer_fused = ggml_backend_alloc_ctx_tensors(model.ctx_fused, model.backend);
            if (!model.buffer_fused) return fail("failed to allocate fused-weight buffer");
            std::vector<uint8_t> tmp;
            auto copy_into = [&](ggml_tensor * dst, size_t off, ggml_tensor * src) {
                tmp.resize(ggml_nbytes(src));
                ggml_backend_tensor_get(src, tmp.data(), 0, tmp.size());
                ggml_backend_tensor_set(dst, tmp.data(), off, tmp.size());
            };
            if (fuse_qkv) {
                for (int i = 0; i < nl; ++i) {
                    auto & l = model.dec_layers[i];
                    const size_t nb = ggml_nbytes(l.q);
                    copy_into(qkv[i], 0, l.q); copy_into(qkv[i], nb, l.k); copy_into(qkv[i], 2 * nb, l.v);
                    l.qkv = qkv[i];
                }
            }
            if (fuse_heads) {
                const size_t nb = ggml_nbytes(model.lm_heads[0]);
                for (int k = 0; k < nq; ++k) copy_into(heads, (size_t) k * nb, model.lm_heads[k]);
                model.lm_head_stacked = heads;
            }
        }
    }

    gguf_free(g);
    ggml_free(ctx_meta);
    return true;
}

void parler_free_model(parler_model & model) {
    ::tts_cpp::detail::sched_fallback_free(model.sched_fb);
    if (model.dac_allocr)   { ggml_gallocr_free(model.dac_allocr); model.dac_allocr = nullptr; }
    if (model.buffer_cross) { ggml_backend_buffer_free(model.buffer_cross); model.buffer_cross = nullptr; }
    if (model.ctx_cross)    { ggml_free(model.ctx_cross); model.ctx_cross = nullptr; }
    if (model.buffer_fused) { ggml_backend_buffer_free(model.buffer_fused); model.buffer_fused = nullptr; }
    if (model.ctx_fused)    { ggml_free(model.ctx_fused); model.ctx_fused = nullptr; }
    if (model.buffer_kv)    { ggml_backend_buffer_free(model.buffer_kv); model.buffer_kv = nullptr; }
    if (model.ctx_kv)       { ggml_free(model.ctx_kv); model.ctx_kv = nullptr; }
    if (model.buffer_w)     { ggml_backend_buffer_free(model.buffer_w); model.buffer_w = nullptr; }
    if (model.ctx_w)        { ggml_free(model.ctx_w); model.ctx_w = nullptr; }
    if (model.backend)      { ggml_backend_free(model.backend); model.backend = nullptr; }
    model.cross_k.clear();
    model.cross_v_t.clear();
    model.cross_len = 0;
}

bool parler_graph_prepare(const parler_model & model, ggml_cgraph * gf,
                          ggml_gallocr_t allocr, bool & use_sched, const char * caller) {
    use_sched = ::tts_cpp::detail::sched_force_enabled() ||
                !::tts_cpp::detail::graph_fully_supported(model.backend, gf);
    if (!use_sched) {
        if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "%s: gallocr alloc failed\n", caller);
            return false;
        }
        return true;
    }
    if (::tts_cpp::detail::graph_has_unsupported_preallocated_op(model.backend, gf)) {
        fprintf(stderr, "%s: op writing a persistent buffer is unsupported; "
                        "sched fallback impossible\n", caller);
        return false;
    }
    if (!::tts_cpp::detail::sched_fallback_ensure(model.sched_fb, model.backend,
            /*graph_size=*/ 2 * PARLER_MAX_NODES,
            {model.buffer_w, model.buffer_kv, model.buffer_cross})) {
        fprintf(stderr, "%s: scheduler creation failed\n", caller);
        return false;
    }
    if (!::tts_cpp::detail::sched_fallback_alloc(model.sched_fb, gf)) {
        fprintf(stderr, "%s: sched graph alloc failed\n", caller);
        return false;
    }
    return true;
}

bool parler_graph_compute(const parler_model & model, ggml_cgraph * gf,
                          bool use_sched, int n_threads, const char * caller) {
    const ggml_status status = use_sched
        ? ::tts_cpp::detail::sched_fallback_compute(model.sched_fb, model.backend, gf, n_threads)
        : ::tts_cpp::detail::direct_compute(model.backend, gf, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed (ggml_status=%d)\n", caller, (int) status);
        return false;
    }
    return true;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
