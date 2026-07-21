// Parler-TTS decoder LM graphs: prefill (prompt embeds prepended to the BOS
// start frame) and single-step decode, with a token-major self-KV slab
// (layout mirrors the T3 cache in main.cpp) and cross-attention against the
// per-description precomputed cross_k / cross_v_t tensors.
//
// HF semantics honoured: 9 codebook embeddings summed; sinusoidal positional
// table rows added over the combined (prompt + audio) sequence; pre-norm
// LayerNorm (weight + bias, eps from GGUF); self-attn q scaled by
// head_dim^-0.5; FFN uses exact-erf GELU ("gelu"); logits = 9 separate heads.

#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w,
                         ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, x, w), b);
}

// self+cross+ffn stack over inpL = [d_model, N] at cache position n_past.
// Returns the final-layer-norm output [d_model, N].
ggml_tensor * build_dec_core(ggml_context * ctx, ggml_cgraph * gf,
                             const parler_model & model,
                             ggml_tensor * inpL, int n_past, int N,
                             ggml_tensor * kq_mask) {
    const parler_hparams & hp = model.hparams;
    const int d  = hp.dec_d_model;
    const int H  = hp.dec_n_head;
    const int HD = d / H;
    const int L  = n_past + N;
    const int Tc = model.cross_len;

    const size_t kv_tok_row  = ggml_row_size(GGML_TYPE_F32, d);
    const size_t kv_head_row = ggml_row_size(GGML_TYPE_F32, HD);
    const size_t kv_layer    = (size_t) hp.n_ctx * kv_tok_row;
    const float  scale       = 1.0f / std::sqrt((float) HD);

    for (int il = 0; il < hp.dec_n_layer; ++il) {
        const parler_dec_layer & l = model.dec_layers[il];

        // ---- self attention ----
        ggml_tensor * cur = layer_norm(ctx, inpL, l.attn_norm_w, l.attn_norm_b, hp.dec_ln_eps);
        ggml_tensor * q = ggml_mul_mat(ctx, l.q, cur); // [d, N]
        ggml_tensor * k = ggml_mul_mat(ctx, l.k, cur);
        ggml_tensor * v = ggml_mul_mat(ctx, l.v, cur);

        const size_t layer_off = (size_t) il * kv_layer;
        {
            ggml_tensor * k_dst = ggml_view_2d(ctx, model.memory_k, d, N, kv_tok_row,
                                               layer_off + (size_t) n_past * kv_tok_row);
            ggml_tensor * v_dst = ggml_view_2d(ctx, model.memory_v, d, N, kv_tok_row,
                                               layer_off + (size_t) n_past * kv_tok_row);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_dst));
        }

        ggml_tensor * q3 = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, q, HD, H, N), 0, 2, 1, 3));            // [HD, N, H]
        ggml_tensor * K = ggml_cont(ctx, ggml_view_3d(ctx, model.memory_k,
            HD, L, H, kv_tok_row, kv_head_row, layer_off));             // [HD, L, H]
        ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx,
            ggml_view_3d(ctx, model.memory_v, HD, L, H, kv_tok_row, kv_head_row, layer_off),
            1, 0, 2, 3));                                               // [L, HD, H]

        ggml_tensor * kq = ggml_mul_mat(ctx, K, q3);                    // [L, N, H]
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, Vt, kq);                  // [HD, N, H]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));       // [HD, H, N]
        kqv = ggml_reshape_2d(ctx, kqv, d, N);
        inpL = ggml_add(ctx, inpL, ggml_mul_mat(ctx, l.o, kqv));

        // ---- cross attention (description; no mask, unpadded) ----
        cur = layer_norm(ctx, inpL, l.cross_norm_w, l.cross_norm_b, hp.dec_ln_eps);
        ggml_tensor * cq3 = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.cq, cur), HD, H, N), 0, 2, 1, 3));
        ggml_tensor * cK = ggml_cont(ctx, ggml_view_3d(ctx, model.cross_k[il],
            HD, Tc, H, model.cross_k[il]->nb[1], kv_head_row, 0));      // [HD, Tc, H]
        ggml_tensor * cVt = ggml_view_3d(ctx, model.cross_v_t[il],
            Tc, HD, H,
            (size_t) Tc * sizeof(float),
            (size_t) Tc * HD * sizeof(float), 0);                       // [Tc, HD, H]

        ggml_tensor * ckq = ggml_mul_mat(ctx, cK, cq3);                 // [Tc, N, H]
        ckq = ggml_soft_max_ext(ctx, ckq, nullptr, scale, 0.0f);
        ggml_tensor * ckqv = ggml_mul_mat(ctx, cVt, ckq);               // [HD, N, H]
        ckqv = ggml_cont(ctx, ggml_permute(ctx, ckqv, 0, 2, 1, 3));
        ckqv = ggml_reshape_2d(ctx, ckqv, d, N);
        inpL = ggml_add(ctx, inpL, ggml_mul_mat(ctx, l.co, ckqv));

        // ---- ffn (exact-erf gelu) ----
        cur = layer_norm(ctx, inpL, l.ffn_norm_w, l.ffn_norm_b, hp.dec_ln_eps);
        cur = ggml_gelu_erf(ctx, ggml_mul_mat(ctx, l.up, cur));
        cur = ggml_mul_mat(ctx, l.down, cur);
        inpL = ggml_add(ctx, inpL, cur);
    }

    inpL = layer_norm(ctx, inpL, model.dec_output_norm_w, model.dec_output_norm_b, hp.dec_ln_eps);
    ggml_set_name(inpL, "hidden");
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);
    return inpL;
}

// stacked per-codebook logits for the LAST position: [vocab, n_codebooks]
void build_dec_heads(ggml_context * ctx, ggml_cgraph * gf,
                     const parler_model & model, ggml_tensor * hidden, int N) {
    const parler_hparams & hp = model.hparams;
    ggml_tensor * last = ggml_view_2d(ctx, hidden, hp.dec_d_model, 1, hidden->nb[1],
                                      (size_t) (N - 1) * hidden->nb[1]);
    last = ggml_cont(ctx, last);
    ggml_tensor * logits = nullptr;
    for (int k = 0; k < hp.n_codebooks; ++k) {
        ggml_tensor * lk = ggml_mul_mat(ctx, model.lm_heads[k], last); // [vocab, 1]
        logits = logits ? ggml_concat(ctx, logits, lk, 1) : lk;
    }
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
}

// summed codebook embeddings for one frame of n_codebooks ids: [d_model, 1]
ggml_tensor * build_frame_embed(ggml_context * ctx, const parler_model & model,
                                ggml_tensor * frame_ids) {
    ggml_tensor * fe = nullptr;
    for (int k = 0; k < model.hparams.n_codebooks; ++k) {
        ggml_tensor * idk = ggml_view_1d(ctx, frame_ids, 1, (size_t) k * sizeof(int32_t));
        ggml_tensor * ek  = ggml_get_rows(ctx, model.dec_embed[k], idk);
        fe = fe ? ggml_add(ctx, fe, ek) : ek;
    }
    return fe;
}

ggml_cgraph * new_parler_graph(ggml_context ** ctx_out) {
    static const size_t buf_size = ggml_tensor_overhead() * PARLER_MAX_NODES +
                                   ggml_graph_overhead_custom(PARLER_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params ip = { buf_size, buf.data(), /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    *ctx_out = ctx;
    return ggml_new_graph_custom(ctx, PARLER_MAX_NODES, false);
}

bool read_logits(ggml_cgraph * gf, const parler_model & model,
                 std::vector<float> & logits_out) {
    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) return false;
    const size_t n = (size_t) model.hparams.dec_vocab * model.hparams.n_codebooks;
    logits_out.resize(n);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, n * sizeof(float));
    return true;
}

} // namespace

bool parler_dec_prefill(const parler_model & model,
                        const std::vector<int32_t> & prompt_ids,
                        const std::vector<int32_t> & start_frame,
                        ggml_gallocr_t allocr, int n_threads,
                        std::vector<float> & logits_out, int & n_past_out) {
    const parler_hparams & hp = model.hparams;
    const int P = (int) prompt_ids.size();
    const int N = P + 1;
    if ((int) start_frame.size() != hp.n_codebooks) {
        fprintf(stderr, "%s: start frame must have %d ids\n", __func__, hp.n_codebooks);
        return false;
    }
    if (N + hp.gen_max_length > hp.n_ctx) {
        fprintf(stderr, "%s: prompt too long (%d tokens; capacity %d)\n",
                __func__, P, hp.n_ctx - hp.gen_max_length - 1);
        return false;
    }
    if (model.cross_len <= 0) {
        fprintf(stderr, "%s: description not encoded\n", __func__);
        return false;
    }

    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = new_parler_graph(&ctx);

    ggml_tensor * pids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, std::max(P, 1));
    ggml_set_name(pids, "prompt_ids"); ggml_set_input(pids);
    ggml_tensor * fids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_codebooks);
    ggml_set_name(fids, "frame_ids"); ggml_set_input(fids);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(pos, "positions"); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    ggml_set_name(mask, "kq_mask"); ggml_set_input(mask);

    ggml_tensor * fe = build_frame_embed(ctx, model, fids);            // [d, 1]
    ggml_tensor * inp;
    if (P > 0) {
        ggml_tensor * pe = ggml_get_rows(ctx, model.embed_prompts, pids); // [d, P]
        inp = ggml_concat(ctx, pe, fe, 1);
    } else {
        inp = fe;
    }
    inp = ggml_add(ctx, inp, ggml_get_rows(ctx, model.embed_positions, pos));

    ggml_tensor * hidden = build_dec_core(ctx, gf, model, inp, /*n_past=*/ 0, N, mask);
    build_dec_heads(ctx, gf, model, hidden, N);
    ggml_free(ctx);

    bool use_sched = false;
    if (!parler_graph_prepare(model, gf, allocr, use_sched, __func__)) return false;

    if (P > 0) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "prompt_ids"),
                                prompt_ids.data(), 0, (size_t) P * sizeof(int32_t));
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "frame_ids"),
                            start_frame.data(), 0, start_frame.size() * sizeof(int32_t));
    std::vector<int32_t> positions(N);
    for (int i = 0; i < N; ++i) positions[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"),
                            positions.data(), 0, positions.size() * sizeof(int32_t));
    std::vector<float> m((size_t) N * N, 0.0f);
    for (int q = 0; q < N; ++q) {
        for (int k = q + 1; k < N; ++k) m[(size_t) q * N + k] = -INFINITY;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "kq_mask"),
                            m.data(), 0, m.size() * sizeof(float));

    if (!parler_graph_compute(model, gf, use_sched, n_threads, __func__)) return false;
    if (!read_logits(gf, model, logits_out)) return false;
    n_past_out = N;
    return true;
}

bool parler_dec_step(const parler_model & model,
                     const std::vector<int32_t> & frame,
                     int n_past,
                     ggml_gallocr_t allocr, int n_threads,
                     std::vector<float> & logits_out) {
    const parler_hparams & hp = model.hparams;
    if ((int) frame.size() != hp.n_codebooks) return false;
    if (n_past + 1 > hp.n_ctx || n_past + 1 > hp.max_position) {
        fprintf(stderr, "%s: cache full (n_past=%d)\n", __func__, n_past);
        return false;
    }

    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = new_parler_graph(&ctx);

    ggml_tensor * fids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_codebooks);
    ggml_set_name(fids, "frame_ids"); ggml_set_input(fids);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos, "positions"); ggml_set_input(pos);

    ggml_tensor * inp = build_frame_embed(ctx, model, fids);
    inp = ggml_add(ctx, inp, ggml_get_rows(ctx, model.embed_positions, pos));

    ggml_tensor * hidden = build_dec_core(ctx, gf, model, inp, n_past, 1, /*mask=*/ nullptr);
    build_dec_heads(ctx, gf, model, hidden, 1);
    ggml_free(ctx);

    bool use_sched = false;
    if (!parler_graph_prepare(model, gf, allocr, use_sched, __func__)) return false;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "frame_ids"),
                            frame.data(), 0, frame.size() * sizeof(int32_t));
    const int32_t p = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), &p, 0, sizeof(p));

    if (!parler_graph_compute(model, gf, use_sched, n_threads, __func__)) return false;
    return read_logits(gf, model, logits_out);
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
