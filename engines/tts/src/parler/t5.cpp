// Parler-TTS T5 text encoder + cross-attention K/V precompute.
//
// T5 specifics honoured here: RMSNorm (eps from GGUF), NO attention scaling
// (scale = 1.0), relative-position bias added pre-softmax as a per-head mask,
// gated-GELU FFN (gelu_new = tanh approx -> ggml_gelu). The optional
// enc_to_dec projection exists only when t5.d_model != dec.d_model (large);
// mini's checkpoint does not even ship it.

#include "internal.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

// HF T5 _relative_position_bucket, bidirectional=True.
int t5_rel_bucket(int relative_position, int num_buckets, int max_distance) {
    int ret = 0;
    int nb = num_buckets / 2;
    if (relative_position > 0) {
        ret += nb;
    }
    int rp = relative_position < 0 ? -relative_position : relative_position;
    const int max_exact = nb / 2;
    if (rp < max_exact) {
        ret += rp;
    } else {
        int large = max_exact +
            (int) (std::log((float) rp / max_exact) / std::log((float) max_distance / max_exact) *
                   (nb - max_exact));
        if (large > nb - 1) large = nb - 1;
        ret += large;
    }
    return ret;
}

struct t5_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
};

t5_graph build_t5_graph(const parler_model & model, int T) {
    const parler_hparams & hp = model.hparams;
    const int inner = hp.t5_n_head * hp.t5_d_kv;
    const int HD = hp.t5_d_kv;
    const int H  = hp.t5_n_head;

    static const size_t buf_size = ggml_tensor_overhead() * PARLER_MAX_NODES +
                                   ggml_graph_overhead_custom(PARLER_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params ip = { buf_size, buf.data(), /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, PARLER_MAX_NODES, false);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(ids, "desc_ids"); ggml_set_input(ids);
    ggml_tensor * buckets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) T * T);
    ggml_set_name(buckets, "rel_buckets"); ggml_set_input(buckets);

    // position bias: gather [H, T*T] from rel_b, reshape to [H, Tk, Tq],
    // permute to [Tk, Tq, H] — used as the soft_max_ext mask with scale 1.0.
    ggml_tensor * bias = ggml_get_rows(ctx, model.t5_rel_b, buckets);
    bias = ggml_reshape_3d(ctx, bias, H, T, T);
    bias = ggml_cont(ctx, ggml_permute(ctx, bias, 2, 0, 1, 3));

    ggml_tensor * x = ggml_get_rows(ctx, model.t5_embed, ids); // [d_model, T]

    for (int il = 0; il < hp.t5_n_layer; ++il) {
        const parler_t5_layer & l = model.t5_layers[il];

        ggml_tensor * cur = ggml_rms_norm(ctx, x, hp.t5_rms_eps);
        cur = ggml_mul(ctx, cur, l.attn_norm);

        ggml_tensor * q = ggml_mul_mat(ctx, l.q, cur); // [inner, T]
        ggml_tensor * k = ggml_mul_mat(ctx, l.k, cur);
        ggml_tensor * v = ggml_mul_mat(ctx, l.v, cur);

        ggml_tensor * q3 = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, q, HD, H, T), 0, 2, 1, 3));          // [HD, T, H]
        ggml_tensor * k3 = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, k, HD, H, T), 0, 2, 1, 3));          // [HD, T, H]
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, v, HD, H, T), 1, 2, 0, 3));          // [T, HD, H]

        ggml_tensor * kq = ggml_mul_mat(ctx, k3, q3);                 // [Tk, Tq, H]
        kq = ggml_soft_max_ext(ctx, kq, bias, 1.0f, 0.0f);            // T5: no 1/sqrt(d)
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);                // [HD, Tq, H]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));     // [HD, H, Tq]
        kqv = ggml_reshape_2d(ctx, kqv, inner, T);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, l.o, kqv));

        cur = ggml_rms_norm(ctx, x, hp.t5_rms_eps);
        cur = ggml_mul(ctx, cur, l.ffn_norm);
        ggml_tensor * gate = ggml_gelu(ctx, ggml_mul_mat(ctx, l.gate, cur));
        ggml_tensor * up   = ggml_mul_mat(ctx, l.up, cur);
        cur = ggml_mul_mat(ctx, l.down, ggml_mul(ctx, gate, up));
        x = ggml_add(ctx, x, cur);
    }

    x = ggml_rms_norm(ctx, x, hp.t5_rms_eps);
    x = ggml_mul(ctx, x, model.t5_output_norm);

    if (hp.enc_to_dec) {
        x = ggml_mul_mat(ctx, model.enc_to_dec_w, x);
        x = ggml_add(ctx, x, model.enc_to_dec_b);
    }
    ggml_set_name(x, "cross_states"); // [dec_d_model, T]
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // per-decoder-layer cross K / V-transposed, written into the persistent
    // cross buffers (model.cross_k / model.cross_v_t must be allocated for T).
    for (int il = 0; il < hp.dec_n_layer; ++il) {
        const parler_dec_layer & dl = model.dec_layers[il];
        ggml_tensor * ck = ggml_mul_mat(ctx, dl.ck, x);               // [dec_d, T]
        ggml_tensor * cv = ggml_mul_mat(ctx, dl.cv, x);               // [dec_d, T]
        ggml_tensor * cvt = ggml_cont(ctx, ggml_transpose(ctx, cv));  // [T, dec_d]
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ck,  model.cross_k[il]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, cvt, model.cross_v_t[il]));
    }

    ggml_free(ctx);
    return { nullptr, gf };
}

} // namespace

bool parler_encode_description(parler_model & model,
                               const std::vector<int32_t> & desc_ids,
                               int n_threads,
                               std::vector<float> * states_out) {
    const parler_hparams & hp = model.hparams;
    const int T = (int) desc_ids.size();
    if (T <= 0) {
        fprintf(stderr, "%s: empty description\n", __func__);
        return false;
    }

    // (re)allocate the persistent cross-KV tensors for this description length
    if (model.buffer_cross) { ggml_backend_buffer_free(model.buffer_cross); model.buffer_cross = nullptr; }
    if (model.ctx_cross)    { ggml_free(model.ctx_cross); model.ctx_cross = nullptr; }
    model.cross_len = 0; // previous cross-KV is gone; any early return below must not leave it usable
    {
        ggml_init_params ip = { (size_t)(2 * hp.dec_n_layer + 2) * ggml_tensor_overhead(),
                                nullptr, /*no_alloc=*/ true };
        model.ctx_cross = ggml_init(ip);
        if (!model.ctx_cross) return false;
        model.cross_k.assign(hp.dec_n_layer, nullptr);
        model.cross_v_t.assign(hp.dec_n_layer, nullptr);
        for (int il = 0; il < hp.dec_n_layer; ++il) {
            model.cross_k[il]   = ggml_new_tensor_2d(model.ctx_cross, GGML_TYPE_F32, hp.dec_d_model, T);
            model.cross_v_t[il] = ggml_new_tensor_2d(model.ctx_cross, GGML_TYPE_F32, T, hp.dec_d_model);
        }
        model.buffer_cross = ggml_backend_alloc_ctx_tensors(model.ctx_cross, model.backend);
        if (!model.buffer_cross) {
            fprintf(stderr, "%s: cross buffer alloc failed\n", __func__);
            return false;
        }
        model.cross_len = T;
    }

    t5_graph tg = build_t5_graph(model, T);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) return false;

    bool ok = false;
    bool use_sched = false;
    do {
        if (!parler_graph_prepare(model, tg.gf, allocr, use_sched, __func__)) break;

        ggml_backend_tensor_set(ggml_graph_get_tensor(tg.gf, "desc_ids"),
                                desc_ids.data(), 0, (size_t) T * sizeof(int32_t));
        std::vector<int32_t> buckets((size_t) T * T);
        for (int q = 0; q < T; ++q) {
            for (int k = 0; k < T; ++k) {
                buckets[(size_t) q * T + k] =
                    t5_rel_bucket(k - q, hp.t5_rel_buckets, hp.t5_rel_max_dist);
            }
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(tg.gf, "rel_buckets"),
                                buckets.data(), 0, buckets.size() * sizeof(int32_t));

        if (!parler_graph_compute(model, tg.gf, use_sched, n_threads, __func__)) break;

        if (states_out) {
            ggml_tensor * st = ggml_graph_get_tensor(tg.gf, "cross_states");
            states_out->resize((size_t) hp.dec_d_model * T);
            ggml_backend_tensor_get(st, states_out->data(), 0,
                                    states_out->size() * sizeof(float));
        }
        ok = true;
    } while (false);

    ggml_gallocr_free(allocr);
    if (!ok) model.cross_len = 0;
    return ok;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
