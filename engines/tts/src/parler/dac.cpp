#include "internal.h"

#include "ggml-alloc.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

// F32 conv1d via im2col + mul_mat (ggml_conv_1d's F16 im2col loses too much
// precision over the 26-conv DAC stack).  kernel ne=[K, IC, OC].
ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                         int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

// ggml_conv_transpose_1d asserts p0 == 0, so run unpadded and trim `padding`
// elements off each end.  kernel ne=[K, OC, IC], input ne=[L, IC, 1].
ggml_tensor * conv_transpose_1d_trim(ggml_context * ctx, ggml_tensor * kernel,
                                     ggml_tensor * input, int stride, int padding) {
    ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, stride, 0, 1);
    if (padding == 0) return out;
    const int64_t l_new = out->ne[0] - 2 * padding;
    ggml_tensor * v = ggml_view_3d(ctx, out, l_new, out->ne[1], out->ne[2],
                                   out->nb[1], out->nb[2], (size_t) padding * out->nb[0]);
    return ggml_cont(ctx, v);
}

// bias ne=[C] broadcast-added over the length dim of x ne=[L, C, 1]
ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    return ggml_add(ctx, x, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
}

// snake(x, alpha) = x + (alpha + 1e-9)^-1 * sin(alpha*x)^2; alpha ne=[1, C, 1]
ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * eps) {
    ggml_tensor * s2 = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, x, alpha)));
    return ggml_add(ctx, x, ggml_div(ctx, s2, ggml_add(ctx, alpha, eps)));
}

} // namespace

bool parler_dac_decode(const parler_model & model, const int32_t * codes, int n_frames,
                       int n_threads, std::vector<float> & pcm_out,
                       std::vector<float> * latent_out) {
    const parler_hparams & hp = model.hparams;
    const int n_q = hp.dac_n_q;

    if (n_frames <= 0) {
        fprintf(stderr, "%s: invalid n_frames=%d\n", __func__, n_frames);
        return false;
    }
    for (int64_t i = 0; i < (int64_t) n_q * n_frames; ++i) {
        if (codes[i] < 0 || codes[i] >= hp.dac_codebook_size) {
            fprintf(stderr, "%s: code %d at index %lld out of range [0, %d)\n",
                    __func__, codes[i], (long long) i, hp.dac_codebook_size);
            return false;
        }
    }

    const size_t ctx_size = ggml_tensor_overhead() * PARLER_MAX_NODES +
                            ggml_graph_overhead_custom(PARLER_MAX_NODES, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init failed\n", __func__);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, PARLER_MAX_NODES, false);

    ggml_tensor * codes_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_frames, n_q);
    ggml_set_name(codes_t, "dac_codes");
    ggml_set_input(codes_t);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(eps, "snake_eps");
    ggml_set_input(eps);

    // RVQ from_codes: latent = sum_k out_proj_k(codebook_k[codes_k])
    ggml_tensor * latent = nullptr;
    for (int k = 0; k < n_q; ++k) {
        const parler_dac_quant & q = model.dac_quant[k];
        ggml_tensor * ids = ggml_view_1d(ctx, codes_t, n_frames, (size_t) k * codes_t->nb[1]);
        ggml_tensor * z   = ggml_get_rows(ctx, q.codebook, ids);                   // [8, T]
        ggml_tensor * w2  = ggml_reshape_2d(ctx, q.out_proj_w,
                                            q.out_proj_w->ne[1], q.out_proj_w->ne[2]);
        ggml_tensor * p   = ggml_add(ctx, ggml_mul_mat(ctx, w2, z), q.out_proj_b); // [1024, T]
        latent = latent ? ggml_add(ctx, latent, p) : p;
    }
    ggml_set_name(latent, "latent");
    ggml_set_output(latent);

    // [latent_dim, T] -> [T, latent_dim, 1] for the conv stack
    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, latent));
    x = ggml_reshape_3d(ctx, x, n_frames, hp.dac_latent, 1);

    x = add_bias(ctx, conv1d_f32(ctx, model.dac_conv_in_w, x, 1, 3, 1), model.dac_conv_in_b);

    static const int dilations[3] = { 1, 3, 9 };
    for (const parler_dac_block & blk : model.dac_blocks) {
        const int s = blk.stride;
        x = snake(ctx, x, blk.snake_alpha, eps);
        const int64_t t_in = x->ne[0];
        x = conv_transpose_1d_trim(ctx, blk.convt_w, x, s, s / 2);
        GGML_ASSERT(x->ne[0] == t_in * s);
        x = add_bias(ctx, x, blk.convt_b);
        for (int j = 0; j < 3; ++j) {
            const parler_dac_residual & r = blk.res[j];
            const int d = dilations[j];
            ggml_tensor * y = snake(ctx, x, r.snake1_alpha, eps);
            y = add_bias(ctx, conv1d_f32(ctx, r.conv1_w, y, 1, 3 * d, d), r.conv1_b);
            y = snake(ctx, y, r.snake2_alpha, eps);
            y = add_bias(ctx, conv1d_f32(ctx, r.conv2_w, y, 1, 0, 1), r.conv2_b);
            x = ggml_add(ctx, x, y);
        }
    }

    x = snake(ctx, x, model.dac_snake_out_alpha, eps);
    x = add_bias(ctx, conv1d_f32(ctx, model.dac_conv_out_w, x, 1, 3, 1), model.dac_conv_out_b);
    ggml_tensor * wav = ggml_tanh(ctx, x);
    ggml_set_name(wav, "wav");
    ggml_set_output(wav);
    ggml_build_forward_expand(gf, wav);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) {
        fprintf(stderr, "%s: gallocr creation failed\n", __func__);
        ggml_free(ctx);
        return false;
    }

    bool ok = false;
    bool use_sched = false;
    do {
        if (!parler_graph_prepare(model, gf, allocr, use_sched, __func__)) break;

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dac_codes"), codes, 0,
                                (size_t) n_q * n_frames * sizeof(int32_t));
        const float eps_val = 1e-9f;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "snake_eps"), &eps_val, 0, sizeof(eps_val));

        if (!parler_graph_compute(model, gf, use_sched, n_threads, __func__)) break;

        ggml_tensor * wav_t = ggml_graph_get_tensor(gf, "wav");
        pcm_out.resize(ggml_nelements(wav_t));
        ggml_backend_tensor_get(wav_t, pcm_out.data(), 0, ggml_nbytes(wav_t));
        if (latent_out) {
            ggml_tensor * lat_t = ggml_graph_get_tensor(gf, "latent");
            latent_out->resize(ggml_nelements(lat_t));
            ggml_backend_tensor_get(lat_t, latent_out->data(), 0, ggml_nbytes(lat_t));
        }
        ok = true;
    } while (false);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return ok;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
