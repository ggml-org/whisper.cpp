#include "internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

// residual dilations, shared by the graph builder and the receptive-field derivation
const int dilations[3] = { 1, 3, 9 };

// F32 conv1d via im2col + mul_mat (ggml_conv_1d's F16 im2col loses too much
// precision over the 26-conv DAC stack).  kernel ne=[K, IC, OC].
//
// Keeping im2col in F32 only helps if the matmul that consumes it stays in F32 too, so
// the contraction asks for it explicitly: backends are free to multiply f32 operands in
// fp16 for GGML_PREC_DEFAULT, and over 26 convolutions that dominates the output error.
ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                         int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
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

// Transposed conv1d (K=2*stride) as two phase-GEMMs + shifted overlap-add on the
// fast matmul path. kernel ne=[K,OC,IC]; input ne=[IL,IC,1]; returns [IL*stride,OC,1].
ggml_tensor * conv_transpose_1d_matmul(ggml_context * ctx, ggml_tensor * kernel,
                                       ggml_tensor * input, int stride) {
    const int64_t K  = kernel->ne[0];
    const int64_t OC = kernel->ne[1];
    const int64_t IC = kernel->ne[2];
    const int64_t IL = input->ne[0];
    const int64_t s  = stride;
    GGML_ASSERT(K == 2 * s);
    GGML_ASSERT(s % 2 == 0); // s/2 trim + IL*s length assume even stride (DAC rates 8,8,4,2)

    ggml_tensor * x2 = ggml_cont(ctx, ggml_transpose(ctx,
        ggml_reshape_2d(ctx, input, IL, IC)));                          // [IC, IL]

    auto phase = [&](int64_t k0) -> ggml_tensor * {
        ggml_tensor * Wv = ggml_view_3d(ctx, kernel, s, OC, IC,
            kernel->nb[1], kernel->nb[2], (size_t) k0 * kernel->nb[0]); // [s, OC, IC]
        ggml_tensor * Wr = ggml_reshape_2d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, Wv, 1, 2, 0, 3)), IC, s * OC); // [IC, s*OC]
        ggml_tensor * M  = ggml_mul_mat(ctx, Wr, x2);                   // [s*OC, IL]
        ggml_mul_mat_set_prec(M, GGML_PREC_F32); // deep-IC contraction; DAC is precision-sensitive
        ggml_tensor * Mi = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, M, s, OC, IL), 0, 2, 1, 3));           // [s, IL, OC]
        return ggml_reshape_2d(ctx, Mi, IL * s, OC);                    // [IL*s, OC] ol=i*s+p
    };
    ggml_tensor * A = phase(0);                                         // ol = i*s + p
    ggml_tensor * B = phase(s);                                         // ol = i*s + p + s

    // A occupies [0, IL*s); B is shifted +s to [s, IL*s+s). Pad both to the full
    // length and overlap-add.
    ggml_tensor * A_full = ggml_pad_ext(ctx, A, 0, (int) s, 0, 0, 0, 0, 0, 0); // [IL*s+s, OC]
    ggml_tensor * B_full = ggml_pad_ext(ctx, B, (int) s, 0, 0, 0, 0, 0, 0, 0); // [IL*s+s, OC]
    ggml_tensor * out    = ggml_add(ctx, A_full, B_full);                     // [IL*s+s, OC]

    ggml_tensor * trimmed = ggml_cont(ctx, ggml_view_2d(ctx, out,
        IL * s, OC, out->nb[1], (size_t) (s / 2) * out->nb[0]));       // [IL*s, OC]
    return ggml_reshape_3d(ctx, trimmed, IL * s, OC, 1);
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

// Builds the DAC graph for a window of `n_win` frames. Inputs are the named
// tensors "dac_codes" ([n_win, n_q] I32) and "snake_eps"; outputs are "latent"
// and "wav" (n_win * dac_hop samples).
ggml_cgraph * build_dac_graph(ggml_context * ctx, const parler_model & model,
                              int n_win, bool convt_mm) {
    const parler_hparams & hp = model.hparams;
    const int n_q = hp.dac_n_q;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, PARLER_MAX_NODES, false);

    ggml_tensor * codes_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_win, n_q);
    ggml_set_name(codes_t, "dac_codes");
    ggml_set_input(codes_t);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(eps, "snake_eps");
    ggml_set_input(eps);

    // RVQ from_codes: latent = sum_k out_proj_k(codebook_k[codes_k])
    ggml_tensor * latent = nullptr;
    for (int k = 0; k < n_q; ++k) {
        const parler_dac_quant & q = model.dac_quant[k];
        ggml_tensor * ids = ggml_view_1d(ctx, codes_t, n_win, (size_t) k * codes_t->nb[1]);
        ggml_tensor * z   = ggml_get_rows(ctx, q.codebook, ids);                   // [8, T]
        ggml_tensor * w2  = ggml_reshape_2d(ctx, q.out_proj_w,
                                            q.out_proj_w->ne[1], q.out_proj_w->ne[2]);
        ggml_tensor * proj = ggml_mul_mat(ctx, w2, z);
        ggml_mul_mat_set_prec(proj, GGML_PREC_F32); // latent feeds the precision-sensitive conv stack
        ggml_tensor * p   = ggml_add(ctx, proj, q.out_proj_b);                     // [1024, T]
        latent = latent ? ggml_add(ctx, latent, p) : p;
    }
    ggml_set_name(latent, "latent");
    ggml_set_output(latent);

    // [latent_dim, T] -> [T, latent_dim, 1] for the conv stack
    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, latent));
    x = ggml_reshape_3d(ctx, x, n_win, hp.dac_latent, 1);

    x = add_bias(ctx, conv1d_f32(ctx, model.dac_conv_in_w, x, 1, 3, 1), model.dac_conv_in_b);

    for (const parler_dac_block & blk : model.dac_blocks) {
        const int s = blk.stride;
        x = snake(ctx, x, blk.snake_alpha, eps);
        const int64_t t_in = x->ne[0];
        x = convt_mm ? conv_transpose_1d_matmul(ctx, blk.convt_w, x, s)
                     : conv_transpose_1d_trim(ctx, blk.convt_w, x, s, s / 2);
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

    return gf;
}

// Decodes frames [w0, w1) of the full [n_q, n_frames] `codes` and appends the
// sub-span [keep0, keep1) (absolute frame indices, w0 <= keep0 <= keep1 <= w1)
// to pcm_out / latent_out.
bool decode_window(const parler_model & model, const int32_t * codes, int n_frames,
                   int w0, int w1, int keep0, int keep1, bool convt_mm,
                   ggml_gallocr_t allocr, int n_threads,
                   std::vector<float> & pcm_out, std::vector<float> * latent_out,
                   std::string & err) {
    const parler_hparams & hp = model.hparams;
    const int n_q   = hp.dac_n_q;
    const int n_win = w1 - w0;

    const size_t ctx_size = ggml_tensor_overhead() * PARLER_MAX_NODES +
                            ggml_graph_overhead_custom(PARLER_MAX_NODES, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        err = "ggml_init failed";
        fprintf(stderr, "parler_dac_decode: %s\n", err.c_str());
        return false;
    }

    // the graph must be freshly built per allocation pass (sched_dispatch contract)
    ggml_cgraph * gf = build_dac_graph(ctx, model, n_win, convt_mm);

    // gather the window's codes into the [n_win, n_q] layout the graph expects
    std::vector<int32_t> win_codes((size_t) n_q * n_win);
    for (int k = 0; k < n_q; ++k) {
        memcpy(&win_codes[(size_t) k * n_win], &codes[(size_t) k * n_frames + w0],
               (size_t) n_win * sizeof(int32_t));
    }

    bool ok = false;
    bool use_sched = false;
    do {
        if (!parler_graph_prepare(model, gf, allocr, use_sched, "parler_dac_decode")) {
            // distinguishes an allocation failure from a compute/GPU fault, which
            // otherwise look identical to the caller
            err = "graph prepare failed for a " + std::to_string(n_win) +
                  "-frame window (" +
                  std::to_string(ggml_gallocr_get_buffer_size(allocr, 0) >> 20) + " MiB)";
            break;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dac_codes"), win_codes.data(), 0,
                                win_codes.size() * sizeof(int32_t));
        const float eps_val = 1e-9f;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "snake_eps"), &eps_val, 0, sizeof(eps_val));

        if (!parler_graph_compute(model, gf, use_sched, n_threads, "parler_dac_decode")) {
            err = "graph compute failed for a " + std::to_string(n_win) + "-frame window";
            break;
        }

        ggml_tensor * wav_t = ggml_graph_get_tensor(gf, "wav");
        if (ggml_nelements(wav_t) != (int64_t) n_win * hp.dac_hop) {
            err = "wav length " + std::to_string(ggml_nelements(wav_t)) + " != n_win " +
                  std::to_string(n_win) + " * hop " + std::to_string(hp.dac_hop) +
                  " (dac_hop disagrees with dac_rates?)";
            fprintf(stderr, "parler_dac_decode: %s\n", err.c_str());
            break;
        }
        const size_t off = (size_t) (keep0 - w0) * hp.dac_hop * sizeof(float);
        const size_t n   = (size_t) (keep1 - keep0) * hp.dac_hop;
        const size_t at  = pcm_out.size();
        pcm_out.resize(at + n);
        ggml_backend_tensor_get(wav_t, pcm_out.data() + at, off, n * sizeof(float));

        if (latent_out) {
            ggml_tensor * lat_t = ggml_graph_get_tensor(gf, "latent");
            const size_t loff = (size_t) (keep0 - w0) * hp.dac_latent * sizeof(float);
            const size_t ln   = (size_t) (keep1 - keep0) * hp.dac_latent;
            const size_t lat  = latent_out->size();
            latent_out->resize(lat + ln);
            ggml_backend_tensor_get(lat_t, latent_out->data() + lat, loff, ln * sizeof(float));
        }
        ok = true;
    } while (false);

    ggml_free(ctx);
    return ok;
}

} // namespace

// Propagates the support radius forward through the conv stack, in samples at each
// stage's own rate, then converts to latent frames. Conservative: the transposed
// conv's spread is bounded by 3*stride/2.
int parler_dac_rf_frames(const parler_model & model) {
    const parler_hparams & hp = model.hparams;

    int64_t r = (model.dac_conv_in_w->ne[0] - 1) / 2;             // conv_in, pad (k-1)/2
    for (const parler_dac_block & blk : model.dac_blocks) {
        const int64_t s = blk.stride;
        r = r * s + (3 * s + 1) / 2;                              // upsample + convt spread
        for (int j = 0; j < 3; ++j) {
            r += (blk.res[j].conv1_w->ne[0] - 1) / 2 * dilations[j];
        }
    }
    r += (model.dac_conv_out_w->ne[0] - 1) / 2;                   // conv_out

    const int64_t hop = hp.dac_hop > 0 ? hp.dac_hop : 1;
    return (int) ((r + hop - 1) / hop);
}

// Peak DAC compute-buffer bytes a decode of n_frames would need. Measured without
// allocating, so the bounded-memory test is cheap on every backend.
size_t parler_dac_compute_buffer_size(const parler_model & model, int n_frames) {
    const bool convt_mm = model.on_gpu || std::getenv("PARLER_DAC_CONVT_MATMUL") != nullptr;
    const int  rf    = parler_dac_rf_frames(model);
    const int  n_win = std::min(n_frames, PARLER_DAC_WINDOW_FRAMES + 2 * rf);

    const size_t ctx_size = ggml_tensor_overhead() * PARLER_MAX_NODES +
                            ggml_graph_overhead_custom(PARLER_MAX_NODES, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 0;

    ggml_cgraph * gf = build_dac_graph(ctx, model, n_win, convt_mm);
    size_t size = 0;
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (allocr) {
        ggml_gallocr_reserve_n_size(allocr, gf, nullptr, nullptr, &size);
        ggml_gallocr_free(allocr);
    }
    ggml_free(ctx);
    return size;
}

bool parler_dac_decode(const parler_model & model, const int32_t * codes, int n_frames,
                       int n_threads, std::vector<float> & pcm_out,
                       std::vector<float> * latent_out,
                       int out_begin, int out_end, std::string * err_out) {
    std::string err;
    const parler_hparams & hp = model.hparams;
    const int n_q = hp.dac_n_q;
    // GPU: transposed-conv via phase-matmuls (fast GEMM). Env flag lets the CPU
    // reference-parity test exercise the same path for validation.
    const bool convt_mm = model.on_gpu || std::getenv("PARLER_DAC_CONVT_MATMUL") != nullptr;

    if (n_frames <= 0) {
        err = "invalid n_frames=" + std::to_string(n_frames);
        fprintf(stderr, "%s: %s\n", __func__, err.c_str());
        if (err_out) *err_out = err;
        return false;
    }
    if (out_end < 0) out_end = n_frames;
    if (out_begin < 0 || out_begin > out_end || out_end > n_frames) {
        err = "invalid output range [" + std::to_string(out_begin) + ", " +
              std::to_string(out_end) + ") for n_frames=" + std::to_string(n_frames);
        fprintf(stderr, "%s: %s\n", __func__, err.c_str());
        if (err_out) *err_out = err;
        return false;
    }
    for (int64_t i = 0; i < (int64_t) n_q * n_frames; ++i) {
        if (codes[i] < 0 || codes[i] >= hp.dac_codebook_size) {
            err = "code " + std::to_string(codes[i]) + " at index " + std::to_string(i) +
                  " out of range [0, " + std::to_string(hp.dac_codebook_size) + ")";
            fprintf(stderr, "%s: %s\n", __func__, err.c_str());
            if (err_out) *err_out = err;
            return false;
        }
    }

    if (!model.dac_allocr) {
        model.dac_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    }
    ggml_gallocr_t allocr = model.dac_allocr;
    if (!allocr) {
        err = "gallocr creation failed";
        fprintf(stderr, "%s: %s\n", __func__, err.c_str());
        if (err_out) *err_out = err;
        return false;
    }

    pcm_out.clear();
    pcm_out.reserve((size_t) (out_end - out_begin) * hp.dac_hop);
    if (latent_out) {
        latent_out->clear();
        latent_out->reserve((size_t) (out_end - out_begin) * hp.dac_latent);
    }

    // One graph per window, each padded with a receptive field of real context so
    // the kept interior matches a whole-sequence decode exactly. No op here reduces
    // along the sequence, so window length changes only how many outputs are
    // computed, never the arithmetic that produces each one.
    const int rf = parler_dac_rf_frames(model);
    bool ok = true;
    for (int a = out_begin; a < out_end && ok; a += PARLER_DAC_WINDOW_FRAMES) {
        const int b  = std::min(a + PARLER_DAC_WINDOW_FRAMES, out_end);
        const int w0 = std::max(0,        a - rf);
        const int w1 = std::min(n_frames, b + rf);
        ok = decode_window(model, codes, n_frames, w0, w1, a, b, convt_mm,
                           allocr, n_threads, pcm_out, latent_out, err);
    }

    if (!ok) {
        pcm_out.clear();
        if (latent_out) latent_out->clear();
        if (err_out) *err_out = err;
    }
    return ok;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
