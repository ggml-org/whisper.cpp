#include "denoiser_ggml.h"

#include "../backend_selection.h"
#include "../backend_util.h"

#include "ggml-backend.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {
namespace detail {

// [n, B] slice of rows [r0, r0+n) from a [3H, B] gate tensor, as a strided view
// (the consuming add/mul emit a contiguous result, so no cont is needed).
static ggml_tensor * gate_rows(ggml_context * ctx, ggml_tensor * g, int64_t r0, int64_t n) {
    const int64_t B = g->ne[1];
    return ggml_view_2d(ctx, g, n, B, g->nb[1], (size_t) r0 * g->nb[0]);
}

ggml_tensor * gru_batched(ggml_context * ctx, ggml_tensor * x, ggml_tensor * Wih,
                          ggml_tensor * Whh, ggml_tensor * Bih, ggml_tensor * Bhh,
                          bool reverse, bool fused) {
    const int64_t I = x->ne[0];
    const int64_t B = x->ne[1];
    const int64_t L = x->ne[2];
    const int64_t H = Whh->ne[0];               // Whh is [H, 3H]
    ggml_tensor * bih = ggml_reshape_2d(ctx, Bih, 3 * H, 1);
    ggml_tensor * bhh = ggml_reshape_2d(ctx, Bhh, 3 * H, 1);

    // Precompute the state-independent input transform Wih*x for ALL timesteps in one wide
    // [3H, B*L] GEMM (bias broadcast once); bit-identical to the per-step transform.
    ggml_tensor * gi_all = ggml_add(ctx, ggml_mul_mat(ctx, Wih, ggml_reshape_2d(ctx, x, I, B * L)), bih);
    gi_all               = ggml_reshape_3d(ctx, gi_all, 3 * H, B, L);          // [3H, B, L]

    // Fused GRU op: the whole recurrent sweep in one dispatch; the per-step path below is
    // the fallback for backends without ggml_gru.
    if (fused) {
        return ggml_gru(ctx, Whh, gi_all, Bhh, reverse);                        // [H, B, L]
    }

    ggml_tensor *              h = nullptr;      // [H, B] running state (zero at start)
    std::vector<ggml_tensor *> steps((size_t) L, nullptr);
    for (int64_t s = 0; s < L; ++s) {
        const int64_t t  = reverse ? (L - 1 - s) : s;
        ggml_tensor * gi = ggml_view_2d(ctx, gi_all, 3 * H, B, gi_all->nb[1],
                                        (size_t) t * gi_all->nb[2]);           // [3H, B] at time t
        // h == 0 on the first step: Whh*0 = 0, so gh = Bhh broadcast over B.
        ggml_tensor * gh = h ? ggml_add(ctx, ggml_mul_mat(ctx, Whh, h), bhh)
                             : ggml_add(ctx, ggml_scale(ctx, gi, 0.0f), bhh);   // [3H, B]

        ggml_tensor * r = ggml_sigmoid(ctx, ggml_add(ctx, gate_rows(ctx, gi, 0, H),
                                                          gate_rows(ctx, gh, 0, H)));
        ggml_tensor * z = ggml_sigmoid(ctx, ggml_add(ctx, gate_rows(ctx, gi, H, H),
                                                          gate_rows(ctx, gh, H, H)));
        ggml_tensor * ncand = ggml_tanh(ctx, ggml_add(ctx, gate_rows(ctx, gi, 2 * H, H),
                                                           ggml_mul(ctx, r, gate_rows(ctx, gh, 2 * H, H))));
        // h_new = (1-z)*n + z*h.  First step (h==0): (1-z)*n = n - z*n.
        ggml_tensor * h_new = h ? ggml_add(ctx, ncand, ggml_mul(ctx, z, ggml_sub(ctx, h, ncand)))
                                : ggml_sub(ctx, ncand, ggml_mul(ctx, z, ncand));
        h                   = h_new;
        steps[(size_t) t]   = ggml_reshape_3d(ctx, h_new, H, B, 1);
    }
    // Stitch the L steps [H,B,1] into [H,B,L] with a BALANCED concat tree (associative along
    // ne2, so bit-identical to a left-fold) -> O(L*logL) data moved instead of O(L^2).
    std::vector<ggml_tensor *> lvl = steps;
    while (lvl.size() > 1) {
        std::vector<ggml_tensor *> nxt;
        nxt.reserve((lvl.size() + 1) / 2);
        for (size_t i = 0; i + 1 < lvl.size(); i += 2)
            nxt.push_back(ggml_concat(ctx, lvl[i], lvl[i + 1], 2));
        if (lvl.size() & 1) nxt.push_back(lvl.back());
        lvl.swap(nxt);
    }
    return lvl[0];                                                                 // [H, B, L]
}

ggml_tensor * conv2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * W,
                     ggml_tensor * bias, int stride_f, int pad_f, int groups, bool fused) {
    const int64_t kf = W->ne[0], kt = W->ne[1], Cing = W->ne[2], Cout = W->ne[3];
    const int64_t T  = x->ne[1];
    // Fold both pads INTO the conv (bit-identical to an explicit zero-pad): symmetric freq pad,
    // causal time via symmetric p1=kt-1 then keep the first T frames.
    const int     tp = (int) (kt - 1);
    ggml_tensor * y;
    if (groups == 1) {
        y = ggml_conv_2d_direct(ctx, W, x, stride_f, 1, pad_f, tp, 1, 1);    // [Fout, T+tp, Cout, Bc]
    } else if (fused && Cing == 1 && groups == Cout) {
        // fused depthwise: one CONV_2D_DW op replaces the C-way per-channel loop + concats.  The op's
        // whcn kernel requires a contiguous data tensor (the per-group loop's cont guaranteed this).
        ggml_tensor * xin = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
        y = ggml_conv_2d_dw_direct(ctx, W, xin, stride_f, 1, pad_f, tp, 1, 1); // [Fout, T+tp, Cout, Bc]
    } else {
        const int64_t oc_g = Cout / groups;                                  // out chans per group
        y = nullptr;
        for (int g = 0; g < groups; ++g) {
            ggml_tensor * xg = ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], x->ne[1], Cing, x->ne[3],
                                                           x->nb[1], x->nb[2], x->nb[3],
                                                           (size_t) g * Cing * x->nb[2]));
            ggml_tensor * wg = ggml_cont(ctx, ggml_view_4d(ctx, W, kf, kt, Cing, oc_g,
                                                           W->nb[1], W->nb[2], W->nb[3],
                                                           (size_t) g * oc_g * W->nb[3]));
            ggml_tensor * yg = ggml_conv_2d_direct(ctx, wg, xg, stride_f, 1, pad_f, tp, 1, 1);
            y = y ? ggml_concat(ctx, y, yg, 2) : yg;                         // stack out chans
        }
    }
    if (kt > 1) {  // causal slice: keep first T time frames (drop the symmetric-pad tail)
        y = ggml_cont(ctx, ggml_view_4d(ctx, y, y->ne[0], T, y->ne[2], y->ne[3],
                                        y->nb[1], y->nb[2], y->nb[3], 0));
    }
    if (bias) y = ggml_add(ctx, y, ggml_reshape_3d(ctx, bias, 1, 1, Cout));  // per-out-channel
    return y;
}

// Insert stride-1 zeros between freq (ne0) samples: [F,T,C,Bc] -> [(F-1)*s+1, T, C, Bc].
// Fallback transposes so the pad runs on a big contiguous ne0 (few, coalesced workgroups).
static ggml_tensor * zero_upsample_freq(ggml_context * ctx, ggml_tensor * x, int s, bool fused) {
    if (s == 1) return x;
    if (fused) return ggml_zero_upsample(ctx, x, s);  // one scatter op vs transpose/pad/transpose/cont
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2], Bc = x->ne[3];
    const int64_t K = T * C * Bc;
    ggml_tensor * r = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, F, K))); // [K, F]
    r               = ggml_reshape_3d(ctx, r, K, 1, F);                  // [K, 1, F]
    r               = ggml_pad_ext(ctx, r, 0, 0, 0, s - 1, 0, 0, 0, 0);  // pad ne1 1->s: [K, s, F]
    r               = ggml_reshape_2d(ctx, r, K, s * F);                 // [K, s*F], x at m=s*f
    r               = ggml_cont(ctx, ggml_transpose(ctx, r));            // [s*F, K]
    r               = ggml_reshape_4d(ctx, r, s * F, T, C, Bc);          // [s*F, T, C, Bc], x at f*s
    const int64_t Fu = (F - 1) * s + 1;
    return ggml_cont(ctx, ggml_view_4d(ctx, r, Fu, T, C, Bc, r->nb[1], r->nb[2], r->nb[3], 0));
}

ggml_tensor * conv_transpose2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * Wc,
                               ggml_tensor * bias, int stride_f, int pad_f, int groups, bool fused) {
    const int64_t kf = Wc->ne[0];
    ggml_tensor * u  = zero_upsample_freq(ctx, x, stride_f, fused);
    return conv2d(ctx, u, Wc, bias, 1, (int) (kf - 1 - pad_f), groups, fused);
}

// LayerNorm over the combined (ne0*ne1) per (ne2,ne3) column, affine g,b [ne0*ne1].
// x:[A, Fw, T, Bc] -> [A, Fw, T, Bc].  Matches Runner::layernorm_fc (norm axis = F*C).
static ggml_tensor * layernorm_fc(ggml_context * ctx, ggml_tensor * x, ggml_tensor * g,
                                  ggml_tensor * b, float eps) {
    const int64_t A = x->ne[0], Fw = x->ne[1], T = x->ne[2], Bc = x->ne[3];
    ggml_tensor * r = ggml_reshape_2d(ctx, ggml_cont(ctx, x), A * Fw, T * Bc);  // [A*Fw, T*Bc]
    r               = ggml_norm(ctx, r, eps);
    r               = ggml_mul(ctx, r, ggml_reshape_2d(ctx, g, A * Fw, 1));
    r               = ggml_add(ctx, r, ggml_reshape_2d(ctx, b, A * Fw, 1));
    return ggml_reshape_4d(ctx, r, A, Fw, T, Bc);
}

static ggml_tensor * gru_uni(ggml_context * ctx, ggml_tensor * x, const std::string & p,
                             const WResolver & W, bool fused) {
    return gru_batched(ctx, x, W(p + ".weight_ih_l0"), W(p + ".weight_hh_l0"),
                       W(p + ".bias_ih_l0"), W(p + ".bias_hh_l0"), false, fused);
}

static ggml_tensor * gru_bi(ggml_context * ctx, ggml_tensor * x, const std::string & p,
                            const WResolver & W, bool fused) {
    ggml_tensor * f = gru_batched(ctx, x, W(p + ".weight_ih_l0"), W(p + ".weight_hh_l0"),
                                  W(p + ".bias_ih_l0"), W(p + ".bias_hh_l0"), false, fused);
    ggml_tensor * r = gru_batched(ctx, x, W(p + ".weight_ih_l0_reverse"),
                                  W(p + ".weight_hh_l0_reverse"), W(p + ".bias_ih_l0_reverse"),
                                  W(p + ".bias_hh_l0_reverse"), true, fused);
    return ggml_concat(ctx, f, r, 0);                                          // [2H, B, L]
}

// Grouped RNN: split the feature (ne0) in half, run rnn1/rnn2, concat.  x:[I,B,L].
static ggml_tensor * grnn(ggml_context * ctx, ggml_tensor * x, const std::string & p,
                          const WResolver & W, bool bidir, bool fused) {
    const int64_t I = x->ne[0], B = x->ne[1], L = x->ne[2], half = I / 2;
    ggml_tensor * x1 = ggml_cont(ctx, ggml_view_3d(ctx, x, half, B, L, x->nb[1], x->nb[2], 0));
    ggml_tensor * x2 = ggml_cont(ctx, ggml_view_3d(ctx, x, half, B, L, x->nb[1], x->nb[2],
                                                   (size_t) half * x->nb[0]));
    ggml_tensor * y1 = bidir ? gru_bi(ctx, x1, p + ".rnn1", W, fused) : gru_uni(ctx, x1, p + ".rnn1", W, fused);
    ggml_tensor * y2 = bidir ? gru_bi(ctx, x2, p + ".rnn2", W, fused) : gru_uni(ctx, x2, p + ".rnn2", W, fused);
    return ggml_concat(ctx, y1, y2, 0);                                        // [hy, B, L]
}

// FC over the hidden axis (ne0): y[hy,B,L] -> [Cout,B,L].  Wfc ggml [hy,Cout].
static ggml_tensor * fc_hidden(ggml_context * ctx, ggml_tensor * y, ggml_tensor * Wfc,
                               ggml_tensor * bias) {
    const int64_t hy = y->ne[0], B = y->ne[1], L = y->ne[2], Cout = Wfc->ne[1];
    ggml_tensor * r  = ggml_mul_mat(ctx, Wfc, ggml_reshape_2d(ctx, ggml_cont(ctx, y), hy, B * L)); // [Cout, B*L]
    if (bias) r = ggml_add(ctx, r, ggml_reshape_2d(ctx, bias, Cout, 1));
    return ggml_reshape_3d(ctx, r, Cout, B, L);                                // [Cout, B, L]
}

ggml_tensor * dpgrnn(ggml_context * ctx, ggml_tensor * x, const std::string & prefix,
                     const WResolver & W, float ln_eps, bool fused) {
    // canonical bottleneck layout xc:[C, F, T, Bc] (from FTC [F,T,C,Bc]).  The
    // chunk-batch Bc folds into each GRU's batch dim (T*Bc intra, F*Bc inter).
    ggml_tensor * xc = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));       // [C, F, T, Bc]
    const int64_t C = xc->ne[0], Fw = xc->ne[1], T = xc->ne[2], Bc = xc->ne[3];

    // ---- intra (over freq, batch = time*chunk): GRU input [C, T*Bc, F] ----
    ggml_tensor * xi = ggml_cont(ctx, ggml_permute(ctx, xc, 0, 3, 1, 2));      // [C, T, Bc, F]
    xi               = ggml_reshape_3d(ctx, xi, C, T * Bc, Fw);
    ggml_tensor * gi = grnn(ctx, xi, prefix + ".intra_rnn", W, /*bidir=*/true, fused); // [hy, T*Bc, F]
    ggml_tensor * ai = fc_hidden(ctx, gi, W(prefix + ".intra_fc.weight"),
                                 W(prefix + ".intra_fc.bias"));                // [C, T*Bc, F]
    ai = ggml_reshape_4d(ctx, ai, C, T, Bc, Fw);                               // [C, T, Bc, F]
    ai = ggml_cont(ctx, ggml_permute(ctx, ai, 0, 2, 3, 1));                    // [C, F, T, Bc]
    ai = layernorm_fc(ctx, ai, W(prefix + ".intra_ln.weight"), W(prefix + ".intra_ln.bias"), ln_eps);
    ggml_tensor * intra = ggml_add(ctx, ai, xc);                              // residual, [C, F, T, Bc]

    // ---- inter (over time, batch = freq*chunk): GRU input [C, F*Bc, T] ----
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, intra, 0, 1, 3, 2));   // [C, F, Bc, T]
    xt               = ggml_reshape_3d(ctx, xt, C, Fw * Bc, T);
    ggml_tensor * gt = grnn(ctx, xt, prefix + ".inter_rnn", W, /*bidir=*/false, fused); // [hy2, F*Bc, T]
    ggml_tensor * at = fc_hidden(ctx, gt, W(prefix + ".inter_fc.weight"),
                                 W(prefix + ".inter_fc.bias"));                // [C, F*Bc, T]
    at = ggml_reshape_4d(ctx, at, C, Fw, Bc, T);                               // [C, F, Bc, T]
    at = ggml_cont(ctx, ggml_permute(ctx, at, 0, 1, 3, 2));                    // [C, F, T, Bc]
    at = layernorm_fc(ctx, at, W(prefix + ".inter_ln.weight"), W(prefix + ".inter_ln.bias"), ln_eps);
    ggml_tensor * inter = ggml_add(ctx, at, intra);                           // residual, [C, F, T, Bc]

    return ggml_cont(ctx, ggml_permute(ctx, inter, 2, 0, 1, 3));              // back to [F, T, C, Bc]
}

ggml_tensor * ctfa(ggml_context * ctx, ggml_tensor * x, const std::string & p,
                   const WResolver & W, int r, bool fused) {
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2], Bc = x->ne[3];

    // ---- TA: energy-mean over freq -> GRU over time (batch=chunk) -> FC -> sigmoid ----
    ggml_tensor * zt   = ggml_mean(ctx, ggml_sqr(ctx, x));                    // [1, T, C, Bc]
    ggml_tensor * ztin = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, zt, 3, 2, 0, 1)),
                                         C, Bc, T);                          // [C, Bc, T]
    ggml_tensor * at   = gru_batched(ctx, ztin, W(p + ".ta_gru.weight_ih_l0"),
                                     W(p + ".ta_gru.weight_hh_l0"), W(p + ".ta_gru.bias_ih_l0"),
                                     W(p + ".ta_gru.bias_hh_l0"), false, fused); // [2C, Bc, T]
    at = fc_hidden(ctx, at, W(p + ".ta_fc.weight"), W(p + ".ta_fc.bias"));    // [C, Bc, T]
    at = ggml_sigmoid(ctx, at);
    ggml_tensor * at_b = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, at, C, Bc, T, 1),
                                                     2, 3, 1, 0));           // [1, T, C, Bc]

    // ---- FA: energy-mean over chan -> fold freq by r -> BiGRU (batch=time*chunk) -> FC -> sigmoid ----
    ggml_tensor * xcf = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));     // [C, T, F, Bc]
    ggml_tensor * ec  = ggml_mean(ctx, ggml_sqr(ctx, xcf));                   // [1, T, F, Bc]
    ggml_tensor * ecF = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, ec, 3, 1, 0, 2)),
                                        F, T * Bc);                          // [F, T*Bc]
    const int     pad_len = (r - (int) (F % r)) % r;
    const int64_t Fp = F + pad_len, Hh = Fp / r;
    ggml_tensor * ecp  = ggml_pad_ext(ctx, ecF, 0, pad_len, 0, 0, 0, 0, 0, 0); // [Fp, T*Bc]
    ggml_tensor * fold = ggml_reshape_3d(ctx, ecp, r, Hh, T * Bc);           // [r, Hh, T*Bc]
    ggml_tensor * fain = ggml_cont(ctx, ggml_permute(ctx, fold, 0, 2, 1, 3)); // [r, T*Bc, Hh]
    ggml_tensor * y    = gru_bi(ctx, fain, p + ".fa.gru", W, fused);         // [2*Hfa, T*Bc, Hh]
    y                  = fc_hidden(ctx, y, W(p + ".fa.fc.weight"), W(p + ".fa.fc.bias")); // [r, T*Bc, Hh]
    y                  = ggml_sigmoid(ctx, y);
    ggml_tensor * yh   = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));   // [r, Hh, T*Bc]
    ggml_tensor * afp  = ggml_reshape_2d(ctx, yh, Fp, T * Bc);               // [Fp, T*Bc]
    ggml_tensor * afF  = ggml_cont(ctx, ggml_view_2d(ctx, afp, F, T * Bc, afp->nb[1], 0)); // [F, T*Bc]
    ggml_tensor * af_b = ggml_reshape_4d(ctx, afF, F, T, 1, Bc);             // [F, T, 1, Bc]

    return ggml_mul(ctx, ggml_mul(ctx, x, at_b), af_b);                      // [F, T, C, Bc]
}

// ERB band-merge: keep `low` linear bins, compress the rest to `hi` via E[in,hi].
// x:[F,T,C,Bc] -> [low+hi, T, C, Bc].  E = ggml (erb.erb_fc.weight [hi,in] -> [in,hi]).
static ggml_tensor * erb_bm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * E, int low, int hi) {
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2], Bc = x->ne[3], in = F - low;
    ggml_tensor * lo = ggml_cont(ctx, ggml_view_4d(ctx, x, low, T, C, Bc, x->nb[1], x->nb[2], x->nb[3], 0));
    ggml_tensor * hg = ggml_cont(ctx, ggml_view_4d(ctx, x, in, T, C, Bc, x->nb[1], x->nb[2], x->nb[3],
                                                   (size_t) low * x->nb[0]));            // [in, T, C, Bc]
    ggml_tensor * hc = ggml_mul_mat(ctx, E, ggml_reshape_2d(ctx, hg, in, T * C * Bc));   // [hi, T*C*Bc]
    return ggml_concat(ctx, lo, ggml_reshape_4d(ctx, hc, hi, T, C, Bc), 0);              // [low+hi, T, C, Bc]
}

// ERB band-split (inverse): keep `low` bins, expand `hi` bands to `out` via E[hi,out].
static ggml_tensor * erb_bs(ggml_context * ctx, ggml_tensor * m, ggml_tensor * E, int low, int hi) {
    const int64_t T = m->ne[1], C = m->ne[2], Bc = m->ne[3], out = E->ne[1];
    ggml_tensor * lo = ggml_cont(ctx, ggml_view_4d(ctx, m, low, T, C, Bc, m->nb[1], m->nb[2], m->nb[3], 0));
    ggml_tensor * hg = ggml_cont(ctx, ggml_view_4d(ctx, m, hi, T, C, Bc, m->nb[1], m->nb[2], m->nb[3],
                                                   (size_t) low * m->nb[0]));            // [hi, T, C, Bc]
    ggml_tensor * hc = ggml_mul_mat(ctx, E, ggml_reshape_2d(ctx, hg, hi, T * C * Bc));   // [out, T*C*Bc]
    return ggml_concat(ctx, lo, ggml_reshape_4d(ctx, hc, out, T, C, Bc), 0);             // [low+out, T, C, Bc]
}

// BatchNorm folded to per-channel scale/shift (precomputed at load).  x:[F,T,C].
static ggml_tensor * batchnorm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * shift) {
    const int64_t C = x->ne[2];
    return ggml_add(ctx, ggml_mul(ctx, x, ggml_reshape_3d(ctx, scale, 1, 1, C)),
                    ggml_reshape_3d(ctx, shift, 1, 1, C));
}

// Per-(c,f) affine + per-channel PReLU.  aw,ab: ggml [F,C] (PyTorch [C,F]); slope [C].
ggml_tensor * affine_prelu(ggml_context * ctx, ggml_tensor * x, ggml_tensor * aw,
                           ggml_tensor * ab, ggml_tensor * slope, bool fused) {
    if (fused) return ggml_affine_prelu(ctx, x, aw, ab, slope);  // one fused op vs 7 elementwise
    const int64_t F = x->ne[0], C = x->ne[2];
    ggml_tensor * relu   = ggml_relu(ctx, x);
    ggml_tensor * neg    = ggml_sub(ctx, x, relu);                                       // min(x,0)
    ggml_tensor * prelu  = ggml_add(ctx, relu, ggml_mul(ctx, neg, ggml_reshape_3d(ctx, slope, 1, 1, C)));
    ggml_tensor * affine = ggml_add(ctx, ggml_mul(ctx, x, ggml_reshape_3d(ctx, aw, F, 1, C)),
                                    ggml_reshape_3d(ctx, ab, F, 1, C));
    return ggml_add(ctx, affine, prelu);
}

// Channel shuffle (2 groups): o[2c]=x[c], o[2c+1]=x[half+c].  x:[F,T,C,Bc].  The
// non-channel axes fold to one so the (group,idx) transpose fits ggml's 4D limit.
ggml_tensor * shuffle2(ggml_context * ctx, ggml_tensor * x, bool fused) {
    if (fused) return ggml_channel_shuffle(ctx, x, 2);  // one plane-copy op vs 3 permute-conts
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2], Bc = x->ne[3], half = C / 2;
    const int64_t S  = T * F * Bc;
    ggml_tensor * xc = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));                 // [C, T, F, Bc]
    ggml_tensor * r  = ggml_reshape_3d(ctx, xc, half, 2, S);                             // [half, 2, T*F*Bc]
    r                = ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3));                 // [2, half, S]
    r                = ggml_reshape_4d(ctx, r, C, T, F, Bc);                             // [C, T, F, Bc] shuffled
    return ggml_cont(ctx, ggml_permute(ctx, r, 2, 1, 0, 3));                             // [F, T, C, Bc]
}

// One encoder/decoder block (denoiser_core.cpp run_block).  type 0=XConv 1=XDWS 2=XMB.
static ggml_tensor * run_block(ggml_context * ctx, ggml_tensor * x, const std::string & base,
                               int type, int cout, int kf, int stride, int groups, bool deconv,
                               bool is_last, const WResolver & W, int r, bool fused) {
    (void) cout;
    const int pf = kf / 2;
    auto conv = [&](ggml_tensor * in, const std::string & wn, int st, int grp) -> ggml_tensor * {
        return deconv ? conv_transpose2d(ctx, in, W(wn + ".ct"), W(wn + ".bias"), st, pf, grp, fused)
                      : conv2d(ctx, in, W(wn + ".weight"), W(wn + ".bias"), st, pf, grp, fused);
    };
    auto bn = [&](ggml_tensor * h, const std::string & p) {
        return batchnorm(ctx, h, W(p + ".bn_scale"), W(p + ".bn_shift"));
    };
    auto ap = [&](ggml_tensor * h, const std::string & p) {
        return affine_prelu(ctx, h, W(p + ".affine_weight"), W(p + ".affine_bias"), W(p + ".slope_weight"), fused);
    };
    if (type == 0) { // XConv
        ggml_tensor * h = conv(x, base + ".ops.1", stride, groups);
        h               = bn(h, base + ".ops.2");
        if (!is_last) h = ap(h, base + ".ops.3");
        h               = ctfa(ctx, h, base + ".ops.4", W, r, fused);
        if (!is_last && groups == 2) h = shuffle2(ctx, h, fused);
        return h;
    }
    if (type == 1) { // XDWS: pconv(1x1) + depthwise dconv
        ggml_tensor * h = conv2d(ctx, x, W(base + ".pconv.0.weight"), W(base + ".pconv.0.bias"), 1, 0, groups, fused);
        h               = bn(h, base + ".pconv.1");
        h               = ap(h, base + ".pconv.2");
        if (groups == 2) h = shuffle2(ctx, h, fused);
        h = conv(h, base + ".dconv.1", stride, (int) h->ne[2]); // depthwise (groups == channels)
        h = bn(h, base + ".dconv.2");
        if (!is_last) h = ap(h, base + ".dconv.3");
        return ctfa(ctx, h, base + ".dconv.4", W, r, fused);
    }
    // type == 2: XMB
    ggml_tensor * h = conv2d(ctx, x, W(base + ".pconv1.0.weight"), W(base + ".pconv1.0.bias"), 1, 0, groups, fused);
    h               = bn(h, base + ".pconv1.1");
    h               = ap(h, base + ".pconv1.2");
    if (groups == 2) h = shuffle2(ctx, h, fused);
    h = conv(h, base + ".dconv.1", stride, (int) h->ne[2]); // depthwise
    h = bn(h, base + ".dconv.2");
    h = ap(h, base + ".dconv.3");
    h = conv2d(ctx, h, W(base + ".pconv2.0.weight"), W(base + ".pconv2.0.bias"), 1, 0, groups, fused);
    h = bn(h, base + ".pconv2.1");
    h = ctfa(ctx, h, base + ".pconv2.2", W, r, fused);
    if (h->ne[0] == x->ne[0] && h->ne[1] == x->ne[1] && h->ne[2] == x->ne[2]) h = ggml_add(ctx, h, x);
    if (!is_last && groups == 2) h = shuffle2(ctx, h, fused);
    return h;
}

} // namespace detail

namespace {

// Decoder-block config (mirrors denoiser_net_forward), indexed by encoder index i.
constexpr int kTypes[5]   = { 0, 2, 1, 2, 1 };
constexpr int kStrides[5] = { 2, 2, 1, 1, 1 };
constexpr int kGroups[5]  = { 1, 2, 2, 2, 2 };
constexpr int kChans[5]   = { 12, 24, 24, 32, 16 };
constexpr int kKf[5]      = { 3, 3, 3, 5, 5 };

// Host reindex: PyTorch transpose kernel Wt[Cin,Cout/g,kt,kf] -> regular-conv Wc
// [Cout,Cin/g,kt,kf] (IC<->OC swap + kt/kf flip).  Fed to conv2d by conv_transpose2d.
std::vector<float> reindex_ct(const std::vector<float> & Wt, int Cin, int Coutg, int kt, int kf, int groups) {
    const int          Cout = Coutg * groups, ing = Cin / groups;
    std::vector<float> Wc((size_t) Cout * ing * kt * kf);
    for (int oc = 0; oc < Cout; oc++) {
        const int grp = oc / Coutg, ocg = oc - grp * Coutg;
        for (int ici = 0; ici < ing; ici++)
            for (int it = 0; it < kt; it++)
                for (int jf = 0; jf < kf; jf++) {
                    const int ic = grp * ing + ici;
                    Wc[(((size_t) oc * ing + ici) * kt + it) * kf + jf] =
                        Wt[(((size_t) ic * Coutg + ocg) * kt + (kt - 1 - it)) * kf + (kf - 1 - jf)];
                }
    }
    return Wc;
}

struct Pend { std::string name; std::vector<float> data; int64_t ne[4]; int nd; };

struct GraphIO { ggml_tensor * feat_in, * re, * ie, * ro, * io; };

constexpr int MAX_NODES = 262144;

// Stage all weights host-side: raw F32 tensors, BatchNorms folded to per-channel
// scale/shift, decoder transpose-conv kernels reindexed for conv_transpose2d.
std::vector<Pend> stage_weights(const DenoiserWeights & w) {
    std::vector<Pend> pend;
    auto push = [&](const std::string & name, std::vector<float> data, const std::vector<int> & cshape) {
        Pend p; p.name = name; p.data = std::move(data);
        p.nd = cshape.empty() ? 1 : (int) cshape.size();
        for (int i = 0; i < 4; i++) p.ne[i] = 1;
        for (int i = 0; i < (int) cshape.size(); i++) p.ne[i] = cshape[(int) cshape.size() - 1 - i];
        pend.push_back(std::move(p));
    };
    for (const auto & kv : w.t) push(kv.first, kv.second.data, kv.second.shape);

    const std::string suf = ".running_var";
    for (const auto & kv : w.t) {
        const std::string & n = kv.first;
        if (n.size() <= suf.size() || n.compare(n.size() - suf.size(), suf.size(), suf) != 0) continue;
        const std::string    p  = n.substr(0, n.size() - suf.size());
        const std::vector<float> &g = w.get(p + ".weight").data, &b = w.get(p + ".bias").data;
        const std::vector<float> &rm = w.get(p + ".running_mean").data, &rv = kv.second.data;
        const int C = (int) rv.size();
        std::vector<float> scale(C), shift(C);
        for (int c = 0; c < C; c++) { float s = g[c] / std::sqrt(rv[c] + w.bn_eps); scale[c] = s; shift[c] = b[c] - rm[c] * s; }
        push(p + ".bn_scale", scale, { C }); push(p + ".bn_shift", shift, { C });
    }

    auto add_ct = [&](const std::string & wn, int groups) {
        const DnTensor & Wt = w.get(wn + ".weight");                       // [Cin, Cout/g, kt, kf]
        const int        Cin = Wt.shape[0], Coutg = Wt.shape[1], kt = Wt.shape[2], kf = Wt.shape[3];
        push(wn + ".ct", reindex_ct(Wt.data, Cin, Coutg, kt, kf, groups),
             { Coutg * groups, Cin / groups, kt, kf });
    };
    { int j = 0;
      for (int i = 4; i >= 1; i--, j++)
          add_ct("decoder.de_convs." + std::to_string(j) + ".dconv.1", kChans[i - 1]); // depthwise groups=cout
      add_ct("decoder.de_convs." + std::to_string(j) + ".ops.1", kGroups[0]);          // final XConv groups=1
    }
    return pend;
}

// log-magnitude feature on the host (no LOG kernel on OpenCL; trivial per-element work).
// Bit-identical to denoiser_net_forward.
std::vector<float> compute_log_features(const std::vector<float> & real_in,
                                        const std::vector<float> & imag_in, size_t n) {
    std::vector<float> feat(n);
    for (size_t i = 0; i < n; i++) {
        const float rr = real_in[i], ii = imag_in[i];
        float       mg = std::sqrt(rr * rr + ii * ii);
        if (mg < 1e-12f) mg = 1e-12f;
        feat[i] = std::log10(mg);
    }
    return feat;
}

} // namespace

struct DenoiserGgml::Impl {
    int   spec_bins = 257, erb_low = 65, erb_high = 64, freq_comp_ratio = 4, chunk_frames = 63;
    float bn_eps = 1e-5f, ln_eps = 1e-8f;

    ggml_backend_t        backend = nullptr;
    bool                  gpu = false, opencl = false;
    ggml_context *        wctx   = nullptr;
    ggml_backend_buffer_t wbuf   = nullptr;
    ggml_gallocr_t        galloc = nullptr;
    std::map<std::string, ggml_tensor *> wt;

    ggml_tensor * W(const std::string & n) const {
        auto it = wt.find(n);
        return it == wt.end() ? nullptr : it->second;
    }

    void    init_backend(int n_gpu_layers, bool verbose);
    void    upload_weights(const std::vector<Pend> & pend);
    GraphIO build_graph(ggml_context * ctx, ggml_cgraph * gf, int L, int Bc) const;
    void    preflight_ops(ggml_cgraph * gf) const;
    void    run(ggml_cgraph * gf, const GraphIO & gio, const std::vector<float> & feat_host,
                const std::vector<float> & real_in, const std::vector<float> & imag_in,
                int L, int Bc, std::vector<float> & real_out, std::vector<float> & imag_out);

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
    }
};

void DenoiserGgml::Impl::init_backend(int n_gpu_layers, bool verbose) {
    if (n_gpu_layers > 0) {
        bool unused = false;
        backend = ::tts_cpp::detail::init_gpu_backend(n_gpu_layers, verbose, "lavasr-dn", 0, false, &unused);
        gpu     = backend != nullptr;
    }
    if (!backend) { backend = ::tts_cpp::detail::init_cpu_backend(); gpu = false; }
    if (!backend) throw std::runtime_error("denoiser_ggml: no backend available");
    opencl = ::tts_cpp::detail::backend_is_opencl(backend);
}

void DenoiserGgml::Impl::upload_weights(const std::vector<Pend> & pend) {
    const size_t     overhead = ggml_tensor_overhead() * (pend.size() + 8);
    ggml_init_params wp       = { overhead, nullptr, true };
    wctx = ggml_init(wp);
    if (!wctx) throw std::runtime_error("denoiser_ggml: ggml_init(weights) failed");
    for (const auto & p : pend) {
        ggml_tensor * t = ggml_new_tensor(wctx, GGML_TYPE_F32, p.nd, p.ne);
        ggml_set_name(t, p.name.c_str());
        wt[p.name] = t;
    }
    wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) throw std::runtime_error("denoiser_ggml: weight buffer alloc failed");
    for (const auto & p : pend)
        ggml_backend_tensor_set(wt[p.name], p.data.data(), 0, ggml_nbytes(wt[p.name]));
    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) throw std::runtime_error("denoiser_ggml: gallocr_new failed");
}

GraphIO DenoiserGgml::Impl::build_graph(ggml_context * ctx, ggml_cgraph * gf, int L, int Bc) const {
    const int F = spec_bins, r = freq_comp_ratio;
    // Fused ops where the backend implements them (OpenCL + CPU); Metal/Vulkan use the
    // standard-op fallback, which the supports_op preflight would otherwise reject.
    const bool fused = opencl || !gpu;
    auto       W     = [this](const std::string & n) { return this->W(n); };

    // Inputs must come from the graph ctx so gallocr assigns them backing buffers.
    ggml_tensor * feat_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, F, L, 1, Bc); ggml_set_input(feat_in);
    ggml_tensor * re      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, F, L, 1, Bc); ggml_set_input(re);
    ggml_tensor * ie      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, F, L, 1, Bc); ggml_set_input(ie);

    ggml_tensor * x = detail::erb_bm(ctx, feat_in, W("erb.erb_fc.weight"), erb_low, erb_high);

    std::vector<ggml_tensor *> en;
    for (int i = 0; i < 5; i++) {
        x = detail::run_block(ctx, x, "encoder.en_convs." + std::to_string(i), kTypes[i], kChans[i],
                              kKf[i], kStrides[i], kGroups[i], false, false, W, r, fused);
        en.push_back(x);
    }
    x = detail::dpgrnn(ctx, x, "dpgrnn.0", W, ln_eps, fused);
    x = detail::dpgrnn(ctx, x, "dpgrnn.1", W, ln_eps, fused);

    int j = 0;
    for (int i = 4; i >= 1; i--, j++) {
        x = ggml_add(ctx, x, en[4 - j]);
        x = detail::run_block(ctx, x, "decoder.de_convs." + std::to_string(j), kTypes[i], kChans[i - 1],
                              kKf[i], kStrides[i], kGroups[i], true, false, W, r, fused);
    }
    x = ggml_add(ctx, x, en[0]);
    x = detail::run_block(ctx, x, "decoder.de_convs." + std::to_string(j), kTypes[0], 1,
                          kKf[0], kStrides[0], kGroups[0], true, true, W, r, fused);

    x                = ggml_sigmoid(ctx, x);                                        // ratio mask [129, L, 1]
    ggml_tensor * m  = detail::erb_bs(ctx, x, W("erb.ierb_fc.weight"), erb_low, erb_high); // [F, L, 1]
    ggml_tensor * ro = ggml_mul(ctx, re, m);
    ggml_tensor * io = ggml_mul(ctx, ie, m);
    ggml_set_output(ro); ggml_set_output(io);
    ggml_build_forward_expand(gf, ro);
    ggml_build_forward_expand(gf, io);
    return { feat_in, re, ie, ro, io };
}

void DenoiserGgml::Impl::preflight_ops(ggml_cgraph * gf) const {
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, node))
            throw std::runtime_error(std::string("denoiser_ggml: op '") + ggml_op_name(node->op) +
                                     "' unsupported on backend '" + ggml_backend_name(backend) + "'");
    }
}

void DenoiserGgml::Impl::run(ggml_cgraph * gf, const GraphIO & gio, const std::vector<float> & feat_host,
                             const std::vector<float> & real_in, const std::vector<float> & imag_in,
                             int L, int Bc, std::vector<float> & real_out, std::vector<float> & imag_out) {
    // Input uploads are only valid after gallocr has assigned the graph's buffers.
    if (!ggml_gallocr_alloc_graph(galloc, gf))
        throw std::runtime_error("denoiser_ggml: gallocr_alloc_graph failed");
    const size_t n = (size_t) L * spec_bins * Bc;
    ggml_backend_tensor_set(gio.feat_in, feat_host.data(), 0, feat_host.size() * sizeof(float));
    ggml_backend_tensor_set(gio.re, real_in.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(gio.ie, imag_in.data(), 0, n * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("denoiser_ggml: graph_compute failed");
    real_out.resize(n);
    imag_out.resize(n);
    ggml_backend_tensor_get(gio.ro, real_out.data(), 0, real_out.size() * sizeof(float));
    ggml_backend_tensor_get(gio.io, imag_out.data(), 0, imag_out.size() * sizeof(float));
}

DenoiserGgml::DenoiserGgml() : impl_(std::make_unique<Impl>()) {}
DenoiserGgml::~DenoiserGgml() = default;

std::unique_ptr<DenoiserGgml> DenoiserGgml::create(const DenoiserWeights & w, int n_gpu_layers, bool verbose) {
    std::unique_ptr<DenoiserGgml> e(new DenoiserGgml());
    Impl & im = *e->impl_;
    im.spec_bins = w.spec_bins; im.erb_low = w.erb_low; im.erb_high = w.erb_high;
    im.freq_comp_ratio = w.freq_comp_ratio; im.chunk_frames = w.chunk_frames;
    im.bn_eps = w.bn_eps; im.ln_eps = w.ln_eps;

    im.init_backend(n_gpu_layers, verbose);
    im.upload_weights(stage_weights(w));
    return e;
}

void DenoiserGgml::chunk_forward(const std::vector<float> & real_in, const std::vector<float> & imag_in,
                                 int L, std::vector<float> & real_out, std::vector<float> & imag_out) {
    batch_forward(real_in, imag_in, L, 1, real_out, imag_out);
}

void DenoiserGgml::batch_forward(const std::vector<float> & real_in, const std::vector<float> & imag_in,
                                 int L, int Bc, std::vector<float> & real_out, std::vector<float> & imag_out) {
    Impl & im = *impl_;

    std::vector<float> feat_host = compute_log_features(real_in, imag_in, (size_t) L * im.spec_bins * Bc);

    const size_t         buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params     gp  = { buf_size, buf.data(), true };
    ggml_context *       ctx = ggml_init(gp);
    if (!ctx) throw std::runtime_error("denoiser_ggml: ggml_init(graph) failed");
    std::unique_ptr<ggml_context, void (*)(ggml_context *)> guard(ctx, ggml_free);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

    const GraphIO gio = im.build_graph(ctx, gf, L, Bc);
    im.preflight_ops(gf);
    im.run(gf, gio, feat_host, real_in, imag_in, L, Bc, real_out, imag_out);
}

bool         DenoiserGgml::is_gpu() const { return impl_->gpu; }
const char * DenoiserGgml::backend_name() const {
    return impl_->backend ? ggml_backend_name(impl_->backend) : "(none)";
}

} // namespace tts_cpp::lavasr
