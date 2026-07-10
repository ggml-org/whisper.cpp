// Parity tests for the LavaSR denoiser ggml graph (denoiser_ggml.cpp) vs the scalar
// reference (denoiser_core.cpp).  Pure host math on the CPU backend — no model/fixture.

#include "lavasr/denoiser_ggml.h"

#include "lavasr/denoiser.h"      // internal: denoise_with_core / denoise_with_batch_core
#include "lavasr/denoiser_gguf.h" // internal: load_denoiser_gguf

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace tts_cpp::lavasr;

static int g_failures = 0;
#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);\
        }                                                                        \
    } while (0)

// Case name + fused mode, so a failure identifies which graph variant drifted.
static const char * fmsg(const char * name, bool fused) {
    static char b[160];
    std::snprintf(b, sizeof b, "%s [fused=%d]", name, (int) fused);
    return b;
}

// Scalar PyTorch GRU (gate order r,z,n), zero init state — a verbatim copy of
// denoiser_core.cpp gru_seq, the golden reference.  Wih:[3H,I], Whh:[3H,H].
static std::vector<float> ref_gru_seq(const std::vector<float> & x, int L, int I, int H,
                                      const std::vector<float> & Wih, const std::vector<float> & Whh,
                                      const std::vector<float> & Bih, const std::vector<float> & Bhh) {
    std::vector<float> h(H, 0.0f), ys((size_t) L * H, 0.0f), gi(3 * H), gh(3 * H);
    for (int t = 0; t < L; t++) {
        const float * xt = x.data() + (size_t) t * I;
        for (int o = 0; o < 3 * H; o++) {
            float acc = Bih[o];
            for (int i = 0; i < I; i++) acc += Wih[(size_t) o * I + i] * xt[i];
            gi[o] = acc;
        }
        for (int o = 0; o < 3 * H; o++) {
            float acc = Bhh[o];
            for (int i = 0; i < H; i++) acc += Whh[(size_t) o * H + i] * h[i];
            gh[o] = acc;
        }
        for (int k = 0; k < H; k++) {
            const float r = 1.0f / (1.0f + std::exp(-(gi[k] + gh[k])));
            const float z = 1.0f / (1.0f + std::exp(-(gi[H + k] + gh[H + k])));
            const float n = std::tanh(gi[2 * H + k] + r * gh[2 * H + k]);
            h[k]                  = (1.0f - z) * n + z * h[k];
            ys[(size_t) t * H + k] = h[k];
        }
    }
    return ys;
}

// Backward reference matching gru_bi's convention: run on reversed input, then
// index b[L-1-t] -> hidden at original position t.
static std::vector<float> ref_gru_rev(const std::vector<float> & x, int L, int I, int H,
                                      const std::vector<float> & Wih, const std::vector<float> & Whh,
                                      const std::vector<float> & Bih, const std::vector<float> & Bhh) {
    std::vector<float> xr((size_t) L * I);
    for (int t = 0; t < L; t++)
        for (int i = 0; i < I; i++) xr[(size_t) t * I + i] = x[(size_t) (L - 1 - t) * I + i];
    std::vector<float> b = ref_gru_seq(xr, L, I, H, Wih, Whh, Bih, Bhh);
    std::vector<float> y((size_t) L * H);
    for (int t = 0; t < L; t++)
        for (int k = 0; k < H; k++) y[(size_t) t * H + k] = b[(size_t) (L - 1 - t) * H + k];
    return y;
}

// Run detail::gru_batched on the CPU backend and return y flat [L][B][H]
// (index (l*B+b)*H+k), matching the ggml [H,B,L] memory layout.
static std::vector<float> run_batched(const std::vector<float> & x, int I, int B, int L, int H,
                                      const std::vector<float> & Wih, const std::vector<float> & Whh,
                                      const std::vector<float> & Bih, const std::vector<float> & Bhh,
                                      bool reverse, bool bidir,
                                      const std::vector<float> & WihR, const std::vector<float> & WhhR,
                                      const std::vector<float> & BihR, const std::vector<float> & BhhR,
                                      bool fused) {
    ggml_init_params p{ (size_t) 128 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, I, B, L);
    ggml_tensor *    tWih = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I, 3 * H);
    ggml_tensor *    tWhh = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 3 * H);
    ggml_tensor *    tBih = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
    ggml_tensor *    tBhh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::memcpy(tWih->data, Wih.data(), ggml_nbytes(tWih));
    std::memcpy(tWhh->data, Whh.data(), ggml_nbytes(tWhh));
    std::memcpy(tBih->data, Bih.data(), ggml_nbytes(tBih));
    std::memcpy(tBhh->data, Bhh.data(), ggml_nbytes(tBhh));

    ggml_tensor * y = detail::gru_batched(ctx, tx, tWih, tWhh, tBih, tBhh, reverse, fused);
    if (bidir) {
        ggml_tensor * tWihR = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I, 3 * H);
        ggml_tensor * tWhhR = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 3 * H);
        ggml_tensor * tBihR = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
        ggml_tensor * tBhhR = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
        std::memcpy(tWihR->data, WihR.data(), ggml_nbytes(tWihR));
        std::memcpy(tWhhR->data, WhhR.data(), ggml_nbytes(tWhhR));
        std::memcpy(tBihR->data, BihR.data(), ggml_nbytes(tBihR));
        std::memcpy(tBhhR->data, BhhR.data(), ggml_nbytes(tBhhR));
        ggml_tensor * yr = detail::gru_batched(ctx, tx, tWihR, tWhhR, tBihR, tBhhR, true, fused);
        y                = ggml_concat(ctx, y, yr, 0); // [2H, B, L]
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void rand_fill(std::vector<float> & v, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d(-0.8f, 0.8f);
    for (float & x : v) x = d(rng);
}

static void test_gru_batched() {
    const int I = 3, H = 4, L = 5, B = 2;
    std::mt19937 rng(1234);
    std::vector<float> Wih(3 * H * I), Whh(3 * H * H), Bih(3 * H), Bhh(3 * H);
    std::vector<float> WihR(3 * H * I), WhhR(3 * H * H), BihR(3 * H), BhhR(3 * H);
    std::vector<float> x((size_t) L * B * I); // [L][B][I]
    rand_fill(Wih, rng); rand_fill(Whh, rng); rand_fill(Bih, rng); rand_fill(Bhh, rng);
    rand_fill(WihR, rng); rand_fill(WhhR, rng); rand_fill(BihR, rng); rand_fill(BhhR, rng);
    rand_fill(x, rng);

    auto x_of = [&](int b) {
        std::vector<float> xb((size_t) L * I);
        for (int l = 0; l < L; l++)
            for (int i = 0; i < I; i++) xb[(size_t) l * I + i] = x[((size_t) l * B + b) * I + i];
        return xb;
    };

    // Forward: y[(l*B+b)*H+k] must equal per-sequence ref_gru_seq.
    for (bool fused : { true, false }) {
        std::vector<float> y = run_batched(x, I, B, L, H, Wih, Whh, Bih, Bhh, false, false,
                                           {}, {}, {}, {}, fused);
        float m = 0.0f;
        for (int b = 0; b < B; b++) {
            std::vector<float> ref = ref_gru_seq(x_of(b), L, I, H, Wih, Whh, Bih, Bhh);
            for (int l = 0; l < L; l++)
                for (int k = 0; k < H; k++)
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * H + k] - ref[(size_t) l * H + k]));
        }
        CHECK(m < 1e-5f, fmsg("gru_batched forward matches scalar gru_seq", fused));
    }
    // Reverse: matches gru_bi's backward convention (hidden at original position).
    for (bool fused : { true, false }) {
        std::vector<float> y = run_batched(x, I, B, L, H, Wih, Whh, Bih, Bhh, true, false,
                                           {}, {}, {}, {}, fused);
        float m = 0.0f;
        for (int b = 0; b < B; b++) {
            std::vector<float> ref = ref_gru_rev(x_of(b), L, I, H, Wih, Whh, Bih, Bhh);
            for (int l = 0; l < L; l++)
                for (int k = 0; k < H; k++)
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * H + k] - ref[(size_t) l * H + k]));
        }
        CHECK(m < 1e-5f, fmsg("gru_batched reverse matches scalar backward", fused));
    }
    // BiGRU: concat(forward, reverse) along ne0 == scalar [fwd | bwd].
    for (bool fused : { true, false }) {
        std::vector<float> y = run_batched(x, I, B, L, H, Wih, Whh, Bih, Bhh, false, true,
                                           WihR, WhhR, BihR, BhhR, fused);
        float m = 0.0f;
        for (int b = 0; b < B; b++) {
            std::vector<float> f = ref_gru_seq(x_of(b), L, I, H, Wih, Whh, Bih, Bhh);
            std::vector<float> r = ref_gru_rev(x_of(b), L, I, H, WihR, WhhR, BihR, BhhR);
            for (int l = 0; l < L; l++) {
                for (int k = 0; k < H; k++) {
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * 2 * H + k] - f[(size_t) l * H + k]));
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * 2 * H + H + k] - r[(size_t) l * H + k]));
                }
            }
        }
        CHECK(m < 1e-5f, fmsg("gru_batched BiGRU concat matches scalar gru_bi", fused));
    }
}

// Scalar conv2d — verbatim math of denoiser_core.cpp Runner::conv2d on flat
// [C][T][F] buffers (which are byte-identical to ggml [F,T,C]).  W:[Cout,Cin/g,kt,kf].
static std::vector<float> ref_conv2d(const std::vector<float> & x, int Cin, int T, int F,
                                     const std::vector<float> & W, int Cout, int Cing, int kt, int kf,
                                     const std::vector<float> * bias, int stride_f, int pad_f, int groups) {
    const int pad_t_top = kt - 1;
    const int Fout      = (F + 2 * pad_f - kf) / stride_f + 1;
    std::vector<float> out((size_t) Cout * T * Fout, 0.0f);
    auto X = [&](int c, int t, int f) { return x[((size_t) c * T + t) * F + f]; };
    auto O = [&](int c, int t, int f) -> float & { return out[((size_t) c * T + t) * Fout + f]; };
    const int og = Cout / groups;
    for (int oc = 0; oc < Cout; oc++) {
        const int   grp = oc / og, ic0 = grp * Cing;
        const float b   = bias ? (*bias)[oc] : 0.0f;
        for (int t = 0; t < T; t++)
            for (int f = 0; f < Fout; f++) O(oc, t, f) = b;
        for (int ici = 0; ici < Cing; ici++) {
            const int     ic = ic0 + ici;
            const float * wk = W.data() + (((size_t) oc * Cing) + ici) * kt * kf;
            for (int it = 0; it < kt; it++)
                for (int jf = 0; jf < kf; jf++) {
                    const float wv = wk[it * kf + jf];
                    for (int t = 0; t < T; t++) {
                        const int ti = t + it - pad_t_top;
                        if (ti < 0 || ti >= T) continue;
                        for (int f = 0; f < Fout; f++) {
                            const int fi = f * stride_f + jf - pad_f;
                            if (fi < 0 || fi >= F) continue;
                            O(oc, t, f) += wv * X(ic, ti, fi);
                        }
                    }
                }
        }
    }
    return out;
}

static std::vector<float> run_conv2d(const std::vector<float> & x, int Cin, int T, int F,
                                     const std::vector<float> & W, int Cout, int Cing, int kt, int kf,
                                     const std::vector<float> & bias, int stride_f, int pad_f, int groups, bool fused) {
    ggml_init_params p{ (size_t) 128 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, Cin);
    ggml_tensor *    tW  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kf, kt, Cing, Cout);
    ggml_tensor *    tb  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Cout);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::memcpy(tW->data, W.data(), ggml_nbytes(tW));
    std::memcpy(tb->data, bias.data(), ggml_nbytes(tb));
    ggml_tensor * y  = detail::conv2d(ctx, tx, tW, tb, stride_f, pad_f, groups, fused);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void test_conv2d() {
    std::mt19937 rng(777);
    struct Cfg { int Cin, Cout, kt, kf, stride, pad, groups; const char * name; };
    const Cfg cfgs[] = {
        { 3, 12, 3, 3, 2, 1, 1, "conv2d groups=1 stride=2" },       // encoder block 0
        { 12, 24, 2, 3, 2, 1, 2, "conv2d groups=2 stride=2" },      // grouped
        { 8, 8, 1, 5, 1, 2, 8, "conv2d depthwise stride=1" },       // depthwise (groups==Cout)
    };
    const int T = 6, F = 9;
    for (const Cfg & c : cfgs) {
        const int Cing = c.Cin / c.groups;
        std::vector<float> x((size_t) c.Cin * T * F), W((size_t) c.Cout * Cing * c.kt * c.kf), b(c.Cout);
        rand_fill(x, rng); rand_fill(W, rng); rand_fill(b, rng);
        std::vector<float> ref = ref_conv2d(x, c.Cin, T, F, W, c.Cout, Cing, c.kt, c.kf, &b, c.stride, c.pad, c.groups);
        for (bool fused : { true, false }) {
            std::vector<float> got = run_conv2d(x, c.Cin, T, F, W, c.Cout, Cing, c.kt, c.kf, b, c.stride, c.pad, c.groups, fused);
            float m = 0.0f;
            CHECK(got.size() == ref.size(), fmsg(c.name, fused));
            for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
            CHECK(m < 1e-4f, fmsg(c.name, fused));
        }
    }
}

// Scalar transposed conv — verbatim math of denoiser_core.cpp Runner::conv_transpose2d.
// W (PyTorch transpose kernel): [Cin, Cout/g, kt, kf].
static std::vector<float> ref_conv_transpose2d(const std::vector<float> & x, int Cin, int T, int F,
                                               const std::vector<float> & W, int Coutg, int kt, int kf,
                                               const std::vector<float> * bias, int stride_f, int pad_f, int groups) {
    const int Cout = Coutg * groups;
    const int pt_top = kt - 1, pad_t = kt - 1;
    const int Tp = T + pt_top, Tfull = (Tp - 1) + kt, Ffull = (F - 1) * stride_f + kf;
    std::vector<float> full((size_t) Cout * Tfull * Ffull, 0.0f);
    auto X  = [&](int c, int t, int f) { return x[((size_t) c * T + t) * F + f]; };
    auto FL = [&](int c, int t, int f) -> float & { return full[((size_t) c * Tfull + t) * Ffull + f]; };
    const int ing = Cin / groups;
    for (int ic = 0; ic < Cin; ic++) {
        const int grp = ic / ing, oc0 = grp * Coutg;
        for (int ocg = 0; ocg < Coutg; ocg++) {
            const int     oc = oc0 + ocg;
            const float * wk = W.data() + (((size_t) ic * Coutg) + ocg) * kt * kf;
            for (int t = 0; t < Tp; t++) {
                const int xt = t - pt_top;
                for (int f = 0; f < F; f++) {
                    const float v = (xt < 0) ? 0.0f : X(ic, xt, f);
                    for (int it = 0; it < kt; it++)
                        for (int jf = 0; jf < kf; jf++) FL(oc, t + it, f * stride_f + jf) += v * wk[it * kf + jf];
                }
            }
        }
    }
    if (bias)
        for (int oc = 0; oc < Cout; oc++) {
            const float b = (*bias)[oc];
            for (int t = 0; t < Tfull; t++)
                for (int f = 0; f < Ffull; f++) FL(oc, t, f) += b;
        }
    const int          Tout = Tfull - 2 * pad_t, Fout = Ffull - 2 * pad_f;
    std::vector<float> out((size_t) Cout * Tout * Fout, 0.0f);
    for (int oc = 0; oc < Cout; oc++)
        for (int t = 0; t < Tout; t++)
            for (int f = 0; f < Fout; f++) out[((size_t) oc * Tout + t) * Fout + f] = FL(oc, t + pad_t, f + pad_f);
    return out;
}

// Host reindex: PyTorch transpose kernel Wt[Cin,Cout/g,kt,kf] -> regular-conv Wc
// [Cout,Cin/g,kt,kf] with IC<->OC swap and kt/kf flip (the create()-time transform).
static std::vector<float> reindex_ct(const std::vector<float> & Wt, int Cin, int Coutg, int kt, int kf, int groups) {
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

static std::vector<float> run_conv_transpose2d(const std::vector<float> & x, int Cin, int T, int F,
                                               const std::vector<float> & Wc, int Cout, int ing, int kt, int kf,
                                               const std::vector<float> & bias, int stride_f, int pad_f, int groups, bool fused) {
    ggml_init_params p{ (size_t) 128 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, Cin);
    ggml_tensor *    tW  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kf, kt, ing, Cout);
    ggml_tensor *    tb  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Cout);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::memcpy(tW->data, Wc.data(), ggml_nbytes(tW));
    std::memcpy(tb->data, bias.data(), ggml_nbytes(tb));
    ggml_tensor * y  = detail::conv_transpose2d(ctx, tx, tW, tb, stride_f, pad_f, groups, fused);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void test_conv_transpose2d() {
    std::mt19937 rng(555);
    struct Cfg { int Cin, Coutg, kt, kf, stride, pad, groups; const char * name; };
    const Cfg cfgs[] = {
        { 8, 1, 2, 3, 2, 1, 8, "convT depthwise stride=2" },  // decoder i=1 dconv
        { 8, 1, 1, 5, 1, 2, 8, "convT depthwise stride=1" },  // decoder i=4 dconv
        { 8, 4, 3, 3, 2, 1, 1, "convT groups=1 stride=2" },   // decoder i=0 final ops.1
    };
    const int T = 6, F = 7;
    for (const Cfg & c : cfgs) {
        const int Cout = c.Coutg * c.groups, ing = c.Cin / c.groups;
        std::vector<float> x((size_t) c.Cin * T * F), Wt((size_t) c.Cin * c.Coutg * c.kt * c.kf), b(Cout);
        rand_fill(x, rng); rand_fill(Wt, rng); rand_fill(b, rng);
        std::vector<float> ref = ref_conv_transpose2d(x, c.Cin, T, F, Wt, c.Coutg, c.kt, c.kf, &b, c.stride, c.pad, c.groups);
        std::vector<float> Wc  = reindex_ct(Wt, c.Cin, c.Coutg, c.kt, c.kf, c.groups);
        for (bool fused : { true, false }) {
            std::vector<float> got = run_conv_transpose2d(x, c.Cin, T, F, Wc, Cout, ing, c.kt, c.kf, b, c.stride, c.pad, c.groups, fused);
            float m = 0.0f;
            CHECK(got.size() == ref.size(), fmsg(c.name, fused));
            for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
            CHECK(m < 1e-4f, fmsg(c.name, fused));
        }
    }
}

static void test_affine_prelu() {
    std::mt19937 rng(4242);
    const int F = 7, T = 5, C = 6;
    std::vector<float> x((size_t) C * T * F), aw((size_t) C * F), ab((size_t) C * F), sl(C);
    rand_fill(x, rng); rand_fill(aw, rng); rand_fill(ab, rng); rand_fill(sl, rng);
    std::vector<float> ref(x.size());
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                const size_t i = ((size_t) c * T + t) * F + f;
                const float  v = x[i];
                ref[i] = (v * aw[(size_t) c * F + f] + ab[(size_t) c * F + f]) + (v > 0.0f ? v : sl[c] * v);
            }
    for (bool fused : { true, false }) {
        ggml_init_params p{ (size_t) 32 * 1024 * 1024, nullptr, false };
        ggml_context *   ctx = ggml_init(p);
        ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
        ggml_tensor *    ta  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, C);
        ggml_tensor *    tb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, C);
        ggml_tensor *    ts  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
        std::memcpy(ta->data, aw.data(), ggml_nbytes(ta));
        std::memcpy(tb->data, ab.data(), ggml_nbytes(tb));
        std::memcpy(ts->data, sl.data(), ggml_nbytes(ts));
        ggml_tensor * y  = detail::affine_prelu(ctx, tx, ta, tb, ts, fused);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        float m = 0.0f;
        CHECK((size_t) ggml_nelements(y) == ref.size(), fmsg("affine_prelu output size", fused));
        const float * out = (const float *) y->data;
        for (size_t i = 0; i < ref.size(); i++) m = std::max(m, std::fabs(out[i] - ref[i]));
        CHECK(m < 1e-5f, fmsg("affine_prelu matches scalar reference", fused));
        ggml_free(ctx);
    }
}

static void test_shuffle2() {
    std::mt19937 rng(5151);
    const int F = 5, T = 4, C = 6, half = C / 2;
    std::vector<float> x((size_t) C * T * F);
    rand_fill(x, rng);
    std::vector<float> ref(x.size());
    for (int c = 0; c < half; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                ref[((size_t) (2 * c) * T + t) * F + f]     = x[((size_t) c * T + t) * F + f];
                ref[((size_t) (2 * c + 1) * T + t) * F + f] = x[((size_t) (half + c) * T + t) * F + f];
            }
    for (bool fused : { true, false }) {
        ggml_init_params p{ (size_t) 32 * 1024 * 1024, nullptr, false };
        ggml_context *   ctx = ggml_init(p);
        ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
        std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
        ggml_tensor * y  = detail::shuffle2(ctx, tx, fused);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        float m = -1.0f;
        CHECK((size_t) ggml_nelements(y) == ref.size(), fmsg("shuffle2 output size", fused));
        const float * out = (const float *) y->data;
        m = 0.0f;
        for (size_t i = 0; i < ref.size(); i++) m = std::max(m, std::fabs(out[i] - ref[i]));
        CHECK(m == 0.0f, fmsg("shuffle2 is an exact channel interleave", fused));
        ggml_free(ctx);
    }
}

// ---- DPGRNN parity: scalar reference (denoiser_core.cpp dpgrnn) vs detail::dpgrnn ----
#include <map>
#include <string>

struct WBag {
    std::map<std::string, std::vector<float>> data;
    std::map<std::string, std::vector<int>>   shape; // C-order
    const std::vector<float> & d(const std::string & n) const { return data.at(n); }
    bool                       has(const std::string & n) const { return data.count(n) != 0; }
    int                        dim(const std::string & n, int i) const { return shape.at(n)[i]; }
};

static std::vector<float> ref_linear2d(const std::vector<float> & x, int N, int c_in,
                                       const std::vector<float> & W, int c_out, const std::vector<float> * b) {
    std::vector<float> y((size_t) N * c_out, 0.0f);
    for (int n = 0; n < N; n++)
        for (int o = 0; o < c_out; o++) {
            float acc = b ? (*b)[o] : 0.0f;
            for (int i = 0; i < c_in; i++) acc += x[(size_t) n * c_in + i] * W[(size_t) o * c_in + i];
            y[(size_t) n * c_out + o] = acc;
        }
    return y;
}

static void ref_layernorm_fc(std::vector<float> & x, int T, int Fw, int C,
                             const std::vector<float> & g, const std::vector<float> & b, float eps) {
    const int n = Fw * C;
    for (int t = 0; t < T; t++) {
        float * row  = x.data() + (size_t) t * n;
        double  mean = 0.0;
        for (int i = 0; i < n; i++) mean += row[i];
        mean /= n;
        double var = 0.0;
        for (int i = 0; i < n; i++) { double dd = row[i] - mean; var += dd * dd; }
        var /= n;
        const float inv = 1.0f / std::sqrt((float) var + eps);
        for (int i = 0; i < n; i++) row[i] = ((float) (row[i] - mean) * inv) * g[i] + b[i];
    }
}

static std::vector<float> ref_grnn(const std::vector<float> & x, int L, int I,
                                   const std::string & p, const WBag & W, bool bidir) {
    const int          half = I / 2;
    std::vector<float> x1((size_t) L * half), x2((size_t) L * half);
    for (int t = 0; t < L; t++)
        for (int i = 0; i < half; i++) {
            x1[(size_t) t * half + i] = x[(size_t) t * I + i];
            x2[(size_t) t * half + i] = x[(size_t) t * I + half + i];
        }
    auto run = [&](const std::vector<float> & xi, const std::string & pp) {
        const int H = W.dim(pp + ".weight_hh_l0", 1);
        if (!bidir)
            return ref_gru_seq(xi, L, half, H, W.d(pp + ".weight_ih_l0"), W.d(pp + ".weight_hh_l0"),
                               W.d(pp + ".bias_ih_l0"), W.d(pp + ".bias_hh_l0"));
        std::vector<float> f = ref_gru_seq(xi, L, half, H, W.d(pp + ".weight_ih_l0"), W.d(pp + ".weight_hh_l0"),
                                           W.d(pp + ".bias_ih_l0"), W.d(pp + ".bias_hh_l0"));
        std::vector<float> r = ref_gru_rev(xi, L, half, H, W.d(pp + ".weight_ih_l0_reverse"),
                                           W.d(pp + ".weight_hh_l0_reverse"), W.d(pp + ".bias_ih_l0_reverse"),
                                           W.d(pp + ".bias_hh_l0_reverse"));
        std::vector<float> y((size_t) L * 2 * H);
        for (int t = 0; t < L; t++)
            for (int k = 0; k < H; k++) {
                y[(size_t) t * 2 * H + k]     = f[(size_t) t * H + k];
                y[(size_t) t * 2 * H + H + k] = r[(size_t) t * H + k];
            }
        return y;
    };
    std::vector<float> y1 = run(x1, p + ".rnn1"), y2 = run(x2, p + ".rnn2");
    const int          h1 = (int) (y1.size() / L), h2 = (int) (y2.size() / L);
    std::vector<float> y((size_t) L * (h1 + h2));
    for (int t = 0; t < L; t++) {
        for (int i = 0; i < h1; i++) y[(size_t) t * (h1 + h2) + i]      = y1[(size_t) t * h1 + i];
        for (int i = 0; i < h2; i++) y[(size_t) t * (h1 + h2) + h1 + i] = y2[(size_t) t * h2 + i];
    }
    return y;
}

static std::vector<float> ref_dpgrnn(const std::vector<float> & x, int C, int T, int F,
                                     const std::string & pre, const WBag & W, float eps) {
    auto fcb = [&](const std::string & n) { return W.has(n) ? &W.d(n) : nullptr; };
    std::vector<float> xt((size_t) T * F * C);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            for (int c = 0; c < C; c++) xt[((size_t) t * F + f) * C + c] = x[((size_t) c * T + t) * F + f];
    std::vector<float> intra((size_t) T * F * C);
    for (int t = 0; t < T; t++) {
        std::vector<float> seq(xt.begin() + (size_t) t * F * C, xt.begin() + (size_t) (t + 1) * F * C);
        std::vector<float> y  = ref_grnn(seq, F, C, pre + ".intra_rnn", W, true);
        const int          hy = (int) (y.size() / F);
        y = ref_linear2d(y, F, hy, W.d(pre + ".intra_fc.weight"), C, fcb(pre + ".intra_fc.bias"));
        std::copy(y.begin(), y.end(), intra.begin() + (size_t) t * F * C);
    }
    ref_layernorm_fc(intra, T, F, C, W.d(pre + ".intra_ln.weight"), W.d(pre + ".intra_ln.bias"), eps);
    for (size_t i = 0; i < intra.size(); i++) intra[i] += xt[i];
    std::vector<float> inter((size_t) T * F * C);
    for (int f = 0; f < F; f++) {
        std::vector<float> seq((size_t) T * C);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) seq[(size_t) t * C + c] = intra[((size_t) t * F + f) * C + c];
        std::vector<float> y  = ref_grnn(seq, T, C, pre + ".inter_rnn", W, false);
        const int          hy = (int) (y.size() / T);
        y = ref_linear2d(y, T, hy, W.d(pre + ".inter_fc.weight"), C, fcb(pre + ".inter_fc.bias"));
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) inter[((size_t) t * F + f) * C + c] = y[(size_t) t * C + c];
    }
    ref_layernorm_fc(inter, T, F, C, W.d(pre + ".inter_ln.weight"), W.d(pre + ".inter_ln.bias"), eps);
    for (size_t i = 0; i < inter.size(); i++) inter[i] += intra[i];
    std::vector<float> out((size_t) C * T * F);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            for (int c = 0; c < C; c++) out[((size_t) c * T + t) * F + f] = inter[((size_t) t * F + f) * C + c];
    return out;
}

static std::vector<float> run_dpgrnn(const std::vector<float> & x, int C, int T, int F,
                                     const WBag & bag, float eps, bool fused) {
    ggml_init_params p{ (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::map<std::string, ggml_tensor *> tmap;
    for (const auto & kv : bag.data) {
        const std::vector<int> & shp = bag.shape.at(kv.first);
        int64_t                  ne[4] = { 1, 1, 1, 1 };
        const int                nd    = (int) shp.size();
        for (int i = 0; i < nd; i++) ne[i] = shp[nd - 1 - i];
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        std::memcpy(t->data, kv.second.data(), ggml_nbytes(t));
        tmap[kv.first] = t;
    }
    detail::WResolver Wr = [&](const std::string & n) -> ggml_tensor * {
        auto it = tmap.find(n);
        return it == tmap.end() ? nullptr : it->second;
    };
    ggml_tensor * y  = detail::dpgrnn(ctx, tx, "dp", Wr, eps, fused);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void test_dpgrnn() {
    const int    C = 4, F = 5, T = 3, Hg = 3, Hi = 4;
    const float  eps = 1e-8f;
    std::mt19937 rng(2468);
    WBag         bag;
    auto add = [&](const std::string & n, std::vector<int> shp) {
        size_t sz = 1;
        for (int s : shp) sz *= (size_t) s;
        std::vector<float> v(sz);
        rand_fill(v, rng);
        bag.data[n]  = v;
        bag.shape[n] = std::move(shp);
    };
    auto add_gru = [&](const std::string & p, int I, int H, bool rev) {
        add(p + ".weight_ih_l0", { 3 * H, I }); add(p + ".weight_hh_l0", { 3 * H, H });
        add(p + ".bias_ih_l0", { 3 * H }); add(p + ".bias_hh_l0", { 3 * H });
        if (rev) {
            add(p + ".weight_ih_l0_reverse", { 3 * H, I }); add(p + ".weight_hh_l0_reverse", { 3 * H, H });
            add(p + ".bias_ih_l0_reverse", { 3 * H }); add(p + ".bias_hh_l0_reverse", { 3 * H });
        }
    };
    add_gru("dp.intra_rnn.rnn1", C / 2, Hg, true);
    add_gru("dp.intra_rnn.rnn2", C / 2, Hg, true);
    add("dp.intra_fc.weight", { C, 4 * Hg }); add("dp.intra_fc.bias", { C });
    add("dp.intra_ln.weight", { F * C }); add("dp.intra_ln.bias", { F * C });
    add_gru("dp.inter_rnn.rnn1", C / 2, Hi, false);
    add_gru("dp.inter_rnn.rnn2", C / 2, Hi, false);
    add("dp.inter_fc.weight", { C, 2 * Hi }); add("dp.inter_fc.bias", { C });
    add("dp.inter_ln.weight", { F * C }); add("dp.inter_ln.bias", { F * C });

    std::vector<float> x((size_t) C * T * F);
    rand_fill(x, rng);
    std::vector<float> ref = ref_dpgrnn(x, C, T, F, "dp", bag, eps);
    for (bool fused : { true, false }) {
        std::vector<float> got = run_dpgrnn(x, C, T, F, bag, eps, fused);
        float m = 0.0f;
        CHECK(got.size() == ref.size(), fmsg("dpgrnn output size", fused));
        for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
        CHECK(m < 2e-4f, fmsg("dpgrnn matches scalar denoiser_core dpgrnn", fused));
    }
}

// Generic CPU-backend runner for a resolver-driven graph builder.
static std::vector<float> run_graph(const std::vector<float> & x, int F, int T, int C, const WBag & bag,
                                    const std::function<ggml_tensor *(ggml_context *, ggml_tensor *, const detail::WResolver &)> & build) {
    ggml_init_params p{ (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::map<std::string, ggml_tensor *> tmap;
    for (const auto & kv : bag.data) {
        const std::vector<int> & shp = bag.shape.at(kv.first);
        int64_t                  ne[4] = { 1, 1, 1, 1 };
        const int                nd    = (int) shp.size();
        for (int i = 0; i < nd; i++) ne[i] = shp[nd - 1 - i];
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        std::memcpy(t->data, kv.second.data(), ggml_nbytes(t));
        tmap[kv.first] = t;
    }
    detail::WResolver Wr = [&](const std::string & n) -> ggml_tensor * {
        auto it = tmap.find(n);
        return it == tmap.end() ? nullptr : it->second;
    };
    ggml_tensor * y  = build(ctx, tx, Wr);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static std::vector<float> ref_ctfa(const std::vector<float> & x, int C, int T, int F,
                                   const std::string & p, const WBag & W, int r) {
    auto X   = [&](int c, int t, int f) { return x[((size_t) c * T + t) * F + f]; };
    auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    auto fcb = [&](const std::string & n) { return W.has(n) ? &W.d(n) : nullptr; };
    // TA
    std::vector<float> zt((size_t) T * C, 0.0f);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) {
            double s = 0.0;
            for (int f = 0; f < F; f++) { float v = X(c, t, f); s += (double) v * v; }
            zt[(size_t) t * C + c] = (float) (s / F);
        }
    const int          Hta = W.dim(p + ".ta_gru.weight_hh_l0", 1);
    std::vector<float> at  = ref_gru_seq(zt, T, C, Hta, W.d(p + ".ta_gru.weight_ih_l0"), W.d(p + ".ta_gru.weight_hh_l0"),
                                         W.d(p + ".ta_gru.bias_ih_l0"), W.d(p + ".ta_gru.bias_hh_l0"));
    at = ref_linear2d(at, T, Hta, W.d(p + ".ta_fc.weight"), C, fcb(p + ".ta_fc.bias"));
    for (float & v : at) v = sig(v);
    // FA
    const int          pad_len = (r - F % r) % r, Fp = F + pad_len, Hh = Fp / r;
    const int          Hfa     = W.dim(p + ".fa.gru.weight_hh_l0", 1);
    std::vector<float> af((size_t) T * F, 0.0f);
    for (int t = 0; t < T; t++) {
        std::vector<float> row((size_t) Hh * r, 0.0f);
        for (int h = 0; h < Hh; h++)
            for (int rr = 0; rr < r; rr++) {
                const int f = h * r + rr;
                double    s = 0.0;
                if (f < F) { for (int c = 0; c < C; c++) { float v = X(c, t, f); s += (double) v * v; } s /= C; }
                row[(size_t) h * r + rr] = (float) s;
            }
        std::vector<float> fwd = ref_gru_seq(row, Hh, r, Hfa, W.d(p + ".fa.gru.weight_ih_l0"), W.d(p + ".fa.gru.weight_hh_l0"),
                                             W.d(p + ".fa.gru.bias_ih_l0"), W.d(p + ".fa.gru.bias_hh_l0"));
        std::vector<float> rev = ref_gru_rev(row, Hh, r, Hfa, W.d(p + ".fa.gru.weight_ih_l0_reverse"), W.d(p + ".fa.gru.weight_hh_l0_reverse"),
                                             W.d(p + ".fa.gru.bias_ih_l0_reverse"), W.d(p + ".fa.gru.bias_hh_l0_reverse"));
        std::vector<float> y((size_t) Hh * 2 * Hfa);
        for (int h = 0; h < Hh; h++)
            for (int k = 0; k < Hfa; k++) {
                y[(size_t) h * 2 * Hfa + k]       = fwd[(size_t) h * Hfa + k];
                y[(size_t) h * 2 * Hfa + Hfa + k] = rev[(size_t) h * Hfa + k];
            }
        y = ref_linear2d(y, Hh, 2 * Hfa, W.d(p + ".fa.fc.weight"), r, fcb(p + ".fa.fc.bias"));
        for (int h = 0; h < Hh; h++)
            for (int rr = 0; rr < r; rr++) {
                const int f = h * r + rr;
                if (f < F) af[(size_t) t * F + f] = sig(y[(size_t) h * r + rr]);
            }
    }
    std::vector<float> o((size_t) C * T * F);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                o[((size_t) c * T + t) * F + f] = at[(size_t) t * C + c] * X(c, t, f) * af[(size_t) t * F + f];
    return o;
}

static void test_ctfa() {
    const int    C = 4, T = 3, F = 7, r = 4;
    std::mt19937 rng(13579);
    WBag         bag;
    auto add = [&](const std::string & n, std::vector<int> shp) {
        size_t sz = 1;
        for (int s : shp) sz *= (size_t) s;
        std::vector<float> v(sz);
        rand_fill(v, rng);
        bag.data[n]  = v;
        bag.shape[n] = std::move(shp);
    };
    const int Hta = 2 * C, Hfa = r;
    add("ct.ta_gru.weight_ih_l0", { 3 * Hta, C }); add("ct.ta_gru.weight_hh_l0", { 3 * Hta, Hta });
    add("ct.ta_gru.bias_ih_l0", { 3 * Hta }); add("ct.ta_gru.bias_hh_l0", { 3 * Hta });
    add("ct.ta_fc.weight", { C, Hta }); add("ct.ta_fc.bias", { C });
    for (const char * suf : { "", "_reverse" }) {
        add(std::string("ct.fa.gru.weight_ih_l0") + suf, { 3 * Hfa, r });
        add(std::string("ct.fa.gru.weight_hh_l0") + suf, { 3 * Hfa, Hfa });
        add(std::string("ct.fa.gru.bias_ih_l0") + suf, { 3 * Hfa });
        add(std::string("ct.fa.gru.bias_hh_l0") + suf, { 3 * Hfa });
    }
    add("ct.fa.fc.weight", { r, 2 * Hfa }); add("ct.fa.fc.bias", { r });

    std::vector<float> x((size_t) C * T * F);
    rand_fill(x, rng);
    std::vector<float> ref = ref_ctfa(x, C, T, F, "ct", bag, r);
    for (bool fused : { true, false }) {
        std::vector<float> got = run_graph(x, F, T, C, bag,
                                           [&](ggml_context * ctx, ggml_tensor * tx, const detail::WResolver & Wr) {
                                               return detail::ctfa(ctx, tx, "ct", Wr, r, fused);
                                           });
        float m = 0.0f;
        CHECK(got.size() == ref.size(), fmsg("ctfa output size", fused));
        for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
        CHECK(m < 2e-4f, fmsg("ctfa matches scalar denoiser_core ctfa", fused));
    }
}

// End-to-end parity: DenoiserGgml::chunk_forward (ggml-CPU) vs scalar
// denoiser_net_forward on one zero-state chunk from the real GGUF.
static void test_e2e(const char * gguf_path) {
    tts_cpp::lavasr::DenoiserWeights w;
    std::string                      err;
    if (!tts_cpp::lavasr::load_denoiser_gguf(gguf_path, w, &err)) {
        std::fprintf(stderr, "SKIP e2e: cannot load %s (%s)\n", gguf_path, err.c_str());
        return;
    }
    const int    L = w.chunk_frames, F = w.spec_bins;
    std::mt19937 rng(20260708);
    auto         df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) L * F), im((size_t) L * F);
    df(re); df(im);
    std::vector<float> sr, si;
    tts_cpp::lavasr::denoiser_net_forward(w, re, im, L, sr, si);
    auto               g = tts_cpp::lavasr::DenoiserGgml::create(w, -1);
    std::vector<float> gr, gi;
    g->chunk_forward(re, im, L, gr, gi);
    auto rel = [](const std::vector<float> & a, const std::vector<float> & b, float & mx) {
        double num = 0, den = 0;
        mx = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float d = std::fabs(a[i] - b[i]);
            mx      = std::max(mx, d);
            num += (double) d * d;
            den += (double) a[i] * a[i];
        }
        return (float) std::sqrt(num / std::max(den, 1e-20));
    };
    float mr, mi;
    float rr = rel(sr, gr, mr), ri = rel(si, gi, mi);
    std::printf("e2e: real max_abs=%.3e nrmse=%.3e | imag max_abs=%.3e nrmse=%.3e\n", mr, rr, mi, ri);
    CHECK(gr.size() == sr.size() && gi.size() == si.size(), "e2e output size matches scalar");
    CHECK(rr < 2e-3f && ri < 2e-3f, "denoiser ggml chunk_forward matches scalar denoiser_net_forward");
}

// Chunk-batch consistency: batch_forward on N stacked chunks must equal
// chunk_forward on each chunk individually (proves the ne3 batching is correct).
static void test_batch(const char * gguf_path) {
    tts_cpp::lavasr::DenoiserWeights w;
    std::string                      err;
    if (!tts_cpp::lavasr::load_denoiser_gguf(gguf_path, w, &err)) return;
    const int    L = w.chunk_frames, F = w.spec_bins, N = 3;
    std::mt19937 rng(31337);
    auto         df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) N * L * F), im((size_t) N * L * F);
    df(re); df(im);
    auto g = tts_cpp::lavasr::DenoiserGgml::create(w, -1);
    std::vector<float> br, bi;
    g->batch_forward(re, im, L, N, br, bi);
    float m = 0.0f;
    for (int c = 0; c < N; c++) {
        std::vector<float> cre(re.begin() + (size_t) c * L * F, re.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> cim(im.begin() + (size_t) c * L * F, im.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> sr, si;
        g->chunk_forward(cre, cim, L, sr, si);
        for (size_t i = 0; i < sr.size(); i++) {
            m = std::max(m, std::fabs(br[(size_t) c * L * F + i] - sr[i]));
            m = std::max(m, std::fabs(bi[(size_t) c * L * F + i] - si[i]));
        }
    }
    std::printf("batch-consistency (N=%d): max_abs=%.3e\n", N, m);
    CHECK(m < 1e-4f, "batch_forward matches per-chunk chunk_forward");
}

// Uniform(-scale, scale) fill; '*.running_var' instead gets uniform(0.25, 1) —
// the BN fold computes g/sqrt(rv+eps), so rv must stay positive (scale ignored).
static void add_dn(DenoiserWeights & w, const std::string & n, std::vector<int> shape,
                   std::mt19937 & rng, float scale) {
    size_t sz = 1;
    for (int s : shape) sz *= (size_t) s;
    DnTensor t;
    t.data.resize(sz);
    const std::string suf = ".running_var";
    const bool var = n.size() > suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0;
    std::uniform_real_distribution<float> d(var ? 0.25f : -scale, var ? 1.0f : scale);
    for (float & v : t.data) v = d(rng);
    t.shape = std::move(shape);
    w.t[n]  = std::move(t);
}

// Small but complete synthetic UL-UNAS: spec_bins=17, erb 5+8 (Fe=13 -> F1=7 -> F2=4),
// chunk 8/3.  0.5/sqrt(fan_in) weights keep the deep chain conditioned (mask unsaturated).
static DenoiserWeights make_synthetic_denoiser_weights(std::mt19937 & rng) {
    DenoiserWeights w;
    w.n_fft = 32; w.hop = 16; w.win = 32; w.spec_bins = 17; w.work_sample_rate = 16000;
    w.erb_low = 5; w.erb_high = 8; w.freq_comp_ratio = 4;
    w.chunk_frames = 8; w.chunk_hop = 3;
    w.bn_eps = 1e-5f; w.ln_eps = 1e-8f;

    const int r = w.freq_comp_ratio, F1 = 7, F2 = 4, Cd = 16;
    auto ws  = [](int fan_in) { return 0.5f / std::sqrt((float) fan_in); };
    auto add = [&](const std::string & n, std::vector<int> shp, float sc) { add_dn(w, n, std::move(shp), rng, sc); };
    auto bnw = [&](const std::string & p, int C) {
        add(p + ".weight", { C }, 0.1f); add(p + ".bias", { C }, 0.1f);
        add(p + ".running_mean", { C }, 0.1f); add(p + ".running_var", { C }, 0.0f);
    };
    auto apw = [&](const std::string & p, int C, int F) {
        add(p + ".affine_weight", { C, F }, 0.1f); add(p + ".affine_bias", { C, F }, 0.1f);
        add(p + ".slope_weight", { C }, 0.1f);
    };
    auto gruw = [&](const std::string & p, int I, int H, bool rev) {
        for (const char * suf : { "", "_reverse" }) {
            add(p + ".weight_ih_l0" + suf, { 3 * H, I }, ws(I));
            add(p + ".weight_hh_l0" + suf, { 3 * H, H }, ws(H));
            add(p + ".bias_ih_l0" + suf, { 3 * H }, 0.1f);
            add(p + ".bias_hh_l0" + suf, { 3 * H }, 0.1f);
            if (!rev) break;
        }
    };
    auto ctfaw = [&](const std::string & p, int C) {
        gruw(p + ".ta_gru", C, 2 * C, false);
        add(p + ".ta_fc.weight", { C, 2 * C }, ws(2 * C)); add(p + ".ta_fc.bias", { C }, 0.1f);
        gruw(p + ".fa.gru", r, r, true);
        add(p + ".fa.fc.weight", { r, 2 * r }, ws(2 * r)); add(p + ".fa.fc.bias", { r }, 0.1f);
    };
    auto fcw = [&](const std::string & p, int out, int in) {
        add(p + ".weight", { out, in }, ws(in)); add(p + ".bias", { out }, 0.1f);
    };
    auto convw = [&](const std::string & p, int cout, int cing, int kt, int kf) {
        add(p + ".weight", { cout, cing, kt, kf }, ws(cing * kt * kf)); add(p + ".bias", { cout }, 0.1f);
    };
    // Transposed kernels in raw PyTorch layout [Cin, Cout/g, kt, kf]; create() derives the .ct.
    auto convtw = [&](const std::string & p, int cin, int coutg, int kt, int kf, int g) {
        add(p + ".weight", { cin, coutg, kt, kf }, ws(cin / g * kt * kf));
        add(p + ".bias", { coutg * g }, 0.1f);
    };

    add("erb.erb_fc.weight", { w.erb_high, w.spec_bins - w.erb_low }, ws(w.spec_bins - w.erb_low));
    add("erb.ierb_fc.weight", { w.spec_bins - w.erb_low, w.erb_high }, ws(w.erb_high));
    { // encoder: XConv 1->12 s2, XMB 12->24 s2, XDWS 24->24, XMB 24->32, XDWS 32->16
        const std::string e0 = "encoder.en_convs.0";
        convw(e0 + ".ops.1", 12, 1, 3, 3);
        bnw(e0 + ".ops.2", 12); apw(e0 + ".ops.3", 12, F1); ctfaw(e0 + ".ops.4", 12);
        const std::string e1 = "encoder.en_convs.1";
        convw(e1 + ".pconv1.0", 24, 6, 1, 1); bnw(e1 + ".pconv1.1", 24); apw(e1 + ".pconv1.2", 24, F1);
        convw(e1 + ".dconv.1", 24, 1, 2, 3); bnw(e1 + ".dconv.2", 24); apw(e1 + ".dconv.3", 24, F2);
        convw(e1 + ".pconv2.0", 24, 12, 1, 1); bnw(e1 + ".pconv2.1", 24); ctfaw(e1 + ".pconv2.2", 24);
        const std::string e2 = "encoder.en_convs.2";
        convw(e2 + ".pconv.0", 24, 12, 1, 1); bnw(e2 + ".pconv.1", 24); apw(e2 + ".pconv.2", 24, F2);
        convw(e2 + ".dconv.1", 24, 1, 2, 3); bnw(e2 + ".dconv.2", 24); apw(e2 + ".dconv.3", 24, F2);
        ctfaw(e2 + ".dconv.4", 24);
        const std::string e3 = "encoder.en_convs.3";
        convw(e3 + ".pconv1.0", 32, 12, 1, 1); bnw(e3 + ".pconv1.1", 32); apw(e3 + ".pconv1.2", 32, F2);
        convw(e3 + ".dconv.1", 32, 1, 1, 5); bnw(e3 + ".dconv.2", 32); apw(e3 + ".dconv.3", 32, F2);
        convw(e3 + ".pconv2.0", 32, 16, 1, 1); bnw(e3 + ".pconv2.1", 32); ctfaw(e3 + ".pconv2.2", 32);
        const std::string e4 = "encoder.en_convs.4";
        convw(e4 + ".pconv.0", 16, 16, 1, 1); bnw(e4 + ".pconv.1", 16); apw(e4 + ".pconv.2", 16, F2);
        convw(e4 + ".dconv.1", 16, 1, 1, 5); bnw(e4 + ".dconv.2", 16); apw(e4 + ".dconv.3", 16, F2);
        ctfaw(e4 + ".dconv.4", 16);
    }
    for (int i = 0; i < 2; i++) { // bottleneck: 2x DPGRNN (C=16, F=F2)
        const std::string p = "dpgrnn." + std::to_string(i);
        gruw(p + ".intra_rnn.rnn1", Cd / 2, 4, true); gruw(p + ".intra_rnn.rnn2", Cd / 2, 4, true);
        fcw(p + ".intra_fc", Cd, Cd);
        add(p + ".intra_ln.weight", { F2 * Cd }, 0.1f); add(p + ".intra_ln.bias", { F2 * Cd }, 0.1f);
        gruw(p + ".inter_rnn.rnn1", Cd / 2, 8, false); gruw(p + ".inter_rnn.rnn2", Cd / 2, 8, false);
        fcw(p + ".inter_fc", Cd, Cd);
        add(p + ".inter_ln.weight", { F2 * Cd }, 0.1f); add(p + ".inter_ln.bias", { F2 * Cd }, 0.1f);
    }
    { // decoder mirror: XDWS 16->32, XMB 32->24, XDWS 24->24, XMB 24->12 s2, final XConv 12->1 s2
        const std::string d0 = "decoder.de_convs.0";
        convw(d0 + ".pconv.0", 32, 8, 1, 1); bnw(d0 + ".pconv.1", 32); apw(d0 + ".pconv.2", 32, F2);
        convtw(d0 + ".dconv.1", 32, 1, 1, 5, 32); bnw(d0 + ".dconv.2", 32); apw(d0 + ".dconv.3", 32, F2);
        ctfaw(d0 + ".dconv.4", 32);
        const std::string d1 = "decoder.de_convs.1";
        convw(d1 + ".pconv1.0", 24, 16, 1, 1); bnw(d1 + ".pconv1.1", 24); apw(d1 + ".pconv1.2", 24, F2);
        convtw(d1 + ".dconv.1", 24, 1, 1, 5, 24); bnw(d1 + ".dconv.2", 24); apw(d1 + ".dconv.3", 24, F2);
        convw(d1 + ".pconv2.0", 24, 12, 1, 1); bnw(d1 + ".pconv2.1", 24); ctfaw(d1 + ".pconv2.2", 24);
        const std::string d2 = "decoder.de_convs.2";
        convw(d2 + ".pconv.0", 24, 12, 1, 1); bnw(d2 + ".pconv.1", 24); apw(d2 + ".pconv.2", 24, F2);
        convtw(d2 + ".dconv.1", 24, 1, 2, 3, 24); bnw(d2 + ".dconv.2", 24); apw(d2 + ".dconv.3", 24, F2);
        ctfaw(d2 + ".dconv.4", 24);
        const std::string d3 = "decoder.de_convs.3";
        convw(d3 + ".pconv1.0", 12, 12, 1, 1); bnw(d3 + ".pconv1.1", 12); apw(d3 + ".pconv1.2", 12, F2);
        convtw(d3 + ".dconv.1", 12, 1, 2, 3, 12); bnw(d3 + ".dconv.2", 12); apw(d3 + ".dconv.3", 12, F1);
        convw(d3 + ".pconv2.0", 12, 6, 1, 1); bnw(d3 + ".pconv2.1", 12); ctfaw(d3 + ".pconv2.2", 12);
        const std::string d4 = "decoder.de_convs.4"; // is_last: no .ops.3 affine-PReLU
        convtw(d4 + ".ops.1", 12, 1, 3, 3, 1); bnw(d4 + ".ops.2", 1); ctfaw(d4 + ".ops.4", 1);
    }
    return w;
}

// Always-on e2e parity (no fixture): ggml chunk_forward vs the scalar
// denoiser_net_forward on the synthetic model, same metrics as test_e2e.
static void test_e2e_synthetic() {
    std::mt19937    rng(20260710);
    DenoiserWeights w = make_synthetic_denoiser_weights(rng);
    const int       L = w.chunk_frames, F = w.spec_bins;
    auto            df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) L * F), im((size_t) L * F);
    df(re); df(im);
    std::vector<float> sr, si;
    tts_cpp::lavasr::denoiser_net_forward(w, re, im, L, sr, si);
    auto               g = tts_cpp::lavasr::DenoiserGgml::create(w, -1);
    std::vector<float> gr, gi;
    g->chunk_forward(re, im, L, gr, gi);
    auto rel = [](const std::vector<float> & a, const std::vector<float> & b, float & mx) {
        double num = 0, den = 0;
        mx = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float d = std::fabs(a[i] - b[i]);
            mx      = std::max(mx, d);
            num += (double) d * d;
            den += (double) a[i] * a[i];
        }
        return (float) std::sqrt(num / std::max(den, 1e-20));
    };
    float mr, mi;
    float rr = rel(sr, gr, mr), ri = rel(si, gi, mi);
    float  omax = 0.0f;
    double oss  = 0.0;
    for (float v : sr) { omax = std::max(omax, std::fabs(v)); oss += (double) v * v; }
    const float orms = (float) std::sqrt(oss / std::max<double>((double) sr.size(), 1.0));
    std::printf("e2e-synthetic: real max_abs=%.3e nrmse=%.3e | imag max_abs=%.3e nrmse=%.3e | out rms=%.3e max=%.3e\n",
                mr, rr, mi, ri, orms, omax);
    CHECK(gr.size() == sr.size() && gi.size() == si.size(), "synthetic e2e output size matches scalar");
    CHECK(rr < 2e-3f && ri < 2e-3f, "synthetic denoiser ggml chunk_forward matches scalar");
    CHECK(omax > 1e-3f, "synthetic outputs are not all ~0 (mask not degenerate)");
}

// Always-on chunk-batch consistency on the synthetic model (style of test_batch).
static void test_batch_synthetic() {
    std::mt19937    rng(20260710);
    DenoiserWeights w = make_synthetic_denoiser_weights(rng);
    const int       L = w.chunk_frames, F = w.spec_bins, N = 3;
    auto            df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) N * L * F), im((size_t) N * L * F);
    df(re); df(im);
    auto g = tts_cpp::lavasr::DenoiserGgml::create(w, -1);
    std::vector<float> br, bi;
    g->batch_forward(re, im, L, N, br, bi);
    float m = 0.0f;
    for (int c = 0; c < N; c++) {
        std::vector<float> cre(re.begin() + (size_t) c * L * F, re.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> cim(im.begin() + (size_t) c * L * F, im.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> sr, si;
        g->chunk_forward(cre, cim, L, sr, si);
        for (size_t i = 0; i < sr.size(); i++) {
            m = std::max(m, std::fabs(br[(size_t) c * L * F + i] - sr[i]));
            m = std::max(m, std::fabs(bi[(size_t) c * L * F + i] - si[i]));
        }
    }
    std::printf("batch-consistency synthetic (N=%d): max_abs=%.3e\n", N, m);
    CHECK(m < 1e-4f, "synthetic batch_forward matches per-chunk chunk_forward");
}

// Pipeline parity: denoise_with_batch_core must be BIT-EXACT vs denoise_with_core
// when both wrap the same element-wise core (metadata-only weights, no tensors).
static void test_pipeline_batch_parity() {
    DenoiserWeights w; // defaults = shipped model: 512/256 STFT, F=257, chunk 63/21, 16 kHz

    auto elemwise = [](const std::vector<float> & re, const std::vector<float> & im,
                       std::vector<float> & orr, std::vector<float> & oii) {
        orr.resize(re.size());
        oii.resize(im.size());
        for (size_t i = 0; i < re.size(); i++) {
            orr[i] = 0.5f * re[i] - 0.25f * im[i];
            oii[i] = 0.5f * im[i] + 0.25f * re[i];
        }
    };
    DenoiseChunkCore chunk_core = [&](const std::vector<float> & re, const std::vector<float> & im,
                                      int /*L*/, std::vector<float> & orr, std::vector<float> & oii) {
        elemwise(re, im, orr, oii);
    };
    DenoiseBatchCore batch_core = [&](const std::vector<float> & re, const std::vector<float> & im,
                                      int /*L*/, int /*n_chunks*/,
                                      std::vector<float> & orr, std::vector<float> & oii) {
        elemwise(re, im, orr, oii);
    };

    struct Run { int n, sr; const char * name; };
    const Run runs[] = {
        { 32500, 16000, "multi-chunk forced-tail" }, // T=127: starts 0,21,42,63 + forced 64
        { 4000, 16000, "single-chunk T<=L" },        // T=16 <= 63
        { 48000, 24000, "resampled 24kHz" },         // real resampling; T=126 multi-chunk
    };
    for (const Run & r : runs) {
        std::vector<float> pcm((size_t) r.n);
        for (int i = 0; i < r.n; i++) {
            const double ph = 2.0 * 3.14159265358979323846 * i / r.sr;
            pcm[i] = (float) (0.5 * std::sin(ph * 440.0) + 0.3 * std::sin(ph * 1337.0) +
                              0.2 * std::sin(ph * 3200.0));
        }
        const std::vector<float> a = denoise_with_core(w, pcm, r.sr, chunk_core);
        const std::vector<float> b = denoise_with_batch_core(w, pcm, r.sr, batch_core);
        CHECK(a.size() == pcm.size(), "pipeline parity: output length preserved");
        CHECK(a.size() == b.size(), "pipeline parity: sizes equal");
        float m = 0.0f;
        for (size_t i = 0; i < a.size() && i < b.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
        const bool bitexact = a.size() == b.size() &&
                              std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        std::printf("pipeline-parity %s: n=%zu bitexact=%d max_abs=%.3e\n",
                    r.name, a.size(), bitexact ? 1 : 0, m);
        CHECK(bitexact, "denoise_with_batch_core bit-exact vs denoise_with_core");
    }
}

int main(int argc, char ** argv) {
    test_gru_batched();
    test_conv2d();
    test_conv_transpose2d();
    test_affine_prelu();
    test_shuffle2();
    test_dpgrnn();
    test_ctfa();
    test_e2e_synthetic();
    test_batch_synthetic();
    test_pipeline_batch_parity();
    if (argc > 1) { test_e2e(argv[1]); test_batch(argv[1]); }
    if (g_failures == 0) {
        std::printf("OK: all LavaSR denoiser ggml tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d LavaSR denoiser ggml test(s) failed\n", g_failures);
    return 1;
}
