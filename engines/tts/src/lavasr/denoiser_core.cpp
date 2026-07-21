#include "denoiser_core.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

const DnTensor & DenoiserWeights::get(const std::string & name) const {
    auto it = t.find(name);
    if (it == t.end()) {
        throw std::runtime_error("lavasr denoiser: missing tensor '" + name + "'");
    }
    return it->second;
}

namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Channel-major 3-D activation tensor [C][T][F], index ((c*T)+t)*F + f.
struct T3 {
    std::vector<float> d;
    int                C = 0, T = 0, F = 0;
    void   alloc(int c, int t, int f) { C = c; T = t; F = f; d.assign(static_cast<size_t>(c) * t * f, 0.0f); }
    float & at(int c, int t, int f) { return d[(static_cast<size_t>(c) * T + t) * F + f]; }
    float   at(int c, int t, int f) const { return d[(static_cast<size_t>(c) * T + t) * F + f]; }
};

// y[t,o] = b[o] + sum_i x[t,i] * W[o,i].  x:[N][c_in] flat, W:[c_out][c_in].
std::vector<float> linear2d(const std::vector<float> & x, int N, int c_in,
                            const DnTensor & W, int c_out, const DnTensor * b) {
    std::vector<float> y(static_cast<size_t>(N) * c_out, 0.0f);
    for (int n = 0; n < N; n++) {
        const float * xr = x.data() + static_cast<size_t>(n) * c_in;
        float *       yr = y.data() + static_cast<size_t>(n) * c_out;
        for (int o = 0; o < c_out; o++) {
            const float * wr  = W.data.data() + static_cast<size_t>(o) * c_in;
            float         acc = b ? b->data[o] : 0.0f;
            for (int i = 0; i < c_in; i++) {
                acc += xr[i] * wr[i];
            }
            yr[o] = acc;
        }
    }
    return y;
}

// One PyTorch GRU (gate order r,z,n) over x:[L][I] -> y:[L][H], zero init state.
std::vector<float> gru_seq(const std::vector<float> & x, int L, int I,
                           const DnTensor & Wih, const DnTensor & Whh,
                           const DnTensor & Bih, const DnTensor & Bhh) {
    const int H = Whh.shape[1];
    std::vector<float> h(H, 0.0f), ys(static_cast<size_t>(L) * H, 0.0f);
    std::vector<float> gi(static_cast<size_t>(3) * H), gh(static_cast<size_t>(3) * H);
    for (int t = 0; t < L; t++) {
        const float * xt = x.data() + static_cast<size_t>(t) * I;
        for (int o = 0; o < 3 * H; o++) {
            const float * wr  = Wih.data.data() + static_cast<size_t>(o) * I;
            float         acc = Bih.data[o];
            for (int i = 0; i < I; i++) {
                acc += wr[i] * xt[i];
            }
            gi[o] = acc;
        }
        for (int o = 0; o < 3 * H; o++) {
            const float * wr  = Whh.data.data() + static_cast<size_t>(o) * H;
            float         acc = Bhh.data[o];
            for (int i = 0; i < H; i++) {
                acc += wr[i] * h[i];
            }
            gh[o] = acc;
        }
        for (int k = 0; k < H; k++) {
            const float r = sigmoidf(gi[k] + gh[k]);
            const float z = sigmoidf(gi[H + k] + gh[H + k]);
            const float n = std::tanh(gi[2 * H + k] + r * gh[2 * H + k]);
            h[k]                                       = (1.0f - z) * n + z * h[k];
            ys[static_cast<size_t>(t) * H + k]         = h[k];
        }
    }
    return ys;
}

// The forward runner bundles the const weights so the block helpers can resolve
// tensors by name (mirrors ulunas.py module structure).
struct Runner {
    const DenoiserWeights & w;

    const DnTensor & W(const std::string & n) const { return w.get(n); }
    const DnTensor * Wopt(const std::string & n) const {
        return w.has(n) ? &w.get(n) : nullptr;
    }

    // Grouped/depthwise 2-D conv, causal in time (pad top by kt-1), "same" time
    // length, freq stride + symmetric freq pad.  W:[Cout,Cin/g,kt,kf].
    T3 conv2d(const T3 & x, const DnTensor & Wt, const DnTensor * B, int stride_f,
              int pad_f, int groups) const {
        const int Cin = x.C, T = x.T, F = x.F;
        const int Cout = Wt.shape[0], Cing = Wt.shape[1], kt = Wt.shape[2], kf = Wt.shape[3];
        const int pad_t_top = kt - 1;
        const int Fout      = (F + 2 * pad_f - kf) / stride_f + 1;
        T3 out; out.alloc(Cout, T, Fout);
        const int og = Cout / groups;
        for (int oc = 0; oc < Cout; oc++) {
            const int   grp = oc / og, ic0 = grp * Cing;
            const float b   = B ? B->data[oc] : 0.0f;
            for (int t = 0; t < T; t++)
                for (int f = 0; f < Fout; f++) out.at(oc, t, f) = b;
            for (int ici = 0; ici < Cing; ici++) {
                const int     ic = ic0 + ici;
                const float * wk = Wt.data.data() +
                                   ((static_cast<size_t>(oc) * Cing) + ici) * kt * kf;
                for (int it = 0; it < kt; it++) {
                    for (int jf = 0; jf < kf; jf++) {
                        const float wv = wk[it * kf + jf];
                        if (wv == 0.0f) continue;
                        for (int t = 0; t < T; t++) {
                            const int ti = t + it - pad_t_top;
                            if (ti < 0 || ti >= T) continue;
                            for (int f = 0; f < Fout; f++) {
                                const int fi = f * stride_f + jf - pad_f;
                                if (fi < 0 || fi >= F) continue;
                                out.at(oc, t, f) += wv * x.at(ic, ti, fi);
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    // Grouped/depthwise 2-D transposed conv (decoder upsampling), causal time.
    // W:[Cin,Cout/g,kt,kf].  pad_t=kt-1 crops the causal-pad back to length T.
    T3 conv_transpose2d(const T3 & x, const DnTensor & Wt, const DnTensor * B,
                        int stride_f, int pad_f, int groups) const {
        const int Cin = x.C, T = x.T, F = x.F;
        const int Coutg = Wt.shape[1], kt = Wt.shape[2], kf = Wt.shape[3];
        const int Cout = Coutg * groups;
        const int pt_top = kt - 1, pad_t = kt - 1;
        const int Tp    = T + pt_top;
        const int Tfull = (Tp - 1) + kt;
        const int Ffull = (F - 1) * stride_f + kf;
        T3 full; full.alloc(Cout, Tfull, Ffull);
        const int ing = Cin / groups;
        for (int ic = 0; ic < Cin; ic++) {
            const int grp = ic / ing, oc0 = grp * Coutg;
            for (int ocg = 0; ocg < Coutg; ocg++) {
                const int     oc = oc0 + ocg;
                const float * wk = Wt.data.data() +
                                   ((static_cast<size_t>(ic) * Coutg) + ocg) * kt * kf;
                for (int t = 0; t < Tp; t++) {
                    const int xt = t - pt_top;
                    for (int f = 0; f < F; f++) {
                        const float v = (xt < 0) ? 0.0f : x.at(ic, xt, f);
                        if (v == 0.0f) continue;
                        for (int it = 0; it < kt; it++)
                            for (int jf = 0; jf < kf; jf++)
                                full.at(oc, t + it, f * stride_f + jf) += v * wk[it * kf + jf];
                    }
                }
            }
        }
        if (B) {
            for (int oc = 0; oc < Cout; oc++) {
                const float b = B->data[oc];
                for (int t = 0; t < Tfull; t++)
                    for (int f = 0; f < Ffull; f++) full.at(oc, t, f) += b;
            }
        }
        const int Tout = Tfull - 2 * pad_t, Fout = Ffull - 2 * pad_f;
        T3 out; out.alloc(Cout, Tout, Fout);
        for (int oc = 0; oc < Cout; oc++)
            for (int t = 0; t < Tout; t++)
                for (int f = 0; f < Fout; f++)
                    out.at(oc, t, f) = full.at(oc, t + pad_t, f + pad_f);
        return out;
    }

    void batchnorm(T3 & x, const std::string & p) const {
        const DnTensor & g  = W(p + ".weight");
        const DnTensor & b  = W(p + ".bias");
        const DnTensor & rm = W(p + ".running_mean");
        const DnTensor & rv = W(p + ".running_var");
        for (int c = 0; c < x.C; c++) {
            const float scale = g.data[c] / std::sqrt(rv.data[c] + w.bn_eps);
            const float shift = b.data[c] - rm.data[c] * scale;
            for (int t = 0; t < x.T; t++)
                for (int f = 0; f < x.F; f++) x.at(c, t, f) = x.at(c, t, f) * scale + shift;
        }
    }

    void affine_prelu(T3 & x, const std::string & p) const {
        const DnTensor & aw = W(p + ".affine_weight"); // [C,F]
        const DnTensor & ab = W(p + ".affine_bias");
        const DnTensor & sl = W(p + ".slope_weight");  // [C]
        if (aw.shape.size() != 2 || aw.shape[0] != x.C || aw.shape[1] != x.F) {
            throw std::runtime_error("lavasr denoiser: affine_prelu shape mismatch at '" + p + "'");
        }
        for (int c = 0; c < x.C; c++) {
            const float * awr = aw.data.data() + static_cast<size_t>(c) * x.F;
            const float * abr = ab.data.data() + static_cast<size_t>(c) * x.F;
            const float   s   = sl.data[c];
            for (int t = 0; t < x.T; t++)
                for (int f = 0; f < x.F; f++) {
                    const float v = x.at(c, t, f);
                    x.at(c, t, f) = awr[f] * v + abr[f] + (v > 0.0f ? v : s * v);
                }
        }
    }

    T3 shuffle(const T3 & x) const {
        const int half = x.C / 2;
        T3 o; o.alloc(x.C, x.T, x.F);
        for (int c = 0; c < half; c++)
            for (int t = 0; t < x.T; t++)
                for (int f = 0; f < x.F; f++) {
                    o.at(2 * c, t, f)     = x.at(c, t, f);
                    o.at(2 * c + 1, t, f) = x.at(half + c, t, f);
                }
        return o;
    }

    std::vector<float> gru_uni(const std::vector<float> & x, int L, int I,
                               const std::string & p) const {
        return gru_seq(x, L, I, W(p + ".weight_ih_l0"), W(p + ".weight_hh_l0"),
                       W(p + ".bias_ih_l0"), W(p + ".bias_hh_l0"));
    }

    std::vector<float> gru_bi(const std::vector<float> & x, int L, int I,
                              const std::string & p) const {
        std::vector<float> f = gru_seq(x, L, I, W(p + ".weight_ih_l0"),
                                       W(p + ".weight_hh_l0"), W(p + ".bias_ih_l0"),
                                       W(p + ".bias_hh_l0"));
        std::vector<float> xr(static_cast<size_t>(L) * I);
        for (int t = 0; t < L; t++)
            for (int i = 0; i < I; i++)
                xr[static_cast<size_t>(t) * I + i] = x[static_cast<size_t>(L - 1 - t) * I + i];
        std::vector<float> b = gru_seq(xr, L, I, W(p + ".weight_ih_l0_reverse"),
                                       W(p + ".weight_hh_l0_reverse"),
                                       W(p + ".bias_ih_l0_reverse"),
                                       W(p + ".bias_hh_l0_reverse"));
        const int          H = W(p + ".weight_hh_l0").shape[1];
        std::vector<float> y(static_cast<size_t>(L) * 2 * H);
        for (int t = 0; t < L; t++) {
            for (int k = 0; k < H; k++) {
                y[static_cast<size_t>(t) * 2 * H + k]     = f[static_cast<size_t>(t) * H + k];
                y[static_cast<size_t>(t) * 2 * H + H + k] = b[static_cast<size_t>(L - 1 - t) * H + k];
            }
        }
        return y;
    }

    // causal time-frequency attention: at(temporal) * x * af(frequency).
    T3 ctfa(const T3 & x, const std::string & p) const {
        const int C = x.C, T = x.T, F = x.F;
        // temporal: energy-mean (mean of squares) over freq -> GRU(C->2C) -> FC(2C->C) -> sigmoid
        std::vector<float> zt(static_cast<size_t>(T) * C, 0.0f);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) {
                double s = 0.0;
                for (int f = 0; f < F; f++) { const float v = x.at(c, t, f); s += static_cast<double>(v) * v; }
                zt[static_cast<size_t>(t) * C + c] = static_cast<float>(s / F);
            }
        std::vector<float> at = gru_uni(zt, T, C, p + ".ta_gru");
        at = linear2d(at, T, 2 * C, W(p + ".ta_fc.weight"), C, Wopt(p + ".ta_fc.bias"));
        for (float & v : at) v = sigmoidf(v);
        // frequency: energy-mean (mean of squares) over chan -> fold by r -> BiGRU(r->r) -> FC(2r->r) -> sigmoid
        const int r = w.freq_comp_ratio;
        const int pad_len = (r - (F % r)) % r;
        const int Fp = F + pad_len, Hh = Fp / r;
        std::vector<float> af(static_cast<size_t>(T) * F, 0.0f);
        std::vector<float> row(static_cast<size_t>(Hh) * r);
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < Hh; h++)
                for (int rr = 0; rr < r; rr++) {
                    const int f = h * r + rr;
                    double    s = 0.0;
                    if (f < F) {
                        for (int c = 0; c < C; c++) { const float v = x.at(c, t, f); s += static_cast<double>(v) * v; }
                        s /= C;
                    }
                    row[static_cast<size_t>(h) * r + rr] = static_cast<float>(s);
                }
            std::vector<float> y = gru_bi(row, Hh, r, p + ".fa.gru");
            y = linear2d(y, Hh, 2 * r, W(p + ".fa.fc.weight"), r, Wopt(p + ".fa.fc.bias"));
            for (int h = 0; h < Hh; h++)
                for (int rr = 0; rr < r; rr++) {
                    const int f = h * r + rr;
                    if (f < F) af[static_cast<size_t>(t) * F + f] = sigmoidf(y[static_cast<size_t>(h) * r + rr]);
                }
        }
        T3 o; o.alloc(C, T, F);
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++)
                    o.at(c, t, f) = at[static_cast<size_t>(t) * C + c] * x.at(c, t, f) *
                                    af[static_cast<size_t>(t) * F + f];
        return o;
    }

    // grouped RNN: split feature in half, rnn1/rnn2, concat.
    std::vector<float> grnn(const std::vector<float> & x, int L, int I,
                            const std::string & p, bool bidir) const {
        const int          half = I / 2;
        std::vector<float> x1(static_cast<size_t>(L) * half), x2(static_cast<size_t>(L) * half);
        for (int t = 0; t < L; t++)
            for (int i = 0; i < half; i++) {
                x1[static_cast<size_t>(t) * half + i] = x[static_cast<size_t>(t) * I + i];
                x2[static_cast<size_t>(t) * half + i] = x[static_cast<size_t>(t) * I + half + i];
            }
        std::vector<float> y1 = bidir ? gru_bi(x1, L, half, p + ".rnn1") : gru_uni(x1, L, half, p + ".rnn1");
        std::vector<float> y2 = bidir ? gru_bi(x2, L, half, p + ".rnn2") : gru_uni(x2, L, half, p + ".rnn2");
        const int          h1 = static_cast<int>(y1.size() / L), h2 = static_cast<int>(y2.size() / L);
        std::vector<float> y(static_cast<size_t>(L) * (h1 + h2));
        for (int t = 0; t < L; t++) {
            for (int i = 0; i < h1; i++) y[static_cast<size_t>(t) * (h1 + h2) + i] = y1[static_cast<size_t>(t) * h1 + i];
            for (int i = 0; i < h2; i++) y[static_cast<size_t>(t) * (h1 + h2) + h1 + i] = y2[static_cast<size_t>(t) * h2 + i];
        }
        return y;
    }

    // LayerNorm over the last two dims (F,C) per time step, x flat [T][F][C].
    void layernorm_fc(std::vector<float> & x, int T, int Fw, int C,
                      const DnTensor & g, const DnTensor & b) const {
        const int    n = Fw * C;
        for (int t = 0; t < T; t++) {
            float * row  = x.data() + static_cast<size_t>(t) * n;
            double  mean = 0.0;
            for (int i = 0; i < n; i++) mean += row[i];
            mean /= n;
            double var = 0.0;
            for (int i = 0; i < n; i++) { const double d = row[i] - mean; var += d * d; }
            var /= n;
            const float inv = 1.0f / std::sqrt(static_cast<float>(var) + w.ln_eps);
            for (int i = 0; i < n; i++)
                row[i] = (static_cast<float>(row[i] - mean) * inv) * g.data[i] + b.data[i];
        }
    }

    // dual-path grouped RNN bottleneck.  x:(C,T,F).
    T3 dpgrnn(const T3 & x, const std::string & p) const {
        const int C = x.C, T = x.T, Fw = x.F;
        // ---- intra (over freq) ----
        std::vector<float> xt(static_cast<size_t>(T) * Fw * C); // [T][F][C]
        for (int t = 0; t < T; t++)
            for (int f = 0; f < Fw; f++)
                for (int c = 0; c < C; c++)
                    xt[(static_cast<size_t>(t) * Fw + f) * C + c] = x.at(c, t, f);
        std::vector<float> intra(static_cast<size_t>(T) * Fw * C);
        for (int t = 0; t < T; t++) {
            std::vector<float> seq(xt.begin() + static_cast<size_t>(t) * Fw * C,
                                   xt.begin() + static_cast<size_t>(t + 1) * Fw * C);
            std::vector<float> y = grnn(seq, Fw, C, p + ".intra_rnn", /*bidir=*/true);
            const int          hy = static_cast<int>(y.size() / Fw);
            y = linear2d(y, Fw, hy, W(p + ".intra_fc.weight"), C, Wopt(p + ".intra_fc.bias"));
            std::copy(y.begin(), y.end(), intra.begin() + static_cast<size_t>(t) * Fw * C);
        }
        layernorm_fc(intra, T, Fw, C, W(p + ".intra_ln.weight"), W(p + ".intra_ln.bias"));
        for (size_t i = 0; i < intra.size(); i++) intra[i] += xt[i]; // residual
        // ---- inter (over time) ----
        std::vector<float> inter(static_cast<size_t>(T) * Fw * C);
        for (int f = 0; f < Fw; f++) {
            std::vector<float> seq(static_cast<size_t>(T) * C);
            for (int t = 0; t < T; t++)
                for (int c = 0; c < C; c++)
                    seq[static_cast<size_t>(t) * C + c] = intra[(static_cast<size_t>(t) * Fw + f) * C + c];
            std::vector<float> y = grnn(seq, T, C, p + ".inter_rnn", /*bidir=*/false);
            const int          hy = static_cast<int>(y.size() / T);
            y = linear2d(y, T, hy, W(p + ".inter_fc.weight"), C, Wopt(p + ".inter_fc.bias"));
            for (int t = 0; t < T; t++)
                for (int c = 0; c < C; c++)
                    inter[(static_cast<size_t>(t) * Fw + f) * C + c] = y[static_cast<size_t>(t) * C + c];
        }
        layernorm_fc(inter, T, Fw, C, W(p + ".inter_ln.weight"), W(p + ".inter_ln.bias"));
        for (size_t i = 0; i < inter.size(); i++) inter[i] += intra[i]; // residual
        // ---- back to (C,T,F) ----
        T3 o; o.alloc(C, T, Fw);
        for (int t = 0; t < T; t++)
            for (int f = 0; f < Fw; f++)
                for (int c = 0; c < C; c++)
                    o.at(c, t, f) = inter[(static_cast<size_t>(t) * Fw + f) * C + c];
        return o;
    }

    T3 erb_bm(const T3 & x) const {
        const DnTensor & E   = W("erb.erb_fc.weight"); // [erb_high, in]
        const int        low = w.erb_low, hi = w.erb_high, in = x.F - low;
        T3 o; o.alloc(x.C, x.T, low + hi);
        for (int c = 0; c < x.C; c++)
            for (int t = 0; t < x.T; t++) {
                for (int f = 0; f < low; f++) o.at(c, t, f) = x.at(c, t, f);
                for (int j = 0; j < hi; j++) {
                    const float * er = E.data.data() + static_cast<size_t>(j) * in;
                    double        s  = 0.0;
                    for (int k = 0; k < in; k++) s += static_cast<double>(er[k]) * x.at(c, t, low + k);
                    o.at(c, t, low + j) = static_cast<float>(s);
                }
            }
        return o;
    }

    T3 erb_bs(const T3 & m) const {
        const DnTensor & E   = W("erb.ierb_fc.weight"); // [out, erb_high]
        const int        low = w.erb_low, hi = w.erb_high, out = E.shape[0];
        T3 o; o.alloc(m.C, m.T, low + out);
        for (int c = 0; c < m.C; c++)
            for (int t = 0; t < m.T; t++) {
                for (int f = 0; f < low; f++) o.at(c, t, f) = m.at(c, t, f);
                for (int j = 0; j < out; j++) {
                    const float * er = E.data.data() + static_cast<size_t>(j) * hi;
                    double        s  = 0.0;
                    for (int k = 0; k < hi; k++) s += static_cast<double>(er[k]) * m.at(c, t, low + k);
                    o.at(c, t, low + j) = static_cast<float>(s);
                }
            }
        return o;
    }

    // One encoder/decoder block.  type: 0=XConv 1=XDWS 2=XMB.
    T3 run_block(const T3 & x, const std::string & base, int type, int cout,
                 int kt, int kf, int stride, int groups, bool deconv, bool is_last) const {
        const int pf = kf / 2;
        auto conv = [&](const T3 & in, const std::string & wn, int st, int grp) -> T3 {
            const DnTensor & Wt = W(wn + ".weight");
            const DnTensor * B  = Wopt(wn + ".bias");
            return deconv ? conv_transpose2d(in, Wt, B, st, pf, grp)
                          : conv2d(in, Wt, B, st, pf, grp);
        };
        if (type == 0) { // XConvBlock: conv, BN, affinePReLU?, cTFA, shuffle?
            T3 h = conv(x, base + ".ops.1", stride, groups);
            batchnorm(h, base + ".ops.2");
            if (!is_last) affine_prelu(h, base + ".ops.3");
            h = ctfa(h, base + ".ops.4");
            if (!is_last && groups == 2) h = shuffle(h);
            return h;
        }
        if (type == 1) { // XDWSBlock: pconv(1x1) + depthwise dconv
            T3 h = conv2d(x, W(base + ".pconv.0.weight"), Wopt(base + ".pconv.0.bias"), 1, 0, groups);
            batchnorm(h, base + ".pconv.1");
            affine_prelu(h, base + ".pconv.2");
            if (groups == 2) h = shuffle(h);
            h = conv(h, base + ".dconv.1", stride, cout); // depthwise (groups==cout)
            batchnorm(h, base + ".dconv.2");
            if (!is_last) affine_prelu(h, base + ".dconv.3");
            h = ctfa(h, base + ".dconv.4");
            return h;
        }
        // type == 2: XMBBlocks: pconv1 + depthwise dconv + pconv2, residual, shuffle?
        T3 h = conv2d(x, W(base + ".pconv1.0.weight"), Wopt(base + ".pconv1.0.bias"), 1, 0, groups);
        batchnorm(h, base + ".pconv1.1");
        affine_prelu(h, base + ".pconv1.2");
        if (groups == 2) h = shuffle(h);
        h = conv(h, base + ".dconv.1", stride, cout); // depthwise
        batchnorm(h, base + ".dconv.2");
        affine_prelu(h, base + ".dconv.3");
        h = conv2d(h, W(base + ".pconv2.0.weight"), Wopt(base + ".pconv2.0.bias"), 1, 0, groups);
        batchnorm(h, base + ".pconv2.1");
        h = ctfa(h, base + ".pconv2.2");
        if (h.C == x.C && h.T == x.T && h.F == x.F)
            for (size_t i = 0; i < h.d.size(); i++) h.d[i] += x.d[i];
        if (!is_last && groups == 2) h = shuffle(h);
        return h;
    }
};

} // namespace

void denoiser_net_forward(const DenoiserWeights & w,
                          const std::vector<float> & real_in,
                          const std::vector<float> & imag_in, int T,
                          std::vector<float> & real_out,
                          std::vector<float> & imag_out) {
    const int F = w.spec_bins;
    Runner    R{w};

    // log-magnitude feature (1,T,F).
    T3 feat; feat.alloc(1, T, F);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++) {
            const float re  = real_in[static_cast<size_t>(t) * F + f];
            const float im  = imag_in[static_cast<size_t>(t) * F + f];
            float       mag = std::sqrt(re * re + im * im);
            if (mag < 1e-12f) mag = 1e-12f;
            feat.at(0, t, f) = std::log10(mag);
        }

    T3 x = R.erb_bm(feat);

    // encoder (default ULUNAS config).
    static const int types[5]    = {0, 2, 1, 2, 1};
    static const int strides[5]  = {2, 2, 1, 1, 1};
    static const int groups_[5]  = {1, 2, 2, 2, 2};
    static const int channels[5] = {12, 24, 24, 32, 16};
    static const int kt_[5]      = {3, 2, 2, 1, 1};
    static const int kf_[5]      = {3, 3, 3, 5, 5};

    std::vector<T3> en;
    for (int i = 0; i < 5; i++) {
        x = R.run_block(x, "encoder.en_convs." + std::to_string(i), types[i],
                        channels[i], kt_[i], kf_[i], strides[i], groups_[i],
                        /*deconv=*/false, /*is_last=*/false);
        en.push_back(x);
    }

    // dual-path RNN bottleneck.
    x = R.dpgrnn(x, "dpgrnn.0");
    x = R.dpgrnn(x, "dpgrnn.1");

    // decoder: mirror the encoder (i = 4..1 deconv blocks, then final block 0).
    int j = 0;
    for (int i = 4; i >= 1; i--, j++) {
        for (size_t k = 0; k < x.d.size(); k++) x.d[k] += en[4 - j].d[k]; // additive skip
        x = R.run_block(x, "decoder.de_convs." + std::to_string(j), types[i],
                        channels[i - 1], kt_[i], kf_[i], strides[i], groups_[i],
                        /*deconv=*/true, /*is_last=*/false);
    }
    for (size_t k = 0; k < x.d.size(); k++) x.d[k] += en[0].d[k];
    x = R.run_block(x, "decoder.de_convs." + std::to_string(j), types[0], 1,
                    kt_[0], kf_[0], strides[0], groups_[0], /*deconv=*/true,
                    /*is_last=*/true);

    for (float & v : x.d) v = sigmoidf(v); // real ratio mask (1,T,129)

    T3 m = R.erb_bs(x); // (1,T,F)

    real_out.assign(static_cast<size_t>(T) * F, 0.0f);
    imag_out.assign(static_cast<size_t>(T) * F, 0.0f);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++) {
            const float mm                          = m.at(0, t, f);
            real_out[static_cast<size_t>(t) * F + f] = real_in[static_cast<size_t>(t) * F + f] * mm;
            imag_out[static_cast<size_t>(t) * F + f] = imag_in[static_cast<size_t>(t) * F + f] * mm;
        }
}

} // namespace tts_cpp::lavasr
