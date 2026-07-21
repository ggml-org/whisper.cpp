#include "enhancer_core.h"

#include <cmath>
#include <stdexcept>

namespace tts_cpp::lavasr {

const EnhTensor & EnhancerWeights::get(const std::string & name) const {
    auto it = t.find(name);
    if (it == t.end()) {
        throw std::runtime_error("lavasr enhancer: missing weight tensor '" + name + "'");
    }
    return it->second;
}

namespace {

// General 1-D convolution with "same" output length (pad both sides) and
// grouped channels.  Activations are channel-major: x[c * T + t].
//   weight: [c_out, c_in/groups, K] (C order)
//   bias:   [c_out]
//   out:    [c_out * T]
std::vector<float> conv1d(const std::vector<float> & x, int c_in, int T,
                          const std::vector<float> & weight, int c_out, int K,
                          int pad, int groups, const std::vector<float> & bias) {
    std::vector<float> out(static_cast<size_t>(c_out) * T, 0.0f);
    const int c_in_g  = c_in / groups;
    const int c_out_g = c_out / groups;

    for (int o = 0; o < c_out; o++) {
        const int    g    = o / c_out_g;
        const float  b    = bias.empty() ? 0.0f : bias[o];
        float *      dst  = out.data() + static_cast<size_t>(o) * T;
        for (int t = 0; t < T; t++) {
            dst[t] = b;
        }
        for (int ic = 0; ic < c_in_g; ic++) {
            const float * src = x.data() + static_cast<size_t>(g * c_in_g + ic) * T;
            const float * wk =
                weight.data() + (static_cast<size_t>(o) * c_in_g + ic) * K;
            for (int k = 0; k < K; k++) {
                const float wv = wk[k];
                // input index for output t is (t + k - pad)
                const int shift = k - pad;
                int       tstart = 0;
                int       tend   = T;
                if (shift < 0) {
                    tstart = -shift;
                }
                if (shift > 0) {
                    tend = T - shift;
                }
                for (int t = tstart; t < tend; t++) {
                    dst[t] += wv * src[t + shift];
                }
            }
        }
    }
    return out;
}

// LayerNorm over the last (channel) dim.  x laid out [T][C]: x[t * C + c].
void layernorm_lastdim(std::vector<float> & x, int T, int C,
                       const std::vector<float> & g, const std::vector<float> & b,
                       float eps) {
    for (int t = 0; t < T; t++) {
        float * row  = x.data() + static_cast<size_t>(t) * C;
        double  mean = 0.0;
        for (int c = 0; c < C; c++) {
            mean += row[c];
        }
        mean /= C;
        double var = 0.0;
        for (int c = 0; c < C; c++) {
            const double d = row[c] - mean;
            var += d * d;
        }
        var /= C;
        const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (int c = 0; c < C; c++) {
            row[c] = (static_cast<float>(row[c] - mean) * inv) * g[c] + b[c];
        }
    }
}

inline float gelu_erf(float v) {
    return v * 0.5f * (std::erf(v * 0.70710678118654752440f) + 1.0f);
}

// y[t,o] = b[o] + sum_i x[t,i] * W[o,i].  x:[T][c_in], W:[c_out][c_in].
std::vector<float> linear(const std::vector<float> & x, int T, int c_in,
                          const std::vector<float> & W, int c_out,
                          const std::vector<float> & b) {
    std::vector<float> y(static_cast<size_t>(T) * c_out, 0.0f);
    for (int t = 0; t < T; t++) {
        const float * xr = x.data() + static_cast<size_t>(t) * c_in;
        float *       yr = y.data() + static_cast<size_t>(t) * c_out;
        for (int o = 0; o < c_out; o++) {
            const float * wr = W.data() + static_cast<size_t>(o) * c_in;
            float         acc = b.empty() ? 0.0f : b[o];
            for (int i = 0; i < c_in; i++) {
                acc += xr[i] * wr[i];
            }
            yr[o] = acc;
        }
    }
    return y;
}

// [C][T] -> [T][C]
std::vector<float> ct_to_tc(const std::vector<float> & x, int C, int T) {
    std::vector<float> y(static_cast<size_t>(T) * C);
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < T; t++) {
            y[static_cast<size_t>(t) * C + c] = x[static_cast<size_t>(c) * T + t];
        }
    }
    return y;
}

// [T][C] -> [C][T]
std::vector<float> tc_to_ct(const std::vector<float> & x, int T, int C) {
    std::vector<float> y(static_cast<size_t>(C) * T);
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            y[static_cast<size_t>(c) * T + t] = x[static_cast<size_t>(t) * C + c];
        }
    }
    return y;
}

} // namespace

void enhancer_spec_forward(const EnhancerWeights & w,
                           const std::vector<float> & mel, int T,
                           std::vector<float> & real, std::vector<float> & imag) {
    const int C = w.dim;
    const int F = w.ffn_dim;

    auto vec = [&](const std::string & n) -> const std::vector<float> & {
        return w.get(n).data;
    };

    // embed: Conv1d(n_mels -> C, k) "same"; mel is [n_mels][T].
    std::vector<float> x_ct =
        conv1d(mel, w.n_mels, T, vec("enhancer.embed.weight"), C, w.kernel,
               w.kernel / 2, 1, vec("enhancer.embed.bias")); // [C][T]

    std::vector<float> x_tc = ct_to_tc(x_ct, C, T);
    layernorm_lastdim(x_tc, T, C, vec("enhancer.norm.weight"),
                      vec("enhancer.norm.bias"), w.ln_eps);
    x_ct = tc_to_ct(x_tc, T, C);

    for (int blk = 0; blk < w.n_blocks; blk++) {
        const std::string p = "enhancer.block." + std::to_string(blk) + ".";
        std::vector<float> residual = x_ct; // [C][T]

        // depthwise conv (groups == C)
        std::vector<float> y_ct =
            conv1d(x_ct, C, T, vec(p + "dwconv.weight"), C, w.kernel, w.kernel / 2,
                   C, vec(p + "dwconv.bias")); // [C][T]

        std::vector<float> y_tc = ct_to_tc(y_ct, C, T);
        layernorm_lastdim(y_tc, T, C, vec(p + "norm.weight"), vec(p + "norm.bias"),
                          w.ln_eps);

        // pwconv1: [T][C] -> [T][F]
        std::vector<float> h =
            linear(y_tc, T, C, vec(p + "pwconv1.weight"), F, vec(p + "pwconv1.bias"));
        for (float & v : h) {
            v = gelu_erf(v);
        }
        // pwconv2: [T][F] -> [T][C]
        std::vector<float> z =
            linear(h, T, F, vec(p + "pwconv2.weight"), C, vec(p + "pwconv2.bias"));

        // gamma (channel-last) then transpose + residual add.
        const std::vector<float> & gamma = vec(p + "gamma");
        for (int t = 0; t < T; t++) {
            float * zr = z.data() + static_cast<size_t>(t) * C;
            for (int c = 0; c < C; c++) {
                zr[c] *= gamma[c];
            }
        }
        std::vector<float> z_ct = tc_to_ct(z, T, C);
        for (size_t i = 0; i < x_ct.size(); i++) {
            x_ct[i] = residual[i] + z_ct[i];
        }
    }

    std::vector<float> hidden = ct_to_tc(x_ct, C, T); // [T][C]
    layernorm_lastdim(hidden, T, C, vec("enhancer.final_norm.weight"),
                      vec("enhancer.final_norm.bias"), w.ln_eps);

    // spec head: Linear(C -> 2*spec_bins), split log-mag / phase.
    const int          B   = w.spec_bins;
    std::vector<float> lin = linear(hidden, T, C, vec("spec_head.out.weight"), 2 * B,
                                    vec("spec_head.out.bias")); // [T][2B]

    real.assign(static_cast<size_t>(B) * T, 0.0f);
    imag.assign(static_cast<size_t>(B) * T, 0.0f);
    for (int t = 0; t < T; t++) {
        const float * lr = lin.data() + static_cast<size_t>(t) * (2 * B);
        for (int f = 0; f < B; f++) {
            float mag = std::exp(lr[f]);
            if (mag > w.clip_max) {
                mag = w.clip_max;
            }
            const float phase = lr[B + f];
            real[static_cast<size_t>(f) * T + t] = mag * std::cos(phase);
            imag[static_cast<size_t>(f) * T + t] = mag * std::sin(phase);
        }
    }
}

} // namespace tts_cpp::lavasr
