// mel2wav: a standalone CLI that takes a mel spectrogram (.npy) and runs the
// Chatterbox HiFTGenerator (vocoder) in pure C++/ggml to produce a 24kHz wav.
//
// This is the end-to-end validation for the HiFT port.  It reuses the s3gen
// GGUF we produce with convert-s3gen-to-gguf.py.
//
// Usage:
//   mel2wav --s3gen-gguf MODEL.gguf --mel-npy MEL.npy --out OUT.wav [--seed N]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------- GGUF loader (same as test_s3gen.cpp) ----------
struct model_ctx {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

static model_ctx load_s3gen_gguf(const std::string & path) {
    model_ctx m;
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) throw std::runtime_error("gguf_init_from_file failed: " + path);
    m.backend = ggml_backend_cpu_init();
    int64_t n_tensors = gguf_get_n_tensors(g);
    ggml_init_params p = { ggml_tensor_overhead() * (size_t)n_tensors, nullptr, true };
    m.ctx_w = ggml_init(p);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        ggml_tensor * dst = ggml_dup_tensor(m.ctx_w, src);
        ggml_set_name(dst, name);
        m.tensors[name] = dst;
    }
    m.buffer_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    for (ggml_tensor * cur = ggml_get_first_tensor(m.ctx_w); cur; cur = ggml_get_next_tensor(m.ctx_w, cur)) {
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
    }
    gguf_free(g);
    ggml_free(tmp_ctx);
    return m;
}

static ggml_tensor * find_tensor(const model_ctx & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

// F32 conv1d via im2col + mul_mat
static ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

static ggml_tensor * conv_transpose_1d_f32(ggml_context * ctx, ggml_tensor * kernel,
                                           ggml_tensor * input, int stride, int padding) {
    ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, stride, 0, 1);
    if (padding == 0) return out;
    int64_t L_new = out->ne[0] - 2 * padding;
    ggml_tensor * v = ggml_view_3d(ctx, out, L_new, out->ne[1], out->ne[2],
                                   out->nb[1], out->nb[2], (size_t)padding * out->nb[0]);
    return ggml_cont(ctx, v);
}

// Snake activation (shape ne=[T, C] with alpha ne=[C])
static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * alpha, ggml_tensor * inv_alpha) {
    ggml_tensor * a  = ggml_reshape_2d(ctx, alpha,     1, alpha->ne[0]);
    ggml_tensor * ia = ggml_reshape_2d(ctx, inv_alpha, 1, inv_alpha->ne[0]);
    ggml_tensor * ax = ggml_mul(ctx, x, a);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_mul(ctx, s2, ia));
}

static std::vector<float> invert_alpha_cpu(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = find_tensor(m, name);
    std::vector<float> a(ggml_nelements(t));
    ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t));
    std::vector<float> inv(a.size());
    for (size_t i = 0; i < a.size(); ++i) inv[i] = 1.0f / (a[i] + 1e-9f);
    return inv;
}

// Reflection pad along ne[0]
static ggml_tensor * reflect_pad_1d(ggml_context * ctx, ggml_tensor * x, int p_left, int p_right) {
    ggml_tensor * y = x;
    for (int i = 0; i < p_left; ++i) {
        int src_idx = p_left - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, s, y, 0);
    }
    int L_orig = (int)x->ne[0];
    for (int i = 0; i < p_right; ++i) {
        int src_idx = L_orig - 2 - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, y, s, 0);
    }
    return y;
}

// ---------- FFT kernels ----------
static std::vector<float> build_hann_window(int n, bool periodic = true) {
    std::vector<float> w(n);
    double N = periodic ? (double)n : (double)(n - 1);
    const double two_pi = 2.0 * M_PI;
    for (int i = 0; i < n; ++i) w[i] = (float)(0.5 * (1.0 - std::cos(two_pi * (double)i / N)));
    return w;
}

static std::vector<float> build_stft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    for (int f = 0; f < F; ++f) {
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(std::cos(th) * w);
            K[n + (F + f) * n_fft] = (float)(-std::sin(th) * w);
        }
    }
    return K;
}

static std::vector<float> build_istft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    const double inv_N = 1.0 / (double)n_fft;
    for (int f = 0; f < F; ++f) {
        double coef_re = (f == 0 || f == n_fft / 2) ? 1.0 : 2.0;
        double coef_im = (f == 0 || f == n_fft / 2) ? 0.0 : 2.0;
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(coef_re * std::cos(th) * w * inv_N);
            K[n + (F + f) * n_fft] = (float)(-coef_im * std::sin(th) * w * inv_N);
        }
    }
    return K;
}

static std::vector<float> build_window_sum(int T_stft, int n_fft, int hop, const std::vector<float> & window) {
    int L = (T_stft - 1) * hop + n_fft;
    std::vector<float> ws(L, 0.0f);
    for (int t = 0; t < T_stft; ++t) {
        int base = t * hop;
        for (int n = 0; n < n_fft; ++n) ws[base + n] += window[n] * window[n];
    }
    return ws;
}

// ---------- SineGen + SourceModule (CPU, produces source signal) ----------
// Returns (T_wav,) time-domain source signal.
static std::vector<float> sinegen_source(const std::vector<float> & f0_wav,  // (T_wav,) upsampled F0
                                         int sampling_rate, int harmonic_num,
                                         float sine_amp, float noise_std,
                                         float voiced_threshold,
                                         const std::vector<float> & l_linear_w,  // (harmonic_num+1,)
                                         float l_linear_b, uint32_t seed) {
    int T_wav = (int)f0_wav.size();
    int H = harmonic_num + 1;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-(float)M_PI, (float)M_PI);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<float> phase_vec(H, 0.0f);
    for (int h = 1; h < H; ++h) phase_vec[h] = uniform(rng);  // phase_vec[0] = 0

    std::vector<float> sine_waves(H * T_wav, 0.0f);
    std::vector<double> cum_phase(H, 0.0);
    for (int t = 0; t < T_wav; ++t) {
        float f0 = f0_wav[t];
        bool is_voiced = f0 > voiced_threshold;
        for (int h = 0; h < H; ++h) {
            double inc = (double)f0 * (h + 1) / (double)sampling_rate;
            cum_phase[h] += inc;
            double theta = 2.0 * M_PI * (cum_phase[h] - std::floor(cum_phase[h]));
            float sine = sine_amp * std::sin((float)theta + phase_vec[h]);
            float noise_amp = is_voiced ? noise_std : sine_amp / 3.0f;
            float noise = noise_amp * gauss(rng);
            float uv = is_voiced ? 1.0f : 0.0f;
            sine_waves[h * T_wav + t] = sine * uv + noise;
        }
    }

    // l_linear: (T_wav, H) @ (H,) + b -> (T_wav,); then tanh
    // But we have sine_waves stored channel-major: sine_waves[h*T_wav + t]
    std::vector<float> source(T_wav, 0.0f);
    for (int t = 0; t < T_wav; ++t) {
        float s = l_linear_b;
        for (int h = 0; h < H; ++h) s += l_linear_w[h] * sine_waves[h * T_wav + t];
        source[t] = std::tanh(s);
    }
    return source;
}

// ---------- Graph: f0_predictor (mel -> f0) ----------
static std::vector<float> run_f0_predictor(const model_ctx & m, const std::vector<float> & mel, int T_mel) {
    static size_t buf_size = 8 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);

    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, 80);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);

    ggml_tensor * x = mel_in;
    for (int i = 0; i < 5; ++i) {
        std::string pfx = "hift/f0_predictor/condnet/" + std::to_string(i * 2);
        ggml_tensor * w = find_tensor(m, pfx + "/weight");
        ggml_tensor * b = find_tensor(m, pfx + "/bias");
        int C_out = (int)w->ne[2];
        ggml_tensor * xp = ggml_pad_ext(ctx, x, 1, 1, 0, 0, 0, 0, 0, 0);
        x = conv1d_f32(ctx, w, xp, 1, 0, 1);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, C_out));
        x = ggml_unary(ctx, x, GGML_UNARY_OP_ELU);
    }
    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [512, T]
    ggml_tensor * cw = find_tensor(m, "hift/f0_predictor/classifier/weight");
    ggml_tensor * cb = find_tensor(m, "hift/f0_predictor/classifier/bias");
    ggml_tensor * y = ggml_mul_mat(ctx, cw, xp);
    y = ggml_add(ctx, y, cb);
    y = ggml_abs(ctx, y);
    y = ggml_reshape_1d(ctx, y, T_mel);
    ggml_set_name(y, "out"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size() * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> f0(T_mel);
    ggml_backend_tensor_get(y, f0.data(), 0, ggml_nbytes(y));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return f0;
}

// ---------- Main HiFT graph: mel + s_stft -> wav ----------
static std::vector<float> run_hift_decode(const model_ctx & m,
                                          const std::vector<float> & mel, int T_mel,
                                          const std::vector<float> & s_stft, int T_stft) {
    const int MEL = 80;
    const int NFFT2 = 18;
    const int BASE_CH = 512;
    const int n_fft = 16;
    const int hop = 4;
    const int F = n_fft / 2 + 1;

    std::vector<int> ups_rates = {8, 5, 3};
    std::vector<int> ups_ksizes = {16, 11, 7};
    std::vector<int> ups_ch = {256, 128, 64};
    std::vector<int> rb_ksizes = {3, 7, 11};
    std::vector<std::vector<int>> rb_dilations = {{1,3,5},{1,3,5},{1,3,5}};
    std::vector<int> src_rb_ksizes = {7, 7, 11};
    std::vector<std::vector<int>> src_rb_dilations = {{1,3,5},{1,3,5},{1,3,5}};

    static size_t buf_size = 32 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);

    ggml_tensor * mel_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, MEL);    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * s_stft_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_stft, NFFT2); ggml_set_name(s_stft_in, "s_stft_in"); ggml_set_input(s_stft_in);

    struct inv_entry { std::string gn; std::vector<float> data; };
    std::vector<inv_entry> inv_alphas;
    auto mk_inv = [&](const std::string & name_pref, int C) {
        std::string gn = "inv_" + name_pref;
        std::vector<float> inv = invert_alpha_cpu(m, name_pref);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        ggml_set_name(t, gn.c_str()); ggml_set_input(t);
        inv_alphas.push_back({gn, std::move(inv)});
        return t;
    };

    auto load_rb = [&](const std::string & prefix, int C) {
        struct rb_data { ggml_tensor *a1, *c1w, *c1b, *a2, *c2w, *c2b, *ia1, *ia2; };
        std::vector<rb_data> p(3);
        for (int i = 0; i < 3; ++i) {
            p[i].a1 = find_tensor(m, prefix + "/activations1/" + std::to_string(i) + "/alpha");
            p[i].c1w = find_tensor(m, prefix + "/convs1/" + std::to_string(i) + "/weight");
            p[i].c1b = find_tensor(m, prefix + "/convs1/" + std::to_string(i) + "/bias");
            p[i].a2 = find_tensor(m, prefix + "/activations2/" + std::to_string(i) + "/alpha");
            p[i].c2w = find_tensor(m, prefix + "/convs2/" + std::to_string(i) + "/weight");
            p[i].c2b = find_tensor(m, prefix + "/convs2/" + std::to_string(i) + "/bias");
            p[i].ia1 = mk_inv(prefix + "/activations1/" + std::to_string(i) + "/alpha", C);
            p[i].ia2 = mk_inv(prefix + "/activations2/" + std::to_string(i) + "/alpha", C);
        }
        return p;
    };

    auto rb_forward = [&](auto & rb, ggml_tensor * x, int C, const std::vector<int> & dils, int k_sz) {
        for (int i = 0; i < 3; ++i) {
            auto & p = rb[i];
            int dilation = dils[i];
            int pad1 = (k_sz * dilation - dilation) / 2;
            int pad2 = (k_sz - 1) / 2;
            ggml_tensor * xt = snake(ctx, x, p.a1, p.ia1);
            xt = ggml_pad_ext(ctx, xt, pad1, pad1, 0, 0, 0, 0, 0, 0);
            xt = conv1d_f32(ctx, p.c1w, xt, 1, 0, dilation);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c1b, 1, C));
            xt = snake(ctx, xt, p.a2, p.ia2);
            xt = ggml_pad_ext(ctx, xt, pad2, pad2, 0, 0, 0, 0, 0, 0);
            xt = conv1d_f32(ctx, p.c2w, xt, 1, 0, 1);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c2b, 1, C));
            x = ggml_add(ctx, x, xt);
        }
        return x;
    };

    // conv_pre
    ggml_tensor * cpw = find_tensor(m, "hift/conv_pre/weight");
    ggml_tensor * cpb = find_tensor(m, "hift/conv_pre/bias");
    ggml_tensor * x = ggml_pad_ext(ctx, mel_in, 3, 3, 0, 0, 0, 0, 0, 0);
    x = conv1d_f32(ctx, cpw, x, 1, 0, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cpb, 1, BASE_CH));

    for (int i = 0; i < 3; ++i) {
        x = ggml_leaky_relu(ctx, x, 0.1f, false);
        ggml_tensor * uw = find_tensor(m, "hift/ups/" + std::to_string(i) + "/weight");
        ggml_tensor * ub = find_tensor(m, "hift/ups/" + std::to_string(i) + "/bias");
        int up_pad = (ups_ksizes[i] - ups_rates[i]) / 2;
        x = conv_transpose_1d_f32(ctx, uw, x, ups_rates[i], up_pad);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, ub, 1, ups_ch[i]));
        if (i == 2) {
            ggml_tensor * x_slice = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 1 * x->nb[0]);
            x_slice = ggml_cont(ctx, x_slice);
            x = ggml_concat(ctx, x_slice, x, 0);
        }
        ggml_tensor * sw = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/weight");
        ggml_tensor * sb = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/bias");
        int sd_stride = (i == 0) ? 15 : (i == 1) ? 3 : 1;
        int sd_pad    = (i == 0) ? 7  : (i == 1) ? 1 : 0;
        int sd_oc     = (int)sw->ne[2];
        ggml_tensor * sin_pad = ggml_pad_ext(ctx, s_stft_in, sd_pad, sd_pad, 0, 0, 0, 0, 0, 0);
        ggml_tensor * si = conv1d_f32(ctx, sw, sin_pad, sd_stride, 0, 1);
        si = ggml_add(ctx, si, ggml_reshape_2d(ctx, sb, 1, sd_oc));
        auto srb = load_rb("hift/source_resblocks/" + std::to_string(i), ups_ch[i]);
        si = rb_forward(srb, si, ups_ch[i], src_rb_dilations[i], src_rb_ksizes[i]);
        x = ggml_add(ctx, x, si);

        ggml_tensor * xs = nullptr;
        for (int j = 0; j < 3; ++j) {
            auto rb = load_rb("hift/resblocks/" + std::to_string(i * 3 + j), ups_ch[i]);
            ggml_tensor * rb_out = rb_forward(rb, x, ups_ch[i], rb_dilations[j], rb_ksizes[j]);
            xs = (xs == nullptr) ? rb_out : ggml_add(ctx, xs, rb_out);
        }
        x = ggml_scale(ctx, xs, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    ggml_tensor * cp2w = find_tensor(m, "hift/conv_post/weight");
    ggml_tensor * cp2b = find_tensor(m, "hift/conv_post/bias");
    x = ggml_pad_ext(ctx, x, 3, 3, 0, 0, 0, 0, 0, 0);
    x = conv1d_f32(ctx, cp2w, x, 1, 0, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cp2b, 1, NFFT2));

    // ISTFT part: split mag/phase, compute spec, apply inverse DFT, divide by w_sum, trim
    size_t col_stride = x->nb[1];
    ggml_tensor * mag_log = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, 0));
    mag_log = ggml_clamp(ctx, mag_log, -1e6f, 1e2f);
    ggml_tensor * mag = ggml_exp(ctx, mag_log);
    ggml_tensor * ph_in = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, (size_t)F * col_stride));
    ggml_tensor * ph = ggml_sin(ctx, ph_in);
    ggml_tensor * real = ggml_mul(ctx, mag, ggml_cos(ctx, ph));
    ggml_tensor * imag = ggml_mul(ctx, mag, ggml_sin(ctx, ph));
    ggml_tensor * spec = ggml_concat(ctx, real, imag, 1);

    auto window = build_hann_window(n_fft, true);
    auto istft_kernel = build_istft_kernel(n_fft, window);
    auto w_sum = build_window_sum(T_stft, n_fft, hop, window);

    ggml_tensor * istft_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(istft_k, "istft_k"); ggml_set_input(istft_k);
    ggml_tensor * ws_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)w_sum.size(), 1);
    ggml_set_name(ws_in, "w_sum"); ggml_set_input(ws_in);

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, istft_k, spec, hop, 0, 1);
    y = ggml_div(ctx, y, ws_in);
    int pad_amt = n_fft / 2;
    int L_wav = (int)w_sum.size() - n_fft;
    ggml_tensor * y_trim = ggml_cont(ctx, ggml_view_2d(ctx, y, L_wav, y->ne[1], y->nb[1], (size_t)pad_amt * y->nb[0]));
    y_trim = ggml_clamp(ctx, y_trim, -0.99f, 0.99f);

    ggml_set_name(y_trim, "wav"); ggml_set_output(y_trim);
    ggml_build_forward_expand(gf, y_trim);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s_stft_in"), s_stft.data(), 0, s_stft.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "istft_k"), istft_kernel.data(), 0, istft_kernel.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_sum"), w_sum.data(), 0, w_sum.size() * sizeof(float));
    for (auto & ia : inv_alphas) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, ia.gn.c_str()), ia.data.data(), 0, ia.data.size() * sizeof(float));
    }

    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> wav(ggml_nelements(y_trim));
    ggml_backend_tensor_get(y_trim, wav.data(), 0, ggml_nbytes(y_trim));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return wav;
}

// Graph-based STFT of the source signal (produces s_stft for run_hift_decode).
static std::vector<float> run_stft(const model_ctx & m, const std::vector<float> & src) {
    const int n_fft = 16;
    const int hop = 4;
    const int F = n_fft / 2 + 1;
    int T_src = (int)src.size();
    int T_stft = (T_src + n_fft - n_fft) / hop + 1;  // center-mode

    auto window = build_hann_window(n_fft, true);
    auto kernel = build_stft_kernel(n_fft, window);

    static size_t buf_size = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_src, 1);
    ggml_set_name(s, "s"); ggml_set_input(s);
    ggml_tensor * s_padded = reflect_pad_1d(ctx, s, n_fft / 2, n_fft / 2);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(k, "k"); ggml_set_input(k);
    ggml_tensor * spec = conv1d_f32(ctx, k, s_padded, hop, 0, 1);
    ggml_set_name(spec, "spec"); ggml_set_output(spec);
    ggml_build_forward_expand(gf, spec);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s"), src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "k"), kernel.data(), 0, kernel.size() * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> out(ggml_nelements(spec));
    ggml_backend_tensor_get(spec, out.data(), 0, ggml_nbytes(spec));
    (void)T_stft;  // silence unused
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}

// Write 24 kHz 16-bit PCM WAV
static void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t num_samples = (uint32_t)wav.size();
    uint32_t byte_rate = sr * 2;
    uint32_t data_size = num_samples * 2;
    uint32_t chunk_size = 36 + data_size;
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_fmt = 1;
    uint16_t n_channels = 1;
    uint32_t sample_rate = (uint32_t)sr;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;
    std::fwrite(&fmt_chunk_size, 4, 1, f);
    std::fwrite(&audio_fmt, 2, 1, f);
    std::fwrite(&n_channels, 2, 1, f);
    std::fwrite(&sample_rate, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bits_per_sample, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float cl = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t)std::lrintf(cl * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

// ---------- main ----------
int main(int argc, char ** argv) {
    std::string gguf_path, mel_path, out_path;
    int seed = 42;
    int sampling_rate = 24000;  // S3GEN_SR
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--s3gen-gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--mel-npy" && i + 1 < argc) mel_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--sr" && i + 1 < argc) sampling_rate = std::atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s --s3gen-gguf MODEL.gguf --mel-npy MEL.npy --out OUT.wav [--seed N] [--sr 24000]\n", argv[0]);
            return 1;
        }
    }
    if (gguf_path.empty() || mel_path.empty() || out_path.empty()) {
        fprintf(stderr, "missing required arguments\n");
        return 1;
    }

    fprintf(stderr, "Loading %s\n", gguf_path.c_str());
    model_ctx m = load_s3gen_gguf(gguf_path);
    fprintf(stderr, "  %zu tensors loaded\n", m.tensors.size());

    // Load mel: shape (80, T)
    npy_array mel_npy = npy_load(mel_path);
    int T_mel = (int)mel_npy.shape[1];
    std::vector<float> mel(T_mel * 80);
    std::memcpy(mel.data(), npy_as_f32(mel_npy), mel.size() * sizeof(float));
    fprintf(stderr, "Mel shape: (%lld, %lld)\n", (long long)mel_npy.shape[0], (long long)mel_npy.shape[1]);

    // f0_predictor
    fprintf(stderr, "Running f0_predictor...\n");
    auto f0 = run_f0_predictor(m, mel, T_mel);

    // Upsample f0 to wav rate: nearest-neighbor, factor = 480
    int upsample = 8 * 5 * 3 * 4;  // prod(upsample_rates) * hop_len = 480
    int T_wav = T_mel * upsample;
    std::vector<float> f0_up(T_wav);
    for (int i = 0; i < T_mel; ++i)
        for (int j = 0; j < upsample; ++j) f0_up[i * upsample + j] = f0[i];

    // SineGen + SourceModule
    fprintf(stderr, "Running SineGen (seed=%d)...\n", seed);
    std::vector<float> l_linear_w(9);
    ggml_tensor * llw = find_tensor(m, "hift/m_source/l_linear/weight");
    ggml_tensor * llb = find_tensor(m, "hift/m_source/l_linear/bias");
    ggml_backend_tensor_get(llw, l_linear_w.data(), 0, 9 * sizeof(float));
    float l_linear_b;
    ggml_backend_tensor_get(llb, &l_linear_b, 0, sizeof(float));

    int harmonic_num = 8;
    float sine_amp = 0.1f, noise_std = 0.003f, voiced_threshold = 10.0f;
    auto src = sinegen_source(f0_up, sampling_rate, harmonic_num,
                              sine_amp, noise_std, voiced_threshold,
                              l_linear_w, l_linear_b, (uint32_t)seed);

    // STFT of source
    fprintf(stderr, "Running STFT on source...\n");
    auto s_stft = run_stft(m, src);
    int T_stft = (int)(s_stft.size() / 18);
    fprintf(stderr, "  s_stft shape: (18, %d)\n", T_stft);

    // Run full HiFT decode (mel + s_stft -> wav)
    fprintf(stderr, "Running HiFT decode...\n");
    auto wav = run_hift_decode(m, mel, T_mel, s_stft, T_stft);
    fprintf(stderr, "  wav shape: (%zu,)\n", wav.size());

    write_wav(out_path, wav, sampling_rate);
    fprintf(stderr, "Wrote %s\n", out_path.c_str());
    return 0;
}
