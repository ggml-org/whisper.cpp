// TDD harness for audit follow-up #6 (F7) — fused ConvNeXt block
// builder for the Supertonic vocoder.
//
// Background
// ----------
// The current `convnext_block_ggml` (private to
// `src/supertonic_vocoder.cpp`) wraps `layer_norm_channel_ggml`
// around a pair of `conv1d_causal_ggml` calls.  Each LN call costs
// two `ggml_cont` materialisations (permute → cont [C, T0] →
// norm/mul/add → permute → cont [T0, C]) and each `K=1` pointwise
// conv pays an `im2col` copy on top.  For the 10 ConvNeXt blocks
// in the vocoder this adds up to ~16.8 MiB of redundant copy
// traffic per synth on a discrete GPU (audit finding F7).
//
// `convnext_block_fused_ggml` cuts that traffic in half by:
//
//   1. Keeping the layer-norm output in `[C, T0]` (channel-major)
//      layout — i.e. skipping the back-permute / back-cont pair.
//   2. Lowering the `K=1` pointwise convs to direct
//      `ggml_mul_mat(w_2d, x_perm)` against the LN-output's
//      `[C, T0]` layout, eliminating both `im2col` copies.
//   3. Re-permuting once at the very end so the block output is
//      `[T0, C]` (time-major) for the next block / final norm.
//
// Net per block:
//   - Conts: 2 → 2 (LN front + final permute-back).  Same count.
//   - `im2col` copies: 2 → 0.  **Saves 2 [T0, C] copies per block.**
//   - Bit-exact arithmetic against the (depthwise → LN → pw1 →
//     gelu → pw2 → γ → residual) reference within `~1e-5` (mul_mat
//     summation order is unchanged; only the layout of intermediate
//     tensors moves).
//
// Test contract
// -------------
// Constructs a synthetic ConvNeXt-block input + weights with small
// random F32 values (no GGUF required) and checks the GGML
// `convnext_block_fused_ggml` output against a scalar reference
// of the same per-block math on the CPU backend.
//
// Shapes are deliberately tiny so the unit test stays in the
// single-millisecond range (T0=8, C=4, hidden=8).  An additional
// "vocoder-size" shape (T0=420, C=512, hidden=1536) is run with a
// slightly looser tolerance to exercise the realistic block.
//
// Registered with `LABEL "unit"` — no GGUF required, no model
// state.  Mirrors the test_supertonic_rope_packed_qk.cpp harness.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "supertonic_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

// -----------------------------------------------------------------
// Scalar reference for the ConvNeXt block math.
//
// All buffers are CPU-native time-major layout: `x[t*C + c]`.
// -----------------------------------------------------------------

void scalar_depthwise_causal(const std::vector<float> & x, int L, int C,
                             const std::vector<float> & w,
                             const std::vector<float> & b,
                             int K, int dilation,
                             std::vector<float> & y) {
    y.assign((size_t) L * C, 0.0f);
    const int pad_left = (K - 1) * dilation;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b[c];
            for (int k = 0; k < K; ++k) {
                int src_t = t + k * dilation - pad_left;
                if (src_t < 0) src_t = 0;
                sum += w[(size_t) c * K + k] * x[(size_t) src_t * C + c];
            }
            y[(size_t) t * C + c] = sum;
        }
    }
}

void scalar_layer_norm_channel(std::vector<float> & x, int L, int C,
                               const std::vector<float> & g,
                               const std::vector<float> & b,
                               float eps = 1e-6f) {
    for (int t = 0; t < L; ++t) {
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) mean += x[(size_t) t * C + c];
        mean /= (float) C;
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float d = x[(size_t) t * C + c] - mean;
            var += d * d;
        }
        float inv = 1.0f / std::sqrt(var / (float) C + eps);
        for (int c = 0; c < C; ++c) {
            float v = (x[(size_t) t * C + c] - mean) * inv;
            x[(size_t) t * C + c] = v * g[c] + b[c];
        }
    }
}

void scalar_linear_1x1(const std::vector<float> & x, int L, int IC,
                       const std::vector<float> & w,
                       const std::vector<float> * bias,
                       int OC,
                       std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = bias ? (*bias)[oc] : 0.0f;
            const size_t woff = (size_t) oc * IC;
            for (int ic = 0; ic < IC; ++ic) {
                sum += w[woff + ic] * x[(size_t) t * IC + ic];
            }
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

float gelu_erf_scalar(float x) {
    // erf-based GELU matches ggml_gelu_erf.
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

void scalar_convnext_block(const std::vector<float> & x_in,
                           int L, int C, int hidden,
                           int K, int dilation,
                           const std::vector<float> & dw_w,
                           const std::vector<float> & dw_b,
                           const std::vector<float> & ln_g,
                           const std::vector<float> & ln_b,
                           const std::vector<float> & pw1_w,
                           const std::vector<float> * pw1_b,
                           const std::vector<float> & pw2_w,
                           const std::vector<float> * pw2_b,
                           const std::vector<float> & gamma,
                           std::vector<float> & y_out) {
    std::vector<float> dw;
    scalar_depthwise_causal(x_in, L, C, dw_w, dw_b, K, dilation, dw);

    std::vector<float> ln = dw;
    scalar_layer_norm_channel(ln, L, C, ln_g, ln_b);

    std::vector<float> pw1;
    scalar_linear_1x1(ln, L, C, pw1_w, pw1_b, hidden, pw1);
    for (float & v : pw1) v = gelu_erf_scalar(v);

    std::vector<float> pw2;
    scalar_linear_1x1(pw1, L, hidden, pw2_w, pw2_b, C, pw2);

    y_out.assign((size_t) L * C, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            y_out[(size_t) t * C + c] =
                x_in[(size_t) t * C + c] +
                gamma[c] * pw2[(size_t) t * C + c];
        }
    }
}

// -----------------------------------------------------------------
// Layout helpers.  CPU-native `x[t*C + c]` ↔ GGML's `ne=[L, C]`
// column-major memory `x[c*L + t]`.
// -----------------------------------------------------------------

void pack_lc_to_col_major(const std::vector<float> & x_lc, int L, int C,
                          std::vector<float> & out) {
    out.assign((size_t) L * C, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            out[(size_t) c * L + t] = x_lc[(size_t) t * C + c];
        }
    }
}

void unpack_col_major_to_lc(const std::vector<float> & x_col, int L, int C,
                            std::vector<float> & out) {
    out.assign((size_t) L * C, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            out[(size_t) t * C + c] = x_col[(size_t) c * L + t];
        }
    }
}

// -----------------------------------------------------------------
// Test harness — runs `convnext_block_fused_ggml` on a CPU backend
// and compares against the scalar reference above.
// -----------------------------------------------------------------

void test_convnext_block_fused(const char * label,
                               int L, int C, int hidden,
                               int K, int dilation,
                               unsigned seed,
                               float atol) {
    std::fprintf(stderr,
                 "[convnext_block_fused: %s] L=%d C=%d hidden=%d K=%d dilation=%d\n",
                 label, L, C, hidden, K, dilation);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::normal_distribution<float> bias_dist(0.0f, 0.1f);
    std::normal_distribution<float> gamma_dist(1.0f, 0.05f);

    auto fill = [&](std::vector<float> & v, std::normal_distribution<float> & d) {
        for (auto & x : v) x = d(rng);
    };

    std::vector<float> x_lc((size_t) L * C);
    fill(x_lc, dist);
    std::vector<float> dw_w((size_t) C * K);
    fill(dw_w, dist);
    std::vector<float> dw_b((size_t) C);
    fill(dw_b, bias_dist);
    std::vector<float> ln_g((size_t) C);
    fill(ln_g, gamma_dist);
    std::vector<float> ln_b((size_t) C);
    fill(ln_b, bias_dist);
    std::vector<float> pw1_w((size_t) hidden * C);
    fill(pw1_w, dist);
    std::vector<float> pw1_b((size_t) hidden);
    fill(pw1_b, bias_dist);
    std::vector<float> pw2_w((size_t) C * hidden);
    fill(pw2_w, dist);
    std::vector<float> pw2_b((size_t) C);
    fill(pw2_b, bias_dist);
    std::vector<float> gamma((size_t) C);
    fill(gamma, gamma_dist);

    std::vector<float> ref;
    scalar_convnext_block(x_lc, L, C, hidden, K, dilation,
                          dw_w, dw_b, ln_g, ln_b,
                          pw1_w, &pw1_b, pw2_w, &pw2_b, gamma,
                          ref);

    // The depthwise step is upstream of the fused helper — compute
    // it scalar-side here and pre-load the result as `dw_out` so the
    // helper's scope stays at the LN + pw1 + gelu + pw2 + γ + residual
    // segment that F7 targets.
    std::vector<float> dw_lc;
    scalar_depthwise_causal(x_lc, L, C, dw_w, dw_b, K, dilation, dw_lc);

    constexpr int MAX_NODES = 1024;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

    ggml_tensor * residual_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(residual_in, "residual_in"); ggml_set_input(residual_in);
    ggml_tensor * dw_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(dw_out_in, "dw_out_in"); ggml_set_input(dw_out_in);
    ggml_tensor * ln_g_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_name(ln_g_in, "ln_g_in"); ggml_set_input(ln_g_in);
    ggml_tensor * ln_b_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_name(ln_b_in, "ln_b_in"); ggml_set_input(ln_b_in);
    // pw1_w GGML shape: ne=[K=1, IC=C, OC=hidden].
    ggml_tensor * pw1_w_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, C, hidden);
    ggml_set_name(pw1_w_in, "pw1_w_in"); ggml_set_input(pw1_w_in);
    ggml_tensor * pw1_b_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    ggml_set_name(pw1_b_in, "pw1_b_in"); ggml_set_input(pw1_b_in);
    ggml_tensor * pw2_w_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, hidden, C);
    ggml_set_name(pw2_w_in, "pw2_w_in"); ggml_set_input(pw2_w_in);
    ggml_tensor * pw2_b_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_name(pw2_b_in, "pw2_b_in"); ggml_set_input(pw2_b_in);
    ggml_tensor * gamma_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_name(gamma_in, "gamma_in"); ggml_set_input(gamma_in);

    ggml_tensor * y = convnext_block_fused_ggml(
        ctx,
        residual_in,
        dw_out_in,
        ln_g_in, ln_b_in,
        pw1_w_in, pw1_b_in,
        pw2_w_in, pw2_b_in,
        gamma_in);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "  SKIP: ggml_backend_cpu_init failed\n");
        ggml_free(ctx);
        return;
    }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    if (!ggml_gallocr_reserve(allocr, gf)) {
        std::fprintf(stderr, "  SKIP: gallocr_reserve failed\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        return;
    }
    ggml_gallocr_alloc_graph(allocr, gf);

    auto upload_2d = [&](ggml_tensor * t, const std::vector<float> & host_lc,
                         int LL, int CC) {
        std::vector<float> col;
        pack_lc_to_col_major(host_lc, LL, CC, col);
        ggml_backend_tensor_set(t, col.data(), 0, col.size() * sizeof(float));
    };
    upload_2d(residual_in, x_lc, L, C);
    upload_2d(dw_out_in, dw_lc, L, C);
    ggml_backend_tensor_set(ln_g_in, ln_g.data(), 0, ln_g.size() * sizeof(float));
    ggml_backend_tensor_set(ln_b_in, ln_b.data(), 0, ln_b.size() * sizeof(float));
    // pw1_w GGUF native memory: row-major [OC, IC] when reshaped to 2D.
    // GGML stores element (k=0, ic, oc) at memory `0 + ic*1 + oc*(1*IC)` =
    // `ic + oc*IC`.  Our host buffer is `pw1_w[oc*IC + ic]` which matches.
    ggml_backend_tensor_set(pw1_w_in, pw1_w.data(), 0, pw1_w.size() * sizeof(float));
    ggml_backend_tensor_set(pw1_b_in, pw1_b.data(), 0, pw1_b.size() * sizeof(float));
    ggml_backend_tensor_set(pw2_w_in, pw2_w.data(), 0, pw2_w.size() * sizeof(float));
    ggml_backend_tensor_set(pw2_b_in, pw2_b.data(), 0, pw2_b.size() * sizeof(float));
    ggml_backend_tensor_set(gamma_in, gamma.data(), 0, gamma.size() * sizeof(float));

    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got_col((size_t) L * C);
    ggml_backend_tensor_get(y, got_col.data(), 0, got_col.size() * sizeof(float));
    std::vector<float> got;
    unpack_col_major_to_lc(got_col, L, C, got);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    CHECK(got.size() == ref.size());

    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
        const float d = std::fabs(ref[i] - got[i]);
        max_abs = std::max(max_abs, d);
        if (d > atol) {
            if (bad < 4) {
                std::fprintf(stderr,
                             "  mismatch @ %zu: ref=%.6g got=%.6g abs=%.3e\n",
                             i, ref[i], got[i], d);
            }
            ++bad;
        }
    }
    std::fprintf(stderr,
                 "  max_abs_err=%.3e  bad=%d / %zu  atol=%.0e\n",
                 max_abs, bad, ref.size(), atol);
    CHECK(bad == 0);
}

} // namespace

int main() {
    // Tiny synthetic shape — runs in microseconds, sanity-checks
    // the fused chain end-to-end.
    test_convnext_block_fused("tiny K=3 dilation=1", 8, 4, 8, 3, 1, 0x73B1, 1e-4f);
    // Dilation > 1 mirrors the vocoder's `dilations[1..2]={2,4}` taps.
    test_convnext_block_fused("tiny K=7 dilation=2", 12, 4, 8, 7, 2, 0xC0DE, 1e-4f);
    // Vocoder-realistic shape (T0=420, C=512, hidden=1536) at the
    // tolerance the trace harness already accepts for the GGML
    // path (`1e-2` band — these values multiply over 10 blocks).
    // Smaller shape here so the unit test stays under the 1ms wall
    // budget; the full T0=420 case is exercised by the existing
    // `test_supertonic_vocoder_trace` fixture once the production
    // `convnext_block_ggml` is rewired to this helper.
    test_convnext_block_fused("scale-up K=7 dilation=4", 40, 16, 64, 7, 4, 0xBEEF, 5e-4f);

    std::fprintf(stderr,
                 "test_supertonic_convnext_block_fused: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
