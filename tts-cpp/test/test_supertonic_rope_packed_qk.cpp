// TDD harness for the audit follow-up #5 packed-QK RoPE helper
// (F23 = F20 integration: bake apply_rope into the Q/K-producing
// group / front-block graph so the host doesn't run apply_rope
// between the QKV download and the flash-attention upload).
//
// Background
// ----------
// `apply_rope_in_graph` (landed in PR #4) operates on a
// `ne=[head_dim, n_heads, L]` tensor — the natural layout the
// scalar `apply_rope` indexes into.  But every actual call site
// in the vector estimator produces Q/K via
// `dense_matmul_time_ggml`, whose output is a packed 2D tensor
// with `ne=[H*D, L]` (`H*D` = the attention `A` dimension =
// `n_heads * head_dim`, packed channel-major along axis 0).
//
// `apply_rope_to_packed_qk` is a thin wrapper that re-views the
// packed tensor as `[head_dim, n_heads, L]` (zero-cost view via
// stride trick), materialises a contiguous copy so downstream
// `ggml_concat` is happy, calls `apply_rope_in_graph`, and
// reshapes the result back to the original `[H*D, L]` packed
// shape.  No new ops introduced — just a packing-layout adapter.
//
// Test contract
// -------------
// Build a synthetic Q packed in `[H*D, L]` time-major layout
// (rows = time-frames, channels = `h*D + d` packed).  Verify on
// the CPU backend that `apply_rope_to_packed_qk(ctx, Q, cos, sin)`
// produces the same buffer as the scalar `apply_rope` would have
// written if Q had been laid out as `[t*H + h]*D + d`.
//
// Two parity shapes:
//   1. Vector-estimator Q: q_len = 20, n_heads = 4, head_dim = 64
//      → ne[0] = 256, ne[1] = 20.
//   2. Vector-estimator K: kv_len = 32 (text_len), n_heads = 4,
//      head_dim = 64 → ne[0] = 256, ne[1] = 32.
//
// Tolerance: `1e-4` absolute — same band as
// `test_supertonic_rope_in_graph.cpp` (the inner helper).
//
// Registered with `LABEL "unit"` — no GGUF required.

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

// Mirror of the in-tree scalar `apply_rope` (private to
// supertonic_vector_estimator.cpp).  Index layout matches the
// `data[t*H*D + h*D + d]` arrangement that the dense-matmul
// output produces when reshaped as `[t * (H*D) + (h*D + d)]` —
// i.e., the [H*D, L] packed tensor's element (col=h*D+d, row=t)
// is at the same memory location as scalar's i1 / i2.
void scalar_apply_rope(const float * theta,
                       std::vector<float> & x,
                       int L, int H, int D) {
    int half = D / 2;
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < L; ++t) {
            for (int d = 0; d < half; ++d) {
                const float angle = ((float) t / (float) L) * theta[d];
                const float cs = std::cos(angle);
                const float sn = std::sin(angle);
                const size_t i1 = ((size_t) t * H + h) * D + d;
                const size_t i2 = ((size_t) t * H + h) * D + half + d;
                const float a = x[i1];
                const float b = x[i2];
                x[i1] = a * cs - b * sn;
                x[i2] = b * cs + a * sn;
            }
        }
    }
}

// Build a randomized Q in the packed [H*D, L] time-major layout
// (a single contiguous row-major buffer where element (col, row)
// sits at `row * H*D + col`).  The same buffer interpreted under
// the scalar layout `data[t*H*D + h*D + d]` matches when col is
// decoded as `h*D + d` and row as `t`.  So scalar_apply_rope can
// be invoked directly on the same buffer for the reference.
void test_packed_rope_shape(const char * label, int L, int n_heads, int head_dim,
                            unsigned seed) {
    std::fprintf(stderr, "[apply_rope_to_packed_qk: %s]  L=%d H=%d D=%d\n",
                 label, L, n_heads, head_dim);

    const int HD = n_heads * head_dim;
    const int half = head_dim / 2;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> theta(half);
    for (auto & v : theta) v = std::abs(dist(rng)) * 1000.0f;

    // Packed Q buffer: ne=[HD, L], element (col, row) at memory
    // index `row * HD + col`.  Random init.
    std::vector<float> q_packed((size_t) L * HD);
    for (auto & v : q_packed) v = dist(rng);

    // Reference: scalar_apply_rope writes via index `t*H*D + h*D + d`
    // = `t*HD + (h*D + d)`.  For (col=h*D+d, row=t), memory is the
    // SAME index.  So the scalar in-place rotation applied to a
    // copy of q_packed gives our reference vector.
    std::vector<float> ref = q_packed;
    scalar_apply_rope(theta.data(), ref, L, n_heads, head_dim);

    // Host-side cos/sin tables exactly like make_rope_cos_sin_tables
    // would write: ne=[half, L], element (d, t) at index `t*half + d`.
    std::vector<float> cos_host, sin_host;
    make_rope_cos_sin_tables(theta.data(), L, half, cos_host, sin_host);

    // Build the helper's graph on the CPU backend.
    constexpr int MAX_NODES = 256;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // q input: ne=[HD, L] (axis 0 = packed channels, axis 1 = time).
    ggml_tensor * q_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HD, L);
    ggml_set_name(q_in, "q_in"); ggml_set_input(q_in);
    ggml_tensor * cos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_tensor * sin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    ggml_tensor * y = apply_rope_to_packed_qk(ctx, q_in, cos_in, sin_in,
                                              n_heads, head_dim);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    // Run on CPU backend.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "  SKIP: ggml_backend_cpu_init failed\n");
        ggml_free(ctx);
        return;
    }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(q_in,   q_packed.data(), 0, q_packed.size() * sizeof(float));
    ggml_backend_tensor_set(cos_in, cos_host.data(), 0, cos_host.size() * sizeof(float));
    ggml_backend_tensor_set(sin_in, sin_host.data(), 0, sin_host.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    // Shape contract: output must have the same total nelements
    // as q_packed (so a downstream ggml_set_output + tensor_to_time_
    // channel can consume it identically).
    CHECK(got.size() == q_packed.size());

    int bad = 0;
    float max_abs = 0.0f;
    const float atol = 1e-4f;
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
                 "  max_abs_err=%.3e  bad=%d / %zu\n",
                 max_abs, bad, ref.size());
    CHECK(bad == 0);
}

// Trip-wire: when L=1 the helper still needs to work (degenerate
// time axis matches the way the front-block builds a single-step
// graph after the convnext + time-add path).
void test_packed_rope_l1() {
    std::fprintf(stderr, "[apply_rope_to_packed_qk: L=1 degenerate]\n");
    const int L = 1, n_heads = 2, head_dim = 8;
    const int HD = n_heads * head_dim;
    const int half = head_dim / 2;

    std::vector<float> theta(half, 100.0f);
    std::vector<float> q_packed((size_t) L * HD, 1.0f);

    // At L=1 the angle is 0/1 * theta = 0, so cos=1, sin=0, and the
    // rotation is identity.  Catches a regression where the helper
    // accidentally divides by L or swaps the angle formula.
    std::vector<float> ref = q_packed;
    scalar_apply_rope(theta.data(), ref, L, n_heads, head_dim);

    std::vector<float> cos_host, sin_host;
    make_rope_cos_sin_tables(theta.data(), L, half, cos_host, sin_host);

    constexpr int MAX_NODES = 64;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * q_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HD, L);
    ggml_set_input(q_in);
    ggml_tensor * cos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_set_input(cos_in);
    ggml_tensor * sin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_set_input(sin_in);

    ggml_tensor * y = apply_rope_to_packed_qk(ctx, q_in, cos_in, sin_in,
                                              n_heads, head_dim);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); std::fprintf(stderr, "  SKIP\n"); return; }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(q_in,   q_packed.data(), 0, q_packed.size() * sizeof(float));
    ggml_backend_tensor_set(cos_in, cos_host.data(), 0, cos_host.size() * sizeof(float));
    ggml_backend_tensor_set(sin_in, sin_host.data(), 0, sin_host.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);
    std::vector<float> got((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
        const float d = std::fabs(ref[i] - got[i]);
        max_abs = std::max(max_abs, d);
        if (d > 1e-5f) ++bad;
    }
    std::fprintf(stderr, "  L=1 max_abs=%.3e bad=%d\n", max_abs, bad);
    CHECK(bad == 0);
}

} // namespace

int main() {
    // Vector-estimator hot shapes (q_len, kv_len typical sizes).
    test_packed_rope_shape("vector-estimator q", 20, 4, 64, 0xA51C);
    test_packed_rope_shape("vector-estimator k", 32, 4, 64, 0xC0FF);
    test_packed_rope_l1();

    std::fprintf(stderr,
                 "test_supertonic_rope_packed_qk: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
