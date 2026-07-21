// TDD harness for the audit follow-up #4 RoPE-in-graph helper
// (F20 partial, Phase 2H in `aiDocs/PLAN_SUPERTONIC_OPENCL.md`).
//
// Background
// ----------
// The vector estimator's `apply_rope` is the last hot-path op
// still running on the CPU between two GPU graph computes.  Every
// per-step / per-attention-site sequence is:
//
//     QKV graph compute  → host download Q,K
//     CPU apply_rope on Q (40 calls / synth on the default
//                         5-step × 4-group + 1-front-block schedule)
//     CPU apply_rope on K
//     host upload Q,K  →  flash-attention graph compute
//
// Supertonic's `apply_rope` is non-standard:
//
//     angle = (t / L) * theta[d]          // ← `t/L`, not `t * base^(-2i/D)`
//     cs = cos(angle), sn = sin(angle)
//     i1 = (t*H + h)*D + d                // d in [0, half)
//     i2 = (t*H + h)*D + half + d
//     x[i1], x[i2] := x[i1]*cs - x[i2]*sn,
//                     x[i2]*cs + x[i1]*sn
//
// `ggml_rope` / `ggml_rope_ext` compute their own θ from
// `(position, base, freq_scale)` — they CAN'T match this formula
// directly because the angle scales with `t/L` (position fraction
// of total length, not absolute position).  The partial F20 lands
// here is the host-precomputed-cos/sin variant:
//
//   1. Host precomputes `cos[half, L] = cos((t/L) * theta[d])`
//      and `sin[half, L]` once per (L, θ) and uploads as graph
//      inputs.
//   2. `apply_rope_in_graph(ctx, x, cos_table, sin_table)` runs
//      the rotation entirely with universally-supported ops
//      (`view`, `repeat`, `mul`, `sub`, `add`, `concat`) — no
//      patched `ggml_sin` / `ggml_cos` / `ggml_rope` needed, so
//      it runs on baseline upstream OpenCL too.
//
// Test contract
// -------------
// Build two graphs over the same synthetic Q on the CPU backend:
//   A. Reference: input + identity (Q stays unrotated) → download
//      → host scalar apply_rope → that's our reference vector.
//   B. In-graph: input + cos/sin inputs → `apply_rope_in_graph`
//      → download.
//
// Then assert B == A within F32 tolerance.  Bit-exact is too
// tight (cos/sin precision + add-order rounding) — chatterbox's
// CHATTERBOX_F16_CFM ships at `1e-3` abs; we use `1e-4` here for
// the CPU backend (F32 throughout, only round-order drift).
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

// Scalar reference: matches the in-tree `apply_rope` exactly so
// any divergence between in-graph and reference is a real
// regression, not a "different RoPE formula" mismatch.  Kept
// here as a private copy so the test stays self-contained — the
// production scalar function lives behind a file-static `namespace
// {}` boundary in `supertonic_vector_estimator.cpp` and isn't
// reachable from this TU.
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

// Test 1 — Parity vs. scalar reference on a realistic
// vector-estimator attention shape (q_len = 20, n_heads = 4,
// head_dim = 64).  Tolerance 1e-4 absolute.
void test_rope_parity_vector_estimator_shape() {
    std::fprintf(stderr, "[apply_rope_in_graph: vector-estimator shape]\n");

    const int q_len    = 20;
    const int n_heads  = 4;
    const int head_dim = 64;
    const int half     = head_dim / 2;

    std::mt19937 rng(0xC0DE);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> theta(half);
    for (auto & v : theta) v = std::abs(dist(rng)) * 1000.0f; // RoPE θ is positive, model-typical range

    std::vector<float> x_host((size_t) q_len * n_heads * head_dim);
    for (auto & v : x_host) v = dist(rng);

    // Reference: scalar apply_rope on host copy.
    std::vector<float> ref = x_host;
    scalar_apply_rope(theta.data(), ref, q_len, n_heads, head_dim);

    // Host-precompute cos / sin tables: ne=[half, L].  Element
    // (d, t) at offset t*half + d so the natural row-major upload
    // matches the GGML tensor's ne[0]=half (inner) layout.
    std::vector<float> cos_host((size_t) q_len * half);
    std::vector<float> sin_host((size_t) q_len * half);
    for (int t = 0; t < q_len; ++t) {
        for (int d = 0; d < half; ++d) {
            const float angle = ((float) t / (float) q_len) * theta[d];
            cos_host[(size_t) t * half + d] = std::cos(angle);
            sin_host[(size_t) t * half + d] = std::sin(angle);
        }
    }

    // Build the in-graph rotation graph.
    constexpr int MAX_NODES = 256;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // x has ne=[head_dim, n_heads, L] in GGML order, matching the
    // scalar layout's memory pattern data[t*H*D + h*D + d].  GGML
    // ne[0] is innermost; with the data laid out as in `ref` /
    // `x_host`, element (d, h, t) is at data[t*H*D + h*D + d].
    // Strides: nb=[4, 4*D, 4*D*H].
    ggml_tensor * x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, q_len);
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * cos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, q_len);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_tensor * sin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, q_len);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    ggml_tensor * y = apply_rope_in_graph(ctx, x_in, cos_in, sin_in);
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

    ggml_backend_tensor_set(x_in,   x_host.data(),   0, x_host.size()   * sizeof(float));
    ggml_backend_tensor_set(cos_in, cos_host.data(), 0, cos_host.size() * sizeof(float));
    ggml_backend_tensor_set(sin_in, sin_host.data(), 0, sin_host.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    // Compare.
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
                 "  shape q_len=%d H=%d D=%d  max_abs_err=%.3e  bad=%d / %zu\n",
                 q_len, n_heads, head_dim, max_abs, bad, ref.size());
    CHECK(bad == 0);
}

// Test 2 — Different L (kv_len style: text_len = 32) to confirm
// the helper isn't accidentally hard-coded to a single length.
void test_rope_parity_text_len_shape() {
    std::fprintf(stderr, "[apply_rope_in_graph: kv-len shape]\n");

    const int kv_len   = 32;   // text_len = ~30 in real synth
    const int n_heads  = 4;
    const int head_dim = 64;
    const int half     = head_dim / 2;

    std::mt19937 rng(0xBEEF);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> theta(half);
    for (auto & v : theta) v = std::abs(dist(rng)) * 1000.0f;

    std::vector<float> x_host((size_t) kv_len * n_heads * head_dim);
    for (auto & v : x_host) v = dist(rng);

    std::vector<float> ref = x_host;
    scalar_apply_rope(theta.data(), ref, kv_len, n_heads, head_dim);

    std::vector<float> cos_host((size_t) kv_len * half);
    std::vector<float> sin_host((size_t) kv_len * half);
    for (int t = 0; t < kv_len; ++t) {
        for (int d = 0; d < half; ++d) {
            const float angle = ((float) t / (float) kv_len) * theta[d];
            cos_host[(size_t) t * half + d] = std::cos(angle);
            sin_host[(size_t) t * half + d] = std::sin(angle);
        }
    }

    constexpr int MAX_NODES = 256;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, kv_len);
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * cos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, kv_len);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_tensor * sin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, kv_len);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    ggml_tensor * y = apply_rope_in_graph(ctx, x_in, cos_in, sin_in);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); std::fprintf(stderr, "  SKIP\n"); return; }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(x_in,   x_host.data(),   0, x_host.size()   * sizeof(float));
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
    const float atol = 1e-4f;
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
        const float d = std::fabs(ref[i] - got[i]);
        max_abs = std::max(max_abs, d);
        if (d > atol) ++bad;
    }
    std::fprintf(stderr,
                 "  shape kv_len=%d H=%d D=%d  max_abs_err=%.3e  bad=%d / %zu\n",
                 kv_len, n_heads, head_dim, max_abs, bad, ref.size());
    CHECK(bad == 0);
}

// Test 3 — Identity check: when θ is all zeros (degenerate), the
// rotation is the identity and output must equal input exactly
// (no F32 drift since cos(0)=1, sin(0)=0).  Catches a regression
// where the lower/upper split + concat path accidentally permutes
// the channel axis.
void test_rope_identity_zero_theta() {
    std::fprintf(stderr, "[apply_rope_in_graph: zero-θ identity]\n");

    const int q_len    = 8;
    const int n_heads  = 2;
    const int head_dim = 8;
    const int half     = head_dim / 2;

    std::mt19937 rng(0xDEAD);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x_host((size_t) q_len * n_heads * head_dim);
    for (auto & v : x_host) v = dist(rng);

    // θ = 0 → all angles are 0 → cos=1, sin=0 → output = input.
    std::vector<float> cos_host((size_t) q_len * half, 1.0f);
    std::vector<float> sin_host((size_t) q_len * half, 0.0f);

    constexpr int MAX_NODES = 64;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, q_len);
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * cos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, q_len);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_tensor * sin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, q_len);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    ggml_tensor * y = apply_rope_in_graph(ctx, x_in, cos_in, sin_in);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); std::fprintf(stderr, "  SKIP\n"); return; }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(x_in,   x_host.data(),   0, x_host.size()   * sizeof(float));
    ggml_backend_tensor_set(cos_in, cos_host.data(), 0, cos_host.size() * sizeof(float));
    ggml_backend_tensor_set(sin_in, sin_host.data(), 0, sin_host.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);
    std::vector<float> got((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    int bad = 0;
    for (size_t i = 0; i < x_host.size() && i < got.size(); ++i) {
        if (x_host[i] != got[i]) ++bad;
    }
    std::fprintf(stderr, "  identity bad=%d / %zu\n", bad, x_host.size());
    CHECK(bad == 0);
}

} // namespace

int main() {
    test_rope_parity_vector_estimator_shape();
    test_rope_parity_text_len_shape();
    test_rope_identity_zero_theta();

    std::fprintf(stderr,
                 "test_supertonic_rope_in_graph: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
