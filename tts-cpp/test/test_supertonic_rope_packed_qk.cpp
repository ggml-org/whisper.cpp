// QVAC-18966 — CPU regression fix for `apply_rope_to_packed_qk`
// (also covers the Vulkan / OpenCL synth-path regression on this
// branch — same root cause; rounds 8 / 9's GPU bridges only run
// past round 11 once this helper produces the right shape).
//
// Background
// ----------
// `apply_rope_to_packed_qk` is the layout adapter between the
// natural `ne=[head_dim, n_heads, L]` contract of
// `apply_rope_in_graph` (PR #4) and the **production** call sites'
// Q/K-producing matmul output.  Both PR #16 ("RoPE in-graph
// integration F23") and rounds 8 / 9 (front-block + style GPU
// bridges) plumb the result of this helper through to
// `vector_text_attention_cache::q_tc_in` via either
// `ggml_backend_tensor_copy` (GPU bridge, production) or
// `ggml_backend_tensor_set` from a host vector (legacy bridge,
// trace-mode + non-RoPE GGUFs).
//
// The original test (PR #16, follow-up #5) built Q under a
// `ne=[H*D, L]` "channel-fastest-in-memory" assumption.  That
// matched the helper's INTERNAL layout assumption (view-as-
// `[D, H, L]` with `nb=[elem, D*elem, HD*elem]`), but it
// CONTRADICTED what `dense_matmul_time_ggml` actually produces:
// every Q/K matmul site in the vector estimator hands the helper
// a tensor with `ne=[L, HD]` (axis 0 = L = time-fastest along
// natural strides), so memory layout is **channel-major-flat**
// (`data[t + c*L]`) — the transpose of what the helper expects.
//
// On any backend (CPU, OpenCL, Vulkan), the synth path therefore
// either:
//   - Crashes on the helper's `GGML_ASSERT(HD == n_heads *
//     head_dim)` (the new assertion catches the shape mismatch
//     before the view trick produces garbage), OR
//   - Pre-assertion, would have produced TRANSPOSED bytes and
//     silently fed wrong-layout Q / K into
//     `ggml_flash_attn_ext`.
//
// This test reproduces the real production layout end-to-end on
// the CPU backend (which has no probe-gating and no per-backend
// kernel paths to confuse the picture) and verifies the helper:
//   1. Accepts `ne=[L, HD]` matmul-shaped Q without aborting.
//   2. Returns post-rotation bytes in the **time-major-flat**
//      layout (`out[t*HD + c]`) that:
//        - Matches the scalar `apply_rope(theta, x, L, H, D)`
//          reference (the SOLE source of truth — every host-side
//          comparison in the codebase indexes through `t*H*D +
//          h*D + d` flat).
//        - Can be uploaded byte-for-byte into
//          `q_tc_in = ggml_new_tensor_2d(F32, A, L)` whose
//          natural strides are `nb=[elem, A*elem]` → same flat
//          layout `data[c + t*A]`.
//
// The L=1 trip-wire is kept (catches a future regression where
// the helper silently divides by L or swaps the angle formula).
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
// supertonic_vector_estimator.cpp).  Indexes a single flat buffer
// as `data[t*H*D + h*D + d]` — the time-major-flat layout every
// scalar comparison in the vector estimator uses (and the layout
// `q_tc_in` reads via `ggml_backend_tensor_copy` of
// `ggml_nbytes(q_tc_in)` bytes).
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

// Run `apply_rope_to_packed_qk` on a Q with the production matmul
// shape ne=[L, HD] (channel-major-flat memory `data[t + c*L]`)
// and verify the rotated output matches the scalar reference's
// time-major-flat layout (`out[t*HD + c]`) bit-for-bit on the CPU
// backend.
//
// Production-layout parity test (matches `dense_matmul_time_ggml`
// output on every backend).  Reference is built in time-major-
// flat layout; upload transposes to channel-major-flat so the
// graph input matches matmul's contract bit-for-bit.  Scalar
// apply_rope is applied in-place on the time-major-flat buffer,
// then compared to the helper's downloaded bytes.  Helper must
// produce bytes in time-major-flat layout so:
//   - `ggml_backend_tensor_copy(q_rope, q_tc_in)` blits matching
//     bytes (q_tc_in has the same `ne=[HD, L]` natural layout).
//   - The legacy host-bridge path's `tensor_raw_f32` download
//     yields a `std::vector<float>` indexable as `out[t*HD + c]`.
void test_production_layout(const char * label, int L, int n_heads, int head_dim,
                            unsigned seed) {
    std::fprintf(stderr,
                 "[apply_rope_to_packed_qk production layout: %s]  "
                 "L=%d H=%d D=%d  (matmul ne=[L, HD])\n",
                 label, L, n_heads, head_dim);

    const int HD = n_heads * head_dim;
    const int half = head_dim / 2;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> theta(half);
    for (auto & v : theta) v = std::abs(dist(rng)) * 1000.0f;

    // Reference: time-major-flat buffer `ref[t*HD + c]`.  Random
    // init.  This is the source of truth — `scalar_apply_rope`
    // indexes through `(t*H + h)*D + d` = `t*HD + (h*D + d)`.
    std::vector<float> ref((size_t) L * HD);
    for (auto & v : ref) v = dist(rng);

    // Transpose to channel-major-flat for upload to a tensor with
    // ne=[L, HD] (natural strides nb=[elem, L*elem]).  Element
    // (t, c) in matmul layout lives at flat index `t + c*L` —
    // contiguous in t for fixed c.
    std::vector<float> q_in_buf((size_t) L * HD);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < HD; ++c) {
            q_in_buf[(size_t) t + (size_t) c * L] =
                ref[(size_t) t * HD + c];
        }
    }

    // Scalar reference in-place rotation on the time-major-flat
    // buffer.
    scalar_apply_rope(theta.data(), ref, L, n_heads, head_dim);

    // Cos/sin tables exactly like `make_rope_cos_sin_tables`
    // writes.
    std::vector<float> cos_host, sin_host;
    make_rope_cos_sin_tables(theta.data(), L, half, cos_host, sin_host);

    // Build the graph on the CPU backend.  Max nodes generous
    // for the transpose + cont + view chain inside the helper.
    constexpr int MAX_NODES = 512;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // q input with the production matmul shape.  ne=[L, HD]
    // explicitly DIFFERENT from the pre-fix test's ne=[HD, L].
    ggml_tensor * q_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, HD);
    ggml_set_name(q_in, "q_in"); ggml_set_input(q_in);
    ggml_tensor * cos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_tensor * sin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    ggml_tensor * y = apply_rope_to_packed_qk(ctx, q_in, cos_in, sin_in,
                                              n_heads, head_dim);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    // Output-shape contract.  The helper MUST produce ne=[HD, L]
    // (axis 0 = HD = channels-fastest, axis 1 = L = time-slowest)
    // for `ggml_backend_tensor_copy(y, q_tc_in)` to hit the
    // matching shape in `vector_text_attention_cache::q_tc_in`.
    CHECK((int) y->ne[0] == HD);
    CHECK((int) y->ne[1] == L);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "  SKIP: ggml_backend_cpu_init failed\n");
        ggml_free(ctx);
        return;
    }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(q_in,   q_in_buf.data(), 0, q_in_buf.size() * sizeof(float));
    ggml_backend_tensor_set(cos_in, cos_host.data(), 0, cos_host.size() * sizeof(float));
    ggml_backend_tensor_set(sin_in, sin_host.data(), 0, sin_host.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    // Memory-layout contract: helper's output bytes should equal
    // scalar reference's time-major-flat bytes element-wise.
    CHECK(got.size() == ref.size());

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

// L=1 trip-wire (preserved from the original test).  At L=1 the
// angle is 0/1 * theta = 0, so cos=1, sin=0 and rotation is the
// identity.  Catches a regression where the helper accidentally
// divides by L or swaps the angle formula.  Re-cast under the
// production ne=[L, HD] contract.
void test_production_layout_l1() {
    std::fprintf(stderr,
                 "[apply_rope_to_packed_qk production layout: L=1 degenerate]\n");
    const int L = 1, n_heads = 2, head_dim = 8;
    const int HD = n_heads * head_dim;
    const int half = head_dim / 2;

    std::vector<float> theta(half, 100.0f);

    // Time-major-flat reference; channel-major-flat upload.
    std::vector<float> ref((size_t) L * HD, 1.0f);
    std::vector<float> q_in_buf((size_t) L * HD);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < HD; ++c) {
            q_in_buf[(size_t) t + (size_t) c * L] =
                ref[(size_t) t * HD + c];
        }
    }
    // Identity rotation at L=1.
    scalar_apply_rope(theta.data(), ref, L, n_heads, head_dim);

    std::vector<float> cos_host, sin_host;
    make_rope_cos_sin_tables(theta.data(), L, half, cos_host, sin_host);

    constexpr int MAX_NODES = 128;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * q_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, HD);
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
    ggml_backend_tensor_set(q_in,   q_in_buf.data(), 0, q_in_buf.size() * sizeof(float));
    ggml_backend_tensor_set(cos_in, cos_host.data(), 0, cos_host.size() * sizeof(float));
    ggml_backend_tensor_set(sin_in, sin_host.data(), 0, sin_host.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);
    std::vector<float> got((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    CHECK((int) y->ne[0] == HD);
    CHECK((int) y->ne[1] == L);

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

// Output-shape regression check.  Even if the helper ever gets
// re-plumbed to a different internal pipeline, the public contract
// must remain `ne[0] = n_heads * head_dim`, `ne[1] = L` so the
// downstream `ggml_backend_tensor_copy` blit into
// `vector_text_attention_cache::q_tc_in` stays bit-exact.
void test_output_shape_contract() {
    std::fprintf(stderr,
                 "[apply_rope_to_packed_qk output-shape contract]\n");
    const int L = 20, n_heads = 4, head_dim = 64;
    const int HD = n_heads * head_dim;
    const int half = head_dim / 2;
    const size_t buf_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * q_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, HD);
    ggml_tensor * cos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_tensor * sin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, L);
    ggml_tensor * y = apply_rope_to_packed_qk(ctx, q_in, cos_in, sin_in,
                                              n_heads, head_dim);
    CHECK((int) y->ne[0] == HD);
    CHECK((int) y->ne[1] == L);
    CHECK(ggml_nelements(y) == (int64_t) L * HD);
    ggml_free(ctx);
}

} // namespace

int main() {
    // Vector-estimator hot shapes (q_len, kv_len typical sizes).
    test_production_layout("vector-estimator q", 20, 4, 64, 0xA51C);
    test_production_layout("vector-estimator k", 32, 4, 64, 0xC0FF);
    test_production_layout_l1();
    test_output_shape_contract();

    std::fprintf(stderr,
                 "test_supertonic_rope_packed_qk: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
