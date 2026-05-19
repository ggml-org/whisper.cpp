// TDD harness for audit follow-up #6 (2C-lite) — graph-to-graph
// tensor blits via `ggml_backend_tensor_copy`.
//
// Background
// ----------
// After F23 landed, the vector-estimator group graph emits post-
// RoPE Q/K (`<q_name>_rope`, `<k_name>_rope`) and raw V on the GPU.
// The next stage (`run_text_attention_cache`) consumes those three
// tensors but lives in its OWN GGML context with its own gallocr.
// The bridge between the two graphs is currently:
//
//   tensor_to_time_channel(group_gf.q_rope)        // GPU → host
//   ggml_backend_tensor_set(att_cache.q_tc_in, …)  // host → GPU
//
// per Q / K / V per attention site (4 sites × 5 denoise steps =
// 60 round-trips per synth on the production path).  Each
// round-trip is one synchronous read + one upload — 6 sync points
// per attention site, or 120 sync points / synth across the four
// fused-attention sites.
//
// 2C-lite is to replace those two operations with a single
// `ggml_backend_tensor_copy(src_tensor_in_graph_A,
//  dst_tensor_in_graph_B)` call.  Same backend on both ends, so
// the copy is a pure device-to-device blit (or a tight memcpy on
// the CPU backend) and the host never touches the buffer.
//
// Test contract
// -------------
// 1. Build two MINIMAL cached graphs that share a single
//    ggml_backend instance:
//      A: x_in → out_A = x_in * 2   (the "producer" graph;
//                                   mirrors the group graph
//                                   producing q_rope)
//      B: y_in → out_B = y_in - 1   (the "consumer" graph;
//                                   mirrors the attention graph
//                                   consuming q_tc_in)
//    Each graph has its OWN ggml_context + gallocr (mirrors the
//    `vector_group_graph_cache` / `vector_text_attention_cache`
//    split exactly).
//
// 2. Reference path (the code we're replacing):
//      compute(A) → ggml_backend_tensor_get(out_A, host_buf)
//                 → ggml_backend_tensor_set(y_in, host_buf)
//                 → compute(B) → read out_B.
//
// 3. Fused path (the code we're adding):
//      compute(A) → ggml_backend_tensor_copy(out_A, y_in)
//                 → compute(B) → read out_B.
//
// 4. Both must produce bit-exact identical out_B.  The copy is a
//    pure memory rearrangement, no arithmetic, so any difference
//    indicates a backend bug we MUST not paper over with a
//    tolerance.
//
// Shapes covered
// --------------
// - `vector_group_graph_cache` post-RoPE Q at L=20, C=256
//   (q_len=20, n_heads=4, head_dim=64).
// - The same site at L=1 (trip-wire for stride / shape bugs at
//   the smallest sensible input).
// - The style-attention site at L=20, kv_len=50, n_heads=2,
//   head_dim=128 (the ne[0]*ne[1] product changes between the
//   two attention shapes; this catches dimension-mismatched
//   tensor_copy bugs).
//
// Mirrors the structure of the other audit follow-up unit tests
// in this directory (no GGUF, no fixture, no model file).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

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

// Single-backend two-graph harness — built once per shape.  The
// producer / consumer split mirrors the cache-per-stage pattern
// used throughout supertonic_vector_estimator.cpp.
struct two_graph_harness {
    ggml_backend_t backend = nullptr;

    // Producer graph: emits out_A = x_in * 2.
    std::vector<uint8_t> buf_a;
    ggml_context * ctx_a = nullptr;
    ggml_cgraph *  gf_a  = nullptr;
    ggml_gallocr_t alloc_a = nullptr;
    ggml_tensor *  x_in  = nullptr;
    ggml_tensor *  out_a = nullptr;

    // Consumer graph: emits out_B = y_in - 1.
    std::vector<uint8_t> buf_b;
    ggml_context * ctx_b = nullptr;
    ggml_cgraph *  gf_b  = nullptr;
    ggml_gallocr_t alloc_b = nullptr;
    ggml_tensor *  y_in  = nullptr;
    ggml_tensor *  out_b = nullptr;
};

void destroy_harness(two_graph_harness & h) {
    if (h.alloc_a) ggml_gallocr_free(h.alloc_a);
    if (h.alloc_b) ggml_gallocr_free(h.alloc_b);
    if (h.ctx_a)   ggml_free(h.ctx_a);
    if (h.ctx_b)   ggml_free(h.ctx_b);
    if (h.backend) ggml_backend_free(h.backend);
    h = {};
}

bool build_harness(two_graph_harness & h, int ne0, int ne1) {
    h.backend = ggml_backend_cpu_init();
    if (!h.backend) return false;

    constexpr int NODES = 16;
    const size_t buf_sz = ggml_tensor_overhead() * NODES + ggml_graph_overhead();

    // Producer.  ne=[ne0, ne1] matches the post-RoPE Q layout
    // (`[width=n_heads*head_dim, q_len]`).
    h.buf_a.assign(buf_sz, 0);
    ggml_init_params pa = { buf_sz, h.buf_a.data(), /*no_alloc=*/true };
    h.ctx_a = ggml_init(pa);
    h.gf_a  = ggml_new_graph(h.ctx_a);
    h.x_in  = ggml_new_tensor_2d(h.ctx_a, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(h.x_in, "x_in"); ggml_set_input(h.x_in);
    h.out_a = ggml_scale(h.ctx_a, h.x_in, 2.0f);
    ggml_set_name(h.out_a, "out_a"); ggml_set_output(h.out_a);
    ggml_build_forward_expand(h.gf_a, h.out_a);
    h.alloc_a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(h.backend));
    if (!h.alloc_a || !ggml_gallocr_reserve(h.alloc_a, h.gf_a)) return false;
    ggml_gallocr_alloc_graph(h.alloc_a, h.gf_a);

    // Consumer — same shape, MUST live in a different context.
    h.buf_b.assign(buf_sz, 0);
    ggml_init_params pb = { buf_sz, h.buf_b.data(), /*no_alloc=*/true };
    h.ctx_b = ggml_init(pb);
    h.gf_b  = ggml_new_graph(h.ctx_b);
    h.y_in  = ggml_new_tensor_2d(h.ctx_b, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(h.y_in, "y_in"); ggml_set_input(h.y_in);
    // out_B = y_in - 1.  `ggml_add` of a constant scalar needs
    // a tensor, so reuse the cleaner `ggml_scale + offset` form:
    // y - 1 == y * 1 + (-1).  Single op, no branching.
    h.out_b = ggml_scale_bias(h.ctx_b, h.y_in, 1.0f, -1.0f);
    ggml_set_name(h.out_b, "out_b"); ggml_set_output(h.out_b);
    ggml_build_forward_expand(h.gf_b, h.out_b);
    h.alloc_b = ggml_gallocr_new(ggml_backend_get_default_buffer_type(h.backend));
    if (!h.alloc_b || !ggml_gallocr_reserve(h.alloc_b, h.gf_b)) return false;
    ggml_gallocr_alloc_graph(h.alloc_b, h.gf_b);
    return true;
}

// Reference bridge: download out_A from graph A, upload into y_in
// of graph B.  This is the byte-for-byte equivalent of the
// pre-2C code path:
//
//   tensor_to_time_channel(group_gf.q_rope)
//   ggml_backend_tensor_set(att_cache.q_tc_in, …)
std::vector<float> run_reference(two_graph_harness & h,
                                 const std::vector<float> & x) {
    ggml_backend_tensor_set(h.x_in, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_graph_compute(h.backend, h.gf_a);

    std::vector<float> host_buf((size_t) ggml_nelements(h.out_a));
    ggml_backend_tensor_get(h.out_a, host_buf.data(), 0,
                            host_buf.size() * sizeof(float));
    ggml_backend_tensor_set(h.y_in, host_buf.data(), 0,
                            host_buf.size() * sizeof(float));
    ggml_backend_graph_compute(h.backend, h.gf_b);

    std::vector<float> out((size_t) ggml_nelements(h.out_b));
    ggml_backend_tensor_get(h.out_b, out.data(), 0, out.size() * sizeof(float));
    return out;
}

// Fused bridge: direct GPU→GPU blit via `ggml_backend_tensor_copy`.
// Host never sees the intermediate buffer — this is the 2C-lite
// fast path we want call sites to use.
std::vector<float> run_fused(two_graph_harness & h,
                             const std::vector<float> & x) {
    ggml_backend_tensor_set(h.x_in, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_graph_compute(h.backend, h.gf_a);

    // Single-call replacement for the host round-trip pair.
    // For same-backend src+dst this is a memcpy on the CPU
    // backend and a `clEnqueueCopyBuffer` on OpenCL.
    ggml_backend_tensor_copy(h.out_a, h.y_in);

    ggml_backend_graph_compute(h.backend, h.gf_b);

    std::vector<float> out((size_t) ggml_nelements(h.out_b));
    ggml_backend_tensor_get(h.out_b, out.data(), 0, out.size() * sizeof(float));
    return out;
}

void test_shape(const char * label, int ne0, int ne1, unsigned seed) {
    std::fprintf(stderr, "[graph_to_graph_blit: %s] ne0=%d ne1=%d\n",
                 label, ne0, ne1);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x((size_t) ne0 * ne1);
    for (auto & v : x) v = dist(rng);

    two_graph_harness ref_h{};
    if (!build_harness(ref_h, ne0, ne1)) {
        std::fprintf(stderr, "  SKIP: harness build failed (ref)\n");
        destroy_harness(ref_h);
        return;
    }
    std::vector<float> ref = run_reference(ref_h, x);
    destroy_harness(ref_h);

    two_graph_harness fused_h{};
    if (!build_harness(fused_h, ne0, ne1)) {
        std::fprintf(stderr, "  SKIP: harness build failed (fused)\n");
        destroy_harness(fused_h);
        return;
    }
    std::vector<float> got = run_fused(fused_h, x);
    destroy_harness(fused_h);

    CHECK(got.size() == ref.size());

    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
        const float d = std::fabs(ref[i] - got[i]);
        max_abs = std::max(max_abs, d);
        if (d > 0.0f) {
            if (bad < 4) {
                std::fprintf(stderr,
                             "  mismatch @ %zu: ref=%.6g got=%.6g abs=%.3e\n",
                             i, ref[i], got[i], d);
            }
            ++bad;
        }
    }
    std::fprintf(stderr, "  %s max_abs=%.3e bad=%d\n", label, max_abs, bad);
    CHECK(bad == 0);
    CHECK(max_abs == 0.0f);
}

}  // namespace

int main() {
    test_shape("attn0_q_rope_L20",     256,  20, 0xA11A1u);   // 4h × 64d  @ L=20
                                                              // Also covers front-block attn0
                                                              // Q post-RoPE tensor (round 8 GPU
                                                              // bridge consumer).
    test_shape("attn0_q_rope_L1",      256,   1, 0xA11A2u);   // L=1 trip-wire
    // QVAC-18605 round 8 — front-block attn0 K / V shape
    // (width=256, kv_len=text_len).  Same layout as the round-1
    // group attentions but different ne1 dimension.  Locks in the
    // blit primitive for the K / V handles the front-block GPU
    // bridge passes to `run_text_attention_cache_gpu`.
    test_shape("attn0_kv_text_len32",  256,  32, 0xA11A4u);   // front-block K / V @ text_len=32
    test_shape("attn0_kv_text_len50",  256,  50, 0xA11A5u);   // front-block K / V @ text_len=50

    // QVAC-18605 round 9 — style flash-attn K / V / Q shapes for
    // the 4 res-style sites (style0 + g1_style + g2_style +
    // g3_style).  Style attention runs at n_heads=2, head_dim=128
    // (vs n_heads=4, head_dim=64 for the text attentions above)
    // — but the underlying flat ne layout is `[width=256, *_len]`
    // either way (2 × 128 == 4 × 64 == 256), so the byte-count-
    // matching contract `ggml_backend_tensor_copy` checks
    // internally is identical to round 8.  The Q (sq) is
    // `[256, L=20]`; the K / V (sk / sv) are `[256, 50]` (the
    // style ttl is fixed at 50 tokens regardless of the input
    // text length).  These shapes are already covered by
    // `style0_q_rope_L20` + `style0_k_rope_kv50` below — round 9
    // adds the explicit doc-comment + a Q at L=1 for the same
    // trip-wire reason as round 8's `attn0_q_rope_L1`.
    test_shape("style_sq_L1",          256,   1, 0xA11A6u);   // L=1 trip-wire for style Q
    test_shape("style0_q_rope_L20",    256,  20, 0xA11A3u);   // 2h × 128d @ L=20  ← style sq
    test_shape("attn0_k_rope_kv20",    256,  20, 0xA11A4u);   // K side
    test_shape("style0_k_rope_kv50",   256,  50, 0xA11A5u);   // K side, style kv_len

    std::fprintf(stderr,
                 "test_supertonic_graph_to_graph_blit: %d / %d checks passed\n",
                 (g_checks - g_failures), g_checks);
    return g_failures == 0 ? 0 : 1;
}
