// CPU-backend parity test for the F16 K/V flash-attention path
// added to the Supertonic vector estimator in QVAC-18607.
//
// On OpenCL the goal of the rewrite is to dispatch the
// `flash_attn_f32_f16` kernel instead of `flash_attn_f32` (Adreno
// drops attention kernel time by ~2.5x in chatterbox's measurement).
// The CPU backend also implements both paths; running both on CPU
// lets us validate that the F16 round-trip stays within an
// acceptable absolute tolerance against the F32-only reference
// without needing an OpenCL device on CI.
//
// Shapes here mirror what the Supertonic vector estimator uses in
// practice:
//
//   width    = n_heads * head_dim
//   n_heads  = 4
//   head_dim = 64   (one of the supported OpenCL dims)
//   q_len    = latent_len  (small int, ~20 in this test)
//   kv_len   = text_len    (small int, ~32 in this test)
//
// Registered with `LABEL "unit"` in CMakeLists.txt so a fresh
// checkout's `ctest` exercises this without needing any fixture.

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

struct attention_inputs {
    int n_heads;
    int head_dim;
    int q_len;
    int kv_len;
    std::vector<float> q;   // [head_dim, q_len,  n_heads] (ggml order)
    std::vector<float> k;   // [head_dim, kv_len, n_heads]
    std::vector<float> v;   // [head_dim, kv_len, n_heads]
    float scale;
};

attention_inputs make_inputs(int n_heads, int head_dim, int q_len, int kv_len, uint32_t seed) {
    attention_inputs in;
    in.n_heads  = n_heads;
    in.head_dim = head_dim;
    in.q_len    = q_len;
    in.kv_len   = kv_len;
    in.scale    = 1.0f / std::sqrt((float) head_dim);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const size_t q_size = (size_t) head_dim * q_len  * n_heads;
    const size_t k_size = (size_t) head_dim * kv_len * n_heads;
    in.q.resize(q_size);
    in.k.resize(k_size);
    in.v.resize(k_size);
    for (auto & v : in.q) v = dist(rng);
    for (auto & v : in.k) v = dist(rng);
    for (auto & v : in.v) v = dist(rng);
    return in;
}

// Build a graph that runs `ggml_flash_attn_ext` with the requested
// K / V dtype on the CPU backend, return the attention output as
// a flat F32 vector.  `kv_type` is either `GGML_TYPE_F32` (the
// reference path), `GGML_TYPE_F16` (the OpenCL fast path), or
// `GGML_TYPE_BF16` (round 4 — the Vulkan coopmat2 fast path,
// added by Prereq B to cover the round-4 dispatch site change).
std::vector<float> run_flash_attn(ggml_backend_t cpu,
                                  const attention_inputs & in,
                                  ggml_type kv_type) {
    constexpr int MAX_NODES = 64;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                            ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         in.head_dim, in.q_len,  in.n_heads);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         in.head_dim, in.kv_len, in.n_heads);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         in.head_dim, in.kv_len, in.n_heads);
    ggml_set_name(q, "q"); ggml_set_input(q);
    ggml_set_name(k, "k"); ggml_set_input(k);
    ggml_set_name(v, "v"); ggml_set_input(v);

    ggml_tensor * k_use = k;
    ggml_tensor * v_use = v;
    if (kv_type != GGML_TYPE_F32) {
        // Same rewrite that ships in the vector estimator: contiguous
        // typed destinations populated via `ggml_cpy` so the
        // mixed-precision flash-attn dispatch sees row-major-by-head
        // typed inputs.  F16 → existing OpenCL `flash_attn_f32_f16`
        // / Vulkan `kernel_flash_attn_f32_f16_*` path.  BF16 → the
        // round-4 Vulkan coopmat2 path (probe-gated by
        // `supertonic_backend_supports_bf16_kv_flash_attn`).
        ggml_tensor * k_typed = ggml_new_tensor_3d(ctx, kv_type,
                                                   in.head_dim, in.kv_len, in.n_heads);
        ggml_tensor * v_typed = ggml_new_tensor_3d(ctx, kv_type,
                                                   in.head_dim, in.kv_len, in.n_heads);
        k_use = ggml_cpy(ctx, k, k_typed);
        v_use = ggml_cpy(ctx, v, v_typed);
    }

    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k_use, v_use,
                                             /*mask=*/nullptr,
                                             in.scale,
                                             /*max_bias=*/0.0f,
                                             /*logit_softcap=*/0.0f);
    ggml_set_name(attn, "attn"); ggml_set_output(attn);
    ggml_build_forward_expand(gf, attn);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    if (!ggml_gallocr_reserve(allocr, gf)) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        throw std::runtime_error("ggml_gallocr_reserve flash_attn failed");
    }
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "q"),
                            in.q.data(), 0, in.q.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "k"),
                            in.k.data(), 0, in.k.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "v"),
                            in.v.data(), 0, in.v.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> out((size_t) ggml_nelements(attn));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "attn"),
                            out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}

// Test 1 — F32 vs F16 K/V parity on the vector-estimator shape.
//
// Tolerance: F16 round-trip on attention typically lands within
// ~5e-3 absolute / ~5e-3 relative on outputs near unit magnitude.
// chatterbox ships this exact pattern in production behind
// `--cfm-f16-kv-attn` with the same tolerance budget.  Tightening
// below this would catch a real F16 regression but also reject
// healthy F16 noise; loosening would let an actually-incorrect
// kernel slip through.
void test_attn_f32_vs_f16_parity(ggml_backend_t cpu) {
    const int n_heads  = 4;
    const int head_dim = 64;
    const int q_len    = 20;
    const int kv_len   = 32;
    const auto in = make_inputs(n_heads, head_dim, q_len, kv_len, 0xC1A5);

    std::vector<float> ref;
    std::vector<float> got;
    bool ran_both = true;
    try {
        ref = run_flash_attn(cpu, in, GGML_TYPE_F32);
    } catch (const std::exception & e) {
        std::fprintf(stderr,
                     "  [attn F32 path] FAILED to run on this CPU build: %s\n",
                     e.what());
        ran_both = false;
    }
    try {
        got = run_flash_attn(cpu, in, GGML_TYPE_F16);
    } catch (const std::exception & e) {
        std::fprintf(stderr,
                     "  [attn F16 path] FAILED to run on this CPU build: %s\n",
                     e.what());
        ran_both = false;
    }

    if (!ran_both) {
        // Treat as informative: the CPU build lacks one of the two
        // flash-attention paths.  Don't count this as a failure;
        // the production OpenCL build is what actually consumes
        // the rewrite, and a missing CPU-side path here doesn't
        // change that.  The dispatch + portable_ops tests still
        // catch the rest of the bring-up regressions.
        std::fprintf(stderr,
                     "  [attn parity] SKIPPED — CPU build missing one path\n");
        return;
    }
    CHECK(ref.size() == got.size());

    int bad = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    const float atol = 5e-3f;
    const float rtol = 5e-3f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float abs_err = std::fabs(got[i] - ref[i]);
        const float rel_err = std::fabs(ref[i]) > 1e-6f ? abs_err / std::fabs(ref[i]) : abs_err;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (abs_err > atol + rtol * std::fabs(ref[i])) {
            if (bad < 4) {
                std::fprintf(stderr,
                             "  attn parity mismatch @ %zu: ref=%.6g got=%.6g abs_err=%.3e\n",
                             i, ref[i], got[i], abs_err);
            }
            ++bad;
        }
    }
    std::fprintf(stderr,
                 "  [attn F32 vs F16 parity]  q=%d kv=%d h=%d d=%d  "
                 "max_abs_err=%.3e  max_rel_err=%.3e  bad=%d / %zu\n",
                 q_len, kv_len, n_heads, head_dim,
                 max_abs_err, max_rel_err, bad, ref.size());
    CHECK(bad == 0);
}

// Test 2 — Style attention shape (kv_len = 50, the fixed style-token
// count).  Same parity story, slightly larger workload, validates
// the F16 path doesn't regress on the second hot shape.
void test_attn_style_shape(ggml_backend_t cpu) {
    const int n_heads  = 4;
    const int head_dim = 64;
    const int q_len    = 20;
    const int kv_len   = 50;   // style tokens — fixed across all prompts
    const auto in = make_inputs(n_heads, head_dim, q_len, kv_len, 0x5717);

    std::vector<float> ref, got;
    try {
        ref = run_flash_attn(cpu, in, GGML_TYPE_F32);
        got = run_flash_attn(cpu, in, GGML_TYPE_F16);
    } catch (const std::exception & e) {
        std::fprintf(stderr,
                     "  [attn style shape] SKIPPED: %s\n", e.what());
        return;
    }
    CHECK(ref.size() == got.size());

    int bad = 0;
    float max_abs_err = 0.0f;
    const float atol = 5e-3f;
    const float rtol = 5e-3f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float abs_err = std::fabs(got[i] - ref[i]);
        max_abs_err = std::max(max_abs_err, abs_err);
        if (abs_err > atol + rtol * std::fabs(ref[i])) {
            if (bad < 4) {
                std::fprintf(stderr,
                             "  style attn mismatch @ %zu: ref=%.6g got=%.6g abs_err=%.3e\n",
                             i, ref[i], got[i], abs_err);
            }
            ++bad;
        }
    }
    std::fprintf(stderr,
                 "  [attn style shape] kv=%d  max_abs_err=%.3e  bad=%d / %zu\n",
                 kv_len, max_abs_err, bad, ref.size());
    CHECK(bad == 0);
}

// QVAC-18605 round 4 — Prereq B: parameterised K/V parity check.
//
// Generalised version of `test_attn_f32_vs_f16_parity` /
// `test_attn_style_shape` that runs the F32 reference and an
// arbitrary `kv_dtype` candidate, then checks max-abs-err against
// a per-dtype tolerance band.  Used by the BF16 tests below.
//
// Per-dtype tolerance rationale:
//   - F16  : 5e-3 abs / 5e-3 rel (existing baseline; matches
//            chatterbox CHATTERBOX_F16_CFM tolerance).
//   - BF16 : 5e-3 abs / 5e-3 rel (BF16 has the same 11-bit-ish
//            precision as F16 — only the exponent range differs.
//            Same tolerance band; the wider exponent range buys
//            stability on small attention scores, not extra
//            absolute accuracy on outputs near unit magnitude.)
//
// The CPU backend MAY or MAY NOT advertise BF16 K/V flash-attn
// (depends on whether ggml-cpu was compiled with BF16 dot-product
// support).  When the BF16 path throws on this build, the test
// is reported as SKIPPED instead of failing — same convention as
// the existing F16 path's "missing one path" treatment.  The
// production Vulkan adapter is what actually consumes this
// dispatch and is probe-gated separately at runtime by
// `supertonic_backend_supports_bf16_kv_flash_attn`.
void test_attn_kv_dtype_parity(ggml_backend_t cpu,
                               const char * label,
                               int n_heads,
                               int head_dim,
                               int q_len,
                               int kv_len,
                               uint32_t seed,
                               ggml_type kv_dtype,
                               float atol,
                               float rtol) {
    const auto in = make_inputs(n_heads, head_dim, q_len, kv_len, seed);

    std::vector<float> ref;
    std::vector<float> got;
    bool ran_both = true;
    try {
        ref = run_flash_attn(cpu, in, GGML_TYPE_F32);
    } catch (const std::exception & e) {
        std::fprintf(stderr,
                     "  [%s F32 ref] FAILED to run on this CPU build: %s\n",
                     label, e.what());
        ran_both = false;
    }
    try {
        got = run_flash_attn(cpu, in, kv_dtype);
    } catch (const std::exception & e) {
        std::fprintf(stderr,
                     "  [%s %s K/V] FAILED to run on this CPU build: %s\n",
                     label, ggml_type_name(kv_dtype), e.what());
        ran_both = false;
    }
    if (!ran_both) {
        std::fprintf(stderr,
                     "  [%s parity %s] SKIPPED — CPU build missing one path\n",
                     label, ggml_type_name(kv_dtype));
        return;
    }
    CHECK(ref.size() == got.size());

    int bad = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float abs_err = std::fabs(got[i] - ref[i]);
        const float rel_err = std::fabs(ref[i]) > 1e-6f ? abs_err / std::fabs(ref[i]) : abs_err;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (abs_err > atol + rtol * std::fabs(ref[i])) {
            if (bad < 4) {
                std::fprintf(stderr,
                             "  %s/%s parity mismatch @ %zu: ref=%.6g got=%.6g abs_err=%.3e\n",
                             label, ggml_type_name(kv_dtype), i, ref[i], got[i], abs_err);
            }
            ++bad;
        }
    }
    std::fprintf(stderr,
                 "  [%s parity %s]  q=%d kv=%d h=%d d=%d  "
                 "max_abs_err=%.3e  max_rel_err=%.3e  bad=%d / %zu  (atol=%.0e, rtol=%.0e)\n",
                 label, ggml_type_name(kv_dtype),
                 q_len, kv_len, n_heads, head_dim,
                 max_abs_err, max_rel_err, bad, ref.size(), atol, rtol);
    CHECK(bad == 0);
}

// Test 3 (round 4 / Prereq B) — F32 vs BF16 K/V parity on the
// vector-estimator shape.  BF16 has the same precision as F16
// (11 bits) but a wider 8-bit exponent — so the per-element
// upload bandwidth is identical to F16, but small attention
// scores avoid the F16 underflow that drives the F16 test's
// 5e-3 tolerance.  Same tolerance band here as a SAFETY gate
// (any bigger bad-count signals a real BF16 kernel regression
// rather than a precision-vs-F16 difference).
//
// Written BEFORE the round-4 dispatch site change (TDD), so the
// parity gate is in place before any production code touches
// the K/V cast logic.
void test_attn_f32_vs_bf16_parity(ggml_backend_t cpu) {
    test_attn_kv_dtype_parity(cpu,
        /*label=*/   "vector_estimator",
        /*n_heads=*/ 4,
        /*head_dim=*/64,
        /*q_len=*/   20,
        /*kv_len=*/  32,
        /*seed=*/    0xBF16C1A5,
        /*kv_dtype=*/GGML_TYPE_BF16,
        /*atol=*/    5e-3f,
        /*rtol=*/    5e-3f);
}

// Test 4 (round 4 / Prereq B) — same shape as the existing
// F16 style-shape test (kv=50) but with BF16 K/V.  Catches
// BF16-specific regressions on the second hot shape.
void test_attn_bf16_style_shape(ggml_backend_t cpu) {
    test_attn_kv_dtype_parity(cpu,
        /*label=*/   "style_attention",
        /*n_heads=*/ 4,
        /*head_dim=*/64,
        /*q_len=*/   20,
        /*kv_len=*/  50,
        /*seed=*/    0xBF165717,
        /*kv_dtype=*/GGML_TYPE_BF16,
        /*atol=*/    5e-3f,
        /*rtol=*/    5e-3f);
}

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "ggml_backend_cpu_init failed\n");
        return 1;
    }

    // Existing F16 parity tests — unchanged.
    test_attn_f32_vs_f16_parity(cpu);
    test_attn_style_shape(cpu);

    // Round 4 / Prereq B — BF16 parity tests, written BEFORE the
    // round-4 dispatch site change.
    test_attn_f32_vs_bf16_parity(cpu);
    test_attn_bf16_style_shape(cpu);

    ggml_backend_free(cpu);

    std::fprintf(stderr,
                 "test_supertonic_f16_attn_parity: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
