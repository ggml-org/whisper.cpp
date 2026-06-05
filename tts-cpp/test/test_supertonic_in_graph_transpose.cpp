// TDD harness for audit follow-up #6 (F12) — in-graph transpose
// helper for the vector / text / duration estimator graph caches.
//
// Background
// ----------
// Every `run_*_cache` site in supertonic_vector_estimator.cpp
// (and a few mirror sites in the text encoder / duration / vocoder
// caches) carries a host-side `pack_time_channel_for_ggml(x_tc,
// L, C)` loop that transposes CPU-native time-major data
// (`x_tc[t*C + c]`) into the channel-major layout GGML stores
// `ne=[L, C]` tensors in (`buf[c*L + t]`).  Audit finding F12 —
// these add up to "dozens of small CPU transposes" per synth +
// they serialise the host-side dispatch on the GPU path.
//
// `transpose_time_channel_ggml(ctx, x_tc_input)` is the audit's
// recommended fix.  The cache exposes the raw upload buffer as a
// GGML tensor with `ne=[C, L]` (channels on axis 0, time on
// axis 1) so the caller can upload `x_tc` BYTE-FOR-BYTE without
// any CPU transpose, then the graph immediately does
// `ggml_cont(ctx, ggml_transpose(ctx, x_tc_in))` to recover the
// `[L, C]` layout the rest of the graph builders expect.  Net
// effect: one CPU O(L*C) loop replaced by one device-side
// `ggml_cont` of the same `L*C` bytes — on a GPU this is far
// faster (and runs in parallel with subsequent kernels under the
// graph scheduler).
//
// Test contract
// -------------
// Build a small synthetic time-channel buffer `x_tc` and verify
// the in-graph transpose helper produces the exact same memory
// layout the existing `pack_time_channel_for_ggml` host loop
// produces, then read back the resulting `[L, C]` tensor and
// confirm element-by-element parity (bit-exact — transpose+cont
// is a pure memory rearrangement, no arithmetic).
//
// Two parity shapes:
//   1. `vector_group_graph_cache`'s hot path: L=20, C=256.
//   2. `vector_tail_graph_cache`'s noise input: L=20, Cin=24.
//
// Registered with `LABEL "unit"` — no GGUF required.  Mirrors the
// pattern used by `test_supertonic_rope_packed_qk.cpp`.

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

// Reference CPU pack — bit-identical to
// `pack_time_channel_for_ggml` in supertonic_vector_estimator.cpp.
// Converts CPU-native time-major `x[t*C + c]` to GGML's
// column-major (channel-slow) storage `out[c*L + t]`.  This is
// the buffer the existing call sites upload directly into a
// `ne=[L, C]` cache input.
std::vector<float> pack_time_channel_reference(const std::vector<float> & x,
                                               int L, int C) {
    std::vector<float> out((size_t) L * C);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            out[(size_t) c * L + t] = x[(size_t) t * C + c];
        }
    }
    return out;
}

void test_transpose_shape(const char * label, int L, int C, unsigned seed) {
    std::fprintf(stderr, "[transpose_time_channel: %s] L=%d C=%d\n",
                 label, L, C);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x_tc((size_t) L * C);
    for (auto & v : x_tc) v = dist(rng);

    std::vector<float> ref = pack_time_channel_reference(x_tc, L, C);

    constexpr int MAX_NODES = 64;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // `x_tc_in`: ne=[C, L].  Caller uploads CPU-native `x_tc` as-
    // is (no CPU pack).  GGML interprets memory byte `i` (= 4-byte
    // float index `i`) as element (c=i%C, l=i/C), which matches
    // x_tc's `x[t*C + c]` layout (the element x_tc[t*C+c] lands at
    // GGML logical (c=c, l=t)).
    ggml_tensor * x_tc_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, L);
    ggml_set_name(x_tc_in, "x_tc_in"); ggml_set_input(x_tc_in);

    // The fix: transpose to ne=[L, C] then cont to materialise the
    // natural-stride layout.  After the cont, memory at index
    // `l + c*L` carries the value at original logical (l, c), which
    // is element x_tc[l*C + c] — the exact same byte sequence as
    // `pack_time_channel_reference(x_tc, L, C)` writes.
    ggml_tensor * x_lc = transpose_time_channel_ggml(ctx, x_tc_in);
    ggml_set_name(x_lc, "x_lc"); ggml_set_output(x_lc);
    ggml_build_forward_expand(gf, x_lc);

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

    // Upload `x_tc` directly — no CPU pack, no memcpy, no copy.
    ggml_backend_tensor_set(x_tc_in, x_tc.data(), 0, x_tc.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got((size_t) ggml_nelements(x_lc));
    ggml_backend_tensor_get(x_lc, got.data(), 0, got.size() * sizeof(float));

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    CHECK(got.size() == ref.size());

    // Bit-exact comparison — transpose+cont is a pure memory
    // rearrangement, no arithmetic.  Any mismatch indicates a
    // stride / shape bug, not a floating-point rounding issue.
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
    std::fprintf(stderr, "  max_abs_err=%.3e  bad=%d / %zu\n",
                 max_abs, bad, ref.size());
    CHECK(bad == 0);
}

// Trip-wire: ne[1] = 1 (single-time-step) is the degenerate shape
// that the front-block / duration caches build for inference-time
// `latent_len = 1` smoke harnesses.  Catches strides that assume
// `L > 1`.
void test_transpose_l1() {
    std::fprintf(stderr, "[transpose_time_channel: L=1 degenerate]\n");
    const int L = 1, C = 8;
    std::vector<float> x_tc((size_t) L * C);
    for (int i = 0; i < (int) x_tc.size(); ++i) x_tc[i] = (float) i + 0.5f;

    std::vector<float> ref = pack_time_channel_reference(x_tc, L, C);

    constexpr int MAX_NODES = 32;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x_tc_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, L);
    ggml_set_input(x_tc_in);
    ggml_tensor * x_lc = transpose_time_channel_ggml(ctx, x_tc_in);
    ggml_set_output(x_lc);
    ggml_build_forward_expand(gf, x_lc);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); std::fprintf(stderr, "  SKIP\n"); return; }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(x_tc_in, x_tc.data(), 0, x_tc.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got((size_t) ggml_nelements(x_lc));
    ggml_backend_tensor_get(x_lc, got.data(), 0, got.size() * sizeof(float));

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    int bad = 0;
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
        if (ref[i] != got[i]) ++bad;
    }
    std::fprintf(stderr, "  L=1 bad=%d\n", bad);
    CHECK(bad == 0);

    // Output ne shape must be [L, C] — the layout downstream
    // graph builders expect.
    CHECK(x_lc->ne[0] == L);
    CHECK(x_lc->ne[1] == C);
}

} // namespace

int main() {
    // Vector-estimator group-graph hot shape (audit example).
    test_transpose_shape("group_graph L=20 C=256", 20, 256, 0xC0DE);
    // Tail-graph noise shape (Cin=24 < L typical).
    test_transpose_shape("tail noise   L=20 C=24",  20,  24, 0xBEEF);
    // Vocoder-realistic shape (T0=420, C=512) — exercises the
    // wider channel buffer to catch a stride wraparound bug.
    test_transpose_shape("vocoder      T0=420 C=64", 420, 64, 0x73B1);
    test_transpose_l1();

    std::fprintf(stderr,
                 "test_supertonic_in_graph_transpose: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
