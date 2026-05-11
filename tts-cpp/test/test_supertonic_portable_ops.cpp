// CPU-backend parity tests for the portable op rewrites landed in the
// Supertonic OpenCL bring-up.  Each test builds two GGML graphs with
// the same input data on the CPU backend:
//
//   - Reference graph: the original op (e.g. `ggml_leaky_relu`).
//   - Portable graph : the GPU-friendly rewrite that
//     `supertonic_internal.h` exposes (e.g.
//     `leaky_relu_portable_ggml` with `supertonic_use_cpu_custom_ops()`
//     forced to `false` via the dispatch scope).
//
// Then it asserts the outputs match within F32 tolerance.  Math
// equivalence is the contract; running both lowerings on the CPU
// backend lets us validate that contract without needing an
// OpenCL device on CI.
//
// Registered with `LABEL "unit"` in CMakeLists.txt so a fresh
// checkout's `ctest` exercises this without needing any fixture.

#include "supertonic_internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
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

// Pick a relative-+-absolute tolerance that covers F32 rounding for the
// portable decomposition.  The rewrite computes
// `(1-α)·relu(x) + α·x` as three separate rounding steps where the
// original `ggml_leaky_relu` is one branch + one multiply, so we
// expect ~3 ULPs of slack on the largest |x|.  Keeping the same
// shape as `close_enough()` in `test_metal_ops.cpp` for consistency.
bool close_enough(float a, float b, float atol = 1e-6f, float rtol = 1e-5f) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// Build a 2-D F32 input tensor [W, H], allocate it on `backend`, run
// the graph constructed by `build_op`, return the contents of its
// last output tensor.  The `build_op` callback receives the graph
// context + the input tensor and returns the output tensor it wants
// observed.
std::vector<float> run_one_op(
    ggml_backend_t backend,
    const std::vector<float> & input,
    int W, int H,
    ggml_tensor * (*build_op)(ggml_context *, ggml_tensor *, float),
    float alpha) {

    constexpr int MAX_NODES = 64;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                            ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, H);
    ggml_set_name(x, "x"); ggml_set_input(x);

    ggml_tensor * y = build_op(ctx, x, alpha);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"),
                            input.data(), 0, input.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "y"),
                            out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}

ggml_tensor * build_reference(ggml_context * ctx, ggml_tensor * x, float alpha) {
    // Direct fused builtin — the lowering used on the CPU backend.
    return ggml_leaky_relu(ctx, x, alpha, /*inplace=*/false);
}

ggml_tensor * build_portable(ggml_context * ctx, ggml_tensor * x, float alpha) {
    // Same lowering the dispatch helper picks when
    // `supertonic_use_cpu_custom_ops()` is false; we call into the
    // shared inline definition so a future change to the rewrite
    // would automatically be exercised here too.  The dispatch
    // scope around the call below forces the GPU branch even
    // though we're physically running on the CPU backend.
    return leaky_relu_portable_ggml(ctx, x, alpha);
}

// Test 1 — Sign-pattern coverage.
//
// LeakyReLU has different paths for `x >= 0` and `x < 0`; the
// portable decomposition collapses them into a single algebraic
// form.  Feed an input that exercises both halves and the boundary.
void test_leaky_relu_signs(ggml_backend_t cpu) {
    const int W = 64, H = 4;
    std::vector<float> input((size_t) W * H);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (auto & v : input) v = dist(rng);
    // Plant the boundary explicitly.
    input[0] = 0.0f;
    input[1] = -0.0f;
    input[2] = 1e-10f;
    input[3] = -1e-10f;

    // Forcing the GPU lowering needs a "GPU-looking" model with a
    // dispatch scope around the portable graph build.  The reference
    // build runs without any scope so it picks the default
    // `supertonic_use_cpu_custom_ops() == true` path, which routes
    // through the CPU fused builtin.
    supertonic_model gpu_model;
    gpu_model.backend_is_cpu = false;
    gpu_model.use_f16_attn   = false;

    for (float alpha : { 0.0f, 0.01f, 0.05f, 0.1f, 0.5f, 0.99f, 1.0f }) {
        auto ref = run_one_op(cpu, input, W, H, build_reference, alpha);
        std::vector<float> got;
        {
            supertonic_op_dispatch_scope scope(gpu_model);
            got = run_one_op(cpu, input, W, H, build_portable, alpha);
        }

        int bad = 0;
        float worst = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!close_enough(got[i], ref[i])) {
                if (bad < 4) {
                    std::fprintf(stderr,
                                 "  alpha=%.3f i=%zu  ref=%.6g  portable=%.6g\n",
                                 alpha, i, ref[i], got[i]);
                }
                ++bad;
            }
            worst = std::max(worst, std::fabs(got[i] - ref[i]));
        }
        CHECK(bad == 0);
        std::fprintf(stderr,
                     "  [leaky_relu signs alpha=%.3f] max_abs_err=%.3e %s\n",
                     alpha, worst, bad == 0 ? "PASS" : "FAIL");
    }
}

// Test 2 — Dispatch scope actually routes through the portable path.
//
// Belt-and-braces: even if `close_enough()` accidentally permitted
// any input → any output, the runtime should still observe the same
// number of graph nodes in the portable build (1 RELU + 2 SCALE
// + 1 ADD = 4 nodes) vs the reference build (1 LEAKY_RELU node).
// Inspecting node count is fragile but cheap; it guards against
// `leaky_relu_portable_ggml` regressing back to a `ggml_leaky_relu`
// passthrough on GPU.
void test_dispatch_actually_routes(ggml_backend_t cpu) {
    const int W = 8, H = 1;
    std::vector<float> input((size_t) W * H);
    for (int i = 0; i < W; ++i) input[i] = (float) i - 3.5f;

    auto count_nodes = [&](ggml_tensor * (*build)(ggml_context *, ggml_tensor *, float)) {
        constexpr int MAX_NODES = 64;
        const size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                                ggml_graph_overhead();
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, H);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * y = build(ctx, x, 0.1f);
        ggml_set_name(y, "y"); ggml_set_output(y);
        ggml_build_forward_expand(gf, y);

        int n = ggml_graph_n_nodes(gf);
        ggml_free(ctx);
        (void) cpu;
        return n;
    };

    supertonic_model cpu_model;
    cpu_model.backend_is_cpu = true;
    supertonic_model gpu_model;
    gpu_model.backend_is_cpu = false;

    int n_ref = 0;
    int n_portable_cpu = 0;
    int n_portable_gpu = 0;
    {
        n_ref = count_nodes(build_reference);
    }
    {
        supertonic_op_dispatch_scope scope(cpu_model);
        n_portable_cpu = count_nodes(build_portable);
    }
    {
        supertonic_op_dispatch_scope scope(gpu_model);
        n_portable_gpu = count_nodes(build_portable);
    }

    std::fprintf(stderr,
                 "  [dispatch routing] ref=%d  portable(cpu)=%d  portable(gpu)=%d\n",
                 n_ref, n_portable_cpu, n_portable_gpu);

    // Reference is the fused builtin: exactly one op.
    CHECK(n_ref == 1);
    // Portable on the CPU dispatch picks the same fused builtin too,
    // so the node count must match the reference.
    CHECK(n_portable_cpu == n_ref);
    // Portable on the GPU dispatch decomposes into RELU + SCALE +
    // SCALE + ADD = 4 ops.  Asserting equality here would couple
    // the test to today's exact lowering; assert "strictly more
    // than 1" instead so a future fused-but-still-portable
    // rewrite stays green.
    CHECK(n_portable_gpu > n_ref);
}

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "ggml_backend_cpu_init failed\n");
        return 1;
    }

    test_leaky_relu_signs(cpu);
    test_dispatch_actually_routes(cpu);

    ggml_backend_free(cpu);

    std::fprintf(stderr,
                 "test_supertonic_portable_ops: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
