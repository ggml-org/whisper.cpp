// Unit tests for the shared dual-path dispatch helpers in
// src/sched_dispatch.{h,cpp} (per-op GPU->CPU fallback for the T3 eval
// paths): graph_fully_supported walk, TTS_CPP_FORCE_SCHED escape
// hatch, sched_fallback lifecycle (ensure/alloc/compute/free) and both
// branches of the pre-allocated-op abort guard (see main for the
// --sched-abort-repro mode that pins the GGML_ABORT side).
//
// No GGUF / model file required — every test builds a tiny graph on the
// CPU backend, so the scheduler is exercised as the single-backend
// pass-through case (exactly what a CPU-primary model would create).
//
// Registered with `LABEL "unit"` in CMakeLists.txt so a fresh
// checkout's `ctest` exercises this without needing any fixture.

#include "backend_selection.h"
#include "sched_dispatch.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace tts_cpp::detail;

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

// Tiny 2-node graph: out = (a + b) scaled by 2.  `a` is a runtime input
// (ggml_set_input, data uploaded after alloc — same contract as the T3
// eval sites); `b` is a second input so the graph has more than one leaf.
struct toy_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_tensor *  a   = nullptr;
    ggml_tensor *  b   = nullptr;
    ggml_tensor *  out = nullptr;

    explicit toy_graph(int n) {
        ggml_init_params p = {
            ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
            nullptr,
            /*no_alloc=*/true,
        };
        ctx = ggml_init(p);
        gf  = ggml_new_graph(ctx);
        a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(a, "a"); ggml_set_input(a);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(b, "b"); ggml_set_input(b);
        out = ggml_scale(ctx, ggml_add(ctx, a, b), 2.0f);
        ggml_set_name(out, "out"); ggml_set_output(out);
        ggml_build_forward_expand(gf, out);
    }
    ~toy_graph() { if (ctx) ggml_free(ctx); }
};

void test_walk_and_force_env(ggml_backend_t cpu) {
    toy_graph g(8);
    // CPU supports every op of the toy graph.
    CHECK(graph_fully_supported(cpu, g.gf));
    // Null-safety.
    CHECK(!graph_fully_supported(nullptr, g.gf));
    CHECK(!graph_fully_supported(cpu, nullptr));

    // The force env var is read per call (NOT latched in a static), so a
    // test can toggle it mid-process.
    unsetenv("TTS_CPP_FORCE_SCHED");
    CHECK(!sched_force_enabled());
    setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    CHECK(sched_force_enabled());
    setenv("TTS_CPP_FORCE_SCHED", "0", 1);
    CHECK(!sched_force_enabled());
    unsetenv("TTS_CPP_FORCE_SCHED");

    // No node of the toy graph is bound to a pre-allocated buffer, so the
    // abort guard must pass.
    CHECK(!graph_has_unsupported_preallocated_op(cpu, g.gf));
}

void test_sched_lifecycle_and_compute(ggml_backend_t cpu) {
    const int n = 8;

    sched_fallback fb;
    CHECK(sched_fallback_ensure(fb, cpu, /*graph_size=*/2048, {}));
    // CPU primary => single-backend pass-through: no extra CPU backend.
    CHECK(fb.sched != nullptr);
    CHECK(fb.cpu_backend == nullptr);
    // Second ensure is a no-op success.
    CHECK(sched_fallback_ensure(fb, cpu, 2048, {}));

    // alloc -> upload inputs -> compute -> download (the T3 site ordering).
    toy_graph g(n);
    CHECK(sched_fallback_alloc(fb, g.gf));

    std::vector<float> va(n), vb(n);
    for (int i = 0; i < n; ++i) { va[i] = (float) i; vb[i] = 10.0f * (float) i; }
    ggml_backend_tensor_set(g.a, va.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(g.b, vb.data(), 0, n * sizeof(float));

    CHECK(sched_fallback_compute(fb, cpu, g.gf, /*n_threads=*/2) == GGML_STATUS_SUCCESS);

    std::vector<float> vo(n);
    ggml_backend_tensor_get(g.out, vo.data(), 0, n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        CHECK(vo[i] == 2.0f * (va[i] + vb[i]));
    }

    // direct_compute twin on an identical fresh graph gives the same result.
    {
        toy_graph g2(n);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        CHECK(ggml_gallocr_alloc_graph(allocr, g2.gf));
        ggml_backend_tensor_set(g2.a, va.data(), 0, n * sizeof(float));
        ggml_backend_tensor_set(g2.b, vb.data(), 0, n * sizeof(float));
        CHECK(direct_compute(cpu, g2.gf, /*n_threads=*/2) == GGML_STATUS_SUCCESS);
        std::vector<float> vo2(n);
        ggml_backend_tensor_get(g2.out, vo2.data(), 0, n * sizeof(float));
        CHECK(std::memcmp(vo.data(), vo2.data(), n * sizeof(float)) == 0);
        ggml_gallocr_free(allocr);
    }

    // free is idempotent and re-arms: a later ensure rebuilds.
    sched_fallback_free(fb);
    CHECK(fb.sched == nullptr && fb.cpu_backend == nullptr);
    sched_fallback_free(fb);
    CHECK(sched_fallback_ensure(fb, cpu, 2048, {}));
    CHECK(fb.sched != nullptr);
    sched_fallback_free(fb);

    // Degenerate-argument behaviour: no sched -> alloc/compute fail cleanly.
    sched_fallback fb2;
    toy_graph g3(n);
    CHECK(!sched_fallback_alloc(fb2, g3.gf));
    CHECK(sched_fallback_compute(fb2, cpu, g3.gf, 2) == GGML_STATUS_FAILED);
    CHECK(!sched_fallback_ensure(fb2, nullptr, 2048, {}));
}

// KV-slab shape: set_rows writes rows of a pre-allocated slab `a`; the
// node is a view of `a`, so its dst buffer is fixed before the sched runs.
struct setrows_graph {
    ggml_context *        lctx     = nullptr;  // slab (leaf) ctx
    ggml_context *        gctx     = nullptr;  // graph ctx
    ggml_backend_buffer_t slab_buf = nullptr;
    ggml_cgraph *         gf       = nullptr;
    ggml_tensor *         a        = nullptr;
    ggml_tensor *         b        = nullptr;
    ggml_tensor *         c        = nullptr;
    ggml_tensor *         out      = nullptr;

    // alloc_on == nullptr builds the graph without allocating the slab
    // (enough for supports_op probes; the guard needs a real buffer).
    setrows_graph(ggml_type dst_type, ggml_backend_t alloc_on) {
        ggml_init_params lp = { ggml_tensor_overhead() * 2, nullptr, /*no_alloc=*/true };
        lctx = ggml_init(lp);
        a = ggml_new_tensor_2d(lctx, dst_type, 256, 4);
        ggml_set_name(a, "slab");
        if (alloc_on) {
            slab_buf = ggml_backend_alloc_ctx_tensors(lctx, alloc_on);
        }

        ggml_init_params gp = {
            ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
            nullptr,
            /*no_alloc=*/true,
        };
        gctx = ggml_init(gp);
        gf = ggml_new_graph(gctx);
        b = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 256, 2);
        ggml_set_name(b, "rows"); ggml_set_input(b);
        c = ggml_new_tensor_1d(gctx, GGML_TYPE_I64, 2);
        ggml_set_name(c, "idx"); ggml_set_input(c);
        out = ggml_set_rows(gctx, a, b, c);
        ggml_set_name(out, "set_rows_out");
        ggml_build_forward_expand(gf, out);
    }
    ~setrows_graph() {
        if (gctx) ggml_free(gctx);
        if (slab_buf) ggml_backend_buffer_free(slab_buf);
        if (lctx) ggml_free(lctx);
    }
};

// First dst type whose set_rows the CPU backend rejects (IQ quants lack
// from_float today); GGML_TYPE_COUNT if a future ggml supports them all.
ggml_type find_cpu_rejected_setrows_type(ggml_backend_t cpu) {
    const ggml_type candidates[] = {
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,   GGML_TYPE_IQ1_S,  GGML_TYPE_IQ1_M,
    };
    for (ggml_type t : candidates) {
        setrows_graph g(t, /*alloc_on=*/nullptr);
        if (!ggml_backend_supports_op(cpu, g.out)) return t;
    }
    return GGML_TYPE_COUNT;
}

void test_abort_guard_both_branches(ggml_backend_t cpu) {
    // True branch: pre-allocated dst nobody can run -> guard must trip
    // (callers then fail gracefully instead of entering the sched).
    const ggml_type bad = find_cpu_rejected_setrows_type(cpu);
    if (bad == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "SKIP: no cpu-unsupported set_rows dst type\n");
    } else {
        setrows_graph g(bad, cpu);
        CHECK(g.slab_buf != nullptr);
        CHECK(graph_has_unsupported_preallocated_op(cpu, g.gf));
        CHECK(!graph_fully_supported(cpu, g.gf));
    }

    // False-branch twin: same shape with an F32 dst passes the guard and
    // runs the full graceful path through the sched.
    setrows_graph g(GGML_TYPE_F32, cpu);
    CHECK(g.slab_buf != nullptr);
    CHECK(!graph_has_unsupported_preallocated_op(cpu, g.gf));

    std::vector<float> zeros(256 * 4, 0.0f);
    ggml_backend_tensor_set(g.a, zeros.data(), 0, zeros.size() * sizeof(float));

    sched_fallback fb;
    CHECK(sched_fallback_ensure(fb, cpu, 2048, {}));
    CHECK(sched_fallback_alloc(fb, g.gf));

    std::vector<float> rows(256 * 2);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (float) i;
    const int64_t idx[2] = { 1, 3 };
    ggml_backend_tensor_set(g.b, rows.data(), 0, rows.size() * sizeof(float));
    ggml_backend_tensor_set(g.c, idx, 0, sizeof(idx));
    CHECK(sched_fallback_compute(fb, cpu, g.gf, 2) == GGML_STATUS_SUCCESS);

    std::vector<float> slab(256 * 4);
    ggml_backend_tensor_get(g.a, slab.data(), 0, slab.size() * sizeof(float));
    CHECK(std::memcmp(slab.data() + 256 * 1, rows.data(),       256 * sizeof(float)) == 0);
    CHECK(std::memcmp(slab.data() + 256 * 3, rows.data() + 256, 256 * sizeof(float)) == 0);
    CHECK(slab[0] == 0.0f && slab[256 * 2] == 0.0f);

    sched_fallback_free(fb);
}

// Child half of the repro: bypass the guard, feed the rejected graph to the
// sched — ggml is expected to GGML_ABORT here.
int run_sched_abort_repro_child(ggml_backend_t cpu, ggml_type bad) {
    setrows_graph g(bad, cpu);
    sched_fallback fb;
    if (!g.slab_buf || !sched_fallback_ensure(fb, cpu, 2048, {})) {
        std::fprintf(stderr, "ERROR: repro setup failed\n");
        return 1;
    }
    sched_fallback_alloc(fb, g.gf);  // expected to abort inside ggml
    std::fprintf(stderr, "ERROR: sched alloc did not abort on the rejected graph\n");
    return 1;
}

// Parent half: re-exec self so the child's SIGABRT doesn't fail ctest; the
// abort message reaches ctest's output for PASS_REGULAR_EXPRESSION.
int run_sched_abort_repro(ggml_backend_t cpu, const char * self) {
    const ggml_type bad = find_cpu_rejected_setrows_type(cpu);
    if (bad == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "SKIP: no cpu-unsupported set_rows dst type\n");
        return 0;
    }
    const std::string cmd = std::string("\"") + self + "\" --sched-abort-repro-child "
                          + std::to_string((int) bad);
    if (std::system(cmd.c_str()) == 0) {
        std::fprintf(stderr, "ERROR: abort-repro child exited cleanly\n");
        return 1;
    }
    std::fprintf(stderr, "abort-repro: child aborted as expected\n");
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_t cpu = tts_cpp::detail::init_cpu_backend();
    if (!cpu) {
        std::fprintf(stderr, "test-t3-sched-dispatch: no CPU backend registered\n");
        return 2;
    }

    if (argc > 1 && std::strcmp(argv[1], "--sched-abort-repro") == 0) {
        return run_sched_abort_repro(cpu, argv[0]);
    }
    if (argc > 2 && std::strcmp(argv[1], "--sched-abort-repro-child") == 0) {
        return run_sched_abort_repro_child(cpu, (ggml_type) std::atoi(argv[2]));
    }

    test_walk_and_force_env(cpu);
    test_sched_lifecycle_and_compute(cpu);
    test_abort_guard_both_branches(cpu);

    ggml_backend_free(cpu);

    std::fprintf(stderr, "test-t3-sched-dispatch: %d checks, %d failures\n",
                 g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
