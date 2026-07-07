// Real per-op migration test for src/sched_dispatch.{h,cpp} (PR #81 review
// follow-up, finding #1): every other sched test in this repo runs with a
// CPU primary, where the scheduler degenerates to a single-backend
// pass-through split and no op ever migrates.  This test uses the BLAS
// backend (device type ACCEL, shipped in ggml-speech) as the PRIMARY: its
// supports_op accepts only MUL_MAT / OUT_PROD (f32, contiguous, dims >= 32)
// and view-class ops, so every ADD / SCALE / CPY node in the graph is
// genuinely rejected and must migrate to the CPU-last backend — a real
// multi-split schedule with cross-backend traffic, exercised through the
// production helper end to end:
//
//   graph_fully_supported() rejects  ->  sched_fallback_ensure() builds the
//   [BLAS, CPU] sched  ->  alloc/upload/compute across the split  ->
//   per-step ggml_cpy writes into a PRE-ALLOCATED host tensor (the KV-slab
//   analogue) accumulate correct state across sched_reset cycles.
//
// A mock supports_op-rejecting backend is not an option here: the system
// ggml-speech package installs no ggml-backend-impl.h, so backend/device
// interface structs are not visible to this repo.  BLAS is strictly better
// anyway — a real backend with a real rejection predicate.
//
// Numerics: the sched pipeline is asserted DETERMINISTIC (two runs
// bit-identical) and compared to a pure-CPU direct reference within a tight
// relative tolerance.  It is NOT bit-compared to the CPU reference by
// design: the MUL_MAT lands on BLAS (Accelerate sgemm), whose f32 rounding
// legitimately differs from ggml-cpu's matmul in the last ulp.  The
// bit-exact direct-vs-sched bar is enforced by test_t3_sched_equivalence,
// where both paths share one backend.
//
// Self-skipping: if no ACCEL device is registered in this build (e.g. a
// GGML_BLAS=OFF configuration), the test prints SKIP and exits 0.

#include "backend_selection.h"
#include "sched_dispatch.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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

constexpr int N       = 64;  // >= BLAS's min_batch=32 on every mul_mat dim
constexpr int N_STEPS = 3;

// Per-step graph, T3-eval shaped: transient ctx, inputs marked, a mul_mat
// (BLAS-supported) feeding add + scale (BLAS-rejected -> CPU), and a cpy of
// the result into a view of the PRE-ALLOCATED `state` slab at column range
// [step*N, (step+1)*N).
struct step_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_tensor *  inp = nullptr;
    ggml_tensor *  out = nullptr;

    step_graph(ggml_tensor * W, ggml_tensor * bias, ggml_tensor * state, int step) {
        ggml_init_params p = {
            ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
            nullptr,
            /*no_alloc=*/true,
        };
        ctx = ggml_init(p);
        gf  = ggml_new_graph(ctx);
        inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
        ggml_set_name(inp, "inp"); ggml_set_input(inp);

        ggml_tensor * mm = ggml_mul_mat(ctx, W, inp);            // BLAS-eligible
        ggml_tensor * ad = ggml_add(ctx, mm, bias);              // BLAS-rejected
        out = ggml_scale(ctx, ad, 0.5f);                         // BLAS-rejected
        ggml_set_name(out, "out"); ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        // KV-append analogue: contiguous write into the persistent slab.
        ggml_tensor * dst = ggml_view_2d(ctx, state, N, N,
                                         state->nb[1],
                                         (size_t) step * N * state->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
    }
    ~step_graph() { if (ctx) ggml_free(ctx); }
};

void fill_inputs(std::vector<float> & W, std::vector<float> & bias,
                 std::vector<float> & inp) {
    W.resize((size_t) N * N);
    bias.resize((size_t) N * N);
    inp.resize((size_t) N * N);
    for (size_t i = 0; i < W.size(); ++i) {
        W[i]    = 0.01f * (float) ((i * 31 % 97) - 48);
        bias[i] = 0.10f * (float) ((i * 17 % 13) - 6);
        inp[i]  = 0.02f * (float) ((i * 7 % 59) - 29);
    }
}

// One full N_STEPS pipeline run.  `use_helper_sched=true` drives the
// production sched_dispatch path with `primary`; false computes a direct
// single-backend CPU reference.  Returns the final state slab contents.
bool run_pipeline(ggml_backend_t primary, bool use_helper_sched,
                  std::vector<float> & state_out) {
    // Persistent tensors (weights + state slab), pre-allocated in a HOST
    // buffer on the primary backend — like T3's buffer_w / buffer_kv.
    ggml_init_params sp = { ggml_tensor_overhead() * 8, nullptr, /*no_alloc=*/true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * W     = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, N, N);
    ggml_tensor * bias  = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, N, N);
    ggml_tensor * state = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, N, N * N_STEPS);
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, primary);
    if (!buf_static) { ggml_free(ctx_static); return false; }

    std::vector<float> vW, vb, vi;
    fill_inputs(vW, vb, vi);
    ggml_backend_tensor_set(W,    vW.data(), 0, vW.size() * sizeof(float));
    ggml_backend_tensor_set(bias, vb.data(), 0, vb.size() * sizeof(float));
    {
        std::vector<float> zero((size_t) N * N * N_STEPS, 0.0f);
        ggml_backend_tensor_set(state, zero.data(), 0, zero.size() * sizeof(float));
    }

    sched_fallback fb;
    ggml_gallocr_t allocr = nullptr;
    bool ok = true;

    for (int step = 0; step < N_STEPS && ok; ++step) {
        step_graph g(W, bias, state, step);

        if (use_helper_sched) {
            // The migration premise: the primary genuinely rejects part of
            // the graph, so the production walk must route it to the sched.
            CHECK(!graph_fully_supported(primary, g.gf));
            // The pre-allocated writes (state slab, host buffer) are
            // CPU-runnable — the abort guard must NOT fire.
            CHECK(!graph_has_unsupported_preallocated_op(primary, g.gf));
            ok = sched_fallback_ensure(fb, primary, /*graph_size=*/2048, {buf_static}) &&
                 sched_fallback_alloc(fb, g.gf);
            if (ok) {
                // Real 2-backend sched: primary is ACCEL, so a CPU fallback
                // backend must have been created.
                CHECK(fb.cpu_backend != nullptr);
            }
        } else {
            if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(primary));
            ok = ggml_gallocr_alloc_graph(allocr, g.gf);
        }
        if (!ok) break;

        ggml_backend_tensor_set(g.inp, vi.data(), 0, vi.size() * sizeof(float));

        const ggml_status st = use_helper_sched
            ? sched_fallback_compute(fb, primary, g.gf, /*n_threads=*/2)
            : direct_compute(primary, g.gf, /*n_threads=*/2);
        ok = (st == GGML_STATUS_SUCCESS);
    }

    if (ok) {
        state_out.resize((size_t) N * N * N_STEPS);
        ggml_backend_tensor_get(state, state_out.data(), 0,
                                state_out.size() * sizeof(float));
    }

    if (allocr) ggml_gallocr_free(allocr);
    sched_fallback_free(fb);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    return ok;
}

bool allclose(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff  = std::fabs(a[i] - b[i]);
        const float scale = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
        if (diff > 1e-6f + 1e-5f * scale) {
            std::fprintf(stderr, "  mismatch @%zu: %.9g vs %.9g\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    // Find the ACCEL (BLAS) device.
    ggml_backend_dev_t accel_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            accel_dev = dev;
            break;
        }
    }
    if (!accel_dev) {
        std::fprintf(stderr, "test-t3-sched-migration: SKIP — no ACCEL (BLAS) device "
                             "in this ggml build; per-op migration needs a primary "
                             "that rejects ops\n");
        return 0;
    }

    ggml_backend_t blas = ggml_backend_dev_init(accel_dev, nullptr);
    if (!blas) {
        std::fprintf(stderr, "test-t3-sched-migration: SKIP — ACCEL device failed to init\n");
        return 0;
    }
    std::fprintf(stderr, "test-t3-sched-migration: primary = %s\n",
                 ggml_backend_name(blas));

    ggml_backend_t cpu = init_cpu_backend();
    if (!cpu) {
        std::fprintf(stderr, "test-t3-sched-migration: no CPU backend registered\n");
        ggml_backend_free(blas);
        return 2;
    }

    // Migrated pipeline (BLAS primary through the helper), twice — the
    // scheduler path must be deterministic (bit-identical).
    std::vector<float> mig1, mig2, ref;
    CHECK(run_pipeline(blas, /*use_helper_sched=*/true, mig1));
    CHECK(run_pipeline(blas, /*use_helper_sched=*/true, mig2));
    CHECK(mig1.size() == mig2.size() &&
          std::memcmp(mig1.data(), mig2.data(), mig1.size() * sizeof(float)) == 0);

    // Pure-CPU direct reference — tight tolerance (see header comment for
    // why not bit-exact: the MUL_MAT runs on BLAS by design).
    CHECK(run_pipeline(cpu, /*use_helper_sched=*/false, ref));
    CHECK(allclose(mig1, ref));

    // Non-degenerate sanity: state slab was actually written each step.
    bool any_nonzero = false;
    for (float x : mig1) if (x != 0.0f) { any_nonzero = true; break; }
    CHECK(any_nonzero);

    ggml_backend_free(cpu);
    ggml_backend_free(blas);

    std::fprintf(stderr, "test-t3-sched-migration: %d checks, %d failures\n",
                 g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
