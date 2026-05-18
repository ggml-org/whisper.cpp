// Standalone validation for the Metal kernels we added/fixed in ggml:
//   - GGML_OP_DIAG_MASK_INF
//   - GGML_OP_PAD with non-zero front-pad offsets (lp0..lp3)
//   - GGML_OP_MUL_MAT + GGML_OP_ADD(bias) [+ GGML_OP_UNARY(GELU_ERF)]
//     fusion in kernel_mul_mm (PROGRESS §3.27, §3.28)
//
// Runs each op twice (once on CPU, once on Metal) with the same input and
// compares element-by-element.  Exits non-zero on mismatch.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool close_enough(float a, float b, float atol = 1e-5f) {
    if (std::isinf(a) && std::isinf(b) && ((a < 0) == (b < 0))) return true;
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::fabs(a - b) <= atol + 1e-5f * std::fabs(b);
}

static std::vector<float> run_graph(ggml_backend_t backend,
                                    ggml_cgraph * gf,
                                    ggml_tensor * out) {
    auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(allocr);
    return res;
}

static int test_diag_mask_inf(ggml_backend_t cpu, ggml_backend_t gpu) {
    fprintf(stderr, "[diag_mask_inf] ");
    const int N = 37, H = 5;  // 5 heads
    const int n_past = 4;

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<float> src(N * N * H);
    for (auto & x : src) x = dist(rng);

    auto run_one = [&](ggml_backend_t backend) {
        static size_t buf_size = 4 * 1024 * 1024;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N, N, H);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * y = ggml_diag_mask_inf(ctx, x, n_past);
        ggml_set_name(y, "y"); ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"),
                                src.data(), 0, src.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        std::vector<float> res(N * N * H);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "y"),
                                res.data(), 0, res.size() * sizeof(float));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return res;
    };

    auto ref = run_one(cpu);
    auto got = run_one(gpu);

    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!close_enough(got[i], ref[i])) {
            if (bad < 8) {
                fprintf(stderr, "\n  mismatch @ %zu: cpu=%.6g gpu=%.6g", i, ref[i], got[i]);
            }
            ++bad;
        }
    }
    if (bad == 0) {
        fprintf(stderr, "OK (N=%d, H=%d, n_past=%d)\n", N, H, n_past);
        return 0;
    }
    fprintf(stderr, "\n[diag_mask_inf] FAIL: %d / %zu mismatched\n", bad, ref.size());
    return 1;
}

static int test_pad_ext(ggml_backend_t cpu, ggml_backend_t gpu) {
    fprintf(stderr, "[pad_ext]        ");
    const int L = 17, C = 4;
    const int lp0 = 3, rp0 = 2, lp1 = 1, rp1 = 0;

    std::mt19937 rng(2);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<float> src(L * C);
    for (auto & x : src) x = dist(rng);

    auto run_one = [&](ggml_backend_t backend) {
        static size_t buf_size = 4 * 1024 * 1024;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * y = ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, 0, 0, 0, 0);
        ggml_set_name(y, "y"); ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"),
                                src.data(), 0, src.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        const int L_out = L + lp0 + rp0;
        const int C_out = C + lp1 + rp1;
        std::vector<float> res(L_out * C_out);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "y"),
                                res.data(), 0, res.size() * sizeof(float));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return res;
    };

    auto ref = run_one(cpu);
    auto got = run_one(gpu);

    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!close_enough(got[i], ref[i])) {
            if (bad < 8) {
                fprintf(stderr, "\n  mismatch @ %zu: cpu=%.6g gpu=%.6g", i, ref[i], got[i]);
            }
            ++bad;
        }
    }
    if (bad == 0) {
        fprintf(stderr, "OK (L=%d, C=%d, lp0=%d, rp0=%d, lp1=%d, rp1=%d)\n",
                L, C, lp0, rp0, lp1, rp1);
        return 0;
    }
    fprintf(stderr, "\n[pad_ext] FAIL: %d / %zu mismatched\n", bad, ref.size());
    return 1;
}

static int test_conv_transpose_1d(ggml_backend_t cpu, ggml_backend_t gpu,
                                  int IL, int IC, int OC, int K, int s0,
                                  const char * label) {
    fprintf(stderr, "[conv_transp_1d %s] ", label);

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> kdata(K * OC * IC);
    std::vector<float> xdata(IL * IC);
    for (auto & v : kdata) v = dist(rng);
    for (auto & v : xdata) v = dist(rng);

    auto run_one = [&](ggml_backend_t backend) {
        static size_t buf_size = 64 * 1024 * 1024;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, OC, IC);
        ggml_set_name(k, "k"); ggml_set_input(k);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IL, IC);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * y = ggml_conv_transpose_1d(ctx, k, x, s0, 0, 1);
        ggml_set_name(y, "y"); ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "k"), kdata.data(), 0, kdata.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), xdata.data(), 0, xdata.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_tensor * out = ggml_graph_get_tensor(gf, "y");
        std::vector<float> res(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return res;
    };

    auto ref = run_one(cpu);
    auto got = run_one(gpu);

    int bad = 0;
    float max_err = 0.f, max_rel = 0.f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        const float r = d / std::max(std::fabs(ref[i]), 1e-6f);
        if (d > max_err) max_err = d;
        if (r > max_rel) max_rel = r;
        if (d > 1e-3f) {
            if (bad < 5) {
                fprintf(stderr, "\n  mismatch @ %zu: cpu=%.6g gpu=%.6g", i, ref[i], got[i]);
            }
            ++bad;
        }
    }
    if (bad == 0) {
        fprintf(stderr, "OK (IL=%d IC=%d OC=%d K=%d s0=%d, max_abs=%.1e max_rel=%.1e)\n",
                IL, IC, OC, K, s0, max_err, max_rel);
        return 0;
    }
    fprintf(stderr, "\n[conv_transp_1d] FAIL: %d / %zu mismatched (max_err=%.3e)\n",
            bad, ref.size(), max_err);
    return 1;
}

// Test the MUL_MAT + ADD(bias) [+ GELU_ERF] fusion in kernel_mul_mm.
// Builds the 2- or 3-op subgraph on both CPU and GPU backends, dispatches,
// and compares output element-wise.  On the GPU side, ggml-metal's fusion
// system (FC_MUL_MM + 2 / +3 / +4, PROGRESS §3.27 / §3.28) collapses these
// into a single `kernel_mul_mm_..._bias=1_res=X_gelu=Y` dispatch; the CPU
// path is always the unfused triple.  Any numerical drift beyond atol
// indicates either a kernel bug or a shape-handling mismatch.
//
// Uses Q4_0 weights to match the chatterbox CFM hot path — that's the
// shape the fused kernel is specifically targeting.  K must be %32 for
// Q4_0 blocks; N / T are unconstrained.
//
// fuse_mode: 0 = MUL_MAT + ADD(bias), 1 = MUL_MAT + ADD(bias) + GELU_ERF.
static int test_mul_mm_fused(ggml_backend_t cpu, ggml_backend_t gpu,
                             int K, int N, int T, int B, int fuse_mode,
                             const char * label) {
    fprintf(stderr, "[mul_mm_fused %s] ", label);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    // W: (K, N) in ggml layout → src0 of shape [K, N] = ggml ne=[K, N].
    //    Quantized to Q4_0 — block of 32 in the K (innermost) dim.
    // X: (K, T, B) → src1 of shape [K, T, B] in ggml ne=[K, T, B].
    // Output: (N, T, B).
    // bias: (N,) — broadcast over T, B.
    std::vector<float> W_f32(K * N);
    std::vector<float> X_f32(K * T * B);
    std::vector<float> bias_f32(N);
    for (auto & v : W_f32)    v = dist(rng);
    for (auto & v : X_f32)    v = dist(rng);
    for (auto & v : bias_f32) v = dist(rng);

    auto run_one = [&](ggml_backend_t backend) {
        static size_t buf_size = 32 * 1024 * 1024;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * W    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
        ggml_tensor * X    = (B == 1) ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T)
                                      : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, T, B);
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_set_name(W,    "W");    ggml_set_input(W);
        ggml_set_name(X,    "X");    ggml_set_input(X);
        ggml_set_name(bias, "bias"); ggml_set_input(bias);

        ggml_tensor * mm   = ggml_mul_mat(ctx, W, X);
        ggml_tensor * mmb  = ggml_add(ctx, mm, bias);
        ggml_tensor * out  = (fuse_mode == 1) ? ggml_gelu_erf(ctx, mmb) : mmb;
        ggml_set_name(out, "out"); ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        // Quantise W to Q4_0 into the backend buffer.
        {
            std::vector<uint8_t> qbuf(ggml_nbytes(ggml_graph_get_tensor(gf, "W")));
            ggml_quantize_chunk(GGML_TYPE_Q4_0, W_f32.data(), qbuf.data(), 0, N, K, nullptr);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "W"),
                                    qbuf.data(), 0, qbuf.size());
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "X"),    X_f32.data(),    0, X_f32.size()    * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bias"), bias_f32.data(), 0, bias_f32.size() * sizeof(float));

        ggml_backend_graph_compute(backend, gf);
        ggml_tensor * out_t = ggml_graph_get_tensor(gf, "out");
        std::vector<float> res(ggml_nelements(out_t));
        ggml_backend_tensor_get(out_t, res.data(), 0, ggml_nbytes(out_t));

        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return res;
    };

    auto ref = run_one(cpu);
    auto got = run_one(gpu);

    int bad = 0;
    float max_err = 0.f, max_rel = 0.f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        const float r = d / std::max(std::fabs(ref[i]), 1e-6f);
        if (d > max_err) max_err = d;
        if (r > max_rel) max_rel = r;
        // Tolerance: the CPU reference and the GPU kernel both dequantize
        // Q4_0 then do f32 mul_mat, but in different accumulation orders
        // (CPU walks rows scalarly, Metal kernel_mul_mm uses cooperative
        // matmul on 8x8 tiles).  Observed max abs ~5e-3 on Q4_0 shapes
        // in the 256..1024 range.  Fail only if abs diff exceeds 2e-2
        // — that's 4x the Q4_0 noise floor, catches real kernel bugs
        // (like §3.29's reverted direct-store RMW which would have
        // shown up as wholesale >1e-1 drift) without flagging
        // accumulation-order drift.
        if (d > 2e-2f) {
            if (bad < 5) {
                fprintf(stderr, "\n  mismatch @ %zu: cpu=%.6g gpu=%.6g diff=%.3e rel=%.3e",
                        i, ref[i], got[i], d, r);
            }
            ++bad;
        }
    }
    if (bad == 0) {
        fprintf(stderr, "OK (K=%d N=%d T=%d B=%d fuse=%s, max_abs=%.1e max_rel=%.1e)\n",
                K, N, T, B, fuse_mode == 1 ? "gelu" : "bias", max_err, max_rel);
        return 0;
    }
    fprintf(stderr, "\n[mul_mm_fused %s] FAIL: %d / %zu mismatched (max_err=%.3e max_rel=%.3e)\n",
            label, bad, ref.size(), max_err, max_rel);
    return 1;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "CPU backend init failed\n"); return 1; }

    ggml_backend_t gpu = nullptr;
#ifdef GGML_USE_METAL
    gpu = ggml_backend_metal_init();
    fprintf(stderr, "Using Metal backend\n");
#endif
    if (!gpu) {
        fprintf(stderr, "No GPU backend compiled in; nothing to validate.\n");
        return 0;
    }

    int rc = 0;
    rc |= test_diag_mask_inf(cpu, gpu);
    rc |= test_pad_ext(cpu, gpu);
    // HiFT-sized shapes:
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/130, /*IC=*/512, /*OC=*/256, /*K=*/16, /*s0=*/8,  "ups[0]");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/1040, /*IC=*/256, /*OC=*/128, /*K=*/15, /*s0=*/5, "ups[1]");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/5200, /*IC=*/128, /*OC=*/64,  /*K=*/11, /*s0=*/3, "ups[2]");
    // A small sanity case too.
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/10,   /*IC=*/3,   /*OC=*/4,   /*K=*/5,  /*s0=*/2, "tiny");

    // MUL_MAT + ADD(bias) fusion (PROGRESS §3.27): CFM transformer hot shapes.
    //   K=256, N=256 — attn to_q / to_k / to_v
    //   K=256, N=512 — attn to_out
    //   K=256, N=1024 — FF gate (ff0; also tested with gelu)
    //   K=1024, N=256 — FF down (ff2)
    // T=87, B=2 matches CFM's use_b2=true configuration.
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/ 256, /*N=*/ 256, /*T=*/87, /*B=*/2, /*fuse=*/0, "cfm-attn-qkv");
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/ 256, /*N=*/ 512, /*T=*/87, /*B=*/2, /*fuse=*/0, "cfm-attn-out");
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/ 256, /*N=*/1024, /*T=*/87, /*B=*/2, /*fuse=*/0, "cfm-ff-gate-bias");
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/ 256, /*N=*/1024, /*T=*/87, /*B=*/2, /*fuse=*/1, "cfm-ff-gate-bias+gelu");
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/1024, /*N=*/ 256, /*T=*/87, /*B=*/2, /*fuse=*/0, "cfm-ff-down");
    // Batch=1 sanity — exercises the non-batch path of the dispatcher.
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/ 256, /*N=*/ 512, /*T=*/87, /*B=*/1, /*fuse=*/0, "cfm-b1");
    // Non-64-multiple N to exercise the bounds-checked (bco=1) shmem path.
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/ 256, /*N=*/ 320, /*T=*/87, /*B=*/2, /*fuse=*/0, "bco-bias");
    rc |= test_mul_mm_fused(cpu, gpu, /*K=*/ 256, /*N=*/ 320, /*T=*/87, /*B=*/2, /*fuse=*/1, "bco-gelu");

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return rc;
}
