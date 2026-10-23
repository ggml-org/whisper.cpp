#pragma once

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Max latent frames per vocoder graph. The im2col columns of the full-rate
// convolutions scale with decoded length (a 689-frame window needs a 1.29 GiB
// compute buffer, which does not fit next to the weights on a 10 GiB GPU), so
// longer windows decode as overlapped tiles. MM3_VOC_OVERLAP exceeds the conv
// stack's receptive field (8 frames leak, 16 do not), so a tile interior never
// sees past its context: it is bit-identical to a single-shot decode on the CPU
// and matches it to rounding on GPU backends, whose GEMM kernels depend on the
// product shape.
#define MM3_VOC_CHUNK   256

#define MM3_VOC_OVERLAP 32

#define MM3_VOC_MAX_NODES 2048

struct MM3VocPrepRes {
    ggml_tensor * inv1 = nullptr;
    ggml_tensor * inv2 = nullptr;
};

struct MM3VocPrepBlk {
    ggml_tensor *              inv     = nullptr;
    ggml_tensor *              convt_w = nullptr;
    std::vector<MM3VocPrepRes> res;
};

struct MM3VocGraph {

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            prep        = {};

    const void * weights_token = nullptr;

    std::vector<MM3VocPrepBlk> blk;
    ggml_tensor *              inv_out = nullptr;

    ggml_context * gctx    = nullptr;
    uint8_t *      gbuf    = nullptr;
    ggml_cgraph *  graph   = nullptr;
    ggml_tensor *  input   = nullptr;
    ggml_tensor *  output  = nullptr;
    int64_t        graph_L = 0;
};

static bool mm3_voc_readback(const ggml_tensor * t, std::vector<float> * out, std::string * err,
                             const char * what) {
    if (!t) {
        if (err) {
            *err = std::string("vocoder tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("vocoder tensor '") + what + "' is not F32 (type " + std::to_string((int) t->type) +
                   "); the layout contract pins the vocoder to F32";
        }
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

static bool mm3_voc_expected_length(int64_t frames, int64_t upsample, int64_t * length,
                                    std::string * error) {
    if (!length || frames <= 0 || upsample <= 0 ||
        frames > std::numeric_limits<int64_t>::max() / upsample) {
        if (error) {
            *error = "vocoder output length is invalid or overflows int64";
        }
        return false;
    }
    *length = frames * upsample;
    return true;
}

static bool mm3_voc_validate_output(const ggml_tensor * output, int64_t expected_length,
                                    std::string * error) {
    if (!output) {
        if (error) {
            *error = "vocoder output tensor is missing";
        }
        return false;
    }
    if (output->type != GGML_TYPE_F32) {
        if (error) {
            *error = "vocoder output tensor must be F32";
        }
        return false;
    }
    const std::string shape_error = tts_cpp::minimax::detail::vocoder_output_shape_error(
        output->ne[0], output->ne[1], output->ne[2], output->ne[3],
        ggml_nelements(output), ggml_nbytes(output), expected_length);
    if (!shape_error.empty()) {
        if (error) {
            *error = shape_error;
        }
        return false;
    }
    return true;
}

static ggml_tensor * mm3_voc_new_prep(WeightCtx * wctx, int64_t ne0, int64_t ne1, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    return t;
}

static ggml_tensor * mm3_voc_new_inv(WeightCtx * wctx, const ggml_tensor * alpha, const char * name) {
    return mm3_voc_new_prep(wctx, 1, ggml_nelements(alpha), name);
}

static ggml_tensor * mm3_voc_new_convt_gemm(WeightCtx * wctx, const ggml_tensor * w, const char * name) {
    return mm3_voc_new_prep(wctx, w->ne[2], w->ne[0] * w->ne[1], name);
}

static void mm3_voc_stage(WeightCtx * wctx, ggml_tensor * dst, std::unique_ptr<float[]> data) {
    wctx_queue_copy(wctx, dst, data.get(), ggml_nbytes(dst), 0);
    wctx->staging.push_back(std::move(data));
}

static void mm3_voc_fill_inv(const std::vector<float> & alpha, float eps, float * inv) {
    for (size_t i = 0; i < alpha.size(); i++) {
        inv[i] = 1.0f / (alpha[i] + eps);
    }
}

static bool mm3_voc_stage_inv(WeightCtx * wctx, ggml_tensor * dst, const ggml_tensor * alpha, float eps,
                              std::string * err) {
    std::vector<float> a;
    if (!mm3_voc_readback(alpha, &a, err, ggml_get_name(dst))) {
        return false;
    }
    auto inv = std::make_unique<float[]>(a.size());
    mm3_voc_fill_inv(a, eps, inv.get());
    mm3_voc_stage(wctx, dst, std::move(inv));
    return true;
}

static void mm3_voc_repack_rows(const float * src, int64_t K, int64_t OC, int64_t IC, float * dst) {
    for (int64_t ic = 0; ic < IC; ic++) {
        const float * s = src + ic * K * OC;
        for (int64_t oc = 0; oc < OC; oc++) {
            for (int64_t k = 0; k < K; k++) {
                dst[(size_t) (ic + (k + oc * K) * IC)] = s[(size_t) (k + oc * K)];
            }
        }
    }
}

static bool mm3_voc_stage_convt(WeightCtx * wctx, ggml_tensor * dst, const ggml_tensor * w, std::string * err) {
    std::vector<float> src;
    if (!mm3_voc_readback(w, &src, err, ggml_get_name(dst))) {
        return false;
    }
    auto repacked = std::make_unique<float[]>(src.size());
    mm3_voc_repack_rows(src.data(), w->ne[0], w->ne[1], w->ne[2], repacked.get());
    mm3_voc_stage(wctx, dst, std::move(repacked));
    return true;
}

static int mm3_voc_prep_tensor_count(const MM3VocConfig & vc) {
    return (int) vc.upsample_rates.size() * (1 + 1 + (int) vc.res_dilations.size() * 2) + 1;
}

static void mm3_voc_define_res_prep(WeightCtx * prep, const MM3VocBlock & wb, int b, MM3VocPrepBlk & pb) {
    char nm[96];
    for (size_t r = 0; r < pb.res.size(); r++) {
        snprintf(nm, sizeof(nm), "voc.blk.%d.res.%zu.snake1.inv_alpha", b, r);
        pb.res[r].inv1 = mm3_voc_new_inv(prep, wb.res[r].snake1_alpha, nm);
        snprintf(nm, sizeof(nm), "voc.blk.%d.res.%zu.snake2.inv_alpha", b, r);
        pb.res[r].inv2 = mm3_voc_new_inv(prep, wb.res[r].snake2_alpha, nm);
    }
}

static void mm3_voc_define_block_prep(WeightCtx * prep, const MM3Model & m, MM3VocGraph * g) {
    const MM3VocConfig &  vc = m.synth_cfg.voc;
    const MM3VocWeights & vw = m.synth.voc;
    char                  nm[96];
    g->blk.assign(vc.upsample_rates.size(), MM3VocPrepBlk{});
    for (size_t b = 0; b < g->blk.size(); b++) {
        MM3VocPrepBlk & pb = g->blk[b];
        snprintf(nm, sizeof(nm), "voc.blk.%zu.snake.inv_alpha", b);
        pb.inv = mm3_voc_new_inv(prep, vw.blk[b].snake_alpha, nm);
        snprintf(nm, sizeof(nm), "voc.blk.%zu.convt.gemm", b);
        pb.convt_w = mm3_voc_new_convt_gemm(prep, vw.blk[b].convt_w, nm);
        pb.res.assign(vc.res_dilations.size(), MM3VocPrepRes{});
        mm3_voc_define_res_prep(prep, vw.blk[b], (int) b, pb);
    }
}

static void mm3_voc_define_prep(const MM3Model & m, MM3VocGraph * g) {
    wctx_init(&g->prep, mm3_voc_prep_tensor_count(m.synth_cfg.voc));
    mm3_voc_define_block_prep(&g->prep, m, g);
    g->inv_out = mm3_voc_new_inv(&g->prep, m.synth.voc.snake_out_alpha, "voc.snake_out.inv_alpha");
}

static bool mm3_voc_stage_res_prep(WeightCtx * prep, const MM3VocBlock & wb, const MM3VocPrepBlk & pb, float eps,
                                   std::string * err) {
    for (size_t r = 0; r < pb.res.size(); r++) {
        if (!mm3_voc_stage_inv(prep, pb.res[r].inv1, wb.res[r].snake1_alpha, eps, err) ||
            !mm3_voc_stage_inv(prep, pb.res[r].inv2, wb.res[r].snake2_alpha, eps, err)) {
            return false;
        }
    }
    return true;
}

static bool mm3_voc_stage_block_prep(WeightCtx * prep, const MM3VocWeights & vw, const MM3VocGraph & g, float eps,
                                     std::string * err) {
    for (size_t b = 0; b < g.blk.size(); b++) {
        const MM3VocPrepBlk & pb = g.blk[b];
        if (!mm3_voc_stage_inv(prep, pb.inv, vw.blk[b].snake_alpha, eps, err) ||
            !mm3_voc_stage_convt(prep, pb.convt_w, vw.blk[b].convt_w, err) ||
            !mm3_voc_stage_res_prep(prep, vw.blk[b], pb, eps, err)) {
            return false;
        }
    }
    return true;
}

static float mm3_voc_snake_eps(const MM3VocConfig & vc) {
    return vc.snake_eps > 0.0f ? vc.snake_eps : 1e-9f;
}

static bool mm3_voc_stage_prep(const MM3Model & m, MM3VocGraph * g, std::string * err) {
    const float eps = mm3_voc_snake_eps(m.synth_cfg.voc);
    return mm3_voc_stage_block_prep(&g->prep, m.synth.voc, *g, eps, err) &&
           mm3_voc_stage_inv(&g->prep, g->inv_out, m.synth.voc.snake_out_alpha, eps, err);
}

static bool mm3_voc_require_f32(const ggml_tensor * t, std::string * err) {
    if (!t) {
        if (err) {
            *err = "vocoder tensor missing";
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("vocoder tensor '") + ggml_get_name(t) + "' is not F32 (type " +
                   std::to_string((int) t->type) + "); the layout contract pins the vocoder to F32";
        }
        return false;
    }
    return true;
}

static bool mm3_voc_validate_res(const MM3VocBlock & wb, std::string * err) {
    for (const MM3VocResUnit & ru : wb.res) {
        if (!mm3_voc_require_f32(ru.snake1_alpha, err) || !mm3_voc_require_f32(ru.snake2_alpha, err)) {
            return false;
        }
    }
    return true;
}

static bool mm3_voc_validate_blocks(const MM3VocWeights & vw, std::string * err) {
    for (const MM3VocBlock & wb : vw.blk) {
        if (!mm3_voc_require_f32(wb.snake_alpha, err) || !mm3_voc_require_f32(wb.convt_w, err) ||
            !mm3_voc_validate_res(wb, err)) {
            return false;
        }
    }
    return true;
}

static bool mm3_voc_validate(const MM3Model & m, std::string * err) {
    return mm3_voc_validate_blocks(m.synth.voc, err) && mm3_voc_require_f32(m.synth.voc.snake_out_alpha, err);
}

static void mm3_vocoder_free_graph(MM3VocGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx    = nullptr;
    g->gbuf    = nullptr;
    g->graph   = nullptr;
    g->input   = nullptr;
    g->output  = nullptr;
    g->graph_L = 0;
}

static void mm3_vocoder_free(MM3VocGraph * g) {
    mm3_vocoder_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->blk.clear();
    g->inv_out       = nullptr;
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static bool mm3_vocoder_prepare(const MM3Model & m, MM3VocGraph * g, std::string * err) {
    if (!m.loaded) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const void * token = (const void *) m.wctx_synth.buffer;
    if (g->weights_token == token && g->sched) {
        return true;
    }
    mm3_vocoder_free(g);

    if (!mm3_voc_validate(m, err)) {
        return false;
    }

    BackendPair bp = backend_init("MM3-Voc");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    mm3_voc_define_prep(m, g);
    std::string e;
    bool        ok = mm3_voc_stage_prep(m, g, &e);
    if (ok) {
        ok = wctx_alloc(&g->prep, g->backend);
        if (!ok) {
            e = "backend buffer allocation failed for the vocoder derived weights";
        }
    }
    if (!ok) {
        if (err) {
            *err = e.empty() ? "vocoder prepare failed" : e;
        }
        mm3_vocoder_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, MM3_VOC_MAX_NODES * 2);
    g->weights_token = token;

    const size_t prep_bytes = g->prep.buffer ? ggml_backend_buffer_get_size(g->prep.buffer) : 0;
    fprintf(stderr, "[MM3-Voc] Prepared: %d derived tensors, %.1f MB (inv-alpha + repacked convT)\n",
            mm3_voc_prep_tensor_count(m.synth_cfg.voc), (double) prep_bytes / (1024.0 * 1024.0));
    return true;
}

static ggml_tensor * mm3_voc_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha,
                                   ggml_tensor * inv_alpha) {
    ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_sqr(ctx, s);
    ggml_tensor * d  = ggml_mul(ctx, s2, inv_alpha);
    return ggml_add(ctx, x, d);
}

static ggml_tensor * mm3_voc_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
                                    int pad, int dilation) {
    ggml_tensor * col = ggml_im2col(ctx, w, x,  1,  0, pad, 0, dilation, 0,  false,
                                    GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                   ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

static ggml_tensor * mm3_voc_convt(ggml_context * ctx, ggml_tensor * wg, ggml_tensor * b, ggml_tensor * x,
                                   int stride, int64_t oc) {
    const int64_t T_in   = x->ne[0];
    const int     S      = stride;
    const int     K      = 2 * S;
    const int64_t T_full = (T_in - 1) * S + K;
    const int     p      = (S + 1) / 2;

    ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * col = ggml_mul_mat(ctx, wg, xt);
    ggml_tensor * c3  = ggml_reshape_3d(ctx, col, K, oc, T_in);

    ggml_tensor * lo = ggml_view_3d(ctx, c3, S, oc, T_in, c3->nb[1], c3->nb[2], 0);
    lo               = ggml_cont(ctx, ggml_permute(ctx, lo, 0, 2, 1, 3));
    lo               = ggml_reshape_2d(ctx, lo, (int64_t) S * T_in, oc);
    ggml_tensor * y  = ggml_pad(ctx, lo, S, 0, 0, 0);

    ggml_tensor * hi = ggml_view_3d(ctx, c3, S, oc, T_in, c3->nb[1], c3->nb[2], (size_t) S * c3->nb[0]);
    hi               = ggml_cont(ctx, ggml_permute(ctx, hi, 0, 2, 1, 3));
    hi               = ggml_reshape_2d(ctx, hi, (int64_t) S * T_in, oc);
    y = ggml_acc(ctx, y, hi, y->nb[1], y->nb[2], y->nb[3], (size_t) S * sizeof(float));

    y = ggml_cont(ctx, ggml_view_2d(ctx, y, T_full - 2 * p, oc, y->nb[1], (size_t) p * sizeof(float)));

    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

static ggml_tensor * mm3_voc_res_unit(ggml_context * ctx, const MM3VocResUnit & w, const MM3VocPrepRes & p,
                                      ggml_tensor * x, int dilation) {
    ggml_tensor * skip = x;
    ggml_tensor * y    = mm3_voc_snake(ctx, x, w.snake1_alpha, p.inv1);
    y                  = mm3_voc_conv1d(ctx, w.conv1_w, w.conv1_b, y, 3 * dilation, dilation);
    y                  = mm3_voc_snake(ctx, y, w.snake2_alpha, p.inv2);
    y                  = mm3_voc_conv1d(ctx, w.conv2_w, w.conv2_b, y, 0, 1);
    if (y->ne[0] != skip->ne[0]) {
        const int64_t off = (skip->ne[0] - y->ne[0]) / 2;
        skip = ggml_cont(ctx, ggml_view_2d(ctx, skip, y->ne[0], skip->ne[1], skip->nb[1],
                                           (size_t) off * sizeof(float)));
    }
    return ggml_add(ctx, skip, y);
}

static ggml_tensor * mm3_voc_build(ggml_context * ctx, const MM3Model & m, const MM3VocGraph & g,
                                   ggml_tensor * latent) {
    const MM3VocConfig &  vc = m.synth_cfg.voc;
    const MM3VocWeights & vw = m.synth.voc;

    ggml_tensor * x = mm3_voc_conv1d(ctx, vw.dec_in_w, vw.dec_in_b, latent, 0, 1);
    x               = mm3_voc_conv1d(ctx, vw.conv_in_w, vw.conv_in_b, x, 3, 1);

    for (size_t b = 0; b < vc.upsample_rates.size(); b++) {
        const MM3VocBlock &   wb = vw.blk[b];
        const MM3VocPrepBlk & pb = g.blk[b];
        const int             s  = (int) vc.upsample_rates[b];
        const int64_t         oc = wb.convt_w->ne[1];

        x = mm3_voc_snake(ctx, x, wb.snake_alpha, pb.inv);
        x = mm3_voc_convt(ctx, pb.convt_w, wb.convt_b, x, s, oc);
        for (size_t r = 0; r < vc.res_dilations.size(); r++) {
            x = mm3_voc_res_unit(ctx, wb.res[r], pb.res[r], x, (int) vc.res_dilations[r]);
        }
    }

    x = mm3_voc_snake(ctx, x, vw.snake_out_alpha, g.inv_out);
    x = mm3_voc_conv1d(ctx, vw.conv_out_w, vw.conv_out_b, x, 3, 1);
    if (vc.final_tanh) {
        x = ggml_tanh(ctx, x);
    }
    return x;
}

static size_t mm3_voc_graph_context_bytes() {
    return ggml_tensor_overhead() * (MM3_VOC_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_VOC_MAX_NODES, false);
}

static ggml_context * mm3_voc_graph_context(MM3VocGraph * g, std::string * err) {
    const size_t ctx_bytes = mm3_voc_graph_context_bytes();
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the vocoder graph context";
        }
        return nullptr;
    }
    ggml_init_params ip = { ctx_bytes, g->gbuf,  true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the vocoder graph context";
        }
        return nullptr;
    }
    return ctx;
}

static ggml_cgraph * mm3_voc_define_graph(ggml_context * ctx, const MM3Model & m, MM3VocGraph * g, int64_t L) {
    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, (int64_t) m.synth_cfg.voc.fold_channels);
    ggml_set_name(g->input, "mm3_voc_in");
    ggml_set_input(g->input);

    g->output = mm3_voc_build(ctx, m, *g, g->input);
    ggml_set_name(g->output, "mm3_voc_out");
    ggml_set_output(g->output);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, MM3_VOC_MAX_NODES, false);
    ggml_build_forward_expand(graph, g->output);
    return graph;
}

static bool mm3_voc_ensure_graph(const MM3Model & m, MM3VocGraph * g, int64_t L, std::string * err) {
    if (g->graph && g->graph_L == L) {
        return true;
    }
    const auto rebuild_started = std::chrono::steady_clock::now();
    mm3_vocoder_free_graph(g);

    ggml_context * ctx = mm3_voc_graph_context(g, err);
    if (!ctx) {
        return false;
    }
    g->graph = mm3_voc_define_graph(ctx, m, g, L);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "vocoder graph allocation failed (out of VRAM?) for L=" + std::to_string(L);
        }
        return false;
    }

    g->gctx    = ctx;
    g->graph_L = L;

    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    const double rebuild_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rebuild_started).count();
    fprintf(stderr,
            "[MM3-Voc] Graph: L=%lld -> %lld samples, %d nodes, %d splits, compute buffer %.0f MB, "
            "built in %.0f ms\n",
            (long long) L, (long long) g->output->ne[0], ggml_graph_n_nodes(g->graph),
            ggml_backend_sched_get_n_splits(g->sched), (double) compute_bytes / (1024.0 * 1024.0), rebuild_ms);
    return true;
}

static bool mm3_voc_run(const MM3Model & m, MM3VocGraph * g, const float * src, int64_t L, float * dst,
                        std::string * err) {
    if (!src || !dst) {
        if (err) {
            *err = "vocoder source and destination buffers must not be null";
        }
        return false;
    }
    if (!mm3_voc_ensure_graph(m, g, L, err)) {
        return false;
    }
    int64_t expected_length = 0;
    if (!mm3_voc_expected_length(L, static_cast<int64_t>(m.synth_cfg.voc.total_upsample),
                                 &expected_length, err) ||
        !mm3_voc_validate_output(g->output, expected_length, err)) {
        return false;
    }
    ggml_backend_tensor_set(g->input, src, 0, ggml_nbytes(g->input));
    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "vocoder graph compute failed";
        }
        return false;
    }
    ggml_backend_tensor_get(g->output, dst, 0, ggml_nbytes(g->output));
    return true;
}

static MM3VocGraph g_mm3_voc;

struct MM3VocTile {
    int64_t core_end = 0;
    int64_t start    = 0;
    int64_t end      = 0;
};

static bool mm3_voc_single_shot(int64_t L, int64_t chunk) {
    return L <= chunk;
}

static int64_t mm3_voc_tile_core(int64_t chunk, int64_t overlap) {
    return chunk - 2 * overlap;
}

static MM3VocTile mm3_voc_tile(int64_t core_start, int64_t L, int64_t core, int64_t overlap) {
    MM3VocTile tile;
    tile.core_end = core_start + core < L ? core_start + core : L;
    tile.start    = core_start - overlap > 0 ? core_start - overlap : 0;
    tile.end      = tile.core_end + overlap < L ? tile.core_end + overlap : L;
    return tile;
}

static bool mm3_vocoder_decode_tiled(const MM3Model & m, const std::vector<float> & latents, int64_t L,
                                     std::vector<float> & out_stereo, int64_t chunk, int64_t overlap,
                                     std::string * err);

static bool mm3_vocoder_decode(const MM3Model & m, const std::vector<float> & latents, int64_t L,
                               std::vector<float> & out_stereo, std::string * err = nullptr) {
#ifdef MM3_COREML_RUNTIME_ENABLED
    if (m.ane_validation) {
        mm3_ane_validation::vocoder(*m.ane_validation, latents, L, out_stereo);
        return true;
    }
#endif
    if (L <= 0) {
        if (err) {
            *err = "frames must be > 0";
        }
        return false;
    }
    if (!mm3_vocoder_prepare(m, &g_mm3_voc, err)) {
        return false;
    }

    const MM3VocConfig & vc  = m.synth_cfg.voc;
    const int64_t        up  = (int64_t) vc.total_upsample;
    const int64_t        FC  = (int64_t) vc.fold_channels;
    int64_t T = 0;
    const uint64_t fold_channels = FC > 0 ? static_cast<uint64_t>(FC) : 0;
    const uint64_t latent_elements =
        fold_channels > 0 && static_cast<uint64_t>(L) <=
                                 static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                                     (2 * fold_channels)
            ? static_cast<uint64_t>(L) * 2 * fold_channels
            : 0;
    if (!mm3_voc_expected_length(L, up, &T, err) ||
        latent_elements == 0 || latents.size() != static_cast<size_t>(latent_elements) ||
        static_cast<uint64_t>(T) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 2)) {
        if (err && err->empty()) {
            *err = "vocoder latent or waveform buffer size is invalid";
        }
        return false;
    }
    out_stereo.assign((size_t) (2 * T), 0.0f);
    return mm3_vocoder_decode_tiled(m, latents, L, out_stereo, MM3_VOC_CHUNK, MM3_VOC_OVERLAP, err);
}

// Tiled decode with explicit chunk/overlap so the tiling is testable against a
// single-shot decode; production callers use the MM3_VOC_* constants.
static bool mm3_vocoder_decode_tiled(const MM3Model & m, const std::vector<float> & latents, int64_t L,
                                     std::vector<float> & out_stereo, int64_t chunk, int64_t overlap,
                                     std::string * err) {
    const MM3VocConfig & vc = m.synth_cfg.voc;
    const int64_t        up = (int64_t) vc.total_upsample;
    const int64_t        FC = (int64_t) vc.fold_channels;
    int64_t T = 0;
    if (!mm3_voc_expected_length(L, up, &T, err)) {
        return false;
    }

    MM3VocGraph * g = &g_mm3_voc;

    if (mm3_voc_single_shot(L, chunk)) {
        for (int ch = 0; ch < 2; ch++) {
            if (!mm3_voc_run(m, g, latents.data() + (size_t) (ch * FC * L), L,
                             out_stereo.data() + (size_t) (ch * T),
                             err)) {
                return false;
            }
        }
        return true;
    }

    const int64_t      core = mm3_voc_tile_core(chunk, overlap);
    std::vector<float> win;
    std::vector<float> tile;
    for (int64_t cs = 0; cs < L; cs += core) {
        const MM3VocTile span = mm3_voc_tile(cs, L, core, overlap);
        const int64_t    ce   = span.core_end;
        const int64_t    ws   = span.start;
        const int64_t    wl   = span.end - span.start;

        win.resize((size_t) (FC * wl));
        tile.resize((size_t) (wl * up));
        for (int ch = 0; ch < 2; ch++) {
            for (int64_t c = 0; c < FC; c++) {
                const int64_t source_offset = (ch * FC + c) * L + ws;
                const int64_t destination_offset = c * wl;
                if (!tts_cpp::minimax::detail::copy_range_fits(
                        latents.size(), win.size(), source_offset, destination_offset, wl)) {
                    if (err) {
                        *err = "vocoder latent window copy range is invalid";
                    }
                    return false;
                }
                memcpy(win.data() + (size_t) destination_offset,
                       latents.data() + (size_t) source_offset,
                       (size_t) wl * sizeof(float));
            }
            if (!mm3_voc_run(m, g, win.data(), wl, tile.data(), err)) {
                return false;
            }

            const int64_t source_offset = (cs - ws) * up;
            const int64_t destination_offset = ch * T + cs * up;
            const int64_t copy_length = (ce - cs) * up;
            if (!tts_cpp::minimax::detail::copy_range_fits(
                    tile.size(), out_stereo.size(), source_offset, destination_offset, copy_length)) {
                if (err) {
                    *err = "vocoder waveform tile copy range is invalid";
                }
                return false;
            }
            memcpy(out_stereo.data() + (size_t) destination_offset,
                   tile.data() + (size_t) source_offset,
                   (size_t) copy_length * sizeof(float));
        }
    }
    return true;
}
