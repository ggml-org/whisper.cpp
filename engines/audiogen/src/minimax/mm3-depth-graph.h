#pragma once

#include "mm3-linear.h"
#include "mm3-model.h"
#include "mm3-sample.h"

#include "backend.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define MM3_DEPTH_MAX_NODES 512

#define MM3_DEPTH_MAX_STEPS 16

#define MM3_DEPTH_SEED_POSITIONS 2

struct MM3DepthStep {
    ggml_backend_sched_t sched = nullptr;
    ggml_context *       gctx  = nullptr;
    uint8_t *            gbuf  = nullptr;
    ggml_cgraph *        graph = nullptr;

    ggml_tensor * in_hidden = nullptr;
    ggml_tensor * in_sem    = nullptr;
    ggml_tensor * in_ac     = nullptr;
    ggml_tensor * in_rows   = nullptr;
    ggml_tensor * out_fused = nullptr;

    ggml_tensor * mask = nullptr;
    int64_t       S    = 0;
    size_t        compute_bytes = 0;
    int           n_nodes       = 0;
};

struct MM3DepthGraph {
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    bool           backend_ref = false;
    WeightCtx      prep        = {};

    const void * synth_token = nullptr;
    const void * lm_token    = nullptr;

    // One K/V cache per layer, sized for the whole codebook sequence. Every
    // frame rewrites positions 0..NC in order, so no step ever reads a position
    // left over from the previous frame. F32 where the LM cache is F16: at
    // eight positions the whole cache is 2 MB, the attention here is
    // launch-bound rather than bandwidth-bound, and F16 would add a cast per
    // layer per step while changing the numerics the old full-prefix path had.
    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k;
    std::vector<ggml_tensor *> kv_v;

    MM3DepthStep step[MM3_DEPTH_MAX_STEPS];
    int          n_steps = 0;
};

struct MM3DepthFrame {
    int32_t            codes[MM3_DEPTH_MAX_STEPS] = { 0 };
    std::vector<float> hiddens;
    std::vector<float> logits_cond;
    std::vector<float> logits_uncond;
    int                n_codes = 0;
    double             ms      = 0.0;
};

static void mm3_depth_free_step(MM3DepthStep * s) {
    if (s->gctx) {
        if (s->sched) {
            ggml_backend_sched_reset(s->sched);
        }
        ggml_free(s->gctx);
        free(s->gbuf);
    }
    if (s->sched) {
        ggml_backend_sched_free(s->sched);
    }
    *s = MM3DepthStep{};
}

static void mm3_depth_free_kv(MM3DepthGraph * g) {
    if (g->kv_buf) {
        ggml_backend_buffer_free(g->kv_buf);
    }
    if (g->kv_ctx) {
        ggml_free(g->kv_ctx);
    }
    g->kv_buf = nullptr;
    g->kv_ctx = nullptr;
    g->kv_k.clear();
    g->kv_v.clear();
}

static void mm3_depth_free(MM3DepthGraph * g) {
    for (int i = 0; i < MM3_DEPTH_MAX_STEPS; i++) {
        mm3_depth_free_step(&g->step[i]);
    }
    mm3_depth_free_kv(g);
    wctx_free(&g->prep);
    g->n_steps     = 0;
    g->synth_token = nullptr;
    g->lm_token    = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static ggml_tensor * mm3_depth_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Attends the step's new token(s) against the layer's K/V cache. `window` is
// how many cached positions are live for this step; everything beyond it
// belongs to a later step and must not be read.
static ggml_tensor * mm3_depth_block(ggml_context * ctx, ggml_cgraph * gf, const MM3DepthConfig & c,
                                     const MM3DepthLayer & w, ggml_tensor * h, ggml_tensor * mask,
                                     ggml_tensor * kcache, ggml_tensor * vcache, ggml_tensor * rows,
                                     int64_t window) {
    const int64_t H  = (int64_t) c.embedding_length;
    const int64_t D  = (int64_t) c.head_dim;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t T  = h->ne[1];
    const int64_t B  = h->ne[2];

    ggml_tensor * n = mm3_depth_norm(ctx, h, w.attn_norm, c.rms_eps);

    const MM3AttentionInputs inputs = { w.attn_qkv, w.attn_q, w.attn_k, w.attn_v };
    const MM3HeadProjections heads  = mm3_project_heads(ctx, inputs, n, D, Nh, Nh, GGML_PREC_F32);

    ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, heads.q, 0, 2, 1, 3));
    ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, heads.k, 0, 2, 1, 3));
    ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, heads.v, 0, 2, 1, 3));

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kcache, k, rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vcache, v, rows));

    ggml_tensor * k_win =
        ggml_view_4d(ctx, kcache, D, window, Nh, B, kcache->nb[1], kcache->nb[2], kcache->nb[3], 0);
    ggml_tensor * v_win =
        ggml_view_4d(ctx, vcache, D, window, Nh, B, vcache->nb[1], vcache->nb[2], vcache->nb[3], 0);

    const float   scale  = 1.0f / sqrtf((float) D);
    ggml_tensor * scores = ggml_mul_mat(ctx, k_win, q);
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

    ggml_tensor * vt   = ggml_cont(ctx, ggml_transpose(ctx, v_win));
    ggml_tensor * attn = ggml_mul_mat(ctx, vt, scores);
    attn               = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn               = ggml_reshape_3d(ctx, attn, H, T, B);

    h = ggml_add(ctx, h, mm3_linear(ctx, w.attn_output, attn, GGML_PREC_F32));

    ggml_tensor * n2    = mm3_depth_norm(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gated = mm3_gated_ffn_input(ctx, w.ffn_gate_up, w.ffn_gate, w.ffn_up, n2, GGML_PREC_F32);
    ggml_tensor * y     = mm3_linear(ctx, w.ffn_down, gated, GGML_PREC_F32);
    return ggml_add(ctx, h, y);
}

static void mm3_depth_add_kv_tensors(MM3DepthGraph * g, int64_t D, int64_t positions, int64_t Nh) {
    for (size_t i = 0; i < g->kv_k.size(); i++) {
        char nm[64];
        g->kv_k[i] = ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F32, D, positions, Nh, 2);
        snprintf(nm, sizeof(nm), "mm3.depth.kv_k.%zu", i);
        ggml_set_name(g->kv_k[i], nm);
        g->kv_v[i] = ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F32, D, positions, Nh, 2);
        snprintf(nm, sizeof(nm), "mm3.depth.kv_v.%zu", i);
        ggml_set_name(g->kv_v[i], nm);
    }
}

static bool mm3_depth_define_kv(const MM3DepthConfig & c, MM3DepthGraph * g, int n_steps, std::string * err) {
    const size_t layers = (size_t) c.block_count;

    ggml_init_params ip = { layers * 2 * ggml_tensor_overhead() + 1024, NULL, true };
    g->kv_ctx           = ggml_init(ip);
    if (!g->kv_ctx) {
        if (err) {
            *err = "ggml_init failed for the depth KV cache context";
        }
        return false;
    }
    g->kv_k.assign(layers, nullptr);
    g->kv_v.assign(layers, nullptr);
    mm3_depth_add_kv_tensors(g, (int64_t) c.head_dim, (int64_t) n_steps + 1, (int64_t) c.head_count);
    return true;
}

static bool mm3_depth_alloc_kv(const MM3Model & m, MM3DepthGraph * g, int n_steps, std::string * err) {
    if (!mm3_depth_define_kv(m.synth_cfg.depth, g, n_steps, err)) {
        return false;
    }
    g->kv_buf = ggml_backend_alloc_ctx_tensors(g->kv_ctx, g->backend);
    if (!g->kv_buf) {
        mm3_depth_free_kv(g);
        if (err) {
            *err = "backend buffer allocation failed for the depth KV cache";
        }
        return false;
    }
    ggml_backend_buffer_clear(g->kv_buf, 0);
    return true;
}

static size_t mm3_depth_step_context_bytes() {
    return ggml_tensor_overhead() * (MM3_DEPTH_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_DEPTH_MAX_NODES, false);
}

static ggml_context * mm3_depth_step_context(MM3DepthStep * s, std::string * err) {
    const size_t ctx_bytes = mm3_depth_step_context_bytes();
    s->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!s->gbuf) {
        if (err) {
            *err = "out of host memory allocating the depth graph context";
        }
        return nullptr;
    }
    ggml_init_params ip  = { ctx_bytes, s->gbuf,  true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(s->gbuf);
        s->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the depth graph context";
        }
        return nullptr;
    }
    return ctx;
}

static ggml_cgraph * mm3_depth_step_graph(ggml_context * ctx, const MM3Model & m, const MM3DepthGraph & g,
                                          MM3DepthStep * s, int cb) {
    const MM3DepthConfig &  c  = m.synth_cfg.depth;
    const MM3DepthWeights & w  = m.synth.depth;
    const int64_t           H  = (int64_t) c.embedding_length;
    const int64_t           S  = cb + 1;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MM3_DEPTH_MAX_NODES, false);

    const tts_cpp::minimax::detail::DepthStepLayout layout =
        tts_cpp::minimax::detail::depth_step_layout(cb);
    const int64_t T = layout.tokens;

    s->in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    ggml_set_name(s->in_rows, "mm3_depth_kv_rows");
    ggml_set_input(s->in_rows);

    ggml_tensor * seq = nullptr;
    if (cb == 1) {
        s->in_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 2);
        ggml_set_name(s->in_hidden, "mm3_depth_lm_hidden");
        ggml_set_input(s->in_hidden);

        s->in_sem = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(s->in_sem, "mm3_depth_semantic_id");
        ggml_set_input(s->in_sem);

        ggml_tensor * semantic = ggml_get_rows(ctx, m.lm.token_embd, s->in_sem);
        ggml_tensor * tok0     = mm3_linear(ctx, w.proj, s->in_hidden, GGML_PREC_F32);
        ggml_tensor * shared   = mm3_linear(ctx, w.proj, semantic, GGML_PREC_F32);
        shared                 = ggml_concat(ctx, shared, shared, 2);
        seq                    = ggml_concat(ctx, tok0, shared, 1);
    } else {
        s->in_ac = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(s->in_ac, "mm3_depth_acoustic_rows");
        ggml_set_input(s->in_ac);

        ggml_tensor * acoustic = ggml_get_rows(ctx, w.audio_embd, s->in_ac);
        ggml_tensor * ac       = mm3_linear(ctx, w.proj, acoustic, GGML_PREC_F32);
        seq                    = ggml_concat(ctx, ac, ac, 2);
    }

    ggml_tensor * pos =
        ggml_view_2d(ctx, w.pos_embd, H, T, w.pos_embd->nb[1], (size_t) layout.first * w.pos_embd->nb[1]);
    ggml_tensor * h = ggml_add(ctx, seq, pos);

    for (size_t i = 0; i < w.blk.size(); i++) {
        h = mm3_depth_block(ctx, gf, c, w.blk[i], h, s->mask, g.kv_k[i], g.kv_v[i], s->in_rows, S);
    }
    h = mm3_depth_norm(ctx, h, w.output_norm, c.rms_eps);

    ggml_tensor * last = ggml_cont(
        ctx, ggml_view_3d(ctx, h, H, 1, 2, h->nb[1], h->nb[2], (size_t) (T - 1) * h->nb[1]));

    ggml_tensor * logits = mm3_linear(ctx, w.head[(size_t) (cb - 1)], last, GGML_PREC_F32);
    ggml_set_name(last, "mm3_depth_hidden");
    ggml_set_name(logits, "mm3_depth_logits");

    // One fused output per step: [hidden row0, hidden row1, logits row0,
    // logits row1], so the host needs a single readback instead of two.
    ggml_tensor * hid_flat   = ggml_reshape_1d(ctx, last, ggml_nelements(last));
    ggml_tensor * logit_flat = ggml_reshape_1d(ctx, logits, ggml_nelements(logits));
    s->out_fused             = ggml_concat(ctx, hid_flat, logit_flat, 0);
    ggml_set_name(s->out_fused, "mm3_depth_fused");
    ggml_set_output(s->out_fused);

    ggml_build_forward_expand(gf, s->out_fused);
    return gf;
}

static bool mm3_depth_build_step(const MM3Model & m, MM3DepthGraph * g, int cb, std::string * err) {
    MM3DepthStep * s   = &g->step[cb - 1];
    ggml_context * ctx = mm3_depth_step_context(s, err);
    if (!ctx) {
        return false;
    }
    s->graph = mm3_depth_step_graph(ctx, m, *g, s, cb);

    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, s->graph)) {
        ggml_free(ctx);
        free(s->gbuf);
        s->gbuf  = nullptr;
        s->graph = nullptr;
        if (err) {
            *err = "depth graph allocation failed (out of VRAM?) for codebook " + std::to_string(cb);
        }
        return false;
    }

    s->gctx          = ctx;
    s->S             = cb + 1;
    s->n_nodes       = ggml_graph_n_nodes(s->graph);
    s->compute_bytes = ggml_backend_sched_get_buffer_size(s->sched, g->backend);
    return true;
}

static int mm3_depth_step_count(const MM3DepthConfig & c) {
    return (int) c.num_codebooks - 1;
}

static bool mm3_depth_validate(const MM3Model & m, std::string * err) {
    const MM3DepthConfig & c = m.synth_cfg.depth;
    if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 || c.head_dim == 0) {
        if (err) {
            *err = "depth config is empty — mm3.depth.* KVs missing from the synth GGUF";
        }
        return false;
    }
    if (c.rope) {
        if (err) {
            *err = "mm3.depth.rope is true; this port implements the documented learned-absolute-position variant only";
        }
        return false;
    }
    if (!c.causal) {
        if (err) {
            *err = "mm3.depth.causal is false; this port implements the documented causal variant only";
        }
        return false;
    }
    const int NC = mm3_depth_step_count(c);
    if (NC < 1 || NC >= MM3_DEPTH_MAX_STEPS) {
        if (err) {
            *err = "mm3.depth.num_codebooks - 1 = " + std::to_string(NC) + " is outside 1.." +
                   std::to_string(MM3_DEPTH_MAX_STEPS - 1);
        }
        return false;
    }
    if ((int64_t) c.max_position < (int64_t) NC + 1) {
        if (err) {
            *err = "depth.pos_embd has " + std::to_string(c.max_position) + " rows, need " + std::to_string(NC + 1);
        }
        return false;
    }
    if (!m.lm.token_embd) {
        if (err) {
            *err = "the LM token embedding is not resident; the depth decoder needs it for the semantic code";
        }
        return false;
    }

    if (!m.synth.depth.pos_embd || m.synth.depth.pos_embd->type != GGML_TYPE_F32) {
        if (err) {
            *err = "depth.pos_embd.weight is not F32; the layout contract pins it to F32";
        }
        return false;
    }
    return true;
}

// Only the seeding step writes more than one position, so it is the only
// one that needs a mask; every later step has a single query that may
// attend to every position in its window.
static ggml_tensor * mm3_depth_mask_tensor(WeightCtx * prep) {
    ggml_tensor * t = ggml_new_tensor_2d(prep->ctx, GGML_TYPE_F32, MM3_DEPTH_SEED_POSITIONS, MM3_DEPTH_SEED_POSITIONS);
    ggml_set_name(t, "mm3.depth.causal_mask");
    return t;
}

static void mm3_depth_fill_causal_mask(float * mask, int64_t positions) {
    for (int64_t q = 0; q < positions; q++) {
        for (int64_t k = 0; k < positions; k++) {
            mask[(size_t) (k + q * positions)] = k <= q ? 0.0f : -INFINITY;
        }
    }
}

static void mm3_depth_stage_mask(WeightCtx * prep, ggml_tensor * mask) {
    const size_t elements = (size_t) (MM3_DEPTH_SEED_POSITIONS * MM3_DEPTH_SEED_POSITIONS);
    auto         d        = std::make_unique<float[]>(elements);
    mm3_depth_fill_causal_mask(d.get(), MM3_DEPTH_SEED_POSITIONS);
    wctx_queue_copy(prep, mask, d.get(), elements * sizeof(float), 0);
    prep->staging.push_back(std::move(d));
}

static bool mm3_depth_prepare(const MM3Model & m, MM3DepthGraph * g, std::string * err) {
    if (!m.loaded) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const void * st = (const void *) m.wctx_synth.buffer;
    const void * lt = (const void *) m.wctx_lm.buffer;
    if (g->synth_token == st && g->lm_token == lt && g->n_steps > 0) {
        return true;
    }
    mm3_depth_free(g);

    if (!mm3_depth_validate(m, err)) {
        return false;
    }
    const MM3DepthConfig & c  = m.synth_cfg.depth;
    const int              NC = mm3_depth_step_count(c);

    BackendPair bp = backend_init("MM3-Depth");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    wctx_init(&g->prep, 1);
    g->step[0].mask = mm3_depth_mask_tensor(&g->prep);
    mm3_depth_stage_mask(&g->prep, g->step[0].mask);
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the depth causal mask";
        }
        mm3_depth_free(g);
        return false;
    }

    if (!mm3_depth_alloc_kv(m, g, NC, err)) {
        mm3_depth_free(g);
        return false;
    }

    for (int cb = 1; cb <= NC; cb++) {
        g->step[cb - 1].sched = backend_sched_new(bp, MM3_DEPTH_MAX_NODES * 2);
        if (!mm3_depth_build_step(m, g, cb, err)) {
            mm3_depth_free(g);
            return false;
        }
    }
    g->n_steps     = NC;
    g->synth_token = st;
    g->lm_token    = lt;

    size_t total = 0;
    for (int i = 0; i < NC; i++) {
        total += g->step[i].compute_bytes;
    }
    fprintf(stderr,
            "[MM3-Depth] Prepared: %u blocks, %u heads x %u, %d step graphs (%d nodes each), "
            "compute buffers %.1f MB, KV cache %d positions %.1f MB\n",
            c.block_count, c.head_count, c.head_dim, NC, g->step[NC - 1].n_nodes,
            (double) total / (1024.0 * 1024.0), NC + 1,
            (double) ggml_backend_buffer_get_size(g->kv_buf) / (1024.0 * 1024.0));
    return true;
}

static MM3DepthGraph g_mm3_depth;

static bool mm3_depth_decode_frame(const MM3Model & m, const float * lm_hidden_cond, const float * lm_hidden_uncond,
                                   int32_t semantic_code, const int32_t * forced_codes, MM3DepthFrame * out,
                                   std::string * err = nullptr, std::mt19937_64 * rng = nullptr, int top_k = 0) {
#ifdef MM3_COREML_RUNTIME_ENABLED
    if (m.ane_validation) {
        mm3_ane_validation::depth(*m.ane_validation, lm_hidden_cond, lm_hidden_uncond, semantic_code,
                                   forced_codes, out, rng, top_k);
        return true;
    }
#endif
    if (!mm3_depth_prepare(m, &g_mm3_depth, err)) {
        return false;
    }
    const MM3DepthConfig & c  = m.synth_cfg.depth;
    const MM3LmConfig &    lc = m.lm_cfg;
    const int64_t          H  = (int64_t) c.embedding_length;
    const int64_t          V  = (int64_t) c.audio_vocab_size;
    const int              NC = g_mm3_depth.n_steps;

    if (semantic_code < 0 || (uint32_t) semantic_code >= lc.semantic_vocab_size) {
        if (err) {
            *err = "semantic code " + std::to_string(semantic_code) + " is outside [0, " +
                   std::to_string(lc.semantic_vocab_size) + ")";
        }
        return false;
    }
    if (forced_codes) {
        for (int i = 0; i < NC; i++) {
            if (forced_codes[i] < 0 || (int64_t) forced_codes[i] >= V) {
                if (err) {
                    *err = "forced code " + std::to_string(forced_codes[i]) + " for codebook " + std::to_string(i + 1) +
                           " is outside [0, " + std::to_string(V) + ")";
                }
                return false;
            }
        }
    }

    out->n_codes = NC;
    out->hiddens.assign((size_t) (NC * H), 0.0f);
    out->logits_cond.assign((size_t) (NC * V), 0.0f);
    out->logits_uncond.assign((size_t) (NC * V), 0.0f);

    const int32_t      sem_id  = semantic_code + (int32_t) lc.semantic_vocab_offset;
    const float        cfg     = lc.ar_cfg_scale > 0.0f ? lc.ar_cfg_scale : 1.5f;
    std::vector<int32_t> ac_rows((size_t) NC, 0);
    std::vector<float>   fused_buf((size_t) (H * 2 + V * 2));
    std::vector<float>   samp_scratch;

    const auto t0 = std::chrono::steady_clock::now();
    for (int cb = 1; cb <= NC; cb++) {
        MM3DepthStep * s = &g_mm3_depth.step[cb - 1];

        // Uploaded every frame even though the rows depend only on the step:
        // in_rows is a graph input, and the graph allocator hands its storage
        // to later nodes once the set_rows consumers have run, so a value set
        // once at build time does not survive the first compute on a backend
        // whose inputs share the compute buffer. The LM uploads its rows the
        // same way.
        const auto    layout     = tts_cpp::minimax::detail::depth_step_layout(cb);
        const int64_t kv_rows[2] = { layout.first, layout.first + 1 };
        ggml_backend_tensor_set(s->in_rows, kv_rows, 0, (size_t) layout.tokens * sizeof(int64_t));

        if (s->in_hidden) {
            ggml_backend_tensor_set(s->in_hidden, lm_hidden_cond, 0, (size_t) H * sizeof(float));
            ggml_backend_tensor_set(s->in_hidden, lm_hidden_uncond, (size_t) H * sizeof(float),
                                    (size_t) H * sizeof(float));
        }
        if (s->in_sem) {
            ggml_backend_tensor_set(s->in_sem, &sem_id, 0, sizeof(int32_t));
        }
        if (s->in_ac) {
            ggml_backend_tensor_set(s->in_ac, &ac_rows[(size_t) (cb - 2)], 0, sizeof(int32_t));
        }

        if (ggml_backend_sched_graph_compute(s->sched, s->graph) != GGML_STATUS_SUCCESS) {
            if (err) {
                *err = "depth graph compute failed at codebook " + std::to_string(cb);
            }
            return false;
        }

        ggml_backend_tensor_get(s->out_fused, fused_buf.data(), 0, fused_buf.size() * sizeof(float));

        // Offsets follow the fused layout built in mm3_depth_build_step.
        float * hid_buf   = fused_buf.data();
        float * logit_buf = fused_buf.data() + (size_t) (H * 2);

        float * lc_row = out->logits_cond.data() + (size_t) ((cb - 1) * V);
        float * lu_row = out->logits_uncond.data() + (size_t) ((cb - 1) * V);
        memcpy(lc_row, logit_buf, (size_t) V * sizeof(float));
        memcpy(lu_row, logit_buf + V, (size_t) V * sizeof(float));

        memcpy(out->hiddens.data() + (size_t) ((cb - 1) * H), hid_buf, (size_t) H * sizeof(float));

        int32_t code;
        if (forced_codes) {
            code = forced_codes[cb - 1];
        } else {

            float * guided = logit_buf + V;
            for (int64_t i = 0; i < V; i++) {
                const float u = lu_row[i];
                guided[i]     = u + cfg * (lc_row[i] - u);
            }
            if (rng) {
                code = (int32_t) mm3_sample_top_k(guided, V, top_k, *rng, &samp_scratch);
            } else {
                int32_t best_i = 0;
                float   best_v = -INFINITY;
                for (int64_t i = 0; i < V; i++) {
                    if (guided[i] > best_v) {
                        best_v = guided[i];
                        best_i = (int32_t) i;
                    }
                }
                code = best_i;
            }
        }
        out->codes[cb - 1] = code;

        ac_rows[(size_t) (cb - 1)] = code + (int32_t) ((cb - 1) * V);
    }
    out->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}
