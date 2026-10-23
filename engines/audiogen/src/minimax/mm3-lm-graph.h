#pragma once

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"
#include "mm3-flash-attn.h"
#include "mm3-linear.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define MM3_LM_MAX_NODES 4096

#define MM3_LM_KV_BUCKET 256

#define MM3_LM_CFG_ROWS 2

struct MM3LmSlot {
    ggml_backend_sched_t sched = nullptr;
    ggml_context *       gctx  = nullptr;
    uint8_t *            gbuf  = nullptr;
    ggml_cgraph *        graph = nullptr;

    ggml_tensor * in_ids  = nullptr;

    ggml_tensor * in_pos  = nullptr;
    ggml_tensor * in_rows = nullptr;
    ggml_tensor * in_mask = nullptr;

    ggml_tensor * out_hidden   = nullptr;
    ggml_tensor * out_logits   = nullptr;
    ggml_tensor * out_feedback = nullptr;

    int64_t T             = 0;
    int64_t n_kv_pad      = 0;
    size_t  compute_bytes = 0;
    int     n_nodes       = 0;
};

struct MM3LmGraph {
    BackendPair    bp             = {};
    ggml_backend_t backend        = nullptr;
    ggml_backend_t cpu_backend    = nullptr;
    bool           backend_ref    = false;
    bool           use_flash_attn = false;

    const void * lm_token    = nullptr;
    const void * synth_token = nullptr;

    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k;
    std::vector<ggml_tensor *> kv_v;
    int64_t                    n_ctx    = 0;
    size_t                     kv_bytes = 0;
    int64_t                    kv_pos   = 0;

    MM3LmSlot prefill;
    MM3LmSlot decode;

    std::vector<int32_t>  ids_host;
    std::vector<int32_t>  pos_host;
    std::vector<int64_t>  rows_host;
    std::vector<uint16_t> mask_host;
};

static void mm3_lm_free_slot(MM3LmSlot * s) {
    if (s->gctx) {
        if (s->sched) {
            ggml_backend_sched_reset(s->sched);
        }
        ggml_free(s->gctx);
        free(s->gbuf);
    }
    ggml_backend_sched_t keep = s->sched;
    *s                        = MM3LmSlot{};
    s->sched                  = keep;
}

static void mm3_lm_free_slot_all(MM3LmSlot * s) {
    mm3_lm_free_slot(s);
    if (s->sched) {
        ggml_backend_sched_free(s->sched);
        s->sched = nullptr;
    }
}

static void mm3_lm_free_kv(MM3LmGraph * g) {
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
    g->n_ctx    = 0;
    g->kv_bytes = 0;
    g->kv_pos   = 0;
}

static void mm3_lm_free(MM3LmGraph * g) {
    mm3_lm_free_slot_all(&g->prefill);
    mm3_lm_free_slot_all(&g->decode);
    mm3_lm_free_kv(g);
    g->lm_token    = nullptr;
    g->synth_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static ggml_tensor * mm3_lm_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

static ggml_tensor * mm3_lm_attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                     ggml_tensor * mask, float scale) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Decode steps project two columns (the CFG pair), which the GPU matrix-vector
// kernels serve in f32 at no cost; the prefill's wide products keep the default
// precision and its tensor-core GEMMs.
static ggml_prec mm3_lm_linear_precision(bool decode) {
    return decode ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
}

static ggml_tensor * mm3_lm_rope(ggml_context * ctx, const MM3LmConfig & c, ggml_tensor * x, ggml_tensor * positions) {
    return ggml_rope_ext(ctx, x, positions, NULL, (int) c.key_length, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f,
                         0.0f, 1.0f, 0.0f, 0.0f);
}

// [D, heads, T * B] -> the cache's [D * heads, T * B] row layout.
static ggml_tensor * mm3_lm_cache_rows(ggml_context * ctx, ggml_tensor * x) {
    return ggml_view_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->nb[2], 0);
}

// The live [D, n_kv, heads, B] window of a cache laid out as one
// [D * heads] row per position, the positions of CFG row b starting at row
// b * n_ctx.
static ggml_tensor * mm3_lm_cache_window(ggml_context * ctx, ggml_tensor * cache, int64_t D, int64_t heads,
                                         int64_t n_kv, int64_t B) {
    const int64_t n_ctx = cache->ne[1] / B;
    return ggml_view_4d(ctx, cache, D, n_kv, heads, B, cache->nb[1], (size_t) D * cache->nb[0],
                        (size_t) n_ctx * cache->nb[1], 0);
}

static ggml_tensor * mm3_lm_block(ggml_context * ctx, ggml_cgraph * gf, const MM3LmConfig & c, const MM3LmLayer & w,
                                  ggml_tensor * h, ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * rows,
                                  ggml_tensor * kcache, ggml_tensor * vcache, int64_t n_kv_pad, bool use_flash,
                                  ggml_prec precision) {
    const int64_t H   = (int64_t) c.embedding_length;
    const int64_t D   = (int64_t) c.key_length;
    const int64_t Nh  = (int64_t) c.head_count;
    const int64_t Nkv = (int64_t) c.head_count_kv;
    const int64_t T   = h->ne[1];
    const int64_t B   = h->ne[2];

    // Tokens of both CFG rows run as one flat axis of T * B columns, so RoPE
    // sees three dimensions and the cache writes need no permute.
    ggml_tensor * n = ggml_reshape_2d(ctx, mm3_lm_rms(ctx, h, w.attn_norm, c.rms_eps), H, T * B);

    const MM3AttentionInputs inputs = { w.attn_qkv, w.attn_q, w.attn_k, w.attn_v };
    const MM3HeadProjections heads  = mm3_project_heads(ctx, inputs, n, D, Nh, Nkv, precision);

    ggml_tensor * q = ggml_mul(ctx, ggml_rms_norm(ctx, heads.q, c.rms_eps), w.attn_q_norm);
    ggml_tensor * k = ggml_mul(ctx, ggml_rms_norm(ctx, heads.k, c.rms_eps), w.attn_k_norm);
    q               = mm3_lm_rope(ctx, c, q, positions);
    k               = mm3_lm_rope(ctx, c, k, positions);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kcache, mm3_lm_cache_rows(ctx, k), rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vcache, mm3_lm_cache_rows(ctx, heads.v), rows));

    ggml_tensor * k_win = mm3_lm_cache_window(ctx, kcache, D, Nkv, n_kv_pad, B);
    ggml_tensor * v_win = mm3_lm_cache_window(ctx, vcache, D, Nkv, n_kv_pad, B);
    ggml_tensor * q4    = ggml_view_4d(ctx, q, D, T, Nh, B, q->nb[2], q->nb[1], (size_t) T * q->nb[2], 0);

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn;
    if (use_flash) {
        attn = ggml_flash_attn_ext(ctx, q4, k_win, v_win, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = mm3_lm_attn_f32(ctx, q4, k_win, v_win, mask, scale);
    }
    attn = ggml_reshape_3d(ctx, attn, H, T, B);

    h = ggml_add(ctx, h, mm3_linear(ctx, w.attn_output, attn, precision));

    ggml_tensor * n2    = mm3_lm_rms(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gated = mm3_gated_ffn_input(ctx, w.ffn_gate_up, w.ffn_gate, w.ffn_up, n2, precision);
    return ggml_add(ctx, h, mm3_linear(ctx, w.ffn_down, gated, precision));
}

static size_t mm3_lm_slot_context_bytes() {
    return ggml_tensor_overhead() * (MM3_LM_MAX_NODES + 256) + ggml_graph_overhead_custom(MM3_LM_MAX_NODES, false);
}

static ggml_context * mm3_lm_slot_context(MM3LmSlot * s, std::string * err) {
    const size_t ctx_bytes = mm3_lm_slot_context_bytes();
    s->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!s->gbuf) {
        if (err) {
            *err = "out of host memory allocating the MM3 LM graph context";
        }
        return nullptr;
    }
    ggml_init_params ip  = { ctx_bytes, s->gbuf,  true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(s->gbuf);
        s->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the MM3 LM graph context";
        }
        return nullptr;
    }
    return ctx;
}

static ggml_cgraph * mm3_lm_slot_graph(ggml_context * ctx, const MM3Model & m, const MM3LmGraph & g, MM3LmSlot * s,
                                       int64_t T, int64_t n_kv_pad, bool decode) {
    const MM3LmConfig & c  = m.lm_cfg;
    const int64_t       H  = (int64_t) c.embedding_length;
    const int64_t       B  = MM3_LM_CFG_ROWS;
    const int64_t       NC = (int64_t) c.num_codebooks - 1;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MM3_LM_MAX_NODES, false);

    s->in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T * B);
    ggml_set_name(s->in_pos, "mm3_lm_positions");
    ggml_set_input(s->in_pos);

    s->in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T * B);
    ggml_set_name(s->in_rows, "mm3_lm_kv_rows");
    ggml_set_input(s->in_rows);

    s->in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
    ggml_set_name(s->in_mask, "mm3_lm_mask");
    ggml_set_input(s->in_mask);

    ggml_tensor * h = nullptr;
    if (!decode) {

        s->in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T * B);
        ggml_set_name(s->in_ids, "mm3_lm_token_ids");
        ggml_set_input(s->in_ids);
        h = ggml_get_rows(ctx, m.lm.token_embd, s->in_ids);
        h = ggml_reshape_3d(ctx, h, H, T, B);
    } else {

        s->in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NC + 1);
        ggml_set_name(s->in_ids, "mm3_lm_feedback_rows");
        ggml_set_input(s->in_ids);

        ggml_tensor * sem_idx = ggml_view_1d(ctx, s->in_ids, 1, 0);
        ggml_tensor * ac_idx  = ggml_view_1d(ctx, s->in_ids, NC, ggml_type_size(GGML_TYPE_I32));

        ggml_tensor * e_sem = ggml_get_rows(ctx, m.lm.token_embd, sem_idx);
        ggml_tensor * e_ac  = ggml_get_rows(ctx, m.synth.depth.audio_embd, ac_idx);

        ggml_tensor * acc = ggml_view_2d(ctx, e_ac, H, 1, e_ac->nb[1], 0);
        for (int64_t i = 1; i < NC; i++) {
            acc = ggml_add(ctx, acc, ggml_view_2d(ctx, e_ac, H, 1, e_ac->nb[1], (size_t) i * e_ac->nb[1]));
        }
        ggml_tensor * fb = ggml_scale(ctx, ggml_add(ctx, e_sem, acc), c.ar_embedding_scale);

        s->out_feedback = fb;
        ggml_set_name(s->out_feedback, "mm3_lm_feedback");
        ggml_set_output(s->out_feedback);

        h = ggml_concat(ctx, fb, fb, 2);
    }

    for (size_t i = 0; i < m.lm.blk.size(); i++) {
        h = mm3_lm_block(ctx, gf, c, m.lm.blk[i], h, s->in_pos, s->in_mask, s->in_rows, g.kv_k[i], g.kv_v[i],
                         n_kv_pad, g.use_flash_attn, mm3_lm_linear_precision(decode));
    }
    h = mm3_lm_rms(ctx, h, m.lm.output_norm, c.rms_eps);

    ggml_tensor * last =
        ggml_cont(ctx, ggml_view_3d(ctx, h, H, 1, B, h->nb[1], h->nb[2], (size_t) (T - 1) * h->nb[1]));

    s->out_hidden = last;
    ggml_set_name(s->out_hidden, "mm3_lm_last_hidden");
    ggml_set_output(s->out_hidden);

    s->out_logits = mm3_linear(ctx, m.lm.output_compact, last, mm3_lm_linear_precision(decode));
    ggml_set_name(s->out_logits, "mm3_lm_logits");
    ggml_set_output(s->out_logits);

    ggml_build_forward_expand(gf, s->out_hidden);
    ggml_build_forward_expand(gf, s->out_logits);
    if (s->out_feedback) {
        ggml_build_forward_expand(gf, s->out_feedback);
    }
    return gf;
}

static bool mm3_lm_build_slot(const MM3Model & m, MM3LmGraph * g, MM3LmSlot * s, int64_t T, int64_t n_kv_pad,
                              bool decode, std::string * err) {
    ggml_context * ctx = mm3_lm_slot_context(s, err);
    if (!ctx) {
        return false;
    }
    s->graph = mm3_lm_slot_graph(ctx, m, *g, s, T, n_kv_pad, decode);

    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, s->graph)) {
        ggml_free(ctx);
        free(s->gbuf);
        s->gbuf  = nullptr;
        s->graph = nullptr;
        if (err) {
            *err = std::string("MM3 LM graph allocation failed (out of VRAM?) for the ") +
                   (decode ? "decode" : "prefill") + " slot at T=" + std::to_string(T) +
                   ", kv=" + std::to_string(n_kv_pad);
        }
        return false;
    }

    s->gctx          = ctx;
    s->T             = T;
    s->n_kv_pad      = n_kv_pad;
    s->n_nodes       = ggml_graph_n_nodes(s->graph);
    s->compute_bytes = ggml_backend_sched_get_buffer_size(s->sched, g->backend);
    return true;
}

static bool mm3_lm_bucket_for_context(int64_t positions, int64_t context_length, int64_t * bucket) {
    if (!bucket || positions <= 0 || context_length <= 0 || positions > context_length) {
        return false;
    }
    const int64_t remainder = positions % MM3_LM_KV_BUCKET;
    const int64_t padding   = remainder == 0 ? 0 : MM3_LM_KV_BUCKET - remainder;
    *bucket = padding <= context_length - positions ? positions + padding : context_length;
    return true;
}

static bool mm3_lm_validate(const MM3Model & m, std::string * err) {
    const MM3LmConfig & c = m.lm_cfg;
    if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 || c.key_length == 0) {
        if (err) {
            *err = "LM config is empty — qwen3.* KVs missing from the LM GGUF";
        }
        return false;
    }
    if (!m.lm.token_embd || !m.lm.output_compact || !m.lm.output_norm) {
        if (err) {
            *err = "the LM weights are not resident";
        }
        return false;
    }
    if (!m.synth.depth.audio_embd) {
        if (err) {
            *err = "depth.audio_embd is not resident; the AR feedback embedding needs it";
        }
        return false;
    }
    return true;
}

static bool mm3_lm_kv_positions(const MM3LmConfig & c, int64_t n_ctx_needed, int64_t * positions, std::string * err) {
    if (!mm3_lm_bucket_for_context(n_ctx_needed, c.context_length, positions)) {
        if (err) {
            *err = "requested LM KV positions exceed qwen3.context_length";
        }
        return false;
    }
    return true;
}

static bool mm3_lm_use_flash_attn(const BackendPair & bp) {
    return mm3_use_flash_attn(bp.has_gpu, mm3_lm_flash_attn_default(tts_cpp::acestep::backend_reg_name(bp.backend)),
                              "MM3_LM_NO_FLASH", "MM3_LM_FLASH");
}

static void mm3_lm_add_kv_tensors(MM3LmGraph * g, int64_t row_width, int64_t rows) {
    for (size_t i = 0; i < g->kv_k.size(); i++) {
        char nm[64];
        g->kv_k[i] = ggml_new_tensor_2d(g->kv_ctx, GGML_TYPE_F16, row_width, rows);
        snprintf(nm, sizeof(nm), "mm3.lm.kv_k.%zu", i);
        ggml_set_name(g->kv_k[i], nm);
        g->kv_v[i] = ggml_new_tensor_2d(g->kv_ctx, GGML_TYPE_F16, row_width, rows);
        snprintf(nm, sizeof(nm), "mm3.lm.kv_v.%zu", i);
        ggml_set_name(g->kv_v[i], nm);
    }
}

static bool mm3_lm_define_kv(const MM3LmConfig & c, MM3LmGraph * g, int64_t positions, std::string * err) {
    const size_t layers = (size_t) c.block_count;

    ggml_init_params ip = { layers * 2 * ggml_tensor_overhead() + 1024, NULL,  true };
    g->kv_ctx           = ggml_init(ip);
    if (!g->kv_ctx) {
        if (err) {
            *err = "ggml_init failed for the MM3 LM KV cache context";
        }
        return false;
    }
    g->kv_k.assign(layers, nullptr);
    g->kv_v.assign(layers, nullptr);
    mm3_lm_add_kv_tensors(g, (int64_t) c.key_length * (int64_t) c.head_count_kv, positions * MM3_LM_CFG_ROWS);
    return true;
}

static bool mm3_lm_prepare(const MM3Model & m, MM3LmGraph * g, int64_t n_ctx_needed, std::string * err) {
#ifdef MM3_ANE_VALIDATION
    if (m.ane_validation) {
        int64_t positions=0;
        if (!mm3_lm_bucket_for_context(n_ctx_needed,m.lm_cfg.context_length,&positions)) {
            if (err) *err="requested positions exceed the original LM context";
            return false;
        }
        if (positions!=mm3_ane_validation::context_positions(*m.ane_validation)) {
            if (err) *err="compile the ANE validation profile for the existing native cache bucket";
            return false;
        }
        g->n_ctx = positions;
        g->kv_pos = 0;
        return true;
    }
#endif
    if (!m.loaded) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const MM3LmConfig & c = m.lm_cfg;
    int64_t want = 0;
    if (!mm3_lm_validate(m, err) || !mm3_lm_kv_positions(c, n_ctx_needed, &want, err)) {
        return false;
    }

    const void * lt = (const void *) m.wctx_lm.buffer;
    const void * st = (const void *) m.wctx_synth.buffer;
    if (g->lm_token != lt || g->synth_token != st) {
        mm3_lm_free(g);
    }

    if (!g->backend_ref) {
        BackendPair bp = backend_init("MM3-LM");
        g->bp          = bp;
        g->backend     = bp.backend;
        g->cpu_backend = bp.cpu_backend;
        g->backend_ref = true;

        g->use_flash_attn = mm3_lm_use_flash_attn(bp);
        g->lm_token    = lt;
        g->synth_token = st;
    }

    if (g->n_ctx >= want) {
        return true;
    }

    mm3_lm_free_slot(&g->prefill);
    mm3_lm_free_slot(&g->decode);
    mm3_lm_free_kv(g);

    if (!mm3_lm_define_kv(c, g, want, err)) {
        return false;
    }
    g->kv_buf = ggml_backend_alloc_ctx_tensors(g->kv_ctx, g->backend);
    if (!g->kv_buf) {
        ggml_free(g->kv_ctx);
        g->kv_ctx = nullptr;
        g->kv_k.clear();
        g->kv_v.clear();
        if (err) {
            *err = "backend buffer allocation failed for the MM3 LM KV cache (" +
                   std::to_string((long long) want) + " positions, out of VRAM?)";
        }
        return false;
    }

    ggml_backend_buffer_clear(g->kv_buf, 0);

    g->n_ctx    = want;
    g->kv_bytes = ggml_backend_buffer_get_size(g->kv_buf);
    g->kv_pos   = 0;

    fprintf(stderr, "[MM3-LM] KV cache: %lld positions x %u layers x 2 rows = %.2f GB (%.0f kB/position), flash=%s\n",
            (long long) want, c.block_count, (double) g->kv_bytes / (1024.0 * 1024.0 * 1024.0),
            (double) g->kv_bytes / (double) want / 1024.0, g->use_flash_attn ? "yes" : "no");
    return true;
}

static void mm3_lm_reset(MM3LmGraph * g) {
    g->kv_pos = 0;
}

static void mm3_lm_fill_mask(MM3LmGraph * g, int64_t T, int64_t n_kv_pad) {
    g->mask_host.resize((size_t) (n_kv_pad * T));
    for (int64_t i = 0; i < T; i++) {
        const int64_t abs = g->kv_pos + i;
        for (int64_t j = 0; j < n_kv_pad; j++) {
            g->mask_host[(size_t) (i * n_kv_pad + j)] = ggml_fp32_to_fp16(j <= abs ? 0.0f : -INFINITY);
        }
    }
}

// Positions and cache rows for the flat T * B token axis: column b * T + i is
// token i of CFG row b, at position kv_pos + i and cache row b * n_ctx + that.
static void mm3_lm_fill_positions(MM3LmGraph * g, int64_t T) {
    const int64_t columns = T * MM3_LM_CFG_ROWS;
    g->pos_host.resize((size_t) columns);
    g->rows_host.resize((size_t) columns);
    for (int64_t column = 0; column < columns; column++) {
        const int64_t position        = g->kv_pos + column % T;
        g->pos_host[(size_t) column]  = (int32_t) position;
        g->rows_host[(size_t) column] = (column / T) * g->n_ctx + position;
    }
}

static void mm3_lm_upload_step(MM3LmGraph * g, MM3LmSlot * s, int64_t T, int64_t n_kv_pad) {
    mm3_lm_fill_positions(g, T);
    mm3_lm_fill_mask(g, T, n_kv_pad);
    ggml_backend_tensor_set(s->in_pos, g->pos_host.data(), 0, g->pos_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(s->in_rows, g->rows_host.data(), 0, g->rows_host.size() * sizeof(int64_t));
    ggml_backend_tensor_set(s->in_mask, g->mask_host.data(), 0, (size_t) (n_kv_pad * T) * sizeof(uint16_t));
}

static void mm3_lm_read_outputs(const MM3LmConfig & c, const MM3LmSlot & s, float * out_hidden, float * out_logits,
                                float * out_feedback) {
    const size_t H  = (size_t) c.embedding_length;
    const size_t CV = (size_t) tts_cpp::minimax::detail::compact_head_row_count((int64_t) c.semantic_vocab_size);
    if (out_hidden) {
        ggml_backend_tensor_get(s.out_hidden, out_hidden, 0, H * MM3_LM_CFG_ROWS * sizeof(float));
    }
    if (out_logits) {
        ggml_backend_tensor_get(s.out_logits, out_logits, 0, CV * MM3_LM_CFG_ROWS * sizeof(float));
    }
    if (out_feedback && s.out_feedback) {
        ggml_backend_tensor_get(s.out_feedback, out_feedback, 0, H * sizeof(float));
    }
}

static bool mm3_lm_prefill(const MM3Model & m, MM3LmGraph * g, const int32_t * ids_cond, const int32_t * ids_uncond,
                           int64_t n_prompt, float * out_hidden, float * out_logits, std::string * err) {
#ifdef MM3_ANE_VALIDATION
    if (m.ane_validation) {
        mm3_ane_validation::reset(*m.ane_validation);
        std::vector<float> input;
        for (int64_t position=0; position<n_prompt; ++position) {
            mm3_ane_validation::lm_embedding(*m.ane_validation, ids_cond[position], ids_uncond[position], input);
            const auto started=std::chrono::steady_clock::now();
            mm3_ane_validation::lm_step(*m.ane_validation, input.data(), position, out_hidden, out_logits);
            fprintf(stderr,"[MM3-ANE] Prefill %lld/%lld: %.3f ms\n",(long long)(position+1),(long long)n_prompt,
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
        }
        g->kv_pos = n_prompt;
        return true;
    }
#endif
    const MM3LmConfig & c = m.lm_cfg;
    if (n_prompt <= 0) {
        if (err) {
            *err = "the prompt is empty";
        }
        return false;
    }
    if (n_prompt > g->n_ctx) {
        if (err) {
            *err = "prompt of " + std::to_string((long long) n_prompt) + " tokens exceeds the KV cache (" +
                   std::to_string((long long) g->n_ctx) + ")";
        }
        return false;
    }

    int64_t n_kv_pad = 0;
    if (!mm3_lm_bucket_for_context(n_prompt, g->n_ctx, &n_kv_pad)) {
        if (err) {
            *err = "prompt positions cannot be represented by the KV cache";
        }
        return false;
    }
    if (!g->prefill.graph || g->prefill.T != n_prompt || g->prefill.n_kv_pad != n_kv_pad) {
        mm3_lm_free_slot(&g->prefill);
        if (!g->prefill.sched) {
            g->prefill.sched = backend_sched_new(g->bp, MM3_LM_MAX_NODES * 2);
        }
        if (!mm3_lm_build_slot(m, g, &g->prefill, n_prompt, n_kv_pad,  false, err)) {
            return false;
        }
        fprintf(stderr, "[MM3-LM] Prefill graph: T=%lld kv=%lld, %d nodes, %.1f MB compute\n", (long long) n_prompt,
                (long long) n_kv_pad, g->prefill.n_nodes, (double) g->prefill.compute_bytes / (1024.0 * 1024.0));
    }

    g->kv_pos = 0;

    g->ids_host.resize((size_t) (n_prompt * MM3_LM_CFG_ROWS));
    memcpy(g->ids_host.data(), ids_cond, (size_t) n_prompt * sizeof(int32_t));
    memcpy(g->ids_host.data() + n_prompt, ids_uncond, (size_t) n_prompt * sizeof(int32_t));
    ggml_backend_tensor_set(g->prefill.in_ids, g->ids_host.data(), 0,
                            (size_t) (n_prompt * MM3_LM_CFG_ROWS) * sizeof(int32_t));
    mm3_lm_upload_step(g, &g->prefill, n_prompt, n_kv_pad);

    if (ggml_backend_sched_graph_compute(g->prefill.sched, g->prefill.graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "MM3 LM prefill graph compute failed";
        }
        return false;
    }
    mm3_lm_read_outputs(c, g->prefill, out_hidden, out_logits, nullptr);
    g->kv_pos = n_prompt;
    return true;
}

static bool mm3_lm_decode(const MM3Model & m, MM3LmGraph * g, int32_t sem_token_id, const int32_t * acoustic_rows,
                          float * out_hidden, float * out_logits, float * out_feedback, std::string * err) {
#ifdef MM3_ANE_VALIDATION
    if (m.ane_validation) {
        const int64_t H = m.lm_cfg.embedding_length;
        std::vector<float> input(2*H);
        mm3_ane_validation::lm_feedback(*m.ane_validation, sem_token_id, acoustic_rows, input.data());
        std::copy_n(input.data(), H, input.data()+H);
        if (out_feedback) std::copy_n(input.data(), H, out_feedback);
        mm3_ane_validation::lm_step(*m.ane_validation, input.data(), g->kv_pos, out_hidden, out_logits);
        ++g->kv_pos;
        return true;
    }
#endif
    const MM3LmConfig & c  = m.lm_cfg;
    const int64_t       NC = (int64_t) c.num_codebooks - 1;

    if (!mm3_lm_positions_fit(g->n_ctx, g->kv_pos, 1)) {
        if (err) {
            *err = "the KV cache is full at " + std::to_string((long long) g->n_ctx) + " positions";
        }
        return false;
    }

    const int64_t next_position = g->kv_pos + 1;
    int64_t       n_kv_pad      = 0;
    if (!mm3_lm_bucket_for_context(next_position, g->n_ctx, &n_kv_pad)) {
        if (err) {
            *err = "decode positions cannot be represented by the KV cache";
        }
        return false;
    }
    if (!g->decode.graph || g->decode.n_kv_pad != n_kv_pad) {
        mm3_lm_free_slot(&g->decode);
        if (!g->decode.sched) {
            g->decode.sched = backend_sched_new(g->bp, MM3_LM_MAX_NODES * 2);
        }
        if (!mm3_lm_build_slot(m, g, &g->decode, 1, n_kv_pad,  true, err)) {
            return false;
        }
        fprintf(stderr, "[MM3-LM] Decode graph: kv=%lld, %d nodes, %.1f MB compute\n", (long long) n_kv_pad,
                g->decode.n_nodes, (double) g->decode.compute_bytes / (1024.0 * 1024.0));
    }

    g->ids_host.resize((size_t) (NC + 1));
    g->ids_host[0] = sem_token_id;
    memcpy(g->ids_host.data() + 1, acoustic_rows, (size_t) NC * sizeof(int32_t));
    ggml_backend_tensor_set(g->decode.in_ids, g->ids_host.data(), 0, (size_t) (NC + 1) * sizeof(int32_t));
    mm3_lm_upload_step(g, &g->decode, 1, n_kv_pad);

    if (ggml_backend_sched_graph_compute(g->decode.sched, g->decode.graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "MM3 LM decode graph compute failed";
        }
        return false;
    }
    mm3_lm_read_outputs(c, g->decode, out_hidden, out_logits, out_feedback);
    g->kv_pos++;
    return true;
}
