#pragma once

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"
#include "logic.h"
#include "mm3-flash-attn.h"
#include "mm3-flow-runtime.h"
#include "mm3-linear.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define MM3_DIT_MAX_NODES 4096

#define MM3_DIT_MAX_FRAMES 4096

#define MM3_DIT_CONDITIONED_GATE 1.0f

#define MM3_DIT_UNCONDITIONED_GATE 0.0f

struct MM3DitGraph {

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    bool                 use_flash_attn = false;

    const void *         weights_token = nullptr;

    std::vector<float>   fourier_w;

    ggml_context * gctx    = nullptr;
    uint8_t *      gbuf    = nullptr;
    ggml_cgraph *  graph   = nullptr;
    ggml_tensor *  in_lat  = nullptr;
    ggml_tensor *  in_cond = nullptr;
    ggml_tensor *  in_temb = nullptr;
    ggml_tensor *  in_gate = nullptr;
    ggml_tensor *  in_pos  = nullptr;
    ggml_tensor *  output  = nullptr;
    int64_t        graph_L = 0;
    int64_t        graph_branches = 0;
    size_t         compute_bytes = 0;
    int            n_nodes       = 0;
};

// One classifier-free-guidance branch of a DiT forward: the condition gate it
// runs under and where its velocity is written.
struct MM3DitBranch {
    float   gate     = MM3_DIT_CONDITIONED_GATE;
    float * velocity = nullptr;
};

struct MM3FlowStats {
    int    steps         = 0;
    int    forwards      = 0;
    double total_ms      = 0.0;
    double forward_ms    = 0.0;
    double first_ms      = 0.0;
    double last_ms       = 0.0;
    size_t compute_bytes = 0;
};

static bool mm3_dit_require_f32(const ggml_tensor * t, std::string * err, const char * what) {
    if (!t) {
        if (err) {
            *err = std::string("DiT tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("DiT tensor '") + what + "' is not F32 (type " + std::to_string((int) t->type) +
                   "); the layout contract pins it to F32";
        }
        return false;
    }
    return true;
}

static bool mm3_dit_readback_f32(const ggml_tensor * t, std::vector<float> * out, std::string * err,
                                 const char * what) {
    if (!mm3_dit_require_f32(t, err, what)) {
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

static void mm3_dit_free_graph(MM3DitGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx          = nullptr;
    g->gbuf          = nullptr;
    g->graph         = nullptr;
    g->in_lat        = nullptr;
    g->in_cond       = nullptr;
    g->in_temb       = nullptr;
    g->in_gate       = nullptr;
    g->in_pos        = nullptr;
    g->output        = nullptr;
    g->graph_L       = 0;
    g->graph_branches = 0;
}

static void mm3_dit_free(MM3DitGraph * g) {
    mm3_dit_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    g->fourier_w.clear();
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static bool mm3_dit_validate(const MM3Model & m, std::string * err) {
    const MM3DitConfig & c = m.synth_cfg.dit;
    if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 || c.head_dim == 0) {
        if (err) {
            *err = "DiT config is empty — mm3.dit.* KVs missing from the synth GGUF";
        }
        return false;
    }
    if (c.rope_type != "neox") {
        if (err) {
            *err = "mm3.dit.rope_type is '" + c.rope_type + "', this port only implements 'neox'";
        }
        return false;
    }
    if (c.glu_order != "value_gate") {
        if (err) {
            *err = "mm3.dit.glu_order is '" + c.glu_order + "', this port only implements 'value_gate'";
        }
        return false;
    }
    const ggml_tensor * fourier = m.synth.dit.time_fourier;
    if (!mm3_dit_require_f32(fourier, err, "dit.time_fourier.weight")) {
        return false;
    }
    if (ggml_nelements(fourier) * 2 != (int64_t) c.fourier_dim) {
        if (err) {
            *err = "dit.time_fourier.weight has " + std::to_string(ggml_nelements(fourier)) +
                   " elements, expected fourier_dim/2 = " + std::to_string(c.fourier_dim / 2);
        }
        return false;
    }
    return true;
}

static bool mm3_dit_use_flash_attn(bool has_gpu) {
    return mm3_use_flash_attn(has_gpu, /*default_on=*/true, "MM3_DIT_NO_FLASH", nullptr);
}

static bool mm3_dit_prepare(const MM3Model & m, MM3DitGraph * g, std::string * err) {
#ifdef MM3_COREML_RUNTIME_ENABLED
    if (m.ane_validation) return true;
#endif
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
    mm3_dit_free(g);

    if (!mm3_dit_validate(m, err)) {
        return false;
    }
    const MM3DitConfig & c = m.synth_cfg.dit;

    BackendPair bp = backend_init("MM3-DiT");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    g->use_flash_attn = mm3_dit_use_flash_attn(bp.has_gpu);

    std::string e;
    if (!mm3_dit_readback_f32(m.synth.dit.time_fourier, &g->fourier_w, &e, "dit.time_fourier.weight")) {
        if (err) {
            *err = e;
        }
        mm3_dit_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, MM3_DIT_MAX_NODES * 2);
    g->weights_token = token;

    fprintf(stderr, "[MM3-DiT] Prepared: %u blocks, %u heads x %u, rope %u/%.0f neox, flash_attn=%s\n",
            c.block_count, c.head_count, c.head_dim, c.rope_dim, (double) c.rope_theta,
            g->use_flash_attn ? "yes" : "no");
    return true;
}

static ggml_tensor * mm3_dit_ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * n = ggml_norm(ctx, x, eps);
    n               = ggml_mul(ctx, n, w);
    return ggml_add(ctx, n, b);
}

static ggml_tensor * mm3_dit_attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                      float scale) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores               = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static ggml_tensor * mm3_dit_rope(ggml_context * ctx, const MM3DitConfig & c, ggml_tensor * x,
                                  ggml_tensor * positions) {
    return ggml_rope_ext(ctx, x, positions, NULL, (int) c.rope_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f,
                         1.0f, 0.0f, 0.0f);
}

static ggml_tensor * mm3_dit_block(ggml_context * ctx, const MM3DitGraph & g, const MM3DitConfig & c,
                                   const MM3DitBlock & w, ggml_tensor * h, ggml_tensor * positions) {
    const int64_t E  = (int64_t) c.embedding_length;
    const int64_t D  = (int64_t) c.head_dim;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t S  = h->ne[1];
    const int64_t B  = h->ne[2];

    ggml_tensor * n   = mm3_dit_ln(ctx, h, w.attn_norm_w, w.attn_norm_b, c.layer_norm_eps);
    ggml_tensor * qkv = mm3_linear(ctx, w.attn_qkv, n);

    ggml_tensor * q = mm3_dit_rope(ctx, c, mm3_head_view(ctx, qkv, D, Nh, 0), positions);
    ggml_tensor * k = mm3_dit_rope(ctx, c, mm3_head_view(ctx, qkv, D, Nh, E), positions);
    ggml_tensor * v = mm3_head_view(ctx, qkv, D, Nh, 2 * E);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn;
    if (g.use_flash_attn) {

        attn = ggml_flash_attn_ext(ctx, q, ggml_cast(ctx, k, GGML_TYPE_F16), ggml_cast(ctx, v, GGML_TYPE_F16), NULL,
                                   scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = mm3_dit_attn_f32(ctx, q, k, v, scale);
    }
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, B);
    h    = ggml_add(ctx, h, mm3_linear(ctx, w.attn_output, attn));

    ggml_tensor * n2 = mm3_dit_ln(ctx, h, w.ffn_norm_w, w.ffn_norm_b, c.layer_norm_eps);
    ggml_tensor * f  = mm3_linear_bias(ctx, w.ffn_in_w, w.ffn_in_b, n2);

    // value_gate order: the value half comes first, so the swapped SwiGLU
    // computes value * silu(gate) without splitting the halves apart.
    ggml_tensor * y = ggml_swiglu_swapped(ctx, f);

    y = mm3_linear_bias(ctx, w.ffn_out_w, w.ffn_out_b, y);
    return ggml_add(ctx, h, y);
}

static ggml_tensor * mm3_dit_build(ggml_context * ctx, const MM3Model & m, const MM3DitGraph & g) {
    const MM3DitConfig &  c = m.synth_cfg.dit;
    const MM3DitWeights & w = m.synth.dit;

    const int64_t E  = (int64_t) c.embedding_length;
    const int64_t IC = (int64_t) c.in_channels;
    const int64_t CC = (int64_t) c.concat_channels;
    const int64_t CD = (int64_t) c.condition_dim;
    const int64_t L  = g.in_lat->ne[0];
    const int64_t B  = g.in_gate->ne[2];

    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, g.in_lat));
    x               = ggml_repeat_4d(ctx, x, IC, L, B, 1);

    ggml_tensor * zeros = ggml_scale(ctx, x, 0.0f);

    ggml_tensor * cond = ggml_mul(ctx, ggml_repeat_4d(ctx, g.in_cond, CD, L, B, 1), g.in_gate);

    ggml_tensor * full = ggml_concat(ctx, ggml_concat(ctx, x, zeros, 0), cond, 0);

    ggml_tensor * pre_w = ggml_reshape_2d(ctx, w.preprocess_conv, CC, CC);
    full                = ggml_add(ctx, mm3_linear(ctx, pre_w, full), full);

    ggml_tensor * h = mm3_linear(ctx, w.proj_in, full);

    ggml_tensor * temb = ggml_add(ctx, ggml_mul_mat(ctx, w.time_embd_w[0], g.in_temb), w.time_embd_b[0]);
    temb               = ggml_silu(ctx, temb);
    temb               = ggml_add(ctx, ggml_mul_mat(ctx, w.time_embd_w[1], temb), w.time_embd_b[1]);
    temb               = ggml_repeat_4d(ctx, temb, E, 1, B, 1);

    h = ggml_concat(ctx, temb, h, 1);

    for (size_t i = 0; i < w.blk.size(); i++) {
        h = mm3_dit_block(ctx, g, c, w.blk[i], h, g.in_pos);
    }

    // The output projections act per column, so they also run on the prepended
    // timestep token, which is dropped afterwards where the tensor is narrow.
    ggml_tensor * out = mm3_linear(ctx, w.proj_out, h);

    ggml_tensor * post_w = ggml_reshape_2d(ctx, w.postprocess_conv, IC, IC);
    out                  = ggml_add(ctx, mm3_linear(ctx, post_w, out), out);
    out                  = ggml_view_3d(ctx, out, IC, L, B, out->nb[1], out->nb[2], out->nb[1]);

    return ggml_cont(ctx, ggml_transpose(ctx, out));
}

static size_t mm3_dit_graph_context_bytes() {
    return ggml_tensor_overhead() * (MM3_DIT_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_DIT_MAX_NODES, false);
}

static ggml_context * mm3_dit_graph_context(MM3DitGraph * g, std::string * err) {
    const size_t ctx_bytes = mm3_dit_graph_context_bytes();
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the DiT graph context";
        }
        return nullptr;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf,  true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the DiT graph context";
        }
        return nullptr;
    }
    return ctx;
}

static ggml_cgraph * mm3_dit_define_graph(ggml_context * ctx, const MM3Model & m, MM3DitGraph * g, int64_t L,
                                          int64_t branches) {
    const MM3DitConfig & c = m.synth_cfg.dit;

    g->in_lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, (int64_t) c.in_channels);
    ggml_set_name(g->in_lat, "mm3_dit_latents");
    ggml_set_input(g->in_lat);

    g->in_cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.condition_dim, L);
    ggml_set_name(g->in_cond, "mm3_dit_cond");
    ggml_set_input(g->in_cond);

    g->in_temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.fourier_dim, 1);
    ggml_set_name(g->in_temb, "mm3_dit_fourier");
    ggml_set_input(g->in_temb);

    g->in_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, branches);
    ggml_set_name(g->in_gate, "mm3_dit_cond_gate");
    ggml_set_input(g->in_gate);

    g->in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L + 1);
    ggml_set_name(g->in_pos, "mm3_dit_positions");
    ggml_set_input(g->in_pos);

    g->output = mm3_dit_build(ctx, m, *g);
    ggml_set_name(g->output, "mm3_dit_velocity");
    ggml_set_output(g->output);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, MM3_DIT_MAX_NODES, false);
    ggml_build_forward_expand(graph, g->output);
    return graph;
}

static bool mm3_dit_ensure_graph(const MM3Model & m, MM3DitGraph * g, int64_t L, int64_t branches,
                                 std::string * err) {
    if (g->graph && g->graph_L == L && g->graph_branches == branches) {
        return true;
    }
    mm3_dit_free_graph(g);

    ggml_context * ctx = mm3_dit_graph_context(g, err);
    if (!ctx) {
        return false;
    }
    g->graph = mm3_dit_define_graph(ctx, m, g, L, branches);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "DiT graph allocation failed (out of VRAM?) for L=" + std::to_string(L) + " x " +
                   std::to_string(branches) + " branches";
        }
        return false;
    }

    g->gctx           = ctx;
    g->graph_L        = L;
    g->graph_branches = branches;
    g->n_nodes        = ggml_graph_n_nodes(g->graph);
    g->compute_bytes  = ggml_backend_sched_get_buffer_size(g->sched, g->backend);

    fprintf(stderr,
            "[MM3-DiT] Graph: L=%lld (S=%lld) x %lld branches, %d nodes, %d splits, compute buffer %.0f MB\n",
            (long long) L, (long long) (L + 1), (long long) branches, g->n_nodes,
            ggml_backend_sched_get_n_splits(g->sched), (double) g->compute_bytes / (1024.0 * 1024.0));
    return true;
}

static MM3DitGraph g_mm3_dit;

static void mm3_dit_upload_gates(MM3DitGraph * g, const std::vector<MM3DitBranch> & branches) {
    std::vector<float> gates(branches.size());
    for (size_t i = 0; i < branches.size(); i++) {
        gates[i] = branches[i].gate;
    }
    ggml_backend_tensor_set(g->in_gate, gates.data(), 0, gates.size() * sizeof(float));
}

static bool mm3_dit_read_branches(const MM3Model & m, MM3DitGraph * g, const std::vector<MM3DitBranch> & branches,
                                  int64_t L, std::string * err) {
    const size_t output_count =
        static_cast<size_t>(m.synth_cfg.dit.in_channels) * static_cast<size_t>(L);
    for (size_t i = 0; i < branches.size(); i++) {
        const size_t        offset   = i * output_count * sizeof(float);
        const MM3DitReadback readback = [g, offset](float * destination, size_t bytes, std::string *) {
            ggml_backend_tensor_get(g->output, destination, offset, bytes);
            return true;
        };
        if (!mm3_read_dit_output(branches[i].velocity, output_count, readback, err)) {
            return false;
        }
    }
    return true;
}

// Runs every CFG branch in one batched forward; the branches share the
// latents, timestep and condition and differ only in their condition gate.
static bool mm3_dit_run(const MM3Model & m, MM3DitGraph * g, const float * latents, const float * cond,
                        const std::vector<MM3DitBranch> & branches, float t, int64_t L, std::string * err) {
#ifdef MM3_COREML_RUNTIME_ENABLED
    if (m.ane_validation) {
        mm3_ane_validation::dit(*m.ane_validation, latents, cond, branches, t, L);
        return true;
    }
#endif
    if (branches.empty()) {
        if (err) {
            *err = "mm3_dit_run needs at least one CFG branch";
        }
        return false;
    }
    if (!mm3_dit_ensure_graph(m, g, L, static_cast<int64_t>(branches.size()), err)) {
        return false;
    }

    const size_t       H = g->fourier_w.size();
    std::vector<float> temb(2 * H);
    for (size_t i = 0; i < H; i++) {
        const double a = 2.0 * 3.14159265358979323846 * (double) t * (double) g->fourier_w[i];
        temb[i]        = (float) std::cos(a);
        temb[H + i]    = (float) std::sin(a);
    }
    ggml_backend_tensor_set(g->in_temb, temb.data(), 0, temb.size() * sizeof(float));

    std::vector<int32_t> pos((size_t) (L + 1));
    for (int64_t i = 0; i <= L; i++) {
        pos[(size_t) i] = (int32_t) i;
    }
    ggml_backend_tensor_set(g->in_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

    if (!cond) {
        if (err) {
            *err = "mm3_dit_run needs the window condition on every call: the graph "
                   "allocator recycles input blocks between computes, so uploads do "
                   "not survive a forward";
        }
        return false;
    }
    mm3_dit_upload_gates(g, branches);
    ggml_backend_tensor_set(g->in_lat, latents, 0, ggml_nbytes(g->in_lat));
    ggml_backend_tensor_set(g->in_cond, cond, 0, ggml_nbytes(g->in_cond));

    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "DiT graph compute failed";
        }
        return false;
    }
    return mm3_dit_read_branches(m, g, branches, L, err);
}

static bool mm3_dit_run_cfg(const MM3Model & m, MM3DitGraph * g, const float * latents, const float * cond, float t,
                            int64_t L, float * conditional, float * unconditional, std::string * err) {
    const std::vector<MM3DitBranch> branches = {
        { MM3_DIT_CONDITIONED_GATE, conditional },
        { MM3_DIT_UNCONDITIONED_GATE, unconditional },
    };
    return mm3_dit_run(m, g, latents, cond, branches, t, L, err);
}

static bool mm3_dit_forward(const MM3Model & m, const float * latents, const float * cond, int64_t L, float t,
                            std::vector<float> & out_velocity, std::string * err = nullptr) {
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_DIT_MAX_FRAMES);
        }
        return false;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, err)) {
        return false;
    }
    out_velocity.assign((size_t) ((int64_t) m.synth_cfg.dit.in_channels * L), 0.0f);
    const std::vector<MM3DitBranch> branches = { { MM3_DIT_CONDITIONED_GATE, out_velocity.data() } };
    return mm3_dit_run(m, &g_mm3_dit, latents, cond, branches, t, L, err);
}

static void mm3_flow_sigmas(int steps, std::vector<float> * sigmas, std::vector<float> * timesteps) {
    tts_cpp::minimax::detail::flow_schedule(steps, *sigmas, *timesteps);
}

static bool mm3_flow_sample(const MM3Model & m, const float * noise, const float * cond, int64_t L, int steps,
                            float cfg_scale, std::vector<float> & out_latents, MM3FlowStats * stats = nullptr,
                            std::string * err = nullptr) {
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_DIT_MAX_FRAMES);
        }
        return false;
    }
    if (steps <= 0 || steps > 1000) {
        if (err) {
            *err = "steps must be in 1..1000";
        }
        return false;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, err)) {
        return false;
    }

    const int64_t N = (int64_t) m.synth_cfg.dit.in_channels * L;
    out_latents.assign((size_t) N, 0.0f);
    memcpy(out_latents.data(), noise, (size_t) N * sizeof(float));

    std::vector<float> sigmas, timesteps;
    mm3_flow_sigmas(steps, &sigmas, &timesteps);

    std::vector<float> pred_c((size_t) N);
    std::vector<float> pred_u((size_t) N);

    const auto t_all = std::chrono::steady_clock::now();
    double     fwd_ms = 0.0, first_ms = 0.0, last_ms = 0.0;

    for (int i = 0; i < steps; i++) {
        const float t = timesteps[(size_t) i];
        const auto  t0 = std::chrono::steady_clock::now();

        // Re-upload the condition every forward — see mm3_flow_sample_chunk.
        if (!mm3_dit_run_cfg(m, &g_mm3_dit, out_latents.data(), cond, t, L, pred_c.data(), pred_u.data(), err)) {
            return false;
        }

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fwd_ms += ms;
        if (i == 0) {
            first_ms = ms;
        }
        last_ms = ms;

        if (!mm3_integrate_flow_step(out_latents, pred_c, pred_u, sigmas,
                                     static_cast<size_t>(i), cfg_scale, err)) {
            return false;
        }
    }

    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all).count();
    if (stats) {
        stats->steps         = steps;
        stats->forwards      = steps * 2;
        stats->total_ms      = total_ms;
        stats->forward_ms    = fwd_ms;
        stats->first_ms      = first_ms;
        stats->last_ms       = last_ms;
        stats->compute_bytes = g_mm3_dit.compute_bytes;
    }
    fprintf(stderr,
            "[MM3-DiT] Flow sample: L=%lld, %d steps, cfg %.2f, t %.6f..%.6f -> %.0f ms "
            "(%.0f ms/step, %.0f ms/forward)\n",
            (long long) L, steps, (double) cfg_scale, (double) timesteps.front(), (double) timesteps.back(), total_ms,
            total_ms / (double) steps, fwd_ms / (double) (steps * 2));
    return true;
}
