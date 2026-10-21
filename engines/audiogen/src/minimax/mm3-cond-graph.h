#pragma once

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"
#include "logic.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define MM3_COND_MAX_NODES 256

#define MM3_COND_MAX_FRAMES 4096

struct MM3CondGraph {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            prep        = {};
    const void *         weights_token = nullptr;

    ggml_tensor * mix = nullptr;

    ggml_context * gctx   = nullptr;
    uint8_t *      gbuf   = nullptr;
    ggml_cgraph *  graph  = nullptr;
    ggml_tensor *  input  = nullptr;
    ggml_tensor *  in_idx = nullptr;
    ggml_tensor *  output = nullptr;
    int64_t        graph_F = 0;
    int64_t        graph_L = 0;
    size_t         compute_bytes = 0;
};

static int64_t mm3_cond_latent_length(const MM3CondConfig & c, int64_t frames) {
    tts_cpp::minimax::detail::ConditionRate rate;
    rate.input_sampling_rate = (int) c.input_sampling_rate;
    rate.input_hop_length = (int) c.input_hop_length;
    rate.output_sampling_rate = (int) c.output_sampling_rate;
    rate.output_hop_length = (int) c.output_hop_length;
    return tts_cpp::minimax::detail::condition_latent_length(rate, frames);
}

static void mm3_cond_free_graph(MM3CondGraph * g) {
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
    g->in_idx  = nullptr;
    g->output  = nullptr;
    g->graph_F = 0;
    g->graph_L = 0;
}

static void mm3_cond_free(MM3CondGraph * g) {
    mm3_cond_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->mix           = nullptr;
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static bool mm3_cond_require_f32(const ggml_tensor * t, std::string * err, const char * what) {
    if (!t) {
        if (err) {
            *err = std::string("condition-encoder tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("condition-encoder tensor '") + what + "' is not F32 (type " +
                   std::to_string((int) t->type) + "); the layout contract pins it to F32";
        }
        return false;
    }
    return true;
}

static bool mm3_cond_readback_f32(const ggml_tensor * t, std::vector<float> * out, std::string * err,
                                  const char * what) {
    if (!mm3_cond_require_f32(t, err, what)) {
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

static bool mm3_cond_validate(const MM3Model & m, std::string * err) {
    const MM3CondConfig & c = m.synth_cfg.cond;
    if (c.num_layers == 0 || c.hidden_dim == 0 || c.out_dim == 0) {
        if (err) {
            *err = "condition-encoder config is empty — mm3.cond.* KVs missing from the synth GGUF";
        }
        return false;
    }
    if (c.layer_mix != "softmax") {
        if (err) {
            *err = "mm3.cond.layer_mix is '" + c.layer_mix + "', this port only implements 'softmax'";
        }
        return false;
    }
    if (c.interpolation != "nearest") {
        if (err) {
            *err = "mm3.cond.interpolation is '" + c.interpolation + "', this port only implements 'nearest'";
        }
        return false;
    }
    if (c.kernel_size != 3 || c.padding != 1) {
        if (err) {
            *err = "mm3.cond kernel/padding is " + std::to_string(c.kernel_size) + "/" + std::to_string(c.padding) +
                   ", this port only implements 3/1";
        }
        return false;
    }
    return mm3_cond_require_f32(m.synth.cond.layer_logits, err, "cond.layer_logits") &&
           mm3_cond_require_f32(m.synth.cond.layer_scale, err, "cond.layer_scale");
}

static ggml_tensor * mm3_cond_mix_tensor(WeightCtx * prep, int64_t layers) {
    ggml_tensor * mix = ggml_new_tensor_2d(prep->ctx, GGML_TYPE_F32, layers, 1);
    ggml_set_name(mix, "cond.layer_mix");
    return mix;
}

static bool mm3_cond_prepare(const MM3Model & m, MM3CondGraph * g, std::string * err) {
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
    mm3_cond_free(g);

    if (!mm3_cond_validate(m, err)) {
        return false;
    }
    const MM3CondConfig &  c = m.synth_cfg.cond;
    const MM3CondWeights & w = m.synth.cond;

    BackendPair bp = backend_init("MM3-Cond");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    std::vector<float> logits, scale;
    std::string        e;
    if (!mm3_cond_readback_f32(w.layer_logits, &logits, &e, "cond.layer_logits") ||
        !mm3_cond_readback_f32(w.layer_scale, &scale, &e, "cond.layer_scale")) {
        if (err) {
            *err = e;
        }
        mm3_cond_free(g);
        return false;
    }
    if (logits.size() != (size_t) c.num_layers || scale.size() != 1) {
        if (err) {
            *err = "cond.layer_logits/layer_scale have unexpected element counts";
        }
        mm3_cond_free(g);
        return false;
    }

    double mx = -1e30;
    for (float v : logits) {
        mx = v > mx ? (double) v : mx;
    }
    double sum = 0.0;
    for (float v : logits) {
        sum += std::exp((double) v - mx);
    }
    auto mix = std::make_unique<float[]>(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        mix[i] = (float) (std::exp((double) logits[i] - mx) / sum * (double) scale[0]);
    }

    wctx_init(&g->prep, 1);
    g->mix = mm3_cond_mix_tensor(&g->prep, (int64_t) logits.size());
    wctx_queue_copy(&g->prep, g->mix, mix.get(), logits.size() * sizeof(float), 0);
    g->prep.staging.push_back(std::move(mix));
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the condition-encoder mix vector";
        }
        mm3_cond_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, MM3_COND_MAX_NODES * 2);
    g->weights_token = token;

    fprintf(stderr, "[MM3-Cond] Prepared: %u layers -> %u -> %u, conv k=%u p=%u, %s resample %u/%u Hz\n",
            c.num_layers, c.hidden_dim, c.out_dim, c.kernel_size, c.padding, c.interpolation.c_str(),
            c.input_sampling_rate, c.output_sampling_rate);
    return true;
}

static ggml_tensor * mm3_cond_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int pad) {
    ggml_tensor * col = ggml_im2col(ctx, w, x,  1,  0, pad, 0,  1, 0,  false,
                                    GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]),
                                   ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]));
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

static ggml_tensor * mm3_cond_build(ggml_context * ctx, const MM3Model & m, const MM3CondGraph & g) {
    const MM3CondConfig &  c = m.synth_cfg.cond;
    const MM3CondWeights & w = m.synth.cond;
    const int64_t          F = g.input->ne[2];

    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, g.input, 1, 0, 2, 3));
    ggml_tensor * y  = ggml_mul_mat(ctx, g.mix, xp);
    y                = ggml_reshape_2d(ctx, y, (int64_t) c.hidden_dim, F);

    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    y = mm3_cond_conv1d(ctx, w.proj_w, w.proj_b, y, (int) c.padding);

    return ggml_get_rows(ctx, y, g.in_idx);
}

static size_t mm3_cond_graph_context_bytes() {
    return ggml_tensor_overhead() * (MM3_COND_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_COND_MAX_NODES, false);
}

static ggml_context * mm3_cond_graph_context(MM3CondGraph * g, std::string * err) {
    const size_t ctx_bytes = mm3_cond_graph_context_bytes();
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the condition-encoder graph context";
        }
        return nullptr;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf,  true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the condition-encoder graph context";
        }
        return nullptr;
    }
    return ctx;
}

static ggml_cgraph * mm3_cond_define_graph(ggml_context * ctx, const MM3Model & m, MM3CondGraph * g, int64_t F) {
    const MM3CondConfig & c = m.synth_cfg.cond;
    const int64_t         L = mm3_cond_latent_length(c, F);

    g->input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t) c.hidden_dim, (int64_t) c.num_layers, F);
    ggml_set_name(g->input, "mm3_cond_in");
    ggml_set_input(g->input);

    g->in_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(g->in_idx, "mm3_cond_resample_idx");
    ggml_set_input(g->in_idx);

    g->output = mm3_cond_build(ctx, m, *g);
    ggml_set_name(g->output, "mm3_cond_out");
    ggml_set_output(g->output);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, MM3_COND_MAX_NODES, false);
    ggml_build_forward_expand(graph, g->output);
    return graph;
}

static bool mm3_cond_ensure_graph(const MM3Model & m, MM3CondGraph * g, int64_t F, std::string * err) {
    if (g->graph && g->graph_F == F) {
        return true;
    }
    mm3_cond_free_graph(g);

    ggml_context * ctx = mm3_cond_graph_context(g, err);
    if (!ctx) {
        return false;
    }
    g->graph = mm3_cond_define_graph(ctx, m, g, F);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "condition-encoder graph allocation failed (out of VRAM?) for F=" + std::to_string(F);
        }
        return false;
    }

    g->gctx          = ctx;
    g->graph_F       = F;
    g->graph_L       = mm3_cond_latent_length(m.synth_cfg.cond, F);
    g->compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);

    fprintf(stderr, "[MM3-Cond] Graph: F=%lld -> L=%lld, %d nodes, %d splits, compute buffer %.0f MB\n", (long long) F,
            (long long) g->graph_L, ggml_graph_n_nodes(g->graph), ggml_backend_sched_get_n_splits(g->sched),
            (double) g->compute_bytes / (1024.0 * 1024.0));
    return true;
}

static MM3CondGraph g_mm3_cond;

static bool mm3_cond_encode(const MM3Model & m, const float * hiddens, int64_t F, std::vector<float> & out,
                            int64_t * out_L, std::string * err = nullptr) {
    if (F <= 0 || F > MM3_COND_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_COND_MAX_FRAMES);
        }
        return false;
    }
    if (!mm3_cond_prepare(m, &g_mm3_cond, err)) {
        return false;
    }
    MM3CondGraph * g = &g_mm3_cond;
    if (m.coreml_conditioner) {
        const auto & c = m.synth_cfg.cond;
        std::vector<float> mix;
        if (!mm3_cond_readback_f32(g->mix, &mix, err, "cond.layer_mix")) return false;
        std::vector<float> mixed((size_t) F * c.hidden_dim, 0.0f);
        for (int64_t frame = 0; frame < F; ++frame) {
            float * dst = mixed.data() + frame * c.hidden_dim;
            for (uint32_t layer = 0; layer < c.num_layers; ++layer) {
                const float * src = hiddens + (frame * c.num_layers + layer) * c.hidden_dim;
                for (uint32_t channel = 0; channel < c.hidden_dim; ++channel) {
                    dst[channel] += mix[layer] * src[channel];
                }
            }
        }
        std::vector<float> projected;
        if (!mm3_coreml_conditioner_project(m.coreml_conditioner, mixed.data(), F, projected, err)) {
            return false;
        }
        const int64_t L = mm3_cond_latent_length(c, F);
        out.resize((size_t) L * c.out_dim);
        for (int64_t i = 0; i < L; ++i) {
            const int64_t source = std::min(i * F / L, F - 1);
            std::copy_n(projected.data() + source * c.out_dim, c.out_dim, out.data() + i * c.out_dim);
        }
        if (out_L) *out_L = L;
        return true;
    }
    if (!mm3_cond_ensure_graph(m, g, F, err)) {
        return false;
    }

    const int64_t L = g->graph_L;

    std::vector<int32_t> idx((size_t) L);
    for (int64_t i = 0; i < L; i++) {
        int64_t s = i * F / L;
        idx[(size_t) i] = (int32_t) (s < F ? s : F - 1);
    }
    ggml_backend_tensor_set(g->in_idx, idx.data(), 0, idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g->input, hiddens, 0, ggml_nbytes(g->input));

    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "condition-encoder graph compute failed";
        }
        return false;
    }

    out.assign((size_t) ((int64_t) m.synth_cfg.cond.out_dim * L), 0.0f);
    ggml_backend_tensor_get(g->output, out.data(), 0, out.size() * sizeof(float));
    if (out_L) {
        *out_L = L;
    }
    return true;
}
