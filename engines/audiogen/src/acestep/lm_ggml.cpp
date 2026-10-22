#include "lm_ggml.h"

#include "bpe_tokenizer.h"
#include "fit_measure.h"
#include "qwen3_block.h"  // shared Qwen3 loaders + builders + DitGGUF IO

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ACE-Step LM core. Port from acestep.cpp/src/qwen3-lm.h, simplified for CPU:
// single KV set, per-call graph build, F32 soft_max attention, tied LM head.
// The KV cache lives in a persistent f16 buffer written via set_rows.

namespace tts_cpp::acestep {

struct LMHeadChunk {
    ggml_tensor * weight = nullptr;
    int           row0   = 0;
    int           rows   = 0;
    ggml_tensor * logits = nullptr;
};

// Reused forward graph: the key fields fully determine the graph shape, and
// n_kv_pad moves in 256-row steps, so one graph serves ~256 decode steps.
struct LMGraphCache {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor *  t_ids = nullptr, *positions = nullptr, *kv_rows = nullptr,
                *  mask = nullptr, *lgt = nullptr, *embd = nullptr;
    std::vector<LMHeadChunk> head_logits;
    // key
    bool          batch    = false;
    int           S        = -1;   // tokens per stream (batch: always 1)
    int           n_kv_pad = -1;
    int           set0     = -1;   // kv set (batch: first set)
    int           n_batch  = -1;
    ggml_tensor * head     = nullptr;  // tied or compact lm head in the graph
    int           head_lo  = -1;
    int           head_hi  = -1;

    void release() {
        if (ga)  ggml_gallocr_free(ga);
        if (ctx) ggml_free(ctx);
        *this = LMGraphCache{};
    }
};

struct LMModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    // CPU map-in-place: verbatim weights backed by `gguf`'s mmap via `map_buf`
    // (see dit_gguf_cpu_map_buffer). `gguf` is kept open for the model's lifetime.
    DitGGUF               gguf;
    ggml_backend_buffer_t map_buf = nullptr;
    bool                  mapped  = false;
    size_t                mapped_bytes = 0;  // sum of mmapped weight nbytes

    LMConfig    cfg;
    Qwen3Config q3;

    ggml_tensor *           embed_tokens = nullptr;  // [H, V] (also tied lm_head)
    ggml_tensor *           final_norm   = nullptr;  // [H] F32
    std::vector<Qwen3Layer> layers;

    ggml_type                embed_type = GGML_TYPE_COUNT;
    const uint8_t *          host_embed = nullptr;
    std::vector<LMHeadChunk> head_chunks;

    // Load-time fused projections (q|k|v and gate|up rows concatenated): one
    // GEMM per group instead of three/two. Empty = unfused (CPU-mapped path,
    // non-Vulkan backends, or ACESTEP_LM_NO_FUSED).
    std::vector<ggml_tensor *> qkv_fused;
    std::vector<ggml_tensor *> gateup_fused;

    // Contiguous tied-head row range [offset, offset+count): suffix rows for
    // Phase-2 batched CFG, prefix rows for FSM-constrained Phase 1. Quantized
    // tensors cannot be safely sliced with ggml_view_2d on every GPU backend.
    ggml_context *        lm_head_ctx           = nullptr;
    ggml_backend_buffer_t lm_head_buf           = nullptr;
    ggml_tensor *         lm_head_partial       = nullptr;
    int                   lm_head_offset        = -1;
    int                   lm_head_count         = -1;
    int                   lm_head_failed_offset = -1;
    int                   lm_head_failed_count  = -1;

    // KV cache: n_sets independent caches, each per-layer [D, max_seq, Nkv] f16.
    // Indexed kv_k[set * n_layers + layer]. Set 0 is the default; CFG uses set 1
    // for the unconditional stream.
    ggml_context *             kv_ctx  = nullptr;
    ggml_backend_buffer_t      kv_buf  = nullptr;
    int                        n_sets  = 1;
    std::vector<ggml_tensor *> kv_k4;
    std::vector<ggml_tensor *> kv_v4;
    std::vector<ggml_tensor *> kv_k;
    std::vector<ggml_tensor *> kv_v;
    std::vector<int>           kv_pos;  // per set
    bool                       use_flash_attn = false;

    bool measuring = false;  // metadata-only load: weights/KV sized, never read

    LMGraphCache graph_cache;
};

static Qwen3Config to_q3(const LMConfig & c) {
    Qwen3Config q;
    q.hidden_size  = c.hidden_size;
    q.n_heads      = c.n_heads;
    q.n_kv_heads   = c.n_kv_heads;
    q.head_dim     = c.head_dim;
    q.n_layers     = c.n_layers;
    q.rope_theta   = c.rope_theta;
    q.rms_norm_eps = c.rms_norm_eps;
    q.is_causal    = true;
    // The ACE-Step LM carries massive activations (~1.9e6 by layer 2, measured),
    // which is far outside fp16 range. Without F32 precision the fp16-based GPU
    // matmul paths clamp them at 65504 and generation degenerates.
    q.prec         = GGML_PREC_F32;
    return q;
}

static bool lm_backend_supports_flash_attn(ggml_backend_t backend, const Qwen3Config & c) {
  return q3_backend_supports_flash_attention(backend, c);
}

static void lm_partial_head_clear(LMModel * m) {
    // A cached graph may reference the freed head tensor, and a reallocation
    // can reuse its address; drop the cache so pointer identity stays sound.
    m->graph_cache.release();
    if (m->lm_head_buf) ggml_backend_buffer_free(m->lm_head_buf);
    if (m->lm_head_ctx) ggml_free(m->lm_head_ctx);
    m->lm_head_buf     = nullptr;
    m->lm_head_ctx     = nullptr;
    m->lm_head_partial = nullptr;
    m->lm_head_offset  = -1;
    m->lm_head_count   = -1;
}

// The copied rows must byte-match the source range; sampling the edge rows
// catches a wrong-offset or unsupported-copy backend before the head is used.
static bool lm_verify_head_rows(LMModel * m, int offset, int count, size_t row_bytes) {
    std::vector<uint8_t> src_row(row_bytes);
    std::vector<uint8_t> dst_row(row_bytes);
    const int checks[2] = { 0, count - 1 };
    for (int k = 0; k < 2; k++) {
        const int r = checks[k];
        ggml_backend_tensor_get(m->embed_tokens, src_row.data(), (size_t) (offset + r) * row_bytes, row_bytes);
        ggml_backend_tensor_get(m->lm_head_partial, dst_row.data(), (size_t) r * row_bytes, row_bytes);
        if (memcmp(src_row.data(), dst_row.data(), row_bytes) != 0) return false;
    }
    return true;
}

// Device-side copy of the tied-embedding row range into the compact head. The
// range is one contiguous byte span, so a row-aligned view is a legal copy
// SOURCE (it never feeds MUL_MAT - that guard stands); backends without a
// device copy path bounce through the host inside ggml_backend_tensor_copy.
static bool lm_copy_head_rows_device(LMModel * m, int offset, int count, size_t row_bytes) {
    if (!m->embed_tokens->buffer || !m->embed_tokens->data) return false;
    ggml_tensor * view = ggml_view_2d(m->lm_head_ctx, m->embed_tokens, m->cfg.hidden_size, count,
                                      m->embed_tokens->nb[1], (size_t) offset * row_bytes);
    if (!view || ggml_backend_view_init(view) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_tensor_copy(view, m->lm_head_partial);
    return lm_verify_head_rows(m, offset, count, row_bytes);
}

// Copy tied-embedding rows [offset, offset+count) into a contiguous tensor. A
// direct ggml_view_2d is only byte-addressable for non-quantized rows on all
// backends; a compact Q4/Q8 view may otherwise feed wrong blocks to GPU MUL_MAT.
static bool lm_build_partial_head(LMModel * m, int offset, int count) {
    if (!m || offset < 0 || count <= 0 || offset + count > m->cfg.vocab_size) return false;
    if (m->lm_head_partial && m->lm_head_offset == offset && m->lm_head_count == count) return true;
    if (m->lm_head_failed_offset == offset && m->lm_head_failed_count == count) return false;

    lm_partial_head_clear(m);

    const int H = m->cfg.hidden_size;
    ggml_init_params hp{ ggml_tensor_overhead() * 3 + 16, nullptr, true };
    m->lm_head_ctx = ggml_init(hp);
    if (!m->lm_head_ctx) {
        m->lm_head_failed_offset = offset;
        m->lm_head_failed_count  = count;
        return false;
    }

    m->lm_head_partial = ggml_new_tensor_2d(m->lm_head_ctx, m->embed_tokens->type, H, count);
    ggml_set_name(m->lm_head_partial, "lm_head_partial");
    m->lm_head_buf = ggml_backend_alloc_ctx_tensors_from_buft(m->lm_head_ctx, dit_gguf_weight_buffer_type(m->backend));
    if (!m->lm_head_buf) {
        m->lm_head_failed_offset = offset;
        m->lm_head_failed_count  = count;
        lm_partial_head_clear(m);
        return false;
    }
    ggml_backend_buffer_set_usage(m->lm_head_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t row_bytes = ggml_row_size(m->embed_tokens->type, H);
    if (row_bytes == 0 || (size_t) count > SIZE_MAX / row_bytes ||
        (size_t) offset > SIZE_MAX / row_bytes) {
        m->lm_head_failed_offset = offset;
        m->lm_head_failed_count  = count;
        lm_partial_head_clear(m);
        return false;
    }
    const size_t nbytes = (size_t) count * row_bytes;
    const auto   t0     = std::chrono::steady_clock::now();
    if (!lm_copy_head_rows_device(m, offset, count, row_bytes)) {
        std::vector<uint8_t> tmp(nbytes);
        ggml_backend_tensor_get(m->embed_tokens, tmp.data(), (size_t) offset * row_bytes, nbytes);
        ggml_backend_tensor_set(m->lm_head_partial, tmp.data(), 0, nbytes);
    }
    m->lm_head_offset = offset;
    m->lm_head_count  = count;
    if (std::getenv("ACESTEP_LM_TIMING"))
        fprintf(stderr, "[lm-timing] partial-head build %.1f ms (%d rows, %.1f MB)\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), count,
                nbytes / 1048576.0);
    return true;
}

static bool lm_backend_runs_embedding_lookup(ggml_backend_t backend, ggml_tensor * embed) {
    ggml_init_params ip{ ggml_tensor_overhead() * 2, nullptr, /*no_alloc=*/true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * ids       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    const bool    supported = ggml_backend_supports_op(backend, ggml_get_rows(ctx, embed, ids));
    ggml_free(ctx);
    return supported;
}

static constexpr int LM_HEAD_CHUNK_MIN_ROWS = 32;
static constexpr int LM_LOGIT_RANGE_EDGES[] = { TOKEN_IM_END, AUDIO_CODE_BASE };

static bool lm_backend_runs_head(ggml_backend_t backend, const Qwen3Config & c, ggml_type type, int rows,
                                 int n_cols) {
    ggml_init_params ip{ ggml_tensor_overhead() * 3, nullptr, /*no_alloc=*/true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * w         = ggml_new_tensor_2d(ctx, type, c.hidden_size, rows);
    ggml_tensor * x         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden_size, n_cols);
    const bool    supported = ggml_backend_supports_op(backend, q3_linear(ctx, w, x, c.prec));
    ggml_free(ctx);
    return supported;
}

static int lm_largest_power_of_two_below(int n) {
    int p = 1;
    while (p * 2 < n) p *= 2;
    return p;
}

static int lm_largest_head_chunk_rows(ggml_backend_t backend, const Qwen3Config & c, ggml_type type, int vocab,
                                      int n_cols) {
    if (lm_backend_runs_head(backend, c, type, vocab, n_cols)) return vocab;
    for (int rows = lm_largest_power_of_two_below(vocab); rows >= LM_HEAD_CHUNK_MIN_ROWS; rows /= 2) {
        if (lm_backend_runs_head(backend, c, type, rows, n_cols)) return rows;
    }
    return 0;
}

static void lm_append_head_chunks(int row0, int row1, int max_rows, std::vector<LMHeadChunk> & chunks) {
    for (int r = row0; r < row1; r += max_rows) {
        LMHeadChunk chunk;
        chunk.row0 = r;
        chunk.rows = std::min(max_rows, row1 - r);
        chunks.push_back(chunk);
    }
}

static std::vector<int> lm_head_segment_edges(int vocab) {
    std::vector<int> edges = { 0 };
    for (int edge : LM_LOGIT_RANGE_EDGES) {
        if (edge > edges.back() && edge < vocab) edges.push_back(edge);
    }
    edges.push_back(vocab);
    return edges;
}

static std::vector<LMHeadChunk> lm_plan_head_chunks(int vocab, int max_rows) {
    const std::vector<int>   edges = lm_head_segment_edges(vocab);
    std::vector<LMHeadChunk> chunks;
    for (size_t i = 0; i + 1 < edges.size(); i++) lm_append_head_chunks(edges[i], edges[i + 1], max_rows, chunks);
    return chunks;
}

static void lm_create_head_chunks(ggml_context * ctx, ggml_type type, int hidden, std::vector<LMHeadChunk> & chunks) {
    for (LMHeadChunk & chunk : chunks) {
        chunk.weight = ggml_new_tensor_2d(ctx, type, hidden, chunk.rows);
        ggml_set_name(chunk.weight, ("lm_head_rows_" + std::to_string(chunk.row0)).c_str());
    }
}

static void lm_load_head_chunks(const std::vector<LMHeadChunk> & chunks, const uint8_t * table, size_t row_bytes) {
    for (const LMHeadChunk & chunk : chunks) {
        ggml_backend_tensor_set(chunk.weight, table + (size_t) chunk.row0 * row_bytes, 0,
                                (size_t) chunk.rows * row_bytes);
    }
}

static void lm_dequantize_row(ggml_type type, const uint8_t * row, float * out, int n) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(out, row, (size_t) n * sizeof(float));
        return;
    }
    ggml_get_type_traits(type)->to_float(row, out, n);
}

static bool lm_lookup_embeddings(const LMModel * m, const int32_t * ids, int n, std::vector<float> & out) {
    const int    H         = m->cfg.hidden_size;
    const size_t row_bytes = ggml_row_size(m->embed_type, H);
    out.resize((size_t) n * H);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= m->cfg.vocab_size) return false;
        lm_dequantize_row(m->embed_type, m->host_embed + (size_t) ids[i] * row_bytes, out.data() + (size_t) i * H, H);
    }
    return true;
}

static ggml_tensor * lm_new_token_input(ggml_context * ctx, const LMModel * m, int n, ggml_tensor ** ids,
                                        ggml_tensor ** embd) {
    if (m->embed_tokens) {
        *ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(*ids);
        return ggml_get_rows(ctx, m->embed_tokens, *ids);
    }
    *embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m->cfg.hidden_size, n);
    ggml_set_input(*embd);
    return *embd;
}

static bool lm_set_token_input(const LMModel * m, ggml_tensor * ids, ggml_tensor * embd, const int32_t * token_ids,
                               int n) {
    if (ids) {
        ggml_backend_tensor_set(ids, token_ids, 0, (size_t) n * sizeof(int32_t));
        return true;
    }
    std::vector<float> rows;
    if (!lm_lookup_embeddings(m, token_ids, n, rows)) {
        fprintf(stderr, "[acestep-lm] token id outside the %d-row vocabulary\n", m->cfg.vocab_size);
        return false;
    }
    ggml_backend_tensor_set(embd, rows.data(), 0, rows.size() * sizeof(float));
    return true;
}

static std::vector<LMHeadChunk> lm_build_split_head(ggml_context * ctx, ggml_cgraph * gf, const LMModel * m,
                                                    ggml_tensor * hidden, int lo, int hi) {
    std::vector<LMHeadChunk> used;
    for (const LMHeadChunk & chunk : m->head_chunks) {
        if (chunk.row0 >= hi || chunk.row0 + chunk.rows <= lo) continue;
        LMHeadChunk projected = chunk;
        projected.logits      = q3_linear(ctx, chunk.weight, hidden, m->q3.prec);
        ggml_set_output(projected.logits);
        ggml_build_forward_expand(gf, projected.logits);
        used.push_back(projected);
    }
    return used;
}

static void lm_read_chunk_logits(const LMHeadChunk & chunk, int lo, int hi, int n_cols, float * out) {
    const int first = std::max(lo, chunk.row0);
    const int last  = std::min(hi, chunk.row0 + chunk.rows);
    for (int col = 0; col < n_cols; col++) {
        ggml_backend_tensor_get(chunk.logits, out + (size_t) col * (hi - lo) + (first - lo),
                                ((size_t) col * chunk.rows + (first - chunk.row0)) * sizeof(float),
                                (size_t) (last - first) * sizeof(float));
    }
}

static void lm_read_split_logits(const std::vector<LMHeadChunk> & chunks, int lo, int hi, int n_cols,
                                 std::vector<float> & out) {
    out.resize((size_t) (hi - lo) * n_cols);
    for (const LMHeadChunk & chunk : chunks) lm_read_chunk_logits(chunk, lo, hi, n_cols, out.data());
}

// Fused-layer creation: norms and o/down as usual, q|k|v and gate|up as single
// row-concatenated tensors. Returns false when the GGUF shapes/types cannot fuse.
static bool lm_create_layer_fused(ggml_context * ctx, const DitGGUF & g, const std::string & prefix, Qwen3Layer & ly,
                                  ggml_tensor ** qkv_out, ggml_tensor ** gateup_out) {
    ly.input_norm = q3_create_f32_like(ctx, g, prefix + ".input_layernorm.weight");
    ly.post_norm  = q3_create_f32_like(ctx, g, prefix + ".post_attention_layernorm.weight");
    ly.q_norm     = q3_create_f32_like(ctx, g, prefix + ".self_attn.q_norm.weight");
    ly.k_norm     = q3_create_f32_like(ctx, g, prefix + ".self_attn.k_norm.weight");
    ly.o_proj     = q3_create_like(ctx, g, prefix + ".self_attn.o_proj.weight");
    ly.down_proj  = q3_create_like(ctx, g, prefix + ".mlp.down_proj.weight");

    ggml_tensor * qm = dit_gmeta(g, prefix + ".self_attn.q_proj.weight");
    ggml_tensor * km = dit_gmeta(g, prefix + ".self_attn.k_proj.weight");
    ggml_tensor * vm = dit_gmeta(g, prefix + ".self_attn.v_proj.weight");
    ggml_tensor * gm = dit_gmeta(g, prefix + ".mlp.gate_proj.weight");
    ggml_tensor * um = dit_gmeta(g, prefix + ".mlp.up_proj.weight");
    if (!qm || !km || !vm || !gm || !um) return false;
    if (km->type != qm->type || vm->type != qm->type || um->type != gm->type) return false;
    if (km->ne[0] != qm->ne[0] || vm->ne[0] != qm->ne[0] || um->ne[0] != gm->ne[0]) return false;

    *qkv_out = ggml_new_tensor_2d(ctx, qm->type, qm->ne[0], qm->ne[1] + km->ne[1] + vm->ne[1]);
    ggml_set_name(*qkv_out, (prefix + ".self_attn.qkv_fused").c_str());
    *gateup_out = ggml_new_tensor_2d(ctx, gm->type, gm->ne[0], gm->ne[1] + um->ne[1]);
    ggml_set_name(*gateup_out, (prefix + ".mlp.gateup_fused").c_str());
    return true;
}

bool lm_load_row_block(ggml_tensor * dst, size_t & off, const DitGGUF & g, const std::string & name) {
    const void *  src = dit_gdata(g, name);
    ggml_tensor * mt  = dit_gmeta(g, name);
    if (!src || !mt) {
        fprintf(stderr, "[acestep-lm] cannot load %s\n", name.c_str());
        return false;
    }
    ggml_backend_tensor_set(dst, src, off, ggml_nbytes(mt));
    off += ggml_nbytes(mt);
    return true;
}

bool lm_load_layer_fused(const DitGGUF & g, const std::string & prefix, Qwen3Layer & ly, ggml_tensor * qkv,
                         ggml_tensor * gateup) {
    q3_load_f32(ly.input_norm, g, prefix + ".input_layernorm.weight");
    q3_load_f32(ly.post_norm, g, prefix + ".post_attention_layernorm.weight");
    q3_load_f32(ly.q_norm, g, prefix + ".self_attn.q_norm.weight");
    q3_load_f32(ly.k_norm, g, prefix + ".self_attn.k_norm.weight");
    q3_load_raw(ly.o_proj, g, prefix + ".self_attn.o_proj.weight");
    q3_load_raw(ly.down_proj, g, prefix + ".mlp.down_proj.weight");
    // A skipped block would byte-shift every later block in the fused tensor,
    // so a missing tensor must fail the whole layer, not fall through.
    size_t off = 0;
    bool   ok  = lm_load_row_block(qkv, off, g, prefix + ".self_attn.q_proj.weight") &&
              lm_load_row_block(qkv, off, g, prefix + ".self_attn.k_proj.weight") &&
              lm_load_row_block(qkv, off, g, prefix + ".self_attn.v_proj.weight");
    off = 0;
    return ok && lm_load_row_block(gateup, off, g, prefix + ".mlp.gate_proj.weight") &&
           lm_load_row_block(gateup, off, g, prefix + ".mlp.up_proj.weight");
}

// Shared body of lm_model_load and lm_model_load_metadata_only. When `measure`
// is non-null the load is metadata-only: the weight and KV allocations are
// sized into `measure` instead of performed and no tensor data is read.
static LMModel * lm_model_load_impl(const std::string & path, ggml_backend_t backend, int max_seq_len, bool verbose,
                                    int n_kv_sets, AcestepStageMeasure * measure, int forced_head_chunk_rows = 0) {
    DitGGUF g;
    if (!dit_gguf_open(g, path)) {
        fprintf(stderr, "[acestep-lm] failed to parse %s\n", path.c_str());
        return nullptr;
    }
    dit_gguf_require_hexagon_weights(g, path, backend, "acestep-lm");

    LMModel * m = new LMModel();
    m->backend  = backend;
    m->n_sets   = n_kv_sets > 0 ? n_kv_sets : 1;

    // Derive config from tensor shapes.
    ggml_tensor * embed = dit_gmeta(g, "model.embed_tokens.weight");
    if (!embed) {
        fprintf(stderr, "[acestep-lm] missing model.embed_tokens.weight\n");
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }
    LMConfig & c   = m->cfg;
    c.hidden_size  = (int) embed->ne[0];
    c.vocab_size   = (int) embed->ne[1];
    c.head_dim     = 128;  // Qwen3 fixed
    // count layers
    int L = 0;
    while (dit_gguf_has(g, "model.layers." + std::to_string(L) + ".input_layernorm.weight")) L++;
    c.n_layers = L;
    // head counts from projection shapes (ne1 = heads*head_dim)
    ggml_tensor * qw = dit_gmeta(g, "model.layers.0.self_attn.q_proj.weight");
    ggml_tensor * kw = dit_gmeta(g, "model.layers.0.self_attn.k_proj.weight");
    c.n_heads    = (qw ? (int) qw->ne[1] : 2048) / c.head_dim;
    c.n_kv_heads = (kw ? (int) kw->ne[1] : 1024) / c.head_dim;
    c.rope_theta   = 1000000.0f;
    c.rms_norm_eps = 1e-6f;
    c.max_seq_len  = max_seq_len > 0 ? max_seq_len : 4096;
    m->q3 = to_q3(c);

    if (c.n_layers <= 0 || c.hidden_size <= 0 || c.vocab_size <= 0 || c.n_heads <= 0 || c.n_kv_heads <= 0) {
        fprintf(stderr, "[acestep-lm] bad derived config (L=%d H=%d V=%d Nh=%d Nkv=%d)\n", c.n_layers, c.hidden_size,
                c.vocab_size, c.n_heads, c.n_kv_heads);
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }
    m->use_flash_attn = lm_backend_supports_flash_attn(backend, m->q3);
    m->embed_type     = embed->type;

    const int head_rows = forced_head_chunk_rows > 0
                              ? forced_head_chunk_rows
                              : lm_largest_head_chunk_rows(backend, m->q3, embed->type, c.vocab_size, m->n_sets);
    if (head_rows <= 0) {
        fprintf(stderr, "[acestep-lm] %s cannot project the %s LM head; place the LM on another backend\n",
                ggml_backend_name(backend), ggml_type_name(embed->type));
        dit_gguf_close(g);
        delete m;
        return nullptr;
    }
    const bool split_table = forced_head_chunk_rows > 0 || head_rows < c.vocab_size ||
                             !lm_backend_runs_embedding_lookup(backend, embed);
    if (split_table) m->head_chunks = lm_plan_head_chunks(c.vocab_size, head_rows);

    // CPU backend: map the quantised weights straight off the mmap (no dirty RAM).
    const bool            mapped  = dit_gguf_backend_maps_weights(backend);
    ggml_backend_buffer_t map_buf = mapped ? dit_gguf_cpu_map_buffer(g) : nullptr;

    // Allocate + load weights.
    const size_t n_tensors = (size_t) 2 + (size_t) c.n_layers * 11 + 8 + m->head_chunks.size();
    ggml_init_params ip{ ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc=*/true };
    m->weight_ctx = ggml_init(ip);
    ggml_context * ctx = m->weight_ctx;

    if (split_table) {
        lm_create_head_chunks(ctx, embed->type, c.hidden_size, m->head_chunks);
    } else {
        m->embed_tokens = q3_create_like(ctx, g, "model.embed_tokens.weight", map_buf);
    }
    m->final_norm   = q3_create_f32_like(ctx, g, "model.norm.weight");
    m->layers.resize(c.n_layers);
    // Fusion is Vulkan-only for now (validated there; other GPU backends keep
    // the split layout until their strided-view attention paths are verified).
    bool fuse = !mapped && std::strncmp(ggml_backend_name(backend), "Vulkan", 6) == 0 &&
                !std::getenv("ACESTEP_LM_NO_FUSED");
    if (fuse) {
        Qwen3Layer    probe;
        ggml_tensor * pq = nullptr, *pg = nullptr;
        ggml_init_params pp{ ggml_tensor_overhead() * 16, nullptr, true };
        ggml_context *   pctx = ggml_init(pp);
        fuse = pctx && lm_create_layer_fused(pctx, g, "model.layers.0", probe, &pq, &pg);
        if (pctx) ggml_free(pctx);
    }
    if (fuse) {
        m->qkv_fused.assign((size_t) c.n_layers, nullptr);
        m->gateup_fused.assign((size_t) c.n_layers, nullptr);
    }
    for (int i = 0; i < c.n_layers; i++) {
        const std::string prefix = "model.layers." + std::to_string(i);
        if (fuse) {
            if (!lm_create_layer_fused(ctx, g, prefix, m->layers[i], &m->qkv_fused[i], &m->gateup_fused[i])) {
                fprintf(stderr, "[acestep-lm] fused layer create failed (layer %d)\n", i);
                if (map_buf) ggml_backend_buffer_free(map_buf);
                ggml_free(ctx);
                dit_gguf_close(g);
                delete m;
                return nullptr;
            }
        } else {
            q3_create_layer(ctx, g, prefix, m->layers[i], map_buf);
        }
    }

    if (measure) {
        measure->weights_alloc_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            ctx, dit_gguf_weight_buffer_type(backend));
        m->measuring = true;
    } else {
        // NB: ggml_backend_alloc_ctx_tensors returns NULL if EVERY tensor is already
        // allocated (i.e. all mapped). Safe to treat as failure here because each
        // stage always has F32 norms that need real allocation; revisit this guard if
        // an all-quantised stage is ever added.
        m->weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, dit_gguf_weight_buffer_type(backend));
        if (!m->weight_buf) {
            fprintf(stderr, "[acestep-lm] failed to allocate weight buffer\n");
            if (map_buf) ggml_backend_buffer_free(map_buf);
            ggml_free(ctx);
            dit_gguf_close(g);
            delete m;
            return nullptr;
        }

        q3_load_raw(m->embed_tokens, g, "model.embed_tokens.weight");
        lm_load_head_chunks(m->head_chunks, (const uint8_t *) dit_gdata(g, "model.embed_tokens.weight"),
                            ggml_row_size(m->embed_type, c.hidden_size));
        q3_load_f32(m->final_norm, g, "model.norm.weight");
        for (int i = 0; i < c.n_layers; i++) {
            const std::string prefix = "model.layers." + std::to_string(i);
            if (fuse) {
                if (!lm_load_layer_fused(g, prefix, m->layers[i], m->qkv_fused[i], m->gateup_fused[i])) {
                    fprintf(stderr, "[acestep-lm] fused weight load failed (layer %d)\n", i);
                    if (map_buf) ggml_backend_buffer_free(map_buf);
                    ggml_backend_buffer_free(m->weight_buf);  // ggml_free(ctx) drops descriptors, not the buffer
                    ggml_free(ctx);
                    dit_gguf_close(g);
                    delete m;
                    return nullptr;
                }
            } else {
                q3_load_layer(g, prefix, m->layers[i]);
            }
        }
    }
    // Exact mmapped weight footprint (see dit_gguf_mapped_bytes); reported by
    // lm_model_weight_bytes below so mapped loads don't look like a few-MB stub.
    m->mapped_bytes = mapped ? dit_gguf_mapped_bytes(ctx, g) : 0;
    if (measure) {
        measure->weights_mapped_bytes = m->mapped_bytes;
        // Mark still-unallocated weights externally-allocated so graph sizing
        // excludes them (see textenc_model_load_impl). No data is read after.
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (!t->data && !t->view_src) {
                t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
        }
    }

    // KV cache: n_sets * n_layers tensors.
    const int D = c.head_dim, Nkv = c.n_kv_heads, S = c.max_seq_len, Lc = c.n_layers, NS = m->n_sets;
    ggml_init_params kp{
        ggml_tensor_overhead() * (size_t) (Lc * 2 + NS * Lc * 2 + 4), nullptr, /*no_alloc=*/true
    };
    m->kv_ctx = ggml_init(kp);
    m->kv_k4.resize((size_t) Lc);
    m->kv_v4.resize((size_t) Lc);
    m->kv_k.resize((size_t) NS * Lc);
    m->kv_v.resize((size_t) NS * Lc);
    for (int l = 0; l < Lc; l++) {
        m->kv_k4[l] = ggml_new_tensor_4d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv, NS);
        m->kv_v4[l] = ggml_new_tensor_4d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv, NS);
        ggml_set_name(m->kv_k4[l], ("kv_k4_" + std::to_string(l)).c_str());
        ggml_set_name(m->kv_v4[l], ("kv_v4_" + std::to_string(l)).c_str());
        for (int s = 0; s < NS; s++) {
            int idx      = s * Lc + l;
            const size_t off = (size_t) s * m->kv_k4[l]->nb[3];
            m->kv_k[idx] = ggml_view_3d(m->kv_ctx, m->kv_k4[l], D, S, Nkv,
                                        m->kv_k4[l]->nb[1], m->kv_k4[l]->nb[2], off);
            m->kv_v[idx] = ggml_view_3d(m->kv_ctx, m->kv_v4[l], D, S, Nkv,
                                        m->kv_v4[l]->nb[1], m->kv_v4[l]->nb[2], off);
            ggml_set_name(m->kv_k[idx], ("kv_k_" + std::to_string(s) + "_" + std::to_string(l)).c_str());
            ggml_set_name(m->kv_v[idx], ("kv_v_" + std::to_string(s) + "_" + std::to_string(l)).c_str());
        }
    }
    if (measure) {
        measure->kv_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            m->kv_ctx, ggml_backend_get_default_buffer_type(backend));
        // Mark the KV bases externally-allocated so measure graphs can view
        // them; the per-set views keep view_src and are resolved through it.
        for (ggml_tensor * t = ggml_get_first_tensor(m->kv_ctx); t; t = ggml_get_next_tensor(m->kv_ctx, t)) {
            if (!t->data && !t->view_src) {
                t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
        }
    } else {
        m->kv_buf = ggml_backend_alloc_ctx_tensors(m->kv_ctx, backend);
        if (!m->kv_buf) {
            fprintf(stderr, "[acestep-lm] failed to allocate KV cache\n");
            if (map_buf) ggml_backend_buffer_free(map_buf);
            if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);  // ggml_free(ctx) drops descriptors, not the buffer
            ggml_free(ctx);
            ggml_free(m->kv_ctx);
            dit_gguf_close(g);
            delete m;
            return nullptr;
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_buffer_clear(m->kv_buf, 0);
            if (std::getenv("ACESTEP_LM_TIMING"))
                fprintf(stderr, "[lm-timing] kv-clear %.1f ms (%.1f MB)\n",
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                        ggml_backend_buffer_get_size(m->kv_buf) / 1048576.0);
        }
    }
    m->kv_pos.assign(NS, 0);

    if (verbose) {
        const size_t kv_bytes = (size_t) NS * Lc * 2 * D * S * Nkv * sizeof(uint16_t);
        fprintf(stderr,
                "[acestep-lm] loaded %s: %.1f MB weights, %d layers H=%d V=%d Nh=%d/%d D=%d, KV %.1f MB (%d sets), FA=%s\n",
                path.c_str(), lm_model_weight_bytes(m) / 1048576.0, c.n_layers, c.hidden_size, c.vocab_size, c.n_heads,
                c.n_kv_heads, c.head_dim, kv_bytes / 1048576.0, NS, m->use_flash_attn ? "on" : "off");
        if (split_table) {
            fprintf(stderr, "[acestep-lm] embedding lookup on the host, LM head in %zu chunks of <= %d rows\n",
                    m->head_chunks.size(), head_rows);
        }
    }

    const bool host_lookup = split_table && !measure;
    if (host_lookup) m->host_embed = (const uint8_t *) dit_gdata(g, "model.embed_tokens.weight");
    if (mapped || host_lookup) {
        m->mapped  = mapped;
        m->map_buf = map_buf;
        m->gguf    = g;  // keep the mmap alive; mapped weights point into it
    } else {
        dit_gguf_close(g);
    }
    return m;
}

LMModel * lm_model_load(const std::string & path, ggml_backend_t backend, int max_seq_len, bool verbose,
                        int n_kv_sets) {
    return lm_model_load_impl(path, backend, max_seq_len, verbose, n_kv_sets, /*measure=*/nullptr);
}

LMModel * lm_model_load_metadata_only(const std::string & path, ggml_backend_t backend, int max_seq_len,
                                      bool verbose, int n_kv_sets, AcestepStageMeasure & measure) {
    measure = AcestepStageMeasure{};
    return lm_model_load_impl(path, backend, max_seq_len, verbose, n_kv_sets, &measure);
}

LMModel * lm_model_load_split_table(const std::string & path, ggml_backend_t backend, int max_seq_len, int n_kv_sets,
                                    int head_chunk_rows) {
    if (head_chunk_rows <= 0) return nullptr;
    return lm_model_load_impl(path, backend, max_seq_len, /*verbose=*/false, n_kv_sets, /*measure=*/nullptr,
                              head_chunk_rows);
}

std::vector<std::pair<int, int>> lm_head_chunk_ranges(int vocab, int max_rows) {
    std::vector<std::pair<int, int>> ranges;
    for (const LMHeadChunk & chunk : lm_plan_head_chunks(vocab, max_rows)) ranges.emplace_back(chunk.row0, chunk.rows);
    return ranges;
}

size_t lm_model_kv_bytes(const LMModel * m) {
    return (m && m->kv_buf) ? ggml_backend_buffer_get_size(m->kv_buf) : 0;
}

size_t lm_model_compute_buffer_bytes(const LMModel * m) {
    return (m && m->graph_cache.ga) ? ggml_gallocr_get_buffer_size(m->graph_cache.ga, 0) : 0;
}

size_t lm_measure_partial_head_bytes(const LMModel * m, int count) {
    if (!m || !m->embed_tokens || count <= 0) return 0;
    // Same tensor lm_build_partial_head creates, sized instead of allocated.
    ggml_init_params hp{ ggml_tensor_overhead() * 3 + 16, nullptr, /*no_alloc=*/true };
    ggml_context *   ctx = ggml_init(hp);
    if (!ctx) return 0;
    ggml_new_tensor_2d(ctx, m->embed_tokens->type, m->cfg.hidden_size, count);
    const size_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
        ctx, dit_gguf_weight_buffer_type(m->backend));
    ggml_free(ctx);
    return bytes;
}

void lm_model_free(LMModel * m) {
    if (!m) return;
    m->graph_cache.release();
    lm_partial_head_clear(m);
    if (m->kv_buf) ggml_backend_buffer_free(m->kv_buf);
    if (m->kv_ctx) ggml_free(m->kv_ctx);
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    if (m->map_buf) ggml_backend_buffer_free(m->map_buf);
    if (m->mapped || m->host_embed) dit_gguf_close(m->gguf);
    delete m;
}

const LMConfig & lm_model_config(const LMModel * m) { return m->cfg; }
size_t           lm_model_weight_bytes(const LMModel * m) {
    if (!m) return 0;
    const size_t alloc = m->weight_buf ? ggml_backend_buffer_get_size(m->weight_buf) : 0;
    const size_t partial = m->lm_head_buf ? ggml_backend_buffer_get_size(m->lm_head_buf) : 0;
    return alloc + m->mapped_bytes + partial;  // allocated + mmapped + compact tied head
}
int  lm_num_kv_sets(const LMModel * m) { return m->n_sets; }
bool lm_model_embeddings_quantized(const LMModel * m) {
    return m && ggml_is_quantized(m->embed_type);
}
bool lm_model_supports_batched_decode(const LMModel * m) {
    return m && m->use_flash_attn && m->n_sets > 1;
}
void lm_reset(LMModel * m, int set) { if (set >= 0 && set < m->n_sets) m->kv_pos[set] = 0; }
int  lm_kv_pos(const LMModel * m, int set) { return (set >= 0 && set < m->n_sets) ? m->kv_pos[set] : 0; }

// One fused-QKV projection sliced into strided [D, heads, S] views; matches the
// split path's reshape_3d layouts except for the larger dim-2 stride.
static void lm_qkv_views(ggml_context * ctx, const Qwen3Config & c, ggml_tensor * qkv_fused, ggml_tensor * x, int S,
                         ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v) {
    const int     D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    ggml_tensor * qkv = q3_linear(ctx, qkv_fused, x, c.prec);  // [(Nh+2*Nkv)*D, S]
    const size_t  es  = ggml_element_size(qkv);
    *q = ggml_view_3d(ctx, qkv, D, Nh, S, (size_t) D * es, qkv->nb[1], 0);
    *k = ggml_view_3d(ctx, qkv, D, Nkv, S, (size_t) D * es, qkv->nb[1], (size_t) Nh * D * es);
    *v = ggml_view_3d(ctx, qkv, D, Nkv, S, (size_t) D * es, qkv->nb[1], (size_t) (Nh + Nkv) * D * es);
}

// MLP with the optional fused gate|up projection (gate rows first, so the
// single-tensor ggml_swiglu split matches swiglu_split(gate, up)).
static ggml_tensor * lm_mlp(ggml_context * ctx, const Qwen3Config & c, Qwen3Layer * ly, ggml_tensor * gateup_fused,
                            ggml_tensor * x) {
    if (!gateup_fused) return q3_build_mlp(ctx, ly, x, c.prec);
    ggml_tensor * gu = q3_linear(ctx, gateup_fused, x, c.prec);  // [2*FFN, S]
    return q3_linear(ctx, ly->down_proj, ggml_swiglu(ctx, gu), c.prec);
}

// KV-cache self-attention for one layer. x [H, S]. Writes S fresh rows at
// [kv_pos..kv_pos+S) then reads the [0, n_kv_pad) window. F32 attention.
static ggml_tensor * lm_attn(ggml_context * ctx, ggml_cgraph * gf, const Qwen3Config & c, Qwen3Layer * ly,
                             ggml_tensor * qkv_fused, ggml_tensor * x, ggml_tensor * positions, ggml_tensor * mask,
                             ggml_tensor * kv_rows, ggml_tensor * cache_k, ggml_tensor * cache_v, int n_kv_pad, int S,
                             bool use_flash_attn) {
    const int D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;

    ggml_tensor *q, *k, *v;
    if (qkv_fused) {
        lm_qkv_views(ctx, c, qkv_fused, x, S, &q, &k, &v);
    } else {
        q = ggml_reshape_3d(ctx, q3_linear(ctx, ly->q_proj, x, c.prec), D, Nh, S);
        k = ggml_reshape_3d(ctx, q3_linear(ctx, ly->k_proj, x, c.prec), D, Nkv, S);
        v = ggml_reshape_3d(ctx, q3_linear(ctx, ly->v_proj, x, c.prec), D, Nkv, S);
    }

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), q3_as_f32(ctx, ly->q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), q3_as_f32(ctx, ly->k_norm));

    q = ggml_rope_ext(ctx, q, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);                   // [D, S, Nh]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D, S, Nkv]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [D, S, Nkv]

    // Write K,V into the persistent f16 cache at kv_rows (broadcast over Nkv).
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, k, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, v, kv_rows));

    // Read the padded window [0, n_kv_pad). Both views are strided in dim 2: nb[2] is the
    // full cache channel stride (max_seq rows), so consecutive KV heads sit max_seq rows
    // apart while only n_kv_pad rows of each are live. Passed to the backends as-is.
    //
    // This used to pack K through ggml_cont at prefill. The Vulkan tiled matmul was
    // deriving its channel stride as ne00*ne01 rather than from nb[2], so every KV head
    // past the first read the previous head's rows and the LM degenerated on GPU while
    // CPU stayed correct. That was a backend bug, not a property of this graph, and it is
    // fixed in ggml (ggml_vk_channel_stride_elements); packing here only hid it for one
    // caller while leaving every other strided-src0 matmul broken.
    ggml_tensor * k_full = ggml_view_3d(ctx, cache_k, D, n_kv_pad, Nkv, cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * v_full = ggml_view_3d(ctx, cache_v, D, n_kv_pad, Nkv, cache_v->nb[1], cache_v->nb[2], 0);

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn;
    if (use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q, k_full, v_full, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, c.prec);
    } else {
        attn = q3_attn(ctx, ggml_cont(ctx, q), k_full, v_full, mask, scale, c.prec);
    }
    attn                = ggml_reshape_2d(ctx, attn, (int64_t) Nh * D, S);
    return q3_linear(ctx, ly->o_proj, attn, c.prec);
}

bool lm_model_forward(LMModel * m, const int32_t * token_ids, int n_tokens, std::vector<float> & logits_out, int set,
                      std::vector<float> * layer_states_out, int logit_limit, size_t * measure_compute) {
    const Qwen3Config & c   = m->q3;
    const LMConfig &    lc  = m->cfg;
    const int           H   = c.hidden_size;
    const int           S   = n_tokens;
    if (set < 0 || set >= m->n_sets) {
        fprintf(stderr, "[acestep-lm] invalid kv set %d (n_sets=%d)\n", set, m->n_sets);
        return false;
    }
    const int           kv0 = m->kv_pos[set];
    const int           kv_len = kv0 + S;

    if (kv_len > lc.max_seq_len) {
        fprintf(stderr, "[acestep-lm] kv_len %d > max_seq %d\n", kv_len, lc.max_seq_len);
        return false;
    }
    int n_kv_pad = (int) GGML_PAD(kv_len, 256);
    if (n_kv_pad > lc.max_seq_len) n_kv_pad = lc.max_seq_len;

    // Prefix head [0, logit_limit): reads fewer tied-head rows when the caller
    // (FSM-constrained phase 1) can never select tokens past the limit.
    // Measure mode never builds (or copies into) the real compact head; it
    // stands a same-shape descriptor into the measure graph instead (below).
    const bool    split_head = !m->head_chunks.empty();
    ggml_tensor * head_w     = m->embed_tokens;
    int           out_V      = lc.vocab_size;
    if (logit_limit > 0 && logit_limit < lc.vocab_size) {
        if (split_head) {
            out_V = logit_limit;
        } else if (measure_compute) {
            head_w = nullptr;  // created in the graph ctx below
            out_V  = logit_limit;
        } else if (!lm_build_partial_head(m, 0, logit_limit)) {
            fprintf(stderr, "[acestep-lm] failed to build prefix head (limit=%d)\n", logit_limit);
            return false;
        } else {
            head_w = m->lm_head_partial;
            out_V  = logit_limit;
        }
    }

    const bool     cacheable = layer_states_out == nullptr && measure_compute == nullptr;
    LMGraphCache & gc        = m->graph_cache;
    const bool     cache_hit = cacheable && gc.ctx != nullptr && !gc.batch && gc.S == S &&
                               gc.n_kv_pad == n_kv_pad && gc.set0 == set && gc.head == head_w &&
                               gc.head_lo == 0 && gc.head_hi == out_V;

    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor *t_ids = nullptr, *positions = nullptr, *kv_rows = nullptr, *mask = nullptr, *lgt = nullptr,
                *embd = nullptr;
    std::vector<LMHeadChunk>   head_logits;
    std::vector<ggml_tensor *> layer_states;

    if (cache_hit) {
        ctx   = gc.ctx;
        gf    = gc.gf;
        ga    = gc.ga;
        t_ids = gc.t_ids; positions = gc.positions; kv_rows = gc.kv_rows;
        mask  = gc.mask;  lgt = gc.lgt; embd = gc.embd;
        head_logits = gc.head_logits;
    } else {
        if (cacheable) gc.release();

        const size_t nodes = 16384;
        ggml_init_params gp{ ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
        ctx = ggml_init(gp);

        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
        ggml_set_input(positions);
        kv_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, S);
        ggml_set_input(kv_rows);
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, S);
        ggml_set_input(mask);

        if (measure_compute && !head_w && !split_head) {
            // Same shape/type as the compact head lm_build_partial_head would
            // allocate; marked externally-allocated so graph sizing excludes
            // it from the compute buffer, like every other weight.
            head_w       = ggml_new_tensor_2d(ctx, m->embed_tokens->type, H, out_V);
            head_w->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }

        gf                   = ggml_new_graph_custom(ctx, nodes, false);
        ggml_tensor * hidden = lm_new_token_input(ctx, m, S, &t_ids, &embd);  // [H, S]
        for (int l = 0; l < c.n_layers; l++) {
            Qwen3Layer *  ly   = &m->layers[l];
            ggml_tensor * norm = q3_rms_norm_w(ctx, hidden, ly->input_norm, c.rms_norm_eps);
            int           idx  = set * c.n_layers + l;
            ggml_tensor * qkvf = m->qkv_fused.empty() ? nullptr : m->qkv_fused[l];
            ggml_tensor * guf  = m->gateup_fused.empty() ? nullptr : m->gateup_fused[l];
            ggml_tensor * attn = lm_attn(ctx, gf, c, ly, qkvf, norm, positions, mask, kv_rows, m->kv_k[idx],
                                         m->kv_v[idx], n_kv_pad, S, m->use_flash_attn);
            hidden             = ggml_add(ctx, hidden, attn);
            norm               = q3_rms_norm_w(ctx, hidden, ly->post_norm, c.rms_norm_eps);
            ggml_tensor * mlp  = lm_mlp(ctx, c, ly, guf, norm);
            hidden             = ggml_add(ctx, hidden, mlp);
            if (layer_states_out) {
                ggml_set_output(hidden);
                ggml_build_forward_expand(gf, hidden);
                layer_states.push_back(hidden);
            }
        }
        hidden = q3_rms_norm_w(ctx, hidden, m->final_norm, c.rms_norm_eps);
        if (S > 1) {
            hidden = ggml_view_1d(ctx, hidden, H, (int64_t) (S - 1) * H * sizeof(float));  // last token
        }
        if (split_head) {
            head_logits = lm_build_split_head(ctx, gf, m, hidden, 0, out_V);
        } else {
            lgt = q3_linear(ctx, head_w, hidden, c.prec);  // [out_V, 1]
            ggml_set_output(lgt);
            ggml_build_forward_expand(gf, lgt);
        }

        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
        if (!ga) {
            fprintf(stderr, "[acestep-lm] forward gallocr init failed (n=%d)\n", S);
            ggml_free(ctx);
            return false;
        }
        if (measure_compute) {
            // Size-only twin of the alloc below; KV positions stay untouched.
            *measure_compute = 0;
            ggml_gallocr_reserve_n_size(ga, gf, nullptr, nullptr, measure_compute);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return true;
        }
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            fprintf(stderr, "[acestep-lm] forward alloc failed (n=%d)\n", S);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return false;
        }

        if (cacheable) {
            gc.ctx   = ctx;
            gc.gf    = gf;
            gc.ga    = ga;
            gc.t_ids = t_ids; gc.positions = positions; gc.kv_rows = kv_rows;
            gc.mask  = mask;  gc.lgt = lgt; gc.embd = embd;
            gc.head_logits = head_logits;
            gc.batch = false; gc.S = S; gc.n_kv_pad = n_kv_pad; gc.set0 = set;
            gc.n_batch = -1;  gc.head = head_w; gc.head_lo = 0; gc.head_hi = out_V;
        }
    }

    if (!lm_set_token_input(m, t_ids, embd, token_ids, S)) {
        if (!cacheable) {
            ggml_gallocr_free(ga);
            ggml_free(ctx);
        }
        return false;
    }
    std::vector<int32_t> pos(S);
    std::vector<int64_t> rows(S);
    for (int i = 0; i < S; i++) { pos[i] = kv0 + i; rows[i] = (int64_t) (kv0 + i); }
    ggml_backend_tensor_set(positions, pos.data(), 0, (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(kv_rows, rows.data(), 0, (size_t) S * sizeof(int64_t));

    std::vector<uint16_t> md((size_t) n_kv_pad * S);
    for (int i = 0; i < S; i++) {
        int qpos = kv0 + i;
        for (int j = 0; j < n_kv_pad; j++)
            md[(size_t) i * n_kv_pad + j] = ggml_fp32_to_fp16((j <= qpos) ? 0.0f : -INFINITY);
    }
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(uint16_t));

    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
        if (cacheable) {
            gc.release();
        } else {
            ggml_gallocr_free(ga);
            ggml_free(ctx);
        }
        return false;
    }

    if (split_head) {
        lm_read_split_logits(head_logits, 0, out_V, 1, logits_out);
    } else {
        logits_out.resize((size_t) out_V);
        ggml_backend_tensor_get(lgt, logits_out.data(), 0, (size_t) out_V * sizeof(float));
    }

    if (layer_states_out) {
        const size_t per_layer = (size_t) H * S;
        layer_states_out->resize(per_layer * layer_states.size());
        for (size_t i = 0; i < layer_states.size(); i++) {
            ggml_backend_tensor_get(layer_states[i], layer_states_out->data() + i * per_layer, 0,
                                    per_layer * sizeof(float));
        }
    }

    if (!cacheable) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    m->kv_pos[set] += S;
    return true;
}

bool lm_model_forward_batch(LMModel * m, const int32_t * token_ids, const int * sets, int N,
                            std::vector<float> & logits_out, int logit_offset,
                            size_t * measure_compute) {
    if (!m || !m->use_flash_attn || N < 1 || N > m->n_sets || logit_offset < 0 ||
        logit_offset >= m->cfg.vocab_size) {
        return false;
    }

    const int s0 = sets[0];
    for (int i = 0; i < N; i++) {
        if (sets[i] != s0 + i || sets[i] < 0 || sets[i] >= m->n_sets) return false;
    }

    const Qwen3Config & c = m->q3;
    const LMConfig & lc = m->cfg;
    const int H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const int requested_out_V = lc.vocab_size - logit_offset;
    const bool split_head      = !m->head_chunks.empty();
    // Measure mode assumes the compact head builds (the real path only falls
    // back to the full tied head when its allocation fails, and the verdict
    // this measurement feeds exists to prevent exactly that situation); the
    // head weights themselves are priced via lm_measure_partial_head_bytes.
    const bool compact_head =
        !split_head && logit_offset > 0 &&
        (measure_compute != nullptr ||
         lm_build_partial_head(m, logit_offset, m->cfg.vocab_size - logit_offset));
    const int graph_out_V = (compact_head || split_head) ? requested_out_V : lc.vocab_size;
    int max_kv_len = 0;
    for (int i = 0; i < N; i++) {
        const int kv_len = m->kv_pos[sets[i]] + 1;
        if (kv_len > lc.max_seq_len) return false;
        max_kv_len = std::max(max_kv_len, kv_len);
    }
    int n_kv_pad = (int) GGML_PAD(max_kv_len, 256);
    if (n_kv_pad > lc.max_seq_len) n_kv_pad = lc.max_seq_len;

    LMGraphCache & gc        = m->graph_cache;
    ggml_tensor *  lm_weight = compact_head ? (measure_compute ? nullptr : m->lm_head_partial)
                                            : m->embed_tokens;
    const bool     cache_hit = !measure_compute &&
                               gc.ctx != nullptr && gc.batch && gc.n_kv_pad == n_kv_pad &&
                               gc.set0 == s0 && gc.n_batch == N && gc.head == lm_weight &&
                               gc.head_lo == logit_offset && gc.head_hi == lc.vocab_size;

    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor *t_ids = nullptr, *positions = nullptr, *kv_rows = nullptr, *mask = nullptr, *lgt = nullptr,
                *embd = nullptr;
    std::vector<LMHeadChunk> head_logits;

    if (cache_hit) {
        ctx   = gc.ctx;
        gf    = gc.gf;
        ga    = gc.ga;
        t_ids = gc.t_ids; positions = gc.positions; kv_rows = gc.kv_rows;
        mask  = gc.mask;  lgt = gc.lgt; embd = gc.embd;
        head_logits = gc.head_logits;
    } else {
        if (!measure_compute) gc.release();

        const size_t nodes = 16384;
        ggml_init_params gp{ ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(nodes, false), nullptr, true };
        ctx = ggml_init(gp);

        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_set_input(positions);
        kv_rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I64, 1, 1, N);
        ggml_set_input(kv_rows);
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv_pad, 1, 1, N);
        ggml_set_input(mask);

        if (measure_compute && !lm_weight && !split_head) {
            // Same shape/type as the compact head lm_build_partial_head would
            // allocate; externally-allocated so graph sizing excludes it.
            lm_weight       = ggml_new_tensor_2d(ctx, m->embed_tokens->type, H, graph_out_V);
            lm_weight->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }

        gf = ggml_new_graph_custom(ctx, nodes, false);
        ggml_tensor * hidden = lm_new_token_input(ctx, m, N, &t_ids, &embd);  // [H, N]

        for (int l = 0; l < c.n_layers; l++) {
            Qwen3Layer * ly = &m->layers[l];
            ggml_tensor * norm = q3_rms_norm_w(ctx, hidden, ly->input_norm, c.rms_norm_eps);

            ggml_tensor *q, *k, *v;
            if (!m->qkv_fused.empty()) {
                lm_qkv_views(ctx, c, m->qkv_fused[l], norm, N, &q, &k, &v);
            } else {
                q = ggml_reshape_3d(ctx, q3_linear(ctx, ly->q_proj, norm, c.prec), D, Nh, N);
                k = ggml_reshape_3d(ctx, q3_linear(ctx, ly->k_proj, norm, c.prec), D, Nkv, N);
                v = ggml_reshape_3d(ctx, q3_linear(ctx, ly->v_proj, norm, c.prec), D, Nkv, N);
            }

            q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), q3_as_f32(ctx, ly->q_norm));
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), q3_as_f32(ctx, ly->k_norm));
            q = ggml_rope_ext(ctx, q, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, positions, nullptr, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            q = ggml_cont(ctx, q);
            k = ggml_cont(ctx, k);
            v = ggml_cont(ctx, v);

            const size_t off = (size_t) s0 * m->kv_k4[l]->nb[3];
            ggml_tensor * k_sets = ggml_view_4d(ctx, m->kv_k4[l], D, lc.max_seq_len, Nkv, N,
                                                m->kv_k4[l]->nb[1], m->kv_k4[l]->nb[2],
                                                m->kv_k4[l]->nb[3], off);
            ggml_tensor * v_sets = ggml_view_4d(ctx, m->kv_v4[l], D, lc.max_seq_len, Nkv, N,
                                                m->kv_v4[l]->nb[1], m->kv_v4[l]->nb[2],
                                                m->kv_v4[l]->nb[3], off);
            ggml_tensor * k_new = ggml_reshape_4d(ctx, k, D, 1, Nkv, N);
            ggml_tensor * v_new = ggml_reshape_4d(ctx, v, D, 1, Nkv, N);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_sets, k_new, kv_rows));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_sets, v_new, kv_rows));

            ggml_tensor * q4 = ggml_reshape_4d(ctx, q, D, 1, Nh, N);
            ggml_tensor * k_batch = ggml_view_4d(ctx, m->kv_k4[l], D, n_kv_pad, Nkv, N,
                                                 m->kv_k4[l]->nb[1], m->kv_k4[l]->nb[2],
                                                 m->kv_k4[l]->nb[3], off);
            ggml_tensor * v_batch = ggml_view_4d(ctx, m->kv_v4[l], D, n_kv_pad, Nkv, N,
                                                 m->kv_v4[l]->nb[1], m->kv_v4[l]->nb[2],
                                                 m->kv_v4[l]->nb[3], off);
            const float scale = 1.0f / sqrtf((float) D);
            ggml_tensor * attn = ggml_flash_attn_ext(ctx, q4, k_batch, v_batch, mask, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, c.prec);
            attn = ggml_reshape_2d(ctx, attn, (int64_t) Nh * D, N);

            hidden = ggml_add(ctx, hidden, q3_linear(ctx, ly->o_proj, attn, c.prec));
            norm = q3_rms_norm_w(ctx, hidden, ly->post_norm, c.rms_norm_eps);
            hidden = ggml_add(ctx, hidden,
                              lm_mlp(ctx, c, ly, m->gateup_fused.empty() ? nullptr : m->gateup_fused[l], norm));
        }

        hidden = q3_rms_norm_w(ctx, hidden, m->final_norm, c.rms_norm_eps);
        if (split_head) {
            head_logits = lm_build_split_head(ctx, gf, m, hidden, logit_offset, lc.vocab_size);
        } else {
            lgt = q3_linear(ctx, lm_weight, hidden, c.prec);  // [graph_out_V, N]
            ggml_set_output(lgt);
            ggml_build_forward_expand(gf, lgt);
        }

        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
        if (!ga) {
            ggml_free(ctx);
            return false;
        }
        if (measure_compute) {
            // Size-only twin of the alloc below; KV positions stay untouched.
            *measure_compute = 0;
            ggml_gallocr_reserve_n_size(ga, gf, nullptr, nullptr, measure_compute);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return true;
        }
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return false;
        }

        gc.ctx   = ctx;
        gc.gf    = gf;
        gc.ga    = ga;
        gc.t_ids = t_ids; gc.positions = positions; gc.kv_rows = kv_rows;
        gc.mask  = mask;  gc.lgt = lgt; gc.embd = embd;
        gc.head_logits = head_logits;
        gc.batch = true; gc.S = 1; gc.n_kv_pad = n_kv_pad; gc.set0 = s0;
        gc.n_batch = N;  gc.head = lm_weight; gc.head_lo = logit_offset; gc.head_hi = lc.vocab_size;
    }

    if (!lm_set_token_input(m, t_ids, embd, token_ids, N)) return false;
    std::vector<int32_t> pos(N);
    std::vector<int64_t> rows(N);
    std::vector<uint16_t> md((size_t) n_kv_pad * N);
    for (int i = 0; i < N; i++) {
        const int kv_pos = m->kv_pos[sets[i]];
        pos[i] = kv_pos;
        rows[i] = (int64_t) kv_pos;
        for (int j = 0; j < n_kv_pad; j++) {
            md[(size_t) i * n_kv_pad + j] = ggml_fp32_to_fp16(j <= kv_pos ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(positions, pos.data(), 0, (size_t) N * sizeof(int32_t));
    ggml_backend_tensor_set(kv_rows, rows.data(), 0, (size_t) N * sizeof(int64_t));
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(uint16_t));

    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
        gc.release();
        return false;
    }

    logits_out.resize((size_t) requested_out_V * N);
    if (split_head) {
        lm_read_split_logits(head_logits, logit_offset, lc.vocab_size, N, logits_out);
    } else if (logit_offset == 0 || compact_head) {
        ggml_backend_tensor_get(lgt, logits_out.data(), 0, logits_out.size() * sizeof(float));
    } else {
        // Allocation of the compact head is an optimization, not a correctness
        // requirement. Fall back to the full tied head and slice host logits.
        std::vector<float> full_logits((size_t) graph_out_V * N);
        ggml_backend_tensor_get(lgt, full_logits.data(), 0, full_logits.size() * sizeof(float));
        for (int i = 0; i < N; ++i) {
            std::memcpy(logits_out.data() + (size_t) i * requested_out_V,
                        full_logits.data() + (size_t) i * graph_out_V + logit_offset,
                        (size_t) requested_out_V * sizeof(float));
        }
    }
    for (int i = 0; i < N; i++) m->kv_pos[sets[i]]++;
    return true;
}

// ── Size-only measure wrappers (memory-fit preflight) ───────────────────────
// The graph shape depends on the KV position, so each wrapper stages the
// projected position, runs the measure-mode forward (which never advances or
// writes the cache), and restores it.

bool lm_model_measure_prefill(LMModel * m, int n_tokens, int logit_limit, size_t & compute_bytes) {
    if (!m || n_tokens < 1 || n_tokens > m->cfg.max_seq_len) return false;
    const int saved  = m->kv_pos[0];
    m->kv_pos[0]     = 0;
    std::vector<float> dummy;
    const bool ok = lm_model_forward(m, /*token_ids=*/nullptr, n_tokens, dummy, /*set=*/0,
                                     /*layer_states_out=*/nullptr, logit_limit, &compute_bytes);
    m->kv_pos[0] = saved;
    return ok;
}

bool lm_model_measure_decode(LMModel * m, int kv_len, int logit_limit, size_t & compute_bytes) {
    if (!m || kv_len < 1 || kv_len > m->cfg.max_seq_len) return false;
    const int saved = m->kv_pos[0];
    m->kv_pos[0]    = kv_len - 1;  // the decode step that reaches kv_len
    std::vector<float> dummy;
    const bool ok = lm_model_forward(m, /*token_ids=*/nullptr, 1, dummy, /*set=*/0,
                                     /*layer_states_out=*/nullptr, logit_limit, &compute_bytes);
    m->kv_pos[0] = saved;
    return ok;
}

bool lm_model_measure_decode_batch(LMModel * m, int N, int kv_len, int logit_offset, size_t & compute_bytes) {
    if (!m || N < 1 || N > m->n_sets || kv_len < 1 || kv_len > m->cfg.max_seq_len) return false;
    std::vector<int> saved(m->kv_pos);
    std::vector<int> sets((size_t) N);
    for (int i = 0; i < N; i++) {
        sets[i]        = i;
        m->kv_pos[i]   = kv_len - 1;
    }
    std::vector<float> dummy;
    const bool ok = lm_model_forward_batch(m, /*token_ids=*/nullptr, sets.data(), N, dummy,
                                           logit_offset, &compute_bytes);
    m->kv_pos = saved;
    return ok;
}

} // namespace tts_cpp::acestep
