// TDT greedy decode, runtime weight prep, and ggml or CPU decoder paths.

#include "parakeet_tdt.h"
#include "parakeet_log.h"
#include "sentencepiece_bpe.h"
#include "backend_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace parakeet {

constexpr int kWrongTransducerModel = 10;

TransducerGraphOutputs build_transducer_argmax_outputs(ggml_context * ctx,
                                                       ggml_tensor * logits,
                                                       int token_count,
                                                       int duration_count) {
    ggml_tensor * token_logits = ggml_view_2d(
        ctx, logits, token_count, 1,
        static_cast<size_t>(token_count) * sizeof(float), 0);
    token_logits = ggml_cont(ctx, token_logits);

    TransducerGraphOutputs outputs;
    outputs.token = ggml_argmax(ctx, token_logits);
    if (duration_count <= 0) return outputs;

    ggml_tensor * duration_logits = ggml_view_2d(
        ctx, logits, duration_count, 1,
        static_cast<size_t>(duration_count) * sizeof(float),
        static_cast<size_t>(token_count) * sizeof(float));
    duration_logits = ggml_cont(ctx, duration_logits);
    outputs.duration = ggml_argmax(ctx, duration_logits);
    return outputs;
}

namespace {

int argmax_f32(const float * data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Vectorisable f32 gemv: y[i] = (b ? b[i] : 0) + sum_j W[i,j] * x[j]
//
// Rationale: parakeet's TDT / EOU / Sortformer decoders run all
// projection / LSTM / joint matmuls as f32 host-side gemvs (the
// quantised weights are dequantised once at `*_prepare_runtime`
// time -- see `dequantize_to_f32`). With `__restrict` + `#pragma
// omp simd` (or, equivalently, gcc's `-O3 -ffast-math` auto-
// vectoriser, which the project uses) gcc-13 picks AVX2/AVX-512
// FMA on x86_64 and lifts the inner loop from ~1 FMA/cycle to
// ~8/cycle. Same shape as ggml-cpu's vec.cpp but specialised to
// the gemv access pattern the decoder hits at every emitted token.
void gemv_f32(const float * __restrict W, const float * __restrict x,
              const float * __restrict b, float * __restrict y,
              int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        const float * __restrict row = W + (size_t) i * in_dim;
        float acc = b ? b[i] : 0.0f;
        #pragma GCC ivdep
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
}

void gemv_add_f32(const float * __restrict W, const float * __restrict x,
                  float * __restrict y, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        const float * __restrict row = W + (size_t) i * in_dim;
        float acc = 0.0f;
        #pragma GCC ivdep
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] += acc;
    }
}

// Dequantise a GGUF tensor (f32, f16, q8_0, etc.) into a host float vector.
// Backend-aware dequantise of a GGUF tensor (f32, f16, q8_0, etc.) into a
// host float vector. Goes through ggml_backend_tensor_get so it works on
// Vulkan / CUDA / Metal where t->data is a device handle, not a host
// pointer. Mirrors the same fix in parakeet_eou.cpp / parakeet_sortformer.cpp
// (origin/main 4fcea2b — "Fix Vulkan segfault in TDT/EOU/Sortformer
// prepare_runtime"). For quantised types we fetch raw device bytes into a
// host scratch buffer first and run to_float on the host buffer.
void dequantize_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) throw std::runtime_error("tdt_prepare_runtime: missing tensor");
    const size_t n = (size_t) ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return;
    }
    const auto * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error(std::string("tdt_prepare_runtime: no to_float for type ") +
                                 ggml_type_name(t->type));
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> host_raw(nbytes);
    ggml_backend_tensor_get(t, host_raw.data(), 0, nbytes);
    tr->to_float(host_raw.data(), out.data(), (int64_t) n);
}

// ---- Scalar host-side LSTM step (CPU fallback) ----
//
// `layer_input_scratch` is a caller-owned reusable buffer matching the
// EOU path. The CPU fallback path runs on CPU-only Engine builds
// where the per-step graph dispatch latency dominates GPU graphs;
// this lifts ~250 emission-step `std::vector<float>(H_pred=640)`
// allocations per utterance out of the hot loop. gemv_f32 writes
// every output byte before any read, so re-using the scratch buffer
// is byte-equal to the per-call allocation.
void host_lstm_step(const TdtRuntimeWeights & W,
                    const float * __restrict x_input,
                    float * __restrict h_state,
                    float * __restrict c_state,
                    std::vector<float> & scratch,
                    std::vector<float> & layer_input_scratch) {
    const int H = W.H_pred;
    const int L = W.L;
    const int G = 4 * H;
    scratch.resize((size_t) G);
    layer_input_scratch.resize((size_t) H);

    const float * x = x_input;

    for (int layer = 0; layer < L; ++layer) {
        const auto & w = W.host_lstm[layer];
        const float * h_l = h_state + (size_t) layer * H;
        float       * c_l = c_state + (size_t) layer * H;

        gemv_f32(w.w_ih.data(), x, w.b_ih.data(), scratch.data(), G, H);
        for (int i = 0; i < G; ++i) scratch[i] += w.b_hh[i];
        gemv_add_f32(w.w_hh.data(), h_l, scratch.data(), G, H);

        float * h_new = h_state + (size_t) layer * H;
        for (int i = 0; i < H; ++i) {
            const float i_g = sigmoidf(scratch[0 * H + i]);
            const float f_g = sigmoidf(scratch[1 * H + i]);
            const float g_g = std::tanh (scratch[2 * H + i]);
            const float o_g = sigmoidf(scratch[3 * H + i]);
            const float c_new = f_g * c_l[i] + i_g * g_g;
            c_l[i] = c_new;
            h_new[i] = o_g * std::tanh(c_new);
        }
        std::memcpy(layer_input_scratch.data(), h_new, (size_t) H * sizeof(float));
        x = layer_input_scratch.data();
    }
}

// ---- Scalar host-side joint step (CPU fallback) ----
//
// Recomputes joint_enc @ enc_frame per emission step. Profiling showed that
// hoisting this matmul to a full-window precompute regresses on CPU due to
// loss of cache locality for small (~250) windows; the original per-step
// path is faster on M-series CPUs.
// Same caller-owned-scratch pattern as host_lstm_step. gemv_f32
// writes every output byte before any read, so re-using
// `tmp_scratch` across emission steps is byte-equal to the per-call
// allocation.
void host_joint_step(const TdtRuntimeWeights & W,
                     const float * __restrict enc_frame,
                     const float * __restrict pred,
                     std::vector<float> & hidden,
                     std::vector<float> & logits,
                     std::vector<float> & tmp_scratch) {
    const int H  = W.H_joint;
    const int Hp = W.H_pred;
    const int De = W.D_enc;
    const int Vo = W.V_out;

    hidden.resize(H);
    tmp_scratch.resize(H);
    gemv_f32(W.host_joint_enc_w.data(), enc_frame, W.host_joint_enc_b.data(),
             hidden.data(), H, De);
    gemv_f32(W.host_joint_pred_w.data(), pred, W.host_joint_pred_b.data(),
             tmp_scratch.data(), H, Hp);
    for (int i = 0; i < H; ++i) hidden[i] += tmp_scratch[i];
    for (int i = 0; i < H; ++i) hidden[i] = std::max(0.0f, hidden[i]);

    logits.resize(Vo);
    gemv_f32(W.host_joint_out_w.data(), hidden.data(), W.host_joint_out_b.data(),
             logits.data(), Vo, H);
}

// A layer can share one GEMV only when both matrices agree on type and shape
// and the input width is a whole number of quantisation blocks (otherwise the
// row-wise stack would need requantisation).
bool lstm_layer_concatenable(const TdtLstmLayer & w, int H) {
    if (!w.w_ih || !w.w_hh || !w.b_ih || !w.b_hh) return false;
    if (w.w_ih->type != w.w_hh->type) return false;
    if (!ggml_is_contiguous(w.w_ih) || !ggml_is_contiguous(w.w_hh)) return false;
    if (w.w_ih->ne[0] != H || w.w_hh->ne[0] != H) return false;
    if (w.w_ih->ne[1] != 4 * (int64_t) H || w.w_hh->ne[1] != 4 * (int64_t) H) return false;
    if (ggml_nelements(w.b_ih) != 4 * (int64_t) H) return false;
    if (ggml_nelements(w.b_hh) != 4 * (int64_t) H) return false;
    return H % ggml_blck_size(w.w_ih->type) == 0;
}

void upload_lstm_cat_weight(ggml_tensor * dst, const TdtLstmLayer & src) {
    std::vector<uint8_t> ih(ggml_nbytes(src.w_ih));
    std::vector<uint8_t> hh(ggml_nbytes(src.w_hh));
    ggml_backend_tensor_get(src.w_ih, ih.data(), 0, ih.size());
    ggml_backend_tensor_get(src.w_hh, hh.data(), 0, hh.size());

    std::vector<uint8_t> packed(ggml_nbytes(dst));
    pack_lstm_input_rows(src.w_ih->ne[1],
                         ggml_row_size(src.w_ih->type, src.w_ih->ne[0]),
                         ggml_row_size(src.w_hh->type, src.w_hh->ne[0]),
                         ih.data(), hh.data(), packed.data());
    ggml_backend_tensor_set(dst, packed.data(), 0, packed.size());
}

void upload_lstm_cat_bias(ggml_tensor * dst, const TdtLstmLayer & src) {
    std::vector<float> b_ih;
    std::vector<float> b_hh;
    dequantize_to_f32(src.b_ih, b_ih);
    dequantize_to_f32(src.b_hh, b_hh);
    for (size_t i = 0; i < b_ih.size(); ++i) b_ih[i] += b_hh[i];
    ggml_backend_tensor_set(dst, b_ih.data(), 0, b_ih.size() * sizeof(float));
}

void release_lstm_input_concat(TdtRuntimeWeights & W) {
    if (W.lstm_cat_buffer) { ggml_backend_buffer_free(W.lstm_cat_buffer); W.lstm_cat_buffer = nullptr; }
    if (W.lstm_cat_ctx)    { ggml_free(W.lstm_cat_ctx);                  W.lstm_cat_ctx    = nullptr; }
    W.lstm_cat.clear();
}

// The decoder graphs execute directly on the backend, so every stacked GEMV
// must be one the backend accepts with its weight where it was allocated.
bool backend_runs_lstm_cat_gemv(const TdtRuntimeWeights & W) {
    ggml_init_params ip = { ggml_tensor_overhead() * 2 * W.lstm_cat.size(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    bool ok = true;
    for (const TdtLstmCatWeights & cat : W.lstm_cat) {
        ggml_tensor * xh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cat.w->ne[0]);
        ok = ok && ggml_backend_supports_op(W.backend, ggml_mul_mat(ctx, cat.w, xh));
    }
    ggml_free(ctx);
    return ok;
}

// Allocate and fill the per-layer [w_ih | w_hh] stacks on the active backend,
// in the buffer type the loader gives weights there (Hexagon runs quantized
// GEMVs only from its repack buffer). Returns false (leaving concat disabled)
// when any layer is ineligible or the backend cannot run a stacked GEMV.
//
// Measure mode (`measure_bytes != nullptr`, fit projection): the eligibility
// decision and tensor shapes are identical, but the backend buffer is *sized*
// into `*measure_bytes` instead of allocated and no weight data is read
// (uploads dequantise/read the source weights, which a metadata-only model
// cannot). The lstm_cat tensors are marked externally-allocated so the LSTM
// graphs that reference them measure exactly like the real ones.
bool build_lstm_input_concat(TdtRuntimeWeights & W, size_t * measure_bytes = nullptr) {
    const int H = W.H_pred;
    for (int l = 0; l < W.L; ++l) {
        if (!lstm_layer_concatenable(W.weights->lstm[l], H)) return false;
    }

    ggml_init_params p = {};
    p.mem_size   = ggml_tensor_overhead() * (size_t) (2 * W.L + 2);
    p.mem_buffer = nullptr;
    p.no_alloc   = true;
    W.lstm_cat_ctx = ggml_init(p);
    if (!W.lstm_cat_ctx) return false;

    W.lstm_cat.resize(W.L);
    for (int l = 0; l < W.L; ++l) {
        const TdtLstmLayer & src = W.weights->lstm[l];
        W.lstm_cat[l].w = ggml_new_tensor_2d(W.lstm_cat_ctx, src.w_ih->type, 2 * H, 4 * H);
        W.lstm_cat[l].b = ggml_new_tensor_1d(W.lstm_cat_ctx, GGML_TYPE_F32, 4 * H);
        const std::string base = "tdt.lstm_cat." + std::to_string(l);
        ggml_set_name(W.lstm_cat[l].w, (base + ".w").c_str());
        ggml_set_name(W.lstm_cat[l].b, (base + ".b").c_str());
    }

    const ggml_backend_buffer_type_t buft = weight_buffer_type(W.backend);
    if (measure_bytes) {
        *measure_bytes += ggml_backend_alloc_ctx_tensors_from_buft_size(W.lstm_cat_ctx, buft);
        for (ggml_tensor * t = ggml_get_first_tensor(W.lstm_cat_ctx); t;
             t = ggml_get_next_tensor(W.lstm_cat_ctx, t)) {
            if (!t->data && !t->view_src) {
                t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
        }
        return true;
    }

    W.lstm_cat_buffer = ggml_backend_alloc_ctx_tensors_from_buft(W.lstm_cat_ctx, buft);
    if (!W.lstm_cat_buffer || !backend_runs_lstm_cat_gemv(W)) {
        release_lstm_input_concat(W);
        return false;
    }

    for (int l = 0; l < W.L; ++l) {
        upload_lstm_cat_weight(W.lstm_cat[l].w, W.weights->lstm[l]);
        upload_lstm_cat_bias  (W.lstm_cat[l].b, W.weights->lstm[l]);
    }
    return true;
}

// Offsets into the packed [h | c] state row of layer `l`.
size_t state_row_offset(int H, int l)   { return (size_t) l * 2 * H * sizeof(float); }
size_t state_c_offset  (int H, int l)   { return state_row_offset(H, l) + (size_t) H * sizeof(float); }

// Append the LSTM body to `gctx` and return:
//   - cpy nodes that write the freshly computed state into rt.hc_persist in
//     place, and
//   - the last layer's h write, which the joint reads as the fresh pred.
//
// `token_in` must be an i32[1] input tensor in `gctx`. The graph builder
// is shared between the init-only `g_lstm` graph and the fused
// `g_lstm_joint` graph so they stay numerically identical.
struct LstmBodyOuts {
    ggml_tensor * pred_cpy;
    // ALL layers' state-write cpy nodes. Every one must be marked as a graph
    // output and forward-expanded, otherwise ggml_build_forward_expand prunes the
    // writes for layers whose result no downstream node reads (i.e. every layer
    // except the last): those layers' hc_persist rows never update on the
    // device path and the predictor loses its recurrence -> dropped word-final
    // subwords. The host path loops all layers, which is why it stays correct.
    std::vector<ggml_tensor *> state_cpy;
};

// gates = w_ih @ x + b_ih + b_hh + w_hh @ h_prev   ->  [4H]
ggml_tensor * build_lstm_gates_split(ggml_context * gctx,
                                     const TdtLstmLayer & w,
                                     ggml_tensor * x,
                                     ggml_tensor * h_l_in) {
    ggml_tensor * gates = ggml_mul_mat(gctx, w.w_ih, x);
    gates = ggml_add(gctx, gates, w.b_ih);
    gates = ggml_add(gctx, gates, w.b_hh);
    return ggml_add(gctx, gates, ggml_mul_mat(gctx, w.w_hh, h_l_in));
}

// Same gates from one GEMV over the load-time [w_ih | w_hh] stack and the
// pre-summed bias. Cuts three nodes per layer; float summation order differs
// from build_lstm_gates_split, so it is not bit-exact with it.
ggml_tensor * build_lstm_gates_concat(ggml_context * gctx,
                                      const TdtLstmCatWeights & w,
                                      ggml_tensor * x,
                                      ggml_tensor * h_l_in) {
    ggml_tensor * xh = ggml_concat(gctx, x, h_l_in, 0);
    return ggml_add(gctx, ggml_mul_mat(gctx, w.w, xh), w.b);
}

// One ggml_cpy per layer: ggml_lstm_cell already returns [h | c], which is
// exactly the packed state row.
std::vector<ggml_tensor *> write_back_fused_state(ggml_context * gctx,
                                                  ggml_tensor * hc_persist,
                                                  const std::vector<ggml_tensor *> & hc_per_layer,
                                                  int H) {
    std::vector<ggml_tensor *> cpy;
    for (size_t l = 0; l < hc_per_layer.size(); ++l) {
        ggml_tensor * dst = ggml_view_1d(gctx, hc_persist, 2 * H, state_row_offset(H, (int) l));
        cpy.push_back(ggml_cpy(gctx, hc_per_layer[l], dst));
    }
    return cpy;
}

// Decomposed path: h and c land in the two halves of the layer's state row.
std::vector<ggml_tensor *> write_back_split_state(ggml_context * gctx,
                                                  ggml_tensor * hc_persist,
                                                  const std::vector<ggml_tensor *> & h_per_layer,
                                                  const std::vector<ggml_tensor *> & c_per_layer,
                                                  int H) {
    std::vector<ggml_tensor *> cpy;
    for (size_t l = 0; l < h_per_layer.size(); ++l) {
        ggml_tensor * h_dst = ggml_view_1d(gctx, hc_persist, H, state_row_offset(H, (int) l));
        ggml_tensor * c_dst = ggml_view_1d(gctx, hc_persist, H, state_c_offset(H, (int) l));
        cpy.push_back(ggml_cpy(gctx, h_per_layer[l], h_dst));
        cpy.push_back(ggml_cpy(gctx, c_per_layer[l], c_dst));
    }
    return cpy;
}

// Previous h and c for each layer, kept as separate [H] tensors so the LSTM
// body reads them the same way whether they live in hc_persist or in the
// packed rows an unrolled step produced.
struct LstmStateRows {
    std::vector<ggml_tensor *> h;
    std::vector<ggml_tensor *> c;
};

// Fresh per-layer state from one LSTM pass: the fused cell already returns the
// packed [h | c] row, the decomposed path returns the halves separately.
struct LstmCellOuts {
    std::vector<ggml_tensor *> hc;
    std::vector<ggml_tensor *> h;
    std::vector<ggml_tensor *> c;
};

// Per-step select for the fused cell: a layer keeps rows[l] when `update` is
// zero, so the state update needs no blend afterwards.
struct LstmCellMask {
    const std::vector<ggml_tensor *> * rows = nullptr;
    ggml_tensor * update = nullptr;
};

LstmStateRows persist_state_rows(ggml_context * gctx, ggml_tensor * hc_persist, int H, int L) {
    LstmStateRows rows;
    for (int l = 0; l < L; ++l) {
        rows.h.push_back(ggml_view_1d(gctx, hc_persist, H, state_row_offset(H, l)));
        rows.c.push_back(ggml_view_1d(gctx, hc_persist, H, state_c_offset(H, l)));
    }
    return rows;
}

// Same split, over packed [2H] rows that are graph tensors rather than views
// of the persistent buffer.
LstmStateRows packed_state_rows(ggml_context * gctx,
                                const std::vector<ggml_tensor *> & packed, int H) {
    LstmStateRows rows;
    for (ggml_tensor * row : packed) {
        rows.h.push_back(ggml_view_1d(gctx, row, H, 0));
        rows.c.push_back(ggml_view_1d(gctx, row, H, (size_t) H * sizeof(float)));
    }
    return rows;
}

LstmCellOuts build_lstm_cells(TdtRuntimeWeights & rt,
                              ggml_context * gctx,
                              ggml_tensor * token_in,
                              const LstmStateRows & prev,
                              const LstmCellMask & mask = {}) {
    const int H = rt.H_pred;
    const int L = rt.L;

    // Embedding lookup. predict_embed has ne[0]=H, ne[1]=vocab+1; result
    // is [H, 1]. Reshape to [H] for the per-step LSTM input.
    ggml_tensor * x = ggml_get_rows(gctx, rt.graph_embed, token_in);
    x = ggml_reshape_1d(gctx, x, H);

    LstmCellOuts outs;

    for (int l = 0; l < L; ++l) {
        const auto & w = rt.weights->lstm[l];
        ggml_tensor * h_l_in = prev.h[l];
        ggml_tensor * c_l_in = prev.c[l];

        ggml_tensor * gates = rt.concat_lstm_input
                                ? build_lstm_gates_concat(gctx, rt.lstm_cat[l], x, h_l_in)
                                : build_lstm_gates_split(gctx, w, x, h_l_in);

        if (rt.fused_lstm_cell) {
            ggml_tensor * hc = mask.update
                                 ? ggml_lstm_cell_masked(gctx, gates, (*mask.rows)[l], mask.update)
                                 : ggml_lstm_cell(gctx, gates, c_l_in);
            outs.hc.push_back(hc);
            x = ggml_view_1d(gctx, hc, H, 0);
            continue;
        }

        const size_t H_bytes = (size_t) H * sizeof(float);
        ggml_tensor * i_part = ggml_view_1d(gctx, gates, H, 0 * H_bytes);
        ggml_tensor * f_part = ggml_view_1d(gctx, gates, H, 1 * H_bytes);
        ggml_tensor * g_part = ggml_view_1d(gctx, gates, H, 2 * H_bytes);
        ggml_tensor * o_part = ggml_view_1d(gctx, gates, H, 3 * H_bytes);

        ggml_tensor * i_g = ggml_sigmoid(gctx, i_part);
        ggml_tensor * f_g = ggml_sigmoid(gctx, f_part);
        ggml_tensor * g_g = ggml_tanh   (gctx, g_part);
        ggml_tensor * o_g = ggml_sigmoid(gctx, o_part);

        // c_new = f * c_prev + i * g
        ggml_tensor * c_new = ggml_add(gctx,
                                        ggml_mul(gctx, f_g, c_l_in),
                                        ggml_mul(gctx, i_g, g_g));
        // h_new = o * tanh(c_new)
        ggml_tensor * h_new = ggml_mul(gctx, o_g, ggml_tanh(gctx, c_new));

        // ggml_cpy / next-layer mul_mat want contiguous sources. Some
        // intermediates above are view-typed.
        h_new = ggml_cont(gctx, h_new);
        c_new = ggml_cont(gctx, c_new);

        outs.h.push_back(h_new);
        outs.c.push_back(c_new);

        // Next layer feeds on this layer's hidden output.
        x = h_new;
    }

    return outs;
}

LstmBodyOuts build_lstm_body(TdtRuntimeWeights & rt,
                             ggml_context * gctx,
                             ggml_tensor * token_in) {
    const int H = rt.H_pred;

    LstmCellOuts cells = build_lstm_cells(rt, gctx, token_in,
                                          persist_state_rows(gctx, rt.hc_persist, H, rt.L));

    LstmBodyOuts out{};
    out.state_cpy = rt.fused_lstm_cell
                      ? write_back_fused_state(gctx, rt.hc_persist, cells.hc, H)
                      : write_back_split_state(gctx, rt.hc_persist, cells.h, cells.c, H);
    // The joint must read pred through the write-back node, not through
    // rt.pred_persist, so gallocr orders the LSTM update before the joint
    // mat_muls inside one compute_graph.
    out.pred_cpy = rt.fused_lstm_cell
                     ? ggml_view_1d(gctx, out.state_cpy.back(), H, 0)
                     : out.state_cpy[out.state_cpy.size() - 2];
    return out;
}

// Joint-network body. `pred_src` should be the tensor that carries the
// up-to-date pred — pred_persist for the joint-only graph, or the
// `pred_cpy` node returned by build_lstm_body for the fused graph (so
// gallocr orders LSTM-cpy → joint-read correctly within one compute_graph).
//
// Returns the (token_argmax, dur_argmax) pair as i32[1] tensors that
// the host reads out per step.  Logits stay on the backend; only 2 ×
// 4 B comes back to the host instead of V_out × 4 B (~32 KB at V_out
// = 8198).  On Apple unified memory the difference is small (the 32 KB
// readback is ~17 us); on a discrete GPU PCIe bus it's an
// order-of-magnitude saving per emission step (~250 / call).
TransducerGraphOutputs build_joint_body(TdtRuntimeWeights & rt,
                                        ggml_context * gctx,
                                        ggml_tensor * pred_src,
                                        ggml_tensor * frame_idx_in) {
    const int H_joint = rt.H_joint;
    const int V_p1    = rt.V_plus_1;
    const int D_n     = rt.num_durations;

    // pred_proj = W_pred @ pred + b_pred
    ggml_tensor * pred_proj = ggml_mul_mat(gctx, rt.weights->joint_pred_w, pred_src);
    pred_proj = ggml_add(gctx, pred_proj, rt.weights->joint_pred_b);

    // enc_proj_row = enc_proj_persist[frame_idx] -> [H_joint, 1]
    ggml_tensor * enc_proj_row = ggml_get_rows(gctx, rt.enc_proj_persist, frame_idx_in);
    enc_proj_row = ggml_reshape_1d(gctx, enc_proj_row, H_joint);

    // hidden = relu(enc_proj_row + pred_proj)
    ggml_tensor * hidden = ggml_add(gctx, enc_proj_row, pred_proj);
    hidden = ggml_relu(gctx, hidden);

    // logits = W_out @ hidden + b_out -> shape (V_out, 1)
    ggml_tensor * logits = ggml_mul_mat(gctx, rt.weights->joint_out_w, hidden);
    logits = ggml_add(gctx, logits, rt.weights->joint_out_b);

    // ggml_argmax requires a matrix (ne[2] = ne[3] = 1) and reduces along
    // ne[0].  Carve token / duration halves out of the contiguous (V_out,
    // 1) logits tensor as ggml_view_2d slices, force contiguity (the
    // duration slice has a non-zero offset, so its base pointer differs
    // from the logits buffer's start, but its single row is otherwise
    // contiguous; Metal's argmax kernel walks rows by nb01 anyway), then
    // argmax each.
    ggml_tensor * tok_logits = ggml_view_2d(gctx, logits,
                                            V_p1, 1,
                                            (size_t) V_p1 * sizeof(float),
                                            (size_t) 0);
    tok_logits = ggml_cont(gctx, tok_logits);

    if (rt.argmax_on_gpu) {
        return build_transducer_argmax_outputs(gctx, logits, V_p1, D_n);
    }

    TransducerGraphOutputs outs;
    outs.token = tok_logits;
    if (D_n <= 0) return outs;

    ggml_tensor * dur_logits = ggml_view_2d(
        gctx, logits, D_n, 1,
        static_cast<size_t>(D_n) * sizeof(float),
        static_cast<size_t>(V_p1) * sizeof(float));
    outs.duration = ggml_cont(gctx, dur_logits);
    return outs;
}

// (1) Init-only LSTM graph. Used once per call (tdt_init_state) to seed
//     pred_persist after zeroing h/c. The hot loop never dispatches this.
void build_lstm_graph(TdtRuntimeWeights & rt) {
    ggml_context * gctx = rt.gctx;

    rt.lstm_token_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(rt.lstm_token_in, "lstm.token_in");
    ggml_set_input(rt.lstm_token_in);

    LstmBodyOuts outs = build_lstm_body(rt, gctx, rt.lstm_token_in);
    rt.lstm_pred_out = outs.pred_cpy;
    ggml_set_name(rt.lstm_pred_out, "lstm.pred_out");
    // Mark EVERY layer's state write as an output (not just the last), else
    // forward_expand prunes the non-last layers' writes and their state never
    // advances. See LstmBodyOuts::state_cpy.
    for (ggml_tensor * n : outs.state_cpy) ggml_set_output(n);

    rt.g_lstm = ggml_new_graph_custom(gctx, /*size*/ 256, /*grads*/ false);
    for (ggml_tensor * n : outs.state_cpy) ggml_build_forward_expand(rt.g_lstm, n);
}

// (2) Joint-only graph. Used after a blank emission, when pred_persist
//     is unchanged from the previous step. Pred is read straight from the
//     persistent buffer, enc_proj_row is sliced via ggml_get_rows on a
//     host-supplied frame index — only 4 B uploaded per step.  Token +
//     duration argmax are computed on-device so the readback is 2 × 4 B
//     (i32 indices) instead of V_out × 4 B (~32 KB) full logits.
void build_joint_graph(TdtRuntimeWeights & rt) {
    ggml_context * gctx = rt.gctx;

    rt.joint_frame_idx_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(rt.joint_frame_idx_in, "joint.frame_idx_in");
    ggml_set_input(rt.joint_frame_idx_in);

    TransducerGraphOutputs outs =
        build_joint_body(rt, gctx, rt.pred_persist, rt.joint_frame_idx_in);
    rt.joint_token_out = outs.token;
    rt.joint_dur_out   = outs.duration;
    ggml_set_name(rt.joint_token_out, rt.argmax_on_gpu ? "joint.token_argmax" : "joint.token_logits");
    ggml_set_output(rt.joint_token_out);
    if (rt.joint_dur_out) {
        ggml_set_name(rt.joint_dur_out, rt.argmax_on_gpu ? "joint.dur_argmax" : "joint.dur_logits");
        ggml_set_output(rt.joint_dur_out);
    }

    rt.g_joint = ggml_new_graph_custom(gctx, /*size*/ 96, /*grads*/ false);
    ggml_build_forward_expand(rt.g_joint, rt.joint_token_out);
    if (rt.joint_dur_out) {
        ggml_build_forward_expand(rt.g_joint, rt.joint_dur_out);
    }
}

// (3) Fused LSTM + joint graph. Used after a non-blank emission.
//
// Body order:
//   1. LSTM body reads h_persist / c_persist, computes h_new / c_new /
//      pred_new, ggml_cpy writes them back into the persistent buffer.
//   2. Joint body uses the pred_cpy node (which aliases pred_persist's
//      memory but carries the LSTM dependency) so gallocr orders the
//      LSTM update strictly before the joint mat_muls within the same
//      compute_graph commit.
//
// Net effect: one Metal command-buffer commit per non-blank step instead
// of two. For sample-16k.wav (95 non-blank emissions) that's ~95
// commits * ~150 us = ~14 ms saved per call before any compute change.
void build_lstm_joint_graph(TdtRuntimeWeights & rt) {
    ggml_context * gctx = rt.gctx;

    rt.lj_token_in     = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    rt.lj_frame_idx_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(rt.lj_token_in,     "lstm_joint.token_in");
    ggml_set_name(rt.lj_frame_idx_in, "lstm_joint.frame_idx_in");
    ggml_set_input(rt.lj_token_in);
    ggml_set_input(rt.lj_frame_idx_in);

    LstmBodyOuts lstm_outs = build_lstm_body(rt, gctx, rt.lj_token_in);
    // Use the pred_cpy node (not pred_persist directly) so the joint mat_muls
    // depend on the LSTM update finishing first.
    TransducerGraphOutputs joint_outs =
        build_joint_body(rt, gctx, lstm_outs.pred_cpy, rt.lj_frame_idx_in);
    rt.lj_token_out = joint_outs.token;
    rt.lj_dur_out   = joint_outs.duration;
    ggml_set_name(rt.lj_token_out, rt.argmax_on_gpu ? "lstm_joint.token_argmax" : "lstm_joint.token_logits");
    ggml_set_output(rt.lj_token_out);
    if (rt.lj_dur_out) {
        ggml_set_name(rt.lj_dur_out, rt.argmax_on_gpu ? "lstm_joint.dur_argmax" : "lstm_joint.dur_logits");
        ggml_set_output(rt.lj_dur_out);
    }
    // Mark EVERY layer's state write as an output so gallocr keeps them alive
    // (their memory IS hc_persist). Without this, forward_expand prunes the
    // dead-end intermediate writes for all but the last layer, so those layers'
    // recurrent state never advances on the device path -> dropped word-final
    // subwords. See LstmBodyOuts::state_cpy.
    for (ggml_tensor * n : lstm_outs.state_cpy) ggml_set_output(n);

    rt.g_lstm_joint = ggml_new_graph_custom(gctx, /*size*/ 384, /*grads*/ false);
    ggml_build_forward_expand(rt.g_lstm_joint, rt.lj_token_out);
    if (rt.lj_dur_out) {
        ggml_build_forward_expand(rt.g_lstm_joint, rt.lj_dur_out);
    }
    for (ggml_tensor * n : lstm_outs.state_cpy) ggml_build_forward_expand(rt.g_lstm_joint, n);
}

// ---- (4) unrolled K-step decode graph ----
//
// Every greedy step is joint -> (token, duration) argmax -> LSTM over the
// emitted token -> ggml_tdt_step control -> masked state update. Chaining K of
// them in one graph replaces K command-buffer commits and K host readbacks with
// one of each; steps that fall past the window, and steps that decoded a blank,
// carry a zero update mask and leave the state as it was. The path needs the
// fused LSTM cell: its masked form is what keeps a held step bit-exact.

// Tensor-slot allowances per unrolled step, several times what each part
// builds: the joint body (two GEMVs, biases, relu, views, argmax), one fused
// LSTM layer (GEMV, gates, cell, views, state copy) and the control op with
// its destination row and slot views.
constexpr size_t k_unroll_slots_joint     = 64;
constexpr size_t k_unroll_slots_per_layer = 48;
constexpr size_t k_unroll_slots_control   = 32;
constexpr size_t k_unroll_slots_base      = 64;

size_t unroll_slots_per_step(int L) {
    return k_unroll_slots_joint + k_unroll_slots_per_layer * (size_t) L + k_unroll_slots_control;
}

ggml_tensor * step_out_view(ggml_context * ctx, ggml_tensor * ctl, int slot) {
    return ggml_view_1d(ctx, ctl, 1, (size_t) slot * sizeof(int32_t));
}

// Row `row` of the launch control buffer: row 0 carries the counters the host
// uploads, row k+1 is where step k writes its result.
ggml_tensor * ctl_row(ggml_context * ctx, ggml_tensor * ctl, int row, int slots) {
    return ggml_view_1d(ctx, ctl, slots,
                        (size_t) row * GGML_TDT_STEP_N_OUTS * sizeof(int32_t));
}

// hc_persist's rows as packed [2H] tensors: the state the first step reads.
std::vector<ggml_tensor *> persist_packed_rows(ggml_context * ctx, ggml_tensor * hc_persist,
                                               int H, int L) {
    std::vector<ggml_tensor *> rows;
    for (int l = 0; l < L; ++l) {
        rows.push_back(ggml_view_1d(ctx, hc_persist, 2 * H, state_row_offset(H, l)));
    }
    return rows;
}

std::vector<ggml_tensor *> write_back_packed_rows(ggml_context * ctx, ggml_tensor * hc_persist,
                                                  const std::vector<ggml_tensor *> & rows, int H) {
    std::vector<ggml_tensor *> cpy;
    for (size_t l = 0; l < rows.size(); ++l) {
        ggml_tensor * dst = ggml_view_1d(ctx, hc_persist, 2 * H, state_row_offset(H, (int) l));
        cpy.push_back(ggml_cpy(ctx, rows[l], dst));
    }
    return cpy;
}

struct UnrollStepOuts {
    ggml_tensor * ctl           = nullptr;  // i32[GGML_TDT_STEP_N_OUTS] control row
    ggml_tensor * counters_next = nullptr;  // i32[GGML_TDT_STEP_N_INS]
    ggml_tensor * frame_next    = nullptr;  // i32[1]
    std::vector<ggml_tensor *> state;       // packed [2H] per layer
};

UnrollStepOuts build_unroll_step(TdtRuntimeWeights & rt, ggml_context * ctx,
                                 const std::vector<ggml_tensor *> & state_in,
                                 ggml_tensor * counters_in,
                                 ggml_tensor * frame_idx_in,
                                 ggml_tensor * ctl_dst,
                                 int blank_id, int max_symbols) {
    const int H = rt.H_pred;

    UnrollStepOuts out;
    ggml_tensor * pred = ggml_view_1d(ctx, state_in.back(), H, 0);
    TransducerGraphOutputs joint = build_joint_body(rt, ctx, pred, frame_idx_in);

    // The control op copies the decoded pair into its own row, so nothing else
    // has to write the token and duration index back for the host.
    ggml_tensor * ctl = ggml_tdt_step(ctx, joint.token,
                                      joint.duration ? joint.duration : rt.zero_dur_idx,
                                      counters_in, rt.dur_table, ctl_dst,
                                      blank_id, max_symbols, rt.rnnt_mode ? 1 : 0);

    // The masked cell selects held vs fresh inside the kernel, so the packed
    // [h | c] rows it returns are the next state as is.
    LstmCellMask mask;
    mask.rows   = &state_in;
    mask.update = step_out_view(ctx, ctl, GGML_TDT_STEP_OUT_UPDATE);
    LstmCellOuts cells = build_lstm_cells(rt, ctx, joint.token,
                                          packed_state_rows(ctx, state_in, H), mask);

    out.ctl           = ctl;
    out.state         = cells.hc;
    out.counters_next = ctl_row(ctx, ctl, 0, GGML_TDT_STEP_N_INS);
    out.frame_next    = step_out_view(ctx, ctl, GGML_TDT_STEP_OUT_FRAME);
    return out;
}

void free_unroll_graph(TdtRuntimeWeights & rt) {
    if (rt.alloc_unroll) { ggml_gallocr_free(rt.alloc_unroll); rt.alloc_unroll = nullptr; }
    if (rt.unroll_ctx)   { ggml_free(rt.unroll_ctx);           rt.unroll_ctx   = nullptr; }
    rt.g_unroll        = nullptr;
    rt.un_ctl          = nullptr;
    rt.unroll_blank_id    = -1;
    rt.unroll_max_symbols = 0;
}

bool graph_runs_on_backend(ggml_backend_t backend, ggml_cgraph * cg) {
    for (int i = 0; i < ggml_graph_n_nodes(cg); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(cg, i))) {
            return false;
        }
    }
    return true;
}

// Chain K steps, each one writing its control row into un_ctl, and return the
// per-layer state rows the last step left. Step k reads row k and writes row
// k + 1, so the host uploads one row and reads the rest back in one transfer.
std::vector<ggml_tensor *> chain_unroll_steps(TdtRuntimeWeights & rt, ggml_context * ctx,
                                              int K, int blank_id, int max_symbols,
                                              ggml_tensor ** last_ctl) {
    std::vector<ggml_tensor *> state = persist_packed_rows(ctx, rt.hc_persist, rt.H_pred, rt.L);
    ggml_tensor * counters = ctl_row(ctx, rt.un_ctl, 0, GGML_TDT_STEP_N_INS);
    ggml_tensor * frame    = step_out_view(ctx, counters, GGML_TDT_STEP_IN_T);

    for (int k = 0; k < K; ++k) {
        UnrollStepOuts step = build_unroll_step(rt, ctx, state, counters, frame,
                                                ctl_row(ctx, rt.un_ctl, k + 1,
                                                        GGML_TDT_STEP_N_OUTS),
                                                blank_id, max_symbols);
        state     = std::move(step.state);
        counters  = step.counters_next;
        frame     = step.frame_next;
        *last_ctl = step.ctl;
    }
    return state;
}

void expand_unroll_outputs(TdtRuntimeWeights & rt, ggml_tensor * last_ctl,
                           const std::vector<ggml_tensor *> & state_writes) {
    for (ggml_tensor * n : state_writes) ggml_build_forward_expand(rt.g_unroll, n);
    ggml_build_forward_expand(rt.g_unroll, last_ctl);
}

// `measure_bytes` (fit projection): when non-null the graph is built and its
// compute buffer *sized* into `*measure_bytes` instead of allocated, then the
// graph is freed again (rt keeps no unroll state).
bool build_unroll_graph(TdtRuntimeWeights & rt, int blank_id, int max_symbols,
                        size_t * measure_bytes = nullptr) {
    free_unroll_graph(rt);

    const int K = rt.unroll_steps;
    const int H = rt.H_pred;

    const size_t slots = unroll_slots_per_step(rt.L) * (size_t) K + k_unroll_slots_base;
    ggml_init_params p = {};
    p.mem_size   = ggml_tensor_overhead() * slots + ggml_graph_overhead_custom(slots, false);
    p.mem_buffer = nullptr;
    p.no_alloc   = true;
    rt.unroll_ctx = ggml_init(p);
    if (!rt.unroll_ctx) return false;

    ggml_context * ctx = rt.unroll_ctx;
    rt.un_ctl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, GGML_TDT_STEP_N_OUTS, K + 1);
    ggml_set_name(rt.un_ctl, "unroll.ctl");
    ggml_set_input(rt.un_ctl);
    ggml_set_output(rt.un_ctl);

    // Single write-back: the intermediate steps keep their state in graph
    // tensors, so hc_persist is touched once per graph.
    ggml_tensor * last_ctl = nullptr;
    std::vector<ggml_tensor *> state_writes = write_back_packed_rows(
        ctx, rt.hc_persist, chain_unroll_steps(rt, ctx, K, blank_id, max_symbols, &last_ctl), H);
    for (ggml_tensor * n : state_writes) ggml_set_output(n);

    rt.g_unroll = ggml_new_graph_custom(ctx, slots, /*grads*/ false);
    expand_unroll_outputs(rt, last_ctl, state_writes);

    if (!graph_runs_on_backend(rt.backend, rt.g_unroll)) {
        free_unroll_graph(rt);
        return false;
    }

    if (measure_bytes) {
        *measure_bytes = 0;
        ggml_gallocr_t pricer =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        if (pricer) {
            ggml_gallocr_reserve_n_size(pricer, rt.g_unroll, nullptr, nullptr, measure_bytes);
            ggml_gallocr_free(pricer);
        }
        free_unroll_graph(rt);
        return true;
    }

    rt.alloc_unroll = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
    if (!rt.alloc_unroll || !ggml_gallocr_alloc_graph(rt.alloc_unroll, rt.g_unroll)) {
        free_unroll_graph(rt);
        return false;
    }

    rt.unroll_blank_id    = blank_id;
    rt.unroll_max_symbols = max_symbols;
    return true;
}

// Build (or rebuild) the unrolled graph for this decode's blank id and symbol
// cap. Leaves g_unroll null when the path does not apply, and the sequential
// per-step loop runs instead.
void ensure_unroll_graph(TdtRuntimeWeights & rt, int blank_id, int max_symbols) {
    if (!rt.use_graphs || !rt.fused_lstm_cell || !rt.argmax_on_gpu || rt.unroll_steps < 1 ||
        !rt.dur_table || !rt.zero_dur_idx || max_symbols < 1) {
        free_unroll_graph(rt);
        return;
    }
    if (rt.g_unroll && rt.unroll_blank_id == blank_id && rt.unroll_max_symbols == max_symbols) {
        return;
    }
    build_unroll_graph(rt, blank_id, max_symbols);
}

// Build the full-window encoder-side projection graph for a given frame
// count. Result is ggml_cpy'd straight into rt.enc_proj_persist[:T] so
// per-step joint reads can ggml_get_rows on the persistent buffer
// without any host roundtrip.
//
// Each call allocates its OWN ggml_context (`g.ctx`) sized for the
// ~32 graph nodes this builder produces.  Previous design parented
// these on `rt.gctx`, and the LRU eviction below freed only the
// gallocr — which leaked ~32 gctx slots per evicted entry.  Owning
// the metadata locally and freeing it at eviction keeps streaming
// callers (Mode 3 with varying right-lookahead-ms → many distinct
// T_enc) bounded by the LRU cap regardless of distinct-T churn.
// `measure_bytes` (fit projection): when non-null the graph is built and its
// compute buffer *sized* into `*measure_bytes` instead of allocated; an empty
// EncProjGraph is returned.
TdtRuntimeWeights::EncProjGraph build_enc_proj_graph(TdtRuntimeWeights & rt, int T,
                                                     size_t * measure_bytes = nullptr) {
    TdtRuntimeWeights::EncProjGraph g{};
    g.T = T;

    const int H_joint = rt.H_joint;
    const int D_enc   = rt.D_enc;

    // ~32 graph slots is enough for: 1 input tensor + 1 mul_mat + 1 add +
    // 1 view + 1 cpy + scratch ≈ 6 nodes.  Round up to 64 for headroom.
    const size_t graph_slots = 64;
    const size_t local_overhead = ggml_tensor_overhead() * graph_slots
                                + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params local_p = {};
    local_p.mem_size   = local_overhead;
    local_p.mem_buffer = nullptr;
    local_p.no_alloc   = true;
    g.ctx = ggml_init(local_p);
    if (!g.ctx) {
        std::fprintf(stderr, "tdt: enc_proj ggml_init failed for T=%d\n", T);
        return g;
    }

    g.enc_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D_enc, T);
    ggml_set_name(g.enc_in, "enc_proj.enc_in");
    ggml_set_input(g.enc_in);

    ggml_tensor * proj = ggml_mul_mat(g.ctx, rt.weights->joint_enc_w, g.enc_in);
    proj = ggml_add(g.ctx, proj, rt.weights->joint_enc_b);

    ggml_tensor * dst_view = ggml_view_2d(g.ctx, rt.enc_proj_persist,
                                           H_joint, T,
                                           (size_t) H_joint * sizeof(float),
                                           0);
    g.out = ggml_cpy(g.ctx, proj, dst_view);
    ggml_set_name(g.out, "enc_proj.out_persist");
    ggml_set_output(g.out);

    g.cg = ggml_new_graph_custom(g.ctx, /*size*/ 32, /*grads*/ false);
    ggml_build_forward_expand(g.cg, g.out);

    if (measure_bytes) {
        *measure_bytes = 0;
        ggml_gallocr_t pricer = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        if (pricer) {
            ggml_gallocr_reserve_n_size(pricer, g.cg, nullptr, nullptr, measure_bytes);
            ggml_gallocr_free(pricer);
        }
        ggml_free(g.ctx);
        return TdtRuntimeWeights::EncProjGraph{};
    }

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
    if (!g.alloc || !ggml_gallocr_alloc_graph(g.alloc, g.cg)) {
        std::fprintf(stderr, "tdt: failed to allocate enc_proj graph for T=%d\n", T);
        if (g.alloc) ggml_gallocr_free(g.alloc);
        g.alloc = nullptr;
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }

    return g;
}

void free_enc_proj_graph(TdtRuntimeWeights::EncProjGraph & g) {
    if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
    if (g.ctx)   { ggml_free(g.ctx);           g.ctx   = nullptr; }
    g.cg = nullptr;
    g.enc_in = nullptr;
    g.out = nullptr;
}

const TdtRuntimeWeights::EncProjGraph * get_enc_proj_graph(TdtRuntimeWeights & rt, int T) {
    for (auto & g : rt.enc_proj_cache) {
        if (g.T == T) return &g;
    }
    if (rt.enc_proj_cache.size() >= TdtRuntimeWeights::k_enc_proj_cache_max) {
        // LRU evict: free both the gallocr's backend buffer AND the
        // local ggml_context that owns the cgraph + tensor metadata.
        free_enc_proj_graph(rt.enc_proj_cache.front());
        rt.enc_proj_cache.erase(rt.enc_proj_cache.begin());
    }
    rt.enc_proj_cache.push_back(build_enc_proj_graph(rt, T));
    return &rt.enc_proj_cache.back();
}

bool tdt_host_decode_forced() {
    const char * v = std::getenv("PARAKEET_TDT_HOST_DECODE");
    return v && v[0] == '1';
}

bool compute_graph(TdtRuntimeWeights & rt, ggml_cgraph * cg) {
    if (rt.n_threads > 0 && backend_is_cpu(rt.backend)) {
        backend_set_n_threads(rt.backend, rt.n_threads);
    }
    return ggml_backend_graph_compute(rt.backend, cg) == GGML_STATUS_SUCCESS;
}

}  // anonymous namespace

TdtRuntimeWeights::TdtRuntimeWeights(TdtRuntimeWeights && o) noexcept { *this = std::move(o); }

TdtRuntimeWeights & TdtRuntimeWeights::operator=(TdtRuntimeWeights && o) noexcept {
    if (this == &o) return *this;
    release();

    H_pred       = o.H_pred;
    H_joint      = o.H_joint;
    D_enc        = o.D_enc;
    V_plus_1     = o.V_plus_1;
    V_out        = o.V_out;
    L            = o.L;
    num_durations = o.num_durations;
    weights      = o.weights;        o.weights = nullptr;
    graph_embed  = o.graph_embed;    o.graph_embed = nullptr;
    backend      = o.backend;        o.backend = nullptr;
    n_threads    = o.n_threads;
    use_graphs   = o.use_graphs;
    fused_lstm_cell = o.fused_lstm_cell;
    argmax_on_gpu   = o.argmax_on_gpu;
    embed        = std::move(o.embed);
    host_lstm    = std::move(o.host_lstm);
    host_joint_enc_w  = std::move(o.host_joint_enc_w);
    host_joint_enc_b  = std::move(o.host_joint_enc_b);
    host_joint_pred_w = std::move(o.host_joint_pred_w);
    host_joint_pred_b = std::move(o.host_joint_pred_b);
    host_joint_out_w  = std::move(o.host_joint_out_w);
    host_joint_out_b  = std::move(o.host_joint_out_b);
    concat_lstm_input = o.concat_lstm_input;
    gctx         = o.gctx;           o.gctx = nullptr;
    lstm_cat_ctx    = o.lstm_cat_ctx;    o.lstm_cat_ctx    = nullptr;
    lstm_cat_buffer = o.lstm_cat_buffer; o.lstm_cat_buffer = nullptr;
    lstm_cat        = std::move(o.lstm_cat);
    o.lstm_cat.clear();
    persist_ctx     = o.persist_ctx;     o.persist_ctx = nullptr;
    persist_buffer  = o.persist_buffer;  o.persist_buffer = nullptr;
    hc_persist       = o.hc_persist;       o.hc_persist = nullptr;
    pred_persist     = o.pred_persist;     o.pred_persist = nullptr;
    enc_proj_persist = o.enc_proj_persist; o.enc_proj_persist = nullptr;
    enc_proj_T_max   = o.enc_proj_T_max;
    g_lstm       = o.g_lstm;         o.g_lstm = nullptr;
    alloc_lstm   = o.alloc_lstm;     o.alloc_lstm = nullptr;
    lstm_token_in = o.lstm_token_in; o.lstm_token_in = nullptr;
    lstm_pred_out = o.lstm_pred_out; o.lstm_pred_out = nullptr;
    g_joint      = o.g_joint;        o.g_joint = nullptr;
    alloc_joint  = o.alloc_joint;    o.alloc_joint = nullptr;
    joint_frame_idx_in = o.joint_frame_idx_in; o.joint_frame_idx_in = nullptr;
    joint_token_out    = o.joint_token_out;    o.joint_token_out = nullptr;
    joint_dur_out      = o.joint_dur_out;      o.joint_dur_out = nullptr;
    g_lstm_joint     = o.g_lstm_joint;     o.g_lstm_joint = nullptr;
    alloc_lstm_joint = o.alloc_lstm_joint; o.alloc_lstm_joint = nullptr;
    lj_token_in     = o.lj_token_in;     o.lj_token_in = nullptr;
    lj_frame_idx_in = o.lj_frame_idx_in; o.lj_frame_idx_in = nullptr;
    lj_token_out    = o.lj_token_out;    o.lj_token_out = nullptr;
    lj_dur_out      = o.lj_dur_out;      o.lj_dur_out = nullptr;
    rnnt_mode          = o.rnnt_mode;
    unroll_steps       = o.unroll_steps;
    unroll_blank_id    = o.unroll_blank_id;    o.unroll_blank_id    = -1;
    unroll_max_symbols = o.unroll_max_symbols; o.unroll_max_symbols = 0;
    dur_table       = o.dur_table;       o.dur_table = nullptr;
    zero_dur_idx    = o.zero_dur_idx;    o.zero_dur_idx = nullptr;
    unroll_ctx      = o.unroll_ctx;      o.unroll_ctx = nullptr;
    g_unroll        = o.g_unroll;        o.g_unroll = nullptr;
    alloc_unroll    = o.alloc_unroll;    o.alloc_unroll = nullptr;
    un_ctl          = o.un_ctl;          o.un_ctl = nullptr;
    enc_proj_cache = std::move(o.enc_proj_cache);
    o.enc_proj_cache.clear();
    return *this;
}

TdtRuntimeWeights::~TdtRuntimeWeights() { release(); }

// Frees every ggml handle and empties the caches; the C++ members stay alive so
// the object can be assigned into afterwards.
void TdtRuntimeWeights::release() {
    for (auto & g : enc_proj_cache) {
        free_enc_proj_graph(g);
    }
    enc_proj_cache.clear();
    lstm_cat.clear();
    if (lstm_cat_buffer) { ggml_backend_buffer_free(lstm_cat_buffer); lstm_cat_buffer = nullptr; }
    if (lstm_cat_ctx)    { ggml_free(lstm_cat_ctx);                  lstm_cat_ctx    = nullptr; }
    if (alloc_unroll) { ggml_gallocr_free(alloc_unroll); alloc_unroll = nullptr; }
    if (unroll_ctx)   { ggml_free(unroll_ctx);           unroll_ctx   = nullptr; }
    if (alloc_lstm_joint) { ggml_gallocr_free(alloc_lstm_joint); alloc_lstm_joint = nullptr; }
    if (alloc_joint) { ggml_gallocr_free(alloc_joint); alloc_joint = nullptr; }
    if (alloc_lstm)  { ggml_gallocr_free(alloc_lstm);  alloc_lstm  = nullptr; }
    if (persist_buffer) { ggml_backend_buffer_free(persist_buffer); persist_buffer = nullptr; }
    if (persist_ctx) { ggml_free(persist_ctx); persist_ctx = nullptr; }
    graph_embed = nullptr;
    if (gctx)        { ggml_free(gctx);                gctx        = nullptr; }
    // backend is owned by ParakeetCtcModel::Impl; don't free here.
}

// Whether the joint's argmax over the token and duration logits runs on the backend.
bool backend_runs_transducer_argmax(ggml_backend_t backend, int V_out, int V_p1, int D_n) {
    if (!backend) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, V_out, 1);
    const bool ok = ggml_backend_supports_op(backend, build_transducer_argmax_outputs(ctx, logits, V_p1, D_n).token);
    ggml_free(ctx);
    return ok;
}

bool backend_runs_lstm_cell(ggml_backend_t backend, int H) {
    if (!backend) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * gates = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * H);
    ggml_tensor * c     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    const bool ok = ggml_backend_supports_op(backend, ggml_lstm_cell(ctx, gates, c));
    ggml_free(ctx);
    return ok;
}

bool backend_runs_concat(ggml_backend_t backend, int H) {
    if (!backend) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    const bool ok = ggml_backend_supports_op(backend, ggml_concat(ctx, a, b, 0));
    ggml_free(ctx);
    return ok;
}

// Frame advance per duration index, exactly as the host applies it: an empty
// GGUF duration list means the index itself is the advance.
std::vector<int32_t> build_duration_table(const ParakeetCtcModel & model, int num_durations) {
    std::vector<int32_t> table((size_t) std::max(1, num_durations));
    for (size_t i = 0; i < table.size(); ++i) {
        table[i] = model.tdt_durations.empty() ? (int32_t) i : (int32_t) model.tdt_durations[i];
    }
    return table;
}

// Shared body of tdt_prepare_runtime and tdt_measure_runtime. When `measure`
// is non-null every allocation the real path makes is sized into `measure`
// instead of performed (fit projection; requires a metadata-only model whose
// weight tensors are marked externally-allocated), and W is left non-ready.
// `measure_enc_frames` sizes the worst-case per-window enc-projection graph.
static int tdt_prepare_runtime_impl(const ParakeetCtcModel & model, TdtRuntimeWeights & W,
                                    bool allow_fused_lstm, bool allow_concat_lstm,
                                    DecoderFitMeasure * measure, int measure_enc_frames) {
    W = TdtRuntimeWeights{};

    const bool is_nemotron =
        model.model_type == ParakeetModelType::NEMOTRON;
    const bool is_rnnt =
        model.model_type == ParakeetModelType::RNNT || is_nemotron;
    if (!is_rnnt && model.model_type != ParakeetModelType::TDT) {
        std::fprintf(
            stderr,
            "tdt_prepare_runtime: expected RNNT, Nemotron, or TDT model\n");
        return 1;
    }
    W.H_pred = is_nemotron
        ? model.nemotron_cfg.pred_hidden
        : (is_rnnt
            ? model.encoder_cfg.rnnt_pred_hidden
            : model.encoder_cfg.tdt_pred_hidden);
    W.H_joint = is_nemotron
        ? model.nemotron_cfg.joint_hidden
        : (is_rnnt
            ? model.encoder_cfg.rnnt_joint_hidden
            : model.encoder_cfg.tdt_joint_hidden);
    W.D_enc         = model.encoder_cfg.d_model;
    W.L = is_nemotron
        ? model.nemotron_cfg.pred_rnn_layers
        : (is_rnnt
            ? model.encoder_cfg.rnnt_pred_rnn_layers
            : model.encoder_cfg.tdt_pred_rnn_layers);
    W.num_durations = is_rnnt ? 0 : model.encoder_cfg.tdt_num_durations;
    W.rnnt_mode     = is_rnnt;
    W.V_plus_1      = (int) model.vocab_size + 1;
    W.V_out         = W.V_plus_1 + W.num_durations;

    W.weights = is_nemotron
        ? &model.nemotron.rnnt
        : (is_rnnt ? &model.rnnt : &model.tdt);
    W.backend = model.backend_active();
    if (!W.backend) {
        std::fprintf(stderr, "tdt_prepare_runtime: model has no active backend (call load_from_gguf first)\n");
        return 1;
    }

    // ggml-opencl drops the in-place ggml_cpy writes that advance the LSTM
    // state, so with the encoder on OpenCL the TDT / RNN-T decode graphs run
    // on the CPU backend from the loader's CPU-resident weight copies (the
    // measure pass sizes them from the originals' shapes). Nemotron keeps the
    // host loop below.
    if (!is_nemotron && backend_is_opencl(W.backend) && model.backend_cpu() &&
        (measure || model_tdt_decoder_on_cpu(model))) {
        W.backend = model.backend_cpu();
        if (!measure) W.weights = &model.tdt_cpu;
    }

    // The CPU backend keeps the thread count the encoder configured at load;
    // compute_graph only overrides it when n_threads is set explicitly.
    W.n_threads = 0;

    if (!W.weights->predict_embed || W.weights->lstm.empty() ||
        !W.weights->joint_out_w) {
        std::fprintf(stderr, "tdt_prepare_runtime: GGUF is missing transducer tensors\n");
        return 2;
    }

    // The ggml decode graphs run on every backend, CPU included: the quantised
    // joint matmuls beat the host f32 gemv loop by an order of magnitude there.
    // PARAKEET_TDT_HOST_DECODE=1 keeps the host loop for parity testing.
    W.use_graphs = !tdt_host_decode_forced();

    // ggml-opencl drops the in-place ggml_cpy writes that update the TDT LSTM
    // persistent state (h/c/pred), so the state never advances and the decode
    // emits one constant token per frame. Run the per-step decode on the host on
    // OpenCL; the encoder still runs on the GPU. EOU takes the same host-side
    // fallback on OpenCL (see parakeet_eou.cpp); Sortformer doesn't use this
    // persistent-state pattern and stays on the GPU. This also applies to
    // Nemotron, which shares this decode path via tdt_prepare_runtime.
    if (W.use_graphs && backend_is_opencl(W.backend)) {
        W.use_graphs = false;
    }
    W.fused_lstm_cell = W.use_graphs && allow_fused_lstm && backend_runs_lstm_cell(W.backend, W.H_pred);
    W.argmax_on_gpu   = W.use_graphs && backend_runs_transducer_argmax(W.backend, W.V_out, W.V_plus_1, W.num_durations);

    // The batched Core ML decoder looks tokens up on the host whichever path
    // the sequential decoder takes (a gather is not an ANE op).
    if (!measure && model_coreml_decoder(model) != nullptr) {
        dequantize_to_f32(W.weights->predict_embed, W.embed);
    }

    if (!W.use_graphs) {
        if (measure) {
            // ---- CPU fallback, sized: the dequantised host f32 copies ----
            auto f32_bytes = [](const ggml_tensor * t) {
                return t ? (size_t) ggml_nelements(t) * sizeof(float) : (size_t) 0;
            };
            measure->host_bytes += f32_bytes(W.weights->predict_embed);
            for (const TdtLstmLayer & lyr : W.weights->lstm) {
                measure->host_bytes += f32_bytes(lyr.w_ih) + f32_bytes(lyr.w_hh)
                                     + f32_bytes(lyr.b_ih) + f32_bytes(lyr.b_hh);
            }
            measure->host_bytes += f32_bytes(W.weights->joint_enc_w)
                                 + f32_bytes(W.weights->joint_enc_b)
                                 + f32_bytes(W.weights->joint_pred_w)
                                 + f32_bytes(W.weights->joint_pred_b)
                                 + f32_bytes(W.weights->joint_out_w)
                                 + f32_bytes(W.weights->joint_out_b);
            return 0;
        }
        // ---- CPU fallback: dequantise weights to host f32 ----
        if (W.embed.empty()) {
            dequantize_to_f32(W.weights->predict_embed, W.embed);
        }
        W.host_lstm.clear();
        W.host_lstm.resize(W.L);
        for (int l = 0; l < W.L; ++l) {
            dequantize_to_f32(W.weights->lstm[l].w_ih, W.host_lstm[l].w_ih);
            dequantize_to_f32(W.weights->lstm[l].w_hh, W.host_lstm[l].w_hh);
            dequantize_to_f32(W.weights->lstm[l].b_ih, W.host_lstm[l].b_ih);
            dequantize_to_f32(W.weights->lstm[l].b_hh, W.host_lstm[l].b_hh);
        }
        dequantize_to_f32(W.weights->joint_enc_w,  W.host_joint_enc_w);
        dequantize_to_f32(W.weights->joint_enc_b,  W.host_joint_enc_b);
        dequantize_to_f32(W.weights->joint_pred_w, W.host_joint_pred_w);
        dequantize_to_f32(W.weights->joint_pred_b, W.host_joint_pred_b);
        dequantize_to_f32(W.weights->joint_out_w,  W.host_joint_out_w);
        dequantize_to_f32(W.weights->joint_out_b,  W.host_joint_out_b);
        return 0;
    }

    // ---- GPU path: build ggml graphs against native GGUF weight tensors ----
    //
    // Two ggml_contexts:
    //   (a) persist_ctx + persist_buffer hold the per-call decoder state
    //       (h, c, pred, enc_proj) so all three graphs can read/write them
    //       in place without host roundtrips.
    //   (b) gctx holds graph-node metadata for the three fixed-shape graphs
    //       (g_lstm init, g_joint blank-path, g_lstm_joint non-blank fused
    //       path) and the dynamic enc_proj_cache. gallocrs allocate the
    //       transient compute buffers for each graph.
    {
        ggml_init_params pp = {};
        pp.mem_size   = ggml_tensor_overhead() * 16 + 4 * 1024;
        pp.mem_buffer = nullptr;
        pp.no_alloc   = true;
        W.persist_ctx = ggml_init(pp);
        if (!W.persist_ctx) {
            std::fprintf(stderr, "tdt_prepare_runtime: persist ggml_init failed\n");
            return 3;
        }

        const int H_pred  = W.H_pred;
        const int H_joint = W.H_joint;
        const int L       = W.L;
        const int T_max   = TdtRuntimeWeights::k_enc_proj_T_max;

        W.graph_embed = W.weights->predict_embed;
        if (!backend_runs_embedding_lookup(W.backend, W.graph_embed)) {
            // Hexagon GET_ROWS supports F32 only. Convert once at preparation
            // so the recurrent decoder can keep executing on the backend.
            W.graph_embed = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32,
                                              W.weights->predict_embed->ne[0], W.weights->predict_embed->ne[1]);
            ggml_set_name(W.graph_embed, "tdt.predict.embed.f32");
            if (!backend_runs_embedding_lookup(W.backend, W.graph_embed)) {
                std::fprintf(stderr, "tdt_prepare_runtime: backend cannot read F32 embeddings\n");
                return 3;
            }
        }

        W.hc_persist       = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, 2 * H_pred, L);
        // Pred is the last layer's h, so it is a view rather than its own tensor:
        // the LSTM write-back updates it for free.
        W.pred_persist     = ggml_view_1d(W.persist_ctx, W.hc_persist, H_pred,
                                          (size_t) (L - 1) * 2 * H_pred * sizeof(float));
        W.enc_proj_persist = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, H_joint, T_max);
        ggml_set_name(W.hc_persist,       "tdt.hc_persist");
        ggml_set_name(W.pred_persist,     "tdt.pred_persist");
        ggml_set_name(W.enc_proj_persist, "tdt.enc_proj_persist");

        // Control-op inputs. Left null when the GGUF duration list is shorter
        // than the duration head, which keeps the unrolled decode graph off.
        if (model.tdt_durations.empty() ||
            (int) model.tdt_durations.size() >= W.num_durations) {
            W.dur_table    = ggml_new_tensor_1d(W.persist_ctx, GGML_TYPE_I32,
                                                std::max(1, W.num_durations));
            W.zero_dur_idx = ggml_new_tensor_1d(W.persist_ctx, GGML_TYPE_I32, 1);
            ggml_set_name(W.dur_table,    "tdt.dur_table");
            ggml_set_name(W.zero_dur_idx, "tdt.zero_dur_idx");
        }

        if (measure) {
            // Size the persistent state, then mark it externally-allocated so
            // the graph measurements below don't count it a second time.
            measure->device_state_bytes += ggml_backend_alloc_ctx_tensors_from_buft_size(
                W.persist_ctx, ggml_backend_get_default_buffer_type(W.backend));
            for (ggml_tensor * t = ggml_get_first_tensor(W.persist_ctx); t;
                 t = ggml_get_next_tensor(W.persist_ctx, t)) {
                if (!t->data && !t->view_src) {
                    t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
                }
            }
        } else {
            W.persist_buffer = ggml_backend_alloc_ctx_tensors(W.persist_ctx, W.backend);
            if (!W.persist_buffer) {
                std::fprintf(stderr, "tdt_prepare_runtime: failed to allocate persistent state buffer\n");
                return 4;
            }
        }
        W.enc_proj_T_max = T_max;

        if (!measure && W.graph_embed != W.weights->predict_embed) {
            std::vector<float> embedding;
            dequantize_to_f32(W.weights->predict_embed, embedding);
            ggml_backend_tensor_set(W.graph_embed, embedding.data(), 0, embedding.size() * sizeof(float));
        }

        // Uploads write through real device pointers; a measured runtime has
        // none (its tensors carry the externally-allocated marker only).
        if (W.dur_table && !measure) {
            const std::vector<int32_t> table = build_duration_table(model, W.num_durations);
            const int32_t zero = 0;
            ggml_backend_tensor_set(W.dur_table, table.data(), 0, table.size() * sizeof(int32_t));
            ggml_backend_tensor_set(W.zero_dur_idx, &zero, 0, sizeof(int32_t));
        }
    }

    const size_t graph_slots = 2048;
    const size_t graph_mem = ggml_tensor_overhead() * graph_slots
                           + ggml_graph_overhead_custom(graph_slots, false) * 4
                           + 64 * 1024;
    ggml_init_params gp = {};
    gp.mem_size   = graph_mem;
    gp.mem_buffer = nullptr;
    gp.no_alloc   = true;
    W.gctx = ggml_init(gp);
    if (!W.gctx) {
        std::fprintf(stderr, "tdt_prepare_runtime: ggml_init failed\n");
        return 5;
    }

    size_t lstm_cat_bytes = 0;
    W.concat_lstm_input = allow_concat_lstm &&
                          backend_runs_concat(W.backend, W.H_pred) &&
                          build_lstm_input_concat(W, measure ? &lstm_cat_bytes : nullptr);
    if (!W.concat_lstm_input) W.lstm_cat.clear();
    if (measure) {
        // The concatenated [w_ih | w_hh] stacks live on the device for the
        // runtime's lifetime, alongside the persistent state.
        measure->device_state_bytes += lstm_cat_bytes;
    }

    build_lstm_graph(W);
    build_joint_graph(W);
    build_lstm_joint_graph(W);

    if (measure) {
        // The three fixed-shape gallocrs coexist for the runtime's lifetime,
        // so their buffers sum (they are not shared or maxed).
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(W.backend);
        for (ggml_cgraph * cg : { W.g_lstm, W.g_joint, W.g_lstm_joint }) {
            ggml_gallocr_t pricer = ggml_gallocr_new(buft);
            if (!pricer) {
                std::fprintf(stderr, "tdt_measure_runtime: failed to create pricing gallocr\n");
                return 6;
            }
            size_t sz = 0;
            ggml_gallocr_reserve_n_size(pricer, cg, nullptr, nullptr, &sz);
            ggml_gallocr_free(pricer);
            measure->device_compute_bytes += sz;
        }
        // Worst-case per-window enc-projection graph (built lazily at decode
        // time). Larger windows are projected in <= k_enc_proj_T_max chunks,
        // matching enc_proj_persist's capacity.
        int T = measure_enc_frames;
        if (T > TdtRuntimeWeights::k_enc_proj_T_max) T = TdtRuntimeWeights::k_enc_proj_T_max;
        if (T > 0) {
            size_t sz = 0;
            build_enc_proj_graph(W, T, &sz);
            measure->device_compute_bytes += sz;
        }
        // Unrolled decode graph: built lazily by ensure_unroll_graph under
        // exactly these conditions and resident alongside the fixed graphs.
        // max_symbols mirrors the decode default (TdtDecodeOptions /
        // rnnt_max_symbols_per_step); the graph's footprint is driven by
        // unroll_steps, not the symbol cap.
        if (W.fused_lstm_cell && W.argmax_on_gpu && W.unroll_steps >= 1 &&
            W.dur_table && W.zero_dur_idx) {
            const int max_symbols = W.rnnt_mode
                                        ? model.encoder_cfg.rnnt_max_symbols_per_step
                                        : TdtDecodeOptions{}.max_symbols_per_step;
            if (max_symbols >= 1) {
                size_t sz = 0;
                build_unroll_graph(W, model.blank_id, max_symbols, &sz);
                measure->device_compute_bytes += sz;
            }
        }
        return 0;
    }

    W.alloc_lstm       = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    W.alloc_joint      = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    W.alloc_lstm_joint = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    if (!W.alloc_lstm || !W.alloc_joint || !W.alloc_lstm_joint) {
        std::fprintf(stderr, "tdt_prepare_runtime: failed to create gallocrs\n");
        return 6;
    }
    if (!ggml_gallocr_alloc_graph(W.alloc_lstm,       W.g_lstm) ||
        !ggml_gallocr_alloc_graph(W.alloc_joint,      W.g_joint) ||
        !ggml_gallocr_alloc_graph(W.alloc_lstm_joint, W.g_lstm_joint)) {
        std::fprintf(stderr, "tdt_prepare_runtime: failed to allocate fixed-shape graphs\n");
        return 7;
    }

    return 0;
}

int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & W,
                        bool allow_fused_lstm, bool allow_concat_lstm) {
    return tdt_prepare_runtime_impl(model, W, allow_fused_lstm, allow_concat_lstm,
                                    /*measure=*/nullptr, /*measure_enc_frames=*/0);
}

int tdt_measure_runtime(const ParakeetCtcModel & model, int worst_enc_frames,
                        DecoderFitMeasure & out) {
    out = DecoderFitMeasure{};
    TdtRuntimeWeights W;  // scaffolding only; destructor frees the metadata ctxs
    // Mirror the real Engine call: both graph refinements enabled, with the
    // same backend-capability probes deciding what actually builds.
    return tdt_prepare_runtime_impl(model, W, /*allow_fused_lstm=*/true,
                                    /*allow_concat_lstm=*/true,
                                    &out, worst_enc_frames);
}

void tdt_set_unroll_steps(TdtRuntimeWeights & W, int steps) {
    W.unroll_steps = steps;
    free_unroll_graph(W);
}

namespace {

// Init-only LSTM step (used by tdt_init_state to seed pred_persist).
// Updates h_persist / c_persist / pred_persist via in-graph ggml_cpy.
bool run_lstm_init_step(TdtRuntimeWeights & rt, int token_id) {
    const int32_t tok = (int32_t) token_id;
    ggml_backend_tensor_set(rt.lstm_token_in, &tok, 0, sizeof(int32_t));

    if (!compute_graph(rt, rt.g_lstm)) {
        std::fprintf(stderr, "tdt: LSTM init graph compute failed\n");
        return false;
    }
    return true;
}

// Read the joint token/dur outputs into host ints: i32 argmax indices when
// argmax_on_gpu, else the raw f32 logit slices (ggml-opencl) argmaxed on host.
// thread_local scratch keeps the per-step readback allocation-free.
void resolve_joint_step(TdtRuntimeWeights & rt,
                        ggml_tensor * tok_t, ggml_tensor * dur_t,
                        int * tok_out, int * dur_out) {
    if (rt.argmax_on_gpu) {
        int32_t tok_val = 0, dur_val = 0;
        ggml_backend_tensor_get(tok_t, &tok_val, 0, sizeof(int32_t));
        if (dur_t) {
            ggml_backend_tensor_get(dur_t, &dur_val, 0, sizeof(int32_t));
        }
        *tok_out = (int) tok_val;
        *dur_out = (int) dur_val;
        return;
    }
    static thread_local std::vector<float> tok_logits;
    static thread_local std::vector<float> dur_logits;
    tok_logits.resize((size_t) rt.V_plus_1);
    ggml_backend_tensor_get(tok_t, tok_logits.data(), 0, (size_t) rt.V_plus_1 * sizeof(float));
    *tok_out = argmax_f32(tok_logits.data(), rt.V_plus_1);
    *dur_out = 0;
    if (dur_t && rt.num_durations > 0) {
        dur_logits.resize((size_t) rt.num_durations);
        ggml_backend_tensor_get(
            dur_t, dur_logits.data(), 0,
            (size_t) rt.num_durations * sizeof(float));
        *dur_out = argmax_f32(dur_logits.data(), rt.num_durations);
    }
}

// Joint-only step (used after a blank emission). pred_persist is unchanged
// from the previous step; only enc_proj_persist[frame_idx] varies.  The
// graph runs token + duration argmax on-device, so the host reads
// 2 × 4 B (i32 indices) instead of V_out × 4 B (~32 KB) of logits per
// step.  On Apple unified memory the difference is small; on a discrete
// GPU PCIe bus it's an order-of-magnitude saving per emission.
bool run_joint_step(TdtRuntimeWeights & rt,
                    int frame_idx,
                    int * tok_out,
                    int * dur_out) {
    const int32_t fi = (int32_t) frame_idx;
    ggml_backend_tensor_set(rt.joint_frame_idx_in, &fi, 0, sizeof(int32_t));

    if (!compute_graph(rt, rt.g_joint)) {
        std::fprintf(stderr, "tdt: joint graph compute failed\n");
        return false;
    }

    resolve_joint_step(rt, rt.joint_token_out, rt.joint_dur_out, tok_out, dur_out);
    return true;
}

// Fused LSTM-then-joint step (used after a non-blank emission). One
// command-buffer commit instead of two: LSTM updates pred_persist via
// ggml_cpy, joint mat_muls depend on the cpy node so they read the fresh
// pred in the same graph.  Same on-device argmax shape as run_joint_step.
bool run_lstm_joint_step(TdtRuntimeWeights & rt,
                         int token_id,
                         int frame_idx,
                         int * tok_out,
                         int * dur_out) {
    const int32_t tok = (int32_t) token_id;
    const int32_t fi  = (int32_t) frame_idx;
    ggml_backend_tensor_set(rt.lj_token_in,     &tok, 0, sizeof(int32_t));
    ggml_backend_tensor_set(rt.lj_frame_idx_in, &fi,  0, sizeof(int32_t));

    if (!compute_graph(rt, rt.g_lstm_joint)) {
        std::fprintf(stderr, "tdt: lstm_joint graph compute failed\n");
        return false;
    }

    resolve_joint_step(rt, rt.lj_token_out, rt.lj_dur_out, tok_out, dur_out);
    return true;
}

// Compute one bounded window's encoder-side projection straight into
// rt.enc_proj_persist (no host download). The offline caller splits longer
// sequences before reaching this graph path.
bool run_enc_proj(TdtRuntimeWeights & rt,
                  const float * encoder_out,
                  int T) {
    if (T <= 0) return true;
    if (T > rt.enc_proj_T_max) return false;

    const int D_enc = rt.D_enc;

    const TdtRuntimeWeights::EncProjGraph * g = get_enc_proj_graph(rt, T);
    if (!g || !g->alloc) return false;

    ggml_backend_tensor_set(g->enc_in, encoder_out, 0, (size_t) T * D_enc * sizeof(float));

    if (!compute_graph(rt, g->cg)) {
        std::fprintf(stderr, "tdt: enc_proj graph compute failed\n");
        return false;
    }
    return true;
}

// Read the control rows the steps wrote back in one transfer, skipping the
// uploaded row, and pick the (token, duration index) pair out of each.
void read_unroll_outputs(TdtRuntimeWeights & rt,
                         std::vector<int32_t> & toks,
                         std::vector<int32_t> & durs) {
    const size_t K = (size_t) rt.un_ctl->ne[1] - 1;
    std::vector<int32_t> rows(K * GGML_TDT_STEP_N_OUTS);
    ggml_backend_tensor_get(rt.un_ctl, rows.data(),
                            GGML_TDT_STEP_N_OUTS * sizeof(int32_t),
                            rows.size() * sizeof(int32_t));
    toks.resize(K);
    durs.resize(K);
    for (size_t k = 0; k < K; ++k) {
        toks[k] = rows[k * GGML_TDT_STEP_N_OUTS + GGML_TDT_STEP_OUT_TOKEN];
        durs[k] = rows[k * GGML_TDT_STEP_N_OUTS + GGML_TDT_STEP_OUT_DUR];
    }
}

// One launch of the unrolled graph from the current loop counters.
bool run_unroll_graph(TdtRuntimeWeights & rt, int t, int symbols, int n_frames,
                      std::vector<int32_t> & toks, std::vector<int32_t> & durs) {
    int32_t counters[GGML_TDT_STEP_N_INS];
    counters[GGML_TDT_STEP_IN_T] = t;
    counters[GGML_TDT_STEP_IN_S] = symbols;
    counters[GGML_TDT_STEP_IN_N] = n_frames;

    ggml_backend_tensor_set(rt.un_ctl, counters, 0, sizeof(counters));

    if (!compute_graph(rt, rt.g_unroll)) {
        std::fprintf(stderr, "tdt: unrolled decode graph compute failed\n");
        return false;
    }
    read_unroll_outputs(rt, toks, durs);
    return true;
}

// Replay on the host the integer rules the control op applied on the device,
// emitting tokens and advancing the window cursor. Stops at the first step past
// the window: the device masked those steps into no-ops, so the state it leaves
// behind is the state after the last replayed step. Returns the steps taken.
int replay_unroll_steps(const ParakeetCtcModel & model, const TdtDecodeOptions & opts,
                        const std::vector<int32_t> & toks, const std::vector<int32_t> & durs,
                        int n_frames, TdtDecodeState & state, int & t,
                        std::vector<int32_t> & out_tokens) {
    const bool is_rnnt = model.model_type == ParakeetModelType::RNNT;
    const int  blank   = (int) model.blank_id;

    int steps = 0;
    for (size_t k = 0; k < toks.size() && t < n_frames; ++k) {
        ++steps;
        const int best_dur = is_rnnt
                               ? 0
                               : (model.tdt_durations.empty()
                                    ? durs[k]
                                    : model.tdt_durations[durs[k]]);

        if (toks[k] == blank) {
            t += is_rnnt ? 1 : std::max(1, best_dur);
            state.symbols_this_step = 0;
            continue;
        }

        out_tokens.push_back(toks[k]);
        ++state.symbols_this_step;
        const bool force_advance =
            state.symbols_this_step >= opts.max_symbols_per_step;
        if ((!is_rnnt && best_dur > 0) || force_advance) {
            t += is_rnnt ? 1 : std::max(1, best_dur);
            state.symbols_this_step = 0;
        }
    }
    return steps;
}

// Decode the window K steps per graph: launch from the current counters, then
// replay what came back.
int decode_window_unrolled(const ParakeetCtcModel & model, TdtRuntimeWeights & W,
                           const TdtDecodeOptions & opts, int n_frames,
                           TdtDecodeState & state, int & t,
                           std::vector<int32_t> & out_tokens, int & out_steps) {
    std::vector<int32_t> toks;
    std::vector<int32_t> durs;
    while (t < n_frames) {
        if (!run_unroll_graph(W, t, state.symbols_this_step, n_frames, toks, durs)) return 7;
        out_steps += replay_unroll_steps(model, opts, toks, durs, n_frames,
                                         state, t, out_tokens);
    }
    return 0;
}

}  // anonymous namespace

void tdt_init_state(TdtRuntimeWeights & W, int blank_id, TdtDecodeState & state) {
    const int H = W.H_pred;
    const int L = W.L;

    state.symbols_this_step = 0;
    state.carry_frames      = 0;

    if (W.use_graphs) {
        // Zero the LSTM state on-device (~10 KB memset), then run one blank-token
        // LSTM step so pred_persist holds the canonical "no tokens yet"
        // prediction (matches NeMo's RNNT_TDT init).
        //
        // Backend portability: ggml_backend_tensor_memset is implemented
        // by CPU / Metal / CUDA / Vulkan in the pinned ggml, but
        // ggml-opencl historically has not implemented it on every
        // upstream rev. ggml_backend_tensor_set is a hard-required op
        // for every backend (used by every graph-input upload) and is
        // guaranteed to work, so we fall back to uploading a host
        // zero buffer when memset fails. The cost is one-off per
        // tdt_decode_window call (~5 KB upload) and only paid when the
        // backend doesn't accelerate the memset path -- negligible vs
        // the ~150 us-per-step Metal command-buffer cost.
        const size_t hc_bytes = ggml_nbytes(W.hc_persist);
        std::vector<uint8_t> zeros(hc_bytes, 0);
        ggml_backend_tensor_set(W.hc_persist, zeros.data(), 0, hc_bytes);
        if (!run_lstm_init_step(W, blank_id)) {
            throw std::runtime_error("tdt_init_state: LSTM graph compute failed");
        }
        // Host-side scratch is unused on the GPU path.
        state.h_state.clear();
        state.c_state.clear();
        state.pred_out.clear();
    } else {
        state.h_state.assign((size_t) L * H, 0.0f);
        state.c_state.assign((size_t) L * H, 0.0f);
        state.pred_out.assign(H, 0.0f);
        std::vector<float> scratch;
        std::vector<float> layer_input_scratch;
        const float * embed_row = W.embed.data() + (size_t) blank_id * H;
        host_lstm_step(W, embed_row, state.h_state.data(), state.c_state.data(),
                       scratch, layer_input_scratch);
        std::memcpy(state.pred_out.data(),
                    state.h_state.data() + (size_t) (L - 1) * H,
                    (size_t) H * sizeof(float));
    }

    state.initialized = true;
}

int tdt_decode_window(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      int & out_steps) {
    out_steps = 0;
    if (D_enc != W.D_enc) {
        PARAKEET_LOG_ERROR("tdt_decode_window: encoder d_model mismatch (%d vs %d)\n",
                           D_enc, W.D_enc);
        return 1;
    }
    if (n_frames <= 0) return 0;

    if (!state.initialized) {
        tdt_init_state(W, (int) model.blank_id, state);
    }

    const int H_pred  = W.H_pred;
    const int V_p1    = W.V_plus_1;
    const int D_n     = W.num_durations;
    const int L       = W.L;
    const int blank   = (int) model.blank_id;
    const int V_out   = W.V_out;
    const bool is_rnnt =
        model.model_type == ParakeetModelType::RNNT ||
        model.model_type == ParakeetModelType::NEMOTRON;

    // Graph path: stash the full-window encoder-side projection into
    // enc_proj_persist on the backend once at the top of the window so per-step
    // joint reads can ggml_get_rows directly. The host fallback keeps the
    // per-step gemv inside host_joint_step (better cache locality).
    if (W.use_graphs) {
        if (!run_enc_proj(W, encoder_out_window, n_frames)) return 6;
        ensure_unroll_graph(W, blank, opts.max_symbols_per_step);
    }

    std::vector<float> logits((size_t) V_out);
    std::vector<float> scratch_lstm;
    std::vector<float> scratch_lstm_layer_input;
    std::vector<float> scratch_joint_tmp;
    std::vector<float> scratch_joint_hidden;

    int t = 0;
    if (!is_rnnt && state.carry_frames > 0) {
        t = std::min(state.carry_frames, n_frames);
        state.carry_frames -= t;
    }

    // K greedy steps per graph. Exact by construction: the control op applies
    // the same integer rules the replay below applies, and masked steps leave
    // the predictor state untouched, so the token stream matches the sequential
    // loop.
    if (W.use_graphs && W.g_unroll) {
        if (int rc = decode_window_unrolled(model, W, opts, n_frames, state, t,
                                            out_tokens, out_steps);
            rc != 0) {
            return rc;
        }
        state.carry_frames = is_rnnt ? 0 : std::max(0, t - n_frames);
        return 0;
    }

    // After a non-blank emission the next iteration runs the fused LSTM+joint
    // graph so pred_persist is updated before the joint reads it. Blank steps
    // leave pred_persist unchanged (joint-only graph next).
    //
    // GPU path returns token + duration argmax indices from the graph; CPU
    // fallback argmaxes full host logits.
    int  pending_lstm_token = -1;  // < 0 means "no pending LSTM update"
    while (t < n_frames) {
        int best_token = 0;
        int best_dur_idx = 0;
        if (W.use_graphs) {
            if (pending_lstm_token >= 0) {
                if (!run_lstm_joint_step(W, pending_lstm_token, t, &best_token, &best_dur_idx)) return 7;
                pending_lstm_token = -1;
            } else {
                if (!run_joint_step(W, t, &best_token, &best_dur_idx)) return 8;
            }
        } else {
            const float * enc_frame = encoder_out_window + (size_t) t * D_enc;
            host_joint_step(W, enc_frame, state.pred_out.data(),
                            scratch_joint_hidden, logits, scratch_joint_tmp);
            best_token   = argmax_f32(logits.data(), V_p1);
            if (!is_rnnt) {
                best_dur_idx = argmax_f32(logits.data() + V_p1, D_n);
            }
        }
        ++out_steps;

        const int best_dur = is_rnnt
                               ? 0
                               : (model.tdt_durations.empty()
                                    ? best_dur_idx
                                    : model.tdt_durations[best_dur_idx]);

        if (best_token == blank) {
            t += is_rnnt ? 1 : std::max(1, best_dur);
            state.symbols_this_step = 0;
            continue;
        }

        out_tokens.push_back((int32_t) best_token);

        if (W.use_graphs) {
            // Defer the LSTM update — it'll run fused with the next
            // iteration's joint forward in one compute_graph commit.
            pending_lstm_token = best_token;
        } else {
            const float * embed_row = W.embed.data() + (size_t) best_token * H_pred;
            host_lstm_step(W, embed_row, state.h_state.data(), state.c_state.data(),
                           scratch_lstm, scratch_lstm_layer_input);
            std::memcpy(state.pred_out.data(),
                        state.h_state.data() + (size_t) (L - 1) * H_pred,
                        (size_t) H_pred * sizeof(float));
        }

        ++state.symbols_this_step;
        const bool force_advance =
            state.symbols_this_step >= opts.max_symbols_per_step;
        if ((!is_rnnt && best_dur > 0) || force_advance) {
            t += is_rnnt ? 1 : std::max(1, best_dur);
            state.symbols_this_step = 0;
        }
    }

    // Streaming: if the window ended with a deferred LSTM update (last
    // emission was non-blank but we ran out of frames), flush it so the
    // next decode_window call sees the up-to-date pred_persist before its
    // first joint forward. One extra commit at end-of-window only;
    // amortised over the whole utterance it's negligible.
    if (W.use_graphs && pending_lstm_token >= 0) {
        if (!run_lstm_init_step(W, pending_lstm_token)) return 9;
    }

    state.carry_frames = is_rnnt ? 0 : std::max(0, t - n_frames);
    return 0;
}

TdtLaneGeometry tdt_lane_geometry(const ParakeetCtcModel & model, const TdtRuntimeWeights & W) {
    TdtLaneGeometry geom;
    geom.blank_id         = (int) model.blank_id;
    geom.vocab_plus_blank = W.V_plus_1;
    geom.durations        = model.tdt_durations;
    geom.h_pred           = W.H_pred;
    geom.layers           = W.L;
    geom.d_enc            = W.D_enc;
    return geom;
}

namespace {

// A window's prediction state as it retires: h/c *before* consuming embed,
// the same triple a lane carries, which is what a warm start needs.
struct LaneWindowState {
    bool               valid = false;
    std::vector<float> h, c, embed;
};

struct LaneCursor {
    size_t window  = 0;  // index into the windows, or the idle marker
    int    t       = 0;  // frame cursor within that window
    int    symbols = 0;  // emissions at the current frame
};

// The lane walk's buffers and bookkeeping for one utterance. The stepper's
// graph always steps the LSTM, so a lane carries (h, c, embed) meaning "h/c
// before consuming embed", and the prediction the joint scores against is
// recomputed every dispatch: a blank step leaves all three alone and only
// moves the frame cursor, an emission commits h_out/c_out and swaps in the
// emitted token's embedding row.
struct LaneDecoder {
    const TdtLaneGeometry &            geom;
    const std::vector<float> &         embed_table;
    TdtLaneStepper &                   stepper;
    const std::vector<TdtLaneWindow> & windows;
    const TdtDecodeOptions &           opts;

    const int     L;
    const int64_t h_pred, h_dim, d_enc, v_out;
    const int     n_dur;
    const size_t  idle;

    std::vector<float> embed, h, c, enc, logits, h_out, c_out;
    std::vector<LaneCursor>            lane;
    std::vector<LaneWindowState>       final_state;
    std::vector<std::vector<int32_t>>  tokens;
    int dispatches = 0;

    LaneDecoder(const TdtLaneGeometry & g, const std::vector<float> & table, TdtLaneStepper & s,
                const std::vector<TdtLaneWindow> & w, const TdtDecodeOptions & o)
        : geom(g), embed_table(table), stepper(s), windows(w), opts(o),
          L((int) s.lanes()), h_pred(s.h_pred()), h_dim(s.h_dim()), d_enc(s.d_enc()),
          v_out(s.v_out()), n_dur(geom.durations.empty()
                                      ? (int) (v_out - geom.vocab_plus_blank)
                                      : (int) geom.durations.size()),
          idle(w.size()),
          embed((size_t) L * h_pred, 0.0f), h((size_t) L * h_dim, 0.0f), c((size_t) L * h_dim, 0.0f),
          enc((size_t) L * d_enc, 0.0f), logits((size_t) L * v_out, 0.0f),
          h_out((size_t) L * h_dim, 0.0f), c_out((size_t) L * h_dim, 0.0f),
          lane((size_t) L), final_state(w.size()), tokens(w.size()) {}

    const float * embed_row(int token) const {
        return embed_table.data() + (size_t) token * (size_t) h_pred;
    }

    float * lane_h(int l)     { return h.data()     + (size_t) l * h_dim; }
    float * lane_c(int l)     { return c.data()     + (size_t) l * h_dim; }
    float * lane_embed(int l) { return embed.data() + (size_t) l * h_pred; }

    void seed_lane(int l, const LaneWindowState & seed) {
        if (seed.valid) {
            std::copy(seed.h.begin(), seed.h.end(), lane_h(l));
            std::copy(seed.c.begin(), seed.c.end(), lane_c(l));
            std::copy(seed.embed.begin(), seed.embed.end(), lane_embed(l));
            return;
        }
        // Zero state plus the blank row is the canonical "nothing emitted yet"
        // prediction, the same seed tdt_init_state gives the sequential walk.
        std::fill(lane_h(l), lane_h(l) + h_dim, 0.0f);
        std::fill(lane_c(l), lane_c(l) + h_dim, 0.0f);
        std::copy(embed_row(geom.blank_id), embed_row(geom.blank_id) + h_pred, lane_embed(l));
    }

    // Seats the next window with frames into lane `l`, or idles the lane.
    void seat(int l, const std::vector<size_t> & todo, size_t & next,
              const std::vector<LaneWindowState> & seeds) {
        while (next < todo.size() && windows[todo[next]].n_frames <= 0) ++next;
        lane[(size_t) l] = LaneCursor{};
        lane[(size_t) l].window = next < todo.size() ? todo[next++] : idle;
        if (lane[(size_t) l].window != idle) {
            seed_lane(l, seeds[lane[(size_t) l].window]);
        }
    }

    void retire(int l) {
        LaneWindowState & st = final_state[lane[(size_t) l].window];
        st.h.assign(lane_h(l), lane_h(l) + h_dim);
        st.c.assign(lane_c(l), lane_c(l) + h_dim);
        st.embed.assign(lane_embed(l), lane_embed(l) + h_pred);
        st.valid = true;
    }

    // Copies every active lane's current encoder frame into the dispatch
    // input; returns the number of active lanes.
    int stage_frames() {
        int active = 0;
        for (int l = 0; l < L; ++l) {
            const LaneCursor & cur = lane[(size_t) l];
            if (cur.window == idle) continue;
            ++active;
            const TdtLaneWindow & w = windows[cur.window];
            std::copy(w.enc + (size_t) cur.t * d_enc, w.enc + (size_t) (cur.t + 1) * d_enc,
                      enc.data() + (size_t) l * d_enc);
        }
        return active;
    }

    int duration_for(int dur_idx) const {
        return geom.durations.empty() ? dur_idx : geom.durations[(size_t) dur_idx];
    }

    // Applies one dispatch's logits to lane `l`: the same blank handling,
    // duration jump and max_symbols_per_step guard as tdt_decode_window.
    void advance_lane(int l) {
        LaneCursor & cur = lane[(size_t) l];
        const float * lane_logits = logits.data() + (size_t) l * v_out;
        const int best_token   = argmax_f32(lane_logits, geom.vocab_plus_blank);
        const int best_dur_idx = argmax_f32(lane_logits + geom.vocab_plus_blank, n_dur);
        const int best_dur     = duration_for(best_dur_idx);

        if (best_token == geom.blank_id) {
            cur.t += std::max(1, best_dur);
            cur.symbols = 0;
            return;
        }
        tokens[cur.window].push_back((int32_t) best_token);
        std::copy(h_out.begin() + (size_t) l * h_dim, h_out.begin() + (size_t) (l + 1) * h_dim, lane_h(l));
        std::copy(c_out.begin() + (size_t) l * h_dim, c_out.begin() + (size_t) (l + 1) * h_dim, lane_c(l));
        std::copy(embed_row(best_token), embed_row(best_token) + h_pred, lane_embed(l));

        ++cur.symbols;
        if (best_dur > 0 || cur.symbols >= opts.max_symbols_per_step) {
            cur.t += std::max(1, best_dur);
            cur.symbols = 0;
        }
    }

    // Retires lanes whose window ran out of frames and seats the next ones.
    void refill_lanes(const std::vector<size_t> & todo, size_t & next,
                      const std::vector<LaneWindowState> & seeds) {
        for (int l = 0; l < L; ++l) {
            const LaneCursor & cur = lane[(size_t) l];
            if (cur.window == idle) continue;
            if (cur.t >= windows[cur.window].n_frames) {
                retire(l);
                seat(l, todo, next, seeds);
            }
        }
    }

    void advance_lanes() {
        for (int l = 0; l < L; ++l) {
            if (lane[(size_t) l].window != idle) advance_lane(l);
        }
    }

    // One batched pass over `todo`, seating windows into lanes as lanes free
    // up so the batch stays full until the list runs out. `seeds` supplies a
    // warm start per window; where it has none the lane starts from blank.
    int run_pass(const std::vector<size_t> & todo, const std::vector<LaneWindowState> & seeds) {
        size_t next = 0;
        for (int l = 0; l < L; ++l) seat(l, todo, next, seeds);

        while (stage_frames() > 0) {
            if (int rc = stepper.step(embed.data(), h.data(), c.data(), enc.data(),
                                      logits.data(), h_out.data(), c_out.data()); rc != 0) {
                PARAKEET_LOG_ERROR("tdt: batched decoder dispatch failed (rc=%d)\n", rc);
                return 8;
            }
            ++dispatches;
            advance_lanes();
            refill_lanes(todo, next, seeds);
        }
        return 0;
    }

    // Windows that emitted nothing although a predecessor retired a state.
    // Blank leaves the prediction state untouched, so a window that starts
    // mid-phrase can score blank against the from-scratch state on every frame.
    std::vector<size_t> windows_to_rescue() const {
        std::vector<size_t> rescue;
        for (size_t i = 1; i < windows.size(); ++i) {
            if (tokens[i].empty() && windows[i].n_frames > 0 && final_state[i - 1].valid) {
                rescue.push_back(i);
            }
        }
        return rescue;
    }
};

bool lane_stepper_matches(const TdtLaneGeometry & geom, const TdtLaneStepper & stepper) {
    const int64_t n_dur = geom.durations.empty() ? 0 : (int64_t) geom.durations.size();
    const bool v_out_ok = geom.durations.empty()
        ? stepper.v_out() > geom.vocab_plus_blank
        : stepper.v_out() == geom.vocab_plus_blank + n_dur;
    return stepper.lanes() > 0 &&
           stepper.h_pred() == geom.h_pred &&
           stepper.h_dim()  == (int64_t) geom.h_pred * geom.layers &&
           stepper.d_enc()  == geom.d_enc &&
           v_out_ok;
}

}  // anonymous namespace

int tdt_decode_lanes(const TdtLaneGeometry & geom,
                     const std::vector<float> & embed_table,
                     TdtLaneStepper & stepper,
                     const std::vector<TdtLaneWindow> & windows,
                     const TdtDecodeOptions & opts,
                     std::vector<std::vector<int32_t>> & out_tokens,
                     TdtLaneStats * stats) {
    // Cross-check the stepper against the checkpoint. A sidecar built from a
    // different GGUF would otherwise decode into plausible-looking nonsense.
    if (!lane_stepper_matches(geom, stepper)) {
        PARAKEET_LOG_ERROR("tdt: batched decoder geometry does not match the GGUF "
                           "(h_pred %lld vs %d, h_dim %lld vs %d, d_enc %lld vs %d, v_out %lld)\n",
                           (long long) stepper.h_pred(), geom.h_pred,
                           (long long) stepper.h_dim(), geom.h_pred * geom.layers,
                           (long long) stepper.d_enc(), geom.d_enc, (long long) stepper.v_out());
        return 2;
    }
    if (embed_table.size() < (size_t) geom.vocab_plus_blank * (size_t) geom.h_pred ||
        geom.blank_id < 0 || geom.blank_id >= geom.vocab_plus_blank) {
        PARAKEET_LOG_ERROR("tdt: batched decoder needs the host embedding table\n");
        return 3;
    }

    LaneDecoder decoder(geom, embed_table, stepper, windows, opts);

    std::vector<size_t> all(windows.size());
    for (size_t i = 0; i < windows.size(); ++i) all[i] = i;
    const std::vector<LaneWindowState> no_seeds(windows.size());
    if (int rc = decoder.run_pass(all, no_seeds); rc != 0) return rc;

    // Seeds all come from the first pass, so the rescue is itself one batched pass.
    const std::vector<size_t> rescue = decoder.windows_to_rescue();
    if (!rescue.empty()) {
        std::vector<LaneWindowState> seeds(windows.size());
        for (size_t i : rescue) seeds[i] = decoder.final_state[i - 1];
        if (int rc = decoder.run_pass(rescue, seeds); rc != 0) return rc;
    }

    out_tokens = std::move(decoder.tokens);
    if (stats != nullptr) {
        stats->dispatches = decoder.dispatches;
        stats->rescued    = (int) rescue.size();
    }
    return 0;
}

int tdt_greedy_decode(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeResult & result) {
    const auto t0 = std::chrono::steady_clock::now();

    TdtDecodeState state;
    result.token_ids.clear();
    result.token_ids.reserve(T_enc);
    result.steps = 0;

    // The offline encoder may concatenate many windows, but the graph
    // decoder's persistent projection holds at most enc_proj_T_max frames.
    // Keep one predictor state (including TDT duration carry) across bounded
    // decoder windows. Short inputs retain their single-window path.
    const int window_frames = W.use_graphs ? W.enc_proj_T_max : T_enc;
    if (window_frames > 0 && T_enc > window_frames) {
        for (int start = 0; start < T_enc; start += window_frames) {
            const int frames = std::min(window_frames, T_enc - start);
            int steps = 0;
            if (int rc = tdt_decode_window(
                    model, W, encoder_out + (size_t) start * D_enc,
                    frames, D_enc, opts, state, result.token_ids, steps);
                rc != 0) {
                return rc;
            }
            result.steps += steps;
        }
    } else {
        if (int rc = tdt_decode_window(model, W, encoder_out, T_enc, D_enc,
                                       opts, state, result.token_ids, result.steps);
            rc != 0) {
            return rc;
        }
    }

    result.text = detokenize(model.vocab, result.token_ids);
    const auto t1 = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return 0;
}

int rnnt_decode_window(const ParakeetCtcModel & model,
                       RnntRuntimeWeights & weights,
                       const float * encoder_out_window,
                       int n_frames,
                       int encoder_dim,
                       const RnntDecodeOptions & options,
                       RnntDecodeState & state,
                       std::vector<int32_t> & out_tokens,
                       int & out_steps) {
    if (model.model_type != ParakeetModelType::RNNT &&
        model.model_type != ParakeetModelType::NEMOTRON) {
        return kWrongTransducerModel;
    }
    return tdt_decode_window(
        model, weights, encoder_out_window, n_frames, encoder_dim,
        options, state, out_tokens, out_steps);
}

int rnnt_greedy_decode(const ParakeetCtcModel & model,
                       RnntRuntimeWeights & weights,
                       const float * encoder_out,
                       int encoder_frames,
                       int encoder_dim,
                       const RnntDecodeOptions & options,
                       RnntDecodeResult & result) {
    if (model.model_type != ParakeetModelType::RNNT &&
        model.model_type != ParakeetModelType::NEMOTRON) {
        return kWrongTransducerModel;
    }
    return tdt_greedy_decode(
        model, weights, encoder_out, encoder_frames, encoder_dim,
        options, result);
}

static int rnnt_decode_chunks(const ParakeetCtcModel & model,
                              RnntRuntimeWeights & weights,
                              const float * encoder_out,
                              int encoder_frames,
                              int encoder_dim,
                              int chunk_frames,
                              const RnntDecodeOptions & options,
                              RnntDecodeState & state,
                              RnntDecodeResult & result) {
    for (int offset = 0; offset < encoder_frames; offset += chunk_frames) {
        const int frames = std::min(chunk_frames, encoder_frames - offset);
        int steps = 0;
        if (int rc = rnnt_decode_window(
                model, weights,
                encoder_out + static_cast<size_t>(offset) * encoder_dim,
                frames, encoder_dim, options, state,
                result.token_ids, steps); rc != 0) {
            return rc;
        }
        result.steps += steps;
    }
    return 0;
}

int rnnt_greedy_decode_chunked(const ParakeetCtcModel & model,
                               RnntRuntimeWeights & weights,
                               const float * encoder_out,
                               int encoder_frames,
                               int encoder_dim,
                               int chunk_frames,
                               const RnntDecodeOptions & options,
                               RnntDecodeResult & result) {
    if ((model.model_type != ParakeetModelType::RNNT &&
         model.model_type != ParakeetModelType::NEMOTRON) ||
        !encoder_out || encoder_frames < 0 || chunk_frames <= 0) {
        return kWrongTransducerModel;
    }

    const auto started = std::chrono::steady_clock::now();
    RnntDecodeState state;
    result = RnntDecodeResult{};
    result.token_ids.reserve(encoder_frames);

    if (int rc = rnnt_decode_chunks(
            model, weights, encoder_out, encoder_frames, encoder_dim,
            chunk_frames, options, state, result); rc != 0) {
        return rc;
    }

    result.text = detokenize(model.vocab, result.token_ids);
    result.decode_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count() / 1000.0;
    return 0;
}

}
