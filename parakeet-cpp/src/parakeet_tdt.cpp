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
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace parakeet {

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

// Append the LSTM body to `gctx` and return:
//   - cpy nodes that write the freshly computed h, c into rt.h_persist /
//     rt.c_persist in place, and
//   - the cpy node aliasing rt.pred_persist (last-layer h_new).
//
// `token_in` must be an i32[1] input tensor in `gctx`. The graph builder
// is shared between the init-only `g_lstm` graph and the fused
// `g_lstm_joint` graph so they stay numerically identical.
struct LstmBodyOuts {
    ggml_tensor * h_cpy;
    ggml_tensor * c_cpy;
    ggml_tensor * pred_cpy;
};

LstmBodyOuts build_lstm_body(TdtRuntimeWeights & rt,
                             ggml_context * gctx,
                             ggml_tensor * token_in) {
    const int H = rt.H_pred;
    const int L = rt.L;

    // Embedding lookup. predict_embed has ne[0]=H, ne[1]=vocab+1; result
    // is [H, 1]. Reshape to [H] for the per-step LSTM input.
    ggml_tensor * x = ggml_get_rows(gctx, rt.weights->predict_embed, token_in);
    x = ggml_reshape_1d(gctx, x, H);

    std::vector<ggml_tensor *> h_new_per_layer(L);
    std::vector<ggml_tensor *> c_new_per_layer(L);

    for (int l = 0; l < L; ++l) {
        const auto & w = rt.weights->lstm[l];
        ggml_tensor * h_l_in = ggml_view_1d(gctx, rt.h_persist, H, (size_t) l * H * sizeof(float));
        ggml_tensor * c_l_in = ggml_view_1d(gctx, rt.c_persist, H, (size_t) l * H * sizeof(float));

        // gates = w_ih @ x + b_ih + b_hh + w_hh @ h_prev   ->  [4H]
        ggml_tensor * gates = ggml_mul_mat(gctx, w.w_ih, x);
        gates = ggml_add(gctx, gates, w.b_ih);
        gates = ggml_add(gctx, gates, w.b_hh);
        ggml_tensor * gates_h = ggml_mul_mat(gctx, w.w_hh, h_l_in);
        gates = ggml_add(gctx, gates, gates_h);

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

        h_new_per_layer[l] = h_new;
        c_new_per_layer[l] = c_new;

        // Next layer feeds on this layer's hidden output.
        x = h_new;
    }

    LstmBodyOuts out{};
    for (int l = 0; l < L; ++l) {
        ggml_tensor * h_dst = ggml_view_1d(gctx, rt.h_persist, H, (size_t) l * H * sizeof(float));
        ggml_tensor * c_dst = ggml_view_1d(gctx, rt.c_persist, H, (size_t) l * H * sizeof(float));
        out.h_cpy = ggml_cpy(gctx, h_new_per_layer[l], h_dst);
        out.c_cpy = ggml_cpy(gctx, c_new_per_layer[l], c_dst);
    }
    out.pred_cpy = ggml_cpy(gctx, h_new_per_layer[L - 1], rt.pred_persist);
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
struct JointBodyOuts {
    ggml_tensor * token_argmax;  // i32[1], over logits[0 : V_plus_1]
    ggml_tensor * dur_argmax;    // i32[1], over logits[V_plus_1 : V_plus_1 + num_durations]
};
JointBodyOuts build_joint_body(const TdtRuntimeWeights & rt,
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
    ggml_tensor * dur_logits = ggml_view_2d(gctx, logits,
                                            D_n, 1,
                                            (size_t) D_n * sizeof(float),
                                            (size_t) V_p1 * sizeof(float));
    tok_logits = ggml_cont(gctx, tok_logits);
    dur_logits = ggml_cont(gctx, dur_logits);

    JointBodyOuts outs{};
    outs.token_argmax = ggml_argmax(gctx, tok_logits);  // i32[1]
    outs.dur_argmax   = ggml_argmax(gctx, dur_logits);  // i32[1]
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
    rt.lstm_h_out    = outs.h_cpy;
    rt.lstm_c_out    = outs.c_cpy;
    rt.lstm_pred_out = outs.pred_cpy;
    ggml_set_name(rt.lstm_h_out,    "lstm.h_out");
    ggml_set_name(rt.lstm_c_out,    "lstm.c_out");
    ggml_set_name(rt.lstm_pred_out, "lstm.pred_out");
    ggml_set_output(rt.lstm_h_out);
    ggml_set_output(rt.lstm_c_out);
    ggml_set_output(rt.lstm_pred_out);

    rt.g_lstm = ggml_new_graph_custom(gctx, /*size*/ 256, /*grads*/ false);
    ggml_build_forward_expand(rt.g_lstm, rt.lstm_h_out);
    ggml_build_forward_expand(rt.g_lstm, rt.lstm_c_out);
    ggml_build_forward_expand(rt.g_lstm, rt.lstm_pred_out);
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

    JointBodyOuts outs = build_joint_body(rt, gctx, rt.pred_persist, rt.joint_frame_idx_in);
    rt.joint_token_out = outs.token_argmax;
    rt.joint_dur_out   = outs.dur_argmax;
    ggml_set_name(rt.joint_token_out, "joint.token_argmax");
    ggml_set_name(rt.joint_dur_out,   "joint.dur_argmax");
    ggml_set_output(rt.joint_token_out);
    ggml_set_output(rt.joint_dur_out);

    rt.g_joint = ggml_new_graph_custom(gctx, /*size*/ 96, /*grads*/ false);
    ggml_build_forward_expand(rt.g_joint, rt.joint_token_out);
    ggml_build_forward_expand(rt.g_joint, rt.joint_dur_out);
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
    JointBodyOuts joint_outs = build_joint_body(rt, gctx, lstm_outs.pred_cpy, rt.lj_frame_idx_in);
    rt.lj_token_out = joint_outs.token_argmax;
    rt.lj_dur_out   = joint_outs.dur_argmax;
    ggml_set_name(rt.lj_token_out, "lstm_joint.token_argmax");
    ggml_set_name(rt.lj_dur_out,   "lstm_joint.dur_argmax");
    ggml_set_output(rt.lj_token_out);
    ggml_set_output(rt.lj_dur_out);
    // Mark the LSTM cpy nodes as outputs too so gallocr keeps them alive
    // (their memory IS h_persist / c_persist; without the output flag the
    // gallocr might prune them as dead-end intermediate writes).
    ggml_set_output(lstm_outs.h_cpy);
    ggml_set_output(lstm_outs.c_cpy);

    rt.g_lstm_joint = ggml_new_graph_custom(gctx, /*size*/ 384, /*grads*/ false);
    ggml_build_forward_expand(rt.g_lstm_joint, rt.lj_token_out);
    ggml_build_forward_expand(rt.g_lstm_joint, rt.lj_dur_out);
    ggml_build_forward_expand(rt.g_lstm_joint, lstm_outs.h_cpy);
    ggml_build_forward_expand(rt.g_lstm_joint, lstm_outs.c_cpy);
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
TdtRuntimeWeights::EncProjGraph build_enc_proj_graph(TdtRuntimeWeights & rt, int T) {
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
    this->~TdtRuntimeWeights();

    H_pred       = o.H_pred;
    H_joint      = o.H_joint;
    D_enc        = o.D_enc;
    V_plus_1     = o.V_plus_1;
    V_out        = o.V_out;
    L            = o.L;
    num_durations = o.num_durations;
    weights      = o.weights;        o.weights = nullptr;
    backend      = o.backend;        o.backend = nullptr;
    n_threads    = o.n_threads;
    use_graphs   = o.use_graphs;
    embed        = std::move(o.embed);
    host_lstm    = std::move(o.host_lstm);
    host_joint_enc_w  = std::move(o.host_joint_enc_w);
    host_joint_enc_b  = std::move(o.host_joint_enc_b);
    host_joint_pred_w = std::move(o.host_joint_pred_w);
    host_joint_pred_b = std::move(o.host_joint_pred_b);
    host_joint_out_w  = std::move(o.host_joint_out_w);
    host_joint_out_b  = std::move(o.host_joint_out_b);
    gctx         = o.gctx;           o.gctx = nullptr;
    persist_ctx     = o.persist_ctx;     o.persist_ctx = nullptr;
    persist_buffer  = o.persist_buffer;  o.persist_buffer = nullptr;
    h_persist        = o.h_persist;        o.h_persist = nullptr;
    c_persist        = o.c_persist;        o.c_persist = nullptr;
    pred_persist     = o.pred_persist;     o.pred_persist = nullptr;
    enc_proj_persist = o.enc_proj_persist; o.enc_proj_persist = nullptr;
    enc_proj_T_max   = o.enc_proj_T_max;
    g_lstm       = o.g_lstm;         o.g_lstm = nullptr;
    alloc_lstm   = o.alloc_lstm;     o.alloc_lstm = nullptr;
    lstm_token_in = o.lstm_token_in; o.lstm_token_in = nullptr;
    lstm_h_out   = o.lstm_h_out;     o.lstm_h_out = nullptr;
    lstm_c_out   = o.lstm_c_out;     o.lstm_c_out = nullptr;
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
    enc_proj_cache = std::move(o.enc_proj_cache);
    o.enc_proj_cache.clear();
    return *this;
}

TdtRuntimeWeights::~TdtRuntimeWeights() {
    for (auto & g : enc_proj_cache) {
        free_enc_proj_graph(g);
    }
    enc_proj_cache.clear();
    if (alloc_lstm_joint) { ggml_gallocr_free(alloc_lstm_joint); alloc_lstm_joint = nullptr; }
    if (alloc_joint) { ggml_gallocr_free(alloc_joint); alloc_joint = nullptr; }
    if (alloc_lstm)  { ggml_gallocr_free(alloc_lstm);  alloc_lstm  = nullptr; }
    if (persist_buffer) { ggml_backend_buffer_free(persist_buffer); persist_buffer = nullptr; }
    if (persist_ctx) { ggml_free(persist_ctx); persist_ctx = nullptr; }
    if (gctx)        { ggml_free(gctx);                gctx        = nullptr; }
    // backend is owned by ParakeetCtcModel::Impl; don't free here.
}

int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & W) {
    W = TdtRuntimeWeights{};

    W.H_pred        = model.encoder_cfg.tdt_pred_hidden;
    W.H_joint       = model.encoder_cfg.tdt_joint_hidden;
    W.D_enc         = model.encoder_cfg.d_model;
    W.L             = model.encoder_cfg.tdt_pred_rnn_layers;
    W.num_durations = model.encoder_cfg.tdt_num_durations;
    W.V_plus_1      = (int) model.vocab_size + 1;
    W.V_out         = W.V_plus_1 + W.num_durations;

    W.weights = &model.tdt;
    W.backend = model.backend_active();
    if (!W.backend) {
        std::fprintf(stderr, "tdt_prepare_runtime: model has no active backend (call load_from_gguf first)\n");
        return 1;
    }

    // Defensive thread-count for the rare case where graphs run on a CPU
    // backend (today they don't; CPU goes through the scalar fallback below).
    {
        const unsigned hc = std::thread::hardware_concurrency();
        W.n_threads = hc > 0 ? (int) hc : 4;
    }

    if (!model.tdt.predict_embed || model.tdt.lstm.empty() || !model.tdt.joint_out_w) {
        std::fprintf(stderr, "tdt_prepare_runtime: GGUF is missing TDT tensors\n");
        return 2;
    }

    // Decide the implementation path. The per-step graph dispatch overhead on
    // the CPU backend (thread-pool wakeup x ~250 emission steps) regresses
    // ~6x vs. a hand-rolled scalar gemv loop, so CPU keeps the legacy path.
    // GPU backends (Metal / CUDA / Vulkan) win even with per-step dispatch
    // because of native quantised matmul and faster argmax / large gemvs.
    W.use_graphs = !backend_is_cpu(W.backend);

    if (!W.use_graphs) {
        // ---- CPU fallback: dequantise weights to host f32 ----
        dequantize_to_f32(model.tdt.predict_embed, W.embed);
        W.host_lstm.clear();
        W.host_lstm.resize(W.L);
        for (int l = 0; l < W.L; ++l) {
            dequantize_to_f32(model.tdt.lstm[l].w_ih, W.host_lstm[l].w_ih);
            dequantize_to_f32(model.tdt.lstm[l].w_hh, W.host_lstm[l].w_hh);
            dequantize_to_f32(model.tdt.lstm[l].b_ih, W.host_lstm[l].b_ih);
            dequantize_to_f32(model.tdt.lstm[l].b_hh, W.host_lstm[l].b_hh);
        }
        dequantize_to_f32(model.tdt.joint_enc_w,  W.host_joint_enc_w);
        dequantize_to_f32(model.tdt.joint_enc_b,  W.host_joint_enc_b);
        dequantize_to_f32(model.tdt.joint_pred_w, W.host_joint_pred_w);
        dequantize_to_f32(model.tdt.joint_pred_b, W.host_joint_pred_b);
        dequantize_to_f32(model.tdt.joint_out_w,  W.host_joint_out_w);
        dequantize_to_f32(model.tdt.joint_out_b,  W.host_joint_out_b);
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

        W.h_persist        = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, H_pred,  L);
        W.c_persist        = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, H_pred,  L);
        W.pred_persist     = ggml_new_tensor_1d(W.persist_ctx, GGML_TYPE_F32, H_pred);
        W.enc_proj_persist = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, H_joint, T_max);
        ggml_set_name(W.h_persist,        "tdt.h_persist");
        ggml_set_name(W.c_persist,        "tdt.c_persist");
        ggml_set_name(W.pred_persist,     "tdt.pred_persist");
        ggml_set_name(W.enc_proj_persist, "tdt.enc_proj_persist");

        W.persist_buffer = ggml_backend_alloc_ctx_tensors(W.persist_ctx, W.backend);
        if (!W.persist_buffer) {
            std::fprintf(stderr, "tdt_prepare_runtime: failed to allocate persistent state buffer\n");
            return 4;
        }
        W.enc_proj_T_max = T_max;
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

    build_lstm_graph(W);
    build_joint_graph(W);
    build_lstm_joint_graph(W);

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

    int32_t tok_val = 0, dur_val = 0;
    ggml_backend_tensor_get(rt.joint_token_out, &tok_val, 0, sizeof(int32_t));
    ggml_backend_tensor_get(rt.joint_dur_out,   &dur_val, 0, sizeof(int32_t));
    *tok_out = (int) tok_val;
    *dur_out = (int) dur_val;
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

    int32_t tok_val = 0, dur_val = 0;
    ggml_backend_tensor_get(rt.lj_token_out, &tok_val, 0, sizeof(int32_t));
    ggml_backend_tensor_get(rt.lj_dur_out,   &dur_val, 0, sizeof(int32_t));
    *tok_out = (int) tok_val;
    *dur_out = (int) dur_val;
    return true;
}

// Compute the full-window encoder-side projection straight into
// rt.enc_proj_persist (no host download). Falls back to per-step host
// gemv if T exceeds the persistent buffer size.
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

}  // anonymous namespace

void tdt_init_state(TdtRuntimeWeights & W, int blank_id, TdtDecodeState & state) {
    const int H = W.H_pred;
    const int L = W.L;

    state.symbols_this_step = 0;
    state.carry_frames      = 0;

    if (W.use_graphs) {
        // Zero h, c on-device (~5 KB memset), then run one blank-token
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
        const size_t h_bytes = ggml_nbytes(W.h_persist);
        const size_t c_bytes = ggml_nbytes(W.c_persist);
        std::vector<uint8_t> zeros;
        zeros.assign(std::max(h_bytes, c_bytes), 0);
        ggml_backend_tensor_set(W.h_persist, zeros.data(), 0, h_bytes);
        ggml_backend_tensor_set(W.c_persist, zeros.data(), 0, c_bytes);
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

    // GPU path: stash the full-window encoder-side projection into
    // enc_proj_persist on-device once at the top of the window so per-step
    // joint reads can ggml_get_rows directly. CPU path keeps the original
    // per-step gemv inside host_joint_step (better cache locality).
    if (W.use_graphs) {
        if (!run_enc_proj(W, encoder_out_window, n_frames)) return 6;
    }

    std::vector<float> logits((size_t) V_out);
    std::vector<float> scratch_lstm;
    std::vector<float> scratch_lstm_layer_input;
    std::vector<float> scratch_joint_tmp;
    std::vector<float> scratch_joint_hidden;

    int t = 0;
    if (state.carry_frames > 0) {
        t = std::min(state.carry_frames, n_frames);
        state.carry_frames -= t;
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
            best_dur_idx = argmax_f32(logits.data() + V_p1, D_n);
        }
        ++out_steps;

        const int best_dur     = model.tdt_durations.empty()
                                   ? best_dur_idx
                                   : model.tdt_durations[best_dur_idx];

        if (best_token == blank) {
            t += std::max(1, best_dur);
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
        if (best_dur > 0 || state.symbols_this_step >= opts.max_symbols_per_step) {
            t += std::max(1, best_dur);
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

    state.carry_frames = std::max(0, t - n_frames);
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

    if (int rc = tdt_decode_window(model, W, encoder_out, T_enc, D_enc,
                                   opts, state, result.token_ids, result.steps);
        rc != 0) {
        return rc;
    }

    result.text = detokenize(model.vocab, result.token_ids);
    const auto t1 = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return 0;
}

}
