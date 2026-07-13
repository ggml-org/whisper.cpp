// EOU runtime preparation and greedy decoding (ggml GPU graphs or scalar CPU path).

#include "parakeet_eou.h"
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

void dequantize_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) throw std::runtime_error("eou_prepare_runtime: missing tensor");
    const size_t n = (size_t) ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return;
    }
    const auto * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error(std::string("eou_prepare_runtime: no to_float for type ") +
                                 ggml_type_name(t->type));
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> host_raw(nbytes);
    ggml_backend_tensor_get(t, host_raw.data(), 0, nbytes);
    tr->to_float(host_raw.data(), out.data(), (int64_t) n);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// See `parakeet_tdt.cpp::gemv_f32` for the same vectorisation pattern
// + rationale. EOU is the same shape on a smaller weight matrix
// (1L LSTM 640 vs TDT 2L LSTM 640) but is the only path that runs
// per-token at low single-digit ms latency targets, so the SIMD
// pragma matters here too.
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

// `layer_input_scratch` lifted out of the per-call allocator. EOU
// emits ~250 tokens per 11s utterance, each calling lstm_step once,
// so a single-decode allocation count of 250 vs. 1 at H_pred=640
// (2.5 KB each) shaves ~1.6 MB of malloc/free traffic off the inner
// loop. Byte-equal output: each call resizes to H and the inner gemv
// writes every byte before reading, so no stale state can leak
// between calls.
void lstm_step(const EouRuntimeWeights & W,
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
        const auto & w = W.lstm[layer];
        const float * h_l = h_state + (size_t) layer * H;
        float * c_l = c_state + (size_t) layer * H;

        gemv_f32(w.w_ih.data(), x, w.b_ih.data(), scratch.data(), G, H);
        for (int i = 0; i < G; ++i) scratch[i] += w.b_hh[i];
        gemv_add_f32(w.w_hh.data(), h_l, scratch.data(), G, H);

        float * h_new = (float *) (h_state + (size_t) layer * H);
        for (int i = 0; i < H; ++i) {
            const float i_g = sigmoid(scratch[0 * H + i]);
            const float f_g = sigmoid(scratch[1 * H + i]);
            const float g_g = std::tanh(scratch[2 * H + i]);
            const float o_g = sigmoid(scratch[3 * H + i]);
            const float c_new = f_g * c_l[i] + i_g * g_g;
            c_l[i] = c_new;
            h_new[i] = o_g * std::tanh(c_new);
        }

        std::memcpy(layer_input_scratch.data(), h_new, (size_t) H * sizeof(float));
        x = layer_input_scratch.data();
    }
}

// `tmp_scratch` lifted out of the per-call allocator. Same rationale
// as `lstm_step`: ~250 calls per utterance, each previously
// fresh-allocating an H-sized vector. gemv_f32 writes every output
// byte before any read, so re-using the buffer is byte-equal to the
// per-call allocation.
void joint_step(const EouRuntimeWeights & W,
                const float * __restrict enc,
                const float * __restrict pred,
                std::vector<float> & hidden,
                std::vector<float> & logits,
                std::vector<float> & tmp_scratch) {
    const int H   = W.H_joint;
    const int De  = W.D_enc;
    const int Hp  = W.H_pred;
    const int Vp1 = W.V_plus_1;

    hidden.resize(H);
    tmp_scratch.resize(H);
    gemv_f32(W.joint_enc_w.data(),  enc,  W.joint_enc_b.data(),  hidden.data(), H, De);
    gemv_f32(W.joint_pred_w.data(), pred, W.joint_pred_b.data(), tmp_scratch.data(), H, Hp);
    for (int i = 0; i < H; ++i) hidden[i] += tmp_scratch[i];
    for (int i = 0; i < H; ++i) hidden[i] = std::max(0.0f, hidden[i]);

    logits.resize(Vp1);
    gemv_f32(W.joint_out_w.data(), hidden.data(), W.joint_out_b.data(), logits.data(), Vp1, H);
}

int argmax_f32(const float * data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

std::string trim_spaces(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// ---- GPU graph builders (same design as parakeet_tdt.cpp) ----

// Append the LSTM body to `gctx`. See parakeet_tdt.cpp::build_lstm_body
// for the layer-by-layer walkthrough and the rationale for returning ALL
// layers' state-write cpy nodes (forward_expand would otherwise prune the
// writes for non-last layers and the recurrence never advances).
struct LstmBodyOuts {
    ggml_tensor * pred_cpy;
    std::vector<ggml_tensor *> state_cpy;
};

LstmBodyOuts build_lstm_body(EouRuntimeWeights & rt,
                             ggml_context * gctx,
                             ggml_tensor * token_in) {
    const int H = rt.H_pred;
    const int L = rt.L;

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

        // c_new = f * c_prev + i * g ; h_new = o * tanh(c_new)
        ggml_tensor * c_new = ggml_add(gctx,
                                        ggml_mul(gctx, f_g, c_l_in),
                                        ggml_mul(gctx, i_g, g_g));
        ggml_tensor * h_new = ggml_mul(gctx, o_g, ggml_tanh(gctx, c_new));

        h_new = ggml_cont(gctx, h_new);
        c_new = ggml_cont(gctx, c_new);

        h_new_per_layer[l] = h_new;
        c_new_per_layer[l] = c_new;

        x = h_new;
    }

    LstmBodyOuts out{};
    for (int l = 0; l < L; ++l) {
        ggml_tensor * h_dst = ggml_view_1d(gctx, rt.h_persist, H, (size_t) l * H * sizeof(float));
        ggml_tensor * c_dst = ggml_view_1d(gctx, rt.c_persist, H, (size_t) l * H * sizeof(float));
        out.state_cpy.push_back(ggml_cpy(gctx, h_new_per_layer[l], h_dst));
        out.state_cpy.push_back(ggml_cpy(gctx, c_new_per_layer[l], c_dst));
    }
    out.pred_cpy = ggml_cpy(gctx, h_new_per_layer[L - 1], rt.pred_persist);
    return out;
}

// Span-joint body: scores k_span frames against one pred in a single
// launch. `pred_src` is pred_persist for the joint-only graph, or the
// pred_cpy node from build_lstm_body for the fused graph (carrying the
// LSTM dependency so gallocr orders the update before the joint reads).
//
// `frame_idx_in` is an i32[k_span] get_rows index tensor; the host clamps
// tail indices to the last valid frame and ignores those slots, so one
// fixed-shape graph serves any window position. Unlike TDT there is no
// duration head — the argmax is over the whole [V_plus_1, k_span] logits.
struct JointBodyOuts {
    ggml_tensor * token_out;  // i32[k_span] argmax, or f32[V_plus_1, k_span] logits when !argmax_on_gpu
};

JointBodyOuts build_joint_span_body(EouRuntimeWeights & rt,
                                    ggml_context * gctx,
                                    ggml_tensor * pred_src,
                                    ggml_tensor * frame_idx_in) {
    // pred_proj = W_pred @ pred + b_pred -> [H_joint], shared by all
    // frames in the span (pred only changes on non-blank emissions, and
    // every emission invalidates the span).
    ggml_tensor * pred_proj = ggml_mul_mat(gctx, rt.weights->joint_pred_w, pred_src);
    pred_proj = ggml_add(gctx, pred_proj, rt.weights->joint_pred_b);

    // enc_proj rows for the span -> [H_joint, k_span]
    ggml_tensor * enc_rows = ggml_get_rows(gctx, rt.enc_proj_persist, frame_idx_in);

    // hidden = relu(enc_rows + pred_proj), pred_proj broadcast across the span
    ggml_tensor * hidden = ggml_add(gctx, enc_rows, pred_proj);
    hidden = ggml_relu(gctx, hidden);

    // logits = W_out @ hidden + b_out -> [V_plus_1, k_span]
    ggml_tensor * logits = ggml_mul_mat(gctx, rt.weights->joint_out_w, hidden);
    logits = ggml_add(gctx, logits, rt.weights->joint_out_b);

    // Per-frame token argmax on-device where supported: k_span * 4 B
    // readback instead of k_span * V_plus_1 * 4 B (~64 KB) of logits.
    ggml_tensor * am = ggml_argmax(gctx, logits);  // i32[k_span]
    rt.argmax_on_gpu = ggml_backend_supports_op(rt.backend, am);

    JointBodyOuts outs{};
    outs.token_out = rt.argmax_on_gpu ? am : logits;
    return outs;
}

// (1) LSTM-only graph: init / `<EOU>` reset seeding and end-of-window
//     flush of a deferred update. Not in the per-frame hot loop.
void build_lstm_graph(EouRuntimeWeights & rt) {
    ggml_context * gctx = rt.gctx;

    rt.lstm_token_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_name(rt.lstm_token_in, "eou_lstm.token_in");
    ggml_set_input(rt.lstm_token_in);

    LstmBodyOuts outs = build_lstm_body(rt, gctx, rt.lstm_token_in);
    rt.lstm_pred_out = outs.pred_cpy;
    ggml_set_name(rt.lstm_pred_out, "eou_lstm.pred_out");
    ggml_set_output(rt.lstm_pred_out);
    // Mark EVERY layer's h/c state write as an output (not just the last), else
    // forward_expand prunes the non-last layers' writes and their state never
    // advances. See parakeet_tdt.cpp::LstmBodyOuts::state_cpy.
    for (ggml_tensor * n : outs.state_cpy) ggml_set_output(n);

    rt.g_lstm = ggml_new_graph_custom(gctx, /*size*/ 256, /*grads*/ false);
    for (ggml_tensor * n : outs.state_cpy) ggml_build_forward_expand(rt.g_lstm, n);
    ggml_build_forward_expand(rt.g_lstm, rt.lstm_pred_out);
}

// (2) Joint-only span graph: used while pred_persist is unchanged (blank
//     stretches). One launch scores up to k_span frames.
void build_joint_graph(EouRuntimeWeights & rt) {
    ggml_context * gctx = rt.gctx;

    rt.joint_frame_idx_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, EouRuntimeWeights::k_span);
    ggml_set_name(rt.joint_frame_idx_in, "eou_joint.frame_idx_in");
    ggml_set_input(rt.joint_frame_idx_in);

    JointBodyOuts outs = build_joint_span_body(rt, gctx, rt.pred_persist, rt.joint_frame_idx_in);
    rt.joint_token_out = outs.token_out;
    ggml_set_name(rt.joint_token_out, rt.argmax_on_gpu ? "eou_joint.token_argmax" : "eou_joint.token_logits");
    ggml_set_output(rt.joint_token_out);

    rt.g_joint = ggml_new_graph_custom(gctx, /*size*/ 96, /*grads*/ false);
    ggml_build_forward_expand(rt.g_joint, rt.joint_token_out);
}

// (3) Fused LSTM + span-joint graph: used after a non-blank emission.
//     One command-buffer commit runs the deferred LSTM update and scores
//     the span starting at the current frame against the fresh pred.
void build_lstm_joint_graph(EouRuntimeWeights & rt) {
    ggml_context * gctx = rt.gctx;

    rt.lj_token_in     = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    rt.lj_frame_idx_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, EouRuntimeWeights::k_span);
    ggml_set_name(rt.lj_token_in,     "eou_lstm_joint.token_in");
    ggml_set_name(rt.lj_frame_idx_in, "eou_lstm_joint.frame_idx_in");
    ggml_set_input(rt.lj_token_in);
    ggml_set_input(rt.lj_frame_idx_in);

    LstmBodyOuts lstm_outs = build_lstm_body(rt, gctx, rt.lj_token_in);
    // Use the pred_cpy node (not pred_persist directly) so the joint mat_muls
    // depend on the LSTM update finishing first.
    JointBodyOuts joint_outs = build_joint_span_body(rt, gctx, lstm_outs.pred_cpy, rt.lj_frame_idx_in);
    rt.lj_token_out = joint_outs.token_out;
    ggml_set_name(rt.lj_token_out, rt.argmax_on_gpu ? "eou_lstm_joint.token_argmax" : "eou_lstm_joint.token_logits");
    ggml_set_output(rt.lj_token_out);
    for (ggml_tensor * n : lstm_outs.state_cpy) ggml_set_output(n);

    rt.g_lstm_joint = ggml_new_graph_custom(gctx, /*size*/ 384, /*grads*/ false);
    ggml_build_forward_expand(rt.g_lstm_joint, rt.lj_token_out);
    for (ggml_tensor * n : lstm_outs.state_cpy) ggml_build_forward_expand(rt.g_lstm_joint, n);
}

// Full-window encoder-side projection straight into enc_proj_persist.
// Same per-graph context ownership + LRU rationale as
// parakeet_tdt.cpp::build_enc_proj_graph.
EouRuntimeWeights::EncProjGraph build_enc_proj_graph(EouRuntimeWeights & rt, int T) {
    EouRuntimeWeights::EncProjGraph g{};
    g.T = T;

    const int H_joint = rt.H_joint;
    const int D_enc   = rt.D_enc;

    const size_t graph_slots = 64;
    const size_t local_overhead = ggml_tensor_overhead() * graph_slots
                                + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params local_p = {};
    local_p.mem_size   = local_overhead;
    local_p.mem_buffer = nullptr;
    local_p.no_alloc   = true;
    g.ctx = ggml_init(local_p);
    if (!g.ctx) {
        std::fprintf(stderr, "eou: enc_proj ggml_init failed for T=%d\n", T);
        return g;
    }

    g.enc_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D_enc, T);
    ggml_set_name(g.enc_in, "eou_enc_proj.enc_in");
    ggml_set_input(g.enc_in);

    ggml_tensor * proj = ggml_mul_mat(g.ctx, rt.weights->joint_enc_w, g.enc_in);
    proj = ggml_add(g.ctx, proj, rt.weights->joint_enc_b);

    ggml_tensor * dst_view = ggml_view_2d(g.ctx, rt.enc_proj_persist,
                                           H_joint, T,
                                           (size_t) H_joint * sizeof(float),
                                           0);
    g.out = ggml_cpy(g.ctx, proj, dst_view);
    ggml_set_name(g.out, "eou_enc_proj.out_persist");
    ggml_set_output(g.out);

    g.cg = ggml_new_graph_custom(g.ctx, /*size*/ 32, /*grads*/ false);
    ggml_build_forward_expand(g.cg, g.out);

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
    if (!g.alloc || !ggml_gallocr_alloc_graph(g.alloc, g.cg)) {
        std::fprintf(stderr, "eou: failed to allocate enc_proj graph for T=%d\n", T);
        if (g.alloc) ggml_gallocr_free(g.alloc);
        g.alloc = nullptr;
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }

    return g;
}

void free_enc_proj_graph(EouRuntimeWeights::EncProjGraph & g) {
    if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
    if (g.ctx)   { ggml_free(g.ctx);           g.ctx   = nullptr; }
    g.cg = nullptr;
    g.enc_in = nullptr;
    g.out = nullptr;
}

const EouRuntimeWeights::EncProjGraph * get_enc_proj_graph(EouRuntimeWeights & rt, int T) {
    for (auto & g : rt.enc_proj_cache) {
        if (g.T == T) return &g;
    }
    if (rt.enc_proj_cache.size() >= EouRuntimeWeights::k_enc_proj_cache_max) {
        free_enc_proj_graph(rt.enc_proj_cache.front());
        rt.enc_proj_cache.erase(rt.enc_proj_cache.begin());
    }
    rt.enc_proj_cache.push_back(build_enc_proj_graph(rt, T));
    return &rt.enc_proj_cache.back();
}

bool compute_graph(EouRuntimeWeights & rt, ggml_cgraph * cg) {
    if (rt.n_threads > 0 && backend_is_cpu(rt.backend)) {
        backend_set_n_threads(rt.backend, rt.n_threads);
    }
    return ggml_backend_graph_compute(rt.backend, cg) == GGML_STATUS_SUCCESS;
}

// LSTM-only step (init / `<EOU>` reset seeding, end-of-window flush).
bool run_lstm_step(EouRuntimeWeights & rt, int token_id) {
    const int32_t tok = (int32_t) token_id;
    ggml_backend_tensor_set(rt.lstm_token_in, &tok, 0, sizeof(int32_t));

    if (!compute_graph(rt, rt.g_lstm)) {
        std::fprintf(stderr, "eou: LSTM graph compute failed\n");
        return false;
    }
    return true;
}

// Read the span's per-frame argmax into host ints: i32 indices when
// argmax_on_gpu, else host argmax over the f32 logit columns.
void resolve_joint_span(EouRuntimeWeights & rt,
                        ggml_tensor * tok_t,
                        int32_t * out_tok) {
    constexpr int S = EouRuntimeWeights::k_span;
    if (rt.argmax_on_gpu) {
        ggml_backend_tensor_get(tok_t, out_tok, 0, (size_t) S * sizeof(int32_t));
        return;
    }
    static thread_local std::vector<float> logits;
    logits.resize((size_t) rt.V_plus_1 * S);
    ggml_backend_tensor_get(tok_t, logits.data(), 0, logits.size() * sizeof(float));
    for (int i = 0; i < S; ++i) {
        out_tok[i] = (int32_t) argmax_f32(logits.data() + (size_t) i * rt.V_plus_1, rt.V_plus_1);
    }
}

// Score the span [t0, t0 + k_span) against the current pred. When
// `pending_token >= 0` the deferred LSTM update runs fused in the same
// graph commit before the joint reads pred. Tail indices past the window
// clamp to the last valid frame; the caller never reads those slots.
bool run_joint_span(EouRuntimeWeights & rt,
                    int pending_token,
                    int t0, int n_frames,
                    int32_t * out_tok) {
    constexpr int S = EouRuntimeWeights::k_span;
    int32_t idx[S];
    for (int i = 0; i < S; ++i) {
        idx[i] = (int32_t) std::min(t0 + i, n_frames - 1);
    }

    if (pending_token >= 0) {
        const int32_t tok = (int32_t) pending_token;
        ggml_backend_tensor_set(rt.lj_token_in, &tok, 0, sizeof(int32_t));
        ggml_backend_tensor_set(rt.lj_frame_idx_in, idx, 0, sizeof(idx));
        if (!compute_graph(rt, rt.g_lstm_joint)) {
            std::fprintf(stderr, "eou: lstm_joint graph compute failed\n");
            return false;
        }
        resolve_joint_span(rt, rt.lj_token_out, out_tok);
        return true;
    }

    ggml_backend_tensor_set(rt.joint_frame_idx_in, idx, 0, sizeof(idx));
    if (!compute_graph(rt, rt.g_joint)) {
        std::fprintf(stderr, "eou: joint graph compute failed\n");
        return false;
    }
    resolve_joint_span(rt, rt.joint_token_out, out_tok);
    return true;
}

// Compute the full-window encoder-side projection into enc_proj_persist.
bool run_enc_proj(EouRuntimeWeights & rt,
                  const float * encoder_out,
                  int T) {
    if (T <= 0) return true;
    if (T > rt.enc_proj_T_max) return false;

    const EouRuntimeWeights::EncProjGraph * g = get_enc_proj_graph(rt, T);
    if (!g || !g->alloc) return false;

    ggml_backend_tensor_set(g->enc_in, encoder_out, 0, (size_t) T * rt.D_enc * sizeof(float));

    if (!compute_graph(rt, g->cg)) {
        std::fprintf(stderr, "eou: enc_proj graph compute failed\n");
        return false;
    }
    return true;
}

// Zero h/c on-device and seed pred with one blank-token LSTM step. Used
// by eou_init_state and the `<EOU>` state reset. tensor_set of a host
// zero buffer instead of ggml_backend_tensor_memset for the same backend
// portability reason as tdt_init_state.
bool gpu_reset_predictor(EouRuntimeWeights & W) {
    const size_t h_bytes = ggml_nbytes(W.h_persist);
    const size_t c_bytes = ggml_nbytes(W.c_persist);
    static thread_local std::vector<uint8_t> zeros;
    zeros.assign(std::max(h_bytes, c_bytes), 0);
    ggml_backend_tensor_set(W.h_persist, zeros.data(), 0, h_bytes);
    ggml_backend_tensor_set(W.c_persist, zeros.data(), 0, c_bytes);
    return run_lstm_step(W, W.blank_id);
}

}  // anonymous namespace

EouRuntimeWeights::EouRuntimeWeights(EouRuntimeWeights && o) noexcept { *this = std::move(o); }

EouRuntimeWeights & EouRuntimeWeights::operator=(EouRuntimeWeights && o) noexcept {
    if (this == &o) return *this;
    // Free owned backend resources without ending the object's lifetime
    // (unlike TdtRuntimeWeights' destroy-then-assign, which is UB when the
    // destination was already populated — e.g. a second prepare_runtime on
    // the same object). The vector members are move-assigned below, which
    // frees their previous storage on its own.
    release();

    H_pred   = o.H_pred;
    H_joint  = o.H_joint;
    D_enc    = o.D_enc;
    V_plus_1 = o.V_plus_1;
    L        = o.L;
    blank_id = o.blank_id;
    eou_id   = o.eou_id;
    eob_id   = o.eob_id;
    weights  = o.weights;   o.weights = nullptr;
    backend  = o.backend;   o.backend = nullptr;
    n_threads     = o.n_threads;
    use_graphs    = o.use_graphs;
    argmax_on_gpu = o.argmax_on_gpu;
    embed = std::move(o.embed);
    lstm  = std::move(o.lstm);
    joint_enc_w  = std::move(o.joint_enc_w);
    joint_enc_b  = std::move(o.joint_enc_b);
    joint_pred_w = std::move(o.joint_pred_w);
    joint_pred_b = std::move(o.joint_pred_b);
    joint_out_w  = std::move(o.joint_out_w);
    joint_out_b  = std::move(o.joint_out_b);
    gctx            = o.gctx;            o.gctx = nullptr;
    persist_ctx     = o.persist_ctx;     o.persist_ctx = nullptr;
    persist_buffer  = o.persist_buffer;  o.persist_buffer = nullptr;
    h_persist        = o.h_persist;        o.h_persist = nullptr;
    c_persist        = o.c_persist;        o.c_persist = nullptr;
    pred_persist     = o.pred_persist;     o.pred_persist = nullptr;
    enc_proj_persist = o.enc_proj_persist; o.enc_proj_persist = nullptr;
    enc_proj_T_max   = o.enc_proj_T_max;
    g_lstm        = o.g_lstm;        o.g_lstm = nullptr;
    alloc_lstm    = o.alloc_lstm;    o.alloc_lstm = nullptr;
    lstm_token_in = o.lstm_token_in; o.lstm_token_in = nullptr;
    lstm_pred_out = o.lstm_pred_out; o.lstm_pred_out = nullptr;
    g_joint       = o.g_joint;       o.g_joint = nullptr;
    alloc_joint   = o.alloc_joint;   o.alloc_joint = nullptr;
    joint_frame_idx_in = o.joint_frame_idx_in; o.joint_frame_idx_in = nullptr;
    joint_token_out    = o.joint_token_out;    o.joint_token_out = nullptr;
    g_lstm_joint       = o.g_lstm_joint;       o.g_lstm_joint = nullptr;
    alloc_lstm_joint   = o.alloc_lstm_joint;   o.alloc_lstm_joint = nullptr;
    lj_token_in     = o.lj_token_in;     o.lj_token_in = nullptr;
    lj_frame_idx_in = o.lj_frame_idx_in; o.lj_frame_idx_in = nullptr;
    lj_token_out    = o.lj_token_out;    o.lj_token_out = nullptr;
    enc_proj_cache = std::move(o.enc_proj_cache);
    o.enc_proj_cache.clear();
    return *this;
}

void EouRuntimeWeights::release() noexcept {
    for (auto & g : enc_proj_cache) {
        free_enc_proj_graph(g);
    }
    enc_proj_cache.clear();
    if (alloc_lstm_joint) { ggml_gallocr_free(alloc_lstm_joint); alloc_lstm_joint = nullptr; }
    if (alloc_joint) { ggml_gallocr_free(alloc_joint); alloc_joint = nullptr; }
    if (alloc_lstm)  { ggml_gallocr_free(alloc_lstm);  alloc_lstm  = nullptr; }
    if (persist_buffer) { ggml_backend_buffer_free(persist_buffer); persist_buffer = nullptr; }
    if (persist_ctx) { ggml_free(persist_ctx); persist_ctx = nullptr; }
    if (gctx)        { ggml_free(gctx);        gctx        = nullptr; }
    // Graph/tensor pointers were parented on the freed contexts.
    g_lstm = nullptr;        lstm_token_in = nullptr;      lstm_pred_out = nullptr;
    g_joint = nullptr;       joint_frame_idx_in = nullptr; joint_token_out = nullptr;
    g_lstm_joint = nullptr;  lj_token_in = nullptr;        lj_frame_idx_in = nullptr;
    lj_token_out = nullptr;
    h_persist = nullptr; c_persist = nullptr; pred_persist = nullptr;
    enc_proj_persist = nullptr;
    // backend is owned by ParakeetCtcModel::Impl; don't free here.
}

EouRuntimeWeights::~EouRuntimeWeights() {
    release();
}

int eou_prepare_runtime(const ParakeetCtcModel & model, EouRuntimeWeights & W) {
    if (model.model_type != ParakeetModelType::EOU) {
        return 1;
    }
    W = EouRuntimeWeights{};

    W.H_pred   = model.encoder_cfg.eou_pred_hidden;
    W.H_joint  = model.encoder_cfg.eou_joint_hidden;
    W.D_enc    = model.encoder_cfg.d_model;
    W.L        = model.encoder_cfg.eou_pred_rnn_layers;
    W.V_plus_1 = (int) model.vocab_size + 1;
    W.blank_id = (int) model.blank_id;
    W.eou_id   = model.eou_id >= 0 ? model.eou_id : (int) model.vocab_size - 2;
    W.eob_id   = model.eob_id >= 0 ? model.eob_id : (int) model.vocab_size - 1;

    W.weights = &model.eou;
    W.backend = model.backend_active();
    if (!W.backend) {
        std::fprintf(stderr, "eou_prepare_runtime: model has no active backend (call load_from_gguf first)\n");
        return 2;
    }
    {
        const unsigned hc = std::thread::hardware_concurrency();
        W.n_threads = hc > 0 ? (int) hc : 4;
    }

    if (!model.eou.predict_embed || model.eou.lstm.empty() || !model.eou.joint_out_w) {
        std::fprintf(stderr, "eou_prepare_runtime: GGUF is missing EOU tensors\n");
        return 3;
    }

    // Same path split as tdt_prepare_runtime: per-step graph dispatch on the
    // CPU backend regresses heavily vs. the scalar gemv loop, and ggml-opencl
    // drops the in-place ggml_cpy writes that carry the persistent LSTM state
    // (h/c/pred), so both decode on the host. Other GPU backends
    // (Metal / CUDA / Vulkan) run the graph path.
    W.use_graphs = !backend_is_cpu(W.backend);
    if (W.use_graphs && std::strcmp(backend_reg_name(W.backend), "OpenCL") == 0) {
        W.use_graphs = false;
    }

    if (!W.use_graphs) {
        // ---- CPU fallback: dequantise weights to host f32 ----
        dequantize_to_f32(model.eou.predict_embed, W.embed);

        W.lstm.clear();
        W.lstm.resize(W.L);
        for (int l = 0; l < W.L; ++l) {
            dequantize_to_f32(model.eou.lstm[l].w_ih, W.lstm[l].w_ih);
            dequantize_to_f32(model.eou.lstm[l].w_hh, W.lstm[l].w_hh);
            dequantize_to_f32(model.eou.lstm[l].b_ih, W.lstm[l].b_ih);
            dequantize_to_f32(model.eou.lstm[l].b_hh, W.lstm[l].b_hh);
        }

        dequantize_to_f32(model.eou.joint_enc_w,  W.joint_enc_w);
        dequantize_to_f32(model.eou.joint_enc_b,  W.joint_enc_b);
        dequantize_to_f32(model.eou.joint_pred_w, W.joint_pred_w);
        dequantize_to_f32(model.eou.joint_pred_b, W.joint_pred_b);
        dequantize_to_f32(model.eou.joint_out_w,  W.joint_out_w);
        dequantize_to_f32(model.eou.joint_out_b,  W.joint_out_b);

        return 0;
    }

    // ---- GPU path: build ggml graphs against native GGUF weight tensors ----
    // Same two-context layout as tdt_prepare_runtime: persist_ctx/buffer for
    // the decoder state, gctx for the fixed-shape graph metadata.
    {
        ggml_init_params pp = {};
        pp.mem_size   = ggml_tensor_overhead() * 16 + 4 * 1024;
        pp.mem_buffer = nullptr;
        pp.no_alloc   = true;
        W.persist_ctx = ggml_init(pp);
        if (!W.persist_ctx) {
            std::fprintf(stderr, "eou_prepare_runtime: persist ggml_init failed\n");
            return 4;
        }

        const int T_max = EouRuntimeWeights::k_enc_proj_T_max;

        W.h_persist        = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, W.H_pred,  W.L);
        W.c_persist        = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, W.H_pred,  W.L);
        W.pred_persist     = ggml_new_tensor_1d(W.persist_ctx, GGML_TYPE_F32, W.H_pred);
        W.enc_proj_persist = ggml_new_tensor_2d(W.persist_ctx, GGML_TYPE_F32, W.H_joint, T_max);
        ggml_set_name(W.h_persist,        "eou.h_persist");
        ggml_set_name(W.c_persist,        "eou.c_persist");
        ggml_set_name(W.pred_persist,     "eou.pred_persist");
        ggml_set_name(W.enc_proj_persist, "eou.enc_proj_persist");

        W.persist_buffer = ggml_backend_alloc_ctx_tensors(W.persist_ctx, W.backend);
        if (!W.persist_buffer) {
            std::fprintf(stderr, "eou_prepare_runtime: failed to allocate persistent state buffer\n");
            return 5;
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
        std::fprintf(stderr, "eou_prepare_runtime: ggml_init failed\n");
        return 6;
    }

    build_lstm_graph(W);
    build_joint_graph(W);
    build_lstm_joint_graph(W);

    W.alloc_lstm       = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    W.alloc_joint      = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    W.alloc_lstm_joint = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    if (!W.alloc_lstm || !W.alloc_joint || !W.alloc_lstm_joint) {
        std::fprintf(stderr, "eou_prepare_runtime: failed to create gallocrs\n");
        return 7;
    }
    if (!ggml_gallocr_alloc_graph(W.alloc_lstm,       W.g_lstm) ||
        !ggml_gallocr_alloc_graph(W.alloc_joint,      W.g_joint) ||
        !ggml_gallocr_alloc_graph(W.alloc_lstm_joint, W.g_lstm_joint)) {
        std::fprintf(stderr, "eou_prepare_runtime: failed to allocate fixed-shape graphs\n");
        return 8;
    }

    return 0;
}

int eou_init_state(EouRuntimeWeights & W, EouDecodeState & state) {
    const int H = W.H_pred;
    const int L = W.L;

    state.last_token        = -1;
    state.symbols_this_step = 0;
    state.has_emitted_token_since_last_eou = false;

    // Prime the predictor with the blank token so the first joint call
    // sees a sensible context (matches NeMo's `initialize_state` +
    // first decoder forward with `targets=[blank]`).
    if (W.use_graphs) {
        if (!gpu_reset_predictor(W)) {
            PARAKEET_LOG_ERROR("eou_init_state: LSTM graph compute failed\n");
            return 1;
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
        const float * embed_row = W.embed.data() + (size_t) W.blank_id * H;
        lstm_step(W, embed_row, state.h_state.data(), state.c_state.data(),
                  scratch, layer_input_scratch);
        std::memcpy(state.pred_out.data(),
                    state.h_state.data() + (size_t) (L - 1) * H,
                    (size_t) H * sizeof(float));
    }

    state.initialized = true;
    return 0;
}

namespace {

// GPU span-batched greedy loop. Semantics are identical to the host loop
// in eou_decode_window below; the difference is purely in how joint
// results are produced:
//   - the joint for frames [t, t+k_span) is scored in one graph launch and
//     cached; the cache stays valid across frames because pred only
//     changes on a non-blank emission (blank / `<EOB>` / skipped special
//     tokens leave it untouched),
//   - a non-blank emission defers its LSTM update into the next launch
//     (fused graph), which restarts the span at the current frame so the
//     per-frame symbol cap re-checks against the fresh pred.
int eou_decode_window_gpu(const ParakeetCtcModel & model,
                          EouRuntimeWeights & W,
                          const float * encoder_out_window,
                          int n_frames,
                          const EouDecodeOptions & opts,
                          EouDecodeState & state,
                          std::vector<int32_t> & out_tokens,
                          std::vector<EouSegmentBoundary> & out_segments,
                          int & out_steps) {
    constexpr int S = EouRuntimeWeights::k_span;

    const int blank = W.blank_id;
    const int eou   = W.eou_id;
    const int eob   = W.eob_id;
    const int max_syms = std::max(1, opts.max_symbols_per_step);
    const size_t n_vocab = model.vocab.pieces.size();

    if (!run_enc_proj(W, encoder_out_window, n_frames)) return 6;

    int32_t span_tok[S];
    int span_start = -1;   // < 0 means "no valid cached span"
    int pending_lstm_token = -1;

    for (int t = 0; t < n_frames; ++t) {
        state.symbols_this_step = 0;

        while (state.symbols_this_step < max_syms) {
            if (span_start < 0 || t < span_start || t >= span_start + S) {
                if (!run_joint_span(W, pending_lstm_token, t, n_frames, span_tok)) return 7;
                pending_lstm_token = -1;
                span_start = t;
            }
            const int best = (int) span_tok[t - span_start];
            ++out_steps;

            if (best == blank) {
                break;
            }

            // <EOB>: training-time block boundary; treat as a no-op skip.
            if (best == eob) {
                break;
            }

            // <EOU>: flush the current segment, reset LSTM state, drop
            // back to the blank token as the predictor input. Match the
            // NeMo `eouDecodeChunk` reference: do NOT feed `<EOU>`
            // back into the predictor; reset h/c to zero and lastToken
            // to blank. (A pending LSTM update cannot exist here: any
            // emission invalidates the span, and the next launch —
            // which produced the argmax we just read — consumed it.)
            if (best == eou) {
                if (state.has_emitted_token_since_last_eou) {
                    EouSegmentBoundary boundary;
                    boundary.token_index  = (int) out_tokens.size();
                    boundary.is_eou_flush = true;
                    out_segments.push_back(boundary);
                    state.has_emitted_token_since_last_eou = false;
                }
                if (!gpu_reset_predictor(W)) return 8;
                state.last_token = blank;
                span_start = -1;
                break;
            }

            // Skip any other special token defensively (e.g. <unk>);
            // any vocab piece wrapped in `<...>` is treated as special.
            if (best >= 0 && (size_t) best < n_vocab) {
                const std::string & piece = model.vocab.pieces[best];
                if (!piece.empty() && piece.front() == '<' && piece.back() == '>') {
                    break;
                }
            }

            out_tokens.push_back((int32_t) best);
            state.has_emitted_token_since_last_eou = true;
            state.last_token = best;

            // Defer the LSTM update; the next joint launch runs it fused
            // and re-scores from the current frame with the fresh pred.
            pending_lstm_token = best;
            span_start = -1;
            ++state.symbols_this_step;
        }
    }

    // Streaming: flush a deferred LSTM update so the next decode_window
    // call sees the up-to-date pred_persist before its first joint.
    if (pending_lstm_token >= 0) {
        if (!run_lstm_step(W, pending_lstm_token)) return 9;
    }

    return 0;
}

}  // anonymous namespace

int eou_decode_window(const ParakeetCtcModel & model,
                      EouRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      std::vector<EouSegmentBoundary> & out_segments,
                      int & out_steps) {
    out_steps = 0;
    if (D_enc != W.D_enc) {
        PARAKEET_LOG_ERROR("eou_decode_window: encoder d_model mismatch (%d vs %d)\n",
                           D_enc, W.D_enc);
        return 1;
    }
    if (n_frames <= 0) return 0;
    if (!state.initialized) {
        if (eou_init_state(W, state) != 0) return 10;
    }

    if (W.use_graphs) {
        return eou_decode_window_gpu(model, W, encoder_out_window, n_frames,
                                     opts, state, out_tokens, out_segments, out_steps);
    }

    const int H     = W.H_pred;
    const int L     = W.L;
    const int V_p1  = W.V_plus_1;
    const int blank = W.blank_id;
    const int eou   = W.eou_id;
    const int eob   = W.eob_id;
    const int max_syms = std::max(1, opts.max_symbols_per_step);

    std::vector<float> scratch_lstm;
    std::vector<float> scratch_lstm_layer_input;
    std::vector<float> scratch_joint_tmp;
    std::vector<float> scratch_joint_hidden;
    std::vector<float> scratch_joint_logits;

    const size_t n_vocab = model.vocab.pieces.size();

    for (int t = 0; t < n_frames; ++t) {
        const float * enc_frame = encoder_out_window + (size_t) t * D_enc;
        state.symbols_this_step = 0;

        while (state.symbols_this_step < max_syms) {
            joint_step(W, enc_frame, state.pred_out.data(),
                       scratch_joint_hidden, scratch_joint_logits,
                       scratch_joint_tmp);
            ++out_steps;

            const int best = argmax_f32(scratch_joint_logits.data(), V_p1);
            if (best == blank) {
                break;
            }

            // <EOB>: training-time block boundary; treat as a no-op skip.
            if (best == eob) {
                break;
            }

            // <EOU>: flush the current segment, reset LSTM state, drop
            // back to the blank token as the predictor input. Match the
            // NeMo `eouDecodeChunk` reference: do NOT feed `<EOU>`
            // back into the predictor; reset h/c to zero and lastToken
            // to blank.
            if (best == eou) {
                if (state.has_emitted_token_since_last_eou) {
                    EouSegmentBoundary boundary;
                    boundary.token_index  = (int) out_tokens.size();
                    boundary.is_eou_flush = true;
                    out_segments.push_back(boundary);
                    state.has_emitted_token_since_last_eou = false;
                }
                state.h_state.assign((size_t) L * H, 0.0f);
                state.c_state.assign((size_t) L * H, 0.0f);
                state.last_token = blank;
                const float * embed_row = W.embed.data() + (size_t) blank * H;
                lstm_step(W, embed_row, state.h_state.data(),
                          state.c_state.data(), scratch_lstm,
                          scratch_lstm_layer_input);
                std::memcpy(state.pred_out.data(),
                            state.h_state.data() + (size_t) (L - 1) * H,
                            (size_t) H * sizeof(float));
                break;
            }

            // Skip any other special token defensively (e.g. <unk>);
            // any vocab piece wrapped in `<...>` is treated as special.
            if (best >= 0 && (size_t) best < n_vocab) {
                const std::string & piece = model.vocab.pieces[best];
                if (!piece.empty() && piece.front() == '<' && piece.back() == '>') {
                    break;
                }
            }

            out_tokens.push_back((int32_t) best);
            state.has_emitted_token_since_last_eou = true;

            const float * embed_row = W.embed.data() + (size_t) best * H;
            lstm_step(W, embed_row, state.h_state.data(),
                      state.c_state.data(), scratch_lstm,
                      scratch_lstm_layer_input);
            std::memcpy(state.pred_out.data(),
                        state.h_state.data() + (size_t) (L - 1) * H,
                        (size_t) H * sizeof(float));
            state.last_token = best;
            ++state.symbols_this_step;
        }
    }
    return 0;
}

int eou_greedy_decode(const ParakeetCtcModel & model,
                      EouRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeResult & result) {
    const auto t0 = std::chrono::steady_clock::now();

    EouDecodeState state;
    result.token_ids.clear();
    result.segments.clear();
    result.token_ids.reserve(T_enc);

    if (int rc = eou_decode_window(model, W, encoder_out, T_enc, D_enc,
                                   opts, state,
                                   result.token_ids, result.segments,
                                   result.steps);
        rc != 0) {
        return rc;
    }

    result.eou_count = (int) result.segments.size();

    int seg_start = 0;
    std::string out_text;
    for (size_t i = 0; i <= result.segments.size(); ++i) {
        const int seg_end = (i < result.segments.size())
                              ? result.segments[i].token_index
                              : (int) result.token_ids.size();
        if (seg_end <= seg_start) {
            seg_start = seg_end;
            continue;
        }
        std::vector<int32_t> seg_tokens(
            result.token_ids.begin() + seg_start,
            result.token_ids.begin() + seg_end);
        std::string seg_text = detokenize(model.vocab, seg_tokens);
        seg_text = trim_spaces(seg_text);
        if (!seg_text.empty()) {
            if (!out_text.empty()) out_text += "\n";
            out_text += seg_text;
        }
        seg_start = seg_end;
    }
    result.text = std::move(out_text);

    const auto t1 = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return 0;
}

}
