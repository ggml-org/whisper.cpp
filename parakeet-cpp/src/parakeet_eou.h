#pragma once

// EOU (end-of-utterance) RNN-T decoder: joint emits `<EOU>` / `<EOB>` alongside speech tokens.
//
// Single-layer predictor LSTM, joint network, and greedy decode with per-frame symbol caps.
// Encoder topology (cache-aware FastConformer, LayerNorm-in-conv when metadata says so) is in
// parakeet_ctc.cpp.
//
// Two decode paths, mirroring parakeet_tdt.h:
//   - GPU path  (Metal / CUDA / Vulkan, `use_graphs == true`): predictor LSTM,
//                joint and full-window enc-projection run as ggml graphs on the
//                active backend with native quantised GGUF weights. Unlike TDT
//                (which skips frames via its duration head), EOU evaluates the
//                joint at EVERY encoder frame, so the joint is batched over
//                spans of `k_span` frames per graph launch: the predictor state
//                only changes on non-blank emissions, so blank stretches cost
//                one launch per span instead of one per frame. This keeps the
//                launch count ~O(#emissions), which is what matters on
//                launch-overhead-bound backends (mobile Vulkan) while being
//                free on discrete GPUs.
//   - CPU path  (`use_graphs == false`): weights dequantised at load into host
//                buffers + scalar gemv loops (per-step graph dispatch on the
//                CPU backend has too much overhead; see parakeet_tdt.h).

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace parakeet {

// Per-layer LSTM weights, dequantised to host f32. Used by the CPU
// fallback decode path only.
struct EouRuntimeLstmLayer {
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
};

// Per-decoder runtime context; same dual-path shape as TdtRuntimeWeights.
//
// Move-only: graph scaffolding owns backend resources that must outlive the
// engine but cannot be duplicated. Like TDT, the GPU persistent state lives
// in this runtime, so one runtime supports one live decode state at a time.
struct EouRuntimeWeights {
    int H_pred  = 640;
    int H_joint = 640;
    int D_enc   = 512;
    int V_plus_1 = 1027;
    int L       = 1;

    int blank_id = 1026;
    int eou_id   = 1024;
    int eob_id   = 1025;

    const EouWeights * weights = nullptr;
    ggml_backend_t     backend = nullptr;
    int                n_threads = 0;
    bool               use_graphs = false;
    // Falls back to a host argmax over the span's logits when the backend
    // has no ARGMAX kernel; true elsewhere keeps the argmax on-device so
    // only k_span i32 come back per launch instead of k_span * V_plus_1 f32.
    bool               argmax_on_gpu = true;

    // ---- CPU-fallback host weights (populated only when !use_graphs) ----
    std::vector<float> embed;
    std::vector<EouRuntimeLstmLayer> lstm;

    std::vector<float> joint_enc_w;
    std::vector<float> joint_enc_b;
    std::vector<float> joint_pred_w;
    std::vector<float> joint_pred_b;
    std::vector<float> joint_out_w;
    std::vector<float> joint_out_b;

    // ---- GPU graph scaffolding (populated only when use_graphs) ----
    ggml_context * gctx = nullptr;

    // Persistent decoder state on the GPU backend. h, c, pred and the
    // full-window enc_proj stay in persist_buffer, wired with ggml_cpy,
    // so the per-span joint and per-emission LSTM never roundtrip
    // through the host.
    ggml_context *        persist_ctx    = nullptr;
    ggml_backend_buffer_t persist_buffer = nullptr;
    ggml_tensor *         h_persist        = nullptr;  // [H_pred, L]
    ggml_tensor *         c_persist        = nullptr;  // [H_pred, L]
    ggml_tensor *         pred_persist     = nullptr;  // [H_pred]
    ggml_tensor *         enc_proj_persist = nullptr;  // [H_joint, T_max]
    int                   enc_proj_T_max   = 0;

    // (1) LSTM-only graph: runs the predictor LSTM for one token and
    //     writes h/c/pred back in place. Used to seed pred after zeroing
    //     h/c (init and `<EOU>` reset) and to flush a deferred update at
    //     the end of a window.
    ggml_cgraph *  g_lstm     = nullptr;
    ggml_gallocr_t alloc_lstm = nullptr;
    ggml_tensor *  lstm_token_in = nullptr;  // i32[1]
    ggml_tensor *  lstm_pred_out = nullptr;  // ggml_cpy result aliasing pred_persist

    // (2) Joint-only span graph: scores `k_span` frames against the
    //     current pred_persist in one launch. Frame indices arrive as an
    //     i32[k_span] get_rows index tensor (tail frames repeat the last
    //     valid index; the host ignores those slots), so one fixed-shape
    //     graph serves any window position. Per-frame token argmax runs
    //     on-device; the readback is k_span * 4 B.
    ggml_cgraph *  g_joint     = nullptr;
    ggml_gallocr_t alloc_joint = nullptr;
    ggml_tensor *  joint_frame_idx_in = nullptr;  // i32[k_span]
    ggml_tensor *  joint_token_out    = nullptr;  // i32[k_span] argmax, or f32[V_plus_1, k_span] logits when !argmax_on_gpu

    // (3) Fused LSTM + span-joint graph: used after a non-blank emission.
    //     LSTM updates h/c/pred from the emitted token, then the span
    //     joint reads the *fresh* pred in the same compute_graph commit
    //     (one launch per emission instead of two).
    ggml_cgraph *  g_lstm_joint     = nullptr;
    ggml_gallocr_t alloc_lstm_joint = nullptr;
    ggml_tensor *  lj_token_in        = nullptr;  // i32[1]
    ggml_tensor *  lj_frame_idx_in    = nullptr;  // i32[k_span]
    ggml_tensor *  lj_token_out       = nullptr;  // i32[k_span] argmax, or f32[V_plus_1, k_span] logits when !argmax_on_gpu

    // Cached full-window enc-projection graphs keyed by frame count, LRU
    // capped; same design (and rationale) as TdtRuntimeWeights::EncProjGraph.
    struct EncProjGraph {
        ggml_context * ctx    = nullptr;
        ggml_cgraph *  cg     = nullptr;
        ggml_gallocr_t alloc  = nullptr;
        ggml_tensor *  enc_in = nullptr;
        ggml_tensor *  out    = nullptr;  // ggml_cpy aliasing enc_proj_persist[:T]
        int            T      = 0;
    };
    std::vector<EncProjGraph> enc_proj_cache;
    static constexpr size_t k_enc_proj_cache_max = 3;
    // ~5 minutes of audio at the encoder's 80 ms frame rate; H_joint=640
    // f32 rows -> ~10 MB on-device.
    static constexpr int k_enc_proj_T_max = 4096;

    // Joint span width. Trade-off: silence costs one launch per k_span
    // frames; every emission wastes at most k_span-1 speculatively scored
    // frames (~1.3 MFLOP each). 16 keeps the waste negligible on weak
    // GPUs while covering a typical 1 s streaming window in one launch.
    static constexpr int k_span = 16;

    EouRuntimeWeights() = default;
    EouRuntimeWeights(const EouRuntimeWeights &) = delete;
    EouRuntimeWeights & operator=(const EouRuntimeWeights &) = delete;
    EouRuntimeWeights(EouRuntimeWeights && other) noexcept;
    EouRuntimeWeights & operator=(EouRuntimeWeights && other) noexcept;
    ~EouRuntimeWeights();

    // Frees owned backend resources (graphs, gallocrs, persistent buffer,
    // ggml contexts) and resets the handles; keeps the object alive so it
    // can be re-assigned. Used by the destructor and move-assignment.
    void release() noexcept;
};

struct EouDecodeOptions {
    int max_symbols_per_step = 5;
};

struct EouDecodeState {
    // Host-side predictor state; unused (kept empty) on the GPU path,
    // where the state persists on-device in EouRuntimeWeights.
    std::vector<float> h_state;
    std::vector<float> c_state;
    std::vector<float> pred_out;

    int32_t last_token        = -1;     // last non-blank token fed back into the predictor
    int     symbols_this_step = 0;
    bool    initialized       = false;

    // Tracks whether the predictor has emitted a non-special token
    // since the last `<EOU>` flush. Guards `eou_decode_window` against
    // emitting an empty segment boundary when `<EOU>` fires before any
    // real token has been appended to `out_tokens`.
    bool    has_emitted_token_since_last_eou = false;
};

struct EouSegmentBoundary {
    int  token_index = 0;     // exclusive end-of-segment index in out_tokens
    bool is_eou_flush = true; // currently always true; reserved for future flags
};

struct EouDecodeResult {
    std::vector<int32_t>          token_ids;
    std::vector<EouSegmentBoundary> segments;
    std::string text;          // segments joined with '\n'
    int    steps      = 0;
    int    eou_count  = 0;
    double decode_ms  = 0.0;
};

int eou_prepare_runtime(const ParakeetCtcModel & model, EouRuntimeWeights & out);

// Zeroes the predictor state and primes it with one blank-token LSTM step.
// Returns 0 on success, non-zero if the GPU-path graph compute fails (same
// rc convention as eou_decode_window).
int eou_init_state(EouRuntimeWeights & W, EouDecodeState & state);

// Decode an arbitrary span of encoder frames. State is preserved across
// calls so the same decoder can be driven chunk-by-chunk in
// `EouStreamSession`. `out_tokens` accumulates **non-blank, non-EOU,
// non-EOB** token IDs; `out_segments` records the token-index where
// each `<EOU>` flush occurred so callers can split the transcript by
// utterance.
int eou_decode_window(const ParakeetCtcModel & model,
                      EouRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      std::vector<EouSegmentBoundary> & out_segments,
                      int & out_steps);

int eou_greedy_decode(const ParakeetCtcModel & model,
                      EouRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeResult & result);

}
