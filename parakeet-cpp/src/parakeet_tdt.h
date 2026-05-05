#pragma once

// TDT (token-and-duration transducer) decoder on FastConformer encoder output (run_encoder).
//
// Prediction LSTM, joint MLP, duration head, and greedy decode over encoder frames.
// GPU paths run per-step ops as ggml graphs on the loaded backend; CPU decode uses
// host GEMV/LSTM with weights prepared at load time.
//
// Typical layout (e.g. parakeet-tdt-0.6b-v3): 2-layer LSTM (hidden 640), joint with
// enc/pred projections and duration logits, greedy loop advancing the encoder index
// by predicted duration.

#include "parakeet_ctc.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace parakeet {

// Per-layer host-dequantised LSTM weights, used by the CPU fallback path
// (per-step ggml-graph dispatch on the CPU backend has too much overhead
// for the 200-300 emission steps in a typical 20s utterance).
struct TdtHostLstmLayer {
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
};

// Per-decoder runtime context. Two implementation paths:
//   - GPU path  (Metal / CUDA / Vulkan): per-step LSTM, joint and full-window
//                enc-projection are ggml graphs running on the active backend
//                with native quantised GGUF weights.
//   - CPU path  (`use_graphs == false`): host-dequantised f32 weights + scalar
//                gemv loops, matching the pre-Phase-13 implementation, since
//                per-step graph dispatch on the CPU backend regresses ~6x.
//
// Per-window enc-projection graphs are cached by frame count (up to
// `k_enc_proj_cache_max`) so streaming chunks of common sizes don't pay
// graph-build overhead more than once.
//
// Move-only: graph scaffolding owns backend resources that must outlive the
// engine but cannot be duplicated.
struct TdtRuntimeWeights {
    int H_pred       = 640;
    int H_joint      = 640;
    int D_enc        = 1024;
    int V_plus_1     = 8193;
    int V_out        = 8198;
    int L            = 2;
    int num_durations = 5;

    const TdtWeights * weights = nullptr;
    ggml_backend_t     backend = nullptr;
    int                n_threads = 0;
    bool               use_graphs = false;

    // ---- CPU-fallback host weights (populated only when !use_graphs) ----
    std::vector<float>             embed;
    std::vector<TdtHostLstmLayer>  host_lstm;
    std::vector<float>             host_joint_enc_w;
    std::vector<float>             host_joint_enc_b;
    std::vector<float>             host_joint_pred_w;
    std::vector<float>             host_joint_pred_b;
    std::vector<float>             host_joint_out_w;
    std::vector<float>             host_joint_out_b;

    // ---- GPU graph scaffolding (populated only when use_graphs) ----
    ggml_context * gctx = nullptr;

    // Persistent decoder state on the GPU backend (Metal/CUDA/Vulkan).
    //
    // Each emission step pays command-buffer submit/wait overhead; splitting
    // joint and LSTM into separate graphs per step doubled that cost. Here
    // h, c, pred, and full-window enc_proj stay in persist_buffer, wired with
    // ggml_cpy, plus a fused `g_lstm_joint` graph after non-blank emissions
    // so LSTM update and joint run in one graph commit when possible.
    ggml_context *           persist_ctx    = nullptr;
    ggml_backend_buffer_t    persist_buffer = nullptr;
    ggml_tensor *            h_persist        = nullptr;  // [H_pred, L]
    ggml_tensor *            c_persist        = nullptr;  // [H_pred, L]
    ggml_tensor *            pred_persist     = nullptr;  // [H_pred]
    ggml_tensor *            enc_proj_persist = nullptr;  // [H_joint, T_max]
    int                      enc_proj_T_max   = 0;

    // (1) Init-only LSTM graph: zeroes h/c and runs LSTM with the blank
    //     token to seed pred_persist. Used once per call (tdt_init_state).
    ggml_cgraph *  g_lstm     = nullptr;
    ggml_gallocr_t alloc_lstm = nullptr;
    ggml_tensor *  lstm_token_in = nullptr;
    ggml_tensor *  lstm_h_out    = nullptr;  // ggml_cpy result aliasing h_persist
    ggml_tensor *  lstm_c_out    = nullptr;  // ggml_cpy result aliasing c_persist
    ggml_tensor *  lstm_pred_out = nullptr;  // ggml_cpy result aliasing pred_persist

    // (2) Joint-only graph: used after a blank emission (pred unchanged
    //     from previous iteration). Reads pred_persist + enc_proj_persist
    //     [frame_idx]; emits token + dur argmax i32 indices instead of
    //     full logits to keep PCIe-class backends from paying the
    //     V_out * 4 B readback per step (~32 KB at V_out = 8198, fine on
    //     Apple unified memory at ~17 us / call but order-of-magnitude
    //     worse on a discrete GPU bus; one int32 per step is the right
    //     shape for both).  Token argmax is over logits[0:V_plus_1],
    //     duration argmax is over logits[V_plus_1:V_plus_1+num_durations].
    ggml_cgraph *  g_joint     = nullptr;
    ggml_gallocr_t alloc_joint = nullptr;
    ggml_tensor *  joint_frame_idx_in = nullptr;  // i32[1]
    ggml_tensor *  joint_token_out    = nullptr;  // i32[1] — token argmax
    ggml_tensor *  joint_dur_out      = nullptr;  // i32[1] — duration argmax

    // (3) Fused LSTM + joint graph: used after a non-blank emission.
    //     LSTM updates h/c/pred from the last emitted token, then joint
    //     reads the *fresh* pred and enc_proj_persist[frame_idx] in the
    //     same compute_graph (one command-buffer commit instead of two).
    //     Same on-device argmax as g_joint.
    ggml_cgraph *  g_lstm_joint     = nullptr;
    ggml_gallocr_t alloc_lstm_joint = nullptr;
    ggml_tensor *  lj_token_in        = nullptr;  // i32[1]
    ggml_tensor *  lj_frame_idx_in    = nullptr;  // i32[1]
    ggml_tensor *  lj_token_out       = nullptr;  // i32[1] — token argmax
    ggml_tensor *  lj_dur_out         = nullptr;  // i32[1] — duration argmax

    struct EncProjGraph {
        // Each cached graph owns its own ggml_context for the cgraph + tensor
        // metadata.  Previous design parented these on `gctx` (the long-lived
        // runtime context), and the LRU eviction below freed only the gallocr
        // — which leaks ~32 graph slots from gctx per evicted entry.  Mode 1
        // (single T_enc per call) hits this once and is fine; Mode 3 streaming
        // with varying right-lookahead-ms can chew through the gctx slot pool
        // and silently fail-to-allocate after ~30 distinct T_enc values.
        // Owning the metadata locally and freeing it at eviction keeps Mode 3
        // bounded by the LRU cap regardless of how many distinct T_enc values
        // a session sees.
        ggml_context * ctx    = nullptr;
        ggml_cgraph *  cg     = nullptr;
        ggml_gallocr_t alloc  = nullptr;
        ggml_tensor *  enc_in = nullptr;
        ggml_tensor *  out    = nullptr;  // ggml_cpy aliasing enc_proj_persist[:T]
        int            T      = 0;
    };
    std::vector<EncProjGraph> enc_proj_cache;
    static constexpr size_t k_enc_proj_cache_max = 3;
    // ~5 minutes of audio at the encoder's 80 ms-frame rate fits in 4096
    // rows; H_joint=640 * f32 → ~10 MB, fine for any backend we target.
    // Audio that exceeds this falls back to a per-call dynamic-T allocation.
    static constexpr int k_enc_proj_T_max = 4096;

    TdtRuntimeWeights() = default;
    TdtRuntimeWeights(const TdtRuntimeWeights &) = delete;
    TdtRuntimeWeights & operator=(const TdtRuntimeWeights &) = delete;
    TdtRuntimeWeights(TdtRuntimeWeights && other) noexcept;
    TdtRuntimeWeights & operator=(TdtRuntimeWeights && other) noexcept;
    ~TdtRuntimeWeights();

    bool ready() const { return weights != nullptr; }
};

struct TdtDecodeOptions {
    int max_symbols_per_step = 10;
};

struct TdtDecodeResult {
    std::vector<int32_t> token_ids;
    std::string text;
    int steps = 0;
    double decode_ms = 0.0;
};

struct TdtDecodeState {
    std::vector<float> h_state;
    std::vector<float> c_state;
    std::vector<float> pred_out;

    int  symbols_this_step = 0;
    bool initialized       = false;
    int  carry_frames      = 0;
};

int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & out);

void tdt_init_state(TdtRuntimeWeights & W,
                    int blank_id,
                    TdtDecodeState & state);

int tdt_decode_window(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      int & out_steps);

int tdt_greedy_decode(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeResult & result);

}
