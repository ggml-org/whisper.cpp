#pragma once

// RNN-T and TDT decoders on FastConformer encoder output (run_encoder).
//
// Prediction LSTM, joint MLP, optional duration head, and greedy decode over encoder frames.
// GPU paths run per-step ops as ggml graphs on the loaded backend; CPU decode uses
// host GEMV/LSTM with weights prepared at load time.
//
// Typical layout (e.g. parakeet-tdt-0.6b-v3): 2-layer LSTM (hidden 640), joint with
// enc/pred projections and duration logits, greedy loop advancing the encoder index
// by predicted duration.

#include "parakeet_ctc.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace parakeet {

struct TransducerGraphOutputs {
    ggml_tensor * token = nullptr;
    ggml_tensor * duration = nullptr;
};

TransducerGraphOutputs build_transducer_argmax_outputs(ggml_context * ctx,
                                                       ggml_tensor * logits,
                                                       int token_count,
                                                       int duration_count);

// Load-time [w_ih | w_hh] stack plus the pre-summed bias, so one GEMV per LSTM
// layer replaces the separate input and recurrent matmuls on the graph path.
struct TdtLstmCatWeights {
    ggml_tensor * w = nullptr;  // [2 * H_pred, 4 * H_pred], same type as w_ih
    ggml_tensor * b = nullptr;  // f32 [4 * H_pred], b_ih + b_hh
};

// Stack two weight matrices row-wise: row r of `dst` is w_ih row r followed by
// w_hh row r, so mul_mat(dst, concat(x, h)) == w_ih @ x + w_hh @ h. Quantised
// rows are whole blocks whenever the input width is a multiple of the block
// size, so this is a byte copy with no requantisation.
inline void pack_lstm_input_rows(int64_t n_rows,
                                 size_t ih_row_bytes, size_t hh_row_bytes,
                                 const uint8_t * w_ih, const uint8_t * w_hh,
                                 uint8_t * dst) {
    const size_t dst_row_bytes = ih_row_bytes + hh_row_bytes;
    for (int64_t r = 0; r < n_rows; ++r) {
        uint8_t * row = dst + (size_t) r * dst_row_bytes;
        std::memcpy(row,                w_ih + (size_t) r * ih_row_bytes, ih_row_bytes);
        std::memcpy(row + ih_row_bytes, w_hh + (size_t) r * hh_row_bytes, hh_row_bytes);
    }
}

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
    ggml_tensor *      graph_embed = nullptr; // native embedding or F32 copy in persist_ctx
    ggml_backend_t     backend = nullptr;
    int                n_threads = 0;
    bool               use_graphs = false;
    // false on ggml-opencl (no ARGMAX kernel): the joint graph emits raw logits
    // for the host to argmax; true elsewhere keeps the argmax on-device.
    bool               argmax_on_gpu = true;
    bool               fused_lstm_cell = false;   // GGML_OP_LSTM_CELL replaces the per-gate graph
    // One GEMV over the [w_ih | w_hh] stack replaces the two per-layer matmuls
    // and the two bias adds. Changes float summation order, so it is not
    // bit-exact with the split form; the gate is token-stream equality.
    bool               concat_lstm_input = false;
    bool               rnnt_mode = false;  // no duration head: advance one frame per step

    // Greedy steps unrolled into one decoder graph on the GPU path. Each step
    // costs one command-buffer commit and one host readback when run alone, so
    // running several per graph amortises that fixed cost. 1 keeps one step per
    // graph, 0 or less falls back to the sequential per-step loop.
    static constexpr int k_unroll_steps_default = 8;
    int                unroll_steps = k_unroll_steps_default;

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

    // Derived [w_ih | w_hh] stacks, built once at prepare time from the GGUF
    // tensors and owned here (populated only when concat_lstm_input).
    ggml_context *              lstm_cat_ctx    = nullptr;
    ggml_backend_buffer_t       lstm_cat_buffer = nullptr;
    std::vector<TdtLstmCatWeights> lstm_cat;

    // Persistent decoder state on the GPU backend (Metal/CUDA/Vulkan).
    //
    // Each emission step pays command-buffer submit/wait overhead; splitting
    // joint and LSTM into separate graphs per step doubled that cost. Here
    // the LSTM state and the full-window enc_proj stay in persist_buffer, wired
    // with ggml_cpy, plus a fused `g_lstm_joint` graph after non-blank emissions
    // so LSTM update and joint run in one graph commit when possible.
    ggml_context *           persist_ctx    = nullptr;
    ggml_backend_buffer_t    persist_buffer = nullptr;
    // Row l is the layer's [h | c] pair, matching ggml_lstm_cell's [2H] result
    // so the fused path writes both halves back with a single ggml_cpy.
    ggml_tensor *            hc_persist       = nullptr;  // [2 * H_pred, L]
    ggml_tensor *            pred_persist     = nullptr;  // view of hc_persist row L-1, h half
    ggml_tensor *            enc_proj_persist = nullptr;  // [H_joint, T_max]
    int                      enc_proj_T_max   = 0;

    // (1) Init-only LSTM graph: zeroes h/c and runs LSTM with the blank
    //     token to seed pred_persist. Used once per call (tdt_init_state).
    ggml_cgraph *  g_lstm     = nullptr;
    ggml_gallocr_t alloc_lstm = nullptr;
    ggml_tensor *  lstm_token_in = nullptr;
    ggml_tensor *  lstm_pred_out = nullptr;  // last layer's state write, h half

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
    ggml_tensor *  joint_token_out    = nullptr;  // i32 argmax, or f32 token logits when !argmax_on_gpu
    ggml_tensor *  joint_dur_out      = nullptr;  // i32 argmax, or f32 dur logits when !argmax_on_gpu

    // (3) Fused LSTM + joint graph: used after a non-blank emission.
    //     LSTM updates h/c/pred from the last emitted token, then joint
    //     reads the *fresh* pred and enc_proj_persist[frame_idx] in the
    //     same compute_graph (one command-buffer commit instead of two).
    //     Same on-device argmax as g_joint.
    ggml_cgraph *  g_lstm_joint     = nullptr;
    ggml_gallocr_t alloc_lstm_joint = nullptr;
    ggml_tensor *  lj_token_in        = nullptr;  // i32[1]
    ggml_tensor *  lj_frame_idx_in    = nullptr;  // i32[1]
    ggml_tensor *  lj_token_out       = nullptr;  // i32 argmax, or f32 token logits when !argmax_on_gpu
    ggml_tensor *  lj_dur_out         = nullptr;  // i32 argmax, or f32 dur logits when !argmax_on_gpu

    // Frame advance per duration index and the RNN-T placeholder index, both
    // uploaded once with the persistent state and read by the control op.
    ggml_tensor *  dur_table    = nullptr;  // i32[num_durations]
    ggml_tensor *  zero_dur_idx = nullptr;  // i32[1], always 0

    // (4) Unrolled decode graph: `unroll_steps` greedy steps (joint -> argmax ->
    //     LSTM -> ggml_tdt_step control -> masked state update) in one commit,
    //     with the host reading back the control rows at the end and replaying
    //     the same integer rules over their pairs. Blank id and
    //     max_symbols_per_step only arrive with the decode call, so the graph is
    //     built lazily and rebuilt when either changes. It owns its context so
    //     a rebuild does not consume gctx slots.
    ggml_context * unroll_ctx        = nullptr;
    ggml_cgraph *  g_unroll          = nullptr;
    ggml_gallocr_t alloc_unroll      = nullptr;
    // i32[GGML_TDT_STEP_N_OUTS, unroll_steps + 1]: row 0 is the counters the
    // host uploads, row k + 1 is the control row step k writes.
    ggml_tensor *  un_ctl            = nullptr;
    int            unroll_blank_id    = -1;
    int            unroll_max_symbols = 0;

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
    // Longer offline sequences use bounded windows with shared predictor state.
    static constexpr int k_enc_proj_T_max = 4096;

    TdtRuntimeWeights() = default;
    TdtRuntimeWeights(const TdtRuntimeWeights &) = delete;
    TdtRuntimeWeights & operator=(const TdtRuntimeWeights &) = delete;
    TdtRuntimeWeights(TdtRuntimeWeights && other) noexcept;
    TdtRuntimeWeights & operator=(TdtRuntimeWeights && other) noexcept;
    ~TdtRuntimeWeights();
    void release();

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

using RnntRuntimeWeights = TdtRuntimeWeights;
using RnntDecodeOptions = TdtDecodeOptions;
using RnntDecodeResult = TdtDecodeResult;
using RnntDecodeState = TdtDecodeState;

// ── Batched decode of independent windows ────────────────────────────────
//
// The Core ML TDT decoder sidecar steps several independent windows per
// dispatch (src/coreml/parakeet-decoder.h). tdt_decode_lanes drives it with
// the same walk as tdt_decode_window, `lanes` windows wide, behind the
// TdtLaneStepper interface so the walk itself is unit-tested without Core ML
// (test/test_independent_segments.cpp).

// One independently decoded window: a slice of encoder output that gets its
// own TDT state. Slices point into a buffer the caller owns and must outlive
// the call; nothing here copies encoder output.
struct TdtLaneWindow {
    const float * enc      = nullptr;  // [n_frames][d_enc], row-major
    int           n_frames = 0;
};

// What the lane walk needs to know about the checkpoint. tdt_lane_geometry()
// fills it from a prepared runtime; unit tests fill it by hand.
struct TdtLaneGeometry {
    int              blank_id         = 0;
    int              vocab_plus_blank = 0;  // token logits occupy [0, vocab_plus_blank)
    std::vector<int> durations;             // frame advance per duration logit; empty = index
    int              h_pred           = 0;
    int              layers           = 0;
    int              d_enc            = 0;
};

TdtLaneGeometry tdt_lane_geometry(const ParakeetCtcModel & model, const TdtRuntimeWeights & W);

// One batched prediction step over `lanes` independent decoder states: the
// sidecar in production, a scripted double in tests. The geometry is fixed per
// instance. Buffers are float32, lane-major and densely packed:
//   embed [lanes][h_pred]   h_in/c_in [lanes][h_dim]   enc_step [lanes][d_enc]
//   logits [lanes][v_out]   h_out/c_out [lanes][h_dim]
// where h_dim = h_pred * layers and v_out = vocab_plus_blank + durations.
// Inputs are read before any output is written, so h_out may alias h_in.
struct TdtLaneStepper {
    virtual ~TdtLaneStepper() = default;
    virtual int64_t lanes()  const = 0;
    virtual int64_t h_pred() const = 0;
    virtual int64_t h_dim()  const = 0;
    virtual int64_t d_enc()  const = 0;
    virtual int64_t v_out()  const = 0;
    virtual int step(const float * embed, const float * h_in, const float * c_in,
                     const float * enc_step, float * logits, float * h_out, float * c_out) = 0;
};

struct TdtLaneStats {
    int dispatches = 0;  // batched steps issued
    int rescued    = 0;  // windows re-decoded warm-started from their predecessor
};

// Decodes independent windows, `stepper.lanes()` of them per dispatch, and
// emits one token vector per window in input order so the caller can
// detokenise and join them.
//
// `embed_table` is the prediction network's token embedding as host f32,
// [vocab_plus_blank][h_pred] (the GGUF table dequantised; a gather is not an
// ANE op). Windows are seated into lanes as lanes free up, so the batch stays
// full until the input runs out. A window that emits nothing at all is
// re-decoded in a second batched pass, warm-started from the previous window's
// retired state.
//
// Returns 0 on success. Any non-zero return leaves out_tokens unusable and
// means the caller should fall back to tdt_greedy_decode for the whole
// utterance: a half-decoded lane batch cannot be resumed.
int tdt_decode_lanes(const TdtLaneGeometry & geom,
                     const std::vector<float> & embed_table,
                     TdtLaneStepper & stepper,
                     const std::vector<TdtLaneWindow> & windows,
                     const TdtDecodeOptions & opts,
                     std::vector<std::vector<int32_t>> & out_tokens,
                     TdtLaneStats * stats);

// allow_fused_lstm = false keeps the decomposed per-gate LSTM graph, and
// allow_concat_lstm = false keeps the separate input/recurrent matmuls (parity tests).
int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & out,
                        bool allow_fused_lstm = true, bool allow_concat_lstm = true);

// Greedy steps per decoder graph on the GPU path: 8 by default, 1 for one step
// per graph, 0 or less for the sequential per-step loop. Drops any built graph
// so the next decode rebuilds it.
void tdt_set_unroll_steps(TdtRuntimeWeights & W, int steps);

// Fit projection: size everything tdt_prepare_runtime would allocate (see
// DecoderFitMeasure in parakeet_ctc.h) without allocating. Requires a
// load_from_gguf_metadata_only model. `worst_enc_frames` sizes the worst-case
// per-window enc-projection graph (clamped to k_enc_proj_T_max). Also covers
// RNNT models (same runtime).
int tdt_measure_runtime(const ParakeetCtcModel & model, int worst_enc_frames,
                        DecoderFitMeasure & out);

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

int rnnt_decode_window(const ParakeetCtcModel & model,
                       RnntRuntimeWeights & weights,
                       const float * encoder_out_window,
                       int n_frames,
                       int encoder_dim,
                       const RnntDecodeOptions & options,
                       RnntDecodeState & state,
                       std::vector<int32_t> & out_tokens,
                       int & out_steps);

int rnnt_greedy_decode(const ParakeetCtcModel & model,
                       RnntRuntimeWeights & weights,
                       const float * encoder_out,
                       int encoder_frames,
                       int encoder_dim,
                       const RnntDecodeOptions & options,
                       RnntDecodeResult & result);

// Decode a long encoder sequence through bounded encoder-projection windows
// while preserving one predictor state across every window.
int rnnt_greedy_decode_chunked(const ParakeetCtcModel & model,
                               RnntRuntimeWeights & weights,
                               const float * encoder_out,
                               int encoder_frames,
                               int encoder_dim,
                               int chunk_frames,
                               const RnntDecodeOptions & options,
                               RnntDecodeResult & result);

}
