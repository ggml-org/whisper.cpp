#pragma once

// EOU (end-of-utterance) RNN-T decoder: joint emits `<EOU>` / `<EOB>` alongside speech tokens.
//
// Single-layer predictor LSTM, joint network, and greedy decode with per-frame symbol caps.
// Encoder topology (cache-aware FastConformer, LayerNorm-in-conv when metadata says so) is in
// parakeet_ctc.cpp; weights are dequantized at load into host buffers for the CPU decode path.

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include <cstdint>
#include <string>
#include <vector>

namespace parakeet {

// Per-layer LSTM weights, dequantised to host f32. Local to the EOU
// runtime which still uses the scalar-CPU decode path.
struct EouRuntimeLstmLayer {
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
};

struct EouRuntimeWeights {
    int H_pred  = 640;
    int H_joint = 640;
    int D_enc   = 512;
    int V_plus_1 = 1027;
    int L       = 1;

    int blank_id = 1026;
    int eou_id   = 1024;
    int eob_id   = 1025;

    std::vector<float> embed;
    std::vector<EouRuntimeLstmLayer> lstm;

    std::vector<float> joint_enc_w;
    std::vector<float> joint_enc_b;
    std::vector<float> joint_pred_w;
    std::vector<float> joint_pred_b;
    std::vector<float> joint_out_w;
    std::vector<float> joint_out_b;
};

struct EouDecodeOptions {
    int max_symbols_per_step = 5;
};

struct EouDecodeState {
    std::vector<float> h_state;
    std::vector<float> c_state;
    std::vector<float> pred_out;

    int32_t last_token        = -1;     // last non-blank token fed back into the predictor
    int     symbols_this_step = 0;
    bool    initialized       = false;

    // Index into the running token stream where the current segment
    // started. `eou_decode_window` writes "<EOU>" boundary positions
    // into the result so the caller can later split the transcript.
    int     segment_start_token = 0;
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

void eou_init_state(const EouRuntimeWeights & W, EouDecodeState & state);

// Decode an arbitrary span of encoder frames. State is preserved across
// calls so the same decoder can be driven chunk-by-chunk in
// `EouStreamSession`. `out_tokens` accumulates **non-blank, non-EOU,
// non-EOB** token IDs; `out_segments` records the token-index where
// each `<EOU>` flush occurred so callers can split the transcript by
// utterance.
int eou_decode_window(const ParakeetCtcModel & model,
                      const EouRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      std::vector<EouSegmentBoundary> & out_segments,
                      int & out_steps);

int eou_greedy_decode(const ParakeetCtcModel & model,
                      const EouRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeResult & result);

}
