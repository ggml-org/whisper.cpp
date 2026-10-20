// Core ML TDT decoder sidecar for parakeet-cpp.
//
// Companion to parakeet-encoder.h. Runs the TDT prediction network and joint
// through Core ML. The prediction network is memory-bandwidth bound: every
// emission step reads the decoder weights, so the graph batches independent
// work into each dispatch.
//
// That is what the `lanes` axis is for. Windows cut out of one long clip decode
// independently (see src/independent_segments.h), so N windows can be stepped
// in lockstep by one dispatch that reads the weights once. The graph is
// exported at a fixed lane count by scripts/export-decoder-coreml-ane.py.
//
// What stays on the host: the TDT walk itself (src/parakeet_tdt.cpp,
// tdt_decode_lanes). The model scores frames and steps the LSTM; deciding
// which frame comes next from the emitted token and its duration is control
// flow. So the caller drives a loop, and per lane per step it:
//
//   1. looks up the embedding row for that lane's last emitted token (the
//      table is host-side because a gather is not an ANE op; the engine reads
//      it from the GGUF),
//   2. hands in that row, the lane's LSTM state, and the lane's current
//      encoder frame,
//   3. reads back logits, argmaxes token over [0, vocab+1) and duration over
//      [vocab+1, v_out), and advances that lane's frame cursor by the duration.
//
// Lanes are independent in the graph, so a lane that has run out of frames can
// be left holding its last input; its output is simply ignored. Every dispatch
// computes all `lanes` lanes, including empty ones.
//
// I/O contract, matching the exporter (all Float16, all shapes concrete):
//
//   embed      {lanes, h_pred,  1, 1}
//   h_in       {lanes, h_dim,   1, 1}   h_dim = h_pred * pred_rnn_layers
//   c_in       {lanes, h_dim,   1, 1}
//   enc_step   {lanes, d_enc,   1, width}
//   logits     {lanes, v_out,   1, width}
//   h_out      {lanes, h_dim,   1, 1}
//   c_out      {lanes, h_dim,   1, 1}
//
// `width` is the number of consecutive encoder frames scored per dispatch
// against one prediction state (1 by default). It is a property of the
// exported graph, not of the caller.
//
// This header is Apple-only; it is compiled and referenced solely when the
// PARAKEET_USE_COREML build definition is set. All entry points return failure
// rather than aborting, so the caller can fall back to the ggml decoder.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct parakeet_coreml_decoder;

// Load a compiled Core ML TDT decoder from a `.mlmodelc` directory.
// Returns nullptr on any failure (missing directory, load error, unsupported
// OS, unexpected model interface) so the caller falls back to ggml.
struct parakeet_coreml_decoder * parakeet_coreml_decoder_init(const char * path_mlmodelc);

// Release a decoder created by parakeet_coreml_decoder_init. Safe with nullptr.
void parakeet_coreml_decoder_free(struct parakeet_coreml_decoder * ctx);

// Report the shapes the graph was exported at, so the caller can size its
// buffers and check them against the GGUF's own geometry before using it.
// Any out-pointer may be null. Returns 0 on success.
int parakeet_coreml_decoder_geometry(const struct parakeet_coreml_decoder * ctx,
                                     int64_t * lanes,
                                     int64_t * h_pred,
                                     int64_t * h_dim,
                                     int64_t * d_enc,
                                     int64_t * width,
                                     int64_t * v_out);

// One dispatch: step every lane's LSTM once and score `width` frames per lane.
//
// All buffers are float32, host layout, lane-major and densely packed at the
// sizes parakeet_coreml_decoder_geometry reports:
//
//   embed       [lanes][h_pred]
//   h_in/c_in   [lanes][h_dim]
//   enc_step    [lanes][d_enc][width]   d_enc-major, matching the graph
//   logits      [lanes][v_out][width]
//   h_out/c_out [lanes][h_dim]
//
// h_out/c_out may alias h_in/c_in: the wrapper reads all inputs into the
// model's own buffers before writing any output back, so stepping state in
// place is safe and is the expected usage.
//
// Returns 0 on success; non-zero on a prediction failure, in which case the
// caller falls back to the ggml decoder for the whole utterance (a partially
// decoded lane batch is not recoverable).
int parakeet_coreml_decoder_step(struct parakeet_coreml_decoder * ctx,
                                 const float * embed,
                                 const float * h_in,
                                 const float * c_in,
                                 const float * enc_step,
                                 float       * logits,
                                 float       * h_out,
                                 float       * c_out);

// Human-readable requested-compute label for stats/logging, e.g. "coreml-ane".
// Core ML may still fall back to CPU for individual unsupported operations.
const char * parakeet_coreml_decoder_backend_label(const struct parakeet_coreml_decoder * ctx);

#if defined(__cplusplus)
}
#endif
