// Core ML FastConformer encoder sidecar for parakeet-cpp.
//
// Mirrors the whisper.cpp `whisper_coreml_*` C shim: an opaque context wraps a
// compiled `.mlmodelc` encoder that runs on the Apple Neural Engine, and the
// caller hands log-mel features in / reads encoder hidden states out. The rest
// of the validated pipeline (mel preprocessing, CTC/RNN-T/TDT/EOU decode,
// tokenizer, and the Sortformer transformer/speaker head) stays on ggml.
//
// This header is Apple-only; it is compiled and referenced solely when the
// PARAKEET_USE_COREML build definition is set. All entry points return failure
// (nullptr / non-zero) rather than aborting so the caller can fall back to the
// ggml encoder on any error.
//
// Two export shapes bind here, told apart by their input names; the C entry
// points below are the same for both.
//
// Fixed-window Neural Engine export (scripts/export-encoder-coreml-ane.py):
//   - Three MLMultiArray inputs, all concrete:
//       `mel`       {1, n_mels, 1, window_mel_frames}
//       `pad_mask`  {1, 1, 1, enc_frames}          1.0 valid, 0.0 pad
//       `key_bias`  {1, enc_frames, 1, 1}          0.0 valid, -1e4 pad
//     plus, on exports that mask every subsampling resolution, a fourth:
//       `mel_mask`  {1, 1, 1, window_mel_frames}   1.0 valid, 0.0 pad
//   - One MLMultiArray output {1, d_model, 1, enc_frames}.
//   The graph only ever sees one window: the wrapper zero-pads a shorter input
//   to the window and masks the padded encoder frames, and
//   parakeet_coreml_fixed_mel_frames reports the window so the engine's
//   long-form plan tiles longer audio across overlapping windows.
//   parakeet_coreml_seam_context_frames reports the seam context the export
//   was validated with.
//
// Single-input export (scripts/export-encoder-coreml.py, the mobius `.mlmodelc`):
//   - Exactly one MLMultiArray input  = log-mel features, dims {n_mels, n_mel_frames}
//     (either order; Float16 or Float32; the wrapper adapts). This is the offline FastConformer encoder
//     input, i.e. everything from the subsampling stack through the last Conformer
//     block; it must NOT include a decoder joint projection. The mel time axis may be
//     fixed or flexible. For a fixed shape, an input no longer than the declared
//     capacity is zero-padded to that capacity. Longer inputs must be divided into
//     windows by the caller. Flexible inputs are allocated at the requested length.
//   - Exactly one MLMultiArray output = encoder hidden states, dims {n_enc_frames,
//     d_model} (either order; Float32 or Float16). n_enc_frames is n_mel_frames put
//     through the three stride-2 subsampling convs (matching run_encoder's sizing).
//   - CTC/IndicConformer, Unified RNN-T, and TDT exports are full-context and may
//     pad shorter inputs. The CTC vocabulary projection remains on ggml.
//     Sortformer batch and EOU exports require their exact fixed shape. EOU
//     additionally bakes
//     causal subsampling/convolution and chunk-limited attention into one exact
//     fixed shape; the caller rejects nonmatching EOU shapes before this wrapper.
//   - Sortformer v2.1 AOSC uses a second fixed-capacity sidecar with
//     `pre_encode` and `valid_mask` inputs. The wrapper pads shorter
//     post-subsampling cache/FIFO/chunk slabs and masks the padded attention
//     keys; slabs above capacity fall back to ggml.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct parakeet_coreml_context;

// Load a compiled Core ML encoder from a `.mlmodelc` directory.
// Returns nullptr on any failure (missing directory, load error, unsupported
// OS, unexpected model interface) so the caller falls back to ggml.
struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc);

struct parakeet_coreml_context * parakeet_coreml_init_with_units_env(const char * path_mlmodelc,
                                                                     const char * units_env);

// Release a context created by parakeet_coreml_init. Safe to call with nullptr.
void parakeet_coreml_free(struct parakeet_coreml_context * ctx);

// Return the fixed mel-frame capacity of the model input for `n_mels`.
// Returns 0 for flexible, enumerated-multi-shape, or unrecognised inputs.
// The value describes model capacity, not the current utterance length.
int64_t parakeet_coreml_fixed_mel_frames(
        const struct parakeet_coreml_context * ctx,
        int64_t n_mels);

int64_t parakeet_coreml_fixed_output_rows(
        const struct parakeet_coreml_context * ctx,
        int64_t cols);

// Encoder frames of seam context each long-form window should carry on both
// sides when the caller has no preference of its own: the fixed-window Neural
// Engine export was validated with 16 frames (1.28 s at the 80 ms stride),
// capped at a quarter of its window; PARAKEET_COREML_ANE_CONTEXT_FRAMES
// overrides it. 0 for every other export, meaning "use the engine default".
int64_t parakeet_coreml_seam_context_frames(const struct parakeet_coreml_context * ctx);

// Run the encoder on the Apple Neural Engine.
//
//   mel           log-mel features, row-major (n_mel_frames, n_mels),
//                 n_mels contiguous (the layout `compute_log_mel` emits).
//   n_mel_valid   frames of `mel` that hold audio; the frames from there to
//                 n_mel_frames are the front-end's zeroed padding and the
//                 fixed-window export masks them out at every resolution
//                 (<= 0 or >= n_mel_frames: every frame holds audio). Other
//                 exports take the whole slab and ignore it.
//   encoder_out   destination for the encoder hidden states, row-major
//                 (n_enc_frames, d_model), d_model contiguous. The caller
//                 sizes it to n_enc_frames * d_model floats.
//
// The wrapper adapts the mel/output tensor orientation to whatever the exported
// model declares (features-major or time-major), so the mobius export only has
// to agree on the dimension sizes, not their order.
//
// Returns 0 on success; non-zero on a shape mismatch or prediction failure, in
// which case the caller falls back to the ggml encoder.
int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mel_valid,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out);

// Sortformer v2.1 AOSC forward. `pre_encode` and `encoder_out` are both
// row-major (n_encoder_frames, d_model). The compiled input is a fixed
// capacity; the wrapper supplies the exported `valid_mask` when padding.
int parakeet_coreml_encode_embeddings(
        struct parakeet_coreml_context * ctx,
        int64_t n_encoder_frames,
        int64_t d_model,
        const float * pre_encode,
        float * encoder_out);

// Human-readable requested-compute label for stats/logging, e.g. "coreml-ane".
// Core ML may still fall back to CPU for individual unsupported operations.
const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx);

#if defined(__cplusplus)
}
#endif
