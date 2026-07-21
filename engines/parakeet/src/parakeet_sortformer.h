#pragma once

// Sortformer diarization head: encoder projection, transformer stack, sigmoid speaker logits.
//
// Data flow (ggml graph on the model backend):
//
//   encoder_out (T, D_enc)
//     -> encoder_proj  : Linear(D_enc -> tf_d)
//     -> transformer   : N_tf_layers x post-LN block
//                        (multi-head self-attn -> residual+LN -> FFN -> residual+LN)
//     -> head          : ReLU -> first_hidden_to_hidden(tf_d -> tf_d)
//                        -> ReLU -> single_hidden_to_spks(tf_d -> num_spks)
//                        -> sigmoid
//   speaker_probs (T, num_spks) in [0, 1]
//
// Streaming (v2.1, AOSC) data flow:
//
//   chunk_audio
//     -> run_subsampling                 (mel -> pre_encode_embs, 8x downsample)
//     -> concat [spkcache | fifo | chunk]
//     -> run_encoder(bypass_pre_encode)  (17 FastConformer blocks; full self-attn)
//     -> sortformer_diarize_ggml         (encoder_proj + 18 transformer blocks + head)
//   then `streaming_update`:
//     -> append committed chunk slice to FIFO
//     -> when FIFO overflows, pop pop_out frames into spkcache
//     -> when spkcache overflows, compress via NeMo's _compress_spkcache
//
// All AOSC state (spkcache/fifo/mean_sil_emb/...) is in the *post-subsampling,
// pre-conformer-layers* embedding space (`fc_d_model`, 512 in v2.1).

#include "parakeet_ctc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace parakeet {

struct SortformerDiarizationOptions {
    float threshold = 0.5f;
};

struct SortformerSegment {
    int    speaker_id = 0;
    double start_s    = 0.0;
    double end_s      = 0.0;
};

struct SortformerDiarizationResult {
    int n_frames     = 0;
    int num_spks     = 0;
    double frame_stride_s = 0.08;
    std::vector<float> speaker_probs;
    std::vector<SortformerSegment> segments;
    double decode_ms = 0.0;
};

// AOSC compression policy + cache geometry, ported from NeMo's
// `nemo/collections/asr/modules/sortformer_modules.py` SortformerModules
// __init__ defaults and overridden at inference in
// `examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py`
// (the production inference path). Values below match the e2e inference defaults
// for diar_streaming_sortformer_4spk-v2.1.
struct SortformerStreamingConfig {
    // Cache geometry (encoder-frame units; 1 enc frame = subsampling_factor mel frames).
    int   spkcache_len               = 188;   // total cache rows = ~15s of 80ms frames
    int   fifo_len                   = 188;   // FIFO warmup buffer
    int   chunk_len                  = 6;     // committed encoder frames per step (~480ms)
    int   chunk_left_context         = 1;     // encoder frames of left audio context
    int   chunk_right_context        = 7;     // encoder frames of right audio context
    int   spkcache_update_period     = 144;   // pop_out_len when FIFO overflows
    int   spkcache_sil_frames_per_spk = 3;    // A_silence rows per speaker

    // Compression scoring policy.
    float sil_threshold              = 0.2f;  // sum-of-probs < this => silence frame
    float pred_score_threshold       = 0.25f; // log-arg clamp (NOT segmentation thresh)
    float scores_boost_latest        = 0.05f; // boost for newest frames in score
    float strong_boost_rate          = 0.75f; // K_strong = floor(spkcache_per_spk * 0.75)
    float weak_boost_rate            = 1.5f;  // K_weak   = floor(spkcache_per_spk * 1.5)
    float min_pos_scores_rate        = 0.5f;  // floor(spkcache_per_spk * 0.5)
};

// Per-session AOSC (Audio-Online Speaker Cache) state for v2.1 streaming.
// All embedding buffers are in the post-subsampling, pre-conformer-layers space
// (`fc_d_model`, 512 in v2.1). Predictions are in (T, num_spks) sigmoid space.
//
// Mirrors `StreamingSortformerState` in NeMo's sortformer_modules.py — fields
// here named to match. Async/batched fields collapsed to a single batch=1 case.
struct SortformerSpeakerCache {
    // Long-term speaker cache. Empty until FIFO has popped at least once.
    std::vector<float> spkcache;             // (n_rows, D)
    int                n_rows = 0;
    std::vector<float> spkcache_preds;       // (n_rows, num_spks)
    bool               spkcache_preds_valid = false;

    // FIFO of most-recent committed chunk rows. Sync'd in length with fifo_preds.
    std::vector<float> fifo;                 // (n_fifo, D)
    int                n_fifo = 0;
    std::vector<float> fifo_preds;           // (n_fifo, num_spks)

    // Runtime silence statistics. mean_sil_emb is a running mean of
    // embeddings of frames whose sum-of-speaker-probs is below sil_threshold;
    // disabled rows in compressed cache get filled from here.
    std::vector<float> mean_sil_emb;         // (D,)
    int64_t            n_sil_frames = 0;

    int                chunk_index = 0;
};

// Reset to a fresh empty state. Allocates mean_sil_emb to D zeros.
void sortformer_cache_reset(SortformerSpeakerCache & cache, int D);

// The diarization head backend is resolved internally via model_sortformer_backend
// (CPU on Mali-Vulkan, the active backend otherwise) so callers cannot accidentally
// drive the CPU-resident force-CPU path through the GPU.
int  sortformer_diarize_ggml(const ParakeetCtcModel & model,
                             const float * encoder_out,
                             int T_enc, int D_enc,
                             const SortformerDiarizationOptions & opts,
                             SortformerDiarizationResult & out);

// AOSC streaming step (NeMo-faithful port of `forward_streaming_step` +
// `streaming_update` from sortformer_diar_models.py / sortformer_modules.py).
//
// Inputs:
//   - chunk_pre_encode_embs: pre-subsampled chunk audio in fc_d_model space,
//     containing [left_context | committed_chunk | right_context] enc frames.
//     Shape: (T_chunk_pre, D) where T_chunk_pre = lc + chunk_len_eff + rc.
//   - lc/rc: left/right context encoder frames in chunk_pre_encode_embs.
//     The committed chunk slice is `[lc, lc + chunk_len_eff)`.
//   - chunk_len_eff: number of encoder frames to commit (may be < cfg.chunk_len
//     at session boundaries; computed by the caller from audio availability).
//
// Side effects:
//   - cache is mutated in-place (FIFO append, optional pop+compress, sil profile).
//
// Output:
//   - out.speaker_probs is the (chunk_len_eff, num_spks) committed-chunk slab.
//   - out.segments is thresholded over the committed chunk (time origin = 0).
//   - out.n_frames = chunk_len_eff; out.frame_stride_s reflects the model.
int  sortformer_aosc_step(ParakeetCtcModel & model,
                          const float * chunk_pre_encode_embs,
                          int T_chunk_pre, int D,
                          int lc, int rc, int chunk_len_eff,
                          SortformerSpeakerCache & cache,
                          const SortformerStreamingConfig & cfg,
                          const SortformerDiarizationOptions & opts,
                          SortformerDiarizationResult & out);

}
