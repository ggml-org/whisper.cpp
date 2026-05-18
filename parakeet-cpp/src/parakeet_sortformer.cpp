// Sortformer ggml graph build, speaker probabilities, and thresholded segments.
//
// Streaming AOSC (Audio-Online Speaker Cache, v2.1) is a faithful port of
// NeMo's `sortformer_modules.py` (`_compress_spkcache`, `_get_silence_profile`,
// `streaming_update`) and `sortformer_diar_models.py::forward_streaming_step`.
// All AOSC state lives in the post-subsampling, pre-conformer-layers embedding
// space (`fc_d_model`, 512 in v2.1). The streaming forward concatenates
// `[spkcache | fifo | chunk_pre_encode]` and runs the conformer encoder via
// `run_encoder_bypass_pre_encode` so the diariser sees contextualised rows
// rather than chunk-only post-encoder rows.

#include "parakeet_sortformer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace parakeet {

namespace {

// Threshold speaker probabilities into time-sorted segments.
void sf_threshold_segments(const std::vector<float> & speaker_probs,
                           int T_enc, int num_spks,
                           double frame_stride_s, float threshold,
                           std::vector<SortformerSegment> & segments) {
    segments.clear();
    for (int s = 0; s < num_spks; ++s) {
        bool active = false;
        int  start_frame = 0;
        for (int t = 0; t < T_enc; ++t) {
            const bool a = speaker_probs[(size_t)t * num_spks + s] > threshold;
            if (a && !active)  { start_frame = t; active = true; }
            if (!a && active) {
                SortformerSegment seg;
                seg.speaker_id = s;
                seg.start_s = start_frame * frame_stride_s;
                seg.end_s   = t           * frame_stride_s;
                segments.push_back(seg);
                active = false;
            }
        }
        if (active) {
            SortformerSegment seg;
            seg.speaker_id = s;
            seg.start_s = start_frame * frame_stride_s;
            seg.end_s   = T_enc       * frame_stride_s;
            segments.push_back(seg);
        }
    }
    std::sort(segments.begin(), segments.end(),
              [](const SortformerSegment & a, const SortformerSegment & b) {
                  if (a.start_s != b.start_s) return a.start_s < b.start_s;
                  return a.speaker_id < b.speaker_id;
              });
}

ggml_tensor * sf_layer_norm(ggml_context * ctx, ggml_tensor * x,
                            ggml_tensor * gamma, ggml_tensor * beta, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, gamma);
    x = ggml_add(ctx, x, beta);
    return x;
}

ggml_tensor * sf_transformer_block(ggml_context * ctx, ggml_tensor * x,
                                   const SortformerTransformerBlock & W,
                                   int n_heads, int head_dim, int d_model, int T) {
    // --- multi-head self-attention ---
    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, W.attn_q_w, x), W.attn_q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, W.attn_k_w, x), W.attn_k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, W.attn_v_w, x), W.attn_v_b);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, T);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, T);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, T);

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // (HD, T, H)
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    const float scale = 1.0f / std::sqrt((float) head_dim);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // (T, T, H)
    scores = ggml_scale(ctx, scores, scale);
    ggml_tensor * attn = ggml_soft_max(ctx, scores);

    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor * attn_v = ggml_mul_mat(ctx, v_t, attn);  // (HD, T, H)
    ggml_tensor * merged = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));  // (HD, H, T)
    merged = ggml_reshape_2d(ctx, merged, d_model, T);

    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, W.attn_o_w, merged), W.attn_o_b);

    // residual + LN1 (post-LN)
    ggml_tensor * r1 = ggml_add(ctx, x, attn_out);
    r1 = sf_layer_norm(ctx, r1, W.ln1_w, W.ln1_b, 1e-5f);

    // --- FFN ---
    ggml_tensor * ffn = ggml_add(ctx, ggml_mul_mat(ctx, W.ffn_in_w, r1), W.ffn_in_b);
    ffn = ggml_relu(ctx, ffn);
    ffn = ggml_add(ctx, ggml_mul_mat(ctx, W.ffn_out_w, ffn), W.ffn_out_b);

    // residual + LN2
    ggml_tensor * r2 = ggml_add(ctx, r1, ffn);
    r2 = sf_layer_norm(ctx, r2, W.ln2_w, W.ln2_b, 1e-5f);

    return r2;
}

// Build the full Sortformer ggml graph: encoder_proj -> N transformer blocks
// -> ReLU -> h2h -> ReLU -> h2s -> sigmoid.  Returns the output tensor and
// writes the input placeholder into *inp.
ggml_tensor * sf_build_graph(ggml_context * ctx,
                             const SortformerWeights & sw,
                             int n_layers, int n_heads, int head_dim,
                             int tf_d, int D_in, int T_enc,
                             ggml_tensor ** inp) {
    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_in, T_enc);
    ggml_set_name(x_in, "enc_in");
    ggml_set_input(x_in);
    *inp = x_in;

    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, sw.encoder_proj_w, x_in),
                                sw.encoder_proj_b);

    for (int l = 0; l < n_layers; ++l)
        x = sf_transformer_block(ctx, x, sw.transformer[l],
                                 n_heads, head_dim, tf_d, T_enc);

    x = ggml_relu(ctx, x);
    x = ggml_add(ctx, ggml_mul_mat(ctx, sw.head_h2h_w, x), sw.head_h2h_b);
    x = ggml_relu(ctx, x);
    x = ggml_add(ctx, ggml_mul_mat(ctx, sw.head_h2s_w, x), sw.head_h2s_b);
    x = ggml_sigmoid(ctx, x);

    ggml_set_name(x, "speaker_probs");
    ggml_set_output(x);
    return x;
}

// Allocate, upload input, compute, and download output for a Sortformer graph.
// Returns 0 on success, negative on failure.  Caller must free ctx afterwards.
int sf_exec_graph(ggml_context * ctx, ggml_backend_t backend,
                  ggml_tensor * x_in, ggml_tensor * x_out,
                  const float * encoder_out,
                  int D_in, int T_enc, int num_spks,
                  std::vector<float> & speaker_probs) {
    const size_t graph_slots = 4096;
    ggml_cgraph * cg = ggml_new_graph_custom(ctx, graph_slots, false);
    ggml_build_forward_expand(cg, x_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, cg))  { ggml_gallocr_free(alloc); return -2; }
    if (!ggml_gallocr_alloc_graph(alloc, cg)) { ggml_gallocr_free(alloc); return -3; }

    ggml_backend_tensor_set(x_in, encoder_out, 0,
                            (size_t)D_in * T_enc * sizeof(float));

    if (ggml_backend_graph_compute(backend, cg) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        return -4;
    }

    speaker_probs.resize((size_t)T_enc * num_spks);
    ggml_backend_tensor_get(x_out, speaker_probs.data(), 0,
                            speaker_probs.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    return 0;
}

// =============================================================================
// AOSC helpers: NeMo-faithful ports of sortformer_modules.py utilities.
// =============================================================================

// Running-mean silence embedding update. Mirrors NeMo's _get_silence_profile
// (sortformer_modules.py:636-667). A frame is "silence" iff its sum-of-speaker-
// probabilities is below sil_threshold. The cache's mean_sil_emb is updated as
// the cumulative mean over all silence frames seen so far across all pop-outs.
void update_silence_profile(SortformerSpeakerCache & cache,
                            const float * emb_pop, const float * preds_pop,
                            int n_pop, int num_spks, int D,
                            float sil_threshold) {
    if (n_pop <= 0 || !emb_pop || !preds_pop) return;

    int sil_count = 0;
    std::vector<double> sil_sum((size_t) D, 0.0);
    for (int t = 0; t < n_pop; ++t) {
        float ssum = 0.0f;
        const float * p_row = preds_pop + (size_t) t * num_spks;
        for (int j = 0; j < num_spks; ++j) ssum += p_row[j];
        if (ssum < sil_threshold) {
            ++sil_count;
            const float * e_row = emb_pop + (size_t) t * D;
            for (int d = 0; d < D; ++d) sil_sum[d] += (double) e_row[d];
        }
    }
    if (sil_count == 0) return;

    const double new_n = (double) (cache.n_sil_frames + sil_count);
    if (cache.mean_sil_emb.size() != (size_t) D) {
        cache.mean_sil_emb.assign((size_t) D, 0.0f);
    }
    for (int d = 0; d < D; ++d) {
        const double old_sum = (double) cache.mean_sil_emb[d] * (double) cache.n_sil_frames;
        cache.mean_sil_emb[d] = (float) ((old_sum + sil_sum[d]) / new_n);
    }
    cache.n_sil_frames += (int64_t) sil_count;
}

// Composite per-(t, spk) score, mirroring NeMo's _get_log_pred_scores
// (sortformer_modules.py:669-686):
//   log_probs[t][i]      = log(max(preds[t][i],   pred_score_threshold))
//   log_1_probs[t][i]    = log(max(1-preds[t][i], pred_score_threshold))
//   log_1_probs_sum[t]   = sum_j log_1_probs[t][j]
//   scores[t][i]         = log_probs[t][i] - log_1_probs[t][i]
//                          + log_1_probs_sum[t] - log(0.5)
static void compute_log_pred_scores(const float * preds, int n_frames, int num_spks,
                                    float clamp_min,
                                    std::vector<float> & scores) {
    scores.assign((size_t) n_frames * num_spks, 0.0f);
    const float log_half  = std::log(0.5f);

    std::vector<float> log1ps((size_t) num_spks);
    for (int t = 0; t < n_frames; ++t) {
        const float * p = preds + (size_t) t * num_spks;
        float log1_sum = 0.0f;
        for (int j = 0; j < num_spks; ++j) {
            const float onmp  = std::max(1.0f - p[j],  clamp_min);
            log1ps[j] = std::log(onmp);
            log1_sum += log1ps[j];
        }
        float * s = scores.data() + (size_t) t * num_spks;
        for (int i = 0; i < num_spks; ++i) {
            const float p_i  = std::max(p[i],        clamp_min);
            const float lp   = std::log(p_i);
            const float l1p  = log1ps[i];
            s[i] = lp - l1p + log1_sum - log_half;
        }
    }
}

// _disable_low_scores: non-speech -> -inf; non-positive scores -> -inf when
// the speaker has at least `min_pos_scores_per_spk` positive frames.
// (sortformer_modules.py:782-808)
static void disable_low_scores(std::vector<float> & scores,
                               const float * preds, int n_frames, int num_spks,
                               int min_pos_scores_per_spk) {
    const float neg_inf = -1.0e30f /* very-negative sentinel; -inf is UB with current FP flags */;

    // First pass: non-speech -> -inf.
    for (int t = 0; t < n_frames; ++t) {
        const float * p = preds + (size_t) t * num_spks;
        float * s = scores.data() + (size_t) t * num_spks;
        for (int i = 0; i < num_spks; ++i) {
            if (!(p[i] > 0.5f)) s[i] = neg_inf;
        }
    }

    // Count positive scores per speaker.
    std::vector<int> pos_count((size_t) num_spks, 0);
    for (int t = 0; t < n_frames; ++t) {
        const float * s = scores.data() + (size_t) t * num_spks;
        for (int i = 0; i < num_spks; ++i) {
            if (s[i] > 0.0f) ++pos_count[i];
        }
    }

    // Second pass: if speaker i has enough positive frames, kill its
    // non-positive but still-speech entries.
    for (int t = 0; t < n_frames; ++t) {
        const float * p = preds + (size_t) t * num_spks;
        float * s = scores.data() + (size_t) t * num_spks;
        for (int i = 0; i < num_spks; ++i) {
            const bool is_speech = p[i] > 0.5f;
            const bool is_nonpos = !(s[i] > 0.0f) && (s[i] != neg_inf);
            if (is_speech && is_nonpos && pos_count[i] >= min_pos_scores_per_spk) {
                s[i] = neg_inf;
            }
        }
    }
}

// _boost_topk_scores: pick top-k frames per speaker, add boost to those scores.
// (sortformer_modules.py:611-634). offset = 0.5; boost = -scale * log(0.5).
static void boost_topk_scores(std::vector<float> & scores,
                              int n_frames, int num_spks,
                              int n_boost_per_spk, float scale_factor) {
    if (n_boost_per_spk <= 0 || n_frames <= 0) return;

    const float boost = -scale_factor * std::log(0.5f);

    std::vector<int> idx_buf((size_t) n_frames);
    for (int spk = 0; spk < num_spks; ++spk) {
        std::iota(idx_buf.begin(), idx_buf.end(), 0);
        const int k = std::min(n_boost_per_spk, n_frames);
        std::nth_element(idx_buf.begin(), idx_buf.begin() + k, idx_buf.end(),
                         [&](int a, int b) {
                             const float sa = scores[(size_t) a * num_spks + spk];
                             const float sb = scores[(size_t) b * num_spks + spk];
                             return sa > sb;
                         });
        for (int i = 0; i < k; ++i) {
            const int t = idx_buf[i];
            float & s = scores[(size_t) t * num_spks + spk];
            if (s != -1.0e30f /* very-negative sentinel; -inf is UB with current FP flags */) {
                s += boost;
            }
        }
    }
}

// NeMo's _compress_spkcache (sortformer_modules.py:838-896).
// Compresses (n_frames, D) embedding rows + (n_frames, num_spks) preds into
// (spkcache_len, D) + (spkcache_len, num_spks), retaining the most informative
// rows per speaker plus an A_silence "anchor" budget per speaker filled from
// mean_sil_emb. Output rows are sorted by absolute frame index (Sort Loss
// anchors speaker arrival order).
static void compress_speaker_cache(
    SortformerSpeakerCache & cache,
    const float * emb_in, const float * preds_in,
    int n_frames, int num_spks, int D,
    const SortformerStreamingConfig & cfg) {

    const int spkcache_len = cfg.spkcache_len;
    if (n_frames <= 0 || num_spks <= 0 || D <= 0 || spkcache_len <= 0) {
        cache.spkcache.assign((size_t) spkcache_len * D, 0.0f);
        cache.spkcache_preds.assign((size_t) spkcache_len * num_spks, 0.0f);
        cache.n_rows = spkcache_len;
        cache.spkcache_preds_valid = true;
        return;
    }

    const int A_sil = cfg.spkcache_sil_frames_per_spk;
    const int spkcache_len_per_spk = spkcache_len / num_spks - A_sil;
    const int strong_boost = (int) std::floor((float) spkcache_len_per_spk * cfg.strong_boost_rate);
    const int weak_boost   = (int) std::floor((float) spkcache_len_per_spk * cfg.weak_boost_rate);
    const int min_pos_per  = (int) std::floor((float) spkcache_len_per_spk * cfg.min_pos_scores_rate);

    // 1. Compute composite log scores: (n_frames, num_spks).
    std::vector<float> scores;
    compute_log_pred_scores(preds_in, n_frames, num_spks, cfg.pred_score_threshold, scores);

    // 2. Disable low/non-positive scores.
    disable_low_scores(scores, preds_in, n_frames, num_spks, min_pos_per);

    // 3. Newest-frame boost: rows beyond the first spkcache_len get a small
    //    additive bonus, biasing retention toward recency. (NeMo line 876-877)
    if (cfg.scores_boost_latest > 0.0f && n_frames > spkcache_len) {
        for (int t = spkcache_len; t < n_frames; ++t) {
            float * s = scores.data() + (size_t) t * num_spks;
            for (int i = 0; i < num_spks; ++i) {
                if (s[i] != -1.0e30f /* very-negative sentinel; -inf is UB with current FP flags */) {
                    s[i] += cfg.scores_boost_latest;
                }
            }
        }
    }

    // 4. Strong boost (scale=2, ensures each speaker keeps K_strong rows).
    boost_topk_scores(scores, n_frames, num_spks, strong_boost, 2.0f);
    // 5. Weak boost  (scale=1, mitigates single-speaker dominance).
    boost_topk_scores(scores, n_frames, num_spks, weak_boost,   1.0f);

    // 6. Append A_sil silence-pad rows with score +inf per speaker. These are
    //    virtual frames that always survive top-K and get filled from
    //    mean_sil_emb in step 8.
    const int n_total = n_frames + A_sil;
    if (A_sil > 0) {
        scores.resize((size_t) n_total * num_spks);
        const float pos_inf = 1.0e30f /* very-positive sentinel; +inf is UB with current FP flags */;
        for (int t = n_frames; t < n_total; ++t) {
            float * s = scores.data() + (size_t) t * num_spks;
            for (int i = 0; i < num_spks; ++i) s[i] = pos_inf;
        }
    }

    // 7. Top-K selection: flatten over (speaker, frame), pick top spkcache_len.
    //    NeMo's _get_topk_indices (line 688-719). Indices are (spk * n_total + t).
    //    Scores at -inf are dropped (placeholder index = MAX_INDEX).
    constexpr int MAX_INDEX = std::numeric_limits<int>::max();
    const size_t flat_n = (size_t) n_total * num_spks;
    std::vector<int> flat_idx(flat_n);
    std::iota(flat_idx.begin(), flat_idx.end(), 0);

    auto flat_score = [&](int idx) {
        const int spk = idx / n_total;
        const int t   = idx % n_total;
        return scores[(size_t) t * num_spks + spk];
    };

    const int k = std::min(spkcache_len, (int) flat_n);
    std::nth_element(flat_idx.begin(), flat_idx.begin() + k, flat_idx.end(),
                     [&](int a, int b) { return flat_score(a) > flat_score(b); });
    std::vector<int> topk(flat_idx.begin(), flat_idx.begin() + k);

    // Replace -inf-score picks with the placeholder. Sort to preserve frame
    // order after modulo. (NeMo flattens via `permute(0,2,1).reshape`, putting
    // speaker blocks contiguous; `torch.remainder(idx, n_frames)` returns the
    // frame index; our `idx % n_total` does the same.)
    for (int & idx : topk) {
        if (flat_score(idx) == -1.0e30f /* very-negative sentinel; -inf is UB with current FP flags */) {
            idx = MAX_INDEX;
        }
    }
    std::sort(topk.begin(), topk.end());

    cache.spkcache.assign((size_t) spkcache_len * D, 0.0f);
    cache.spkcache_preds.assign((size_t) spkcache_len * num_spks, 0.0f);

    const int n_frames_no_sil = n_frames;  // frames with index >= n_frames_no_sil are silence-pad
    if (cache.mean_sil_emb.size() != (size_t) D) {
        cache.mean_sil_emb.assign((size_t) D, 0.0f);
    }

    // 8. Gather rows. Disabled (placeholder or silence-pad) -> mean_sil_emb + zero preds.
    for (int r = 0; r < spkcache_len; ++r) {
        if (r >= k) {
            std::memcpy(cache.spkcache.data() + (size_t) r * D,
                        cache.mean_sil_emb.data(),
                        (size_t) D * sizeof(float));
            continue;
        }
        const int idx = topk[r];
        if (idx == MAX_INDEX) {
            std::memcpy(cache.spkcache.data() + (size_t) r * D,
                        cache.mean_sil_emb.data(),
                        (size_t) D * sizeof(float));
            continue;
        }
        const int frame_idx = idx % n_total;
        if (frame_idx >= n_frames_no_sil) {
            std::memcpy(cache.spkcache.data() + (size_t) r * D,
                        cache.mean_sil_emb.data(),
                        (size_t) D * sizeof(float));
            continue;
        }
        std::memcpy(cache.spkcache.data() + (size_t) r * D,
                    emb_in + (size_t) frame_idx * D,
                    (size_t) D * sizeof(float));
        std::memcpy(cache.spkcache_preds.data() + (size_t) r * num_spks,
                    preds_in + (size_t) frame_idx * num_spks,
                    (size_t) num_spks * sizeof(float));
    }

    cache.n_rows = spkcache_len;
    cache.spkcache_preds_valid = true;
}

// streaming_update (sync mode), NeMo sortformer_modules.py:526-609.
// Updates FIFO with the committed-chunk slice, optionally pops `pop_out_len`
// frames into spkcache, optionally compresses spkcache on overflow. Also
// updates the silence profile from popped frames.
//
// preds_full layout: [spkcache_preds (prev_spkcache_n) | fifo_preds (prev_fifo_n)
//                   | chunk_preds (lc + chunk_committed + rc)]
// `lc` is the left-context offset within the chunk region; the committed-chunk
// preds start at index `prev_spkcache_n + prev_fifo_n + lc` and span `chunk_committed`.
static void streaming_update(SortformerSpeakerCache & cache,
                             const float * chunk_pre_encode_lc, int chunk_committed,
                             const float * preds_full,
                             int prev_spkcache_len_at_call, int prev_fifo_len_at_call,
                             int lc,
                             int num_spks, int D,
                             const SortformerStreamingConfig & cfg) {

    const int fifo_off  = prev_spkcache_len_at_call;
    const int chunk_off = prev_spkcache_len_at_call + prev_fifo_len_at_call;

    // Refresh fifo_preds with current model output over the FIFO region
    // (NeMo sortformer_modules.py:562).
    if (prev_fifo_len_at_call > 0) {
        cache.fifo_preds.assign((size_t) prev_fifo_len_at_call * num_spks, 0.0f);
        std::memcpy(cache.fifo_preds.data(),
                    preds_full + (size_t) fifo_off * num_spks,
                    (size_t) prev_fifo_len_at_call * num_spks * sizeof(float));
    } else {
        cache.fifo_preds.clear();
    }

    // Append committed chunk + its preds to FIFO.
    const int new_fifo_after_append = cache.n_fifo + chunk_committed;
    cache.fifo.resize((size_t) new_fifo_after_append * D);
    std::memcpy(cache.fifo.data() + (size_t) cache.n_fifo * D,
                chunk_pre_encode_lc,
                (size_t) chunk_committed * D * sizeof(float));
    cache.fifo_preds.resize((size_t) new_fifo_after_append * num_spks);
    std::memcpy(cache.fifo_preds.data() + (size_t) cache.n_fifo * num_spks,
                preds_full + (size_t) (chunk_off + lc) * num_spks,
                (size_t) chunk_committed * num_spks * sizeof(float));
    cache.n_fifo = new_fifo_after_append;

    // Maybe pop out: NeMo sortformer_modules.py:570-601.
    if (cache.n_fifo > cfg.fifo_len) {
        int pop_out = cfg.spkcache_update_period;
        pop_out = std::max(pop_out, chunk_committed - cfg.fifo_len + prev_fifo_len_at_call);
        pop_out = std::min(pop_out, cache.n_fifo);
        if (pop_out < 1) pop_out = 1;

        // Update mean_sil_emb from popped frames.
        update_silence_profile(cache,
                               cache.fifo.data(),
                               cache.fifo_preds.data(),
                               pop_out, num_spks, D, cfg.sil_threshold);

        // Append popped frames to spkcache.
        const int new_spkcache_n = cache.n_rows + pop_out;
        cache.spkcache.resize((size_t) new_spkcache_n * D);
        std::memcpy(cache.spkcache.data() + (size_t) cache.n_rows * D,
                    cache.fifo.data(),
                    (size_t) pop_out * D * sizeof(float));

        // spkcache_preds: lazy init on first overflow (NeMo lines 589-593).
        if (cache.spkcache_preds_valid) {
            cache.spkcache_preds.resize((size_t) new_spkcache_n * num_spks);
            std::memcpy(cache.spkcache_preds.data() + (size_t) cache.n_rows * num_spks,
                        cache.fifo_preds.data(),
                        (size_t) pop_out * num_spks * sizeof(float));
        } else if (new_spkcache_n > cfg.spkcache_len) {
            // Will compress for the first time -- seed spkcache_preds with the
            // model's predictions for the current spkcache rows plus the popped
            // rows (NeMo line 593).
            cache.spkcache_preds.assign((size_t) new_spkcache_n * num_spks, 0.0f);
            if (cache.n_rows > 0) {
                std::memcpy(cache.spkcache_preds.data(),
                            preds_full,  // first cache.n_rows rows
                            (size_t) cache.n_rows * num_spks * sizeof(float));
            }
            std::memcpy(cache.spkcache_preds.data() + (size_t) cache.n_rows * num_spks,
                        cache.fifo_preds.data(),
                        (size_t) pop_out * num_spks * sizeof(float));
        }

        cache.n_rows = new_spkcache_n;

        // Drop popped frames from FIFO.
        const int remaining = cache.n_fifo - pop_out;
        if (remaining > 0) {
            std::memmove(cache.fifo.data(),
                         cache.fifo.data() + (size_t) pop_out * D,
                         (size_t) remaining * D * sizeof(float));
            std::memmove(cache.fifo_preds.data(),
                         cache.fifo_preds.data() + (size_t) pop_out * num_spks,
                         (size_t) remaining * num_spks * sizeof(float));
        }
        cache.fifo.resize((size_t) remaining * D);
        cache.fifo_preds.resize((size_t) remaining * num_spks);
        cache.n_fifo = remaining;

        // Compress on overflow.
        if (cache.n_rows > cfg.spkcache_len) {
            std::vector<float> emb_in   = std::move(cache.spkcache);
            std::vector<float> preds_in = std::move(cache.spkcache_preds);
            const int n_in = cache.n_rows;
            cache.spkcache.clear();
            cache.spkcache_preds.clear();
            cache.n_rows = 0;
            compress_speaker_cache(cache,
                                   emb_in.data(), preds_in.data(),
                                   n_in, num_spks, D, cfg);
        }
    }
}

}  // namespace

void sortformer_cache_reset(SortformerSpeakerCache & cache, int D) {
    cache = SortformerSpeakerCache{};
    if (D > 0) {
        cache.mean_sil_emb.assign((size_t) D, 0.0f);
    }
}

int sortformer_diarize_ggml(const ParakeetCtcModel & model,
                            const float * encoder_out,
                            int T_enc, int D_enc,
                            ggml_backend_t backend,
                            const SortformerDiarizationOptions & opts,
                            SortformerDiarizationResult & out) {
    const auto & enc   = model.encoder_cfg;
    const int D_in     = enc.sortformer_fc_d_model;
    const int tf_d     = enc.sortformer_tf_d_model;
    const int n_heads  = enc.sortformer_tf_n_heads;
    const int n_layers = enc.sortformer_tf_n_layers;
    const int num_spks = enc.sortformer_num_spks;

    if (D_enc != D_in) {
        std::fprintf(stderr, "sortformer_diarize_ggml: encoder D mismatch %d vs %d\n", D_enc, D_in);
        return 1;
    }
    if (n_heads <= 0 || tf_d % n_heads != 0) {
        std::fprintf(stderr, "sortformer_diarize_ggml: tf_d %d not divisible by n_heads %d\n", tf_d, n_heads);
        return 1;
    }
    if (T_enc <= 0) {
        out.n_frames = 0;  out.num_spks = num_spks;
        out.speaker_probs.clear();  out.segments.clear();
        return 0;
    }

    const int head_dim = tf_d / n_heads;
    const auto t0 = std::chrono::steady_clock::now();

    // 1. Context for graph construction (no-alloc)
    const size_t graph_slots = 4096;
    const size_t overhead = ggml_tensor_overhead() * graph_slots
                          + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params gp = { overhead, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) return -1;

    // 2. Build graph
    ggml_tensor * x_in  = nullptr;
    ggml_tensor * x_out = sf_build_graph(ctx, model.sortformer,
                                         n_layers, n_heads, head_dim,
                                         tf_d, D_in, T_enc, &x_in);

    // 3. Execute on backend
    int rc = sf_exec_graph(ctx, backend, x_in, x_out,
                           encoder_out, D_in, T_enc, num_spks,
                           out.speaker_probs);
    ggml_free(ctx);
    if (rc != 0) return rc;

    // 4. Fill result metadata + threshold segmentation
    out.n_frames = T_enc;
    out.num_spks = num_spks;
    out.frame_stride_s = (double)(model.mel_cfg.hop_length *
                                  model.encoder_cfg.subsampling_factor) /
                         (double)model.mel_cfg.sample_rate;

    sf_threshold_segments(out.speaker_probs, T_enc, num_spks,
                          out.frame_stride_s, opts.threshold, out.segments);

    out.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count() / 1000.0;
    return 0;
}

int sortformer_aosc_step(ParakeetCtcModel & model,
                         const float * chunk_pre_encode_embs,
                         int T_chunk_pre, int D,
                         int lc, int rc, int chunk_len_eff,
                         SortformerSpeakerCache & cache,
                         const SortformerStreamingConfig & cfg,
                         ggml_backend_t backend,
                         const SortformerDiarizationOptions & opts,
                         SortformerDiarizationResult & out) {
    const auto & enc   = model.encoder_cfg;
    const int D_enc    = enc.sortformer_fc_d_model;
    const int num_spks = enc.sortformer_num_spks;

    if (D != D_enc) {
        std::fprintf(stderr,
            "sortformer_aosc_step: D mismatch (cache=%d, fc_d_model=%d)\n",
            D, D_enc);
        return 1;
    }
    if (T_chunk_pre <= 0 || chunk_len_eff <= 0) {
        out.n_frames = 0;
        out.num_spks = num_spks;
        out.speaker_probs.clear();
        out.segments.clear();
        return 0;
    }
    if (lc + chunk_len_eff + rc > T_chunk_pre) {
        std::fprintf(stderr,
            "sortformer_aosc_step: bad slice lc=%d chunk=%d rc=%d > T_chunk_pre=%d\n",
            lc, chunk_len_eff, rc, T_chunk_pre);
        return 1;
    }

    if (cache.mean_sil_emb.size() != (size_t) D) {
        cache.mean_sil_emb.assign((size_t) D, 0.0f);
    }

    const auto t0 = std::chrono::steady_clock::now();

    // 1. Assemble [spkcache | fifo | chunk_pre_encode] in pre-encode space.
    const int prev_spkcache_n = cache.n_rows;
    const int prev_fifo_n     = cache.n_fifo;
    const int T_cat = prev_spkcache_n + prev_fifo_n + T_chunk_pre;

    std::vector<float> cat_pre((size_t) T_cat * D);
    size_t off = 0;
    if (prev_spkcache_n > 0) {
        std::memcpy(cat_pre.data() + off,
                    cache.spkcache.data(),
                    (size_t) prev_spkcache_n * D * sizeof(float));
        off += (size_t) prev_spkcache_n * D;
    }
    if (prev_fifo_n > 0) {
        std::memcpy(cat_pre.data() + off,
                    cache.fifo.data(),
                    (size_t) prev_fifo_n * D * sizeof(float));
        off += (size_t) prev_fifo_n * D;
    }
    std::memcpy(cat_pre.data() + off,
                chunk_pre_encode_embs,
                (size_t) T_chunk_pre * D * sizeof(float));

    // 2. Run the FastConformer encoder layers on the cat (bypass pre_encode).
    EncoderOutputs enc_cat;
    if (int rc_ = run_encoder_bypass_pre_encode(model, cat_pre.data(),
                                                T_cat, D, enc_cat); rc_ != 0) {
        std::fprintf(stderr,
            "sortformer_aosc_step: run_encoder_bypass_pre_encode rc=%d\n", rc_);
        return rc_;
    }
    if (enc_cat.n_enc_frames != T_cat || enc_cat.d_model != D) {
        std::fprintf(stderr,
            "sortformer_aosc_step: unexpected encoder output shape (%d,%d) vs (%d,%d)\n",
            enc_cat.n_enc_frames, enc_cat.d_model, T_cat, D);
        return -2;
    }

    // 3. Run the diariser over the full cat.
    SortformerDiarizationResult diar_cat;
    if (int rc_ = sortformer_diarize_ggml(model, enc_cat.encoder_out.data(),
                                          T_cat, D, backend, opts, diar_cat); rc_ != 0) {
        return rc_;
    }
    if (diar_cat.num_spks != num_spks) {
        std::fprintf(stderr,
            "sortformer_aosc_step: num_spks mismatch (%d vs %d)\n",
            diar_cat.num_spks, num_spks);
        return -3;
    }

    // 4. Slice the committed chunk preds (drop lc context, take chunk_len_eff rows).
    const int chunk_off    = prev_spkcache_n + prev_fifo_n;
    const int committed_at = chunk_off + lc;

    std::vector<float> chunk_probs((size_t) chunk_len_eff * num_spks);
    std::memcpy(chunk_probs.data(),
                diar_cat.speaker_probs.data() + (size_t) committed_at * num_spks,
                (size_t) chunk_len_eff * num_spks * sizeof(float));

    out.n_frames       = chunk_len_eff;
    out.num_spks       = num_spks;
    out.frame_stride_s = diar_cat.frame_stride_s;
    out.segments.clear();
    sf_threshold_segments(chunk_probs, chunk_len_eff, num_spks,
                          out.frame_stride_s, opts.threshold, out.segments);
    out.speaker_probs = std::move(chunk_probs);

    // 5. streaming_update: append committed chunk to FIFO, maybe pop, maybe compress.
    const float * chunk_pre_committed = chunk_pre_encode_embs + (size_t) lc * D;
    streaming_update(cache,
                     chunk_pre_committed, chunk_len_eff,
                     diar_cat.speaker_probs.data(),
                     prev_spkcache_n, prev_fifo_n,
                     lc,
                     num_spks, D, cfg);

    ++cache.chunk_index;

    out.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count() / 1000.0;
    return 0;
}

}
