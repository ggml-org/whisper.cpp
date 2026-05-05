// Sortformer ggml graph build, speaker probabilities, and thresholded segments.

#include "parakeet_sortformer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

}  // namespace

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

}
