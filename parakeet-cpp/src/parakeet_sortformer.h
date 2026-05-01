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

int  sortformer_diarize_ggml(const ParakeetCtcModel & model,
                             const float * encoder_out,
                             int T_enc, int D_enc,
                             ggml_backend_t backend,
                             const SortformerDiarizationOptions & opts,
                             SortformerDiarizationResult & out);

}
