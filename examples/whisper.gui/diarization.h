// Lightweight, dependency-free speaker diarization for the whisper.cpp GUI.
//
// The pipeline is: for each transcript segment, turn its audio span into a
// fixed-length "speaker embedding", then cluster the embeddings so that
// segments spoken by the same voice get the same speaker id.
//
// The embedding is currently computed from classic MFCC statistics (see
// compute_embedding). This is intentionally the one plug-in point: a neural
// speaker-embedding model (ECAPA-TDNN, etc.) can later implement the same
// compute_embedding() contract for much better accuracy, with no change to the
// clustering or the GUI.

#pragma once

#include <vector>

namespace diarize {

// Turn a span of mono PCM audio into a fixed-length speaker embedding.
// Returns an L2-normalized feature vector (empty if there is no usable audio).
//
// PLUG-IN POINT: replace the body of this function (in diarization.cpp) with a
// neural speaker-embedding model to improve accuracy. Everything else is agnostic
// to how the embedding is produced.
std::vector<float> compute_embedding(const float * samples, int n_samples, int sample_rate);

// Assign a speaker id (0..K-1) to each embedding by agglomerative clustering
// with average linkage over cosine distance.
//
//   num_speakers > 0 : merge until exactly that many clusters remain (use this
//                      when the speaker count is known - most reliable).
//   num_speakers <= 0: determine the count automatically, stopping when the
//                      closest two clusters are farther apart than `threshold`.
//
// Returns one id per input embedding (same order). Empty embeddings get id 0.
std::vector<int> cluster(const std::vector<std::vector<float>> & embeddings,
                         int num_speakers, float threshold);

} // namespace diarize
