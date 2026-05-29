#ifndef WHISPER_DIARIZE_H
#define WHISPER_DIARIZE_H

#include "whisper-speaker.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mel-spectrogram computation (80-bin, n_fft=400, hop=160, fmin=0, fmax=8000)
float * whisper_compute_mel_80(const float * samples, int n_samples);

// Get number of mel frames for given sample count
int whisper_get_mel_n_frames(int n_samples);

// Free mel buffer
void whisper_mel_free(float * mel);

// Speaker encoder
struct whisper_speaker_encoder;

// Create encoder for given mel frame count
struct whisper_speaker_encoder * whisper_speaker_encoder_new(
    struct whisper_speaker_model * model,
    int n_frames,
    int device
);

// Free encoder context
void whisper_speaker_encoder_free(struct whisper_speaker_encoder * encoder);

// Run forward pass: mel [T, 80] → embedding [192]
bool whisper_speaker_encoder_compute(
    struct whisper_speaker_encoder * encoder,
    const float * mel,
    float * embedding
);

// Clustering (agglomerative hierarchical)
enum whisper_linkage_type {
    WHISPER_LINKAGE_AVERAGE = 0,   // average linkage (default)
    WHISPER_LINKAGE_COMPLETE = 1   // complete linkage (more conservative)
};

struct whisper_clustering_context {
    int num_segments;
    int embedding_dim;
    double * distance_matrix;
    int * speaker_ids;
    int num_speakers;
};

struct whisper_clustering_context * whisper_clustering_context_create(int num_segments);

void whisper_clustering_context_free(struct whisper_clustering_context * ctx);

double * compute_distance_matrix(
    const float * embeddings,
    int num_segments,
    int embedding_dim
);

// Run agglomerative clustering on embeddings
// target_speakers > 0: force exact count; == 0: auto-detect using threshold
// Returns 0 on success, -1 on error
int whisper_clustering_cluster(
    struct whisper_clustering_context * ctx,
    const float * embeddings,
    int target_speakers,
    float threshold,
    int linkage_type
);

#ifdef __cplusplus
}
#endif

#endif // WHISPER_DIARIZE_H
