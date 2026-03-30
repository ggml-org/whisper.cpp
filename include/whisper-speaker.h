#ifndef WHISPER_SPEAKER_H
#define WHISPER_SPEAKER_H

#include "ggml.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque speaker model context
struct whisper_speaker_model;

// Load speaker model from GGML binary file
struct whisper_speaker_model * whisper_speaker_load_from_file(const char * path_model);

// Free model resources
void whisper_speaker_free(struct whisper_speaker_model * model);

// Print model structure info
void whisper_speaker_validate(struct whisper_speaker_model * model);

// Get embedding dimension (192 for ECAPA-TDNN)
int whisper_speaker_get_embedding_dim(struct whisper_speaker_model * model);

// Get tensor count
int whisper_speaker_get_tensor_count(struct whisper_speaker_model * model);

// Get tensor by index
struct ggml_tensor * whisper_speaker_get_tensor(struct whisper_speaker_model * model, int idx);

// Find tensor by name (e.g. "mods.embedding_model.blocks.0.conv.conv.weight")
struct ggml_tensor * whisper_speaker_find_tensor(struct whisper_speaker_model * model, const char * name);

#ifdef __cplusplus
}
#endif

#endif // WHISPER_SPEAKER_H
