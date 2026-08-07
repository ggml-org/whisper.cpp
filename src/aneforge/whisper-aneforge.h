// ANEForge encoder backend for whisper.cpp: runs the audio encoder directly on the
// Apple Neural Engine via ANEForge's e5rt dispatch shim, in place of the ggml/Metal or
// CoreML encoder. Enabled at runtime by setting ANEFORGE_ENCODER to a bundle directory
// (model.mil + weights.bin + ports.txt + pos.f16), produced by export_encoder.py.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_aneforge_context;

struct whisper_aneforge_context * whisper_aneforge_init(const char * bundle_dir);

// mel: n_mel x n_len fp32 (channel-major, as whisper stores it). out: the encoder
// output, n_ctx x n_state fp32, written in the [n_state, n_ctx] ggml layout.
void whisper_aneforge_encode(struct whisper_aneforge_context * ctx,
                             int64_t n_mel, int64_t n_len, const float * mel, float * out);

void whisper_aneforge_free(struct whisper_aneforge_context * ctx);

#ifdef __cplusplus
}
#endif
