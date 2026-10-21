#pragma once

// Test-only C ABI, version 2. One handle is serialized by its caller. Every
// array is copied during the call; the bridge never retains caller pointers.
// No tokenization, sampling, waveform decoding, or implicit fallback is used.
// Build as a CMake MODULE linked PRIVATE to tts-cpp, with src in include dirs.
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define PARLER_VALIDATION_API __declspec(dllexport)
#else
#define PARLER_VALIDATION_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

PARLER_VALIDATION_API int parler_validation_abi_version(void);
PARLER_VALIDATION_API void * parler_validation_create(
    const char * gguf, int n_gpu_layers, int use_coreml, int threads,
    const char * backend, char * error, size_t error_size);
PARLER_VALIDATION_API void parler_validation_destroy(void * handle);
PARLER_VALIDATION_API int parler_validation_dimensions(
    void * handle, int * codebooks, int * vocab, int * on_gpu,
    char * error, size_t error_size);

// Description encoding AND native cross-K/V projection run here. Call outside
// decoder wall time. No previous self-KV is consumed; prefill below resets it.
// max_frames is the exact delayed trace length INCLUDING the BOS input frame.
PARLER_VALIDATION_API int parler_validation_prepare(
    void * handle, const int32_t * description, size_t description_count,
    const int32_t * prompt, size_t prompt_count, int max_frames,
    char * error, size_t error_size);
PARLER_VALIDATION_API int parler_validation_prefill(
    void * handle, const int32_t * start, size_t start_count,
    float * logits, size_t logits_count, char * error, size_t error_size);
PARLER_VALIDATION_API int parler_validation_step(
    void * handle, const int32_t * frame, size_t frame_count,
    float * logits, size_t logits_count, char * error, size_t error_size);
PARLER_VALIDATION_API int parler_validation_synchronize(
    void * handle, char * error, size_t error_size);
PARLER_VALIDATION_API int parler_validation_backend(
    void * handle, char * output, size_t output_size, char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
