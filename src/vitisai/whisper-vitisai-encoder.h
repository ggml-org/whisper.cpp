// Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdbool>
#include <cstdint>

#if __cplusplus
extern "C" {
#endif

struct whisper_vitisai_context;

struct whisper_vitisai_context * whisper_vitisai_init(const char * path_model);
void whisper_vitisai_free(struct whisper_vitisai_context * ctx);

// Function to mmap rai file for Linux and MapViewOfFile for Windows
bool map_rai_file(const char * path, uint8_t ** buffer, size_t * size);
// Function to unmap rai file for Linux and UnmapViewOfFile for Windows
void unmap_rai_file(uint8_t * buffer, size_t size);

struct ggml_tensor;

int whisper_vitisai_encode(
    struct whisper_vitisai_context * ctx,
    struct ggml_tensor * mel,
    struct ggml_tensor * out);

#if __cplusplus
}
#endif
