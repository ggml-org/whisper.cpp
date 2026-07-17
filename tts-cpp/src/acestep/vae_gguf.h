#pragma once

// ACE-Step Oobleck VAE — GGUF weight IO (QVAC-21921).
//
// The only ACE-Step VAE translation unit that touches gguf directly: mmaps the
// GGUF produced by acestep.cpp's convert.py and fuses the weight_norm-parameterised
// conv weights (w = g * v / ||v||) into pre-allocated backend tensors at load time.
// Weights are bf16 in the file; conv weights are uploaded as F16, snake/bias as F32.

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>

namespace tts_cpp::acestep {

// Parsed GGUF: mmapped file + ggml metadata context (tensor descriptors, no data).
struct VaeGGUF {
    gguf_context * ctx      = nullptr;
    ggml_context * meta     = nullptr;
    uint8_t *      map      = nullptr;
    size_t         fsize    = 0;
    size_t         data_off = 0;
    int            fd       = -1;
};

bool           vae_gguf_open(VaeGGUF & g, const std::string & path);
void           vae_gguf_close(VaeGGUF & g);
const void *   vae_gdata(const VaeGGUF & g, const std::string & name);  // raw mmap ptr, nullptr if absent
ggml_tensor *  vae_gmeta(const VaeGGUF & g, const std::string & name);  // metadata tensor, nullptr if absent
bool           vae_gguf_has(const VaeGGUF & g, const std::string & name);

// weight_norm fusion into a pre-allocated backend tensor (dst must already be
// backed by a buffer; data is uploaded via ggml_backend_tensor_set).
void vae_fuse_wn(ggml_tensor * dst, const VaeGGUF & g, const std::string & pfx);     // conv1d
void vae_fuse_wn_ct(ggml_tensor * dst, const VaeGGUF & g, const std::string & pfx);  // conv-transpose (-> [IC, K*OC])
void vae_load_snake(ggml_tensor * dst, const VaeGGUF & g, const std::string & name, bool inv);  // exp(a) or 1/exp(b)
void vae_load_bias(ggml_tensor * dst, const VaeGGUF & g, const std::string & name);

} // namespace tts_cpp::acestep
