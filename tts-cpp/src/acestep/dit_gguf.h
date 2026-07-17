#pragma once

// ACE-Step DiT — GGUF weight IO (QVAC-21921).
//
// mmaps the DiT GGUF (acestep.cpp convert.py) and uploads weights into
// pre-allocated backend tensors. Unlike the VAE (weight_norm fusion), the DiT
// loader:
//   - reads config from GGUF metadata (acestep-dit.* / acestep.*),
//   - optionally fuses Q/K/V (self+cross attn) and gate/up (MLP) into single
//     tensors when the source quant types match (matmul throughput),
//   - pre-permutes proj_in ([H,in_ch,P] -> [in_ch*P, H]) and proj_out
//     ([H,out_ch,P] -> [H, out_ch*P]) at load time to drop runtime permutes.
//
// Kept as a separate translation unit (mirrors vae_gguf) so the graph builder in
// dit_ggml.cpp never touches gguf directly.

#include "dit_ggml.h"  // DitConfig

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>

namespace tts_cpp::acestep {

// Parsed GGUF: mmapped file + ggml metadata context (descriptors, no data).
struct DitGGUF {
    gguf_context * ctx      = nullptr;
    ggml_context * meta     = nullptr;
    uint8_t *      map      = nullptr;
    size_t         fsize    = 0;
    size_t         data_off = 0;
    int            fd       = -1;
};

bool          dit_gguf_open(DitGGUF & g, const std::string & path);
void          dit_gguf_close(DitGGUF & g);
const void *  dit_gdata(const DitGGUF & g, const std::string & name);  // raw mmap ptr, nullptr if absent
ggml_tensor * dit_gmeta(const DitGGUF & g, const std::string & name);  // metadata tensor, nullptr if absent
bool          dit_gguf_has(const DitGGUF & g, const std::string & name);

// Metadata accessors (throw std::runtime_error on missing key).
uint32_t dit_gguf_u32(const DitGGUF & g, const std::string & key);
float    dit_gguf_f32(const DitGGUF & g, const std::string & key);
bool     dit_gguf_bool(const DitGGUF & g, const std::string & key, bool def);  // def if key missing

// Read the full DiT config from metadata. Returns false if any key is missing.
bool dit_gguf_read_config(const DitGGUF & g, DitConfig & cfg);

} // namespace tts_cpp::acestep
