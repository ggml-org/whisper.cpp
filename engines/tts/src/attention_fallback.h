#pragma once

// Attention for backends that decline GGML_OP_FLASH_ATTN_EXT.  ggml-opencl
// declines it on Adreno by policy (its kernels miscompute some inputs there);
// a scheduler fallback would then run every attention on the CPU and copy the
// KV cache across on every decode step.

#include "ggml.h"
#include "ggml-backend.h"

namespace tts_cpp::detail {

// Same result and layout as ggml_flash_attn_ext(q, k, v, mask, scale, 0, 0),
// ne=[D, H, N, B], from contiguous score and value products and a masked
// softmax.  q ne=[D, N, H, B]; k and v ne=[D, L, H_kv, B] with H % H_kv == 0;
// mask is null or F16/F32 with ne=[L, >= N].
ggml_tensor * unfused_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                ggml_tensor * mask, float scale);

// ggml_flash_attn_ext when `backend` runs that node (or is null), otherwise
// unfused_attention over the same inputs.
ggml_tensor * attention_for_backend(ggml_context * ctx, ggml_backend_t backend, ggml_tensor * q,
                                    ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, float scale);

} // namespace tts_cpp::detail
