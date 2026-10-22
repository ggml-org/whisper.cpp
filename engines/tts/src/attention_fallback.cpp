#include "attention_fallback.h"

namespace tts_cpp::detail {

namespace {

ggml_tensor * contiguous(ggml_context * ctx, ggml_tensor * t) {
    return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

} // namespace

ggml_tensor * unfused_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                ggml_tensor * mask, float scale) {
    ggml_tensor * queries         = contiguous(ctx, q);
    ggml_tensor * keys            = contiguous(ctx, k);
    ggml_tensor * values_by_key   = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor * scores          = ggml_mul_mat(ctx, keys, queries);
    ggml_tensor * weights         = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * heads_by_query  = ggml_mul_mat(ctx, values_by_key, weights);
    return ggml_cont(ctx, ggml_permute(ctx, heads_by_query, 0, 2, 1, 3));
}

ggml_tensor * attention_for_backend(ggml_context * ctx, ggml_backend_t backend, ggml_tensor * q,
                                    ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, float scale) {
    ggml_tensor * fused = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    if (backend == nullptr || ggml_backend_supports_op(backend, fused)) return fused;
    return unfused_attention(ctx, q, k, v, mask, scale);
}

} // namespace tts_cpp::detail
