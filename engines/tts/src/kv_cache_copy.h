#pragma once

#include "ggml-backend.h"
#include "ggml.h"

namespace tts_cpp::detail {

inline ggml_tensor *
build_kv_cache_copy(ggml_context *ctx, ggml_backend_t backend, ggml_tensor *source, ggml_tensor *destination,
                    bool (*supports_op)(ggml_backend_t, const ggml_tensor *) = ggml_backend_supports_op) {
    ggml_tensor *copy = ggml_cpy(ctx, source, destination);
    if (!ggml_is_contiguous(source) && !supports_op(backend, copy)) {
        copy = ggml_cpy(ctx, ggml_cont(ctx, source), destination);
    }
    return copy;
}

}
