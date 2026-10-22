#pragma once

// Exact per-output-channel affine codes carried in unit-scale GGML blocks.
// These are not Q4_0/Q8_0 model weights: the architecture tag is deliberately
// distinct so older engines refuse them. Reconstruction is a transient graph
// operation, with FP32 min/step and an explicit FP16 rounding boundary.
#include "ggml.h"
#include "gguf.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <unordered_map>

namespace parakeet {
struct PackedAffineWeight {
    ggml_tensor * minimum;
    ggml_tensor * step;
    int shift;
};
using PackedAffineWeights = std::unordered_map<const ggml_tensor *, PackedAffineWeight>;

inline PackedAffineWeights load_packed_affine(gguf_context * file, ggml_context * weights) {
    PackedAffineWeights result;
    const int version = gguf_find_key(file, "parakeet.affine.version");
    if (version < 0) return result;
    const int names = gguf_find_key(file, "parakeet.affine.names");
    if (gguf_get_kv_type(file, version) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(file, version) != 1 || names < 0 ||
        gguf_get_kv_type(file, names) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(file, names) != GGUF_TYPE_STRING || gguf_get_arr_n(file, names) == 0) {
        throw std::runtime_error("invalid packed-affine schema");
    }
    for (size_t i = 0; i < gguf_get_arr_n(file, names); ++i) {
        const std::string name = gguf_get_arr_str(file, names, i);
        const bool encoder = name.rfind("encoder.blk.", 0) == 0;
        const bool matrix = name.find(".ff1.linear") != std::string::npos ||
                            name.find(".ff2.linear") != std::string::npos ||
                            name.find(".attn.") != std::string::npos ||
                            name.find(".conv.pw") != std::string::npos;
        if (!encoder || !matrix || name.size() < 7 ||
            name.compare(name.size() - 7, 7, ".weight") != 0) {
            throw std::runtime_error("unsupported packed-affine target: " + name);
        }
        auto * codes = ggml_get_tensor(weights, name.c_str());
        auto * minimum = ggml_get_tensor(weights, (name + ".affine.minimum").c_str());
        auto * step = ggml_get_tensor(weights, (name + ".affine.step").c_str());
        if (!codes || !minimum || !step || ggml_n_dims(codes) != 2 ||
            (codes->type != GGML_TYPE_Q4_0 && codes->type != GGML_TYPE_Q8_0) ||
            minimum->type != GGML_TYPE_F32 || step->type != GGML_TYPE_F32 ||
            ggml_nelements(minimum) != codes->ne[1] || ggml_nelements(step) != codes->ne[1] ||
            result.count(codes)) {
            throw std::runtime_error("invalid packed-affine tensor: " + name);
        }
        result.emplace(codes, PackedAffineWeight{minimum, step,
                       codes->type == GGML_TYPE_Q4_0 ? 8 : 128});
    }
    return result;
}

inline bool affine_fused_requested() {
    const char * value = std::getenv("PARAKEET_AFFINE_FUSED");
    return value && std::strcmp(value, "1") == 0;
}

struct PackedAffineGraphScope;
inline thread_local PackedAffineGraphScope * current_affine_graph = nullptr;
struct PackedAffineGraphScope {
    ggml_context * context;
    ggml_cgraph * graph;
    const PackedAffineWeights & weights;
    PackedAffineGraphScope * previous;
    bool fused_enabled;
    size_t fused_ops = 0;
    PackedAffineGraphScope(ggml_context * c, ggml_cgraph * g, const PackedAffineWeights & w, bool fused = false)
        : context(c), graph(g), weights(w), previous(current_affine_graph), fused_enabled(fused) {
#ifndef GGML_AFFINE_METAL_V1
        if (fused) throw std::runtime_error("fused affine requires the experimental GGML Metal backend");
#endif
        current_affine_graph = this;
    }
    ~PackedAffineGraphScope() {
        if (fused_enabled && std::getenv("PARAKEET_DEBUG_AFFINE"))
            std::fprintf(stderr, "parakeet: encoder graph contains %zu fused affine matmuls\n", fused_ops);
        current_affine_graph = previous;
    }
    PackedAffineGraphScope(const PackedAffineGraphScope &) = delete;
    PackedAffineGraphScope & operator=(const PackedAffineGraphScope &) = delete;
};

inline ggml_tensor * affine_weight(ggml_context * ctx, ggml_tensor * weight,
                                   ggml_tensor * activation) {
    auto * scope = current_affine_graph;
    if (!scope || scope->context != ctx) return weight;
    const auto found = scope->weights.find(weight);
    if (found == scope->weights.end()) return weight;
    // Schedule the activation prefix before this reconstruction. Otherwise the
    // default src0-first DFS can materialize every dense weight before layer 0.
    ggml_build_forward_expand(scope->graph, activation);
    const auto & record = found->second;
    auto * codes = ggml_cast(ctx, weight, GGML_TYPE_F32);
    codes = ggml_scale_bias(ctx, codes, 1.0f, float(record.shift));
    auto * step = ggml_reshape_2d(ctx, record.step, 1, weight->ne[1]);
    auto * minimum = ggml_reshape_2d(ctx, record.minimum, 1, weight->ne[1]);
    auto * scaled = ggml_mul(ctx, codes, step);
    // Keep multiply/add as distinct FP32 operations, matching the saved decoder.
    scaled = ggml_cont(ctx, scaled);
    auto * restored = ggml_cast(ctx, ggml_add(ctx, scaled, minimum), GGML_TYPE_F16);
    ggml_format_name(restored, "%s.restored", ggml_get_name(weight));
    return restored;
}
// Pointwise callers inspect the weight shape before matmul; preserve the
// carrier identity there when this graph will use the fused backend.
inline ggml_tensor * affine_weight_for_view(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x) {
    auto * scope = current_affine_graph;
    if (scope && scope->context == ctx && scope->fused_enabled && scope->weights.count(weight)) return weight;
    return affine_weight(ctx, weight, x);
}
inline ggml_tensor * affine_mul_mat(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x) {
#ifdef GGML_AFFINE_METAL_V1
    auto * scope = current_affine_graph;
    if (scope && scope->context == ctx && scope->fused_enabled) {
        const auto found = scope->weights.find(weight);
        if (found != scope->weights.end()) {
            if (x->type != GGML_TYPE_F32 || ggml_n_dims(x) > 2)
                throw std::runtime_error("unsupported fused affine activation");
            auto * contiguous = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
            auto * result = ggml_mul_mat_affine_v1(ctx, weight, contiguous, found->second.step, found->second.minimum);
            ggml_format_name(result, "%s.fused_affine", ggml_get_name(weight));
            scope->fused_ops++;
            return result;
        }
    }
#endif
    return ggml_mul_mat(ctx, affine_weight(ctx, weight, x), x);
}
} // namespace parakeet
