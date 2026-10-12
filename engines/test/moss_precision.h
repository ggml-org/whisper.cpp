#pragma once

#include "ggml.h"
#include "gguf.h"
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

// Re-encode generated F32 fixtures using the same mixed-precision layout as
// the converters: matrix weights change type; biases, norms and assets do not.
// Tiny rows that cannot hold a quantization block remain F16.
inline void moss_fixture_precision(const std::filesystem::path & path, ggml_type precision) {
    if (precision == GGML_TYPE_F32) return;
    ggml_context * raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> input(
            gguf_init_from_file(path.string().c_str(), {false, &raw}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> source(raw, ggml_free);
    if (!input || !source) throw std::runtime_error("cannot read precision fixture");
    const size_t arena = std::filesystem::file_size(path) +
            (size_t) gguf_get_n_tensors(input.get()) * ggml_tensor_overhead() + 1024 * 1024;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> destination(
            ggml_init({arena, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    if (!destination || !output) throw std::runtime_error("cannot allocate precision fixture");
    gguf_set_kv(output.get(), input.get());
    for (auto * src = ggml_get_first_tensor(source.get()); src; src = ggml_get_next_tensor(source.get(), src)) {
        const std::string name = ggml_get_name(src);
        const bool matrix = src->ne[0] > 1 && src->ne[1] > 1 &&
                name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0;
        if (!matrix || src->type != GGML_TYPE_F32) {
            gguf_add_tensor(output.get(), src);
            continue;
        }
        ggml_type type = precision;
        if (ggml_is_quantized(type) && (src->ne[0] % ggml_blck_size(type) || src->ne[2] != 1)) {
            type = GGML_TYPE_F16;
        }
        auto * dst = ggml_new_tensor(destination.get(), type, GGML_MAX_DIMS, src->ne);
        ggml_set_name(dst, name.c_str());
        const size_t bytes = ggml_quantize_chunk(type, static_cast<const float *>(src->data), dst->data,
                0, ggml_nelements(src) / src->ne[0], src->ne[0], nullptr);
        if (bytes != ggml_nbytes(dst)) throw std::runtime_error("fixture quantization size mismatch");
        gguf_add_tensor(output.get(), dst);
    }
    if (!gguf_write_to_file(output.get(), path.string().c_str(), false)) {
        throw std::runtime_error("cannot write precision fixture");
    }
}
