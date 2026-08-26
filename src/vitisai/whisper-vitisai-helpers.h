#pragma once

#include "FlexMLClient.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace whisper_vitisai_helpers {

bool map_rai_file(const char * path, uint8_t ** buffer, size_t * size);
void unmap_rai_file(uint8_t * buffer, size_t size);

const char * whisper_kv_type_name(ggml_type type);
const char * whisper_flexml_dtype_name(flexmlrt::client::DataType type);
bool whisper_flexml_dtype_to_ggml_type(
        flexmlrt::client::DataType type,
        ggml_type * ggml_dtype);

bool whisper_validate_cross_shape(
        const char * tensor_name,
        const std::vector<std::uint32_t> & model_shape,
        int n_text_layer,
        int n_ctx,
        int n_state);

bool whisper_vitisai_bind_tensor_data(
        const char * tensor_name,
        struct ggml_tensor * runtime_tensor,
        const std::vector<size_t> & expected_shape,
        flexmlrt::client::ErtTensorType & io_tensor);

#if defined(WHISPER_DEBUG)
template <typename T>
void whisper_vitisai_print_shape(const std::vector<T> & shape) {
    std::fprintf(stderr, "[");
    for (size_t i = 0; i < shape.size(); ++i) {
        std::fprintf(stderr, "%s%lld", i == 0 ? "" : ", ", (long long) shape[i]);
    }
    std::fprintf(stderr, "]");
}
#endif

// Model IO tensor indices and metadata sizes resolved once at init time.
struct whisper_vitisai_io_binding {
    int    mel_in_idx              = -1;
    int    embd_enc_out_idx        = -1;
    int    cross_k_out_idx         = -1;
    int    cross_v_out_idx         = -1;
    size_t mel_in_expected_bytes   = 0;
    size_t embd_enc_expected_bytes = 0;
    size_t cross_k_expected_bytes  = 0;
    size_t cross_v_expected_bytes  = 0;
};

// Warnings are printed with the caller's name; hard failures are returned in *error
// so the caller can decide how to report them.
bool whisper_vitisai_resolve_io_binding(
        const char * caller,
        const std::vector<flexmlrt::client::ErtTensorType> & input_tensors,
        const std::vector<flexmlrt::client::ErtTensorType> & output_tensors,
        whisper_vitisai_io_binding * binding,
        std::string * error);

bool whisper_vitisai_all_tensors_claimed(
        const char * caller,
        const char * tensor_kind,
        const std::vector<flexmlrt::client::ErtTensorType> & tensors,
        const std::vector<bool> & claimed);

// Geometry of one cross K/V transfer from the model output (always f32, contiguous
// [ctx, state] per layer) into the runtime kv cache.
struct whisper_kv_cross_layout {
    int    n_layer          = 0;
    int    n_ctx            = 0;
    int    n_state          = 0;
    size_t src_layer_elems  = 0; // f32 elements per layer in the model output buffer
    size_t layer_elems      = 0; // elements per layer transferred into the kv cache
    size_t dst_layer_stride = 0; // bytes per layer in the kv cache
    float  kscale           = 1.0f;
};

void whisper_kv_cross_scale_k_f32(
        float * k_data,
        size_t count,
        float kscale);

void whisper_kv_cross_store_layers_f32(
        const float * src_k,
        const float * src_v,
        uint8_t * dst_k,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout);

void whisper_kv_cross_store_layers_f16(
        const float * src_k,
        const float * src_v,
        uint8_t * dst_k,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout);

void whisper_kv_cross_transpose_v_layers_f32(
        const float * src_v,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout);

void whisper_kv_cross_store_k_transpose_v_layers_f16(
        const float * src_k,
        const float * src_v,
        uint8_t * dst_k,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout);

} // namespace whisper_vitisai_helpers
