#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "vitisai/whisper-vitisai-helpers.h"

#include <algorithm>
#include <cstdio>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
#endif
#include <string>
#include <utility>

namespace whisper_vitisai_helpers {

bool map_rai_file(const char * path, uint8_t ** buffer, size_t * size) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "%s: %d: Failed to open rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to get file size for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, fileSize.QuadPart, NULL);
    if (hMapping == NULL) {
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to create file mapping for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    *buffer = (uint8_t *) MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, fileSize.QuadPart);
    if (*buffer == NULL) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to map rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }
    CloseHandle(hMapping);
    CloseHandle(hFile);
    *size = fileSize.QuadPart;
    return true;
#else
    FILE * fd = fopen(path, "rb");
    if (!fd) {
        std::fprintf(stderr, "%s: %d: Failed to open rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    struct stat st;
    if (fstat(fileno(fd), &st) == -1) {
        fclose(fd);
        std::fprintf(stderr, "%s: %d: Failed to get file size for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    *buffer = (uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fileno(fd), 0);
    if (*buffer == MAP_FAILED) {
        fclose(fd);
        std::fprintf(stderr, "%s: %d: Failed to mmap rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }
    fclose(fd);
    *size = st.st_size;
    return true;
#endif // _WIN32
}

void unmap_rai_file(uint8_t * buffer, size_t size) {
#ifdef _WIN32
    UnmapViewOfFile(buffer);
#else
    munmap(buffer, size);
#endif // _WIN32
}

const char * whisper_kv_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        default:            return "unsupported";
    }
}

const char * whisper_flexml_dtype_name(flexmlrt::client::DataType type) {
    switch (type) {
        case flexmlrt::client::DataType::Float32:  return "Float32";
        case flexmlrt::client::DataType::Int8:     return "Int8";
        case flexmlrt::client::DataType::UInt8:    return "UInt8";
        case flexmlrt::client::DataType::Int16:    return "Int16";
        case flexmlrt::client::DataType::UInt16:   return "UInt16";
        case flexmlrt::client::DataType::BFloat16: return "BFloat16";
        case flexmlrt::client::DataType::Bool:     return "Bool";
        case flexmlrt::client::DataType::Float16:  return "Float16";
        case flexmlrt::client::DataType::Int32:    return "Int32";
        case flexmlrt::client::DataType::UInt32:   return "UInt32";
        default:                                    return "Unknown";
    }
}

bool whisper_flexml_dtype_to_ggml_type(
        flexmlrt::client::DataType type,
        ggml_type * ggml_dtype) {
    switch (type) {
        case flexmlrt::client::DataType::Float32:
            if (ggml_dtype) {
                *ggml_dtype = GGML_TYPE_F32;
            }
            return true;
        case flexmlrt::client::DataType::Float16:
            if (ggml_dtype) {
                *ggml_dtype = GGML_TYPE_F16;
            }
            return true;
        case flexmlrt::client::DataType::BFloat16:
            if (ggml_dtype) {
                *ggml_dtype = GGML_TYPE_BF16;
            }
            return true;
        default:
            return false;
    }
}

static bool whisper_vitisai_validate_tensor_dtype(
        const char * tensor_name,
        flexmlrt::client::DataType model_dtype,
        ggml_type runtime_dtype) {
    ggml_type expected_runtime_dtype = GGML_TYPE_COUNT;
    if (!whisper_flexml_dtype_to_ggml_type(model_dtype, &expected_runtime_dtype)) {
        std::fprintf(stderr,
                "%s: unsupported model dtype for %s: %s (supported: Float32/Float16/BFloat16)\n",
                __func__, tensor_name, whisper_flexml_dtype_name(model_dtype));
        return false;
    }

    if (runtime_dtype != expected_runtime_dtype) {
        std::fprintf(stderr,
                "%s: %s dtype mismatch (runtime=%s, model=%s)\n",
                __func__, tensor_name, ggml_type_name(runtime_dtype), whisper_flexml_dtype_name(model_dtype));
        return false;
    }

    return true;
}

static std::string whisper_shape_to_string(const std::vector<size_t> & shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

static std::vector<size_t> whisper_canonical_shape(const std::vector<std::uint32_t> & shape) {
    std::vector<size_t> canonical;
    canonical.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        const size_t dim = (size_t) shape[i];
        if (dim != 1) {
            canonical.push_back(dim);
        }
    }
    if (canonical.empty()) {
        canonical.push_back(1);
    }
    return canonical;
}

static bool whisper_validate_shape(
        const char * tensor_name,
        const std::vector<std::uint32_t> & model_shape,
        const std::vector<size_t> & expected_shape) {
    const std::vector<size_t> shape = whisper_canonical_shape(model_shape);
    if (shape != expected_shape) {
        std::fprintf(stderr,
                "%s: %s shape mismatch (runtime expected=%s, model=%s)\n",
                __func__,
                tensor_name,
                whisper_shape_to_string(expected_shape).c_str(),
                whisper_shape_to_string(shape).c_str());
        return false;
    }
    return true;
}

bool whisper_validate_cross_shape(
        const char * tensor_name,
        const std::vector<std::uint32_t> & model_shape,
        int n_text_layer,
        int n_ctx,
        int n_state) {
    const std::vector<size_t> expected = {
        (size_t) n_text_layer,
        (size_t) n_ctx,
        (size_t) n_state,
    };
    return whisper_validate_shape(tensor_name, model_shape, expected);
}

bool whisper_vitisai_bind_tensor_data(
        const char * tensor_name,
        struct ggml_tensor * runtime_tensor,
        const std::vector<size_t> & expected_shape,
        flexmlrt::client::ErtTensorType & io_tensor) {
    const auto & meta = io_tensor.getMetadata();
    if (!whisper_vitisai_validate_tensor_dtype(tensor_name, meta.type, runtime_tensor->type)) {
        return false;
    }
    if (!whisper_validate_shape(tensor_name, meta.shape, expected_shape)) {
        return false;
    }

    const size_t model_bytes = meta.size;
    const size_t runtime_bytes = ggml_nbytes(runtime_tensor);
    if (model_bytes == 0 || runtime_bytes == 0) {
        std::fprintf(stderr, "%s: %s sizes must be non-zero (model=%zu, runtime=%zu)\n",
                __func__, tensor_name, model_bytes, runtime_bytes);
        return false;
    }
    if (runtime_bytes != model_bytes) {
        std::fprintf(stderr,
                "%s: %s tensor size mismatch (runtime=%zu B, model=%zu B). "
                "VitisAI .rai requires exact context match; use matching -ac/model artifact.\n",
                __func__, tensor_name, runtime_bytes, model_bytes);
        return false;
    }

    io_tensor.data = runtime_tensor->data;
    return true;
}

bool whisper_vitisai_resolve_io_binding(
        [[maybe_unused]] const char * caller,
        const std::vector<flexmlrt::client::ErtTensorType> & input_tensors,
        const std::vector<flexmlrt::client::ErtTensorType> & output_tensors,
        whisper_vitisai_io_binding * binding,
        std::string * error) {
    const auto fail = [error](std::string message) {
        if (error) {
            *error = std::move(message);
        }
        return false;
    };

    if (input_tensors.empty()) {
        return fail("Model has no input tensors");
    }

    binding->mel_in_idx = 0;
    bool found_named_mel = false;
    for (int i = 0; i < (int) input_tensors.size(); ++i) {
        const std::string & name = input_tensors[i].getMetadata().name;
        if (name == "input" || name == "mel") {
            binding->mel_in_idx = i;
            found_named_mel = true;
            break;
        }
    }
    if (!found_named_mel) {
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: WARNING: mel input not found by name; falling back to input[0]\n", caller);
#endif
    }

    if (output_tensors.empty()) {
        return fail("Model has no output tensors");
    }

    for (int i = 0; i < (int) output_tensors.size(); ++i) {
        const std::string & name = output_tensors[i].getMetadata().name;
        if (name == "embd_enc") {
            binding->embd_enc_out_idx = i;
        } else if (name == "cross_k") {
            binding->cross_k_out_idx = i;
        } else if (name == "cross_v") {
            binding->cross_v_out_idx = i;
        }
    }

    if (binding->embd_enc_out_idx < 0) {
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: WARNING: embd_enc output not found by name; falling back to output[0]\n", caller);
#endif
        binding->embd_enc_out_idx = 0;
    }

    const bool has_cross_k = binding->cross_k_out_idx >= 0;
    const bool has_cross_v = binding->cross_v_out_idx >= 0;
    if (has_cross_k != has_cross_v) {
        return fail("Incomplete cross-projection contract: both cross_k and cross_v outputs are required");
    }

    if (has_cross_k && (binding->cross_k_out_idx == binding->cross_v_out_idx ||
                        binding->cross_k_out_idx == binding->embd_enc_out_idx ||
                        binding->cross_v_out_idx == binding->embd_enc_out_idx)) {
        return fail("Invalid output mapping: embd_enc/cross_k/cross_v indices overlap");
    }

    const auto & mel_meta = input_tensors[binding->mel_in_idx].getMetadata();
    if (!whisper_flexml_dtype_to_ggml_type(mel_meta.type, nullptr)) {
        return fail(
                std::string("Unsupported mel input type: ") +
                whisper_flexml_dtype_name(mel_meta.type) + " (supported: Float32/Float16/BFloat16)");
    }
    binding->mel_in_expected_bytes = mel_meta.size;

    const auto & embd_meta = output_tensors[binding->embd_enc_out_idx].getMetadata();
    if (!whisper_flexml_dtype_to_ggml_type(embd_meta.type, nullptr)) {
        return fail(
                std::string("Unsupported embd_enc output type: ") +
                whisper_flexml_dtype_name(embd_meta.type) + " (supported: Float32/Float16/BFloat16)");
    }
    binding->embd_enc_expected_bytes = embd_meta.size;

    if (has_cross_k) {
        const auto & cross_k_meta = output_tensors[binding->cross_k_out_idx].getMetadata();
        const auto & cross_v_meta = output_tensors[binding->cross_v_out_idx].getMetadata();
        if (cross_k_meta.type != flexmlrt::client::DataType::Float32 ||
                cross_v_meta.type != flexmlrt::client::DataType::Float32) {
            return fail(
                    std::string("Unsupported cross output type(s): cross_k=") +
                    whisper_flexml_dtype_name(cross_k_meta.type) + ", cross_v=" +
                    whisper_flexml_dtype_name(cross_v_meta.type) + " (cross path currently requires Float32)");
        }
        if (cross_k_meta.size != cross_v_meta.size) {
            return fail("cross_k and cross_v output sizes do not match");
        }
        binding->cross_k_expected_bytes = cross_k_meta.size;
        binding->cross_v_expected_bytes = cross_v_meta.size;
    }

    return true;
}

bool whisper_vitisai_all_tensors_claimed(
        const char * caller,
        const char * tensor_kind,
        const std::vector<flexmlrt::client::ErtTensorType> & tensors,
        const std::vector<bool> & claimed) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!claimed[i]) {
            std::fprintf(stderr,
                    "%s: unsupported extra %s tensor at index %zu (name='%s'); strict contract expects only mapped %ss\n",
                    caller, tensor_kind, i, tensors[i].getMetadata().name.c_str(), tensor_kind);
            return false;
        }
    }
    return true;
}

void whisper_kv_cross_scale_k_f32(
        float * k_data,
        size_t count,
        float kscale) {
    for (size_t i = 0; i < count; ++i) {
        k_data[i] *= kscale;
    }
}

void whisper_kv_cross_store_layers_f32(
        const float * src_k,
        const float * src_v,
        uint8_t * dst_k,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout) {
    for (int il = 0; il < layout.n_layer; ++il) {
        const float * layer_src_k = src_k + (size_t)il * layout.src_layer_elems;
        const float * layer_src_v = src_v + (size_t)il * layout.src_layer_elems;
        float * dk = (float *)(dst_k + layout.dst_layer_stride * (size_t)il);
        float * dv = (float *)(dst_v + layout.dst_layer_stride * (size_t)il);
        for (size_t i = 0; i < layout.layer_elems; ++i) {
            dk[i] = layer_src_k[i] * layout.kscale;
            dv[i] = layer_src_v[i];
        }
    }
}

void whisper_kv_cross_store_layers_f16(
        const float * src_k,
        const float * src_v,
        uint8_t * dst_k,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout) {
    for (int il = 0; il < layout.n_layer; ++il) {
        const float * layer_src_k = src_k + (size_t)il * layout.src_layer_elems;
        const float * layer_src_v = src_v + (size_t)il * layout.src_layer_elems;
        ggml_fp16_t * dk = (ggml_fp16_t *)(dst_k + layout.dst_layer_stride * (size_t)il);
        ggml_fp16_t * dv = (ggml_fp16_t *)(dst_v + layout.dst_layer_stride * (size_t)il);
        for (size_t i = 0; i < layout.layer_elems; ++i) {
            dk[i] = ggml_fp32_to_fp16(layer_src_k[i] * layout.kscale);
            dv[i] = ggml_fp32_to_fp16(layer_src_v[i]);
        }
    }
}

void whisper_kv_cross_transpose_v_layers_f32(
        const float * src_v,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout) {
    const int n_ctx   = layout.n_ctx;
    const int n_state = layout.n_state;

    const int BLOCK = 32;
    for (int il = 0; il < layout.n_layer; ++il) {
        const float * layer_src_v = src_v + (size_t)il * layout.src_layer_elems;
        float * dv = (float *)(dst_v + layout.dst_layer_stride * (size_t)il);

        for (int ic = 0; ic < n_ctx; ic += BLOCK) {
            for (int is = 0; is < n_state; is += BLOCK) {
                const int ic_end = std::min(ic + BLOCK, n_ctx);
                const int is_end = std::min(is + BLOCK, n_state);
                for (int i = ic; i < ic_end; ++i) {
                    for (int j = is; j < is_end; ++j) {
                        dv[j * n_ctx + i] = layer_src_v[i * n_state + j];
                    }
                }
            }
        }
    }
}

void whisper_kv_cross_store_k_transpose_v_layers_f16(
        const float * src_k,
        const float * src_v,
        uint8_t * dst_k,
        uint8_t * dst_v,
        const whisper_kv_cross_layout & layout) {
    const int n_ctx   = layout.n_ctx;
    const int n_state = layout.n_state;

    const int BLOCK = 32;
    for (int il = 0; il < layout.n_layer; ++il) {
        const float * layer_src_k = src_k + (size_t)il * layout.src_layer_elems;
        const float * layer_src_v = src_v + (size_t)il * layout.src_layer_elems;
        ggml_fp16_t * dk = (ggml_fp16_t *)(dst_k + layout.dst_layer_stride * (size_t)il);
        ggml_fp16_t * dv = (ggml_fp16_t *)(dst_v + layout.dst_layer_stride * (size_t)il);
        for (size_t i = 0; i < layout.layer_elems; ++i) {
            dk[i] = ggml_fp32_to_fp16(layer_src_k[i] * layout.kscale);
        }

        for (int ic = 0; ic < n_ctx; ic += BLOCK) {
            for (int is = 0; is < n_state; is += BLOCK) {
                const int ic_end = std::min(ic + BLOCK, n_ctx);
                const int is_end = std::min(is + BLOCK, n_state);
                for (int i = ic; i < ic_end; ++i) {
                    for (int j = is; j < is_end; ++j) {
                        dv[j * n_ctx + i] = ggml_fp32_to_fp16(layer_src_v[i * n_state + j]);
                    }
                }
            }
        }
    }
}

} // namespace whisper_vitisai_helpers
