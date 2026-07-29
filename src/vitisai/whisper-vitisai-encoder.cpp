#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "vitisai/whisper-vitisai-encoder.h"
#include "FlexMLClient.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(WHISPER_DEBUG)
#define WHISPER_DBG_TIMER(name) const int64_t name = ggml_time_us()
#else
#define WHISPER_DBG_TIMER(name) do {} while (0)
#endif

#if defined(WHISPER_DEBUG)
template <typename T>
static void whisper_vitisai_print_shape(const std::vector<T> & shape) {
    std::fprintf(stderr, "[");
    for (size_t i = 0; i < shape.size(); ++i) {
        std::fprintf(stderr, "%s%lld", i == 0 ? "" : ", ", (long long) shape[i]);
    }
    std::fprintf(stderr, "]");
}
#endif

struct whisper_vitisai_context {
    std::string model_path;
    std::shared_ptr<flexmlrt::client::Model> runner;
    uint8_t * fbs_buffer = nullptr;
    size_t fbs_buffer_size = 0;

    std::vector<float> cross_k_staging;
    std::vector<float> cross_v_staging;

    int embd_enc_out_idx = -1;
    int cross_k_out_idx = -1;
    int cross_v_out_idx = -1;

    std::vector<flexmlrt::client::ErtTensorType> cached_input_tensors;
    std::vector<flexmlrt::client::ErtTensorType> cached_output_tensors;
};

// Function to mmap rai file for Linux and MapViewOfFile for Windows
static bool map_rai_file(const char * path, uint8_t ** buffer, size_t * size) {
#ifdef _WIN32
    // Open the file
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "%s: %d: Failed to open rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Get the file size
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to get file size for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Create a file mapping object
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, fileSize.QuadPart, NULL);
    if (hMapping == NULL) {
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to create file mapping for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Map the file
    *buffer = (uint8_t *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, fileSize.QuadPart);
    if (*buffer == NULL) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to map rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }
    *size = fileSize.QuadPart;
    return true;
#else
    // Open the file
    FILE * fd = fopen(path, "rb");
    if (!fd) {
        std::fprintf(stderr, "%s: %d: Failed to open rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Get the file size
    struct stat st;
    if (fstat(fileno(fd), &st) == -1) {
        fclose(fd);
        std::fprintf(stderr, "%s: %d: Failed to get file size for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Mmap the file
    *buffer = (uint8_t *)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fileno(fd), 0);
    if (*buffer == MAP_FAILED) {
        fclose(fd);
        std::fprintf(stderr, "%s: %d: Failed to mmap rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }
    *size = st.st_size;
    return true;
#endif // _WIN32
}

static void unmap_rai_file(uint8_t * buffer, size_t size) {
#ifdef _WIN32
    UnmapViewOfFile(buffer);
#else
    munmap(buffer, size);
#endif // _WIN32
}

bool whisper_vitisai_file_exists(const char * path) {
    if (!path) {
        return false;
    }

    FILE * file = fopen(path, "rb");
    if (!file) {
        return false;
    }
    fclose(file);
    return true;
}

// Reuse cached tensor descriptors to avoid repeated getIOTensors() lookups.
static bool whisper_vitisai_get_io_tensors(
        struct whisper_vitisai_context * ctx,
        std::vector<flexmlrt::client::ErtTensorType> & input_tensors,
        std::vector<flexmlrt::client::ErtTensorType> & output_tensors) {
    if (!ctx || !ctx->runner) {
        return false;
    }

    if (ctx->cached_input_tensors.empty() || ctx->cached_output_tensors.empty()) {
        ctx->cached_input_tensors  = ctx->runner->getIOTensors("input", false);
        ctx->cached_output_tensors = ctx->runner->getIOTensors("output", false);
    }

    input_tensors  = ctx->cached_input_tensors;
    output_tensors = ctx->cached_output_tensors;
    return true;
}

struct whisper_vitisai_context * whisper_vitisai_init(const char * path_model) {
    if (!path_model) {
        std::fprintf(stderr, "%s: path_model is null\n", __func__);
        return nullptr;
    }

    auto * ctx = new whisper_vitisai_context;
    ctx->model_path = path_model;

    // Override the model path with the environment variable if it is set
    if (const char * env_model_path = std::getenv("OVERRIDE_VITISAI_MODEL_PATH")) {
        if (env_model_path[0] != '\0') {
            ctx->model_path = env_model_path;
        }
    }

    // Step 1: Set up the model
    flexmlrt::client::Options options;
    options.modelPath = ctx->model_path;
    options.debug = false;
    options.executeMode = 2;
    options.extOptions["enable_preemption"] = true;

    // Check if model_path is rai file and if so, add fbs_buffer and fbs_buffer_size to the options
    if (ctx->model_path.find(".rai") != std::string::npos) {
        if (map_rai_file(ctx->model_path.c_str(), &ctx->fbs_buffer, &ctx->fbs_buffer_size)) {
            options.extOptions["fbs_buffer"] = ctx->fbs_buffer;
            options.extOptions["fbs_buffer_size"] = ctx->fbs_buffer_size;
            options.extOptions["cache_dir"] = std::string(".");
        } else {
            std::fprintf(stderr, "%s: Failed to mmap rai file '%s'\n", __func__, ctx->model_path.c_str());
            delete ctx;
            return nullptr;
        }
    } else {
        options.deviceName = "stx";
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: Using default device name 'stx'\n", __func__);
#endif
    }

    const bool model_is_rai = ctx->model_path.find(".rai") != std::string::npos;
    if (model_is_rai) {
#if WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES
        options.deviceName = "stx";
        options.subgraphName = "vaiml_par_0";
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr,
                "%s: legacy FlexMLRT compile configuration detected; applying RAI overrides (device='stx', subgraph='vaiml_par_0')\n",
                __func__);
#endif // defined(WHISPER_DEBUG)
#endif
    }

    try {
        ctx->runner = std::make_shared<flexmlrt::client::Model>(options);

        if (!ctx->runner->good()) {
            throw std::runtime_error("Runner creation ran into an error");
        }

        ctx->cached_input_tensors  = ctx->runner->getIOTensors("input", false);
        ctx->cached_output_tensors = ctx->runner->getIOTensors("output", false);

        auto & output_tensors = ctx->cached_output_tensors;
        for (int i = 0; i < (int) output_tensors.size(); ++i) {
            const std::string & name = output_tensors[i].getMetadata().name;
            if (name == "embd_enc") {
                ctx->embd_enc_out_idx = i;
            } else if (name == "cross_k") {
                ctx->cross_k_out_idx = i;
            } else if (name == "cross_v") {
                ctx->cross_v_out_idx = i;
            }
        }

        if (ctx->embd_enc_out_idx < 0) {
            std::fprintf(stderr, "%s: WARNING: embd_enc output not found by name; falling back to output[0]\n", __func__);
            ctx->embd_enc_out_idx = 0;
        }

#if defined(WHISPER_DEBUG)
        {
            auto & input_tensors = ctx->cached_input_tensors;

            std::fprintf(stderr, "%s: model has %zu input tensor(s)\n", __func__, input_tensors.size());
            for (int i = 0; i < (int) input_tensors.size(); ++i) {
                const auto & meta = input_tensors[i].getMetadata();
                std::fprintf(stderr, "%s:   input[%d] name='%s' size=%zu shape=",
                        __func__, i, meta.name.c_str(), (size_t) meta.size);
                whisper_vitisai_print_shape(meta.shape);
                std::fprintf(stderr, "\n");
            }

            std::fprintf(stderr, "%s: model has %zu output tensor(s)\n", __func__, output_tensors.size());
            for (int i = 0; i < (int) output_tensors.size(); ++i) {
                const auto & meta = output_tensors[i].getMetadata();
                std::fprintf(stderr, "%s:   output[%d] name='%s' size=%zu shape=",
                        __func__, i, meta.name.c_str(), (size_t) meta.size);
                whisper_vitisai_print_shape(meta.shape);
                std::fprintf(stderr, "\n");
            }

            std::fprintf(stderr, "%s: output indices: embd_enc=%d cross_k=%d cross_v=%d\n",
                    __func__, ctx->embd_enc_out_idx, ctx->cross_k_out_idx, ctx->cross_v_out_idx);
        }
#endif
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: Exception during Vitis AI runner creation: %s\n", __func__, e.what());
        delete ctx;
        return nullptr;
    }
    return ctx;
}

bool whisper_vitisai_has_cross_proj(const struct whisper_vitisai_context * ctx) {
    return ctx && ctx->cross_k_out_idx >= 0 && ctx->cross_v_out_idx >= 0;
}

void whisper_vitisai_free(struct whisper_vitisai_context * ctx) {
    if (!ctx) {
        return;
    }

#if defined(WHISPER_DEBUG)
    std::fprintf(stderr, "%s: releasing Vitis AI context for model '%s'\n", __func__, ctx->model_path.c_str());
#endif
    if (ctx->fbs_buffer) {
        unmap_rai_file(ctx->fbs_buffer, ctx->fbs_buffer_size);
    }
    delete ctx;
}

int whisper_vitisai_encode(struct whisper_vitisai_context * ctx, struct ggml_tensor * mel, struct ggml_tensor * out) {
    if (!ctx || !mel || !out) {
        std::fprintf(stderr, "%s: ctx/mel/out must not be null\n", __func__);
        return 0;
    }

    if (ggml_n_dims(mel) != 2) {
        std::fprintf(stderr, "%s: mel tensor expected to have 2 dims, got %d\n", __func__, ggml_n_dims(mel));
        return 0;
    }

    if (ggml_n_dims(out) != 2) {
        std::fprintf(stderr, "%s: out tensor expected to have 2 dims, got %d\n", __func__, ggml_n_dims(out));
        return 0;
    }

    // setup input and output tensors for Vitis AI model
    std::vector<flexmlrt::client::ErtTensorType> input_tensors, output_tensors;
    auto model = ctx->runner;

    if (!whisper_vitisai_get_io_tensors(ctx, input_tensors, output_tensors)) {
        std::fprintf(stderr, "%s: failed to acquire Vitis AI I/O tensors\n", __func__);
        return 0;
    }

    // TODO: add assert checks for tensor numbers and shapes

    if (ctx->embd_enc_out_idx < 0 || ctx->embd_enc_out_idx >= (int) output_tensors.size()) {
        std::fprintf(stderr, "%s: invalid embd_enc output index %d for %zu output tensor(s)\n",
                __func__, ctx->embd_enc_out_idx, output_tensors.size());
        return 0;
    }

    input_tensors[0].data = mel->data;
    output_tensors[ctx->embd_enc_out_idx].data = out->data;

    try {
        model->forward(input_tensors, output_tensors);
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: Vitis AI model inference completed.\n", __func__);
#endif
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: Exception during model inference: %s\n", __func__, e.what());
        return 0;
    }

    return 1;
}

int whisper_vitisai_run_enc_cross(
        struct whisper_vitisai_context * ctx,
        struct ggml_tensor * mel,
        struct ggml_tensor * out,
        void * cross_v_data,
        void * cross_k_data) {
    if (!ctx || !mel || !out || !cross_v_data || !cross_k_data) {
        std::fprintf(stderr, "%s: ctx/mel/out/cross_v_data/cross_k_data must not be null\n", __func__);
        return 0;
    }

    if (ggml_n_dims(mel) != 2) {
        std::fprintf(stderr, "%s: mel tensor expected to have 2 dims, got %d\n", __func__, ggml_n_dims(mel));
        return 0;
    }

    if (ggml_n_dims(out) != 2) {
        std::fprintf(stderr, "%s: out tensor expected to have 2 dims, got %d\n", __func__, ggml_n_dims(out));
        return 0;
    }

    std::vector<flexmlrt::client::ErtTensorType> input_tensors, output_tensors;
    auto model = ctx->runner;

    if (!whisper_vitisai_get_io_tensors(ctx, input_tensors, output_tensors)) {
        std::fprintf(stderr, "%s: failed to acquire Vitis AI I/O tensors\n", __func__);
        return 0;
    }

    if (output_tensors.size() != 3) {
        std::fprintf(stderr, "%s: expected 3 output tensors, got %zu\n", __func__, output_tensors.size());
        return 0;
    }

    if (ctx->embd_enc_out_idx < 0 || ctx->embd_enc_out_idx >= (int) output_tensors.size() ||
            ctx->cross_k_out_idx < 0 || ctx->cross_k_out_idx >= (int) output_tensors.size() ||
            ctx->cross_v_out_idx < 0 || ctx->cross_v_out_idx >= (int) output_tensors.size()) {
        std::fprintf(stderr, "%s: invalid output indices embd_enc=%d cross_k=%d cross_v=%d for %zu output tensor(s)\n",
                __func__, ctx->embd_enc_out_idx, ctx->cross_k_out_idx, ctx->cross_v_out_idx, output_tensors.size());
        return 0;
    }

    input_tensors[0].data = mel->data;
    output_tensors[ctx->embd_enc_out_idx].data = out->data;
    output_tensors[ctx->cross_v_out_idx].data = cross_v_data;
    output_tensors[ctx->cross_k_out_idx].data = cross_k_data;

    try {
        model->forward(input_tensors, output_tensors);
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: Vitis AI model inference (encoder + cross proj) completed.\n", __func__);
#endif
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: Exception during model inference: %s\n", __func__, e.what());
        return 0;
    }

    return 1;
}

// Ensure persistent staging buffers are large enough for the given dimensions.
static void ensure_staging_buffers(
        struct whisper_vitisai_context * ctx,
        size_t count, bool need_k) {
    if (need_k && ctx->cross_k_staging.size() < count) {
        ctx->cross_k_staging.resize(count);
    }
    if (ctx->cross_v_staging.size() < count) {
        ctx->cross_v_staging.resize(count);
    }
}

int whisper_vitisai_encode_with_cross(
        struct whisper_vitisai_context * ctx,
        struct ggml_tensor * mel,
        struct ggml_tensor * embd_enc,
        struct ggml_tensor * kv_cross_k,
        struct ggml_tensor * kv_cross_v,
        int n_text_layer,
        int n_ctx,
        int n_text_state,
        int n_text_head,
        bool flash_attn) {
    if (!ctx || !mel || !embd_enc || !kv_cross_k || !kv_cross_v) {
        std::fprintf(stderr, "%s: null argument\n", __func__);
        return 0;
    }

    const int n_state = n_text_state;
    const int n_state_head = n_state / n_text_head;
    const int n_ctx_pad = (n_ctx + 255) & ~255; // GGML_PAD(n_ctx, 256)

    const float Kscale = pow(float(n_state_head), -0.25f);
    const ggml_type kv_type = kv_cross_k->type;
    const size_t elem_size = ggml_type_size(kv_type);
    const size_t layer_elems = (size_t)n_ctx * n_state;
    const size_t buf_count = (size_t)n_text_layer * layer_elems;

    if (flash_attn) {
        WHISPER_DBG_TIMER(t_fwd_start);

        if (n_ctx_pad == n_ctx) {
            // No padding gap -- plugin writes directly into kv_cross.
            if (!whisper_vitisai_run_enc_cross(
                    ctx, mel, embd_enc,
                    kv_cross_v->data, kv_cross_k->data)) {
                return 0;
            }

            WHISPER_DBG_TIMER(t_fwd_end);
            WHISPER_DBG_TIMER(t_post_start);

            if (kv_type == GGML_TYPE_F32) {
                float * kdata = (float *)kv_cross_k->data;
                for (size_t i = 0; i < buf_count; ++i) {
                    kdata[i] *= Kscale;
                }
            } else if (kv_type == GGML_TYPE_F16) {
                ggml_fp16_t * kdata = (ggml_fp16_t *)kv_cross_k->data;
                for (size_t i = 0; i < buf_count; ++i) {
                    kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * Kscale);
                }
            }

            WHISPER_DBG_TIMER(t_post_end);

#if defined(WHISPER_DEBUG)
            std::fprintf(stderr, "%s: vitisai enc+cross forward time = %8.2f ms\n", __func__, (t_fwd_end - t_fwd_start) / 1000.0f);
            std::fprintf(stderr, "%s: kv_cross post-process time     = %8.2f ms (flash, no-pad direct)\n", __func__, (t_post_end - t_post_start) / 1000.0f);
#endif
        } else {
            // Padding gap -- use persistent staging buffers.
            ensure_staging_buffers(ctx, buf_count, true);
            float * cross_k_buf = ctx->cross_k_staging.data();
            float * cross_v_buf = ctx->cross_v_staging.data();

            if (!whisper_vitisai_run_enc_cross(
                    ctx, mel, embd_enc,
                    cross_v_buf, cross_k_buf)) {
                return 0;
            }

            WHISPER_DBG_TIMER(t_fwd_end);
            WHISPER_DBG_TIMER(t_post_start);

            // Combined per-layer K+V scatter for better cache locality.
            const size_t padded_layer_stride = elem_size * n_state * n_ctx_pad;

            for (int il = 0; il < n_text_layer; ++il) {
                const float * src_k = cross_k_buf + (size_t)il * layer_elems;
                const float * src_v = cross_v_buf + (size_t)il * layer_elems;
                uint8_t * dst_k = (uint8_t *)kv_cross_k->data + padded_layer_stride * il;
                uint8_t * dst_v = (uint8_t *)kv_cross_v->data + padded_layer_stride * il;

                if (kv_type == GGML_TYPE_F32) {
                    float * dk = (float *)dst_k;
                    for (size_t i = 0; i < layer_elems; ++i) {
                        dk[i] = src_k[i] * Kscale;
                    }
                    memcpy(dst_v, src_v, layer_elems * sizeof(float));
                } else if (kv_type == GGML_TYPE_F16) {
                    ggml_fp16_t * dk = (ggml_fp16_t *)dst_k;
                    ggml_fp16_t * dv = (ggml_fp16_t *)dst_v;
                    for (size_t i = 0; i < layer_elems; ++i) {
                        dk[i] = ggml_fp32_to_fp16(src_k[i] * Kscale);
                        dv[i] = ggml_fp32_to_fp16(src_v[i]);
                    }
                }
            }

            WHISPER_DBG_TIMER(t_post_end);

#if defined(WHISPER_DEBUG)
            std::fprintf(stderr, "%s: vitisai enc+cross forward time = %8.2f ms\n", __func__, (t_fwd_end - t_fwd_start) / 1000.0f);
            std::fprintf(stderr, "%s: kv_cross post-process time     = %8.2f ms (flash, padded, n_ctx=%d, n_ctx_pad=%d, kv_type=%s)\n",
                __func__, (t_post_end - t_post_start) / 1000.0f,
                n_ctx, n_ctx_pad,
                kv_type == GGML_TYPE_F32 ? "F32" : kv_type == GGML_TYPE_F16 ? "F16" : "other");
#endif
        }
    } else {
        // Non-flash: layers are contiguous (stride = n_state * n_ctx).
        // K: plugin writes directly into kv_cross_k, then in-place Kscale.
        // V: persistent staging buffer + cache-friendly blocked transpose.
        ensure_staging_buffers(ctx, buf_count, false);
        float * cross_v_buf = ctx->cross_v_staging.data();

        WHISPER_DBG_TIMER(t_fwd_start);

        if (!whisper_vitisai_run_enc_cross(
                ctx, mel, embd_enc,
                cross_v_buf, kv_cross_k->data)) {
            return 0;
        }

        WHISPER_DBG_TIMER(t_fwd_end);
        WHISPER_DBG_TIMER(t_post_start);

        if (kv_type == GGML_TYPE_F32) {
            float * kdata = (float *)kv_cross_k->data;
            for (size_t i = 0; i < buf_count; ++i) {
                kdata[i] *= Kscale;
            }

            const int BLOCK = 32;
            for (int il = 0; il < n_text_layer; ++il) {
                const float * src_v = cross_v_buf + (size_t)il * layer_elems;
                float * dst_v = (float *)kv_cross_v->data + (size_t)il * layer_elems;

                for (int ic = 0; ic < n_ctx; ic += BLOCK) {
                    for (int is = 0; is < n_state; is += BLOCK) {
                        const int ic_end = std::min(ic + BLOCK, n_ctx);
                        const int is_end = std::min(is + BLOCK, n_state);
                        for (int i = ic; i < ic_end; ++i) {
                            for (int j = is; j < is_end; ++j) {
                                dst_v[j * n_ctx + i] = src_v[i * n_state + j];
                            }
                        }
                    }
                }
            }
        } else if (kv_type == GGML_TYPE_F16) {
            ggml_fp16_t * kdata = (ggml_fp16_t *)kv_cross_k->data;
            for (size_t i = 0; i < buf_count; ++i) {
                kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * Kscale);
            }

            const int BLOCK = 32;
            for (int il = 0; il < n_text_layer; ++il) {
                const float * src_v = cross_v_buf + (size_t)il * layer_elems;
                ggml_fp16_t * dst_v = (ggml_fp16_t *)((uint8_t *)kv_cross_v->data + elem_size * n_state * n_ctx * il);

                for (int ic = 0; ic < n_ctx; ic += BLOCK) {
                    for (int is = 0; is < n_state; is += BLOCK) {
                        const int ic_end = std::min(ic + BLOCK, n_ctx);
                        const int is_end = std::min(is + BLOCK, n_state);
                        for (int i = ic; i < ic_end; ++i) {
                            for (int j = is; j < is_end; ++j) {
                                dst_v[j * n_ctx + i] = ggml_fp32_to_fp16(src_v[i * n_state + j]);
                            }
                        }
                    }
                }
            }
        }

        WHISPER_DBG_TIMER(t_post_end);

#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: vitisai enc+cross forward time = %8.2f ms\n", __func__, (t_fwd_end - t_fwd_start) / 1000.0f);
        std::fprintf(stderr, "%s: kv_cross post-process time     = %8.2f ms (non-flash)\n", __func__, (t_post_end - t_post_start) / 1000.0f);
#endif
    }

    return 1;
}
