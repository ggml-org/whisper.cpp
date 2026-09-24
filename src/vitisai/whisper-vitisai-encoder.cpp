#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "vitisai/whisper-vitisai-encoder.h"
#include "vitisai/whisper-vitisai-helpers.h"
#include "FlexMLClient.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#if defined(WHISPER_DEBUG)
#define WHISPER_DBG_TIMER(name) const int64_t name = ggml_time_us()
#else
#define WHISPER_DBG_TIMER(name) do {} while (0)
#endif

struct whisper_vitisai_context {
    std::string model_path;
    std::shared_ptr<flexmlrt::client::Model> runner;
    uint8_t * fbs_buffer = nullptr;
    size_t fbs_buffer_size = 0;

    std::vector<float> cross_k_staging;
    std::vector<float> cross_v_staging;

    int mel_in_idx = -1;
    int embd_enc_out_idx = -1;
    int cross_k_out_idx = -1;
    int cross_v_out_idx = -1;
    size_t mel_in_expected_bytes = 0;
    size_t embd_enc_expected_bytes = 0;
    size_t cross_k_expected_bytes = 0;
    size_t cross_v_expected_bytes = 0;

    std::vector<flexmlrt::client::ErtTensorType> cached_input_tensors;
    std::vector<flexmlrt::client::ErtTensorType> cached_output_tensors;
};

// Return cached IO tensor descriptors by reference to avoid per-call deep copies.
static bool whisper_vitisai_get_cached_io_tensors(
        struct whisper_vitisai_context * ctx,
        std::vector<flexmlrt::client::ErtTensorType> *& input_tensors,
        std::vector<flexmlrt::client::ErtTensorType> *& output_tensors) {
    if (!ctx || !ctx->runner) {
        return false;
    }

    if (ctx->cached_input_tensors.empty() || ctx->cached_output_tensors.empty()) {
        ctx->cached_input_tensors  = ctx->runner->getIOTensors("input", false);
        ctx->cached_output_tensors = ctx->runner->getIOTensors("output", false);
    }

    input_tensors = &ctx->cached_input_tensors;
    output_tensors = &ctx->cached_output_tensors;
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

    const bool model_is_rai = ctx->model_path.find(".rai") != std::string::npos;

    // Check if model_path is rai file and if so, add fbs_buffer and fbs_buffer_size to the options
    if (model_is_rai) {
        if (whisper_vitisai_helpers::map_rai_file(ctx->model_path.c_str(), &ctx->fbs_buffer, &ctx->fbs_buffer_size)) {
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
        if (!ctx->runner || !ctx->runner->good()) {
            throw std::runtime_error("Runner creation ran into an error");
        }

        ctx->cached_input_tensors  = ctx->runner->getIOTensors("input", false);
        ctx->cached_output_tensors = ctx->runner->getIOTensors("output", false);

        auto & input_tensors = ctx->cached_input_tensors;
        auto & output_tensors = ctx->cached_output_tensors;

        whisper_vitisai_helpers::whisper_vitisai_io_binding binding;
        std::string binding_error;
        if (!whisper_vitisai_helpers::whisper_vitisai_resolve_io_binding(
                __func__, input_tensors, output_tensors, &binding, &binding_error)) {
            throw std::runtime_error(binding_error);
        }

        ctx->mel_in_idx              = binding.mel_in_idx;
        ctx->embd_enc_out_idx        = binding.embd_enc_out_idx;
        ctx->cross_k_out_idx         = binding.cross_k_out_idx;
        ctx->cross_v_out_idx         = binding.cross_v_out_idx;
        ctx->mel_in_expected_bytes   = binding.mel_in_expected_bytes;
        ctx->embd_enc_expected_bytes = binding.embd_enc_expected_bytes;
        ctx->cross_k_expected_bytes  = binding.cross_k_expected_bytes;
        ctx->cross_v_expected_bytes  = binding.cross_v_expected_bytes;

#if defined(WHISPER_DEBUG)
        {
            std::fprintf(stderr, "%s: model has %zu input tensor(s)\n", __func__, input_tensors.size());
            for (int i = 0; i < (int) input_tensors.size(); ++i) {
                const auto & meta = input_tensors[i].getMetadata();
                std::fprintf(stderr, "%s:   input[%d] name='%s' size=%zu shape=",
                        __func__, i, meta.name.c_str(), (size_t) meta.size);
                whisper_vitisai_helpers::whisper_vitisai_print_shape(meta.shape);
                std::fprintf(stderr, "\n");
            }

            std::fprintf(stderr, "%s: model has %zu output tensor(s)\n", __func__, output_tensors.size());
            for (int i = 0; i < (int) output_tensors.size(); ++i) {
                const auto & meta = output_tensors[i].getMetadata();
                std::fprintf(stderr, "%s:   output[%d] name='%s' size=%zu shape=",
                        __func__, i, meta.name.c_str(), (size_t) meta.size);
                whisper_vitisai_helpers::whisper_vitisai_print_shape(meta.shape);
                std::fprintf(stderr, "\n");
            }

            std::fprintf(stderr, "%s: input index: mel=%d\n", __func__, ctx->mel_in_idx);
            std::fprintf(stderr, "%s: output indices: embd_enc=%d cross_k=%d cross_v=%d\n",
                    __func__, ctx->embd_enc_out_idx, ctx->cross_k_out_idx, ctx->cross_v_out_idx);
        }
#endif
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: Exception during Vitis AI runner creation: %s\n", __func__, e.what());
        whisper_vitisai_free(ctx);
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
        whisper_vitisai_helpers::unmap_rai_file(ctx->fbs_buffer, ctx->fbs_buffer_size);
    }
    delete ctx;
}

static int whisper_vitisai_forward_impl(
        struct whisper_vitisai_context * ctx,
        struct ggml_tensor * mel,
        struct ggml_tensor * out,
        std::vector<flexmlrt::client::ErtTensorType> & input_tensors,
        std::vector<flexmlrt::client::ErtTensorType> & output_tensors,
        void * cross_k_data,
        void * cross_v_data) {
    if (!ctx || !mel || !out) {
        std::fprintf(stderr, "%s: ctx/mel/out must not be null\n", __func__);
        return 0;
    }

    const bool with_cross = (cross_k_data != nullptr || cross_v_data != nullptr);
    if (with_cross && (!cross_k_data || !cross_v_data)) {
        std::fprintf(stderr, "%s: cross_k_data/cross_v_data must both be set\n", __func__);
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

    if (ctx->embd_enc_out_idx < 0 || ctx->embd_enc_out_idx >= (int) output_tensors.size()) {
        std::fprintf(stderr, "%s: invalid embd_enc output index %d for %zu output tensor(s)\n",
                __func__, ctx->embd_enc_out_idx, output_tensors.size());
        return 0;
    }

    if (ctx->mel_in_idx < 0 || ctx->mel_in_idx >= (int) input_tensors.size()) {
        std::fprintf(stderr, "%s: invalid mel input index %d for %zu input tensor(s)\n",
                __func__, ctx->mel_in_idx, input_tensors.size());
        return 0;
    }

    if (!whisper_vitisai_helpers::whisper_vitisai_bind_tensor_data(
            "mel input",
            mel,
            { (size_t) mel->ne[1], (size_t) mel->ne[0] },
            input_tensors[ctx->mel_in_idx])) {
        return 0;
    }

    if (!whisper_vitisai_helpers::whisper_vitisai_bind_tensor_data(
            "embd_enc output",
            out,
            { (size_t) out->ne[1], (size_t) out->ne[0] },
            output_tensors[ctx->embd_enc_out_idx])) {
        return 0;
    }

    std::vector<bool> claimed_inputs(input_tensors.size(), false);
    claimed_inputs[ctx->mel_in_idx] = true;
    if (!whisper_vitisai_helpers::whisper_vitisai_all_tensors_claimed(
            __func__, "input", input_tensors, claimed_inputs)) {
        return 0;
    }

    std::vector<bool> claimed_outputs(output_tensors.size(), false);
    claimed_outputs[ctx->embd_enc_out_idx] = true;
    if (with_cross) {
        if (ctx->cross_k_out_idx < 0 || ctx->cross_k_out_idx >= (int) output_tensors.size() ||
                ctx->cross_v_out_idx < 0 || ctx->cross_v_out_idx >= (int) output_tensors.size()) {
            std::fprintf(stderr, "%s: invalid cross output indices cross_k=%d cross_v=%d for %zu output tensor(s)\n",
                    __func__, ctx->cross_k_out_idx, ctx->cross_v_out_idx, output_tensors.size());
            return 0;
        }
        output_tensors[ctx->cross_k_out_idx].data = cross_k_data;
        output_tensors[ctx->cross_v_out_idx].data = cross_v_data;
        claimed_outputs[ctx->cross_k_out_idx] = true;
        claimed_outputs[ctx->cross_v_out_idx] = true;
    }
    if (!whisper_vitisai_helpers::whisper_vitisai_all_tensors_claimed(
            __func__, "output", output_tensors, claimed_outputs)) {
        return 0;
    }

    auto clear_bound_data = [&]() {
        input_tensors[ctx->mel_in_idx].data = nullptr;
        output_tensors[ctx->embd_enc_out_idx].data = nullptr;
        if (with_cross) {
            output_tensors[ctx->cross_k_out_idx].data = nullptr;
            output_tensors[ctx->cross_v_out_idx].data = nullptr;
        }
    };

    try {
        ctx->runner->forward(input_tensors, output_tensors);
        clear_bound_data();
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: Vitis AI model inference %scompleted.\n",
                __func__, with_cross ? "(encoder + cross proj) " : "");
#endif
    } catch (const std::exception & e) {
        clear_bound_data();
        std::fprintf(stderr, "%s: Exception during model inference: %s\n", __func__, e.what());
        return 0;
    }

    return 1;
}

int whisper_vitisai_encode(struct whisper_vitisai_context * ctx, struct ggml_tensor * mel, struct ggml_tensor * out) {
    std::vector<flexmlrt::client::ErtTensorType> * input_tensors_cached = nullptr;
    std::vector<flexmlrt::client::ErtTensorType> * output_tensors_cached = nullptr;
    if (!whisper_vitisai_get_cached_io_tensors(ctx, input_tensors_cached, output_tensors_cached)) {
        std::fprintf(stderr, "%s: failed to acquire Vitis AI I/O tensors\n", __func__);
        return 0;
    }

    std::vector<flexmlrt::client::ErtTensorType> input_tensors = *input_tensors_cached;
    std::vector<flexmlrt::client::ErtTensorType> output_tensors = *output_tensors_cached;

    return whisper_vitisai_forward_impl(
            ctx,
            mel,
            out,
            input_tensors,
            output_tensors,
            nullptr,
            nullptr);
}

// Ensure persistent staging buffers are large enough for the given dimensions.
static void ensure_staging_buffers(
        struct whisper_vitisai_context * ctx,
        size_t count,
        bool need_k,
        bool need_v) {
    if (need_k && ctx->cross_k_staging.size() < count) {
        ctx->cross_k_staging.resize(count);
    }
    if (need_v && ctx->cross_v_staging.size() < count) {
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
        std::fprintf(stderr, "%s: ctx/mel/embd_enc/kv_cross_k/kv_cross_v must not be null\n", __func__);
        return 0;
    }

    if (n_text_layer <= 0 || n_ctx <= 0 || n_text_state <= 0 || n_text_head <= 0) {
        std::fprintf(stderr, "%s: invalid shape parameters layer=%d ctx=%d state=%d head=%d\n",
                __func__, n_text_layer, n_ctx, n_text_state, n_text_head);
        return 0;
    }

    if ((n_text_state % n_text_head) != 0) {
        std::fprintf(stderr, "%s: invalid head configuration state=%d head=%d\n",
                __func__, n_text_state, n_text_head);
        return 0;
    }

    if (kv_cross_k->type != kv_cross_v->type) {
        std::fprintf(stderr, "%s: kv_cross type mismatch k=%s v=%s\n",
                __func__,
                whisper_vitisai_helpers::whisper_kv_type_name(kv_cross_k->type),
                whisper_vitisai_helpers::whisper_kv_type_name(kv_cross_v->type));
        return 0;
    }

    const int n_state = n_text_state;
    const int n_state_head = n_state / n_text_head;
    const int n_ctx_pad = (n_ctx + 255) & ~255; // GGML_PAD(n_ctx, 256)

    const float Kscale = pow(float(n_state_head), -0.25f);
    const ggml_type kv_type = kv_cross_k->type;
    const bool kv_is_f32 = kv_type == GGML_TYPE_F32;
    const bool kv_is_f16 = kv_type == GGML_TYPE_F16;
    if (!kv_is_f32 && !kv_is_f16) {
        std::fprintf(stderr, "%s: unsupported kv_cross tensor type '%s'\n",
                __func__, whisper_vitisai_helpers::whisper_kv_type_name(kv_type));
        return 0;
    }

    const size_t elem_size = ggml_type_size(kv_type);
    const size_t req_layer_elems = (size_t)n_ctx * (size_t)n_state;

    std::vector<flexmlrt::client::ErtTensorType> * input_tensors_cached = nullptr;
    std::vector<flexmlrt::client::ErtTensorType> * output_tensors_cached = nullptr;
    if (!whisper_vitisai_get_cached_io_tensors(ctx, input_tensors_cached, output_tensors_cached)) {
        std::fprintf(stderr, "%s: failed to acquire Vitis AI I/O tensors\n", __func__);
        return 0;
    }
    std::vector<flexmlrt::client::ErtTensorType> input_tensors = *input_tensors_cached;
    std::vector<flexmlrt::client::ErtTensorType> output_tensors = *output_tensors_cached;

    if (ctx->cross_k_out_idx < 0 || ctx->cross_k_out_idx >= (int) output_tensors.size() ||
            ctx->cross_v_out_idx < 0 || ctx->cross_v_out_idx >= (int) output_tensors.size()) {
        std::fprintf(stderr, "%s: invalid cross output indices cross_k=%d cross_v=%d for %zu output tensor(s)\n",
                __func__, ctx->cross_k_out_idx, ctx->cross_v_out_idx, output_tensors.size());
        return 0;
    }

    const auto & cross_k_meta = output_tensors[ctx->cross_k_out_idx].getMetadata();
    const auto & cross_v_meta = output_tensors[ctx->cross_v_out_idx].getMetadata();
    if (!whisper_vitisai_helpers::whisper_validate_cross_shape("cross_k", cross_k_meta.shape, n_text_layer, n_ctx, n_state) ||
            !whisper_vitisai_helpers::whisper_validate_cross_shape("cross_v", cross_v_meta.shape, n_text_layer, n_ctx, n_state)) {
        return 0;
    }

    if (ctx->cross_k_expected_bytes == 0 || ctx->cross_v_expected_bytes == 0) {
        std::fprintf(stderr, "%s: missing cross output metadata sizes\n", __func__);
        return 0;
    }
    if (ctx->cross_k_expected_bytes != ctx->cross_v_expected_bytes) {
        std::fprintf(stderr, "%s: cross output metadata size mismatch k=%zu v=%zu\n",
                __func__, ctx->cross_k_expected_bytes, ctx->cross_v_expected_bytes);
        return 0;
    }
    const size_t expected_cross_bytes = (size_t) n_text_layer * req_layer_elems * sizeof(float);
    if (ctx->cross_k_expected_bytes != expected_cross_bytes) {
        std::fprintf(stderr,
                "%s: cross output size mismatch (model=%zu B, expected=%zu B for layer=%d ctx=%d state=%d)\n",
                __func__, ctx->cross_k_expected_bytes, expected_cross_bytes, n_text_layer, n_ctx, n_state);
        return 0;
    }

    const size_t model_total_elems = ctx->cross_k_expected_bytes / sizeof(float);
    const size_t model_layer_elems = req_layer_elems;

    const size_t required_kv_bytes = flash_attn
            ? (size_t)n_text_layer * elem_size * (size_t)n_state * (size_t)n_ctx_pad
            : (size_t)n_text_layer * elem_size * req_layer_elems;
    if (ggml_nbytes(kv_cross_k) < required_kv_bytes || ggml_nbytes(kv_cross_v) < required_kv_bytes) {
        std::fprintf(stderr,
                "%s: kv_cross buffers are too small (required=%zu B, k=%zu B, v=%zu B)\n",
                __func__, required_kv_bytes, ggml_nbytes(kv_cross_k), ggml_nbytes(kv_cross_v));
        return 0;
    }

    const bool direct_k_to_kv = kv_is_f32 && (!flash_attn || n_ctx_pad == n_ctx);
    const bool direct_v_to_kv = kv_is_f32 && flash_attn && (n_ctx_pad == n_ctx);
    const bool need_k_staging = !direct_k_to_kv;
    const bool need_v_staging = !direct_v_to_kv;

    if (need_k_staging || need_v_staging) {
        ensure_staging_buffers(ctx, model_total_elems, need_k_staging, need_v_staging);
    }

    void * cross_k_out = direct_k_to_kv
            ? kv_cross_k->data
            : (void *) ctx->cross_k_staging.data();
    void * cross_v_out = direct_v_to_kv
            ? kv_cross_v->data
            : (void *) ctx->cross_v_staging.data();

    whisper_vitisai_helpers::whisper_kv_cross_layout kv_layout;
    kv_layout.n_layer         = n_text_layer;
    kv_layout.n_ctx           = n_ctx;
    kv_layout.n_state         = n_state;
    kv_layout.src_layer_elems = model_layer_elems;
    kv_layout.layer_elems     = req_layer_elems;
    kv_layout.kscale          = Kscale;

    if (flash_attn) {
        WHISPER_DBG_TIMER(t_fwd_start);
        if (!whisper_vitisai_forward_impl(
                ctx, mel, embd_enc, input_tensors, output_tensors, cross_k_out, cross_v_out)) {
            return 0;
        }

        WHISPER_DBG_TIMER(t_fwd_end);
        WHISPER_DBG_TIMER(t_post_start);

        if (n_ctx_pad == n_ctx) {
            kv_layout.dst_layer_stride = req_layer_elems * elem_size;
            if (kv_is_f32) {
                // V was written straight into the kv cache by the runtime; only K needs scaling.
                whisper_vitisai_helpers::whisper_kv_cross_scale_k_f32(
                        (float *)kv_cross_k->data,
                        (size_t) n_text_layer * req_layer_elems,
                        Kscale);
            } else { // kv_is_f16
                whisper_vitisai_helpers::whisper_kv_cross_store_layers_f16(
                        ctx->cross_k_staging.data(),
                        ctx->cross_v_staging.data(),
                        (uint8_t *)kv_cross_k->data,
                        (uint8_t *)kv_cross_v->data,
                        kv_layout);
            }
        } else {
            // Runtime decoder uses padded K/V cache. Copy only requested context, leave the pad tail untouched.
            kv_layout.dst_layer_stride = elem_size * (size_t)n_state * (size_t)n_ctx_pad;
            if (kv_is_f32) {
                whisper_vitisai_helpers::whisper_kv_cross_store_layers_f32(
                        ctx->cross_k_staging.data(),
                        ctx->cross_v_staging.data(),
                        (uint8_t *)kv_cross_k->data,
                        (uint8_t *)kv_cross_v->data,
                        kv_layout);
            } else { // kv_is_f16
                whisper_vitisai_helpers::whisper_kv_cross_store_layers_f16(
                        ctx->cross_k_staging.data(),
                        ctx->cross_v_staging.data(),
                        (uint8_t *)kv_cross_k->data,
                        (uint8_t *)kv_cross_v->data,
                        kv_layout);
            }
        }

        WHISPER_DBG_TIMER(t_post_end);

#if defined(WHISPER_DEBUG)
        const size_t model_ctx = (size_t) n_ctx;
        std::fprintf(stderr, "%s: vitisai enc+cross forward time = %8.2f ms\n", __func__, (t_fwd_end - t_fwd_start) / 1000.0f);
        std::fprintf(stderr, "%s: kv_cross post-process time     = %8.2f ms (flash, req_ctx=%d, model_ctx=%zu, req_ctx_pad=%d, kv_type=%s)\n",
                __func__, (t_post_end - t_post_start) / 1000.0f, n_ctx, model_ctx, n_ctx_pad,
                whisper_vitisai_helpers::whisper_kv_type_name(kv_type));
#endif
    } else {
        // Non-flash: model outputs contiguous [ctx, state] per layer.
        WHISPER_DBG_TIMER(t_fwd_start);
        if (!whisper_vitisai_forward_impl(
                ctx, mel, embd_enc, input_tensors, output_tensors, cross_k_out, cross_v_out)) {
            return 0;
        }
        WHISPER_DBG_TIMER(t_fwd_end);
        WHISPER_DBG_TIMER(t_post_start);

        kv_layout.dst_layer_stride = elem_size * (size_t)n_state * (size_t)n_ctx;
        if (kv_is_f32) {
            // K was written straight into the kv cache by the runtime and is scaled there.
            whisper_vitisai_helpers::whisper_kv_cross_scale_k_f32(
                    (float *)kv_cross_k->data,
                    (size_t) n_text_layer * req_layer_elems,
                    Kscale);

            whisper_vitisai_helpers::whisper_kv_cross_transpose_v_layers_f32(
                    ctx->cross_v_staging.data(),
                    (uint8_t *)kv_cross_v->data,
                    kv_layout);
        } else { // kv_is_f16
            whisper_vitisai_helpers::whisper_kv_cross_store_k_transpose_v_layers_f16(
                    ctx->cross_k_staging.data(),
                    ctx->cross_v_staging.data(),
                    (uint8_t *)kv_cross_k->data,
                    (uint8_t *)kv_cross_v->data,
                    kv_layout);
        }

        WHISPER_DBG_TIMER(t_post_end);

#if defined(WHISPER_DEBUG)
        const size_t model_ctx = (size_t) n_ctx;
        std::fprintf(stderr, "%s: vitisai enc+cross forward time = %8.2f ms\n", __func__, (t_fwd_end - t_fwd_start) / 1000.0f);
        std::fprintf(stderr, "%s: kv_cross post-process time     = %8.2f ms (non-flash, req_ctx=%d, model_ctx=%zu, kv_type=%s)\n",
                __func__, (t_post_end - t_post_start) / 1000.0f, n_ctx, model_ctx,
                whisper_vitisai_helpers::whisper_kv_type_name(kv_type));
#endif
    }

    return 1;
}
