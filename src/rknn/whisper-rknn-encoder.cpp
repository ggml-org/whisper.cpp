#include "rknn/whisper-rknn-encoder.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <rknn_api.h>

#include <cmath>
#include <cstdio>
#include <cstring>

struct whisper_rknn_context {
    rknn_context handle = 0;
    rknn_tensor_attr input = {};
    rknn_tensor_attr output = {};
};

whisper_rknn_context * whisper_rknn_init(const char * path_model) {
    if (!path_model || !path_model[0]) {
        fprintf(stderr, "%s: model path is empty\n", __func__);
        return nullptr;
    }

    whisper_rknn_context * ctx = new whisper_rknn_context;
    // As in Model Zoo's rknpu2 whisper.cc, size=0 loads a file by path.
    int ret = rknn_init(&ctx->handle, const_cast<char *>(path_model), 0, 0, nullptr);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "%s: rknn_init failed: %d\n", __func__, ret);
        whisper_rknn_free(ctx);
        return nullptr;
    }

    rknn_input_output_num io = {};
    ret = rknn_query(ctx->handle, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret != RKNN_SUCC || io.n_input != 1 || io.n_output != 1) {
        fprintf(stderr, "%s: expected one input/output: ret=%d, inputs=%u, outputs=%u\n",
                __func__, ret, io.n_input, io.n_output);
        whisper_rknn_free(ctx);
        return nullptr;
    }

    ret = rknn_query(ctx->handle, RKNN_QUERY_INPUT_ATTR, &ctx->input, sizeof(ctx->input));
    if (ret == RKNN_SUCC) {
        ret = rknn_query(ctx->handle, RKNN_QUERY_OUTPUT_ATTR, &ctx->output, sizeof(ctx->output));
    }
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "%s: rknn_query tensor attributes failed: %d\n", __func__, ret);
        whisper_rknn_free(ctx);
        return nullptr;
    }

    const auto & in = ctx->input;
    const auto & out = ctx->output;
    if (in.n_dims != 3 || in.dims[0] != 1 || in.dims[1] != 80 || in.dims[2] != 3000 ||
        out.n_dims != 3 || out.dims[0] != 1 || out.dims[1] != 1500 || out.dims[2] != 512 ||
        in.n_elems != 80*3000 || out.n_elems != 1500*512 ||
        (in.type != RKNN_TENSOR_FLOAT16 && in.type != RKNN_TENSOR_FLOAT32) ||
        (out.type != RKNN_TENSOR_FLOAT16 && out.type != RKNN_TENSOR_FLOAT32) ||
        (in.fmt != RKNN_TENSOR_NCHW && in.fmt != RKNN_TENSOR_UNDEFINED) ||
        (out.fmt != RKNN_TENSOR_NCHW && out.fmt != RKNN_TENSOR_UNDEFINED)) {
        fprintf(stderr, "%s: expected base.en [1,80,3000] -> [1,1500,512] float tensors\n", __func__);
        whisper_rknn_free(ctx);
        return nullptr;
    }

    fprintf(stderr, "%s: input [1,80,3000] %s %s, output [1,1500,512] %s %s\n",
            __func__, get_type_string(in.type), get_format_string(in.fmt),
            get_type_string(out.type), get_format_string(out.fmt));
    return ctx;
}

void whisper_rknn_free(whisper_rknn_context * ctx) {
    if (ctx) {
        if (ctx->handle) {
            rknn_destroy(ctx->handle);
        }
        delete ctx;
    }
}

int whisper_rknn_encode(whisper_rknn_context * ctx, ggml_tensor * mel, ggml_tensor * out) {
    const int64_t t_start_us = ggml_time_us();
    if (!ctx || !mel || !out || mel->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        ggml_n_dims(mel) != 2 || ggml_n_dims(out) != 2 ||
        mel->ne[0] != 3000 || mel->ne[1] != 80 || out->ne[0] != 512 || out->ne[1] != 1500 ||
        !ggml_is_contiguous(mel) || !ggml_is_contiguous(out) ||
        !mel->data || !out->data || !mel->buffer || !out->buffer ||
        !ggml_backend_buffer_is_host(mel->buffer) || !ggml_backend_buffer_is_host(out->buffer)) {
        fprintf(stderr, "%s: expected contiguous host float32 mel [3000,80], output [512,1500]\n", __func__);
        return 0;
    }

    // ggml's time axis is contiguous: already RKNN [batch,mel,time].
    // pass_through=0 lets librknnrt convert float32 mel to the model's float16.
    rknn_input input = {};
    input.type = RKNN_TENSOR_FLOAT32;
    input.fmt = RKNN_TENSOR_NCHW;
    input.size = ggml_nbytes(mel);
    input.buf = mel->data;
    int ret = rknn_inputs_set(ctx->handle, 1, &input);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "%s: rknn_inputs_set failed: %d\n", __func__, ret);
        return 0;
    }

    const int64_t t_run_us = ggml_time_us();
    ret = rknn_run(ctx->handle, nullptr);
    const int64_t t_run_end_us = ggml_time_us();
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "%s: rknn_run failed: %d\n", __func__, ret);
        return 0;
    }

    rknn_output output = {};
    output.want_float = 1;
    ret = rknn_outputs_get(ctx->handle, 1, &output, nullptr);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "%s: rknn_outputs_get failed: %d\n", __func__, ret);
        return 0;
    }

    // RKNN [1,time,state] and ggml [state,time] have the same linear order.
    bool valid = output.buf && output.size == ggml_nbytes(out);
    if (valid) {
        const float * data = static_cast<const float *>(output.buf);
        for (int64_t i = 0; i < ggml_nelements(out); ++i) {
            if (!std::isfinite(data[i])) {
                valid = false;
                break;
            }
        }
        if (valid) {
            memcpy(out->data, output.buf, ggml_nbytes(out));
        }
    }
    ret = rknn_outputs_release(ctx->handle, 1, &output);
    if (!valid || ret != RKNN_SUCC) {
        fprintf(stderr, "%s: invalid output or rknn_outputs_release failed: %d\n", __func__, ret);
        return 0;
    }

    fprintf(stderr, "%s: encoder %.3f ms (rknn_run %.3f ms), output [1,1500,512] float32, all finite\n",
            __func__, (ggml_time_us() - t_start_us)/1000.0, (t_run_end_us - t_run_us)/1000.0);
    return 1;
}
