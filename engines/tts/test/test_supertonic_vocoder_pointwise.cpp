#include "supertonic_internal.h"
#include "npy.h"
#include "ggml-alloc.h"

#if defined(TTS_CPP_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(TTS_CPP_USE_CBLAS)
#include <cblas.h>
#endif

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

struct f32_tensor {
    std::vector<float> data;
    int64_t ne[4] = {1, 1, 1, 1};
};

f32_tensor read_tensor(ggml_tensor * t) {
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

const supertonic_trace_tensor & require_trace(
    const std::vector<supertonic_trace_tensor> & trace,
    const std::string & name) {
    for (const auto & t : trace) {
        if (t.name == name) return t;
    }
    throw std::runtime_error("missing trace tensor: " + name);
}

void scalar_linear_oc_ic(const std::vector<float> & x, int L, int IC,
                         const f32_tensor & w, const f32_tensor * b,
                         int OC, std::vector<float> & y) {
    y.assign((size_t)L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[(size_t)oc] : 0.0f;
            const size_t woff = (size_t)oc * IC;
            for (int ic = 0; ic < IC; ++ic) {
                sum += w.data[woff + ic] * x[(size_t)t * IC + ic];
            }
            y[(size_t)t * OC + oc] = sum;
        }
    }
}

void scalar_linear_ic_oc(const std::vector<float> & x, int L, int IC,
                         const f32_tensor & w, const f32_tensor * b,
                         int OC, std::vector<float> & y) {
    y.assign((size_t)L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[(size_t)oc] : 0.0f;
            for (int ic = 0; ic < IC; ++ic) {
                sum += w.data[(size_t)ic * OC + oc] * x[(size_t)t * IC + ic];
            }
            y[(size_t)t * OC + oc] = sum;
        }
    }
}

void pack_lc_to_col_major(const std::vector<float> & x, int L, int C, std::vector<float> & out) {
    out.assign((size_t)L * C, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            out[(size_t)c * L + t] = x[(size_t)t * C + c];
        }
    }
}

void unpack_col_major_to_lc(const std::vector<float> & x, int L, int C, std::vector<float> & out) {
    out.assign((size_t)L * C, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            out[(size_t)t * C + c] = x[(size_t)c * L + t];
        }
    }
}

#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
ggml_tensor * repeat_like_harness(ggml_context * ctx, ggml_tensor * v, ggml_tensor * like) {
    if (ggml_n_dims(v) == 1 && ggml_n_dims(like) >= 2) {
        if (like->ne[0] == v->ne[0]) v = ggml_reshape_2d(ctx, v, v->ne[0], 1);
        else if (like->ne[1] == v->ne[0]) v = ggml_reshape_2d(ctx, v, 1, v->ne[1]);
    }
    if (!ggml_can_repeat(v, like)) throw std::runtime_error("mini graph repeat_like failed");
    return ggml_repeat(ctx, v, like);
}

void graph_pointwise_op(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) return;
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * weight = dst->src[1];
    const ggml_tensor * bias = dst->src[2];
    const int L = (int)src->ne[0];
    const int IC = (int)src->ne[1];
    const int OC = (int)weight->ne[2];
    const float * src_data = static_cast<const float *>(src->data);
    const float * weight_data = static_cast<const float *>(weight->data);
    float * dst_data = static_cast<float *>(dst->data);
    const int lda = (int)(src->nb[1] / sizeof(float));
    const int ldb = (int)(weight->nb[2] / sizeof(float));
    const int ldc = (int)(dst->nb[1] / sizeof(float));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                L, OC, IC,
                1.0f,
                src_data, lda,
                weight_data, ldb,
                0.0f,
                dst_data, ldc);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    if (bias) {
        const auto * bias_base = static_cast<const uint8_t *>(bias->data);
        for (int oc = 0; oc < OC; ++oc) {
            const float bv = *reinterpret_cast<const float *>(bias_base + (size_t)oc * bias->nb[0]);
            float * col = dst_data + (size_t)oc * ldc;
            for (int t = 0; t < L; ++t) col[t] += bv;
        }
    }
}

ggml_tensor * graph_pointwise_node(ggml_context * ctx,
                                   ggml_tensor * x,
                                   ggml_tensor * w,
                                   ggml_tensor * b,
                                   int OC) {
    ggml_tensor * args_with_bias[] = { x, w, b };
    ggml_tensor * args_no_bias[] = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], OC, x->ne[2], x->ne[3],
                          b ? args_with_bias : args_no_bias,
                          b ? 3 : 2,
                          graph_pointwise_op,
                          1,
                          nullptr);
}

std::vector<float> ggml_custom_graph_candidate(const supertonic_model & model,
                                               const std::vector<float> & x_lc,
                                               int L,
                                               int IC,
                                               ggml_tensor * w,
                                               ggml_tensor * b,
                                               int OC) {
    constexpr int NODES = 64;
    std::vector<uint8_t> buf(ggml_tensor_overhead() * NODES + ggml_graph_overhead_custom(NODES, false), 0);
    ggml_init_params p = { buf.size(), buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, NODES, false);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, IC);
    ggml_set_name(x, "x"); ggml_set_input(x);
    ggml_tensor * y = graph_pointwise_node(ctx, x, w, b, OC);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) throw std::runtime_error("ggml_gallocr_new failed");
    if (!ggml_gallocr_reserve(allocr, gf)) {
        ggml_gallocr_free(allocr);
        throw std::runtime_error("ggml_gallocr_reserve failed");
    }
    ggml_gallocr_alloc_graph(allocr, gf);

    std::vector<float> x_col;
    pack_lc_to_col_major(x_lc, L, IC, x_col);
    ggml_backend_tensor_set(x, x_col.data(), 0, x_col.size() * sizeof(float));
    supertonic_graph_compute(model, gf);
    std::vector<float> y_col((size_t)L * OC);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "y"), y_col.data(), 0, y_col.size() * sizeof(float));
    std::vector<float> y_lc;
    unpack_col_major_to_lc(y_col, L, OC, y_lc);

    ggml_gallocr_free(allocr);
    return y_lc;
}

struct mini_graph_outputs {
    std::vector<float> pw1;
    std::vector<float> gelu;
    std::vector<float> pw2;
    std::vector<float> out;
};

mini_graph_outputs ggml_block0_mini_graph_candidate(const supertonic_model & model,
                                                    const std::vector<float> & norm_lc,
                                                    const std::vector<float> & residual_lc,
                                                    int L,
                                                    int C,
                                                    int H) {
    constexpr int NODES = 256;
    std::vector<uint8_t> buf(ggml_tensor_overhead() * NODES + ggml_graph_overhead_custom(NODES, false), 0);
    ggml_init_params p = { buf.size(), buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, NODES, false);

    ggml_tensor * norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(norm, "mini_norm"); ggml_set_input(norm);
    ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C);
    ggml_set_name(residual, "mini_residual"); ggml_set_input(residual);

    const auto & cw = model.vocoder.convnext[0];
    ggml_tensor * norm_copy = ggml_cont(ctx, norm);
    ggml_set_name(norm_copy, "mini_norm_copy"); ggml_set_output(norm_copy); ggml_build_forward_expand(gf, norm_copy);
    ggml_tensor * pw1 = graph_pointwise_node(ctx, norm_copy, cw.pw1_w, cw.pw1_b, H);
    ggml_set_name(pw1, "mini_pw1"); ggml_set_output(pw1); ggml_build_forward_expand(gf, pw1);
    ggml_tensor * pw1_copy = ggml_cont(ctx, pw1);
    ggml_set_name(pw1_copy, "mini_pw1_copy"); ggml_build_forward_expand(gf, pw1_copy);
    ggml_tensor * gelu = ggml_gelu_erf(ctx, pw1_copy);
    ggml_set_name(gelu, "mini_gelu"); ggml_set_output(gelu); ggml_build_forward_expand(gf, gelu);
    ggml_tensor * gelu_copy = ggml_cont(ctx, gelu);
    ggml_set_name(gelu_copy, "mini_gelu_copy"); ggml_build_forward_expand(gf, gelu_copy);
    ggml_tensor * pw2 = graph_pointwise_node(ctx, gelu_copy, cw.pw2_w, cw.pw2_b, C);
    ggml_set_name(pw2, "mini_pw2"); ggml_set_output(pw2); ggml_build_forward_expand(gf, pw2);
    ggml_tensor * pw2_copy = ggml_cont(ctx, pw2);
    ggml_set_name(pw2_copy, "mini_pw2_copy"); ggml_build_forward_expand(gf, pw2_copy);
    ggml_tensor * scaled = ggml_mul(ctx, pw2_copy, repeat_like_harness(ctx, cw.gamma, pw2_copy));
    ggml_tensor * out = ggml_add(ctx, residual, scaled);
    ggml_set_name(out, "mini_out"); ggml_set_output(out); ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) throw std::runtime_error("ggml_gallocr_new mini failed");
    if (!ggml_gallocr_reserve(allocr, gf)) {
        ggml_gallocr_free(allocr);
        throw std::runtime_error("ggml_gallocr_reserve mini failed");
    }
    ggml_gallocr_alloc_graph(allocr, gf);

    std::vector<float> norm_col, residual_col;
    pack_lc_to_col_major(norm_lc, L, C, norm_col);
    pack_lc_to_col_major(residual_lc, L, C, residual_col);
    ggml_backend_tensor_set(norm, norm_col.data(), 0, norm_col.size() * sizeof(float));
    ggml_backend_tensor_set(residual, residual_col.data(), 0, residual_col.size() * sizeof(float));
    supertonic_graph_compute(model, gf);

    auto read_lc = [&](const char * name, int channels) {
        std::vector<float> col((size_t)L * channels);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, name), col.data(), 0, col.size() * sizeof(float));
        std::vector<float> lc;
        unpack_col_major_to_lc(col, L, channels, lc);
        return lc;
    };

    mini_graph_outputs outputs;
    outputs.pw1 = read_lc("mini_pw1", H);
    outputs.gelu = read_lc("mini_gelu", H);
    outputs.pw2 = read_lc("mini_pw2", C);
    outputs.out = read_lc("mini_out", C);
    ggml_gallocr_free(allocr);
    return outputs;
}

void blas_col_nn(const std::vector<float> & x_lc, int L, int IC,
                 const f32_tensor & w, const f32_tensor * b,
                 int OC, std::vector<float> & y_lc) {
    std::vector<float> x_col;
    std::vector<float> y_col((size_t)L * OC, 0.0f);
    pack_lc_to_col_major(x_lc, L, IC, x_col);
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                L, OC, IC,
                1.0f,
                x_col.data(), L,
                w.data.data(), IC,
                0.0f,
                y_col.data(), L);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    if (b) {
        for (int oc = 0; oc < OC; ++oc) {
            float * col = y_col.data() + (size_t)oc * L;
            const float bias = b->data[(size_t)oc];
            for (int t = 0; t < L; ++t) col[t] += bias;
        }
    }
    unpack_col_major_to_lc(y_col, L, OC, y_lc);
}

void blas_col_nt(const std::vector<float> & x_lc, int L, int IC,
                 const f32_tensor & w, const f32_tensor * b,
                 int OC, std::vector<float> & y_lc) {
    std::vector<float> x_col;
    std::vector<float> y_col((size_t)L * OC, 0.0f);
    pack_lc_to_col_major(x_lc, L, IC, x_col);
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                L, OC, IC,
                1.0f,
                x_col.data(), L,
                w.data.data(), OC,
                0.0f,
                y_col.data(), L);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    if (b) {
        for (int oc = 0; oc < OC; ++oc) {
            float * col = y_col.data() + (size_t)oc * L;
            const float bias = b->data[(size_t)oc];
            for (int t = 0; t < L; ++t) col[t] += bias;
        }
    }
    unpack_col_major_to_lc(y_col, L, OC, y_lc);
}
#endif

void print_candidate(const char * name,
                     const std::vector<float> & got,
                     const std::vector<float> & expected) {
    compare_stats st = compare_f32(got.data(), expected.data(), got.size());
    print_compare(name, st);
}

void test_linear_family(const char * prefix,
                        const supertonic_model & model,
                        const std::vector<float> & x,
                        int L,
                        int IC,
                        ggml_tensor * w_tensor,
                        ggml_tensor * b_tensor,
                        int OC,
                        const std::vector<float> & expected) {
    f32_tensor w = read_tensor(w_tensor);
    f32_tensor b;
    f32_tensor * bp = nullptr;
    if (b_tensor) {
        b = read_tensor(b_tensor);
        bp = &b;
    }

    std::fprintf(stderr, "\n%s shapes: L=%d IC=%d OC=%d w_ne=[%lld,%lld,%lld,%lld]\n",
                 prefix, L, IC, OC,
                 (long long)w.ne[0], (long long)w.ne[1], (long long)w.ne[2], (long long)w.ne[3]);

    std::vector<float> y;
    scalar_linear_oc_ic(x, L, IC, w, bp, OC, y);
    print_candidate((std::string(prefix) + "_scalar_oc_ic").c_str(), y, expected);

    scalar_linear_ic_oc(x, L, IC, w, bp, OC, y);
    print_candidate((std::string(prefix) + "_scalar_ic_oc").c_str(), y, expected);

#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
    blas_col_nn(x, L, IC, w, bp, OC, y);
    print_candidate((std::string(prefix) + "_blas_col_nn").c_str(), y, expected);

    blas_col_nt(x, L, IC, w, bp, OC, y);
    print_candidate((std::string(prefix) + "_blas_col_nt").c_str(), y, expected);

    y = ggml_custom_graph_candidate(model, x, L, IC, w_tensor, b_tensor, OC);
    print_candidate((std::string(prefix) + "_ggml_custom_graph").c_str(), y, expected);
#else
    std::fprintf(stderr, "  [%s] BLAS candidates skipped: TTS_CPP_USE_ACCELERATE/TTS_CPP_USE_CBLAS not enabled\n", prefix);
#endif
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf REF_DIR\n", argv[0]);
        return 2;
    }

    supertonic_model model;
    if (!load_supertonic_gguf(argv[1], model)) return 1;

    int rc = 0;
    try {
        npy_array latent_npy = npy_load(std::string(argv[2]) + "/final_latent.npy");
        if (latent_npy.dtype != "<f4" || latent_npy.shape.size() != 3 || latent_npy.shape[0] != 1) {
            throw std::runtime_error("unexpected final_latent.npy shape/dtype");
        }
        const int latent_len = (int)latent_npy.shape[2];

        std::vector<supertonic_trace_tensor> scalar;
        std::string error;
        if (!supertonic_vocoder_trace_scalar(model, npy_as_f32(latent_npy), latent_len, scalar, &error)) {
            throw std::runtime_error("scalar trace failed: " + error);
        }

        const auto & embed = require_trace(scalar, "embed");
        const auto & block0_norm = require_trace(scalar, "block0_norm");
        const auto & block0_pw1 = require_trace(scalar, "block0_pw1");
        const auto & block0_gelu = require_trace(scalar, "block0_gelu");
        const auto & block0_pw2 = require_trace(scalar, "block0_pw2");
        const auto & block0_out = require_trace(scalar, "block0_out");
        const auto & final_norm = require_trace(scalar, "final_norm");
        const auto & head1 = require_trace(scalar, "head1");
        const auto & prelu = require_trace(scalar, "prelu");
        const auto & wav = require_trace(scalar, "wav");

        const int L = (int)block0_norm.shape[0];
        const int C = (int)block0_norm.shape[1];
        const int H = (int)block0_pw1.shape[1];

        test_linear_family("block0_pw1", model,
                           block0_norm.data, L, C,
                           model.vocoder.convnext[0].pw1_w,
                           model.vocoder.convnext[0].pw1_b,
                           H, block0_pw1.data);
        test_linear_family("block0_pw2", model,
                           block0_gelu.data, L, H,
                           model.vocoder.convnext[0].pw2_w,
                           model.vocoder.convnext[0].pw2_b,
                           C, block0_pw2.data);
        test_linear_family("head1", model,
                           final_norm.data, L, C,
                           model.vocoder.head1_w,
                           model.vocoder.head1_b,
                           (int)head1.shape[1], head1.data);
        test_linear_family("head2", model,
                           prelu.data, L, (int)prelu.shape[1],
                           model.vocoder.head2_w,
                           nullptr,
                           (int)wav.shape[1], wav.data);

#if defined(TTS_CPP_USE_ACCELERATE) || defined(TTS_CPP_USE_CBLAS)
        std::fprintf(stderr, "\nblock0 mini graph candidate\n");
        mini_graph_outputs mini = ggml_block0_mini_graph_candidate(
            model, block0_norm.data, embed.data, L, C, H);
        print_candidate("mini_block0_pw1", mini.pw1, block0_pw1.data);
        print_candidate("mini_block0_gelu", mini.gelu, block0_gelu.data);
        print_candidate("mini_block0_pw2", mini.pw2, block0_pw2.data);
        print_candidate("mini_block0_out", mini.out, block0_out.data);
#endif
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    free_supertonic_model(model);
    return rc;
}
