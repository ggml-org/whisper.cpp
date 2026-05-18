#include "supertonic_internal.h"
#include "npy.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

static const supertonic_trace_tensor * find_trace(
    const std::vector<supertonic_trace_tensor> & trace,
    const std::string & name) {
    for (const auto & t : trace) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf REF_DIR\n", argv[0]);
        return 2;
    }

    supertonic_model model;
    if (!load_supertonic_gguf(argv[1], model)) return 1;

    int rc = 0;
    try {
        npy_array noise = npy_load(std::string(argv[2]) + "/noise.npy");
        npy_array text_emb = npy_load(std::string(argv[2]) + "/text_emb.npy");
        npy_array style_ttl = npy_load(std::string(argv[2]) + "/style_ttl.npy");
        npy_array latent_mask = npy_load(std::string(argv[2]) + "/latent_mask.npy");
        if (noise.dtype != "<f4" || noise.shape.size() != 3 || noise.shape[0] != 1) {
            throw std::runtime_error("bad noise.npy");
        }
        if (text_emb.dtype != "<f4" || text_emb.shape.size() != 3 || text_emb.shape[0] != 1) {
            throw std::runtime_error("bad text_emb.npy");
        }
        if (style_ttl.dtype != "<f4" || style_ttl.n_elements() != 50*256) {
            throw std::runtime_error("bad style_ttl.npy");
        }
        if (latent_mask.dtype != "<f4" || latent_mask.n_elements() != (size_t) noise.shape[2]) {
            throw std::runtime_error("bad latent_mask.npy");
        }

        std::string error;
        std::vector<supertonic_trace_tensor> scalar;
        std::vector<supertonic_trace_tensor> ggml;
        if (!supertonic_vector_trace_proj_ggml(model, npy_as_f32(noise), npy_as_f32(text_emb),
                                               (int) text_emb.shape[2], npy_as_f32(style_ttl),
                                               npy_as_f32(latent_mask),
                                               (int) noise.shape[2], 0, 5, scalar, ggml, &error)) {
            throw std::runtime_error("vector proj trace failed: " + error);
        }

        for (const auto & s : scalar) {
            const auto * g = find_trace(ggml, s.name);
            if (!g) {
                fprintf(stderr, "missing ggml trace tensor: %s\n", s.name.c_str());
                rc = 1;
                continue;
            }
            if (g->shape != s.shape || g->data.size() != s.data.size()) {
                fprintf(stderr, "[%s] shape/size mismatch\n", s.name.c_str());
                rc = 1;
                continue;
            }
            compare_stats st = compare_f32(g->data.data(), s.data.data(), s.data.size());
            print_compare(("vector_trace_" + s.name).c_str(), st);
            if (st.max_abs_err > 2e-3) rc = 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    free_supertonic_model(model);
    return rc;
}
