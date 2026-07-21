#include "supertonic_internal.h"
#include "npy.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

static const supertonic_trace_tensor * find_trace(
    const std::vector<supertonic_trace_tensor> & trace,
    const std::string & name) {
    for (const auto & t : trace) if (t.name == name) return &t;
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
        npy_array text_ids = npy_load(std::string(argv[2]) + "/text_ids.npy");
        if (text_ids.dtype != "<i8" || text_ids.shape.size() != 2 || text_ids.shape[0] != 1) {
            throw std::runtime_error("unexpected text_ids.npy");
        }
        std::vector<supertonic_trace_tensor> scalar, ggml;
        std::string error;
        if (!supertonic_text_encoder_trace_ggml(model,
                                                reinterpret_cast<const int64_t *>(text_ids.data.data()),
                                                (int) text_ids.shape[1],
                                                scalar, ggml, &error)) {
            throw std::runtime_error("text encoder trace failed: " + error);
        }
        for (const auto & s : scalar) {
            const auto * g = find_trace(ggml, s.name);
            if (!g) { fprintf(stderr, "missing ggml trace tensor: %s\n", s.name.c_str()); rc = 1; continue; }
            if (g->shape != s.shape || g->data.size() != s.data.size()) {
                fprintf(stderr, "[%s] shape/size mismatch\n", s.name.c_str());
                rc = 1;
                continue;
            }
            compare_stats st = compare_f32(g->data.data(), s.data.data(), s.data.size());
            print_compare(("text_encoder_trace_" + s.name).c_str(), st);
            if (st.max_abs_err > 2e-3) rc = 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }
    free_supertonic_model(model);
    return rc;
}
