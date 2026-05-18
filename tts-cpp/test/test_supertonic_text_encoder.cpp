#include "supertonic_internal.h"
#include "npy.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

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
        npy_array style_ttl = npy_load(std::string(argv[2]) + "/style_ttl.npy");
        npy_array ref = npy_load(std::string(argv[2]) + "/text_emb.npy");
        if (text_ids.dtype != "<i8" || text_ids.shape.size() != 2 || text_ids.shape[0] != 1) {
            throw std::runtime_error("unexpected text_ids.npy");
        }
        if (style_ttl.dtype != "<f4" || style_ttl.n_elements() != 50 * 256) {
            throw std::runtime_error("unexpected style_ttl.npy");
        }
        if (ref.dtype != "<f4" || ref.shape.size() != 3 || ref.shape[0] != 1 || ref.shape[1] != 256) {
            throw std::runtime_error("unexpected text_emb.npy");
        }

        std::vector<float> got;
        std::string error;
        if (!supertonic_text_encoder_forward_ggml(
                model,
                reinterpret_cast<const int64_t *>(text_ids.data.data()),
                (int) text_ids.shape[1],
                npy_as_f32(style_ttl),
                got,
                &error)) {
            throw std::runtime_error("text encoder failed: " + error);
        }
        if (got.size() != ref.n_elements()) {
            fprintf(stderr, "text_emb size mismatch: got %zu ref %zu\n", got.size(), ref.n_elements());
            rc = 1;
        } else {
            compare_stats s = compare_f32(got.data(), npy_as_f32(ref), got.size());
            print_compare("supertonic_text_encoder", s);
            if (s.max_abs_err > 2e-3) {
                fprintf(stderr, "supertonic text encoder parity FAILED\n");
                rc = 1;
            } else {
                fprintf(stderr, "supertonic text encoder parity PASS\n");
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    free_supertonic_model(model);
    return rc;
}
