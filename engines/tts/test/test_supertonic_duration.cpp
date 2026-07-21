#include "supertonic_internal.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <string>

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
        npy_array style_dp = npy_load(std::string(argv[2]) + "/style_dp.npy");
        npy_array duration = npy_load(std::string(argv[2]) + "/duration_raw.npy");
        if (text_ids.dtype != "<i8" || text_ids.shape.size() != 2 || text_ids.shape[0] != 1) {
            throw std::runtime_error("unexpected text_ids.npy");
        }
        if (style_dp.dtype != "<f4" || style_dp.n_elements() != 128) {
            throw std::runtime_error("unexpected style_dp.npy");
        }
        if (duration.dtype != "<f4" || duration.n_elements() != 1) {
            throw std::runtime_error("unexpected duration_raw.npy");
        }

        float got = 0.0f;
        std::string error;
        if (!supertonic_duration_forward_cpu(
                model,
                reinterpret_cast<const int64_t *>(text_ids.data.data()),
                (int) text_ids.shape[1],
                npy_as_f32(style_dp),
                got,
                &error)) {
            throw std::runtime_error("duration failed: " + error);
        }
        const float ref = npy_as_f32(duration)[0];
        const float abs_err = std::fabs(got - ref);
        fprintf(stderr, "supertonic duration: got=%.8f ref=%.8f abs=%.3e\n", got, ref, abs_err);
        if (abs_err > 2e-4f) {
            fprintf(stderr, "supertonic duration parity FAILED\n");
            rc = 1;
        } else {
            fprintf(stderr, "supertonic duration parity PASS\n");
        }

        float got_ggml = 0.0f;
        if (!supertonic_duration_forward_ggml(
                model,
                reinterpret_cast<const int64_t *>(text_ids.data.data()),
                (int) text_ids.shape[1],
                npy_as_f32(style_dp),
                got_ggml,
                &error)) {
            throw std::runtime_error("duration ggml failed: " + error);
        }
        const float abs_err_ggml = std::fabs(got_ggml - ref);
        fprintf(stderr, "supertonic duration ggml: got=%.8f ref=%.8f abs=%.3e\n", got_ggml, ref, abs_err_ggml);
        if (abs_err_ggml > 2e-4f) {
            fprintf(stderr, "supertonic duration GGML parity FAILED\n");
            rc = 1;
        } else {
            fprintf(stderr, "supertonic duration GGML parity PASS\n");
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    free_supertonic_model(model);
    return rc;
}
