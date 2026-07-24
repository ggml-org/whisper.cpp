// T5 encoder parity vs HF fixtures (case*_desc_ids -> case*_cross_states).

#include "parler/internal.h"
#include "npy.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

// PARLER_TEST_REPORT_ONLY=1: print metrics without enforcing the tolerance
// bars (for measuring quantized GGUFs). Non-finite output still fails.
static bool g_report_only = false;

static bool run_case(parler_model & model, const std::string & ref_dir, const char * name) {
    npy_array ids_npy = npy_load(ref_dir + "/" + name + "_desc_ids.npy");
    if (ids_npy.dtype != "<i8") {
        fprintf(stderr, "%s: unexpected desc_ids dtype %s\n", name, ids_npy.dtype.c_str());
        return false;
    }
    const int64_t * ids64 = reinterpret_cast<const int64_t *>(ids_npy.data.data());
    std::vector<int32_t> ids(ids_npy.n_elements());
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) ids64[i];

    std::vector<float> states;
    if (!parler_encode_description(model, ids, /*n_threads=*/ 4, &states)) {
        fprintf(stderr, "%s: encode failed\n", name);
        return false;
    }

    npy_array ref = npy_load(ref_dir + "/" + name + "_cross_states.npy"); // [1, T, d]
    if (ref.n_elements() != states.size()) {
        fprintf(stderr, "%s: size mismatch got %zu ref %zu\n", name,
                states.size(), ref.n_elements());
        return false;
    }
    // numpy [1, T, d] C-order flat index == ggml [d, T] flat index
    compare_stats s = compare_f32(states.data(), npy_as_f32(ref), states.size());
    print_compare(name, s);
    // NaN guard: comparisons against NaN are false, so a NaN output would
    // otherwise sail under the max_abs bar with max_abs == 0.
    if (!std::isfinite(s.mean_abs_err)) {
        fprintf(stderr, "%s: FAIL non-finite output\n", name);
        return false;
    }
    if (s.max_abs_err > 2e-3 || s.rel_err > 2e-3) {
        if (g_report_only) {
            fprintf(stderr, "%s: REPORT over-tolerance (max_abs %.3e rel %.3e)\n",
                    name, s.max_abs_err, s.rel_err);
            return true;
        }
        fprintf(stderr, "%s: FAIL tolerance (max_abs %.3e rel %.3e)\n",
                name, s.max_abs_err, s.rel_err);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf REF_DIR\n", argv[0]);
        return 2;
    }
    g_report_only = getenv("PARLER_TEST_REPORT_ONLY") != nullptr;
    if (g_report_only) fprintf(stderr, "REPORT-ONLY MODE: tolerance bars not enforced\n");
    parler_model model;
    std::string err;
    if (!parler_load_gguf(argv[1], model, 0, &err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    int rc = 0;
    if (!run_case(model, argv[2], "case0")) rc = 1;
    if (!run_case(model, argv[2], "case1")) rc = 1;
    parler_free_model(model);
    if (rc == 0) fprintf(stderr, g_report_only ? "parler t5: REPORT DONE\n" : "parler t5: PASS\n");
    return rc;
}
