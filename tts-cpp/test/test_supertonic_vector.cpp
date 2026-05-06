#include "supertonic_internal.h"
#include "npy.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.gguf REF_DIR\n", argv[0]); return 2; }
    supertonic_model model;
    if (!load_supertonic_gguf(argv[1], model)) return 1;
    int rc = 0;
    try {
        npy_array noise = npy_load(std::string(argv[2]) + "/noise.npy");
        npy_array text_emb = npy_load(std::string(argv[2]) + "/text_emb.npy");
        npy_array style_ttl = npy_load(std::string(argv[2]) + "/style_ttl.npy");
        npy_array latent_mask = npy_load(std::string(argv[2]) + "/latent_mask.npy");
        npy_array ref = npy_load(std::string(argv[2]) + "/vector_step_00.npy");
        if (noise.dtype != "<f4" || noise.shape.size() != 3 || noise.shape[0] != 1) throw std::runtime_error("bad noise.npy");
        if (text_emb.dtype != "<f4" || text_emb.shape.size() != 3 || text_emb.shape[0] != 1) throw std::runtime_error("bad text_emb.npy");
        if (style_ttl.dtype != "<f4" || style_ttl.n_elements() != 50*256) throw std::runtime_error("bad style_ttl.npy");
        if (latent_mask.dtype != "<f4" || latent_mask.n_elements() != (size_t)noise.shape[2]) throw std::runtime_error("bad latent_mask.npy");
        std::vector<float> got;
        std::string error;
        if (!supertonic_vector_step_cpu(model, npy_as_f32(noise), (int)noise.shape[2],
                                        npy_as_f32(text_emb), (int)text_emb.shape[2],
                                        npy_as_f32(style_ttl), npy_as_f32(latent_mask),
                                        0, 5, got, &error)) {
            throw std::runtime_error("vector failed: " + error);
        }
        compare_stats s = compare_f32(got.data(), npy_as_f32(ref), got.size());
        print_compare("supertonic_vector_step0", s);
        if (s.max_abs_err > 5e-3) { fprintf(stderr, "supertonic vector parity FAILED\n"); rc = 1; }
        else fprintf(stderr, "supertonic vector parity PASS\n");

        std::vector<float> got_ggml;
        if (!supertonic_vector_step_ggml(model, npy_as_f32(noise), (int)noise.shape[2],
                                         npy_as_f32(text_emb), (int)text_emb.shape[2],
                                         npy_as_f32(style_ttl), npy_as_f32(latent_mask),
                                         0, 5, got_ggml, &error)) {
            throw std::runtime_error("vector ggml failed: " + error);
        }
        compare_stats sg = compare_f32(got_ggml.data(), npy_as_f32(ref), got_ggml.size());
        print_compare("supertonic_vector_step0_ggml", sg);
        if (sg.max_abs_err > 5e-3) { fprintf(stderr, "supertonic vector GGML parity FAILED\n"); rc = 1; }
        else fprintf(stderr, "supertonic vector GGML parity PASS\n");
    } catch (const std::exception & e) { fprintf(stderr, "test failed: %s\n", e.what()); rc = 1; }
    free_supertonic_model(model);
    return rc;
}
