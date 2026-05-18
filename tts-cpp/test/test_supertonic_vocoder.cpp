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
    if (!load_supertonic_gguf(argv[1], model)) {
        return 1;
    }

    int rc = 0;
    try {
        npy_array latent_npy = npy_load(std::string(argv[2]) + "/final_latent.npy");
        npy_array wav_npy = npy_load(std::string(argv[2]) + "/wav_full.npy");
        if (latent_npy.dtype != "<f4" || latent_npy.shape.size() != 3 || latent_npy.shape[0] != 1) {
            throw std::runtime_error("unexpected final_latent.npy shape/dtype");
        }
        if (wav_npy.dtype != "<f4" || wav_npy.shape.size() != 2 || wav_npy.shape[0] != 1) {
            throw std::runtime_error("unexpected wav_full.npy shape/dtype");
        }
        const int latent_len = (int) latent_npy.shape[2];
        std::vector<float> wav;
        std::string error;
        if (!supertonic_vocoder_forward_cpu(model, npy_as_f32(latent_npy), latent_len, wav, &error)) {
            throw std::runtime_error("vocoder failed: " + error);
        }
        if (wav.size() != wav_npy.n_elements()) {
            fprintf(stderr, "wav size mismatch: got %zu ref %zu\n", wav.size(), wav_npy.n_elements());
            rc = 1;
        } else {
            compare_stats s = compare_f32(wav.data(), npy_as_f32(wav_npy), wav.size());
            print_compare("supertonic_vocoder", s);
            if (s.max_abs_err > 2e-3) {
                fprintf(stderr, "supertonic vocoder parity FAILED\n");
                rc = 1;
            } else {
                fprintf(stderr, "supertonic vocoder parity PASS\n");
            }
        }
        std::vector<float> wav_ggml;
        if (!supertonic_vocoder_forward_ggml(model, npy_as_f32(latent_npy), latent_len, wav_ggml, &error)) {
            throw std::runtime_error("vocoder ggml failed: " + error);
        }
        if (wav_ggml.size() != wav_npy.n_elements()) {
            fprintf(stderr, "wav ggml size mismatch: got %zu ref %zu\n", wav_ggml.size(), wav_npy.n_elements());
            rc = 1;
        } else {
            compare_stats s = compare_f32(wav_ggml.data(), npy_as_f32(wav_npy), wav_ggml.size());
            print_compare("supertonic_vocoder_ggml", s);
            if (s.max_abs_err > 2e-3) {
                fprintf(stderr, "supertonic vocoder GGML parity FAILED\n");
                rc = 1;
            } else {
                fprintf(stderr, "supertonic vocoder GGML parity PASS\n");
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    free_supertonic_model(model);
    return rc;
}
