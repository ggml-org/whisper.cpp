// End-to-end pipeline parity test: text_encoder -> denoising loop -> vocoder
// Uses reference noise/style/text_ids/text_mask/latent_mask from artifacts/supertonic-ref-quick
// and compares the final waveform against wav_full.npy.
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
        npy_array noise        = npy_load(std::string(argv[2]) + "/noise.npy");
        npy_array text_ids_npy = npy_load(std::string(argv[2]) + "/text_ids.npy");
        npy_array style_ttl    = npy_load(std::string(argv[2]) + "/style_ttl.npy");
        npy_array latent_mask  = npy_load(std::string(argv[2]) + "/latent_mask.npy");
        npy_array text_mask    = npy_load(std::string(argv[2]) + "/text_mask.npy"); (void) text_mask;
        npy_array wav_ref      = npy_load(std::string(argv[2]) + "/wav_full.npy");

        if (noise.dtype != "<f4" || noise.shape.size() != 3 || noise.shape[0] != 1)
            throw std::runtime_error("bad noise.npy");
        if (text_ids_npy.dtype != "<i8" && text_ids_npy.dtype != "<i4")
            throw std::runtime_error("bad text_ids.npy dtype");

        const int latent_len = (int) noise.shape[2];
        const int text_len   = (int) text_ids_npy.shape.back();

        std::vector<int64_t> text_ids(text_len);
        if (text_ids_npy.dtype == "<i8") {
            std::memcpy(text_ids.data(), text_ids_npy.data.data(), text_len * sizeof(int64_t));
        } else {
            const int32_t * src = reinterpret_cast<const int32_t*>(text_ids_npy.data.data());
            for (int i = 0; i < text_len; ++i) text_ids[i] = src[i];
        }

        std::vector<float> text_emb;
        std::string error;
        if (!supertonic_text_encoder_forward_ggml(model, text_ids.data(), text_len,
                                                  npy_as_f32(style_ttl), text_emb, &error)) {
            throw std::runtime_error("text encoder failed: " + error);
        }

        const int n_steps = 5; // matches reference dump
        const int channels = model.hparams.latent_channels;
        std::vector<float> latent(noise.n_elements());
        std::memcpy(latent.data(), npy_as_f32(noise), latent.size() * sizeof(float));

        std::vector<float> next;
        for (int step = 0; step < n_steps; ++step) {
            if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                             text_emb.data(), text_len,
                                             npy_as_f32(style_ttl), npy_as_f32(latent_mask),
                                             step, n_steps, next, &error)) {
                throw std::runtime_error("vector step " + std::to_string(step) + " failed: " + error);
            }
            latent.swap(next);
        }

        std::vector<float> wav;
        if (!supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, wav, &error)) {
            throw std::runtime_error("vocoder failed: " + error);
        }

        const size_t n = std::min((size_t) wav.size(), wav_ref.n_elements());
        compare_stats s = compare_f32(wav.data(), npy_as_f32(wav_ref), n);
        print_compare("supertonic_pipeline_wav", s);
        if (s.max_abs_err > 1e-3) { fprintf(stderr, "supertonic pipeline parity FAILED\n"); rc = 1; }
        else fprintf(stderr, "supertonic pipeline parity PASS\n");
    } catch (const std::exception & e) { fprintf(stderr, "test failed: %s\n", e.what()); rc = 1; }
    free_supertonic_model(model);
    return rc;
}
