#pragma once

// Core ML sidecar for Parler's DAC decoder (scripts/parler/export-dac-coreml.py).
//
// The sidecar is `<gguf stem minus quant tag>-dac.mlmodelc` next to the GGUF.
// It decodes one fixed window of summed RVQ latents, latent [1, latent_dim, W]
// plus a 0/1 frame mask valid [1, 1, W], to pcm [1, 1, W * hop]; frames past
// the end of the utterance are masked out, which reproduces the zero padding
// of a shorter decode. The quantizer runs on the host from the GGUF's
// codebooks, and parler_dac_decode walks the utterance in windows with
// parler_dac_rf_frames of real context on each side, exactly like the ggml
// decode, so a sidecar decode matches a whole-sequence decode within fp16
// tolerance.
//
// Environment:
//   PARLER_COREML_DISABLE        set: never open the sidecar
//   PARLER_COREML_STRICT         set: a missing or failing sidecar is an error
//                                instead of a fallback to ggml
//   PARLER_COREML_COMPUTE_UNITS  all | cpu_and_ane | cpu_and_gpu | cpu_only

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

struct parler_model;

constexpr const char * PARLER_COREML_DISABLE_ENV = "PARLER_COREML_DISABLE";
constexpr const char * PARLER_COREML_STRICT_ENV  = "PARLER_COREML_STRICT";
constexpr const char * PARLER_COREML_UNITS_ENV   = "PARLER_COREML_COMPUTE_UNITS";
constexpr const char * PARLER_COREML_DEFAULT_UNITS = "all";
constexpr const char * PARLER_DAC_SIDECAR_SUFFIX = "-dac";
constexpr const char * PARLER_DAC_GGML_BACKEND   = "ggml";
constexpr const char * PARLER_DAC_MIXED_BACKEND  = "mixed";

std::string parler_dac_sidecar_path(const std::string & gguf_path);
bool parler_coreml_disabled();
bool parler_coreml_strict();
// Whether an Engine on `gguf_path` would try the sidecar: Core ML compiled
// in, not disabled, and `<stem>-dac.mlmodelc` present.
bool parler_dac_sidecar_enabled(const std::string & gguf_path);

// One fixed-window DAC decode. Implementations: the Core ML model, and test
// doubles that exercise the fallback without Core ML.
class parler_dac_sidecar {
public:
    virtual ~parler_dac_sidecar() = default;
    // Frames per call, context included.
    virtual int window() const = 0;
    // "coreml-all", "coreml-ane", ... -- reported as the DAC backend.
    virtual const char * label() const = 0;
    // latent: [latent_dim, window()] channel-major; valid: [window()] 0/1;
    // pcm: [window() * hop]. False on any failure.
    virtual bool decode(const float * latent, const float * valid, float * pcm) = 0;
};

// Host copy of the residual vector quantizer: per codebook the code vectors
// [codebook_size, cb_dim], the 1x1 output projection [latent_dim, cb_dim] and
// its bias [latent_dim].
struct parler_dac_rvq {
    int n_q = 0, codebook_size = 0, cb_dim = 0, latent_dim = 0;
    std::vector<std::vector<float>> codebook, proj, bias;

    // Sums every codebook's projection for frames [f0, f1) of the
    // [n_q, n_frames] code matrix into out[c * stride + (t - f0)].
    void latent(const int32_t * codes, int n_frames, int f0, int f1,
                float * out, int stride) const;
};

// Reads the quantizer tensors from a loaded model (any backend). False when
// they are not F32 or are missing.
bool parler_dac_rvq_load(const parler_model & model, parler_dac_rvq & rvq, std::string * err);

// Opens the sidecar next to `gguf_path` and checks its feature shapes against
// the model. nullptr when Core ML is not compiled in, the sidecar is absent or
// disabled, or it does not match (with the reason in *why, if given).
std::shared_ptr<parler_dac_sidecar> parler_open_dac_sidecar(const std::string & gguf_path,
                                                            const parler_model & model,
                                                            std::string * why = nullptr);

// Attaches `sidecar` (may be null) to the model, loading the host quantizer it
// needs. False, leaving the model on ggml, when the quantizer cannot be read.
bool parler_attach_dac_sidecar(parler_model & model, std::shared_ptr<parler_dac_sidecar> sidecar,
                               std::string * err = nullptr);

// Opens and attaches the sidecar next to `gguf_path`. Returns whether one is
// attached; under PARLER_COREML_STRICT a missing or unusable sidecar throws.
bool parler_attach_dac_sidecar(parler_model & model, const std::string & gguf_path);

// The sidecar decode of frames [out_begin, out_end) (see parler_dac_decode).
// False on any sidecar failure; pcm_out / latent_out are then unspecified.
bool parler_dac_decode_sidecar(const parler_model & model, const int32_t * codes, int n_frames,
                               std::vector<float> & pcm_out, std::vector<float> * latent_out,
                               int out_begin, int out_end);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
