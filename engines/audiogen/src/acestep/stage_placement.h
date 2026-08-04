#pragma once

// Stage-placement policy for the ACE-Step pipeline: given the registry name of
// the active GPU backend plus the environment escape hatches, decide which of
// the encoder / LM / FSQ-detokenizer stages run on it and which fall back to
// the CPU. Engine::create() applies the answer (engine.cpp).
//
// This lives apart from backend_registry.h, and takes a plain `const char *`
// rather than a ggml_backend_t, so the policy is exercisable with no GPU, no
// ggml context and no GGUF -- see test/test_acestep_units.cpp. The placement
// decides which numerical path generated audio takes, so it is regression
// tested rather than only observed on a device lane.

#include <cstdlib>
#include <cstring>

namespace tts_cpp::acestep {

// Backend predicates over the REGISTRY name (backend_reg_name()), which carries
// no device-index suffix -- `ggml_backend_name()` returns "Vulkan0" / "MTL0"
// and would never match here.
inline bool backend_name_is_vulkan(const char * name) {
    return name && std::strcmp(name, "Vulkan") == 0;
}

// ggml-metal registers as "MTL"; older ggml reported "Metal". Match both, or the
// check is silently dead on one of them.
inline bool backend_name_is_metal(const char * name) {
    return name && (std::strcmp(name, "MTL") == 0 || std::strcmp(name, "Metal") == 0);
}

// Environment escape hatches, read once at create(). Presence is what counts:
// ACESTEP_LM_CPU=0 still forces the LM to the CPU, matching the getenv() checks
// this replaced.
struct PlacementOverrides {
    bool lm_gpu       = false;  // ACESTEP_LM_GPU
    bool lm_cpu       = false;  // ACESTEP_LM_CPU
    bool detok_gpu    = false;  // ACESTEP_DETOK_GPU
    bool detok_cpu    = false;  // ACESTEP_DETOK_CPU
    bool encoders_cpu = false;  // ACESTEP_ENCODERS_CPU
};

// Per-stage answer. Defaults are the "GPU is active and allowlisted" case; the
// DiT and the VAE are not represented because they always run on the GPU.
struct StagePlacement {
    bool enc_on_gpu   = true;
    bool lm_on_gpu    = true;
    bool detok_on_gpu = true;
};

// Allowlist, then the overrides. Only meaningful when a GPU backend actually
// initialised -- Engine::create() does not consult it on the CPU-only path.
//
// The LM and the detokenizer are allowlisted rather than denylisted: a backend
// nobody has measured keeps the CPU placement and cannot silently regress
// generated audio. Widen the list once a backend is measured; ACESTEP_LM_GPU /
// ACESTEP_DETOK_GPU take that measurement without a rebuild.
inline StagePlacement resolve_stage_placement(const char * reg_name, const PlacementOverrides & ov) {
    StagePlacement p;

    if (!backend_name_is_vulkan(reg_name) && !backend_name_is_metal(reg_name)) {
        p.lm_on_gpu    = false;
        p.detok_on_gpu = false;
    }

    // Applied after the allowlist; CPU wins if both are set for a stage.
    if (ov.lm_gpu)    p.lm_on_gpu    = true;
    if (ov.lm_cpu)    p.lm_on_gpu    = false;
    if (ov.detok_gpu) p.detok_on_gpu = true;
    if (ov.detok_cpu) p.detok_on_gpu = false;

    if (ov.encoders_cpu) p.enc_on_gpu = false;

    return p;
}

inline PlacementOverrides placement_overrides_from_env() {
    PlacementOverrides ov;
    ov.lm_gpu       = std::getenv("ACESTEP_LM_GPU") != nullptr;
    ov.lm_cpu       = std::getenv("ACESTEP_LM_CPU") != nullptr;
    ov.detok_gpu    = std::getenv("ACESTEP_DETOK_GPU") != nullptr;
    ov.detok_cpu    = std::getenv("ACESTEP_DETOK_CPU") != nullptr;
    ov.encoders_cpu = std::getenv("ACESTEP_ENCODERS_CPU") != nullptr;
    return ov;
}

}  // namespace tts_cpp::acestep
