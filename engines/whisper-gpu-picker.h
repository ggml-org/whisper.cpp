#pragma once

// GPU-picker helper for callers of the vendored whisper.cpp
// (third_party/whisper.cpp). Returns the value the caller should pass as
// `whisper_context_params.gpu_device`.
//
// WHY THIS EXISTS
// ---------------
// whisper's own `whisper_backend_init_gpu` walks the ggml-backend registry
// linearly and picks the Nth GPU-or-IGPU device (where N is
// `params.gpu_device`, default 0). That default silently loses on two setups
// speech callers care about:
//
//   1. NVIDIA host with both CUDA and Vulkan compiled in — ggml usually
//      enumerates Vulkan first, so `gpu_device=0` picks the Vulkan adapter
//      on the same NVIDIA card. CUDA is vendor-native and measurably faster;
//      that is what llm-llamacpp / diffusion / tts / parakeet / audiogen all
//      prefer, so whisper callers should too.
//   2. Hybrid dGPU + iGPU host (e.g. NVIDIA discrete + Intel integrated) —
//      ggml enumeration order is not stable across driver installs, and
//      `gpu_device=0` may land on the iGPU. All other QVAC speech engines
//      prefer discrete over integrated.
//
// The vendored whisper subtree is gated by third_party/whisper.cpp/PATCHES.md
// against direct edits, so the selection lives here on the caller side.
// Consumers in the qvac monorepo (packages/asr-ggml) wire the return value
// into whisper_context_params.gpu_device before whisper_init_from_file_*.
//
// USAGE
// -----
//     #include "whisper-gpu-picker.h"
//     whisper_context_params p = whisper_context_default_params();
//     p.use_gpu    = true;
//     p.gpu_device = qvac::speech::pick_whisper_gpu_device();
//     ctx = whisper_init_from_file_with_params(path, p);
//
// The helper never calls into whisper.cpp itself — it only reads the shared
// ggml-backend registry, so it links cleanly under both GGML_BACKEND_DL=ON
// (dlopened per-arch backends) and OFF (static-linked) builds.

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

namespace qvac::speech {

// Adreno generation from a device name/description: "Adreno (TM) 740" -> 740,
// the Snapdragon-X "Adreno X1-85" naming -> 800 (7xx/8xx-tier silicon), else
// -1. Mirrors parse_adreno_version in engines/tts/src/backend_selection.cpp
// and engines/audiogen/src/acestep/backend_registry.h so the four speech
// stacks (tts / audiogen / parakeet / whisper-caller) reach the same verdict
// on the same hardware. Hand-rolled rather than <regex> to keep this header
// cheap to include.
inline int parse_adreno_version_whisper_picker(const char * s) {
    if (!s) return -1;
    std::string t(s);
    for (char & c : t) c = (char) std::tolower((unsigned char) c);
    int best = -1;
    for (size_t p = t.find("dreno"); p != std::string::npos; p = t.find("dreno", p + 1)) {
        for (size_t i = p + 5; i < t.size(); ++i) {
            if (t[i] == 'x' && i + 1 < t.size() && std::isdigit((unsigned char) t[i + 1])) {
                if (best < 800) best = 800;
                break;
            }
            if (!std::isdigit((unsigned char) t[i])) continue;
            size_t j = i;
            while (j < t.size() && std::isdigit((unsigned char) t[j])) ++j;
            const size_t len = j - i;
            if (len >= 3 && len <= 4) {
                const int v = std::stoi(t.substr(i, len));
                if (v > best) best = v;
            }
            break;
        }
    }
    return best;
}

// Return the index-among-GPU-devices — using whisper.cpp's exact enumeration
// order (GPU + IGPU, in ggml registry order) — that the QVAC speech tier
// policy prefers.
//
//   1. Adreno 700+ OpenCL — Snapdragon 8 Gen 2/3/4 ship validated ggml-opencl
//      kernels that outperform the same card's Vulkan adapter. Matches the
//      top tier used by parakeet / audiogen / tts on the same hardware; the
//      packages/asr-ggml addon that consumes this helper runs on Android, so
//      the tier is load-bearing there. (Adreno and NVIDIA CUDA never coexist
//      on the same host — Adreno is Snapdragon-only — so putting Adreno above
//      CUDA is safe.)
//   2. First CUDA device on a discrete GPU (typical NVIDIA).
//   3. First CUDA device on an integrated GPU (Tegra/Jetson).
//   4. Otherwise, first discrete (GPU) non-CUDA, non-OpenCL device —
//      Vulkan/Metal/WebGPU on a dGPU, etc.
//   5. Otherwise, first integrated (IGPU) non-OpenCL device — UMA Vulkan,
//      Mali iGPU, Intel iGPU, Apple integrated, etc.
//   6. Otherwise, first OpenCL device that did not qualify for the Adreno
//      700+ tier — Adreno 6xx (known-broken kernels), desktop OpenCL, or any
//      other unrecognised OpenCL adapter. Ranked last so a Vulkan/Metal
//      sibling on the same host is preferred; matches the `opencl_other`
//      last-resort tier in parakeet / audiogen / tts.
//   7. Otherwise 0 (whisper's own default; whisper_backend_init_gpu will
//      report "no GPU found" and return nullptr, and the caller falls back
//      to CPU as before).
//
// Returns 0 when the registry has not yet been loaded (no
// `ggml_backend_load_all*` call has happened): with an empty device list
// there is nothing to prefer, and the default preserves whisper's own
// no-GPU-found path.
inline int pick_whisper_gpu_device() {
    const size_t n_dev = ggml_backend_dev_count();

    // Whisper counts GPU-or-IGPU devices linearly; we mirror that counting
    // so the index we return means the same thing on the whisper side.
    // Track the first hit in each tier as we walk.
    int idx_adreno_opencl   = -1;
    int idx_cuda_discrete   = -1;
    int idx_cuda_integrated = -1;
    int idx_discrete        = -1;
    int idx_integrated      = -1;
    int idx_opencl_other    = -1;

    int cnt = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
        const char *       reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        const bool         is_cuda   = reg_name && std::strcmp(reg_name, "CUDA")   == 0;
        const bool         is_opencl = reg_name && std::strcmp(reg_name, "OpenCL") == 0;
        const bool         is_integrated = (type == GGML_BACKEND_DEVICE_TYPE_IGPU);
        const int          adreno_v =
            is_opencl ? std::max(parse_adreno_version_whisper_picker(ggml_backend_dev_name(dev)),
                                 parse_adreno_version_whisper_picker(ggml_backend_dev_description(dev)))
                      : -1;

        // OpenCL that did not qualify for Adreno-700+ (Adreno 6xx broken,
        // desktop OpenCL, or unrecognised) is routed to the last-resort tier
        // rather than the generic discrete/integrated buckets. This matches
        // parakeet's `opencl_other` bucket (parakeet_ctc.cpp) and keeps a
        // Vulkan/Metal sibling on the same host preferred over a driver we
        // don't want to pick by default.
        if      (is_opencl && adreno_v >= 700 && idx_adreno_opencl   < 0) idx_adreno_opencl   = cnt;
        else if (is_opencl                    && idx_opencl_other    < 0) idx_opencl_other    = cnt;
        else if (is_cuda   && !is_integrated  && idx_cuda_discrete   < 0) idx_cuda_discrete   = cnt;
        else if (is_cuda   &&  is_integrated  && idx_cuda_integrated < 0) idx_cuda_integrated = cnt;
        else if (!is_cuda  && !is_integrated  && idx_discrete        < 0) idx_discrete        = cnt;
        else if (!is_cuda  &&  is_integrated  && idx_integrated      < 0) idx_integrated      = cnt;

        ++cnt;
    }

    if (idx_adreno_opencl   >= 0) return idx_adreno_opencl;
    if (idx_cuda_discrete   >= 0) return idx_cuda_discrete;
    if (idx_cuda_integrated >= 0) return idx_cuda_integrated;
    if (idx_discrete        >= 0) return idx_discrete;
    if (idx_integrated      >= 0) return idx_integrated;
    if (idx_opencl_other    >= 0) return idx_opencl_other;
    return 0;
}

// Testable form of the tier policy: takes an explicit device list (registry
// name, is-integrated, adreno_version) and returns the index the policy
// prefers. Kept alongside `pick_whisper_gpu_device` so both call the same
// logic; the process-scoped version above is a thin wrapper that reads the
// ggml registry and delegates here. Exposed for unit tests that simulate a
// fake device topology without depending on the host's real GPU set.
struct PickerDevice {
    const char * reg_name;       // registry name, e.g. "CUDA" / "Vulkan" / "Metal" / "WebGPU" / "OpenCL"
    bool         is_integrated;  // true for GGML_BACKEND_DEVICE_TYPE_IGPU
    int          adreno_version; // parsed Adreno generation (700, 740, 800, ...); -1 for non-Adreno
};

inline int pick_whisper_gpu_device_from(const PickerDevice * devs, size_t n) {
    int idx_adreno_opencl   = -1;
    int idx_cuda_discrete   = -1;
    int idx_cuda_integrated = -1;
    int idx_discrete        = -1;
    int idx_integrated      = -1;
    int idx_opencl_other    = -1;

    for (size_t i = 0; i < n; ++i) {
        const PickerDevice & d = devs[i];
        const bool is_cuda   = d.reg_name && std::strcmp(d.reg_name, "CUDA")   == 0;
        const bool is_opencl = d.reg_name && std::strcmp(d.reg_name, "OpenCL") == 0;
        const int  idx       = static_cast<int>(i);
        // See pick_whisper_gpu_device above: OpenCL that misses the Adreno-700+
        // gate goes to the last-resort tier so a Vulkan/Metal sibling wins.
        if      (is_opencl && d.adreno_version >= 700 && idx_adreno_opencl   < 0) idx_adreno_opencl   = idx;
        else if (is_opencl                             && idx_opencl_other   < 0) idx_opencl_other    = idx;
        else if (is_cuda   && !d.is_integrated         && idx_cuda_discrete   < 0) idx_cuda_discrete   = idx;
        else if (is_cuda   &&  d.is_integrated         && idx_cuda_integrated < 0) idx_cuda_integrated = idx;
        else if (!is_cuda  && !d.is_integrated         && idx_discrete        < 0) idx_discrete        = idx;
        else if (!is_cuda  &&  d.is_integrated         && idx_integrated      < 0) idx_integrated      = idx;
    }

    if (idx_adreno_opencl   >= 0) return idx_adreno_opencl;
    if (idx_cuda_discrete   >= 0) return idx_cuda_discrete;
    if (idx_cuda_integrated >= 0) return idx_cuda_integrated;
    if (idx_discrete        >= 0) return idx_discrete;
    if (idx_integrated      >= 0) return idx_integrated;
    if (idx_opencl_other    >= 0) return idx_opencl_other;
    return 0;
}

}  // namespace qvac::speech
