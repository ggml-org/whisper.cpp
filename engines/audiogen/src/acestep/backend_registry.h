#pragma once

// ACE-Step backend acquisition via the ggml backend registry.
//
// WHY THIS EXISTS (not just `ggml_backend_cpu_init()`):
// On desktop x86-64 / Apple the ggml-speech port static-links the CPU backend,
// so `ggml_backend_cpu_init` / `ggml_backend_cpu_set_n_threads` are defined in
// the addon's `.bare`. On arm64 (Android + Linux) the port builds the CPU
// backend as per-microarch dlopen MODULE .so files
// (`GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON`), so those two symbols live
// ONLY in the lazily-loaded backend .so and are left UNDEFINED in the `.bare`.
// A `.bare` with UND engine symbols and no DT_NEEDED provider dlopen-crashes on
// device (SIGABRT) the moment it is co-loaded in the SDK -- and the
// verify-prebuild-symbols CI guard rejects it before it ships. See
// qvac/packages/classification-ggml/docs/architecture.md and the
// @qvac/tts-ggml@0.2.1 regression that guard was written for.
//
// The fix is to go through the registry, which resolves against whichever CPU
// backend was loaded (static or dlopen'd), using only ggml-base symbols that are
// always statically present:
//   * `ggml_backend_load_all_from_path(dir)` loads the dlopen backend modules
//     from the addon's per-arch prebuilds subdir (no-op on static-only builds);
//   * `ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, ...)` returns the
//     CPU backend from the registry;
//   * the thread count is set through the generic `ggml_backend_set_n_threads`
//     proc-address fetched from the backend's registry entry (the same pattern
//     llama.cpp uses for multi-variant CPU backends).

#include "audiogen-cpp/gpu_fallback.h"
#include "acestep/stage_placement.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

namespace tts_cpp::acestep {

inline constexpr char BACKEND_REQUEST_AUTO[]    = "auto";
inline constexpr char BACKEND_REQUEST_CPU[]     = "cpu";
inline constexpr char BACKEND_REQUEST_OPENCL[]  = "opencl";
inline constexpr char BACKEND_REQUEST_WEBGPU[]  = "webgpu";
inline constexpr char BACKEND_REQUEST_HEXAGON[] = "hexagon";

inline constexpr char HEXAGON_PRIMARY_DEVICE[] = "HTP0";
inline constexpr char OPENCL_REG_NAME[]        = "OpenCL";

inline constexpr char DSP_LIBRARY_PATH_SEPARATOR = ';';
inline constexpr char DSP_LIBRARY_PATH_ENV[]        = "DSP_LIBRARY_PATH";
inline constexpr char DSP_LIBRARY_PATH_LEGACY_ENV[] = "ADSP_LIBRARY_PATH";

inline std::string prepend_dsp_library_directory(const std::string & dir, const std::string & paths) {
    if (dir.empty()) return paths;
    const std::string sep(1, DSP_LIBRARY_PATH_SEPARATOR);
    if ((sep + paths + sep).find(sep + dir + sep) != std::string::npos) return paths;
    return paths.empty() ? dir : dir + sep + paths;
}

inline void prepare_dsp_library_path(const std::string & dir) {
#ifdef __ANDROID__
    if (dir.empty()) return;
    if (dir.find(DSP_LIBRARY_PATH_SEPARATOR) != std::string::npos) {
        fprintf(stderr, "[acestep-engine] backends_dir contains ';'; not added to %s\n", DSP_LIBRARY_PATH_ENV);
        return;
    }
    const char * prior = std::getenv(DSP_LIBRARY_PATH_ENV);
    if (!prior) prior = std::getenv(DSP_LIBRARY_PATH_LEGACY_ENV);
    const std::string path = prepend_dsp_library_directory(dir, prior ? prior : "");
    if (setenv(DSP_LIBRARY_PATH_ENV, path.c_str(), 1) != 0) {
        fprintf(stderr, "[acestep-engine] failed to set %s\n", DSP_LIBRARY_PATH_ENV);
    }
#else
    (void) dir;
#endif
}

// Load the dlopen'd ggml backend modules (CPU micro-arch variants, Vulkan,
// OpenCL, ...) that the addon staged next to its `.bare` in `dir`. Idempotent
// and safe to call more than once; a no-op when `dir` is empty or on
// static-only builds where the registry is already populated at load time.
inline void load_backends(const std::string & dir) {
    if (dir.empty()) return;
    prepare_dsp_library_path(dir);
    ggml_backend_load_all_from_path(dir.c_str());
}

inline bool backend_request_is_auto(const std::string & requested) {
    return requested.empty() || requested == BACKEND_REQUEST_AUTO;
}

inline bool backend_request_matches(const std::string & requested, const char * reg_name, const char * device_name,
                                    enum ggml_backend_dev_type type) {
    if (backend_request_is_auto(requested)) return false;
    if (requested == BACKEND_REQUEST_CPU) return type == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (requested == BACKEND_REQUEST_OPENCL) return reg_name && std::strcmp(reg_name, OPENCL_REG_NAME) == 0;
    if (requested == BACKEND_REQUEST_WEBGPU) return backend_name_is_webgpu(reg_name);
    if (requested == BACKEND_REQUEST_HEXAGON) {
        return backend_name_is_hexagon(reg_name) && device_name && std::strcmp(device_name, HEXAGON_PRIMARY_DEVICE) == 0;
    }
    return device_name && requested == device_name;
}

inline const char * backend_dev_reg_name(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg  = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char *       name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name ? name : "";
}

inline bool backend_device_matches_request(const std::string & requested, ggml_backend_dev_t dev) {
    return dev && backend_request_matches(requested, backend_dev_reg_name(dev), ggml_backend_dev_name(dev),
                                          ggml_backend_dev_type(dev));
}

inline ggml_backend_dev_t backend_requested_device(const std::string & requested) {
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (backend_device_matches_request(requested, dev)) return dev;
    }
    return nullptr;
}

struct RequestedBackendInit {
    ggml_backend_t backend    = nullptr;
    bool           saw_device = false;
};

inline RequestedBackendInit backend_init_first_matching(const std::string & requested) {
    RequestedBackendInit result;
    const size_t         n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev && !result.backend; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!backend_device_matches_request(requested, dev)) continue;
        result.saw_device = true;
        result.backend    = ggml_backend_dev_init(dev, nullptr);
    }
    return result;
}

inline GpuFallbackReason backend_request_outcome(const RequestedBackendInit & init) {
    if (init.backend) return GpuFallbackReason::none;
    return init.saw_device ? GpuFallbackReason::init_failed : GpuFallbackReason::no_devices;
}

inline ggml_backend_t backend_requested_init(const std::string & requested, GpuFallbackReason * reason = nullptr) {
    const RequestedBackendInit init = backend_init_first_matching(requested);
    if (reason) *reason = backend_request_outcome(init);
    return init.backend;
}

inline bool backend_is_cpu_device(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// CPU backend from the registry -- resolves the same on static and dlopen
// (GGML_CPU_ALL_VARIANTS) builds, unlike the CPU-backend-only
// `ggml_backend_cpu_init()`. Call `load_backends()` first on DL platforms.
inline ggml_backend_t backend_cpu_init() {
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

// Both discrete and integrated GPUs are valid compute devices. Vulkan reports
// UMA adapters (including Android Mali) as IGPU, so
// ggml_backend_init_by_type(GPU) alone silently misses them.
inline bool backend_device_type_is_gpu(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
           type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

inline bool backend_reg_name_is_validated_gpu(const char * name) {
    return name && (std::strcmp(name, "Vulkan") == 0 ||
                    std::strcmp(name, "MTL") == 0 ||
                    std::strcmp(name, "Metal") == 0 ||
                    std::strcmp(name, "CUDA") == 0);
}

// Adreno generation from a device name/description: "Adreno (TM) 740" -> 740, the
// Snapdragon-X "Adreno X1-85" naming -> 800 (7xx/8xx-tier silicon), else -1.
// Mirrors `parse_adreno_version` in engines/tts/src/backend_selection.cpp so the
// speech stacks reach the same verdict on the same hardware; hand-rolled rather than
// <regex> to keep this header cheap to include.
inline int parse_adreno_version(const char * s) {
    if (!s) return -1;
    std::string t(s);
    for (char & c : t) c = (char) std::tolower((unsigned char) c);
    int best = -1;
    for (size_t p = t.find("dreno"); p != std::string::npos; p = t.find("dreno", p + 1)) {
        // Scan forward over non-digits; a 3-4 digit run is the model number, an
        // "x<digit>" token is the Snapdragon-X naming. A shorter digit run (the
        // "3.0" of an embedded "OpenCL 3.0") ends this candidate, as in the regex.
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

// A device this build would rather drive through ggml-opencl than Vulkan: on
// Adreno 700+ the OpenCL kernels are validated and faster, but ggml enumerates
// Vulkan first, so the generic passes below would always hand back Vulkan.
inline bool backend_dev_prefers_opencl(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char *       rn  = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (!rn || std::strcmp(rn, "OpenCL") != 0) return false;
    return std::max(parse_adreno_version(ggml_backend_dev_name(dev)),
                    parse_adreno_version(ggml_backend_dev_description(dev))) >= 700;
}

// Tier ranking for GPU selection (lower = preferred). Kept as a pure function
// on (registry name, ggml device type, parsed Adreno version) so unit tests
// can exercise the policy against a synthesised device topology without a
// live ggml registry.
//
// Ordering mirrors backend_gpu_init below:
//   0. Adreno 700+ OpenCL — validated + faster than Vulkan on Snapdragon.
//   1. CUDA on a discrete GPU — vendor-native, preferred over Vulkan on NVIDIA.
//   2. CUDA on an integrated GPU (Tegra) — same rationale, discrete-first tie-break.
//   3. Validated (Vulkan/Metal/CUDA-shaped) GPU on a discrete adapter.
//   4. Validated GPU on an integrated adapter (UMA Vulkan, Apple iGPU, ...).
//   5. Any other GPU on a discrete adapter (unvalidated backends).
//   6. Any other GPU on an integrated adapter.
//   7. Not a GPU / non-selectable (CPU, accelerators, the Hexagon NPU, which
//      runs only on an explicit backend request) — never picked.
enum class GpuTier {
    AdrenoOpenCL700Plus     = 0,
    CudaDiscrete            = 1,
    CudaIntegrated          = 2,
    ValidatedDiscrete       = 3,
    ValidatedIntegrated     = 4,
    OtherDiscrete           = 5,
    OtherIntegrated         = 6,
    NotSelectable           = 7,
};

inline GpuTier gpu_tier_for(const char *                     reg_name,
                            enum ggml_backend_dev_type       dev_type,
                            int                              adreno_version) {
    if (!backend_device_type_is_gpu(dev_type) || backend_name_is_hexagon(reg_name)) return GpuTier::NotSelectable;
    const bool integrated = (dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (reg_name && std::strcmp(reg_name, "OpenCL") == 0 && adreno_version >= 700) {
        return GpuTier::AdrenoOpenCL700Plus;
    }
    if (reg_name && std::strcmp(reg_name, "CUDA") == 0) {
        return integrated ? GpuTier::CudaIntegrated : GpuTier::CudaDiscrete;
    }
    if (backend_reg_name_is_validated_gpu(reg_name)) {
        return integrated ? GpuTier::ValidatedIntegrated : GpuTier::ValidatedDiscrete;
    }
    return integrated ? GpuTier::OtherIntegrated : GpuTier::OtherDiscrete;
}

// GPU backend from the registry. Prefer Adreno 700+ OpenCL, then CUDA on
// NVIDIA (vendor-native, measurably faster than the same card's Vulkan
// adapter), then a validated Vulkan/Metal device, then a discrete adapter,
// while still preserving the historical fallback to another GPU backend
// when none of those exist. Off Adreno / non-NVIDIA nothing changes: Vulkan
// stays preferred over OpenCL because the complete ACE-Step pipeline,
// including the VAE custom ops, is validated there. Accepting IGPU is
// required for UMA adapters such as Pixel's Mali GPU, Apple integrated GPUs
// and Adreno itself. The {require_validated, {GPU, IGPU}} nest below already
// prefers discrete over integrated (dGPU tier tried before iGPU tier at the
// same validated level).
//
// Try every matching device so one adapter failing to initialise does not hide
// another usable one.
// `reason`, when given, receives why no backend was returned, so a caller can
// tell "no GPU device was enumerated" from "a device was found and refused to
// initialise" instead of inferring from a null.
inline ggml_backend_t backend_gpu_init(GpuFallbackReason * reason = nullptr) {
    bool saw_device = false;

    const size_t n_dev_opencl = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev_opencl; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !backend_device_type_is_gpu(ggml_backend_dev_type(dev))) continue;
        if (!backend_dev_prefers_opencl(dev)) continue;
        saw_device = true;
        if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
            if (reason) *reason = GpuFallbackReason::none;
            return backend;
        }
    }

    // CUDA-first pass on NVIDIA. When a build carries both CUDA and Vulkan
    // and the same NVIDIA card is exposed through both, the generic
    // validated-GPU loop below would race to whichever ggml enumerates first
    // (typically Vulkan). CUDA is vendor-native and measurably faster on
    // that card, so try every visible CUDA device first — GPU tier before
    // IGPU tier (CUDA on tegra-class hardware surfaces as IGPU) to keep the
    // discrete-first preference.
    for (enum ggml_backend_dev_type wanted :
         {GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU}) {
        const size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev) continue;
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (!backend_device_type_is_gpu(type) || type != wanted) continue;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
            if (!reg_name || std::strcmp(reg_name, "CUDA") != 0) continue;
            saw_device = true;
            if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
                if (reason) *reason = GpuFallbackReason::none;
                return backend;
            }
        }
    }

    for (bool require_validated : {true, false}) {
        for (enum ggml_backend_dev_type wanted :
             {GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU}) {
            const size_t n_dev = ggml_backend_dev_count();
            for (size_t i = 0; i < n_dev; ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (!dev) continue;
                const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
                if (!backend_device_type_is_gpu(type) || type != wanted) continue;

                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
                if (backend_name_is_hexagon(reg_name)) continue;
                if (backend_reg_name_is_validated_gpu(reg_name) != require_validated) continue;

                saw_device = true;
                if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
                    if (reason) *reason = GpuFallbackReason::none;
                    return backend;
                }
            }
        }
    }

    if (reason) {
        *reason = saw_device ? GpuFallbackReason::init_failed : GpuFallbackReason::no_devices;
    }
    return nullptr;
}

// Set the compute thread count via the backend's generic
// `ggml_backend_set_n_threads` proc-address (works for the dlopen'd CPU
// variant); silently no-ops if the backend does not expose the setter.
inline void backend_set_n_threads(ggml_backend_t backend, int n_threads) {
    if (!backend) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return;
    auto set_n_threads =
        (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads) set_n_threads(backend, n_threads);
}

inline ggml_backend_t backend_init_for_tool(const char * requested, bool gpu, int n_threads) {
    ggml_backend_t backend = nullptr;
    if (requested && !backend_request_is_auto(requested)) {
        backend = backend_requested_init(requested);
    } else if (gpu) {
        backend = backend_gpu_init();
    } else {
        backend = backend_cpu_init();
    }
    if (backend && backend_is_cpu_device(backend)) backend_set_n_threads(backend, n_threads);
    return backend;
}

// Registry name of the backend implementation ("CPU", "Vulkan", "MTL", ...).
// Unlike `ggml_backend_name()` this carries no device-index suffix ("Vulkan0"), so
// stage-placement policies can compare it exactly. Mirrors the helper in
// engines/tts/src/backend_util.h; duplicated so audiogen stays self-contained.
// The policy that consumes it lives in stage_placement.h.
inline const char * backend_reg_name(ggml_backend_t backend) {
    if (!backend) return "";
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name ? name : "";
}

// Device description of the backend's active device; feeds the device-scoped
// placement decisions in stage_placement.h. Empty when unavailable.
inline const char * backend_dev_description(ggml_backend_t backend) {
    if (!backend) return "";
    ggml_backend_dev_t dev  = ggml_backend_get_device(backend);
    const char *       desc = dev ? ggml_backend_dev_description(dev) : nullptr;
    return desc ? desc : "";
}

}  // namespace tts_cpp::acestep
