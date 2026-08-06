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

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <string>

namespace tts_cpp::acestep {

// Load the dlopen'd ggml backend modules (CPU micro-arch variants, Vulkan,
// OpenCL, ...) that the addon staged next to its `.bare` in `dir`. Idempotent
// and safe to call more than once; a no-op when `dir` is empty or on
// static-only builds where the registry is already populated at load time.
inline void load_backends(const std::string & dir) {
    if (!dir.empty()) ggml_backend_load_all_from_path(dir.c_str());
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
                    std::strcmp(name, "Metal") == 0);
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

// GPU backend from the registry. Prefer Adreno 700+ OpenCL, then a measured
// Vulkan/Metal device, then a discrete adapter, while still preserving the
// historical fallback to another GPU backend when none of those exist. Off
// Adreno nothing changes: Vulkan stays preferred over OpenCL because the
// complete ACE-Step pipeline, including the VAE custom ops, is validated there.
// Accepting IGPU is required for UMA adapters such as Pixel's Mali GPU, Apple
// integrated GPUs and Adreno itself.
//
// Try every matching device so one adapter failing to initialise does not hide
// another usable one.
inline ggml_backend_t backend_gpu_init() {
    const size_t n_dev_opencl = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev_opencl; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !backend_device_type_is_gpu(ggml_backend_dev_type(dev))) continue;
        if (!backend_dev_prefers_opencl(dev)) continue;
        if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
            return backend;
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
                if (backend_reg_name_is_validated_gpu(reg_name) != require_validated) continue;

                if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
                    return backend;
                }
            }
        }
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

}  // namespace tts_cpp::acestep
