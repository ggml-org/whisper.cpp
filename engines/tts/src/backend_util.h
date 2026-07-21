#pragma once

// Backend-introspection helpers that work uniformly under both
// GGML_BACKEND_DL=ON and GGML_BACKEND_DL=OFF. The legacy
// ggml_backend_is_cpu / ggml_backend_is_metal entry points live in
// the per-backend shared libraries (libggml-cpu.* / libggml-metal.*),
// so they are unlinkable from libtts-cpp under the dynamic-loader
// build mode embedded host applications typically ship with. Routing
// through the registry (ggml_backend_get_device + ggml_backend_dev_*)
// reaches the same answer in both modes.
//
// Mirrors parakeet-cpp/src/backend_util.h 1:1 (same QVAC speech-stack
// pattern); kept in a tts-cpp namespace so the two libraries can be
// vendored side-by-side without ODR collisions on the helpers.

#include "ggml-backend.h"

#include <cstring>

namespace tts_cpp::detail {

inline const char * backend_reg_name(ggml_backend_t b) {
    if (!b) return "";
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return "";
    const char * n = ggml_backend_reg_name(reg);
    return n ? n : "";
}

inline bool backend_is_cpu(ggml_backend_t b) {
    if (!b) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

inline bool backend_is_metal(ggml_backend_t b) {
    return std::strcmp(backend_reg_name(b), "Metal") == 0;
}

inline bool backend_is_vulkan(ggml_backend_t b) {
    return std::strcmp(backend_reg_name(b), "Vulkan") == 0;
}

inline bool backend_is_opencl(ggml_backend_t b) {
    return std::strcmp(backend_reg_name(b), "OpenCL") == 0;
}

// Null-safe ASCII case-insensitive substring match: device name capitalisation
// varies across OpenCL CL_DEVICE_NAME and Vulkan deviceName.
inline bool str_contains_ci(const char * hay, const char * needle) {
    if (!hay || !needle) return false;
    for (const char * h = hay; *h; ++h) {
        const char * a = h;
        const char * b = needle;
        while (*a && *b) {
            const char ca = (*a >= 'A' && *a <= 'Z') ? char(*a + 32) : *a;
            const char cb = (*b >= 'A' && *b <= 'Z') ? char(*b + 32) : *b;
            if (ca != cb) break;
            ++a;
            ++b;
        }
        if (!*b) return true;
    }
    return false;
}

// Single matcher for ARM Mali/Immortalis (Valhall), shared by GPU allowlist and pad gate so they can't disagree.
// Valhall miscomputes small-output-dim mul_mat; the st_mul_mat output-pad works around it.
inline bool desc_or_name_is_arm_mali(const char * name, const char * desc) {
    return str_contains_ci(name, "mali")       || str_contains_ci(desc, "mali") ||
           str_contains_ci(name, "immortalis") || str_contains_ci(desc, "immortalis");
}

// Gates the st_mul_mat output-pad: true iff `b` is ggml-vulkan on an ARM Mali/Immortalis GPU.
// DL-safe, no compute: reads device name via core ggml accessors, so it links under GGML_BACKEND_DL=ON.
inline bool backend_is_arm_mali_vulkan(ggml_backend_t b) {
    if (!backend_is_vulkan(b)) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return false;
    return desc_or_name_is_arm_mali(ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
}

inline void backend_set_n_threads(ggml_backend_t b, int n_threads) {
    if (!b || n_threads <= 0) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return;
    auto fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(b, n_threads);
}

} // namespace tts_cpp::detail
