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

// Registry feature lookup via ggml_backend_get_features ("AVX2", "NEON", "LLAMAFILE", ...).
inline bool backend_reg_has_feature(ggml_backend_reg_t reg, const char * name) {
    if (!reg || !name) return false;
    auto get_features = (ggml_backend_get_features_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (!get_features) return false;
    for (const ggml_backend_feature * f = get_features(reg); f && f->name; ++f) {
        if (std::strcmp(f->name, name) == 0) return true;
    }
    return false;
}

inline bool backend_has_feature(ggml_backend_t b, const char * name) {
    if (!b) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    return dev && backend_reg_has_feature(ggml_backend_dev_backend_reg(dev), name);
}

inline bool cpu_backend_has_feature(const char * name) {
    return backend_reg_has_feature(ggml_backend_reg_by_name("CPU"), name);
}

// False with tinyBLAS (GGML_LLAMAFILE), whose GEMM rounding depends on the operand shapes.
inline bool cpu_matmul_is_shape_exact() {
    return !cpu_backend_has_feature("LLAMAFILE");
}

// The name-level predicates exist because device selection knows a registry name
// before it has a backend handle to ask, and both readings have to agree.
inline bool reg_name_is_metal(const char * n) {
    // ggml-metal's registry name is "MTL" (GGML_METAL_NAME); older ggml reported "Metal".
    return n && (std::strcmp(n, "MTL") == 0 || std::strcmp(n, "Metal") == 0);
}

inline bool reg_name_is_vulkan(const char * n) {
    return n && std::strcmp(n, "Vulkan") == 0;
}

inline bool reg_name_is_opencl(const char * n) {
    return n && std::strcmp(n, "OpenCL") == 0;
}

inline bool reg_name_is_cuda(const char * n) {
    return n && std::strcmp(n, "CUDA") == 0;
}

// ggml-webgpu registers as "WebGPU" (GGML_WEBGPU_NAME) and names its single
// device the same way.
inline bool reg_name_is_webgpu(const char * n) {
    return n && std::strcmp(n, "WebGPU") == 0;
}

inline bool reg_name_is_hexagon(const char * n) {
    return n && std::strcmp(n, "HTP") == 0;
}

// Which backend a resident tensor's buffer belongs to. Graph builders see the
// weight but not the backend that holds it, and the buffer carries the answer.
inline const char * buft_reg_name(ggml_backend_buffer_type_t buft) {
    if (!buft) return "";
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) return "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return "";
    const char * n = ggml_backend_reg_name(reg);
    return n ? n : "";
}

inline bool backend_is_metal(ggml_backend_t b) {
    return reg_name_is_metal(backend_reg_name(b));
}

inline bool backend_is_vulkan(ggml_backend_t b) {
    return reg_name_is_vulkan(backend_reg_name(b));
}

inline bool backend_is_opencl(ggml_backend_t b) {
    return reg_name_is_opencl(backend_reg_name(b));
}

inline bool backend_is_cuda(ggml_backend_t b) {
    return reg_name_is_cuda(backend_reg_name(b));
}

inline bool backend_is_webgpu(ggml_backend_t b) {
    return reg_name_is_webgpu(backend_reg_name(b));
}

inline bool backend_is_hexagon(ggml_backend_t b) {
    return reg_name_is_hexagon(backend_reg_name(b));
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

// True iff `b` is ggml-vulkan on an NVIDIA GPU.  NVIDIA's matmul accumulates
// the same way for either operand order, so the channel-major Supertonic step
// can keep the weight as src0 there; other Vulkan drivers (measured on RADV)
// reorder the reduction and need the activation-first form to stay on the F32
// reference.  DL-safe: device name only, no compute.
inline bool backend_is_nvidia_vulkan(ggml_backend_t b) {
    if (!backend_is_vulkan(b)) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return false;
    return str_contains_ci(ggml_backend_dev_name(dev), "nvidia") ||
           str_contains_ci(ggml_backend_dev_description(dev), "nvidia");
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
