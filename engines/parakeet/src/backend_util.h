#pragma once

// Backend-introspection helpers that work uniformly under both
// GGML_BACKEND_DL=ON and GGML_BACKEND_DL=OFF. The legacy
// ggml_backend_is_cpu / ggml_backend_is_metal entry points live in
// the per-backend shared libraries (libggml-cpu.* / libggml-metal.*),
// so they are unlinkable from libparakeet under the dynamic-loader
// build mode embedded host applications typically ship with. Routing
// through the registry (ggml_backend_get_device + ggml_backend_dev_*)
// reaches the same answer in both modes.

#include "ggml-backend.h"

#include <cstring>

namespace parakeet {

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

}
