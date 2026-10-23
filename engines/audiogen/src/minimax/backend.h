#pragma once

#include "acestep/backend_registry.h"
#include "ggml-backend.h"
#include "logic.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

struct BackendPair {
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    bool has_gpu = false;
};

inline BackendPair g_backend_cache = {};
inline int g_backend_refs = 0;
inline int g_backend_threads = 1;
inline std::string g_backend_modules_dir;
inline std::string g_backend_device = "cpu";

// Set by backend_init on the one shared acquisition; read back through
// Engine::gpu_fallback_reason().
inline tts_cpp::GpuFallbackReason g_backend_gpu_fallback_reason = tts_cpp::GpuFallbackReason::not_requested;

static void backend_configure_cpu(int n_threads, const std::string & modules_dir) {
    const int requested_threads = n_threads > 0 ? n_threads : 1;
    if (!tts_cpp::minimax::detail::backend_configuration_matches(
            g_backend_refs, g_backend_threads, g_backend_modules_dir, requested_threads, modules_dir)) {
        throw std::runtime_error(
            "minimax engine: active CPU backend uses different thread or backend directory options");
    }
    g_backend_threads = requested_threads;
    g_backend_modules_dir = modules_dir;
}

// Resolve the requested compute device: an explicit option wins, then the
// MM3_DEVICE environment variable, then "cpu". Values: cpu | gpu | auto.
static std::string backend_resolve_device_request(const std::string & device) {
    if (!device.empty()) {
        return device;
    }
    const char * env = std::getenv("MM3_DEVICE");
    return env && *env ? env : "cpu";
}

static bool backend_device_request_valid(const std::string & device) {
    return device == "cpu" || device == "gpu" || device == "auto";
}

static void backend_configure_device(const std::string & device) {
    const std::string requested = backend_resolve_device_request(device);
    if (!backend_device_request_valid(requested)) {
        throw std::runtime_error("minimax engine: device must be cpu, gpu or auto, got '" + requested + "'");
    }
    if (g_backend_refs > 0 && requested != g_backend_device) {
        throw std::runtime_error("minimax engine: active backend already initialised with device '" +
                                 g_backend_device + "'");
    }
    g_backend_device = requested;
}

static void backend_set_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    if (!registry) {
        return;
    }
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
    if (set_threads) {
        set_threads(backend, n_threads);
    }
}

static void backend_set_abort_handler(ggml_backend_t backend, ggml_abort_callback callback, void * user_data) {
    if (!backend) {
        return;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    if (!registry) {
        return;
    }
    auto set_abort = reinterpret_cast<ggml_backend_set_abort_callback_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_abort_callback"));
    if (set_abort) {
        set_abort(backend, callback, user_data);
    }
}

static ggml_backend_t backend_create_cpu() {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (backend) {
        backend_set_threads(backend, g_backend_threads);
    }
    return backend;
}

static void backend_load_modules(const std::string & modules_dir) {
    if (!modules_dir.empty()) {
        ggml_backend_load_all_from_path(modules_dir.c_str());
    } else {
        ggml_backend_load_all();
    }
}

static BackendPair backend_create_pair(const std::string & device, const char * tag,
                                       tts_cpp::GpuFallbackReason * fallback_reason) {
    BackendPair pair;
    pair.cpu_backend = backend_create_cpu();
    pair.backend = pair.cpu_backend;
    if (!pair.cpu_backend) {
        throw std::runtime_error("minimax engine: CPU backend initialization failed");
    }
    if (device == "gpu" || device == "auto") {
        if (ggml_backend_t gpu = tts_cpp::acestep::backend_gpu_init(fallback_reason)) {
            pair.backend = gpu;
            pair.has_gpu = true;
            fprintf(stderr, "[%s] Using GPU backend %s (%s); CPU handles unsupported ops\n", tag,
                    tts_cpp::acestep::backend_reg_name(gpu), ggml_backend_name(gpu));
        } else if (device == "gpu") {
            // An explicit GPU request must not silently degrade into a run that
            // is orders of magnitude slower; only device=auto may fall back.
            const char * reason = tts_cpp::gpu_fallback_reason_name(*fallback_reason);
            ggml_backend_free(pair.cpu_backend);
            throw std::runtime_error(
                std::string("minimax engine: device=gpu but no usable GPU backend was found (") + reason +
                "); use device=auto for CPU fallback");
        } else {
            fprintf(stderr, "[%s] device=auto found no usable GPU backend (%s); using CPU\n", tag,
                    tts_cpp::gpu_fallback_reason_name(*fallback_reason));
        }
    } else {
        *fallback_reason = tts_cpp::GpuFallbackReason::not_requested;
    }
    return pair;
}

static void backend_free_pair(BackendPair & pair) {
    if (pair.backend && pair.backend != pair.cpu_backend) {
        ggml_backend_free(pair.backend);
    }
    if (pair.cpu_backend) {
        ggml_backend_free(pair.cpu_backend);
    }
    pair = {};
}

static BackendPair backend_init(const char * tag) {
    if (g_backend_refs > 0) {
        ++g_backend_refs;
        return g_backend_cache;
    }
    backend_load_modules(g_backend_modules_dir);
    BackendPair pair = backend_create_pair(g_backend_device, tag, &g_backend_gpu_fallback_reason);
    g_backend_cache = pair;
    g_backend_refs = 1;
    return pair;
}

static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    if (g_backend_refs <= 0) {
        return;
    }
    --g_backend_refs;
    if (g_backend_refs != 0) {
        return;
    }
    BackendPair pair = { backend, cpu_backend, backend && backend != cpu_backend };
    backend_free_pair(pair);
    g_backend_cache = {};
}

static ggml_backend_sched_t backend_sched_new(BackendPair pair, int max_nodes) {
    ggml_backend_t backends[2] = {pair.backend, pair.cpu_backend};
    ggml_backend_buffer_type_t buffer_types[2] = {ggml_backend_get_default_buffer_type(pair.backend),
                                                  ggml_backend_get_default_buffer_type(pair.cpu_backend)};
    const int n_backends = pair.has_gpu ? 2 : 1;
    ggml_backend_sched_t scheduler =
        ggml_backend_sched_new(backends, buffer_types, n_backends, max_nodes, false, true);
    if (!scheduler) {
        throw std::runtime_error("minimax engine: scheduler initialization failed");
    }
    return scheduler;
}
