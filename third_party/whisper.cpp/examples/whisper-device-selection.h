#pragma once

// QVAC (see PATCHES.md): shared device selection for native Whisper examples.
// gpu_device is an ordinal among GPU/IGPU devices, not a registry index.
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace whisper_device_selection {

struct device {
    std::string name;
    enum ggml_backend_dev_type type;
    std::string description;
};

inline bool is_gpu(const device & dev) {
    return dev.type == GGML_BACKEND_DEVICE_TYPE_GPU || dev.type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

inline std::vector<device> devices() {
    std::vector<device> result;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        result.push_back({ggml_backend_dev_name(dev), ggml_backend_dev_type(dev), ggml_backend_dev_description(dev)});
    }
    return result;
}

// The explicit candidate overload also allows tests without GPU hardware.
inline bool resolve(const std::string & target, const std::vector<device> & candidates,
                    int & gpu_device, std::string & error) {
    if (target.empty()) {
        error = "device must be a non-negative GPU index or an exact device name";
        return false;
    }

    int requested_index = -1;
    if (target[0] == '+' || target[0] == '-' || (target[0] >= '0' && target[0] <= '9')) {
        // No partial parses, whitespace, negative values or overflowing indices.
        size_t pos = target[0] == '+' ? 1 : 0;
        int value = 0;
        if (pos == target.size()) {
            error = "invalid GPU device index: '" + target + "'";
            return false;
        }
        for (; pos < target.size(); ++pos) {
            const char c = target[pos];
            if (c < '0' || c > '9' || value > (std::numeric_limits<int>::max() - (c - '0')) / 10) {
                error = "invalid GPU device index: '" + target + "' (expected a non-negative integer)";
                return false;
            }
            value = value * 10 + (c - '0');
        }
        requested_index = value;
    }

    int ordinal = 0;
    int match = -1;
    bool non_gpu_match = false;
    for (const device & dev : candidates) {
        if (!is_gpu(dev)) {
            non_gpu_match = non_gpu_match || dev.name == target;
            continue;
        }
        if ((requested_index >= 0 && requested_index == ordinal) ||
            (requested_index < 0 && dev.name == target)) {
            if (match >= 0) {
                error = "ambiguous GPU device name: '" + target + "'; use its numeric index";
                return false;
            }
            match = ordinal;
        }
        ++ordinal;
    }
    if (match < 0) {
        error = non_gpu_match ? "device '" + target + "' is not a selectable GPU/IGPU device; use --no-gpu for CPU" :
                                "GPU device '" + target + "' is unavailable; use --list-devices";
        return false;
    }
    gpu_device = match;
    error.clear();
    return true;
}

inline bool resolve(const std::string & target, int & gpu_device, std::string & error) {
    return resolve(target, devices(), gpu_device, error);
}

inline bool apply(const std::string & requested, bool explicit_target, bool use_gpu,
                  int & gpu_device, std::string & error, const std::vector<device> & candidates) {
    error.clear();
    if (explicit_target && !use_gpu) {
        error = "--device/--gpu-device cannot be combined with --no-gpu";
        return false;
    }
    if (!use_gpu) {
        return true;
    }
    if (explicit_target) {
        return resolve(requested, candidates, gpu_device, error);
    }
    if (const char * env = std::getenv("WHISPER_ARG_DEVICE")) {
        if (!resolve(env, candidates, gpu_device, error)) {
            error = "WHISPER_ARG_DEVICE: " + error;
            return false;
        }
        return true;
    }
    // Prefer the registered Hexagon device for automatic native CLI selection.
    // Missing/ambiguous HTP0 leaves the caller's existing default unchanged.
    int htp_device = 0;
    std::string ignored;
    if (resolve("HTP0", candidates, htp_device, ignored)) {
        gpu_device = htp_device;
    }
    return true;
}

inline bool apply(const std::string & requested, bool explicit_target, bool use_gpu,
                  int & gpu_device, std::string & error) {
    return apply(requested, explicit_target, use_gpu, gpu_device, error,
                 use_gpu ? devices() : std::vector<device>{});
}

// Call after apply(): an explicit flag/environment request must not silently
// become a CPU projection or inference after device initialization fails.
inline std::string requested_backend(bool explicit_target, bool use_gpu, int gpu_device,
                                     const std::vector<device> & candidates) {
    if (!use_gpu || (!explicit_target && std::getenv("WHISPER_ARG_DEVICE") == nullptr)) {
        return {};
    }
    int ordinal = 0;
    for (const device & dev : candidates) {
        if (is_gpu(dev) && ordinal++ == gpu_device) {
            return dev.name;
        }
    }
    return {};
}

inline std::string requested_backend(bool explicit_target, bool use_gpu, int gpu_device) {
    return requested_backend(explicit_target, use_gpu, gpu_device, devices());
}

inline bool check_primary_backend(const std::string & requested, const char * actual,
                                  std::string & error) {
    error.clear();
    if (!requested.empty() && (actual == nullptr || requested != actual)) {
        error = "requested backend '" + requested + "' did not initialize (primary backend: " +
                (actual ? actual : "none") + ")";
        return false;
    }
    return true;
}

inline void list(FILE * out) {
    fprintf(out, "Available GGML devices (GPU indices count only GPU/IGPU devices):\n");
    int ordinal = 0;
    for (const device & dev : devices()) {
        if (is_gpu(dev)) {
            fprintf(out, "  --device %d or --device \"%s\": %s (%s)\n", ordinal++, dev.name.c_str(),
                    dev.description.c_str(), dev.type == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" : "IGPU");
        } else {
            fprintf(out, "  %s: %s (%s)\n", dev.name.c_str(), dev.description.c_str(),
                    dev.type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU; select with --no-gpu" : "ACCEL; managed automatically");
        }
    }
    if (ordinal == 0) {
        fprintf(out, "  No selectable GPU/IGPU devices found.\n");
    }
}

} // namespace whisper_device_selection
