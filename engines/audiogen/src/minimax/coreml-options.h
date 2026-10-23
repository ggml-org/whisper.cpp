#pragma once

#include "audiogen-cpp/minimax/engine.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>

namespace tts_cpp::minimax::detail {

inline bool coreml_requested(const EngineOptions & options) {
    return !options.coreml_profile_dir.empty() || !options.coreml_compute_units.empty();
}

inline bool coreml_compute_units_valid(std::string_view value) {
    constexpr std::array<std::string_view, 4> values = {"CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"};
    return std::find(values.begin(), values.end(), value) != values.end();
}

inline void validate_coreml_options(const EngineOptions & options) {
    if (!coreml_requested(options)) {
        return;
    }
    if (options.coreml_profile_dir.empty() || options.coreml_compute_units.empty()) {
        throw std::invalid_argument("MiniMax Core ML requires both "
                                    "coreml_profile_dir and coreml_compute_units");
    }
    if (options.coreml_profile_dir.find('\0') != std::string::npos) {
        throw std::invalid_argument("MiniMax Core ML profile path contains a null character");
    }
    if (!coreml_compute_units_valid(options.coreml_compute_units)) {
        throw std::invalid_argument("MiniMax coreml_compute_units must be "
                                    "CPU_ONLY, CPU_AND_GPU, CPU_AND_NE or ALL");
    }
}

} // namespace tts_cpp::minimax::detail
