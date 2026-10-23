#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace tts_cpp { namespace parler { namespace detail {

constexpr const char * PARLER_COREML_DISABLE_ENV = "PARLER_COREML_DISABLE";
constexpr const char * PARLER_COREML_STRICT_ENV = "PARLER_COREML_STRICT";
constexpr const char * PARLER_COREML_UNITS_ENV = "PARLER_COREML_COMPUTE_UNITS";
constexpr const char * PARLER_COREML_DEFAULT_UNITS = "all";

// Resolve once per Engine, before opening either sidecar. Explicit options
// never write the environment; inherited values are copied, not read again
// during synthesis. DISABLE remains a process-wide load-time kill switch.
struct parler_coreml_policy {
    std::string compute_units = PARLER_COREML_DEFAULT_UNITS;
    bool strict = false;
    bool disabled = false;
    std::string invalid_inherited_units; // DAC defaults to all; decoder retains its legacy rejection
};

inline bool parler_coreml_valid_units(const std::string & units) {
    return units == "all" || units == "cpu_and_ane" ||
           units == "cpu_and_gpu" || units == "cpu_only";
}

inline parler_coreml_policy parler_resolve_coreml_policy(
    const std::string & compute_units = {}, std::optional<bool> strict = std::nullopt) {
    parler_coreml_policy policy;
    policy.disabled = std::getenv(PARLER_COREML_DISABLE_ENV) != nullptr;
    policy.strict = strict.has_value() ? *strict : std::getenv(PARLER_COREML_STRICT_ENV) != nullptr;
    if (!compute_units.empty()) {
        if (!parler_coreml_valid_units(compute_units)) {
            throw std::runtime_error("parler: invalid coreml_compute_units='" + compute_units +
                                     "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
        }
        policy.compute_units = compute_units;
    } else if (const char * env = std::getenv(PARLER_COREML_UNITS_ENV); env && *env) {
        // Retain the DAC's legacy warning-and-default behavior for inherited
        // values, now shared by the decoder as well.
        policy.compute_units = env;
        if (!parler_coreml_valid_units(policy.compute_units)) {
            fprintf(stderr, "parler: ignoring %s=%s (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)\n",
                    PARLER_COREML_UNITS_ENV, policy.compute_units.c_str());
            policy.invalid_inherited_units = policy.compute_units;
            policy.compute_units = PARLER_COREML_DEFAULT_UNITS;
        }
    }
    return policy;
}

}}} // namespace tts_cpp::parler::detail
