// Compute-unit selection shared by the Core ML encoder and decoder wrappers.
//
// Apple-only, Objective-C++ header: included by the .mm wrappers under
// PARAKEET_USE_COREML only.

#pragma once

#import <CoreML/CoreML.h>

#include <cstdlib>
#include <cstring>
#include <string>

// Resolves the compute units an `.mlmodelc` is loaded with from the
// environment variable `units_env` (`all`, `cpu_and_gpu`, `cpu_only`,
// `cpu_and_ane`), writing the matching stats label.
//
// Without an override, `prefer_neural_engine` picks CPU and Neural Engine:
// the graphs exported for the Neural Engine (the fixed-window encoder and the
// batched TDT decoder) are fully ANE-shaped, and MLComputeUnitsAll is only a
// preference that may resolve to the GPU. The generic encoder exports keep
// MLComputeUnitsAll, since their few GPU-preferred operations would otherwise
// fall back to the CPU.
inline MLComputeUnits parakeet_coreml_requested_units(const char * units_env,
                                                      bool prefer_neural_engine,
                                                      std::string * label) {
    const char * env = units_env != nullptr ? std::getenv(units_env) : nullptr;
    if (env != nullptr && std::strcmp(env, "all") == 0) {
        *label = "coreml-all";
        return MLComputeUnitsAll;
    }
    if (env != nullptr && std::strcmp(env, "cpu_and_gpu") == 0) {
        *label = "coreml-cpu-gpu";
        return MLComputeUnitsCPUAndGPU;
    }
    if (env != nullptr && std::strcmp(env, "cpu_only") == 0) {
        *label = "coreml-cpu";
        return MLComputeUnitsCPUOnly;
    }
    const bool want_ane =
        prefer_neural_engine || (env != nullptr && std::strcmp(env, "cpu_and_ane") == 0);
    if (want_ane) {
        if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
            *label = "coreml-ane";
            return MLComputeUnitsCPUAndNeuralEngine;
        }
    }
    *label = "coreml-all";
    return MLComputeUnitsAll;
}
