// No models, Core ML SDK, ggml backend or inference required.
#include "parler/coreml_policy.h"
#include "tts-cpp/parler/engine.h"
#include "tts-cpp/parler/fit.h"
#include "../test_env_portable.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using namespace tts_cpp::parler;
using namespace tts_cpp::parler::detail;

namespace {
int failures = 0;
void check(bool ok, const char * message) {
    fprintf(stderr, "[%s] %s\n", ok ? "PASS" : "FAIL", message);
    if (!ok) ++failures;
}
struct saved_env {
    const char * name;
    std::optional<std::string> value;
    explicit saved_env(const char * key) : name(key) {
        if (const char * v = std::getenv(key)) value = v;
        unsetenv(key);
    }
    ~saved_env() {
        if (value) setenv(name, value->c_str(), 1);
        else unsetenv(name);
    }
};
parler_coreml_policy resolve(const EngineOptions & options) {
    return parler_resolve_coreml_policy(options.coreml_compute_units, options.coreml_strict);
}
}

int main() {
    saved_env units(PARLER_COREML_UNITS_ENV), strict(PARLER_COREML_STRICT_ENV),
              disabled(PARLER_COREML_DISABLE_ENV);
    EngineOptions options;
    const auto defaults = resolve(options);
    check(options.coreml_compute_units.empty() && !options.coreml_strict.has_value() &&
          !options.coreml_decoder && !FitOptions{}.coreml_decoder,
          "public defaults inherit policy and keep Engine/Fit decoder opt-in false");
    check(defaults.compute_units == "all" && !defaults.strict && !defaults.disabled,
          "unset environment resolves legacy defaults");

    for (const char * value : {"all", "cpu_and_ane", "cpu_and_gpu", "cpu_only"}) {
        setenv(PARLER_COREML_UNITS_ENV, value, 1);
        check(resolve(options).compute_units == value, "each supported environment unit is inherited");
        options.coreml_compute_units = value;
        setenv(PARLER_COREML_UNITS_ENV, "invalid-env", 1);
        check(resolve(options).compute_units == value, "each explicit unit overrides environment");
        options.coreml_compute_units.clear();
    }
    check(resolve(options).compute_units == "all", "invalid inherited units retain DAC warning/default behavior");
    check(resolve(options).invalid_inherited_units == "invalid-env",
          "invalid inherited units are retained for decoder rejection");
    setenv(PARLER_COREML_UNITS_ENV, "", 1);
    check(resolve(options).compute_units == "all", "empty inherited units use all");
    for (const char * value : {"1", "0", "false"}) {
        setenv(PARLER_COREML_STRICT_ENV, value, 1);
        check(resolve(options).strict, "legacy strict environment uses presence, even 0/false");
        options.coreml_strict = false;
        check(!resolve(options).strict, "explicit false overrides inherited strict true");
        options.coreml_strict.reset();
    }
#ifndef _MSC_VER
    setenv(PARLER_COREML_STRICT_ENV, "", 1);
    check(resolve(options).strict, "empty but present strict environment is true");
#endif
    unsetenv(PARLER_COREML_STRICT_ENV);
    options.coreml_strict = true;
    check(resolve(options).strict, "explicit true overrides absent strict environment");

    options.coreml_compute_units = "cpu_only";
    const auto first = resolve(options);
    setenv(PARLER_COREML_UNITS_ENV, "cpu_and_ane", 1);
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    options.coreml_compute_units.clear();
    options.coreml_strict = false;
    const auto second = resolve(options);
    check(first.compute_units == "cpu_only" && first.strict &&
          second.compute_units == "cpu_and_ane" && !second.strict,
          "independent captured policies survive environment and option changes");
    check(std::string(std::getenv(PARLER_COREML_UNITS_ENV)) == "cpu_and_ane" &&
          std::string(std::getenv(PARLER_COREML_STRICT_ENV)) == "1",
          "explicit options never modify process environment");

    setenv(PARLER_COREML_DISABLE_ENV, "0", 1);
    check(resolve(options).disabled && !first.disabled && !second.disabled,
          "global disable uses legacy presence semantics and is captured at construction");
    for (const char * value : {"ane", "ALL", "cpu_only ", "auto"}) {
        options.coreml_compute_units = value;
        bool rejected = false;
        try { resolve(options); }
        catch (const std::runtime_error & e) {
            const std::string error = e.what();
            rejected = error.find("coreml_compute_units") != std::string::npos &&
                       error.find(value) != std::string::npos &&
                       error.find("cpu_and_ane") != std::string::npos;
        }
        check(rejected, "invalid explicit units reject with value and allowed names even when disabled");
    }
    unsetenv(PARLER_COREML_DISABLE_ENV);

    // Parallel instances use different explicit policies against one stable
    // environment. No test thread writes environment variables.
    std::atomic<bool> isolated{true};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([i, &isolated] {
            EngineOptions local;
            local.coreml_compute_units = i % 2 ? "cpu_only" : "cpu_and_gpu";
            local.coreml_strict = i % 2 != 0;
            for (int n = 0; n < 1000; ++n) {
                const auto policy = resolve(local);
                if (policy.compute_units != local.coreml_compute_units ||
                    policy.strict != *local.coreml_strict || policy.disabled) isolated = false;
            }
        });
    }
    for (auto & thread : threads) thread.join();
    check(isolated && std::string(std::getenv(PARLER_COREML_UNITS_ENV)) == "cpu_and_ane" &&
          std::string(std::getenv(PARLER_COREML_STRICT_ENV)) == "1",
          "concurrent explicit instance policies remain independent without environment writes");
    return failures ? 1 : 0;
}
