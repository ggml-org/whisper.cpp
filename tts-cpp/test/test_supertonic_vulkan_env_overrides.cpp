// round 7 — CPU-only TDD test for the Vulkan env-var
// passthrough mechanism.
//
// Background
// ----------
// ggml-vulkan reads numerous `GGML_VK_*` env vars at backend init
// time to configure adapter selection, coopmat / bf16 toggles, the
// perf logger, etc.  Operators currently have to set these env
// vars in the shell before invoking supertonic-cli / tts-cli /
// supertonic-bench, which is awkward when the env is managed by a
// service supervisor or when the operator wants to A/B-compare
// settings without losing their shell state.
//
// Round 7 adds:
//
//   1. A new `EngineOptions::vulkan_env_overrides` field
//      (std::map<std::string, std::string>) that the engine
//      applies just before backend init.
//
//   2. A public helper `apply_vulkan_env_overrides(map)` declared
//      in `supertonic_internal.h`, defined in `supertonic_gguf.cpp`,
//      that:
//        - validates each key starts with `GGML_VK_`
//          (throws std::runtime_error on a bad key — guards
//           against operator-config typos like
//           `GMML_VK_PREFER_HOST_MEMORY`);
//        - calls `set_env_if_unset(key, value)` so an
//          operator-set env var still wins over the EngineOptions
//          override (lets operators force a setting from the
//          shell without recompiling).
//
//   3. CLI flags on supertonic-cli / tts-cli / supertonic-bench
//      that map friendly names to `GGML_VK_*` env var keys:
//
//        --vulkan-prefer-host-memory  → GGML_VK_PREFER_HOST_MEMORY=1
//        --vulkan-disable-coopmat2    → GGML_VK_DISABLE_COOPMAT2=1
//        --vulkan-disable-bfloat16    → GGML_VK_DISABLE_BFLOAT16=1
//        --vulkan-perf-logger         → GGML_VK_PERF_LOGGER=1
//        --vulkan-async-transfer      → GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1
//
//      Each flag inserts the corresponding entry into
//      EngineOptions::vulkan_env_overrides; the engine then
//      applies them via `apply_vulkan_env_overrides()` before
//      `init_supertonic_backend()` runs.
//
// This test is the TDD gate for the EngineOptions field + the
// public helper.  CLI parsing is exercised by separate smoke
// tests on each binary's `--help` output (visual; no test gate
// — same as every other CLI flag added in rounds 1-6).
//
// Whole TU MUST fail to compile before the symbols are added,
// then pass after.

#include "tts-cpp/supertonic/engine.h"
#include "supertonic_internal.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

using tts_cpp::supertonic::detail::apply_vulkan_env_overrides;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

template <typename F>
bool throws_runtime_error(F && fn) {
    try { fn(); return false; }
    catch (const std::runtime_error &) { return true; }
    catch (...) { return false; }
}

// SFINAE: assert the EngineOptions field exists.
template <typename T>
auto has_vulkan_env_overrides(int) -> decltype(
    std::declval<T &>().vulkan_env_overrides, std::true_type{});
template <typename T>
auto has_vulkan_env_overrides(...) -> std::false_type;

void unsetenv_safe(const char * name) {
#if defined(_WIN32)
    _putenv_s(name, "");  // empty value treated as unset by ggml-vulkan's getenv check
#else
    unsetenv(name);
#endif
}

// Test 1 — `EngineOptions::vulkan_env_overrides` field exists and
// has the expected type, default-constructs empty, accepts
// assignment.
void test_engine_options_field_exists() {
    using namespace tts_cpp::supertonic;
    static_assert(
        decltype(has_vulkan_env_overrides<EngineOptions>(0))::value,
        "EngineOptions must declare vulkan_env_overrides "
        "(std::map<std::string, std::string>)");

    EngineOptions opts;
    CHECK(opts.vulkan_env_overrides.empty());

    opts.vulkan_env_overrides["GGML_VK_PREFER_HOST_MEMORY"] = "1";
    opts.vulkan_env_overrides["GGML_VK_DISABLE_COOPMAT2"]   = "1";
    CHECK(opts.vulkan_env_overrides.size() == 2);
    CHECK(opts.vulkan_env_overrides["GGML_VK_PREFER_HOST_MEMORY"] == "1");

    // Round-3 + round-4 + round-6 baseline regression guard.
    EngineOptions baseline;
    CHECK(baseline.vulkan_env_overrides.empty());
    CHECK(baseline.kv_attn_type == -1);
    CHECK(baseline.f16_attn == -1);
    CHECK(baseline.f16_weights == -1);
    CHECK(baseline.f16_weights_deny_list.empty());
    CHECK(baseline.vulkan_device == 0);
    CHECK(baseline.prewarm_text.empty());
}

// Test 2 — `apply_vulkan_env_overrides({})` is a no-op (regression
// guard against the helper accidentally touching the env on the
// default empty path).
void test_empty_map_is_noop() {
    // Pre-condition: a unique, never-set env var must read back null.
    const char * unique = "GGML_VK_TEST_R7_EMPTY_NOOP_KEY";
    unsetenv_safe(unique);
    CHECK(std::getenv(unique) == nullptr);

    std::map<std::string, std::string> empty;
    apply_vulkan_env_overrides(empty);

    // Helper must NOT have invented a value for our unique key.
    CHECK(std::getenv(unique) == nullptr);
}

// Test 3 — `apply_vulkan_env_overrides({{"GGML_VK_*", "v"}})` calls
// `set_env_if_unset` so the env var becomes set on a clean env.
void test_single_entry_sets_env() {
    const char * key = "GGML_VK_TEST_R7_SETS_ENV";
    unsetenv_safe(key);
    CHECK(std::getenv(key) == nullptr);

    apply_vulkan_env_overrides({{key, "value_a"}});

    const char * actual = std::getenv(key);
    CHECK(actual != nullptr);
    if (actual) CHECK(std::string(actual) == "value_a");

    unsetenv_safe(key);
}

// Test 4 — operator-set env wins over the EngineOptions override.
//
// This pins the `set_env_if_unset` semantics: an operator who
// has already exported `GGML_VK_DISABLE_COOPMAT2=0` in their shell
// must NOT have it overwritten by an EngineOptions override.
// Lets a debugging operator force-disable a setting from the
// command line without recompiling.
void test_operator_env_wins() {
    const char * key = "GGML_VK_TEST_R7_OPERATOR_WINS";
#if defined(_WIN32)
    _putenv_s(key, "operator_set");
#else
    setenv(key, "operator_set", 1);
#endif
    CHECK(std::string(std::getenv(key) ? std::getenv(key) : "") == "operator_set");

    apply_vulkan_env_overrides({{key, "engine_override"}});

    const char * after = std::getenv(key);
    CHECK(after != nullptr);
    if (after) CHECK(std::string(after) == "operator_set");

    unsetenv_safe(key);
}

// Test 5 — invalid key (no `GGML_VK_` prefix) throws.
//
// Loud-failure for operator-config typos — same convention as
// `--kv-attn-type bogus` (round 4) and `--vulkan-device -2`
// (round 3 reserved-negative throw).  An operator that types
// `GMML_VK_PREFER_HOST_MEMORY` in their config gets a clean
// error message instead of silently setting an env var that
// ggml-vulkan won't read.
void test_invalid_key_throws() {
    CHECK(throws_runtime_error([] {
        apply_vulkan_env_overrides({{"GMML_VK_PREFER_HOST_MEMORY", "1"}});
    }));
    CHECK(throws_runtime_error([] {
        apply_vulkan_env_overrides({{"PATH", "1"}});
    }));
    CHECK(throws_runtime_error([] {
        apply_vulkan_env_overrides({{"", "1"}});
    }));
    CHECK(throws_runtime_error([] {
        apply_vulkan_env_overrides({{"GGML_", "1"}});  // close but missing _VK_
    }));
    CHECK(throws_runtime_error([] {
        apply_vulkan_env_overrides({{"GGML_VK", "1"}}); // missing trailing underscore
    }));
}

// Test 6 — when a single bad entry is in a map with several good
// entries, the throw fires AT the bad entry; the helper must NOT
// silently apply the good entries before the throw lands (ALL or
// NOTHING semantics so a partial-success doesn't leave the env
// in a half-applied state).
void test_all_or_nothing_on_invalid_key() {
    const char * good_a = "GGML_VK_TEST_R7_AON_A";
    const char * good_b = "GGML_VK_TEST_R7_AON_B";
    unsetenv_safe(good_a);
    unsetenv_safe(good_b);

    std::map<std::string, std::string> mixed = {
        {good_a, "1"},
        {"BAD_KEY", "should_throw"},
        {good_b, "1"},
    };
    CHECK(throws_runtime_error([&] {
        apply_vulkan_env_overrides(mixed);
    }));

    // Neither good key should have been applied.
    CHECK(std::getenv(good_a) == nullptr);
    CHECK(std::getenv(good_b) == nullptr);
}

// Test 7 — multi-entry happy path.
void test_multi_entry_all_applied() {
    const char * a = "GGML_VK_TEST_R7_MULTI_A";
    const char * b = "GGML_VK_TEST_R7_MULTI_B";
    const char * c = "GGML_VK_TEST_R7_MULTI_C";
    unsetenv_safe(a);
    unsetenv_safe(b);
    unsetenv_safe(c);

    apply_vulkan_env_overrides({
        {a, "alpha"},
        {b, "beta"},
        {c, "gamma"},
    });

    CHECK(std::string(std::getenv(a) ? std::getenv(a) : "") == "alpha");
    CHECK(std::string(std::getenv(b) ? std::getenv(b) : "") == "beta");
    CHECK(std::string(std::getenv(c) ? std::getenv(c) : "") == "gamma");

    unsetenv_safe(a);
    unsetenv_safe(b);
    unsetenv_safe(c);
}

} // namespace

int main() {
    test_engine_options_field_exists();
    test_empty_map_is_noop();
    test_single_entry_sets_env();
    test_operator_env_wins();
    test_invalid_key_throws();
    test_all_or_nothing_on_invalid_key();
    test_multi_entry_all_applied();

    std::fprintf(stderr,
                 "test_supertonic_vulkan_env_overrides: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
