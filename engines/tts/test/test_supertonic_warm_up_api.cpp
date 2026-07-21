// follow-up — CPU-only API-surface test for the
// first-synth pre-warm hook added alongside the Vulkan bring-up:
//
//   - `tts_cpp::supertonic::EngineOptions::prewarm_text` exists,
//     defaults to empty, and accepts a std::string assignment.
//
//   - `tts_cpp::supertonic::Engine::warm_up(const std::string &)`
//     exists in the public API and is callable.
//
// We intentionally don't construct a real `Engine` here — that
// requires a GGUF fixture and the engine surface is exercised
// end-to-end by `test-supertonic-pipeline` (LABEL "fixture").
// This file's job is to lock in the *compile-time contract* of
// the new fields / methods so a future refactor that renames or
// removes them breaks this test before the downstream
// integration / fixture tests have a chance to drift.
//
// The harness compiles + links + runs in <1 ms; on a fresh
// checkout `ctest -L unit` exercises it without any model file.

#include "tts-cpp/supertonic/engine.h"

#include <cstdio>
#include <string>
#include <type_traits>

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

// Test 1 — `prewarm_text` exists, defaults to empty, accepts
// std::string.
//
// Compile-time + runtime: a default-constructed EngineOptions
// has an empty `prewarm_text`, and we can write a non-empty
// string to it without surprises.  This locks in the field's
// type (std::string, not const char*, not std::string_view) and
// default state.
void test_prewarm_text_default_empty() {
    tts_cpp::supertonic::EngineOptions opts;
    CHECK(opts.prewarm_text.empty());

    opts.prewarm_text = "Hello world";
    CHECK(opts.prewarm_text == "Hello world");

    opts.prewarm_text.clear();
    CHECK(opts.prewarm_text.empty());

    static_assert(std::is_same<decltype(opts.prewarm_text), std::string>::value,
                  "EngineOptions::prewarm_text must be std::string");
}

// Test 2 — `Engine::warm_up(const std::string &)` exists in the
// public API.
//
// Asserts the method's existence and signature via SFINAE.  We
// don't actually call it (would require a constructed Engine
// which would need a GGUF fixture); the goal is just to fail
// compilation if the public symbol disappears.
template <typename E, typename = void>
struct has_warm_up : std::false_type {};

template <typename E>
struct has_warm_up<E,
                   std::void_t<decltype(std::declval<E &>().warm_up(std::declval<const std::string &>()))>>
    : std::true_type {};

void test_warm_up_method_exists() {
    static_assert(has_warm_up<tts_cpp::supertonic::Engine>::value,
                  "Engine::warm_up(const std::string &) must exist in the public API");
    CHECK(true);  // tally one runtime check so the harness reports a count
}

// Test 3 — Field-by-field default state of EngineOptions.
//
// Documents the defaults the engine relies on so a regression
// like "prewarm_text accidentally defaults to a hard-coded
// sample text" (which would silently slow down every CPU caller
// by the prewarm cost — even though warm_up is a no-op on CPU,
// the OptionsCheck would surface it in a debug log).
void test_engine_options_defaults() {
    tts_cpp::supertonic::EngineOptions o;
    CHECK(o.model_gguf_path.empty());
    CHECK(o.prewarm_text.empty());
    CHECK(o.vulkan_device == 0);
    // follow-up — the default values for f16_attn /
    // f16_weights are -1 (auto: gated on the new probe set).
    // The probes themselves are exercised by
    // test_supertonic_capability_cache.cpp; here we just lock
    // in the auto-policy default so nobody accidentally flips
    // the engine to "force on" or "force off" by changing the
    // sentinel value.
    CHECK(o.f16_attn    == -1);
    CHECK(o.f16_weights == -1);
}

} // namespace

int main() {
    test_prewarm_text_default_empty();
    test_warm_up_method_exists();
    test_engine_options_defaults();

    std::fprintf(stderr,
                 "test_supertonic_warm_up_api: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
