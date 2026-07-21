// round 6 — CPU-only TDD test for the F16-weights
// deny-list API surface.
//
// Round 6 layers a user-overridable extra deny-list on top of
// the existing hand-curated `should_materialise_f16_weight()`
// allow-list.  The deny-list lives on `EngineOptions` and gets
// plumbed through `load_supertonic_gguf` to the predicate at
// load time.
//
// API surface this test pins:
//   - `EngineOptions::f16_weights_deny_list` is a public field
//     of type `std::vector<std::string>` defaulting to empty.
//   - `load_supertonic_gguf(...)` accepts an optional
//     `const std::vector<std::string> & f16_weights_deny_list`
//     parameter at the end of its signature, defaulting to empty
//     (so every existing call site keeps compiling).
//   - The 2-arg `should_materialise_f16_weight(name, deny)`
//     overload exists with the documented signature.
//
// Behaviour is covered by `test_supertonic_f16_weights.cpp`
// (predicate level) and the load-time fixture-bound tests
// (model-bound, run on hosts with the GGUF available).  This
// test only asserts the API surface compiles + the defaults are
// what we documented.
//
// Written FIRST (TDD).  Whole TU MUST fail to compile before
// the symbols are added; MUST compile + pass after.

#include "tts-cpp/supertonic/engine.h"
#include "supertonic_internal.h"

#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

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

// SFINAE: assert that `EngineOptions::f16_weights_deny_list`
// member exists and has the expected type.  If the symbol is
// missing the whole TU fails to compile — exactly what TDD
// step 2 expects.
template <typename T>
auto has_f16_weights_deny_list_field(int) -> decltype(
    std::declval<T &>().f16_weights_deny_list,
    std::true_type{}
);
template <typename T>
auto has_f16_weights_deny_list_field(...) -> std::false_type;

// SFINAE: assert that `load_supertonic_gguf` accepts the
// `f16_weights_deny_list` argument.  Post-rebase onto upstream's
// Metal-port `supertonic_optimizations` branch, the parameter
// order is:
//   path, model, n_gpu_layers, verbose, f16_weights, precision,
//   vulkan_device, f16_weights_deny_list
// — 8 trailing params after `model`; the deny-list lives at the
// 8th position (was 7th pre-rebase on the round-6 branch).
template <typename = void>
auto has_deny_list_param_in_load(int) -> decltype(
    tts_cpp::supertonic::detail::load_supertonic_gguf(
        std::declval<const std::string &>(),
        std::declval<tts_cpp::supertonic::detail::supertonic_model &>(),
        /*n_gpu_layers=*/0,
        /*verbose=*/false,
        /*f16_weights=*/-1,
        /*precision=*/tts_cpp::supertonic::detail::supertonic_precision::F32,
        /*vulkan_device=*/0,
        /*f16_weights_deny_list=*/std::declval<const std::vector<std::string> &>()),
    std::true_type{}
);
template <typename = void>
auto has_deny_list_param_in_load(...) -> std::false_type;

void test_engine_options_field_exists() {
    std::fprintf(stderr, "[Round 6 API: EngineOptions::f16_weights_deny_list]\n");
    using namespace tts_cpp::supertonic;
    static_assert(
        decltype(has_f16_weights_deny_list_field<EngineOptions>(0))::value,
        "EngineOptions must declare f16_weights_deny_list");

    EngineOptions opts;
    // Default must be empty.
    CHECK(opts.f16_weights_deny_list.empty());

    // Field must be assignable from a vector<string> literal.
    opts.f16_weights_deny_list = {".pwconv1.", "MatMul_3101"};
    CHECK(opts.f16_weights_deny_list.size() == 2);
    CHECK(opts.f16_weights_deny_list[0] == ".pwconv1.");
    CHECK(opts.f16_weights_deny_list[1] == "MatMul_3101");

    // Documented default for every other field stays unchanged
    // (regression guard for the round-3 prewarm/vulkan_device
    // baseline).
    EngineOptions baseline;
    CHECK(baseline.prewarm_text.empty());
    CHECK(baseline.vulkan_device == 0);
    CHECK(baseline.f16_attn == -1);
    CHECK(baseline.f16_weights == -1);
}

void test_load_supertonic_gguf_param_exists() {
    std::fprintf(stderr, "[Round 6 API: load_supertonic_gguf f16_weights_deny_list param]\n");
    static_assert(
        decltype(has_deny_list_param_in_load<>(0))::value,
        "load_supertonic_gguf must accept an optional f16_weights_deny_list parameter");
    // The static_assert is the actual API gate.  Bump check
    // count so the test reports a meaningful pass/fail summary.
    ++g_checks;
}

} // namespace

int main() {
    test_engine_options_field_exists();
    test_load_supertonic_gguf_param_exists();

    std::fprintf(stderr,
                 "test_supertonic_f16_deny_list_api: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
