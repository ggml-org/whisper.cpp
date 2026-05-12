// QVAC-18605 round 4 — CPU-only TDD test for the multi-dtype
// K/V flash-attention API surface.
//
// Pins:
//   1. `EngineOptions::kv_attn_type` int field exists, defaults to -1
//      (auto), and accepts assignment to the documented values
//      0..3 (f32, f16, bf16, q8_0).
//   2. `supertonic_model::kv_attn_type` (`detail::kv_attn_dtype`)
//      field exists, defaults to `kv_attn_dtype::f32` (no
//      surprise dispatch on a default-constructed model).
//   3. `supertonic_kv_attn_type()` thread-local accessor exists
//      and returns the currently-active dispatch dtype.  Default
//      (no scope active) is `kv_attn_dtype::f32`.
//   4. `supertonic_op_dispatch_scope::prev_kv_attn_type` field
//      exists so the RAII teardown restores the right value.
//   5. The round-3 baseline EngineOptions defaults
//      (prewarm_text empty, vulkan_device 0, f16_attn -1,
//      f16_weights -1, f16_weights_deny_list empty) are unchanged
//      — regression guard against accidental ABI churn.
//
// Whole TU MUST fail to compile before the symbols are added,
// then pass after.

#include "tts-cpp/supertonic/engine.h"
#include "supertonic_internal.h"

#include <cstdio>
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

// SFINAE: assert the EngineOptions field exists.
template <typename T>
auto has_kv_attn_type_field(int) -> decltype(
    std::declval<T &>().kv_attn_type, std::true_type{});
template <typename T>
auto has_kv_attn_type_field(...) -> std::false_type;

// SFINAE: assert the dispatch-scope field exists.
template <typename T>
auto has_prev_kv_attn_type(int) -> decltype(
    std::declval<T &>().prev_kv_attn_type, std::true_type{});
template <typename T>
auto has_prev_kv_attn_type(...) -> std::false_type;

// SFINAE: assert the model field exists.
template <typename T>
auto has_model_kv_attn_type(int) -> decltype(
    std::declval<T &>().kv_attn_type, std::true_type{});
template <typename T>
auto has_model_kv_attn_type(...) -> std::false_type;

void test_engine_options_field_exists() {
    using namespace tts_cpp::supertonic;
    static_assert(
        decltype(has_kv_attn_type_field<EngineOptions>(0))::value,
        "EngineOptions must declare kv_attn_type (int, default -1 = auto)");

    EngineOptions opts;
    // Default = -1 (auto) — matches the f16_attn / f16_weights /
    // vulkan_device convention.
    CHECK(opts.kv_attn_type == -1);

    // Field accepts the documented values.
    opts.kv_attn_type = 0; CHECK(opts.kv_attn_type == 0);
    opts.kv_attn_type = 1; CHECK(opts.kv_attn_type == 1);
    opts.kv_attn_type = 2; CHECK(opts.kv_attn_type == 2);
    opts.kv_attn_type = 3; CHECK(opts.kv_attn_type == 3);
    opts.kv_attn_type = -1; CHECK(opts.kv_attn_type == -1);

    // Round-3 + earlier defaults — regression guard.
    EngineOptions baseline;
    CHECK(baseline.kv_attn_type == -1);
    CHECK(baseline.prewarm_text.empty());
    CHECK(baseline.vulkan_device == 0);
    CHECK(baseline.f16_attn == -1);
    CHECK(baseline.f16_weights == -1);
    CHECK(baseline.f16_weights_deny_list.empty());
}

void test_supertonic_model_field_exists() {
    using namespace tts_cpp::supertonic::detail;
    static_assert(
        decltype(has_model_kv_attn_type<supertonic_model>(0))::value,
        "supertonic_model must declare kv_attn_type (kv_attn_dtype)");

    supertonic_model model;
    // Default = f32 — a default-constructed model must NOT
    // accidentally dispatch the F16 path before
    // `load_supertonic_gguf` resolves the policy.
    CHECK(model.kv_attn_type == kv_attn_dtype::f32);
}

void test_dispatch_scope_field_exists() {
    using namespace tts_cpp::supertonic::detail;
    static_assert(
        decltype(has_prev_kv_attn_type<supertonic_op_dispatch_scope>(0))::value,
        "supertonic_op_dispatch_scope must declare prev_kv_attn_type "
        "for RAII teardown of the thread-local kv_attn_type flag");
    // Static assert IS the gate.  Bump check count for the
    // pass/fail summary.
    ++g_checks;
}

void test_thread_local_accessor_default() {
    using namespace tts_cpp::supertonic::detail;
    // No scope active → default dtype must be f32 (matches the
    // model default; ensures graph builders called outside a
    // scope don't accidentally take the F16 path).
    CHECK(supertonic_kv_attn_type() == kv_attn_dtype::f32);
}

void test_dispatch_scope_restores_on_teardown() {
    using namespace tts_cpp::supertonic::detail;
    // Baseline.
    CHECK(supertonic_kv_attn_type() == kv_attn_dtype::f32);

    // A scope built from a model with a non-default dtype must
    // flip the thread-local; teardown must restore it.
    {
        supertonic_model m;
        m.kv_attn_type = kv_attn_dtype::bf16;
        // Other fields stay at their defaults; constructor must
        // not require backend / tensors / hparams.
        supertonic_op_dispatch_scope scope(m);
        CHECK(supertonic_kv_attn_type() == kv_attn_dtype::bf16);
    }
    // RAII restored.
    CHECK(supertonic_kv_attn_type() == kv_attn_dtype::f32);
}

} // namespace

int main() {
    test_engine_options_field_exists();
    test_supertonic_model_field_exists();
    test_dispatch_scope_field_exists();
    test_thread_local_accessor_default();
    test_dispatch_scope_restores_on_teardown();

    std::fprintf(stderr,
                 "test_supertonic_kv_attn_type_api: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
