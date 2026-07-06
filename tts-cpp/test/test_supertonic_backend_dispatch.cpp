// Unit tests for the OpenCL bring-up dispatch helpers landed in
// `supertonic_op_dispatch_scope`, the thread-local
// `supertonic_use_cpu_custom_ops()` / `supertonic_use_f16_attn()`
// queries, and the `supertonic_model::backend_is_cpu`
// + `supertonic_model::use_f16_attn` fields they mirror.
//
// No GGUF / model file required — every test instantiates a bare
// `supertonic_model` POD on the stack with the two relevant flags set
// by hand, opens an RAII scope around it, and re-asserts the
// thread-local query state matches what the scope was constructed
// with.  This is what every public `supertonic_*_forward_ggml` /
// `*_trace_ggml` entry point does, so a regression here would mean a
// regression in the *real* dispatch path.
//
// Registered with `LABEL "unit"` in CMakeLists.txt so a fresh
// checkout's `ctest` exercises this without needing any fixture.

#include "supertonic_internal.h"

#include <cstdio>
#include <stdexcept>

using namespace tts_cpp::supertonic::detail;

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

// Test 1 — Default thread-local state.
//
// Every thread enters with CPU custom ops enabled (the historical
// CPU-only Supertonic path keeps working unchanged) and F16 K/V
// attention disabled (the CPU CBLAS attention path is the cheaper
// choice on a CPU backend, so the auto-policy lands here).
void test_default_flags() {
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 2 — Scope mirrors a CPU model.
//
// A CPU-backend model toggles nothing: defaults already match.
// The point of this test is to catch a "scope leaked the wrong
// previous-value back into the thread-local on dtor" regression by
// also asserting the default state after teardown.
void test_scope_mirrors_cpu_model() {
    supertonic_model model;
    model.backend_is_cpu = true;
    model.use_f16_attn   = false;
    {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == true);
        CHECK(supertonic_use_f16_attn() == false);
    }
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 3 — Scope mirrors a GPU model + restores defaults after.
//
// A GPU-backend engine (OpenCL / CUDA / Metal / Vulkan) sets both
// flags via the dispatch scope; the cblas-backed `ggml_custom_4d`
// fast paths in the vocoder + vector estimator must see `false`
// inside the scope, then `true` again after teardown so a
// CPU-only second engine in the same thread isn't poisoned.
void test_scope_mirrors_gpu_model() {
    supertonic_model model;
    model.backend_is_cpu = false;
    model.use_f16_attn   = true;
    {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
    }
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 4 — RAII teardown on exception.
//
// The forward functions wrap the rest of their body in try / catch;
// if the body throws (e.g. invalid voice, GGML buffer alloc failure),
// the scope must still restore the previous flags so the next
// engine's call sees a clean slate.
void test_scope_unwinds_on_exception() {
    supertonic_model model;
    model.backend_is_cpu = false;
    model.use_f16_attn   = true;
    bool caught = false;
    try {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
        throw std::runtime_error("simulated forward failure");
    } catch (const std::runtime_error &) {
        caught = true;
    }
    CHECK(caught);
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 5 — Nested scopes stack and unwind correctly.
//
// This is the harness for the "host destroyed engine_a then
// immediately invoked synthesize on engine_b on the same thread"
// path the alive-id registry already covers for gallocr free.
// Here we verify the dispatch flags don't get crossed during the
// brief window where both scopes exist (e.g. one forward function
// calling another's helper synchronously).
void test_nested_scopes() {
    supertonic_model gpu_model;
    gpu_model.backend_is_cpu = false;
    gpu_model.use_f16_attn   = true;

    supertonic_model cpu_model;
    cpu_model.backend_is_cpu = true;
    cpu_model.use_f16_attn   = false;

    {
        supertonic_op_dispatch_scope outer(gpu_model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
        {
            supertonic_op_dispatch_scope inner(cpu_model);
            CHECK(supertonic_use_cpu_custom_ops() == true);
            CHECK(supertonic_use_f16_attn() == false);
        }
        // After inner unwinds, outer's state restored.
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
    }
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 6 — Independent flags.
//
// `use_f16_attn = true` on a CPU model is a valid configuration
// (the user can `--f16-attn 1` even on CPU for parity testing),
// and `use_f16_attn = false` on a GPU model is the manual opt-out.
// Make sure the two flags are mirrored independently.
void test_independent_flags() {
    supertonic_model m;
    m.backend_is_cpu = true;
    m.use_f16_attn   = true;
    {
        supertonic_op_dispatch_scope scope(m);
        CHECK(supertonic_use_cpu_custom_ops() == true);
        CHECK(supertonic_use_f16_attn() == true);
    }

    m.backend_is_cpu = false;
    m.use_f16_attn   = false;
    {
        supertonic_op_dispatch_scope scope(m);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == false);
    }
}

} // namespace

int main() {
    test_default_flags();
    test_scope_mirrors_cpu_model();
    test_scope_mirrors_gpu_model();
    test_scope_unwinds_on_exception();
    test_nested_scopes();
    test_independent_flags();

    std::fprintf(stderr,
                 "test_supertonic_backend_dispatch: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
