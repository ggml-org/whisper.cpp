// CPU-only unit test for the Vulkan-specific dispatch
// additions landed alongside the Vulkan bring-up:
//
//   1. `supertonic_model::backend_is_vk` — informational flag set
//      from `ggml_backend_is_vk()` at GGUF load.  Carried through
//      to engine.cpp / supertonic_bench.cpp's backend-name
//      annotator (verified by inspection; not under unit test).
//   2. `supertonic_model::use_native_leaky_relu` — true when the
//      resolved backend supports `GGML_OP_LEAKY_RELU` natively.
//      Mirrored into the thread-local `g_supertonic_use_native_leaky_relu`
//      by `supertonic_op_dispatch_scope`; consulted by
//      `leaky_relu_portable_ggml` to skip the RELU+SCALE+ADD
//      decomposition when the fused op is available.
//   3. `supertonic_backend_supports_f16_kv_flash_attn(backend)` —
//      load-time backend probe used by engine + bench to gate the
//      `use_f16_attn` auto-policy.
//
// All three additions are CPU-only-testable: the flags are POD on
// `supertonic_model`, the dispatch scope is a thread-local mirror,
// and the probe takes any `ggml_backend_t` (CPU works fine — it
// supports `LEAKY_RELU` natively, and the F16-K/V flash-attn op
// support depends on whether the CPU backend was built with the
// flash-attn kernel).
//
// No GGUF / model file required.  Registered with `LABEL "unit"` in
// CMakeLists.txt so a fresh checkout's `ctest` exercises this without
// any fixture.
//
// Companion to `test_supertonic_backend_dispatch.cpp` (the OpenCL
// bring-up's tests for `op_dispatch_scope`); this file extends the
// same harness with the new `use_native_leaky_relu` mirror and adds
// a probe smoke test.

#include "supertonic_internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

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

// Test 1 — Default thread-local state for the new query.
//
// Every thread enters with `use_native_leaky_relu` defaulted to
// `true` (matches the historical CPU-only path: CPU has the fused
// op natively, so we want callers without a scope active to keep
// emitting it).  Same default-true contract as
// `supertonic_use_cpu_custom_ops()`.
void test_default_native_leaky_relu_flag() {
    CHECK(supertonic_use_native_leaky_relu() == true);
}

// Test 2 — Scope mirrors a CPU model.
//
// CPU explicitly sets `use_native_leaky_relu = true` (the load-time
// probe always returns true on CPU); the dispatch scope must
// mirror that without flipping anything.
void test_scope_mirrors_cpu_model() {
    supertonic_model model;
    model.backend_is_cpu        = true;
    model.backend_is_vk         = false;
    model.use_native_leaky_relu = true;
    {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == true);
        CHECK(supertonic_use_native_leaky_relu() == true);
    }
    CHECK(supertonic_use_native_leaky_relu() == true);
}

// Test 3 — Scope mirrors a Vulkan-style model.
//
// On Vulkan the load-time probe sets `backend_is_cpu = false`,
// `backend_is_vk = true`, and `use_native_leaky_relu = true`
// (ggml-vulkan's `pipeline_leaky_relu_f32` natively implements the
// op).  `leaky_relu_portable_ggml` should emit the fused builtin
// inside this scope, not the RELU+SCALE+ADD decomposition.
void test_scope_mirrors_vulkan_model() {
    supertonic_model model;
    model.backend_is_cpu        = false;
    model.backend_is_vk         = true;
    model.use_native_leaky_relu = true;
    {
        supertonic_op_dispatch_scope scope(model);
        // CPU custom ops disabled (it's a non-CPU backend), but the
        // native LEAKY_RELU dispatch is on (Vulkan supports it).
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_native_leaky_relu() == true);
    }
    // After teardown, defaults restored.
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_native_leaky_relu() == true);
}

// Test 4 — Scope mirrors an OpenCL-style model (probe = false).
//
// Plain upstream ggml-opencl rejects `GGML_OP_LEAKY_RELU` (only
// chatterbox's vendored patch adds it).  When the load-time probe
// returns false we expect the dispatch helper to take the
// RELU+SCALE+ADD decomposition path instead — the scope must
// faithfully transport that bit.
void test_scope_mirrors_opencl_model() {
    supertonic_model model;
    model.backend_is_cpu        = false;
    model.backend_is_vk         = false;
    model.use_native_leaky_relu = false;
    {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_native_leaky_relu() == false);
    }
    // After teardown, defaults restored — the next CPU engine in
    // the same thread sees the full fused-ops path again.
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_native_leaky_relu() == true);
}

// Test 5 — RAII teardown on exception (extends the OpenCL bring-up
// test to cover the new flag).
//
// If a forward-pass body throws (invalid voice, GGML buffer alloc
// failure, …), the scope must still restore the previous
// `use_native_leaky_relu` so the next engine's call sees a clean
// slate.
void test_scope_unwinds_on_exception() {
    supertonic_model model;
    model.backend_is_cpu        = false;
    model.backend_is_vk         = true;
    model.use_native_leaky_relu = true;
    bool caught = false;
    try {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_native_leaky_relu() == true);
        throw std::runtime_error("simulated forward failure");
    } catch (const std::runtime_error &) {
        caught = true;
    }
    CHECK(caught);
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_native_leaky_relu() == true);
}

// Test 6 — Nested scopes stack and unwind correctly for the new flag.
//
// Mirrors `test_nested_scopes` in `test_supertonic_backend_dispatch.cpp`
// for the new bit so a regression in the dtor restore order shows up
// here as well as in the older test.
void test_nested_scopes() {
    supertonic_model vk_model;
    vk_model.backend_is_cpu        = false;
    vk_model.backend_is_vk         = true;
    vk_model.use_native_leaky_relu = true;

    supertonic_model cl_model;  // OpenCL-style: probe returned false
    cl_model.backend_is_cpu        = false;
    cl_model.backend_is_vk         = false;
    cl_model.use_native_leaky_relu = false;

    {
        supertonic_op_dispatch_scope outer(vk_model);
        CHECK(supertonic_use_native_leaky_relu() == true);
        {
            supertonic_op_dispatch_scope inner(cl_model);
            CHECK(supertonic_use_native_leaky_relu() == false);
        }
        // Inner unwound — outer's bit restored.
        CHECK(supertonic_use_native_leaky_relu() == true);
    }
    CHECK(supertonic_use_native_leaky_relu() == true);
}

// Test 7 — F16-K/V flash-attn backend probe smoke test.
//
// Loads the CPU backend (always available) and asks the probe
// whether it would accept a Supertonic-shaped F16-K/V flash-attn
// node.  We don't assert a specific true/false — the answer
// depends on the CPU backend's build (some upstream builds support
// F16-K/V flash-attn via the cblas reference path; some don't).
// What we assert is:
//   1. The probe returns `false` on a null backend (defensive).
//   2. The probe doesn't crash on the CPU backend.
//   3. Whatever the probe returns, calling it twice returns the
//      same value (it's pure / cacheable).
void test_f16_kv_flash_attn_probe_smoke() {
    CHECK(supertonic_backend_supports_f16_kv_flash_attn(nullptr) == false);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "skip: CPU backend init failed\n");
        return;
    }
    bool a = supertonic_backend_supports_f16_kv_flash_attn(cpu);
    bool b = supertonic_backend_supports_f16_kv_flash_attn(cpu);
    CHECK(a == b);
    std::fprintf(stderr, "probe(F16-K/V flash-attn, CPU) = %s\n",
                 a ? "true" : "false");
    ggml_backend_free(cpu);
}

// Test 8 — Independent flag mutation.
//
// The three flags are independent dimensions: a user might force
// `--f16-attn 1` on a CPU backend (for parity testing), or
// auto-disable `use_native_leaky_relu` on a CPU model (for parity
// testing the GPU decomposition path).  Make sure
// `op_dispatch_scope` round-trips each combination without
// crossing wires.
void test_independent_flags() {
    // CPU + force F16 attn + force decomposed leaky-relu.
    supertonic_model m;
    m.backend_is_cpu        = true;
    m.backend_is_vk         = false;
    m.use_f16_attn          = true;
    m.use_native_leaky_relu = false;
    {
        supertonic_op_dispatch_scope scope(m);
        CHECK(supertonic_use_cpu_custom_ops()    == true);
        CHECK(supertonic_use_f16_attn()          == true);
        CHECK(supertonic_use_native_leaky_relu() == false);
    }

    // Vulkan + force F32 attn + force native leaky-relu.
    m.backend_is_cpu        = false;
    m.backend_is_vk         = true;
    m.use_f16_attn          = false;
    m.use_native_leaky_relu = true;
    {
        supertonic_op_dispatch_scope scope(m);
        CHECK(supertonic_use_cpu_custom_ops()    == false);
        CHECK(supertonic_use_f16_attn()          == false);
        CHECK(supertonic_use_native_leaky_relu() == true);
    }
}

} // namespace

int main() {
    test_default_native_leaky_relu_flag();
    test_scope_mirrors_cpu_model();
    test_scope_mirrors_vulkan_model();
    test_scope_mirrors_opencl_model();
    test_scope_unwinds_on_exception();
    test_nested_scopes();
    test_f16_kv_flash_attn_probe_smoke();
    test_independent_flags();

    std::fprintf(stderr,
                 "test_supertonic_vulkan_dispatch: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
