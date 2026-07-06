// round 12 #5 — CPU-only TDD test for the pinned-host-
// buffer input-allocation helper.
//
// Background
// ----------
// Round 3 shipped the capability probe
// `supertonic_backend_supports_pinned_host_buffer`, which returns
// `true` iff `ggml_backend_vk_host_buffer_type()` is non-null on the
// resolved backend.  The probe primed the cache + bench surface
// but the actual per-engine input-scratchpad refactor that would
// USE the host-pinned buffer to skip ggml-vulkan's internal
// staging-buffer hop was deferred.
//
// Round 12 #5 lands that refactor as a thin helper:
//
//   ggml_backend_buffer_t try_alloc_inputs_in_pinned_host_buffer(
//       const supertonic_model & model,
//       ggml_context * input_ctx);
//
// Callers create a small `ggml_context` containing ONLY the hot
// per-step input tensors (e.g. front-block `x_in` / `mask_in` /
// `t_emb_in`), then call the helper.  The helper:
//
//   - Returns `nullptr` if the backend doesn't expose
//     `ggml_backend_vk_host_buffer_type()` (CPU, Metal, OpenCL,
//     and any future backend that lacks the API).  Caller falls
//     back to letting `ggml_gallocr_alloc_graph` handle the
//     input tensors via the default buffer type — same memory
//     layout, just one staging-buffer hop per upload.
//
//   - Allocates a buffer from `ggml_backend_vk_host_buffer_type()`
//     and binds every tensor in `input_ctx` to it on success.
//     `ggml_backend_tensor_set` writes from the host buffer
//     directly into the BAR-mapped GPU memory without an
//     intermediate staging-buffer copy.
//
// Per synth wins (RTX 5090, 5-step CFM):
//   - 4 attention-feeding caches × per-step inputs:
//       front_block: x_in (~80 KB), mask_in (~80 B), t_emb_in (~256 B)
//       g1 / g2 / g3 group:  x_in, temb_in
//   - 5 denoise steps × ~3 small uploads = ~15 staging-hops saved
//     per synth.  Each hop is ~5-15 us on the test rig; net
//     ~75-225 us / synth.
//
// What this test pins (CPU-only)
// ------------------------------
// 1. The helper symbol exists with the documented signature
//    (compile-time SFINAE).
//
// 2. On a CPU backend (no Vulkan host-buffer API), the helper
//    returns `nullptr` — and does so WITHOUT crashing when
//    handed a context with no tensors, or a context with a
//    couple of synthetic input tensors.
//
// 3. Repeated calls on the same input context against a CPU
//    backend are idempotent (no leak on null return; no
//    double-free on the second call).
//
// What is NOT testable in this CPU-only unit test:
//   - The actual host-buffer allocation behaviour (requires a
//     real Vulkan adapter).  Validated end-to-end by the
//     model-fixture synth runs + the per-step bench.
//   - The wiring at the production cache sites (validated by
//     `ctest -L unit` running every other test green + the
//     end-to-end Vulkan synth).
//
// Registered with `LABEL "unit"` — no GGUF required.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "supertonic_internal.h"

#include <cstdio>
#include <type_traits>
#include <vector>

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

// SFINAE — the helper symbol exists with the expected signature.
// Compile-fails before implementation lands; compile-passes after.
template <typename = void>
auto has_try_alloc_helper(int)
    -> decltype(try_alloc_inputs_in_pinned_host_buffer(
        std::declval<const supertonic_model &>(),
        std::declval<ggml_context *>()),
        std::true_type{});
template <typename = void>
auto has_try_alloc_helper(...) -> std::false_type;

void test_helper_symbol_exists() {
    std::fprintf(stderr, "[Round 12 #5: try_alloc_inputs_in_pinned_host_buffer symbol]\n");
    static_assert(
        decltype(has_try_alloc_helper<>(0))::value,
        "try_alloc_inputs_in_pinned_host_buffer must exist with the documented signature");
    ++g_checks;
}

// Build a minimal supertonic_model carrying only the backend
// pointer the helper needs.  Synth code paths aren't exercised
// here — the helper just queries `model.backend` for the host-
// buffer-type capability.
supertonic_model make_cpu_model() {
    supertonic_model m;
    m.backend = ggml_backend_cpu_init();
    return m;
}

void free_cpu_model(supertonic_model & m) {
    if (m.backend) ggml_backend_free(m.backend);
    m = {};
}

// Round-12 #5 contract on CPU backend: helper returns nullptr
// (no Vulkan host-buffer API available).  Caller proceeds with
// the default gallocr path.
void test_cpu_backend_returns_nullptr() {
    std::fprintf(stderr, "[Round 12 #5: CPU backend → nullptr]\n");
    supertonic_model model = make_cpu_model();
    CHECK(model.backend != nullptr);

    // Empty input ctx — should still return nullptr without
    // crashing.
    {
        const size_t buf_size = ggml_tensor_overhead() * 16;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
        ggml_context * ctx = ggml_init(p);
        CHECK(ctx != nullptr);
        ggml_backend_buffer_t res = try_alloc_inputs_in_pinned_host_buffer(model, ctx);
        CHECK(res == nullptr);
        ggml_free(ctx);
    }

    // Input ctx with a handful of small synthetic input tensors.
    // The helper must still return nullptr cleanly when the
    // backend doesn't expose the host-buffer type.
    {
        const size_t buf_size = ggml_tensor_overhead() * 32;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
        ggml_context * ctx = ggml_init(p);
        (void) ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 20);   // ~x_in
        (void) ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 20);       // ~mask_in
        (void) ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);       // ~t_emb_in
        ggml_backend_buffer_t res = try_alloc_inputs_in_pinned_host_buffer(model, ctx);
        CHECK(res == nullptr);
        ggml_free(ctx);
    }

    free_cpu_model(model);
}

// Round-12 #5: idempotency.  Calling the helper twice on the same
// (model, ctx) pair against a backend that returns nullptr each
// time must be safe (no internal state leakage, no double-free
// path triggered).  Catches a regression where the helper
// accidentally caches the buffer in `model` or `ctx` extras and
// double-frees on the second call.
void test_idempotent_on_cpu_backend() {
    std::fprintf(stderr, "[Round 12 #5: idempotent on CPU backend]\n");
    supertonic_model model = make_cpu_model();
    const size_t buf_size = ggml_tensor_overhead() * 32;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    (void) ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 20);

    ggml_backend_buffer_t res1 = try_alloc_inputs_in_pinned_host_buffer(model, ctx);
    ggml_backend_buffer_t res2 = try_alloc_inputs_in_pinned_host_buffer(model, ctx);
    CHECK(res1 == nullptr);
    CHECK(res2 == nullptr);
    CHECK(res1 == res2);

    ggml_free(ctx);
    free_cpu_model(model);
}

// Round-12 #5: null-backend safety.  If the caller hands the
// helper a `supertonic_model` whose `.backend` is null (e.g., a
// half-constructed model in an error path), the helper must
// return nullptr instead of dereferencing.  Conservative
// failure mode beats SIGSEGV in error-handler code paths.
void test_null_backend_returns_nullptr() {
    std::fprintf(stderr, "[Round 12 #5: null backend → nullptr]\n");
    supertonic_model model;  // .backend = nullptr by default
    CHECK(model.backend == nullptr);
    const size_t buf_size = ggml_tensor_overhead() * 16;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_backend_buffer_t res = try_alloc_inputs_in_pinned_host_buffer(model, ctx);
    CHECK(res == nullptr);
    ggml_free(ctx);
}

// Round-12 #5: null-ctx safety.  Same conservative contract as
// the null-backend test — pass a real backend with a null
// ctx and verify the helper returns nullptr without crashing.
void test_null_ctx_returns_nullptr() {
    std::fprintf(stderr, "[Round 12 #5: null ctx → nullptr]\n");
    supertonic_model model = make_cpu_model();
    ggml_backend_buffer_t res = try_alloc_inputs_in_pinned_host_buffer(model, nullptr);
    CHECK(res == nullptr);
    free_cpu_model(model);
}

} // namespace

int main() {
    test_helper_symbol_exists();
    test_cpu_backend_returns_nullptr();
    test_idempotent_on_cpu_backend();
    test_null_backend_returns_nullptr();
    test_null_ctx_returns_nullptr();

    std::fprintf(stderr,
                 "test_supertonic_pinned_host_buffer: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
