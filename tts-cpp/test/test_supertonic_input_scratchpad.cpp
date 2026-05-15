// QVAC-18605 round 13 #1 — CPU-only TDD test for the
// `alloc_input_scratchpad_or_throw` helper.
//
// Background
// ----------
// Round 12 #5 shipped `try_alloc_inputs_in_pinned_host_buffer` and
// applied it via a dual-context allocation pattern at 4 cache
// sites (front-block + 3 group caches).  Each application
// repeats the same boilerplate:
//
//     cache.input_buf = try_alloc_inputs_in_pinned_host_buffer(
//         model, cache.input_ctx);
//     if (!cache.input_buf) {
//         cache.input_buf = ggml_backend_alloc_ctx_tensors(
//             cache.input_ctx, model.backend);
//         if (!cache.input_buf) {
//             // teardown + throw
//         }
//     }
//
// Round 13 #1 needs to extend this to several more caches (the
// unrolled CFM loop's `vector_loop_one_graph_cache`, the
// vocoder cache, the style residual + QKV caches, and the
// merged speech-prompted cache).  Rather than 5x copy-paste,
// factor the fallback pattern out:
//
//     ggml_backend_buffer_t alloc_input_scratchpad_or_throw(
//         const supertonic_model & model,
//         ggml_context * input_ctx,
//         const char * cache_name);
//
// Contract:
//   - Tries `try_alloc_inputs_in_pinned_host_buffer(model, ctx)`
//     first.  Returns on success.
//   - On failure (CPU / non-Vulkan / probe miss), falls back to
//     `ggml_backend_alloc_ctx_tensors(ctx, model.backend)`.
//     Returns on success.
//   - On BOTH failing (system resource exhaustion, dead
//     backend), throws `std::runtime_error` with a message
//     that includes `cache_name` so operators can attribute
//     the failure.
//   - Defensive: null `model.backend` / null `input_ctx` / null
//     `cache_name` cases all throw rather than crash.
//
// What this test pins (CPU-only)
// ------------------------------
// 1. Helper symbol exists with the documented signature
//    (compile-time SFINAE).
// 2. On a CPU backend (no Vulkan host buffer), helper falls
//    through to `ggml_backend_alloc_ctx_tensors` and returns a
//    valid buffer.  The returned buffer holds the input ctx's
//    tensors bound to addressable memory (ggml_backend_tensor_set
//    + ggml_backend_tensor_get round-trips correctly).
// 3. Defensive throws on null model.backend / null input_ctx /
//    null cache_name.
// 4. Caller owns the returned buffer; double-free safety via
//    paired `ggml_backend_buffer_free` on the success path.
//
// Registered with `LABEL "unit"` — no GGUF required.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "supertonic_internal.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
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

template <typename F>
bool throws_runtime_error(F && fn) {
    try {
        fn();
        return false;
    } catch (const std::runtime_error &) {
        return true;
    } catch (...) {
        return false;
    }
}

// SFINAE — the helper exists with the documented signature.
template <typename = void>
auto has_alloc_scratchpad(int)
    -> decltype(alloc_input_scratchpad_or_throw(
        std::declval<const supertonic_model &>(),
        std::declval<ggml_context *>(),
        std::declval<const char *>()),
        std::true_type{});
template <typename = void>
auto has_alloc_scratchpad(...) -> std::false_type;

void test_helper_symbol_exists() {
    std::fprintf(stderr, "[Round 13 #1: alloc_input_scratchpad_or_throw symbol]\n");
    static_assert(
        decltype(has_alloc_scratchpad<>(0))::value,
        "alloc_input_scratchpad_or_throw must exist with the documented signature");
    ++g_checks;
}

supertonic_model make_cpu_model() {
    supertonic_model m;
    m.backend = ggml_backend_cpu_init();
    return m;
}

void free_cpu_model(supertonic_model & m) {
    if (m.backend) ggml_backend_free(m.backend);
    m = {};
}

// On CPU backend the pinned-host path returns null; helper MUST
// fall through to `ggml_backend_alloc_ctx_tensors` and produce a
// valid buffer.  Round-trip a test tensor through the buffer to
// confirm the binding actually works (not just non-null).
void test_cpu_fallback_returns_valid_buffer() {
    std::fprintf(stderr, "[Round 13 #1: CPU backend falls through to default-backend alloc]\n");
    supertonic_model model = make_cpu_model();
    CHECK(model.backend != nullptr);

    const size_t buf_size = ggml_tensor_overhead() * 16;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);

    // Synthetic per-step inputs (mimicking the vector_loop one-
    // graph cache layout: a couple of float tensors).
    ggml_tensor * x_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);  // ~512 B
    ggml_tensor * temb_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);     // 256 B
    (void) x_in; (void) temb_in;

    ggml_backend_buffer_t scratchpad =
        alloc_input_scratchpad_or_throw(model, ctx, "test_cpu_fallback");
    CHECK(scratchpad != nullptr);
    if (scratchpad) {
        // Confirm the tensors were actually bound to addressable memory.
        std::vector<float> payload(128, 1.0f);
        ggml_backend_tensor_set(x_in, payload.data(),
                                 0, payload.size() * sizeof(float));
        std::vector<float> readback(128, 0.0f);
        ggml_backend_tensor_get(x_in, readback.data(),
                                 0, readback.size() * sizeof(float));
        bool roundtrip_ok = true;
        for (size_t i = 0; i < payload.size(); ++i) {
            if (readback[i] != payload[i]) { roundtrip_ok = false; break; }
        }
        CHECK(roundtrip_ok);
        ggml_backend_buffer_free(scratchpad);
    }
    ggml_free(ctx);
    free_cpu_model(model);
}

// Empty input_ctx (no tensors) is an edge case — a caller
// shouldn't ever invoke the helper with no inputs to allocate
// (it's a caller bug), but the helper's failure mode on this
// input should be "loud throw with the cache_name in the
// message" so debuggers can identify the misbehaving caller.
//
// Background: `ggml_backend_alloc_ctx_tensors` returns null for
// an empty ctx (no tensors → zero-sized buffer is treated as
// failure on most backends).  Combined with
// `try_alloc_inputs_in_pinned_host_buffer` returning null on CPU,
// both paths fail and the helper throws.  That's the desired
// contract: caller-bug guards in error paths > silent success.
void test_empty_ctx_throws_loud_with_name() {
    std::fprintf(stderr, "[Round 13 #1: empty input_ctx throws with cache_name]\n");
    supertonic_model model = make_cpu_model();
    const size_t buf_size = ggml_tensor_overhead() * 8;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    bool threw_with_name = false;
    try {
        (void) alloc_input_scratchpad_or_throw(model, ctx, "empty_ctx_test");
    } catch (const std::runtime_error & e) {
        const std::string what = e.what();
        threw_with_name = (what.find("empty_ctx_test") != std::string::npos);
    } catch (...) {
        // wrong exception type — caught + reported as a CHECK failure below.
    }
    CHECK(threw_with_name);
    ggml_free(ctx);
    free_cpu_model(model);
}

// Defensive throws — null model.backend, null input_ctx, null
// cache_name.  Each must produce a `std::runtime_error` with a
// message that mentions the failing condition.  These are
// caller-bug guards in error-handler paths.
void test_null_arguments_throw() {
    std::fprintf(stderr, "[Round 13 #1: null arguments throw runtime_error]\n");

    // Null model.backend.
    {
        supertonic_model model;  // backend = nullptr by default
        const size_t buf_size = ggml_tensor_overhead() * 4;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        CHECK(throws_runtime_error([&] {
            (void) alloc_input_scratchpad_or_throw(model, ctx, "null_backend");
        }));
        ggml_free(ctx);
    }

    // Null input_ctx.
    {
        supertonic_model model = make_cpu_model();
        CHECK(throws_runtime_error([&] {
            (void) alloc_input_scratchpad_or_throw(model, nullptr, "null_ctx");
        }));
        free_cpu_model(model);
    }

    // Null cache_name — keep the error message useful; throw
    // rather than dereference a null format-string later.
    {
        supertonic_model model = make_cpu_model();
        const size_t buf_size = ggml_tensor_overhead() * 4;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        CHECK(throws_runtime_error([&] {
            (void) alloc_input_scratchpad_or_throw(model, ctx, nullptr);
        }));
        ggml_free(ctx);
        free_cpu_model(model);
    }
}

// Idempotency — calling the helper twice on the same input
// ctx is a caller bug (only one buffer should ever back the
// inputs) but must not crash.  ggml's
// `ggml_backend_alloc_ctx_tensors` re-allocates the same
// tensors, leaking the first buffer; the contract is the
// caller frees the first.  Test the second call returns a
// distinct (or null) buffer without crashing.
void test_repeated_calls_safe() {
    std::fprintf(stderr, "[Round 13 #1: repeated calls do not crash]\n");
    supertonic_model model = make_cpu_model();
    const size_t buf_size = ggml_tensor_overhead() * 8;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    (void) ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_backend_buffer_t b1 =
        alloc_input_scratchpad_or_throw(model, ctx, "repeat_first");
    CHECK(b1 != nullptr);
    // Second call: don't assert specific behaviour, just ensure
    // we don't crash.  If it returns a buffer, free it.  If it
    // throws, that's also acceptable (caller bug).
    ggml_backend_buffer_t b2 = nullptr;
    bool b2_threw = throws_runtime_error([&] {
        b2 = alloc_input_scratchpad_or_throw(model, ctx, "repeat_second");
    });
    (void) b2_threw;  // either outcome OK
    if (b2 && b2 != b1) ggml_backend_buffer_free(b2);
    if (b1) ggml_backend_buffer_free(b1);
    ggml_free(ctx);
    free_cpu_model(model);
}

} // namespace

int main() {
    test_helper_symbol_exists();
    test_cpu_fallback_returns_valid_buffer();
    test_empty_ctx_throws_loud_with_name();
    test_null_arguments_throw();
    test_repeated_calls_safe();

    std::fprintf(stderr,
                 "test_supertonic_input_scratchpad: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
