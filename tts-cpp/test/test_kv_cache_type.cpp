// KV-cache dtype selection + capability-probe coverage (QVAC-19557).
//
// Exercises the two pure helpers that gate the chatterbox T3 KV-cache
// dtype, without needing a model fixture:
//
//   chatterbox_kv_type_from_str  — string -> ggml_type, with the
//       unknown-string -> F32 guard so a typo can't silently change
//       numerics.
//   chatterbox_resolve_kv_type   — backend capability probe: returns
//       the requested dtype when the backend's flash_attn_ext accepts
//       K/V of that type at the model's head geometry, else falls back
//       to F32 (so an f16/q8_0 request can't assert deep inside a
//       backend that rejects quantized K/V).
//
// The CPU backend supports F32/F16/Q8_0 flash attention, so the probe
// retains each on CPU; the fallback branch is covered structurally via
// the null-backend short-circuit (no backend can run the op).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "chatterbox_t3_internal.h"

#include <cstdio>

using namespace tts_cpp::chatterbox::detail;

static int g_failures = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        ++g_failures;                                          \
    }                                                          \
} while (0)

int main() {
    // ---- string -> type ----
    CHECK(chatterbox_kv_type_from_str("")     == GGML_TYPE_F32,  "empty -> f32");
    CHECK(chatterbox_kv_type_from_str("f32")  == GGML_TYPE_F32,  "f32");
    CHECK(chatterbox_kv_type_from_str("f16")  == GGML_TYPE_F16,  "f16");
    CHECK(chatterbox_kv_type_from_str("q8_0") == GGML_TYPE_Q8_0, "q8_0");
    // Unknown / typo falls back to F32 rather than silently mis-parsing.
    CHECK(chatterbox_kv_type_from_str("q4_0")    == GGML_TYPE_F32, "unknown q4_0 -> f32");
    CHECK(chatterbox_kv_type_from_str("garbage") == GGML_TYPE_F32, "garbage -> f32");

    // Turbo head geometry (n_head == n_kv_head); head_dim = n_embd / n_head.
    const int head_dim = 64, n_head = 16, n_kv_head = 16;

    // ---- null backend: nothing can run the op -> F32 ----
    CHECK(chatterbox_resolve_kv_type(nullptr, GGML_TYPE_Q8_0, head_dim, n_head, n_kv_head)
              == GGML_TYPE_F32, "null backend -> f32");

    // ---- F32 request short-circuits (always supported, no probe) ----
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu != nullptr, "cpu backend init");
    CHECK(chatterbox_resolve_kv_type(cpu, GGML_TYPE_F32, head_dim, n_head, n_kv_head)
              == GGML_TYPE_F32, "f32 request stays f32 on cpu");

    // ---- CPU flash-attn supports F16 + Q8_0 K/V -> requested type retained ----
    CHECK(chatterbox_resolve_kv_type(cpu, GGML_TYPE_F16, head_dim, n_head, n_kv_head)
              == GGML_TYPE_F16, "cpu retains f16 KV");
    CHECK(chatterbox_resolve_kv_type(cpu, GGML_TYPE_Q8_0, head_dim, n_head, n_kv_head)
              == GGML_TYPE_Q8_0, "cpu retains q8_0 KV");

    ggml_backend_free(cpu);

    if (g_failures) {
        fprintf(stderr, "test-kv-cache-type: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-kv-cache-type: all checks passed\n");
    return 0;
}
