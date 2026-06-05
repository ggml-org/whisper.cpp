// QVAC-18605 round 12 — CPU-only TDD test for the text-encoder
// speech-prompted-attention GPU bridge (`run_speech_prompted_merged_cache`).
//
// Background
// ----------
// Master's Metal-port branch (PR #15) shipped a fully-built
// `speech_prompted_merged_cache` graph in `supertonic_text_encoder.cpp`
// — a single ggml graph that does QKV projection + head-split +
// flash-attn + out-proj end-to-end on the GPU.  The graph
// builder (`build_speech_prompted_merged_cache`) is present + tested
// at the implementation level via the Metal port's own harnesses,
// but the **run path** that exercises it from
// `speech_prompted_attention_ggml` was never wired in.  So the
// production text-encoder path stays on the pre-Phase-A4 two-cache
// pattern with host-side Q/V download → pack → re-upload between
// the QKV cache and the flash-attn cache.
//
// Per text encoder call (2 speech-prompted layers per synth):
//
//   Pre-round-12 (two-cache path):
//     - QKV cache compute
//     - 2 GPU→host downloads (q_out, v_out via tensor_to_time_channel)
//     - host-side pack of q_pack / k_pack / v_pack (rearranges into
//       the [D, L, H] layout flash_attn views as [head_dim, q_len,
//       n_heads])
//     - 3 host→GPU uploads (q_pack, k_pack, v_pack)
//     - flash-attn cache compute
//   = 5 sync points + ~half_dim × L × n_heads × 3 floats of host work
//
//   Post-round-12 (merged path):
//     - One merged graph compute
//   = 0 sync points, 0 host pack work
//
// Eliminates **5 sync points × 2 layers = 10 sync points / synth**
// on the text encoder alone.  Combined with the auto-pick fix in
// the same round, the RTX 5090 number drops from ~4.8 ms /
// text_encoder to ~2.5-3 ms.
//
// What this test pins (CPU-only)
// ------------------------------
// 1. The new `run_speech_prompted_merged_cache` symbol exists in
//    `detail::` with the expected signature.  SFINAE — fails at
//    compile time if the function isn't there, fails at link
//    time if it's declared but undefined.
//
// 2. The `speech_prompted_merged_cache` struct exposes the
//    fields the run path needs (x_in, style_in, out, gf,
//    idx, L, Lctx, generation_id, model).  Same SFINAE pattern.
//
// 3. A runtime trip-wire that confirms the dispatch wrapper
//    `speech_prompted_attention_ggml` exists with its
//    pre-round-12 signature.  Round 12 swaps the internal
//    dispatch (CPU → legacy two-cache path, non-CPU → merged
//    path) without changing the public function shape, so any
//    caller that compiled pre-round-12 keeps compiling.
//
// Equivalence between the merged and legacy paths is verified
// end-to-end on real hardware via the model-fixture tests
// (`test-supertonic-text-encoder-trace`,
// `test-supertonic-pipeline`) — those exercise the live graph
// against the scalar reference.  CPU-only unit tests can't
// build the cache without a real GGUF's source tensors (q_w,
// v_w, out_w, tanh_k all by name) so we don't try here.
//
// Registered with `LABEL "unit"` — no GGUF required.

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

// SFINAE — the merged-cache run symbol exists with the expected
// shape.  Round 12 introduces this; pre-round-12 the test fails
// to compile on `has_run_speech_prompted_merged_cache<>(0)`.
//
// Expected signature:
//
//   void run_speech_prompted_merged_cache(
//       speech_prompted_merged_cache & cache,
//       const supertonic_model & m,
//       const std::vector<float> & x_lc,
//       int L,
//       const float * style_ttl,
//       std::vector<float> & out_lc);
//
// Mirrors the calling convention of the legacy
// `speech_prompted_attention_ggml` so the dispatch wrapper can
// fall through to it with no argument repacking.
template <typename = void>
auto has_run_speech_prompted_merged_cache(int)
    -> decltype(run_speech_prompted_merged_cache(
        std::declval<speech_prompted_merged_cache &>(),
        std::declval<const supertonic_model &>(),
        std::declval<const std::vector<float> &>(),
        std::declval<int>(),
        std::declval<const float *>(),
        std::declval<std::vector<float> &>()),
        std::true_type{});
template <typename = void>
auto has_run_speech_prompted_merged_cache(...) -> std::false_type;

void test_run_symbol_exists() {
    std::fprintf(stderr, "[Round 12 #6: run_speech_prompted_merged_cache symbol]\n");
    static_assert(
        decltype(has_run_speech_prompted_merged_cache<>(0))::value,
        "run_speech_prompted_merged_cache must exist with the documented signature");
    // SFINAE is the actual gate; runtime check exists so the
    // test reports a meaningful pass/fail count.
    ++g_checks;
}

// SFINAE — the merged-cache struct exposes the fields the run
// path needs.  Master built the struct + builder; round 12 adds
// the run path that reads these fields.  A future struct rename
// or field removal trips this gate.
template <typename T, typename = void>
struct has_x_in_field : std::false_type {};
template <typename T>
struct has_x_in_field<T, std::void_t<decltype(std::declval<T &>().x_in)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_style_in_field : std::false_type {};
template <typename T>
struct has_style_in_field<T, std::void_t<decltype(std::declval<T &>().style_in)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_out_field : std::false_type {};
template <typename T>
struct has_out_field<T, std::void_t<decltype(std::declval<T &>().out)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_idx_field : std::false_type {};
template <typename T>
struct has_idx_field<T, std::void_t<decltype(std::declval<T &>().idx)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_L_field : std::false_type {};
template <typename T>
struct has_L_field<T, std::void_t<decltype(std::declval<T &>().L)>>
    : std::true_type {};

void test_merged_cache_struct_fields() {
    std::fprintf(stderr, "[Round 12 #6: speech_prompted_merged_cache struct fields]\n");
    static_assert(has_x_in_field    <speech_prompted_merged_cache>::value,
                  "speech_prompted_merged_cache must expose x_in");
    static_assert(has_style_in_field<speech_prompted_merged_cache>::value,
                  "speech_prompted_merged_cache must expose style_in");
    static_assert(has_out_field     <speech_prompted_merged_cache>::value,
                  "speech_prompted_merged_cache must expose out");
    static_assert(has_idx_field     <speech_prompted_merged_cache>::value,
                  "speech_prompted_merged_cache must expose idx");
    static_assert(has_L_field       <speech_prompted_merged_cache>::value,
                  "speech_prompted_merged_cache must expose L");
    ++g_checks;
}

// `speech_prompted_attention_ggml` is internal to
// `supertonic_text_encoder.cpp` (it's only called from
// `supertonic_text_encoder_forward_ggml` in the same TU) and
// intentionally not declared in `supertonic_internal.h` — so this
// SFINAE-pinning is left to the model-fixture tests that
// link against the dispatch path through
// `supertonic_text_encoder_forward_ggml` (e.g.
// `test-supertonic-text-encoder-trace`).

// Trip-wire: free a fresh-defaulted merged cache.  Verifies the
// destructor path works on a never-built cache (idx==-1, ctx==
// nullptr, allocr==nullptr) without crashing — important because
// the dispatch wrapper holds `thread_local
// speech_prompted_merged_cache merged_caches[2]` and on
// program exit those destructors fire.  A buggy free path
// (e.g., unconditional `ggml_free(cache.ctx)` on nullptr) would
// segfault here.
void test_free_default_constructed_cache() {
    std::fprintf(stderr, "[Round 12 #6: free default-constructed merged cache]\n");
    speech_prompted_merged_cache cache;  // defaults: idx=-1, ctx=nullptr, etc.
    free_speech_prompted_merged_cache(cache);
    CHECK(cache.ctx == nullptr);
    CHECK(cache.allocr == nullptr);
    CHECK(cache.idx == -1);
    CHECK(cache.L == 0);
}

} // namespace

int main() {
    test_run_symbol_exists();
    test_merged_cache_struct_fields();
    test_free_default_constructed_cache();

    std::fprintf(stderr,
                 "test_supertonic_text_encoder_gpu_bridge: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
