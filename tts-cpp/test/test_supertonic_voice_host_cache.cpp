// QVAC-18605 round 7 — CPU-only TDD test for the voice ttl/dp host
// cache.
//
// Background
// ----------
// `Engine::Impl::synthesize()` currently downloads the per-voice
// style tensors (`ttl`, `dp`) from the GGUF on EVERY call:
//
//   std::vector<float> style_ttl = read_tensor_f32(vit->second.ttl);
//   std::vector<float> style_dp  = read_tensor_f32(vit->second.dp);
//
// Each `read_tensor_f32` is one synchronous GPU→host download +
// one host vector allocation.  On Vulkan / OpenCL backends this
// is a sync point per call per voice, which doesn't change across
// calls (voice tensors are part of the load-time GGUF state — they
// never mutate after load).  Caching them per-engine keyed by
// voice name eliminates 2 sync points per `synthesize()` call on
// every call after the first per-voice.
//
// Round 7 introduces a small standalone helper
// `tts_cpp::supertonic::detail::voice_host_cache` so the lookup-
// or-load semantics are testable on CPU without instantiating a
// full `Engine::Impl`.  The Engine::Impl wiring is a thin caller
// of this helper.
//
// API contract:
//
//   struct voice_host_cache {
//       struct entry {
//           std::vector<float> ttl;
//           std::vector<float> dp;
//       };
//
//       // Returns a stable reference to the cached entry for
//       // `voice_name`.  On cache miss, calls `read_tensor_f32`
//       // on `ttl_tensor` and `dp_tensor`, stores the result,
//       // and returns the new entry.  On cache hit, returns the
//       // existing entry without touching the GGML tensors at
//       // all (the host vectors are reused as-is).
//       //
//       // Reference is stable across subsequent `get_or_load`
//       // calls for OTHER voices (std::unordered_map's
//       // reference-stability guarantee on insert).  Caller may
//       // hold the reference across the next `get_or_load` on
//       // the same instance, BUT must NOT call `clear()` on the
//       // cache while holding the reference.
//       const entry & get_or_load(const std::string & voice_name,
//                                 ggml_tensor * ttl_tensor,
//                                 ggml_tensor * dp_tensor);
//
//       // Drops every cached entry.  Called by Engine::Impl on
//       // backend reset (currently unreachable — included for
//       // forward-compat with hot-swap scenarios).
//       void clear();
//
//       // Diagnostic — number of entries currently cached.  Used
//       // by the test to assert lookup-vs-load semantics.
//       size_t size() const;
//   };
//
// Whole TU MUST fail to compile before the symbol is added,
// then pass after.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "supertonic_internal.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using tts_cpp::supertonic::detail::voice_host_cache;

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

// Build a tiny F32 tensor with the supplied scalar payload
// allocated on `cpu`.  Mirrors the shape of a real voice
// tensor (ttl is [256, 50, 1], dp is [16, 8, 1]) without
// requiring a real model.  Caller owns the returned context +
// buffer; tensor is valid until ggml_free + ggml_backend_buffer_free.
struct stub_tensor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * tensor = nullptr;

    ~stub_tensor() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
    stub_tensor() = default;
    stub_tensor(const stub_tensor &)             = delete;
    stub_tensor & operator=(const stub_tensor &) = delete;
};

void make_stub_tensor(ggml_backend_t cpu,
                      stub_tensor & out,
                      int ne0, int ne1, int ne2,
                      const std::vector<float> & payload) {
    constexpr int MAX_NODES = 4;
    const size_t buf_size = ggml_tensor_overhead() * MAX_NODES;
    ggml_init_params p{ buf_size, nullptr, /*no_alloc=*/true };
    out.ctx = ggml_init(p);
    if (!out.ctx) throw std::runtime_error("ggml_init failed");
    out.tensor = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, cpu);
    if (!out.buf) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
    if ((size_t) ggml_nelements(out.tensor) != payload.size()) {
        throw std::runtime_error("payload size mismatch in test stub");
    }
    ggml_backend_tensor_set(out.tensor, payload.data(), 0,
                            payload.size() * sizeof(float));
}

// Test 1 — empty cache reports size 0; clear is a no-op on empty.
void test_empty_cache() {
    voice_host_cache cache;
    CHECK(cache.size() == 0);
    cache.clear();  // must not throw
    CHECK(cache.size() == 0);
}

// Test 2 — first `get_or_load` populates from the GGML tensors;
// returned vectors carry the exact payload.
void test_first_load_populates(ggml_backend_t cpu) {
    voice_host_cache cache;

    std::vector<float> ttl_payload(8, 1.5f);
    for (size_t i = 0; i < ttl_payload.size(); ++i) ttl_payload[i] = (float) i + 0.25f;
    std::vector<float> dp_payload(4, 2.5f);
    for (size_t i = 0; i < dp_payload.size(); ++i) dp_payload[i] = (float) i - 0.5f;

    stub_tensor ttl_t; make_stub_tensor(cpu, ttl_t, 8, 1, 1, ttl_payload);
    stub_tensor dp_t;  make_stub_tensor(cpu, dp_t,  4, 1, 1, dp_payload);

    const auto & e = cache.get_or_load("F1", ttl_t.tensor, dp_t.tensor);
    CHECK(e.ttl == ttl_payload);
    CHECK(e.dp  == dp_payload);
    CHECK(cache.size() == 1);
}

// Test 3 — second `get_or_load` for the same voice returns the
// same entry WITHOUT touching the GGML tensors.  We verify the
// "no-touch" property by passing nullptr for ttl/dp on the second
// call: a real load attempt would crash; a cache hit returns the
// previously-stored entry.
void test_second_load_hits_cache(ggml_backend_t cpu) {
    voice_host_cache cache;

    std::vector<float> ttl_payload(6, 0.0f);
    for (size_t i = 0; i < ttl_payload.size(); ++i) ttl_payload[i] = (float) i;
    std::vector<float> dp_payload(3, 0.0f);
    for (size_t i = 0; i < dp_payload.size(); ++i) dp_payload[i] = -(float) i;

    stub_tensor ttl_t; make_stub_tensor(cpu, ttl_t, 6, 1, 1, ttl_payload);
    stub_tensor dp_t;  make_stub_tensor(cpu, dp_t,  3, 1, 1, dp_payload);

    const auto & first  = cache.get_or_load("M1", ttl_t.tensor, dp_t.tensor);
    CHECK(first.ttl == ttl_payload);

    // Pass nullptr — if the cache TRIED to re-load, this would
    // crash inside `read_tensor_f32`.  A clean cache hit returns
    // the prior entry untouched.
    const auto & second = cache.get_or_load("M1", nullptr, nullptr);
    CHECK(&first == &second);  // reference identity
    CHECK(second.ttl == ttl_payload);
    CHECK(second.dp  == dp_payload);
    CHECK(cache.size() == 1);
}

// Test 4 — multiple voices coexist; each entry is independent;
// reference stability holds across subsequent get_or_load calls
// for OTHER voices.
void test_multiple_voices(ggml_backend_t cpu) {
    voice_host_cache cache;

    stub_tensor ttl_a; make_stub_tensor(cpu, ttl_a, 4, 1, 1, {1, 2, 3, 4});
    stub_tensor dp_a;  make_stub_tensor(cpu, dp_a,  2, 1, 1, {10, 20});
    stub_tensor ttl_b; make_stub_tensor(cpu, ttl_b, 4, 1, 1, {5, 6, 7, 8});
    stub_tensor dp_b;  make_stub_tensor(cpu, dp_b,  2, 1, 1, {30, 40});
    stub_tensor ttl_c; make_stub_tensor(cpu, ttl_c, 4, 1, 1, {9, 9, 9, 9});
    stub_tensor dp_c;  make_stub_tensor(cpu, dp_c,  2, 1, 1, {50, 60});

    const auto & a1 = cache.get_or_load("A", ttl_a.tensor, dp_a.tensor);
    const auto & b1 = cache.get_or_load("B", ttl_b.tensor, dp_b.tensor);
    const auto & c1 = cache.get_or_load("C", ttl_c.tensor, dp_c.tensor);

    CHECK(a1.ttl == std::vector<float>({1, 2, 3, 4}));
    CHECK(b1.ttl == std::vector<float>({5, 6, 7, 8}));
    CHECK(c1.ttl == std::vector<float>({9, 9, 9, 9}));
    CHECK(a1.dp  == std::vector<float>({10, 20}));
    CHECK(b1.dp  == std::vector<float>({30, 40}));
    CHECK(c1.dp  == std::vector<float>({50, 60}));
    CHECK(cache.size() == 3);

    // Reference stability — looking up A again must yield the
    // SAME object the original lookup returned.  std::unordered_map
    // guarantees stable references on insert (no rehash needed
    // because we're not exceeding any bucket threshold).  This
    // matters for the production Engine::Impl call site: it
    // captures the ttl/dp pointers from `e.ttl.data()` /
    // `e.dp.data()` and forwards them to the synthesis pipeline,
    // which expects them to stay valid for the duration of the
    // call.
    const auto & a2 = cache.get_or_load("A", nullptr, nullptr);
    CHECK(&a1 == &a2);
}

// Test 5 — `clear()` drops every entry; subsequent get_or_load
// re-loads from the tensors.
void test_clear_drops_entries(ggml_backend_t cpu) {
    voice_host_cache cache;

    std::vector<float> ttl_payload(4, 7.0f);
    std::vector<float> dp_payload(2, -3.0f);
    stub_tensor ttl_t; make_stub_tensor(cpu, ttl_t, 4, 1, 1, ttl_payload);
    stub_tensor dp_t;  make_stub_tensor(cpu, dp_t,  2, 1, 1, dp_payload);

    cache.get_or_load("V", ttl_t.tensor, dp_t.tensor);
    CHECK(cache.size() == 1);
    cache.clear();
    CHECK(cache.size() == 0);

    // Re-load must succeed and produce the same payload.
    const auto & e = cache.get_or_load("V", ttl_t.tensor, dp_t.tensor);
    CHECK(e.ttl == ttl_payload);
    CHECK(e.dp  == dp_payload);
    CHECK(cache.size() == 1);
}

// Test 6 — null tensor pointers throw on cache miss (loud
// failure for an Impl bug; never expected to fire on the
// production path because Impl validates `voices.find()` before
// calling the cache).
void test_null_tensors_on_miss_throws(ggml_backend_t /*cpu*/) {
    voice_host_cache cache;
    bool threw = false;
    try {
        cache.get_or_load("ghost", nullptr, nullptr);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(cache.size() == 0);
}

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "ggml_backend_cpu_init failed\n");
        return 1;
    }

    test_empty_cache();
    test_first_load_populates(cpu);
    test_second_load_hits_cache(cpu);
    test_multiple_voices(cpu);
    test_clear_drops_entries(cpu);
    test_null_tensors_on_miss_throws(cpu);

    ggml_backend_free(cpu);

    std::fprintf(stderr,
                 "test_supertonic_voice_host_cache: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
