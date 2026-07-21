#pragma once

// Shared dual-path graph dispatch: direct single-backend compute when the
// primary backend supports every op in a graph, ggml_backend_sched over
// [primary, CPU-last] when it does not — the scheduler then routes the
// unsupported ops to CPU per-node instead of the backend asserting inside
// graph_compute.  Shared by all three engines: T3 (main.cpp/t3_mtl.cpp),
// S3Gen (chatterbox_tts.cpp s3gen_sched_prepare/s3gen_sched_compute) and
// Supertonic (supertonic_gguf.cpp supertonic_sched_*).
//
// The compute entry points return the backend's ggml_status so callers can
// surface a graceful compute failure instead of silently consuming
// whatever half-written output the graph left behind.
//
// Threading: NOT thread-safe by design.  Synthesis on one model is
// serialized (single-threaded CLI/engine synthesis loops); lazy creation is
// guarded by a plain null check, matching that contract.

#include "ggml-backend.h"

#include <vector>

namespace tts_cpp::detail {

// Lazily-created [primary, CPU-last] scheduler bundle; one per model.
//
// Ordering contract: the sched holds references into its backends, so a
// sched_fallback MUST be freed (sched_fallback_free) or destroyed BEFORE
// ggml_backend_free(primary).  Every production teardown site calls
// sched_fallback_free explicitly in that order; the destructor is a safety
// net for paths that never freed the primary at all (tests, early exits).
// Non-copyable/non-movable: a copy would double-free, and nothing moves it.
struct sched_fallback {
    ggml_backend_t       cpu_backend = nullptr;  // owned; null when primary is CPU
    ggml_backend_sched_t sched       = nullptr;  // owned

    sched_fallback() = default;
    sched_fallback(const sched_fallback &)             = delete;
    sched_fallback & operator=(const sched_fallback &) = delete;
    sched_fallback(sched_fallback &&)                  = delete;
    sched_fallback & operator=(sched_fallback &&)      = delete;
    ~sched_fallback();
};

// True iff `backend` supports every node op in `gf`.  Read-only walk with
// early exit — safe to probe cached graphs.  Same node-only walk as the
// S3Gen HiFT and Supertonic dispatch gates (supports_op itself answers
// VIEW/NONE-class nodes).
bool graph_fully_supported(ggml_backend_t backend, const ggml_cgraph * gf);

// Test/debug escape hatch: force the sched path even when the walk passes.
// Reads TTS_CPP_FORCE_SCHED on every call (NOT cached in a static) so
// tests can toggle it per-phase within one process.
bool sched_force_enabled();

// Pre-sched abort guard: true if some node is BOTH bound to a pre-allocated
// buffer (node->buffer, or view_src->buffer — e.g. an op writing a
// persistent KV-cache slab) AND runnable by neither the primary backend nor
// the CPU device for that buffer type.  ggml_backend_sched would GGML_ABORT
// on such a node ("pre-allocated tensor in a buffer that cannot run the
// operation") instead of falling back; callers must fail gracefully rather
// than enter the scheduler.
bool graph_has_unsupported_preallocated_op(ggml_backend_t primary, const ggml_cgraph * gf);

// Lazy one-time creation, guarded by the null check (see threading note
// above): marks each non-null buffer in `weight_buffers` USAGE_WEIGHTS (so
// sched may copy a primary-resident weight to CPU for a CPU-routed op),
// creates cpu_backend (skipped when the primary is the CPU) and the sched
// over [primary, cpu-last] with parallel=false, op_offload=false.
// Returns false with a stderr log on failure; a failed attempt leaves the
// bundle empty, so a later call retries naturally.
bool sched_fallback_ensure(sched_fallback & fb, ggml_backend_t primary,
                           size_t graph_size,
                           const std::vector<ggml_backend_buffer_t> & weight_buffers);

// Reset-at-head + allocate: sched allocates at alloc time, so callers set
// input data AFTER this and read outputs after compute, before the next
// reset.  Returns false on allocation failure.
//
// GRAPH LIFETIME: `gf` must be FRESHLY BUILT for every call — sched graphs
// are single-use for allocation (ggml-backend.h: alloc rewires node->src[]
// into sched-owned copies that the next pass frees; tensor buffer/data
// bindings survive the reset).  Feeding a previously sched-allocated graph
// back in computes garbage deterministically or crashes.  T3, HiFT and
// every Supertonic dual-path site rebuild before each sched pass.
bool sched_fallback_alloc(sched_fallback & fb, ggml_cgraph * gf);

// Set the CPU-side thread count, run the graph through the scheduler and
// return its status (the scheduler propagates the failing split's status;
// it does not retry).
ggml_status sched_fallback_compute(sched_fallback & fb, ggml_backend_t primary,
                                   ggml_cgraph * gf, int n_threads);

// Direct-path twin so call sites get a status check without duplicating the
// n_threads plumbing: registry-routed set_n_threads + graph_compute.
ggml_status direct_compute(ggml_backend_t backend, ggml_cgraph * gf, int n_threads);

// Free sched then cpu_backend (that order — the sched holds backend refs).
// MUST run before ggml_backend_free(primary).  Idempotent; a later
// ensure() can rebuild.
void sched_fallback_free(sched_fallback & fb);

} // namespace tts_cpp::detail
