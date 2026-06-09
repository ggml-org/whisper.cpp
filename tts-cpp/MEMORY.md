# tts-cpp memory model

Steady-state RSS / VRAM behaviour for the two persistent engines and
the per-process caches they share.  Numbers below were measured on
Apple M2 (macOS 15.7, arm64) with the supertonic CPU + Metal paths;
RSS is `mach_task_basic_info::phys_footprint`.  The measurement
harness is `test/test_supertonic_engine_cycle.cpp`
(`build/test-supertonic-engine-cycle`).

## Acceptance criteria (CI gate)

Both engines must satisfy:

- **No monotonic RSS growth** across 100+ `synthesize → destroy` cycles
  (drift < 5 MB).  Enforced for supertonic by
  `test-supertonic-engine-cycle`.  Chatterbox does not currently
  have a fixture-free cycle harness in tree; the lifecycle invariants
  it relies on are documented below and the existing tests
  (`test-t3-caches`, `test-cpu-caches`) cover the cache contracts.
- **Backend-buffer release ordering**:  per-stage caches that hold
  `ggml_gallocr_t` handles must be torn down BEFORE
  `ggml_backend_free`.  Reversing that order asserts inside the
  GPU-backend dylib finaliser (ggml-metal's
  `[rsets->data count] == 0` check is the canonical example).
- **Cancel-mid-flight cleanup**:  `cancel()` throws out of
  `synthesize()` at the next stage boundary; local scratch buffers
  unwind via RAII and the populated caches stay valid (they're keyed
  on `(model.generation_id, shape)` so a subsequent `synthesize()`
  reuses them without rebuilds).

## Supertonic engine — measured

| Phase                           | RSS (CPU) | RSS (Metal) | Notes                                       |
| ------------------------------- | --------: | ----------: | ------------------------------------------- |
| Process baseline (no engine)    |    2.5 MB |      2.5 MB | ggml/tts-cpp dylibs not yet resolved        |
| After first `synthesize()`      |   311 MB  |     207 MB  | model + caches + backend init               |
| After Engine destruction        |   293 MB  |     202 MB  | one-shot inits stay; per-engine state freed |
| Steady-state cycle drift (n=100)|  ±5 MB   |     ±5 MB  | within allocator + page-pool noise          |

Where the ~290 MB residual goes after the Engine is destroyed:

- **ggml-metal library + device state** (~100 MB on Metal, ~0 on CPU).
  The metal kernel library is compiled+loaded ONCE per process; the
  resulting `MTLLibrary` and pipeline cache live on a static device
  list inside `libqvac-speech-ggml-metal.dylib`.
- **MoltenVK / ggml-vulkan ICD** (~30 MB; non-zero even when
  n_gpu_layers=0 because the ggml backend registry walks every
  registered backend at first `Engine` ctor).
- **ggml backend registry, OpenMP pool, libstdc++ allocator pools**
  (~60 MB combined).
- **Supertonic GGUF residuals** — model weights are freed but the
  underlying mmap'd pages may stay resident under macOS memory
  pressure semantics (the allocator returns the region but the
  OS may keep pages mapped until pressure rises).  Counts toward
  RSS, not "leaked" — releases on `madvise` / OS-level reclaim.
- **Per-stage thread_local cache CPU buffers** (~2.5 MB per worker
  thread that has ever run a Supertonic synth).  These hold ggml
  init bufs + graph metadata; intentionally retained across Engine
  cycles for fast graph rebuild on the next synth.  Released by
  the `release_*_thread_local_caches` machinery at engine
  destruction on the synth-owning thread.

## Supertonic engine — load/unload cycle invariants

`free_supertonic_model` (called from `Engine::~Impl`) executes in this
order, which is the documented invariant downstream callers should
not break:

1. `release_vector_estimator_thread_local_caches()` /
   `release_text_encoder_thread_local_caches()` /
   `release_vocoder_thread_local_caches()` /
   `release_duration_thread_local_caches()` — drives every
   per-stage thread_local cache populated on the calling thread
   through its normal `free_*_cache` path WHILE the backend is
   still alive, so each gallocr inside gets a complete
   `ggml_gallocr_free` (no skip).
2. `unregister_supertonic_alive(model.generation_id)` — any
   thread_local cache on OTHER threads (cross-thread Engine
   destruction, not supported per the contract) now sees its
   generation as no-longer-alive and the skip-and-leak path in
   `supertonic_safe_gallocr_free` kicks in to keep us from
   crashing.
3. `ggml_backend_sched_free(model.sched)` — the QVAC-19254
   scheduler holds non-owning pointers to the backends, must die
   first.
4. `ggml_backend_buffer_free(model.buffer_w_extra)` /
   `model.buffer_w` — weight buffers, allocated against the
   primary backend.
5. `ggml_backend_free(model.backend)` — primary compute backend.
6. `ggml_backend_free(model.cpu_backend)` — CPU fallback backend
   (allocated only when primary is non-CPU).
7. `ggml_free(model.ctx_w_extra)` / `model.ctx_w` — tensor
   metadata contexts (CPU memory only).

## Supertonic — intentional retained CPU memory after destruction

Per worker thread that has ever run a Supertonic synth, the
following caches' **CPU-side metadata** persists from first call
until thread death:

| Cache TU                           | Per-thread size (typical) |
| ---------------------------------- | -------------------------: |
| supertonic_vector_estimator.cpp    | ~2.0 MB (23 caches × ~85 KB ctx+buf each, after engine cycle frees the gallocators) |
| supertonic_text_encoder.cpp        | ~0.4 MB (6 caches; 5 are arrays of 2–4 instances) |
| supertonic_vocoder.cpp             | ~64 KB (1 cache) |
| supertonic_duration.cpp            | ~32 KB (1 cache) |

These are the `buf` storage vector + `ggml_context` headers that the
caches reuse across synths for fast graph rebuild.  The `ggml_gallocr_t`
inside each cache IS freed at engine destruction (post-fix) — only
the cache wrapper's CPU memory stays around so the next engine cycle
hits the `cache.generation_id != model.generation_id` rebuild path
with all metadata already allocated.

If you need a thread-pool host to fully reclaim this after engines
go idle, the cheapest path is to terminate the worker (thread_local
destructors run at thread death; `cache.buf.~vector()` releases the
heap storage).  There is currently no public API to release the
retained CPU metadata without thread termination.

## Chatterbox engine — lifecycle invariants (code review only)

We do not have local chatterbox GGUFs to run a cycle bench, but the
teardown code in `src/chatterbox_engine.cpp` + `src/chatterbox_tts.cpp`
+ `src/t3_mtl.cpp` follows these documented invariants:

1. `wait_for_preload(s3gen_preload_thread)` — block until the
   background S3Gen GGUF load thread completes; otherwise the
   destructor races the preload thread inside ggml-metal's
   `ggml_metal_buffer_type_shared_alloc_buffer`.
2. `s3gen_unload()` — refcount-protected.  When the count reaches
   zero, calls `s3gen_model_cache_release()` which
   `s3gen_release_synth_caches()` first (cfm_estimator, encoder,
   hift, f0, stft, time-mlp results, weight CPU mirrors,
   pos_emb / inv_alpha scaffolding, hann_window / istft_kernel /
   window_sum / stft_kernel) BEFORE freeing the s3gen
   `model_ctx`'s scheduler + backend.
3. `ggml_gallocr_free(allocr)` — the T3 prompt+step gallocr.
4. `free_model()` — calls `t3_stack_unregister` (pull the
   `(buffer_stack, ctx_stack)` pair out of the process-wide
   `t3_stack_registry` BEFORE local free), then
   `t3_release_caches()` (drains the T3 step-graph cache —
   process-global, mutex-protected LRU bounded at 256 entries
   when `CHATTERBOX_T3_STEP_CACHE=1`), then frees T3 buffers +
   backend + contexts in that order.

### Chatterbox process-singleton caches

- **S3Gen model cache** (`g_s3gen_cache_entry` in chatterbox_tts.cpp).
  One `model_ctx` per (path, n_gpu_layers) pair, refcounted across
  multiple Engine instances sharing the same S3Gen GGUF.  Released
  by the last `s3gen_unload()` or by the `atexit` hook
  (`s3gen_model_cache_release`) on process exit.
- **T3 step-graph cache** (LRU, bounded at 256 entries, opt-in via
  `CHATTERBOX_T3_STEP_CACHE=1`).  Each entry holds a
  `ggml_context *` (one per `(n_past, is_uncond)` cache key) +
  the graph metadata `buf` (~MB-scale on the multilingual model).
  Cap @ 256 → roughly 256 × graph-metadata bytes upper bound;
  measured ~270 MB peak on the multilingual model when fully
  saturated.  Cleared by `detail::t3_release_caches()` (called
  from Engine destructor) or by the `atexit` hook
  (`t3_step_cache_release_atexit`) at process exit.
- **T3 stack registry** (`t3_stack_registry` in t3_mtl.cpp).
  Holds `(ggml_backend_buffer_t, ggml_context *)` pairs registered
  by load-time MTL build paths so the `atexit` hook can free them
  before the GPU dylib's static finaliser runs.  Each Engine's
  destructor pulls its registered entries before its own backend
  free.

### Chatterbox follow-up — known concern

`src/chatterbox_tts.cpp::time_mlp_cache` (line 1270, inside
`compute_time_mlp`) is a thread_local cache whose destructor calls
`ggml_gallocr_free(allocr)` directly (no safe-skip helper).  If a
worker thread outlives the Engine that populated it and the backend
has been freed, the thread_local destructor at thread death will
attempt `ggml_gallocr_free` against a dead backend and assert inside
the dylib finaliser.

Empirically this doesn't trigger on chatterbox CLI runs (single
thread, backend lives until process exit) but it's a latent risk
for thread-pool hosts (SDK / Bare addon).  A follow-up fix would
mirror the supertonic registry pattern for chatterbox or move the
time_mlp_cache into `s3gen_release_synth_caches()`'s sweep.

## Per-Engine and per-process state inventory

| State                                    | Scope                | Released by                            |
| ---------------------------------------- | -------------------- | -------------------------------------- |
| Supertonic model.backend + buffers       | Engine               | `free_supertonic_model`                |
| Supertonic per-stage thread_local caches | Per-thread, per-Engine generation | `release_*_thread_local_caches` (synth thread) on dtor, OR thread death |
| Supertonic alive registry                | Process              | atomic — Engine ctor adds, dtor drops  |
| Chatterbox T3 model.backend + buffers    | Engine               | `Engine::~Impl::free_model`            |
| Chatterbox S3Gen model_ctx               | Process (refcount)   | last `s3gen_unload()` / process atexit |
| Chatterbox T3 step-graph cache           | Process              | `t3_release_caches()` / process atexit |
| Chatterbox T3 stack registry             | Process              | `t3_stack_unregister` / process atexit |
| ggml backend registry (Metal, Vulkan, …) | Process              | Process exit (dylib finaliser)         |
| OpenMP / MoltenVK / Metal compiler pools | Process              | Process exit                           |
| GGUF mmap pages                          | Process (until OS pressure) | `madvise` on free + OS-level reclaim |

## Cycle harness usage

```
build/test-supertonic-engine-cycle <supertonic.gguf> [REF_DIR_ignored] [n_cycles=20] [n_gpu_layers=0]
```

- `n_cycles >= 2` required.  First cycle's RSS captures
  one-time process inits; cycles 2..N are compared against it
  with a 5 MB tolerance.
- `n_gpu_layers=99` exercises the Metal / Vulkan / OpenCL path
  (whichever the backend cascade resolves to first).
- Exits non-zero if max RSS across cycles 2..N exceeds first-cycle
  RSS + 5 MB.
- The synthesis is a fixed sentence ("The quick brown fox …")
  so this is a memory-only assertion; numerical parity is
  covered by the `test-supertonic-{pipeline,vector,text-encoder,
  vocoder,duration}` harnesses.
