# Supertonic OpenCL Optimization Plan (QVAC-18607)

R&D log + concrete next steps for taking Supertonic from "runs on OpenCL"
to "competitive on OpenCL".  Companion to `PROGRESS_SUPERTONIC.md`
("GPU bring-up: OpenCL (May 2026)" section); this file is the planning
+ test-strategy doc.

---

## Status

### Phase 1 — OpenCL correctness + first optimization (LANDED in QVAC-18607)

- `supertonic_model::backend_is_cpu` flag set from `ggml_backend_is_cpu()`.
- `supertonic_op_dispatch_scope` thread-local RAII scope set at every
  public `*_forward_ggml` / `*_trace_ggml` entry point.
- Every `ggml_custom_4d` (CBLAS-backed) site in the vocoder + vector
  estimator gated on `supertonic_use_cpu_custom_ops()` — falls through
  to the existing pure-GGML (`ggml_im2col + ggml_mul_mat`, `ggml_norm`,
  etc.) paths on any non-CPU backend.
- Portable `leaky_relu_portable_ggml()` — fused builtin on CPU,
  `RELU + SCALE + ADD` decomposition on GPU.
- F16 K/V flash-attention in the vector estimator (`use_f16_attn` flag);
  dispatches OpenCL's `flash_attn_f32_f16` instead of the F32-only
  kernel.  Auto-enables on GPU, off on CPU; CLI flag `--f16-attn 0|1`.

### Phase 1 testing (this PR)

- `test/test_supertonic_backend_dispatch.cpp` — unit test for
  `supertonic_op_dispatch_scope` (no GGUF required).  Covers default
  flag values, scope set/unwind on normal exit, exception safety,
  nested scope stacking.
- `test/test_supertonic_portable_ops.cpp` — CPU-backend parity test
  for `leaky_relu_portable_ggml()` vs `ggml_leaky_relu`.  Validates
  bit-exact F32 equivalence of the GPU-friendly rewrite.
- `test/test_supertonic_f16_attn_parity.cpp` — CPU-backend parity test
  for the F16-K/V `ggml_flash_attn_ext` path vs F32 K/V on synthetic
  Q/K/V tensors with the same shape used by the vector estimator.
  Validates that the F16 round-trip stays within ~1e-3 absolute error,
  which is the same tolerance chatterbox's `--cfm-f16-kv-attn` flag
  ships against.

---

## Phase 2 — Next optimization rounds

### 2A. F16 weight materialization for hot matmuls

**Motivation.**  Chatterbox's `OpenCL optimization log` (PROGRESS.md
§ Adreno bring-up) called out `kernel_mul_mm_f32_f32` as the second-
largest CFM bottleneck at ~138 ms / step after `flash_attn_f32_f16`
landed.  F16 weights halve the bandwidth into that matmul; chatterbox's
C1 path (`CHATTERBOX_F16_CFM`) gates a load-time F32→F16 conversion of
the CFM linear weights on a single env var.

For Supertonic the analogous hot list is:

| TU | Tensor pattern | Role |
|----|----------------|------|
| `supertonic_vector_estimator.cpp` | `vector_estimator:onnx::MatMul_*` | Q / K / V / out per group |
| `supertonic_vector_estimator.cpp` | `vector_estimator:tts.ttl.vector_field.main_blocks.*.convnext.*.pwconv{1,2}.weight` | pointwise conv 512↔1024 |
| `supertonic_vector_estimator.cpp` | `vector_estimator:tts.ttl.vector_field.last_convnext.convnext.*.pwconv{1,2}.weight` | tail pointwise conv |
| `supertonic_vocoder.cpp` | `vocoder:tts.ae.decoder.convnext.*.pwconv{1,2}.weight` | vocoder pointwise conv |
| `supertonic_vocoder.cpp` | `vocoder:tts.ae.decoder.head.layer{1,2}.net.weight` | head conv |
| `supertonic_vocoder.cpp` | `vocoder:onnx::Conv_1440` (embed_w) | vocoder embed |
| `supertonic_text_encoder.cpp` | `text_encoder:tts.ttl.text_encoder.*.*.weight` (transformer linears) | text encoder linears |

**Plan.**
1. Extend `supertonic_internal.h::supertonic_model` with a small
   `bool use_f16_weights` field plus an inline tensor predicate
   `should_materialise_f16_weight(const std::string & gguf_name)` whose
   pattern set matches the table above.
2. In `load_supertonic_gguf()`, when `use_f16_weights == true`, switch
   the `ggml_new_tensor` allocation type for matching tensors to
   `GGML_TYPE_F16` and convert the F32 GGUF payload to F16 before
   `ggml_backend_tensor_set`.  Reuse the existing `should_expand_*`
   path: today it forces *F16/Q8_0 → F32*; we add the inverse for the
   hot-weights list when the flag is on.
3. Engine option: `EngineOptions::f16_weights` (`-1` auto, `0`/`1`
   force).  Auto-enables on GPU, off on CPU (the cblas custom paths
   need F32).
4. CLI: `--f16-weights 0|1`.

**Acceptance.**
- New `test/test_supertonic_f16_weights.cpp` reloads the same GGUF
  with `use_f16_weights=true` and `=false`, runs
  `supertonic_pipeline_forward` on both, asserts:
  - Max abs error vs F32 baseline ≤ `2e-3` (looser than the
    pipeline-parity test's `1e-3` because of F16 weight rounding).
  - Audio cosine similarity ≥ `0.999`.
- Existing `test-supertonic-pipeline` still passes with the F32 path.
- `supertonic-bench --f16-weights 1` reports the matmul wall-time
  drop on the GPU path (manual perf gate, not a CTest one).

**Risk.**  Same as chatterbox C1: F16 rounding can drift on long
sequences.  Mitigation: keep `attn.W_out` and `proj_out` weights in
F32 (only the bulk linears go F16) — matches chatterbox's choice
to keep the projection-after-attention in F32.

### 2B. Pre-quantized GGUF Q8_0 weights

**Motivation.**  Chatterbox's Adreno log shows Q8_0 wasn't optimal
for the CFM (Q4_0 won there), but Q8_0 was the safe drop-in for the
S3Gen path.  Supertonic's bulk linears are dim-512 → 1024 (well above
Q4_0's quality knee for dense models), so Q8_0 is the right first
quantization to ship.

**Plan.**
1. Extend `scripts/convert-supertonic2-to-gguf.py` with a
   `--quantize Q8_0` flag that quantizes the same hot-weights list
   defined for 2A.
2. C++ side: nothing.  GGUF loader already round-trips Q8_0 (see
   `expand_supertonic_tensor_to_f32` in `supertonic_gguf.cpp` —
   today it *expands* Q8_0 to F32; we keep tensors at Q8_0 when on
   GPU and the GGUF was authored that way).
3. Add `model.hparams.weight_quant` to surface the quantization on
   `--verbose` and in `supertonic-bench` JSON output.

**Acceptance.**
- New `test/test_supertonic_q8_weights.cpp` runs the same pipeline
  on F32 + Q8_0 GGUFs of the same model.
  - Max abs error ≤ `3e-3`.
  - Audio cosine similarity ≥ `0.998`.
- New CI step or convert-script doc note for producing a `.q8.gguf`
  alongside the standard one.

**Risk.**  Q8_0 affects audio quality at the margins; need a sign-
off listening test on the canonical English + Portuguese smoke
prompts before shipping.

### 2C. Reduce host↔GPU sync round-trips in the vector estimator

**Motivation.**  Today each "island" in `supertonic_vector_step_ggml`
(text attention, style attention, group convnext × 4, tail) does its
own:

```
ggml_backend_tensor_set(...)     # upload host vector
supertonic_graph_compute(model, gf)
ggml_backend_tensor_get(...)     # download to host vector
```

On a discrete GPU (or even a unified-memory GPU running the OpenCL
driver) each `tensor_set / get` pair is a synchronisation point.  A
profile of the vector step on a typical English prompt shows 28
distinct `tensor_set` / `tensor_get` calls per step × 5 steps =
**140 host↔device round-trips** per synth, all blocking.

**Plan.**
1. Stitch the per-step island graphs into one larger
   `vector_step_graph_cache` per (latent_len, text_len, step,
   total_steps).  Intermediate tensors stay on GPU.
2. Keep the existing trace path (which intentionally peels each
   island off as a separately-allocated output) on the small
   per-island caches so the parity tests don't change.
3. Profile-mode opt-out (`SUPERTONIC_VECTOR_ROUNDTRIP_FALLBACK=1`)
   for debugging.

**Acceptance.**
- Existing `test-supertonic-vector` + `test-supertonic-pipeline`
  still pass bit-exactly (graph fusion must not change op order or
  introduce stride-induced rounding).
- New `test/test_supertonic_vector_roundtrip.cpp`:
  - Counts `tensor_set` / `tensor_get` calls via a backend hook in
    the test wrapper.
  - Asserts the count drops by ≥10× on the production path while
    staying identical on the trace path.
- `supertonic-bench` GPU vector wall time drops by the round-trip
  saving (manual perf gate).

**Risk.**  Large refactor.  Defer until 2A / 2B / F16 attn are
benchmarked and the per-island view is no longer worth its parity
guarantees.  Pre-req: the alive-id `gallocr_free` work from §7
("Persistent graph/allocr caches") needs to stay consistent
across the merged cache.

### 2D. OpenCL kernel-time profile mode

**Motivation.**  Today `SUPERTONIC_VECTOR_PROFILE=1` reports
per-island GGML wall time, but on OpenCL the wall time includes
queue submission + the actual kernel time + the readback.  Per-
kernel breakdown is what tells us whether the next bottleneck is
the matmul, the flash-attention, or the host→GPU copy.

**Plan.**
1. Add `SUPERTONIC_OPENCL_PROFILE=PATH.csv` env var, parsed in
   `init_supertonic_backend`.  When set on an OpenCL build, open
   the `CL_QUEUE_PROFILING_ENABLE` profile-enabled command queue
   (already supported by `ggml-opencl`) and write per-kernel
   `start_ns, end_ns, kernel_name` rows.
2. Convert PROGRESS.md's chatterbox `cl_profiling_*.csv` parser
   into a small Python helper under `scripts/` so the same
   tooling works for both repos.

**Acceptance.**
- Bench harness gains a `--cl-profile FILE.csv` arg that round-
  trips through env-var.
- Manual smoke: the CSV emitted on a quick English prompt has
  the expected ~50-70 kernel dispatches per step + ~600 per
  vocoder.

**Risk.**  None functional; profile-enabled queues have a small
overhead that doesn't matter for tuning runs.

### 2E. Eliminate host-side latent unpack

**Motivation.**  `unpack_latent_ggml_layout` in
`supertonic_vocoder.cpp` does a CPU-side `[1, 144, L] → [144, L*6]`
transpose before the GPU graph runs.  Replacing with
`ggml_permute + ggml_cont` keeps the unpack on the device.

**Plan.**
1. Build the unpack as the first ops of the vocoder graph.
2. The input tensor becomes `[1, latent_channels, latent_len]`
   directly.
3. Update the cache key.

**Acceptance.**
- `test-supertonic-vocoder` and `test-supertonic-vocoder-trace`
  still pass bit-exactly (the unpack is a pure permutation, so
  bit-exact is reasonable to demand).

**Risk.**  Low.  Small change, well-tested input shape.

---

## Test coverage matrix

| Phase | Test | Type | Requires GGUF? | Status |
|-------|------|------|----------------|--------|
| 1     | `test_supertonic_backend_dispatch` | unit | no  | NEW (this PR) |
| 1     | `test_supertonic_portable_ops`     | unit | no  | NEW (this PR) |
| 1     | `test_supertonic_f16_attn_parity`  | unit | no  | NEW (this PR) |
| 2A    | `test_supertonic_f16_weights`      | fixture | yes | planned |
| 2B    | `test_supertonic_q8_weights`       | fixture | yes (.q8.gguf) | planned |
| 2C    | `test_supertonic_vector_roundtrip` | fixture | yes | planned |
| 2D    | (manual perf gate via bench)       | manual | yes | planned |
| 2E    | extend `test_supertonic_vocoder`   | fixture | yes | planned |

CI labels: every Phase-1 unit test is `LABEL "unit"` (CPU-only, runs
without any fixture).  Every Phase-2 fixture test is `LABEL "fixture"`
and `REQUIRES` the model GGUF — auto-disabled with a clear message
when the model isn't present.

---

## Sequencing

Ordering for review + ship:

1. **This PR** — Phase 1 + the three unit tests above.  Lands the
   correctness + first OpenCL optimization (F16 K/V attn).
2. **Phase 2A** — F16 weights.  Independent change, easy to gate
   behind a flag, biggest expected single-flag win on Adreno
   after the F16 attn one.
3. **Phase 2E** — Vocoder unpack-on-GPU.  Small, useful housekeeping.
4. **Phase 2D** — OpenCL kernel profile mode.  Unblocks the next
   round of tuning by giving us per-kernel attribution.
5. **Phase 2B** — Q8_0 weights.  Needs convert-script work + an
   audio listening sign-off before ship.
6. **Phase 2C** — Round-trip elimination.  Biggest refactor; only
   justified once 2A/2B/2D have shifted the bottleneck onto the
   per-step host sync cost.

Each phase has the test gate spelled out above.  Tests are written
**before** the implementation lands (TDD); they remain red until
the corresponding optimization is in place, then turn green and
stay there.
