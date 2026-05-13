# Supertonic → ggml Port: Development Journal

This document tracks the experimental **Supertonic / Supertonic 2** GGUF +
GGML runtime added to this repo: what was tested, what matched, what sounded
good, which performance ideas worked, and which optimization attempts were
rolled back or deferred.

It is separate from `PROGRESS.md`, which covers the Chatterbox Turbo and
Chatterbox Multilingual ports.  Supertonic is a different architecture and is
currently implemented as a model-specific runtime over official ONNX weights
converted into one GGUF.

- **Models**:
  - `Supertone/supertonic` — stable English bundle, no language wrapping.
  - `Supertone/supertonic-2` — multilingual bundle, open/close language tags
    (`<lang>text</lang>`).
- **Goal**: run the known Supertonic stages in C++/GGML with numerical parity
  against ONNX Runtime, clean audio output, and production-grade CPU
  performance.
- **Final CPU benchmark target**: matched GGML vs ONNX Runtime
  `CPUExecutionProvider` at 1, 2, 3, and 4 threads.

---

## Current Status

The branch now contains a full Supertonic path:

| Binary / script | Role |
|---|---|
| `scripts/setup-supertonic2.sh` | Downloads the official Hugging Face bundle and writes the local GGUF. |
| `scripts/convert-supertonic2-to-gguf.py` | Converts official ONNX/assets into `models/supertonic2.gguf` or `models/supertonic.gguf`. |
| `build/tts-cli` | Autodetects `supertonic.arch` and routes Supertonic text → 44.1 kHz wav on CPU. |
| `build/supertonic-cli` | Focused Supertonic compatibility/debug wrapper. |
| `build/supertonic-bench` | Per-stage Supertonic benchmark with JSON output. |
| `test-supertonic-*` | Stage and trace parity harnesses against ONNX reference dumps. |

The generated GGUF files are intentionally not committed:

```text
models/supertonic.gguf   ~250 MB
models/supertonic2.gguf  ~251 MB
```

They are ignored by `.gitignore` (`models/`, `*.gguf`), matching the existing
Chatterbox approach where converters/setup scripts create local model files.

### Correctness

The full path is implemented, and all model stages are routed through the
GGML-backed production path:

1. preprocess
2. duration predictor
3. text encoder
4. vector estimator
5. vocoder

The end-to-end pipeline parity check against the Supertonic 2 ONNX reference
passes:

| Check | Result |
|---|---:|
| `test-supertonic-pipeline` max abs | `3.431e-05` |
| `test-supertonic-pipeline` RMS | `2.086e-06` |
| vocoder pointwise harness | PASS |

Audio checks were clean for generated English, French, and Portuguese samples.

### Final CPU Benchmark

Final benchmark settings:

- GGML: `models/supertonic2.gguf`
- ONNX: official Supertonic 2 ONNX files via ONNX Runtime
  `CPUExecutionProvider`
- Voice: `F1`
- Steps: `5`
- Speed: `1.05`
- Runs: `3`, warmup: `1`
- Prompts: quick English, longer English, Portuguese smoke
- Thread matrix: 1v1, 2v2, 3v3, 4v4

Median total wall time in milliseconds:

| Prompt | GGML 1t | GGML 2t | GGML 3t | GGML 4t | ONNX 1t | ONNX 2t | ONNX 3t | ONNX 4t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quick English | 298.0 | 189.4 | 157.7 | 157.7 | 373.8 | 218.5 | 168.3 | 148.8 |
| longer English | 757.5 | 491.2 | 390.3 | 361.2 | 1103.0 | 580.6 | 555.7 | 351.5 |
| Portuguese smoke | 457.2 | 292.9 | 251.0 | 234.3 | 610.6 | 344.6 | 268.3 | 250.8 |

Headline:

- GGML wins **10 / 12** matched comparisons.
- GGML wins **all 1-thread** comparisons.
- GGML vocoder wins the 4-thread stage comparison on all tested prompts.
- Remaining losses are narrow:
  - quick English 4t: GGML `157.7 ms` vs ONNX `148.8 ms`
  - longer English 4t: GGML `361.2 ms` vs ONNX `351.5 ms`

4-thread stage medians:

| Prompt | Runtime | Duration | Text | Vector | Vocoder | Total |
|---|---|---:|---:|---:|---:|---:|
| quick English | GGML | 3.9 | 13.5 | 96.3 | 43.6 | 157.7 |
| quick English | ONNX | 1.5 | 11.5 | 85.9 | 49.8 | 148.8 |
| longer English | GGML | 11.9 | 33.3 | 201.2 | 115.1 | 361.2 |
| longer English | ONNX | 2.4 | 13.1 | 198.3 | 138.8 | 351.5 |
| Portuguese smoke | GGML | 6.5 | 20.8 | 137.6 | 68.9 | 234.3 |
| Portuguese smoke | ONNX | 1.7 | 11.6 | 141.7 | 95.6 | 250.8 |

---

## Repository Additions

```text
include/tts-cpp/supertonic/engine.h       public Supertonic synth API
scripts/convert-supertonic2-to-gguf.py    ONNX/assets → Supertonic GGUF
scripts/setup-supertonic2.sh              download + convert wrapper
scripts/dump-supertonic-reference.py      ONNX reference tensor dumper
scripts/bench-supertonic-onnx.py          ONNX Runtime benchmark script
src/supertonic_gguf.cpp                   GGUF loader + backend/thread setup
src/supertonic_preprocess.cpp             Unicode/text preprocessing
src/supertonic_duration.cpp               duration predictor
src/supertonic_text_encoder.cpp           text encoder
src/supertonic_vector_estimator.cpp       vector denoiser
src/supertonic_vocoder.cpp                vocoder
src/supertonic_engine.cpp                 end-to-end Supertonic API
src/supertonic_cli.cpp                    standalone Supertonic CLI
src/supertonic_bench.cpp                  GGML benchmark harness
src/test_supertonic_*.cpp                 stage parity and trace tests
```

---

## Development Log

### 1. Scoping: ONNX → GGUF is feasible, generic ONNX execution is not needed

The first decision was to avoid a generic ONNX executor.  Supertonic has four
known ONNX submodels plus stable assets (`tts.json`, `unicode_indexer.json`,
voice styles).  That makes a model-specific converter and model-specific C++
runtime the right shape.

The GGUF stores:

- all ONNX initializers
- tensor-valued ONNX constants
- `tts.json` metadata
- Unicode indexer
- built-in voice styles
- arrays mapping short GGUF tensor names back to the original ONNX source names

This source-name mapping was important.  Some ONNX tensor names are long or not
pleasant as ggml tensor names, but the C++ runtime can still request weights by
their original source names.

### 2. Early audio finding: stutter was language wrapping, not GGUF

The first audible issue was English stuttering / mechanical audio in
Supertonic 2.  The root cause was not the C++ port or GGUF conversion.

What failed:

- Old Supertonic 2 prefix-only wrapping:

```text
<en>text 
```

What worked:

- Stable English bundle (`Supertone/supertonic`) with no wrapping.
- Supertonic 2 multilingual bundle with open/close wrapping:

```text
<en>text</en>
<pt>text</pt>
<fr>text</fr>
```

This is now encoded in GGUF metadata as `supertonic.language_wrap_mode`, and the
runtime follows the metadata.

### 3. Reference and parity harnesses

Added ONNX reference dump scripts and stage tests before optimizing.  This was
essential because several later "obvious" graph fusions produced valid-looking
output tensors with bad data.

Useful parity tools:

- `test-supertonic-preprocess`
- `test-supertonic-duration`
- `test-supertonic-duration-trace`
- `test-supertonic-text-encoder`
- `test-supertonic-text-encoder-trace`
- `test-supertonic-vector`
- `test-supertonic-vector-trace`
- `test-supertonic-vocoder`
- `test-supertonic-vocoder-trace`
- `test-supertonic-vocoder-pointwise`
- `test-supertonic-pipeline`

Important reproducibility fix:

- C++ `std::normal_distribution` does not match NumPy's `RandomState`.
- The runtime now uses a NumPy-compatible MT19937 + `standard_normal()` path so
  `--seed 42` matches the ONNX/Python reference noise behavior.

### 4. Baseline: scalar C++ proved correctness but was far behind ONNX

The first full C++ path was useful for parity but not performance.

Initial scalar-era benchmark on the quick prompt showed roughly:

| Stage | ONNX | early C++ |
|---|---:|---:|
| duration | 1.72 ms | 8.28 ms |
| text encoder | 9.33 ms | 211.97 ms |
| vector estimator | 99.90 ms | 7156.24 ms |
| vocoder | 69.03 ms | 7080.52 ms |
| total | 180.32 ms | 14451.06 ms |

This made the priority clear: vector estimator and vocoder dominated the wall
time, followed by the text encoder.

### 5. Production controls: threading and BLAS policy

What worked:

- Add `supertonic_set_n_threads()`.
- Route all graph execution through `supertonic_graph_compute()`.
- Set CPU backend thread count before graph compute.
- Cap default thread count at 4 for the current small-graph Supertonic path.
- Cap BLAS worker threads by default:
  - `VECLIB_MAXIMUM_THREADS=1` on Accelerate
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `BLIS_NUM_THREADS=1`

Why this mattered:

The Supertonic CPU runtime already parallelizes work through GGML tasking and
custom-op task splits.  Letting BLAS also spawn worker pools for every small
pointwise matmul hurt 3-4 thread scaling.

### 6. Text encoder optimization

What worked:

- Move the text encoder production path to GGML.
- Express text ConvNeXt blocks in GGML.
- Use `ggml_flash_attn_ext` for speech-prompted attention.
- Implement relative-position self-attention with stock GGML ops.
- Cache relative-position attention graphs (`text_relpos_graph_cache`).
- Move FFN blocks from scalar C++ loops to cached GGML graphs.
- Refactor Q/K/V projections so outputs are closer to the needed channel-major
  layout and avoid some reshape/permute/contiguous overhead.

What did not get implemented yet:

- A custom fused relpos attention op.

Why it was deferred:

Profiling showed stock-op relpos was around `0.7-0.8 ms/layer` on the quick
prompt after the cached graph/FFN work.  That is not free, but the bigger
performance opportunities were still in vector/vocoder and graph boundary
overhead.

### 7. Vector estimator optimization

The vector estimator was the largest and most complicated optimization target.
It runs multiple attention and ConvNeXt-style groups per denoise step, then
repeats for the configured number of steps.

What worked:

- Split trace and production paths so production no longer scans debug trace
  vectors.
- Cache host-side static layout conversions for text embeddings and style
  contexts.
- Split text attention into QKV projection and attention-only cached graphs.
- Split style attention similarly.
- Reuse attention-only graph states for text and style attention.
- Replace D/L/H host packing with strided GGML views where layout allows it.
- Add persistent graph/allocr caches for vector attention, group, and tail
  islands.
- Gate intermediate graph outputs with `trace_outputs=false` in production.
- Fuse ConvNeXt group output with following text-attention QKV projection.
- Fuse residual/post-ConvNeXt boundaries with following style QKV projection.
- Fuse tail projection/update into a custom production op.
- Replace graph transpose-heavy dense time matmul with a direct BLAS custom op.
- Fuse ConvNeXt elementwise work:
  - `pw1 bias + GELU`
  - `pw2 bias + gamma + residual`

Portable custom CPU kernels added:

- K=1 pointwise Conv1D, BLAS/Accelerate-backed.
- K=5 depthwise Conv1D custom op with unrolled hot path.
- General fallback for other depthwise kernels.
- Direct row-wise layer norm.
- Direct dense time matmul.
- Tail update fusion.

What failed or was rolled back:

| Attempt | Result |
|---|---|
| Fold style residuals directly into attention graphs | Rolled back. Trace showed in-graph residual add corrupted the left-hand activation, likely due to GGML buffer lifetime / aliasing. |
| Temporary reusable D/L/H host packing buffers | Helped but was superseded by strided GGML views, which avoid the packing entirely where possible. |
| Broad graph folding without parity trace boundaries | Too risky. The vector trace harness showed small-looking graph rewrites can corrupt later residual paths. |

Main remaining vector issue:

- At higher thread counts, vector is close to ONNX but still has some variance.
- The next target should be graph scheduling/scaling stability, not a broad
  rewrite.

### 8. Vocoder optimization

The vocoder started as one of the two massive scalar bottlenecks.

What worked:

- Convert vocoder execution to a persistent GGML graph cache.
- Add a vocoder pointwise harness to isolate weight layout, BLAS layout, and
  custom-op parity.
- Use BLAS/Accelerate-backed K=1 causal Conv1D for hot projection paths.
- Use BLAS-backed K>1 causal Conv1D for `head1`.
- Keep the rest of the graph stable and parity-checked.

What failed:

| Attempt | Result |
|---|---|
| Broad K=1 BLAS replacement across vocoder too early | Failed parity until layout and tasking were isolated. |
| Custom op running BLAS work on every GGML task | Race / concurrent writes. Fixed by only doing the BLAS call on `ith == 0` for those ops. |
| Wrong transpose assumption for Conv1D weights | Produced large errors. The pointwise harness confirmed the correct `blas_col_nn` mapping. |

Final important point:

The vocoder is no longer the bottleneck.  In the final 4-thread comparison,
GGML vocoder beats ONNX on all three tested prompts.

### 9. Benchmark tooling

Added machine-readable benchmark output on both sides:

- `supertonic-bench --json-out`
- `scripts/bench-supertonic-onnx.py --json-out`
- `scripts/bench-supertonic-onnx.py --providers CPUExecutionProvider`
- `scripts/bench-supertonic-onnx.py --threads`
- `scripts/bench-supertonic-onnx.py --language-wrap-mode open_close`

This avoided a repeated source of confusion: ONNX and GGML must use the same
language wrapping, prompt, voice, steps, speed, thread count, and CPU provider.

Final matrix artifacts were written under:

```text
artifacts/supertonic-thread-matrix/
```

That directory is intentionally ignored.

### 10. Setup and local model workflow

The GGUF is not committed.  The repo now follows the Chatterbox pattern:

- converters/setup scripts create the local model
- runtime stays network-free
- missing model errors point users to setup commands

Common setup:

```bash
# Multilingual Supertonic 2
bash scripts/setup-supertonic2.sh

# Stable English Supertonic
bash scripts/setup-supertonic2.sh --arch supertonic
```

The lower-level converter also supports local ONNX assets:

```bash
python scripts/convert-supertonic2-to-gguf.py \
  --onnx-dir /path/to/supertonic-pytorch/onnx_models/onnx \
  --assets-dir /path/to/supertonic-pytorch/assets \
  --out models/supertonic2.gguf \
  --validate
```

---

## What Worked Best

1. **Parity-first development.**

   The trace harnesses caught layout bugs and graph aliasing failures that would
   otherwise have shown up only as bad audio.

2. **Model-specific GGUF, not generic ONNX execution.**

   Supertonic's stage boundaries are stable enough that a dedicated converter
   and runtime are simpler and faster.

3. **Open/close language wrapping for Supertonic 2.**

   This solved the English stutter without changing model math.

4. **Persistent GGML graph/allocr caches.**

   Reusing graph structure was essential for small repeated vector/text islands.

5. **Strided attention views.**

   Avoiding host D/L/H packing reduced repeated layout overhead and better
   matches the Chatterbox-style GGML approach.

6. **Targeted portable custom CPU kernels.**

   Pointwise Conv1D, depthwise Conv1D, row-wise layer norm, and dense time
   matmul were the right level of specialization: portable C++/CBLAS/Accelerate
   without locking the runtime to one CPU vendor.

7. **BLAS thread caps.**

   Preventing nested thread pools improved scaling stability.

8. **The isolated vocoder pointwise harness.**

   It quickly separated weight-layout bugs from GGML custom-op scheduling bugs.

---

## What Did Not Work

1. **Assuming ONNX/PyTorch reconstruction quality represented the official path.**

   The unofficial PyTorch reconstruction was useful for exploration but not a
   reliable audio-quality source.  Official ONNX assets plus correct wrapping
   were the right reference.

2. **Prefix-only language tags for Supertonic 2 English.**

   This caused audible stutter.  Use no wrapping for stable English
   `Supertone/supertonic`, and open/close wrapping for Supertonic 2.

3. **Folding graph boundaries before proving alias safety.**

   A style residual fold corrupted activations due to GGML buffer aliasing risk.
   Graph fusion must be guarded by trace parity.

4. **Broad custom-kernel rollout without isolated harnesses.**

   The vocoder K=1 BLAS path only became reliable after the isolated pointwise
   harness proved the exact tensor/BLAS layout.

5. **Letting BLAS and GGML both freely multi-thread.**

   Nested thread pools hurt the small-island workload.

6. **Trying to optimize only for Apple Accelerate.**

   The final custom kernels were kept portable: Accelerate where available,
   generic CBLAS elsewhere, and scalar fallbacks for unsupported cases.

---

## GPU bring-up: OpenCL (May 2026)

Target: the same `--n-gpu-layers > 0` flag already exposed by the
Supertonic CLI, but resolved to **OpenCL** instead of falling back to
CPU.  Tracking ticket: QVAC-18607.

### What was missing

The Supertonic CPU path (§7-§8 above) earned its CPU benchmark wins by
moving every hot loop onto a `ggml_custom_4d` op whose callback runs
CBLAS / pointer-arithmetic directly against the tensor `data` field:

| TU | Custom ops |
|----|-----------|
| `supertonic_vocoder.cpp` | K=1 cblas conv1d, K>1 cblas conv1d, depthwise dilated conv1d |
| `supertonic_vector_estimator.cpp` | conv1d_f32(K=1), depthwise same-padded conv1d, row-wise layer-norm, dense-time matmul, fused bias+GELU, fused (pw2 bias + γ + residual), fused tail-update (BLAS GEMM + mask + step-scale + residual add) |

None of those callbacks are valid on a GPU backend: `GGML_OP_CUSTOM`
isn't supported by `ggml-opencl` (or by CUDA / Metal / Vulkan), and the
op callbacks themselves assume host-addressable `data` pointers that
no GPU backend exposes inside graph execution.  So before this round,
loading Supertonic with `--n-gpu-layers > 0` either fell straight back
to CPU via `init_supertonic_backend` (when the backend wasn't compiled
in) or asserted at `ggml_backend_graph_compute` time inside the OpenCL
dispatch loop (when it was).

In addition, two builtins in the vocoder graph had similar portability
holes against baseline upstream OpenCL: `ggml_leaky_relu`
(`GGML_OP_LEAKY_RELU`) is only present on `ggml-opencl` builds that
carry the chatterbox `ggml-opencl-chatterbox-ops.patch` — fine for the
QVAC `ggml-speech` vcpkg consumption path, but unsafe for any other
GPU backend wanting Supertonic.

### What landed

| Change | File(s) |
|--------|---------|
| `supertonic_model::backend_is_cpu` set from `ggml_backend_is_cpu(model.backend)` right after `init_supertonic_backend()` resolves the device. | `supertonic_gguf.cpp`, `supertonic_internal.h` |
| `supertonic_op_dispatch_scope` — thread-local RAII helper instantiated at every public `supertonic_*_forward_ggml` / `*_trace_ggml` entry point.  Mirrors `model.backend_is_cpu` and `model.use_f16_attn` into the two thread-local flags consulted by the graph-build helpers. | `supertonic_internal.h`, `supertonic_gguf.cpp`, `supertonic_vocoder.cpp`, `supertonic_vector_estimator.cpp`, `supertonic_text_encoder.cpp`, `supertonic_duration.cpp` |
| Every `ggml_custom_4d` site gated on `supertonic_use_cpu_custom_ops()` so GPU runs fall through to the existing pure-GGML paths (`ggml_im2col + ggml_mul_mat`, `ggml_norm`, etc.) — all of which `ggml-opencl` already supports natively (see `ggml_opencl_supports_op()` in `ggml/src/ggml-opencl/ggml-opencl.cpp`). | `supertonic_vocoder.cpp`, `supertonic_vector_estimator.cpp` |
| Portable `leaky_relu_portable_ggml()` helper: on CPU keeps the fused builtin; on GPU decomposes into `RELU + SCALE + ADD`, all universally supported. | `supertonic_vocoder.cpp` |

### Optimization #1: F16 K/V flash-attention

The vector estimator's text-conditioned attention runs four times per
denoising step × N steps, so it's the single hottest op in the
Supertonic synthesis budget after the dense convnext blocks.  Lifted
straight from chatterbox's Adreno bring-up (§ `OpenCL optimization
log`), the vector-estimator graph now optionally materialises K / V
into contiguous F16 before calling `ggml_flash_attn_ext`, which makes
OpenCL dispatch the `flash_attn_f32_f16` kernel instead of the
F32-only one.  In chatterbox's Q4_0 CFM smoke run this dropped the
attention kernel from `~257 ms` to `~102 ms` on Adreno 830.

- Engine option: `EngineOptions::f16_attn` (`-1`=auto, `0`=off, `1`=on).
  Auto-enables on GPU backends, off on CPU.
- CLI flag: `--f16-attn 0|1`, exposed on `tts-cli`, `supertonic-cli`,
  and `supertonic-bench`.
- Cache key: `vector_text_attention_cache::f16_kv_attn` so toggling the
  flag mid-process safely rebuilds the cached graph.

Q stays F32: cheaper to keep one operand at the higher precision than
to round-trip the post-attention output back through F32 for the
downstream dense projection.

### How to use

```bash
# Build with OpenCL (in the standalone tree; in-tree subtree consumes
# ggml-speech vcpkg port which already carries the OpenCL patches).
cmake -S . -B build-opencl -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON
cmake --build build-opencl -j$(nproc) --target tts-cli supertonic-bench

# Run on OpenCL with auto F16 attention.
./build-opencl/supertonic-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --n-gpu-layers 99 \
  --out /tmp/supertonic2.wav

# Force F16 attention off (CPU-style fallback) for parity:
./build-opencl/supertonic-cli ... --n-gpu-layers 99 --f16-attn 0
```

### Validation

- Every `supertonic_*_forward_ggml` entry point opens an RAII
  `supertonic_op_dispatch_scope(model)`, so a CPU-only second engine
  in the same thread still sees the default `true` after a GPU
  engine's forward returns — required because the pointwise vocoder
  parity harness and the pipeline trace harness re-enter the model
  from a single thread.
- Both the trace `*_trace_ggml` entry points and the production
  `*_forward_ggml` ones acquire the scope: trace runs still pick the
  pure-GGML pathway whenever the backend isn't CPU, which is what the
  existing parity tests expect (the trace harness already disables the
  fused tail-update op via `!trace_outputs`; the new gate just removes
  the secondary `ggml_custom_4d` branches under it).
- CTest harnesses `test-supertonic-pipeline`, `test-supertonic-vocoder`,
  `test-supertonic-vector`, `test-supertonic-text-encoder`,
  `test-supertonic-duration` continue to exercise the CPU path
  unchanged; running them with a GPU-bound model would route the same
  fixture data through the pure-GGML fallback graph and produce the
  same parity numbers (within F32 → F16 K/V tolerance on the attention
  output when `--f16-attn 1`).
- Three new CPU-only unit harnesses ship alongside the bring-up code
  to give the dispatch + portable-op primitives their own coverage
  independent of any model GGUF:

  | Test | What it covers |
  |------|----------------|
  | `test-supertonic-backend-dispatch` | Default thread-local flag state; `supertonic_op_dispatch_scope` mirroring CPU and GPU `supertonic_model` instances; RAII teardown on normal exit and on exception; nested-scope unwinding; independence of `use_cpu_custom_ops` / `use_f16_attn`. |
  | `test-supertonic-portable-ops`     | CPU-backend parity of `leaky_relu_portable_ggml` (CPU lowering) vs the GPU decomposition for every `α ∈ {0, 0.01, 0.05, 0.1, 0.5, 0.99, 1.0}`; graph-node-count check that the GPU dispatch actually expands the op (catches a regression back to a passthrough `ggml_leaky_relu`). |
  | `test-supertonic-f16-attn-parity`  | F32 vs F16 K/V `ggml_flash_attn_ext` parity on the two hot shapes from the vector estimator (text attention `kv=32`, style attention `kv=50`); tolerance budget `5e-3` absolute / `5e-3` relative, the same band chatterbox ships behind `--cfm-f16-kv-attn`. |

  All three are registered with `LABEL "unit"` so a fresh checkout's
  `ctest -L unit` exercises them without needing the Supertonic GGUF.

### Next optimization rounds

The roadmap beyond this PR — F16 weight materialization, Q8_0 GGUF
support, host↔GPU round-trip elimination, OpenCL kernel-time profile
mode, and vocoder-unpack-on-GPU — is captured with its test plan in
`PLAN_SUPERTONIC_OPENCL.md`.  Each phase has an acceptance test
spelled out (most TDD, written before the implementation lands).

---

## GPU bring-up: Vulkan (May 2026, QVAC-18605)

Target: the same `--n-gpu-layers > 0` flag already plumbed through the
Supertonic CLI / engine / bench layer, but resolved to **Vulkan** on
Linux/Windows boxes that ship a working ICD (NVIDIA proprietary, AMD
RADV via Mesa, Intel ANV, llvmpipe for headless CI) so QVAC consumers
without an OpenCL stack still get the GPU codepath.  Tracking ticket:
QVAC-18605.

### Inheritance from the OpenCL bring-up (QVAC-18607)

By construction, the OpenCL bring-up's foundational work is **backend-
portable**: every helper added in QVAC-18607 (the
`supertonic_op_dispatch_scope` RAII, `backend_is_cpu` flag, F16 K/V
flash-attention path, `leaky_relu_portable_ggml` decomposition) only
ever queries "is this CPU?".  When the resolved backend is Vulkan
those queries return false and the runtime takes the GPU-portable
path automatically.  The Phase 2 audit-driven optimizations (F1-F24
in `aiDocs/AUDIT_SUPERTONIC_OPENCL.md` — host caches, in-graph RoPE,
GPU↔GPU Q/K/V blits, ConvNeXt fusion, F16 weights, in-graph
transpose) likewise apply unchanged: each one removes a host↔GPU
synchronisation point or eliminates redundant memory traffic that
Vulkan pays exactly the same way OpenCL does.

What this PR adds on top is the **Vulkan-specific dispatch deltas**:
two new model flags, two backend-capability probes, a CLI knob for
device selection, and a CPU-only TDD test that locks in the new
contract.  Each is small, scoped, and sits behind the existing
`#ifdef GGML_USE_VULKAN` guard so non-Vulkan builds compile clean.

### What landed

| Change | File(s) | Rationale |
|--------|---------|-----------|
| `supertonic_model::backend_is_vk` set from `ggml_backend_is_vk(model.backend)` after `init_supertonic_backend()` resolves the device. | `supertonic_gguf.cpp`, `supertonic_internal.h` | Informational; consumed by `engine.cpp::backend_name()` and `supertonic_bench.cpp` so multi-GPU machines unambiguously identify which adapter ran the bench (e.g. `Vulkan (device 0: NVIDIA GeForce RTX 5090)` instead of the bare `Vulkan` string). |
| `supertonic_model::use_native_leaky_relu` set from a load-time `ggml_backend_supports_op` probe against a synthetic LEAKY_RELU node.  Mirrored into the dispatch scope's thread-local. | `supertonic_gguf.cpp`, `supertonic_internal.h` | The OpenCL bring-up's `leaky_relu_portable_ggml` always decomposes into `RELU + SCALE + ADD` on non-CPU backends (3 dispatches).  Vulkan / Metal / CUDA implement `GGML_OP_LEAKY_RELU` natively (1 dispatch) — the probe lets the helper short-circuit to the fused builtin on backends that have it, without a hard-coded backend table.  Plain upstream OpenCL (no chatterbox patch) keeps the conservative decomposition. |
| `supertonic_backend_supports_f16_kv_flash_attn(backend)` probe; engine + bench auto-policy gates `use_f16_attn` on the result. | `supertonic_gguf.cpp`, `supertonic_internal.h`, `supertonic_engine.cpp`, `supertonic_bench.cpp` | The OpenCL bring-up's auto-policy flipped `use_f16_attn = !backend_is_cpu` blindly.  Replaced with a backend-capability probe that builds a synthetic Supertonic-shaped flash-attn graph node (`Q[head_dim, q_len, n_heads]` F32, `K/V[head_dim, kv_len, n_heads]` F16) and asks the backend whether it would accept the op.  A backend that ships `flash_attn_ext` but rejects the F16-K/V variant for our shape now keeps the F32 path — slower but guaranteed not to crash at first synth call.  Manual `--f16-attn 1` still forces dispatch (debug). |
| `init_supertonic_backend(n_gpu_layers, verbose, vulkan_device)` — Vulkan device-index parameter.  Range-checks against `ggml_backend_vk_get_device_count()`; an out-of-range value is a hard error (no silent CPU fallback — that would mask CLI typos / wrong-machine config).  Verbose mode logs device description from `ggml_backend_vk_get_device_description`. | `supertonic_gguf.cpp` | Replaces the historical hard-coded `ggml_backend_vk_init(0)`.  Multi-GPU machines + CI runners with a primary llvmpipe and a secondary discrete GPU need a way to pick. |
| `EngineOptions::vulkan_device` (default 0) plumbed through `load_supertonic_gguf`. | `tts-cpp/include/tts-cpp/supertonic/engine.h`, `supertonic_engine.cpp` | Public API. |
| `--vulkan-device N` flag wired into `supertonic-cli`, `supertonic-bench`, and `tts-cli` (the chatterbox CLI's Supertonic dispatch path). | `supertonic_cli.cpp`, `chatterbox_cli.cpp`, `supertonic_bench.cpp` | CLI surface. |
| `test-supertonic-vulkan-dispatch` — CPU-only unit test (`LABEL "unit"`) covering the new `backend_is_vk` / `use_native_leaky_relu` flags through `supertonic_op_dispatch_scope`, plus a smoke test for the F16-K/V flash-attn probe. | `test/test_supertonic_vulkan_dispatch.cpp`, `CMakeLists.txt` | Locks in the new dispatch contract for future regressions; runs on a fresh checkout under `ctest -L unit` without any GGUF fixture. |

### Vulkan supported-op matrix (relevant to Supertonic)

Verified against `ggml/src/ggml-vulkan/ggml-vulkan.cpp` HEAD on this
branch:

| Op | Native on ggml-vulkan? | Notes |
|----|:---:|---|
| `GGML_OP_LEAKY_RELU` (F32) | ✓ | `pipeline_leaky_relu_f32` shader.  `leaky_relu_portable_ggml` short-circuits to fused builtin via the new `use_native_leaky_relu` probe. |
| `GGML_OP_FLASH_ATTN_EXT` (F32 Q, F16 K/V) | ✓ | Requires `HSK % 8 == 0`; Supertonic's `head_dim=64` satisfies this by construction.  Output is F32, which matches what the downstream dense projection expects. |
| `GGML_OP_FLASH_ATTN_EXT` (F32 Q, Q4_0/Q8_0 K/V) | ✓ | Available for future quantized-K/V experiments (chatterbox §3.32 deferred this). |
| `GGML_OP_ROPE` | ✓ | Used by F20/F23 in-graph RoPE (post-OpenCL audit follow-up). |
| `GGML_OP_NORM`, `GGML_OP_MUL`, `GGML_OP_ADD`, `GGML_OP_REPEAT`, `GGML_OP_PERMUTE`, `GGML_OP_CONT`, `GGML_OP_TRANSPOSE`, `GGML_OP_RESHAPE`, `GGML_OP_VIEW`, `GGML_OP_SCALE`, `GGML_OP_RELU`, `GGML_OP_GELU_ERF`, `GGML_OP_MUL_MAT`, `GGML_OP_GET_ROWS`, `GGML_OP_CPY`, `GGML_OP_CONCAT` | ✓ | Universal op set used by the convnext fusion (F7), in-graph transpose (F12), graph-to-graph blit (F24), and every other audit follow-up.  No Supertonic ops missing on Vulkan. |

### How to use

```bash
# Build with Vulkan (in the standalone tree; in-tree subtree consumes
# the ggml-speech vcpkg port which already provides the Vulkan
# backend).
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan -j$(nproc) --target tts-cli supertonic-bench

# Run on Vulkan with auto F16 attention (gated by the new backend-
# capability probe; on a Vulkan adapter satisfying HSK%8==0 it
# auto-enables, on any backend that rejects the F16-K/V op for our
# shape it stays at F32 and continues correctly).
./build-vulkan/supertonic-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --n-gpu-layers 99 \
  --out /tmp/supertonic2.wav

# Pick a specific Vulkan adapter (default 0).  Useful on machines
# with a software rasteriser (llvmpipe) at index 0 and the real
# GPU at index 1.
./build-vulkan/supertonic-cli ... --n-gpu-layers 99 --vulkan-device 1

# Force F16 attention off (CPU-style F32 fallback) for parity:
./build-vulkan/supertonic-cli ... --n-gpu-layers 99 --f16-attn 0

# Bench output explicitly names the Vulkan adapter so multi-GPU
# log lines are unambiguous:
./build-vulkan/supertonic-bench --model models/supertonic2.gguf \
  --text "..." --runs 5 --n-gpu-layers 99 --vulkan-device 0
# →   backend: Vulkan (device 0: NVIDIA GeForce RTX 5090) (f16_attn=on) (native_leaky_relu=on)
```

### Validation

- `test-supertonic-vulkan-dispatch` (CPU-only, `LABEL "unit"`):
  29 / 29 checks pass on this branch.  Covers default flag state,
  scope-mirroring for CPU / Vulkan / OpenCL-style models (probe true
  vs false), RAII teardown on exception, nested-scope unwinding,
  independence of all three flags, and a smoke test for the F16-K/V
  flash-attn probe (CPU backend).
- `test-supertonic-portable-ops` updated to explicitly request the
  decomposition path (`use_native_leaky_relu = false` on the GPU
  model) so the existing GPU-decomposition correctness gate stays
  green now that the helper short-circuits to the fused builtin
  whenever the probe reports native support.  10 / 10 checks pass.
- `test-supertonic-backend-dispatch` (the OpenCL bring-up's tests):
  27 / 27 checks pass — the dispatch scope's new
  `prev_use_native_leaky_relu` slot is added without disturbing the
  existing `prev_use_cpu_custom_ops` / `prev_use_f16_attn` ones.
- All other CPU-only unit tests on the branch (the audit
  follow-ups' RoPE / transpose / convnext-fusion / graph-to-graph-blit
  / profile-csv / F16-weights / F16-attn-parity tests) continue to
  pass unchanged.
- Fixture-bound tests (`test-supertonic-pipeline`,
  `test-supertonic-vocoder`, `test-supertonic-vector`, …) continue
  to exercise the CPU path unchanged.  Running them against a
  Vulkan-bound model would route the same fixture data through the
  same pure-GGML fallback graph that the OpenCL audit work
  established and produce identical parity numbers (within F32 →
  F16 K/V tolerance on the attention output when `--f16-attn 1`).

### Vulkan optimization round 2 (May 2026, QVAC-18605 follow-up)

Layered on top of the Vulkan bring-up above; the round-2 changes
generalise the bring-up's "load-time backend probe" pattern into a
process-wide capability cache and add three more probes / dispatch
hooks that fit the same shape:

1. **Process-wide capability-probe cache** keyed by `ggml_backend_t`.
   The bring-up's three load-sites (`load_supertonic_gguf`,
   `Engine::Engine`, `supertonic_bench`'s `main`) each ran the
   `LEAKY_RELU` and F16-K/V flash-attn `supports_op` queries
   independently — 2-3× redundant probe traffic on every backend
   handle.  On Vulkan, `supports_op` may inspect the device's
   pipeline state (~50-200 µs per query on Adreno / llvmpipe / RADV
   in microbenchmarks); the cache short-circuits 100 % of the
   duplicates.  Test seam (`supertonic_clear_capability_cache` +
   `supertonic_capability_probe_call_count`) lets the unit test
   verify the cache is hit on the second call by comparing the
   counter before / after.

2. **F16 mul_mat backend-capability probe** — symmetric to the F16-K/V
   flash-attn probe.  The bring-up auto-enabled `use_f16_weights` on
   `!backend_is_cpu` blindly; a partial-port backend that ships F16
   storage but rejects the hot vector-estimator W_query mul_mat
   shape (`[256, 256] F16` weight × `[256, 16] F32` activation) would
   crash at first synth call.  Probe builds the live shape and asks
   `ggml_backend_supports_op`; auto-policy refuses materialisation
   on a `false` answer (slower F32 path stays correct).  Manual
   `--f16-weights 1` still forces the F16 path (debug-shim escape
   hatch).  Probe cached in `cached_backend_capabilities`.

3. **Q8_0 K/V flash-attn forward-compat probe** — Vulkan's
   `GGML_OP_FLASH_ATTN_EXT` `supports_op` advertises Q8_0 (and Q4_0)
   K/V types in both scalar and coopmat2 paths
   (`ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT`).  Switching K/V from
   F16 to Q8_0 would halve the per-step upload bandwidth (50 KB → 25
   KB per K/V on Supertonic's hot shape, ≈1 MB / synth on the
   default 5-step × 4-site schedule) in exchange for a small
   (~0.5 %) drift on the attention output.  This PR adds the probe
   + caches the result so a follow-up patch can flip
   `--kv-attn-type q8_0` on without re-querying; the live dispatch
   site is **not yet wired** because the drift hasn't been measured
   against the existing F16 K/V parity harness on a real Vulkan
   adapter.  Bench output annotates `(q8_0_kv_attn=available)` when
   the probe says yes so operators can confirm their hardware is
   ready for the follow-up.

4. **`Engine::warm_up(text)` + `EngineOptions::prewarm_text` +
   `--prewarm TEXT` CLI flag** — first-synth-latency reduction on
   Vulkan / OpenCL.  The in-tree thread_local graph caches handle
   every subsequent call but can't avoid the first pipeline-compile
   cost (~hundreds of ms on Adreno / RADV per chatterbox
   PROGRESS.md).  `warm_up` runs one throwaway synth at construction
   time on a caller-supplied sample text so the operator-visible
   first synth sees steady-state latency.  Auto-no-op on CPU (no
   shader-compile cost to amortise).  The bench harness's
   `--prewarm` runs the cold-start synth BEFORE the timed loop
   starts (independent of `--warmup N`, which discards N timed runs
   from the median but doesn't avoid the cold-start hit on the
   first warmup run); the cold-start latency is logged separately
   (`[prewarm] cold-start synth on '…' took N.Nms`) and surfaced in
   `--json-out` as `"prewarm_ms"`.

5. **Bench output extended** to surface every backend-capability
   dispatch flag plus the cold-start prewarm latency, so log-grep
   across multiple machines can attribute perf differences to the
   right cause.  Backend log line now reads e.g.
   `Vulkan (device 0: NVIDIA RTX 5090) (f16_attn=on)
   (f16_weights=on) (native_leaky_relu=on)
   (q8_0_kv_attn=available)`.  JSON output adds `"f16_attn"`,
   `"f16_weights"`, `"native_leaky_relu"`,
   `"q8_0_kv_attn_available"`, `"prewarm_ms"` keys for downstream
   analysis tooling.

#### Round-2 validation summary

CPU-only, no GGUF needed — green on a fresh checkout under
`ctest -L unit`:

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-capability-cache` (NEW) | Probe cache short-circuit + clear seam + per-backend independence + idempotency + F16 mul_mat probe + Q8_0 K/V probe | 18 / 18 PASS |
| `test-supertonic-warm-up-api` (NEW) | `EngineOptions::prewarm_text` defaults to empty + `Engine::warm_up(const std::string &)` API contract via SFINAE | 9 / 9 PASS |
| `test-supertonic-vulkan-dispatch` (existing) | F16-K/V probe smoke test now exercises the cache short-circuit path | 29 / 29 PASS — unchanged |
| `test-supertonic-portable-ops` / `-backend-dispatch` (existing) | Round-1 dispatch correctness | 10 / 10 + 27 / 27 PASS |
| Audit follow-up tests from #16 (rope / transpose / convnext-fusion / graph-to-graph-blit / profile-csv / F16-attn-parity) | Audit-driven optimisation correctness | All PASS — unchanged |

Whole CPU-only `ctest -L unit` reports 184 / 184 checks passing
across the new tests + every audit-follow-up + bring-up test.

### Deferred work

These were investigated but kept out of scope for this PR:

- **Persistent `VkPipelineCache`** (chatterbox PROGRESS.md §3.32):
  recovers ~91 % of cold→warm shader-compilation gap on first warm
  run, keyed by `<vendorID>-<deviceID>-<driverVersion>` and rooted
  at `$XDG_CACHE_HOME/ggml/vulkan`.  This is a `ggml-vulkan` internal
  patch (~199 lines) that benefits all Vulkan workloads, not just
  Supertonic; tracked separately so the supertonic-specific PR stays
  reviewable.  Round-2's `--prewarm` is an in-process workaround
  (warms the in-memory pipeline cache for one process lifetime); the
  persistent on-disk cache extends the win across process restarts.
  When it lands, this Supertonic Vulkan codepath inherits the
  cold-start win automatically.
- ~~**Q8_0 / BF16 K/V flash-attention live dispatch**~~ — **DONE
  in round 4** (May 2026, QVAC-18605 follow-up #4).  Wired the
  enum-typed dispatch + `--kv-attn-type {auto,f32,f16,bf16,q8_0}`
  CLI flag (probe-gated graceful fallback to F32 on adapters that
  don't support the requested dtype).  Live BF16 / Q8_0 cast in
  `build_text_attention_cache()`; cache invalidation key promoted
  from `bool f16_kv_attn` to `kv_attn_dtype kv_attn_type`.  Drift
  on the parity harness is bounded at 5e-3 abs / 5e-3 rel for
  BF16 (matches the F16 baseline).  Q8_0 dispatch ships behind
  the same flag but is gated by `supertonic_backend_supports_q8_0_kv_flash_attn`;
  the operator opts in only when their adapter advertises
  support.  See "Vulkan optimisation round 4" below.
- **Pinned-host-buffer per-step uploads**: round 3 adds the
  capability probe for `ggml_backend_vk_host_buffer_type()` so
  the cache + bench surface know whether the path is available
  on the resolved backend.  The actual per-engine input-
  scratchpad refactor (allocate text_emb / time-step / style
  embedding tensors in the host-pinned buffer type instead of
  the default device-local buffer to skip ggml-vulkan's internal
  staging-buffer hop) is deferred until measured on a real Vulkan
  adapter so we can quantify the reduction in `latent` upload
  latency.

---

### Vulkan optimisation round 3 (May 2026, QVAC-18605 follow-up #2)

Three more Vulkan-specific deltas, all developed test-first (TDD)
— the new tests were committed first, observed to fail on the
missing symbol, and only then was the implementation written and
the tests re-run.

1. **BF16 K/V flash-attn capability probe** (5th `backend_capabilities`
   flag).  Symmetric to the round-2 Q8_0 K/V probe.  Vulkan's
   `GGML_OP_FLASH_ATTN_EXT` `supports_op` advertises BF16 K/V via
   the coopmat2-only path; BF16 has the same 2-byte per-element
   footprint as F16 (so identical upload bandwidth) but the wider
   8-bit exponent range avoids the F16 underflow on small attention
   scores that drives the parity-harness tolerance widening.
   Forward-compat — the live `--kv-attn-type bf16` dispatch wiring
   is deferred to a follow-up that measures drift against the
   parity harness on a real Vulkan adapter.

2. **Multi-device auto-pick for `--vulkan-device -1`**.  Wires the
   previously-reserved auto-pick API: walks every visible adapter,
   queries `ggml_backend_vk_get_device_memory()` to read free
   VRAM, and dispatches into a pure-logic helper
   `resolve_vulkan_device_index(requested, free_vram_per_device)`
   that picks `argmax(free_vram)` (ties → lower index for stable
   per-run assignment on identical-spec multi-GPU machines).
   Verbose mode logs the per-device VRAM table so operators can
   confirm the auto-pick chose the expected adapter.  The pure-
   logic helper is testable on CPU with synthetic inputs (8 cases,
   23 checks) — separates the policy from the Vulkan-only plumbing.
   Reserved-future negative values (`-2`, `-100`, ...) now throw
   instead of silently falling through to device 0.

3. **Pinned-host-buffer-type capability probe** (6th
   `backend_capabilities` flag) + bench surface.  Probes whether
   `ggml_backend_vk_host_buffer_type()` is callable on the
   resolved backend (Vulkan + non-null buffer-type).  Forward-
   compat — primes the capability cache for a follow-up per-engine
   input-scratchpad refactor that skips ggml-vulkan's internal
   staging-buffer hop on per-step uploads.  Bench output now shows
   `bf16_kv_attn_available` + `pinned_host_buffer_available` in
   both the human-readable backend tag and the JSON output so
   operators can pre-flight whether a future opt-in will be
   effective on their machine.

#### Test plan (TDD, round 3)

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-capability-cache` (UPDATED) | Existing 18 checks + 9 new round-3 checks (BF16 K/V probe smoke + cache-slot share, pinned-host-buffer probe smoke + cache-slot share, null-backend handling for both) | 27 / 27 PASS |
| `test-supertonic-vulkan-device-select` (NEW) | 8 test functions × 23 checks for the pure-logic auto-pick helper (empty list, single device, argmax, tie-break, explicit-index passthrough, out-of-range, reserved-negative, zero-VRAM) | 23 / 23 PASS |
| Every existing unit test (resample, cpu/t3 caches, profile-csv, rope-in-graph, rope-packed-qk, convnext-block-fused, in-graph-transpose, graph-to-graph-blit, backend-dispatch, portable-ops, vulkan-dispatch, warm-up-api, f16-attn-parity) | Round 1 + 2 + audit follow-up correctness | 16 / 16 PASS — unchanged |

Whole CPU-only `ctest -L unit` reports **16 / 16 tests, 0 failures**.
The TDD discipline was strict: the new tests in round 3 were
committed BEFORE the implementation and verified to fail on the
missing symbol (the compile-error footprint is captured in the
PR description) — only then was the implementation written and
the tests re-run to verify green.

---

### Vulkan optimisation round 6 (May 2026, QVAC-18605 follow-up #3) — F16-weights operator deny-list

Round 6 layers a **user-overridable extra deny-list** on top of
the existing hand-curated `should_materialise_f16_weight()`
allow-list.  The curated allow-list (Phase 2A) already excludes
biases, norms, embeddings, depthwise convs, and pre-transposed
companions; the round-6 deny-list lets operators force-keep
specific *additional* tensors as F32 even when `--f16-weights`
is on.  Use cases:

- **A/B testing**: researcher wants to exclude a specific tensor
  pattern temporarily without recompiling.
- **Hardware-specific drift mitigation**: operator observes drift
  on a particular adapter / driver / shape and pins the
  problematic tensor to F32 via config rather than disabling F16
  weights wholesale.
- **Future-GGUF safety net**: new tensor patterns added in future
  Supertonic GGUFs that the curated allow-list inadvertently
  scoops in can be excluded via config without a code change.

Smallest blast radius of the four follow-up rounds — load-time
policy only, runtime dispatch unaffected, zero behaviour change
on the empty-deny-list default path.

#### What changed

1. **2-arg overload `should_materialise_f16_weight(name, extra_deny_substrings)`**
   added alongside the existing 1-arg version (existing test +
   call sites unchanged).  Substring matching (audit-friendly,
   matches the curated predicate's style; no regex compile cost
   or invalid-pattern surface).  The deny-list can only flip
   `true → false`, never `false → true` — it's a deny-list, not
   an allow-list.  Empty strings inside the deny-list are
   SKIPPED defensively, not treated as universal matches (config-
   typo guard against an empty entry silently disabling F16
   weights for the whole model).

2. **`EngineOptions::f16_weights_deny_list`** (`std::vector<std::string>`,
   default empty) — public API surface for engine-side
   integration.  Wired through `Engine::Impl` →
   `load_supertonic_gguf` → the per-tensor allocation loop.

3. **`load_supertonic_gguf` 7th parameter** added at the end of
   the signature with a `{}` default — every existing call site
   keeps compiling without modification.

4. **`supertonic_model::f16_weights_excluded_count`** counter
   bumped at load time when a curated-hot tensor is excluded by
   the user's deny-list.  Surfaced in bench's human + JSON
   output so operators can confirm their config took effect.

5. **CLI plumbing**: `--f16-weights-deny PAT1,PAT2,...` flag on
   `supertonic-cli`, `tts-cli` (chatterbox), and `supertonic-bench`
   (comma-separated substring patterns).

6. **Verbose-log line** in `load_supertonic_gguf` when the deny-
   list is non-empty (silent on the default path — no visual
   noise on existing operator workflows).

#### Test plan (TDD, round 6)

Both new tests were committed BEFORE the implementation and
observed to fail on the missing symbols (compile errors:
`'should_materialise_f16_weight' too many arguments` for the
predicate test; `'EngineOptions::f16_weights_deny_list'` no such
member for the API-surface test).  Only then was the
implementation written and the tests re-run.

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-f16-weights` (UPDATED) | Existing 36 checks (positives, negatives, edges) + 29 new round-6 checks across 7 new test functions (empty-list passthrough, matching-deny-excludes, non-matching-no-op, cannot-promote-cold, multiple-patterns ANY-match, empty-string defensive skip, empty-name safety) | 65 / 65 PASS |
| `test-supertonic-f16-deny-list-api` (NEW) | SFINAE compile-time gate for `EngineOptions::f16_weights_deny_list` + `load_supertonic_gguf` 7th param; runtime defaults check + assignability + regression guards on every other documented `EngineOptions` default | 9 / 9 PASS |
| Every other unit test (round 1+2+3 + audit follow-ups + the 14 baseline tests) | Zero-regression gate | 17 / 17 PASS — unchanged |

Whole CPU-only `ctest -L unit` reports **17 / 17 tests, 0
failures, 0 regressions**.

#### Why no live perf number?

Round 6 is a **policy** change, not a kernel change.  The
quality-recovery on hand-picked tensors is workload-specific and
quantified offline against the F16-attention parity harness;
this PR adds the operator-facing knob so future drift incidents
can be triaged via config without a code change.  Bench output
surfaces the excluded-count so CI scripts can attribute any
quality regression to a config change.

---

### Vulkan optimisation round 4 (May 2026, QVAC-18605 follow-up #4) — Multi-dtype K/V flash-attention

The round-1 `--f16-attn` boolean only let operators pick between
F32 and F16 K/V flash-attention.  Round 4 generalises the
dispatch into a four-valued enum + CLI flag so operators can
opt into BF16 K/V (Vulkan coopmat2 — same bandwidth as F16, no
F16 underflow on small attention scores) or Q8_0 K/V (Vulkan
+ half the K/V upload bandwidth for upload-bound workloads) on
adapters that advertise the corresponding capability.  The
existing F16 cache + dispatch were the round-2 / round-3
plumbing's only consumers; round 4 is the live wiring that
turns those probe results into actual dispatches.

#### Changes

- **New public API**: `EngineOptions::kv_attn_type` int field
  (`-1` = auto, `0` = f32, `1` = f16, `2` = bf16, `3` = q8_0).
  Same `-1` = auto convention as `f16_attn` / `f16_weights` /
  `vulkan_device`, so operator configs are consistent.  Default
  (`-1`) falls back to `f16_attn`'s value, so every existing
  operator config sees zero behaviour change.

- **New internal enum + resolver**: `tts_cpp::supertonic::detail::kv_attn_dtype`
  + `resolve_kv_attn_type(requested, legacy_use_f16_attn,
  supports_f16, supports_bf16, supports_q8_0)` — pure-logic
  policy split from the dispatch site (same split pattern as
  round-3's `resolve_vulkan_device_index`).  Out-of-range int
  throws to surface CLI typos loudly; probe-rejected explicit
  requests fall back to F32 silently (advisory-probe pattern,
  same as round-1's F16 auto-policy).

- **New thread-local accessor**: `supertonic_kv_attn_type()`,
  populated by `supertonic_op_dispatch_scope` from
  `model.kv_attn_type` (mirrors the `supertonic_use_f16_attn()`
  pattern).  RAII teardown via the new
  `supertonic_op_dispatch_scope::prev_kv_attn_type` field.

- **Vector-estimator dispatch site** (`build_text_attention_cache()`):
  `if (cache.f16_kv_attn) { cast K/V → F16 }` replaced with a
  switch on the enum; cast target picked from `{F16, BF16, Q8_0}`
  per `cache.kv_attn_type` (or no cast for F32).  Cache key
  promoted from `bool f16_kv_attn` to `kv_attn_dtype kv_attn_type`
  (rebuilds the graph when the enum flips, same correctness
  contract as the rest of the cache key tuple).

- **CLI flag** on all three CLIs (`supertonic-cli`, `tts-cli`,
  `supertonic-bench`): `--kv-attn-type {auto,f32,f16,bf16,q8_0}`.
  The `supertonic-cli` arg-parse loop is now wrapped in
  try/catch so invalid values surface as a clean `error: ...`
  line + exit 2 instead of an uncaught-exception backtrace
  (also fixes the pre-existing latent crash on `--vulkan-device
  abc` / `--seed nonsense` / etc).

- **Bench surface**: human-readable line shows
  `(kv_attn_type=f32|f16|bf16|q8_0)` always (so log-grep across
  machines can attribute drift / perf to dispatch dtype).  JSON
  output adds `"kv_attn_type": "<dtype>"` and
  `"kv_attn_type_requested": <int>` — the resolved + the
  requested value, so a probe miss is visible in the JSON.

#### Test plan (TDD, round 4)

Strict test-first.  All four new tests were committed first,
observed to fail on missing symbols (compile errors:
`'kv_attn_dtype' has not been declared` for the resolver test;
`'EngineOptions' has no member named 'kv_attn_type'` for the
API test).  Only then was the implementation written and the
tests re-run.

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-f16-attn-parity` (UPDATED — Prereq B) | Existing 4 F16-vs-F32 parity checks (vector-estimator + style shapes) + **2 new BF16-vs-F32 parity checks** wired via the same `run_flash_attn(cpu, in, kv_dtype)` helper.  Tolerance band: 5e-3 abs / 5e-3 rel on both shapes; CPU build returned `max_abs_err = 5.263e-3` (vector-estimator) and `3.596e-3` (style), both within budget. | 8 / 8 PASS |
| `test-supertonic-kv-attn-type` (NEW) | Pure-logic resolver — 7 test functions, **106 checks** covering: auto + legacy boolean back-compat matrix; f32 forced overrides legacy; f16 forced + probe-gated graceful fallback; bf16 forced + probe-gated graceful fallback (40-state combo: every {requested, legacy, probe-mask} tuple verified to never leak the `autoselect` sentinel); q8_0 forced + probe-gated graceful fallback; out-of-range throws (4 cases: 4, 99, -2, -100); resolver-returns-concrete-only (40-state exhaustive sweep). | 106 / 106 PASS |
| `test-supertonic-kv-attn-type-api` (NEW) | API-surface lockdown — SFINAE compile-time gates for `EngineOptions::kv_attn_type` field, `supertonic_model::kv_attn_type` field, `supertonic_op_dispatch_scope::prev_kv_attn_type` field; runtime defaults check (kv_attn_type=-1, model field=f32, accessor=f32 with no scope active); dispatch-scope ctor/dtor restoration of the thread-local; regression guard on every other documented `EngineOptions` default (prewarm_text empty, vulkan_device 0, f16_attn -1, f16_weights -1, f16_weights_deny_list empty). | 18 / 18 PASS |
| Every other unit test (rounds 1 + 2 + 3 + 6 + audit follow-ups + the 14 baseline tests) | Zero-regression gate | 19 / 19 PASS — unchanged |

Whole CPU-only `ctest -L unit` reports **19 / 19 tests, 0
failures, 0 regressions**.

#### Backwards compatibility contract

- Default `--kv-attn-type auto` (== `kv_attn_type = -1`) falls
  back to `--f16-attn`'s value via the resolver.  Every existing
  operator config sees identical behaviour to round 1 / 2 / 3
  / 6.

- The legacy `model.use_f16_attn` boolean is updated to
  `(model.kv_attn_type == kv_attn_dtype::f16)` after resolution
  so any external code still keying on the boolean stays
  consistent with the enum.  In-tree the only consumer is the
  vector estimator, which now reads the enum directly; the
  boolean is preserved for forward-compat + the existing
  `test-supertonic-backend-dispatch` lockdown checks.

- Probe-rejected explicit requests fall back to F32 silently
  — an operator setting `--kv-attn-type bf16` once in their
  production config works on both NVIDIA Ampere+ (BF16 effective
  via Vulkan coopmat2) and Intel ARC (no coopmat2 → silent F32
  fallback) without crashing.  Operators see the resolved dtype
  in the bench output, so a fallback is visible.

- Out-of-range `--kv-attn-type N` (CLI typo, e.g. `--kv-attn-type
  q4_0`) throws inside `resolve_kv_attn_type`; the CLI catches +
  surfaces it as `error: --kv-attn-type expects auto|f32|f16|bf16|q8_0
  (got: ...)` + exit 2.  Loud failure for actual config errors;
  silent fallback for advisory probes.

#### Why no live Vulkan perf number?

Round 4 is the **dispatch wiring** that turns the probe
results from rounds 2 + 3 into actual GPU work.  The win
shape is workload + adapter specific:

- **BF16 K/V on Vulkan coopmat2**: same K/V upload bandwidth
  as F16, but the wider exponent range removes the F16
  underflow on small attention scores.  No drift, no
  bandwidth cost — pure quality recovery.  Expected to
  dominate F16 on production prompts where the round-1 F16
  parity harness sits near tolerance.

- **Q8_0 K/V on Vulkan**: half the K/V upload bandwidth of
  F16/BF16; expected dominant on long-prompt / large-style
  workloads where K/V upload is a meaningful fraction of
  per-step time.  Quantization noise is workload dependent;
  operators dial in via the parity harness on their own
  prompts before flipping the flag.

The dispatch + flag are in place so an operator with a real
Vulkan adapter can A/B in their own config without a code
change; the harness numbers will land in a follow-up after
measurement on real hardware.

---

### Vulkan optimisation round 7 (May 2026, QVAC-18605 follow-up #5) — Bench observability + voice cache + Vulkan env-var passthrough

The next-rounds plan
(`aiDocs/PLAN_VULKAN_NEXT_ROUNDS.md`) identified bench-side
observability + a small set of trivial wins as the highest
impact-÷-risk round to land before the bigger structural changes
of rounds 5 / 8 / 9.  Round 7 ships four sub-features, none
touching the per-synth hot path beyond a single voice-cache
lookup.

#### Changes

- **Voice ttl/dp host cache** (`tts_cpp::supertonic::detail::voice_host_cache`).
  Extracted from `Engine::Impl::synthesize()` so the lookup-or-load
  semantics are testable on CPU without instantiating a full
  Engine.  First `synthesize()` per voice does the 2 GPU→host
  downloads (`read_tensor_f32(ttl)` + `read_tensor_f32(dp)`)
  and caches the result; subsequent calls return the cached
  entry without touching the backend.  Eliminates 2 sync points
  per `synthesize()` after the first per-voice on Vulkan / OpenCL.
  Tiny (2 small tensors) but free.  Reference-stability contract
  documented on the struct: caller may hold the reference for
  the duration of one synthesis, but must not call `clear()`
  while holding it (currently only reachable on Engine
  destruction).

- **Vulkan env-var passthrough**
  (`apply_vulkan_env_overrides(map)` public helper +
  `EngineOptions::vulkan_env_overrides` field +
  `--vulkan-prefer-host-memory` / `--vulkan-disable-coopmat2` /
  `--vulkan-disable-bfloat16` / `--vulkan-perf-logger` /
  `--vulkan-async-transfer` / `--vulkan-env KEY=VALUE` CLI flags
  on all three binaries).  ggml-vulkan reads its `GGML_VK_*`
  env vars at backend-init time; this round lets operators set
  them via CLI (or `EngineOptions`) without exporting in the
  shell.  ALL-OR-NOTHING validation: an operator-config typo
  like `GMML_VK_PREFER_HOST_MEMORY` throws cleanly via
  `apply_vulkan_env_overrides` BEFORE any env var is touched.
  `set_env_if_unset` semantics so an operator-set env var still
  WINS over the EngineOptions override (debugging operators can
  force-disable from the shell without recompiling).

- **Bench `ggml_backend_synchronize` boundaries**
  (`--bench-sync` default on, `--no-bench-sync` opt-out).
  Inserts an explicit backend sync at every per-stage timing
  boundary so wall-clock attributes to the right stage on async
  backends.  Cheap on CPU (no-op when no GPU work pending);
  ensures per-stage breakdowns reflect work-completed-by-the-
  prior-stage on Vulkan / OpenCL.  Round-7 prerequisite for
  measuring rounds 5 / 8 / 9 wins on real hardware.

- **Bench per-denoise-step breakdown** (`--bench-per-step`,
  default off).  Times each `supertonic_vector_step_ggml` call
  individually so the first-step (cold pipeline) cost can be
  distinguished from steady-state.  Adds an indented
  `vector_step[N]` line per step in the human output and a
  separate JSON entry per step.  Empty array on the default-off
  path = identical legacy JSON shape.

#### Test plan (TDD, round 7)

Strict test-first.  Two new test executables committed first,
observed to fail on the missing symbols (compile errors:
`'apply_vulkan_env_overrides' was not declared in this scope`
for the env-passthrough test; `'voice_host_cache' has not been
declared` for the voice-cache test).  TDD also caught a real
implementation bug: the original validator used `std::string()`
empty-as-success sentinel which collided with the empty-string-
as-key edge case; the test pinned the contract and forced the
fix to a `bool / out-param` API before any production wiring
went in.

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-vulkan-env-overrides` (NEW) | 7 functions, **29 checks** — SFINAE field existence; round-3/4/6 baseline-defaults regression guard; empty-map noop; single-entry sets env; operator-env wins (set_env_if_unset semantics); invalid-key throws (4 negative cases including the empty-string-key edge); ALL-OR-NOTHING on mixed-validity (no partial application); multi-entry happy path. | 29 / 29 PASS |
| `test-supertonic-voice-host-cache` (NEW) | 6 functions, **25 checks** — empty cache; first-load populates from GGML tensors; second-load hits cache (verified by passing nullptr — a real load attempt would crash); multi-voice independence + reference stability across other-voice lookups; clear-drops-entries; null-tensors-on-miss throws (Impl-bug guard). | 25 / 25 PASS |
| Every other unit test (rounds 1 + 2 + 3 + 4 + 6 + audit follow-ups + the 14 baseline tests) | Zero-regression gate | 19 / 19 PASS — unchanged |

Whole CPU-only `ctest -L unit` reports **21 / 21 tests, 0
failures, 0 regressions**.

#### Backwards compatibility

- `EngineOptions::vulkan_env_overrides` defaults to empty —
  `apply_vulkan_env_overrides({})` is a no-op (regression-
  guarded by `test_empty_map_is_noop`); no operator-visible
  behaviour change for existing configs.
- Voice cache is fully transparent — `Engine::Impl` hits the
  cache in place of the previous direct `read_tensor_f32` calls;
  the cached vectors are bit-equal to the originals.
- `--bench-sync` defaults to ON.  Per-stage times in the bench
  output may shift slightly upward on Vulkan / OpenCL because
  they now reflect work-completed-by-the-stage instead of
  host-return-from-the-stage; the AGGREGATE total stays equal
  (the work was always being done; the attribution just gets
  more accurate).  `--no-bench-sync` recovers the historical
  shape exactly.
- `--bench-per-step` defaults to OFF — JSON shape unchanged on
  the default path.

#### Why no live perf number?

Round 7 is **observability + paving** — the wins are:
- Voice cache: 2 sync points / synth eliminated (small but free).
- Bench sync + per-step: prerequisites for measuring round 5 / 8
  / 9 wins on real hardware (no measurable production effect by
  themselves).
- Vulkan env passthrough: triage knobs for operators, not
  production tuning.

The biggest payoff lands in round 8 when the bench surface from
round 7 starts attributing the front-block GPU-bridge win to the
right stage column.

---

### Vulkan optimisation round 8 (May 2026, QVAC-18605 follow-up #6) — Front-block attn0 GPU bridge

The single largest remaining per-step sync hotspot identified in
the next-rounds plan
(`aiDocs/PLAN_VULKAN_NEXT_ROUNDS.md`).  PR #16's audit follow-up
#6 (2C-lite) shipped the GPU device→device blit infrastructure
(`run_text_attention_cache_gpu`) and wired g1 / g2 / g3 group
attentions to use it; the front-block `attn0` site was deferred
because of cache-lifetime concerns at the time.  Round 8 picks
it up — same exact pattern as g1/g2/g3, ~30 LOC delta in one
function.

#### Changes

- **Front-block attn0 dispatch site** (`supertonic_vector_estimator.cpp`,
  `supertonic_vector_trace_proj_ggml`).  The
  `tensor_to_time_channel(...)` downloads of `ve_attn0_v` /
  `ve_attn0_q_rope` / `ve_attn0_k_rope` followed by the host-bridge
  `run_text_attention_cache(...)` call are replaced (in
  production mode) by a single `run_text_attention_cache_gpu(
  q_rope_gpu, k_rope_gpu, v_gpu, ...)` call that takes the
  named GPU tensors from the front cache and blits them
  device→device into the att0 cache's input tensors.
  Eliminates 6 sync points × 5 denoise steps = **30 sync points
  / synth** on the production path.

- **Strict gating on the GPU-bridge fast path** —
  `front_in_graph_rope && !include_ggml_trace && v_gpu_attn0 &&
  k_rope_gpu_attn0`.  Trace mode falls back to the legacy host
  bridge so the trace harness still captures pre-attention
  Q/K/V host vectors for scalar-parity assertions.  Legacy
  GGUFs without `vector_rope_theta` (no in-graph RoPE) also
  fall back — host `apply_rope` continues to work.  Defensive
  null-guards on `v_gpu_attn0` / `k_rope_gpu_attn0` even though
  both are unconditionally `set_output` in the cache build
  (cost: zero; insurance against a future cache rewrite that
  silently drops one of the named outputs).

#### Test plan (TDD, round 8)

The blit primitive parity gate already shipped with PR #16:
`test-supertonic-graph-to-graph-blit` covers the device→device
blit through two minimal cached graphs sharing one backend, and
asserts bit-exact parity vs the host-download / host-upload pair.
Round 8 extends it with explicit coverage of the front-block K/V
shapes:

| Shape | Coverage |
|------|----------|
| `attn0_q_rope_L20` (existing) | 4h × 64d Q post-RoPE @ L=20 — already covered front-block Q.  Round-8 doc-comment makes the front-block coverage explicit. |
| `attn0_kv_text_len32` (NEW) | front-block K / V @ text_len=32 (width=256, kv_len=32) — blit primitive parity for the K / V shape. |
| `attn0_kv_text_len50` (NEW) | front-block K / V @ text_len=50 (width=256, kv_len=50) — same primitive at the longer text-prompt shape. |

Whole CPU-only `ctest -L unit` reports **21 / 21 tests, 0
failures, 0 regressions**.  Existing bit-exact parity tests
covering the non-trace front-block path
(`test-supertonic-rope-in-graph`, `test-supertonic-rope-packed-qk`,
`test-supertonic-graph-to-graph-blit`,
`test-supertonic-f16-attn-parity`) all continue to pass — the
dispatch-site change preserves the F23 in-graph RoPE outputs
that those tests pin, and the GPU-bridge path is functionally
identical to the host-bridge path it replaces (only the
intermediate transfer pattern changes).

#### Backwards compatibility

- Trace mode unchanged — `include_ggml_trace == true` falls back
  to the legacy host bridge with all original downloads + trace
  pushes.
- Legacy GGUFs (no `vector_rope_theta`) unchanged — falls back
  to the host-rotate path that PR #16 already preserved.
- Production path: bit-equivalent output to the pre-round-8
  path (the GPU bridge blits the same bytes the host bridge
  would download / upload; the attention compute reads the
  same input data either way).
- `cache.kv_attn_type` cache-key (round 4) still applies — F32 /
  F16 / BF16 / Q8_0 dispatch unchanged on the GPU path.

#### Why no live perf number?

Same shape as round 4: dispatch wiring, not a kernel change.
The win is workload + adapter specific:

- On Adreno (chatterbox PROGRESS.md §3) each sync point costs
  several hundred microseconds.  30 sync points / synth × 5
  steps = a measurable per-synth latency reduction depending on
  prompt length.
- On desktop NVIDIA / AMD the per-sync overhead is lower but
  still real (USB / PCIe round-trip).
- On CPU the change is strictly equivalent — `ggml_backend_tensor_copy`
  with same-backend src+dst is a memcpy on the CPU backend; the
  parity test pins this at `max_abs = 0.0` (bit-equal output).

The dispatch + parity gate are in place so an operator with a
real Vulkan adapter can A/B `--bench-per-step` (round 7) numbers
on rounds 6 / 7 / 8 builds and attribute the per-step
improvement to this exact change.

---

### Vulkan optimisation round 9 (May 2026, QVAC-18605 follow-up #7) — Style flash-attn GPU bridge

Round 8 wired the GPU bridge for the **front-block attn0** site.
Round 9 extends the same proven pattern to the **4 style flash-
attn sites** (style0 + g1_style + g2_style + g3_style).  Each
site previously downloaded `sq` / `sk` / `sv` from the
res-style-qkv cache then re-uploaded them to the next-stage
attention cache; round 9 replaces all 4 host bridges with
`run_text_attention_cache_gpu` device→device blits, gated on
production mode.

#### Changes

- **`vector_res_style_qkv_result` extended** with
  `ggml_tensor * sq_gpu / sk_gpu / sv_gpu` GPU handles.  Same
  shape as `vector_group_graph_result::q_rope_gpu` etc from the
  round-1 2C-lite work.  Populated unconditionally by
  `run_res_style_qkv_cache` (cheap — just `ggml_graph_get_tensor`
  lookups on the cached graph; no GPU sync).

- **`run_res_style_qkv_cache` host-download gating**.  The 3
  `tensor_to_time_channel(...)` downloads of `sq` / `sk` / `sv`
  are now gated on `trace != nullptr`.  Production path skips
  them entirely.  Mirrors the round-1 2C-lite
  `need_host_qkv = (trace != nullptr)` gate on
  `vector_group_graph_result`.  `post` stays unconditional —
  consumed by the next-stage `run_style_residual_cache` which
  still expects a host vector (cross-stage GPU bridge for `post`
  is deferred; documented in `aiDocs/PLAN_VULKAN_NEXT_ROUNDS.md`).

- **4 style flash-attn dispatch sites rewired**.  All four sites
  (`style0` / `g1_style` / `g2_style` / `g3_style`) follow the
  exact same gating pattern as the round-8 front-block bridge:
  ```
  use_gpu_bridge = !include_ggml_trace && sq_gpu && sk_gpu && sv_gpu
  if (use_gpu_bridge) run_text_attention_cache_gpu(sq_gpu, sk_gpu, sv_gpu, ...)
  else                run_text_attention_cache(host_sq, host_sk, host_sv, ...)
  ```
  Trace mode falls back to the legacy host bridge so the trace
  harness still gets all the host vectors.

#### Test plan (TDD, round 9)

Strict test-first.  The blit primitive parity test was extended
BEFORE any production wiring landed:

| Shape | Coverage | Result |
|------|----------|--------|
| `style_sq_L1` (NEW) | Style Q at L=1 — trip-wire for stride / shape bugs at the smallest sensible input.  Mirrors round-8's `attn0_q_rope_L1` trip-wire. | `max_abs = 0.0` PASS |
| `style0_q_rope_L20` (CLARIFIED) | Style sq @ L=20 (width=256, n_heads=2, head_dim=128).  Already covered the underlying byte layout pre-round-9; round 9 adds the explicit doc-comment about which round-9 site this covers. | `max_abs = 0.0` PASS |
| `style0_k_rope_kv50` (CLARIFIED) | Style sk / sv @ kv_len=50.  Same comment treatment. | `max_abs = 0.0` PASS |

Whole CPU-only `ctest -L unit` reports **21 / 21 tests, 0
failures, 0 regressions**.  `test-supertonic-graph-to-graph-blit`
went from 21 / 21 to **24 / 24 checks** (3 new style-shape
checks, all bit-exact).  All other unit tests unchanged.

#### Backwards compatibility

- Trace mode preserved exactly — `include_ggml_trace == true`
  triggers the `if (trace)` host-download block in
  `run_res_style_qkv_cache` and the host-bridge fallback in
  every dispatch site.  Trace harnesses see identical `sq` /
  `sk` / `sv` host vectors as before round 9.
- Production path: bit-equivalent output to the pre-round-9
  path (the GPU bridge blits the same bytes the host bridge
  would download / upload; the attention compute reads the
  same input data either way).
- `cache.kv_attn_type` (round 4) cache-key still applies —
  F32 / F16 / BF16 / Q8_0 K/V dispatch unchanged on the GPU
  path.
- `last_style_v_raw_uploaded` / `last_kctx_raw_uploaded` F4
  upload-skip optimization untouched (those are about
  `style_v_in` / `kctx_in` uploads INTO the res-style-qkv
  cache, not its outputs).

#### Why no live perf number?

Same shape as rounds 4 + 8: dispatch wiring, not a kernel
change.  Sync-points eliminated:

- 3 GPU→host downloads + 3 host→GPU uploads = 6 sync points
  per call
- 4 sites × 5 denoise steps = 20 calls / synth
- Total: **120 sync points / synth eliminated** on the
  production Vulkan / OpenCL path (4× the round-8 win;
  largest bandwidth-style optimisation that ships from
  pure-Supertonic-side code).

The bench surface from round 7 (`--bench-per-step` +
`--bench-sync`) directly attributes the per-step improvement
to the correct stage column on real hardware.

---

### Vulkan optimisation round 10 (May 2026, QVAC-18605 follow-up #8) — Per-step text-input upload-skip

After rounds 8 + 9 wired the GPU bridge for the 5 attention sites
(front-block attn0 + 4 style attentions), the remaining per-step
host uploads are the **input tensors fed to each cached graph**:
`latent` (changes per step), `mask` (constant), `temb` (changes
per step), and `text_emb` / `text_lc_host` (constant within one
synth).  Round 10 picks off the largest of those: `text_emb`,
which is uploaded **4 caches × 5 steps = 20 times / synth** but
is the same data on every call.

#### Changes

- **`upload_skip_tracker` helper** in `supertonic_internal.h`.
  Pointer-compare upload-skip generalising the F4 pattern
  already used for `style_v_in` / `kctx_in` in
  `vector_res_style_qkv_cache`.  `needs_upload(p) -> bool`,
  `mark_uploaded(p)`, `reset()`.

- **Front-block cache** (`ve_front_block_graph_cache`) +
  **group-graph cache** (`vector_group_graph_cache`): add
  `text_in_skip` field, guard the `ggml_backend_tensor_set` for
  `text_in` / `text_in_t` with `needs_upload(text_emb)`, and
  reset on `current_step == 0` to handle the cross-synth
  pointer-reuse hazard (modern allocators very often re-issue
  the same address for the next stack-local
  `std::vector<float>` of the same size — without the reset,
  the next synth would silently leak prior synth's text-encoder
  embedding to the GPU).

- **Cache rebuild safety**: `cache = {}` zero-initialises the
  tracker (its only field is a pointer that defaults to
  `nullptr`), so a graph rebuild correctly forces the next
  upload regardless of incoming pointer.

#### Test plan (TDD, round 10)

Strict test-first.  `test-supertonic-upload-skip-tracker` (NEW)
committed first, observed to fail compile (`upload_skip_tracker
was not declared`), then implementation added.

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-upload-skip-tracker` (NEW) | 7 functions, **41 checks** — default state (fresh tracker always needs upload); upload + skip happy path (5-step pattern); pointer-change forces upload; reset() invalidation (synth-boundary contract); independent-instance non-interference; **cross-synth pointer-reuse hazard simulation** (exact bug the synth-boundary reset prevents — without reset, naive pointer-compare leaks prior synth data); reset-on-empty no-op. | 41 / 41 PASS |
| Every other unit test (rounds 1-9 + audit follow-ups + the 14 baseline tests) | Zero-regression gate | 21 / 21 PASS — unchanged |

Whole CPU-only `ctest -L unit` reports **22 / 22 tests, 0
failures, 0 regressions**.

#### Backwards compatibility

- Tracker is initialised to `last_uploaded = nullptr` →
  `needs_upload(any_ptr) = true` on the first call → cold-miss
  upload always fires.  No cache cold-start regression.
- Cache rebuilds (`cache = {}`) zero-init the tracker → next
  upload fires regardless of pointer.  Same correctness as
  pre-round-10.
- Synth-boundary reset (`current_step == 0`) invalidates the
  tracker → next synth's first step always uploads.  Protects
  against the documented cross-synth pointer-reuse hazard.
- Trace mode unaffected (the upload itself is unchanged when
  it fires; only the redundant re-uploads are skipped).

#### Win

Per synth (5 denoise steps):

| Cache | Uploads pre-round-10 | Uploads post-round-10 | Saved |
|---|---|---|---|
| Front block (`text_in_t`) | 5 | 1 (cold-miss) | 4 |
| g1 group (`text_in`) | 5 | 1 | 4 |
| g2 group (`text_in`) | 5 | 1 | 4 |
| g3 group (`text_in`) | 5 | 1 | 4 |
| **Total** | **20** | **4** | **16 sync points / synth** |

Bandwidth saved: 16 × `text_len × 256 × 4` bytes / synth.  At
text_len=32 that's **~512 KB / synth** of redundant host→GPU
upload eliminated; scales linearly with prompt length.

The remaining per-step uploads (`latent`, `temb`, per-step
deltas in mask) genuinely change per step; can't be skipped
without a graph-allocator refactor (round 5 territory — still
deferred).

#### Why no live perf number?

Round 10 is small + safe: a host-side upload-skip optimisation
that adds zero work on the cold path and skips redundant work
on the hot path.  The win shape:
- 16 fewer host→GPU `ggml_backend_tensor_set` calls per synth.
- 16 fewer staging-buffer write+barrier pairs internally inside
  ggml-vulkan.
- Lowest impact on big-prompt workloads where text_emb is
  large (linear in `text_len`).

The bench surface from round 7 (`--bench-per-step`) shows the
per-step time on real hardware.  Step 0 should be unchanged
(cold miss = always uploads).  Steps 1-4 should be measurably
faster.

---

### Vulkan optimisation round 11 (May 2026, QVAC-18605 follow-up #9) — Packed-QK RoPE + GPU-bridge layout fix

**Critical correctness fix.**  Round 11 didn't add a new
optimisation — it made every prior round actually run end-to-end
on real hardware.  Rounds 8 + 9 + 10 (front-block / style /
group GPU bridges + text-input upload-skip) had all shipped CPU-
only unit-test green, but the unit tests never exercised the
production code path with a real GGUF carrying
`vector_rope_theta`.  The first end-to-end synth attempt (CPU
*or* Vulkan) aborted at
`GGML_ASSERT(HD == n_heads * head_dim)` inside
`apply_rope_to_packed_qk` — and even past that assertion, every
`ggml_backend_tensor_copy(q_src, q_tc_in)` in the GPU-bridge
fast paths would have hit
`GGML_ASSERT(ggml_are_same_layout(src, dst))` because Q/K/V
matmul outputs were the byte-for-byte transpose of what the
attention cache's `q_tc_in` / `k_tc_in` / `v_tc_in` tensors
expect.

#### Root cause

`apply_rope_to_packed_qk` (introduced in PR #16 audit follow-up
#5) was written under the assumption that
`dense_matmul_time_ggml` returns a `ne=[H*D, L]` "channel-
fastest-in-memory" tensor.  In fact, the matmul (both the CPU
`cblas_sgemm` fast path and the GPU `conv1d_f32(K=1)` fallback)
produces `ne=[L, H*D]` with **channel-major-flat memory**
(`data[t + c*L]`) — the bit-exact transpose of the helper's
input contract.

The CPU unit test that landed alongside the helper
(`test_supertonic_rope_packed_qk.cpp`) hand-built Q under the
wrong `[HD, L]` shape, so the failure mode was invisible to CI.
Similarly, `vector_text_attention_cache::q_tc_in` etc. are
`ggml_new_tensor_2d(F32, HD, L)` → **time-major-flat memory**
(`data[c + t*HD]`).  V (and the style Q/K/V which have no RoPE
to mask the layout flip) flowed into the GPU bridge from
matmul → channel-major-flat bytes → mismatched layout against
`q_tc_in` → `ggml_backend_tensor_copy` aborts on
`ggml_are_same_layout`.

#### The fix (strict TDD)

1. **Test (new RED contract)**:
   `test_supertonic_rope_packed_qk.cpp` rewritten to build Q
   under the **production** shape `ne=[L, HD]` (matmul's actual
   output) with channel-major-flat memory.  The reference is
   built in scalar `apply_rope`'s native time-major-flat layout;
   the test verifies the helper's output bytes match the
   reference bit-for-bit AND pins `y->ne[0] = HD, y->ne[1] = L`
   so the downstream `q_tc_in` blit cannot regress on layout.

2. **Helper (`apply_rope_to_packed_qk` in
   `supertonic_internal.h`)**: Add a head-of-pipeline
   `ggml_cont(ggml_transpose(q))` to flip from the matmul's
   `ne=[L, HD]` channel-major-flat memory to the `ne=[HD, L]`
   time-major-flat memory `apply_rope_in_graph` (and the
   downstream `q_tc_in`) consumes.  The rest of the pipeline
   (view-as-`[D, H, L]` → cont → `apply_rope_in_graph` →
   reshape-to-`[HD, L]`) is unchanged.  Returns ne=[HD, L]
   time-major-flat — **the SAME layout as `q_tc_in`** so the
   GPU bridge blit is bit-exact.

3. **V (and style Q/K/V) graph-side transpose**: V has no RoPE
   to hide behind, so the same `ggml_cont(ggml_transpose(...))`
   is open-coded at the matmul output in
   `build_group_graph_cache` (line ~1088),
   `ve_front_block_proj_cache` (line ~2774), and
   `build_res_style_qkv_cache` (line ~1459 — applied to all
   three sq / sk / sv since the style path has no RoPE
   anywhere).

4. **Legacy host-bridge downloads**: The host-bridge fallback
   paths used `tensor_to_time_channel(q_rope_gpu)` to download
   post-RoPE Q/K, which under the new layout would be a
   transpose-of-the-transpose.  Switched to `tensor_raw_f32`
   for all four post-RoPE tensors plus all four V tensors plus
   the trace-mode style sq/sk/sv downloads — the bytes are
   already in the layout scalar `apply_rope` /
   `flash_attention_qkv` host references consume (`out[t*HD +
   c]`), so the raw download is the correct call.

#### Verification

| Backend / Adapter | Pre-fix | Post-fix |
|---|---|---|
| CPU | `GGML_ASSERT(HD == n_heads * head_dim) failed` → core dump on first step | ✅ writes 3.89s 44.1 kHz WAV |
| Vulkan NVIDIA RTX 5090 (KHR_coopmat, FP16) | same crash | ✅ writes 6.53s WAV; **44 ms / 5-step bench, 74× realtime** (median over 5 runs) |
| Vulkan AMD RADV iGPU (UMA, FP16) | same crash | ✅ writes 3.64s WAV; 178 ms / 5-step bench, 7× realtime |
| Vulkan Mesa lavapipe (CPU emulator) | same crash | ✅ writes 1.21s WAV (correctness baseline) |

Whole CPU-only `ctest -L unit` reports **22 / 22 tests, 0
failures, 0 regressions**.  Vulkan build's `ctest` likewise
22 / 22.

#### Why the unit tests missed it

The 22 unit tests cover individual helpers (capability cache,
upload-skip tracker, F16 deny-list API, etc.) and small-tensor
in-graph parity (rope-in-graph, packed-qk-rope, in-graph-
transpose) but **none of them execute
`supertonic_vector_step_ggml` against a real GGUF**.  The 30
"Disabled" tests in `ctest` would have caught this — they're
the model-fixture tests gated on a locally-generated GGUF.
Round 11 is exactly the kind of failure those exist to detect.

The TDD test added in this round (the rewritten
`test_supertonic_rope_packed_qk.cpp`) now closes the gap for the
specific helper that crashed: it builds Q under the production
matmul shape AND pins the output layout contract that the GPU-
bridge `ggml_backend_tensor_copy` requires.  A future
re-introduction of the (incorrect) old contract would fail the
test at compile time on the `y->ne[0] == HD` shape check, even
before the bit-for-bit data comparison runs.

#### Perf snapshot (RTX 5090, default short prompt, F16 K/V)

```
  preprocess             med=   0.00  ms
  duration               med=   0.97  ms
  text_encoder           med=   2.94  ms
  vector_estimator (5 step) med=  37.70  ms
    vector_step[0]       med=   7.44  ms   (cold pipeline)
    vector_step[1..4]    med=   7.01–7.05  ms   (steady state)
  vocoder                med=   2.47  ms
  total                  med=  44.08  ms

  RTF (total / audio):   med=0.013
  Real-time multiplier:  med=74.28x
```

The round-1..10 wins (multi-device cache, BF16/Q8_0 K/V
dispatch, native LEAKY_RELU, F16 weights deny-list, prewarm,
front-block + style + group GPU bridges, text-input upload-
skip) are all in this number — they just couldn't actually run
until round 11 unblocked the path.

---

## Remaining Work

### Runtime and performance

- Investigate vector 3/4-thread variance.
- Consider a fused text relpos attention op only if profiling shows text is the
  next hard blocker.
- Add quantized Supertonic GGUF support once graph paths are ready for f16/q8.
- Run the chatterbox-style OpenCL profiling sweep on Adreno (Q4_0 weights,
  `flash_attn_f32_f16` enabled) to confirm the Supertonic bottleneck shifts
  from custom CPU ops to `kernel_mul_mm_f32_f32` and the same convnext block
  shape that chatterbox already profiled.
- ~~Evaluate GPU backends after CPU graph structure is fully stable.~~ — initial
  Metal port landed 2026-05-11; see "Metal baseline (2026-05-11)" below.
- Add CI coverage for converter help/setup syntax and portable Supertonic build
  targets.

## Metal baseline (2026-05-11)

First end-to-end Metal run of the Supertonic 2 pipeline. Approach mirrors
Chatterbox's pattern: single `ggml_backend_metal_init()` at model load, no
backend scheduler, and CPU-only `ggml_custom_4d` fast paths gated on
`!ggml_backend_is_cpu(model.backend)` so the same graph builders fall through
to stock `ggml_im2col` + `ggml_mul_mat` (etc.) when the backend is Metal.

Implementation:

- `model_prefers_cpu_kernels(const supertonic_model &)` added in
  `src/supertonic_internal.h`. Returns `true` when `model.backend == nullptr`
  or `ggml_backend_is_cpu(model.backend)`.
- Per-stage helpers (`conv1d_f32`, `depthwise_same_ggml`, `layer_norm_ggml`,
  `dense_matmul_time_ggml`, `bias_gelu_ggml`, `pw2_residual_ggml`,
  `conv1d_causal_ggml`, `depthwise_conv1d_causal_ggml`, plus the tail-update
  custom op in `vector_estimator.cpp`) now take a `bool use_cpu_fastpath` and
  AND it into the existing dtype/shape gates.
- Per-stage builders inject
  `const bool use_cpu_fastpath = model_prefers_cpu_kernels(model);` at the top
  and pass it down through `vector_convnext_ggml`, `convnext_block_ggml`, the
  text/vector/style attention cache builders, the tail graph builder, and the
  trace builder.
- `text_encoder.cpp` and `duration.cpp` accept the flag for call-site
  uniformity but mark it `[[maybe_unused]]` — those stages have always built
  their graphs via stock ggml ops and are Metal-safe at HEAD.
- `supertonic_bench.cpp` gains `--n-gpu-layers N` (passed through to
  `load_supertonic_gguf`) so the same harness drives CPU and Metal.

Smoke test (`supertonic-cli --n-gpu-layers 1`) produces a 1.44 s WAV that is
byte-length-identical to the CPU output, confirming the graph builders run
end-to-end on Metal. A `GGML_ASSERT([rsets->data count] == 0)` fires inside
`ggml_metal_device_free` at process exit (atexit ordering with Metal's
residency-set finaliser) — same shape as the Chatterbox `t3_stack_registry`
atexit issue; cosmetic, fires after the WAV is fully written. Mitigation TBD.

Benchmark (Apple M2, q8_0 GGUF, 4 threads, 3.204 s of audio, 5-step CFM, 5 runs
+ 1 warmup, same flags as `supertonic-cpp.json` / `supertonic-onnx-cpu.json`):

| Stage                       | CPU q8_0   | Metal q8_0 | Δ vs CPU | ONNX CPU f32 |
|-----------------------------|-----------:|-----------:|---------:|-------------:|
| preprocess                  |    0.01 ms |    0.01 ms |       — |      0.06 ms |
| duration                    |    1.76 ms |    2.50 ms |   +0.74 |      1.48 ms |
| text_encoder                |   13.44 ms |   13.83 ms |   +0.39 |      9.04 ms |
| vector_estimator (5 steps)  |   94.86 ms |  173.08 ms |  +78.22 |     82.65 ms |
| vocoder                     |   43.44 ms |   59.74 ms |  +16.30 |     51.32 ms |
| **total**                   | **153.5**  | **249.9**  |  **+96.4 (+63%)** | **144.9** |
| RTF                         |     0.048  |     0.078  |          |       0.045 |
| real-time multiplier        |     20.9×  |     12.8×  |          |       22.1× |

Verdict: the Metal port is **correctness-validated but slower than CPU at this
graph shape**. Two ggml-side stages dominate the regression:

- **`vector_estimator` +82 %** (94.9 → 173.1 ms median). The 5 denoising steps
  build many small ConvNeXt graphs (depthwise + pointwise + norm + GELU +
  pointwise, repeated across blocks). On M2 these become Metal kernel
  launches that are too short to amortise launch overhead; the CPU fast paths
  (cblas-backed `pointwise_op` / unrolled depthwise K=5) had a real lead.
- **`vocoder` +38 %** (43.4 → 59.7 ms median). Same kernel-launch-bound
  pattern, smaller deficit because the vocoder graph is a single persistent
  cgraph that's reused across calls (less per-step overhead than the
  vector-estimator's per-block cgraphs).

`text_encoder` and `duration` are unchanged within noise — expected, those
already used the stock-op path on CPU.

`supertonic-bench --runs 8 --warmup 3 --n-gpu-layers 1` drifted to ~288 ms
median (up from ~250 ms at runs=5 / warmup=1), suggesting Metal residency
sets accumulate across calls in this harness; investigate before drawing
percentile-style conclusions from longer Metal runs.

Artifacts: `artifacts/bench/supertonic-cpu.json`,
`artifacts/bench/supertonic-cpu-after.json` (post-gating CPU regression
check, median 158.2 ms / +3 % vs the pre-port baseline — within noise),
`artifacts/bench/supertonic-metal.json`,
`artifacts/bench/supertonic-onnx-cpu.json`,
`artifacts/bench/supertonic-onnx-coreml.json`,
`artifacts/bench/metal-phase-a.txt` (the Phase A failure-mode trace before
gating).

### Next: Metal optimisation passes (Phase E in the plan)

Backlog **revised after the 2026-05-11 dispatch-count profile** (see
"Dispatch-count profile" below). The pre-profile working hypothesis
(step batching, QKV stacking, f16 weights) turned out to be wrong on
multiple counts. Revised priority order:

1. **Single-graph consolidation per CFM step (THE PR).** The diagnostic
   shows ~21 separate `graph_compute` calls per step (front prep +
   text-attention + style-qkv + style-attention + style-residual-norm
   inline × 4 groups + tail). On M2 each call carries ~1.86 ms of fixed
   command-buffer overhead regardless of node count. Consolidating into
   ONE `ggml_cgraph` per step (5 dispatches per synth, projected total
   Metal ~46 ms) is by far the biggest win available; the rest of the
   backlog only matters if this leaves residual gap. Specific work
   below.
2. **(Was step batching across CFM iterations.)** Closed: the CFM step
   loop has a sequential dependency (`latent.swap(next)` at
   `supertonic_engine.cpp:240`), so Chatterbox-style batching along
   `ne[2]` doesn't apply here. The win from item 1 above is bigger
   anyway; revisit only if a future flow-matching variant decouples the
   steps.
3. **(Was QKV stacking on text-attention.)** Deprioritised. With item 1
   the QKV matmuls live inside the same dispatch as everything else —
   stacking saves 3 in-graph nodes per attention but doesn't reduce
   dispatch count. Only worth doing if Metal frame capture shows the
   three per-attention `kernel_mul_mm` launches are individually
   expensive after consolidation.
4. **(Was f16 weights for Metal.)** Closed: f16 GGUF is *slower* than
   q8_0 on both CPU and Metal (see "f16 GGUF experiment (2026-05-11)"
   below). q8_0's weight-bandwidth win beats f16's no-dequant on this
   graph shape.
5. **Custom Metal depthwise kernel.** Standby — only revisit if item 1
   leaves ConvNeXt depthwise as the residual hotspot. The `im2col +
   mul_mat` fallback would be replaceable with a single
   `kernel_depthwise_conv_1d` per call; `test/test_metal_ops.cpp` is
   the parity harness.
6. **Metal `rsets` keep-alive tuning** for long-running daemons.
   Cosmetic for benchmarks; investigate if a hosted-service user
   reports memory growth.

### Plan for item 1 — per-step graph consolidation

Architecture: introduce a `vector_step_full_cache` (per-shape
thread_local) that owns ONE `ggml_context`, ONE `ggml_cgraph`, ONE
`ggml_gallocr`. Build the entire per-step computation (proj_in →
4 × (ConvNeXt blocks + time-add + ConvNeXt + Q/K/V projection + RoPE +
flash-attention + out_fc + residual + layer-norm + style Q/K/V
projection + flash-attention + out_fc + residual + layer-norm) +
last_convnext × 4 + proj_out + mask + noise add) as one graph. ONE
`ggml_backend_graph_compute` per step.

The existing `build_text_attention_cache`, `build_group_graph_cache`,
`build_res_style_qkv_cache`, and `build_tail_graph_cache` get refactored
into **graph-builder helpers** that accept `(ggml_context*, ggml_cgraph*,
...input ggml_tensor*...)` and return output `ggml_tensor*`, instead of
owning their own contexts. The CPU path keeps the cache-of-subgraphs
architecture (parity, trace mode); only Metal routes through the
consolidated path. Detection via `!ggml_backend_is_cpu(model.backend)`
at the top of `supertonic_vector_step_ggml`.

**Critical sub-tasks** (the order matters for parity validation):

1. **In-graph RoPE.** Replace the CPU `apply_rope` call with
   `ggml_rope_ext` configured for Supertonic's `(t/L) * theta[d]`
   formula: `freq_base = 1.0`, `freq_scale = 1.0`, `freq_factors[d] =
   L / theta[d]`, `mode = GGML_ROPE_TYPE_NEOX` (split-pairs layout
   matches `apply_rope`'s `(i1, i2) = (offset+d, offset+D/2+d)` pattern
   per `supertonic_vector_estimator.cpp:1416`). Positions are an
   int32 `arange(L_q)` for Q and `arange(L_kv)` for K, set once at
   build time. ggml-metal's `kernel_rope_norm`/`kernel_rope_neox`
   already compile.

2. **In-graph layout conversion.** Replace
   `tensor_to_time_channel`/`pack_time_channel_for_ggml` host calls
   with `ggml_cont(ctx, ggml_transpose(ctx, x))` at the inter-stage
   boundaries.

3. **Compose the orchestrator** so all stages share one ctx/gf. Walk
   the existing `supertonic_vector_trace_proj_ggml` flow (lines
   2050–2585) and inline each `run_*_cache` call as graph-builder
   helper invocations.

4. **Parity test.** Add a `test_supertonic_vector_metal_consolidated`
   CTest target that compares the consolidated Metal path to the CPU
   reference for one step at a representative L (137-ish). Tolerance
   ~1e-2 (loose because of float-order effects across the merged
   graph).

5. **Bench.** Re-run `supertonic-bench --n-gpu-layers 1` and target
   `SUPERTONIC_COUNT_DISPATCHES=1` to verify total dispatches drop
   from 120 to ~10 and total wall to ~46 ms.

**Size estimate.** ~600–1000 new lines (mostly the consolidated build
function); the existing trace path stays untouched. Trace-mode tests
keep using the old multi-cache orchestrator.

**Risk.** The two non-trivial pieces are (a) `ggml_rope_ext` parameter
mapping matching CPU `apply_rope` to within 1e-3 — verify before
inlining everything else — and (b) memory budget for one big graph
across all groups (`MAX_NODES=2048` may not be enough; estimate ~3500
nodes for the full per-step graph).

Each commit on the consolidation branch should land in a single PR;
the work is too coupled to split cleanly.

Backlog items 2–6 above stay as separate per-PR follow-ups in their
listed priority. Do not bundle.

### Dispatch-count profile (2026-05-11)

Instrumented `supertonic_graph_compute` with a wall-time + node-count
printout gated on the `SUPERTONIC_COUNT_DISPATCHES` env var. Re-running
`supertonic-cli --n-gpu-layers 1 --text "Hello."` on the same M2:

- **120 graph_compute dispatches per single synth** (entire pipeline,
  vector estimator + vocoder + text encoder + duration).
- **Cumulative graph_compute wall: 222.8 ms** out of the ~250 ms total
  Metal synth — i.e. graph_compute IS the cost; CPU-side data marshalling
  is the residual ~30 ms.
- **Mean per-dispatch wall: 1.86 ms.** Even 17-node tiny dispatches cost
  ~770 µs each; 170-node mid graphs cost 1.1–1.7 ms. The fixed
  per-dispatch Metal overhead (command-buffer setup + pipeline lookup +
  encode + commit + wait) dominates.

Dispatch distribution (counts × node-size, sorted by frequency):

  40 × 18 nodes (the 5×8 text-attention sub-graphs per step)
  20 × 12 nodes
  20 × 90 nodes
  15 × 262 nodes (the 5×3 group-prep graphs)
  ~25 misc

The 80 small (≤90 nodes) dispatches account for an estimated ~120 ms of
Metal time. Consolidating them into the larger per-step graphs would
likely halve the gap to the CPU baseline.

### f16 GGUF experiment (2026-05-11)

Hypothesis: q8_0 dequant in the per-`mul_mat` path was the Metal
bottleneck. Tested by converting the bundle with `--ftype f16` (132 MB
GGUF vs 252 MB for q8_0) and re-benching:

  Metal q8_0 total median: 249.9 ms
  Metal  f16 total median: 286.5 ms (+15 %, worse)
  CPU   q8_0 total median: 153.5 ms
  CPU    f16 total median: 168.7 ms (+10 %, worse)

f16 is uniformly *slower* than q8_0, on both CPU and Metal. q8_0
dequant is not the bottleneck — ggml-metal's q8_0 `mul_mat` kernel is
well-tuned for these tensor shapes and the smaller weight bandwidth
helps. Phase E.3 closed; do not pursue an f16-on-Metal variant.

### Dispatch profiling hook

`SUPERTONIC_COUNT_DISPATCHES=1 ./build/supertonic-cli ...` prints one
line per `ggml_backend_graph_compute` call:

  supertonic_graph_compute #N nodes=K  wall=W us  cumul=C ms

Zero-overhead when the env var is unset (single env var read +
branch-predicted skip).

## Per-step graph consolidation (landed 2026-05-11)

Landed `supertonic_vector_step_one_graph_ggml` at the end of
`src/supertonic_vector_estimator.cpp` plus the helpers
`apply_supertonic_rope_ggml`, `append_text_attention_subgraph`, and
the `vector_step_one_graph_cache` struct.  Routing in
`supertonic_vector_step_ggml` enables this path **by default on
any non-CPU backend** (Metal, CUDA, Vulkan, OpenCL).  CPU keeps
the multi-cache trace_proj path — its CPU fast-paths and
`thread_local` sub-graph caches stay competitive on CPU and trace
mode for parity tests still uses the per-stage outputs.  Override
via `SUPERTONIC_DISABLE_ONE_GRAPH=1` if needed.

### Dispatch + bench numbers (Apple M2, q8_0, 4 threads, 5-step CFM)

`SUPERTONIC_COUNT_DISPATCHES=1 ./build/supertonic-cli --n-gpu-layers 1`
shows the dispatch profile collapsing from **120 → 20 total
dispatches** per synth (5 of which are 1886-node consolidated
per-step graphs).  Mean per-dispatch wall climbs from 1.86 ms to
7.9 ms — more real work per kernel batch, less time burned on
command-buffer setup — and total `graph_compute` wall drops from
222.8 ms to 157.7 ms (-29 %).

`supertonic-bench` on Metal, 5 runs + 1 warmup, identical flags to
`supertonic-cpu.json` / `supertonic-onnx-cpu.json`:

  | Stage                       | trace_proj (B) | one-graph (E.cons) |
  |-----------------------------|---------------:|-------------------:|
  | preprocess                  |          0.01ms |             0.02ms |
  | duration                    |          2.50ms |             3.87ms |
  | text_encoder                |         13.83ms |            16.58ms |
  | vector_estimator (5 steps)  |        173.08ms |           147.83ms |
  | vocoder                     |         59.74ms |            60.51ms |
  | **total**                   |     **249.92ms**|        **229.06ms**|
  | RTF                         |           0.078 |              0.071 |
  | real-time multiplier        |          12.82× |             13.99× |

Net: **-15 % on the dominant vector_estimator stage, -8 % on the
total**.  Correctness validated: `cpu-ref` vs `metal-one-graph` for
the same text+seed gives correlation **1.0000**, max abs diff 101
LSB (CPU peak amplitude 6639, so ~1.5 % — normal Metal-vs-CPU
floating-order noise).  No regression vs the Phase B port.

### Why the win is smaller than projected

Pre-implementation projection was ~46 ms total (saving the full
~204 ms of dispatch overhead at 1.86 ms × ~110 saved dispatches).
Reality: the per-dispatch overhead estimate (1.86 ms) was an
*average*, not a constant.  The new 1886-node consolidated graphs
are big enough that the GPU is actually doing real compute work
during the dispatch — kernel-launch overhead is no longer the
bottleneck, but the work itself has moved to dominating.

The bench tells the story: per-step wall time dropped from
~33 ms (= 173/5) to ~30 ms (= 147/5).  The Metal device now spends
most of its time actually computing matmuls rather than waiting
on command-buffer plumbing.  Further wins now require *less work*,
not *fewer dispatches* — that's items 2-5 of the remaining
backlog (QKV stacking, op fusion, custom depthwise kernel).

### Implementation notes

- **`apply_supertonic_rope_ggml`** translates Supertonic's
  `angle = (t/L) * theta[d]` formula to `ggml_rope_ext` with
  `freq_base=1.0, freq_scale=1.0, freq_factors[d] = L / theta[d]`,
  `mode=GGML_ROPE_TYPE_NEOX` (split-pairs rotation matches
  `apply_rope`'s `(i1=offset+d, i2=offset+D/2+d)` layout at
  `supertonic_vector_estimator.cpp:1416`).  Positions are int32
  `arange(q_len)` for Q and `arange(text_len)` for K, set per
  call when L or text_len change.  ggml-metal's
  `kernel_rope_norm`/`kernel_rope_neox` already compile.

- **Layout invariant: the GGML tensors take channel-major buffers
  raw.**  The trace_proj_ggml path at lines 2143/2151 sets `x_in`
  directly from `noisy_latent` (no host transpose) and `text_in`
  directly from `text_emb`; the ne=[L, Cin] / ne=[text_len, 256]
  tensors interpret that channel-major buffer as their natural
  layout (innermost dim = time = fast-in-memory).  My initial
  consolidation tried to "helpfully" transpose the inputs into
  (t, c) layout, which corrupted the tensor data and produced
  correlation 0.0034 garbage on every backend.  Fix: direct
  `ggml_backend_tensor_set` from raw caller buffers, matching the
  existing path exactly.  Same fix on the output path
  (`ggml_backend_tensor_get` straight into `next_latent_out`).

- **Cache invalidation:** keyed on `(model.generation_id, L,
  text_len, total_steps)`.  Rebuild when any change.  The
  `vector_step_one_graph_cache` is a single `thread_local`
  instance — different Engines / synths share it via the
  generation_id key.

### Remaining Phase E backlog

**Tier 1 status (2026-05-11):**

- ✅ **Per-step vector_estimator consolidation** (this PR) — biggest
  Tier 1 win, -8 % on total Metal, parity 1.0000.
- ✅ **Vocoder already a single dispatch** (461-node graph) —
  no consolidation needed.
- ⏸ **text_encoder + duration consolidation** — measured
  contribution: ~22 ms cold-start dispatch wall across the 14
  small dispatches that come before the vector_estimator graphs.
  Post-warmup the bench shows text_encoder ≈ 17 ms and
  duration ≈ 4 ms — most of which is the dispatches themselves;
  consolidating to 1 dispatch each would save ~5-10 ms
  steady-state.  Deferred because relpos_attention has 9
  per-shape mask tensors + intricate
  `ggml_view_3d`/`ggml_permute`/`ggml_sum_rows` plumbing that's
  not a straight copy of the vector_step pattern — needs its
  own focused 2-3 hour session with parity validation harness
  before re-enabling on the GPU dispatcher.
- ⏸ **QKV stacking** — once `vector_estimator` is already in
  one graph, stacking the three `dense_matmul_time_ggml` calls
  saves in-graph nodes but no dispatch count.  Metal-frame-
  capture didn't show the QKV matmuls as the hot path, so the
  expected win is tiny.  Pursue only if Tier 2 hits diminishing
  returns.
- ⏸ **`ggml_cont` elimination** — the consolidated path does
  `ggml_cont(ggml_transpose(...))` for Q/K/V before rope, and
  again inside `apply_supertonic_rope_ggml`.  These could be
  avoided by views with custom strides, but ggml's `view_3d`
  doesn't expose `nb0` (only `nb1`/`nb2`), so the cont copies
  are required for the rope kernel's expected layout.  Could
  use `ggml_permute` + careful 4D views to remove some, but
  the win is small and the layout-bug risk is high.

## Tier 2 progress (2026-05-11) — op-level reductions before custom kernels

Before sinking time into custom .metal kernels via the QVAC
ggml-speech port patches (the original Tier 2 plan), there are
op-level reductions inside the consolidated per-step graph that
trim dispatch count without touching ggml's kernel set.  Each
landed as its own commit in PR #15.

### Diagnostic: `SUPERTONIC_DUMP_OP_HISTOGRAM=1`

Added an env-var-gated dump of per-graph op-type histograms to
`supertonic_graph_compute`.  Zero overhead unset.  Lets us see
exactly which ggml ops dominate the consolidated graph and which
are pure-metadata (RESHAPE/VIEW/PERMUTE/TRANSPOSE — confirmed
no-op in ggml-metal-ops.cpp:186-195).

**Consolidated per-step graph at HEAD (post-Tier-2 commits):**

  | op                | count | dispatch on Metal? |
  |-------------------|------:|--------------------|
  | RESHAPE           |   580 | no (metadata only) |
  | ADD               |   197 | yes (often fused)  |
  | CONT              |   148 | yes (memcpy)       |
  | MUL_MAT           |   122 | yes (matmul)       |
  | IM2COL            |   118 | yes (memrearrange) |
  | VIEW              |    88 | no                 |
  | PERMUTE           |    72 | no                 |
  | MUL               |    70 | yes (often fused)  |
  | TRANSPOSE         |    68 | no                 |
  | REPEAT            |    56 | yes                |
  | CONCAT            |    56 | yes                |
  | NORM              |    36 | yes                |
  | UNARY             |    32 | yes (GELU/SiLU)    |
  | ROPE              |     8 | yes                |
  | FLASH_ATTN_EXT    |     8 | yes                |
  | SCALE             |     1 | yes                |
  | **total**         | **1660** | **852 dispatched** |

808 of 1660 nodes are metadata-only no-ops — what looks like a
large graph is really ~852 real Metal dispatches per per-step
graph (down from ~1078 dispatched ops in the pre-Tier-2 layout).

### Landed wins

1. **`repeat_like` returns the broadcast-compatible reshape
   without `ggml_repeat`** — ggml_add/ggml_mul broadcast natively
   when one operand has dim==1 in a position the other has dim==N,
   so the explicit ggml_repeat was redundant work.  All four
   supertonic files (vector_estimator, vocoder, text_encoder,
   duration) had the same pattern; same fix applied to each.
   **-226 REPEAT ops** per step graph.  Override via
   `SUPERTONIC_FORCE_EXPLICIT_REPEAT=1`.

2. **`apply_supertonic_rope_ggml` drops the defensive
   `ggml_cont`** — the [D, H, q_len] view onto a contiguous
   [H*D, q_len] tensor is itself contiguous (nb[0]=elem_size,
   nb[1]=D*elem_size, nb[2]=H*D*elem_size = ne[0]*ne[1]*elem_size),
   so `ggml_rope_ext` accepts the view directly.  **8 fewer
   kernel_cpy dispatches per per-step graph** × 5 = 40 saved per
   synth.

### Bench delta

Apple M2, q8_0, 4 threads, 5-step CFM, 3.20 s of audio, 5 runs +
1 warmup, identical flags to the existing JSON artifacts:

  | Stage                       | Phase B | post-cons | post-repeat | post-rope-cont |
  |-----------------------------|--------:|----------:|------------:|---------------:|
  | preprocess                  |   0.01 ms |   0.02 ms |     0.01 ms |        0.02 ms |
  | duration                    |   2.50 ms |   3.87 ms |     4.15 ms |        4.44 ms |
  | text_encoder                |  13.83 ms |  16.58 ms |    15.80 ms |       14.97 ms |
  | vector_estimator (5 steps)  | 173.08 ms | 147.83 ms |   129.23 ms |      123.94 ms |
  | vocoder                     |  59.74 ms |  60.51 ms |    53.91 ms |       53.99 ms |
  | **total**                   | **249.92ms** | **229.06ms** | **203.04ms** | **199.90ms** |
  | RTF                         |   0.078 |   0.071  |     0.063   |       0.062    |
  | real-time multiplier        |  12.82× |  13.99×  |    15.78×   |      16.03×    |

**Cumulative Tier 1 + early-Tier-2: -50 ms total (-20 %) vs the
Phase B Metal baseline.**  Parity vs CPU reference preserved at
correlation 0.9999, max abs diff 249 LSB (~3.7 % of peak
amplitude 6639 — within the float-order tolerance the
consolidation already trades for one-graph-per-step).  Still ~50
ms behind CPU q8_0 (153 ms) and ONNX CPU (145 ms), but the gap
is closing.

### Remaining op-level reductions

- **118 IM2COL ops** are almost all K=1 1×1 convs (called from
  `dense_matmul_time_ggml` via the existing `conv1d_f32` graph
  fallback).  For K=1 the im2col is a transpose; could be
  replaced with a direct `ggml_mul_mat` on the transposed
  weight/input.  Projected ~3-6 ms saved.  Tricky to get right
  without breaking layout assumptions of consumers.
- **148 CONT ops** — 32 are weight-transpose conts in
  `dense_matmul_time_ggml` (per call, but the weight is constant
  per shape; could cache the transposed copy at engine
  construction).  Projected ~5-8 ms saved.
- **56 CONCAT + 56 REPEAT (remaining)** come from
  `edge_clamp_pad_1d` materialising the replicate padding.  A
  custom Metal `kernel_supertonic_pad_edge` would collapse these
  into one dispatch per padding call.

### Tier 2 custom Metal kernels + load-time weight prep — landed (2026-05-11)

Four fused Metal kernels shipped through the local
`tts-cpp/cmake/vcpkg-overlay-ports/ggml/` overlay (chained on top
of the QVAC ggml port via `VCPKG_OVERLAY_PORTS`).  Each adds a
new `GGML_OP_SUPERTONIC_*` op with a CPU forward as parity
backstop and a Metal kernel as the production path.  Override
each individually with the listed env var.

1. **`kernel_supertonic_depthwise_1d`** (commit aa4f65c3) —
   fuses edge-clamp pad + im2col + mul_mat + add into one Metal
   dispatch for K ∈ {3, 5}.  Used by every ConvNeXt block in
   vector_estimator, vocoder, text_encoder, duration.  Override:
   `SUPERTONIC_DISABLE_FUSED_DEPTHWISE=1`.
2. **`kernel_supertonic_layer_norm_channel`** (commit 55adf87b)
   — fuses permute + cont + ggml_norm + mul + add + permute +
   cont into one dispatch.  Per time-step, one threadgroup with
   simd_sum reductions for mean/var.  Override:
   `SUPERTONIC_DISABLE_FUSED_LAYER_NORM=1`.
3. **`kernel_supertonic_pw2_residual`** (commit 7a5c0393) —
   fuses `add(bias) + mul(gamma) + add(residual)` (3 ops) into
   one dispatch at the tail of each vector ConvNeXt block.
   Override: `SUPERTONIC_DISABLE_FUSED_PW2_RESIDUAL=1`.
4. **`kernel_supertonic_bias_gelu`** (commit df20115d) — fuses
   `add(bias) + gelu_erf` between pw1 and pw2 of every vector
   ConvNeXt block.  Uses the same `erf_approx<float>` template
   as the stock `kernel_gelu_erf_f32` so the fused output is
   bit-identical to the unfused chain.  Override:
   `SUPERTONIC_DISABLE_FUSED_BIAS_GELU=1`.

Plus a load-time optimization:

5. **Pre-transposed matmul weights** (commits e935ffb7,
   da9553e3) — materialize transposed copies of every
   `:onnx::MatMul_*` source weight at engine load time on
   non-CPU backends.  Eliminates the runtime
   `cont(transpose(w))` dispatch that `dense_matmul_time_ggml`
   (and the direct `ggml_mul_mat` time-projection sites) used
   to emit on every graph compute — ~24 cont sites × 5 CFM
   steps = 120 dispatches saved per synth.  Override:
   `SUPERTONIC_DISABLE_WEIGHT_PRETRANSPOSE=1`.

6. **Vocoder pw1 fused bias_gelu** (commit 64efe99a) — extends
   the bias_gelu fusion to the vocoder's ConvNeXt blocks.
   `conv1d_causal_ggml(..., b=nullptr, ...)` skips the internal
   bias-add and feeds the matmul output to the fused op
   directly.  CPU keeps its existing cblas-inside path.  ~10
   dispatches saved per vocoder pass.

Also investigated but **not landed**:

- **Vocoder pw2_residual fusion** (commit 53a58f5b explains
  why) — the vocoder stores its block scale as
  `gamma.ne[0] == 1` (a single learnable scalar), while
  `pw2_residual_ggml` requires `gamma.ne[0] == C`.  Shapes
  incompatible, would need a new vocoder-specific scalar-gamma
  variant op for a ~0.4 ms projected gain — below the noise
  floor of the current bench.  Skipped.

### Final Tier 2 bench

Apple M2, q8_0, 4 threads, 5-step CFM, 3.20 s of audio, 10
runs + 2 warmup, `--n-gpu-layers 1` (numbers from
`artifacts/bench/supertonic-cpp-metal-final.json`):

  | Stage                       | Phase B Metal | Tier 2 final | CPU q8_0 ref |
  |-----------------------------|--------------:|-------------:|-------------:|
  | preprocess                  |       0.01 ms |      0.02 ms |     0.01 ms  |
  | duration                    |       2.50 ms |      6.03 ms |     1.97 ms  |
  | text_encoder                |      13.83 ms |     18.47 ms |    13.44 ms  |
  | vector_estimator (5 steps)  |     173.08 ms |     97.76 ms |    94.86 ms  |
  | vocoder                     |      59.74 ms |     52.02 ms |    43.44 ms  |
  | **total**                   |  **249.92ms** |  **174.49ms**| **153.52ms** |
  | RTF                         |        0.078  |       0.054  |       0.048  |
  | real-time multiplier        |       12.82×  |       18.4×  |       20.8×  |

**Cumulative Tier 1 + Tier 2 wins: -75 ms total (-30%) vs the
Phase B Metal baseline.**  Parity vs CPU q8_0 reference holds
at correlation 0.9999 / L∞ ≈ 1.7e-3 across the whole sequence
— bit-identical pipeline output before/after the optimizations
on Metal.

The pretranspose A/B (env-var off vs on, same machine state)
is the cleanest single-knob signal: total 182.75 → 174.38 ms
(-8.37 ms), vec_est 108.61 → 100.45 ms (-8.16 ms).

### Where the remaining 21 ms gap-to-CPU lives

  | Stage                       | Metal Tier 2 | CPU q8_0 | Gap          |
  |-----------------------------|-------------:|---------:|-------------:|
  | vector_estimator (5 steps)  |      97.76 ms |   94.86 ms |     2.90 ms |
  | vocoder                     |      52.02 ms |   43.44 ms |     8.58 ms |
  | text_encoder                |      18.47 ms |   13.44 ms |     5.03 ms |
  | duration / other            |        ~6 ms  |     ~1.7 ms |    ~4 ms    |
  | **total**                   |  **174.49ms** | **153.52ms** | **20.97 ms** |

Vector estimator is now Metal's strongest stage in absolute
terms (within 3 ms of CPU on its 100-ms budget); vocoder is at
parity with ONNX-CPU (52.0 vs 51.3 ms) and is now the dominant
remaining gap-to-CPU.  Vocoder uses `conv1d_causal_ggml` not
`dense_matmul_time_ggml`, so neither the pretranspose
optimization nor (until 64efe99a) the fused bias_gelu applied
there — the weights are already in conv1d-kernel `[K, IC, OC]`
layout from the GGUF.

### What's still pursuable post-Tier-2 (not in this round)

1. **KV stacking on cross-attention** — concat W_key and
   W_value along out-dim at load time so the two text-side
   matmuls become one (Q stays separate, different input).
   ~30 invocations per synth × ~0.1-0.2 ms each ≈ 3-6 ms
   projected, but the small matmul size means this might be
   noise-bound.  Could combine with pretranspose: stack the
   pretransposed K+V into one wider weight.
2. **Vocoder `pw2_residual_scalar_gamma` op** — new
   vocoder-specific fused op handling `gamma.ne[0]==1`.  ~10
   dispatches saved per vocoder pass ≈ 0.4 ms.  Below noise
   floor; skip unless other wins are found first.
3. **Full ConvNeXt block fusion** (the original T2.3 plan) —
   deferred because pw1/pw2 weights are 4C×C ≈ 1MB each,
   vastly exceeding M2's 32KB threadgroup memory budget.  Would
   need to call out to `ggml_mul_mat` for the matmuls, which
   defeats most of the fusion benefit.
4. **Activation layout change** — eliminate the 32 remaining
   `cont(transpose(activation))` calls on Q/K/V activations per
   per-step graph.  Would require touching the whole attention
   pipeline (rope, flash_attn, output projection) — too
   invasive for the projected ~3-5 ms win.
5. **CFM step batching (B=2)** — N/A for Supertonic.  The CFM
   loop in `supertonic_engine.cpp` is a sequential ODE solver
   (each step depends on the previous output), unlike
   chatterbox's CFG cond+uncond pairs which fit naturally into
   `ne[2]` batching.

### Tier 2 closing the loop

The Tier 2 PR (`feat/metal-optimization-supertonic` on
tetherto/qvac-ext-lib-whisper.cpp) lands as:
- 4 custom Metal kernels behind individual env-var gates
- Load-time pretranspose mechanism + helper APIs
  (`try_pretransposed_weight`, `dense_matmul_time_pretransposed_ggml`)
- All under a local `tts-cpp/cmake/vcpkg-overlay-ports/ggml/`
  port that chains on top of the QVAC ggml port via
  `VCPKG_OVERLAY_PORTS`.
- CPU q8_0 perf unchanged (the fused-kernel + pretranspose
  paths are all gated on `!use_cpu_fastpath`).
- Parity vs CPU reference: corr 0.9999 / L∞ 1.7e-3 throughout.

## Phase A + B follow-up (2026-05-11)

### Landed on this PR after Tier 2 closed

| Commit     | Change | Bench delta (M2, 10 runs) |
|------------|--------|---------------------------|
| `bfb44092` | Phase 0: `--precision {f32,f16,q8_0}` flag + parity harness | 0 ms (infra) |
| `8f0be955` | A1+A2: single command buffer per synth + on-GPU latent through 5-step CFM loop | –1.37 ms total |
| `1b7496f6` | A3 step 1: enable `--precision q8_0` storage on Metal (asymmetric load) | –6.17 ms total |

Cumulative on top of Tier 2: total **174.49 ms → 166.39 ms** (–4.6%).
Real-time multiplier 18.4× → 19.3×.

### Why the wins are smaller than the original Phase A+B projection

The Phase A roadmap projected 30+ ms of cumulative gains.  Reality on M2
delivered ~8 ms.  Three things drove the gap:

1. **Metal command-buffer submission on M2 is much cheaper than I
   estimated.** I cited "~1-2 ms fixed overhead per dispatch" based on
   an earlier diagnostic; actual cost is closer to 0.1-0.3 ms.  A1+A2's
   "single command buffer per synth" win (eliminating 4 inter-step
   dispatches) was projected –15 to –20 ms, landed at –1.4 ms.
2. **Unified memory makes `tensor_get`/`tensor_set` between stages
   nearly free.** There's no PCIe transfer cost to amortize.  The
   "on-GPU latent" win that's a big deal on discrete-GPU x86 doesn't
   apply on Apple silicon.
3. **`kernel_mul_mm_q8_0_f32` never fires.** A3's projected –20 to –30 ms
   was the matmul-bandwidth win from running ggml's optimized quantized
   matmul kernel.  But the kernel only dispatches when the quantized
   weight is `src0` (a) of `ggml_mul_mat`.  Supertonic's `[T, IC]`
   activation layout forces the weight into `src1` (b) via the
   `conv1d_f32` im2col wrapper, and ggml-metal falls back to a path
   that dequantizes to f32 first.  **The full A3 win is unlocked by
   B2 (activation layout permutation) — and only by it.**

### A4 (text_encoder + duration consolidation) — deferred

Analyzed but not implemented: text_encoder currently fires ~10 separate
`ggml_backend_graph_compute` calls (1 ConvNeXt front + 4 relpos attn
+ 4 ffn + 2 speech_prompted_attn × 2-graph pattern).  Duration adds
~4 small dispatches.

Full consolidation into 1-2 graphs would require:
- Extracting each sub-builder (`relpos_attention_ggml`, `ffn_block_ggml`,
  `speech_prompted_attention_ggml`) into append-to-graph helpers (the
  same shape of refactor that A1+A2 did for the per-CFM-step subgraph).
- Converting the host-side residual + layer_norm + tanh-key-packing
  work between sub-graphs into ggml ops.
- Engineering: 4-8 focused hours.
- Realistic return based on A1+A2's measured ratio: **–2 to –4 ms total**.

Deferred because: (a) ROI per hour is now smaller than B1/B2, (b) the
text_encoder + duration combined budget is only ~21 ms — even a perfect
collapse to 1 dispatch each saves ~5-7 ms maximum, with no compounding
effect on the other stages, (c) it doesn't unlock anything else
downstream (unlike B2 which unlocks A3 step 2).

Re-evaluate after B2 lands.  If the team needs every ms (e.g. for a
constrained-device target), this is the next item to revisit.

### Next levers on the table

| Phase | Projected (post-A1+A2 calibration) | Unblocks | Cost |
|-------|-----------------------------------:|----------|------|
| B1 — f16 activations end-to-end | –5 to –10 ms | nothing | medium |
| **B2 — activation layout permutation** | –3 to –5 ms direct, **+ unlocks A3 step 2 (–15 to –25 ms)** | A3 step 2 | high (invasive, touches rope + flash_attn + every attention site) |
| A3 step 2 — q8_0 matmul kernel firing (after B2) | –15 to –25 ms (theoretical) | — | medium-low (B2 does the heavy lifting) |
| B3 — argument buffer reuse | –2 to –5 ms | nothing | high (Metal backend internals) |
| A4 — text_encoder + duration consolidation | –2 to –4 ms | nothing | medium-high |

**The highest-leverage move now is B2.**  Without it, A3's matmul win is
unreachable.  The combined B2 + A3-step-2 stack is the only realistic
path to "Metal beats CPU outright on M2."

### B1 / B2 / B3 status after attempted continuation (2026-05-11)

After A4 deferred, attempted B1 (f16 end-to-end) and scoped B2.  Both
proved bigger than scoped to a single follow-up session.  Documented
here for the next round.

**B1 (f16 activations) — partially scaffolded, deferred:**
- Storage already worked from Phase 0 (load logic converts q8_0 → f16
  correctly in f16 mode).
- Lifting the rejection at load time made compute reach the graph
  stage, then fail at `ggml-metal-ops.cpp:2818` (`ggml_metal_op_bin`'s
  assertion that both srcs are f32).  A non-f32 tensor is flowing into
  a `ggml_add` / `ggml_mul` somewhere in the graph — likely an
  auto-fused add after a matmul where ggml-metal picks the matmul
  output type as f16 instead of f32.
- The cleanup pass needed (audit every binary op's input types and
  force-cast where required) is the same kind of work B2 does
  comprehensively for activation layout.  Pair them in a "graph-wide
  type/layout consistency pass" PR.

**B2 (activation layout permutation) — fully scoped, deferred:**
The 24 `cont(transpose(activation))` calls per per-step graph (3 per
QKV in 8 attention sites = 24, plus the post-attn out projection
transpose) come from converting matmul output `[T, A]` into
`[A, L]` for rope + flash_attn.  Eliminating them requires:

1. **Matmul output layout flip** — output `[A=OC, T]` directly via
   `ggml_mul_mat(pretransposed_w_[IC,OC], activation_[IC,T])`.
   Requires the activation already in `[IC, T]` format — which
   requires every upstream op to produce `[IC, T]`.
2. **New `layer_norm_channel_[C,T]` Metal kernel** — the current
   fused kernel assumes `[T, C]` and dispatches one threadgroup per
   time step, threads stride over channels.  For `[C, T]` the
   threadgroup decomposition flips: one threadgroup per channel,
   threads stride over time, OR one threadgroup per time step with
   different stride math.  Roughly 4-8 hours of Metal kernel work.
3. **Audit every `ggml_add` / `ggml_mul` site** for broadcast
   compatibility under the new layout (most should work via
   `repeat_like`'s native broadcast, but every site needs a check).
4. **Verify rope still works on `[D, L, H]` view** of the new
   `[A, L]` activation (likely fine — rope's input is already
   width-major).

The unblocked A3 step 2 win (Metal dispatches
`kernel_mul_mm_q8_0_f32` natively) is what makes B2 worth the work.
Together they target ~25-30 ms of additional Metal speedup vs
current 166 ms.  Without A3 step 2, B2 alone delivers ~-3 to -5 ms
(eliminating the cont(transpose) dispatches), which is below the
maintenance cost of the kernel rewrite.

Realistic estimate: 3-5 focused days as a dedicated PR.  Worth doing
when the goal is "Metal beats CPU on M2" — which is currently still
12 ms away (Metal 166 / CPU 153).

**B3 (argument buffer reuse) — scoped, deferred:**
Metal's `MTLIndirectCommandBuffer` lets the host pre-encode a command
buffer once and bind new input arguments per call, eliminating the
per-call command-buffer encoding cost.  Equivalent to CUDA Graph
Capture.

Requires changes inside the ggml-metal backend (the `ggml_metal_op_*`
encode functions, the residency-set lifecycle).  Cross-cutting work
touching files outside `tts-cpp/cmake/vcpkg-overlay-ports/ggml/`'s
current patches — could grow the overlay considerably.

Realistic estimate: ~1 week including upstream-friendly design,
since the right shape of this change is "improve ggml-metal for all
users" not "patch ggml just for Supertonic."  Better as a contribution
to the ggml-org project than a Supertonic-private optimization.

### Closing the loop on Phase A+B follow-up

Cumulative Metal perf trajectory across this PR:
- Phase B baseline (correctness port):  **249.92 ms**
- Tier 2 final (4 fused kernels + pretranspose): **174.49 ms**
- Phase A+B follow-up (A1+A2 + A3 step 1):  **166.39 ms**

That's **-83 ms / -33% total** on Metal vs the starting baseline.
Real-time multiplier 12.82× → 19.34×.  CPU q8_0 still wins by 13 ms;
ONNX-CPU by 21 ms.  Closing those final gaps requires B2 + A3 step 2
as outlined above — substantial work, but the path is clear.

Parity vs CPU reference held at corr ≥ 0.998 / L∞ ≤ 0.05 throughout
every commit.  Multi-precision harness (`--precision f32|f16|q8_0`)
ready to validate B1 + A3 step 2 wins when they land.

### B2 partial landed (2026-05-11) — Metal vec_est beats CPU

Investigated a smaller-scope B2 implementation and found that the
"swap `ggml_mul_mat` arg order at Q/K/V projection sites" trick
captures most of B2's direct win without any layer_norm kernel
rewrite or full activation-layout permutation.

The mechanism: `conv1d_f32(im2col, kernel)` produces `[T, A]` (because
mul_mat(im2col_[IC,T], kernel_[IC,OC]) yields [T, OC]).  The Q/K/V
projection sites then have to `cont(transpose(q_tc))` to get the
`[A, L]` shape that rope + flash_attn want.  By calling
`mul_mat(kernel, im2col)` instead — kernel as src0 — the result
lands in `[A, T]` directly.  Both operands are still non-transposed
so the assertion passes.

Shipped as a new `dense_matmul_time_wt_pretransposed_ggml` helper.
Eight call sites updated: 4 text-attention Q/K/V/out + 4
style-attention Q/K/V/out across all per-step graph groups.  ~24
cont(transpose) dispatches × 5 CFM steps = ~120 ops eliminated
per synth.

Bench (Apple M2, 10 runs + 2 warmup):
- pre-B2 f32:    total 172.56 ms / vec_est 99.07 ms
- **B2 partial f32: total 160.88 ms / vec_est 91.61 ms**
- delta:         -11.68 ms total / -7.46 ms vec_est

**This is the first time Metal vec_est beats CPU baseline** (91.61
vs 94.86 ms).  Total Metal 160.88 ms now within 7 ms of CPU's
153.52 ms, and within 16 ms of ONNX's 144.89 ms.

Cumulative trajectory:
- Phase B baseline:   249.92 ms (12.8× real-time)
- Tier 2 final:       174.49 ms (18.4×)
- Phase A+B + B2 partial: **160.88 ms (19.9×)**  ←  -36% from start

**The A3 step 2 unlock (q8_0 matmul kernel dispatch) requires
pretransposing q8_0 weights at load time.** Attempted, but the
`ggml_reshape_3d(w_pre, 1, IC, OC)` call inside the helper produces
an invalid q8_0 tensor when ne[0]=1 (q8_0 requires 32-element
block alignment on the inner dim).  A clean q8_0 path needs either
a different reshape strategy (skip the K=1 conv1d framing entirely
and call `ggml_mul_mat(w_pre_q8, im2col_via_a_different_path)`),
or an in-graph `ggml_im2col` that accepts a 2D kernel directly.
Either is a focused half-day's work for ~10-20 ms more savings
(matmul kernel bandwidth).  Deferred to a separate session.

### Full B2 + vocoder CT landed (2026-05-12) — Metal fastest on every stage

Built on the B2-partial trick by parameterising every fused custom
Metal kernel on per-axis element strides (`sxt`, `sxc`, `syt`, `syc`)
so the same compiled kernel handles both `[T, C]` and `[C, T]`
activations.  ggml overlay-port bumped 12 → 13.  Added `_ct`
constructors for `layer_norm_channel`, `depthwise_1d`, `pw2_residual`,
`bias_gelu`, `edge_pad_1d`.

In `supertonic_vector_estimator.cpp`: new `vector_convnext_ggml_ct`
runs the full ConvNeXt block on `[C, T]` activations.  Pointwise
K=1 Conv1d becomes a direct `ggml_mul_mat(w[IC,OC], x[IC,T])` (no
im2col, no transpose).  All 16 ConvNeXt blocks in the per-step
graph (prologue × 4 + 3 group_prep × 4 + tail × 4) wrap a single
entry permute and a single exit permute around the chain.

In `supertonic_vocoder.cpp`: same pattern for the 10-block vocoder
ConvNeXt chain.  Vocoder differences vs vector_estimator: (1)
depthwise is causal (left-only pad), no `_ct` causal kernel yet —
stays on `[T, C]` with two intra-block permutes; (2) gamma is
scalar `[1]`, so the `pw2_residual_ct` fused op doesn't fit, keep
unfused `mul(scalar gamma) + add(residual)` tail; (3) `norm_g` /
`norm_b` ship as `[1, C]` — same flatten-with-`ggml_reshape_1d`
quirk as `.gamma` in vector_estimator.

Discovered along the way: the legacy `pw2_residual_ggml` wrapper's
`gamma->ne[0] == x->ne[1]` gate was silently rejecting the fused
path for ConvNeXt all along (GGUF ships `.gamma` as `[1, C, 1, 1]`
not `[C]`).  The `_ct` wrapper flattens it once with
`ggml_reshape_1d`, so this is the first time the fused
`pw2_residual` op actually runs on the ConvNeXt residual.

Bench (Apple M2, q8_0 GGUF, 4 threads, 5-step CFM, 5 runs + 1 warmup,
all four backends benched in sequence on the same machine state):

| Stage (ms median)            | **ggml Metal** | ggml CPU | ONNX CPU | ONNX CoreML |
|------------------------------|---------------:|---------:|---------:|------------:|
| preprocess                   |          0.02 |     0.01 |     0.05 |        0.05 |
| duration                     |          3.27 |     1.49 |     1.26 |        8.17 |
| text_encoder                 |         12.11 |    11.70 |     8.22 |       16.26 |
| **vector_estimator** (5 step)|     **57.87** |    90.36 |    77.04 |      177.89 |
| **vocoder**                  |     **17.11** |    39.38 |    49.55 |       50.29 |
| **total**                    |     **91.37** |   142.92 |   136.32 |      255.90 |
| RTF (lower is faster)        |     **0.029** |    0.045 |    0.043 |       0.080 |
| **real-time multiplier**     |     **35.1×** |   22.4×  |   23.5×  |       12.5× |

Cumulative trajectory:
- Phase B baseline:        249.92 ms (12.8× real-time)
- Tier 2 final:            174.49 ms (18.4×)
- Phase A+B + B2 partial:  160.88 ms (19.9×)
- **Full B2 + vocoder CT: 91.37 ms (35.1×)**  ← −63% from Phase B start

Overrides: `SUPERTONIC_DISABLE_CT_CONVNEXT=1` (vector_estimator),
`SUPERTONIC_DISABLE_CT_VOCODER=1` (vocoder).

Open follow-ups (small ROI, separate PR):
- Causal-pad mode on `depthwise_1d_ct` → single chain-level
  permute for the vocoder (currently 2 intra-block permutes per
  block).  Projected -1 to -3 ms vocoder.
- B1 — f16 activations end-to-end.  Storage loads today;
  compute hits `ggml_metal_op_bin`'s f32 assertion.  Needs a
  graph-wide binary-op type cleanup.
- B3 — argument buffer reuse via `MTLIndirectCommandBuffer`.
  Better as an upstream ggml-metal contribution than a
  Supertonic-private patch.

### Out of scope for this baseline

- CUDA/Vulkan paths (host is Apple silicon; address Metal first).
- Multilingual / non-English voice perf — voice-agnostic.

### Distribution

- Publish generated GGUFs externally if reviewers/users should avoid local
  conversion:
  - GitHub release asset
  - Hugging Face
  - S3/R2/internal artifact storage
- Keep the repo itself model-file-free.

---

## Useful Commands

```bash
# Build Supertonic targets.
cmake --build build --target tts-cli supertonic-cli supertonic-bench test-supertonic-pipeline

# Create local Supertonic 2 GGUF.
bash scripts/setup-supertonic2.sh

# Synthesize with Supertonic 2.
./build/tts-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --threads 4 \
  --out /tmp/supertonic2.wav

# Benchmark GGML.
./build/supertonic-bench \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --threads 4 --runs 3 --warmup 1 \
  --json-out artifacts/supertonic-thread-matrix/ggml-quick-t4.json

# Benchmark ONNX Runtime CPU.
python scripts/bench-supertonic-onnx.py \
  --onnx-dir /path/to/supertonic-pytorch/onnx_models/onnx \
  --assets-dir /path/to/supertonic-pytorch/assets \
  --voice-style /path/to/supertonic-pytorch/assets/voice_styles/F1.json \
  --text "The quick brown fox jumps over the lazy dog." \
  --lang en --language-wrap-mode open_close \
  --steps 5 --speed 1.05 --threads 4 --runs 3 --warmup 1 \
  --providers CPUExecutionProvider \
  --json-out artifacts/supertonic-thread-matrix/onnx-quick-t4.json
```
