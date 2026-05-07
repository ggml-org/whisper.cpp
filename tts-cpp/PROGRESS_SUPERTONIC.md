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

## Remaining Work

### Runtime and performance

- Investigate vector 3/4-thread variance.
- Consider a fused text relpos attention op only if profiling shows text is the
  next hard blocker.
- Add quantized Supertonic GGUF support once graph paths are ready for f16/q8.
- Evaluate GPU backends after CPU graph structure is fully stable.
- Add CI coverage for converter help/setup syntax and portable Supertonic build
  targets.

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
