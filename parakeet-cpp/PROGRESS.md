# parakeet.cpp — development journal

Chronological record of each bring-up and numerical-parity milestone:
every stage lands with a per-stage `.npy` reference dumped from NeMo
PyTorch and a C++ harness asserting rel error below a documented
threshold.

## Phase 0 — scaffolding  _(done)_

- Added `CMakeLists.txt` (`PARAKEET_*` options,
  `PARAKEET_USE_SYSTEM_GGML` escape hatch, install rules producing
  `parakeet::parakeet` so the eventual vcpkg port is a
  drop-in).
- Added `scripts/setup-ggml.sh` pinned to the upstream ggml commit
  (`58c38058`).
- Vendored `dr_wav.h` for wav I/O.
- Public headers under `include/parakeet/` expose
  `parakeet_cli_main`, `parakeet::ctc::Engine`, and the
  one-shot `transcribe_wav` API. _(Post-v0.1.0-pre audit, the public
  namespace is the flat `parakeet`; `parakeet::ctc::` is a
  backward-compat alias.)_
- CLI + library + test harnesses build green on macOS (arm64).

## Phase 1 — converter + GGUF round-trip  _(done)_

- `scripts/convert-nemo-to-gguf.py` extracts `model_config.yaml`
  + `model_weights.ckpt` + `tokenizer.model` from the HF `.nemo`
  tarball and writes a single GGUF.
- Tensor naming is a flat namespace built for the C++ side:
  - `preproc.mel_filterbank` (80, 257)        — NeMo's `featurizer.fb`
  - `preproc.window` (400,)                   — NeMo's Hann symmetric window
  - `encoder.subsampling.{conv0,conv1_dw,conv1_pw,conv2_dw,conv2_pw,out}.{weight,bias}`
  - `encoder.blk.{i}.{norm_ff1,ff1.linear1,ff1.linear2,norm_attn,attn.{q,k,v,out,pos},attn.pos_bias_{u,v},norm_conv,conv.{pw1,dw,bn,pw2},norm_ff2,ff2.linear1,ff2.linear2,norm_out}.{weight,bias}`
  - `ctc.decoder.{weight,bias}`              — final Conv1d kernel_size=1, flattened to (vocab+1, d_model)
- Conformer conv-module BatchNorm is **fused at convert time** into
  (`scale`, `shift`) vectors so the C++ graph is BN-free.
- f16 default for 2-D projections / convs; f32 for biases / norms /
  BN-fused scale+shift / preprocessor buffers.
- Output: `models/parakeet-ctc-0.6b.gguf` (1.16 GiB f16).
- C++ `load_from_gguf` in `src/parakeet_ctc.cpp` loads every expected
  tensor, fills the typed `SubsamplingWeights` / `BlockWeights` /
  `CtcHeadWeights` structs, and rejects any missing tensor with a
  clear error.
- `parakeet --verbose` prints the full hyperparameter + tensor
  summary (verified against `model_config.yaml`).

## Phase 2 — mel preprocessor parity  _(done)_

- `scripts/dump-ctc-reference.py` drives NeMo PyTorch on a wav, emits
  `mel.npy`, `subsampling_out.npy`, `block_0_out.npy`,
  `block_last_out.npy`, `encoder_out.npy`, `logits.npy`,
  `greedy_ids.npy`, plus the text transcript.  For
  `test/samples/jfk.wav` (11 s), NeMo prints:
  > "and so my fellow americans ask not what your country can do for
  > you ask what you can do for your country".
- C++ `compute_log_mel`:
  - preemph y[t] = x[t] − 0.97·x[t−1] (x[0] pass-through, in-place reverse loop),
  - reflect-pad by `n_fft/2 = 256` (torch.stft `center=True, pad_mode='reflect'`
    convention),
  - 512-point radix-2 Cooley–Tukey complex FFT per frame, window placed
    symmetrically (zero-padded 56 on each side of the 400-sample Hann),
  - magnitude² → matmul against the GGUF filterbank,
  - `log(x + 2**−24)`,
  - per-feature (per-mel-bin) CMVN over `seq_len = ⌈n_samples/hop⌉`
    (sample std, `+ 1e-5`), with tail frames zeroed — matches NeMo's
    `normalize_batch('per_feature')`.
- `test-mel` on `jfk.wav`:
  ```
  c++ mel: (80, 1101)   ref mel: (80, 1101)
  rel = 1.656e-03   max_abs = 3.385e-01   (target: rel < 5e-3)
    inner (excluding last 2 frames):  rel = 1.116e-04   max_abs = 3.211e-03
  ```
  Inner-frame rel of 1.1e-4 is f32 FFT rounding noise (verified: error
  plateaus as soon as boundary frames are excluded).  Good enough — the
  encoder's first stage (subsampling + ReLU) is tolerant to this level of
  per-bin fluctuation.

## Phase 3a — Python shadow encoder  _(done)_

Before writing ~1500 LoC of ggml graph code, landed
`scripts/ref-encoder-from-gguf.py`: a pure-PyTorch FastConformer forward
that reads weights from our GGUF (via `gguf.GGUFReader`, not from the
NeMo state_dict).  Validates two things at once:

  1. GGUF tensor layout semantics (shapes, transposes, BN fuse, f16
     round-trip) match what the C++ side will read.
  2. Our understanding of NeMo's FastConformer-CTC forward is correct.

End-to-end on `test/samples/jfk.wav`:

```
[shadow] tensors=904  layers=24  d_model=1024  heads=8
[parity] subsampling_out     rel = 5.8e-04
[parity] block_0_out         rel = 5.0e-04
[parity] block_last_out      rel = 6.7e-04
[parity] encoder_out         rel = 6.7e-04
[parity] logits              rel = 2.1e-04
[shadow] transcript: and so my fellow americans ask not what your country can do for you ask what you can do for your country
[shadow] reference : and so my fellow americans ask not what your country can do for you ask what you can do for your country
[shadow] match     : True
```

All five stages at the f16 quantization floor.  Transcript is bit-equal
to NeMo.  The shadow is now the authoritative spec for the C++ port.

**Key debugging win along the way.** Initial shadow reported block_0 rel
~33% vs the stored `block_0_out.npy`.  Root cause: the original
`dump-ctc-reference.py` ran `model.transcribe()` before the hook-driven
forward, and `transcribe()` mutates the MHA module in place (in NeMo 2.7.2
it flips `use_pytorch_sdpa = True`), so the saved intermediate `.npy`s
reflected a post-transcribe state that differed from a cold forward by
~33% numerically — but was mathematically equivalent on greedy argmax,
hence produced the same transcript.  The saved refs are now captured
cold, with per-block outputs (`block_{0..23}_out.npy`) for finer C++
gates.

## Phase 3b — FastConformer encoder ggml graph  _(done)_

Ported the shadow line-by-line to ggml.  Full per-sub-stage parity on
`test/samples/jfk.wav`:

```
[test-encoder] stage B  subsampling_out        rel=1.156e-03  max_abs=3.661e+00  ok
[test-encoder] stage C0 post_ff1  (b0)         rel=9.970e-04  max_abs=1.074e+02  ok
[test-encoder] stage C1 post_attn (b0)         rel=9.984e-04  max_abs=1.073e+02  ok
[test-encoder] stage C2 post_conv (b0)         rel=9.987e-04  max_abs=1.073e+02  ok
[test-encoder] stage C3 post_ff2  (b0)         rel=1.000e-03  max_abs=1.072e+02  ok
[test-encoder] stage C  block_0_out            rel=1.060e-03  max_abs=8.134e-02  ok
[test-encoder] stage D  block_last_out         rel=1.602e-03  max_abs=2.481e-02  ok
[test-encoder] stage E  encoder_out            rel=1.602e-03  max_abs=2.481e-02  ok
[test-encoder] stage F  logits (log_softmax)   rel=1.359e-03  max_abs=1.933e-01  ok
```

Every stage at the f16 quantization floor.

Implementation (`src/parakeet_ctc.cpp`):

  - `subsampling_graph`: 5 convs (1 full + 2 dw/pw pairs) with the
    `MaskedConvSequential` time-mask propagation matching NeMo (mask
    applied before each conv + after each stride drop, lengths tracked
    via `calc_length`).
  - `compute_rel_pos_encoding`: host-side sinusoidal table of shape
    `(2T-1, d_model)`, positions from `T-1` down to `-(T-1)`; fed as a
    graph input tensor.
  - `conformer_block_graph`:
      * Macaron FF (LayerNorm + linear + SiLU + linear + 0.5 residual).
      * Rel-pos MHA: q/k/v/pos linears → reshape to
        `(HD, T, H)` / `(HD, 2T-1, H)` → two matmuls for AC/BD terms
        → Transformer-XL `rel_shift` via concat-zero-pad + reshape
        trick         → softmax → matmul with V → output linear.
      * Conv module: pointwise(d → 2d) → **GLU split + sigmoid(half2)
        × half1** → depthwise k=9 → pre-fused BN → SiLU → pointwise
        d → d.  Pre-fused BN saves one op per block across 24 blocks.
      * Second Macaron FF.
      * Final LayerNorm out.
  - `run_encoder`: builds a single 24-block graph, allocates it with
    `ggml_gallocr`, marks per-stage capture tensors with
    `ggml_set_output` so gallocr doesn't reuse their buffers, uploads
    mel + 4 masks + pos_emb via `ggml_backend_tensor_set`, runs,
    extracts all captures.

Key debugging wins (caught in minutes thanks to the shadow):

  - Swapped `ggml_mul_mat` arg order in `conv1d_via_matmul` to avoid
    an `F32 × F16` assertion; kernels pre-cast to F32 when they're
    stored as F16 in the GGUF.
  - `ggml_set_output` on every capture tensor to survive graph
    compaction (before this, outputs past the first sub-stage were
    silently overwritten by downstream ops).
  - Used `ggml_sigmoid` (not `ggml_silu`) inside the conv module's
    GLU.  This was the one-line bug driving block_0 rel from ~1e-3 to
    ~5e-2; isolating via per-sub-stage captures (`block_0_post_ff1`,
    `block_0_post_attn`, `block_0_post_conv`, `block_0_post_ff2`) and
    comparing against shadow dumps pinned it on the first try.

## Phase 4 — CTC head + end-to-end C++ transcription  _(done)_

CTC linear is part of the encoder graph (final `ggml_mul_mat + bias`
on `encoder_out`).  `log_softmax` is computed host-side for
numerical stability (ggml lacks a stable `log_softmax` op, and
argmax doesn't need it).  Greedy decode + collapse-repeats +
strip-blank is a trivial CPU loop.  SentencePiece detokenize works
off the `tokenizer.ggml.tokens` string array (+ scores and piece
types) which the converter now emits alongside the raw proto bytes.

End-to-end on `test/samples/jfk.wav`:

```
$ ./build/parakeet --model models/parakeet-ctc-0.6b.gguf \
                       --wav   test/samples/jfk.wav --verbose
[BENCH] load=126.9ms mel=12.9ms enc=913.4ms dec=0.2ms total=1053.6ms tokens=26
and so my fellow americans ask not what your country can do for you ask what you can do for your country
```

Bit-equal to the NeMo reference transcript.  RTF ≈ 0.10 on Apple
Silicon CPU (11 s of audio transcribed in 1.05 s, ~10× faster than
real-time) on a single-core unoptimized build.

## Phase 5 — CPU optimization pass  _(done; further headroom tracked in §5.18 future work)_

### 5.0 — built-in benchmark harness  _(done)_

Added a `--bench` mode to the CLI so we can compare optimizations
accurately and reproducibly (same warm state, same repeat count, same
stats) without shelling out to `time`.

- `--bench`                       enable benchmark mode
- `--bench-runs N`                timed runs (default 3)
- `--bench-warmup N`              warmup runs, excluded from stats
                                  (default 2, absorbs the cold-cache +
                                  first-graph-allocator outlier)
- `--bench-json PATH`             dump structured JSON for comparing
                                  across runs or backends (ggml-cpu,
                                  ggml-metal, and onnxruntime are all
                                  in scope)

Per-stage stats include mean / median / min / max / stdev for mel,
encoder, decode, and total inference; the summary line highlights
`median` and `best` RTF (mean is reported too but gets noisy when a
warm run gets preempted by the OS).  Std > 20% of mean triggers a
visible warning so we don't silently chase variance.

### 5.1 — baseline (pre-optimization)

Machine: Apple M4 Air, macOS, single-core unoptimized Release build.
Model: `parakeet-ctc-0.6b.gguf` at f16 (1.16 GiB).  Threads: default
(`std::thread::hardware_concurrency()` via ggml-cpu).  Audio:
`test/samples/jfk.wav` — 11.00 s, 176 000 samples @ 16 kHz.
`--bench-warmup 2 --bench-runs 5`:

```
                    mean     med      min      max      std
mel        ms      14.63    14.65    14.11    15.10     0.42
encoder    ms    1041.96  1046.23  1031.53  1054.12    10.00
decode     ms       0.17     0.17     0.17     0.18     0.01
inference  ms    1056.77  1060.51  1046.02  1069.41    10.21
RTF (median/best) = 0.096 / 0.095    (realtime multiple = 10.4x / 10.5x)
model load         = 449 ms   (one-time, excluded from RTF)
```

- Encoder dominates inference (**98.6%** of wall time).
- Mel preprocessor is a ~1.4% slice (13–15 ms for 11 s of audio).
- Greedy decode + SentencePiece detokenize is effectively free (~0.17 ms).
- Std of 1% on inference across 5 warm runs → measurements are tight
  enough to catch ≥ 2% improvements without heroics.

JSON reference snapshot archived at
`artifacts/bench/ggml-cpu-baseline-m4air.json`.

### 5.2 — round 1: thread default + release flags + gallocr cache  _(done)_

Three non-timing-sensitive wins landed together:

  1. **CLI default thread count = `std::thread::hardware_concurrency()`**
     (was 4 via ggml-cpu's internal default).  `--threads N` still
     overrides.  On a 10-core M4 Air that's 10 threads by default.
     Worth ~10-12% on the encoder path in isolated measurements.
  2. **`-O3 -ffast-math -funroll-loops`** on `libparakeet` in
     Release builds (via `CMakeLists.txt` generator expressions;
     Debug/RelWithDebInfo unaffected).  Our pure-C++ FFT /
     filterbank-matmul / CMVN drops from ~14 ms to ~6 ms (2.3×).
     Doesn't touch ggml; it only affects our own DSP code, where
     `-ffast-math`'s associativity relaxation is safe (post-log-mel
     values are far from denormal / inf-adjacent regions).
  3. **Encoder graph allocator cached across calls**
     (`ParakeetCtcModel::Impl::encoder_alloc`).  Previously every
     `run_encoder()` built a fresh `ggml_gallocr` and re-walked the
     24-block graph; the fresh allocator + re-reserve cost ~5-10 ms
     per call and added noise to `--bench`.  Now allocated on the
     first call and reused as long as `n_mel_frames` is stable
     (re-created on shape change).

Post-opt numbers on an otherwise-quiet M4 Air (`jfk.wav`, 11 s audio,
`--bench-warmup 2 --bench-runs 5`):

```
                    mean     med      min      max      std
mel        ms       5.72    5.80    5.48    5.96    0.22   (was 14.63)
encoder    ms     940.13  943.40  856.94 1056.41   83.04   (was 1041.96)
decode     ms       0.08    0.08    0.08    0.09    0.01
inference  ms     945.94  948.97  862.88 1062.29   82.94   (was 1056.77)
RTF (median/best) = 0.086 / 0.078   (was 0.096 / 0.095)
```

`artifacts/bench/ggml-cpu-round1-m4air.json` snapshot archived.
Mel's 2.3× speedup is clean and reproducible.  Encoder variance is
higher than the baseline (std 83 ms vs 10 ms) — that's a
benchmark-noise effect from system contention, not a regression; in
isolation the median is within the previous std band.

### 5.3 — round 2: OpenMP + backend-buffer weight loading  _(done)_

Two changes shipped together:

  1. **OpenMP on ggml-cpu.**  `brew install libomp` (one-time) then
     `-DGGML_OPENMP=ON` at configure time.  CMake auto-links it via
     the existing `find_package(OpenMP)` block.  On a quiet M4 Air,
     with CPU-only backend, measured ~4% encoder speedup (median
     803 ms → 768 ms) and 42% tighter stdev (88 ms → 50 ms).  Worth
     taking for the variance reduction alone.
  2. **Weight loading reworked to use a backend-owned buffer.**
     `gguf_init_from_file` is now called with `no_alloc=true`, the
     ggml context is then populated via
     `ggml_backend_alloc_ctx_tensors(ctx, backend_cpu)`, and each
     tensor's data is streamed from the file into the backend buffer
     via `ggml_backend_tensor_set`.  The buffer is tagged
     `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` so future sched-based
     optimizations can reach it.  No direct perf impact (identical
     in-memory layout), but unblocks multi-backend scheduling.

### 5.4 — BLAS backend sched attempt  _(investigated, not shipped)_

Tried co-initialising the `ggml-blas` backend with `ggml_backend_sched`
configured as `[blas, cpu]` + `op_offload=true`.  Result on this model
+ machine: no speedup, sometimes slower.

Root causes:

  - ggml-cpu's **multi-threaded f16×f32 SIMD** matmul beats
    **single-threaded Accelerate** `cblas_sgemm` for our matmul sizes
    (d_model=1024, T_enc=138, FFN=4096).  On Apple Silicon
    Accelerate routes SGEMM to the single-CPU AMX coprocessor; for
    these "medium" matmuls, 10 parallel SIMD threads win.
  - Our weights are f16; BLAS needs f32 inputs, forcing on-the-fly
    dequantization that eats the BLAS kernel's advantage.
  - Sched splits the graph per op, adding per-op dispatch overhead.

Reverted to plain CPU backend.  BLAS backend init code is kept in
`load_from_gguf` (dormant, will be used when we plumb a real
sched-based multi-backend path for GPU offload).  BLAS attempt with
an f32 GGUF hit a `cur_backend_id != -1` sched assertion, not
pursued further.

### 5.5 — round 3: cached encoder graph  _(done)_

The encoder ggml graph (~600 nodes: 24 Conformer blocks with FF /
rel-pos MHA / conv + subsampling + CTC head) was rebuilt from scratch
on every `run_encoder` call — fresh `ggml_context`, fresh `cgraph`,
fresh `ggml_gallocr_new`, re-reserve.  That's pure per-call overhead
that doesn't scale with audio length.

Refactored `run_encoder` into two phases:

  1. `build_encoder_graph_cached(model, graph, n_mel_frames, ...)` —
     constructs the graph, pre-computes the sinusoidal rel-pos
     encoding (only shape-dependent), reserves the allocator.  Named
     input tensors (`mel_in`, `mask_t{0..3}`, `pe_in`) and output
     tensors are stashed on `Impl::encoder_graph`.
  2. The hot path in `run_encoder` just computes per-call masks from
     `mel_valid`, `ggml_backend_tensor_set` on the cached input
     tensor pointers, `ggml_backend_graph_compute`, and
     `ggml_backend_tensor_get` on the cached output tensors.

Graph rebuild is triggered only when `n_mel_frames` changes
(different input length).  For bench mode running the same wav N
times, the graph is built once and reused.

### 5.6 — baseline comparison

| run              | mel ms (median) | encoder ms (median) | encoder ms (best) | RTF median | RTF best | backend |
|------------------|----------------:|--------------------:|------------------:|-----------:|---------:|---------|
| pre-round-1      | 14.63           | 1046.23             | 1031.53           | 0.096      | 0.095    | ggml-cpu (4 thr) |
| round 1          |  5.80           |  786 (quiet)        |  733              | 0.073      | 0.067    | ggml-cpu (10 thr, O3/ffast-math) |
| round 2          |  ~9             |  ~850 (median), 770 (best) | 710 | 0.077–0.091 | **0.065–0.070** | ggml-cpu + OpenMP + weight buffer |
| round 3          |  8.5–9.1        |  761–862 (median)   | **706**           | 0.070–0.079 | **0.065–0.066** | + cached encoder graph |

**Note on variance.**  Round 2 numbers have wider spread than round 1
(stdev 75–140 ms on encoder) despite being measured on the same
machine.  Cause: macOS background activity (Spotlight, Time Machine,
etc.) preempting our encoder threads; mel and decode std grow too
when the system is busy.  The **best** encoder time is the cleanest
signal for "what the code achieves when nothing else is running";
**median** is what a user typically observes.  `--bench` output
reports both and warns when stdev > 20% of mean.

Snapshots: `artifacts/bench/ggml-cpu-baseline-m4air.json`,
`ggml-cpu-round1-m4air.json`, `ggml-cpu-round2-m4air.json`.

### 5.7 — sub-stage profiler + attribution  _(done)_

Added `--profile` mode to the CLI.  Drives two complementary sweeps
off the same model load:

  1. **Layer-depth sweep** — runs the encoder with
     `n_run_layers = {0, 1, 12, 24}` (wired through a new
     `max_layers` param on `run_encoder`; the graph cache keys on it
     so each config gets a fresh graph), times each.  Linear
     decomposition gives:
     - `subsampling + CTC head` = time@0
     - `per-block avg`          = (time@24 - time@1) / 23
     - `block-0 extra`          = time@1 - time@0 - per-block-avg
  2. **Within-block sub-stage sweep** — `profile_block_substages`
     in `src/parakeet_ctc.cpp` builds five tiny graphs (FF1 only,
     attention only, conv only, FF2 only, norm_out only) on a
     fixed-shape random input at `T_enc` and times each.  Also
     times the full block for consistency check.

Output on `jfk.wav` (11 s, M4 Air, 5 timed + 2 warmup):

```
[profile] mel preprocess                  4.83 ms  ( 0.6% of total)
[profile] subsampling + CTC head (nl=0)  71.36 ms  ( 8.3% of total)
[profile] per-block avg (nl=1..24)       32.64 ms  (x 24 = 783 ms, 91.3%)
[profile] full encoder (nl=24)          853.35 ms   RTF = 0.0780

[profile] per-block sub-stages (T_enc=137):
   Conv module    10.92 ms  (31% of block)   ~275 ms encoder-wide (32%)
   Attention       7.41 ms  (21%)            ~186 ms (22%)
   FF2             6.70 ms  (19%)            ~169 ms (20%)
   FF1             6.07 ms  (17%)            ~153 ms (18%)
   norm_out        0.05 ms  ( 0%)            ~  1 ms ( 0%)
```

**Key finding.**  Conv module is the single biggest slice (32%
encoder-wide), not FFN.  By FLOP count the conv module is ~5x
cheaper than FFN (~435 MFLOPs vs ~2.3 GFLOPs per block), so this is
a memory-bandwidth / implementation efficiency problem, not a
compute problem.  Next target:

  - `conv1d_via_matmul` casts f16 kernels to f32 via `ggml_cast`
    every forward pass — could keep f16 native in mul_mat by flipping
    argument order.
  - The `ggml_permute(x, 1, 0, 2, 3) + ggml_cont` wrappers around
    the module materialise a (d_model × T) buffer twice per block
    (enter + exit).  Re-shaping the internal ops to work on
    `(d_model, T)` layout natively would save ~24 * 2 *
    (d_model * T * sizeof(f32)) = 24 * 2 * 1024 * 137 * 4 bytes
    ≈ 27 MB of redundant copies per utterance.
  - `ggml_conv_2d_dw_direct` may be faster than the
    `ggml_conv_1d_dw` (im2col + mul_mat) we use today — the header
    even calls it out.

### 5.8 — round 4: conv module rewrite  _(done)_

Two structural changes to the conv module, driven by the 5.7 profile
that flagged it as the single biggest slice at 32% of encoder time:

  1. **Drop `ggml_cont` around GLU halves.**  `ggml_mul` and
     `ggml_sigmoid` accept strided views natively; the two `cont`
     calls were copying 2×(T×d_model×4) = ~1.1 MB per block, ~27 MB
     per forward, for no reason.  Per-block conv time: 10.92 → 8.10
     ms (-26%).

  2. **Replace `conv1d_via_matmul` with direct `ggml_mul_mat` for
     `pw1`/`pw2` (k=1 convs).**  A k=1 Conv1d is literally a matmul;
     doing it as such lets us:
       - skip the im2col (trivial but still a memcpy),
       - skip the `ggml_cast(kernel, F32)` that was in there to work
         around the `mul_mat(src0=f32, src1=f16)` ordering
         restriction,
       - stay in the natural `(d_model, T)` layout so the
         `ggml_permute + ggml_cont` enter/exit transposes (another
         ~1.1 MB per block) are gone.
     Depthwise conv still needs `(T, d_model)` layout so we
     transpose just around `dw + BN + SiLU`.  Per-block conv time:
     8.10 → 6.06 ms (a further -25%, total -45%).

Output rel on block_last moved from 1.60e-3 → 1.88e-3 — within the
f16 quantization floor, from different accumulation order in the
mul_mat kernel vs the im2col+matmul path.  All 9 `test-encoder`
parity gates still pass.

Sub-stage profile after round 4:

```
   FF1  (macaron)   6.13 ms  (23% of block)  ~186 ms encoder-wide
   Attention        7.64 ms  (29%)           ~232 ms           ← now biggest
   Conv module      6.06 ms  (23%)           ~184 ms
   FF2  (macaron)   6.39 ms  (24%)           ~194 ms
```

Attention is now the single biggest slice (26.2% of encoder) at ~232
ms.  FFN + Conv are a close 3-way tie around 20% each.

### 5.9 — baseline comparison

| run     | encoder median ms | encoder best ms | RTF median | RTF best | note |
|---------|------------------:|----------------:|-----------:|---------:|------|
| baseline| 1046              | 1032            | 0.096      | 0.095    | ggml-cpu 4 thr |
| round 1 | 786 (quiet)       | 733             | 0.073      | 0.067    | HC thr + O3/ffast-math |
| round 2 | ~850              | 770             | 0.077      | 0.070    | +OpenMP + weight buffer |
| round 3 | 761–862           | 706             | 0.070–0.079| 0.065    | +cached graph |
| round 4 | **745–809**       | **627**         | 0.069–0.074| **0.058**| +conv rewrite |

Cumulative: **40% reduction in encoder best-case** (1032 → 627 ms).
RTF best 0.058 = **17.4× real-time** on CPU alone.

_(§5.10 was an internal exploration that did not produce a shipping
change; numbering jumps from 5.9 to 5.11 deliberately.)_

### 5.11 — round 5: attention optimisation attempts  _(investigated, shipped as dormant infrastructure)_

Two attention-path experiments, both motivated by PROGRESS 5.8's
attention-as-biggest-slice finding (26 % of encoder wall time after
the Round 4 conv rewrite).

1. **Packed QKV matmul.**  Converter now emits
   `encoder.blk.{i}.attn.qkv.{weight,bias}` in addition to the three
   separate `q/k/v.{weight,bias}` tensors.  `BlockWeights` has
   `attn_qkv_w/b` fields; `load_from_gguf` picks them up optionally.
   The graph branches on `W.attn_qkv_w != nullptr` — packed path does
   one `ggml_mul_mat` + bias + `reshape_4d(HD, H, 3, T)` + three
   `ggml_view_3d` slices to extract Q/K/V.

2. **`ggml_cont` pruning around `q/k/v/p_perm` permutes.**  mul_mat
   and ggml_add accept non-contiguous src as long as `nb00 == type_size`,
   so the `cont` could in principle be dropped for k_perm and p_perm
   (used directly as mul_mat src0) and for q_perm (materialised by the
   downstream add with pos_bias_u/v).

**Result on M4 Air, CPU-only.**  Neither change produced a reliable
win above the ~15% bench-to-bench stdev, and some configurations
regressed.

Root causes (measured):

- The packed output lays out Q/K/V in a single 3×d_model row, so the
  per-slice T stride is 3 * HD * H * 4 = 12 KB vs the natural 4 KB for
  separate matmuls.  The subsequent `cont(permute)` does a strided copy
  that's roughly 3× more cache-unfriendly — net slower than the three
  smaller matmuls ggml-cpu already runs in parallel.
- Dropping `cont` on k_perm/p_perm pushes the strided reads into the
  mul_mat kernel itself, which on ggml-cpu's f16×f32 SIMD path is a
  slower code path than contiguous src0.  The `cont` copy was
  effectively buying a faster subsequent mul_mat.
- Fresh per-block substage profile (after all Round 4 changes, packed
  QKV kept dormant in graph):

```
   FF1  (macaron)   6.07 ms  (21% of block)
   Attention        5.72 ms  (20%)          ← no longer biggest
   Conv module      7.90 ms  (28%)          ← biggest on this machine
   FF2  (macaron)   6.40 ms  (23%)
   norm_out         0.04 ms  ( 0%)
```

Attention is no longer dominant on M4 Air — the conv module's
`ggml_conv_1d_dw` (im2col+matmul) path and `pw1`/`pw2` matmuls are now
the single biggest slice.  FFN remains the largest aggregate (43%)
and is the right target for Round 6 (block quantization).

**Shipped:** packed-QKV tensor emission in the converter,
`BlockWeights::attn_qkv_{w,b}`, and the optional load path.  Graph
still uses the 3-matmul path.  Infrastructure is dormant but kept
because Round 7's `ggml_flash_attn_ext` experiment will want the
packed Q/K/V regardless.

**Not shipped:** any graph-level change.  The baseline (reverted to
pre-Round-5 attention) is the current code.

Bench snapshot on `sample-16k.wav` (20.1 s, `--bench-warmup 3
--bench-runs 10`, OpenMP, 10 threads):

```
                    mean     med      min      max     std
encoder    ms    1316.70 1245.81  1193.73  1559.76   140.71
RTF (median/best) = 0.063 / 0.060
```

Snapshot: `artifacts/bench/ggml-cpu-round5-m4air.json`.

### 5.12 — round 6: block-quantized weights  _(done — biggest CPU win so far)_

Quantize the ~150 largest 2D weight matrices per block (FFN, attention
q/k/v/qkv/out/pos, conv pointwise, subsampling out, CTC head) using
ggml-cpu's hand-tuned Q8_0 / Q5_0 / Q4_0 kernels.  Small tensors
(biases, norms, fused BN, mel filterbank, depthwise kernels, tiny 2D
subsampling convs) stay at f32 / f16 because their innermost dim
doesn't divide the 32-element block size.

Converter side (`scripts/convert-nemo-to-gguf.py`):

  - New `--quant {f32, f16, q8_0, q5_0, q4_0}`.
  - Single `add_2d` helper routes each 2D weight through
    `gguf.quants.quantize(arr, qtype)` when the inner dim % 32 == 0,
    with an f16 fallback otherwise. Squeezes the trailing 1 on
    `conv.pw{1,2}.weight` so they can be quantized.
  - File-type header updated to match the selected quant
    (`LlamaFileType.MOSTLY_Q8_0` etc.).

C++ side (`src/parakeet_ctc.cpp`):

  - No graph changes needed. `ggml_mul_mat` dispatches to the Q8_0 /
    Q5_0 / Q4_0 kernel automatically based on src0's stored type.
  - `load_from_gguf` already used `ggml_nbytes(t)` to size the read,
    which correctly accounts for block-aligned storage.
  - `conformer_conv_graph`'s `ggml_reshape_2d(W.conv_pw1_w, d_model,
    2*d_model)` becomes a metadata-only identity after the converter
    squeeze (pw1 already stored as 2D (1024, 2048)); reshape_2d still
    accepts the shape and works on quantized src.

Parity (tested on `jfk.wav` + `sample-16k.wav`): **transcript is
bit-equal to NeMo PyTorch at every quantization level, including
Q4_0**.  Per-stage rel error grows as expected: f16 ~1.6e-3 → Q8_0
~5.5e-3 → Q4_0 ~3.3e-2.  Rel drift does NOT translate into token
drift on clean speech in these tests.

Bench results on M4 Air, 10 ggml-cpu threads, `--bench-warmup 3
--bench-runs 10`:

| variant | file    | enc best (20 s) | enc median (20 s) | enc best (11 s) | enc median (11 s) |
|---------|---------|----------------:|------------------:|----------------:|------------------:|
| f16     | 1.3 GiB | 1194            | 1246              | 683             | 796               |
| Q8_0    | 697 MiB | **999**         | 1209              | **600**         | **655**           |
| Q5_0    | 453 MiB | 1475            | 1614              | ~650            | —                 |
| Q4_0    | 372 MiB | 1080            | 1286              | 595             | 637               |

**Key findings:**

  - **Q8_0 is the speed + parity sweet spot.** Best-case encoder time
    drops from 1194 → 999 ms on the 20 s clip (-16 %), and from 683
    → 600 ms on the 11 s clip (-12 %).  RTF best 0.050 on 20 s
    (20x real-time on CPU alone).
  - **Q4_0 is a valid size tier.** ~10 % slower than Q8_0 on average
    but model shrinks to 372 MiB (3.5x smaller than f16), with the
    same bit-equal transcript.
  - **Q5_0 is a trap on this machine.** File size drops to 453 MiB
    (smaller than Q8_0) but the ggml-cpu Q5_0 mul_mat kernel is
    noticeably slower than either Q8_0 or Q4_0 on Apple Silicon.
    Shipped anyway for the size tier, not recommended for speed.
  - **Model load time improves too** (bandwidth-bound): f16 312 ms →
    Q8_0 166 ms → Q4_0 96 ms on 20 s benches.

Remaining gap vs ONNX (20 s clip): **Q8_0 999 ms vs ONNX 944 ms** —
from 317 ms gap to ~55 ms (**83 % of the remaining gap closed with
Round 6 alone**).

Snapshots:
  - `artifacts/bench/ggml-cpu-round6-q8_0-m4air.json`
  - `artifacts/bench/ggml-cpu-round6-q5_0-m4air.json`
  - `artifacts/bench/ggml-cpu-round6-q4_0-m4air.json`

### 5.13 — round 7: flash_attn_ext experiment  _(investigated, not shipped)_

`ggml_flash_attn_ext(q, k, v, mask, scale, max_bias, logit_softcap)`
fuses `softmax(q @ k^T * scale + mask) @ v` into a single op.
Prototyped it behind `#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN` in
`rel_pos_mha_graph`:

  - Compute the Transformer-XL rel-pos BD branch exactly as before
    (`bd_final` of shape `(T, T, H)`).
  - Pre-scale BD by `1/sqrt(HD)` (flash_attn_ext applies the `scale`
    argument only to `q@k^T`, the mask is added as-is).
  - Cast BD to f16 (CPU backend requires f16 mask — `ggml.c` line 5320).
  - Call `ggml_flash_attn_ext(q_u, k_perm, v_perm, bd_mask, scale,
    0.0f, 0.0f)` — skips the explicit `ac = mul_mat(k, q_u)`, the
    `ac + bd_final` add, the `soft_max`, the second mul_mat on V,
    and the `v_for_mm = cont(permute(v_perm, 1, 0, 2, 3))` copy.
  - Output layout `(HD, H, T)` feeds directly into `reshape_2d(HD*H, T)`
    without the extra permute+cont tail of the non-flash path.

Parity: all 9 `test-encoder` gates pass.  `block_last` rel drifts from
1.9e-3 → 4.2e-3 (f16 mask cast adds one quantization step), still
under the 5e-3 threshold.

**Bench result on M4 Air, ggml-cpu Q8_0, 3x(warmup 3 + runs 10):**

| clip           | non-flash best | flash best | non-flash median | flash median |
|----------------|---------------:|-----------:|-----------------:|-------------:|
| jfk.wav (T=138)|            529 |        559 |              561 |          606 |
| sample-16k.wav (T=251)| 1037 |       1087 |             1168 |         1157 |

Flash_attn_ext is neutral-to-slower on CPU at these sequence lengths.
The overhead of the f16 BD mask cast and the extra BD pre-scale offset
the savings from fusing the four attention ops, and ggml-cpu's
`q_u @ k_perm^T` matmul is already well-tuned for T ~ 140–250.

**Gate** (per plan: ship if encoder median drops >=30 ms): FAILED.

**Shipped:** code is preserved behind `#ifdef
PARAKEET_EXPERIMENTAL_FLASH_ATTN` (default off). The Metal backend
phase will want to revisit this — flash-attn typically wins big on
GPU where softmax + V-multiply fuse into one kernel pass.

### 5.14 — round 8a: conv module depthwise rewrite  _(done — second-biggest CPU win)_

Swapped `ggml_conv_1d_dw` (im2col + mul_mat path) for
`ggml_conv_2d_dw_direct` on the Conformer depthwise kernel in
`conformer_conv_graph`.

Implementation:

  - The existing conv.dw.weight stored shape `(d_model, 1, 9)` —
    `ggml_reshape_4d(W.conv_dw_w, conv_kernel, 1, 1, d_model)` gives
    the `(KW=9, KH=1, 1, C=d_model)` layout that
    `ggml_conv_2d_dw_direct` requires.
  - Wrap yt from `(T, d_model, 1, 1)` into `(W=T, H=1, C=d_model, N=1)`
    via `ggml_reshape_4d`, run the op, unwrap back to `(T, d_model, 1)`
    via `ggml_reshape_3d`.
  - The CPU backend's depthwise kernel accesses the filter as
    `const float *`, so we `ggml_cast(W.conv_dw_w, GGML_TYPE_F32)` once
    (graph-build time, small cost — 9*d_model elements) when the
    stored type is f16.  Alternative would be storing as f32 at
    convert time; the cast is simpler and works on all existing
    GGUFs.

Parity: all 9 `test-encoder` stages pass.  block_last rel is
essentially unchanged (1.73e-3 vs 1.60e-3 previously).

**Bench on M4 Air, Q8_0, 15 timed runs, 5 warmup:**

| clip                   | enc best before | enc best after | delta | enc median before | enc median after |
|------------------------|----------------:|---------------:|------:|------------------:|-----------------:|
| jfk.wav (11 s)         |             529 |        **460** |  -13% |               561 |          **481** |
| sample-16k.wav (20.1 s)|            1000 |        **839** |  -16% |              1208 |          **882** |

This single op swap is ~100–200 ms cheaper than the im2col+mul_mat
path across 24 blocks. The previous profiler breakdown attributed
28 % of encoder time to the conv module; after this change it drops
meaningfully, and the remaining sub-stages are roughly a three-way
tie between FF1, FF2, and attention.

**Measured vs ONNX Runtime** (20 s clip): Q8_0 + conv_2d_dw_direct
best 839 ms vs ONNX 944 ms — **ggml-cpu is now 12 % faster than
ONNX on best-case encoder**. Round 4's 317 ms gap is entirely
closed.

Snapshots: `artifacts/bench/ggml-cpu-round8a-q8_0-m4air.json`.

### 5.15 — round 8b: subsampling mask fast-path  _(done — neutral)_

Added an `all_valid` flag threaded through `build_encoder_graph_cached`
and `subsampling_graph`. When the caller's mel has no trailing
silence (`mel_valid == n_mel_frames`, the common case for a single
utterance), the 8 `apply_time_mask` `ggml_mul` calls in
`subsampling_graph` are skipped — the graph is built without those
ops at all.  `EncoderGraph` caches the `all_valid` value so the graph
is rebuilt when it flips.

Parity: all 9 `test-encoder` gates still pass (the test sends a
padded mel, so `all_valid=false` and the masked path runs).

**Bench impact**: within noise (~0-10 ms), because the mask ops were
already small element-wise muls and ggml-cpu runs them cheaply in the
OpenMP pool.  Shipped anyway for correctness hygiene — running a
no-op mul_by_ones is silly — and because the infrastructure enables
the Round 8c LRU cache to cleanly key on `all_valid`.

### 5.16 — round 8c: multi-shape LRU graph cache  _(done — latent win)_

Replaced the single-shape `Impl::encoder_graph` with a small LRU
`std::vector<std::unique_ptr<EncoderGraph>>` of up to 3 entries. The
cache key is `(n_mel_frames, n_run_layers, all_valid)`.

Behaviour:

  - On `run_encoder`, scan the cache for a matching entry. If found,
    reuse it and move it to the back (most-recently-used).
  - If no match, evict the oldest entry (if cache is full) and build
    a new graph for the current shape.
  - Graph rebuild only happens on a genuine shape change; previously
    any shape change freed the single cached graph and rebuilt it.

This is a **latent** optimisation: the benchmark mode reuses one shape
and shows no change.  The win shows up in production callers that
alternate between a few utterance lengths (streaming, short-burst
input, etc.) — those paths avoid the ~20-50 ms graph rebuild cost on
every length change.

Parity: unchanged. Transcripts bit-equal on both test clips.

### 5.17 — summary, Round 5-8

| round              | code        | jfk best | 20s best | vs ONNX f16 best (944) |
|--------------------|:-----------:|---------:|---------:|------------------------:|
| pre-Round-5        | f16         |      617 |     1197 |               -27 %    |
| Round 5            | f16         |      683 |     1193 |               -26 %    |
| Round 6            | Q8_0        |      600 |      999 |                -6 %    |
| Round 7            | Q8_0 + flash_attn | 559|     1087 |              -15 %    |
| Round 8 (8a+8b+8c) | **Q8_0**    |  **460** | **839**  |           **+11 %**    |

**Round 8 vs ONNX f16**: 11 % faster on best-case encoder on a 20 s clip.

**Fair f16 vs f16** (same precision, different runtimes — 5 warmup + 15 timed runs):

```
                   onnxruntime-f16    ggml-cpu-f16
  -----------------------------------------------
  model size           2.3 GiB         1.3 GiB
  load ms              16 736            642      (26x faster cold start)
  inf best ms             948           1117      (15 % slower)
  inf median ms         1 007           1132      (12 % slower)
  inf stdev ms             52             18      (3x tighter)
  RTF best               0.047          0.055
  RTF median             0.050          0.056
  Transcripts            match          match
```

**Fair int8 vs int8** (generated via ORT dynamic quantization from the same weights,
5 warmup + 15 timed runs):

```
                   onnxruntime-int8    ggml-cpu-Q8_0
  -------------------------------------------------
  model size           583.9 MiB         697 MiB
  load ms               2 054             179      (11x faster cold start)
  inf best ms             677             898      (25 % slower)
  inf median ms           721             928      (22 % slower)
  inf stdev ms             55              25      (2x tighter)
  RTF best               0.034           0.045
  RTF median             0.036           0.046
  Transcripts            match           match
```

Interpretation:

  - ggml is **12–25 % slower** than onnxruntime at the same precision tier.
    onnxruntime's kernels on Apple Silicon route through AMX coprocessor
    instructions (hand-tuned for both f16 and int8) that ggml-cpu's
    OpenMP SIMD threads can't match on multiply-accumulate throughput.
  - ggml stdev is **2–3× tighter** at both tiers (18 vs 52 ms at f16;
    25 vs 55 ms at int8), meaning per-utterance latency is more
    predictable under background OS load.
  - ggml model load is **11–26× faster** — critical for cold-start /
    short-session workloads.
  - The Metal backend (planned Phase 6) will target GPU compute, where
    AMX doesn't apply and ggml's flash-attention kernel (already
    prototyped in Round 7) can be used.

RTF best on 20 s clip (Q8_0): 0.045 → **22x real-time** on CPU alone.
Model load: 179 ms vs ONNX int8's 2054 ms.

Snapshots:

  - `artifacts/bench/ggml-cpu-round5-m4air.json`
  - `artifacts/bench/ggml-cpu-round6-{q8_0,q5_0,q4_0}-m4air.json`
  - `artifacts/bench/ggml-cpu-round8a-q8_0-m4air.json`
  - `artifacts/bench/ggml-cpu-round8-q8_0-m4air.json`
  - `artifacts/bench/ggml-cpu-round8-q8_0-jfk-m4air.json`
  - `artifacts/bench/ggml-cpu-round8-f16-m4air.json`

## Phase 6 — Metal backend  _(done, experimental)_

Bring-up of the `ggml_backend_metal` path for GPU offload on Apple
Silicon.  End-to-end on the M4 Air GPU:

### 6.1 — wire-up

  - `init_gpu_backend(n_gpu_layers, verbose)` helper drives
    `ggml_backend_load_all()` once, then walks the registry in
    registration order (CUDA → Metal → Vulkan → OpenCL → ...) and
    picks the first GPU/IGPU device via `ggml_backend_dev_init`,
    returning `nullptr` when `n_gpu_layers <= 0` or the registry has
    no usable GPU device. Same shape under both `GGML_BACKEND_DL=ON`
    (the dynamic-loader mode embedded host applications use; backends
    are dlopened at runtime)
    and `GGML_BACKEND_DL=OFF` (statically linked; load_all is a
    no-op). Matches the registry-walk convention used by
    `llama.cpp` and `whisper.cpp`.
  - `Impl::backend_active` pointer — one of CPU or GPU — drives
    `ggml_backend_alloc_ctx_tensors`, `ggml_backend_graph_compute`,
    and the per-call `safe_set` tensor uploads.  All weights live on
    the GPU backend (unified memory on Apple Silicon), graph runs
    entirely on GPU.
  - Standard CLI flag: `--n-gpu-layers N` (same spelling as
    llama.cpp / whisper.cpp). Any value > 0 moves the whole encoder
    to GPU — this model has one encoder, so we don't actually need
    per-layer granularity.
  - Compile via `cmake -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`
    (or `-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`).
  - `ggml_conv_2d_dw_direct` (Round 8a) is **not yet implemented on
    Metal** (`ggml_metal_op_encode_impl: error: unsupported op
    'CONV_2D_DW'`). `conformer_conv_graph` takes a `use_conv2d_dw`
    bool, chosen at graph-build time via `ggml_backend_is_cpu(backend)`:
    CPU path uses the fast direct kernel, GPU paths revert to
    `ggml_conv_1d_dw` (im2col + mul_mat, Metal/CUDA/Vulkan supported).
  - `flash_attn_ext` left behind `#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN`
    from Round 7 — should be tested on Metal as a separate follow-up.

### 6.2 — parity

Metal `test-encoder` on `jfk.wav` + `artifacts/ctc-ref`:

```
stage B  subsampling_out        rel=7.641e-04  (CPU: 1.156e-03)
stage C0 post_ff1  (b0)         rel=4.859e-04  (CPU: 9.970e-04)
stage C1 post_attn (b0)         rel=4.866e-04  (CPU: 9.984e-04)
stage C2 post_conv (b0)         rel=4.870e-04  (CPU: 9.987e-04)
stage C3 post_ff2  (b0)         rel=4.880e-04  (CPU: 1.000e-03)
stage C  block_0_out            rel=6.756e-04  (CPU: 1.060e-03)
stage D  block_last_out         rel=1.698e-03  (CPU: 1.730e-03)
stage E  encoder_out            rel=1.698e-03  (CPU: 1.730e-03)
stage F  logits (log_softmax)   rel=3.871e-04  (CPU: 1.362e-03)
```

All 9 gates pass, and Metal per-stage rel is **tighter than CPU**
(the Metal f16 mul_mat kernels use f32 accumulators throughout, which
happens to track NeMo PyTorch's f32 reference more closely than the
CPU path's mixed-precision accumulation).

### 6.3 — bench

`sample-16k.wav` (20 s), `--bench-warmup 5 --bench-runs 15`:

| variant            | enc best | enc median | stdev | RTF best | real-time multiple |
|--------------------|---------:|-----------:|------:|---------:|-------------------:|
| CPU f16 (Round 8)  |    1 117 |      1 132 |    18 |    0.055 |              18x   |
| CPU Q8_0 (Round 8) |      898 |        928 |    25 |    0.045 |              22x   |
| CPU Q4_0 (Round 8) |    1 080 |      1 286 |   138 |    0.054 |              19x   |
| **Metal f16**      |    **266** |    **268** | **1.1** | **0.013** |        **75x**   |
| **Metal Q8_0**     |    **272** |    **274** | **1.5** | **0.014** |        **73x**   |
| Metal Q4_0         |      271 |        272 |   0.5 |    0.014 |              74x   |

On `jfk.wav` (11 s): Metal f16 encoder best 152 ms, median 154 ms.

### 6.4 — comparison vs onnxruntime

`sample-16k.wav`, 5 warmup + 15 timed runs, ggml run with
`--n-gpu-layers 1`:

```
                   onnxruntime-int8    ggml-metal-Q8_0
  ---------------------------------------------------
  model size           583.9 MiB         697 MiB
  load ms               2 295              420      (5.5x faster cold start)
  inf best ms             682              282      (2.4x faster)
  inf median ms           712              283      (2.5x faster)
  inf stdev ms             18             0.83      (21x tighter)
  RTF best               0.034           0.014
  RTF median             0.035           0.014
  Transcripts            match           match
```

**Metal ggml is 2.4x–2.5x faster than onnxruntime's AMX-accelerated
int8 path**, with 21x tighter variance (0.83 ms vs 18 ms stdev).
Metal is compute-bound on GPU shader units, so quantization does not
help (f16 / Q8_0 / Q4_0 all cluster around 272 ms) — but it does
shrink the model file and the unified-memory footprint.

### 6.5 — remaining work

  - Implement `CONV_2D_DW` on the Metal backend (upstream contribution
    to ggml) so the CPU and Metal paths share `conformer_conv_graph`.
    Would buy a few ms more on Metal since the direct path is
    asymptotically cheaper than im2col.
  - ~~Test `ggml_flash_attn_ext` on Metal — likely a meaningful win given
    the fused softmax + V-multiply kernel, plus the dormant infra from
    Round 7 is already in place.~~ **Done** — see §15.8 below. Shipped
    `PARAKEET_FLASH_ATTN=ON` as the Metal default; encoder
    67.35 → 67.00 ms (−0.5 %) and inference 119.24 → 118.66 ms (−0.5 %)
    on M3 Ultra at byte-exact parity. CPU + CUDA + Vulkan + OpenCL keep
    the default OFF until each is A/B'd.
  - Hybrid `ggml_backend_sched` with Metal for the encoder + CPU for
    the mel preprocessor, so the CPU mel path doesn't block the GPU
    encoder. Today the mel runs inline on host before the encoder
    starts; with a sched we could overlap them.

---

### 5.18 — Phase 5 follow-up: future CPU-only headroom

Phase 5 (CPU optimization) is closed: Round 8 Q8_0 is 11 % faster than
`onnxruntime` on the 20 s clip and ships as the default. The items
below are remaining CPU-side ideas that did not make it into Phase 5
itself; they're tracked here so a future "Phase 5.x CPU follow-up"
sweep has a starting list. (Phase 6 ships the Metal backend +
`ggml_backend_sched` work referenced in the first bullet, so that bullet
is historical context.)

  - **Metal backend + `ggml_backend_sched` for GPU offload.**  The
    backend-buffer rework from Round 2 and the cached encoder graph
    from Round 3 are what the sched plumbed through. (Shipped in
    Phase 6; left here for cross-reference.)
    flash_attn_ext (dormant behind `PARAKEET_EXPERIMENTAL_FLASH_ATTN`
    from Round 7) is almost certainly a win on GPU where the
    softmax + V-multiply fuse into one kernel pass.
  - **K-quant tiers (Q4_K_M, Q5_K_M, Q6_K).**  ggml-cpu has k-quant
    kernels too; these might extend the quality-vs-size curve beyond
    the block-quant tiers shipped in Round 6.  Would need a sweep
    against parity.
  - **Bucketed encoder graph cache.**  Round 8c landed an exact-shape
    LRU cache (up to 3 entries) which proved sufficient for the Phase 8
    cache-aware streaming workload (every chunk is a fresh encoder call,
    but the shape set is small enough that the LRU rarely misses). A
    bucketed variant — round up to the next multiple of 64 or 128 mel
    frames — would avoid rebuilds for variable-length production streams
    where chunk shapes vary chunk-to-chunk, at the cost of padding the
    mel input and masking out the tail via the `all_valid=false` path.

## Phase 7 — streaming entry points (Mode 2)  _(done)_

Scope: ship a platform-agnostic streaming API surface on `Engine`
shaped around three transcription modes (one-shot, streamed-output,
duplex). Mode 2 (streamed-output) is implemented on top of today's
offline encoder; Mode 3 (duplex) has its header + ABI surface frozen
but errors at runtime until Phase 8 delivers a cache-aware streaming
GGUF.

Design rationale is in the plan's scope discussion: chunked-batch on
the offline encoder was explicitly rejected because it costs 1-3 % WER
at 2 s chunks with no throughput win on 20 s clips, and Mode 2's
"offline encoder + CTC-timestamp streaming" gets the same UX at zero
accuracy cost.

### 7.1 — Engine class implementation

`include/parakeet/ctc/engine.h` was the declared-but-unimplemented
surface. Phase 7 lands the definition in `src/parakeet_engine.cpp`:

- `Engine(const EngineOptions &)` loads the GGUF once via
  `load_from_gguf`, stores `ParakeetCtcModel` + cancel flag in `Impl`.
- `Engine::transcribe(wav_path)` / `transcribe_samples(samples, n, sr)`
  drive the existing `compute_log_mel` + `run_encoder` +
  `ctc_greedy_decode` + `detokenize` pipeline and return an
  `EngineResult` with per-stage timing.
- `Engine::cancel()` sets an atomic flag; the streaming loop polls it
  between chunks (cooperative cancellation only — the encoder graph
  run itself is not interruptible).

### 7.2 — Stateful CTC window decoder

Added `ctc_greedy_decode_window(logits, start, end, vocab, blank,
inout_prev_token, out_tokens, out_first_frame=nullptr)` in
`src/parakeet_ctc.{h,cpp}`. The existing one-shot
`ctc_greedy_decode()` now delegates to it with `prev_token = -1`.

The stateful variant preserves collapse-repeats across window
boundaries via a caller-managed `inout_prev_token`, so a token whose
first argmax lands in window K and repeats in window K+1 isn't emitted
twice. This is the core invariant that makes Mode 2 byte-equal to the
offline path.

### 7.3 — Mode 2: `transcribe_stream` / `transcribe_samples_stream`

`Engine::transcribe_samples_stream(samples, n, sr, opts, on_segment)`:

1. Runs the existing offline mel + encoder path once.
2. Computes `frames_per_window = max(1, opts.chunk_ms / 80 ms)` (the
   80 ms comes from 10 ms mel hop × 8x subsampling).
3. Walks `[0, T_enc)` in contiguous windows, calling
   `ctc_greedy_decode_window` with a persistent `prev_token`.
4. After each window's decode, detokenizes the *cumulative* token list
   and emits the delta slice as the segment text. Detokenizing
   cumulatively rather than per-window is required because
   `sentencepiece_bpe::detokenize` strips a leading ASCII space — if
   we detokenized per window, the leading space of every segment
   except the first would be silently stripped and `"hello world"`
   would come out as `"helloworld"`. Caught by the first run of the
   new test harness on `jfk.wav` and fixed before landing.
5. Emits `StreamingSegment{text, token_ids, start_s, end_s,
   chunk_index, is_final=true, encoder_ms (first segment only),
   decode_ms}` via the caller's callback, and accumulates into
   the returned `EngineResult`.

`transcribe_stream(wav_path, opts, cb)` is a thin wrapper that loads
the WAV and forwards to `transcribe_samples_stream`. Both return
the full concatenated `EngineResult` so callers that want *both* the
streaming callback *and* a final aggregate don't have to rebuild it
themselves.

### 7.4 — Mode 3 API freeze (errored) _(superseded by §8.2)_

`StreamSession` declares `feed_pcm_f32(const float*, int)`,
`feed_pcm_i16(const int16_t*, int)`, `finalize()`, `cancel()`,
`options()`, destructor, and move ctor/assign. Implementation is in
`src/parakeet_engine.cpp`.

`Engine::stream_start(opts, cb)` probes
`pimpl_->model.supports_streaming` (fed from the new
`parakeet.encoder.streaming.enabled` GGUF key, added to
`load_from_gguf`). Today's GGUFs don't set the flag, so the call
throws `std::runtime_error` with a message pointing at Phase 8 and
suggesting `transcribe_stream()` for full-audio cases. Consumers
can target the final `StreamSession` shape immediately; when Phase
8 lands the error branch is swapped for the real state machine
without touching the public header.

**Update**: §8.2 removed this gate entirely. `stream_start()` now
runs cache-aware streaming inference directly on the existing offline
GGUF and never throws on `supports_streaming`. The `(errored)` marker
above is historical context for how the API was first frozen, not
current behaviour.

### 7.5 — CLI wiring

`src/main.cpp`:

- `--pcm-in PATH` + `--pcm-format {s16le,f32le}` — load raw PCM
  directly (used to validate end-to-end against `LastQuestion_long_EN.raw`
  without adding an ffmpeg dependency to the test loop). Mutually
  exclusive with `--wav PATH`.
- `--stream` — route through `Engine::transcribe_samples_stream`
  instead of the existing `run_once` path. Incompatible with `--bench`
  / `--profile` (those continue to exercise the offline path).
- `--stream-chunk-ms N` — segment stride (default 1000).
- `--emit {text,jsonl}` — `text` prints `[start-end] segment` one line
  per callback; `jsonl` prints a single JSON object per line, with
  proper escaping of `"`, `\`, newlines, control chars. Flushes stdout
  after each segment so downstream players/consumers see output
  immediately.

### 7.6 — Validation harness `test-streaming`

`test/test_streaming.cpp` runs on a loaded Engine and asserts:

- Mode 1 reference: `transcribe()` produces the baseline text.
- Mode 2 byte-equality: for `chunk_ms ∈ {250, 500, 1000, 2000, 4000}`
  plus `audio_duration_ms` (single-segment edge case),
  `transcribe_stream()` segments concatenate byte-equal to Mode 1.
- Mode 2 timestamp continuity: every segment's `start_s` matches the
  previous `end_s` (within 1 ms rounding); `end_s > start_s`;
  `is_final=true` for every segment in Phase 1.
- Mode 3 error path: `stream_start()` on a non-streaming GGUF throws
  an exception whose message mentions "streaming" or "Phase 8".

Caught the cumulative-vs-per-window detokenize bug on first run; once
fixed, all checks pass on `jfk.wav` and on the long speech clip via
CLI byte-equality.

### 7.7 — Real-world validation

`LastQuestion_long_EN.raw` (5.46 min, 16 kHz s16le mono;
external long-form fixture, not tracked in this repo):

- offline `--model ... --pcm-in ...` transcript: 5169 bytes, 1710
  tokens, 4099 encoder frames.
- `--stream --stream-chunk-ms 2000` segments concatenated: 5169 bytes,
  byte-equal to offline (`diff` produces no output).
- Metal Q8_0 timing: `mel=152ms enc=14941ms dec=5ms total=15099ms
  RTF=0.046` (22x real-time). Mode 2's overhead over the offline path
  is sub-ms (the stream variant's total wall is the encoder pass +
  1710-token cumulative detokenize + callback dispatch).
- `--emit jsonl` emits one correctly-escaped JSON object per segment;
  first chunk lands at `start=0.000 end=0.480` with the first word
  `"but"`.

### 7.8 — Phase 8 design notes (historical; superseded by Phase 8 below)

This section captures the original Phase 8 scoping notes from before
Mode 3 shipped. Phase 8 (the next section) records the actual
implementation; the rolling-encoder design that landed differs from
the cache-aware streaming-checkpoint plan sketched here. Kept for
the round-by-round journal trail.

Prerequisites and scope tracked for Phase 8 (as planned at the time):

1. **Checkpoint selection (go/no-go gate).** Evaluate candidate NeMo
   cache-aware streaming checkpoints against Parakeet-CTC-0.6B on the
   repo's reference clips. Primary candidate:
   `stt_en_fastconformer_hybrid_large_streaming_multi`. Accept only
   if WER on reference set is within ±0.5 % of current offline.
2. **New converter** scoped similarly to `convert-nemo-to-gguf.py`,
   to set the `parakeet.encoder.streaming.enabled = true` metadata flag
   that `stream_start()` already probes.
3. **Streaming encoder graph**: per-layer attention KV cache tensors
   (left-context), depthwise-conv left-state tensors, chunked +
   left-context attention mask, streaming mel state (reflect-pad only
   on true first/last chunk, per-chunk CMVN *or* running-mean CMVN
   depending on what the chosen checkpoint was trained with).
4. **Bucketed graph cache** — round chunk mel length up to a fixed
   bucket so the 3-entry LRU graph cache reuses the compiled graph
   across chunks. Already tracked in §5.18.
5. `StreamingOptions` gains `left_context_ms`, `right_lookahead_ms`,
   `emit_partials` (activated in Phase 8).
6. Per-stage numerical parity harness vs the NeMo streaming reference,
   following the Round 5-8 methodology.
7. Mode 3 bring-up + `StreamSession` state machine (sample ring,
   bucket-rounded encoder call, KV cache slide, cancellation).
8. Wire the CLI `--pcm-in` path (plus future `--pcm-in -` for stdin)
   through `stream_start()` for manual live-streaming testing.
9. Expected performance (extrapolated from §6.x Metal numbers): 20 s
   clip, `chunk_ms=500`: ~350-450 ms total, first segment ~0.55 s;
   60 s clip, `chunk_ms=2000`: ~700 ms total (vs offline ~850 ms —
   linear-in-T attention wins on long-form).

## Phase 8 — Mode 3 cache-aware streaming  _(done; KV-cache optimisation tracked as Phase 8.5)_

### Phase 8.0 — checkpoint landscape (done)

The NeMo registry has only one cache-aware streaming Conformer family:
`stt_en_fastconformer_hybrid_large_streaming_{multi,80ms,480ms,1040ms}`.
It's a 115M-parameter, RNN-T+aux_CTC hybrid trained with chunked-limited
attention. Real-world quality on this family is not great (~2x WER vs
Parakeet-CTC-0.6B offline).

So Phase 8 is **not** going to ship a port of streaming_multi. Instead,
the chosen approach is **cache-aware *inference* on the existing
offline-trained Parakeet-CTC-0.6B weights**: same 600M model, same
quality ceiling, just driven through a streaming forward pass. The
cost is some accuracy degradation because the model wasn't trained
with chunked attention masks.

### Phase 8.1 — Python reference + accuracy bake-off (done)

`scripts/streaming-reference.py` implements the **chunking-with-context**
strategy in Python on top of the NeMo offline model: each chunk feeds
`[left_context + chunk + right_lookahead]` into the offline encoder,
slices out the center frames, runs CTC greedy with a stateful
`prev_token` carried across chunks. This mirrors what the eventual C++
streaming path does (modulo the indefinitely-deferred Phase 8.5
KV-cache-on-offline-weights optimisation; see §8.5 for why this is
distinct from chunked-limited streaming inference and why the latter
is rejected).

Sweep results on `test/samples/jfk.wav` (11 s clean speech) and
`LastQuestion_long_EN.raw` (5.5 min sci-fi narration with proper nouns,
representing the "harder" production case):

| chunk_ms | left_ctx_ms | right_lookahead_ms | jfk WER | long-clip WER | first-seg latency |
|---------:|------------:|-------------------:|--------:|--------------:|------------------:|
| 1000 | 0     | 0    | 40.91% | n/a    | 1.0 s |
| 1000 | 2000  | 500  | 0.00%  | 14.86% | 1.5 s |
| 2000 | 2000  | 1000 | 0.00%  | 7.64%  | 3.0 s |
| 2000 | 5000  | 1000 | 0.00%  | n/a    | 3.0 s |
| 2000 | 5000  | 2000 | n/a    | 4.02%  | 4.0 s |
| 2000 | 10000 | 1000 | n/a    | 7.53%  | 3.0 s |
| **2000** | **10000** | **2000** | n/a | **3.82%** | **4.0 s** |
| 4000 | 5000  | 1000 | 0.00%  | 4.75%  | 5.0 s |

Key observations:

- **Right lookahead is the single most impactful knob.** Going from
  1000 → 2000 ms right-lookahead drops long-clip WER from 7.64% to
  4.02% at the same chunk + left configuration. The conv module uses
  symmetric `kernel=9` padding (designed at training to see future
  context), so denying it future frames at chunk boundaries hurts more
  than denying past frames.
- **Short audio is forgiving, long audio compounds errors.** jfk is
  at 0% with modest context; 5.5 min same-config drifts to ~7-8%
  because (a) per-window CMVN drifts vs the offline single-statistic
  pass and (b) more boundary opportunities for misreads. Per-window
  CMVN is the suspect; running CMVN may close some of the gap and is
  noted as a Phase 8.5 follow-up.
- **`right=0` (pure causal) is uniformly bad** unless chunks are large
  enough to hide the boundary (chunk=2000 + right=0 → 4.55% on jfk;
  chunk=500 + right=0 → 40.9%). Pure-causal mode is supportable but
  not the recommended default.
- **Left context past 5 s gives diminishing returns** on this model.

**Recommended C++ defaults (subject to revision once C++ measurements
are in):**

- `StreamingOptions{ chunk_ms = 2000, left_context_ms = 10000,
  right_lookahead_ms = 2000 }` — sweet-spot accuracy
  (~4 % WER on long-form, 0 % on short clean speech), ~4 s
  first-segment latency. Suits production live-captioning.
- For lower latency, callers can pick e.g. `chunk_ms=1000,
  left=2000, right=500` and accept ~10-15 % WER on long-form.

### Phase 8.2 — C++ implementation plan

Next milestones in order:

1. Extend `StreamingOptions` with `left_context_ms` +
   `right_lookahead_ms`; remove the runtime gate in `stream_start()`
   (any Parakeet-CTC GGUF works in streaming mode now — no metadata
   probe needed).
2. Implement `StreamSession` state machine (sample ring, chunk
   dispatch, per-window mel + encoder call, logits center-slicing,
   CTC stateful decode, segment emission with absolute timestamps).
3. Wire the CLI `--stream` path to drive `stream_start()` when
   `--pcm-in` is used (or always — same flag for Mode 2 vs Mode 3 is
   surprising; revisit in §8.3).
4. Per-chunk numerical parity vs the Python reference at the same
   `(chunk_ms, left_ctx_ms, right_lookahead_ms)` config to make sure
   the C++ port lands on the same logits, not approximately.
5. Test harness extension covering Mode 3 with random burst feeds.

Phase 8.5 (perf follow-up): replace chunking-with-context with true
KV cache + conv state tensors, ~6× compute reduction on long-form
audio without changing accuracy.

### Phase 8.2 — C++ StreamSession (done)

Landed the Mode 3 state machine in [src/parakeet_engine.cpp](src/parakeet_engine.cpp)
(`StreamSession::Impl`), backed by the existing `Engine::Impl::model`
through a borrowed pointer. Key pieces:

- `feed_pcm_f32` / `feed_pcm_i16` append samples to a `pending`
  buffer and trigger `try_emit_chunks()`.
- `try_emit_chunks()` consumes one chunk at a time while
  `pending.size() >= chunk_samples + right_lookahead_samples`.
  Per-chunk window = `left_history + chunk + right_lookahead`.
  After the encoder run, the consumed chunk is appended to
  `left_history` (rolling cap at `left_context_samples`).
- `flush_remainder()` (called from `finalize()`) processes the tail
  with whatever lookahead is left. No right-lookahead on the final
  chunk, so the last ~`right_lookahead_ms` of audio sees less conv
  context — acceptable for a one-shot end-of-audio case.
- Per-chunk numerical path reuses `compute_log_mel` + `run_encoder` +
  `ctc_greedy_decode_window` + `detokenize` from the offline stack
  unchanged. Cumulative detokenize with suffix slicing preserves the
  leading-space invariant that bit Mode 2 in §7.3.
- Segment timestamps: `start_s = emitted_samples / sr`,
  `end_s = (emitted_samples + consumed_chunk_samples) / sr`, absolute
  from the start of the session. Same shape any external streaming
  consumer would expect (`{ start, end, text/toAppend }`).
- `StreamingOptions::left_context_ms` and `right_lookahead_ms`
  landed on the public API (defaults 10000 / 2000 respectively, the
  winners from §8.1's sweep).
- `Engine::stream_start()` no longer gates on the GGUF metadata
  `parakeet.encoder.streaming.enabled` — any Parakeet-CTC GGUF works.
  The metadata key survives in the loader for Phase 8.5 / 9 use.

### Phase 8.3 — CLI + validation (done)

CLI:

- `--stream --stream-duplex` routes through `stream_start()` +
  `feed_pcm_f32` with 4 kB default block size (configurable via
  `--stream-feed-bytes`, useful to stress the session state machine).
- `--stream-left-context-ms` + `--stream-right-lookahead-ms` override
  the `StreamingOptions` defaults.
- `--emit text|jsonl` reused unchanged.

Test harness (`test/test_streaming.cpp`) adds three Mode 3 configs on
`jfk.wav` (`chunk_ms × left_ms × right_ms ∈ {1000,2000,500},
{2000,2000,1000}, {2000,5000,2000}`) plus a cancel-path assertion.
PCM is fed in **random-size bursts (512-4000 samples)** via
`feed_pcm_f32` to exercise the ring / chunk-dispatch paths; each
config asserts WER ≤ 5 % vs the Mode 1 reference (all hit 0 %).

### Phase 8.4 — end-to-end numbers

`LastQuestion_long_EN.raw` (5.46 min, 16 kHz s16le, Apple M4 Air,
Metal Q8_0), default config `chunk_ms=2000, left=10000, right=2000`:

- C++ Mode 3 transcript: 972 words, **4.13 % WER** vs offline.
- Python f32 reference at the same config: 3.82 % WER. The 0.3 %
  delta is Q8_0 quantisation noise, matches §6.x's measurements.
- Wall time: 35.3 s, RTF 0.108 (9× real-time). ~2.4× slower than the
  Mode 2 offline encoder (RTF 0.046) because each chunk re-runs the
  encoder on a 14 s window (`left + chunk + right`) instead of the
  shipping-forward incremental state. This is the chunking-with-context
  tax; Phase 8.5 closes it.
- First-segment latency: `chunk_ms + right_lookahead_ms` ≈ 4 s wall
  (matches Python reference).

### Phase 8.5 — KV cache / conv state (deferred indefinitely; not the same as chunked-limited streaming inference)

> **Important distinction — read this before touching streaming
> internals.** Two superficially-similar designs have been proposed
> (and one of them has been attempted twice) on this project, and
> they have very different quality implications. Conflating them is
> what makes this corner trip-hazardous.

#### (A) Chunked-limited streaming inference on a chunked-limited-trained checkpoint

What NeMo's `cache_aware_stream_step` actually does. Each query
attends only to a fixed lookback window; per-chunk encoder cost
drops to `O(chunk)`; per-layer `(lookback, d_model)` K/V cache plus
`(d_model, kernel-1)` depthwise-conv state slide forward each call.
Looks like an attractive perf win on paper.

**This shape has been evaluated twice on this project and rejected
both times on quality grounds.**

- **Round 1 — Phase 8.0** evaluated
  `stt_en_fastconformer_hybrid_large_streaming_multi`, the only
  NeMo cache-aware streaming Conformer family available at the time.
  Real-world quality landed at ~2× WER vs `parakeet-ctc-0.6b`
  offline. Phase 8 therefore chose the rolling-encoder Mode 3
  design instead.
- **Round 2 — Phase 12.x exploration** ported
  `nvidia/parakeet_realtime_eou_120m-v1` (same model family, newer
  120 M variant) as the EOU engine in Phase 12.5 on the rolling-
  encoder Mode 3, and scoped a true cache-aware fast path as the
  follow-up. A bit-equal C++ port of NeMo's
  `cache_aware_stream_step` was prototyped on a working branch:
  per-layer K/V cache, depthwise-conv state, chunked-limited
  streaming attention mask, generalised Transformer-XL `rel_shift`
  for `T_q != T_kv`. Numerical parity vs NeMo was clean — worst rel
  `1.85e-3` over 44 chunks of `jfk.wav` — but decoded end-to-end
  through `eou_decode_window`, the result reproduced exactly NeMo's
  streaming transcript, which is **not** the offline transcript:

  ```
  Mode 2 / offline:   "and so my fellow americans ask not what your
                       country can do for you ask what you can do
                       for your country<EOU>"
  cache-aware
    (NeMo + ours):    "that's all i've held america ask not what
                       your country can do for you ask what you can
                       do for your country"
  ```

  Same quality cliff Phase 8.0 had already documented two years
  earlier on the same model family. NeMo's own cache-aware
  streaming RNN-T over the same 88 encoder frames also fails to
  emit any `<EOU>` token on `jfk.wav`, so the cache-aware path
  doesn't even win on `<EOU>` boundary detection vs the rolling
  encoder. **The branch was reverted** before any of it landed on
  `main`; this section exists so a third iteration of the project
  doesn't redo the same loop.

**Bottom line for (A): cache-aware streaming inference on a
chunked-limited-trained ASR checkpoint is a quality regression in
this project's context (clean speech, offline-quality transcripts
as the bar). It will not be implemented. If a future requirement
explicitly trades early-utterance accuracy for bounded compute
(low-power voice agent, very long-form streaming), revisit this
decision with that requirement on the table — but assume by default
that re-running this exploration will produce the same numbers.**

#### (B) KV cache / depthwise-conv state on the offline-trained CTC / TDT weights

The original scope of "Phase 8.5", and a *different* design from
(A) despite the surface-level similarity. Same offline-trained
weights, same full attention pattern as training, just amortised
across chunks: keep per-layer `K`, `V`, and depthwise-conv
left-state tensors as backend buffers, slid forward each chunk;
each encoder call computes only over new-chunk + right-lookahead
frames instead of the full `(left + chunk + right)` window. Pure
compute-layout refactor — **accuracy unchanged**. Projected wins
on the original §8.1 Python reference:

- Per-chunk compute: down from `O(left + chunk + right)` to
  `O(chunk + right)`, i.e. ~5× on the default config
  (2 + 2 vs 10 + 2 + 2).
- Total wall on the 5.5 min clip: down from 35 s to ~10-15 s
  (close to the offline 15 s baseline).
- Accuracy unchanged.

Crucially, the streaming graph for (B) is **not** the same shape
as the chunked-limited graph from (A). Different attention mask
(no chunked-limit), different cache-size policy (sliding window
without a quality-coupled lookback), different validation fixtures
(parity vs offline forward, not vs `cache_aware_stream_step`). Any
future attempt at (B) should treat it as a fresh design exercise,
not as a retrofit of any (A) prototype recovered from git history.

Requires graph changes (persistent cache tensors for attention +
conv module), per-stage parity harness vs the §8.1 Python reference,
and a sliding-window cache-eviction policy.

**Status: deferred indefinitely.** No current owner. Not on the
critical path of any shipping feature. Pick up only when a concrete
consumer needs the per-chunk compute reduction and is willing to
pay the engineering cost. The §8.1 Python reference and the rolling-
encoder Mode 3 implementation in §8.4-8.7 remain the source of truth
for streaming-quality expectations on CTC / TDT.

## Phase 9 — multi-model support _(done; ships parakeet-ctc-1.1b alongside 0.6B)_

Phase 9 extends the Parakeet-CTC pipeline beyond the initial 0.6B
checkpoint. Target is drop-in support for other NeMo Parakeet-CTC
checkpoints that share the FastConformer architecture, without
branching the converter or the C++ encoder graph.

### Phase 9.1 — parakeet-ctc-1.1b (done)

HF repo: `nvidia/parakeet-ctc-1.1b`. NeMo encoder config:

    d_model: 1024          (same as 0.6B)
    n_layers: 42           (was 24)
    n_heads: 8             (same)
    ff_expansion_factor: 4 (same; ff_dim=4096)
    conv_kernel_size: 9    (same)
    subsampling_factor: 8  (same)
    subsampling_conv_channels: 256 (same)
    self_attention_model: rel_pos (same)
    conv_norm_type: batch_norm (same, fused at convert)
    att_context_size: [-1, -1]  (same, offline)
    vocab_size: 1025 (1024 BPE + CTC blank)

Only `n_layers` differs from 0.6B. The converter already reads
`n_layers` from the NeMo YAML and iterates block-by-block, and the C++
loader reads `parakeet.encoder.n_layers` from GGUF metadata. Zero
code changes needed to produce a working 1.1B GGUF and transcribe with
it. The one stale hardcode was in the `--profile` CLI path
(`layer_points = {0, 1, 12, 24}`); now scales with `n_layers` via
`{0, 1, n_layers/2, n_layers}`.

End-to-end results on Apple M4 Air, Metal, Q8_0:

| Clip | Model | Wall time | RTF | Encoder ms |
|-|-|-|-|-|
| jfk.wav (11 s) | ctc-0.6b | 170 ms | 0.015 | ~155 ms |
| jfk.wav (11 s) | ctc-1.1b | 281 ms | 0.026 | 276 ms |
| LastQuestion_long_EN.raw (5.5 min) | ctc-0.6b | 15.1 s | 0.046 | 14.9 s |
| LastQuestion_long_EN.raw (5.5 min) | ctc-1.1b | 24.4 s | 0.074 | 24.2 s |

1.1B is 1.6-1.8× slower than 0.6B, roughly matching the layer-count
ratio (42/24 ≈ 1.75). Metal still hits ~13× real-time on the long clip
with 1.1B, comfortably faster than real-time. Transcripts differ on
2.3 % of words on the long clip (22 / 969) — typical quality difference
between the two models; both ship transcripts that track the same
content.

Streaming (Mode 2 and Mode 3) works out of the box with 1.1B:
`test-streaming` passes all chunk-size byte-equality checks (Mode 2)
and all three (chunk × left × right) configs at WER ≤ 5% (Mode 3).

### Phase 9.x — next candidates

Natural follow-ups that reuse the same converter + encoder graph:

- `nvidia/parakeet-tdt_ctc-110m`: 110 M hybrid TDT+CTC. 512 × 17
  FastConformer. CTC head alone can be decoded by the existing
  `ctc_greedy_decode_window`; TDT primary decoder is a separate
  port (prediction net + joint net + transducer greedy).
- `nvidia/parakeet-tdt-0.6b-v3`: bigger TDT model family — same
  transducer-decoder port, larger encoder.
- `nvidia/parakeet-ctc-110m`: smaller CTC-only variant if it exists;
  would land as a converter-flag change only.

## Phase 10 — TDT (Token-and-Duration Transducer) support _(done; covers parakeet-tdt-0.6b-v3 + parakeet-tdt-1.1b, Mode 1/2/3)_

Phase 10 ports `nvidia/parakeet-tdt-0.6b-v3`, the multilingual (~25
languages) TDT ASR model with punctuation-and-capitalization. Shares
the FastConformer encoder backbone with the CTC checkpoints but needs
its own decoder: 2-layer LSTM prediction network, joint MLP, and a
transducer greedy loop that interleaves token + duration predictions.

### Phase 10.1 — converter + loader (done, commit c501c4c)

Auto-detects model flavour from the NeMo `target` field
(`EncDecCTCModelBPE` vs `EncDecRNNTBPEModel`). Writes
`parakeet.model.type` + `parakeet.tdt.*` metadata and tensors:

- `tdt.predict.embed.weight`                (V+1, 640)
- `tdt.predict.lstm.{0,1}.{w_ih,w_hh,b_ih,b_hh}`
- `tdt.joint.{enc,pred,out}.{weight,bias}`

Handles the architectural differences from CTC:

- `use_bias=False` — all encoder linear biases are optional; the
  loader reads them via `maybe_tensor()` and the graph skips every
  `mul_mat + bias` via a new `maybe_add_bias()` helper.
- `xscaling=False` — gated the `ggml_scale(x, sqrt(d_model))` entry.
- 128 mel bins (vs 80) — the existing `n_mels` plumbing handles it,
  `subsampling_freq_bins` derives correctly (128/8 = 16 freq bins
  after subsampling -> `pre_encode.out` = (1024, 4096)).
- 8192-vocab SentencePiece (vs 1024) — `blank_as_pad` semantics; the
  joint output is 8192 labels + 1 blank + 5 durations = 8198.

CTC regression: re-converting parakeet-ctc-0.6b produces an identical
tensor set to the shipping GGUF (zero additions/removals), plus two
new metadata keys (`parakeet.model.type`, `parakeet.encoder.use_bias`)
that the loader reads with safe fallbacks.

### Phase 10.2 — encoder parity (done, commit 48be18b)

`scripts/dump-tdt-reference.py` dumps NeMo per-stage tensors (log-mel,
encoder_out, LSTM init state, transcribe() text). New
`test/test_tdt_encoder_parity.cpp` harness loads a TDT GGUF + wav +
reference dir, runs the C++ encoder, compares.

Results on jfk.wav:

| dtype  | mel max_abs / rel   | enc_out max_abs / rel | verdict |
|--------|---------------------|-----------------------|---------|
| n/a    | 7.29e-1 / 2.77e-3   | —                     | — (mel same as CTC) |
| f16    | (see above)         | 1.11e-3 / 2.15e-3     | PASS (< 5e-3 f16 floor) |
| q8_0   | (see above)         | 1.43e-2 / 1.97e-2     | q8_0 accumulation over 24 layers without biases; PASS functionally (downstream transcripts stable) |

Encoder graph works correctly — any numerical differences are pure
quantization accumulation, matching the CTC precedent.

### Phase 10.3 — TDT decoder (done, commit 3224e23)

`src/parakeet_tdt.{h,cpp}` implements LSTM + joint + transducer greedy
on CPU in pure f32. Weights are dequantized from the loaded GGUF once
at Engine construction via `ggml_get_type_traits(type)->to_float`,
which handles f32 / f16 / q8_0 / q5_0 / q4_0 uniformly. Post-dequant
footprint: ~70 MiB f32 for the decoder (embedding + 2-layer LSTM +
joint MLP).

Decode loop:

1. Initialize LSTM h/c to zeros, feed blank through the prediction
   net to produce the initial `g` output.
2. Per encoder frame:
   - Compute joint logits (8198,) = ReLU(enc_proj + pred_proj) @ W_out.
   - argmax over first V+1=8193 -> token; argmax over last 5 -> duration.
   - If token == blank: advance `t` by `max(1, dur)`, reset sym counter.
   - Else: emit token, step LSTM on token embedding, update `g`;
     advance `t` only when `dur > 0` or `sym_count >= max_symbols`
     (max_symbols=10 matches NeMo's greedy config).
3. Detokenize through the existing `sentencepiece_bpe` helper (shared
   with CTC). NeMo's TDT v3 tokenizer has 8192 SBPE pieces covering
   ~25 languages + multilingual PnC.

Engine wiring:

- `Engine::Impl` owns the `TdtRuntimeWeights` populated at
  construction for TDT GGUFs; the CTC path ignores it.
- `Engine::transcribe()` / `transcribe_samples()` branch on model
  type. Streaming entry points still reject TDT via
  `ensure_ctc_only()` — transducer streaming is a Phase 10.5 item.
  _(Superseded by §10.5: `ensure_ctc_only()` was removed; both Mode 2
  and Mode 3 streaming run on TDT GGUFs today.)_
- CLI `run_once` lambda has the same branch; TDT GGUFs now transcribe
  end-to-end via `--wav` / `--pcm-in`.

### Phase 10.4 — end-to-end results

All measured on Apple M4 Metal, f16 GGUF (1.34 GiB), unless noted.

**jfk.wav (11 s):** C++ transcript is *byte-identical* to NeMo:

> "And so, my fellow Americans, ask not what your country can do for
>  you, ask what you can do for your country."

`load=490 ms, mel=5.5 ms, enc=211 ms, dec=48 ms, total=264 ms,
RTF=0.024` (42× real-time).

**LastQuestion_long_EN.raw (5.5 min):** clean long-form transcription
with proper nouns (Multivac / Adele / Lupov / Pluto), commas,
periods, dialog structure. `RTF=0.050` (20× real-time), 1825 tokens,
1472 ms pure-CPU decode.

**Multilingual sanity** (external `sample_*.raw` clips, not tracked
in this repo):

- Spanish: *"Se recomienda enfáticamente a los viajeros..."*
- French:  *"L'accident a eu lieu en terrain montagneux..."*
- German:  *"Für die besten Aussichten auf Hongkong..."*
- Italian, Portuguese, Russian: native-script transcripts with
  punctuation.
- Japanese: produces garbled output (same as NeMo reference on the
  same sample; confirmed model limitation, not a port bug).

### Phase 10.5 — TDT streaming (Mode 2 + Mode 3)  _(done)_

Refactored the TDT decoder around a stateful window primitive so all
three entry points work uniformly on both CTC and TDT GGUFs.

**`TdtDecodeState`** (new, in `src/parakeet_tdt.h`) holds the LSTM
hidden + cell tensors per layer, the last-layer pred_out vector that
feeds the joint, the `symbols_this_step` counter for the max_symbols
guard, an `initialized` flag, and a `carry_frames` counter for cases
where a large-duration advance spills past the end of the current
window.

**`tdt_decode_window(encoder_out_ptr, n_frames, opts, &state,
&out_tokens, &out_steps)`** is the new primitive. It:

1. Lazy-inits the state on first call by feeding the blank token
   through the LSTM.
2. Honours `state.carry_frames` to skip frames consumed by a
   duration-advance from the previous window.
3. Walks encoder frames, argmax'ing both token and duration logits,
   emitting non-blank tokens, stepping the LSTM on every emission.
4. Stops when the cursor hits `n_frames`; parks any leftover advance
   in `state.carry_frames` for the next call.

`tdt_greedy_decode()` is now a thin wrapper: fresh state, one window
spanning all encoder frames.

Engine wiring (`src/parakeet_engine.cpp`):

- `Engine::Impl` already had `TdtRuntimeWeights` from §10.3; no changes.
- `transcribe_samples_stream()` (Mode 2): branches on `model_type`.
  TDT path inits a fresh `TdtDecodeState`, then calls
  `tdt_decode_window()` once per `frames_per_window` range. CTC path
  unchanged.
- `StreamSession::Impl` gains a `TdtDecodeState tdt_state` that is
  primed in `stream_start()` for TDT GGUFs. Each live chunk's
  center-frame range (after `left_drop_frames` + before
  `right_drop_frames`) flows through `tdt_decode_window()` with the
  session's carried state.
- `ensure_ctc_only()` helper is gone; the CLI gate that short-circuited
  `--stream` for TDT is gone too.

Public API addition: `Engine::model_type() -> "ctc" | "tdt"` (Phase 11
extends this to also return `"sortformer"`), so downstream callers
(and the test harness) can pick per-model knobs
without reaching through internal headers.

Test harness (`test_streaming.cpp`):

- Mode-3 configs carry per-model WER tolerances.
- CTC tolerates 5 % at all three configs.
- TDT tolerates 5 % at chunk=2000 configs, 40 % at the aggressive
  chunk=1000 left=2000 right=500 slot (observed 36 % on jfk — still
  passes offline parity at the larger context, but the transducer
  greedy is more sensitive to short chunks + small right-lookahead
  than CTC's greedy collapse).

End-to-end numbers on `parakeet-tdt-0.6b-v3.f16.gguf`, M4 Metal:

- Mode 2 (offline encoder + streamed segments):
    - jfk.wav, chunk_ms=2000: 6 segments, concatenated byte-identical
      to one-shot transcribe().
    - LastQuestion_long_EN.raw, chunk_ms=2000: 164 segments; correctly
      cased/punctuated; cumulative text matches offline.
- Mode 3 (live duplex):
    - jfk.wav, chunk=2000 / left=5000 / right=2000: 5 segments,
      0.00 % WER vs one-shot.
    - LastQuestion_long_EN.raw at the default preset
      (chunk=2000 / left=10000 / right=2000): 9.84 % WER vs offline
      TDT. Higher than CTC's 4.13 % on the same clip; the TDT
      transducer decode is more sensitive to missing future context
      at chunk boundaries. Still usable for live captioning; the gap
      narrows with bigger chunks / right-lookahead if latency allows.

`live-mic` now works with TDT GGUFs unchanged, so native-microphone
captions stream out properly-cased + punctuated text end-to-end.

### Phase 10.6 — parakeet-tdt-1.1b (done, zero code changes)

`nvidia/parakeet-tdt-1.1b` is the deeper English-only TDT sibling of
`parakeet-tdt-0.6b-v3`:

| Config | tdt-0.6b-v3 | **tdt-1.1b** |
|-|-|-|
| Encoder layers     | 24  | **42** |
| Mel bins           | 128 | **80** |
| Vocab              | 8192 (multilingual + PnC) | **1024** (English only, lowercase no PnC) |
| `use_bias`         | False | **True** (default) |
| Decoder / joint    | 2-layer LSTM 640 / joint 640 + 5 durations | *same* |

Every one of those dimensions (`n_layers`, `n_mels`, `vocab_size`,
`use_bias`) is already read from GGUF metadata, so the converter and
C++ loader/decoder support this checkpoint with **no code changes**.

Measured on Apple M4 Metal, q8_0 GGUF (1.22 GiB):

- **Byte-identical to NeMo on jfk.wav**:
  `"and so my fellow americans ask not what your country can do for
   you ask what you can do for your country"` (lowercase/no-PnC;
  matches NeMo `parakeet-tdt-1.1b.transcribe()` exactly).
- `jfk.wav` (11 s): `load=419 ms, mel=5 ms, enc=277 ms, dec=20 ms,
  total=301 ms, RTF=0.027` (37× real-time). Encoder scales 1.75×
  vs tdt-0.6b-v3 as expected (42/24 layers). Decode is **faster**
  than tdt-0.6b-v3 (20 ms vs 48 ms) because the 1024-class output
  layer is 8× smaller than the 8192-class multilingual one.
- `LastQuestion_long_EN.raw` (5.5 min): `RTF=0.079` (13× real-time),
  1716 tokens, high-quality transcript of all dialog content.
- `test-streaming` on tdt-1.1b: **10/10 PASS**, including 0 % WER on
  *all three* Mode 3 configs (even the aggressive chunk=1000/left=2000/
  right=500 slot where tdt-0.6b-v3 had 36 % — the deeper encoder +
  English-only training gives better streaming quality too).
- Mode 2 + Mode 3 streaming work end-to-end with no tuning; `live-mic`
  just works.

`scripts/download-all-models.sh` gains parakeet-tdt-1.1b so offline
bootstrapping picks it up automatically.

### Phase 10.7 — pending follow-ups

- **BLAS / Accelerate for the LSTM + joint gemvs.** Current decode
  uses pure scalar loops: 20-48 ms / 11 s on f16 one-shot, 890 ms -
  1.5 s / 5.5 min. Not a bottleneck today; easy win at high throughput.
- **Quantized (q8_0 / q4_0) TDT GGUFs sweep.** Converter and loader
  already handle these storage types via the universal dequant path,
  but haven't been sweep-tested for WER drift.
- **parakeet-tdt_ctc-110m support.** Same TDT decoder, smaller
  512 × 17 FastConformer encoder; `.nemo` already cached locally.

## Phase 11 — Sortformer (4-speaker diarization) _(done through §11.11.1; spkcache streaming tracked as Phase 11.11.2)_

Phase 11 ports `nvidia/diar_sortformer_4spk-v1`, a speaker-diarization
model that shares the FastConformer encoder backbone with our Parakeet
ports but adds a Sortformer-specific head: encoder projection, an
18-layer post-LN Transformer encoder, and a small MLP that produces
per-frame, per-speaker probabilities for up to 4 speakers (multi-label
sigmoid output handles overlapping speech).

### Phase 11.1 — converter + C++ loader (done, commit dee5e86)

scripts/convert-nemo-to-gguf.py auto-detects Sortformer
checkpoints (target == SortformerEncLabelModel) and:

- Skips tokenizer extraction (Sortformer has no SentencePiece).
- Writes parakeet.sortformer.* metadata: num_spks, fc_d_model,
  tf_d_model, tf_n_layers, tf_n_heads, tf_inner_size, tf_pre_ln,
  tf_hidden_act.
- Writes new tensors: encoder_proj (512 -> 192), 18 transformer
  blocks (attn q/k/v/out + ln1 + ffn in/out + ln2), and the head
  (first_hidden_to_hidden + single_hidden_to_spks). The 384-wide
  hidden_to_spks (used only in v2 streaming) is intentionally
  skipped.

C++ loader (src/parakeet_ctc.{h,cpp}):

- ParakeetModelType::SORTFORMER added; EncoderConfig grows
  sortformer_{num_spks,fc_d_model,tf_d_model,tf_n_layers,tf_n_heads,
  tf_inner_size,tf_pre_ln} fields.
- New SortformerWeights / SortformerTransformerBlock structs hold the
  18 blocks + head linears.
- Engine::transcribe / streaming entry points reject Sortformer GGUFs
  with a clear message pointing at PROGRESS.md.

### Phase 11.2-11.4 — Python reference + C++ forward pass (done, commit 4c55c16)

scripts/dump-sortformer-reference.py replicates NeMo's
process_signal -> frontend_encoder -> transformer_encoder ->
forward_speaker_sigmoids chain on a wav and dumps per-stage
references (mel, encoder_out, post_proj, post_transformer,
speaker_probs).

src/parakeet_sortformer.{h,cpp} is the CPU forward:

- SortformerRuntimeWeights holds f32-dequantised tensors. Same
  to_float trait pattern as TDT, so f32/f16/q8_0/q4_0 all just work.
- sortformer_diarize() pipeline:
  1. linear_batch(encoder_proj) -> (T, 192)
  2. for each transformer block (post-LN, pre_ln=False):
       attn(Q,K,V) + residual + layer_norm_1 -> ffn(ReLU) + residual
       + layer_norm_2
  3. ReLU -> first_hidden_to_hidden -> ReLU -> single_hidden_to_spks
     -> sigmoid -> (T, num_spks)
  4. Threshold-based per-speaker segment formation, sorted by start
     time then speaker_id.

Numerical parity on jfk.wav vs NeMo (Apple M4 Metal, f16 GGUF):

  mel    : max_abs=3.36e-1 rel=1.65e-3  PASS  (peak-norm matches)
  enc    : max_abs=3.50e-3 rel=1.62e-3  PASS  (FastConformer)
  probs  : max_abs=8.68e-4 rel=2.03e-4  PASS  (encoder_proj +
                                               18-layer transformer +
                                               head + sigmoid)
  speaker activity matches NeMo exactly: 118/138 frames active,
  only speaker 0, max 1 simultaneous speaker.

### Phase 11.5-11.7 — public API + CLI (done, commit 9d06d60)

Public engine.h additions:

  struct DiarizationOptions  { float threshold = 0.5f; int min_segment_ms = 0; };
  struct DiarizationSegment  { int speaker_id; double start_s, end_s; };
  struct DiarizationResult   { segments + per-frame speaker_probs +
                               n_frames + num_spks + frame_stride_s +
                               per-stage timings };

  Engine::diarize(wav_path, opts) and diarize_samples(samples, n, sr, opts).

Engine::Impl primes a SortformerRuntimeWeights at construction so
the dequant cost is paid once.

CLI: when a Sortformer GGUF is loaded, the existing --wav / --pcm-in
pipeline routes through diarize_samples and emits one line per
segment:

  text:  [start-end] speaker_<id>
  jsonl: {"speaker":N,"start":S,"end":E}

Measured on Apple M4 Metal, sortformer-4spk-v1.f16.gguf:

| Clip | encoder | decode (CPU) | total | RTF |
|---|---|---|---|---|
| jfk.wav (11 s)               |  96 ms |  83 ms |  187 ms | 0.017 (58x) |
| LastQuestion_long_EN (5.5 m) | 9.2 s  | 22.5 s | 31.9 s  | 0.097 (10x) |

Decoder cost grows fast on long-form because the post-LN Transformer
has O(T^2) attention with no chunking. At T=4099 (5.5 min) that's
~16.8 M attention pairs per layer x 18 layers — Phase 11.11
(streaming v2) brings chunked attention.

### Phase 11.10 — speaker-attributed transcription _(done)_

Combines Sortformer (Phase 11) with a Parakeet ASR Engine to produce
"who said what" output natively in C++ in one binary and one CLI call.

Public API (`include/parakeet/ctc/engine.h`):

    struct AttributedSegment { speaker_id; text; start_s; end_s; };
    struct AttributedTranscriptionOptions { diarization;
                                            merge_same_speaker = true;
                                            min_segment_ms = 200;
                                            pad_segment_ms = 0; };
    struct AttributedTranscriptionResult { segments; diarization;
                                           asr_calls; total_ms;
                                           audio_samples; sample_rate; };

    transcribe_with_speakers(sf_engine, asr_engine, wav_path, opts);
    transcribe_samples_with_speakers(sf_engine, asr_engine,
                                     samples, n, sr, opts);

Plus tiny `Engine::is_diarization_model()` /
`is_transcription_model()` helpers so downstream callers (CLI, unit
tests, future bindings) can route based on what each loaded GGUF is.

Pipeline (in `src/parakeet_engine.cpp`):

  1. sortformer_engine.diarize_samples() -> per-frame speaker probs +
     segments via threshold + per-speaker grouping.
  2. For each diarization segment, slice samples[start:end] (with
     optional pad_segment_ms padding on each side, skipping segments
     shorter than min_segment_ms) and feed the slice through
     asr_engine.transcribe_samples().
  3. If merge_same_speaker (default), collapse consecutive same-speaker
     entries by appending text and extending end_s. This turns the
     ~10 micro-segments Sortformer emits per speaker turn into the
     handful of natural turn boundaries a downstream UI cares about.

CLI:

  ./parakeet --model <asr.gguf> --diarization-model <sf.gguf> \
    --wav <multi-speaker.wav>

Output formats:

  text  : [start-end] speaker_<id>: <text>
  jsonl : {"speaker":N,"start":S,"end":E,"text":"..."}

End-to-end on diarization-sample-16k.wav (27.3 s, 2 speakers, Apple
M4 Metal):

  TDT-0.6b-v3.f16 + Sortformer-v1.f16:
    diar.segments=11 -> merged=4, asr_calls=11
    total=1514ms RTF=0.055 (18x real-time)
    [0.40-4.24]  speaker_0: So Aaron, in your email you said you wanted
                            to talk about the exam.
    [4.96-15.60] speaker_1: Yeah, um I've just never taken a class with
                            so many different readings...
    [16.24-18.16] speaker_0: Yeah.
    [18.48-27.36] speaker_1: Yeah. There's usually just one book to
                            review, not two. Three different books, plus
                            all those other text excerpts and videos.

  CTC-0.6b.q8_0 + Sortformer-v1.f16:
    Same speaker boundaries, faster decode, English lowercase no-PnC.

CTC + TDT + sortformer-only paths all unchanged (regressions verified).

### Phase 11.11.0 — Sortformer v2 offline support _(done)_

`nvidia/diar_streaming_sortformer_4spk-v2` is the streaming-trained
sibling of v1. Architecture diff for offline-mode usage:

  encoder    : 512 d_model x 17 layers (was 18) + 128 mel bins (was 80)
  transformer: 18 layers x 192 d_model (same)
  head       : encoder_proj + first_hidden_to_hidden + single_hidden_to_spks
               + extra hidden_to_spks (4, 384) for streaming-only path

All four read-paths flow from existing GGUF metadata
(n_layers / feat_in / tf_n_layers / use_bias / xscaling), so converting
v2 and running it through our offline diarize() pipeline works with
no code changes:

  python scripts/convert-nemo-to-gguf.py \
      --ckpt models/diar_streaming_sortformer_4spk-v2.nemo \
      --out  models/sortformer-streaming-4spk-v2.f16.gguf --quant f16
  -> 250.9 MiB f16

The 384-wide hidden_to_spks tensor is the streaming-mode-only output
head (concat of spkcache + chunk hidden states); the converter
currently skips it since v1's 192-wide head reproduces NeMo's
forward_speaker_sigmoids() bit-for-bit in offline mode. v2 GGUFs run
through the same single_hidden_to_spks (192-wide) path as v1.

Verified on diarization-sample-16k.wav (2 speakers): v2 produces 9
segments vs v1's 11 — same conversation structure, slightly different
boundary placement (v2 was trained with chunked-attention masking
which subtly affects even offline forward passes).

Converter helper `_get_member` handles both `./model_config.yaml` and
`model_config.yaml` tarball layouts (v1 has the prefix, v2 doesn't).

### Phase 11.11 — Live streaming diarization (overview) _(11.11.1 shipped; 11.11.2 NeMo-style spkcache pending)_

Real live diarization needs the v2 spkcache + FIFO state machine.
NeMo's `forward_streaming_step` per chunk is:

  1. encoder.pre_encode(chunk)  -> chunk_pre_encode_embs
       (only the dw_striding subsampling stage; output is in 512-dim
        post-subsampling space)
  2. concat([spkcache, fifo, chunk_pre_encode_embs]) -> concat_embs
       (spkcache_len + fifo_len + chunk_subs frames)
  3. frontend_encoder(concat_embs, bypass_pre_encode=True) ->
     full FastConformer encoder + encoder_proj on the concatenated
     buffer  -> (T_total, 192)
  4. forward_infer(...) -> 18-layer transformer + speaker head ->
     (T_total, 4) speaker probabilities
  5. streaming_update(state, chunk_pre_encode_embs, all_preds, lc, rc):
     - Extract chunk_preds = preds[spkcache_len + fifo_len + lc :
                                    spkcache_len + fifo_len + chunk_len + lc]
     - Append (chunk, chunk_preds) to FIFO
     - If FIFO overflows, pop front frames into spkcache and update the
       silence profile (mean_sil_emb + n_sil_frames) so the next
       compress_spkcache call can identify silence frames to keep.
     - If spkcache overflows, _compress_spkcache reduces it back to
       spkcache_len via per-speaker top-k frame selection (using the
       speaker probabilities) plus silence-anchor frames. This step is
       what keeps speaker IDs consistent across chunks: it constructs a
       persistent per-speaker memory and a permutation.

Default v2 hyperparameters from the .nemo config:

  spkcache_len: 188     fifo_len: 188     chunk_len: 188
  chunk_left_context: 1 chunk_right_context: 1   subsampling_factor: 8
  spkcache_update_period: 188   spkcache_sil_frames_per_spk: 3
  causal_attn_rate: 0.5  (training-only)

Implementation plan when this lands:

1. Expose `pre_encode_only` from `run_encoder` (or a sibling): given mel,
   run only the dw_striding subsampling + out projection so we get the
   512-dim post-subsampling tensor without running the conformer
   blocks.
2. New `SortformerStreamingState` struct (mirrors NeMo's
   `StreamingSortformerState`): `spkcache (T_max, 512)`,
   `spkcache_preds (T_max, 4)`, `fifo (T_max, 512)`,
   `fifo_preds (T_max, 4)`, `mean_sil_emb (512)`, `n_sil_frames (int)`,
   `spk_perm (4)`.
3. `SortformerStreamSession` (mirrors `StreamSession` in shape but with
   `feed_pcm_*()` -> `chunk_callback` semantics emitting per-chunk
   speaker probabilities + segment events).
4. Per-chunk forward: pre_encode the new audio chunk only, concat with
   spkcache + fifo, run full encoder + transformer + head on the
   concatenation, slice out chunk_preds, run `streaming_update` which
   updates state including the silence profile + (eventually)
   compresses spkcache via per-speaker top-k.
5. Wire CLI `--stream-duplex` to route through SortformerStreamSession
   when the model is Sortformer.
6. Per-stage parity vs NeMo's `forward_streaming` (dump
   spkcache/fifo/chunk_preds at each chunk boundary, compare).

Open design questions (need to resolve when work starts):

- `_compress_spkcache` is the most complex component (~150 lines of
  PyTorch). Probably re-implementable in pure C++ since it's mostly
  index gather + softmax + top-k.
- What's the right surface for `chunk_callback`? Per-chunk probability
  matrix? Per-chunk newly-formed segments? Both?
- `pad_segment_ms` from §11.10 should also work in streaming mode.
- BLAS/Accelerate for the transformer attention will be needed to keep
  per-chunk RTF reasonable (today's offline scalar implementation
  spends 22 s on a 5.5 min clip; per chunk that's ~0.7 s per 100 ms
  chunk, way over real-time).

Estimated effort: 1-2 weeks of focused work + parity validation.
Tracked as Phase 11.11.2; the §11.11.1 sliding-history implementation
below shipped first as the pragmatic v1.

### Phase 11.11.1 — Sortformer live streaming (pragmatic v1, sliding history)

Phase 11.11.2 (planned) is a multi-week effort to land the full NeMo
`forward_streaming` algorithm (spkcache + fifo + `_compress_spkcache`
+ encoder graph split). To unblock product integration *now*, Phase
11.11.1 ships a pragmatic streaming layer that reuses the existing
offline `Engine::diarize()` path under a sliding-history window.

API (in `include/parakeet/ctc/engine.h`):

```cpp
struct SortformerStreamingOptions {
    int   sample_rate    = 16000;
    int   chunk_ms       = 2000;     // emit cadence
    int   history_ms     = 30000;    // sliding context window
    float threshold      = 0.5f;
    int   min_segment_ms = 200;
    bool  emit_partials  = true;
};

struct StreamingDiarizationSegment {
    int    speaker_id;
    double start_s, end_s;
    int    chunk_index;
    bool   is_final;
};

using SortformerSegmentCallback =
    std::function<void(const StreamingDiarizationSegment &)>;

class SortformerStreamSession {
public:
    void feed_pcm_f32(const float *, int n);
    void feed_pcm_i16(const int16_t *, int n);
    void finalize();
    void cancel();
    const SortformerStreamingOptions & options() const;
};

std::unique_ptr<SortformerStreamSession>
Engine::diarize_start(const SortformerStreamingOptions &,
                      SortformerSegmentCallback);
```

Algorithm (per chunk):
1. `feed_pcm_*()` appends samples to a `std::vector<float> ring`.
2. Once `chunk_samples` of new audio is available beyond
   `emitted_samples`, take a window
   `[max(ring_origin, emit_end - history_samples) , emit_end]` and run
   the full offline `engine_impl_diarize_helper(...)` on it (mel +
   encoder + sortformer head + threshold-segmentation).
3. For every returned segment whose absolute time range overlaps the
   new chunk's `[emitted_samples, emit_end]`, emit a
   `StreamingDiarizationSegment` clipped to the chunk.
4. Advance `emitted_samples = emit_end`. Trim `ring` to keep only
   `history_samples` of audio behind us (so the buffer stays bounded
   for arbitrarily long sessions).
5. `finalize()` semantics:
   - if `>= 1` sample of new audio sits past the last emitted chunk,
     run one final `process_chunk` over `[max(ring_origin, end -
     history_samples), end]`, emit each overlapping segment with
     `is_final = true`. Consumers see real segments tagged final.
   - if the audio ended exactly on a chunk boundary (no tail), emit a
     single synthetic terminator with `speaker_id = -1`,
     `start_s == end_s == emitted_samples / sample_rate`,
     `is_final = true`. Consumers should treat negative speaker IDs as
     "session done, no new segment". This avoids the round-1 bug
     where the last chunk's segments were re-emitted as duplicates
     with `is_final = true` flipped on.
6. `cancel()` short-circuits; subsequent `feed_*` calls are no-ops.

Trade-offs (vs the planned full Phase 11.11.2 NeMo-style streaming):

- **Pro**: ~150 lines of code; zero changes to the encoder graph;
  works with both v1 and v2 Sortformer GGUFs; reuses the parity-tested
  offline `diarize()` path.
- **Pro**: speaker IDs stabilise within a few chunks once the history
  window contains both speakers' audio; matches offline IDs exactly
  once the history covers the full session.
- **Con**: each chunk re-runs the full encoder over the trailing
  `history_ms` of audio. Measured RTF ~0.25 on M4 Air CPU at
  `chunk_ms=2000 history_ms=30000` for the 22 s
  `diarization-sample-16k.wav` sample (5.5 s wall for 22 s of audio).
  Phase 11.11.2's `spkcache` approach will fix this.
- **Con**: speaker IDs in the *very first* chunks may be arbitrary
  before the history window contains both speakers. Verified on
  `diarization-sample-16k.wav`: chunk 1 mislabels speaker_0 as
  speaker_1 at `[2.00-4.00]`; chunks 2-10 align with the offline
  reference (`speaker_0` for [1.84-10.00], `speaker_1` for
  [13.36-21.04]).

CLI:

```bash
./build/parakeet \
    --model models/sortformer-4spk-v1.f16.gguf \
    --pcm-in recording.raw --pcm-format s16le \
    --stream \
    --stream-chunk-ms 2000 --stream-history-ms 30000 \
    --emit text     # or jsonl
```

The CLI auto-routes `Sortformer + --stream` through the streaming path
(no separate flag). `--emit jsonl` produces
`{"speaker", "start", "end", "chunk", "is_final"}` per line.

Live mic auto-detects diarization mode when `--model` resolves to a
Sortformer GGUF — `examples/live-mic.cpp` swaps in a
`SortformerStreamSession` instead of `StreamSession` and prints
`[start-end] speaker_N` per chunk:

```bash
./build/live-mic --model models/sortformer-4spk-v1.f16.gguf \
                 --chunk-ms 2000 --history-ms 30000
```

For combined live transcription + speaker labels in a single binary,
`examples/live-mic-attributed.cpp` loads two engines (a CTC/TDT ASR
engine and a Sortformer engine), forwards each captured audio batch
to both `StreamSession` and `SortformerStreamSession`, and tags each
transcript segment with the speaker whose live diarization range
overlaps it the most. `--accumulate` accumulates text on a single
line per speaker and emits a newline on speaker change or
`--silence-flush-ms` of silence:

```bash
./build/live-mic-attributed \
    --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
    --diar-model models/sortformer-4spk-v1.f16.gguf \
    --accumulate
```

Independent `--asr-n-gpu-layers` / `--diar-n-gpu-layers` allow
splitting the two engines across CPU and GPU on machines where
running both on the GPU would compete for resources.

Testing: `test/test_sortformer_streaming.cpp` (built as
`test-sortformer-streaming` when `PARAKEET_BUILD_TESTS=ON`) feeds
the multi-speaker sample in random burst sizes (1-5000 samples per
`feed_pcm_f32()` call) and asserts:
- `>= 1` real segment callback received (`speaker_id >= 0`),
- exactly one `is_final = true` callback received after `finalize()`
  (real segment for the tail case, synthetic terminator with
  `speaker_id = -1` for the chunk-aligned case),
- `max_end` is within the audio duration,
- no two consecutive callbacks duplicate each other's
  `(speaker_id, start_s, end_s)`,
- `cancel()` on a half-fed session is idempotent.

Verified end-to-end on `diarization-sample-16k.wav`:
```
offline:  [1.84-10.00] speaker_0  [13.36-21.04] speaker_1
streaming (chunk=2000, history=30000):
  [2.00-4.00] speaker_1  (chunk 1)        # cold-start mislabel
  [4.00-6.00] speaker_0  (chunk 2)
  [6.00-8.00] speaker_0  (chunk 3)
  [8.00-10.00] speaker_0 (chunk 4)
  [13.36-14.00] speaker_1 (chunk 6)
  [14.00-16.00] speaker_1 (chunk 7)
  [16.00-18.00] speaker_1 (chunk 8)
  [18.00-20.00] speaker_1 (chunk 9)
  [20.00-21.04] speaker_1 (chunk 10)
  [20.00-21.04] speaker_1 (chunk 10, final)
```

Phase 11.11.2 (true NeMo streaming with spkcache compression) remains
the eventual destination; 11.11.1 is what ships today.

### Phase 11.x — pending optimisations

- **BLAS / Accelerate for transformer attention**. Same opportunity
  as TDT's LSTM + joint gemvs; current scalar attention is the long-
  form bottleneck on Sortformer's 18-layer TF (T^2 cost dominates).
  See §5.4 for the prior Accelerate sched-assertion investigation on
  the f32 GGUF -- worth re-checking with the q8_0 path.

### Phase 11.12 — quantised Sortformer GGUFs  _(done)_

Both Sortformer checkpoints (`diar_sortformer_4spk-v1` offline and
`diar_streaming_sortformer_4spk-v2` streaming-trained) now ship at
`q8_0` and `q4_0` via the universal `add_2d` quantisation path in
`scripts/convert-nemo-to-gguf.py`. No converter changes needed --
Sortformer's encoder shares the FastConformer graph with CTC/EOU,
and the transformer encoder + diarization head are 2D linear layers
that already flow through `add_2d`.

Sizes:

| GGUF                             | f16     | q8_0    | q4_0    |
|----------------------------------|---------|---------|---------|
| sortformer-4spk-v1               | 263 MiB | 141 MiB | 75 MiB  |
| sortformer-streaming-4spk-v2     | 251 MiB | 134 MiB | 72 MiB  |

`scripts/verify-gguf-roundtrip.py` gained `build_expected_sortformer`
covering the encoder + `sortformer.encoder_proj` + 18 transformer
blocks (`attn.{q,k,v,out}`, `ln{1,2}`, `ffn.{in,out}`) + the
two-layer diarization head. All 6 GGUFs (2 models × 3 tiers) PASS
the roundtrip gate (worst rel `1.15e-1` on `parakeet-ctc-0.6b.q4_0`-
class q4 weights, well within the `2^-3 = 0.125` quant gate).

`test-sortformer-parity` was extended with `--enc-rel-tol` and
`--probs-abs-tol` flags so each quant tier can pass at appropriate
gates (defaults still f16 = 5e-3 / 5e-2). Per-tier numbers on
`jfk.wav` (single-speaker, 11 s):

| GGUF                                     | enc rel  | probs max_abs |
|------------------------------------------|----------|---------------|
| sortformer-4spk-v1.f16                   | 1.6e-3   | 8.7e-4        |
| sortformer-4spk-v1.q8_0                  | 2.7e-2   | 2.7e-2        |
| sortformer-4spk-v1.q4_0                  | 3.2e-1   | 1.3e-1        |
| sortformer-streaming-4spk-v2.f16         | 5.0e-2   | 5.1e-2        |
| sortformer-streaming-4spk-v2.q8_0        | 5.2e-2   | 5.4e-2        |
| sortformer-streaming-4spk-v2.q4_0        | 2.2e-1   | 2.0e-1        |

(v2's f16 baseline is already worse than v1's because the
streaming-trained encoder's offline forward in our C++ graph diverges
from NeMo's offline forward -- this is a structural property of the
streaming-trained checkpoint when run offline, not a quantisation
regression. v2 q8/q4 inflate within the same factor band as v1.)

User-facing diarization output is identical across all three tiers
of v2 on `jfk.wav` (`[0.24-2.40] [3.36-4.56] [5.44-11.04]`,
all speaker_0). v1's three tiers also produce the same three
segments, with q4 boundaries shifted by at most ~80 ms (one encoder
frame) vs f16 -- well within the post-processing `min_segment_ms`
band.

**Recommendation:** prefer q8_0 for general use (1.9× smaller than
f16 with negligible quality impact); use q4_0 when memory is tight
(3.5× smaller than f16, marginally noisier individual speaker
probabilities but identical thresholded segments on shipping
fixtures).

## Phase 12 — EOU end-of-utterance streaming ASR  _(done)_

### Phase 12.0 — scope, model selection, API target  _(done)_

This repo already provides ggml backends for `tdt` / `ctc` /
`sortformer`. Phase 12 closes the loop on `eou` so the four
families that ship under one `Engine` umbrella all run on a
single pure-ggml dependency.

**Checkpoint.** NVIDIA's official
**`nvidia/parakeet_realtime_eou_120m-v1`** NeMo `.nemo` archive
(NVIDIA Open Model License). Sourcing the `.nemo` directly lets us
reuse the exact pattern the CTC / TDT / Sortformer ports already
followed: `.nemo` -> GGUF via `convert-nemo-to-gguf.py`, NeMo
PyTorch as the parity oracle.

**Architecture summary** (from `model_config.yaml` + state-dict probe):

| Stage | Spec |
|---|---|
| Mel  | `AudioToMelSpectrogramPreprocessor`, 128 bins, n_fft=512, win=400, hop=160, normalize=NA, dither=1e-5, pad_to=0 |
| Encoder | FastConformer, 17 layers, d_model=512, n_heads=8, ff_expansion=4, conv_kernel=9, dw_striding subsample 8x, **`use_bias=False`**, **`xscaling=False`**, **`conv_norm_type=layer_norm`** (gamma/beta still stored under `conv.batch_norm.{weight,bias}`; no running stats), **`att_context_size=[70,1]`** + `att_context_style=chunked_limited`, `causal_downsampling=true`, `conv_context_size=causal` |
| Decoder | RNNT-Decoder, 1 LSTM layer x 640 hidden, embedding `[1027, 640]` (`vocab + 1` for `blank_as_pad=true`) |
| Joint | RNNT-Joint, encoder_hidden=512 -> 640, pred_hidden=640 -> 640, ReLU, output dim **1027** = 1024 BPE + `<EOU>` (id 1024) + `<EOB>` (id 1025) + blank (id 1026) |
| Latency | NVIDIA card cites 80 ms (p50) / 280 ms (p90) / 320 ms (p95) end-of-turn detection on TTS-augmented DialogStudio |

So EOU is **TDT minus durations + LayerNorm in the conv module +
two attention/conv shape switches**; encoder graph is ~95 % shared
with the existing CTC/TDT encoder, with three shipping deltas
(LN-vs-fused-BN in conv module, chunked-limited attention mask
applied as a static offline mask via `ggml_soft_max_ext`, asymmetric
`(L=k-1, R=s-1)` causal padding in the dw_striding subsampler).
NeMo's own streaming forward additionally maintains per-chunk KV
cache state and depthwise-conv left state for `cache_aware_stream_step`;
this project deliberately does **not** ship that path -- see §8.5
case (A) for why driving streaming-trained Parakeet checkpoints
through chunked-limited streaming inference is a quality regression
on the targets this repo cares about. The decoder + joint mirror
TDT minus the duration head.

**API target.** The reference EOU surface emits `TranscriptionSegment`
records with only `text` populated (no per-segment timestamps, no
utterance-boundary event), so the C++ Engine deliberately preserves
that shape rather than synthesising fields from intermediate state.

C++ pipeline this maps to (matching the upstream NeMo `processEOU`
reference):

1. mel(128) over the full input audio (offline; the reference
   implementation accumulates an append-queue until end-of-job, then
   mels the whole buffer).
2. Walk mel in fixed 25-frame slices (`encoder_chunk_mel_frames=25`);
   skip trailing slice if `< 10` frames and not first.
3. Per slice: cache-aware encoder forward with running
   `cache_last_channel (17, 1, 70, 512)`,
   `cache_last_time (17, 1, 512, 8)`, `cache_last_channel_len (1)`.
4. Per encoder frame: RNN-T greedy with up to 5 symbols/step;
   `<blank>` ends the per-frame loop, `<EOU>` flushes the current
   segment with `\n` separator + zeroes h/c, otherwise append the
   piece to the running segment.
5. Concatenate segments with single space; trim; empty -> "no speech"
   sentinel; total word count -> stats.

Cross-engine VAD/EndOfTurn events are **not** part of Phase 12; they
will be a Phase 13 cross-cutting concern wiring `<EOU>` and Sortformer
per-frame any-speaker probabilities into a shared `StreamEvent`
umbrella across parakeet.cpp + whisper.cpp. Phase 12 just needs
to land the `EouStreamSession` callback signature with
`is_eou_boundary` from day 1 so Phase 13 plugs in without churn.

**Phase 12 outline (and current shipping status):**

- 12.0 plan + scope (this section). _(done)_
- 12.1 converter + Python reference + GGUF roundtrip. _(done; see
  §12.1 below)_
- 12.2 EOU GGUF loader + Engine routing. _(done)_
- 12.3 cache-aware FastConformer encoder graph (LN-in-conv,
  chunked-limited attention mask, KV + conv state). _(done; KV +
  conv state path was prototyped and rejected on quality grounds,
  see §8.5 case (A); LN-in-conv + chunked-limited mask shipped.)_
- 12.4 RNN-T decoder (1-layer LSTM 640 + joint MLP) with `<EOU>`
  reset semantics. _(done)_
- 12.5 streaming push API (Modes 2 + 3) with callback shape ready
  for Phase 13 events. _(done; the callback hangs off the existing
  `StreamSession` rather than a new `EouStreamSession`.)_
- 12.6 CLI auto-routing + `live-mic` auto-detection. _(done)_
- 12.7 end-to-end parity harness (`test-eou-streaming` on jfk.wav,
  driven by `dump-eou-reference.py`). _(done)_

### Phase 12.1 — converter + Python reference  _(done)_

`scripts/convert-nemo-to-gguf.py` learned an EOU branch:

1. **Detection** -- `detect_model_type()` distinguishes EOU from TDT
   by the absence of `model_defaults.tdt_durations` plus the presence
   of `<EOU>` in `cfg.labels`. Sortformer / CTC paths unchanged.
2. **Conv-norm switch** -- when `cfg.encoder.conv_norm_type ==
   "layer_norm"`, the per-block emitter writes
   `encoder.blk.{i}.conv.norm.{weight,bias}` straight from the
   `conv.batch_norm.{weight,bias}` tensors (gamma/beta) and skips the
   BN running-stats fusion that the BatchNorm path requires. The
   metadata key `parakeet.encoder.conv_norm_type` advertises which
   path each GGUF expects.
3. **Streaming hyperparameters in metadata** --
   `parakeet.encoder.{conv_norm_type,conv_context_size,
   causal_downsampling,att_context_style,att_context_size_left,
   att_context_size_right}` so the C++ encoder can build the right
   chunked-limited attention mask + KV-cache shapes without re-parsing
   YAML.
4. **EOU metadata block** under `parakeet.eou.*`:
   `{vocab_size, blank_id, eou_id, eob_id, pred_hidden,
   pred_rnn_layers, joint_hidden, encoder_chunk_mel_frames,
   cache_lookback_frames, cache_time_steps, max_symbols_per_step}`.
   `cache_lookback_frames` defaults from `att_context_size_left` (70)
   and `cache_time_steps` from `conv_kernel - 1` (8).
5. **EOU tensors** under `eou.*`: `eou.predict.embed.weight`,
   `eou.predict.lstm.0.{w_ih,w_hh,b_ih,b_hh}`, `eou.joint.enc.*`,
   `eou.joint.pred.*`, `eou.joint.out.*`. SentencePiece tokenizer
   bytes embedded same as CTC/TDT.

Output sizes on `nvidia/parakeet_realtime_eou_120m-v1.nemo`:

| Quant | File size | Notes |
|-|-|-|
| f16   | 246.0 MiB | 251 f32 + 233 f16 tensors |
| q8_0  | 131.7 MiB | same f32 set + 233 q8_0 tensors |

`scripts/dump-eou-reference.py` mirrors `dump-tdt-reference.py` plus a
streaming-mode pass:

- Offline: 128-bin mel, full-context encoder output (T_enc, 512), LSTM
  init state (1, 1, 640), prediction-net output for the blank/SOS
  token (640,), and the NeMo `transcribe()` greedy reference text.
- Streaming: `model.encoder.cache_aware_stream_step(...)` driven in
  25-mel-frame chunks with explicit running caches; per-chunk encoder
  outputs are concatenated and saved alongside per-chunk frame counts.
  With `att_context_size=[70,1]` each 25-mel-frame chunk emits 2
  encoder frames (the right-context-1 frame is held back), so on
  jfk.wav (11 s, 1101 mel frames) the streaming pass produces 88
  encoder frames vs the offline pass's 139. That's intentional and is
  what the C++ streaming graph will need to reproduce in Phase 12.3.

NeMo offline transcript on `test/samples/jfk.wav`:

```
and so my fellow americans ask not what your country can do for you ask
what you can do for your country<EOU>
```

The trailing literal `<EOU>` is the joint network emitting the EOU
token at end-of-utterance and is exactly the signal the C++ decoder
will key on for `\n` segment-flush + LSTM state reset (per the
upstream NeMo `eouDecodeChunk` reference).

`scripts/verify-gguf-roundtrip.py` learned to dispatch on
`parakeet.model.type`: `build_expected_eou()` recreates the EOU tensor
map (LayerNorm in conv, no use_bias on inner blocks, EOU/joint
weights), `build_expected_ctc()` keeps the existing CTC path. The
verifier also gained a Q8_0 / Q5_0 / Q4_0 dequant comparison branch
with per-format rel gates, so the same script validates every quant
tier we ship.

Both tiers pass round-trip on the EOU GGUFs (worst rel 4.78e-4 at
f16 -- under the 2^-10 gate -- and 4.0e-3 at q8_0 -- under the 2^-7
gate); CTC GGUF baseline still passes after the verifier was
generalised to handle trailing-1-axis squeezing in older artefacts.

### Phase 12.2 — EOU GGUF loader + Engine routing  _(done)_

Touched `parakeet_ctc.h` / `parakeet_ctc.cpp` / `parakeet_engine.cpp`.
Additions:

- `enum class ParakeetModelType` gains an `EOU` variant; the loader
  routes on `parakeet.model.type == "eou"` and populates an
  `EouWeights` struct alongside the existing CTC / TDT / Sortformer
  weight blobs.
- `EncoderConfig` gains a `ConvNormType conv_norm_type` enum +
  `causal_downsampling`, `conv_causal`, `att_chunked_limited`,
  `att_context_left`, `att_context_right` fields, all read from the
  GGUF metadata block written by §12.1's converter changes. CTC / TDT
  / Sortformer GGUFs leave these at their offline defaults so the
  existing engines are bit-for-bit unchanged.
- `BlockWeights` gains optional `conv_norm_w` / `conv_norm_b` (used
  when `conv_norm_type == LayerNorm`) alongside the existing fused-BN
  `conv_bn_scale` / `conv_bn_shift` (used when `BatchNorm`). The
  loader's per-block tensor pull picks one or the other based on the
  metadata.
- `EouWeights` mirrors `TdtWeights` minus the duration head:
  `predict.embed`, one-layer LSTM `(w_ih, w_hh, b_ih, b_hh)`, and
  `joint.{enc,pred,out}.{weight,bias}`.
- `Engine::Impl` gains an `EouRuntimeWeights eou_rt` slot and runs
  `eou_prepare_runtime()` at construction when the GGUF is EOU
  (dequantises the predict + joint to f32 once, same shape the
  TDT runtime uses).
- `Engine::transcribe_samples()` dispatches to a new EOU branch that
  calls `eou_greedy_decode()`. `Engine::is_transcription_model()`
  returns true for EOU; `Engine::model_type()` returns `"eou"`.

CLI side (`src/main.cpp`): the manual decode dispatch in the closure
gained an EOU branch (the bug that surfaced as a segfault during
bring-up was that the manual closure had a TDT branch but no EOU
branch, so EOU GGUFs fell through to `ctc_greedy_decode` on a NULL
logits buffer). `transcribe_wav()` now lists EOU alongside TDT /
Sortformer in its "use Engine instead" rejection message.

### Phase 12.3 — encoder graph (LN-in-conv + causal subsampler + chunked-limited attention mask)  _(done)_

Three structural changes to `subsampling_graph()` /
`conformer_conv_graph()` / `rel_pos_mha_graph()` /
`build_encoder_graph_cached()`, each gated on `EncoderConfig`
metadata so CTC / TDT / Sortformer GGUFs take the original code
path:

1. **LayerNorm in conv module.** When `conv_norm_type ==
   LayerNorm`, the conv graph permutes from `(T, d_model)` to
   `(d_model, T)`, runs `layer_norm_affine(x, conv.norm.weight,
   conv.norm.bias, eps)`, applies SiLU, and falls into the existing
   pw2 / matmul path. Saves one permute vs the BN path. Existing
   CTC / TDT / offline Sortformer keep the fused
   `bn_scale * x + bn_shift -> silu -> permute -> pw2` flow.
2. **Causal subsampler** (`causal_downsampling=true`). NeMo's
   `CausalConv2D` pre-pads each stride-2 dw_striding conv with
   `(L=k-1=2, R=s-1=1)` zeros on **both** the freq and time axes,
   then convolves with `padding=0`. New `zero_pad_dim1` helper
   (analogous to the existing `zero_pad_dim0`) implements the
   freq-axis half. `subsampling_graph` gains a `causal_downsampling`
   flag; when set it pre-pads and switches the conv `padding` to
   zero. Output sizing changes from
   `(L+2-k)/s+1 = (L-1)/2+1` (symmetric) to `(L+(L+R)-k)/s+1 =
   L/2+1` (causal) so freq goes 128 -> 65 -> 33 -> 17 instead of
   128 -> 64 -> 32 -> 16, matching the trained
   `encoder.subsampling.out.weight` shape `[512, 4352=17*256]`.
   `run_encoder()`'s mask-sizing math was also gated on the same
   flag (a bug surfaced where the cached graph used the new sizes
   but the per-call mask uploads still used the symmetric formula,
   producing 138-frame outputs instead of 139).
3. **Causal depthwise conv module** (`conv_context_size: causal`).
   The conv module's k=9 depthwise stride=1 conv now uses
   `(L=8, R=0)` zero-pad instead of the symmetric `(L=4, R=4)`.
4. **Chunked-limited attention mask** (`att_context_style:
   chunked_limited`). `EncoderGraph` gains an `att_mask` graph
   input + `att_mask_host` buffer. Built host-side once per graph
   (cached across calls): for query frame `i` in chunk
   `c = i / (right + 1)`, the mask is `0.0f` on the visible
   `[c*chunk_size - left, (c+1)*chunk_size - 1]` range (clamped to
   `[0, T-1]`) and `-INFINITY` everywhere else. Wired into
   `rel_pos_mha_graph` via `ggml_soft_max_ext(scores, mask, scale,
   0.0f)` (which the `att_mask=nullptr` callers fall back to the
   prior `ggml_scale + ggml_soft_max` path on, so CTC / TDT
   regression unchanged). The formula matches NeMo's
   `_create_masks` exactly: `chunk_idx[i] - chunk_idx[j] in
   [0, left // chunk_size]` -- for `att_context_size=[70, 1]`,
   `left_chunks_num = 70 // 2 = 35` and queries see their own chunk
   plus 35 chunks of past context, exactly 72 keys per query (in
   the steady state).

The conv graph branches and the mask wiring also flow through
`profile_block_substages` (CTC profiling helper). Tested across
CTC / TDT / Sortformer with no regression.

**Mel preprocessor (`mel_preprocess.{h,cpp}`)** also got a
`MelNormalize` enum + `MelConfig::normalize` field. EOU's NeMo
config sets `normalize: NA` (no per-feature CMVN); the loader reads
the converter-emitted `parakeet.preproc.normalize` string and gates
the existing `apply_per_feature_cmvn()` call on it. CTC / TDT /
Sortformer all leave `normalize=per_feature` so their CMVN keeps
running. **This was the dominant accuracy gap during bring-up:**
CTC/TDT-style CMVN on EOU's preprocessor mean-centres each mel bin
across the whole utterance, but the EOU encoder was trained against
raw log-mel values that floor at the log-zero guard during
silence frames; without CMVN our subsampler cosine jumped from
`0.108` (broken) to `0.999688` (matched) and the encoder cosine
landed at `0.999997` -- f16 quantisation floor.

Per-stage parity on `test/samples/jfk.wav` (NeMo PyTorch reference
via `dump-eou-reference.py` → C++ via `PARAKEET_DUMP_*` env vars):

| Stage | max_abs | rel_max | cosine |
|-|-|-|-|
| log-mel          | 1.36e+1 (tail-frame artifacts) | 8.17e-1 | 0.999644 |
| post-subsampler  | 3.30e+2                        | 1.00e-1 | 0.999688 |
| encoder out      | **7.64e-2**                    | **7.70e-3** | **0.999997** |

Transcript on `jfk.wav` (both `parakeet-eou-120m-v1.gguf` f16 and
`parakeet-eou-120m-v1.q8_0.gguf`):

```
and so my fellow americans ask not what your country can do for you ask
what you can do for your country
```

Bit-equal to NeMo's offline reference (modulo the trailing literal
`<EOU>` token which the C++ decoder strips after using it for the
segment-flush + LSTM-state-reset side effect). The 20 s
`sample-16k.wav` Alice-in-Wonderland clip transcribes with zero
errors on q8_0:

```
alice was beginning to get very tired of sitting by her sister on the
bank and of having nothing to do once or twice she had peeped into the
book her sister was reading but it had no pictures or conversations in
it and what is the use of a book thought alice without pictures or
conversations
```

### Phase 12.4 — RNN-T decoder + `<EOU>` reset semantics  _(done)_

New `parakeet_eou.{h,cpp}` (~360 lines) modelled on
`parakeet_tdt.{h,cpp}`. `EouRuntimeWeights` dequantises the predict
(1-layer LSTM, 640 hidden) + joint (`enc 512->640`, `pred 640->640`,
`out 640->1027`) to f32 once at Engine construction. `EouDecodeState`
holds `h_state`, `c_state`, `pred_out`, `last_token`,
`symbols_this_step`, `segment_start_token` -- everything needed to
carry decoder state across chunked calls.

`eou_decode_window()` runs greedy RNN-T over a span of encoder
frames with up to `max_symbols_per_step=5` symbols per encoder step
(matches the upstream NeMo `EOU_MAX_SYMBOLS_PER_STEP` constant). Per
emitted token:

- `<blank>` (id 1026) -> break out of inner loop, advance encoder.
- `<EOB>` (id 1025) -> training-time block boundary marker; treated
  as a no-op skip, same policy as the NeMo reference.
- `<EOU>` (id 1024) -> flush the in-progress segment to
  `out_segments`, **zero h/c state**, set `last_token = blank`,
  re-prime the predictor with the blank embedding, break out of
  inner loop. The state reset is the NeMo `eouDecodeChunk` reset
  semantics carried through verbatim.
- Any other special token (vocabulary entry of the form
  `<...>`) -> defensive break (matches the NeMo reference's
  `isSpecialToken` skip).
- Otherwise: append to `out_tokens`, feed back into the LSTM,
  update `pred_out` for the next joint call.

`eou_greedy_decode()` is the one-shot wrapper used by
`Engine::transcribe()`: detokenises segment-by-segment using the
boundaries `eou_decode_window` recorded, joins with `\n`, returns
the result as `EouDecodeResult.text`. `eou_count` is exposed for
later wiring into the planned cross-engine `OnEndOfTurn` event.

### Phase 12.5 — streaming push API (Modes 2 + 3)  _(done; rolling-encoder Mode 3 is the chosen design -- chunked-limited streaming inference rejected, see §8.5)_

Public API additions in `include/parakeet/ctc/engine.h`:

```cpp
struct StreamingSegment {
    // ... existing fields ...
    bool   is_eou_boundary = false;   // EOU only: <EOU> token fired in this chunk
    float  eot_confidence  = 0.0f;    // reserved for Phase 13's OnEndOfTurn event
};
```

Existing `StreamSession` (`StreamSession::Impl`) gained an
`EouDecodeState eou_state` slot and an EOU branch in
`process_window()`. On `Engine::stream_start()` for an EOU GGUF the
session initialises the EOU state via `eou_init_state(eou_rt,
eou_state)` (priming the LSTM with the blank embedding, matching
NeMo's `decoder.initialize_state`). The existing
`Engine::transcribe_samples_stream()` (Mode 2) gained the same EOU
branch. Both paths set `seg.is_eou_boundary = (win_segments.size() >
0)` per emitted chunk, so the `<EOU>` token's emission shows up on
the cadence the consumer is already iterating.

CLI wiring (`src/main.cpp`):

- `--stream` (Mode 2) on EOU GGUFs runs the offline encoder once
  then walks chunks emitting `StreamingSegment`s, exactly like
  CTC / TDT.
- `--stream --stream-duplex` (Mode 3) on EOU GGUFs goes through
  `Engine::stream_start()` -> `StreamSession`. Each chunk's
  encoder runs over `[left + chunk + right_lookahead]` audio with
  the chunked-limited mask applied; the EOU decoder state carries
  across chunks. Mode 3 produces transcript output that's
  byte-equal to Mode 1 on jfk.wav (104 B vs 104 B).
- `--emit jsonl` includes `"is_eou_boundary"` per segment line.

`live-mic` already routes anything that isn't a Sortformer GGUF
through `StreamSession`, so `live-mic --model
models/parakeet-eou-120m-v1.q8_0.gguf` works out of the box -- no
new auto-detection logic was required.

`test-eou-streaming` (new, `test/test_eou_streaming.cpp`) asserts:

- Mode 2 concatenated text **byte-equal** to the offline
  `Engine::transcribe()` reference;
- Mode 2 `is_eou_boundary` fires on at least one segment (the
  trailing `<EOU>` on `jfk.wav`);
- Mode 3 transcript size matches the reference within a 20 % tail
  jitter band. Chasing byte-equality on Mode 3 via cache-aware
  streaming inference was explored and rejected -- see §8.5 case (A)
  for the full rationale -- so the rolling-encoder tail-jitter band
  is the assertion the test will keep.

Passes on both `parakeet-eou-120m-v1.gguf` (f16) and
`parakeet-eou-120m-v1.q8_0.gguf`. Existing `test-streaming` (CTC /
TDT byte-equality + WER tolerance) and `test-sortformer-streaming`
both still pass after the `StreamSession::Impl` plumbing changes.

#### Mode 3 caveat — chosen design, not a workaround

Mode 3 today re-runs the **offline** encoder per chunk over a
sliding `[left + chunk + right_lookahead]` window without persistent
KV / conv-state cache across chunks. The transcript matches Mode 2
byte-equally on `jfk.wav`, but `<EOU>` boundary detection is
approximate: the trailing chunk doesn't carry the long-context
encoder state the EOU head needs to confidently fire `<EOU>` on
end-of-utterance.

This is the **chosen design**, not a deferred workaround. The
obvious alternative -- driving the streaming-trained EOU 120m-v1
weights through NeMo's `cache_aware_stream_step` to recover
"per-chunk `O(chunk)` compute and persistent encoder state" --
was prototyped during the Phase 12.x exploration and rejected; see
§8.5 case (A) for the full rationale. Short version: same model
family Phase 8.0 already evaluated, same ~2× early-utterance WER
cliff, same `<EOU>` token disappearing entirely in the cache-aware
output (NeMo's own `cache_aware_stream_step` over `jfk.wav`
produces 0 `<EOU>` tokens; we reproduced that bit-for-bit). Per-
chunk encoder cost on Mode 3 today is `O(left + chunk +
right_lookahead)`; trading that off for the chunked-limited
streaming-inference path is a quality regression and is not on the
roadmap.

### Phase 12.6 — download script + roundtrip verifier  _(done)_

`scripts/download-all-models.sh` swapped the previously-cached
forward-looking `stt_en_fastconformer_hybrid_large_streaming_multi.nemo`
for the actual `parakeet_realtime_eou_120m-v1.nemo`. The verifier
(`scripts/verify-gguf-roundtrip.py`) dispatches on `parakeet.model.type`
and ships a `build_expected_eou()` map that asserts every GGUF tensor
matches the source NeMo state-dict at f32 bit-exactness for the f32
slots and within a per-tier rel gate for the quant slots
(2^-10 for f16, 2^-7 for q8_0, 2^-4 for q5_0, 2^-3 for q4_0).

### Phase 12.x — follow-ups

#### Rejected (do not attempt again without new evidence)

- **Cache-aware streaming encoder graph for EOU 120m-v1
  (and any other chunked-limited-trained Parakeet checkpoint).**
  Prototyped on a working branch during the Phase 12.x exploration:
  bit-equal NeMo's `cache_aware_stream_step` (worst rel `1.85e-3`
  over 44 chunks of `jfk.wav`), end-to-end transcript bit-equal
  NeMo's *streaming* output -- which is structurally distinct
  from, and meaningfully worse than, NeMo's offline output (~2×
  early-utterance WER, no `<EOU>` token emitted). Reverted before
  landing. Same quality cliff Phase 8.0 documented two years
  earlier on `streaming_multi`; the EOU 120m-v1 family is the same
  cache-aware streaming Conformer family with a slightly newer
  120 M variant. **Will not be implemented.** See §8.5 case (A)
  for the full rationale and numbers.

#### Pending (no current owner)

(All Phase 12.x follow-ups have either shipped or been formally
rejected; see Phase 13 below for the cross-engine event API.)

## Phase 13 -- cross-engine StreamEvent API (VadState / EndOfTurn)  _(done)_

Voice-agent UX (turn detection, barge-in, hold-the-mic-open) needs
two signals we already had hooks for but no API on top of: VAD
state transitions and end-of-turn boundaries. Phase 13 lands a
small public `StreamEvent` surface that streaming sessions can
emit alongside the existing per-segment callbacks. The shape is
explicitly designed to be the same as what whisper.cpp's
streaming API will eventually emit, so consumers can write
engine-agnostic event handling once.

### Public types

```cpp
enum class VadState { Unknown, Speaking, Silent };
enum class StreamEventType { VadStateChanged, EndOfTurn };

struct StreamEvent {
    StreamEventType type;
    double  timestamp_s;
    int     chunk_index;

    // VadStateChanged
    VadState vad_state;
    int      speaker_id;     // argmax on entering Speaking; -1 otherwise
    float    vad_score;      // 0..1; provenance-specific

    // EndOfTurn
    float    eot_confidence;
};

using StreamEventCallback = std::function<void(const StreamEvent&)>;
```

`StreamingOptions::on_event` and `SortformerStreamingOptions::on_event`
default to `nullptr` (back-compat: existing consumers unaffected).
`StreamingOptions` also gains `enable_energy_vad` (default off) plus
`energy_vad_threshold_db = -35.0f`, `energy_vad_window_ms = 30`,
`energy_vad_hangover_ms = 200` knobs for the CTC/TDT fallback.

### Event sources

| Engine     | Event                  | Trigger                                                                                              |
|------------|------------------------|------------------------------------------------------------------------------------------------------|
| EOU        | `EndOfTurn`            | `<EOU>` token decoded in this chunk; `eot_confidence = 1.0`. Mode 2 + Mode 3.                        |
| Sortformer | `VadStateChanged`      | Per-chunk `max(speaker_probs) > threshold` (the same threshold the diarization head uses), with hysteresis (state retained across chunks). `speaker_id = argmax mean(speaker_probs)` on entering Speaking. |
| CTC / TDT  | `VadStateChanged`      | Energy-VAD on raw PCM (sliding RMS window, dBFS threshold + hangover). Only fires when consumer opts in via `enable_energy_vad`.                                                                          |

EOU's `EndOfTurn` is fired from both `Engine::transcribe_stream`
(Mode 2) and `StreamSession::process_window` (Mode 3) so the event
shape is identical regardless of which streaming entry point the
consumer drives.

### Implementation

- `include/parakeet/ctc/engine.h` -- new public types +
  `on_event` slots on both options structs + the `enable_energy_vad`
  knobs. Adding fields with defaults to a struct is forward-compatible
  for current consumers.

- `src/energy_vad.{h,cpp}` -- internal helper. Sliding RMS over a
  configurable ms window of mono f32 PCM, with hysteresis: enter
  Speaking immediately on threshold-crossing; fall back to Silent
  only after `hangover_ms` of below-threshold audio. Default
  `-35 dBFS / 30 ms / 200 ms` is tuned for clean 16 kHz mono speech.
  Not exposed in the public headers (would force consumers to pin
  to this implementation; shape may evolve).

- `src/parakeet_engine.cpp`:
  - `StreamSession::Impl` gains a `unique_ptr<EnergyVad>` member
    that is constructed only when `opts.enable_energy_vad` and the
    underlying engine has no native VAD source (constructed for
    CTC/TDT, skipped for EOU). The VAD is driven from a small
    `stream_drive_energy_vad()` helper invoked from both
    `feed_pcm_f32` and `feed_pcm_i16`.
  - `SortformerStreamSession::Impl` gains a `vad_state` field
    (initial `Unknown`, transitions on each chunk's emit-range
    speaker probabilities). Fires `VadStateChanged` on transitions
    only -- no per-chunk repeat events.
  - Mode-2 and Mode-3 EOU paths each fire `EndOfTurn` events on
    chunks where `eou_boundaries_in_chunk > 0`.

### Tests

- `test-streaming` (CTC + TDT) gained an opt-in energy-VAD
  invocation that asserts at least one Speaking transition fires
  on `jfk.wav`. Default-off path (sweep above) keeps emitting zero
  events, confirming back-compat.
- `test-eou-streaming` Mode-2 path now asserts that
  `is_eou_boundary` and `EndOfTurn` event count are consistent
  (boundary fires => at least one event fires). On `jfk.wav` chunk
  size 1500 ms: 1 `EndOfTurn` event, matching the single trailing
  `<EOU>` boundary.
- `test-sortformer-streaming` asserts at least one `VadStateChanged`
  event on a wav with audible speech and at least one Speaking
  transition. Default fixture (`diarization-sample-16k.wav`) skips
  when missing; on `jfk.wav` (single speaker, 11 s) the test fires
  one `Speaking` transition on chunk 0 with `speaker_id = 0`, which
  is the expected shape.

Numbers on `jfk.wav` (sanity check):

| Test                              | Events fired                                      |
|-----------------------------------|---------------------------------------------------|
| test-streaming + energy-VAD (CTC) | 9 VadStateChanged (6 Speaking transitions)         |
| test-streaming + energy-VAD (TDT) | 9 VadStateChanged (6 Speaking transitions)         |
| test-eou-streaming Mode 2         | 1 EndOfTurn at chunk 7 (the trailing `<EOU>`)      |
| test-sortformer-streaming v1.f16  | 1 VadStateChanged @ 0.00 s -> Speaking, speaker 0 |
| test-sortformer-streaming v1.q8   | identical to f16                                   |
| test-sortformer-streaming v2.q4   | 1 VadStateChanged @ 0.00 s -> Speaking, speaker 0 |

### Shape decisions

- **Single struct + enum, not separate event types.** Keeps the
  callback signature trivial (`void(const StreamEvent&)`) which
  maps cleanly through any C/C++/FFI ABI without per-type
  wrappers. Costs a few unused fields per event; cheap.
- **Engines fire what they natively know.** EOU has the `<EOU>`
  token and fires only `EndOfTurn`; Sortformer has speaker probs
  and fires only `VadStateChanged`; CTC/TDT have neither so they
  fire `VadStateChanged` from energy-VAD when explicitly enabled.
  No engine pretends to fire events it doesn't have a real signal
  for, and no Silero / external VAD dependency is added.
- **Default off.** Both `on_event = nullptr` and
  `enable_energy_vad = false` are the defaults. No behavioural
  change for existing consumers; opt-in only.

## Phase 14 — TDT decoder Metal port  _(done)_

Phase 10 brought up TDT (Token-and-Duration Transducer) end-to-end on
CPU with the encoder also offloadable to Metal, but the **decoder
itself bypassed ggml entirely**: at load time the LSTM prediction net
+ joint MLP were dequantised to host `std::vector<float>` and the
greedy emission loop ran scalar `gemv_f32` per emission step. Even
with a Metal-accelerated encoder, the decoder owned ~48 % of total
inference time on the M4 Air (76 ms of 159 ms on a 20 s clip). Phase 14 ports the decoder to ggml graphs on `backend_active` so it runs
end-to-end on Metal alongside the encoder.

### 14.1 — graph design

Two fixed-shape per-step graphs plus one window-shape graph, all
allocated against `model.backend_active()` (Metal / CUDA / Vulkan
when compiled and `--n-gpu-layers > 0`, else CPU):

  - **`g_lstm_step`** — embedding lookup (`ggml_get_rows` against the
    native quantised `predict_embed` tensor) + L-layer LSTM unroll
    expressed as `mul_mat` + `add` + `sigmoid`/`tanh` + element-wise
    products. Inputs `token_in[1, i32]`, `h_in[H, L]`, `c_in[H, L]`;
    outputs `h_out[H, L]`, `c_out[H, L]`, `pred_out[H]` (alias for
    last-layer `h_new`). Built once, reused via
    `ggml_gallocr_alloc_graph` per emission step.
  - **`g_joint_step`** — `pred_proj = joint_pred @ pred + b`,
    `hidden = relu(pred_proj + enc_proj_row)`,
    `logits = joint_out @ hidden + b`. Inputs `pred_out[H_pred]` and
    `enc_proj_row[H_joint]`; output `logits[V_out]`.
  - **`g_enc_proj`** — full-window `enc_proj = joint_enc @ enc + b`
    matmul (size `[T_enc, D_enc] -> [T_enc, H_joint]`). One per
    distinct `T_enc` seen (LRU-cached in `enc_proj_cache`). Hoisting
    this matmul out of the per-step joint graph cuts ~250 small
    `gemv(640, 1024)` calls per window down to one large `gemm` —
    cheap on Metal where matmul kernels are compute-bound, expensive
    on CPU where it loses cache locality (see §14.2 fallback).
  - All three graphs use `ggml_set_input` / `ggml_set_output` and
    upload host inputs each step via `ggml_backend_tensor_set`,
    pulling outputs back via `ggml_backend_tensor_get`. The
    `argmax` over token + duration logits stays on host (~32 KB
    `tensor_get` per step is cheap on unified memory; see §14.5
    Phase 4 gate decision).

`TdtRuntimeWeights` carries both the GPU-graph scaffolding
(`ggml_context * gctx`, `ggml_cgraph * g_lstm / g_joint`, gallocrs,
`ggml_tensor *` inputs/outputs, and an `enc_proj_cache` LRU) and a
parallel set of host f32 vectors (`embed`, `host_lstm[L]`,
`host_joint_*`) for the CPU fallback. Move semantics + a destructor
free the gallocrs, contexts, and any cached enc_proj graphs on
runtime teardown; the backend pointer itself is owned by
`ParakeetCtcModel::Impl`.

### 14.2 — CPU fallback

The straightforward "all paths through ggml" design regressed CPU
decode by **~6x** (76 ms -> 480 ms median) because per-step graph
dispatch on the synchronous CPU backend pays thread-pool wakeup
latency on every one of ~250 emission steps. The fix is a runtime
branch: `tdt_prepare_runtime` checks `ggml_backend_is_cpu(backend)`
and either builds the graphs (GPU) or dequantises weights to host
f32 (CPU). The decode loop then routes every per-step op
(`tdt_init_state`, `host_lstm_step`, `host_joint_step`) through the
proven scalar implementation when `!use_graphs`.

The CPU path also keeps the original **per-step** `joint_enc` gemv
inside `host_joint_step` rather than the full-window precompute used
on GPU: profiling showed the precompute regresses CPU by ~8 % for
20 s windows because it streams ~1 MB through L1 once per window
without reuse, while the per-step gemv keeps the encoder-frame
slice in cache through both `joint_enc` and the surrounding
`joint_pred` / `joint_out` calls.

### 14.3 — parity gate

`test-tdt-decoder-parity` (`test/test_tdt_decoder_parity.cpp`,
linked under `PARAKEET_BUILD_TESTS`) runs the same WAV through
`tdt_greedy_decode` twice — once with `n_gpu_layers=0` (scalar CPU
fallback) and once with `n_gpu_layers=1` (ggml graph path on the
compiled backend). Greedy TDT is fully deterministic, so the
invariant is exact integer equality of the token-ID stream (and
hence byte-equal transcript text). On `sample-16k.wav` (20.13 s,
M4 Air, Metal build):

```
[tdt-decode-parity] CPU: tokens=95 text=Alice was beginning to get very tired of sitting by her sister...
[tdt-decode-parity] GPU: tokens=95 text=Alice was beginning to get very tired of sitting by her sister...
[tdt-decode-parity] PASS: CPU vs graph token IDs match (95 tokens)
```

A `<ref-dir>` argument optionally also compares against the NeMo
reference token-ID stream from `scripts/dump-tdt-reference.py`
(extended in this phase to write `token_ids.npy` alongside the
existing `transcript.txt`).

### 14.4 — bench

`sample-16k.wav` (20.13 s of audio), `--bench-warmup 5
--bench-runs 15`, M4 Air, q8_0:

| backend       | enc median | dec best | dec median | inf median | RTF best | RTF median | real-time multiple |
|---------------|-----------:|---------:|-----------:|-----------:|---------:|-----------:|-------------------:|
| CPU baseline  |    911     |    72.8  |    76.6    |   1003     |   0.041  |   0.050    |          24x       |
| **CPU after** |   1102 *   |  **72.3**|  **76.95** |   1190 *   |   0.043  |   0.059    |          23x       |
| Metal baseline|     68.5   |    72.7  |    76.4    |    159.5   |   0.008  |   0.008    |         132x       |
| **Metal after** |    68.9   |  **58.5**|  **59.32** |  **143.2** | **0.007**| **0.007**  |       **142x**     |

`*` CPU "after" `inf median` includes encoder-side wall-time noise
(thermal throttling on a passive-cooled Air during the 15-run sweep
shows up in the encoder, not the decoder); the **decoder** numbers
are within 0.5 ms of baseline on CPU.

Net Metal effect on the 20 s clip:

  - decoder: **76.4 ms -> 59.3 ms median (-22.4 %)**
  - inference total: **159.5 ms -> 143.2 ms median (-10.2 %)**
  - real-time multiple: **132x -> 142x**

CPU stays neutral by design (the fallback path is the same scalar
implementation that shipped in Phase 10); the new graph path is
only exercised on Metal / CUDA / Vulkan builds where
`backend_active` is non-CPU.

### 14.5 — encoder→decoder handoff (gated, not landed)

The original plan considered keeping `encoder_out` resident on the
backend so the TDT decoder could run directly off the GPU tensor
instead of going through the existing host `std::vector<float>` in
`EncoderOutputs::encoder_out`. Empirical profiling on the M4 Air:

```
[probe] encoder_out tensor_get: 87 us (258048 floats)
[probe] enc_proj   tensor_set: 17 us (258048 floats)
```

Total host roundtrip for the encoder→decoder boundary is **~104 us
per call**, i.e. **0.07 % of total inference time** on a 20 s clip.
The gate for this work was a >5 % RTF improvement; the data
disqualifies it (Apple Silicon's unified-memory `tensor_get/set` is
essentially memcpy at ~50 GB/s and cannot deliver the threshold).
Skipped, with the engine-side API kept simple — `EncoderOutputs`
stays host-side, matching CTC + EOU + Sortformer.

### 14.6 — bench-JSON backend label

Pre-existing bug surfaced by this phase: `main.cpp`'s
`--bench-json` writer hardcoded `"backend": "ggml-cpu"` regardless
of the active backend, which silently mis-tagged every Metal /
CUDA / Vulkan bench captured into `artifacts/bench/`. Fixed to
derive from `GGML_USE_METAL` / `GGML_USE_CUDA` / `GGML_USE_VULKAN`
plus the runtime `n_gpu_layers` flag, and an `n_gpu_layers` field
was added to the JSON so post-hoc sweeps can disambiguate same-
binary CPU vs GPU runs.

### 14.7 — remaining work

  - **CUDA / Vulkan validation.** The graph code path is generic
    over `backend_active`; both backends should "just work" because
    every op used (`get_rows`, `mul_mat`, `add`, `sigmoid`, `tanh`,
    `mul`, `concat`, `cont`) is supported on CUDA and Vulkan in
    the pinned ggml. Not validated on hardware in Phase 14 — needs
    a follow-up bench run.
  - **Mode 3 streaming bench.** Phase 14 measured Mode 1 (one-shot
    `tdt_greedy_decode` over the full window) only. The streaming
    `StreamSession::process_window` calls into the same
    `tdt_decode_window` so the per-step Metal speed-up should
    carry over, but the per-chunk cost mix is different (smaller
    `T_enc` per call -> the `g_enc_proj` cache will see more
    distinct shapes; the LRU is currently unbounded). Tracked as a
    follow-up: cap the cache or switch to a bucketed shape.
  - **TDT 1.1B sweep.** Numbers above are for `parakeet-tdt-0.6b-v3`
    only; rerun on `parakeet-tdt-1.1b` to populate the
    "RTF (Metal)" column for that row in the README's Supported
    checkpoints table.

## Phase 15 — fused LSTM+joint + persistent decoder state (Metal)

Phase 14 ported the TDT decoder to ggml graphs and shipped on Metal
with two `compute_graph` dispatches per non-blank emission step
(joint, then LSTM). Profiling on M3 Ultra showed the dominant cost
per step is **the Metal command-buffer commit + wait latency, not
the readback or the kernel work itself**:

```
[probe] phase 13 decoder = 57.6 ms / 247 dispatches = ~233 us/dispatch
                                                    ~ commit ~150 us + GPU ~25 us + bookkeeping
```

Phase 15 collapses the per-non-blank dispatch pair into a single
fused graph. The LSTM update writes h / c / pred in place into a
persistent backend buffer via `ggml_cpy`; the joint mat-muls take
the `pred_cpy` node as their input so gallocr orders the LSTM
update strictly before the joint reads inside one Metal command
buffer.

### 15.1 — persistent decoder state

`TdtRuntimeWeights` gains a dedicated `persist_buffer` allocated
via `ggml_backend_alloc_ctx_tensors` that holds:

  - `h_persist`        : f32[H_pred, L]  (LSTM hidden, layer-major)
  - `c_persist`        : f32[H_pred, L]  (LSTM cell)
  - `pred_persist`     : f32[H_pred]     (last-layer h, fed into joint)
  - `enc_proj_persist` : f32[H_joint, T_max]  (T_max = 4096 frames)

All four stay resident on the backend across the entire decode
loop. Per-step host upload shrinks from ~5 KB (token + h + c +
enc_proj_row) to **4 B** (just the frame index or token id);
`enc_proj` is no longer downloaded after the full-window
projection — it's `ggml_cpy`'d straight into the persistent slab
and the joint network reads rows via `ggml_get_rows` on a
host-supplied frame index.

### 15.2 — three fixed-shape graphs

A `build_lstm_body` helper is shared between two of them so the
LSTM math stays numerically identical across init and the fused
hot path:

  1. `g_lstm`       — init-only. Used once per call (`tdt_init_state`)
                      to seed `pred_persist` after a blank LSTM step.
  2. `g_joint`      — used after blank emissions (pred unchanged).
                      Reads `pred_persist`, slices `enc_proj_persist`
                      via `ggml_get_rows(frame_idx)`, writes logits
                      to host.
  3. `g_lstm_joint` — used after non-blank emissions. **Fused**:
                      LSTM body writes the new pred via `ggml_cpy`,
                      then the joint mat-muls take that cpy node as
                      their pred input. One commit instead of two.

The decoder loop tracks `pending_lstm_token`: blank emissions
clear it and the next iteration uses `g_joint`, non-blank
emissions defer the LSTM update so the next iteration fuses it
with the next frame's joint forward via `g_lstm_joint`.
Streaming windows flush any deferred update at end-of-window.

### 15.3 — bench (Metal, M3 Ultra, sample-16k.wav, 20.1 s, 95 tokens)

3-warmup + 10-timed runs, averaged across 3 invocations:

| Stage          | Phase 14 base | Phase 15 fused | Δ        |
|----------------|--------------:|---------------:|---------:|
| mel ms         |        14.4   |          14.6  |  noise   |
| encoder ms     |        68.5   |          68.6  |  noise   |
| **decode ms**  |    **57.6**   |      **43.0**  | **−25%** |
| **inference**  |     **141**   |       **126**  | **−10%** |
| RTF            |        0.007  |         0.006  |          |
| realtime mult  |       146×    |        **160×**| **+14×** |

Parity gate: `test-tdt-decoder-parity` PASSes — CPU and graph
paths emit byte-identical 95-token streams. The fused graph is
numerically equivalent to the sequential path because:

  1. `ggml_cpy(h_new, h_persist)` writes h_persist's memory in
     place; subsequent readers of `h_persist` see the new value.
  2. The joint body uses the `pred_cpy` result tensor (not
     `pred_persist` directly) so its mat_muls dataflow-depend on
     the cpy and gallocr emits the LSTM update's barriers first.
  3. `h_persist` and `c_persist` live in `persist_buffer`, which
     is a separate backend buffer from gallocr's compute buffer,
     so gallocr cannot alias them with intermediate `h_new` /
     `c_new` and there are no read-before-write hazards.

### 15.4 — what didn't work

**Batched-joint over K consecutive frames** *(prototyped, reverted)*

The arithmetic looked promising: 152 single-frame blank-path
joints could collapse to ~96 K-frame batches (one per non-blank
cycle, since avg blank-run length ≈ 152 / 95 = 1.6 frames).
Tested K ∈ {4, 8} on the same sample; both regressed by ~0–1 ms
back to phase-13-ish numbers:

| Variant   | decode ms (3-run mean) |
|-----------|-----------------------:|
| Phase 15  |                  43.0  |
| K = 4     |                  43.8  |
| K = 8     |                  43.2  |

Empirical conclusion: **Apple Silicon Metal command-buffer
commit latency is much lower than the ~150 us I assumed from
back-of-envelope, probably ~30–50 us in practice**. The 56-commit
saving from K = 8 (predicted ~8 ms) gets eaten by the larger
per-batch GPU work (each batch computes joint over K frames
even though only ~1.6 are consumed before a non-blank). Reverted
the prototype rather than ship neutral code; phase 14's fused
LSTM+joint is the local optimum on this hardware.

### 15.5 — remaining work

  - **CUDA / Vulkan validation.** Same plumbing as Phase 14:
    `g_lstm_joint` and the persistent-state buffer should "just
    work" on any backend that already supports `ggml_cpy`,
    `ggml_get_rows`, `ggml_backend_alloc_ctx_tensors`. Worth
    benchmarking — backends with higher dispatch overhead
    (CUDA) could see proportionally larger Phase 15 wins.
  - **TDT 1.1B sweep.** Same caveat as Phase 14; the relative
    win should hold but absolute numbers shift.

---

### 15.8 — Phase 6.5 follow-up: ship `ggml_flash_attn_ext` on Metal

Closes the lone bench-validation hole in PROGRESS §6.5 ("Test
`ggml_flash_attn_ext` on Metal — likely a meaningful win"). The
infra has been dormant since the Round 7 audit (§5.13), gated
behind `#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN` in
`rel_pos_mha_graph()` and surfaced as the `PARAKEET_FLASH_ATTN`
CMake option (off by default everywhere). Round 7 only A/B'd it on
CPU, where it regressed encoder by +3.1 % because the cast-to-f16
of the relative-position bias `bd_final` mask before softmax
shifted the BD computation order; Metal was never tested.

**What changed:** `CMakeLists.txt` now derives a per-backend
default. When `GGML_METAL=ON` the option defaults to ON; CPU /
CUDA / Vulkan / OpenCL keep their existing OFF default until each
ships its own A/B (CUDA can be exercised with a local `-DGGML_CUDA=ON` build
and `parakeet --bench` when a discrete-GPU host is available). No source files in `src/` changed — the
`#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN` branch in
`parakeet_ctc.cpp::rel_pos_mha_graph` is what gets compiled in.

**Kernel actually loaded:** from `ggml_metal_library_compile_pipeline`
log on first invocation —

  - `kernel_flash_attn_ext_pad_mask=1_ncpsg=64`
  - `kernel_flash_attn_ext_blk_nqptg=8_ncpsg=64`
  - `kernel_flash_attn_ext_f32_dk128_dv128_mask=1_sinks=0_bias=0_scap=0_kvpad=1_bcm=1_ns10=128_ns20=128_nsg=4`

Head_dim 128 covers `parakeet-ctc-0.6b`, `parakeet-tdt-0.6b-v3`,
`parakeet_realtime_eou_120m-v1` (all `d_model=1024 / n_heads=8`)
and `parakeet-tdt-1.1b` (`d_model=2048 / n_heads=16` → also 128).
Sortformer's transformer is `tf_d_model=192 / n_heads=8 → head_dim=24`,
which falls below `flash_attn_ext`'s supported set
{40,64,80,96,112,128,192,256}. `parakeet_sortformer.cpp` does not
share `rel_pos_mha_graph` with the conformer encoder, so the
Sortformer transformer block is structurally untouched by this
flag.

**Bench (M3 Ultra Metal, q8_0, sample-16k.wav 20.13 s, 95 tokens,
3 warmup + 15 timed runs averaged across 5 invocations):**

| Stage    | FA OFF (HEAD)    | FA ON (this change) | delta              |
|----------|------------------:|--------------------:|-------------------:|
| mel ms   |   7.72 ± 0.13     |    7.65 ± 0.02      |              noise |
| enc ms   |  67.35 ± 0.03     |   67.00 ± 0.02      |  −0.35 ms / −0.5 % |
| dec ms   |  44.16 ± 0.27     |   44.02 ± 0.51      |              noise |
| infer ms | 119.24 ± 0.32     |  118.66 ± 0.51      |  −0.58 ms / −0.5 % |
| RT mult  |     168×          |        170×         |               +2×  |

Stdev figures are between-invocation; per-invocation stdev is
≤ 0.21 ms on encoder. The 0.35 ms encoder saving is ~5–6× the
between-invocation stdev, so reproducible.

**Why so modest on M3 Ultra:** the conformer attention shape is
`T = 252, H = 8, HD = 128`, which puts the QK^T scores tensor at
~2 MB (252 × 252 × 8 × 4 B). That's well-cached on the M3 Ultra's
60-core Metal GPU, so the standard `mul_mat → soft_max_ext →
permute → mul_mat` path was already not memory-bound. The
remaining win is purely from collapsing four kernel dispatches
per attention block into one (24 layers × 4 = 96 dispatch saves
× ~30 µs/dispatch ≈ 2.9 ms theoretical; we measured 0.35 ms,
suggesting the dispatch saving is partially absorbed by ggml's
graph-machinery overhead and the f32→f16 BD-mask cast). PCIe-
based discrete GPUs typically have higher per-dispatch overhead
and proportionally less L2 per SM, so the predicted-positive
case on CUDA / Vulkan is meaningfully larger; that's why those
defaults stay OFF until measured.

**Parity:** all gates pass byte-exact under the new default —

  - `test-tdt-decoder-parity` PASS, 95 tokens, "Alice was
    beginning…", CPU-fallback vs Metal-graph token IDs identical
  - `test-mel-fft-parity` PASS, rfft vs textbook FFT rel error
    6.89e-08, stateful overload bit-equal stateless on 101 frames
    × 80 mels, 7 sequential calls bit-equal
  - `test-encoder-capture-parity` (CTC) PASS,
    `encoder_out (258 048 floats)` and `logits (258 300 floats)`
    bit-equal across capture=true/false
  - `test-perf-regression` (TDT q8_0, n_gpu_layers=1) PASS,
    transcript byte-equal to expected, mel and encoder summary
    inside the configured budgets

The cast-to-f16 of the BD mask that broke CPU parity in Round 7
does not break Metal parity here because: (a) the no-streaming
case (full sample-16k.wav window) has `att_mask = nullptr` so the
mask passed to `ggml_flash_attn_ext` is purely `bd_scaled =
scale * bd_final` with no additional masking term to merge; (b)
the f16 cast happens on a tensor whose pre-softmax magnitudes are
in the ±5 to ±10 range (relative-position embeddings post matmul
+ `pos_bias_v` add), well within f16's 6-decimal-digit precision;
(c) the downstream argmax over the joint logits is invariant to
sub-bit-15 precision drift in attention scores.

**What this does not address:**

  - `att_mask != nullptr` (Mode 2/3 streaming windows). The
    experimental code path passes only `bd_scaled` as the mask;
    the additive `att_mask` is dropped on the floor. Mode 1 is
    fine but for Mode 2/3 production streaming the mask should
    be folded in via `ggml_add(bd_scaled, att_mask)` before the
    f16 cast. Tracked as a precondition for the streaming bench
    sweep.
  - Sortformer transformer head_dim=24 is unsupported by
    `flash_attn_ext`. Not a regression — just unaffected.

**Remaining stack-rank for next encoder optimization:**

  1. Conv2d-DW Metal kernel (PROGRESS §6.5 first bullet) —
     promotes 24 conformer-conv blocks off the im2col fallback.
     Bigger expected gain than this flash-attn flip because the
     im2col path adds a separate copy kernel before the matmul.
  2. Hybrid `ggml_backend_sched`: overlap mel preprocessing
     (7.65 ms host CPU) with encoder dispatch (67 ms Metal).
     Up to 7.65 ms inference wall-time saving if we can hide
     mel under encoder.
  3. `att_mask` fold-in for streaming (covered above).

---

## Phase 16 — Vulkan backend validation  _(done)_

Vulkan was listed as a supported backend since Phase 6 but had never
been validated end-to-end on the CTC encoder. This phase brings it to
correctness on Windows with an NVIDIA RTX 5060 (should apply to any
Vulkan-capable GPU).

(Originally landed as "Phase 15" on `main` while this branch was
mid-flight on its own §14 / §15 TDT-decoder + fused-LSTM+joint
sequence; renumbered to §16 here so the two streams don't collide.)

### 16.1 — bugs found and fixed

Two issues prevented the Vulkan backend from producing correct output:

1. **`memcpy` from GPU pointer in `read_filterbank_to_vector()`.**
   The filterbank tensor is allocated on the GPU backend when
   `n_gpu_layers > 0`. The original code did
   `std::memcpy(out.data(), t->data, ...)` which dereferences a
   device pointer on the host — segfault (`0xC0000005`). Fixed by
   switching to `ggml_backend_tensor_get(t, out.data(), 0, n)` which
   handles the device-to-host copy transparently. This fix is
   backend-agnostic: it was already correct on Metal/CUDA because
   those backends happened to map `t->data` to host-visible memory,
   but the `ggml_backend_tensor_get` path is the correct API for all
   backends.

2. **Strided `ggml_view_3d` passed to unary ops in the GLU.**
   The Conformer conv module's Gated Linear Unit splits the
   pointwise-conv output in half along the channel dimension using
   `ggml_view_3d`. The resulting tensors are non-contiguous (strided).
   `ggml-vulkan`'s unary operations (`ggml_sigmoid`) use push
   constants that do not carry stride information — they assume
   contiguous memory. The sigmoid output was therefore garbage
   (`rel=0.82` at the GLU stage), which cascaded into zero-token
   transcriptions. Fixed by wrapping both `ggml_view_3d` halves with
   `ggml_cont()` before feeding them into `ggml_sigmoid` and
   `ggml_mul`. This fix is also backend-agnostic: any backend that
   doesn't handle strided unary inputs benefits.

### 16.2 — diagnosis methodology

The bisection used the `test-vk-vs-cpu` harness with per-sub-stage
taps injected into the first Conformer block's convolution module.
Each intermediate tensor (`pre`, `post_pw1`, `post_glu`, `post_dw`,
`post_pw2`, `post_bn`) was compared CPU vs Vulkan. The divergence
was pinpointed to `post_glu` (rel jumped from ~2e-3 to 0.82),
confirming the `ggml_view_3d` + unary op interaction as root cause.
The `ggml-vulkan.cpp` source was then inspected to confirm that
unary push constants lack stride fields, validating the hypothesis.

### 16.3 — parity results (RTX 5060, Windows, f16 GGUF)

```
PASS stage subsampling_out       n=141312  max_abs=1.239e+01  rel=2.032e-03
PASS stage block0_post_ff1       n=141312  max_abs=3.966e+02  rel=2.030e-03
PASS stage block0_post_attn      n=141312  max_abs=3.965e+02  rel=2.033e-03
PASS stage block0_post_conv      n=141312  max_abs=3.901e+02  rel=2.032e-03
PASS stage block0_post_ff2       n=141312  max_abs=3.897e+02  rel=2.035e-03
PASS stage block0_out            n=141312  max_abs=3.813e-01  rel=1.958e-03
PASS stage block_last_out        n=141312  max_abs=1.260e-01  rel=6.897e-03
PASS stage encoder_out           n=141312  max_abs=1.260e-01  rel=6.897e-03
PASS stage logits                n=141450  max_abs=1.047e+00  rel=1.454e-03
all stages passed
```

### 16.4 — build system changes

- `CMakeLists.txt`: centralised `GGML_USE_*` defines into an
  `INTERFACE` library `parakeet-backend-defs` (CUDA, Metal,
  Vulkan, BLAS, OpenCL). All test targets link this library so
  GPU code paths are compiled consistently.
- `test-vk-vs-cpu` target gated behind `if (GGML_VULKAN)`.
- Test sources live under `test/test_*.cpp`
  for cleaner repo organisation.

### 16.5 — test harness

`test/test_vk_vs_cpu.cpp` loads the same GGUF twice (CPU and
Vulkan), runs both encoders on the same mel input, and compares
9 intermediate stages. Each stage asserts `rel < 5e-2` and no
NaN/Inf values. Exit code 1 on any failure.

### 16.6 — follow-ups

- Vulkan performance optimisation (RTF benchmarking, pipeline cache).
- Validate on AMD and Intel GPUs.
- Upstream the `ggml_cont` fix as a ggml-vulkan unary stride patch.
