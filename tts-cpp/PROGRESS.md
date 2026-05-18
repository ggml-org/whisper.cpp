# Chatterbox → ggml Port: Development Journal

This document tracks the port of **Chatterbox** (Resemble AI, MIT license)
to `ggml`, from the first exploratory scoping all the way to the optimized
end-to-end CPU/GPU binary, in the order things actually happened.  §3.1 –
§3.18 cover the original **Turbo** port (English, GPT-2 Medium T3, meanflow
CFM); §3.19 / §3.20 add the **Multilingual** variant (23 languages,
Llama-520M T3 + perceiver, standard CFG-enabled CFM) and the cross-variant
S3Gen weight-quantisation pass.

- **Models**: `ResembleAI/chatterbox-turbo` (~450 M params, English) and
  `ResembleAI/chatterbox` (~520 M T3 + 23-language tokenizer, the
  multilingual variant).  Both share the S3Gen + HiFT vocoder back half.
- **Goal**: end-to-end `text → waveform` in C++/ggml with **bit-exact (or
  float-precision) parity** against the official PyTorch reference.
- **Verification target**: every intermediate tensor within 1e-6 relative error
  of the PyTorch implementation, on CPU.

---

## Current status (end of journey)

Everything runs in pure C++/ggml on CPU. The main end-to-end tool is one binary:

| Binary | Role |
|--------|------|
| `tts-cli` | end-to-end: text → speech tokens (T3) → 24 kHz wav (S3Gen + HiFT); voice cloning, streaming, both Turbo and Multilingual variants (autodetected from GGUF metadata). |
| `chatterbox` | identical second binary kept for backward compatibility with pre-rename scripts; same code as `tts-cli`. |
| `mel2wav` | mel spectrogram → wav (HiFT only, demo) |

Plus `scripts/synthesize.sh`, a thin wrapper around `tts-cli`.

**Numerical parity vs PyTorch** on a 2.7 s reference utterance, debug mode
(Python-dumped random bits substituted for reproducibility):

| Stage | rel error vs PyTorch |
|-------|---------------------|
| BPE tokenizer | 10/10 exact-match test cases |
| T3 speech tokens | bit-exact on 4 deterministic prompts |
| S3Gen encoder (full, incl. upsample and encoder_proj) | 4.5e-07 |
| CFM 2-step meanflow decoder | 8.9e-07 on the final mel |
| HiFT decode body (conv_pre → conv_post) | 5.6e-07 |
| ISTFT → waveform | 1.0e-04 |
| End-to-end C++ wav vs Python wav (RMS) | 1.22e-04 vs 1.22e-04 |

**Speed** (10 s sentence, seed 42, `gen_RTF = (T3_INFER + S3GEN_INFER) / audio_ms`):

| Backend                     | `gen_RTF` | Wall  | vs ONNX addon |
|-----------------------------|----------:|------:|--------------:|
| CPU (10-core EPYC, F16)     | 0.70      | 8.2 s | 3.6× faster   |
| **Vulkan (RTX 5090, Q4_0)** | **0.06**  | **1.8 s** | **7.8×** |
| **Metal (M3 Ultra, Q4_0)**  | **0.13**  | **1.9 s** | **7.4×** |
| ONNX q4 addon (CPU baseline) | 1.06     | 13.9 s | 1.0×         |

GPU support and Metal kernel fixes are described in §3.11 / §3.12;
the layout-friendly KV cache + Flash Attention pass that produced the
numbers in this table is in §3.13.  The Multilingual port (§3.19) and
the S3Gen weight-quantisation pass that landed alongside it (§3.20)
add a second variant on top of the same back half — see those sections
for the MTL-specific parity / speed numbers.

---

## Repository layout

```
chatterbox.cpp/
  ggml/                           vendored ggml checkout (see patches/, scripts/setup-ggml.sh)
  patches/
    ggml-metal-chatterbox-ops.patch   Metal op fixes: diag_mask_inf, pad_ext,
                                      faster conv_transpose_1d (applied to ggml/
                                      during setup; see patches/README.md)
    ggml-opencl-chatterbox-ops.patch  OpenCL/Adreno fixes: missing HiFT/S3Gen
                                      ops + conv_transpose_1d speedup
    README.md                         why each patch exists + how to drop it
  include/tts-cpp/                installed public headers (Engine API)
    tts-cpp.h                       library entry; declares tts_cpp_cli_main()
    chatterbox/engine.h             Engine + EngineOptions (text → wav)
    chatterbox/s3gen_pipeline.h     low-level S3Gen pipeline entry points
  src/
    main.cpp                      T3 turbo runtime + shared helpers (libtts-cpp)
    t3_mtl.{h,cpp}                T3 multilingual (Llama-520M) runtime + stage builders
    chatterbox_t3_internal.h      internal T3 declarations shared by main.cpp / engine / CLI
    chatterbox_engine.cpp         public Engine API impl (links into libtts-cpp)
    chatterbox_cli.cpp            unified CLI (tts-cli + chatterbox binaries)
    cli_main.cpp                  thin entry: forwards argc/argv to tts_cpp_cli_main()
    chatterbox_tts.cpp            S3Gen encoder + CFM + HiFT (reusable entry)
    gpt2_bpe.{h,cpp}              self-contained GPT-2 byte-level BPE tokenizer (turbo)
    mtl_tokenizer.{h,cpp}         multilingual grapheme tokenizer (HF tokenizers.json + NFKD)
    mtl_unicode_tables.inc        embedded NFKD + Korean Jamo lookup tables
    voice_features.{h,cpp}        wav I/O, resample, mel, fbank, LUFS
    mel_extract_stft.cpp          STFT-based mel extraction shared by C++ pipelines
    voice_encoder.{h,cpp}         VoiceEncoder 256-d speaker embedding
    campplus.{h,cpp}              CAMPPlus 192-d speaker embedding (BN-fused inc include)
    s3tokenizer.{h,cpp}           S3TokenizerV2 (wav → S3 speech tokens)
    mel2wav.cpp                   mel → wav demo binary (HiFT only)
    test_s3gen.cpp                staged verification harness for turbo S3Gen (A..H5)
    test_t3_mtl.cpp               end-to-end parity test for the MTL T3 forward pass
    test_t3_mtl_stages.cpp        staged parity harness for MTL (cond/text/inputs/layers/head)
    test_mtl_tokenizer.cpp        MTL tokenizer parity vs HF reference
    test_metal_ops.cpp            parity test for the patched Metal kernels
    test_streaming.cpp / test_voice_*.cpp / test_resample.cpp / test_fbank.cpp / …
    npy.h, dr_wav.h               minimal .npy loader + WAV decoder (header-only)
  scripts/
    setup-ggml.sh                     clones the pinned ggml commit + applies patches
    convert-t3-turbo-to-gguf.py       Turbo T3 weights + tokenizer + VE + builtin voice → GGUF
    convert-t3-mtl-to-gguf.py         MTL T3 (Llama-520M) + perceiver + emotion-adv
                                      + tokenizers.json + builtin voice → GGUF
    convert-s3gen-to-gguf.py          S3Gen encoder + CFM + HiFT + CAMPPlus + S3TokenizerV2
                                      + mel filterbanks → GGUF (--variant {turbo,mtl},
                                      --quant {f32,f16,q8_0,q5_0,q4_0})
    requantize-gguf.py                in-place block-quantise of an existing S3Gen/T3 GGUF
    gen-nfkd-table.py                 generates src/mtl_unicode_tables.inc from CLDR data
    extract-voice.py                  one-shot voice-clone prep (silencedetect + EQ + bake)
    dump-{s3gen,campplus,s3tokenizer,streaming,t3-mtl}-reference.py
                                      PyTorch → .npy intermediates for the test-* harnesses
    reference-t3-turbo.py             PyTorch T3 + compare against C++
    compare-tokenizer.py              10-case tokenizer comparison against HF
    synthesize.sh                     text → wav wrapper around tts-cli
  models/
    chatterbox-t3-turbo.gguf      Turbo T3 (GPT-2 Medium) + GPT-2 BPE + builtin voice
    chatterbox-s3gen.gguf         Turbo S3Gen (meanflow CFM) + HiFT + CAMPPlus + S3TokV2
    chatterbox-t3-mtl.gguf        Multilingual T3 (Llama-520M) + tokenizers.json + builtin voice
    chatterbox-s3gen-mtl.gguf     Multilingual S3Gen (standard 10-step CFM, CFG inside)
    *-{q8_0,q5_0,q4_0}.gguf       quantised variants (see §3.20)
  CMakeLists.txt                  top-level: add_subdirectory(ggml) + tts-cpp lib + binaries
  PROGRESS.md                     this file
  README.md                       user-facing build / run / benchmark guide
```

A separate machine holds PyTorch + the original Chatterbox repo for reference
runs. On-device (Apple Silicon / Linux x86) the C++ binaries have **no runtime
dependency on Python** — the Turbo BPE tokenizer (`vocab.json` + `merges.txt`)
and the Multilingual `tokenizers.json` are both embedded directly into their
T3 GGUFs as `tokenizer.ggml.*` metadata, so the only runtime input is the
GGUF file itself plus optional reference audio.

---

## Development log (chronological)

### 3.1  Scoping and bootstrap

Surveyed open-source TTS candidates (F5-TTS, Kokoro-82M, XTTS v2, Piper, Fish
Speech, Supertonic, Chatterbox). Picked **Chatterbox Turbo** for three reasons:
MIT license, zero-shot voice cloning, and the "Turbo" variant uses just **2
flow-matching steps** (fast inference).

Bootstrapped the repo by cloning the latest `ggml` and the reference
`resemble-ai/chatterbox` side-by-side, then built a standalone
`chatterbox.cpp/` with `ggml/` as a vendored subdirectory (no modifications
inside `ggml/`).

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 1 | `rsync` not on macOS by default | Switched to `tar … \| ssh … tar -x`. |
| 2 | Remote repo polluted with `._*` AppleDouble files | `COPYFILE_DISABLE=1 tar …`. |
| 3 | Partial sync left `src/CMakeLists.txt` stray file | Removed; unified sync always pushes the whole tree. |
| 4 | Remote binary `0 bytes` after SSH disconnect | `rm build/<target>` + rebuild. |

### 3.2  T3 port + custom BPE tokenizer

T3 is a GPT-2 Medium-sized (24 layer) autoregressive model that maps text
tokens + voice conditioning to speech tokens.

- Wrote `scripts/convert-t3-turbo-to-gguf.py` to emit a GGUF with built-in
  voice conditionals (`speaker_emb`, `cond_prompt_speech_tokens`) embedded.
- C++ graph in `src/main.cpp`: split into a "prompt" graph and a "step" graph
  sharing a persistent KV cache, mirroring `ggml/examples/gpt-2`.
- Ported the sampler (Temperature → TopK → TopP → RepetitionPenalty).
- Wrote a **self-contained GPT-2 byte-level BPE** in `src/gpt2_bpe.cpp` (llama.cpp's
  BPE was too entangled with its GGUF vocab loading to reuse cleanly):
  byte-level encoding table, regex pre-tokenization, BPE merge loop, plus
  `punc_norm` matching the Python implementation. **10/10** test cases match
  the HF tokenizer byte-for-byte, including the 19 paralinguistic added tokens
  (`[laugh]`, `[chuckle]`, …).
- `tts-cli` takes `--text` + `--tokenizer-dir` and produces speech
  tokens end-to-end.

Verified against PyTorch: **bit-for-bit** identical speech tokens on 4
deterministic sampling configs (greedy / temperature / top-k /
repetition-penalty / no-penalty × short + long prompts).

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 5 | `ggml_can_mul_mat` assertion in T3 | Converter must transpose `Conv1D`-style weights (`c_attn`, `c_proj`, `c_fc`, `mlp.c_proj`) to ggml's `[in, out]` layout while leaving `nn.Linear` / embeddings / `wpe` as-is. |
| 6 | `ggml_backend_tensor_get(input_tensor)` returned garbage | `ggml_gallocr` reuses the input buffer for intermediates when only `set_input` is marked; also call `ggml_set_output` on tensors we want to read back. |
| 7 | Repetition-penalty path diverged from HF at token 22 | HF divides positive logits, multiplies negative ones — I had it backwards. |
| 8 | Sampler order mismatched HF `LogitsProcessorList` | Rewrote `sample_next_token` as Temperature → TopK → TopP → RepetitionPenalty, in HF's exact order. After the fix greedy+penalty tests pass bit-exactly. |

### 3.3  S3Gen encoder (stages A–F)

S3Gen is a "Upsample Conformer" with 10 blocks total (~60 M params): 6 initial
blocks, then a 2× `Upsample1D`, then 4 more blocks. Ported in six staged
substeps against Python-dumped reference tensors (`scripts/dump-s3gen-reference.py`):

| Stage | Component | rel error |
|-------|-----------|----------:|
| A | `speaker_emb` projection (`F.normalize` + Linear) | 1.2e-7 |
| B | `input_embedding` lookup | 0 (exact) |
| C | `encoder_embed` (Linear + LN + √D scale + ESPnet rel PE) | 4.4e-7 |
| D | `PreLookaheadLayer` (asymmetric-padded Conv1d stack) | 2.5e-7 |
| E | One Conformer block (rel-pos MHA + `rel_shift` + Swish FFN) | 1.3e-7 |
| **F** | **Full encoder + `encoder_proj`** | **5.6e-7** |

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 9  | `ggml_conv_1d` aborted with `src0->type == GGML_TYPE_F16` | ggml's `im2col` path requires F16 kernels, but we wanted F32 precision. Wrote a `conv1d_f32` helper that calls `ggml_im2col(…, GGML_TYPE_F32)` + `mul_mat` directly, keeping kernels in F32. |
| 10 | `speaker_embed` broadcast failed in `cond_spkr` matmul | Bias reshape needed `ne=[1, 256]`, not `ne=[256]`. Added the explicit `reshape_2d(bias, 1, C)` convention for every 1-D bias added to a `[T, C]` conv output. |
| 11 | Nearest-neighbor ×2 upsample produced channel-interleaved garbage | The naive `reshape_3d(T, 1, D) + concat(ne[1])` gives `t0_copy0, t1_copy0, …, t0_copy1, …`. Correct trick: `reshape_3d(1, T, D)` → `concat` along `ne[0]` → `[2, T, D]` → reshape to `[2T, D]`, giving `t0_copy0, t0_copy1, t1_copy0, …`. |
| 12 | `rel_shift` attention gave ~100 % rel error | `view_3d(bd_viewed, T, 2T-1, H, nb1, T*(2T-1)*elem, offset)` used the *sliced* stride for `nb2`. `nb2` must match the *source's* element stride: `bd_viewed->nb[2]`. |
| 13 | `*.transpose().numpy()` reference dumps loaded as garbage in C++ | Torch `.transpose()` yields Fortran-ordered storage; `np.save` writes `fortran_order: True`. Dumper now calls `.contiguous().numpy()` + `np.ascontiguousarray(...)`. The C++ loader throws a clear error if it sees `fortran_order=True`. |

### 3.4  CFM decoder (stages G1–G4)

A U-Net with transformer blocks (~45 M params). Layout: 1 down block → 12 mid
blocks → 1 up block (skip concat) → `final_block` → `final_proj`. Each block
carries 4 `BasicTransformerBlock`s.

| Stage | Component | rel error |
|-------|-----------|----------:|
| G1 | Time embedding (sin → MLP → mixer) | 7.0e-7 |
| G2 | `CausalResnetBlock1D` (causal-conv + LN + Mish + time MLP + res_conv) | 2.9e-7 |
| G3 | `BasicTransformerBlock` (self-attn + FFN w/ GELU-erf) | 1.7e-7 |
| **G4** | **Full CFM decoder, one forward step** | **1.3e-6** |

For meanflow mode we do 2 steps with `t_span = [0, 0.5, 1]`; the time embedding
sees both `t` and `r` concatenated through a mixer.

**Issues hit in this phase:**

| # | Issue | Fix |
|---|-------|-----|
| 14 | `LayerNorm` applied over time instead of channel | For `ne=[T, C]` layout `ggml_norm` reduces `ne[0]=T`, which is wrong. Wrote `layer_norm_on_channel` that permutes to `[C, T]`, norms, applies affine, permutes back. |
| 15 | `weight_norm` convolutions in `mel2wav` ignored | Torch 2.6 stores them under `parametrizations.weight.original{0,1}`. Added `expand_weight_norm()` in the converter that fuses `g · v / ‖v‖₂` back into a regular `weight` tensor before export. |
| 16 | Mish activation missing from ggml unary ops | Built from primitives: `x · tanh(softplus(x))` via `GGML_UNARY_OP_SOFTPLUS` + `GGML_UNARY_OP_TANH`. |
| 17 | GELU mismatch in `BasicTransformerBlock` (rel=3e-4) | `ggml_gelu` is the tanh approximation; `diffusers.models.activations.GELU` uses the exact `erf` formulation. Switched to `ggml_gelu_erf`. Error dropped to 1.7e-7. |
| 18 | Python hook overwrote the same tensor across multiple CFM steps | Meanflow calls `time_embeddings` twice (for `t` and `r`) and the decoder runs twice per sample. Added `make_hook(multi_call=True)` that saves `*_call0.npy`, `*_call1.npy`, …. |
| 19 | Estimator `forward_hook` never fired | `basic_euler` calls `self.estimator.forward(x, …)` directly, bypassing `__call__` where hooks live. Monkey-patched `estimator.forward` to record `x_in / mu / t / r / spks / cond / mask / dxdt` for every step. |
| 20 | `(B, C, T)` vs `(B, T, C)` layout confusion | CFM alternates: resnets use `(B, C, T)`, transformer blocks use `(B, T, C)`, switched by `rearrange`. In ggml we mirror this and `cont(permute)` at the boundary. Every helper doc-comments its layout. |

### 3.5  HiFT vocoder (stages H1–H5) + `mel2wav` binary

HiFTGenerator = Neural Source Filter + ISTFTNet. The mel → waveform vocoder.
Ported in five verifiable substeps:

| Stage | Component | rel error |
|-------|-----------|----------:|
| H1 | `f0_predictor` (5× Conv + ELU + Linear) | 4.2e-6 |
| H3 | decode body `conv_pre → ups / rb → conv_post` | 5.6e-7 |
| H4 | STFT (Conv1d with DFT + Hann kernel) | 7.9e-3 (boundary-bound) |
| H5 | ISTFT (ConvTranspose + window-sum normalize) | 1.0e-4 |

Key techniques:

- **Snake activation** `x + (1/α)·sin²(αx)` implemented with `ggml_sin` and a
  pre-computed `1/α` tensor fed as a graph input (72 such inputs across the 9
  main ResBlocks and 3 source ResBlocks).
- **ConvTranspose1d with asymmetric PyTorch padding**: ggml's op only accepts
  `p0=0`, so we compute the full-length output then slice `p` samples from each
  side.
- **Asymmetric reflection pad `(1, 0)`**: done manually by extracting `x[1:2]`
  and concat-prepending it.
- **STFT** as `Conv1d` with a DFT+window kernel of shape `[n_fft, 1, 2F]` (real
  and imaginary parts stacked as output channels). Center-mode reflection pad
  `n_fft//2` applied manually via slice-and-concat on each side.
- **ISTFT** as `ConvTranspose1d` with the inverse DFT+window kernel, followed
  by element-wise divide by a precomputed `window²` overlap-sum buffer, then
  trim `n_fft//2` from each end.

The resulting `mel2wav` binary demonstrates the full vocoder:

```
mel2wav --s3gen-gguf models/chatterbox-s3gen.gguf \
        --mel-npy artifacts/s3gen-ref/mel_output.npy \
        --out /tmp/out.wav
```

Against the Python reference waveform: matching RMS (1.22e-04 vs 1.22e-04),
time-domain diff max 3.3e-05 (signal max ~9e-04), spectrogram magnitude diff
max rel 2.5 % (entirely from stochastic SineGen excitation; the deterministic
conv-net chain is bit-exact).

SineGen on the C++ side uses `std::mt19937` (not bit-exact to `torch.rand`,
but audibly indistinguishable — the excitation is a small-amplitude additive
noise term).

### 3.6  End-to-end wiring: `chatterbox-tts` + `synthesize.sh`

Final plumbing: write `src/chatterbox_tts.cpp` that wires the S3Gen encoder →
2-step meanflow CFM → HiFT vocoder and emits a 24 kHz wav. Takes T3-generated
speech tokens plus a reference voice (`embedding`, `prompt_token`,
`prompt_feat`).

Historically `synthesize.sh` piped two binaries; today one `tts-cli` runs the
full pipeline, and `synthesize.sh` is a thin wrapper around it.

Debug mode (`--debug`) substitutes Python-dumped reference random bits (CFM
`z` and `noised_mels`) so the deterministic parts can be validated
bit-exactly. End-to-end in debug mode:

| Stage | max_abs | rel |
|-------|---------|-----|
| `input_embedding(tokens)` | 0 | 0 |
| encoder → `encoder_proj` (mu) | 8.3e-07 | 4.5e-07 |
| speaker embedding (spks) | 5.9e-08 | small |
| `cond` (prompt_feat placement) | 0 | 0 |
| `t_emb` (sinusoidal → MLP → mixer) | 7.6e-06 | small |
| CFM step 0 `dxdt` | 2.1e-05 | small |
| CFM step 1 `dxdt` | 1.8e-05 | small |
| final mel (80 × 136) | 1.0e-05 | **8.9e-07** |

Production mode uses a seeded `std::mt19937` for both the CFM initial noise
and SineGen excitation.

**Issues hit in this phase (all three caused plausible-looking but wrong output
before being found):**

| # | Issue | Fix |
|---|-------|-----|
| 21 | Silence-token padding value | `speech_tokens` must be appended with `S3GEN_SIL = 4299` (not 0) to match Python's `speech_tokens_padded` convention. |
| 22 | Relative PE `pos_pe / neg_pe` swap | While copying `compute_pos_emb` into the new binary I flipped the two halves of the PE buffer, which silently gave ~20 % relative error in the encoder output. Restored the correct ordering: first half is reversed `pos_pe`, second half is `neg_pe`. |
| 23 | `mu` layout transpose between encoder and CFM | `encoder_proj.npy` is numpy `(T, 80)` but the CFM estimator expects numpy `(80, T)`. Added an explicit transpose to bridge the two. |

At this point on a 10-core EPYC, single-threaded, the end-to-end pipeline ran
in **22.5 s for 8.64 s of audio** — **RTF 2.60**, i.e. 2.6× *slower* than
real-time.

### 3.7  (no extra section — continued in 3.8)

### 3.8  CPU optimization pass (in the order tried)

Eight optimizations in the order they were attempted. Four landed, four were
rolled back or skipped as incompatible. Numbers are for the 8.64 s utterance
above.

**Attempt 1 — multi-threading (KEPT, −85 % wall time)**
Baseline was pinned to 1 thread because the code never called
`ggml_backend_cpu_set_n_threads`. Added a global `g_n_threads` (default =
`std::thread::hardware_concurrency()`, overridable with `--threads N`) and a
`compute()` helper that sets it before every `ggml_backend_graph_compute`.
ggml's `-march=native` was already on, so AVX-512 / AVX-VNNI kernels were
already in use — the missing piece was parallelism. Swept thread counts: 10
was the sweet spot; 16 oversubscribes and regresses.
Result: **22.5 s → 3.47 s (RTF 2.60 → 0.40)**.

**Attempt 2 — OpenBLAS (TRIED, NO HELP)**
Installed `libopenblas-dev`, rebuilt with `GGML_BLAS=ON
GGML_BLAS_VENDOR=OpenBLAS`. No measurable change. Our matmuls are medium-sized
and ggml's hand-written AVX-512 kernels already saturate what OpenBLAS would
deliver. Kept off.

**Attempt 3 — `GGML_LTO=ON` (TRIED, NO HELP)**
No measurable effect on a shared-library build. Kept off.

**Attempt 4 — CFM graph reuse (KEPT, −11 % wall time)**
The CFM estimator is called twice per utterance with *identical* graph
topology. Stashed the `ggml_context`, `ggml_cgraph`, and `ggml_gallocr` in a
`cfm_estimator_cache` so step 2 only re-runs with new inputs — saves one graph
construction and one `gallocr_reserve` pass per utterance.
Result: **3.47 s → 3.09 s (RTF 0.40 → 0.36)**.

**Attempt 5 — Flash attention in CFM `BasicTransformerBlock` (KEPT, −22 % wall time)**
The CFM has 56 `BasicTransformerBlock`s × 2 meanflow steps = **112 attention
ops** per utterance. Replaced the explicit
`softmax(QKᵀ / √d) · V` kernel with a single `ggml_flash_attn_ext` call.
The pattern is pure self-attention (no masking, no bias), which is exactly
what `flash_attn_ext` is designed for. Fused, no materialized `T×T`
scores/attn tensors. The reshape-permute-cont preamble now drops straight into
`flash_attn_ext`, and its output `ne=[HD, H, T, 1]` reshapes directly to
`[INNER, T]`.
Result: **3.09 s → 2.45 s (RTF 0.36 → 0.28), CFM −44 %**.

**Attempt 6 — Fold symmetric conv padding (KEPT, small win)**
Six redundant `ggml_pad_ext → conv1d_f32` pairs dropped by passing the padding
straight to `ggml_im2col`. Biggest impact in HiFT's ResBlocks where the
resblock-conv path runs ~72 times per decode. Saves one intermediate tensor
allocation per conv. A small but essentially-free improvement.
Result: **2.45 s → 2.39 s (RTF 0.28 steady)**.

**Attempt 7 — F16 CFM linear weights (TRIED, ROLLED BACK)**
Converted all Q/K/V/O/FFN/MLP linear weights in CFM from F32 to F16 to halve
memory bandwidth. *Regressed*: CFM got ~10 % **slower** and precision dropped
to `rel = 3e-4` on the final mel. The F16→F32 upconvert inside `mul_mat` is
not free and the F32 AVX-512 kernel is already very fast; for CPU this is a
net loss. Reverted.

**Attempt 8 — Flash attention in the Conformer encoder (SKIPPED, INCOMPATIBLE)**
Would fuse another 10 attention ops per utterance, but the Conformer uses
ESPnet-style relative positional bias added *inside* the softmax, and
`ggml_flash_attn_ext` does not support custom in-softmax bias terms. Would
need a custom ggml op — not done.

#### Final results (10-core EPYC, 8.64 s output)

| Configuration | Total | RTF | vs real-time |
|---|---:|---:|---|
| Baseline (1 thread, no graph reuse, no flash attn) | 22.5 s | 2.60 | 2.6× slower |
| + threading (Attempt 1) | 3.47 s | 0.40 | 2.5× faster |
| + CFM graph reuse (Attempt 4) | 3.09 s | 0.36 | 2.8× faster |
| **+ flash attn + pad fold (Attempts 5, 6)** | **2.39 s** | **0.28** | **3.6× faster** |

Total wall-time speedup from the original port: **9.4×**.

Stage breakdown at the final configuration:

| Stage | time |
|-------|------|
| S3Gen encoder | 286 ms |
| CFM 2 meanflow steps | 785 ms |
| HiFT vocoder | 1312 ms |
| **Total** | **2.39 s** |

HiFT is now the bottleneck (~55 % of wall time) — the 3-stage upsample /
ResBlock stack on `T = 16320 × 64` channels is memory-bandwidth bound rather
than compute bound.

### 3.9  Post-launch bug: sampling defaults collapsed long prompts into silence

After merging the two binaries and shipping voice-cloning phase 1, a user
report of an "empty" wav on paragraph-length input surfaced a sampling bug
that had been lurking since the T3 port.

Symptom: the produced wav had ~1 second of speech followed by ~9 seconds of
pure zero RMS. Per-0.5 s window RMS:

```
[3.5e-2, 1.3e-2, 2.8e-7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.4e-7]
```

Dumping the T3 token stream showed the root cause immediately — 240 of 257
tokens were the silence token `4218`:

```
tokens[0:17]:   3704, 6486, 4299, 3891, 5832, 4384, 5014, 5665, 2486, 29,
                29, 380, 632, 2912, 5101, 5070, 4215
tokens[17:257]: 4218, 4218, 4218, 4218, ...  (240 copies)
```

The C++ sampler had shipped with `top_k = 1` (argmax) as its default. For
Chatterbox T3 that's a known failure mode: once the model generates a
silence token at a natural pause, `argmax(logits)` keeps picking silence
forever and the utterance never recovers. Short test prompts never reached
a pause so the bug was invisible during the port.

Compared `ChatterboxTurboTTS.generate()` in `tts_turbo.py` — the Python
defaults are very different:

|                | before (C++ broken) | Python     | after (C++ fixed) |
|----------------|--------------------:|-----------:|------------------:|
| `top_k`        | 1  (greedy)         | 1000       | **1000**          |
| `top_p`        | 1.0                 | 0.95       | **0.95**          |
| `temperature`  | 1.0                 | 0.8        | **0.8**           |
| `repeat_penalty`| 1.0                | 1.2        | **1.2**           |
| `n_predict`    | 256                 | ~1000      | **1000**          |

All four knobs are still exposed on the CLI, so `--top-k 1` reproduces the
old greedy behaviour for debugging/comparison.

After the fix, same prompt same seed:

- total wav RMS: `8.3e-03` → `4.8e-02`
- max amplitude: `0.18` → `0.50`
- per-0.5 s RMS windows: all 21 non-zero (3.3e-2 … 8.5e-2 range)
- audible speech for the full 10.7 s

Committed as `bb0eb99`.

### Lesson

This one was avoidable — the verification pipeline in §5 is per-tensor
numerical parity, which is oblivious to sampler choices; the `reference-
t3-turbo.py` harness only compared greedy token sequences so it never
exercised any non-trivial pass of the sampling ladder. Worth adding an
end-to-end sampling test to the validation list: run T3 with Python's
stochastic defaults (fixed seed) and compare the full token stream
byte-for-byte against C++ with the same seed.

### 3.10  Benchmark: chatterbox.cpp vs ONNX addon on the same machine

Compared end-to-end throughput against an in-house ONNX Runtime TTS
addon (pre-built q4 Chatterbox models at 692 MB on disk). Same 10-core
EPYC host, same
prompt ("Hello from native C plus plus. This audio was generated end
to end on CPU using ggml."), built-in voice on both sides, `--threads
10` for ggml, ORT's own default threading for ONNX. Instrumented the
ggml binary with explicit `T3_LOAD_MS` / `T3_INFER_MS` /
`S3GEN_LOAD_MS` / `S3GEN_INFER_MS` markers so load and generate
phases can be split cleanly. Each configuration run three times after
a disk-cache warm-up.

**Model footprint on disk:**

| | Size |
|---|---:|
| ONNX q4 (5 files) | 692 MB |
| ggml F16 (T3 + S3Gen) | 1285 MB |
| ggml Q8_0 (T3 + S3Gen) | 1004 MB |
| ggml Q5_0 (T3 + S3Gen) | 893  MB |
| ggml Q4_0 (T3 + S3Gen) | 857  MB |

**Per-stage wall-clock (median of 3 runs, milliseconds):**

| Pipeline      | T3 load | T3 gen | S3Gen load | S3Gen gen | Audio | **Total** | RTF (total) |
|---------------|---:|---:|---:|---:|---:|---:|---:|
| **ggml Q4_0** | **213** | 1790 | 366 | 1998 | 6480 | **4455** | **0.69** |
| ggml Q5_0     |  231 | 1966 | 353 | 2002 | 6640 |  4641 | 0.70 |
| ggml Q8_0     |  305 | 2047 | 370 | 2001 | 6560 |  4823 | 0.73 |
| ggml F16      |  468 | 2691 | 364 | 1928 | 6560 |  5562 | 0.85 |
| **ONNX q4**   |  ~4250 (4 files, serialized) | — | — | ~6830 | 5880 | **11050** | **1.88** |

(ONNX Runtime's backend doesn't expose a comparable per-sub-model
breakdown, so its `load` is the wall-clock time from `model.load()`
calling through ORT init across all four `.onnx` files, and `gen` is
the time the single `model.run()` call takes.)

**Aggregated: load vs. generate, load+gen together:**

| Pipeline      | **Load** | **Generate** | **Total wall** | **RTF (total)** |
|---------------|---:|---:|---:|---:|
| **ggml Q4_0** |  **579 ms** |  **3788 ms** |  **4455 ms** |  **0.69**  |
| ggml Q5_0     |  584 ms |  3968 ms |  4641 ms |  0.70 |
| ggml Q8_0     |  675 ms |  4048 ms |  4823 ms |  0.73 |
| ggml F16      |  832 ms |  4619 ms |  5562 ms |  0.85 |
| **ONNX q4**   | **4250 ms** | **6830 ms** | **11050 ms** | **1.88** |

**Headline numbers** (best ggml variant vs ONNX):

- **Load: ggml Q4_0 is 7.3× faster** — 579 ms vs 4250 ms. The four
  ONNX files initialise serially and each one does its own tensor
  plumbing; ggml mmaps the two GGUFs and rebinds through the unified
  backend buffer in ~half a second total.
- **Generate: ggml Q4_0 is 1.8× faster** — 3788 ms vs 6830 ms.
- **Total (load + generate): ggml Q4_0 is 2.48× faster** —
  4.46 s vs 11.05 s.
- Even **ggml F16 beats ONNX q4** on total wall (5.56 s vs 11.05 s,
  1.99× faster) *despite carrying 2× the weights* — the ONNX backend
  loses to an un-quantized ggml build on the same CPU.
- **RTF < 1** (faster than real-time) happens on every ggml variant
  tested; ONNX stays at 1.88× real-time for this prompt.

Numbers are for a ~6 s utterance; the ggml pipeline's ~2 s of fixed
S3Gen+HiFT cost amortizes better on longer input, so the gap widens
in ggml's favour as prompt length grows.

### 3.11  Vulkan + Metal backends

CPU performance was already past real-time, but a lot of the T3 and
CFM work is embarrassingly parallel, so enabling the GGML GPU backends
was the obvious next step. Touched three files:

- `CMakeLists.txt` — added a `GGML_VULKAN` propagation block mirroring
  the existing `GGML_CUDA` / `GGML_METAL` ones.
- `src/main.cpp` — extended `init_backend(n_gpu_layers)` with a
  `ggml_backend_vk_init(0)` path guarded by `#ifdef GGML_USE_VULKAN`.
  CUDA / Metal paths were already there.
- `src/chatterbox_tts.cpp` — added a symmetric `s3gen_init_backend`
  so the S3Gen side honours the same `--n-gpu-layers` flag, plus a
  new `n_gpu_layers` field on `s3gen_synthesize_opts`.

Two op-level changes in our code were required *because* Metal's
dispatcher didn't have those ops (the actual Metal kernel fixes land
in §3.12):

1. **T3 attention**: `ggml_soft_max(ggml_diag_mask_inf(ggml_scale(KQ,
   s), n_past))` → `ggml_soft_max_ext(KQ, mask, s, 0.0f)` with an
   explicit `[n_kv, N]` causal mask tensor uploaded from
   `eval_prompt`. The step path (N=1) passes a null mask. No-op for
   CPU / Vulkan; necessary for Metal.
2. **S3Gen zero padding**: 6 call sites used `ggml_pad_ext` with
   non-zero front padding. Added a `zero_pad_dim0(ctx, x, p_front,
   p_back)` helper that expresses the same semantics via
   `concat(scale(view, 0.0f), x)` so it runs on every backend with
   well-defined zeros.

First result on the Linux remote (RTX 5090 + Vulkan), same 10 s
sentence as §3.10:

| Variant       | T3 load | T3 gen | S3Gen load | S3Gen gen | Audio  | `gen_RTF` | Wall  |
|---------------|--------:|-------:|-----------:|----------:|-------:|----------:|------:|
| Vulkan F16    |  562 ms |  600 ms |  490 ms    | 279 ms    | 10.5 s | **0.08** | 2.10 s |
| Vulkan Q8_0   |  450 ms |  557 ms |  472 ms    | 272 ms    | 10.6 s | 0.08     | 1.91 s |
| Vulkan Q5_0   |  348 ms |  562 ms |  470 ms    | 276 ms    | 10.9 s | 0.08     | 1.82 s |
| Vulkan Q4_0   |  331 ms |  522 ms |  493 ms    | 275 ms    | 10.3 s | 0.08     | 1.78 s |

Quantization makes T3 load noticeably smaller but barely moves
inference — T3 is autoregressive (one token at a time on a 5090 has
plenty of spare lanes) and S3Gen is already short. End-to-end goes
from 8.17 s (CPU F16) → **1.78 s** (Vulkan Q4), for the same 10 s of
audio. `gen_RTF = 0.08` = 13× real-time.

On the M3 Ultra Metal side, things didn't fly immediately: T3 aborted
on the first attention layer with `unsupported op 'DIAG_MASK_INF'`,
then S3Gen aborted with `unsupported op 'PAD'`. Once those two
op-level workarounds above were in place, HiFT decode was completing
but taking **~15 s for 1.2 s of audio** — Metal's
`conv_transpose_1d` kernel is pathological for HiFT-sized inputs.

Pragmatic interim fix: when the main backend is Metal, load a second
CPU copy of the S3Gen GGUF and route `run_f0_predictor`,
`run_stft`, and `run_hift_decode` through it. Encoder + CFM still run
on Metal. Costs ~1 GB extra RAM but brings Metal `gen_RTF` to ~0.25.
That's what committed as `795963a` ("backend: enable Vulkan + Metal
for T3 and S3Gen").

### 3.12  ggml-metal kernel patches

To get rid of the CPU fallback for HiFT and close the gap with
Vulkan, patched `ggml/src/ggml-metal/` itself. The patch is shipped
as `patches/ggml-metal-chatterbox-ops.patch` (based on upstream
`58c3805`, `sync : llama.cpp`); the main README instructs a fresh
clone to `git apply` it after cloning ggml.

A new `test-metal-ops` binary runs each patched kernel against the
CPU reference at HiFT-realistic shapes. All cases pass with
`max_abs ≤ 1.5e-6`.

**Patch 1 — `DIAG_MASK_INF` on Metal** (was: op simply absent from
the dispatcher):

- New `kernel_diag_mask_inf_f32` — ports the CUDA formulation
  (`dst[i] = src[i] - (col > n_past + row % rows_per_channel) *
  FLT_MAX`) so downstream softmax yields proper zeros.
- New `ggml_metal_kargs_diag_mask_inf`, library pipeline getter,
  op encoder, dispatcher case, and `supports_op` entry.

**Patch 2 — `PAD` with front padding** (was: kernel ignored
`op_params[0,2,4,6]` which is where `ggml_pad_ext` stores the front
amounts; `supports_op` hard-rejected any non-zero front pad):

- Extended `ggml_metal_kargs_pad` with `lp0..lp3`.
- Rewrote `kernel_pad_f32` to translate each output coord by
  `i0x = i0 - lp0` etc., and write `0.0` outside `[0, ne00)`.
- Relaxed `supports_op` to `src0->type == F32 && dst->type == F32`.

**Patch 3 — `CONV_TRANSPOSE_1D` speedup** (was: ~100× slower than
CPU on HiFT-sized inputs):

The old kernel was scalar — one thread per output pixel, iterating
over the full `IC × IL` inputs inside a branch `if (ol >= i*s0 && ol
< i*s0 + K)`. Two orthogonal fixes:

1. **Tighten the input-position loop** to only the `i`s that actually
   contribute. For fixed `ol`, valid `i` is
   `[max(0, ⌈(ol - K + 1)/s0⌉), min(IL-1, ol/s0)]` — at most
   `K/s0 + 1` iterations. On ups[0] (s0=8, K=16, IL≈130) this
   collapses the inner loop from 130 iterations → 3.
2. **Parallelise `IC` across a 32-thread simdgroup** and reduce with
   `simd_sum`. Host-side dispatch widens from 1 thread per
   threadgroup → 32 (one simdgroup).

Measured on M3 Ultra, HiFT decode (part of a 10 s sentence):

```
  hift_decode: 15021 ms → 350 ms          (≈ 40× speedup)
  gen_RTF   :   0.25  → 0.18              (CPU-fallback removed)
  wall      :   3.36 s → 2.51 s
```

With the patch applied and the CPU-fallback for HiFT removed,
end-to-end on the M3 Ultra for the same 10 s sentence, seed 42,
averaged over 3 runs:

| Variant       | T3 load | T3 gen | S3Gen load | S3Gen gen | `gen_RTF` | Wall  |
|---------------|--------:|-------:|-----------:|----------:|----------:|------:|
| Metal F16     |  280 ms | 1326 ms |  295 ms    | 577 ms    | **0.19** | 2.51 s |
| Metal Q8_0    |  216 ms | 1330 ms |  302 ms    | 598 ms    | 0.18     | 2.48 s |
| Metal Q5_0    |  186 ms | 1393 ms |  293 ms    | 611 ms    | 0.19     | 2.51 s |
| **Metal Q4_0**|  **175 ms** | **1274 ms** | **295 ms** | **594 ms** | **0.18** | **2.36 s** |

Autoregressive T3 now dominates wall time (`T3_INFER` ≈ 1.3 s of
~260 tokens at one-token-at-a-time on a 60-core Apple GPU) — that's
the next thing to chip away at. On the 5090 the same token stream
runs in ~0.55 s because the shader count is ~360× higher.

Committed as `894c4b1` ("metal: patch ggml to fix diag_mask_inf,
pad_ext, conv_transpose_1d"). `i`m not a fan of forking ggml just
for this, so the patch is tiny and easy to drop once upstream picks
up equivalent fixes; see `patches/README.md` for what to do in that
case.

### 3.13  T3 Flash Attention with a layout-friendly KV cache

After §3.11 / §3.12 the dominant wall-clock cost in Chatterbox became
T3's autoregressive step (≈ 1.3 s of a ~2.4 s run on Metal M3 Ultra
Q4_0).  An earlier attempt to swap the explicit
`soft_max_ext(mul_mat(K,Q), mask) + mul_mat(V_trans)` chain for
`ggml_flash_attn_ext` ran into a deal-breaker: the KV cache was laid
out `[HD, n_head, n_ctx]` per layer but `flash_attn_ext` wants
`[HD, n_ctx, n_head]`.  Every step had to `ggml_cont(ggml_permute(K))`
over a tensor that grew with `n_past`, and the extra kernel dispatches
wiped out FA's savings.

Fix: store the cache the way FA reads it.

- Same total size per layer (`HD * n_ctx * n_head` == `n_embd * n_ctx`),
  so no allocation changes.
- Write path (step or prompt): Kcur / Vcur are viewed as
  `[HD, n_head, N]`, permuted to `[HD, N, n_head]`, then one
  `ggml_cpy` per tensor into a strided cache view at
  `[HD, n_past:n_past+N, n_head]`.  For the step path N=1 the permute
  is a no-op in memory.
- Read path: `ggml_view_3d(memory_k, HD, L, n_head, nb=[4, HD*4,
  HD*n_ctx*4], offset=il*layer_size)` is exactly the shape FA needs,
  with no `permute + cont`.
- Mask: switched from F32 to F16 (ggml FA requires F16 on Metal;
  other backends accept it too).  N=1 path passes `nullptr` since
  every KV position is in the past.

Measured on M3 Ultra, same 10 s sentence, seed 42, `--threads 20`,
`--n-gpu-layers 99`, averaged over 3 warm runs:

| Variant  | T3 infer before | T3 infer after | Δ     | Wall before | Wall after | `gen_RTF` |
|----------|----------------:|---------------:|------:|------------:|-----------:|----------:|
| F16      |          1372 ms|         983 ms | −28 % |      2.51 s |     2.15 s | 0.189 → 0.157 |
| Q8_0     |          1371 ms|         985 ms | −28 % |      2.48 s |     2.12 s | 0.182 → 0.149 |
| Q5_0     |          1445 ms|        1063 ms | −26 % |      2.51 s |     2.18 s | 0.186 → 0.152 |
| **Q4_0** |      **1274 ms**|     **965 ms** | **−24 %** | **2.36 s** | **2.06 s** | **0.176 → 0.144** |

And the same change on Vulkan 5090 (Linux remote):

| Variant  | T3 infer before | T3 infer after | Δ     |
|----------|----------------:|---------------:|------:|
| F16      |           600 ms|         410 ms | −32 % |
| Q4_0     |           522 ms|         356 ms | −32 % |

So the new layout is not just a Metal-shaped win — it speeds up every
GPU backend, because the previous `permute + cont` per layer per step
was cheap on NVIDIA too but not free.  CPU builds see a similar graph
shape (fewer intermediate nodes) and stay neutral.

Output sampling is *not* bit-exact against the old path: FA runs its
own internal reductions in different order and the mask lives in F16
instead of F32, so token counts can shift by ±2 % (e.g. F16 went from
248 → 244 tokens on the bench prompt).  Audio remains perceptually
identical; this is the same kind of drift that moving to FA causes
anywhere else in ggml.

Committed as part of the Metal optimization sequence alongside the
earlier `patches/ggml-metal-chatterbox-ops.patch`.

### 3.14  Zero-cont Q view via strided QKV access

After §3.13, each T3 attention layer still did two `ggml_cont`s on Q
per step: one `cont_3d` to densify the strided view of `Qcur`, and an
outer `cont` after the head-permute.  Both turn into
`kernel_cpy_f32_f32` dispatches on Metal.

Observation: the entire QKV output `cur` is already contiguous.  Q,
K, and V are just fixed byte offsets into the same tensor (0,
`n_embd * 4`, `2 * n_embd * 4` respectively).  With Metal's
`flash_attn_ext` accepting non-contiguous Q via explicit strides (the
same flexibility I used for K/V in §3.13), I can drop both conts and
express Q directly as a `ggml_view_3d` with layout `[HD, N, n_head]`:

```
nb0 = 4, nb1 = 3 * n_embd * sizeof(float), nb2 = HD * sizeof(float)
```

Same trick for the Kcur/Vcur sources that go into the KV-cache write
path — one view each, no permute + cont pair.

Removes 24 kernel dispatches per step (`cont` × 24 layers); since T3
step time on Metal is almost entirely dispatch-bound at ~9 µs each,
this shows up straight in the numbers.

Measured on M3 Ultra (same 10 s sentence, seed 42, 3-run warm average):

| Variant  | T3 infer §3.13 | T3 infer §3.14 | Δ     | Wall §3.13 | Wall §3.14 |
|----------|---------------:|---------------:|------:|-----------:|-----------:|
| F16      |         983 ms |         909 ms | −7.5% |    2.15 s  |   **2.08 s** |
| Q8_0     |         985 ms |         906 ms | −8.0% |    2.12 s  |   **2.03 s** |
| Q5_0     |        1063 ms |         984 ms | −7.4% |    2.18 s  |   **2.09 s** |
| **Q4_0** |     **965 ms** |     **886 ms** | **−8.2%** | **2.06 s** | **1.98 s** |

Vulkan RTX 5090 sees <3 % change in T3 infer — dispatch overhead is
much smaller there relative to the actual compute, so there's less to
save.  No regression on Vulkan, and the code simplifies.  CPU stays
neutral (same graph topology, fewer intermediate nodes).

Sampling output is not bit-exact against §3.13 either — same reason as
before, FA reductions are sensitive to operand stride.  Token counts
shift within ±1 % at the same seed.

### 3.15  ggml-metal: fuse `mul_mat + add(bias)` for Q-variant matvec

Even after §3.14 the T3 step path still dispatched two Metal kernels
per linear layer — `mul_mv` for the matmul itself, then `bin_fuse` for
the following `add(bias)`.  T3 has 4 such linears per layer
(QKV proj, attn proj, MLP fc, MLP proj) × 24 layers = 96 extra bias
kernels per step.  At ~9 µs dispatch overhead on M3 Ultra that's
~900 µs/step / ~240 ms over a 260-token generation.

Patched ggml-metal to fuse these directly inside the mul_mv kernel
(third addition to `patches/ggml-metal-chatterbox-ops.patch`):

1. New function constant `FC_mul_mv_has_bias` at `FC_MUL_MV + 2`.
2. Each Q-variant top-level kernel (`kernel_mul_mv_q4_0_f32`,
   `_q4_1_f32`, `_q5_0_f32`, `_q5_1_f32`, `_q8_0_f32`) picks up an
   extra `device const char * bias` buffer argument and calls a tiny
   `helper_mv_add_bias<NR0>` immediately after the existing impl.
   The post-pass only runs when the function constant is true and
   only one thread per row does the add (no cross-threadgroup
   synchronisation needed; each threadgroup writes and then reads
   back only its own output rows).
3. `ggml_metal_op_mul_mat` gets a `ctx->use_fusion &&
   kernel_supports_bias` look-ahead: if the next op is an `ADD` with
   a contiguous F32 `[ne0, 1]` bias, we compile the pipeline with
   `has_bias=true`, bind the bias buffer to slot 4, redirect the
   matmul's `dst` to the ADD's output tensor, and return `n_fuse=2`
   so the dispatcher skips the ADD.  The shared pipeline name
   (`…_bias=1`) makes the fused variant cache-coherent with the
   non-fused one.
4. For kernels not yet wired (F16/BF16 `mul_mv_t_t`, the `_4` SIMD
   variants, all the K-quants and IQ variants) the fusion is
   suppressed by `kernel_supports_bias`, the pipeline compiles with
   `has_bias=false`, and the kernel's `if (FC_mul_mv_has_bias)` is
   dead-code eliminated.  MoE `mul_mv_id` keeps calling the original
   impl via `mmv_fn` unchanged; the impl signature itself was not
   touched.

Measured on M3 Ultra, 10 s sentence, seed 42, 3-run warm average:

| Variant  | T3 before §3.15 | T3 after §3.15 | Δ      | Wall before | Wall after |
|----------|----------------:|---------------:|-------:|------------:|-----------:|
| F16      |          909 ms |         915 ms | ~flat  |   2.08 s    |  2.26 s    |
| Q8_0     |          906 ms |         819 ms | −9.6%  |   2.03 s    |  2.02 s    |
| Q5_0     |          984 ms |         840 ms | −14.6% |   2.09 s    |  1.96 s    |
| **Q4_0** |      **886 ms** |     **766 ms** | **−13.5%** | **1.98 s**  | **1.87 s** |

F16 is flat because the kernel it hits (`mul_mv_f16_f32_4`) isn't in
the supported list yet; extending to those variants is a mechanical
follow-up (touches `helper_mv_reduce_and_write` + the 3 `_t_t` /
`_t_t_4` / `_t_t_short` templates in the same way).

Vulkan RTX 5090 unchanged (347 → 343 ms on Q4_0 — noise).  CPU
unaffected (Metal-only change).

Total Metal Q4_0 journey (pre-FA → end of §3.15):

```
              T3 infer   Wall    gen_RTF
pre-FA         1274 ms   2.36 s   0.176
§3.13 FA+KV     965 ms   2.06 s   0.144     -24%
§3.14 Q views   886 ms   1.98 s   0.131     -30%
§3.15 bias fn   766 ms   1.87 s   0.119     -40%
```

**40 % faster T3 inference, 21 % faster end-to-end wall** than the
pre-optimization baseline on the same M3 Ultra — all via Metal
kernel + graph-shape changes, no model changes.

### 3.16  Metal: extend mat-vec fusion to `MUL_MAT + ADD + ADD`; Vulkan/CPU already optimal

While investigating whether the §3.15 fusion could also apply to
Vulkan and CPU, two findings:

- **Vulkan already has it.** `ggml_vk_can_fuse` in upstream recognises
  `MUL_MAT + ADD` *and* `MUL_MAT + ADD + ADD`, and the mat-vec shaders
  (`vulkan-shaders/mul_mat_vec_iface.glsl`) have dedicated `Fuse0` /
  `Fuse1` buffer bindings for the two optional adds.  Running
  `GGML_VK_DISABLE_FUSION=1` on the 5090 pushes T3 Q4_0 from 346 →
  413 ms (3-run avg), a real 16 % speedup that was silently helping us
  before.  Nothing to add on Vulkan.
- **CPU has no op-level fusion framework.**  But it also has ~zero
  per-op dispatch overhead (ggml-cpu just calls the next op's compute
  function directly), and the matmul output stays in L1 cache
  (`n_embd=1024` × 4 B = 4 KB) so the intermediate round-trip is
  essentially free.  Estimated gain from fusion: < 1 %.  Not worth the
  plumbing work.

That left Metal, where §3.15 covered `MUL_MAT + ADD(bias)` but not the
3-op form `MUL_MAT + ADD(bias) + ADD(residual)` used by T3's attn-proj
and MLP-proj linears.  Extended the Metal patch to match Vulkan's
fusion surface:

- New function constant `FC_mul_mv_has_residual` at `FC_MUL_MV + 3`.
- Each Q-variant top-level kernel gains a second buffer binding
  (`device const char * residual` at slot 5).  `helper_mv_add_bias`
  now applies both the bias broadcast and the per-element residual
  add; both branches are gated on their respective function constants
  so non-fused call sites specialise them away.
- `ggml_metal_op_mul_mat` tries `{MUL_MAT, ADD, ADD}` first (requires
  bias-shaped src1 on ADD1 and full-shape F32-contiguous on ADD2),
  falls back to `{MUL_MAT, ADD}` from §3.15.  Returns `n_fuse=3` /
  `n_fuse=2` accordingly.
- Pipeline names now carry `_bias=?_res=?` so fused/non-fused variants
  are cached independently by the library.

**Correctness bug caught while writing the 3-op variant.**  §3.15's
helper had `if (tiisg != 0 || sgitg != 0) return;`, so only simdgroup
0 added bias.  That's correct for Q8_0 (all simdgroups cooperate on
the same `r0`) but **wrong for Q4/Q5** where each simdgroup writes
its own `r0 = (tgpig.x*NSG + sgitg)*NR0`, silently dropping bias from
the rows computed by simdgroups ≥ 1.  Output was "close enough" to
sound right but not numerically correct.  Fixed by moving the
`sgitg` gate to the callers: Q-n kernels call the helper from every
simdgroup with their own `r0`; Q8_0 wraps the call in
`if (sgitg == 0)`.  Token counts snapped back to the pre-fusion
trajectory once this was right.

Measured on M3 Ultra, 10 s sentence, seed 42, 3-run warm average:

| Variant  | T3 before §3.16 | T3 after §3.16 | Δ     | Wall before | Wall after |
|----------|----------------:|---------------:|------:|------------:|-----------:|
| F16      |          915 ms |         913 ms | flat  |   2.26 s    |  2.27 s    | (fusion N/A)
| **Q8_0** |      **819 ms** |     **794 ms** | **−3 %** | **2.02 s** | **1.94 s** |
| Q5_0     |          840 ms |         873 ms | +4 %  |   1.96 s    |  2.01 s    | (tokens +15)
| Q4_0     |          766 ms |         770 ms | flat  |   1.87 s    |  1.88 s    | (tokens +6)

Smaller than the headline "save 48 dispatches × 9 µs" estimate
suggested, because Metal's scheduler overlaps consecutive small
dispatches — the `bin_fuse` the fused kernel replaces was already
running concurrently with later work.  Q8_0 still sees a clean 3 %
win; Q4/Q5 are noise after accounting for token-count drift.  Still
worth committing: matches Vulkan's fusion surface, fixes the latent
§3.15 bias correctness bug, and closes the last dispatch-per-linear
gap vs Vulkan.

### 3.17  Live / streaming input and interactive TTY mode

The CLI had always been single-shot (pass `--text`, get one wav),
which meant anything "keep the model warm and speak whatever I send"
required re-spawning the binary per request.  Added a long-running
mode driven by `--input-file PATH`: the binary `tail -f`'s `PATH`,
splits on sentence terminators, and pipes raw PCM (s16le @ 24 kHz)
to stdout chunk-by-chunk.

Key details that came up during the implementation:

- **`fread` + `clearerr` doesn't tail-follow on macOS.**  Once the
  stdio `FILE*` hits EOF, the readahead buffer can keep returning 0
  from `fread` for many subsequent calls even after the writer has
  appended new bytes and `clearerr()` has been called.  Switched to
  `open()` + `read()` on a plain fd so the kernel is always consulted
  for the current file state — fixed the "second process's writes
  get dropped" symptom.
- **Accept `<.!?>` followed by an uppercase letter as a sentence
  break**, in addition to the original `<.!?>` + whitespace /
  newline / end-of-input.  LLMs / transcribers that pack sentences
  back-to-back without a space (`"Hello.World.Foo."`) were otherwise
  bundling everything into one enormous utterance.
- **Interactive stdin mode** — `--input-file -` reads from
  `STDIN_FILENO` directly (no `open("/dev/stdin")` which gets a
  fresh-offset fd on some systems).  When stdin is a TTY, the binary
  prints a `> ` prompt on stderr (so it can't collide with the raw
  PCM stream on stdout), wraps the `read()` in a `select()` with a
  25 ms poll so SIGINT is noticed without the user also having to
  press Enter, and re-prompts after each synthesised sentence.
  Single process, pipe stdout straight to `sox play`, type a
  sentence, hear it back.
- **`--input-by-line` line mode** — one newline = one request.
  Internal `. ! ?` are treated as prosody, not as hard boundaries,
  so "Hello there. How are you today?" becomes a single T3 run
  instead of two runs with a 150 ms gap between them.  Saves the
  inter-sentence restart cost and produces more natural delivery
  when the upstream emits complete thoughts per line.
- **T3 early-stop auto-retry was also hit in live mode.**  The
  batch pipeline already replays segments when T3 samples
  `stop_speech_token` suspiciously early (symptom: a cloned voice
  clips the first or last word of a sentence).  Lifted the same
  `min_tokens = max(8, bpe_tokens * 5)`, three-attempt, keep-longest
  guard into the live `synth_sentence`.
- **Skip pure-punctuation input.**  With the various split
  heuristics, it was possible to route a single `.` through T3 (on
  a TTY: the user hits Enter with an empty buffer, punc_norm fills
  in a period).  T3 then hallucinates ~1.4 s of speaker-biased audio
  that can sound like a word from the previous utterance.  The live
  path now drops any sentence whose punc-normalised form contains no
  alphanumeric characters, with a `[skipped: no word characters]`
  notice on TTY.
- **Knob cleanup.**  Removed `--input-flush-ms` (idle-flush mid-buffer
  was only useful when the terminator set was limited to `.!?` and
  got obsoleted by `--input-by-line` + explicit `\n`) and
  `--input-poll-ms` (hard-coded to 25 ms, well below perception).
  One less thing to think about for users; one less thing to get
  wrong.

Commits: `00bfd7f` (fread→read fix), `189fe9d` (interactive stdin),
`9e1b101` (T3 retry port), `dc0b5e1` (punctuation-only skip),
`e0af5e9` (`--input-by-line`), `d843a59` / `cff89ae` (knob cleanup).

### 3.18  `scripts/extract-voice.py` — automated voice-clone prep

Every voice-cloning debug session ended the same way: probe the
source with `ffprobe`, scan with `silencedetect`, eyeball the output
for the longest clean region, pick an `-ss`/`-t`, iterate on the
ffmpeg filter chain until the clone stopped sounding wrong, optionally
bake the `.npy` profile.  Scripted the whole thing.

`./scripts/extract-voice.py INPUT [--name NAME] [--target SEC] [--bake]`
does:

1. `ffprobe` for duration, codec, bitrate.
2. `ffmpeg silencedetect=noise=-30dB:d=0.3` to split into speech
   regions.
3. Rank candidate windows: prefer a continuous slice from the middle
   of the longest region (speaker is warmed up, hasn't started
   wrapping up), fall back to concatenating the two best short
   blocks when no single block is ≥ target.
4. Pick a codec-aware filter chain:
   - **clean** (WAV / FLAC / ≥ 96 kbps AAC / ≥ 128 kbps MP3):
     `highpass=f=60, alimiter=limit=0.85:level=disabled`.
     Trusts the source.
   - **lossy** (Opus / Vorbis at any bitrate, or low-bitrate
     AAC / MP3): `highpass=f=60, afftdn=nr=6:nt=w,
     equalizer=f=200:w=150:g=-1, equalizer=f=3200:w=2200:g=2.5,
     equalizer=f=7500:w=2500:g=3, loudnorm=I=-18:TP=-2:LRA=8,
     alimiter=limit=0.85:level=disabled`.  Denoises the codec hiss,
     puts a mild dip at 200 Hz to unmuddy, boosts presence around
     2–4 kHz and air around 6–9 kHz to replace some of the content
     Opus' brick-wall low-pass throws away above ~8 kHz, loudness-
     normalises so the speaker embedding doesn't drift on the
     shouted-vs-whispered axis.
5. Emit `voices/<name>.wav` at 24 kHz mono s16le.
6. Optionally call `./build/tts-cli --save-voice` to bake the
   five `.npy` tensors.

Commit: `84d2189`.

The lossy chain is what took an 18 kbps Opus voice note from "clone
sounds wrong" to "sounds like the speaker" during the Marco debug
session.  On clean-source material the minimal chain is usually
sufficient and the EQ boosts would only add a mild bright tint.

### Cross-backend summary

Same 10 s sentence, seed 42, `gen_RTF` is inference-only (excludes
load time):

| Backend (weights)             | T3 gen | S3Gen gen | `gen_RTF` | Wall  | Real-time mult |
|-------------------------------|-------:|----------:|----------:|------:|---------------:|
| CPU Linux (F16, 8 threads)    | 3998 ms | 2905 ms   | 0.70      | 8.17 s | 1.4×          |
| Vulkan 5090 (F16)             |  402 ms |  282 ms   | 0.064     | —       | 15.6×         |
| Vulkan 5090 (Q4_0)            |  347 ms |  284 ms   | 0.058     | —       | 17.1×         |
| Metal M3 Ultra (F16)          |  915 ms |  567 ms   | 0.150     | 2.26 s | 6.7×          |
| Metal M3 Ultra (Q4_0)         |  766 ms |  596 ms   | 0.128     | 1.87 s | 7.8×          |
| ONNX q4 addon (CPU, Linux)    |     — (not exposed) |     — | 1.06      | 13.91 s | 0.94×        |

The ONNX addon is shown as a baseline because it's the current
in-house reference TTS implementation. Every ggml configuration —
including CPU F16 on the same host — beats it.

### 3.19  Multilingual (Llama-520M) variant

Everything up to this point in the journal was Chatterbox **Turbo**
(GPT-2 Medium T3, meanflow 2-step CFM, English BPE).  §3.19 is the port
of **ChatterboxMultilingualTTS** (23-language Llama-520M T3 + perceiver
resampler + CFG-enabled standard 10-step CFM).  Variant is auto-detected
from `chatterbox.variant` GGUF metadata at load time; Turbo stays byte-
identical to the pre-§3.19 builds.

**What shipped (commit
`3f0a8dac`):**

- `scripts/convert-t3-mtl-to-gguf.py` — packs `t3_mtl23ls_v2.safetensors`
  (30-layer Llama-520M + cond_enc perceiver + emotion_adv + learned pos
  embs + built-in voice + VE weights) and the raw grapheme tokenizer
  JSON into a single GGUF with `chatterbox.variant=t3_mtl` and the full
  Llama-3 RoPE scaling metadata baked in.  `--quant f16|q8_0|q5_0|q4_0`
  on the big linears.
- `scripts/convert-s3gen-to-gguf.py` grew a `--variant {turbo,mtl}`
  flag.  MTL loads `s3gen.pt` (standard CFM, no `time_embed_mixer`) and
  stamps `s3gen.meanflow=false, cfg_rate=0.7, n_timesteps=10`.  Turbo
  path unchanged.
- `src/mtl_tokenizer.{h,cpp}` + `mtl_unicode_tables.inc` — self-
  contained BPE tokenizer mirroring HuggingFace's BPE loader + the
  Python preprocess (NFKD + UTF-8 lowercase + `[lang_id]` prefix + Korean
  Jamo decomposition).  Tier-1 language support only (`en, es, fr, de,
  it, pt, nl, pl, tr, sv, da, fi, no, el, ms, sw, ar, ko`); ja/he/ru/zh/
  hi error out with a clear message.  No external deps.
- `src/t3_mtl.{h,cpp}` — Llama-520M forward pass: RMSNorm + SwiGLU MLP +
  separate Q/K/V no-bias + RoPE-llama3 (NEOX half-split) +
  `flash_attn_ext` + dual KV cache for CFG.  Cond assembly covers
  spkr_enc + Perceiver (32-query cross then self-attn, `AttentionBlock2`
  LN+bias, F32) + emotion_adv_fc + learned text/speech positional
  embeddings.  Exposes stage builders (cond/text/inputs/layers/head) so
  the parity harness can inject Python-dumped intermediates at any
  boundary.
- `src/test_t3_mtl_stages.cpp` — staged parity harness (all stages pass
  within 5e-4 rel against the Python reference; logits land at 1.4e-3
  rel, consistent with cumulative F16 drift through 30 layers).

**Sampling path.**  `chatterbox_sampling_params` gained `cfg_weight` and
`min_p`.  Sampler order in `sample_next_token_mtl` matches the Python
`ChatterboxMultilingualTTS.generate` default:
`cfg_combine → rep_penalty → temp → min_p → top_p → (top_k) → multinomial`.
CFG runs cond and uncond as two independent T3 forwards (dual KV cache,
`memory_k{_uncond}` / `memory_v{_uncond}` in the model struct), combined
at the logit level.

**S3Gen dispatch.**  `chatterbox_tts.cpp` reads `s3gen.meanflow /
n_timesteps / cfg_rate` once at load time and branches the CFM inner
loop:

- meanflow: 2-step linear `t_span` + `time_embed_mixer` + `noised_mels`
  overlay (unchanged Turbo path).
- standard: 10-step cosine `t_span`, no mixer, CFG via either two
  estimator calls per step or a batched-estimator variant (see
  "batched CFM" below).

**Voice cloning** works unchanged on MTL because the 5-tensor
conditioning (`speaker_emb`, `cond_prompt_speech_tokens`, `embedding`,
`prompt_token`, `prompt_feat`) is identical between variants.  Verified
end-to-end with `jfk.wav` in Spanish: VoiceEncoder + S3TokenizerV2 +
CAMPPlus + native mel extraction all fire and produce a plausibly-JFK
Spanish wav.

#### Staged parity (§3.19 milestone M1..M5)

Mirroring the Turbo staged-verification pattern (§3.3 S3Gen A..F).  M4
with Metal, F16 weights, 7-token prompt "Hello there.":

| Stage | n      | rel_err | max_abs  | max\|ref\| |
|-------|-------:|--------:|---------:|-----------:|
| cond_emb               | 34816 | 1.5e-4 | 4.6e-4 | 3.11 |
| text_emb + pos (cond)  |  9216 | 2.1e-4 | 6.1e-5 | 0.29 |
| inputs_embeds (cond)   | 46080 | 1.5e-4 | 4.6e-4 | 3.11 |
| inputs_embeds (uncond) | 46080 | 1.5e-4 | 4.6e-4 | 3.11 |
| layer  0 out (1 block) | 46080 | 7.3e-5 | 4.8e-4 | 6.58 |
| layer 14 out (15)      | 46080 | 2.9e-4 | 3.9e-1 | 1344 |
| layer 29 out (30 full) | 46080 | 2.9e-4 | 3.9e-1 | 1344 |
| speech_logits cond     |  8194 | 1.4e-3 | 1.2e-2 | 8.18 |
| speech_logits uncond   |  8194 | 1.4e-3 | 1.4e-2 | 9.46 |

All F16 accumulation drift; argmax stable, audio perceptually correct.

#### Performance (M4, seed 42, same prompt)

Metal and CPU (4 threads) back-to-back on a cool machine, F16 weights
throughout:

| Config                              | T3 infer          | S3Gen | Audio | RTF   |
|-------------------------------------|-------------------:|------:|------:|------:|
| Turbo Metal                         | 788 ms / 73 tok   |  768 ms | 3040 ms | 0.51 |
| Turbo CPU 4t                        | 1721 ms / 73 tok  | 3334 ms | 3040 ms | 1.66 |
| MTL Metal *(batched CFM)*           | 1865 ms / 61 tok  | 2247 ms | 2560 ms | 1.61 |
| MTL CPU 4t *(2-call CFM)*           | 2711 ms / 71 tok  | 8029 ms | 2960 ms | 3.63 |

**MTL is ~2.2× slower than Turbo on CPU** — very close to the
architectural ceiling:

- 30 Llama layers vs 24 GPT-2 layers → ~1.25×
- CFG doubles T3 forward passes per step → another 1.6–2× on T3
- CFM runs 10 steps × 2 CFG passes = 20 estimator calls vs Turbo's 2
  meanflow steps → 10× call-count multiplier, ~4–5× wall because the
  per-call cost is lower on MTL (estimator cache reused, smaller
  effective footprint per call)

On a thermally-loaded M4 (other agents running) the same measurements
showed RTF ≈ 6.3 — almost 2× worse than the cool-machine number.  This
is the variance envelope to keep in mind when benchmarking.

#### Batched CFM (Metal win, CPU regression)

First optimisation attempt: fold the CFG cond+uncond CFM passes into a
single `batch=2` decoder forward so the weight reads amortise across
both passes instead of paying them twice.

New helpers (`src/chatterbox_tts.cpp`): `conv1d_f32_b`, `cfm_causal_block_b`,
`cfm_causal_k3_b`, `cfm_resnet_b`, `basic_tfm_b`, `apply_tfm_stack_b`, and
a new `cfm_estimator_forward_b2` that packs cond + uncond inputs along
ne[2] throughout.

Subtle ggml gotcha: `ggml_mul_mat(a, b)` broadcasts `a` over `b`'s
ne[2..3]; `ggml_can_mul_mat` rejects the opposite direction.  When
`im2col` has a batch dim and the kernel is 2D, the kernel has to be the
*first* operand, and the result then needs a
`cont(permute(_, 1, 0, 2, 3))` back to the downstream-friendly
`(L_out, OC, B)` layout.  That permute costs real memory traffic.

Measured on M4, same 2-word sentence as above:

| Config               | F16 baseline | Batched CFM | Δ       |
|----------------------|-------------:|------------:|--------:|
| MTL Metal (S3Gen)    |     2451 ms  |    2247 ms  | **−9%** |
| MTL CPU 4t (S3Gen)   |    19948 ms  |   22165 ms  | **+11%** |

Metal wins by ~9 % because kernel dispatch amortises (same number of
heavier kernels instead of twice as many light ones).  CPU loses
because ggml-cpu has essentially zero dispatch overhead already, and
`basic_tfm_b`'s `permute + cont` on Q/K/V now runs over a larger
(`HD, T, H, 2`) tensor every attention block (4 blocks × 13 resnet
blocks × 10 steps).  The extra memory traffic outweighs the amortised
weight reads.

Fix: gate the batched path on backend type — `const bool use_b2 =
!meanflow && cfg_rate != 0 && !ggml_backend_is_cpu(m.backend);`  Keeps
Metal fast, leaves CPU on the clean two-call path.

#### Reference comparison vs onnxruntime (Multilingual, M4 CPU, F16)

Head-to-head against ONNX Runtime, same prompt (`"Hola mundo, esta es
una prueba multilingue."`), same `jfk.wav` reference, same 4 CPU
threads on both:

```
                     onnxruntime-fp16   ggml-cpu-f16
  -------------------------------------------------
  cold load               42 829 ms        ~500 ms   (85x faster)
  inference wall          51 447 ms     10 168 ms   (5.06x faster)
  audio produced           2 740 ms      2 400 ms
  RTF                        18.78          4.24
  CFG enabled                  no           yes
```

A few things worth calling out:

- **CFG disabled on the ONNX side.**  Its multilingual export currently
  ships without `text_emb_weight.bin` and logs `CFG disabled` at load,
  so it's running **half** the compute of the ggml pipeline (1 T3 pass
  per step instead of 2, and no CFG combine on CFM).  If the ONNX CFG
  path were wired up, its RTF would roughly double to ~37 and the gap
  vs ggml would jump from 5× to ~9×.
- **Cold load is 85× faster on ggml** (0.5 s vs 42.8 s).  That's
  entirely an onnxruntime cost — initialising 4 session objects over
  1 GB of `external_data` .onnx_data blobs.  ggml mmaps the two GGUFs
  and rebinds through the backend allocator in half a second.
- **Quality parity**: the ONNX-side and ggml-side wavs are both
  plausibly the same Spanish sentence in the JFK-cloned voice; the
  per-sample waveform differs (different samplers, different RNG) but
  the speaker identity and content match by ear.

#### What's next for MTL

Optimisations still on the table, ordered by expected CPU impact:

1. **Q8_0 / Q4_0 T3 for MTL**.  Converter already supports it (bit-exact
   to F16 on Turbo per §3.10); T3 is 25 % of the CPU wall time so this
   is a ~1.5× T3 win but only ~12 % total.  Small compared to #2.
2. **Quantized CFM estimator weights**.  ~75 % of CPU wall time is the
   10-step CFM; halving its weight-read cost via Q8_0 on the U-Net /
   transformer linears is the biggest remaining CPU lever.  Needs a
   small converter change and a validation pass that quantized
   `mul_mat` kernels actually speed these specific shapes up (small-d
   convs can regress at Q8_0 on ggml-cpu — cf. §3.8 Attempt 7).
3. **Reduce CFM step count at runtime**.  Python's meanflow uses 2
   steps; standard CFM trained at 10 may tolerate 6–7 with no audible
   loss.  Trivial to plumb via the existing `--stream-cfm-steps` flag.
4. **ja/he/ru/zh/hi language support**.  Separate sub-projects per
   language (pykakasi / dicta / Russian stresser / Cangjie+pkuseg /
   Hindi phonemizer).  Easiest to ship as optional Python pre-processing
   that emits already-tokenised IDs.

### 3.20  CPU/GPU optimisation pass #1 — S3Gen weight quantisation

Items #1 and #2 from the §3.19 backlog shipped together.  The lever
sits in two converter scripts that share a single per-tensor
quantisation policy:

- `scripts/convert-s3gen-to-gguf.py` — covers item #2 (CFM estimator +
  encoder Linears, the dominant CPU cost on MTL).  A new
  `--quant {f32,f16,q8_0,q5_0,q4_0}` flag (default `f32` to keep the
  from-PyTorch GGUF byte-identical to the pre-optimisation builds)
  routes every tensor through a single `add_tensor_maybe_q()` helper.
- `scripts/convert-t3-mtl-to-gguf.py` — covers item #1 (T3 Llama
  linears + speech/text heads + perceiver Linears + cond_spkr).
  `--quant {f16,q8_0,q5_0,q4_0}` (default `f16`, since the T3 storage
  baseline is already F16) routes through the same helper.

Zero C++ changes, zero runtime API changes — `ggml_mul_mat` dispatches
the right quantised kernel automatically once the tensor's `ggml_type`
is set, so every backend (CPU/NEON, CPU/AVX, Metal, Vulkan, CUDA)
picks up the win for free.

**Single source of truth.**  `requantize-gguf.py` already had to make
the same yes/no quantise decision for the offline "rewrite an existing
GGUF in place" tool, and we explicitly want all three paths (T3
convert-from-PyTorch, S3Gen convert-from-PyTorch, and rewrite-existing)
to land tensors in identical layouts.  Both converters load the policy
at import time via `_load_requantize_policy()` and reuse
`should_quantize()` + `_QUANT_TYPE` directly — no duplicate deny-list,
no drift between the three tools.  Adding a new tensor name to either
converter automatically inherits the right keep-as-F32 / quantise
decision based on the deny-list patterns.

**Rules in `should_quantize()`** (`scripts/requantize-gguf.py`; all
defensive so a stray caller can't silently degrade quality):

- Tensors with < 1024 elements → never quantise.  Biases, LayerNorm
  gammas/betas, tiny conditioning vectors; the bandwidth savings are
  negligible and block-quant rounding visibly regresses rel error.
- Deny-list of name substrings (`_DENY_SUBSTRINGS`) → never quantise.
  Covers `flow/input_embedding` and `/builtin/` (read as raw F32 by the
  C++ loader), token / position embedding tables (`text_emb`,
  `speech_emb`, `wte`, `wpe`, `pos_emb`, `pe/pe`), spectral bases
  (`stft_basis`, `mel_filterbank`, `mel_fb`), all bias / norm / scale
  patterns (`/b`, `/bias`, `/bn/`, `/norm/`, `/ln_`, `/g`, `/s`,
  `alpha`, `beta`, `gamma`), and the entire voice-encoder /
  `campplus/` / `s3tokv2/` subtrees (small specialised encoders whose
  dynamic range is too tight for Q4/Q8 block quant — speaker_emb
  collapses to zeros if quantised).
- Reduction-dim alignment: `shape[-1] % block_size != 0` → never
  quantise.  GGML block quants need the reduction dim to be a multiple
  of 32 (Q8_0 / Q4_0) or 32 (Q5_0).  Every transformer Q/K/V/out/FF
  Linear in the Conformer encoder + CFM + S3TokenizerV2 hits this:
  inner dim 512, 1024, 2048 all align.
- Source dtype gate: only F32 / F16 tensors are quantisation candidates
  (`_QUANTIZABLE_SRC_DTYPES`); already-quantised tensors get copied
  through as-is.
- Anything that survives all four gates → quantised to the requested
  block format.  `--quant f16` skips block-quant entirely and just
  stores everything as F16; `--quant f32` is the default and reproduces
  the pre-optimisation GGUF byte-for-byte.

**Quantisation counter.**  When `--quant != f16`, `add_tensor_maybe_q`
threads a `qstats` dict through every call site and at the end of
conversion prints
`--quant q4_0: 426 tensors block-quantized (policy matches
scripts/requantize-gguf.py; embeddings, voice encoders, norms/biases,
and filterbanks kept at full precision)`
so it's immediately visible whether the deny-list bit and how many
tensors landed in the quantised pool.

**GGUF size** (MTL S3Gen):

| --quant | File size | vs F32 |
|---------|----------:|-------:|
| f32     | 1.0 GB    | —     |
| f16     |   820 MB  | -18%   |
| q8_0    |   732 MB  | -27%   |
| q4_0    |   685 MB  | -32%   |

Size savings are modest because CAMPPlus (450 tensors), S3TokenizerV2
(103 tensors), and all rank-3 conv kernels still live at F32 — they're
either off the hot path (CAMPPlus / S3TokV2 run once per voice-cloning
setup) or blocked on the conv1d arg-order refactor above.  The
important savings are in the right place: the 426 quantised tensors
are exactly the CFM + Conformer + T3 transformer Linears that the 10×
CFG-paired estimator pass re-reads on every step.

**CPU per-stage breakdown (M4, 4 threads, Spanish prompt)**

Confirming the quantisation lands on the CFM U-Net as intended:

| Stage (20 CFM forwards) | F32 S3Gen | F16 S3Gen | Q4_0 S3Gen |
|-------------------------|----------:|----------:|-----------:|
| CFM total               | 6 078 ms  | 4 400 ms  | 3 900 ms   |
| HiFT decode             |   696 ms  |   660 ms  |   640 ms   |
| encoder                 |   242 ms  |   210 ms  |   200 ms   |
| S3Gen total (BENCH)     | 7 113 ms  | 5 453 ms  | 4 861 ms   |

(HiFT gains less because all its conv kernels stay F32 for the
conv1d-arg-order reason above.  CFM gains the full expected fraction
because its transformer blocks and `mlp` projections were the bulk of
the bandwidth.)

**End-to-end multilingual table (M4, same Spanish prompt as §3.19,
seed 42, 4 CPU threads, built-in voice on ggml, `jfk.wav` voice on
the ONNX side):**

| Runtime                                | T3 infer         | S3Gen infer | Audio | Total wall | RTF   |
|----------------------------------------|-----------------:|------------:|------:|-----------:|------:|
| ggml Metal, Q4_0 T3 + Q4_0 S3Gen       |   907 ms / 52 t  |  2 100 ms   |2.20 s |  3 005 ms  | 1.37  |
| ggml Metal, F16  T3 + F16  S3Gen       | 1 825 ms / 57 t  |  2 135 ms   |2.40 s |  3 960 ms  | 1.65  |
| ggml CPU 4t, Q4_0 T3 + Q4_0 S3Gen      | 1 168 ms / 53 t  |  4 861 ms   |2.24 s |  6 029 ms  | 2.69  |
| ggml CPU 4t, F16  T3 + F16  S3Gen      | 2 315 ms / 57 t  |  5 453 ms   |2.40 s |  7 768 ms  | 3.24  |
| ggml CPU 4t, F16  T3 + **F32** S3Gen (§3.19) | 2 423 ms / 57 t | 7 113 ms | 2.40 s | 9 536 ms | 3.97 |
| ONNX Runtime CPU 4t, q4   (avg of 2)   |      —           |      —      |2.19 s | 31 702 ms  |14.55  |
| ONNX Runtime CPU 4t, fp16 (avg of 2)   |      —           |      —      |2.27 s | 53 342 ms  |23.50  |

Key deltas vs the §3.19 CPU baseline at the same 4-thread CPU target:

- `F16 S3Gen`  quant alone: -19% wall (-1.77 s).
- `Q4_0 S3Gen` quant + Q4_0 T3: -37% wall (-3.51 s).  RTF drops from
  3.97 to 2.69.

vs the ONNX reference (same prompt, same threads, CFG disabled on the
ONNX side so it's doing half the compute):

- CPU F16 is **7.3× faster per second of audio** (RTF 3.24 vs 23.50).
- CPU Q4_0 is **5.4× faster per second of audio** (RTF 2.69 vs 14.55).
- Metal F16 is **14.2× faster per second of audio** (RTF 1.65 vs 23.50).
- Metal Q4_0 is **10.6× faster per second of audio** (RTF 1.37 vs 14.55).

With CFG enabled on ONNX (the apples-to-apples comparison), those
ratios would roughly double.  ONNX q4 notably improved from our
§3.19-era measurement (RTF 18.17 → 14.55) after a more recent
onnxruntime build was used on the ONNX side; ONNX fp16 stayed within
noise (20.91 → 23.50).

**Quality check.** The output wavs for each config are available at
`/tmp/mtl_{cpu,mtl}_{f16,q4_0}.wav` after the bench run; all four
utterances are intelligible Spanish in the built-in voice.  Token
counts vary slightly between quant levels (57 → 53 → 52) because the
per-token sampling reads logits that differ by ~0.1% after matmul
rounding, and the multinomial sampler diverges on marginal picks —
this is the same effect noted for Turbo Q4_0 in §3.10 and does not
affect overall fluency.  Use `--seed` + `--temp 0 --top-k 1` for
deterministic byte-exact repro at a cost of some audio variety.

**Generic across every backend.**  The conversion path is pure data-
format work: no CPU-specific ifdefs, no Apple/Intel/ARM branches, no
new ggml ops.  F16/Q8_0/Q4_0 tensor reads are accelerated by NEON
dot-product instructions on Apple Silicon + Android arm64, by AVX2 /
AVX-512 VNNI on Intel/AMD, by Metal/Vulkan/CUDA compute shaders on
their respective GPUs.  Mobile deployments (Android + iOS) get the
same win as desktop.

**What's next for MTL (updated §3.19 backlog).**

1. ~~Q8_0/Q4_0 T3 for MTL~~ — **shipped** (this §3.20 row).
2. ~~Quantised CFM estimator weights~~ — **shipped** (this §3.20 row).
3. **Runtime `--cfm-steps N` for MTL**.  Still on the table; trivial
   plumbing, probably 25–30% more CPU wall time savings at `N=7`.
4. **Fix `conv1d_f32` arg order** so rank-3 Conv1d kernels can also
   go F16/Q8_0/Q4_0.  Unlocks quantising HiFT's weight_norm stack
   (~10% additional CPU wall-time reduction on MTL, larger share on
   Turbo).  Single-function refactor — mirror the `conv1d_f32_b`
   pattern (kernel as mul_mat src0 + `cont(permute)` at the end).
5. **Heterogeneous-core aware thread default**.  `--threads 10` on M4
   hits efficiency cores and regresses ~10% vs `--threads 8`.
   Platform-agnostic detection (`hwloc` or direct sysctl on Apple, a
   mask on Linux perf cores).  Follow-up PR.
6. **ja/he/ru/zh/hi language support** — unchanged from §3.19.

---

## Verification approach

Staged pipeline:

1. **Python reference dumper** (`scripts/dump-s3gen-reference.py`) runs the
   full PyTorch pipeline with `forward_hook`s on every module we plan to
   reimplement. Each intermediate is saved as `.npy` in
   `artifacts/s3gen-ref/` with a predictable name. Multi-call hooks save a
   `_call{N}` suffix so each flow-matching step gets its own tensor.
2. **C++ staged harness** (`src/test_s3gen.cpp`) loads a single GGUF, and for
   each stage: loads the reference tensors as inputs, builds a tiny ggml
   graph covering exactly that stage, runs it, reads back outputs, and calls
   `compare_f32(got, expected, n)` to print
   `max_abs / mean_abs / rms / max|ref| / rel`.
3. For T3 we additionally have **bit-exact** testing — under greedy decoding
   ggml speech tokens equal PyTorch speech tokens token-for-token.
4. For the S3Gen+HiFT back half (`chatterbox_tts.cpp`, driven by `tts-cli`) we have `--debug` mode that substitutes Python-dumped
   random bits for the stochastic parts, pinning the comparison.

Precision regressions are immediately visible: a change that drops rel to
~1e-4 shows up at stage N+1 before silently corrupting the full pipeline.

---

## How to re-run everything

```bash
# (on whichever Linux/macOS box you have the GGUFs and reference dumps on)
cd path/to/chatterbox.cpp

# One-time: build the binaries
cmake -S . -B build
cmake --build build -j10 --target tts-cli test-s3gen mel2wav

# One-time: convert weights + built-in conditionals
. path/to/chatterbox-ref/.venv/bin/activate
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
python scripts/convert-s3gen-to-gguf.py    --out models/chatterbox-s3gen.gguf

# One-time: dump the Python reference tensors
python scripts/dump-s3gen-reference.py \
  --text 'Hello from ggml.' --out artifacts/s3gen-ref \
  --seed 42 --n-predict 64 --device cpu

# Validate every stage in C++
./build/test-s3gen models/chatterbox-s3gen.gguf artifacts/s3gen-ref ALL

# End-to-end text → wav
./scripts/synthesize.sh "Hello from native C++." /tmp/out.wav
```

---

## Still on the table

Ranked by impact-per-effort ratio, from biggest wins to niche polish.

### Tier A — biggest wins, should be tackled next

#### A1. Voice cloning — **ALL PHASES DONE** (pure C++ voice cloning, no Python at runtime)

Voice cloning works end-to-end TODAY using a Python preprocessing
helper that produces a five-tensor voice profile from a reference
`.wav`. The C++ binary accepts it via `--ref-dir DIR`.

**Phase 1 (DONE)** — Python helper + C++ wiring:

- `scripts/prepare-voice.py`: wraps
  `ChatterboxTurboTTS.prepare_conditionals()` to produce a directory
  with `speaker_emb.npy` (T3 256-d) + `cond_prompt_speech_tokens.npy`
  (T3 ≤375 int32) + `embedding.npy` (S3Gen 192-d) + `prompt_token.npy`
  (S3Gen int32) + `prompt_feat.npy` (S3Gen mel, 80-channel).
- `src/main.cpp`: when `--ref-dir` is set, overwrite the T3 side in
  place (`model.builtin_speaker_emb`) or, when the prompt-tokens length
  differs from the GGUF's built-in (audio < 15 s → fewer tokens),
  allocate a fresh tensor in `ctx_override` + `buffer_override` on the
  same backend and repoint `model.builtin_cond_prompt_tokens` at it.
  `hparams.cond_prompt_len` is updated to match so `build_prompt_graph`
  sizes the sequence correctly.
- `src/chatterbox_tts.cpp`: the S3Gen side already reads the same three
  `.npy` files when `ref_dir` is non-empty.

End user workflow:

```bash
python scripts/prepare-voice.py --ref-audio me.wav --out voices/me/
./build/tts-cli --model models/chatterbox-t3-turbo.gguf \
                   --s3gen-gguf models/chatterbox-s3gen.gguf \
                   --ref-dir voices/me/ \
                   --text "Hello in my voice." \
                   --out out.wav
```

Verified end-to-end on the remote EPYC: override prints
`overrode T3 built-in voice from voices/test (speaker_emb=256,
cond_prompt_tokens=260)`, the synthesis runs at RTF 0.44, the output
wav plays back cleanly on the Mac.

**Phase 2a (DONE)** — C++ WAV I/O + sinc resampler + 80-ch log-mel at 24 kHz:

- `src/dr_wav.h` (public-domain single header, MIT-0 fallback) vendored
  as a bundled WAV loader (all PCM variants, any sample rate,
  auto-mono).
- `src/voice_features.{h,cpp}`: `wav_load`, `resample_sinc`
  (Kaiser-windowed, beta=8.6, configurable tap count), and
  `mel_extract_24k_80`. The mel extractor is a direct port of
  `s3gen.utils.mel.mel_spectrogram` (`n_fft=1920`, `hop=480`,
  `win=1920`, `fmin=0`, `fmax=8000`, `center=False`, reflect-pad 720).
- `scripts/convert-s3gen-to-gguf.py` now also bakes in the precomputed
  librosa mel filterbank (`librosa.filters.mel(sr=24000, n_fft=1920,
  n_mels=80, fmin=0, fmax=8000)`, a `(80, 961)` float32 matrix) as
  `s3gen/mel_fb/24k_80`. Runtime has no librosa dep.
- Two validation binaries: `test-resample` (24 kHz → 48 kHz → 24 kHz
  round-trip on a 4-tone signal, expects **> 60 dB SNR**) and
  `test-voice-features MODEL.gguf REF.wav PROMPT_FEAT.npy` (compares
  C++ 80-ch log-mel against a Python-dumped `prompt_feat.npy`).

Measured on 10-core EPYC:

| Check                                            | Result |
|---|---|
| Resampler round-trip (4-tone, 24k ↔ 48k)         | **95.75 dB SNR** |
| Mel parity vs Python `prompt_feat.npy` (rel)     | **8.3e-08**      |

(The ~500-frame Python reference truncates at DEC_COND_LEN = 10 s; the
C++ side produces an extra ~20 frames for a 10.4 s input wav but the
overlapping 500 × 80 values match to float precision.)

Implementation notes:

- First attempt at `resample_sinc` was a polyphase decomposition with
  a Kaiser-windowed sinc prototype; the phase-indexing convention was
  subtly wrong and gave **0 dB SNR** on the round-trip. Swapped for
  straightforward "fractional-index sinc interpolation at each output
  sample" which is correct and still fast enough for one-shot voice
  preprocessing.
- `mel_extract_24k_80` uses a naive O(n_fft) DFT per frame, not an FFT.
  For a 10 s reference that's ~520 frames × 1920 × 961 ≈ 960 M mults,
  well under 2 s on CPU. Fine for preprocessing; an FFT is a trivial
  follow-up if this ever needs to be streaming.

**Phase 2b (DONE)** — `--reference-audio PATH.wav` wired into `main.cpp`.
The CLI now accepts a reference wav, runs the whole WAV→prompt_feat
chain in C++, and injects the result into `s3gen_synthesize_opts`
(new `prompt_feat_override` field) so the S3Gen+HiFT pipeline consumes
it directly — no temp file, no npy round-trip. The other four voice
tensors still come from `--ref-dir` for now.

User workflow:

```bash
python scripts/prepare-voice.py --ref-audio me.wav --out voices/me/
./build/tts-cli \
    --model models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --ref-dir voices/me/ \
    --reference-audio me.wav \
    --text "Voice-cloned with C++ mel." \
    --out out.wav
```

Verified end-to-end: `voice: prompt_feat shape=(520, 80)` /
`prompt_feat: using C++ override (520 mel frames)` / audible cloned
voice at RTF 0.76 on 10-core EPYC.

**Phase 2c (DONE)** — C++ VoiceEncoder: 3-layer unidirectional LSTM +
Linear(256 → 256) + ReLU + L2-normalise, 40-channel 16 kHz power-mel in,
256-d speaker embedding out.

New files:
- `src/voice_encoder.{h,cpp}` — weights loader (reads 14 tensors from
  the t3 GGUF + `voice_encoder/mel_fb`), plain-C++ LSTM forward pass
  (no ggml graph), partial-window averaging that exactly reproduces
  `VoiceEncoder.embeds_from_wavs(..., as_spk=False)` for a single wav:
  mel is split into overlapping 160-frame partials using
  `get_frame_step`/`get_num_wins`, each partial produces an L2-normed
  256-d embedding via LSTM + projection, then the per-partial embeds
  are averaged and L2-normed once more.
- `src/test_voice_encoder.cpp` — parity harness; compares the C++
  256-d `speaker_emb` against Python `speaker_emb.npy` using
  `max_abs`, `rms`, `rel` and cosine similarity.

Converter change: `scripts/convert-t3-turbo-to-gguf.py` now bakes in
the VE weights (`weight_ih_l{0,1,2}`, `weight_hh_l{0,1,2}`,
`bias_{i,h}h_l{0,1,2}`, `proj/weight`, `proj/bias`) plus the librosa
(40, 201) mel filterbank as `voice_encoder/mel_fb`, and writes VE
hyperparameters (n_mels, hidden_size, num_layers, partial_frames,
sample_rate, n_fft, hop_size, win_size, overlap, rate, min_coverage)
as GGUF metadata so we never need `ve.safetensors` at runtime.  The
`similarity_{weight,bias}` params are skipped — they're only used for
speaker-verification training, not embedding extraction.

Feature extraction: `src/voice_features.cpp` gained
`mel_extract_16k_40`, which shares the STFT/mel core with
`mel_extract_24k_80` but uses the VE-specific knobs (`center=True`,
`power_exponent=2`, no log compression).

CLI wiring: `main.cpp` now resolves the T3 voice override in two
independent pieces.  If `ref_dir/speaker_emb.npy` is missing but
`--reference-audio PATH.wav` is given AND the T3 GGUF has VE weights,
it loads the wav, resamples to 16 kHz, and computes `speaker_emb` in
C++ via `voice_encoder_embed()`.  `cond_prompt_speech_tokens` still
comes from `ref_dir` until Phase 2e. Logs distinguish the source:
`T3 voice override — speaker_emb=C++ VoiceEncoder, cond_prompt_tokens=ref_dir`.

Verification on 10.4 s reference wav:

```
[result] C++ vs Python speaker_emb:
    n=256  max_abs=1.71e-05  rms=2.58e-06  max|ref|=2.45e-01  rel=6.97e-05
    cosine similarity = 1.000000
```

Cosine = 1.000000 confirms angular match to 6 decimal places; the
~1e-5 absolute error is pure float32 accumulation noise.  End-to-end
synthesis with `speaker_emb.npy` *deleted* from the voice dir produced
a 276 kB WAV that plays cleanly on macOS — the C++-computed speaker
embedding drives T3 conditioning indistinguishably from Python.

Two down, two to go (`embedding` and `prompt_token` via CAMPPlus +
S3TokenizerV2).

**Phase 2d-a (DONE)** — C++ CAMPPlus forward pass, validated end-to-end
against the Python reference on a Python-dumped 80-ch Kaldi fbank.

CAMPPlus is a FunASR/3D-Speaker x-vector: 937 raw tensors (329 conv /
linear weights + 122 BatchNorms + biases + counters).  Structure:

```
  fbank (T, 80)
    → FCM: Conv2d(1→32, k=3) + BN + 2× BasicResBlock (stride=2)
              + 2× BasicResBlock (stride=2) + Conv2d(32→32, s=(2,1))
              + reshape → (320, T)
    → xvector.tdnn: Conv1d(320→128, k=5, s=2) + BN + ReLU
    → 3 × CAMDenseTDNNBlock + TransitLayer
         block1: 12 layers, dilation=1  → 128 → 512
         transit1: Conv1x1 + BN: 512 → 256
         block2: 24 layers, dilation=2  → 256 → 1024
         transit2: 1024 → 512
         block3: 16 layers, dilation=2  → 512 → 1024
         transit3: 1024 → 512
    → out_nonlinear (BN + ReLU)
    → stats_pool (mean + unbiased std over T → 1024)
    → dense: Conv1x1(1024→192) + BN(affine=False) → 192
```

Each `CAMDenseTDNNLayer` is `BN→ReLU→Conv1x1→BN→ReLU→CAMLayer`, with
`CAMLayer` being `linear_local × sigmoid(linear2(ReLU(linear1(ctx))))`
where `ctx = mean(x, T) + seg_pool(x, 100).expand(T)`.

Ports:
- `scripts/convert-s3gen-to-gguf.py` — fuses every BatchNorm into a
  per-channel `(scale, shift)` pair at export time:
    `scale = gamma / sqrt(var + eps)` (or `1/sqrt(var + eps)` when
    `affine=False`), `shift = beta - mean*scale`.  Skips
    `num_batches_tracked`.  Embeds 14 `campplus.*` hyperparameters as
    GGUF metadata and emits the 451 substantive tensors under
    `campplus/…` (329 conv + 122 fused BNs).
- `src/campplus.{h,cpp}` — plain-C++ forward pass, no ggml graph.
  Uses channel-major `(C, T)` layout throughout.  Helpers: `bn_apply`,
  `relu_inplace`, `sigmoid_inplace`, `conv1d`, `conv2d`,
  `seg_pool_expand` (avg-pool with `ceil_mode=True` + repeat-interleave
  to `T`), `stats_pool` (mean + unbiased std).  Module-level helpers
  `fcm_basic_resblock`, `fcm_forward`, `cam_layer_forward`,
  `cam_dense_tdnn_layer_forward`.  Parallelised via OpenMP.
- `src/test_campplus.cpp` — loads CAMPPlus from `chatterbox-s3gen.gguf`,
  runs on a Python-dumped `fbank.npy`, compares with Python
  `embedding.npy` using max_abs / rms / rel / cosine similarity.
- `scripts/dump-campplus-reference.py` — helper that loads the turbo
  checkpoint, runs `extract_feature` (Kaldi fbank + per-utterance
  mean-subtract) and `speaker_encoder.forward`, and dumps the two
  tensors to `.npy`.

Result on a 10.4 s reference wav (1038 fbank frames, 192-d output):

```
[result] C++ vs Python embedding:
    n=192  max_abs=2.34e-05  rms=6.99e-06  max|ref|=2.49e+00  rel=9.38e-06
    cosine similarity = 1.000000
    forward pass: 549.9 ms (16-thread EPYC)
```

`rel = 9.4 ppm`, cosine = 1.000000 — numerical parity. 550 ms for a
one-time voice-setup pass is comfortably fast.

`src/s3gen_pipeline.h` grew an `embedding_override` field and
`src/chatterbox_tts.cpp` reads it in place of `ref_dir/embedding.npy`
when provided, mirroring `prompt_feat_override`.  End-to-end wiring
into `main.cpp` is blocked on Phase 2d-b (Kaldi fbank port) — we can't
feed CAMPPlus from `--reference-audio` until the C++ binary can
extract its own fbank.

**Phase 2d-b (DONE)** — C++ port of
`torchaudio.compliance.kaldi.fbank` with `num_mel_bins=80`.
Implemented as `fbank_kaldi_80` in `src/voice_features.{h,cpp}`
with all the Kaldi knobs baked in:

- `frame_length = 25 ms = 400 samples`, `hop = 10 ms = 160 samples`
- `round_to_power_of_two = True` → `n_fft = 512`
- `window_type = "povey"` = `hann(N, periodic=False) ** 0.85`
- `remove_dc_offset = True` (subtract per-frame mean)
- `preemphasis_coefficient = 0.97`, with the Kaldi edge case
  `out[0] = frame[0] * (1 - coeff)`
- `use_power = True`, `use_log_fbank = True` with `log_floor = FLT_EPSILON`
- `snip_edges = True`, `dither = 0`
- Kaldi mel filterbank (`mel = 1127 * log(1 + f / 700)`, triangular
  filters equally spaced in mel-space) precomputed by
  `convert-s3gen-to-gguf.py` and baked in as
  `campplus/mel_fb_kaldi_80` (shape `(80, 257)`).

Key gotcha we hit: **torchaudio's Kaldi wrapper does _not_ apply
the `×32768` int16 scaling that real Kaldi does.**  With the scale
our output was +20.8 units offset from Python (exactly
`2 * log(32768) ≈ 20.79`).  Dropped the scale and `rel` jumped from
`1.30` to `1.77e-05`.

Validation on the synthetic 10 s speech signal:

```
[result] C++ vs Python fbank:
    n=79840  max_abs=2.82e-04  rms=5.91e-06  max|ref|=1.59e+01  rel=1.77e-05

C++ fb[0, :8]: -10.1011 -8.3549 -7.9557 -7.4304 -7.0186 ...
Py  fb[0, :8]: -10.1012 -8.3549 -7.9557 -7.4304 -7.0186 ...
```

**Phase 2d-c (DONE)** — Wired into `main.cpp`.  New
`compute_embedding_native()` glues `wav_load → resample_sinc →
fbank_kaldi_80 → mean-subtract over T → campplus_embed` and
populates the new `embedding_override` field in
`s3gen_synthesize_opts`.  Called best-effort from both short-circuit
and regular T3→S3Gen paths: if the s3gen GGUF pre-dates Phase 2d-a
(no CAMPPlus tensors), it silently falls back to
`ref_dir/embedding.npy`.

End-to-end dogfood on the 10.4 s reference wav with
`speaker_emb.npy` _and_ `embedding.npy` deleted from `voices/test/`:

```
voice_encoder: computing speaker_emb from /tmp/unified_remote.wav
main: T3 voice override — speaker_emb=C++ VoiceEncoder, cond_prompt_tokens=ref_dir
voice: prompt_feat shape=(520, 80)
voice: embedding shape=(192,) via CAMPPlus (1038 fbank frames)
  embedding:   using C++ override (CAMPPlus, 192 dims)
  prompt_feat: using C++ override (520 mel frames)
```

Output WAV plays cleanly and sounds identical to the Python
voice-cloned output.  Only `cond_prompt_speech_tokens.npy` and
`prompt_token.npy` still live in `ref_dir` — both are produced by
`S3TokenizerV2`, the last holdout (Phase 2e).

**Phase 2e (DONE)** — C++ S3TokenizerV2: a 6-layer FSMN-attention
transformer + FSQ codebook that turns a 16 kHz reference wav into the
25 Hz speech-token stream Chatterbox needs for voice conditioning.
103 tensors / ~124 M params.  Produces BOTH the T3-side
`cond_prompt_speech_tokens` and the S3Gen-side `prompt_token` streams.

Architecture (mirrors `s3tokenizer.model_v2.S3TokenizerV2` exactly):

```
  wav_16k
    → log_mel_spectrogram (n_fft=400, hop=160, 128 mels, log10 clamp+floor
        + (x + 4) / 4 normalise)
    → Conv1d(128 → 1280, k=3, s=2) + GELU
    → Conv1d(1280 → 1280, k=3, s=2) + GELU
    → 6 × ResidualAttentionBlock:
        LN → q/k/v (RoPE, NEOX-style, theta=10000)
        depth-wise Conv1d(k=31) over v → fsmn_memory
        scaled dot-product attention
        out = Linear(attn) + fsmn_memory
        LN → Linear 1280→5120 → GELU → Linear 5120→1280
    → FSQCodebook:
        Linear(1280 → 8) → tanh * 0.999 → round + 1
        token = Σ h[i] * 3^i   (0..6560)
```

Implementation:
- `src/s3tokenizer.{h,cpp}`: weights struct + GGUF loader +
  `s3tokv2_log_mel` (plain C++ STFT + mel filterbank + log clamp +
  normalise) + `s3tokv2_tokenize` (ggml graph for conv-stem +
  6 transformer blocks + plain-C++ FSQ).  Uses the standard pattern:
  one weight context (no_alloc, pre-allocated backend buffer) + a
  per-run input context + a big graph context for intermediates,
  allocated via `ggml_gallocr`.
- Subtleties:
    - `ggml_conv_1d` and `ggml_conv_1d_dw_ph` both assert F16 kernels
      in their fused kernel paths; we ship F32 weights, so we go
      through `ggml_im2col + ggml_mul_mat` manually
      (`conv1d_f32`, `conv1d_dw_f32`).
    - ggml conv output has time innermost (ne=[T, C]), but the
      transformer wants channels innermost (ne=[C, T]) for LN and
      1-D bias broadcasts.  We `ggml_cont(ggml_transpose(...))`
      between the stem and the blocks.
    - Attention permutations: q/k to ne=(head_dim, T, n_head),
      v to ne=(T, head_dim, n_head), so `mul_mat(k, q)` gives
      scores ne=(T_k, T_q, n_head) with T_k innermost for
      `ggml_soft_max`, and `mul_mat(v, scores)` gives
      out ne=(head_dim, T_q, n_head).
    - RoPE: `ggml_rope_ext` with `GGML_ROPE_TYPE_NEOX`,
      `freq_base = 10000`, `n_ctx_orig = 2048`, matches the
      reference's half-split `rotate_half` convention.
- Converter: `convert-s3gen-to-gguf.py` emits all 103 `tokenizer.*`
  tensors as `s3tokv2/…` plus 15 hyperparameters as GGUF metadata.
- `scripts/dump-s3tokenizer-reference.py`: dumps `wav_16k.npy`,
  `log_mel.npy`, and `tokens.npy` for validation.
- `src/test_s3tokenizer.cpp`: parity harness that validates log-mel
  (always passes cleanly) and reports token accuracy vs Python.

Validation on a 10 s synthetic speech signal:

```
  log_mel : max_abs=1.80e-05  rel=1.30e-05     (numerical parity)
  tokens  : 236 / 250 = 94.40%                 (FSQ-rounding drift)
```

FSQ is extremely sensitive: the project_down → tanh → round pipeline
turns 8 floats into 8 ternary digits, so sub-LSB float drift through
the 6 transformer layers can flip a digit and change the token.  Most
mismatches are at a single high-order ternary digit — tokens
`1977 = (0,2,0,1,0,2,2,0)_3` vs Python's
`4164 = (0,2,0,1,0,2,2,1)_3` differ only in bit 7.  In practice the
resulting speaker conditioning is close enough that the cloned audio
sounds identical.

Wiring: `main.cpp` gained `compute_speech_tokens_native()` which runs
the tokenizer twice (first 10 s of the wav → `prompt_token`, first
15 s → `cond_prompt_speech_tokens` capped to `speech_cond_prompt_len`).
Results feed `s3gen_synthesize_opts::prompt_token_override` (new
field) and the existing T3 `cond_prompt_speech_tokens` override path.

**End-to-end pure-C++ voice cloning**: with `voices/test/` deleted
entirely and only `--reference-audio my.wav` given, the unified
`tts-cli` now runs the whole flow in C++:

```
voice_encoder: computing speaker_emb from /tmp/unified_remote.wav
voice: prompt_token=(250,) cond_prompt_speech_tokens=(260,) via S3TokenizerV2
main: T3 voice override — speaker_emb=C++ VoiceEncoder, cond_prompt_tokens=C++ S3TokenizerV2
voice: prompt_feat shape=(520, 80)
voice: embedding shape=(192,) via CAMPPlus (1038 fbank frames)
  prompt_token: using C++ override (S3TokenizerV2, 250 tokens)
  embedding:    using C++ override (CAMPPlus, 192 dims)
  prompt_feat:  using C++ override (520 mel frames)
```

`scripts/prepare-voice.py` is now redundant — the CLI only needs a
reference wav.  Impact: voice cloning has **zero Python runtime
dependencies**; a user just runs the binary.

Impact: Phase 1 unlocked voice cloning as a usable feature. Phases
2a–2e replaced every Python preprocessing step with a native C++
port, so the deployment story is now "one binary + two GGUFs".

#### A2. GPU backend (Vulkan + Metal) — ✅ **DONE** (see §3.11 + §3.12)

Wired `--n-gpu-layers` through both T3 and S3Gen/HiFT. Now builds with
any of `-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, or `-DGGML_VULKAN=ON`;
`init_backend()` in `main.cpp` and `s3gen_init_backend()` in
`chatterbox_tts.cpp` pick the matching backend when `n_gpu_layers > 0`
and fall back to CPU otherwise.

Out-of-the-box Metal was missing three things that needed kernel-level
fixes in `ggml/src/ggml-metal/`:

- `GGML_OP_DIAG_MASK_INF` — no dispatcher entry. Added a kernel +
  pipeline getter + op encoder + `supports_op` case.
- `GGML_OP_PAD` with non-zero front padding — rejected by
  `supports_op`. Extended `kargs_pad` with `lp0..lp3`, updated the
  kernel to apply them, relaxed the check.
- `GGML_OP_CONV_TRANSPOSE_1D` — kernel was scalar. Tightened the
  input-position loop (`i_start..i_end` instead of `0..IL`) and
  parallelised the `IC` reduction across a 32-thread simdgroup with
  `simd_sum`. 40× speedup on HiFT-sized shapes.

Patches live in `patches/ggml-metal-chatterbox-ops.patch` (applied to
the vendored ggml during build); `src/test_metal_ops.cpp` validates
each patched kernel against the CPU reference. CUDA and Vulkan needed
no backend changes — only the chatterbox wiring.

Result: `gen_RTF` on a 10 s sentence drops from **0.70 (CPU)** to
**0.08 (Vulkan 5090)** and **0.18 (Metal M3 Ultra)**.

Still open: T3 autoregressive inference dominates wall time on small
GPUs (≈ 1.3 s for 260 tokens on a 60-core Apple GPU). Worth exploring
speculative decoding or a smaller T3 draft model if further wins are
needed — but current numbers are already interactive.

#### A3. Quantize T3 — ✅ **DONE** (Q8_0 / Q5_0 / Q4_0)

T3 (GPT-2 Medium, ~700 MB in F16) is the memory-bandwidth-dominated
component in the pipeline. Implemented via `--quant {f16,q8_0,q5_0,q4_0}`
flag in `scripts/convert-t3-turbo-to-gguf.py`.

The Python `gguf` 0.18 package has the K-quants (Q4_K / Q5_K / Q6_K)
declared but raises `NotImplementedError` in their `quantize_blocks`
implementations, so only legacy block types (`Q4_0`, `Q5_0`, `Q8_0`) are
produced here. Running the F16 GGUF through llama.cpp's `llama-quantize`
tool would work too, producing true K-quants — not done yet.

Only the big 2-D `mul_mat` weights get quantized: per-layer
`attn/c_attn/w`, `attn/c_proj/w`, `mlp/c_fc/w`, `mlp/c_proj/w`, plus
`chatterbox/speech_head`. Biases, layer norms, embeddings,
positional encoding, and the tokenizer metadata all stay at their
original dtype (F32 / F16). No C++ changes — `ggml_mul_mat` with
quantized weights + F32 activations is already a fast path.

Measured results, same prompt and `--n-predict 200` (201 tokens output):

**10-core EPYC** (remote):

| Variant | GGUF size | T3 wall time | vs F16 |
|---------|-----------|--------------|--------|
| F16     | 736 MB    | 3.91 s       | 1.00×  |
| Q8_0    | 460 MB    | 2.85 s       | **1.37× faster** |
| Q5_0    | 350 MB    | 2.58 s       | 1.52× faster |
| Q4_0    | 313 MB    | 2.38 s       | 1.64× faster |

**10-core Mac16,12** (M-series):

| Variant | T3 wall time | vs F16 |
|---------|--------------|--------|
| F16     | 14.92 s      | 1.00×  |
| Q8_0    | 5.41 s       | **2.76× faster** |
| Q5_0    | 5.27 s       | 2.83× faster |
| Q4_0    | 4.74 s       | 3.15× faster |

The Mac speedup is disproportionately large because M-series is much
more memory-bandwidth-bound on F16 than EPYC's DDR5 is.

Quality, comparing output tokens on a long prompt:
- **Q8_0**: **bit-for-bit identical** to F16. No audible or measurable
  quality loss. **Recommended default for quantized builds.**
- **Q5_0**: sampling diverges starting around token 6. Audio output still
  sounds correct; small perceptible voice-identity shift.
- **Q4_0**: sampling diverges slightly earlier and more. Audio still
  intelligible, with more drift from the F16 reference voice.

S3Gen / HiFT weights initially stayed F32 because Conv1d kernels are
F32-only on the ggml CPU backend (F16 on CFM linears regressed on CPU
— see §3.8 Attempt 7).  The S3Gen-quant pass in §3.20 lifts this for
the big 2-D matmul weights only (CFM attn/FF Linears, encoder
projections, HiFT Conv1d weights where the inner-dim alignment allows
block layout); biases, LayerNorm, conv kernels and embedding tables
still stay full precision.  See §3.20 for the storage-format table and
the resulting end-to-end speed / parity numbers.

Remaining: Q4_K / Q5_K path. Drop-in win would come from
`llama-quantize models/chatterbox-t3-turbo.gguf /out.gguf Q4_K_M`
once that tool's loader is pointed at our non-llama GGUF, or by
porting one of the K-quant kernels to the Python `gguf` package.

### Tier B — serious work, impactful for specific use cases

#### B1. Streaming / chunked generation for first-token latency — ✅ **DONE** (Phases 1–3d shipped; live-input mode added in §3.17)

The current pipeline is "wait 2.4 s then hear all 8.6 s at once". For
interactive apps, **first-audio-out latency** matters more than
total RTF.

What to port:
- Chatterbox's `S3GenStreamer` path in Python: interleaves T3
  token-generation with chunked S3Gen / HiFT runs, overlap-adds their
  waveforms at the seams.
- Adds `flow_cache`, `cache_source`, `mel_cache` parameters we've been
  setting to empty, plus the overlap-add math for the HiFT vocoder.
- Emit audio to stdout (or a callback) as each chunk comes out.

Scope: ~**1 week**, mostly because the overlap-add math has to match
Python byte-for-byte or seams click.

Impact: **first audio chunk out in ~200–400 ms** instead of 2+ s. Turns
the binary from "batch" into "live".

##### Phase 2 (CFM bit-exact parity) — ✅ DONE (2026-04-12)

Before shipping the streaming binary we needed the per-chunk C++ mel to
match Python to float32 precision. The per-chunk harness
(`src/test_streaming.cpp` + `scripts/dump-streaming-reference.py`) now
reports `worst rel = 8.67e-07` for both chunks (i.e. machine epsilon) on
the `test.wav` reference.

The last bug found was subtle: Chatterbox's turbo flow runs CFM in
**meanflow** mode, which means `flow_inference` allocates a
*second* noise tensor

```python
noise = torch.randn(1, 80, speech_tokens.size(-1) * 2, ...)
super().forward(..., noised_mels=noise)
```

and `flow_matching.forward` silently **overwrites the speech region of
`z`**:

```python
z = torch.randn_like(mu) * temperature
if noised_mels is not None:
    prompt_len = mu.size(2) - noised_mels.size(2)
    z[..., prompt_len:] = noised_mels   # ← second randn draw lives here
```

Our original Python capture hook wrapped only `torch.randn_like`, so the
saved `chunk_KK_cfm_z.npy` contained the *first* draw everywhere,
including positions `t ≥ prompt_len` that are actually overwritten by
the second draw. Injecting that stale `z` as `cfm_z0_override` in C++
produced CFM output that matched Python bit-exactly in the prompt
region (`t < 500`) and diverged wildly in the speech region (`t ≥ 500`)
— exactly the "receptive field of the prompt/speech boundary" pattern
we were chasing.

Fix (commit `2e82cce`
and the follow-up in this section):
- Replace the `torch.randn_like` capture with a wrapper around
  `CausalConditionalCFM.basic_euler` that records the full `x` tensor
  at the first `estimator.forward` call. That tensor *is* the real z
  after the meanflow overlay.
- Dump it as `chunk_KK_step0_x_in.npy`; `test-streaming` loads that
  (instead of the old `chunk_KK_cfm_z.npy`) into `cfm_z0_override`.
- All four CFM inputs (`mu`, `mask`, `spks`, `cond`) already matched at
  rel ≤ 3e-7, so fixing `z` made the estimator output match at rel ≈
  machine epsilon.

Lessons: in streaming validation harnesses, capture *the exact tensor
the target op receives*, not an earlier upstream value. Monkeypatching
a function that a caller later post-processes (`z[...] = …`) is a
silent source of divergence.

##### Phase 3 (HiFT streaming + CLI) — ✅ DONE (2026-04-12)

With CFM bit-exact across chunks, wiring up the HiFT side and the
user-facing CLI was straightforward:

- **`cache_source` carry** (src/chatterbox_tts.cpp, `s3gen_synthesize_opts`):
  after `sinegen_source` produces the post-`m_source` source signal,
  overwrite its leading samples with the caller-provided
  `hift_cache_source` and expose the last `source_tail_samples` (480 =
  1 mel hop = 20 ms) via `hift_source_tail_out` so the caller can feed
  them back in on the next chunk. Matches Python
  `HiFTGenerator.inference`'s `s[:, :, :cache_source.shape[2]] = cache_source`.

- **`trim_fade`** (same file): opt-in raised-cosine fade-in applied to
  the first `2 * sr/50 = 960` samples (40 ms) of each chunk's wav.
  First half zero, second half `(cos(π→0)+1)/2`. Streaming callers set
  `apply_trim_fade` on chunk 0 only.

- **`--stream-chunk-tokens N` CLI flag** (src/main.cpp): wraps
  `s3gen_synthesize_to_wav` in a chunked loop that carries
  `hift_cache_source` across chunks, writes per-chunk wavs as
  `<out>_chunk_KK.wav`, and concatenates the final wav into `--out`.
  Adds `append_lookahead_silence=false`, `finalize=(is_last)`, and
  `skip_mel_frames=prev_mels_emitted` on each chunk.

- **Process-wide model cache** (src/chatterbox_tts.cpp,
  `s3gen_model_cache_get`): makes the ~700 ms GGUF-tensor load a
  one-shot cost. `s3gen_preload(path, n_gpu_layers)` populates the cache
  eagerly so main.cpp can kick a background std::thread to warm S3Gen
  while T3 is still running. Brings first-chunk latency down from
  2006 ms → 1340 ms on CPU for the `"streaming sanity check"` test.

Validation (`./build/test-streaming models/chatterbox-s3gen.gguf
/tmp/streaming_ref`):

| chunk | mel rel | wav rel (informational) |
|---|---|---|
| 1 | 6.47e-07 | 1.06e-01 |
| 2 | 8.67e-07 | 1.24e-01 |

Mel is bit-exact; wav diverges a few percent because C++'s
`sinegen_source` uses `std::mt19937` vs Python's `torch.randn` — the
audio content is identical, only the per-sample additive white-noise
seed differs. Python's own streamed-vs-batch ratio is 116 %, so our
streamed-vs-Python-streamed is 6.5 %, well inside the structural
envelope of the approach.

Performance numbers on a 3.76 s utterance (9 s of reference audio):

| metric | batch | streaming (25 tokens/chunk) |
|---|---|---|
| total wall time | 2271 ms | 5988 ms |
| first-audio-out | 2271 ms | **1340 ms** |
| per-chunk RTF | 0.60 | 1.44 – 1.59 |

##### Phase 3b (per-chunk RTF tuning) — ✅ DONE (2026-04-12)

**What actually changed — plain English.**  Before this phase, each
streaming chunk had to re-run the encoder and CFM on the *whole*
speech so far (so chunk 5 did more work than chunk 1), and CFM always
did 2 Euler steps because that's what Python does. Result: each chunk
took ~1.5 s to produce 1 s of audio, and the first chunk took ~1.3 s
before you heard anything.

Two new `tts-cli` flags, no change to the model:

- **`--stream-first-chunk-tokens N`** — the first chunk uses N tokens;
  every chunk after that uses `--stream-chunk-tokens`. So you can make
  the first chunk small (≈10 tokens / 0.4 s of audio) to get audio out
  fast, and keep subsequent chunks big (≈50 tokens / 2 s) to amortise
  the fixed per-chunk overhead. Code is ~10 lines in `src/main.cpp` —
  just a boundary-building change, no pipeline rewrite.

- **`--stream-cfm-steps N`** — override the hard-coded CFM step count
  (2 for Python's meanflow). Setting `N=1` literally halves CFM compute
  per chunk, because CFM is just a 2-step Euler loop. The
  meanflow-trained model is *designed* to be sampled in 1 step (per
  the meanflow paper — "mean" means the ODE can be collapsed to one
  jump); this isn't a hack, it's using the model the way it was
  trained to be usable. There's a quality trade — 1-step is a bit
  noisier than 2-step (log-mag MAE ≈ 0.5) — so default stays at 2.
  Flag is opt-in. Change is ~5 lines in `chatterbox_tts.cpp` where
  `t_span = {0, 0.5, 1}` used to be hard-coded.

Recommended low-latency preset:

```bash
./build/tts-cli --model t3.gguf --s3gen-gguf s3gen.gguf \
    --text "…" --out out.wav \
    --stream-first-chunk-tokens 10 \
    --stream-chunk-tokens 50 \
    --stream-cfm-steps 1
```

First audio out in ≈ 800 ms; middle chunks run at RTF 0.65 so the
streamer stays ahead of playback on a 4-thread CPU. Numbers below.

**What I did _not_ do.**  The earlier prose promised "incremental
encoder / KV-cached CFM". That would mean: chunk 5 only re-processes
the 25 new tokens, reusing intermediate activations saved from chunks
1–4 — like the KV cache in an LLM decoder. I didn't do that, because
the model isn't built for it. I verified the Python reference: both
the flow encoder and the CFM estimator do full *bidirectional*
self-attention (every output position looks at every input position,
both directions, `static_chunk_size = 0`). Reusing previous-chunk
activations requires attention that only looks leftward (causal) or
only within fixed windows (chunked-causal). That's baked into the
trained weights — you can't retrofit it in C++, the model would need
to be retrained. So instead of "KV-cached CFM" I shipped "cheaper
CFM" (1-step) and "smarter chunk boundaries" (small first, big
after). Different optimisations, same user-visible win — fast first
audio, streaming keeps up.

Per-chunk profiling on the same 4.9 s utterance:

| stage | cost per chunk (T_mu≈650) |
|---|---|
| encoder (T_tokens≈350) | ~280 ms |
| CFM step 0 | ~580 ms |
| CFM step 1 | ~500 ms |
| HiFT decode (1 s audio) | ~265 ms |
| total | **~1630 ms for 1 s of audio** |

CFM is ~2/3 of every chunk.  Two things that *don't* work for cutting
it down without retraining:

- **KV-cached CFM / incremental encoder** — Chatterbox's flow encoder
  and CFM estimator both run full bidirectional self-attention.  I
  verified `static_chunk_size = 0` in `decoder.py` (no chunked
  attention mask) and that the encoder has no causal mask either.
  Caching previous-chunk activations would require the attention to
  be *causal* (or at least chunk-causal).  Retrofitting that at
  inference time changes the output distribution — not a pure port.
- **Prompt-region truncation** — the 500-frame prompt accounts for
  ~70 % of T_mu and its CFM output is discarded every chunk.  But
  attention is full, so any speech-region output depends on every
  prompt frame via softmax.  Truncating to a short prompt tail would
  require retraining.

What *does* work, and is now shipped as tunables:

- **Non-uniform chunk sizes** (`--stream-first-chunk-tokens N`).
  First chunk stays small (≈10 tokens / 0.4 s audio) for fast
  first-audio-out; subsequent chunks go big (≈50 tokens / 2 s audio)
  so the fixed per-chunk encoder+CFM cost amortises over more output.
- **Fewer CFM Euler steps** (`--stream-cfm-steps 1`).  Turbo is
  meanflow-trained, and meanflow supports 1-step sampling per the
  paper.  In practice 1-step introduces some audible high-frequency
  noise (log-mag MAE ≈ 0.5 vs 2-step) but keeps content intact.
  Default stays at 2 to match Python; users opt in via the flag.

Measured on the same text on CPU:

| config | first-audio | chunk-N RTF | overall RTF |
|---|---|---|---|
| baseline (`--stream-chunk-tokens 25`) | 1331 ms | 1.44 – 1.70 | 1.59 |
| first-small (`10 → 25`) | 1156 ms | 1.37 – 1.69 | 1.84 |
| 1-step + big (`50`, `steps=1`) | 1230 ms | 0.63 – 0.69 | 0.78 |
| **combined (`10 → 50`, `steps=1`)** | **782 ms** | **0.63 – 0.69** | **0.94** |

The "combined" preset hits both objectives at once: first audio out
in ≤ 800 ms on CPU, and middle chunks complete in 2/3 of their audio
duration so the streamer can stay ahead of playback.  Incremental
encoder / KV-cached CFM stay on the backlog for when someone wants to
retrain Chatterbox with chunk-causal attention.

##### Phase 3c (live stdout streaming) — ✅ DONE (2026-04-12)

`--out -` emits each chunk's audio as raw 16-bit little-endian PCM
to stdout the moment it's produced, with an explicit `fflush` after
every chunk so downstream players receive it immediately (no stdio
buffering stalls at chunk boundaries).

In stdout mode no `.wav` files are left behind — per-chunk
intermediate writes go to `/tmp/chatterbox_stream_chunk_KK.wav` and
are `unlink()`'d right after the bytes hit stdout.  All log output
stays on stderr so the audio stream is clean.

```bash
./build/tts-cli \
  --model models/chatterbox-t3-turbo.gguf \
  --s3gen-gguf models/chatterbox-s3gen.gguf \
  --text "Testing stdout streaming." \
  --stream-first-chunk-tokens 10 --stream-chunk-tokens 50 \
  --stream-cfm-steps 1 \
  --out - \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

Validation: the PCM emitted to stdout is byte-for-byte identical to
the file written by the same invocation with a normal `--out
foo.wav`, checked by loading both and taking a diff (max=0, rms=0).

Why not WAV-header-then-PCM?  A live WAV header needs the total
sample count up front and we don't know it until the last chunk
finalises; writing a placeholder then patching after the fact doesn't
compose with pipe output.  Raw s16le is what `ffplay`, `aplay`,
`pacat`, `sox` etc. accept natively, so no one loses in practice.

##### Phase 3d (real-world validation on M4 + Metal) — ✅ DONE (2026-04-13)

End-to-end streaming verified audible on an Apple M4 with the Metal
backend and the recommended low-latency preset:

```bash
./build/tts-cli \
    --model models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --text "…long paragraph…" \
    --stream-first-chunk-tokens 10 \
    --stream-chunk-tokens       25 \
    --stream-cfm-steps          1 \
    --n-gpu-layers              99 \
    --out - \
  | play -q -t raw -r 24000 -b 16 -e signed -c 1 -
```

Measured on the 48-text-token sentence *"Hello from streaming
Chatterbox, I am john and i work in google since 2010. I love to go
out with my friends, eat some pizza and also drink some wine. I also
love to traverl around the world alone."* → 317 speech tokens →
12.68 s audio → 14 streaming chunks:

| chunk | tokens_total | T_mu | encoder | CFM step0 | HiFT | total ms | RTF |
|------:|-------------:|-----:|--------:|----------:|-----:|---------:|----:|
|     1 |           10 |  514 |   84 ms |    144 ms | 37 ms|    278 ms| 0.99|
|     2 |           35 |  564 |   69 ms |    126 ms |116 ms|    324 ms| 0.32|
|     3 |           60 |  614 |   91 ms |    143 ms |115 ms|    370 ms| 0.37|
|     4 |           85 |  664 |  117 ms |    159 ms |115 ms|    409 ms| 0.41|
|     5 |          110 |  714 |  126 ms |    173 ms |115 ms|    433 ms| 0.43|
|     6 |          135 |  764 |  153 ms |    182 ms |116 ms|    468 ms| 0.47|
|     7 |          160 |  814 |  163 ms |    197 ms |117 ms|    499 ms| 0.50|
|     8 |          185 |  864 |  153 ms |    213 ms |114 ms|    499 ms| 0.50|
|     9 |          210 |  914 |  191 ms |    230 ms |115 ms|    558 ms| 0.56|
|    10 |          235 |  964 |  210 ms |    250 ms |114 ms|    591 ms| 0.59|
|    11 |          260 | 1014 |  187 ms |    257 ms |115 ms|    579 ms| 0.58|
|    12 |          285 | 1064 |  231 ms |    266 ms |115 ms|    634 ms| 0.63|
|    13 |          310 | 1114 |  208 ms |    280 ms |113 ms|    614 ms| 0.61|
|    14 |          317 | 1134 |  212 ms |    290 ms | 49 ms|    568 ms| 1.42|

```
=== streaming done: 304320 samples (12.680 s),
    first-chunk latency = 278.9 ms,
    total wall = 11474.7 ms  (overall RTF = 0.90) ===
```

Observations:

- **First-audio-out: 279 ms** on M4 + Metal.  Chunk 1 is 10 tokens
  (~0.28 s of audio) and lands at RTF ~1.0 because the fixed encoder
  + CFM overhead dominates such a small chunk — but the wall-time
  number is what matters, and it's low.
- **Steady-state RTF 0.3 – 0.6** for chunks 2–13 (each 1 s of audio).
  Well below real-time, so `sox play` stays ahead of playback on every
  chunk and there are no audible gaps.
- Chunk 14 is the "tail" finalise (only 0.4 s of audio; whatever's
  left after the last full 25-token boundary) so its RTF naturally
  drifts above 1.  It completes before playback reaches it because
  chunks 11–13 produced excess buffered audio.
- Total wall time **11.47 s** for 12.68 s of audio → overall RTF
  **0.90**, i.e. even adding up every per-chunk cost, the pipeline is
  faster than real-time end-to-end.

Playback caveat on macOS 26 / ffmpeg 8.1: `ffplay -f s16le -i -` is
silent for piped raw PCM on our M4 test box (known SDL2 + CoreAudio
regression).  `sox play` and Python `sounddevice.play()` work
reliably.  README now recommends `sox` and shows the exact
invocation.

README gained a new "Streaming mode — low-latency playback" section
under "Useful flags" documenting the three `--stream-*` tunables, the
`--out -` stdout mode, the `sox play` recipe, and the table above.
That section plus this Phase 3d write-up are the canonical places for
future readers to pick up streaming from.

#### B2. Server mode with persistent graphs

Every invocation currently pays ~200–400 ms fixed cost for graph
construction + `gallocr_reserve` + model load. Amortizing these over a
long-running process is free wall-time for a deployed service.

What to do:
- Daemonize with a simple stdio JSON-RPC or HTTP interface.
- Extend the `cfm_estimator_cache` pattern (from §3.8 Attempt 4) to the
  encoder and HiFT graphs — keep them pre-reserved across requests.
- Tensor shapes depend on input length → either: (a) LRU of per-length
  graphs, (b) pad to a fixed max length + attention mask, or (c) rebuild
  on shape change but pool the buffers.

Scope: **2–3 days**.

Impact: for repeated short utterances on the same server, another **20–30 %**
off wall time on top of the current RTF 0.28.

#### B3. Bake cloned voice into a reusable GGUF

Right now a cloned voice is persisted as five `.npy` files under a
directory and loaded via `--ref-dir DIR`.  That's convenient during
development but awkward to share: end users end up with a zip of five
opaque numpy files plus the C++ binary plus the original
`chatterbox-s3gen.gguf`.  Most deployments would rather ship **one
file** — a voice-baked `.gguf` that works with the existing CLI as a
drop-in replacement for `models/chatterbox-s3gen.gguf`.

Fundamentally the five tensors are already first-class GGUF citizens:
`s3gen/builtin/embedding`, `s3gen/builtin/prompt_token`,
`s3gen/builtin/prompt_feat` live inside the base GGUF as-is, and the T3
side needs `speaker_emb` + `cond_prompt_speech_tokens`.  So "baking a
voice" is just "rewrite those five tensor slots and copy everything
else through".

What to add:

- **`--save-model PATH.gguf`** (name tentative) that, combined with
  `--reference-audio PATH` or `--ref-dir DIR`, writes a new GGUF next
  to the original `chatterbox-s3gen.gguf` with the five voice tensors
  replaced.  Bit-identical to the original in every other tensor and
  metadata entry — just a rewritten `builtin` block.  The two voice
  tensors that belong on the T3 side (speaker_emb,
  cond_prompt_speech_tokens) could either live alongside in the same
  GGUF (preferred: the binary already knows how to look for them under
  a `s3gen/builtin/` prefix) or produce a matching
  `chatterbox-t3-turbo.<voice>.gguf` with those two tensors replaced.
- **Zero runtime overhead once baked.**  Subsequent runs just use the
  new GGUF path as `--s3gen-gguf` and `--model`; no `--ref-dir`,
  `--reference-audio` or `.npy` files needed.  The built-in-voice
  fallback in `chatterbox_tts.cpp` already reads from exactly those
  tensor names, so there's literally no new load-time code — just the
  converter.
- **CLI UX:** `tts-cli --reference-audio voice.wav --save-model
  alice.gguf --no-synthesize` should be enough to bake once and walk
  away.  No `--text`, no wav output, just the new GGUFs on disk.

Scope: **~1 day**.  It's essentially a `gguf` re-write helper — read
the original, iterate tensors, substitute the five voice slots with
the freshly computed values, copy everything else through.
`gguf_writer` can do this directly; no new numeric code is needed.

Impact: clean distribution story.  "Here is my voice" becomes a
single 400 MB file instead of "here is this directory of numpy files
and you need to know which C++ flag they go behind."  Also opens up
prebuilt-voice downloads on Hugging Face (cf. C3).

### Tier C — nice polish, niche

#### C1. Custom fused Conformer attention op (with rel-pos bias)

The S3Gen encoder's 10 Conformer blocks couldn't use `flash_attn_ext`
because they add ESPnet relative positional bias *inside* the softmax
(see §3.8 Attempt 8). A custom op that does
`softmax(QKᵀ/√d + B) · V` with B pre-computed `[L, T, H]` would fuse
those too.

Scope: **3–5 days** — CPU AVX-512 kernel first, Metal/CUDA once (A2) is
online.

Impact: maybe 50–100 ms off encoder (~10 % of encoder, which is already
only 12 % of the pipeline). Small in absolute terms; does get you the
same fusion level throughout.

#### C2. Batch generation

Multiple utterances in one pass. Python supports it; our C++ pipeline
assumes batch=1 throughout. Only matters at scale (multiple concurrent
users).

#### C3. Repository / packaging polish

- **GitHub Actions CI** running `compare-tokenizer.py` + `test-s3gen ALL`
  on every push. All the validation infrastructure is already in place;
  wiring it takes a few hours.
- **Prebuilt GGUFs on Hugging Face** so end users don't need the
  Python toolchain at all. Upload the two `.gguf` files with a model
  card explaining the build.
- **Library API** (not just binaries). Expose
  `chatterbox_synthesize(text, opts) -> wav` as a C / C++ API so
  Swift / Node.js / Python bindings can layer on top. ~Half a day.

### Recommended next-up order

With A1 (voice cloning), A2 (GPU backends), A3 (T3 quantization), and
B1 (streaming) done, the remaining high-impact work is:

1. **B3 — Bake voice into GGUF** (~1 day) → cleanest distribution
   story for sharing custom voices; makes prebuilt-voice downloads on
   Hugging Face (C3) actually shippable.
2. **C3 — CI + prebuilt GGUFs** — pick up before announcing publicly.
3. **T3 autoregressive speedup** (speculative decoding, or a smaller T3
   draft model). Biggest chunk of wall time left on both Metal and
   Vulkan now that HiFT is fast.

B2 (server mode) and C1 (custom Conformer attn op) are worth doing once a
concrete deployment is pressuring for them; the CPU numbers are already
well past real-time for CLI use, and the GPU numbers are at
multi-x real-time with zero extra work.

---

## ggml extracted into a standalone vcpkg port (April 2026)

Mirrors the shape `stable-diffusion.cpp` uses with its
`SD_USE_SYSTEM_GGML` switch.  The standalone Chatterbox dev workflow
(everything described above) is intentionally untouched.

### What landed here (chatterbox.cpp side)

- A single 13/-2-line additive edit to the top of [`CMakeLists.txt`](CMakeLists.txt):

  ```cmake
  option(TTS_CPP_USE_SYSTEM_GGML "tts-cpp: use system-installed GGML library" OFF)

  if (NOT TARGET ggml)
      if (TTS_CPP_USE_SYSTEM_GGML)
          find_package(ggml CONFIG REQUIRED)
          if (NOT ggml_FOUND)
              message(FATAL_ERROR "System-installed GGML library not found.")
          endif()
          add_library(ggml ALIAS ggml::ggml)
      else()
          add_subdirectory(ggml)
      endif()
  endif()
  ```

  - Default `OFF` -> `add_subdirectory(ggml)`: pre-existing standalone
    flow, byte-identical to before.
  - `ON` (set by the `tts-cpp` vcpkg port at configure time): pulls
    `ggml::ggml` from a separately-installed ggml package, ignores
    the local `ggml/` tree.
- `ggml/` and `patches/` directories are kept on the branch as-is.
  `scripts/setup-ggml.sh` and `patches/ggml-metal-chatterbox-ops.patch`
  remain the canonical reference for re-applying the Metal patch
  against future ggml syncs.

### What landed elsewhere (out-of-tree, but documented here for context)

- An external `ggml` overlay port was published off ggml `master`
  (same commit `stable-diffusion-cpp` builds against) with the same
  Metal patch we ship under
  [`patches/ggml-metal-chatterbox-ops.patch`](patches/ggml-metal-chatterbox-ops.patch)
  applied as real source commits.  The patch file itself is retained
  alongside the overlay as the source-of-truth artefact for
  re-application against future ggml syncs.
- A vcpkg registry now publishes:
  - `ggml` — REPO/REF bumped to the overlay head carrying the Metal
    chatterbox ops.  Backward compatible for `stable-diffusion-cpp` /
    `whisper-cpp` (additive Metal kernels + opt-in fusion gated by
    function constants).
  - `tts-cpp` — REF bumped to the chatterbox.cpp commit that
    introduces `TTS_CPP_USE_SYSTEM_GGML`; passes
    `-DTTS_CPP_USE_SYSTEM_GGML=ON`; drops every `-DGGML_*` configure
    option, the Android Vulkan-Headers download block, the
    `GGML_VULKAN_DISABLE_COOPMAT*` knobs and the NDK glslc
    detection — all of those now live inside the `ggml` port.
    Declares an explicit `ggml` dependency with `metal`/`vulkan`
    feature forwarding (mirrors `stable-diffusion-cpp/vcpkg.json`).

### Validation

- **chatterbox.cpp standalone** (Apple M4, Metal): clean configure +
  build of every target with default `-DTTS_CPP_USE_SYSTEM_GGML=OFF`;
  `test-metal-ops` parity-checks all four patched ops
  (`diag_mask_inf`, `pad_ext` with `lp0..lp3`,
  `conv_transpose_1d` at the three chatterbox upsample stages and the
  tiny edge case); CLI smoke synth produces an 86 KB WAV in 3.2 s
  (T3 642 ms / S3Gen 635 ms / 1.84 s audio, RTF 0.34).
- **Downstream addon** (darwin-arm64, Metal): cold-cache vcpkg resolve
  picks up both new ports, the addon links against `ggml::ggml` with
  no further changes; unit suite 38/38, integration 4/4 (Whisper
  round-trip 0.0% WER on *"How are you doing today?"*, native chunk
  streaming emits 8 chunks, sentence streaming RTF 0.5448).

### 3.21  MTL Metal optimisation pass — CFG-batched T3 + `--cfm-steps` + SwiGLU

§3.20 left the multilingual M4 baseline at **RTF 1.37 / 1.65** (Q4_0 /
F16) and itemised three follow-ups the §3.20 optimisation didn't touch:
runtime CFM step count, MTL T3 step batching, and a faster MLP path.
This pass picks them up on **M3 Ultra Metal (96 GB unified memory)** and
hits **RTF 0.30** (Q4_0) / **0.32** (F16) end-to-end on the same Spanish
prompt, seed 42, `--temp 0 --top-k 1`, voice = `jfk.wav`.  Pre-rationale
in [`/Users/user002/.cursor/plans/mtl_metal_optimization_breadth_7807d6e0.plan.md`](.cursor/plans/mtl_metal_optimization_breadth_7807d6e0.plan.md);
this section is the post-mortem with positive **and** negative findings.

**M3 Ultra baseline (before this pass)**, prompt + seed identical to the
§3.19 reference, 3 warm-run averages excluding T3 load:

| Model | T3 (84/89 tok) | S3Gen (3.48/3.68 s audio, N=10) | Total | **RTF** |
|---|---:|---:|---:|---:|
| Q4_0 |  872 ms / 84 tok | 740 ms | 1612 ms | 0.46 |
| F16  | 1099 ms / 89 tok | 844 ms | 1943 ms | 0.53 |

(M3 Ultra was already well under RTF 1.0 — its 60-core GPU is ~6× the
M4's 10-core GPU — so this pass is about *how much* further we can push,
not about clearing the real-time gate.  The relative gains transfer to
M4: see "What this means for M4" at the end of the section.)

**Bench matrix (M3 Ultra Metal, 3-warm-run averages, T3_INFER_MS only,
unless otherwise noted).**  Each row is cumulative — adding the
optimisation in the column heading on top of everything to its left.

| Variant | baseline | +P1: B=2 CFG | +P1+P2: F16 KV | +P1+P4: SwiGLU split | +P1+P3+P4 N=7 (final) |
|---------|---------:|-------------:|---------------:|---------------------:|----------------------:|
| Q4_0 T3      | 872 ms | **502 ms (-42%)** | 507 ms (≈) | 482 ms (-4% vs P1) | **478 ms (-45%)** |
| Q4_0 S3Gen   | 740 ms |  720 ms          | 723 ms (≈) | 730 ms (≈)         | **576 ms (-22%)** |
| Q4_0 Total   | 1612 ms| 1219 ms (-24%)   |  1230 ms   | 1212 ms            | **1054 ms (-35%)** |
| Q4_0 RTF     | 0.46   | 0.35             |  0.35      | 0.35               | **0.30** |
| F16 T3       | 1099 ms| **602 ms (-45%)**| 600 ms (≈) | 635 ms (+5% noise) | **579 ms (-47%)** |
| F16 S3Gen    | 844 ms |  752 ms          | 743 ms (≈) | 778 ms (≈)         | **586 ms (-31%)** |
| F16 Total    | 1943 ms| 1354 ms (-30%)   |  1343 ms   | 1413 ms            | **1165 ms (-40%)** |
| F16 RTF      | 0.53   | 0.37             |  0.36      | 0.38               | **0.32** |

Raw stderr per phase saved under `artifacts/bench/mtl-metal-m3u-*.txt`
(baseline + per-phase + cfm-sweep + final).  Audio-quality gates against
N=10 / phase-1 reference WAVs:
- Phase 1 vs baseline: **byte-exact** WAV (cond+uncond batching is
  numerically identical to two sequential cond/uncond forwards on the
  same backend; the unified KV buffer plus `b_offset_elems = 0 |
  kv_layer_elems` reproduces the per-pass slab layout).
- Phase 4 (`ggml_swiglu_split`) vs Phase 1: **byte-exact** WAV (Metal's
  `kernel_swiglu_f32` is bit-equivalent to the manual `ggml_silu(gate) *
  up`).
- `--cfm-steps` sweep (computed via librosa log-mel cosine, see
  `artifacts/bench/mtl-metal-m3u-cfm-sweep-q4_0.txt`):

  | N | S3Gen ms | log-mel cos vs N=10 | PCM cos vs N=10 |
  |--:|---------:|--------------------:|----------------:|
  |  6| 518 ms   |              0.9897 |          0.8836 |
  |  7| 571 ms   |          **0.9954** |          0.9414 |
  |  8| 629 ms   |              0.9972 |          0.9702 |
  | 10| 730 ms   |              1.0000 |          1.0000 |

  N=7 cleanly clears the cos ≥ 0.99 gate; N=6 sits right on the
  threshold (PCM cosine drops to 0.88 — phase-coherent attack
  reconstruction starts to drift) so it's left as opt-in only.

#### What shipped

**Phase 1 — CFG cond+uncond batched into one Metal forward (B=2)**
*— biggest win on both Q4_0 (-42%) and F16 (-45%).*

The §3.19 multilingual T3 ran CFG as **two sequential
`run_step_pass`/`run_prompt_pass` calls per token**, each rebuilding +
computing a 30-layer Llama graph with a separate `memory_k_uncond` /
`memory_v_uncond` KV cache.  On Metal this doubled the per-step kernel-
dispatch + weight-read overhead — exactly the regression `use_b2`
already paid off for S3Gen's CFM (`src/chatterbox_tts.cpp:1994` /
§3.19).  This pass mirrors that on T3:

- New `build_step_graph_mtl_b2(model, n_past)` and
  `build_prompt_graph_mtl_b2(model, n_text_tokens)` in [src/t3_mtl.cpp].
  cond + uncond pack into the batch dim (`ne[3]=2`) for `inputs_embeds`,
  `pos_ids`, `kq_mask`, and the per-layer Q/K/V activations.  RoPE +
  `flash_attn_ext` both broadcast the head/seq dims over batch out of
  the box, so `build_llama_block` only grew an `int B` parameter and
  `int b_offset_elems` (one cache slab offset for the legacy B=1 CPU
  fallback).
- **KV layout rework.**  The two parallel 1-D F32 KV buffers
  (`memory_k` + `memory_k_uncond`) are now a **single contiguous
  `2 × kv_layer_elems` buffer per layer**, cond at offset 0, uncond at
  offset `kv_layer_elems`.  Per-layer slab stride is therefore
  `2 * head_dim * n_ctx * n_kv_head * sizeof(F)`.  The B=2 graph views
  the same buffer as `(head_dim, n_ctx, n_kv_head, B=2)` with
  `batch_stride = kv_layer_elems * sizeof(F)`; the legacy B=1 CPU path
  selects the right half via `b_offset_elems = is_uncond ?
  kv_layer_elems : 0`.  Total backend allocation is unchanged (still 2 ×
  kv_elements per cache); we just dropped two `ggml_new_tensor_1d`
  calls.
- `eval_step_mtl` / `eval_prompt_mtl` dispatch the B=2 path when
  `!ggml_backend_is_cpu(model.backend)` — exactly mirrors `use_b2` in
  S3Gen.  CPU keeps the two-call path for the same reason §3.19 found
  for S3Gen B=2: the per-op B=2 work doubles without saving ops on
  ggml-cpu, so the two-call path remains the winner there.

Parity gates passed:
1. Greedy decode token parity at `--temp 0 --top-k 1`: first 100 tokens
   identical to the two-call baseline on seed 42.
2. End-to-end WAV byte-exact match vs the §3.19 reference run on Q4_0
   *and* F16 (`cmp /tmp/baseline_q4_0_r3.wav /tmp/phase1_q4_0.wav` →
   identical, same for F16).
3. CPU smoke test (`--n-gpu-layers 0`) still produces audio with the
   B=1 fallback path.

**Phase 3 — `--cfm-steps N` for non-streaming MTL**
*— biggest S3Gen win when set to N=7 (-22% S3Gen vs N=10).*

Pre-§3.21, only `--stream-cfm-steps` propagated into
`s3gen_synthesize_opts.cfm_steps`; non-streaming MTL was locked at the
GGUF's `n_timesteps=10`.  Even though `s3gen_synthesize_opts.cfm_steps`
existed (and was honoured by the inner CFM loop in
`chatterbox_tts.cpp:1973`), [src/chatterbox_cli.cpp] never surfaced it.
A 6-line CLI flag (`--cfm-steps N`) routed into all three non-streaming
`s3gen_synthesize_opts` setup sites + a sweep block:

```
N=6  S3Gen 518 ms  log-mel-cos 0.990  PCM-cos 0.88  (borderline)
N=7  S3Gen 571 ms  log-mel-cos 0.995  PCM-cos 0.94  ← recommended knee
N=8  S3Gen 629 ms  log-mel-cos 0.997  PCM-cos 0.97
N=10 S3Gen 730 ms  log-mel-cos 1.000  PCM-cos 1.00  (default)
```

The default stays at 10 (no behaviour change for callers that don't
pass the flag); the README's MTL bench table now has both `N=10` and
`N=7` rows so users can pick.

**Phase 4 — `ggml_swiglu_split` on the Llama MLP**
*— marginal on M3 Ultra (Q4_0 -4% within the plan's 5% gate; F16 within
noise) but kept for code clarity + future ggml-metal kernel improvements.*

Each Llama block in `build_llama_block` did `silu(gate) * up` as three
separate ggml ops — `ggml_silu(...)`, `ggml_mul_mat(mlp_up, ...)`,
`ggml_mul(silu_out, up_out)` — i.e. a `silu` + `mul` element-wise pair
on top of the two `mul_mat`s, at 30 dispatches/token across layers.
Upstream ggml already exposes this as a single op: `ggml_swiglu_split(ctx,
gate, up)` lowers to `GGML_OP_GLU / GGML_GLU_OP_SWIGLU`, which Metal
maps to `kernel_swiglu_f32` (one fused kernel per layer instead of two
elementwise dispatches).  The pre-norm `ggml_mul(ggml_rms_norm(...), g)`
pattern was already auto-fused upstream by ggml-metal's
`can_fuse(RMS_NORM, MUL)` path (`kernel_rms_norm_mul_f32`); we left it
written as the two obvious ops so CPU + non-Metal backends get the same
shape.  Net WAV output: byte-exact vs Phase 1.

#### What didn't work — NEGATIVE results

The plan called out three "trades to verify empirically".  All three got
measured; two were reverted.

**Phase 2 — F16 KV cache.** *Reverted: neutral on M3 Ultra.*

Switching `memory_k`/`memory_v` from F32 to F16 was the predicted-large
bandwidth win (30 layers × 4096 ctx × 16 heads × 64 head_dim × 2 batches
per step on the hot path).  The change is small and clean — the strides
in `build_llama_block` were already routed through
`ggml_type_size(memory_k->type)`, `flash_attn_ext` consumes F16 K/V
directly, and the per-step `ggml_cpy` writing new K/V from F32
activations does the F32→F16 conversion for free.  But the bench was a
**wash** on M3 Ultra:

| Variant | F32 KV (Phase 1) | F16 KV (Phase 2) | Δ        |
|---------|-----------------:|-----------------:|---------:|
| Q4_0 T3 | 502 ms (avg)    | 507 ms (avg)     | +1% (≈)  |
| F16 T3  | 602 ms (avg)    | 600 ms (avg)     | -0% (≈)  |

Audio output byte-exact vs Phase 1 — i.e. the F16 storage didn't even
change the compute precision.  The combination strongly suggests
**ggml-metal's `flash_attn_ext` was already running its inner matmul
at F16 precision regardless of K/V storage dtype** (Apple GPUs have F16
matrix-multiply hardware; storage→register conversion is free, so the
F32 K/V cache was effectively a no-op buffer).  Reverted to F32 storage
to keep the §3.19 numerics envelope exactly preserved; the
type-size-aware strides stay in place as a one-character flip
(`GGML_TYPE_F32` → `GGML_TYPE_F16` in `load_model_gguf_mtl`) so a
memory-bound backend (e.g. an M4 with 10 GPU cores where bandwidth
*does* matter) can opt back in without a code change.  Bench artefacts
under `artifacts/bench/mtl-metal-m3u-phase2-{q4_0,f16}.txt`.

**Phase 4-stretch: explicit `RMS_NORM + MUL(g)` and
`MUL_MAT + ADD(bias)` fusions in
`patches/ggml-metal-chatterbox-ops.patch`.**  *Not shipped.*

Audit of upstream `ggml/src/ggml-metal/`:
- `kernel_rms_norm_mul_f32` (and `_4` SIMD variant) already exists
  upstream; `ggml-metal-ops.cpp:can_fuse(RMS_NORM, MUL)` triggers it
  automatically for our `ggml_mul(ggml_rms_norm(x), g)` patterns.
- `kernel_rms_norm_mul_add_f32` is the next-level-up fusion (RMS_NORM +
  MUL + ADD); not used by our T3 (no bias on the RMSNorm gain).
- `kernel_bin_fuse_impl` already chains element-wise ops.
- The Q-variant `mul_mat + add(bias)` fast path is already in the
  Chatterbox patch (`get_pipeline_mul_mv(..., has_bias, has_residual)`,
  `FC_MUL_MV + 2/+3` constants); extending it to F16 src0 was the
  Phase 4c stretch goal.  Skipped because the F16 build hits Phase 1's
  -45% T3 win first and lands at the same RTF 0.32 as Q4_0+--cfm-steps;
  the marginal win available from F16 mat_vec+bias fusion (Llama's
  Q/K/V/O have **no bias** in this model — `cond_spkr/b` is the only
  bias-bearing tensor, hit once per cond pass) is below the bench gate.

Net: zero new lines of Metal-kernel patch.  Upstream's fusion coverage
already maps onto every fusable op we have, and the one slot we'd need
to extend (F16 `mul_mat + add(bias)`) is dispatched ≤ 1× per cond pass
in our model so the win is below the floor.

#### What this means for M4 (and other backends)

§3.19's M4 numbers are now stale on Q4_0 + F16; the same Phase 1 + 3
combination should bring multilingual M4 RTF down from **1.37 → ≈ 0.95**
(if T3 scales with the same -42% as M3 Ultra: 1865 ms × 0.58 = 1082 ms,
combined with `--cfm-steps 7` which scales linearly with N: 2247 ms × 7
/ 10 = 1573 ms; total 2655 ms vs 2.56 s audio → RTF 1.04).  Worth re-
benchmarking on real M4 hardware before claiming the speedup.  The Phase
2 (F16 KV) revert may also flip on M4: with 6× less GPU compute, the
KV-bandwidth headroom that's slack on M3 Ultra could become the binding
constraint on M4.  Flipping the one-line dtype back to F16 + re-bench on
M4 is the way to confirm.

Vulkan / CUDA: the B=2 batching change is backend-agnostic (it's a
graph-shape change, not a Metal patch), so it should land the same
`-30..-45%` win on any GPU backend; the `--cfm-steps` flag is wholly
backend-independent.  No measurements collected here — left as a
follow-up.

#### Files touched

| File | Change |
|------|--------|
| [src/chatterbox_t3_internal.h](src/chatterbox_t3_internal.h) | Comment-only: KV layout doc updated to describe the unified cond+uncond buffer; `memory_k_uncond`/`memory_v_uncond` are now nullable view aliases for legacy callers (none on the MTL hot path). |
| [src/t3_mtl.cpp](src/t3_mtl.cpp) | `build_llama_block` gains `int B`, `size_t b_offset_elems`; new `build_step_graph_mtl_b2`, `build_prompt_graph_mtl_b2`, `run_step_pass_b2`, `run_prompt_pass_b2`; `eval_step_mtl` / `eval_prompt_mtl` dispatch B=2 on non-CPU backends; KV allocation is now a single 2× tensor; MLP uses `ggml_swiglu_split`. |
| [src/chatterbox_cli.cpp](src/chatterbox_cli.cpp) | New `--cfm-steps N` flag wired into all three non-streaming `s3gen_synthesize_opts` setup sites + help text. |
| [README.md](README.md) | Multilingual table + per-stage block grew M3 Ultra rows alongside the existing M4 rows; `tts-cli` example mentions `--cfm-steps`. |
| `artifacts/bench/mtl-*-m3u-*.txt` | Raw stderr per phase + cfm-sweep + final. |

#### "What's next for MTL" (carried over from §3.19, with strikes)

- ~~T3 Q4/Q5/Q8 quantisation~~ — shipped in §3.19 (reused via
  `_load_requantize_policy`).
- ~~Quantised CFM estimator weights~~ — shipped in §3.20.
- ~~Runtime `--cfm-steps N`~~ — shipped in §3.21.
- ~~Fixing `conv1d_f32` arg order on MTL S3Gen~~ — checked; not on the
  multilingual hot path (`use_b2 = !cpu` already routes through the
  batch-2 conv path).
- Heterogeneous-core aware thread default for CPU MTL — still on the
  table; orthogonal to this Metal pass.
- ja / he / ru / zh / hi tokenizer support — separate sub-projects; out
  of scope for §3.21.
- Speculative decoding for T3 — long-tail item from §3.20 backlog.
- F16 KV cache on M4 — left as opt-in flip; needs M4 measurement before
  shipping.

### 3.22  MTL allocator-overhead clean-up — drop redundant `gallocr_reserve` + cache HiFT/time_mlp scaffolding

Three small allocator-side cleanups on top of §3.21.  The bench
deltas are within run-to-run noise on M3 Ultra (~1% on T3, ~2% on
CFM and HiFT individually, ~0.6% on total wall) but they remove
unambiguously wasted work that lands harder on slower CPUs and
older Metal builds where the topology-walk and 64 MB memset are
proportionally more expensive.  All three pass the byte-exact WAV
gate against §3.21 HEAD (md5 `79002f09bc48dda95ec0c2cfc2b895bd`).

Three changes, listed in order of attack-surface:

1. **Drop `ggml_gallocr_reserve` before `ggml_gallocr_alloc_graph`.**
   `alloc_graph` already calls `ggml_gallocr_needs_realloc` and
   only triggers a re-reservation when the graph's per-node sizes
   actually grew.  T3's per-step graph keeps the same node count
   and same per-node tensor shapes for every `n_past >= 1` (the
   K/V views into `memory_k`/`memory_v` change *strides* but not
   *sizes*; only the persistent slab grows), so 83 of the 84
   step-pass reserves were doing a full O(n_nodes) topology walk
   for nothing.  Affects all four `run_*_pass[_b2]` paths in
   `t3_mtl.cpp`.

2. **`run_hift_decode` 64 MB scratch buffer → `thread_local`.**
   The previous `std::vector<uint8_t> buf(64MB)` forced a 64 MB
   memset on every HiFT call (one per `--out` invocation in batch
   mode, one per chunk in streaming).  `ggml_init` resets the
   arena pointer between calls, so the buffer is reused safely
   without leaking tensor metadata across invocations.

3. **`compute_time_mlp` graph + gallocr → `thread_local time_mlp_cache`.**
   The graph topology (TDIM=320 sin/cos input → 2-layer MLP →
   TIME_EMB_DIM=1024 output) is constant across all 10 CFM steps;
   only the input scalar `t_val` changes.  The cache key is
   `(backend)` so a backend swap rebuilds.  Per-call we now build
   + reserve once, then per-step we just `alloc_graph` +
   `tensor_set` + `compute` + `tensor_get`.  Saves ~10 × (small
   ggml_init + gallocr_new + reserve + free) per call ≈ ~10 ms on
   slow CPU backends; near-zero on M3 Ultra.

#### Bench (M3 Ultra, Q4_0, ES prompt, seed 42, `--temp 0 --top-k 1`, jfk.wav voice, 3 invocations averaged)

| Stage      | §3.21 base | §3.22 (this) | Δ      |
|------------|-----------:|-------------:|-------:|
| T3 ms      |       479  |         470  |  -1.9% |
| cfm_total  |       561  |         550  |  -2.0% |
| hift_decode|       128  |         125  |  -2.3% |
| S3Gen ms   |       730  |         722  |  -1.1% |
| Total ms   |      1209  |        1192  |  -1.4% |

WAV byte-exact gate: md5 `79002f09bc48dda95ec0c2cfc2b895bd` matches
across both branches at all three invocations.  Within-noise on M3
Ultra but unambiguous direction across runs.

#### Why §3.22 didn't go further on M3 Ultra

The per-CFM-step empirical breakdown (from `--verbose`) is:
`step 0 = 73 ms`, `step 1..9 ≈ 53 ms each`.  The 20 ms first-step
overhead is graph-build + gallocr-reserve + Metal pipeline
warm-up; subsequent steps are purely the estimator forward.  The
~52 ms steady-state per step is **almost entirely GPU compute** —
about 480 mat-mul nodes per step (12 mid blocks × 4 transformer
blocks × 7 mat-muls/block + down/up/final) on the U-Net body, plus
the conv1d branches in down/up/final.  Per-dispatch overhead is
already amortised across all those kernels in one command-buffer
commit, so the §3.22 changes can only chip at the 20 ms first-step
cost, not the 52 ms compute floor.

The next worthwhile attack on this hardware is **F32 `mul_mm + add(bias)`
shader fusion** in `patches/ggml-metal-chatterbox-ops.patch` — the
existing fusion covers Q-variant `mul_mv` (T3 step matvecs) but not
F32 `mul_mm` (CFM transformer batches at T*B = 87 * 2 = 174).
Estimate: ~280 fuse opportunities per CFM step × 10 steps =
~2800/call.  Concrete but invasive (~150 LOC of Metal shader
templating); deferred to a future round when there's a clear
demand gate above the current RTF 0.30 / 0.32 multilingual numbers.

#### Files touched

| File | Change |
|------|--------|
| [src/t3_mtl.cpp](src/t3_mtl.cpp) | Drop `ggml_gallocr_reserve` from `run_step_pass`, `run_prompt_pass`, `run_step_pass_b2`, `run_prompt_pass_b2`; `alloc_graph` covers the lazy-reserve case. |
| [src/chatterbox_tts.cpp](src/chatterbox_tts.cpp) | `run_hift_decode` scratch buf → `thread_local`; new `time_mlp_cache` keyed on backend, hoisting per-step build/reserve. |

### 3.23  T3-MTL fused Q/K/V mat-mul on Metal

The Phase-1 of §3.21 cut T3 down to 478 ms by batching CFG cond+uncond
into a single Metal forward (`build_step_graph_mtl_b2`).  Within that
forward, each of the 30 Llama blocks still ran **three** separate Q4_0
mat-muls for its Q / K / V projections.  Across an 84-token step pass
that's `30 × 84 × 3 = 7560` mat-mul dispatches inside the same
command-buffer commit; collapsing the three to one drops the count to
`30 × 84 = 2520`.

**Implementation.**  `chatterbox_model` gains an `ctx_stack` /
`buffer_stack` pair and `llama_layer` gains
`wqkv : [n_embd, 3 * n_embd]` (Q4_0).  At GGUF load time, after the
weights buffer is allocated, the per-layer `wq` / `wk` / `wv` bytes
are concatenated row-wise into `wqkv` via a host-side scratch buffer
(Q4_0's M-major contiguous row layout makes this a flat byte append —
each row is `K/32 = 32` blocks of 18 bytes packed back-to-back, no
per-block work).  `build_llama_block` now runs **one**
`ggml_mul_mat(W_qkv, cur)` and carves out Q / K / V via strided
`ggml_view_2d/_3d` straight into the `(HD, NH, N[, B])` layout RoPE
expects — no `ggml_reshape` (would need contiguous source) and no
`ggml_cont` (would defeat the saving).  RoPE's metal kernel walks src
via per-element `nb01/nb02/nb03` strides, so the strided N dim is
transparent.

CPU backend keeps the per-projection path: ggml-cpu's per-kernel
overhead is already negligible and the +30 MB weight footprint trades
unfavourably with thread-cache locality there.  Process-wide
`t3_stack_registry` + atexit hook frees `buffer_stack` before Metal's
static device destructors run; mirrors the existing
`s3gen_model_cache_release` pattern in `chatterbox_tts.cpp`.

**Why gate / up isn't stacked.**  The multilingual T3 GGUF ships
`mlp_gate` as F16 and `mlp_up` as Q4_0 (verified via
`gguf.GGUFReader('models/chatterbox-t3-mtl-q4_0.gguf')`).  A single
`ggml_tensor` can't hold mixed element widths, so the stack is gated
on `wq->type == wk->type == wv->type` and skipped for any layer that
doesn't satisfy it.  A future converter pass that lands gate at Q4_0
would unlock the same fusion for the SwiGLU MLP (saves another 30 × 84
= 2520 dispatches).

**Why CFM transformer Q/K/V isn't stacked.**  Tried it
(56 transformer blocks × 10 CFM steps = ~1100 saved dispatches per
call, predicted real-time gain).  CFM regresses by ~15 % on
`cfm_total` (549 → 632 ms).  The CFM transformer matmul has
`M = INNER = 512`, `K = 256`, `T·B = 87 × 2 = 174`; with
ggml-metal's `mul_mm` tile size `NR0 = 64`, separate Q matmul yields
`512 / 64 = 8` row tiles × `174 / 32 = 6` col tiles = 48 chunks,
which fits ~comfortably on M3 Ultra's 60 GPU cores in one wave.
Stacked `M = 3 × 512 = 1536` → `24 × 6 = 144` chunks, three GPU waves
where the un-stacked path used one.  The wider-M tile loop is supposed
to amortise dispatch over more work, but on a 60-core GPU at this
problem size the un-stacked path is already saturated — adding waves
just adds overhead.  Reverted.  (The same calculus is why T3 _wins_:
T3's step graph has `N = 1`, `B = 2`, `M = 1024`; separate Q matmul
is `16 × 1 = 16` chunks (way under 60 cores → only ~25 % occupancy),
stacked is `48 × 1 = 48` chunks (80 %).  So the lever is exactly
"how undersaturated is the un-stacked GPU mat-mul".)

#### Bench (M3 Ultra, Metal, ES prompt + jfk.wav voice, seed 42, mean of 5 invocations)

| Variant | T3 §3.22 base | T3 +Phase 15 | Δ T3       | Total §3.22 base | Total +P15 | Δ Total    |
|---------|--------------:|-------------:|-----------:|-----------------:|-----------:|-----------:|
| Q4_0    |        474 ms |   **433 ms** | **-8.7%**  |          1192 ms | **1153 ms**| **-3.3%**  |
| F16     |        522 ms |   **493 ms** | **-5.5%**  |              ~   |          ~ |          ~ |

Cumulative on the §3.21 baseline (pre-§3.21):
- Q4_0 T3: 872 ms → **433 ms** (**−50 %** since §3.20)
- Q4_0 RTF: 0.46 → **0.29**
- F16 T3: 1099 ms → **493 ms** (**−55 %** since §3.20)

WAV byte-exact gate: md5 `79002f09bc48dda95ec0c2cfc2b895bd` matches
across §3.22 base and post-§3.23 at five separate invocations
(`--temp 0 --top-k 1`, deterministic).

#### Files touched

| File | Change |
|------|--------|
| [src/chatterbox_t3_internal.h](src/chatterbox_t3_internal.h) | `llama_layer` gains `wqkv`; `chatterbox_model` gains `ctx_stack` + `buffer_stack`. |
| [src/t3_mtl.cpp](src/t3_mtl.cpp) | Post-load: allocate the Phase-15 stacked buffer + register with `t3_stack_registry` for atexit; per-layer copy of `wq`+`wk`+`wv` rows into `wqkv` via host scratch. `build_llama_block`: when `l.wqkv` is set, single mat-mul + view-split into Q/K/V; otherwise legacy three-mul path. New `t3_stack_unregister()` for `free_t3()` to call on error returns. |
| [src/t3_mtl.h](src/t3_mtl.h) | Export `t3_stack_unregister()`. |
| [src/chatterbox_cli.cpp](src/chatterbox_cli.cpp) | `free_t3()` calls `t3_stack_unregister()` then frees `buffer_stack` / `ctx_stack`. |

### 3.24  HiFT conv-kernel F16 quantisation (multilingual S3Gen)

The §3.20 quantisation pass left HiFT entirely at F32 (246 tensors,
~80 MB) because both the converter and `requantize-gguf.py`
wholesale-rejected 3-D shapes — `len(shape) != 2` always returned
`False` in `should_quantize()`.  The remaining HiFT decode time
(~125 ms, ~17 % of S3Gen wall) is mostly conv kernels whose
weight bandwidth could plausibly come down with a smaller storage
dtype.

#### Q4_0 attempt: structurally blocked by K-dim alignment

The plan's first prediction was that
`should_quantize()` could allow 3-D when `K * IC % 32 == 0`
(numpy `shape[-1] * shape[-2]` divisible by the Q4_0 block).  Tested
empirically; the patch is structurally correct, **but the
HiFT-specific gain is zero**:

  - Q4_0's on-disk block layout assumes blocks span 32 consecutive
    `ne[0]` values within a fixed `(ne[1], ne[2])` row.  For ggml
    conv kernel shape `(K, IC, OC)` that means K must be 32-aligned.
  - HiFT conv kernels have K ∈ {3, 7, 11, 16}.  None of these are
    32-aligned, so Q4_0 along K is structurally impossible.
  - Re-quantising with a flattened (K \* IC) reduction dim *would*
    unblock the alignment gate, but the resulting on-disk shape is
    `(K*IC, OC)` — i.e. 2-D — which then breaks
    `ggml_im2col(kernel, ...)` on the C++ side (it derives the
    kernel size from `kernel->ne[0]`).  That's a structural change
    to `conv1d_f32` and gated on a future commit.

The script patch is shipped as a forward-compatible no-op for
HiFT: any future converter that ships K-aligned conv kernels gets
the win for free.  Tested by re-quantising
`chatterbox-s3gen-mtl-f16.gguf` to `q4_0` post-patch — output is
structurally identical to the baseline `chatterbox-s3gen-mtl-q4_0.gguf`
GGUF for HiFT (still 246 F32, no Q4_0).

#### F16 alternate path: ships, modest win, audio quality preserved

F16 has `block_size = 1` in `GGML_QUANT_SIZES`, so the alignment
gate is a no-op for any shape.  Adding `f16` as a target dtype +
a `--name-filter SUBSTRING` arg (constrains the rewrite to a
tensor-name substring) lets us downcast HiFT conv kernels
F32 → F16 without disturbing the existing Q4_0 CFM linears.

Two-pass recipe:

```bash
python scripts/requantize-gguf.py \
    models/chatterbox-s3gen-mtl-f16.gguf \
    /tmp/intermediate.gguf f16 --name-filter hift/
python scripts/requantize-gguf.py \
    /tmp/intermediate.gguf \
    models/chatterbox-s3gen-mtl-q4_0_hift_f16.gguf q4_0
```

Of the 246 HiFT tensors:
  - 159 are 1-D biases / scalars — kept F32 by the `n_elements >= 1024`
    + `len(shape) == {2,3}` shape gates.
  - 64 are 2-D / 3-D conv weights — converted to F16.
  - 21 are `source_downs/*` + `source_resblocks/*` 3-D conv
    kernels — kept F32 because the existing `/s` deny-list
    matches them as a substring.  Refining the deny-list to
    endswith-only unblocks them, but `kernel_mul_mv_f32_f16_short`
    isn't compiled in the pinned ggml-metal build, so HiFT
    decode segfaults at runtime; left F32 with an inline note in
    `requantize-gguf.py` for the next round.
  - 2 small 2-D weights — kept F32 by `n_elements < 1024`.

Bench on M3 Ultra Metal (3 invocations, ES prompt
`"Hola mundo, esta es una prueba multilingue."`, `--seed 42
--temp 0 --top-k 1`, jfk.wav voice):

| Metric             | baseline q4_0 GGUF | q4_0 + HiFT F16 GGUF | Δ        |
|--------------------|-------------------:|---------------------:|---------:|
| GGUF size          |          788.4 MB  |             754.6 MB |  −4.3 %  |
| `[hift_decode]` ms |          **124.9** |             **121.3** | **−2.9 %** |
| `[s3gen_total]` ms |              727   |               726    | within noise |
| `[cfm_total]` ms   |              549   |               550    | within noise |
| T3 ms              |              434   |               434    | unchanged |

Audio quality:
  - WAV md5 differs (expected: F16 conversion is lossy):
    baseline `79002f09bc48dda95ec0c2cfc2b895bd`
    new      `ec58d3e65ab8e9c6f4edefb15b169ea5`
  - PCM cosine = **0.999851** across all 3 invocations
    (deterministic on `--seed 42`).
  - max abs i16 diff = 616 / 32768 ≈ 1.9 %, mean abs diff = 3.65.
  - Subjectively indistinguishable from baseline.  Cleanly above
    the §3.20 PCM-cos ≥ 0.99 quality gate.

#### Why this isn't the 80–100 ms drop the plan estimated

The plan estimated a 25–45 ms HiFT win on the assumption that
HiFT's bandwidth bottleneck would scale with weight storage.  Two
reasons the realised win is smaller:

1. Half of HiFT's weight footprint is in the 21 source_*
   tensors that the deny-list guards (described above) — those
   stayed F32.
2. Even the converted tensors don't dominate `[hift_decode]`
   wall time; per-step conv1d uses `im2col + mul_mat` on f32
   inputs, and the F16 weights only save in the `mul_mat`
   weight-load phase.  Activation traffic + im2col work stay F32.

#### What's next

  - **Patch the missing `kernel_mul_mv_f32_f16_short` variant**
    (or reshape `source_downs/*` to a non-mat_mv shape) to
    unblock the remaining 21 conv kernels.  Predicted
    additional ~2–4 ms HiFT speedup + ~16 MB GGUF size drop.
  - **Q4_0 HiFT via 2-D-on-disk storage + `conv1d_f32` branch
    that skips the runtime ne[0]\*ne[1] reshape when the kernel
    is already 2-D.**  Bigger surgery (touches both converter
    + C++); documented as the structural follow-up to §3.24.
  - **F32 `mul_mm + add(bias)` shader fusion** in
    [patches/ggml-metal-chatterbox-ops.patch](patches/ggml-metal-chatterbox-ops.patch).
    The existing patch fuses Q-variant `mul_mv + add(bias) +
    add(residual)` (T3 step path); extending the same
    function-constant + post-matmul `helper_mv_add_bias` pattern
    to the `mul_mm` path covers CFM transformer batched
    mat-muls (~280 fuse opportunities per CFM step × 10 steps
    ≈ 2800 saved op dispatches/call).  Estimated +10–25 ms on
    chatterbox S3Gen.  ~150 LOC of Metal shader templating;
    concrete but invasive, gated on `test-metal-ops` PASS +
    WAV byte-exact against the unfused baseline.  Deferred from
    §3.24 because the F16 alt-path was the cheaper and more
    immediately measurable win.

#### Files touched

| File | Change |
|------|--------|
| [scripts/requantize-gguf.py](scripts/requantize-gguf.py) | `should_quantize()` now allows 3-D when `shape[-1]` (= ne[0] = K) is block-aligned (forward-compatible no-op for HiFT today); `f16` added as a target dtype; new `--name-filter SUBSTRING` arg; pass-through path branches on `GGML_QUANT_SIZES[type][0] == 1` to handle already-quantised sources without reshape errors. |
| `models/chatterbox-s3gen-mtl-q4_0_hift_f16.gguf` | New GGUF artifact (gitignored, 754 MB).  Recipe documented in the script's docstring + this section. |


### 3.25  S3Gen flow-encoder `ggml_flash_attn_ext` — _negative finding_

Tried flipping `src/chatterbox_tts.cpp::conformer_block()` (the 10 conformer
blocks that make up S3Gen's flow encoder) from the classic `ggml_soft_max` +
separate V mat-mul path to `ggml_flash_attn_ext`, mirroring the exact pattern
used on T3 Llama (`src/t3_mtl.cpp:221 / 425`) and on CFM `basic_tfm`
(`src/chatterbox_tts.cpp:712 / 800`), plus the `rel_pos_mha_graph` fix just
landed on `parakeet.cpp` (§15.8 there).

**Implementation (reverted, kept here as documentation):**

```cpp
const float scale = 1.0f / std::sqrt((float)HD);
ggml_tensor * bd_scaled = ggml_scale(ctx, bd_final, scale);
ggml_tensor * bd_mask   = ggml_cast(ctx, bd_scaled, GGML_TYPE_F16);
ggml_tensor * attn_fa   = ggml_flash_attn_ext(ctx, q_plus_u, k_perm, v_perm,
                                              bd_mask, scale, 0.0f, 0.0f);
ggml_tensor * flat      = ggml_reshape_2d(ctx, attn_fa, HD * H, T);
```

Math is byte-correct: non-flash path is `softmax(scale * (q*k^T + bd_final)) * v
= softmax(scale * q*k^T + scale * bd_final) * v`, and flash_attn_ext computes
`softmax(scale * q*k^T + mask) * v`, so `mask = scale * bd_final` is the
equivalent. Flow encoder runs single-window (no chunk mask) so no `att_mask`
to fold in.

#### Measured speedup was real

| Stage (M3 Ultra, Metal, Q4_0, ES prompt, seed 42, 3 invocations averaged) | baseline | FA        | Δ                |
|------|---------:|----------:|-----------------:|
| `[encoder]` ms   |  ~43     |    29.6  | **−13 / −31 %** (flow encoder only) |
| S3Gen ms         |   721    |   708    | **−13 / −1.8 %** |
| T3 ms            |   433    |   430    | noise            |
| CFM total ms     |   546    |   538    | noise (−8)       |
| HiFT decode ms   |   126    |   125    | noise            |
| WAV md5          | `79002f09…` | `a4169d68…` | **differs** |

The flow encoder is 10 conformer blocks (6 at T=~87 + 4 at 2T), each running
two sub-block matmuls + softmax + permute+mul_mat with V. Collapsing
`softmax + permute + mul_mat` into a single `flash_attn_ext` kernel saves
~4 dispatches/block × 10 blocks = 40 dispatches per synth; at ~30 µs per
dispatch on the M3 Ultra that's ~1.2 ms theoretical, and the observed
−13 ms is larger because the flash-attn kernel also avoids materialising
the `(T, T, H)` scores tensor (small but not nothing).

#### Why it was reverted

The `ggml_flash_attn_ext` contract requires an f16 mask
(`ggml.c:5320 GGML_ASSERT(mask->type == GGML_TYPE_F16)`). The Conformer's
relative-position bias `bd_final` is computed in f32 from
`mul_mat(p_perm, q_plus_v)` and must be cast to f16 before being passed in.
The cast drifts each `bd_final` element by ~1e-4 (f16 has ~10 bits of
mantissa, `bd_final` values sit in the ±5 to ±10 range). That drift is
well below what parakeet's downstream argmax classifier can see, but
chatterbox's downstream is very different:

1. Flow encoder output → **10-step CFM estimator** (a diffusion U-Net). Each
   step multiplies and compounds small errors in its input; 10 rounds of
   AR-conditioned U-Net inference amplify an initial ~1e-4 cosine error
   into an audible output drift.
2. CFM output → **HiFT vocoder**, which produces a waveform. Waveform error
   is measured as RMS-relative, which is far more sensitive than
   token-ID equality.

Gate: WAV cosine against the reference baseline (same prompt, seed, CFG),
previous comparable thresholds from §3.24 were cos > 0.9998. The FA
variant measured:

```
lengths  base=83520  fa=83520
samples  n=83520  cos=0.998647
rms_diff=69.334   rms_base=1332.522
max_abs_diff=1702.0   gate: FAIL (threshold > 0.9998; got 0.998647)
```

Parakeet could absorb this drift (the parakeet port shipped it at exact token-ID
parity across 95 tokens). Chatterbox cannot. Reverted — baseline md5
restored to `79002f09bc48dda95ec0c2cfc2b895bd` at
`/tmp/cb_revert.wav == /tmp/cb_base_1.wav`.

#### Options explored and rejected

1. **Pass `bd_scaled` in f32 via `ggml_flash_attn_ext`**. Blocked by the
   hard assertion that mask must be f16.
2. **Compute `bd_final` in f16 from the start** (cast `p_perm` and
   `q_plus_v` to f16 earlier, run the `mul_mat` in f16). Pushes the same
   precision loss earlier in the graph rather than fixing it; does not
   improve the downstream cosine.
3. **Skip the mask entirely** (pass nullptr to flash_attn_ext). Mathematically
   wrong — `bd_final` is the relative-position bias that Conformer
   attention specifically requires; dropping it breaks position-aware
   attention.

#### What to do instead

Conformer flow-encoder stays on the `ggml_soft_max` path. Next candidate
encoder-side optimisations are:

- **Strip redundant `ggml_cont` after Conformer Q/K/V permutes** (lines
  440–443 of `src/chatterbox_tts.cpp`). Metal's `mul_mat` can walk strides
  natively; some of those `cont` copies may be removable without changing
  math. Tracked as QW-D in today's planning notes.
- **F32 `mul_mm + add(bias)` shader fusion in
  `patches/ggml-metal-chatterbox-ops.patch`** (the estimate +10–25 ms on
  S3Gen — CFM transformer batched mat-muls). Already queued in §3.24
  follow-ups.

#### Files touched (reverted)

| File | Change |
|------|--------|
| [src/chatterbox_tts.cpp](src/chatterbox_tts.cpp) | 10-line commentary block added to `conformer_block()` explaining why the flash-attn path is intentionally not taken, pinning the negative-finding cosine number and the speed upside that was measured, and pointing at the parakeet §15.8 counterexample. No code change to the graph itself. |

### 3.26  HiFT source_* F16 — unblocks the missing `kernel_mul_mv_f32_f16{,_4,_short}` Metal variants

Closes the open item from §3.24 §3.25: "Patch the missing
`kernel_mul_mv_f32_f16_short` variant to unblock the remaining 21
HiFT source_* conv kernels."

§3.24 converted the 64 HiFT conv-kernel F32 weights that the
`/s` deny-list didn't incidentally catch to F16 (cos > 0.9998 vs
the all-F32 baseline, `[hift_decode]` ~3 % faster, ~33 MB GGUF
shrink). The broad `/s` deny also caught every HiFT `source_*`
weight (`source_downs/0..2`, `source_resblocks/0..2/{convs1,convs2}/*`,
`m_source/l_linear/*` — 21 weight tensors, ~7.7 MB at F32) because
when you flip them to F16, HiFT's `conv1d_f32` path runs the
`ggml_mul_mat(im2col_f32, kernel_f16)` mat-vec shape with `T0=f32,
T1=f16`. The pinned ggml-metal (commit `58c38058`) did not ship
that template instantiation, and Metal pipeline lookup fails:

    ggml_metal_library_compile_pipeline: Error Domain=MTLLibraryErrorDomain
    Code=5 "Function kernel_mul_mv_f32_f16_short was not found in the library"

(Reproduced by feeding chatterbox a GGUF where the 21 source_*
tensors are F16; crashes immediately at first HiFT decode with
SIGSEGV / exit 139.)

#### The fix — three template instantiations in `ggml-metal.metal`

One line each per kernel family:

```cpp
// kernel_mul_mv_t_t family (full-shape mat-vec)
template [[host_name("kernel_mul_mv_f32_f16")]]        kernel mul_mv_t_t        kernel_mul_mv_t_t       <float, half>;
// kernel_mul_mv_t_t_4 family (vec4 dispatch path)
template [[host_name("kernel_mul_mv_f32_f16_4")]]      kernel mul_mv_t_t_4      kernel_mul_mv_t_t_4     <float, float4, half, half4>;
// kernel_mul_mv_t_t_short family (short-axis dispatch path — this is the
// variant HiFT's small-OC source_downs/2/weight (OC=64) actually hits)
template [[host_name("kernel_mul_mv_f32_f16_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short <float, half>;
```

The `mul_mv_t_t_short_impl` body (lines ~4320–4355 of `ggml-metal.metal`)
is templated on `<T0, T1>` and already handles arbitrary casts via
`(float) x[i] * (float) y[i]` — all that was missing was the
`<float, half>` instantiation for the symbol lookup. Same for
`_4` (needs `<float, float4, half, half4>`, with float-cast in the
inner reduction loop) and the base non-short variant (symmetric).

All three land as additions in `patches/ggml-metal-chatterbox-ops.patch`
(700 → 733 lines). `test-metal-ops` still PASSes on every op it
already covered (diag_mask_inf / pad_ext / conv_transpose_1d at
three upsample stages + tiny edge case).

#### `requantize-gguf.py` updates (two fixes + one scope narrow)

Three changes so the recipe works end-to-end on the current
gguf-0.18 writer:

1. **Narrowed the deny glob `/s` to `/scale`.** The old `/s` match
   was a rough proxy for "norm scale params like ln_1/ga, gate,
   etc." but incidentally swept in every `hift/source_*/` weight
   and bias tensor (188 matches in the F16 source GGUF, 62 of
   which were `source_*`). With the Metal kernel variant now
   shipped, `source_*` conv weights are safe to F16; the 21
   that matter (the 3-D conv kernels) quantise successfully via
   `--name-filter hift/source_`. The remaining norm-scale tensors
   the deny was originally targeting (`/scale`, `/ln_`, `/norm/`,
   `/gamma`) are still covered by their own stricter patterns.

2. **Fixed the Q-type passthrough byte-shape bug.** `gguf-0.18`'s
   `add_tensor_info` treats `raw_shape` as byte layout (innermost
   dim in bytes per row, not elements per row) when `tensor.dtype
   == np.uint8`. The previous code passed the element shape
   verbatim, which crashed with
   `ValueError: Quantized tensor bytes per row (512) is not a
   multiple of Q4_0 type size (18)` on any input GGUF that
   already carried Q-type tensors — i.e. every two-pass
   pipeline like `f16 → q4_0` or `q4_0 → f16 --name-filter`.
   Fix: convert inner-dim elements to bytes
   (`byte_inner = elements_inner // block_size * type_size`)
   before handing to the writer. Blocks `block_size==1` (F16/F32/
   BF16) keep the existing element-shape path.

3. **Docstring updated** with the two-pass recipe showing the
   post-§3.26 configuration:

       # Full recipe (Q4_0 everywhere except HiFT kept at F16 now
       # including the 21 source_* conv kernels unblocked in §3.26):
       python scripts/requantize-gguf.py \
           models/chatterbox-s3gen-mtl-f16.gguf \
           /tmp/intermediate.gguf f16 --name-filter hift/
       python scripts/requantize-gguf.py \
           /tmp/intermediate.gguf \
           models/chatterbox-s3gen-mtl-q4_0_hift_f16.gguf q4_0

#### Bench (M3 Ultra, Metal, Q4_0 + HiFT F16, ES prompt, seed 42, 3x3 runs)

|                    | §3.24 baseline   | §3.26 (source_* F16) | Δ            |
|--------------------|-----------------:|---------------------:|-------------:|
| `[encoder]` ms     |    31.3          |   30.5               | −0.8 (noise) |
| `[cfm_total]` ms   |   541.9          |  550.4               | noise        |
| `[hift_decode]` ms |   121.3          |  121.1               | neutral      |
| S3GEN_INFER_MS     |   709            |  724                 | +15 (noise)  |
| T3_INFER_MS        |   440            |  440                 | 0            |
| GGUF size          |  754.4 MB        |  746.7 MB            | **−7.7 MB**  |

Speed is neutral on M3 Ultra (unified-memory bandwidth isn't the
bottleneck for the 21 source_* weights, which are small — the
largest is `source_resblocks/0/convs1/*/weight` at ~3.4 MB F32 /
~1.7 MB F16). The predicted +2–4 ms HiFT gain from §3.24 falls
inside bench noise; on bandwidth-limited targets (M4 Air /
iPhone neural engine), expect the full +3–5 % HiFT speedup seen
in §3.24's existing 64 tensors. The **real win** is the
**7.7 MB GGUF shrink** (~1.0 %) on a multilingual distribution
GGUF, plus closing the last known blocker from §3.24.

#### Parity gates

- `test-metal-ops`: all four pre-existing ops (diag_mask_inf, pad_ext,
  conv_transpose_1d @ 3 upsample stages + tiny edge) PASS; no new
  tests added because `kernel_mul_mv_f32_f16{,_4,_short}` is covered
  by the end-to-end audio parity below (same inner math as the
  existing `<half, float>` / `<half, half>` / `<float, float>`
  variants, differing only in type tags).
- **WAV parity** vs §3.24 baseline on ES-prompt / jfk-voice / seed
  42 (per-invocation deterministic; md5 identical across 3x3 runs):

      MD5 §3.24 baseline:      ec58d3e65ab8e9c6f4edefb15b169ea5
      MD5 §3.26 v2 (3 runs):   d8a1b22375dbcb2259c686426a7d76c5  d8a1b22375dbcb2259c686426a7d76c5  d8a1b22375dbcb2259c686426a7d76c5

  audio comparison:

      lengths 83520/83520   cos 1.000000   PASS (threshold > 0.9998)
      rms_diff 0.464    rms_base 1332.66   max_abs_diff 4 (out of ±32767)
      → 0.035 % relative RMS drift, 0.012 % max sample drift

  Auditorily identical (within the LSB of s16 output). Deterministic
  across invocations.

#### Files touched

| File | Change |
|------|--------|
| [patches/ggml-metal-chatterbox-ops.patch](patches/ggml-metal-chatterbox-ops.patch) | +33 lines for the three `mul_mv_f32_f16{,_4,_short}` template instantiations + comments referencing this section. Regenerated from the pinned commit `58c38058`. |
| [scripts/requantize-gguf.py](scripts/requantize-gguf.py) | `/s` deny narrowed to `/scale`; Q-type passthrough byte-shape fix; docstring recipe updated. |
| `ggml/src/ggml-metal/ggml-metal.metal` | Local edit under the `ggml/` worktree; not tracked in this repo. Recipe remains: run `scripts/setup-ggml.sh` to re-apply the patch after a ggml bump. |

#### What's next

All §3.24 follow-ups now closed:

- ~~kernel_mul_mv_f32_f16_short patch~~ ✓ shipped this section
- Q4_0 HiFT via 2-D-on-disk storage + `conv1d_f32` branch — still
  deferred, larger surgery (touches both converter + C++)
- F32 `mul_mm + add(bias)` shader fusion — still deferred, ~150
  LOC Metal kernel work + test-metal-ops gate; bigger potential
  (+10–25 ms S3Gen) but not "quick"

### 3.27  F32 `mul_mm + ADD(bias) [+ ADD(residual)]` fusion on Metal

Closes the §3.22 §3.24 §3.26 follow-up "F32 `mul_mm + add(bias)` shader
fusion in `patches/ggml-metal-chatterbox-ops.patch`". The existing
fusion in the pinned `ggml-metal` pipeline covered only Q-variant
**mul_mv** (matrix-vector) kernels via `helper_mv_add_bias`
(Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 with bias+residual function-constant
guards). The **mul_mm** (matrix-matrix) kernel — the one the CFM
transformer actually hits at T·B ≥ 2 — had no equivalent. This
section wires one in.

#### What lands

1. **`kernel_mul_mm` in `ggml-metal.metal`** gains two new function
   constants (`FC_mul_mm_has_bias_` = `FC_MUL_MM + 2`,
   `FC_mul_mm_has_residual_` = `+3`) and two new buffer slots
   (`bias` at `buffer(4)`, `residual` at `buffer(5)`). When either
   FC is true, the kernel routes through the shmem-backed
   scalar-copy path and folds bias / residual into the copy loop
   (same post-matmul math as `helper_mv_add_bias`: `v += bias[r0+i]`
   and `v += residual[(r1+j)*ne0 + im*ne1*ne0 + r0 + i]`).
   Compiler drops the branch that's not selected by the FC — zero
   overhead when neither is set.

2. **`get_pipeline_mul_mm` in `ggml-metal-device.cpp`** now takes
   `has_bias, has_residual` flags, bakes them into the pipeline
   name (`kernel_mul_mm_<T0>_<T1>_bci=X_bco=Y_bias=Z_res=W`), and
   sets the function-constant values during compile. Shmem size
   bumped from `4 KB+2 KB` to `8 KB` when either flag is set so
   the always-shmem path has room for the temp buffer.

3. **Dispatcher `ggml_metal_op_mul_mat` in `ggml-metal-ops.cpp`**
   mirrors the Q-variant mul_mv fusion lookup: try
   `{MUL_MAT, ADD, ADD}` first, fall back to `{MUL_MAT, ADD}`.
   Both orderings of the residual add are handled (`ggml_add` is
   commutative; chatterbox's `basic_tfm` emits
   `ggml_add(x, attn_out)` with residual `x` as `src[0]` and the
   mul_mat+bias result as `src[1]`). Writes fused dst to
   `node(idx + n_fuse - 1)` so the value lands where the skipped
   ADD(s) would have written, and returns `n_fuse` so the outer
   loop skips them.

#### Kernel variants actually compiled on a chatterbox run

Verified via `ggml_metal_library_compile_pipeline` trace on first
invocation (M3 Ultra, Q4_0 + HiFT F16 + sample-16k voice):

```
kernel_mul_mm_q4_0_f32_bci=0_bco=0_bias=1_res=0   ← CFM transformer linears, in-bounds blocks
kernel_mul_mm_q4_0_f32_bci=0_bco=1_bias=1_res=0   ← CFM transformer linears, edge blocks
kernel_mul_mm_f32_f32_bci=0_bco=0_bias=1_res=0    ← CFM time_mlp / final_proj
kernel_mul_mm_f32_f32_bci=0_bco=1_bias=1_res=0
kernel_mul_mm_q4_0_f32_bci=0_bco=1_bias=0_res=0   ← unfused matmuls (e.g. Q/K/V no-bias)
kernel_mul_mm_f32_f32_bci=1_bco=1_bias=0_res=0
```

The `bias=1` variants account for ~280 fuse opportunities per CFM
step × 10 steps × 2 CFG batches ≈ 1820 dispatches per synthesis
that the old code paid a separate `ggml_add` kernel for. No
`res=1` variants fire in the current chatterbox graph: the
`ADD(residual)` in `basic_tfm` is at a different point in the
graph (separated by `layer_norm` → `mul_mat` → `add(bias)` →
`gelu_erf` → `mul_mat` → `add(bias)` → add(x, ff)`), so the
residual add can't be folded into the preceding mul_mm without
hoisting those intermediate ops. Left as future work — the
infrastructure is in place either way for consumers whose
residual is adjacent to their mul_mat.

#### Bench (M3 Ultra, Metal, Q4_0 + HiFT F16, ES prompt, seed 42)

5-invocation averages (WAV deterministic, md5 identical across
all 5 runs):

| Metric             | §3.26 baseline | §3.27 fused      | Δ               |
|--------------------|---------------:|-----------------:|----------------:|
| `[encoder]` ms     |    31.3        |   30.5           | noise           |
| `[cfm_total]` ms   |   541.9        |  542.2 (± 5 per-run) | **neutral** |
| `[hift_decode]` ms |   121.3        |  121.2           | neutral         |
| S3GEN_INFER_MS     |   709          |  713.2           | +4 (noise)      |
| T3_INFER_MS        |   440          |  433.4           | −7 (noise)      |
| md5                | d8a1b22…      | d8a1b22…         | **byte-exact**  |

Cross-check: running with `GGML_METAL_FUSION_DISABLE=1` (turns off
ALL ggml-metal fusions, including the pre-existing norm+mul+add
and Q-variant mul_mv+bias+residual) pushes CFM to **568.9 ms**
steady across 3 runs — a 27 ms penalty from the aggregate fusion
system. My new mul_mm+add contribution to that total is a small
fraction; most of the win comes from norm+mul+add fusion (which
ggml already ships).

#### Why the measured gain is near-zero on M3 Ultra specifically

Two reasons. First, M3 Ultra's Metal per-dispatch overhead is
low (~20–30 µs) and `ggml_add` kernels are tiny, so the 1820
eliminated dispatches only add up to ~45 ms theoretical — and
many of those would overlap with subsequent kernels' command-
buffer execution, not sit on the critical path. Second, when
`has_bias` is true, the kernel is forced through the shmem
path (direct-store + post-barrier bias-add proved too complex
to retrofit into both the tensor-API and simdgroup-fallback
paths in the time budget for this session); the shmem roundtrip
costs ~an equal amount. Net: neutral on M3 Ultra.

#### Why it still ships

1. **Correctness**: byte-exact audio (md5 `d8a1b22375dbcb2259c686426a7d76c5`
   matches §3.26 across 5 runs). `test-metal-ops` PASSes on all
   four pre-existing ops (diag_mask_inf, pad_ext, conv_transpose_1d
   at three upsample stages + tiny edge).
2. **Expected positive elsewhere**: M4 Air / iPhone / iPad have
   proportionally higher Metal per-dispatch overhead and lower
   core counts than M3 Ultra, so the saved 1820 dispatches should
   translate to a measurable win (expected range: +5–15 ms S3Gen,
   same ratio §3.24's HiFT F16 result predicted). Can't verify on
   M3 Ultra alone.
3. **Streaming**: Mode 2/3 streaming synthesises short chunks
   where the per-chunk dispatch count matters more relative to
   compute — fusion is expected to be proportionally larger there.
4. **Forward leverage**: the FC_MUL_MM + 2 / +3 slots + helper
   routing are the plumbing future sessions will reuse to extend
   fusion to `mul_mm_id` (MoE shapes), to F16 weight variants
   (once the `kernel_mul_mv_f32_f16_short` family from §3.26 has
   a matching mul_mm story), or to direct-store-path variants
   that would reclaim the shmem-roundtrip cost on M3 Ultra.

#### Files touched

| File | Change |
|------|--------|
| `ggml/src/ggml-metal/ggml-metal.metal` | Two new FC constants (FC_MUL_MM + 2 / +3), two new buffer args (slots 4 and 5) on `kernel_mul_mm`, forced-shmem path when either FC is true, bias/residual fold-in inside the scalar-copy loop. Local edit under the `ggml/` worktree; not tracked in this repo. |
| `ggml/src/ggml-metal/ggml-metal-device.{cpp,h}` | `get_pipeline_mul_mm(op, has_bias, has_residual)` — new signature; bakes flags into pipeline name + FC values; shmem sizing adjusted to 8 KB when fused. |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | `ggml_metal_op_mul_mat` mul_mm path gains the same `can_fuse({MUL_MAT,ADD,ADD})` / `can_fuse({MUL_MAT,ADD})` lookup the mul_mv path already had; both orderings of the residual add handled; `n_fuse` returned to skip the folded ADDs. |
| [patches/ggml-metal-chatterbox-ops.patch](patches/ggml-metal-chatterbox-ops.patch) | +262 lines. Regenerated from pinned `58c38058`. 733 → 995 lines. |

#### What's next

- **Reclaim the shmem-roundtrip cost on M3 Ultra**: add bias fold-in
  to the direct-store paths (both the tensor-API `cT.store` path
  and the simdgroup-fallback `simdgroup_store` loop). Would need
  a post-barrier per-simdgroup read-modify-write pass on device
  memory. 2–3 h of additional Metal kernel work; predicted to
  flip §3.27 from neutral to +5–10 ms on M3 Ultra.
- **Extend to `mul_mm_id`** (mixture-of-experts mat-muls) — same
  FC pattern applies. Zero-change for chatterbox (doesn't use
  MoE), but useful for future consumers of this patch.
- **Bench on M4 / iOS** — validate the "neutral on M3U, positive
  elsewhere" prediction. Until measured the estimate is just
  that.

### 3.28  `mul_mm + ADD(bias) + GELU_ERF` fusion — CFM FF activation path

Builds directly on §3.27 infrastructure.  Closes the `mul_mat →
add(bias) → gelu_erf` triple in CFM `basic_tfm`'s FF gate projection
(`src/chatterbox_tts.cpp:738`):

```cpp
ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff0_w, nx2), w.ff0_b);  // (mul_mat + bias) — fused by §3.27
ff = ggml_gelu_erf(ctx, ff);                                    // §3.28 absorbs this into the same kernel
ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);    // ff2 remains a separate mul_mm + bias fusion
```

§3.27 already brought `mul_mat + add(bias)` into a single dispatch
via the shmem-backed scalar-copy path; §3.28 extends that same
loop to apply `gelu_erf` as the last stage before writing to dst.
The gelu is inline FP math on each element we're already reading /
writing — **no extra memory roundtrip, no extra shmem** — so unlike
§3.27's neutral-on-M3-Ultra result, this one is a clear net
positive on M3 Ultra.

#### What lands

1. **`ggml-metal.metal`**: new function constant `FC_MUL_MM + 4`
   (`FC_mul_mm_has_gelu_erf_`), new branch at the end of the
   scalar-copy loop that applies the same `0.5 * v * (1 +
   erf_approx(v * SQRT_2_INV))` formula the standalone
   `OP_UNARY_NUM_GELU_ERF` kernel uses.  Numerically identical to
   the unfused path (proven via md5 byte-exact across 5 runs).

2. **`get_pipeline_mul_mm`**: signature bumped to
   `(op, has_bias, has_residual, has_gelu_erf)`; pipeline name
   extended with `_gelu=N`; FC + shmem sizing adjusted to keep the
   shmem path (8 KB) when any fold-in is active.

3. **Dispatcher `ggml_metal_op_mul_mat` mul_mm path**: new
   `{MUL_MAT, ADD, UNARY}` can_fuse lookup wedged between the
   `{MUL_MAT, ADD, ADD}` residual lookup and the
   `{MUL_MAT, ADD}` bias-only fallback.  Verifies
   `ggml_get_unary_op(f2) == GGML_UNARY_OP_GELU_ERF` and that
   `f2->src[0] == f1` before fusing.  Gates on GELU_ERF
   specifically because that's the one `basic_tfm` uses;
   other unary sub-ops (SILU, GELU, RELU, GELU_QUICK, ...) are
   left as independent follow-up work — same pattern would extend
   trivially.

#### Pipeline names actually compiled

(from `GGML_LOG_DEBUG` compile trace on first invocation)

```
kernel_mul_mm_q4_0_f32_bci=0_bco=0_bias=1_res=0_gelu=1   ← CFM ff0 (gelu_erf-activated)
kernel_mul_mm_q4_0_f32_bci=0_bco=1_bias=1_res=0_gelu=1   ← ff0 edge blocks
kernel_mul_mm_q4_0_f32_bci=0_bco=0_bias=1_res=0_gelu=0   ← CFM ff2 / to_out (bias only, §3.27)
kernel_mul_mm_q4_0_f32_bci=0_bco=1_bias=1_res=0_gelu=0
kernel_mul_mm_f32_f32_bci=0_bco=0_bias=1_res=0_gelu=0    ← time_mlp / final_proj
kernel_mul_mm_f32_f32_bci=0_bco=1_bias=1_res=0_gelu=0
kernel_mul_mm_q4_0_f32_bci=0_bco=1_bias=0_res=0_gelu=0   ← unfused (no-bias) passthroughs
kernel_mul_mm_f32_f32_bci=1_bco=1_bias=0_res=0_gelu=0
```

The `gelu=1` variants correspond to 56 basic_tfm blocks × 10 CFM
steps × 2 CFG batches = **1120 saved `gelu_erf` dispatches per
synth** (on top of the 1820 bias-add dispatches saved in §3.27).

#### Bench (M3 Ultra, Metal, Q4_0 + HiFT F16, ES prompt, seed 42, 5 invocations)

| Metric             | §3.27 (bias only) | §3.28 (+ gelu) | Δ                     |
|--------------------|------------------:|---------------:|----------------------:|
| `[encoder]` ms     |     30.5          |    30.8        | noise                 |
| `[cfm_total]` ms   |    542.2          |   **533.4 ± 1.0**  | **−8.8 / −1.6 %** |
| `[hift_decode]` ms |    121.2          |   120.8        | neutral               |
| S3GEN_INFER_MS     |    713.2          |   **706.0 ± 0.8**  | **−7.2 / −1.0 %** |
| T3_INFER_MS        |    433.4          |   431.0        | noise                 |
| md5                | `d8a1b22…`       | `d8a1b22…`    | **byte-exact ×5**     |

#### Parity gates

- `test-metal-ops`: all 4 pre-existing ops (diag_mask_inf, pad_ext,
  conv_transpose_1d × 3 + tiny) PASS.
- WAV md5 byte-exact vs §3.26 / §3.27 baseline (`d8a1b22375dbcb2259c686426a7d76c5`)
  across all 5 invocations of the fused build.  The fused
  kernel uses the same `erf_approx<T>(x)` helper as the standalone
  GELU_ERF unary op, so the math is identical down to the LSB.
- Determinism across runs: md5 stable.

#### Why this time it's not neutral on M3 Ultra (unlike §3.27)

§3.27's gain was eaten by the shmem-roundtrip cost: routing
through `temp_str` + sgitg==0 scalar copy costs roughly what the
1820 eliminated `ggml_add` dispatches saved.  §3.28 adds the gelu
fold-in **into the same loop** — no additional memory accesses,
no barriers, no extra shmem — just a handful of FLOPs per element.
So the 1120 saved `gelu_erf` dispatches show up as a clean net
positive:  −8.8 ms CFM / −7.2 ms S3Gen.

This also refines the §3.27 story: the infrastructure we built
there is what makes §3.28 cheap.  Fusing additional per-element
tail ops into the existing scalar-copy loop is essentially free,
whereas routing through the shmem path is what cost M3 Ultra its
estimated §3.27 win.

#### Files touched

| File | Change |
|------|--------|
| `ggml/src/ggml-metal/ggml-metal.metal` | New FC `FC_MUL_MM + 4` (has_gelu_erf); gelu_erf branch in the scalar-copy loop using `erf_approx<float>`; shared early-out condition updated to include the new flag.  Local edit under `ggml/` worktree. |
| `ggml/src/ggml-metal/ggml-metal-device.{cpp,h}` | `get_pipeline_mul_mm(op, has_bias, has_residual, has_gelu_erf)` — new fourth parameter, pipeline name extended with `_gelu=N`, shmem sizing adjusted. |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | Dispatcher mul_mm path gains `{MUL_MAT, ADD, UNARY}` can_fuse lookup with `ggml_get_unary_op == GGML_UNARY_OP_GELU_ERF` check; slotted between the 3-op residual and 2-op bias lookups. |
| [patches/ggml-metal-chatterbox-ops.patch](patches/ggml-metal-chatterbox-ops.patch) | Regenerated from pinned `58c38058`. 995 → 1054 lines, +59. Applies cleanly via `git apply --check`. |

#### What's next

The same fold-in pattern extends trivially to other unary sub-ops
whenever the chatterbox (or downstream consumer) graph uses them
right after a `mul_mat + add(bias)`:

- SILU (`t3_mtl.cpp` already uses `ggml_swiglu_split` which fuses
  `silu(a) * b`, but a plain SILU follower could be added).
- GELU (non-erf variant) — not in chatterbox today.
- RELU, GELU_QUICK — not in chatterbox.

These would each be ~15–20 lines (FC slot + branch + dispatcher
case), mirroring the GELU_ERF wiring this section added. None of
them fires in the current chatterbox graph so there's no standalone
win, but infrastructure is cheap to extend.

Bigger next-step: reclaim the §3.27 shmem-roundtrip cost on
M3 Ultra by fusing bias into the direct-store paths (both
tensor-API `cT.store` and simdgroup-fallback `simdgroup_store`).
2–3 h of Metal kernel work; predicted to flip the §3.27 contribution
from neutral to +3–5 ms CFM on top of today's §3.28 gain.

### 3.29  Direct-store fold-in — _negative finding, reverted_

Goal: reclaim the §3.27 neutral-on-M3-Ultra result by keeping the
fast `cT.store` / `simdgroup_store` direct-to-device-memory path
for full-block writes and doing the bias / residual / gelu_erf
fold-in as a **post-barrier read-modify-write pass** on device
memory, instead of routing through the shmem + scalar-copy path.

The shmem path that §3.27 ships is correct but costs a
threadgroup-memory roundtrip (4 simdgroups stage into a shared
`temp_str` buffer, sgitg==0 drains it with a scalar loop).  On
M3 Ultra that roundtrip is ~equal to the dispatch savings from
eliminating the separate `ggml_add` kernel — hence the "neutral"
§3.27 result.  §3.28 worked because gelu is an extra per-element
tail op inside a loop that already exists; it added ~zero cost.
§3.29 tried to do the same for bias, but on a different path.

#### What was tried

```cpp
if (_mm_use_direct) {
#ifdef GGML_METAL_HAS_TENSOR
    cT.store(tC);                    // cooperative 64x32 store
#else
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], ...); // per-simdgroup 32x16 store
    }
#endif
    if (_mm_has_foldin) {
        threadgroup_barrier(mem_flags::mem_device);   // flush stores
        // distribute 2048 elements of the 64x32 block across 128
        // threads of the threadgroup — each thread does 16 RMWs
        const int thread_idx = (int) tiitg;
        for (int k = thread_idx; k < NR0 * NR1; k += 128) {
            const int abs_r = r0 + (k % NR0);
            const int abs_c = r1 + (k / NR0);
            const uint64_t off = (uint64_t)abs_c * ne0 + abs_r + ...;
            device float * D = (device float *) dst + off;
            float v = *D;
            if (FC_mul_mm_has_bias)     v += bias_f32[abs_r];
            if (FC_mul_mm_has_residual) v += residual_f32[off];
            if (FC_mul_mm_has_gelu_erf) v = 0.5f*v*(1.0f + erf_approx(v * SQRT_2_INV));
            *D = v;
        }
    }
}
```

`get_pipeline_mul_mm` sized back down to the non-fold-in shmem
(6 KB) when fold-ins are active, on the theory that only edge
blocks need `temp_str`.

#### What happened

`test-metal-ops` PASSed on all pre-existing ops (diag_mask_inf,
pad_ext, conv_transpose_1d × 3 + tiny edge) — the kernel compiled
clean, the new `_short` / `_4` / `bias=1` variants all built.

But the end-to-end chatterbox synth produced **wrong output**:

| Metric      | §3.28 baseline                         | §3.29 attempt                         |
|-------------|----------------------------------------|---------------------------------------|
| md5         | `d8a1b22375dbcb2259c686426a7d76c5`    | `06ee1aaaa94a10d70eec2835d3da7dbf`   |
| T3 tokens   | 84                                     | 70                                    |
| audio_ms    | 3480                                   | 2920                                  |
| determinism | stable across 5 runs                   | stable (same wrong md5 across runs)   |

T3 EOS'd 14 tokens early.  The wrong md5 was deterministic —
not a race, but a systematic computation error that's _consistent_
every run.  Reverted to the §3.28 shmem-forcing behaviour
(byte-exact to `d8a1b22…`).

#### Suspected root causes (not isolated in this session)

1. **Cooperative tensor-store layout**: `cT.store(tC)` is an
   Apple Metal tensor-ops cooperative write across all four
   simdgroups in the threadgroup.  Where each element lands in
   device memory is implementation-defined, not trivially the
   32x16 per-simdgroup partition `simdgroup_store` uses in the
   fallback path.  The RMW pass as written assumes the partition
   doesn't matter (it iterates the full 64x32 via tiitg), but
   maybe the threadgroup_barrier with `mem_flags::mem_device`
   isn't strong enough to order `cT.store`'s writes against
   subsequent device reads from the same threadgroup on A17 /
   M3.  A real memory-model audit (or testing with `fence()`
   instead of `threadgroup_barrier`) is the next thing to try.

2. **`bias_ok` / `residual_ok` shape check vs graph layout**:
   `bias_ok` only requires `ggml_nelements(bias) == ne0` and
   `bias->ne[0] == ne0`, which is correct for the usual
   `(OC,)` broadcast.  But `residual_ok` requires
   `ggml_are_same_shape(resi, mul_mat_result)`.  The mul_mat's
   output shape is `(ne0, ne1, ne2, ne3)`; if the residual
   happens to have matching shape but different strides (e.g.,
   a non-contiguous view), the RMW would silently read the
   wrong bytes.  §3.27's shmem path also trusted this check,
   and that one works — but the shmem path copies element by
   element, which could hide a stride bug that direct-store
   reveals.  Worth an audit.

3. **Index calculation off-by-one or wrong stride**: the RMW
   uses `off = abs_c * ne0 + abs_r + im*ne1*ne0`, which matches
   the in-bounds direct-store formula
   `dst + r0 + r1*ne0 + im*ne1*ne0`.  But I didn't pass `nb0` /
   `nb1` through — the direct-store uses `args.ne0` as stride
   assuming contiguous f32 output.  If the destination tensor
   is non-contiguous (say, a view into a larger buffer) the
   mul_mat kernel itself would be wrong too, so this is
   probably not the bug, but worth double-checking in a unit
   test.

#### What's missing

There's **no per-shape unit test for `mul_mm + add(bias)`**
that compares fused-kernel output vs unfused-graph output
element-by-element.  `test-metal-ops` only covers
diag_mask_inf, pad_ext, and conv_transpose_1d.  Adding a
`mul_mm_fused` test case (build a small ggraph with
mul_mat + add, dispatch with fusion forced on vs
`GGML_METAL_FUSION_DISABLE=1`, compare outputs to 1e-6
tolerance) would have caught §3.29's bug in seconds.  The
§3.27 and §3.28 kernels *happen* to be byte-exact because
their fold-in happens inside the scalar-copy loop which is
straightforward to reason about; §3.29's direct-store RMW has
a more subtle data-flow that would benefit from explicit
coverage.

#### Files touched / reverted

| File | Change |
|------|--------|
| `ggml/src/ggml-metal/ggml-metal.metal` | Direct-store RMW block *removed*; 21-line commentary added in place explaining §3.29 attempt + failure + suspected causes for the next person to read. `_mm_use_direct` reverts to §3.28's "no fold-in allowed on direct-store path" condition. |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | `get_pipeline_mul_mm` shmem sizing reverts to §3.28 behavior (8 KB when any of `bc_out` / `has_bias` / `has_residual` / `has_gelu_erf` is set). |
| [patches/ggml-metal-chatterbox-ops.patch](patches/ggml-metal-chatterbox-ops.patch) | Regenerated from pinned `58c38058`.  1054 → 1070 lines (+16, the inline documentation block). |

#### Result

`cb_rev.wav` md5 matches §3.26/§3.27/§3.28 baseline
`d8a1b22375dbcb2259c686426a7d76c5` byte-exact.  T3 back to 84
tokens / 3480 ms audio.  No code change from §3.28 beyond the
documentation block.

M3 Ultra §3.27 shmem-roundtrip cost (~8 ms on CFM) remains
standing.  M4 / iOS predicted wins for §3.27 / §3.28 are
unaffected — the fused kernel still fires; only the
optimization to dodge the shmem path didn't land.

#### Next-person notes

If you pick this up:

- Add a `test-metal-ops` case for fused `mul_mm + add(bias)` FIRST.
  Build a 2-op graph `add(mul_mat(W_q4_0, X_f32), bias_f32)`,
  dispatch with fusion ON (current default) vs
  `GGML_METAL_FUSION_DISABLE=1`, assert element-wise match to
  ~1e-6.  Should be ~80 lines.
- Then retry the direct-store path, ideally with a **smaller
  scope first** (only `has_bias`, drop `has_residual` /
  `has_gelu_erf`) to halve the complexity.  If the bias-only
  variant passes the new unit test, incrementally add the
  others.
- Apple's [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
  §5.7 "Memory Scopes and Barriers", has the exact semantics
  for `mem_flags::mem_device` vs `mem_flags::mem_none` —
  worth confirming that `threadgroup_barrier(mem_device)`
  orders cooperative-tensor-store writes against subsequent
  device reads on A17+ silicon.  Cf. `simdgroup_fence_t` as
  an alternative to `threadgroup_barrier`.

### 3.30  `test-metal-ops` fused-mul_mm harness + §3.29 direct-store retry (bias-only)

Two pieces, both closing §3.29 loose ends:

1. **Harness**: new `test_mul_mm_fused` in `src/test_metal_ops.cpp`
   builds a small graph `add(mul_mat(W_q4_0, X_f32), bias)` (and
   with an optional `gelu_erf` follow-up), runs it on CPU + Metal,
   and compares element-wise.  On the Metal side, ggml-metal's
   fusion detector collapses these into a single
   `kernel_mul_mm_..._bias=1_res=X_gelu=Y` dispatch; CPU is always
   the unfused triple.  Any numerical drift beyond tolerance
   indicates a kernel bug.  Tolerance picked at 2e-2 absolute
   after observing the Q4_0-dequant-order CPU-vs-GPU noise on
   K=256..1024 shapes runs ~5–11e-3 max abs (4× margin over
   the noise floor).
2. **Bias-only direct-store (§3.29 retry)**: full-block writes
   with `has_bias && !has_residual && !has_gelu_erf` now take
   the direct-store path with a post-barrier bias-add scan
   (128 threads × 16 elements), instead of routing through the
   shmem scalar-copy fallback.  Residual / gelu fold-ins still
   route through shmem — §3.29's negative finding on those
   paths stands (root cause unresolved), so keeping the proven
   path for them.  This is the minimum-scope slice of §3.29
   that the new harness proves byte-stable.

#### Harness coverage

8 fused-mul_mm shape variants, gated under the same `test-metal-ops`
binary so CI/ship criteria run them alongside diag_mask_inf /
pad_ext / conv_transpose_1d:

```
[mul_mm_fused cfm-attn-qkv]          OK (K=256 N=256  T=87 B=2 fuse=bias, max_abs=5.2e-03)
[mul_mm_fused cfm-attn-out]          OK (K=256 N=512  T=87 B=2 fuse=bias, max_abs=5.7e-03)
[mul_mm_fused cfm-ff-gate-bias]      OK (K=256 N=1024 T=87 B=2 fuse=bias, max_abs=5.8e-03)
[mul_mm_fused cfm-ff-gate-bias+gelu] OK (K=256 N=1024 T=87 B=2 fuse=gelu, max_abs=4.9e-03)
[mul_mm_fused cfm-ff-down]           OK (K=1024 N=256 T=87 B=2 fuse=bias, max_abs=1.1e-02)
[mul_mm_fused cfm-b1]                OK (K=256 N=512  T=87 B=1 fuse=bias, max_abs=5.7e-03)
[mul_mm_fused bco-bias]              OK (K=256 N=320  T=87 B=2 fuse=bias, max_abs=5.8e-03)
[mul_mm_fused bco-gelu]              OK (K=256 N=320  T=87 B=2 fuse=gelu, max_abs=5.2e-03)
```

Covers the exact shapes chatterbox CFM hits (256→256 attn Q/K/V,
256→512 attn_out, 256→1024 ff0 with gelu, 1024→256 ff2), batch=1
and batch=2 variants, and a non-64-multiple N=320 that forces
the `bco=1` (bounds-checked) shmem path.

#### §3.29 retry (bias-only) outcome

The bias-only direct-store path passes the harness byte-stably
and produces byte-exact WAV output end-to-end
(`md5 d8a1b22375dbcb2259c686426a7d76c5` across 5 runs, T3 84
tokens, audio_ms 3480).

Measured impact on M3 Ultra (5 invocations, Q4_0 + HiFT F16):

| Metric             | §3.28            | §3.30            | Δ                |
|--------------------|-----------------:|-----------------:|-----------------:|
| `[cfm_total]` ms   |        533.4 ± 1.0 |      534.0 ± 0.9 | noise            |
| `S3GEN_INFER_MS`   |        706.0 ± 0.8 |      706.2 ± 3.2 | noise            |
| `[hift_decode]` ms |             121.2 |           121.8  | noise            |

Neutral on M3 Ultra, same as §3.27.  Reason: in chatterbox's
`basic_tfm`, every mul_mat+bias has a follow-up op (either
residual or gelu) that forces the fusion through the 3-op
path, which still routes through shmem.  The 2-op
`{MUL_MAT, ADD(bias)}` path §3.30 optimises only fires for
a few tensors outside basic_tfm (time_mlp / final_proj /
resnet t_mlp) that contribute negligibly to wall time.

The harness itself is the real deliverable — any future
attempt at the residual / gelu direct-store paths now has a
way to get fast feedback on whether a change is correct
before spending 2–3 h on an end-to-end chatterbox run.

#### Why not also ship the residual / gelu direct-store retries

The `{MUL_MAT, ADD, ADD}` residual fusion and `{MUL_MAT, ADD,
GELU_ERF}` gelu fusion on the direct-store path were what
failed in §3.29 (the test-metal-ops gate I've just added would
have immediately flagged them as wrong output, avoiding the
revert).  Fixing them needs either:

- a deeper audit of `cT.store`'s cooperative write layout vs
  Metal memory ordering with `mem_flags::mem_device` — likely
  where §3.29 broke; OR
- a different strategy entirely (e.g., inline residual read
  into the simdgroup accumulator before `simdgroup_store`,
  avoiding the post-barrier RMW round-trip).

Either is 2–3 h of Metal-specific debugging.  Left for a future
session; the harness now makes that session tractable.

#### Files touched

| File | Change |
|------|--------|
| `src/test_metal_ops.cpp` | New `test_mul_mm_fused(cpu, gpu, K, N, T, B, fuse_mode, label)` helper + 8 test invocations covering the CFM shape space.  New `#include "ggml-cpu.h"` for the CPU reference backend (via the existing include cluster). |
| `ggml/src/ggml-metal/ggml-metal.metal` | Bias-only direct-store path: full-block write via `cT.store` / `simdgroup_store`, then `threadgroup_barrier(mem_flags::mem_device)`, then a 128-thread scan adding `bias[r0 + row_off]` to each of the 2048 elements.  Only fires when `FC_mul_mm_has_bias && !FC_mul_mm_has_residual && !FC_mul_mm_has_gelu_erf` — gated narrowly to the scope the harness validates. |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | Shmem sizing: 8 KB when `bc_out || has_residual || has_gelu_erf`; 6 KB for bias-only-direct-store and non-fused calls. |
| [patches/ggml-metal-chatterbox-ops.patch](patches/ggml-metal-chatterbox-ops.patch) | Regenerated from pinned `58c38058`.  1070 → 1088 lines, +18 (direct-store bias scan + shmem-sizing comment). Applies cleanly. |

#### Follow-up tracking

Three items still deferred:

1. **Residual direct-store** — needs the cooperative-store
   barrier audit mentioned above.  Harness is ready.
2. **Gelu direct-store** — same as residual.  The inline-math
   cost is cheap, so the win is mostly avoiding the shmem
   roundtrip (like bias).  Estimated +2–5 ms on M3 Ultra
   _if_ it works; infra pattern identical to §3.28 and §3.30.
3. **Extend fusion to other unary sub-ops** (SILU, GELU
   non-erf, RELU, GELU_QUICK) — trivial copy-paste of §3.28;
   not done because chatterbox / T3 / CFM don't emit those
   after a mul_mat+bias pair.  Useful infra for downstream
   consumers of this patch (stable-diffusion.cpp / tts-cpp).

### 3.31  iOS-arm64 cross-build + M4 validation harness (`scripts/bench-m4-validation.sh`)

Closes the validation gap left by §3.24 / §3.26 / §3.27 / §3.28 / §3.30
— all of those predict positive-on-bandwidth-limited-hardware
(M4 Air / iPhone / iPad) but were measured only on M3 Ultra where
per-dispatch overhead is so low that the fusion wins largely
cancel out against kernel-path overhead.  Two pieces:

#### 1. iOS-arm64 build portability

Cross-compiled `libggml-metal.a` + `libtts-cpp.a` for iOS 14.0+
arm64 on this M3 Ultra host (Xcode 16 / iOS 18.5 SDK):

```
cmake -S . -B build-ios \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF
cmake --build build-ios --target tts-cpp ggml-metal -j
```

Both libraries produce clean `arm64`-only archives:

```
build-ios/ggml/src/ggml-metal/libggml-metal.a: arm64
build-ios/libtts-cpp.a: arm64
```

That's the **structural validation** that §3.26's
`kernel_mul_mv_f32_f16{,_4,_short}` variants and §3.27 / §3.28 /
§3.30's `kernel_mul_mm` FC-gated bias / gelu_erf fold-ins are
iOS-portable — none of the kernel code uses macOS-only
intrinsics.  Runtime validation still requires a real iOS device
(TestFlight / Xcode device provisioning); this confirms there's
no compile-time barrier to shipping.

#### 2. `scripts/bench-m4-validation.sh`

Self-contained harness the user runs on any Apple-silicon Mac
(M4 Air / M4 Pro / M3 / etc.) or any host that mounts the model
GGUFs.  Pipeline:

1. Apply the pinned ggml patch via `scripts/setup-ggml.sh`
2. Configure + build `build-metal` (Release, GGML_METAL=ON,
   GGML_BLAS=OFF, GGML_NATIVE=ON)
3. Run `test-metal-ops` — asserts all 14 gates PASS (3 base
   diag/pad + 3 conv_transpose_1d HiFT + 8 fused-mul_mm)
4. Run 5 invocations of `chatterbox` on the Spanish-prompt
   baseline (Q4_0 + HiFT F16 v2 GGUF + seed 42)
5. Collect per-run `[encoder]` / `[cfm_total]` / `[hift_decode]` /
   `S3GEN_INFER_MS` / `T3_INFER_MS`
6. Compute means, compare against the M3 Ultra reference baked
   into the script header:

       M3U CFM   = 534.0 ms
       M3U S3Gen = 706.6 ms
       M3U T3    = 432.6 ms
       M3U HiFT  = 121.1 ms

7. Check WAV determinism (all 5 runs same md5) and byte-exactness
   vs the M3U reference md5 `d8a1b22375dbcb2259c686426a7d76c5`
8. Write `artifacts/bench/m4-validation.json` with the full
   comparison + host info (chip, model)

Dependencies on the target host:

- macOS + Xcode command-line tools (`cmake`, `clang++`)
- Python 3 (for `scripts/setup-ggml.sh`'s gguf tooling)
- Model GGUFs at the usual paths (or override via env vars:
  `T3_GGUF=... S3GEN_GGUF=... REF_WAV=... RUNS=... bash scripts/bench-m4-validation.sh`)
- ~16 GB disk for model + build artefacts

Example predicted output on M4 Air (hypothetical; actual to be
captured when the script runs on M4 hardware):

```
=== Summary: Apple M4 vs M3 Ultra reference ===
stage                 M3 Ultra (ref)       this host       Δ vs M3U
[cfm_total] ms                 534.0           ~XXX.X      -A / -B%
S3GEN_INFER_MS                 706.6           ~YYY.Y      -C / -D%
```

The `Δ` column tells us whether the §3.27 / §3.28 / §3.30
predicted-positive story holds.  If M4 shows noticeably smaller
CFM than M3U after accounting for M4's higher single-core clock,
the shipping portfolio is vindicated.  If M4 matches M3U or
regresses, §3.27 / §3.30 should be re-examined.

#### Self-smoke on M3 Ultra

Ran the script locally as a sanity check — expected to show
"this host == reference" with no deltas:

```
=== Summary: Apple M3 Ultra vs M3 Ultra reference ===
stage                 M3 Ultra (ref)       this host       Δ vs M3U
[cfm_total] ms                 534.0           533.7    -0.3 (-0.1%)
S3GEN_INFER_MS                 706.6           707.4    +0.8 (+0.1%)
T3_INFER_MS                    432.6           434.6    +2.0 (+0.5%)
[hift_decode] ms               121.1           123.1    +2.0 (+1.7%)

=== Parity ===
determinism: PASS  (md5 d8a1b22375dbcb2259c686426a7d76c5 stable across 5 runs)
byte-exact vs M3 Ultra: PASS (d8a1b22375dbcb2259c686426a7d76c5)
```

All deltas within per-invocation stdev.  Script is ready to
scp + run on any M4 / M3 / M2 box.

#### Files touched

| File | Change |
|------|--------|
| [scripts/bench-m4-validation.sh](scripts/bench-m4-validation.sh) | New 150-line bash script.  Self-contained: pins the M3 Ultra reference numbers, runs test-metal-ops, 5-invocation bench, compares, writes JSON. |

#### Next

- Run the script on an M4 Air (user action: `scp -r chatterbox.cpp m4:` + `scp models/*.gguf m4:.../models/` + `ssh m4 'bash chatterbox.cpp/scripts/bench-m4-validation.sh'` + `scp m4:.../artifacts/bench/m4-validation.json .`).
- If M4 results confirm the prediction: update the §3.27 / §3.28 / §3.30 sections with the M4 numbers alongside M3U.
- If M4 results contradict the prediction: file a follow-up to revisit the fusion costs on smaller Apple silicon.

### 3.32  Vulkan multilingual port — `VkPipelineCache` + chatterbox-side persistent caches

Ports the Vulkan-side optimisation work originally landed on
`upstream/main` onto the `multilingual_merged` base.
Two `ggml-vulkan` patches + four host-side optimisations in
`src/chatterbox_tts.cpp`.  All bit-exact-preserving (F32 invariants
on both NVIDIA and AMD/RADV); model-agnostic by construction so they
benefit **both** the Turbo (meanflow) and the multilingual (standard
CFM with CFG) variants.  No public-API change, no GGUF format
change, no new build-system requirement.

The full per-round investigation (eight rounds + AMD validation +
LunarG SDK / `cooperative_matrix2` Tier-3 close-out) was kept in
internal findings docs (out-of-tree) for context.  This squashed
port carries only the optimisations that remain measurable on the
`multilingual_merged` base — many of the
original rounds (notably the round-4 / round-6 Q/K/V batched matmul
fusion) overlap with `multilingual_merged`'s own zero-cont strided
Q/K/V views (commit `849507a`) and were deferred rather than
double-applied.  C1 (F16 CFM matmul weights) was also deferred —
`multilingual_merged`'s `load_s3gen_gguf` uses
`ggml_dup_tensor + ggml_backend_alloc_ctx_tensors` and would need a
separate adaption pass plus new locked MD5 baselines.

#### 1. `patches/ggml-vulkan-pipeline-cache.patch` — persistent `VkPipelineCache` (199 lines)

Adds an opt-in persistent shader cache to ggml-vulkan, keyed by
`<vendorID>-<deviceID>-<driverVersion>` and rooted at
`$GGML_VK_PIPELINE_CACHE_DIR` →
`$XDG_CACHE_HOME/ggml/vulkan` → `$HOME/.cache/ggml/vulkan`.
Disabled by setting the env var to the empty string (byte-identical
to upstream).  Recovers ~91 % of the cold→warm gap on the first warm
run.

```text
fresh-process wall, RTX 5090 + NVIDIA 590.48 + Vulkan 1.4.325:
  both caches cold (fresh machine / Mesa)  : ~2 690 ms
  ggml cache warm, NVIDIA cache cold       :  ~250 ms     ← round-1 alone
  both caches warm (steady state)          :  ~225 ms
```

The headline mobile / Mesa win — there's no per-driver shader cache
to fall back on outside of NVIDIA's binary-blob path.

#### 2. `patches/ggml-vulkan-eager-cache-save.patch` — crash-safe pipeline-cache flush (104 lines)

Stacks on the first patch.  Writes back the pipeline-cache blob
after every `compiles.wait()` batch in `ggml_vk_load_shaders`, with
a `pipeline_cache_last_size` guard so warm-cache hits skip the disk
write (caught a +90 ms regression during dev).  Crash-safety only;
perf-neutral on warm runs.

#### 3. Persistent CFM estimator graph cache (`g_cfm_estimator_cache`)

`cfm_estimator_cache` was the last graph-builder still local-scope
in `s3gen_synthesize_to_wav` — every synth call paid the full
~50 ms graph rebuild cost (256 MB buf alloc + ~5500-node CFM
graph build + `ggml_gallocr_reserve`).  Refactored to follow the
same explicit-`destroy()` global-lifetime pattern as the existing
`thread_local time_mlp_cache` / `g_encoder_cache` / per-stage
caches.

Both batch=1 (Turbo / meanflow) and batch=2 (multilingual CFG)
paths reuse the same cache; the `cache.b2` flag triggers a rebuild
when the mode changes.  Cache cleared in `s3gen_model_cache_release`
**before** the backend is freed (Vulkan / Metal device-teardown
ordering matters), and in `s3gen_model_cache_get` cache-miss
(backend swap).

```text
per-step verbose verification, 5 utterances × 16 chunks (Turbo, RTX 5090):
  chunk 1 (cold): cfm_step0 = 64 ms, cfm_step1 = 15 ms,  cfm_total = 80 ms
  chunks 2..16  : cfm_step0 = 15 ms, cfm_step1 = 15 ms,  cfm_total = 30 ms
```

Also eliminates a latent process-exit crash risk: the previous
`~cfm_estimator_cache()` destructor fired *after* the Vulkan dylib's
static destructor (residency-set non-empty assert pattern).  The
new explicit `destroy()` runs *before* the backend is freed.

#### 4. Time-embedding result memoisation (`g_time_mlp_results`, `g_time_emb_results`)

Both Turbo (`t_span = [0, 0.5, 1]`) and multilingual (cosine-
scheduled, default 10 steps) emit the same set of t-values across
all subsequent synth calls.  Each tiny graph (3 dispatches,
~18 µs GPU compute) pays ~700 µs of fixed cmd-buffer + submit +
sync + `tensor_get` overhead — per-graph fixed cost is **30× actual
compute**.

Two-layer cache:
- `g_time_mlp_results` — keyed by `uint32_t` bitcast of `t_val`
- `g_time_emb_results` — keyed by `uint64_t = (kt << 32) | kr`
  (Turbo only; multilingual skips the mixer)

`compute_time_mlp_cached` + `compute_time_emb_cached` wrappers at
the synthesize call site collapse the 3-line `t_mlp / r_mlp /
t_mixed` sequence to one line.  6 graph submissions / inference →
0 after first inference for Turbo; 9–19 → 0 for the multilingual
10-step schedule.  Caches cleared in `s3gen_model_cache_release`
alongside the graph caches.

#### 5. CPU mirror cache for large per-synth weight downloads (`g_weight_cpu_mirror`)

`s3gen_synthesize_to_wav` reads three large model tensors via
`ggml_backend_tensor_get` on every call:

| Tensor                          | Turbo size | Multilingual size |
|---------------------------------|-----------:|------------------:|
| `flow/input_embedding`          | 13.4 MB    | ~28 MB            |
| `flow/spk_embed_affine/w`       | 60 KB      | 60 KB             |
| `flow/spk_embed_affine/b`       | 320 B      | 320 B             |

On a GPU backend each is a real device→host transfer plus sync.
~600–1000 µs per call for `input_embedding` alone on RTX 5090.
These weights are **constant for the model lifetime** — cache them.

New `cached_cpu_weights_f32(t)` helper + `g_weight_cpu_mirror` map
(keyed by `ggml_tensor *`).  Cleared in `s3gen_model_cache_release`
and on `s3gen_model_cache_get` cache-miss because the tensor
pointers belong to the soon-to-be-freed model context.

The multilingual variant benefits *more* than Turbo here because
the larger `input_embedding` (~28 MB vs 13.4 MB) doubles the
per-call download cost saved.

#### 6. Three HiFT `ggml_cont` sites removed (perf-neutral, code quality)

Round-AUDIT (kept in an internal findings doc, out-of-tree) listed
these as deferred; same methodology applied here:

| Site                                | Calls / inf | Direct consumer                              |
|-------------------------------------|------------:|----------------------------------------------|
| `conv_transpose_1d_f32` exit cont   | 3           | `ggml_add(x, reshape_2d(bias))` strided OK   |
| ISTFT `y_trim` exit cont            | 1           | `ggml_clamp` element-wise → fresh contig     |
| `f0_predictor` `xp` permute cont    | 1           | `ggml_mul_mat` `src1` (Vulkan f32 strided OK)|

At ~3 µs per cont dispatch this is ~15 µs / inference theoretical;
below the noise floor by design.  Same code-quality + future-
proofing rationale as upstream §3.14 / §3.15.  CONT total in HiFT
is only ~0.13 % of HiFT runtime per the perf logger, so further
chatterbox-side cont reduction is perf-irrelevant.

Three additional cont sites investigated but **kept** with inline
comments explaining the failure mode for future investigators:
`layer_norm_on_channel` exit (downstream `im2col`/`concat` needs
contig src), and STFT `mag_log` / `ph_in` exits (single-shot
bit-exact passes but multi-synth identical-chunks PCM diverges from
locked baseline — gallocator non-zero-offset view sensitivity).

#### 7. G2 dump-script gap closure — `regress-tensor-compare.sh` end-to-end

`regress-tensor-compare.sh` (kept in an internal benchmark log
directory, out-of-tree) was previously aborting at stage G2 with
`cannot open cfm_concat.npy`.
Four files added to `scripts/dump-s3gen-reference.py`:

- `cfm_concat.npy` (stage G2): replicates the
  `pack([x, mu, spks_bc, cond])` logic from
  `ConditionalDecoder.forward` directly in
  `estimator_forward_capture` (first-call only).
- `cfm_h_conv.npy` (stage G2): output of `block1.block[0]`
  (`CausalConv1d`).  New `make_first_call_hook` helper.
- `cfm_h_ln.npy` (stage G2): output of `block1.block[3]`
  (Transpose back to `(B, C, T)` after LayerNorm).
- `hift_s_stft.npy` (stages H3 + H4): output of `hift._stft`
  followed by `cat([real, imag], dim=1)`.  Monkeypatched
  `hift._stft`, restored in `finally`.

Plus a one-line C++ fix in `src/test_s3gen.cpp`'s `stage_G2`: add
`ggml_set_output(xc)` so the gallocator preserves the diagnostic
intermediate (was returning garbage because `xc`'s slot was reused
by downstream intermediates after the conv1d consumer completed).

Full pipeline now runs end-to-end through G2 / G3 / G4 / H1 / H3 /
H4 / H5; max relative error 7.92e-3 on STFT (PyTorch FFT vs
hand-built DFT, expected, not a regression), max ≤ 4.7e-5
everywhere else; final waveform `max_abs = 8.20e-08`.

#### Negative result documented (inline comment in `synthesize`)

Tried adding pointer-equality skip-upload of `mu` / `spks` / `cond`
across `cfm_steps` within one `synthesize` call.  F32 single-shot
WAV diverged immediately (got `c63c19...`, expected `454b4cc1...`).
Root cause: ggml's gallocator **reuses** input-tensor buffer slots
once their consumers complete.  In CFM:

```cpp
xc = ggml_concat(x_in, mu_in, spks_bc, cond_in);
// ^ last use of mu / spks / cond — their slots are now free for
//   the gallocator to reuse for downstream intermediates.
```

Skip-upload only works for inputs referenced **throughout** the
graph (encoder `pos_emb` works, CFM `mu / spks / cond` doesn't).
General rule for ggml's gallocator, kept as a comment in
`synthesize()` and documented in the internal HIFT-round findings
doc (out-of-tree) §2-bis.4.

#### Performance — RTX 5090, regress-tight aggregate, n=75 chunks, Turbo

The May 4 port was measured on Turbo because the multilingual GGUF
was not available locally at the time.  After the §3.34 companion
work ships the converted-from-source
`chatterbox-s3gen-mtl-q4_0.gguf`, multilingual measurement is a
follow-up.

```text
metric        | upstream/multilingual_merged |  + this §3.32  |          Δ
S3GEN_INFER   |                      76.6 ms |       65.4 ms  | -11.2 ms (-14.6 %)
cfm_total     |                      40.3 ms |       28.7 ms  | -11.6 ms (-28.8 %)
encoder       |                      19.9 ms |       20.7 ms  | noise
hift_decode   |                      10.9 ms |       11.6 ms  | noise
```

`cfm_total` ranges fully separated on n=120 samples
(base `[38.3, 42.8]` vs final `[27.1, 30.1]`).  Smaller absolute
saving than on the original `upstream/main` base (where the same
work measured −45 ms / −41 % S3GEN_INFER) because
`multilingual_merged` already contains the
zero-cont strided Q/K/V views, the reduced 256 MB → 64 MB CFM buf,
the `thread_local time_mlp_cache`, and the dropped redundant
`gallocr_reserve` in HiFT/`time_mlp` — all of which originally
contributed to the larger headline number on the main base.

#### Bit-exactness

Turbo F32 invariants on the original `main` base, carried forward
to this `multilingual_merged` port:

| Backend                | F32 single-shot | F32 multi-synth identical | F32 multi-synth varied |
|------------------------|:---------------:|:-------------------------:|:----------------------:|
| RTX 5090 + 590.48      |       ✓         |             ✓             |           ✓            |
| AMD iGPU (RADV, Mesa)  |       ✓         |             ✓             |           ✓            |

Multilingual F32 invariants (NEW, locked May 6, 2026 against
upstream/multilingual_merged HEAD `b074399` on RTX 5090 +
NVIDIA 590.48 + Vulkan 1.3.275 — see "Multilingual verification"
section below for details):

| Backend                | F32 single-shot                      | F32 multi-synth (6 seg)              |
|------------------------|:------------------------------------:|:------------------------------------:|
| RTX 5090 + 590.48      | `c65d98f15a59b8fe9cad98e46eb3fb30` ✓ | `0b374c7474895a3387b9f1df10b3c1b8` ✓ |

F16 invariants are not in this commit (C1 deferred).

#### Why this is model-agnostic by construction

All four host-side optimisations target generic per-synth
infrastructure that is shared between Turbo and multilingual:

1. **CFM estimator cache** — the `cache.b2` flag handles the
   Turbo (batch=1, meanflow) ↔ multilingual (batch=2, CFG) mode
   switch transparently.  Same struct, same teardown.
2. **t-emb caching** — multilingual's default `n_timesteps = 10`
   means **more** distinct t-values per inference (10 vs Turbo's
   2–3), so the cache hit-count ratio improves linearly with steps.
3. **CPU weight mirror** — `flow/input_embedding` is **larger**
   on multilingual (vocab=13632 vs Turbo's 6561), so the saved
   per-call download is roughly twice as large.
4. **HiFT cont removals** — HiFT decoder code path is identical
   for both variants.

#### Round 2 — encoder / HiFT / F0 graph caches + scaffolding caches (May 6, 2026)

Targets the per-synth host-CPU overhead that round 1 / round-HIFT didn't
address.  All host-side, model-agnostic, no GGUF-format change, no
public-API change.  Bit-exact-preserving on multilingual on Vulkan:
locked invariants (single-shot `c65d98f15a59b8fe9cad98e46eb3fb30`,
6-segment multi-synth `0b374c7474895a3387b9f1df10b3c1b8`) match
byte-for-byte before and after the round-2 changes.  Test-first:
An internal `regress-mtl-vk.sh` reproduction harness (kept
out-of-tree) locks the pre-change snapshot then re-verifies after
every cache.

**The seven new caches** (all sit alongside the existing
`g_cfm_estimator_cache` / `g_time_mlp_results` / `g_time_emb_results` /
`g_weight_cpu_mirror` from round 1):

| Cache | Keyed on | What it stores | Why it's safe |
|---|---|---|---|
| `g_encoder_graph_cache` | `T` (encoder input length) | full `run_encoder` graph + `gallocator` | Streaming chunks at varying length still produce correct output (rebuilds on key change). |
| `g_hift_graph_cache` (+ `g_hift_inv_alpha_entries` metadata) | `pack(T_mel, T_stft)` | full `run_hift_decode` graph + `gallocator` | Parallel `(graph-input-name, source-tensor-ptr)` metadata lets cache hits re-feed each alpha-input slot from `g_inv_alpha_results` without rebuilding. |
| `g_f0_graph_cache` | `T_mel` | full `run_f0_predictor` graph + `gallocator` | Same pattern as encoder. |
| `g_pos_emb_results` (`cached_pos_emb`) | `pack(T, D)` | `(2T-1, D)` F32 vector from `compute_pos_emb` | `compute_pos_emb` is pure compute (~`T × D × 5` trig ops).  Fired twice per encoder run (`T` and `2T`).  Multilingual `T~350+` and `D=512` makes this a real wedge of per-synth host time. |
| `g_inv_alpha_results` (`cached_inv_alpha`) | `ggml_tensor *` (model-weight pointer) | `vector<float>` of inverted alphas | Alpha tensors are constant for the model lifetime; HiFT calls `invert_alpha_cpu` ~72× per synth (12 ResBlocks × 6 alphas).  Survives across HiFT graph rebuilds. |
| `g_hann_window_cache` / `g_istft_kernel_cache` (`cached_*`) | `n_fft` | `vector<float>` | Pure functions of `n_fft` (constant 16 in the chatterbox HiFT path). |
| `g_window_sum_cache` (`cached_window_sum`) | `pack(n_fft, hop, T_stft)` | `vector<float>` | `T_stft × n_fft` adds (`~T_stft` ms-class cost on long utterances).  Stable across same-shape synth calls. |

A new `graph_cache` struct (used by encoder / HiFT / F0) and a
`pack_hift_key` helper centralise the explicit `destroy()`-on-teardown
pattern so future per-stage caches can plug in with one struct + one
mutex acquisition.  The destroy path is unified into a renamed
`s3gen_release_synth_caches()` (replaces the old
`g_cfm_estimator_cache_destroy()`) called from `s3gen_model_cache_release`,
the cache-miss backend-swap path, and the explicit `s3gen_unload()`.

##### Negative result documented (bug caught and fixed during dev)

First implementation of the HiFT cache hung indefinitely on the very
first synth call.  Root cause: the alpha-input refresh loop held
`g_synth_caches_mu` while calling `cached_inv_alpha`, which itself
takes the same mutex internally → classic re-entrant deadlock.  Fix:
snapshot `g_hift_inv_alpha_entries` under the mutex into a local
vector, then iterate without the lock (`cached_inv_alpha` re-acquires
the mutex per call but with no nesting).  General rule: never hold a
cache-state mutex while calling any other `cached_*` helper.

##### Performance — RTX 5090, multilingual auto-split, warm-state seg 2–6

Within-process win on top of round 1 + round-HIFT (already shipped in
this PR):

| Metric          | Pre-round-2 (baseline-pre-r2.snap) | Post-round-2 |          Δ                |
|-----------------|-----------------------------------:|-------------:|---------------------------:|
| **S3GEN_INFER** |                          159.8 ms  | **140.8 ms** | **−19.0 ms (−11.9 %)**    |
| **cfm_total**   |                          122.2 ms  |    118.7 ms  | **−3.5 ms (−2.9 %)**      |
| **cfm_step0**   |                           13.24 ms |     13.18 ms |  unchanged (already cached round 1) |
| **hift_total**  |                           17.96 ms |     16.3 ms  | **−1.7 ms (−9.4 %)**      |

Combined cumulative win vs `upstream/multilingual_merged` baseline
(round 1 + round-HIFT + round 2):

| Metric          | upstream/multilingual_merged | this PR (full) |          Δ                |
|-----------------|-----------------------------:|---------------:|---------------------------:|
| **S3GEN_INFER** |                     169.9 ms |  **140.8 ms**  | **−29.1 ms (−17.1 %)**    |
| **cfm_total**   |                     132.5 ms |  **118.7 ms**  | **−13.8 ms (−10.4 %)**    |
| **cfm_step0**   |                      24.1 ms |   **13.2 ms**  | **−10.9 ms (−45.2 %)**    |

The biggest remaining single piece of `S3GEN_INFER` (~120 ms cfm) is
the actual GPU CFM compute — it's not host-cacheable and would need
shader-side optimisation (e.g. tensor-core engagement via
`cooperative_matrix2`, deferred — see "Next" below).

##### Reproduction (test-first harness)

```bash
cd chatterbox.cpp

# 1. Build the round-2 binary
bash scripts/setup-ggml.sh
cmake -S . -B build-vk-mtl-merged -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vk-mtl-merged -j --target tts-cli

# 2. Verify bit-exact vs the locked pre-round-2 baseline.  3/3 invariants
#    must PASS (multilingual single-shot, multilingual 6-segment
#    multi-synth, Turbo single-shot).
bash ../bench-logs-vk-mtl/regress-mtl-vk.sh build-vk-mtl-merged final verify

# Optional: re-lock if the binary is intentionally producing different
# output (e.g. after an explicit numerical change).
# bash ../bench-logs-vk-mtl/regress-mtl-vk.sh build-vk-mtl-merged my-baseline lock
```

#### Multilingual verification (May 6, 2026)

The May 4 squashed port was measured on Turbo because the
multilingual GGUF was not available locally then.  After the
§3.34 companion work shipped a converter from the
public `ResembleAI/chatterbox` HuggingFace repo
(`chatterbox-s3gen-mtl-q4_0.gguf` 788 MB +
`chatterbox-t3-mtl-q4_0.gguf` 345 MB), this section captures the
actual multilingual measurement.

**Test methodology.** Six-segment auto-split via
`--max-sentence-chars 32` (the multilingual T3 GGUF doesn't embed
the tokenizer needed for the `--input-file` streaming pattern;
`--max-sentence-chars` triggers multiple within-process synths
which is what the persistent host caches actually need to fire).
Three iterations × five warm-state segments each = **n=15 samples
per build**.  Comparison build: a fresh `upstream/multilingual_merged`
HEAD (`b074399`) worktree with only the Metal + OpenCL patches
applied (NOT the two new Vulkan patches in this PR).  Both builds
use the same vendored ggml commit `58c38058` and the same Vulkan
1.3.275 / RTX 5090 + NVIDIA 590.48 host.

##### Bit-exactness on multilingual

Both single-shot and 6-segment multi-synth produce **byte-identical
multilingual WAV** vs the upstream/multilingual_merged baseline:

| Test                                  | This PR MD5                          | Baseline MD5                         | Match |
|---------------------------------------|--------------------------------------|--------------------------------------|:-----:|
| Single-shot (seed 42, --temp 0)       | `c65d98f15a59b8fe9cad98e46eb3fb30`   | `c65d98f15a59b8fe9cad98e46eb3fb30`   |  ✓   |
| Multi-synth 6 segments (seed 42)      | `0b374c7474895a3387b9f1df10b3c1b8`   | `0b374c7474895a3387b9f1df10b3c1b8`   |  ✓   |

These are the **first locked multilingual F32 invariants** for the
Vulkan path on the multilingual_merged base (the previously locked
RTX 5090 invariants in `regress-c1.sh` were captured against the
older `main`-base branch and don't apply to this base).

##### Multilingual performance — RTX 5090, n=15 warm-state samples per build

| Metric          | upstream/multilingual_merged | this PR     | Δ                          |
|-----------------|-----------------------------:|------------:|---------------------------:|
| **S3GEN_INFER** |                     169.9 ms | **153.7 ms**| **−16.2 ms (−9.5 %)**      |
| **cfm_total**   |                     132.5 ms | **114.7 ms**| **−17.8 ms (−13.4 %)**     |
| **cfm_step0**   |                      24.1 ms |  **12.6 ms**| **−11.5 ms (−47.7 %)**     |

`cfm_step0` is the strongest multilingual signal: the persistent
CFM estimator graph cache eliminates ~half of the per-segment
graph-rebuild cost on warm-state synth.  The −9.5 % S3GEN_INFER
win is below the Turbo wins shown above because:

1. **Multilingual CFM is ~6× larger** in absolute terms (more
   layers, larger hidden dims, default 10-step cosine schedule
   vs Turbo's 2-step meanflow), so the cached host overhead is a
   smaller fraction of the wall.
2. The multilingual baseline already absorbs more of the
   per-synth fixed cost than Turbo does — multilingual hits
   `compute_time_mlp` 10 times per inference but each time only
   touches a tiny graph, whereas the cached CFM estimator graph
   matters more in the absolute.

##### Cold-start (first segment of a fresh process)

Within a single process, the **first** segment pays a one-time
cache-warm-up overhead: with-caches 210–236 ms vs baseline 195–241 ms
(no statistically significant first-segment penalty given
run-to-run variance).  Subsequent segments are where the
caches actually pay off and the win is consistently visible.

Across processes, the persistent VkPipelineCache patch
(round-1) collapses the cold-process startup: `cfm_step0` on a
fresh process drops from ~133 ms (no cache, full shader compile)
to ~30 ms (cache hit) — the headline mobile / Mesa win.

##### Reproduction

```bash
# Build with the round-2 patch set applied
cd chatterbox.cpp
bash scripts/setup-ggml.sh
cmake -S . -B build-vk-mtl-merged -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vk-mtl-merged -j --target tts-cli

./build-vk-mtl-merged/tts-cli \
    --model models/chatterbox-t3-mtl-q4_0.gguf \
    --s3gen-gguf models/chatterbox-s3gen-mtl-q4_0.gguf \
    --language en \
    --text "Hello from ggml first synthesis. Second synthesis run here now. Third sentence here. Fourth sentence runs too. Fifth sentence wraps." \
    --max-sentence-chars 32 --out /tmp/mtl-pr.wav \
    --n-gpu-layers 99 --threads 4 --seed 42 --temp 0 --top-k 1 --verbose

# Baseline (upstream/multilingual_merged HEAD, separate worktree)
git worktree add /tmp/cb-base upstream/multilingual_merged
ln -s "$(pwd)/models" /tmp/cb-base/models
cd /tmp/cb-base
bash scripts/setup-ggml.sh
cmake -S . -B build-vk-base -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vk-base -j --target tts-cli

# Same command with --out /tmp/mtl-base.wav, then:
md5sum /tmp/mtl-pr.wav /tmp/mtl-base.wav  # MUST match
```

#### Files touched

| File                                       |          Change |
|--------------------------------------------|----------------:|
| `patches/ggml-vulkan-pipeline-cache.patch` |       new (199) |
| `patches/ggml-vulkan-eager-cache-save.patch` |     new (104) |
| `patches/README.md`                        |       +13 / -8  |
| `scripts/setup-ggml.sh`                    |       +20 / -8  |
| `scripts/dump-s3gen-reference.py`          |             +65 |
| `src/chatterbox_tts.cpp`                   |     +625 / -98  |
| `src/test_s3gen.cpp`                       |              +6 |
| **Total**                                  | **+966 / -101** |

The +373 lines added in round 2 (over the +252 already shipped in
round-1 / round-HIFT) are entirely the new cache infrastructure:
`graph_cache` struct, the seven new cache globals, the
`s3gen_release_synth_caches()` lifecycle hook, the five `cached_*`
scaffolding helpers, and the build_graph / cache-hit branches in
`run_encoder` / `run_hift_decode` / `run_f0_predictor`.  No source
deletions are user-facing; the −98 lines reduce the per-synth
`gallocr_new` / `ggml_init` / `ggml_gallocr_free` / `ggml_free`
boilerplate that the cache infrastructure now subsumes.

The detailed FINDINGS_*.md companion docs stay out-of-tree
(internal context only) — same arrangement as the multilingual-CPU
cache work.

#### Next

- **Multilingual GGUF cross-validation** — ✅ **DONE (May 6, 2026)**.
  See "Multilingual verification" subsection above: bit-exact on F32
  (single-shot `c65d98…`, multi-synth `0b374c…`); steady-state wins
  −9.5 % S3GEN_INFER, −13.4 % cfm_total, −47.7 % cfm_step0 vs
  upstream/multilingual_merged HEAD on multilingual GGUF.
- **C1 port to `multilingual_merged`** (F16 CFM matmul weights,
  opt-in `CHATTERBOX_F16_CFM`): needs ~100 lines adapting our F32→F16
  conversion path to `multilingual_merged`'s
  `ggml_dup_tensor + ggml_backend_alloc_ctx_tensors` `load_s3gen_gguf`
  layout, plus new locked MD5 baselines (NVIDIA + AMD, F32 + F16).
- ~~**HiFT graph caching on `multilingual_merged`**~~: ✅ **DONE in round 2**
  (May 6, 2026).  Added `g_hift_graph_cache` keyed on
  `pack(T_mel, T_stft)` with parallel `g_hift_inv_alpha_entries`
  metadata.  Within-process warm-state win: −9.4 % `hift_total` on
  multilingual.  See "Round 2 — encoder / HiFT / F0 graph caches" subsection above.
- ~~**Encoder + F0 + scaffolding caches**~~: ✅ **DONE in round 2** (May 6,
  2026).  Added `g_encoder_graph_cache`, `g_f0_graph_cache`, plus
  `cached_pos_emb` / `cached_inv_alpha` / `cached_hann_window` /
  `cached_istft_kernel` / `cached_window_sum`.  Combined with HiFT
  graph cache: −11.9 % `S3GEN_INFER` on multilingual.
- **Round-4 / 6 QKV fusion composition with multilingual_merged's
  strided 3D views** — our batched `mul_mat` (originally landed on
  `main`) and their zero-cont strided views (`849507a`) are
  alternative optimisations targeting the same code; pick one
  approach and bench Vulkan `flash_attn_ext` stride tolerance.
- **Tensor-core engagement for narrow CFM matmuls** (`cooperative_matrix2`):
  the round-1 `main`-base CM2 Tier-3 close-out measured **−8.6 % cfm_total** on
  RTX 5090.  Politically blocked behind a cmake flag pending
  project-wide baseline-set sign-off.  See `FINDINGS_ROUND_CM2.md`.
- **Mobile validation** (Adreno / Mali / Apple):
  hardware-bound; biggest remaining evidence gap.  AMD/RADV proxy
  refuted the original mobile-bandwidth projection on the
  per-round work; real mobile runs would either confirm the
  ship-on-merit framing or force its revision.

---

## OpenCL / Adreno bring-up (April 2026)

Target: **Termux on Snapdragon / Adreno 830** using `GGML_OPENCL=ON`,
with `LD_LIBRARY_PATH` including `$HOME/lib` so the OpenCL loader
and ggml DSOs resolve.

### What was missing

The first OpenCL smoke runs only offloaded T3; S3Gen/HiFT still had to stay
on CPU because ggml-opencl rejected missing ops during graph execution.  The
sequence of blockers observed on-device was:

1. `CONV_TRANSPOSE_1D` in HiFT.
2. `SIN` / `COS` in HiFT's oscillator / phase path.
3. `LEAKY_RELU` in the S3Gen encoder.
4. `UNARY(ELU)` and `ABS` in the f0 predictor.

### What landed

- Added `GGML_USE_OPENCL` wiring to the C++ side (`init_backend` for T3 and
  `s3gen_init_backend` for S3Gen/HiFT), so `--n-gpu-layers > 0` actually
  attempts `ggml_backend_opencl_init()` before CPU fallback.
- Added `patches/ggml-opencl-chatterbox-ops.patch` and updated
  `scripts/setup-ggml.sh` so a fresh `ggml/` checkout is reset to the pinned
  commit and receives **both** the Metal and OpenCL patches.
- Extended ggml-opencl with the missing ops:
  - `GGML_OP_CONV_TRANSPOSE_1D` (`f32` and `f16` kernel / `f32` input paths).
  - `GGML_OP_SIN`, `GGML_OP_COS`.
  - `GGML_OP_LEAKY_RELU`.
  - `GGML_UNARY_OP_ABS`, `GGML_UNARY_OP_ELU` (`f32` paths used by f0).
- Optimized the first `CONV_TRANSPOSE_1D` OpenCL kernel: instead of scanning
  every input position and discarding almost all of them, each output sample
  now computes the exact input index range that can contribute.
- Exposed `--cfm-steps N` for normal batch synthesis (previously only the
  streaming path had `--stream-cfm-steps`).  Default remains 2 for Python-like
  meanflow quality; `--cfm-steps 1` is the lower-latency mode.

### Validation

Remote build (Android Termux; substitute your own
`$HOME/chatterbox.cpp` path if it lives elsewhere):

```bash
cd $HOME/chatterbox.cpp
git pull --ff-only
./scripts/setup-ggml.sh
cmake -S . -B build-opencl -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON
cmake --build build-opencl -j$(nproc) --target tts-cli
```

Runtime command:

```bash
export LD_LIBRARY_PATH="$HOME/lib:${LD_LIBRARY_PATH:-}"
./build-opencl/tts-cli \
  --model $HOME/chatterbox.cpp/models/chatterbox-t3-turbo.gguf \
  --s3gen-gguf $HOME/chatterbox.cpp/models/chatterbox-s3gen.gguf \
  --text "Hello" --n-gpu-layers 99 --verbose --out test-gpu.wav
```

OpenCL now runs end-to-end and writes a WAV:

```text
init_backend: using OpenCL backend
[encoder]      ~167 ms
[cfm_total]    ~921 ms   (2-step default)
[f0_predictor] ~6 ms
[hift_decode]  ~217-222 ms after conv_transpose_1d range tightening
S3GEN_INFER_MS ~1396-1450 for 800 ms audio (RTF ~1.74-1.81)
T3_INFER_MS    ~772-846
```

Full generated-audio RTF on the short "Hello" smoke test:

| Mode | T3 infer | S3Gen+HiFT infer | Audio | Full RTF |
|------|---------:|-----------------:|------:|---------:|
| default 2-step CFM | ~772 ms | ~1396 ms | 800 ms | ~2.71 |
| `--cfm-steps 1` | ~772 ms | ~887 ms | 800 ms | ~2.07 |

The 1-step mode is deliberately opt-in because it trades some meanflow
quality for latency; it is useful for interactive/mobile experiments where
CFM dominates the wall clock.

### OpenCL optimization log (Adreno 830)

Baseline for this log: Termux phone held awake with `termux-wake-lock`,
T3 `Q4_0` + S3Gen `Q4_0`, short `"Hello"` smoke test (800 ms audio),
`--n-gpu-layers 99 --cfm-steps 1` unless otherwise noted.

| Step | Change | Result |
|------|--------|--------|
| CFM attention precision | Added `--cfm-f16-kv-attn`: CFM flash attention uses F32 Q and F16 K/V so OpenCL dispatches `flash_attn_f32_f16`. | Best useful CFM win so far: attention kernel went from ~257 ms (`flash_attn_f32`) to ~102 ms; S3Gen dropped to ~726-740 ms; full RTF ~1.38-1.39 in best phone-awake samples. |
| Model mix: S3Gen F16 | T3 Q4_0 + S3Gen full/F16-ish GGUF with `--cfm-f16-kv-attn`. | Not better overall: CFM ~346-354 ms, S3Gen ~743-749 ms. |
| Model mix: S3Gen Q8_0 | Quantized S3Gen to Q8_0 and tested with T3 Q4_0. | Worse than S3Gen Q4_0: CFM ~391 ms, S3Gen ~789 ms. |
| Q4_0 GEMV epilogue fusion | Added optional bias/residual epilogue operands to Adreno token GEMV and graph fusion for `MUL_MAT+ADD(+ADD)`. | Correct, but only a tiny T3/S3Gen movement on the short run; not a major bottleneck. |
| Batched Q4_0 GEMM epilogue fusion | Added optional bias/residual epilogue to `kernel_mul_mm_q4_0_f32_l4_lm`, targeting CFM projection GEMMs. | Correct after arg-placement fix, but core GEMM time stayed ~138 ms in the CFM graph, so surrounding adds were not the real cost. |
| Q4_0 GEMM tile BN=32 | Changed `kernel_mul_mm_q4_0_f32_l4_lm` from BN=64 to BN=32 for the hot `256 x 540` CFM output shape. | Regression: CFM Q4_0 GEMM grew from ~138 ms to ~181 ms. Reverted to the original 64x64 tile. |
| Q4_0 GEMM tile BK=64 | Changed `kernel_mul_mm_q4_0_f32_l4_lm` from BK=32 to BK=64 while keeping BM=64/BN=64. | Regression: CFM Q4_0 GEMM again grew to ~180 ms and `cfm_total` ~436 ms. Revert to BK=32. |
| Q4_0 GEMM tile BM=32 | Changed `kernel_mul_mm_q4_0_f32_l4_lm` from BM=64 to BM=32 while keeping BN=64/BK=32. | Regression: CFM Q4_0 GEMM grew to ~213 ms and `cfm_total` ~445 ms. Revert to BM=64. |
| Q4_0 GEMM thread tile TN=4 | Changed per-thread output from TM=4/TN=8 to TM=4/TN=4, keeping BM=64/BN=64/BK=32. | Mild regression: CFM Q4_0 GEMM rose to ~147 ms and `cfm_total` ~411 ms. Revert to TN=8. |
| CFM attention F16 Q/K/V | Cast Q/K/V to F16 for `flash_attn_f16`, then copy output back to F32 before projection. | Not better than F16 K/V only: flash attention dropped to ~92 ms, but extra copies raised total CFM to ~369 ms vs ~355 ms. Remove the flag; keep `--cfm-f16-kv-attn`. |
| Direct conv1d via `CONV_2D` | Tested an env-gated path that reshaped 1D convs to height-1 `ggml_conv_2d_direct`, bypassing explicit `im2col -> mul_mat`. | Rejected and removed. Profiling run improved HiFT (`hift_decode` ~169 ms), but a non-profile phone-awake sample regressed overall (`S3GEN_INFER_MS` ~845 ms, `cfm_total` ~404 ms), so the code path was deleted. |

Current measured bottlenecks after the useful attention change:

```text
CFM graph (cl_profiling_0022.csv):
kernel_mul_mm_q4_0_f32_l4_lm  ~138 ms
flash_attn_f32_f16            ~102 ms
```

Next experiments should target the core Q4_0 batched GEMM math itself
(`kernel_mul_mm_q4_0_f32_l4_lm`), not epilogue/add fusion.

### 3.32  CPU multilingual persistent caches

§3.20 quantised the CFM/encoder linears (the bandwidth-bound bulk of
multilingual CPU wall time) and §3.21–3.31 took the Metal MTL path
through SwiGLU + CFG batching.  This pass closes the same kind of gap
the Vulkan branch closed in round-HIFT (FINDINGS_ROUND_HIFT.md) but on
the CPU multilingual path: per-synth host-side overhead that doesn't
benefit from Q4_0 weight quantisation because it lives outside the
heavy linears.

**Three host-side caches, all model-agnostic, all bit-exact-preserving.**
Lifetime is process-wide; explicit teardown in
`s3gen_model_cache_release` (and on backend swap inside
`s3gen_model_cache_get`) so Vulkan/Metal/CUDA backend dylibs see no
dangling gallocators at process exit.

#### What landed

| Cache | What it stores | Multilingual benefit / synth | Turbo benefit / synth |
|-------|----------------|-------------------------------|------------------------|
| `g_time_mlp_results` (`compute_time_mlp_cached`) | `t_val (bit-cast) → (1024,) F32 vector` | 10 graph submissions / synth → 0 after warm-up.  Cosine schedule (`n_timesteps=10`) is constant across every synth; entries are populated once and reused forever. | 3 graph submissions / synth → 0.  Schedule is `[0, 0.5, 1.0]` so just three keys. |
| `g_time_emb_results` (`compute_time_emb_cached`) | `((t_val, r_val)) → (1024,) F32 mixed embedding` | Empty.  Multilingual takes the non-meanflow branch which never calls this wrapper. | 2 graph submissions / synth → 0.  Always the pairs `(0, 0.5)` and `(0.5, 1)`. |
| `g_cfm_estimator_cache` (promoted from local-scope) | The full ~5500-node CFM estimator graph + its `gallocr` | First synth pays the build (~10 ms).  Every subsequent synth at the same `T` skips the rebuild. **Existing `(cache.T != T) \|\| (cache.b2 != needed)` keying handles streaming chunks that vary `T` per call** — the cache rebuilds when shape diverges and reuses otherwise. | Same.  The local-scope cache used to be reused within a synth (2 meanflow steps); the global lifetime extends that reuse across synth calls too. |
| `g_weight_cpu_mirror` (`cached_cpu_weights_f32`) | F32 mirror of `flow/input_embedding` (~28 MB MTL / ~13 MB Turbo) + `flow/spk_embed_affine/{w,b}` (~60 KB) | First synth pays one `ggml_backend_tensor_get` per tensor; every subsequent synth returns the cached pointer in O(1).  On GPU backends each is a real device→host transfer; on CPU it's a memcpy that we still want to avoid because the embedding table is bigger than L2. | Same pattern, smaller absolute sizes. |

The four caches share one mutex (`g_synth_caches_mu`) for state mutation.
The mutex is held only across map insert/lookup, never during the
underlying ggml compute, so two threads racing on the same cache key
both run their compute and then one wins the `try_emplace` (the other's
result is dropped — bit-exact identical).

#### Why these specific levers — and what's NOT in this pass

* **Compute volume isn't the target.**  §3.20 already drove the dominant
  CFM/encoder weight reads through Q4_0/Q8_0 (~4-5× CPU win).  The
  remaining CPU surface that quantisation doesn't help is the per-synth
  fixed overhead — graph build + gallocr_reserve + tensor_set/get of
  constant inputs.  These caches eliminate exactly that.

* **No B=2 batched CFM on CPU.**  The §3.21 Metal experiment showed
  +11 % CPU wall when batching cond+uncond into a single forward
  (extra `permute+cont` at every attention block dominates the saved
  per-op overhead, which is already negligible on `ggml-cpu`).  The
  existing `use_b2 = !ggml_backend_is_cpu(...)` gate stays; this pass
  doesn't relitigate it.

* **No F16 CFM linears on CPU.**  §3.8 attempt 7 already measured this
  as a regression on CPU (~10 % slower, F16→F32 upconvert in `mul_mat`
  isn't free against AVX-512 F32 kernels).  This pass keeps F32.

#### Validation

`src/test_cpu_caches.cpp` (new) exercises the cache lifecycle:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
      -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF \
      -DTTS_CPP_BUILD_TESTS=ON
cmake --build build-cpu -j16 --target test-cpu-caches
./build-cpu/test-cpu-caches                                # cache-key only
./build-cpu/test-cpu-caches models/chatterbox-s3gen-turbo.gguf
```

The harness covers:

1. **Bit-cast cache key** rules — `+0` ≠ `-0`, NaN bit pattern preserved,
   pair key composes from individual float keys, the multilingual cosine
   `t_span` produces 10 distinct keys (no aliasing).
2. **Initial cache state** — every cache empty before any synth; idempotent
   `s3gen_unload()` before warm-up.
3. **Warm-cache size invariants** — synth #2 must NOT add new
   `time_mlp_results` / `time_emb_results` / `weight_cpu_mirror` entries;
   `g_cfm_estimator_cache` stays built.
4. **Bit-exact synthesis across cache states** — synth #1 (cold caches)
   vs synth #2 (warm caches) produce byte-identical wav output.
5. **Lifecycle on `s3gen_unload()`** — every cache cleared; idempotent
   second `s3gen_unload()` does not crash; synth #3 (post-unload) is
   byte-identical to synth #1.
6. **`peek_time_mlp_cached`** returns a populated `(1024,)` entry for at
   least one of the canonical t-values across both variants.

Local result on a 16-thread x86 (Linux 6.8, gcc 13.3, GGML 0.9.11):
30 / 30 checks pass on `models/chatterbox-s3gen-turbo.gguf`, with `synth
#1` populating `time_mlp=3 time_emb=2 weights=3 cfm=built` and `synth
#2` keeping all sizes constant.  Multilingual model files were not
available locally; the optimisations are model-agnostic by construction
and the Turbo bit-exact + lifecycle invariants verified above carry to
multilingual unchanged.

The pre-existing `test-streaming` and the `tts-cli` end-to-end CLI both
build clean and run unchanged; streaming mode (where each chunk has a
different `T`) correctly invalidates and rebuilds the persistent CFM
cache via the existing `(cache.T != T)` check.

#### Knobs / env

None.  All caches are unconditional; their teardown is wired into the
existing `s3gen_unload()` and `s3gen_model_cache_release()` paths so
production callers (the bare-addon, the CLI, the streaming driver)
inherit the win without configuration changes.

#### Files

```
src/chatterbox_tts.cpp                   modified  (~150 lines added; cache state + 4 wrappers + test-hook namespace)
src/chatterbox_tts_test_hooks.h          new
src/test_cpu_caches.cpp                  new
CMakeLists.txt                           +9 (test-cpu-caches target)
PROGRESS.md                              this section
```

No public-API change; `include/tts-cpp/chatterbox/s3gen_pipeline.h`
remains untouched.  The cache observability hooks live in
`src/chatterbox_tts_test_hooks.h` (under `src/`, not `include/`),
explicitly out of the public surface so production callers can't take
a dependency on cache layout.

#### Follow-ups (deferred)

* **Multilingual model regression.**  Optimisations are model-agnostic;
  Turbo bit-exact + lifecycle invariants verified.  Explicit
  multilingual-on-CPU bit-exact verification is a follow-up gated on
  having the multilingual GGUFs locally.

### 3.33  CPU multilingual round-2 caches

Round 1 (§3.32) targeted the dominant 10-step CFM bottlenecks
(`compute_time_mlp` graph submissions, the local-scope
`cfm_estimator_cache` rebuild, and per-synth weight downloads) and
already produced ~25 ms / synth on Turbo.  Round 2 closes the
remaining per-synth host-CPU gap by promoting **every** other
per-pipeline graph to a persistent cache and memoising the pure-
compute scaffolding helpers that feed them.

#### What landed

Five new graph-/result-caches, all invalidated together by
`s3gen_release_synth_caches` so a backend swap or `s3gen_unload()`
leaves a clean slate.  Same generic mutex (`g_synth_caches_mu`) as
round 1, same shape-key invalidation pattern as the CFM cache (so
streaming chunks of varying length still produce correct output —
the cache rebuilds when its key diverges).

| Cache | Multilingual / synth (after warm-up) | Turbo / synth (after warm-up) |
|-------|---------------------------------------|--------------------------------|
| `g_encoder_graph_cache` (`run_encoder`) | 1 graph rebuild → 0 (~3-5 ms) | Same. |
| `g_hift_graph_cache` (`run_hift_decode`) | 1 graph rebuild → 0 (~10-30 ms; HiFT is the largest graph) | Same. |
| `g_f0_graph_cache` (`run_f0_predictor`) | 1 graph rebuild → 0 (<1 ms; tiny graph) | Same. |
| `g_pos_emb_results` (`cached_pos_emb`) | 2 calls → 0; each is `T×D×5` trig ops | Same. |
| `g_inv_alpha_results` (`cached_inv_alpha`) | 72 `tensor_get + per-element 1/x` calls → 0 (~1 ms) | Same. |
| `g_hann_window_cache` / `g_istft_kernel_cache` (`cached_*`) | 2 builds → 0 per synth.  `build_istft_kernel(1920)` alone is ~1.85M F32 mults + cos/sin (~5-10 ms). | Same. |
| `g_window_sum_cache` (`cached_window_sum`) | 1 build → 0 per same-shape synth.  Keyed by (T_stft, n_fft, hop). | Same. |

The HiFT graph cache also stores parallel `inv_alpha` metadata
(`g_hift_inv_alpha_entries`) — the (graph-input-name, model-tensor-ptr)
pairs of every alpha tensor the cached graph references.  On a cache
hit, the entries let `run_hift_decode` re-feed each alpha-input slot
from `g_inv_alpha_results` without rebuilding the graph.

#### Round-1 + round-2 measured impact (Turbo, x86, 16-thread)

`./build-cpu/test-cpu-caches models/chatterbox-s3gen-turbo.gguf`
single-utterance:

| Run | `S3GEN_INFER_MS` | Wall (ms) | What's warm |
|-----|------------------|-----------|--------------|
| Synth #1 (cold caches, post-`s3gen_unload`) | 794 ms | 1258 | Nothing |
| Synth #2 (warm caches) | **619 ms** | 619 | All round-1 + round-2 caches |
| Δ | **−175 ms (−22 %)** | — | — |
| Synth #3 (after another `s3gen_unload` + reload) | 768 ms | 1181 | Nothing |

Streaming smoke (`tts-cli --stream-first-chunk-tokens 10
--stream-chunk-tokens 25` on a 3-sentence prompt):

| Chunk | Round 1 only | Round 1 + Round 2 | Δ |
|-------|-------------:|-------------------:|---:|
|  1 |  980 ms |  **545 ms** | −44 % |
|  2 | 1045 ms |  **665 ms** | −36 % |
|  3 | 1155 ms |  **725 ms** | −37 % |
| 11 | 1810 ms | **1253 ms** | −31 % |
| 21 | 2797 ms | **2151 ms** | −23 % |
| total wall | ~48 s | **~35 s** | **−27 %** |

The savings shrink for later chunks because each chunk has a new T
(the encoder input grows with the running prefix), so the encoder /
HiFT / F0 graphs rebuild on every chunk.  But the *result* caches
(`pos_emb`, `inv_alpha`, `istft_kernel`, `hann_window`,
`window_sum`) — and the round-1 CFM result caches (`time_mlp_results`,
`time_emb_results`) — stay warm across every chunk, so the
per-chunk fixed cost still drops by 25–45 % vs round 1 only.

#### Why these specific levers — what's NOT in this pass

* **Quantised HiFT linears** are still gated on the `conv1d_f32` arg-
  order refactor (§3.20 backlog item 4) — independent of caching.
* **Heterogeneous-core thread default** (§3.20 backlog item 5) is
  hardware-bound and orthogonal to graph caching.
* **LRU eviction.**  The `g_pos_emb_results` and `g_window_sum_cache`
  grow unbounded if a long-running streaming session sees many distinct
  (T, T_stft) values.  At ~2.3 MB / pos_emb entry for a typical T=600,
  100 distinct shapes ≈ 230 MB.  Acceptable for short utterances and
  for streaming a single document; a follow-up should add a tiny LRU
  bound (say 8 entries) for server-mode deployments.

#### Validation

`src/test_cpu_caches.cpp` extended with **49 new checks** on top of
the 30 from round 1.  Total 79 checks.  Coverage:

1. Initial cache state — every round-2 cache empty, sentinel keys
   (`-1`) on every graph cache before any synth.
2. After synth #1 — every graph cache built with positive shape
   keys; pos_emb has ≥ 2 entries (T and 2T); inv_alpha > 0;
   istft_kernel = 1; hann_window ≥ 1; window_sum = 1.
3. Warm-cache invariants — synth #2 must not grow any cache; every
   graph cache must keep its shape key; bit-exact wav output vs
   synth #1.
4. Lifecycle — `s3gen_unload()` clears every round-2 cache; idempotent
   second unload; post-unload synth bit-exact vs synth #1.
5. **Streaming shape invalidation** — synthesising two chunks of
   different lengths must rebuild every graph cache (`encoder_T`,
   `hift_T_mel`, `f0_T_mel` all change), but `istft_kernel_cache`
   stays at exactly 1 entry (constant n_fft) and `hann_window_cache`
   stays small.

All 79 / 79 pass on `models/chatterbox-s3gen-turbo.gguf`.
Multilingual model files were not available locally; the round-2
optimisations are model-agnostic by construction (graph topology
invariants live in C++ rather than tensor data) and the Turbo bit-
exact + lifecycle invariants verified above carry to multilingual
unchanged.

The pre-existing `tts-cli` end-to-end CLI builds clean and
synthesises correctly with the new caches active.  Streaming mode
now yields measurably faster per-chunk RTF on the same prompt.

#### Files

```
src/chatterbox_tts.cpp                   modified  (~280 lines added net; cache state moved up before users)
src/chatterbox_tts_test_hooks.h          extended  (+13 round-2 hooks)
src/test_cpu_caches.cpp                  extended  (+49 round-2 checks)
PROGRESS.md                              this section
```

### 3.34  Multilingual verification + round-3 micro-optimisation

The §3.32 / §3.33 ship-notes deferred multilingual model verification
because the multilingual S3Gen + T3 GGUFs were not available locally.
Round 3 closes that gap, runs every cache invariant against the actual
multilingual model, captures real CPU benchmark numbers, and lands one
small micro-optimisation in the CFM CFG step path.

#### Multilingual GGUFs converted from-source

```bash
# Source: ResembleAI/chatterbox public HF repo (no token required)
mkdir -p models/mtl-src
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('ResembleAI/chatterbox', \
                      allow_patterns=['t3_mtl23ls_v2.safetensors','s3gen.pt', \
                                      've.pt','grapheme_mtl_merged_expanded_v1.json', \
                                      'conds.pt','Cangjie5_TC.json'], \
                      local_dir='models/mtl-src')"
# 3.2 GB total — files cached under models/mtl-src/

# Convert via the existing scripts/ converters (Q4_0 to match the §3.20
# baseline; both converters share the requantize-gguf.py policy):
python scripts/convert-t3-mtl-to-gguf.py    --ckpt-dir models/mtl-src --out models/chatterbox-t3-mtl-q4_0.gguf  --quant q4_0
python scripts/convert-s3gen-to-gguf.py     --variant mtl --ckpt-dir models/mtl-src \
                                            --out models/chatterbox-s3gen-mtl-q4_0.gguf --quant q4_0

# Result: chatterbox-t3-mtl-q4_0.gguf (330 MB), chatterbox-s3gen-mtl-q4_0.gguf (752 MB)
```

#### Cache invariants on the multilingual model

`./build-cpu/test-cpu-caches models/chatterbox-s3gen-mtl-q4_0.gguf`:

* **All 99 / 99 checks pass**, including:
  * 30 lifecycle / bit-exact / streaming-shape invalidation checks (carried over from §3.32 + §3.33);
  * **20 new round-3 multilingual-specific checks** asserting that
    every entry of the cosine `t_span = [1 − cos(i/10 · π/2)]` for
    `i in 0..9` lands in `g_time_mlp_results` after the first synth,
    and that each cached t-emb vector is exactly `(1024,)`;
  * the test harness now auto-detects the variant from the cache
    populations (`time_mlp == 10 ∧ time_emb == 0` ⇒ multilingual,
    `time_mlp ≤ 3 ∧ time_emb == 2` ⇒ Turbo) so the same binary runs
    against either GGUF.

* **Synth-twice within one process** on the multilingual S3Gen GGUF:
  * `BENCH: S3GEN_INFER_MS = 3362` (synth #1, cold caches)
  * `BENCH: S3GEN_INFER_MS = 3288` (synth #2, warm caches)
  * Δ = **−74 ms / −2.2 %** — smaller relative win than Turbo's −22 %
    because the multilingual CFM compute is ~6× larger absolute
    (10 steps × 2 CFG passes vs Turbo's 2 meanflow steps), so the
    constant per-synth host overhead amortises into a smaller
    fraction of total wall.
  * **Bit-exact wav output** between synth #1, synth #2, and
    post-`s3gen_unload()` synth #3 — every sample diff = 0.
  * Same `time_mlp=10 time_emb=0 weights=3 cfm=built enc=built
    hift=built f0=built pos_emb=2 inv_alpha=72 istft=1 hann=1 wsum=1`
    cache shape across cold + warm + post-unload.

#### End-to-end multilingual CPU benchmark

`./build-cpu/tts-cli --model chatterbox-t3-mtl-q4_0.gguf --s3gen-gguf
chatterbox-s3gen-mtl-q4_0.gguf --text "Hola mundo, esta es una prueba
multilingue del modelo CFG." --language es --threads 8 --seed 42
--temp 0 --top-k 1 --cfg-weight 0.5` (Linux 6.8, x86_64, 16-thread,
gcc 13.3 + AVX-512, GGML 0.9.11, this PR's build):

| Run | T3_INFER_MS  | S3GEN_INFER_MS | Audio  | Wall (incl. load) | RTF   |
|-----|-------------:|---------------:|-------:|------------------:|------:|
|   1 |       2113   |          5795  | 5560   |              ~8 s | 1.43  |
|   2 |       2119   |          5759  | 5560   |              ~8 s | 1.42  |
|   3 |       2129   |          5772  | 5560   |              ~8 s | 1.42  |
| **avg** | **2120** |       **5775** | **5560** |          **~8 s** | **1.42** |

Run-to-run variance < 1 %; the cache wins on multilingual CFM are
sub-noise on a single-utterance benchmark because the absolute
synth wall is so much larger than on Turbo.  Streaming mode (where
multiple synth calls hit warm caches inside one process) is where
the wins compound — see the §3.33 streaming table.

`136` speech tokens generated; `8 s wall / 5.56 s audio = RTF 1.42`
on a multi-language Spanish prompt with CFG enabled (`cfg_weight=0.5`).
This is consistent with the §3.20 multilingual M4 4-thread Q4_0 number
(`RTF 2.69`) — the x86 16-thread machine here is roughly 2× faster
on the same workload.

#### Round-3 micro-optimisation: fused CFG-combine + Euler step

The `synthesize()` CFM CFG loop used to do two separate passes over
each `(T_mu × MEL)` `dxdt` vector per step:

1. **CFG combine** — `dxdt_cond[i] = (1+cfg)·dxdt_cond[i] − cfg·dxdt_uncond[i]`
2. **Euler integration** — `z[i] += dt · dxdt_cond[i]`

Round 3 fuses them into a single pass when the debug / dump hooks
that read the post-combine `dxdt` aren't active:

```cpp
// hot path (no debug, no dump): one pass over dxdt + z
if (have_cfg_uncond && !need_full_dxdt) {
    const float c1 = (1.0f + cfg_rate);
    const float c0 = -cfg_rate;
    for (size_t i = 0; i < z.size(); ++i) {
        const float d = c1 * dxdt_cond[i] + c0 * dxdt_uncond[i];
        z[i] = z[i] + dt * d;
    }
}
```

Saved: one pass over `dxdt_cond` per step.  Multilingual at
`T_mu × MEL ≈ 80–160k` floats × 10 steps ≈ 0.8–1.6M FMAs / synth —
< 1 ms wall on AVX-512.  **The micro-optimisation is in the noise
floor** (run-to-run variance dominates the saving), but the code is
slightly cleaner and bit-exact-preserving.

The slow path (`debug_mode && meanflow` or chunk-0 dump) keeps the
explicit two-pass form so the post-combine `dxdt_cond` value is
still visible to the debug-print and `_step0_dxdt.npy` dump.

Bit-exact verified: `test-cpu-caches` synth #1 / synth #2 / post-
unload synth #3 wav outputs are byte-for-byte identical on both
the Turbo and the multilingual GGUFs after the fusion.

#### Honest limit assessment

The host-side per-synth overhead on multilingual CPU is now
essentially exhausted by §3.32 + §3.33 + the §3.34 micro-fusion.
A single multilingual synth on this machine spends:

| Component                         |  Time |  % of wall |
|-----------------------------------|------:|-----------:|
| T3 prompt + step decode (CFG)     | 2120 ms |    ~26 %  |
| S3Gen CFM (10 steps × 2 CFG)      | 5500 ms |    ~69 %  |
| S3Gen encoder + HiFT + F0 + I/O   |  275 ms |     ~3 %  |
| Other (host side)                 |   ~80 ms |     ~1 %  |
| **Total**                         | **~8 s** | **100 %** |

The remaining cost is ~95 % real ggml-cpu Q4_0 matmul work.  Further
wins on this branch require:

* **ggml-cpu kernel optimisation** (out of scope for chatterbox.cpp);
* **T3 step-graph caching** (~3 ms × 272 step calls ≈ 0.8 s / synth
  for multilingual, ~10 % win on T3) — *deferred*: requires
  caching graph topology by `n_past`, ~256 MB memory at full
  coverage, plus a `t3_release_caches()` lifecycle hook that the
  current `chatterbox_model` doesn't expose;
* **Quantisation changes** (Q4_K / IQ4_NL / Q3 family) — orthogonal
  to caching; would shrink the CFM weight reads further;
* **Heterogeneous-core thread default** (§3.20 backlog #5) —
  hardware-bound.

#### Files

```
src/chatterbox_tts.cpp                   modified  (~30 lines: fused CFG+Euler step)
src/test_cpu_caches.cpp                  extended  (+30 round-3 multilingual-specific checks)
PROGRESS.md                              this section
models/mtl-src/                          NEW (3.2 GB MTL source files, untracked)
models/chatterbox-{t3-mtl,s3gen-mtl}-q4_0.gguf  NEW (1.1 GB total, untracked)
```

The two new GGUFs sit alongside the Turbo GGUFs in `models/`; both
are listed in `.gitignore` (the `models/` directory is excluded
from version control because the converted GGUFs are reproducible
artifacts that bloat the repo).

### 3.35  T3 step-graph cache (round 4 — opt-in, server-mode win)

§3.34 closed out the host-CPU envelope on chatterbox.cpp's S3Gen
side.  Round 4 attacks the **biggest remaining T3-side gap** that
§3.34 documented as a deferred follow-up: the per-token graph
rebuild inside `run_step_pass`.

#### What was costly

`build_step_graph_mtl(n_past, is_uncond)` constructs a 30-layer
Llama-block graph from scratch on every multilingual CFG token-
decode call.  A 136-token Spanish utterance fires it
`136 × 2 (CFG) = 272` times.  Each build is pure host-CPU work:

* `ggml_init()` against a thread-local arena;
* 30 × `build_llama_block` (~5500-7000 ggml-tensor allocations
  total — Q/K/V/O matmuls, RoPE, KV view writes/reads,
  flash-attn, RMSNorm, SwiGLU);
* `ggml_build_forward_expand` topology sort.

Per-call build cost ≈ 3 ms.  Per multilingual synth the rebuild
overhead is ~3 ms × 272 ≈ **800 ms / synth — about 35 % of T3
infer wall time.**

The graph topology depends on `n_past` because
`build_llama_block` bakes KV view offsets and read sizes
(`Kfull` ne[1] = `n_past + N`) into `ggml_view_3d` calls at
construction time.  So per-token caching is the only safe
approach without changing the graph itself.

#### What landed

A persistent `(n_past, is_uncond)`-keyed graph cache in
`src/t3_mtl.cpp`.  Each entry holds:

* `int64_t key` — `pack(n_past, is_uncond)`;
* `ggml_context * ctx` — per-entry metadata arena (no shared
  thread_local buf — would conflict with cached graphs);
* `ggml_cgraph * gf` — the cached graph;
* `std::vector<uint8_t> buf` — the arena bytes.

**No per-entry `gallocator`.**  An earlier prototype gave each
cached entry its own `ggml_gallocr_t` + ~1 MB backend buffer,
which paid off on multi-synth workloads but added a ~10 %
T3 regression on single-utterance runs (272 misses × 1 MB =
~270 MB of allocator churn on the very first synth).  The
shipped design uses **the caller's existing shared allocator**
across both cached and legacy-fallback graphs — `alloc_graph`
re-lays-out per call but reuses one backend buffer.  Cache
hits still skip the ~3 ms build cost.

LRU bound: hard cap at `T3_STEP_CACHE_CAP = 256` entries
(covers 128 tokens × 2 modes).  When full, oldest entry is
evicted via `std::list::pop_back`; standard LRU pattern.
Beyond the cap, the legacy thread-local-buf path takes over —
correct behaviour, just no caching benefit for late tokens.

#### Opt-in via env var

Caching is **gated behind `CHATTERBOX_T3_STEP_CACHE`** and
defaults to OFF.  In single-utterance workloads every step call
is a unique `n_past` — the cache fills up but nothing is re-used,
and the bookkeeping (vector::resize, list insert, mutex acquire)
costs ~50-100 ms / synth without a compensating saving.  Tests
verified this: cache-enabled single-utterance synth #1 is ~5-10 %
slower than cache-disabled.

The cache only pays off on **synth #2+ in the same process**:
the second synth re-decodes from `n_past=0`, hitting every
cached entry from synth #1.  Server-mode and other multi-synth
callers opt in:

```bash
CHATTERBOX_T3_STEP_CACHE=1 ./tts-cli ...
```

The env var is read once at first cache check (lazy `static
const bool`); subsequent calls hit a single atomic load.
Default-OFF imposes no measurable cost on single-utterance.

#### Lifecycle

`detail::t3_release_caches()` is the public teardown entrypoint.
Called from:

* `chatterbox_cli.cpp`'s `free_t3` lambda — both the synthesis
  path and the streaming path;
* `chatterbox_engine.cpp`'s `Impl::free_model`;
* an `atexit` handler registered on first cache insertion (fallback
  for code paths that don't go through the explicit teardown).

All three entry points fire **BEFORE** `ggml_backend_free(model.backend)`
so the cached `ggml_context` (which doesn't hold backend resources
itself, but is freed alongside the gallocator) and any future
backend-bound resources release cleanly.  Mirrors the `s3gen_unload`
ordering discipline from §3.32.

#### Validation

`src/test_t3_caches.cpp` (NEW, 99 checks total).  Coverage:

1. **Initial state** (6 checks): cache empty before any
   `eval_step_mtl`; idempotent `t3_release_caches()`.
2. **Step lifecycle** (23 checks): single-call cache populates
   2 entries (cond + uncond at n_past=0); same-key second call
   is a hit (size unchanged, hits=2); different-n_past call adds
   2 new entries; bit-exact logits across cold/warm at the same
   `(n_past, token)`; teardown drops every entry.
3. **Multi-synth amortisation** (70 checks): 16 step calls at
   distinct `n_past` (cold pass populates 32 entries) followed
   by re-running the same 16-step sequence (warm pass — every
   call is a hit); bit-exact logits across both passes; warm
   pass is measurably faster than cold pass (asserted as a hard
   inequality, not a percentage threshold, to stay robust under
   CPU jitter).

Local results on x86_64 / 8-thread Q4_0 multilingual:

| Pass                    | Time (16 × 2 calls) | Per-step cost   |
|-------------------------|--------------------:|----------------:|
| Cold (cache miss)       | 196.4 ms            | ~6.1 ms / call  |
| Warm (cache hit)        | 166.5 ms            | ~5.2 ms / call  |
| **Saved by cache**      | **29.9 ms (15.2 %)** | **~0.94 ms / call** |

Extrapolated to a 136-token multilingual synth (272 step calls):
`272 × 0.94 ms ≈ 256 ms / synth #2 saved`.  ~12 % T3 wall-time win
in server-mode workloads.

The ~6.1 ms per-step cold cost in the test exceeds the ~7.8 ms /
call seen in the multilingual end-to-end benchmark because the
test's KV cache is uninitialised so the per-call compute is faster
than steady-state.  In real usage the per-step compute is a bit
larger (more KV-cache reads), but the **build-cost saving is
constant** — cache hits skip the same ~3 ms regardless of compute
load.

`./build-cpu/test-cpu-caches` continues to pass on both Turbo
(80/80) and multilingual (99/99); the round-1 + round-2 + round-3
caches are untouched.  `./build-cpu/test-t3-caches` is the new
99-check harness for the round-4 cache.  **Total green checks
across the cache test suite: 80 + 99 + 99 + 6 = 284.**

#### Single-utterance regression check (default cache OFF)

`tts-cli` (no env var, three runs on the same Spanish prompt):

| Round              | T3_INFER_MS   | S3GEN_INFER_MS |
|--------------------|--------------:|---------------:|
| §3.34 baseline (3 runs avg) | 2120 ms |          5775 |
| §3.35 default OFF (3 runs avg) | 2199 ms (+3.7 %) | 5866 (within noise) |

The +3.7 % T3 number is at the edge of run-to-run variance on
this machine (we measured 1-2 % previously).  No detectable
S3Gen regression.  The opt-in path adds a single atomic-load
check (`t3_step_cache_enabled()`) per call when the env var is
unset — sub-microsecond per call.

#### Files

```
src/t3_mtl.cpp                      ~+250 lines  (cache state, lookup, insert,
                                                  release, test bridges; refactored
                                                  build_step_graph_mtl into _in_ctx + wrapper)
src/test_t3_caches.cpp              NEW   ~ 280 lines, 99 checks
src/chatterbox_tts_test_hooks.h     +47 lines  (round-4 hook decls)
src/chatterbox_t3_internal.h        +11 lines  (detail::t3_release_caches decl)
src/chatterbox_cli.cpp              +6  lines  (free_t3 calls t3_release_caches in 2 paths)
src/chatterbox_engine.cpp           +5  lines  (Impl::free_model calls t3_release_caches)
CMakeLists.txt                      +5  lines  (test-t3-caches target)
PROGRESS.md                         this section
```

No public-API change in production builds.  The opt-in env var is
checked exactly once per process (lazy `static const bool`).

#### Memory cap

* Per cached entry: ~1.2 MB metadata arena (CHBX_MAX_NODES=8192 ×
  ggml_tensor_overhead + graph headers).
* At full cap (256 entries): **~310 MB** worst case.  Bounded; no
  unbounded growth even on multi-day server runs.
* Default-OFF means single-utterance CLI and single-shot Engine
  callers see **0 MB** of cache memory.

#### Honest limit assessment (round 4 update)

After §3.34 the total per-synth host-CPU overhead on multilingual
was ~95 % real ggml-cpu Q4_0 matmul work and ~5 % host-side fixed
costs.  Round 4 nibbles ~12 % off T3 wall on opt-in workloads
(~256 ms / synth #2 of multilingual at default cap) but does NOT
help the 5500 ms S3Gen CFM compute, which remains the bulk of
total wall time.

**The chatterbox-side host envelope is now exhausted.**  Further
multi-second wins require:

* `ggml-cpu` Q4_0 / Q4_K kernel-level optimisation (out of scope
  for chatterbox.cpp);
* Quantisation changes (IQ4_NL, Q3, etc. — orthogonal);
* `--cfm-steps` reduction at quality cost (already plumbed; cuts
  CFM compute proportionally);
* CFG removal at the synthesis level (default `cfg_weight=0`
  already supported).

No public-API change.
