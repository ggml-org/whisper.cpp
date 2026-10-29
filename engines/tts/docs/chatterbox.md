# tts engine: Chatterbox

Part of the [tts engine documentation](../README.md).

## Chatterbox

End-to-end inference on a short sentence with voice cloning from an 11 s
reference wav (T3 + S3Gen + HiFT, warm runs, excludes model load):

**Turbo:**

| Backend                              | Wall      | `RTF`  | vs real-time |
|--------------------------------------|----------:|-------:|-------------:|
| Vulkan (CI · Linux x86-64, Q4_0)     |   410 ms  | 0.099  | **10.1×**    |
| Metal (Mac Studio M3 Ultra, Q4_0)    |   985 ms  | 0.16   | 6.4×         |
| CPU (CI · Linux x86-64, Q4_0)        | 5 841 ms  | 1.54   | 0.65×        |
| CPU (Mac Studio M3 Ultra, NEON)      | 7 568 ms  | 1.05   | 0.96×        |

> Rows are independent warm runs on different machines/backends, so **`RTF`** —
> not wall time — is the comparable metric; wall time tracks each run's own
> generated audio length. The Linux x86-64 rows are CI numbers (see the
> [Performance](performance.md#ci-benchmarks-2026-08-12-linux-x86-64) section for the full CI table + provenance).

**Multilingual** (same Spanish prompt, seed 42, built-in voice):

| Backend                              | Wall      | `RTF` | vs real-time |
|--------------------------------------|----------:|------:|-------------:|
| **Metal (M3 Ultra, Q4_0, `--cfm-steps 7`)** | **1.05 s**| **0.30** | **3.3×**     |
| Metal (M3 Ultra, Q4_0)               |  **1.22 s** | 0.35  | 2.9×        |
| Metal (M3 Ultra, F16, `--cfm-steps 7`)| 1.16 s   |  0.32  | 3.2×        |
| Metal (M3 Ultra, F16)                |  1.41 s   |  0.38  | 2.6×        |
| **Metal (M4, Q4_0)**                 |  **3.0 s**| 1.37  | 0.73×        |
| Metal (M4, F16)                      |   4.0 s   | 1.65  | 0.61×        |
| CPU (M4, 4t NEON, Q4_0)              |  10.7 s²  | 4.32² | 0.23×        |
| CPU (M4, 4t NEON, F16)               |  17.1 s²  | 6.70² | 0.15×        |

The M3 Ultra rows reflect the §3.21 optimisation pass — CFG cond+uncond
batched into one Metal forward (B=2) on T3, the new `--cfm-steps N` knob
on the standard 10-step CFM (N=7 is the recommended quality knee, log-mel
cosine vs N=10 = **0.995**), and `ggml_swiglu_split` on the Llama MLP.
The M4 rows are kept for continuity with §3.19/§3.20.

² **CPU multilingual rows re-measured after restoring CFG on the
non-Metal CFM path.**  The previous numbers in this row (`6.0 s / 2.69`
and `7.8 s / 3.24`) were captured between Apr 23–May 4 while the
non-`use_b2` branch in the CFM step loop was silently running only the
conditional pass — i.e. half the CFM compute, no classifier-free
guidance steering on CPU/Vulkan/CUDA.  Fixed in commit `6d9b42b`; the
two rows above are now end-to-end re-measurements on the same M4 host
with CFG correctly applied (12 CFM steps × 2 forward calls).
S3Gen wall-time roughly doubled, RTF went up ~2×.  Metal rows are
unaffected (the `use_b2` branched path always carried the CFG combine).

See the [full benchmark](performance.md#ci-benchmarks-2026-08-12-linux-x86-64) report for the CI benchmark
table, or the [Chatterbox port archive](history/chatterbox-port.md) for the full chronological
development journal — every numerical-parity stage and optimization pass
(T3 Flash Attention, KV-cache layout rework, Metal kernel patches,
CAMPPlus + VoiceEncoder + S3TokenizerV2 ported to ggml graphs, mel
extraction via STFT matmul, T3 Q4/Q5/Q8 quantization, the multilingual
Llama-520M port + CFG dual-cache (§3.19), and the shared S3Gen weight-
quantisation pass that ships in this repo (§3.20)).


### Core ML sidecars

Configure with `-DTTS_CPP_COREML=ON` on Apple platforms; the default build
keeps Core ML disabled. The S3Gen encoder, CFM estimator and HiFT decoder can
each run on a Core ML sidecar next to the S3Gen GGUF, by default on the CPU and
Neural Engine. T3, tokenization, the voice frontend, the time embedding, the
Euler update, the F0 predictor, SineGen, STFT and the inverse STFT stay on
ggml, so T3 keeps the GPU.

Export from the same S3Gen GGUF (`f32` or `f16`; one export serves every
quantization tier of the checkpoint) with the
[Core ML requirements](../../parakeet/scripts/requirements-coreml.txt)
(`torch`, `numpy`, `gguf`, `coremltools`) and Xcode's `coremlcompiler`:

```bash
python3 scripts/export-chatterbox-coreml.py --gguf models/chatterbox-s3gen-f32.gguf \
    --output models --compile
python3 scripts/export-chatterbox-coreml.py --gguf models/chatterbox-s3gen-mtl-f32.gguf \
    --output models --compile
```

`chatterbox-s3gen-q8_0.gguf` then finds `chatterbox-s3gen-encoder.mlmodelc`,
`chatterbox-s3gen-cfm.mlmodelc` and `chatterbox-s3gen-hift.mlmodelc`; the
Multilingual files are `chatterbox-s3gen-mtl-{encoder,cfm,hift}.mlmodelc`. The
quantization tag is dropped by the rule every TTS sidecar uses; do not reuse a
sidecar for different checkpoint weights. `--stage encoder|cfm|hift` exports
one stage. The artifacts target macOS 15 / iOS 18 (multi-function models);
older runtimes keep ggml.

The encoder and CFM estimator are multi-function models with one function per
width: `--frames` (default `512,640,768,896,1024,1280,1536` mel frames, and half
that in tokens for the encoder) covers the voice prompt plus a sentence. A call
takes the narrowest function that holds it, with the padding masked out of
every attention, so it matches a ggml call at its own length; a longer input
keeps ggml for that stage. The Multilingual estimator carries the
classifier-free-guidance pair in its batch. HiFT is one 128-frame window walked
over the utterance with 15 frames of real context on each side (its receptive
field), zero-masked past the end, so the stitched spectrum matches a
whole-utterance decode. Exp, sine and the inverse STFT of its output stay in
float32 on the host. Every stage computes in float16: the Neural Engine has no
sine (Snake is a range-reduced polynomial), runs dilation-5 and stride-15
convolutions on the CPU (they are respelled as undilated/blocked convolutions)
and miscomputes native transposed convolutions (respelled in phase form), as
for the Parler and Audio8 sidecars.

`CHATTERBOX_COREML_COMPUTE_UNITS` (or `EngineOptions::coreml_compute_units`)
accepts `cpu_and_ane` (default), `all`, `cpu_and_gpu` and `cpu_only`;
`cpu_and_ane` measured fastest for every stage and keeps the GPU for T3.
`CHATTERBOX_COREML_DISABLE` keeps every stage on ggml. Before each prediction
every input and output shape the sidecar declares is checked against what the
call packs; a sidecar that declares another shape (an incompatible export, or
the other variant's sidecar) or fails a prediction is retired and ggml takes
over. `CHATTERBOX_COREML_STRICT` (or `EngineOptions::coreml_strict`) makes
that, or an input past the widest function, an error instead. `SynthesisResult::s3gen_backend` reports `ggml`,
the sidecar label (`coreml-ane`, ...) when every stage of the call ran on
Core ML, or `mixed`.

With the CFM and HiFT sidecars present, S3Gen no longer needs the GPU, so the
batch path vocodes sentence *i* while T3 decodes sentence *i + 1* (`tts-cli`
and `Engine` with `max_sentence_chars`; the CLI's default 180 splits). On one
GPU the two only contend, which is why the ggml path stays sequential.

The Neural Engine compiles each function on its first use by an executable
(seconds to tens of seconds per function; the OS caches the result per
executable). After that, loading a function costs about 0.1 to 1 s per process,
growing with the whole multi-function program, so a background thread loads
HiFT and the encoder and CFM functions each sentence (or streamed chunk) is
expected to need, from its text length, while T3 decodes; an input outside the
prediction loads its function on demand.

Measured on an Apple M2 (macOS 15.7, 10-core GPU), `q8_0` T3 and S3Gen, the
built-in voice, seed 42, `tts-cli` with the benchmark prompt (three sentences
split into two segments), median of three warm runs with compiled sidecars,
same binary with and without `CHATTERBOX_COREML_DISABLE`:

| Model | Path | T3 | S3Gen (2 segments) | Synthesis | RTF | Process wall |
|---|---|--:|--:|--:|--:|--:|
| Turbo | Metal | 3.00 s | 1.75 s | 4.73 s | 0.49 | 5.70 s |
| Turbo | Core ML | 2.66 s | 0.72 s (overlapped) | **2.91 s** | **0.30** | 3.68 s |
| Multilingual | Metal | 5.06 s | 3.57 s | 8.63 s | 1.04 | 9.23 s |
| Multilingual | Core ML | 5.22 s | 1.73 s (overlapped) | **5.87 s** | **0.70** | 6.68 s |

Synthesis takes 39% (Turbo) and 32% (Multilingual) less time, 1.63x and 1.47x
the Metal throughput. Per stage, the Turbo encoder takes 47-60 ms against
55-84 ms on Metal, a CFM step 45-57 ms against 89-140 ms, and HiFT 87-190 ms
against 368-798 ms; the Multilingual CFM step (CFG pair) takes 55-81 ms against
105-155 ms. Through `Engine` (`max_sentence_chars = 180`, median of four calls on one
Engine) the same prompt takes 3.08 s against 4.49 s for Turbo (31% less) and
5.66 s against 8.40 s for Multilingual (33% less): T3 runs a few percent slower
while the Neural Engine vocodes the previous sentence, as both share memory
bandwidth. Streaming Turbo (`--stream-chunk-tokens 25`) delivers its first
chunk after 1.94 s against 2.53 s and finishes in 4.58 s against 6.38 s. A
single sentence has nothing to overlap: there the gain is the S3Gen time alone
(Turbo 0.59 s to 0.25 s, Multilingual 1.32 s to 0.65 s on the last sentence
above).

Accuracy against ggml (Metal) on the same `q8_0` GGUF and tokens
(`test-chatterbox-coreml --stages-fixture`: each sidecar alone, then all
three):

| Model | Encoder `mu` | CFM mel | HiFT waveform | All three, mel |
|---|--:|--:|--:|--:|
| Turbo | 0.999992 | 0.999976 | 0.999894 | 0.999973 |
| Multilingual | 0.999990 | 0.999981 | 0.999609 | 0.999980 |

(cosine similarity; the gates are 0.9999, 0.999, 0.995 and 0.999). On the
benchmark prompt the Multilingual HiFT alone reproduces the Metal waveform at
cosine 0.99992 (37 dB SNR). Whisper `tiny` transcribes the Metal and
Core ML outputs of the benchmark prompt identically for both models and for
seeds 1-4; `base.en` heard one word differently on one Multilingual run.

When Core ML is compiled in and any S3Gen sidecar is enabled, `fit_params`
returns `FitStatus::Error` with reason `coreml-memory-estimate-unavailable`:
the native projection cannot account for resident Core ML models and
prediction buffers. `CHATTERBOX_COREML_DISABLE=1` restores it.

`test-chatterbox-coreml` covers the policy, naming, function selection, the
HiFT window plan and the fit refusal without models;
`test/test_export_chatterbox_coreml.py` checks the rebuild on random weights
(padding equivalence, the window walk against a whole decode, the respelled
convolutions and Snake). `test-chatterbox-coreml-shapes` loads tiny sidecars
from `test/prepare_chatterbox_coreml_shape_fixture.py`
(`CHATTERBOX_COREML_TEST_SHAPES`) and checks that one declaring any feature
larger than the runtime packs it is rejected before prediction.
`test-chatterbox-coreml-stages` reads `CHATTERBOX_COREML_TEST_S3GEN` (an S3Gen
GGUF with its exported sidecars); `test-chatterbox-coreml-engine` also reads
`CHATTERBOX_COREML_TEST_T3` and checks the `Engine` sentence overlap: PCM
identical to the segments synthesized one call at a time and crossfaded in
order, the aggregated backend, cancellation mid-request, and a worker's
strict-mode error reaching the caller. Each reports a skip without its inputs.
The macOS CI lane runs the exporter and shape checks.

### Adreno OpenCL

ggml-opencl declines `FLASH_ATTN_EXT` on Adreno (its kernels miscompute some
inputs there unless `GGML_OPENCL_FA_ADRENO=1`), so T3 builds its attention
from score and value products and a masked softmax wherever the backend
declines the fused node; before, the scheduler ran every T3 attention on the
CPU and copied the KV cache across on every step. On Adreno, the Multilingual
cond+uncond pair also runs its weight products without batch broadcasting
(ggml-opencl dequantized each quantized weight to F16 for a broadcast
product): the prompt as one flattened product, each decode step as one
matrix-vector product per half. `test-attention-fallback` pins the unfused
form to `ggml_flash_attn_ext`. T3 decode on a Galaxy S25 (Adreno 830, `q4_0`,
one sentence):

| T3 | Before | After | CPU (6 threads) |
|---|--:|--:|--:|
| Turbo, 70 tokens | 3.9 s | 1.3 s | 1.6 s |
| Multilingual, 61 tokens | 134 s | 1.9 s | 1.4 s |

`test-t3-backend-logits` against the CPU over the same teacher-forced tokens:
Turbo min cosine 0.99985 with 24 of 25 argmaxes equal, Multilingual 0.99914
with 47 of 50. The S3Gen CFM keeps its flash attention on OpenCL, whose
output matches the CPU's (mel rel 2.3e-6 on the `f32` tier). The
Multilingual F32 KV cache at the GGUF's context (1.44 GB per tensor) still
exceeds OpenCL's 1 GB allocation limit on Adreno: cap `n_ctx` (e.g. 1200) or
use the `f16` cache.

### Snapdragon Hexagon NPU

Chatterbox Turbo and Multilingual run on the Snapdragon Hexagon NPU (HTP0;
measured on a Galaxy S25, Snapdragon 8 Elite, Hexagon v79) when it is requested
explicitly: `EngineOptions::backend = "hexagon"` or `--backend hexagon` on
`tts-cli` and `chatterbox-fit-params`. There is no fallback: construction fails
when HTP0 is missing, and the automatic `--n-gpu-layers` walk never selects it.
T3 (the prompt and every decode step) and S3Gen (encoder, CFM, F0 predictor,
STFT and HiFT) run on the NPU. With a reference WAV, the S3Tokenizer and
CAM++ stay on the CPU, as the voice encoder does on every backend. Weights go
into the Hexagon repack buffer type; HTP multiplies only `f32`, `f16`, `q8_0`
and `q4_0` weights, so a T3 or S3Gen GGUF holding another type is refused at
load. It needs a `qvac-ext-ggml@speech` build with the Hexagon `COL2IM_1D`,
F16 `IM2COL`, flash attention (F32 and F16 keys and values), sin, cos and
leaky ReLU kernels (measured at `a6122941`).

On Hexagon every S3Gen graph is checked against the operations HTP reports it
can run (support depends on shapes, e.g. flash attention past its VTCM budget)
and goes through the `[backend, CPU]` scheduler when one is missing; other
backends keep that check for HiFT only, as before. Four Hexagon forms keep the
whole pipeline on the NPU:

- **HiFT upsamplers and iSTFT.** HTP has no transposed convolution, so each
  runs as one GEMM into columns plus `ggml_col2im_1d`
  (`conv_transpose_1d_columns`), pinned to `ggml_conv_transpose_1d` by
  `test-conv-columns`.
- **F16 convolution columns.** HTP has no fast F32 `im2col`, and HMX
  multiplies in F16 regardless, so convolutions whose kernel sits on the NPU
  take F16 columns. HiFT on 36 s of audio drops from 13.5 to 4.2 s. The
  batched (classifier-free-guidance) CFM convolutions keep F32 columns: HTP's
  batched F16 `im2col` is slower than its F32 one.
- **F16 K/V CFM attention.** HTP runs F32 flash attention on HVX; F16 keys
  and values move it onto HMX, 15x faster (7.1 to 0.47 s of attention over
  the 2,306-frame CFM of a 36 s clip). On `f32` S3Gen weights the first CFM
  step's velocity then differs from the CPU's by at most 0.27 (cosine
  0.999996) instead of 0.058; the `q4_0` tier itself moves it by up to 5.1
  from the `f32` weights, identically on the CPU and on the NPU.
- **F16 T3 KV cache.** The DSP maps at most 3.2 GB of buffers per batch of
  operations. The Multilingual F32 cache at the GGUF's context (2.9 GB for the
  cond+uncond pair) overflows it beside the weights, so every decode step
  remapped: 3.1 s of T3 for a 60-token sentence against 0.67 s with the F16
  cache. Hexagon therefore stores the cache as F16 whatever
  `kv_cache_type` requests (Turbo 0.77 GB, Multilingual 1.44 GB).

The conformer encoder keeps its unfused relative-position attention (an F16
mask drifts it past the WAV quality gate; see §3.25 of the
[port archive](history/chatterbox-port.md)), whose F32 score and value
products run on HVX. On a single 36 s segment it takes 5.2 s on the NPU
against 2.4 s on the CPU, so split long text into sentences
(`EngineOptions::max_sentence_chars`, the CLI's default 180).

Accuracy against the CPU on the same GGUF (the CPU and the NPU run different
quantized-matmul paths, so the residual is larger at `q4_0` than at `f32`):

| Check | Result | Gate |
|---|---|---|
| `test-s3gen-agreement-hexagon`, Turbo S3Gen `f32`, 900 tokens | mel rel 4.0e-3, waveform rms ratio 0.056 | 2e-2, 0.15 |
| `test-s3gen-agreement-hexagon-q4_0`, 900 tokens | mel rel 3.2e-2, rms ratio 0.177 | 5e-2, 0.5 |
| Multilingual S3Gen `q4_0`, 900 tokens, 10 CFG steps | mel cosine 0.99998; max abs 1.9, above 1.0 on four of 1,806 frames | (noise of the tier, above) |
| `test-t3-backend-logits-turbo-hexagon` (`q4_0`, prompt + 24 teacher-forced steps) | min cosine 0.99980, argmax 25/25 | 0.995, 0.75 |
| `test-t3-backend-logits-mtl-hexagon` (`q4_0`, cond and uncond) | min cosine 0.99792, argmax 48/50 | 0.995, 0.75 |
| `test-chatterbox-parity-mtl-hexagon` (`q4_0`, greedy) | token counts within 4.3% | duration 1.35x |

Whisper `small.en` transcribes the NPU's Turbo, Multilingual, JFK-cloned and
streamed Multilingual outputs verbatim. Greedy Turbo decoding (`top_k = 1`)
misses the stop token on some sentences on every backend: over 12 sentences
the `q8_0` T3 ran on to the limit for the same three on the CPU and on the
NPU, and the `q4_0` T3 for two on each (different ones). Requesting F32
products for the Hexagon prompt removed one of them at three times the
prefill cost, so the prompt keeps HMX. With the default sampling (seed 42) all
12 sentences stopped on both the CPU and the NPU (at most 87 and 84 tokens).

Speed on the Galaxy S25 (registry `q4_0` T3 and S3Gen, built-in voice,
seed 42, `tts-cli`, each run started at thermal status 0, median of two runs;
inference only, model load excluded). The short text is "The quick brown fox
jumps over the lazy dog."; the paragraph is three sentences (62 words),
synthesised sentence by sentence. The OpenCL Multilingual rows cap the context
at 1,200 (see Adreno OpenCL above):

| Model | Text | Backend | Tokens | T3 | S3Gen (encoder / CFM / HiFT) | Audio | RTF |
|---|---|---|--:|--:|--:|--:|--:|
| Turbo | short | CPU (6 threads) | 69 | 1.73 s | 3.59 s (0.38 s / 2.04 s / 1.16 s) | 2.9 s | 1.85 |
| Turbo | short | OpenCL (Adreno 830) | 70 | 1.28 s | 1.41 s (0.15 s / 0.86 s / 0.40 s) | 2.9 s | 0.92 |
| Turbo | short | Hexagon | 73 | 0.55 s | 0.95 s (0.33 s / 0.29 s / 0.33 s) | 3.0 s | 0.49 |
| Turbo | paragraph | CPU (6 threads) | 427 | 8.51 s | 23.37 s (2.09 s / 11.90 s / 9.37 s) | 17.4 s | 1.83 |
| Turbo | paragraph | OpenCL (Adreno 830) | 422 | 8.16 s | 5.67 s (0.72 s / 2.77 s / 2.15 s) | 17.2 s | 0.80 |
| Turbo | paragraph | Hexagon | 454 | 3.39 s | 4.61 s (1.48 s / 1.11 s / 2.01 s) | 18.5 s | 0.43 |
| Multilingual | short | CPU (6 threads) | 61 | 1.44 s | 14.01 s (0.25 s / 12.57 s / 1.19 s) | 2.6 s | 6.04 |
| Multilingual | short | OpenCL (Adreno 830) | 61 | 1.90 s | 4.55 s (0.10 s / 4.08 s / 0.37 s) | 2.6 s | 2.52 |
| Multilingual | short | Hexagon | 60 | 0.66 s | 2.92 s (0.15 s / 2.47 s / 0.29 s) | 2.5 s | 1.42 |
| Multilingual | paragraph | CPU (6 threads) | 352 | 14.48 s | 91.79 s (1.97 s / 80.51 s / 9.30 s) | 14.4 s | 7.36 |
| Multilingual | paragraph | OpenCL (Adreno 830) | 346 | 12.06 s | 18.76 s (0.39 s / 16.52 s / 1.83 s) | 14.2 s | 2.17 |
| Multilingual | paragraph | Hexagon | 365 | 4.23 s | 12.01 s (0.75 s / 9.61 s / 1.64 s) | 15.0 s | 1.09 |

On the paragraph the NPU runs Turbo faster than real time, at RTF 0.43 (4.3x
the CPU, 1.9x OpenCL), and Multilingual close to it, at 1.09 (6.8x the CPU,
2.0x OpenCL). The Multilingual CFM dominates there: ten Euler steps of the
cond+uncond pair at about 250 ms each for a short sentence, a quarter of it
attention; its batched convolutions keep F32 columns (see above).

---
