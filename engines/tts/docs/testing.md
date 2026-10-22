# tts engine: fixtures and validation

Part of the [tts engine documentation](../README.md).

## Fixtures and static validation

The checkout includes the model-free voice-clone metric fixtures under
`test/fixtures/voiceclone/v1` and the public-domain JFK voice reference under
`test/reference-audio`. Model-dependent fixtures remain optional:

```bash
python scripts/dump-s3gen-reference.py --out artifacts/s3gen-ref
python scripts/dump-supertonic-reference.py \
  --onnx-dir /path/to/supertonic/onnx \
  --assets-dir /path/to/supertonic/assets \
  --voice-style /path/to/supertonic/assets/voice_styles/M1.json \
  --lang en --out artifacts/supertonic-ref-quick
```

For Supertonic 3, use the same existing dumper interface once per language,
pointing `--onnx-dir`, `--assets-dir`, and `--voice-style` at the v3 bundle:

```bash
python scripts/dump-supertonic-reference.py \
  --onnx-dir /path/to/supertonic-3/onnx \
  --assets-dir /path/to/supertonic-3/assets \
  --voice-style /path/to/supertonic-3/assets/voice_styles/M1.json \
  --lang en --out artifacts/supertonic3-ref-en
```

The current dumper accepts the legacy five-language `--lang` choices. Generate
`en`, `ko`, `es`, `pt`, and `fr` for the registered Supertonic 3 parity suite;
the model-free `test-supertonic-languages` target covers the complete v3
language registry.

For quantized Parler inspection, report-only mode is an environment variable
on the individual harness executable:

```bash
PARLER_TEST_REPORT_ONLY=1 ./build/test-parler-t5 MODEL.gguf REF_DIR
PARLER_TEST_REPORT_ONLY=1 ./build/test-parler-decoder MODEL.gguf REF_DIR
```

Do not treat `ctest -N` or disabled fixture registrations as executed tests.

### Optional: validate against Python references

Every stage of the pipeline has a numerical regression test against
Python-dumped reference tensors:

```bash
./build/test-s3gen models/chatterbox-s3gen.gguf artifacts/s3gen-ref ALL
```

Expected output (rel error per stage):

```
Stage A  speaker_emb_affine    rel ≈ 1e-7
Stage B  input_embedded        rel = 0
Stage C  encoder_embed         rel ≈ 4e-7
Stage D  pre_lookahead         rel ≈ 3e-7
Stage E  enc_block0_out        rel ≈ 1e-7
Stage F  encoder_proj (mu)     rel ≈ 5e-7
Stage G1 time_mixer            rel ≈ 7e-7
Stage G2 cfm_resnet_out        rel ≈ 3e-7
Stage G3 tfm_out               rel ≈ 2e-7
Stage G4 cfm_step0_dxdt        rel ≈ 1e-6
Stage H1 f0                    rel ≈ 4e-6
Stage H3 conv_post             rel ≈ 6e-7
Stage H4 stft                  rel ≈ 8e-3 (boundary-bound)
Stage H5 waveform              rel ≈ 1e-4
```

For T3 bit-exact validation against the Python reference:

```bash
python scripts/reference-t3-turbo.py \
  --text "Hello from ggml." \
  --out-dir artifacts \
  --cpp-bin ./build/tts-cli \
  --cpp-model models/chatterbox-t3-turbo.gguf
```

## Recorded GPU validation

The following records describe the validation performed when the backend paths
were introduced. Fixture counts and timings are historical evidence; use the
current CTest registrations to inspect available coverage.

 on two ggml-cuda fixes carried by
[`qvac-ext-ggml@speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech).
Without the first (`im2col` / `pad` striding the grid Y and Z axes, which are
capped at 65535) a chunk past roughly ten seconds aborts the launch; Chatterbox
splits on sentences but readily emits a thirty-second chunk from ordinary
prose, so long-form input reaches it. Without the second (`conv_transpose_1d`
bounded to the kernel taps that reach each output) the HiFT vocoder is slower
on CUDA than on CPU and dominates the pipeline. With both, Chatterbox Turbo
runs at RTF 0.05 against 0.72 on CPU, and Multilingual at 0.08 against 3.23 --
CPU cannot keep up with real time on Multilingual and CUDA is comfortably
ahead of it.

What that row is based on: `test-s3gen` compares every S3Gen and HiFT stage to
the Python reference dump under an explicit per-stage tolerance and returns
non-zero when one is exceeded. It is registered per backend (`test-s3gen-cuda`,
`test-s3gen-vulkan`), each arm pinning its backend through
`TTS_CPP_GPU_BACKEND` and failing rather than falling back to the CPU.

T3 carries a weaker claim, and the shape of it matters. Its `t3-mtl-ref`
fixture cannot be regenerated -- the dump script pins no conditioning, so a
fresh dump disagrees with the C++ by about 4e-2 on CPU and CUDA alike -- so
there is no reference parity for T3 on any backend. Nor do CPU and GPU agree
token for token: T3 decodes autoregressively, so a single differing argmax
re-rolls the remainder and the two emit different counts for the same input
(55 against 56, 64 against 68, 63 against 69 on the three phrases the arm
uses). `test-chatterbox-parity-mtl-{cuda,vulkan}` therefore asserts what does
hold -- both backends produce audio, and within 1.35x of each other's duration,
which truncation, runaway generation and silent failure all break. T3 on CUDA
is gated at that strength and no more.

Parler on CUDA is covered by the same per-stage reference harnesses as the
other GPU backends, registered per backend as `test-parler-dac-{cuda,vulkan}`
and run with the full Parler suite pinned to each: 15/15 on CUDA and on
Vulkan against the mini-v1 fixtures. The Indic checkpoint remains hub-gated,
so its arm carries the mini/large result, as it always has. The DAC
range-equivalence check now states its real contract: bit-identity on the
CPU, and a measured tolerance on GPU backends, where a ranged decode near a
sequence edge builds a shorter graph than the full decode and a GEMM over a
different shape legitimately reduces in a different order (worst observed:
5.12e-4 on CUDA, under 1e-5 on Vulkan, against a 2e-3 bar; the window
arithmetic errors the check exists to catch are hop-scale).

Parler's autoregressive decode fuses the per-layer QKV projections and the
nine LM heads into single matmuls on GPU and on mmap-backed CPU loads
(byte-exact; `PARLER_NO_FUSED` disables fusion), samples each step over the
top-k candidate set with reused scratch, and on host backends samples the
step logits in place from the graph buffer instead of downloading a copy
per step. On CPU the decoder runs flash attention over an F32 KV cache, and
on Apple builds the DAC's convolutions run on Accelerate (`PARLER_NO_FA` and
`PARLER_DAC_NO_ACCEL` opt out). Details in
[docs/parler.md](parler.md); M3 Ultra CPU numbers in
[docs/performance.md](performance.md#parler-tts-on-cpu-mac-studio-m3-ultra).

Audio8 on CUDA required one ggml-cuda fix and a rethink of what its
harnesses measure. The fix: the transpose fast path of the CUDA copy kernel
ignores destination strides, so the V-cache append -- a transposed source
into a strided cache view -- corrupted attention from the first prefill step;
with the guarded kernel the LM's per-step numerics match the Vulkan-vs-CPU
baseline node for node. The rethink: Audio8's networks are expansive enough
that sub-ulp cross-backend rounding grows into large per-element differences,
and its rollouts are free-running, so one differing argmax re-rolls the rest.
The harnesses now gate on observables that are stable across backends -- a
teacher-forced LM trace with per-step numeric bounds and a token-agreement
rate (CPU exact, CUDA 96.9 percent, Vulkan 100), codec codes agreement with
energy-level latent bounds, and rollout audio sanity with correlation
reported for information -- with every bar measured on all three backends.
The same redesign is what fixed test-audio8-codec and test-audio8-lm-vulkan,
which had been failing on unmodified master since the speech ggml moved past
their calibration.

CosyVoice3 supports `f32` weights, `q8_0`/`q4_0` LM and flow weights,
`f16`/`bf16` flow weights, and `f16` HiFT weights. The recommended desktop
GPU combination is a `q8_0` LM, `q8_0` flow, and `f16` HiFT — except on
Metal, where the `f16` flow measured slightly ahead of `q8_0` (the GEMMs
there are compute-bound, not weight-bandwidth-bound), so Apple targets
prefer `q8_0` LM + `f16` flow + `f16` HiFT; on CPU use a
`bf16` flow on AVX512-BF16 hosts and `f16` elsewhere (measured on a 16-core
Zen 5, 16 threads, same pinned 14.8 s utterance: flow+vocoder wall 16.7 s
with f32 weights on a ggml built without tinyBLAS falls to 8.6 s with the
f16 tier and 7.7 s with the bf16 tier on a `GGML_LLAMAFILE=ON` build, the
bundled-ggml default since this change). The optional
`cosyvoice-cli --flow-cut-prompt` flag reduces flow work: only the first DiT
block attends to the voice-prompt frames, and subsequent blocks process the
generated region. It changes reference output and is off by default. See
[CosyVoice3 conversion and usage](cosyvoice3.md) for details and the
unsupported LM `f16` caveat.

CosyVoice3 on Metal takes graph paths the profiler singled out on Apple
GPUs, all gated by the cross-backend and per-backend harnesses: the LM's
single-token decode runs one flash-attention node per layer instead of the
masked matmul/softmax chain (with the all-zeros causal mask elided — the
greedy trajectory stays bit-identical to the CPU's, measured over 1680
consecutive steps), newer LM GGUFs feed it one fused `qkv_proj` matvec, the
DiT's flash attention takes f16 K/V operands, its grouped `conv_pos_embed`
collapses from 64 dispatches per Euler step to one batched im2col + matmul,
and the vocoder's snake activations run as single fused `GGML_OP_SNAKE`
kernels on every backend. Measured against the previous revision on an M3
Ultra with `--greedy`, so both legs decode the identical 550-token trajectory
(22.0 s audio) and every stage is directly comparable, f16 flow + f16 HiFT
throughout: end-to-end RTF 0.214 -> 0.179 (4710 -> 3934 ms, 1.20x, 5.6x real
time), of which LM decode 4.89 -> 3.73 ms/token (1.31x, the flash-attention
step plus the fused `qkv_proj` GGUF), DiT 1463 -> 1359 ms (1.08x) and HiFT
decode 190 -> 163 ms (1.17x). On an M4 the same change moves end-to-end RTF
0.598 -> 0.544 and HiFT decode 838 -> 686 ms (sampled legs, so the
trajectories differ slightly). The remaining DiT time is machine-rate GEMM
and flash attention (13+ TFLOPS measured per op), which is why the f16 tier,
not `q8_0`, is the Metal recommendation: measured on the same machine with
the HiFT tier held at f16, flow f32 -> f16 moves the DiT 1426 -> 1351 ms
while f16 -> `q8_0` moves it back to 1403 ms. The f16 HiFT tier is worth
1.13-1.14x on the vocoder decode there — less than the fused snake above it,
because Metal's `mul_mm` already stages f32 operands as half, so a narrower
weight dtype saves bandwidth rather than arithmetic. The vocoder's host-side
SineGen2 source excitation is threaded over harmonics and the mix arithmetic
on the engine's `n_threads` (the RNG pass stays sequential, so the output is
byte-identical at any thread count): 59.7 -> 34.7 ms on the M3 Ultra and
49.0 -> 34.7 ms on the M4, same pinned trajectories.

CosyVoice3 on CUDA is covered by the same per-stage reference harnesses as
its other GPU backends, each registered per backend --
`test-cosyvoice-{flow,llm,hift,conv1d,frontend,clone}-{cuda,vulkan}` and
`test-s3tokenizer-v3-{cuda,vulkan}[-q8_0]` -- so both desktop backends stay
pinned rather than whichever selection prefers: the full CosyVoice and
s3tokenizer suite passes 27/27 pinned to CUDA and to Vulkan. Greedy long-form
synthesis (67 s of audio, well past the launch-abort threshold the
grid-striding fix removed) produces the identical sample count (1617600) on
CUDA, Vulkan and the CPU -- each measured -- at roughly 12 s wall on either
GPU against 526 s on the CPU. The CUDA column here is engine-level validation; the consumer path --
the published `speech-cpp[cuda]` port and the addon behind it -- ships with
the registry bump that closes this model series, which is when that port and
the qvac workflows exercise it end to end.

Supertonic on CUDA rests on a backend-capability distinction the F16-weight
auto policy now probes: the conv-via-im2col lowerings put the weight in
mul_mat's second operand, and ggml-cuda (and ggml-cpu) accept an F16 weight as
the first operand while rejecting it as the second, where ggml-vulkan accepts
both. Materialising F16 weights on such a backend left the scheduler with a
node no backend claims and aborted the first synthesis. The auto policy
requires both orientations, so CUDA and any future backend in its position
keep F32 weights -- slower but correct, the same trade the policy already
makes on CPU and OpenCL -- while Vulkan and Metal keep the F16 roster. With
that in place all three Supertonic generations synthesize on CUDA at every
shipped tier (f32, f16, q8_0, q4_0), the direct and scheduler paths are
bit-identical per backend (`test-supertonic-sched-equivalence-{cuda,vulkan}`),
and CUDA matches the CPU's output length exactly on the reference sentences.

## Validation harness index

Commands below assume a direct build from `engines/tts`.

`supertonic-bench` is built only by `TTS_CPP_BUILD_TESTS`; it is not in the
executable gate. The tests also produce the following representative
validation harnesses:

| Binary | What it does |
|--------|--------------|
| `build/supertonic-bench` | Per-stage Supertonic benchmark harness (`--text` / `--out` / `--runs`); machine-readable RTF + per-stage timings |
| `build/test-s3gen`            | Staged numerical validation of S3Gen encoder + CFM vs Python dumps |
| `build/test-resample`         | Round-trip SNR of the C++ Kaiser-windowed sinc resampler + output-frequency helpers (validate / passthrough / ratio) |
| `build/test-output-sample-rate` | `--output-sample-rate` on `chatterbox::Engine`: native/16 kHz batch, out-of-range rejection, streaming `pcm == concat(chunks)` invariant (needs the MTL GGUFs) |
| `build/test-voice-features`   | 24 kHz 80-ch mel parity (prompt_feat) |
| `build/test-fbank`            | 16 kHz 80-ch Kaldi fbank parity |
| `build/test-voice-encoder`    | VoiceEncoder 256-d speaker embedding parity |
| `build/test-campplus`         | CAMPPlus 192-d embedding parity |
| `build/test-voice-embedding`  | wav → fbank → CAMPPlus end-to-end parity |
| `build/test-s3tokenizer`      | S3TokenizerV2 log-mel + speech-token parity |
| `build/test-tts-streaming`        | Per-chunk CFM + HiFT parity for the streaming pipeline (B1) |
| `build/test-mtl-tokenizer`    | Multilingual grapheme tokenizer parity vs the HF reference |
| `build/test-t3-mtl`           | End-to-end MTL T3 (Llama-520M) forward-pass parity |
| `build/test-t3-mtl-stages`    | Staged MTL T3 parity (cond/text/inputs/layers/head) |
| `build/test-cpu-caches`       | CPU-side persistent-cache validation (time_mlp / time_emb / cfm_estimator / weight_mirror caches that amortise per-synth overhead on the multilingual CPU path); no-arg invocation runs the bit-cast cache-key + initial-state checks, GGUF arg runs the warm-cache + bit-exact pipeline check |
| `build/test-t3-caches`        | T3 step-graph cache validation (no-arg = initial-state, GGUF arg = warm-cache + bit-exact end-to-end check) |
| `build/test-supertonic-*`     | Per-stage Supertonic parity harnesses (`preprocess`, `vocoder` ± `trace` / `pointwise`, `duration` ± `trace`, `text-encoder` ± `trace`, `vector` ± `trace`, `pipeline`); each takes `MODEL.gguf REF_DIR` |
| `build/test-metal-ops`        | Metal-only: parity check for `diag_mask_inf`, `pad_ext`, and fast `conv_transpose_1d` (only useful when built with `-DGGML_METAL=ON`) |
| `build/test-cosyvoice-time-emb` | Sinusoidal timestep embedding for the CosyVoice3 flow DiT: sin/cos layout, the 1000x scale, the four-decade frequency schedule, batch-row independence (no GGUF) |
| `build/test-conv-columns` | The transposed convolution as GEMM + `col2im_1d` (the Chatterbox HiFT upsamplers and iSTFT on Hexagon) against `ggml_conv_transpose_1d` at HiFT shapes; pass a backend name (e.g. `hexagon`) to also run it there (no GGUF) |
| `build/test-attention-fallback` | The unfused T3 attention (taken where a backend declines `FLASH_ATTN_EXT`, e.g. ggml-opencl on Adreno) against `ggml_flash_attn_ext` for prompt, decode-step, batch-2 and grouped-head shapes over strided KV-cache views; pass a backend name (e.g. `opencl`) to also run it there (no GGUF) |
| `build/test-t3-backend-logits` | Chatterbox T3 (Turbo or Multilingual) prompt and teacher-forced step logits on a named backend against the CPU: min cosine and argmax agreement, and the F16 KV cache on Hexagon (`T3.gguf BACKEND`) |
| `build/test-chatterbox-explicit-backend` | Explicit Chatterbox backend requests: `cpu` matches the automatic CPU path bit for bit; a missing device fails T3, S3Gen, the Engine and the fit projection instead of falling back |
| `build/test-cosyvoice-hexagon-graphs` | CosyVoice3 graph forms Hexagon takes (iSTFT as GEMM + `col2im_1d`, first-head RoPE inside full rows) against the generic ones, the weight types it accepts, and the cloning tokenizer's backend; `COSYVOICE_TEST_BACKEND=hexagon` also runs them on the NPU (no GGUF) |
| `build/test-lavasr-gguf-load`   | Fail-closed GGUF loading for both LavaSR stages: missing, empty, truncated, unmarked, cross-architecture, and tensorless files (no GGUF) |

The test targets register with CTest. From a single-config build directory use
`ctest -L unit` or `ctest -L fixture`; with Visual Studio or another
multi-config generator add `-C Release`.
