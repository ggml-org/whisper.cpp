# tts engine: CosyVoice3

Part of the [tts engine documentation](../README.md).

## CosyVoice3

CosyVoice3 runs a Qwen2.5 speech-token LM, DiT conditional-flow-matching
network, and CausalHiFT vocoder through `tts_cpp::cosyvoice::Engine`. The
validated backends are CPU, Metal (macOS / iOS), OpenCL (Adreno), and Vulkan
(Linux / Windows desktop); `n_gpu_layers > 0` selects the GPU path via the
engine's backend requirement and other GPU backends fall back to CPU. On
Android the requirement stays Metal-or-OpenCL, so Vulkan-only mobile GPUs
(e.g. Mali, Xclipse) keep declining to CPU rather than running unvalidated.
`--vulkan-device N` pins the Vulkan adapter on multi-GPU hosts (`-1`
auto-picks the discrete card with the most free VRAM). The Hexagon NPU runs
only on explicit request (see [Snapdragon Hexagon NPU](#snapdragon-hexagon-npu)).

Use `cosyvoice-cli` for end-to-end synthesis. The `cosyvoice-hift`,
`cosyvoice-flow`, and `cosyvoice-llm` executables isolate stages, and
`cosyvoice-bench` reports per-stage timing. The streaming callback currently
chunks PCM only after the full utterance has been generated, so it preserves
the callback shape but does not reduce first-audio latency.

```bash
./build/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
  --text "Hello from CosyVoice3." --out out.wav
```

### Convert

Everything the engine loads converts from the official
[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
checkpoint plus the 3D-Speaker CAM++ torch checkpoint;
`scripts/download-cosyvoice3-checkpoint.sh` fetches both (Hugging Face CLI:
`pip install -U huggingface_hub`). The LM/flow/HiFT converters need the
[Conversion tooling](build.md#conversion-tooling) environment with `torch`; the
tokenizer converter additionally needs `pip install s3tokenizer` (or
`--s3tokenizer-repo` at a checkout of it).

```bash
scripts/download-cosyvoice3-checkpoint.sh --dir checkpoints/cosyvoice3

python3 scripts/convert-cosyvoice3-llm-to-gguf.py \
    --llm checkpoints/cosyvoice3/llm.pt \
    --outfile cosyvoice3-llm-f32.gguf --dtype f32
python3 scripts/convert-cosyvoice3-flow-to-gguf.py \
    --flow checkpoints/cosyvoice3/flow.pt \
    --config checkpoints/cosyvoice3/cosyvoice3.yaml \
    --outfile cosyvoice3-flow-f32.gguf --dtype f32
python3 scripts/convert-cosyvoice3-hift-to-gguf.py \
    --hift checkpoints/cosyvoice3/hift.pt \
    --outfile cosyvoice3-hift-f32.gguf --dtype f32

# voice-cloning add-on (required only when reference_audio is used)
python3 scripts/convert-s3tokenizer-v3-to-gguf.py \
    --onnx checkpoints/cosyvoice3/speech_tokenizer_v3.onnx \
    --out cosyvoice3-s3tok-f16.gguf --outtype f16
python3 scripts/convert-campplus-to-gguf.py \
    --ckpt checkpoints/cosyvoice3/campplus_cn_common.bin \
    --out cosyvoice3-campplus-f32.gguf
```

The LM converter accepts `--dtype {f32,f16,q8_0,q4_0}`, the flow converter
`--dtype {f32,f16,bf16,q8_0,q4_0}`, and HiFT `f32`/`f16` (its f0 predictor
stays f32 either way). The recommended desktop GPU tier is LM `q8_0` + flow
`q8_0` + HiFT `f16`, except on Metal, where flow `f16` is preferred: the DiT
there is compute-bound at the GPU's f16 GEMM rate, so `q8_0` buys no wall
time and costs a little accuracy. Measured on an M3 Ultra against one pinned
543-token trajectory with HiFT held at `f16`, `f16` leads on the DiT
(dit_euler 1351 vs 1403 ms) and the two tie on the flow+vocoder wall
(1734 vs 1728 ms), so the recommendation rests on the DiT margin and the
accuracy, not on an end-to-end gap. On CPU the float flow tiers beat `q8_0`
once ggml is built with tinyBLAS (`GGML_LLAMAFILE=ON`, the bundled-ggml
default): flow `bf16` is fastest on AVX512-BF16 hosts (Zen 4/5, recent Xeon),
flow `f16` elsewhere, both with HiFT `f16`. On Apple-silicon CPU flow `f16`
is worth 1.57x on the DiT (23848 -> 15186 ms, 20 threads, same pinned
trajectory). The CPU LM decode is weight-bandwidth bound, so LM `q4_0`
roughly halves it against `q8_0` (measured 7.5 -> 4.1 ms/token) where its
output quality is acceptable. A quantized HiFT tier would gain nothing: like
the flow's `q8_0`, quantization applies only to 2-D matmul weights, and the
vocoder is convolutions end to end. The `f16` HiFT tier does help, but
modestly and only off the critical path people expect: 1.13-1.14x on the
vocoder decode on Metal, less than the fused snake activations above it,
because `mul_mm` already stages f32 operands as half there. Avoid LM `f16`:
the engine reads the embedding tables as f32, so a f16 LM is not loadable
today — use the quantized LM tiers instead.
In `q8_0`/`q4_0` mode the flow converter quantizes only the 2D matmul
weights (conv kernels, norms, biases, the token embedding and the baked
`rand_noise` stay float), and it always writes the per-block attention
projections pre-fused as one `to_qkv` tensor; the engine also still loads
older GGUFs with separate `to_q`/`to_k`/`to_v` tensors. `bf16` mode stores
those same 2D matmul weights as bf16 and the conv kernels as f16, keeping
the kernel-typed f16 im2col path. Every reduced-precision tier is gated
against the f32 reference by `test-cosyvoice-{flow,hift}-tier-*`, which
needs only the two GGUFs staged (no PyTorch fixture). It drives the stage
with deliberately synthetic, off-distribution inputs (random-normal mel
around log-mel statistics), which is what keeps it fixture-free but also
amplifies the deviation — so what it measures is as much the host CPU
backend's matmul rounding as it is the tier GGUF, and the same GGUF
measures differently on the x86 and ARM CPU paths. Both are measured, and
the bounds registered in CMakeLists.txt are per-arch:

| flow tier | x86-64 cosine / max abs | arm64 cosine / max abs |
|---|---|---|
| `f16`  | 0.999961 / 0.246 | 0.999855 / 0.697 |
| `bf16` | 0.999671 / 0.989 | 0.998686 / 2.990 (historical scalar path; see below) |
| `q8_0` | 0.999829 / 0.658 | 0.999018 / 2.384 |
| `q4_0` | 0.986591 / 4.845 | 0.971658 / 4.125 |

(x86-64: Ryzen 9 9950X3D; arm64: Apple M3 Ultra, macOS 15.7, the CPU backend
of a Metal build. `GGML_LLAMAFILE` does not move these numbers on either
arch.) Prefer `q8_0` over `q4_0` where the flow size allows. The cosine is
the tier-integrity signal — a mis-loaded or mis-dequantized tensor collapses
it far past either column — and the absolute deviation is the host-dependent
half, which is why the arm64 column gates `q8_0` at max abs 4.0 against the
x86 column's 2.0. An x86-measured bound applied to ARM is what made
`test-cosyvoice-flow-tier-q8_0` (2.38 against a 2.0 bound) and
`-q4_0` (cosine 0.9717 against a 0.98 threshold) fail on Apple silicon.

Neither tinyBLAS nor `ggml_vec_dot_bf16` in the pinned ggml has an ARM bf16
fast path. The ARM CPU loader therefore expands `flow/` bf16 tensors to f32
once per stage load, preserving the stored bf16 values exactly, and runs
the f32 matmul path. Activations and accumulation follow that path's rounding,
so synthesized output need not be identical to the former scalar bf16 path.
The expansion doubles the resident bytes of those tensors; metadata-only
memory preflight uses the same expanded types. Conversion streams through
bounded scratch rather than materializing another full copy of the weights.
GPU loads and non-ARM CPU loads retain bf16. Prefer an f16 flow bundle on
ARM for its lower memory use and faster matmuls. The bf16 tier now also runs
in the ARM quality gate (minimum cosine 0.997, maximum absolute error 4.0).

The HiFT leg pins f0 so the gate measures weight precision rather than
sine-phase noise; its `f16` waveform deviation is 0.999966 / 0.0012 on
x86-64 and 0.999427 / 0.0065 on arm64, both inside the single
0.999 / 0.01 gate it carries on every host.

The LM converter applies the same attention fusion to each layer
(`qkv_proj`, rows q ++ k ++ v): one matvec feeds all three heads per decode
step. Row-wise quantization makes the fused tensor bit-identical to the
separate ones, and the engine still loads older GGUFs with separate
`q/k/v_proj` tensors. The engine rejects a fused tensor whose row count
disagrees with the GGUF's own `n_head` / `n_kv` / `head_dim` metadata: the
Q/K/V slices carry a custom token stride, so a short fused tensor would read
past the projection output during multi-token prefill rather than fail
loudly. Because the layout is chosen per GGUF, `test-cosyvoice-xb` reports
which one the pinned LM carries, so a pass states whether it covered the
fused matvec or the fallback.

### Stage profiling

`cosyvoice-bench` accepts `--llm-gguf`, `--flow-gguf`, and `--hift-gguf` to
override individual components while retaining the voice and tokenizer from
`--model-dir`. The JSON `model_overrides` object records these explicit paths;
an empty string means the component was discovered under `model_dir`.

First record a speech-token trajectory, then replay it with each flow tier:

```bash
./build/cosyvoice-bench --model-dir models/cosyvoice3 --backend cpu \
  --text "Hello from CosyVoice3." --threads 4 --runs 3 --warmup 1 \
  --tokens-out short.tokens --json-out baseline.json
./build/cosyvoice-bench --model-dir models/cosyvoice3 --backend cpu \
  --flow-gguf models/cosyvoice3-flow-f16.gguf --threads 4 \
  --tokens-in short.tokens --runs 3 --warmup 1 --json-out flow-f16.json
```

Repeat with a longer text and a separate trajectory. Keep HiFT fixed when
comparing flow f16, bf16, q8_0 and q4_0. Check `work` for equal speech tokens,
mel frames and samples, then compare `flow_frontend` and `dit_euler` timing.
Pinned runs skip LM generation and cannot measure LM speed. To compare LM
q4_0 and q8_0, omit `--tokens-in`, vary only `--llm-gguf`, and examine prefill,
`lm_decode_per_token` and decode-step counts. The per-token stage divides
each measured run's decode time by its own step count before aggregation;
it has zero observations for pinned runs. Differing generated lengths make raw
end-to-end RTF an unreliable attribution to a particular stage.

On RTX, repeat the same short/long flow trajectories with explicit backend
selection (`--backend CUDA0` or the Vulkan device name exposed by the build).
An unavailable explicit backend fails instead of silently using CPU. Compare
all stage timings before selecting an optimization; do not assign a mixed
LM/flow bundle's performance gap to either stage from end-to-end RTF alone.

### CUDA LM attention

Single-token LM decoding uses `FLASH_ATTN_EXT` on CUDA when the backend
supports the model's head dimensions. The KV cache stays in F32; prefill
keeps regular attention. The same selection is used by memory-fit projection.
Unsupported backends and dimensions retain the regular attention path.

CUDA decode reuses a graph within each 128-token cache window when indexed
F32 cache writes and masked flash attention are supported. Position inputs
select the cache row; an F16 mask excludes unused positions. Cache values and
attention accumulation remain F32. The mask stays allocated across replays,
and memory-fit projection includes it. Forced scheduler execution retains
the per-step graph path. CUDA graph capture also requires a ggml build with
CUDA graphs enabled.

`test-cosyvoice-lm-attention-cuda` needs no model files. It checks the
capability gate and strided-cache attention against a scalar reference across
short, unaligned, and long cache lengths. A synthetic Qwen model also compares
sampled trajectories with replay enabled and disabled across cache-window
boundaries, without external model files. `test-cosyvoice-xb-cuda` additionally
checks exact greedy-token parity against CPU when model fixtures are staged.
Floating-point differences can change sampled trajectories, so compare
`lm_decode_per_token` alongside end-to-end timings when profiling.

### Vulkan flow attention

Flow prefers F16 key/value operands for Vulkan flash attention and retains
F32 accumulation. Model weight precision is unchanged. The backend capability
probe selects F32 operands when F16 is unsupported, and regular attention
when neither flash-attention configuration is supported. Memory-fit projection
uses the same graph builder and selection.

`test-cosyvoice-flow-attention-vulkan` requires no model files. It checks the
operand selection, unsupported-dimension fallback, and batched attention
against an F32 scalar reference at short and unaligned sequence lengths.
The CPU version is registered as `test-cosyvoice-flow-attention`.

### Metal graph paths

On Metal the engine takes graph shapes chosen from a per-op GPU profile, all
gated by `test-cosyvoice-xb` and the per-backend harnesses: one
`FLASH_ATTN_EXT` per layer for single-token LM decode (with the all-zeros
causal mask elided, which is bitwise identical on every backend), f16 K/V
operands for the DiT's flash attention, and the DiT's grouped
`conv_pos_embed` emitted as one batched im2col plus one batched matmul — that
last one takes the layer pair from roughly 64 dispatches per Euler step to 6.
The per-group form stays for backends whose im2col fusion needs an unbatched
2-D signal (Adreno). The vocoder's snake activations emit the fused
`GGML_OP_SNAKE` every backend implements, which is also worth 1.09x on the
CPU vocoder decode.

The vocoder's SineGen2 source excitation (`hift_source`) is host-side work no
backend touches. Its per-harmonic sine tracks and the mix/tanh arithmetic run
on the engine's `n_threads` host threads, while the gauss draws stay one
sequential pass in the original order — the vocoder was trained on this exact
excitation, so the parallel path must stay byte-identical to the serial one
at any thread count, which `test-cosyvoice-sinegen` pins. On a pinned
543-token trajectory this takes `hift_source` from 59.7 to 34.7 ms on an M3
Ultra (20 threads) and from 49.0 to 34.7 ms on an M4 (4 threads); the
remainder is the sequential draw pass itself.

`cosyvoice-cli --flow-cut-prompt` enables an opt-in flow shortcut that treats
the voice-prompt frames as attention conditioning only (the same design
cosyvoice.cpp uses by default). It cuts DiT time by roughly the prompt's share
of the mel sequence but deviates from the PyTorch reference, so it is off by
default; `test-cosyvoice-flow-cut` pins its output against the reference mel
with a looser bound.

The engine will not construct without a baked default voice (`voice.gguf`): it
packs the four prompt tensors of one reference utterance, computed on the
upstream PyTorch stack. The dump scripts load the upstream CosyVoice3 Python
model, so they need a full Hugging Face snapshot (including `campplus.onnx`
and the yaml/config set — the converter inputs fetched above are not enough)
plus a [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) checkout with its
dependencies on `PYTHONPATH`:

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir checkpoints/Fun-CosyVoice3-0.5B

PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \
python3 scripts/dump-cosyvoice3-reference.py \
    --model-dir checkpoints/Fun-CosyVoice3-0.5B \
    --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \
    --prompt-text "希望你以后能够做的比我还好呦。" \
    --out-dir artifacts/cv3-ref
PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \
python3 scripts/dump-cosyvoice3-llm-reference.py \
    --model-dir checkpoints/Fun-CosyVoice3-0.5B \
    --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \
    --prompt-text "希望你以后能够做的比我还好呦。" \
    --out-dir artifacts/llm-ref

python3 scripts/bake-cosyvoice3-voice.py \
    --prompt-stok  artifacts/llm-ref/prompt_stok.npy \
    --prompt-token artifacts/cv3-ref/prompt_token.npy \
    --prompt-feat  artifacts/cv3-ref/prompt_feat.npy \
    --embedding    artifacts/cv3-ref/embedding.npy \
    --prompt-text "希望你以后能够做的比我还好呦。" \
    --outfile voice.gguf
```

Any 5-15 s reference clip works in place of `zero_shot_prompt.wav`; pass its
verbatim transcript as `--prompt-text`. Assemble the model directory the
engine scans:

```bash
python3 scripts/assemble-cosyvoice3-model.py \
    --llm cosyvoice3-llm-f32.gguf --flow cosyvoice3-flow-f32.gguf \
    --hift cosyvoice3-hift-f32.gguf --voice voice.gguf \
    --vocab checkpoints/cosyvoice3/CosyVoice-BlankEN/vocab.json \
    --merges checkpoints/cosyvoice3/CosyVoice-BlankEN/merges.txt \
    --s3tok cosyvoice3-s3tok-f16.gguf --campplus cosyvoice3-campplus-f32.gguf \
    --out models/cosyvoice3-0.5b
```

### Voice cloning

With no reference audio the engine speaks with the baked default voice
(`voice.gguf`). Zero-shot / cross-lingual cloning runs fully natively — the
reference wav is tokenized by a ggml port of `speech_tokenizer_v3` (12-block
FSMN encoder + FSQ, converted by `scripts/convert-s3tokenizer-v3-to-gguf.py`),
the 192-d speaker embedding comes from the CAM++ port
(`scripts/convert-campplus-to-gguf.py`; the same 3D-Speaker checkpoint
CosyVoice ships as `campplus.onnx`), and the prompt mel from the shared
matcha-compatible mel extractor. Both GGUFs are required whenever
`reference_audio` is set (`cosyvoice3-s3tok*.gguf`,
`cosyvoice3-campplus*.gguf`, auto-resolved from the model dir); a missing
model, an unreadable wav, or an out-of-range duration (0.5-30 s hard limits,
5-15 s recommended) fails construction rather than silently keeping the baked
voice.

The reference transcript selects the mode, mirroring the upstream frontends:

```bash
# zero-shot: transcript given, LM prompted with the reference speech tokens
# (best fidelity when synthesizing the reference's own language)
./build/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
  --reference-audio me.wav --prompt-text "verbatim transcript of me.wav" \
  --text "Same language as the reference." --out out.wav

# cross-lingual: no transcript, timbre-only conditioning through the flow
./build/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
  --reference-audio me.wav --text "Any other language." --out out.wav
```

The bake runs once at engine construction (roughly a second of CPU for the
tokenizer + CAM++ + mel on a short clip; the tokenizer graph rides the
engine's GPU backend when one is selected, and the CPU on Hexagon). Instruct
mode composes with a cloned voice: the instruction drives dialect/style while
the cloned tensors supply the timbre.

### Snapdragon Hexagon NPU

CosyVoice3 runs on the Snapdragon Hexagon NPU (HTP0; measured on a Snapdragon
8 Elite, Hexagon v79) when it is requested explicitly:
`EngineOptions::backend = "hexagon"` or `--backend hexagon` on
`cosyvoice-cli`, `cosyvoice-bench` and `cosyvoice-fit-params`. There is no
fallback: construction fails when HTP0 is missing, and the automatic
`--n-gpu-layers` walk never selects it. The LM (prefill and every decode
step), the DiT and the vocoder run on the NPU; the host samples the speech
tokens and builds the vocoder's source excitation. With a reference WAV the
cloning tokenizer stays on the CPU, where CAM++ runs on every backend. Weights
go into the Hexagon repack buffer type. HTP multiplies only `f32`, `f16`,
`q8_0` and `q4_0` weights, so the `bf16` flow is refused at load. It needs a
`qvac-ext-ggml@speech` build with the Hexagon F32 flash attention, nearest
upscale, ELU, ABS, leaky ReLU, sin and cos kernels and the batched-matmul
flattening.

The DiT's attention runs in an F32 HVX kernel rather than the F16 HMX one.
Several DiT layers carry attention-sink activations (|q| and |k| up to about
1,100 under real conditioning) whose logits reach about 1.6e6 while the gaps
that set the softmax are tens: F16 logits overflow or round them away, and F16
queries and keys lose them. The F16 kernel would take the DiT of a 7.8 s clip
from 12.6 to 4.4 s, but the output's log-mel cosine to the CPU falls from
0.9893 to 0.9655. Three graph forms also differ from the other backends: the
DiT rotates the first head inside each full row (one ROPE over the fused QKV
view rather than split, rotate and concatenate), the vocoder's iSTFT runs as
one GEMM into columns plus `ggml_col2im_1d` (HTP has no transposed
convolution), and the vocoder's convolutions run over F16 im2col columns on
HMX, which multiplies in F16 anyway; HTP has no fast F32 im2col.
`test-cosyvoice-hexagon-graphs` pins the first two against the generic forms
on the CPU and, with `COSYVOICE_TEST_BACKEND=hexagon`, on the NPU.

Accuracy, `test-cosyvoice-xb --backend NAME` (each stage over the same
deterministic inputs on the CPU and on the named device; the LM leg compares
the greedy speech-token trajectory over the baked voice, the flow leg the mel,
the HiFT leg the waveform with f0 pinned):

| Check | Hexagon | OpenCL (Adreno 830) |
|---|--:|--:|
| LM `q8_0` (fused QKV), greedy trajectory, 800 tokens | identical | identical |
| LM `q4_0` (separate Q/K/V) | identical | identical |
| flow `q8_0`, mel cosine / max abs | 0.999839 / 0.584 | 0.999859 / 0.463 |
| flow `q4_0` | 0.999229 / 0.647 | 0.999415 / 0.562 |
| flow `f16` | 0.999889 / 0.349 | 0.999926 / 0.381 |
| flow `f32` | 0.999941 / 0.462 | 1.000000 / 0.003 |
| HiFT `f16`, waveform cosine / max abs | 0.999269 / 0.0082 | 0.999282 / 0.0089 |
| HiFT `f32` | 0.999898 / 0.0029 | 1.000000 / 0.0000 |

HMX multiplies in F16 wherever a matmul does not request F32 precision, so
unlike OpenCL even the `f32` tiers are not bit-close to the CPU on Hexagon;
every tier stays well inside the gates (flow 0.995 / 4.0, HiFT 0.999).

On real text with the speech tokens pinned (the three utterances below), the
output's log-mel cosine to the CPU's is 0.9887-0.9899 on Hexagon and
0.9878-0.9889 on OpenCL, and whisper `base.en` transcribes the Hexagon output
verbatim.

Galaxy S25, LM `q8_0` + flow `q8_0` + HiFT `f16`, baked English voice. The
flow and vocoder run over the speech tokens of one seeded CPU run
(`cosyvoice-bench --tokens-in`), median of 2 warm runs, each backend started
from thermal status 0, ms:

| Audio | CPU (6 threads) | OpenCL (Adreno 830) | Hexagon | Hexagon vs OpenCL |
|---|--:|--:|--:|--:|
| 4.3 s (107 speech tokens) | 56,109 | 28,536 | 9,025 | 3.16x |
| 7.8 s (195 speech tokens) | 75,746 | 37,176 | 13,726 | 2.71x |
| 19.8 s (494 speech tokens) | 182,744 | 105,216 | 35,323 | 2.98x |

The DiT is 92% of the Hexagon time: ten Euler steps over the voice prompt's
550 mel frames plus the utterance's, at classifier-free-guidance batch 2. On
the 7.8 s utterance the F32 attention is about three quarters of it (43 ms per
call, 220 calls), so the weight tier barely matters: the flow and vocoder take
13.7 s with the `q8_0` flow, 13.9 s with `q4_0`, 14.0 s with `f16` and 14.1 s
with `f32` (and an `f32` HiFT). The recommended GPU tier (`q8_0` LM and flow,
`f16` HiFT) holds on Hexagon too.

The LM is the one stage where Hexagon trails. Its matmuls request F32
precision (the cross-backend LM gate is exact greedy equality), which keeps
them on HVX instead of HMX, and a decode step is short enough for the host's
per-step work to show: prefill takes 1.4-1.8 s against 0.9-1.1 s on the CPU
and 0.34-0.41 s on OpenCL, and a decode step 15.9-20.3 ms against 8.6-9.1 ms
and 12.6-14.3 ms. With the LM at those rates over the same tokens, Hexagon
renders the three utterances end to end in 12.1, 18.4 and 47.2 s: 2.2-2.5x
faster than OpenCL and 4.0-4.8x faster than the CPU, but still 2.4-2.8x slower
than real time. `GGML_HEXAGON_OPPOLL=1` (the host polls the DSP queue instead
of waiting) cuts LM decoding to 12.9-18.0 ms per token, slows the flow by
about 5% and leaves the phone at thermal status 2, for no net gain.
